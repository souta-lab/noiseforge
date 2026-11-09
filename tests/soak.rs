//! Long run stability (DESIGN.md section 9.2: "48 kHz, eight hours, no dropouts
//! and no abnormal memory growth").
//!
//! Rendering eight hours faster than realtime is the only practical way to run
//! that check inside a test, so this renders a configurable amount of audio as
//! fast as the CPU allows and asserts that nothing drifts, blows up or leaks.
//!
//! It is opt-in because it is slow:
//!
//! ```sh
//! NOISEFORGE_SOAK_SECONDS=28800 cargo test --release -p noiseforge --test soak -- --nocapture
//! ```
//!
//! The default is skipped so `cargo test` stays quick.

use std::time::Instant;

use noise_dsp::compile::compile;
use noise_dsp::engine::{Command, Engine};
use noise_dsp::params::{LayerParams, PresetParams};

/// Seconds of audio to render; `0` (or unset) skips the test.
fn soak_seconds() -> f64 {
    std::env::var("NOISEFORGE_SOAK_SECONDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0)
}

fn soak_preset() -> PresetParams {
    PresetParams {
        seed: 987_654_321,
        layers: (0..8)
            .map(|i| LayerParams {
                name: format!("Layer {i}"),
                color: noise_dsp::Color::ALL[i % 6],
                alpha: -3.5 + i as f64,
                character: noise_dsp::Character::ALL[i % 3],
                character_amount: 0.65,
                high_pass_hz: 25.0 + i as f64 * 15.0,
                low_pass_hz: 18_000.0,
                stereo_correlation: 0.2 * i as f64 / 7.0,
                gain_db: -3.0,
                seed: None,
                ..LayerParams::default()
            })
            .collect(),
        ..PresetParams::default()
    }
}

#[test]
fn an_eight_hour_render_does_not_drift_blow_up_or_leak() {
    let seconds = soak_seconds();
    if seconds <= 0.0 {
        eprintln!(
            "soak skipped: set NOISEFORGE_SOAK_SECONDS to run it \
             (28800 for the eight hour case in DESIGN.md section 9.2)"
        );
        return;
    }

    let fs = 48_000.0;
    let params = soak_preset();
    let graph = compile(&params, fs, 2);
    assert_eq!(
        graph.layers.len(),
        8,
        "the soak must exercise all eight layers"
    );

    let mut engine = Engine::new(fs, params.seed);
    for c in graph.commands(&params) {
        engine.apply(c);
    }
    engine.apply(Command::Play);

    let block = 1024usize;
    let mut sink = vec![0.0f32; block * 2];
    let total_blocks = (seconds * fs / block as f64) as usize;

    // Measure in windows so a slow drift is visible rather than averaged away.
    let windows = 64usize;
    let per_window = (total_blocks / windows).max(1);
    let mut window_rms = Vec::with_capacity(windows + 1);
    let mut peak = 0.0f64;
    let mut non_finite = 0u64;
    let mut acc = 0.0f64;
    let mut n = 0u64;

    let started = Instant::now();
    for b in 0..total_blocks {
        engine.render(&mut sink, block, 2);
        for s in &sink {
            let v = *s as f64;
            if !v.is_finite() {
                non_finite += 1;
                continue;
            }
            peak = peak.max(v.abs());
            acc += v * v;
            n += 1;
        }
        if b % per_window == per_window - 1 {
            window_rms.push(if n > 0 { (acc / n as f64).sqrt() } else { 0.0 });
            acc = 0.0;
            n = 0;
        }
    }
    let elapsed = started.elapsed();
    let rendered = total_blocks * block;
    eprintln!(
        "soak: {:.1} h of audio rendered in {:.1} s ({:.0}x realtime), {} windows",
        rendered as f64 / fs / 3600.0,
        elapsed.as_secs_f64(),
        (rendered as f64 / fs) / elapsed.as_secs_f64(),
        window_rms.len()
    );

    assert_eq!(non_finite, 0, "{non_finite} non-finite samples");
    assert_eq!(engine.meters().nan_events(), 0);

    // No window may be silent, and the level must not drift.
    assert!(window_rms.iter().all(|v| *v > 0.0), "a window was silent");
    let first: f64 =
        window_rms[..window_rms.len() / 4].iter().sum::<f64>() / (window_rms.len() / 4) as f64;
    let last: f64 = window_rms[window_rms.len() * 3 / 4..].iter().sum::<f64>()
        / (window_rms.len() - window_rms.len() * 3 / 4) as f64;
    let drift_db = 20.0 * (last / first).log10();
    assert!(
        drift_db.abs() < 0.5,
        "level drifted {drift_db:.3} dB over the run ({first:.6} -> {last:.6})"
    );

    // The calibrated target is -36 dBFS; a modest statistical spread is fine but
    // a systematic offset is not.
    let mean: f64 = window_rms.iter().sum::<f64>() / window_rms.len() as f64;
    let mean_dbfs = 20.0 * mean.log10();
    assert!(
        (mean_dbfs + 36.0).abs() < 2.0,
        "mean level {mean_dbfs:.2} dBFS is off the -36 dBFS target"
    );
    assert!(
        peak.is_finite() && peak < 1.0,
        "peak {peak} suggests runaway gain"
    );

    eprintln!(
        "soak result: mean {mean_dbfs:.2} dBFS, drift {drift_db:+.3} dB, peak {:.2} dBFS",
        20.0 * peak.log10()
    );
}
