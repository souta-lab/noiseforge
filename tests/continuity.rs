//! Continuity and robustness tests: DESIGN.md sections 9.2 and 9.3.
//!
//! * No setting change may put a step into the output.
//! * Rapid, extreme and adversarial control traffic must never produce NaN,
//!   Inf, a panic or unbounded CPU.
//! * A long run must not drift, blow up or leak filter state.

use noise_dsp::compile::compile;
use noise_dsp::engine::{Command, Engine, LayerFilters, MAX_BLOCK, PlayState};
use noise_dsp::params::{LayerParams, PresetParams};

/// One millisecond sliding RMS in a 64 sample window.
fn sliding_rms(x: &[f32], window: usize) -> Vec<f32> {
    if x.len() < window {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(x.len() - window + 1);
    let mut acc: f64 = x[..window].iter().map(|v| (*v as f64) * (*v as f64)).sum();
    out.push((acc / window as f64).sqrt() as f32);
    for i in window..x.len() {
        let a = x[i - window] as f64;
        let b = x[i] as f64;
        acc += b * b - a * a;
        out.push((acc / window as f64).max(0.0).sqrt() as f32);
    }
    out
}

/// Largest jump between consecutive sliding RMS values, as a ratio.
///
/// Windows below 5% of the loudest window are ignored: a fade legitimately
/// starts from exact silence, and dividing by a near-zero window would report a
/// huge "jump" that is really just the first audible sample of a ramp.
fn max_energy_jump(x: &[f32]) -> f32 {
    let rms = sliding_rms(x, 64);
    let max = rms.iter().copied().fold(0.0f32, f32::max);
    if max <= 0.0 {
        return 0.0;
    }
    let floor = max * 0.05;
    let mut worst = 0.0f32;
    for w in rms.windows(2) {
        if w[0] >= floor {
            worst = worst.max(w[1] / w[0]);
        }
    }
    worst
}

fn configured_engine(params: &PresetParams, fs: f64) -> Engine {
    let graph = compile(params, fs, 2);
    let mut e = Engine::new(fs, params.seed);
    for c in graph.commands(params) {
        e.apply(c);
    }
    e
}

fn preset_with(layers: Vec<LayerParams>) -> PresetParams {
    PresetParams {
        seed: 8,
        layers,
        ..PresetParams::default()
    }
}

#[test]
fn every_control_change_is_click_free() {
    // Two engines fed by the same PRNG stream: one is left alone, the other gets
    // a control change. The difference isolates the processing change, so a
    // click shows up as a spike in the difference energy.
    let params = preset_with(vec![LayerParams {
        color: noise_dsp::Color::Pink,
        ..LayerParams::default()
    }]);
    let fs = 48_000.0;

    let cases: Vec<(&str, Command)> = vec![
        (
            "layer gain",
            Command::SetLayerGain {
                layer: 0,
                gain_db: 9.0,
                ramp_ms: 20.0,
            },
        ),
        (
            "master gain",
            Command::SetMasterGain {
                gain_db: 6.0,
                ramp_ms: 20.0,
                hold_ms: 0.0,
            },
        ),
        (
            "stereo correlation",
            Command::SetStereoCorrelation {
                layer: 0,
                correlation: -0.8,
                ramp_ms: 20.0,
            },
        ),
        (
            "character gain",
            Command::SetCharacterGain {
                layer: 0,
                low_gain: 0.41,
                high_gain: 0.58,
                ramp_ms: 20.0,
            },
        ),
        (
            "character shape",
            Command::SetCharacterShape {
                layer: 0,
                low_shelf: noise_dsp::BiquadCoeffs::one_pole_low_pass(180.0, fs),
                high_shelf: noise_dsp::BiquadCoeffs::one_pole_high_pass(2500.0, fs),
                ramp_ms: 20.0,
            },
        ),
        (
            "colour coefficients",
            Command::SetLayerFilters {
                layer: 0,
                filters: LayerFilters {
                    color: noise_dsp::designer::design_color(2.0, fs).unwrap().coeffs,
                    high_pass: noise_dsp::designer::design_high_pass(35.0, 0.707, fs),
                    low_pass: noise_dsp::designer::design_low_pass(16_000.0, 0.707, fs),
                    expected_rms: 0.0,
                },
                crossfade_ms: 40.0,
            },
        ),
        (
            "layer seed",
            Command::SetLayerSeed {
                layer: 0,
                seed: 0x5EED_1234,
                fade_ms: 20.0,
            },
        ),
        (
            "layer disable",
            Command::SetLayerEnabled {
                layer: 0,
                enabled: false,
                fade_ms: 30.0,
            },
        ),
    ];

    for (name, cmd) in cases {
        let mut test = configured_engine(&params, fs);
        let mut control = configured_engine(&params, fs);
        for e in [&mut test, &mut control] {
            e.apply(Command::Play);
        }
        let _ = control.render_block(4096);
        let _ = test.render_block(4096);

        test.apply(cmd);
        let mut a = Vec::new();
        let mut b = Vec::new();
        for _ in 0..40 {
            a.extend_from_slice(test.render_block(256));
            b.extend_from_slice(control.render_block(256));
        }
        let diff: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x - y).collect();
        assert!(
            diff.iter().all(|v| v.is_finite()),
            "{name}: non-finite output"
        );
        let jump = max_energy_jump(&diff);
        assert!(jump < 1.6, "{name}: difference energy jumped {jump}x");
    }
}

#[test]
fn a_flood_of_control_traffic_stays_bounded() {
    // Replays real daemon traffic: each "preset change" is a full command batch
    // from `compile`, which always carries the recalibrated master gain together
    // with the new coefficients. That is what keeps the level legal while a user
    // drags a slider from `alpha = -4` to `alpha = +4`.
    let fs = 48_000.0;
    let colours = [
        noise_dsp::Color::White,
        noise_dsp::Color::Pink,
        noise_dsp::Color::Brown,
        noise_dsp::Color::Blue,
        noise_dsp::Color::Violet,
        noise_dsp::Color::Custom,
    ];
    let mut batches: Vec<Vec<Command>> = Vec::new();
    for (i, color) in colours.iter().enumerate() {
        let params = PresetParams {
            seed: 8,
            layers: vec![
                LayerParams {
                    color: *color,
                    alpha: -4.0 + i as f64 * 1.6,
                    gain_db: -2.0,
                    ..LayerParams::default()
                },
                LayerParams {
                    name: "b".into(),
                    color: colours[(i + 1) % colours.len()],
                    alpha: 4.0 - i as f64 * 1.6,
                    stereo_correlation: 0.0,
                    ..LayerParams::default()
                },
            ],
            ..PresetParams::default()
        };
        let mut e = configured_engine(&params, fs);
        e.apply(Command::Play);
        batches.push(compile(&params, fs, 2).commands(&params));
    }

    let params = PresetParams {
        seed: 8,
        layers: vec![LayerParams::default()],
        ..PresetParams::default()
    };
    let mut e = configured_engine(&params, fs);
    e.apply(Command::Play);

    let mut sink = vec![0.0f32; MAX_BLOCK * 2];
    let mut peak = 0.0f32;
    for block in 0..1200usize {
        let batch = &batches[block % batches.len()];
        for c in batch {
            e.apply(*c);
        }
        let frames = [32usize, 64, 128, 512, 1024][block % 5];
        e.render(&mut sink, frames, 2);
        for s in &sink[..frames * 2] {
            assert!(s.is_finite(), "block {block} produced {s}");
            peak = peak.max(s.abs());
        }
    }
    assert_eq!(e.meters().nan_events(), 0);
    assert!(peak < 4.0, "peak {peak} suggests runaway gain");
}

#[test]
fn a_filter_swap_without_recalibration_is_reported_not_hidden() {
    // DESIGN.md section 3.3 forbids hidden limiting. If a caller changes the
    // shaping filter without updating the calibration, the engine must reproduce
    // the resulting level faithfully and leave it to the peak monitor to report
    // it, rather than silently squashing it.
    let fs = 48_000.0;
    let params = preset_with(vec![LayerParams::default()]);
    let mut e = configured_engine(&params, fs);
    e.apply(Command::Play);
    for _ in 0..40 {
        e.render_block(256);
    }
    let before = e.meters().peak();

    // `alpha = -4` has over 130 dB of ultrasonic gain; without a matching
    // master gain the output is far hotter.
    let hot = noise_dsp::designer::design_color(-4.0, fs).unwrap();
    e.apply(Command::SetLayerFilters {
        layer: 0,
        filters: LayerFilters {
            color: hot.coeffs,
            high_pass: noise_dsp::designer::design_high_pass(35.0, 0.707, fs),
            low_pass: noise_dsp::designer::design_low_pass(16_000.0, 0.707, fs),
            expected_rms: 0.0,
        },
        crossfade_ms: 20.0,
    });
    let mut peak = 0.0f32;
    for _ in 0..200 {
        for s in e.render_block(256) {
            peak = peak.max(s.abs());
        }
    }
    assert!(
        peak > before as f32 * 4.0,
        "the hotter filter had no effect"
    );
    assert!(peak.is_finite());
    assert_eq!(e.meters().nan_events(), 0);
}

#[test]
fn out_of_range_commands_are_ignored_not_fatal() {
    let params = preset_with(vec![LayerParams::default()]);
    let fs = 48_000.0;
    let mut e = configured_engine(&params, fs);
    e.apply(Command::Play);
    // Layer index past the end.
    e.apply(Command::SetLayerGain {
        layer: 99,
        gain_db: 0.0,
        ramp_ms: 1.0,
    });
    // Non-finite values.
    e.apply(Command::SetLayerGain {
        layer: 0,
        gain_db: f32::NAN,
        ramp_ms: 1.0,
    });
    e.apply(Command::SetMasterGain {
        gain_db: f32::INFINITY,
        ramp_ms: 1.0,
        hold_ms: 0.0,
    });
    e.apply(Command::SetStereoCorrelation {
        layer: 0,
        correlation: f32::NAN,
        ramp_ms: 1.0,
    });
    // An unstable cascade must be rejected.
    e.apply(Command::SetLayerFilters {
        layer: 0,
        filters: LayerFilters {
            color: noise_dsp::SosCoeffs::new(&[noise_dsp::BiquadCoeffs {
                b0: 1.0,
                b1: 0.0,
                b2: 0.0,
                a1: 0.0,
                a2: -1.5,
            }]),
            ..LayerFilters::default()
        },
        crossfade_ms: 1.0,
    });
    let block = e.render_block(512);
    assert!(block.iter().all(|s| s.is_finite()));
    assert!(block.iter().any(|s| *s != 0.0));
    assert_eq!(e.meters().nan_events(), 0);
}

#[test]
fn a_long_run_does_not_drift_or_blow_up() {
    // Proportional stand-in for the eight hour soak in DESIGN.md section 9.2:
    // ten minutes of audio through the full eight layer graph, checking level
    // stability in windows spread across the run.
    let params = preset_with(
        (0..8)
            .map(|i| LayerParams {
                name: format!("L{i}"),
                color: noise_dsp::Color::ALL[i % 6],
                alpha: 1.0 + (i as f64) * 0.3,
                gain_db: -4.0,
                character: noise_dsp::Character::ALL[i % 3],
                character_amount: 0.5,
                ..LayerParams::default()
            })
            .collect(),
    );
    let fs = 48_000.0;
    let mut e = configured_engine(&params, fs);
    e.apply(Command::Play);

    let total_seconds = 600.0;
    let block = 1024usize;
    let blocks = (total_seconds * fs / block as f64) as usize;
    let mut sink = vec![0.0f32; block * 2];
    let mut windows: Vec<f64> = Vec::new();
    let mut acc = 0.0f64;
    let mut n = 0u64;
    for b in 0..blocks {
        e.render(&mut sink, block, 2);
        for s in &sink {
            let v = *s as f64;
            acc += v * v;
            n += 1;
        }
        if b % 2000 == 1999 {
            windows.push((acc / n as f64).sqrt());
            acc = 0.0;
            n = 0;
        }
    }
    windows.push((acc / n.max(1) as f64).sqrt());
    assert!(windows.iter().all(|v| v.is_finite() && *v > 0.0));
    let first_half: f64 =
        windows[..windows.len() / 2].iter().sum::<f64>() / (windows.len() / 2) as f64;
    let second_half: f64 = windows[windows.len() / 2..].iter().sum::<f64>()
        / (windows.len() - windows.len() / 2) as f64;
    let drift_db = 20.0 * (second_half / first_half).log10();
    assert!(
        drift_db.abs() < 1.0,
        "level drifted {drift_db:.2} dB over {total_seconds} s \
         (first {first_half:.6}, second {second_half:.6})"
    );
}

#[test]
fn the_sleep_timer_fades_out_and_pauses() {
    let params = preset_with(vec![LayerParams::default()]);
    let fs = 48_000.0;
    let mut e = configured_engine(&params, fs);
    e.apply(Command::Play);
    e.apply(Command::SetTimer {
        seconds: 0.25,
        fade_out_seconds: 0.1,
    });
    let mut out = Vec::new();
    for _ in 0..40 {
        out.extend_from_slice(e.render_block(512));
    }
    assert_eq!(e.play_state(), PlayState::Paused, "timer did not pause");
    let tail = &out[out.len() - 512..];
    assert!(
        tail.iter().all(|s| s.abs() < 1e-6),
        "output did not fade out"
    );

    // Clearing the timer keeps it playing.
    e.apply(Command::Play);
    e.apply(Command::SetTimer {
        seconds: 100.0,
        fade_out_seconds: 1.0,
    });
    e.apply(Command::ClearTimer);
    for _ in 0..40 {
        e.render_block(512);
    }
    assert_eq!(e.play_state(), PlayState::Playing);
}

#[test]
fn pause_and_resume_are_click_free_and_resume_the_same_stream() {
    let params = preset_with(vec![LayerParams::default()]);
    let fs = 48_000.0;
    let mut e = configured_engine(&params, fs);
    e.apply(Command::Play);
    let mut out = Vec::new();
    for _ in 0..20 {
        out.extend_from_slice(e.render_block(256));
    }
    e.apply(Command::Pause);
    for _ in 0..40 {
        out.extend_from_slice(e.render_block(256));
    }
    let paused = out[out.len() - 256..].to_vec();
    assert!(paused.iter().all(|s| *s == 0.0));
    e.apply(Command::Play);
    for _ in 0..40 {
        out.extend_from_slice(e.render_block(256));
    }
    // The resumed audio must be alive again and still click free.
    assert!(out[out.len() - 256..].iter().any(|s| *s != 0.0));
    let jump = max_energy_jump(&out);
    assert!(jump < 1.6, "resume jumped {jump}x");
}
