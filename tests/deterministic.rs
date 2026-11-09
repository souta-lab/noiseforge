//! Determinism tests: DESIGN.md sections 1.3, 4.1 and 9.2.
//!
//! * The same DSP version, settings, sample rate and seed reproduce the same PCM.
//! * Changing the host buffer size does not change the sample sequence.
//! * Different layers and different seeds are independent.
//! * A preset that is written and read back renders identically.

use noise_dsp::compile::compile;
use noise_dsp::engine::{Command, Engine};
use noise_dsp::params::{LayerParams, MasterParams, PresetParams};
use noise_offline::render::{RenderRequest, render};

fn preset(seed: u64) -> PresetParams {
    PresetParams {
        seed,
        layers: vec![
            LayerParams {
                color: noise_dsp::Color::Pink,
                character: noise_dsp::Character::Smooth,
                character_amount: 0.7,
                seed: None,
                ..LayerParams::default()
            },
            LayerParams {
                name: "Second".into(),
                color: noise_dsp::Color::Brown,
                gain_db: -6.0,
                stereo_correlation: 0.0,
                seed: None,
                ..LayerParams::default()
            },
        ],
        ..PresetParams::default()
    }
}

fn render_with_block(frames: usize, blocks: usize, seed: u64) -> Vec<f32> {
    let params = preset(seed);
    let graph = compile(&params, 48_000.0, 2);
    let mut e = Engine::new(48_000.0, params.seed);
    for c in graph.commands(&params) {
        e.apply(c);
    }
    e.apply(Command::Play);
    let mut out = Vec::with_capacity(frames * blocks * 2);
    for _ in 0..blocks {
        out.extend_from_slice(e.render_block(frames));
    }
    out
}

#[test]
fn the_same_seed_reproduces_the_same_pcm() {
    let a = render_with_block(256, 40, 1234);
    let b = render_with_block(256, 40, 1234);
    assert!(a == b, "the same seed did not reproduce the same PCM");
    assert!(a.iter().any(|s| *s != 0.0));
}

#[test]
fn a_different_seed_gives_different_pcm() {
    let a = render_with_block(256, 20, 1);
    let b = render_with_block(256, 20, 2);
    assert!(
        a != b,
        "seeds 1 and 2 produced the same PCM (first sample {})",
        a[0]
    );
}

#[test]
fn block_size_does_not_change_the_sample_sequence() {
    // 128, 256 and 512 frame quanta must all produce identical PCM. This is the
    // property that lets the PipeWire buffer size change without disturbing the
    // signal (DESIGN.md section 1.3).
    let reference = render_with_block(128, 80, 7);
    for frames in [64usize, 256, 512, 1024] {
        let blocks = reference.len() / 2 / frames;
        let other = render_with_block(frames, blocks, 7);
        assert_eq!(
            other.len(),
            blocks * frames * 2,
            "length mismatch at {frames} frames"
        );
        let mut worst = 0.0f32;
        for (x, y) in reference.iter().zip(other.iter()) {
            worst = worst.max((x - y).abs());
        }
        assert_eq!(worst, 0.0, "block size {frames} changed the PCM by {worst}");
    }
}

#[test]
fn the_offline_renderer_agrees_with_the_engine() {
    // `noiseforge render` must produce exactly what the daemon plays.
    let params = preset(99);
    let req = RenderRequest {
        params: params.clone(),
        sample_rate: 48_000.0,
        seconds: 0.5,
        warmup_seconds: 0.0,
        channels: 2,
        seed: None,
        auto_play: true,
    };
    let offline = render(&req);
    // Render the same number of frames through the engine in 256 frame blocks
    // plus the remainder, which is what a host would do.
    let total = offline.frames;
    let params = preset(99);
    let graph = compile(&params, 48_000.0, 2);
    let mut e = Engine::new(48_000.0, params.seed);
    for c in graph.commands(&params) {
        e.apply(c);
    }
    e.apply(Command::Play);
    let mut engine = Vec::with_capacity(total * 2);
    let mut left = total;
    while left > 0 {
        let n = left.min(256);
        engine.extend_from_slice(e.render_block(n));
        left -= n;
    }
    assert_eq!(offline.samples.len(), engine.len());
    assert_eq!(offline.samples, engine);
}

#[test]
fn layers_with_the_same_settings_but_different_seeds_are_uncorrelated() {
    let p = PresetParams {
        seed: 5,
        master: MasterParams {
            target_rms_dbfs: -30.0,
            ..MasterParams::default()
        },
        layers: vec![LayerParams {
            color: noise_dsp::Color::White,
            high_pass_hz: 0.0,
            low_pass_hz: 0.0,
            seed: Some(1),
            ..LayerParams::default()
        }],
        ..PresetParams::default()
    };
    let q = PresetParams {
        layers: vec![
            p.layers[0].clone(),
            LayerParams {
                name: "b".into(),
                seed: Some(2),
                ..p.layers[0].clone()
            },
        ],
        ..p.clone()
    };
    // Two independent layers add 3 dB of power before the calibration, which
    // the calibration then removes; check that the sum really did add power by
    // comparing the raw mix levels before calibration.
    let g_one = compile(&p, 48_000.0, 2);
    let g_two = compile(&q, 48_000.0, 2);
    let delta = g_two.calibration.mix_rms_dbfs - g_one.calibration.mix_rms_dbfs;
    assert!(
        (delta - 3.0103).abs() < 0.05,
        "two independent layers added {delta:.3} dB instead of 3.010"
    );

    // And the rendered output is genuinely decorrelated between the two layers
    // when they are rendered separately.
    let a = render(&RenderRequest {
        params: p,
        seconds: 1.0,
        ..RenderRequest::default()
    });
    // Render the *second* layer on its own: same settings, different seed.
    let solo_second = PresetParams {
        layers: vec![q.layers[1].clone()],
        ..q.clone()
    };
    let b = render(&RenderRequest {
        params: solo_second,
        seconds: 1.0,
        ..RenderRequest::default()
    });
    let x = a.channel(0);
    let y = b.channel(0);
    let n = x.len().min(y.len()) as f64;
    let (mut sxy, mut sxx, mut syy) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..x.len().min(y.len()) {
        let (a, b) = (x[i] as f64, y[i] as f64);
        sxy += a * b;
        sxx += a * a;
        syy += b * b;
    }
    let corr = sxy / (sxx.sqrt() * syy.sqrt());
    assert!(
        corr.abs() < 0.05,
        "layer seed correlation {corr} over {n} frames"
    );
}

#[test]
fn a_toml_round_trip_renders_identically() {
    let params = preset(2026);
    let text = toml::to_string_pretty(&params).expect("serialise");
    let parsed: PresetParams = toml::from_str(&text).expect("deserialise");
    assert_eq!(params.layers.len(), parsed.layers.len());
    let a = render(&RenderRequest {
        params,
        seconds: 0.25,
        warmup_seconds: 0.0,
        ..RenderRequest::default()
    });
    let b = render(&RenderRequest {
        params: parsed,
        seconds: 0.25,
        warmup_seconds: 0.0,
        ..RenderRequest::default()
    });
    assert_eq!(a.samples, b.samples);
}

#[test]
fn the_first_play_starts_from_silence() {
    // Truncation, not assertion failure: the first sample after Play must be
    // tiny because the fade-in starts at zero.
    let params = preset(3);
    let graph = compile(&params, 48_000.0, 2);
    let mut e = Engine::new(48_000.0, params.seed);
    for c in graph.commands(&params) {
        e.apply(c);
    }
    assert!(e.render_block(64).iter().all(|s| *s == 0.0));
    e.apply(Command::Play);
    let first = e.render_block(64);
    assert!(first[0].abs() < 1e-6, "first sample {}", first[0]);
}
