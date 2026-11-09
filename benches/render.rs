//! Render benchmark: the evidence behind DESIGN.md section 9.2.
//!
//! The design target is "under 5% of one i5-8400 core for 48 kHz stereo, one
//! layer, no FFT display". This measures it instead of claiming it.
//!
//! ```sh
//! cargo bench -p noiseforge
//! ```

use std::hint::black_box;
use std::time::Instant;

use noise_dsp::compile::compile;
use noise_dsp::engine::{Command, Engine};
use noise_dsp::params::{LayerParams, MasterParams, PresetParams};

/// Nanoseconds in one second of one CPU core.
const ONE_CORE_NS_PER_SECOND: f64 = 1.0e9;

struct Case {
    name: &'static str,
    params: PresetParams,
    single_layer: bool,
}

fn base_layer(
    color: noise_dsp::Color,
    character: noise_dsp::Character,
    amount: f64,
) -> LayerParams {
    LayerParams {
        color,
        character,
        character_amount: amount,
        ..LayerParams::default()
    }
}

fn cases() -> Vec<Case> {
    let mut out = Vec::new();
    let mut one = |name: &'static str, params: PresetParams| {
        out.push(Case {
            name,
            params,
            single_layer: true,
        })
    };
    one(
        "1 layer, White, Neutral",
        PresetParams {
            layers: vec![base_layer(
                noise_dsp::Color::White,
                noise_dsp::Character::Neutral,
                0.0,
            )],
            ..PresetParams::default()
        },
    );
    one(
        "1 layer, Pink, Neutral",
        PresetParams {
            layers: vec![base_layer(
                noise_dsp::Color::Pink,
                noise_dsp::Character::Neutral,
                0.0,
            )],
            ..PresetParams::default()
        },
    );
    one(
        "1 layer, Pink, Smooth 0.7",
        PresetParams {
            layers: vec![base_layer(
                noise_dsp::Color::Pink,
                noise_dsp::Character::Smooth,
                0.7,
            )],
            ..PresetParams::default()
        },
    );
    one(
        "1 layer, Pink, Intense 1.0",
        PresetParams {
            layers: vec![base_layer(
                noise_dsp::Color::Pink,
                noise_dsp::Character::Intense,
                1.0,
            )],
            ..PresetParams::default()
        },
    );
    one(
        "1 layer, Brown (7 sos)",
        PresetParams {
            layers: vec![base_layer(
                noise_dsp::Color::Brown,
                noise_dsp::Character::Neutral,
                0.0,
            )],
            ..PresetParams::default()
        },
    );
    one(
        "1 layer, limiter explicitly on",
        PresetParams {
            layers: vec![base_layer(
                noise_dsp::Color::Pink,
                noise_dsp::Character::Neutral,
                0.0,
            )],
            master: MasterParams {
                limiter_enabled: true,
                ..Default::default()
            },
            ..PresetParams::default()
        },
    );
    out.push(Case {
        name: "8 layers, mixed colours",
        params: PresetParams {
            seed: 5,
            layers: (0..8)
                .map(|i| LayerParams {
                    name: format!("L{i}"),
                    color: noise_dsp::Color::ALL[i % 6],
                    alpha: 1.0 + i as f64 * 0.25,
                    character: noise_dsp::Character::ALL[i % 3],
                    character_amount: 0.6,
                    gain_db: -3.0,
                    stereo_correlation: 0.3,
                    ..LayerParams::default()
                })
                .collect(),
            ..PresetParams::default()
        },
        single_layer: false,
    });
    out.push(Case {
        name: "8 layers, all Intense 1.0",
        params: PresetParams {
            seed: 6,
            layers: (0..8)
                .map(|i| LayerParams {
                    name: format!("L{i}"),
                    color: noise_dsp::Color::ALL[i % 6],
                    character: noise_dsp::Character::Intense,
                    character_amount: 1.0,
                    stereo_correlation: 0.0,
                    ..LayerParams::default()
                })
                .collect(),
            ..PresetParams::default()
        },
        single_layer: false,
    });
    out
}

fn measure(case: &Case, sample_rate: f64, seconds: f64, block: usize) -> f64 {
    let graph = compile(&case.params, sample_rate, 2);
    let mut engine = Engine::new(sample_rate, case.params.seed);
    for c in graph.commands(&case.params) {
        engine.apply(c);
    }
    engine.apply(Command::Play);

    let mut sink = vec![0.0f32; block * 2];
    // Warm up: settle the fades and let the CPU ramp its clocks.
    let mut warm = 0usize;
    while warm < sample_rate as usize {
        engine.render(&mut sink, block, 2);
        warm += block;
    }

    let iterations = ((seconds * sample_rate) as usize / block).max(1);
    let start = Instant::now();
    for _ in 0..iterations {
        engine.render(black_box(&mut sink), block, 2);
    }
    let elapsed = start.elapsed().as_secs_f64();
    black_box(sink[0]);
    (iterations * block) as f64 / elapsed
}

fn main() {
    let sample_rate = 48_000.0;
    let seconds = 4.0;
    println!("NoiseForge render benchmark");
    println!(
        "{sample_rate:.0} Hz stereo, {:.0} s per case, filter state in f64",
        seconds
    );
    println!();
    println!(
        "{:<32} {:>12} {:>11} {:>16}",
        "configuration", "frames/s", "ns/frame", "CPU % of 1 core"
    );
    println!("{}", "-".repeat(74));

    let mut worst_single = 0.0f64;
    let mut worst_all = 0.0f64;
    for case in cases() {
        let fps = measure(&case, sample_rate, seconds, 1024);
        let ns_per_frame = 1.0e9 / fps;
        // One second of audio needs `sample_rate` frames, so the fraction of a
        // core is that many frames' worth of CPU over one second.
        let cpu_percent = ns_per_frame * sample_rate / ONE_CORE_NS_PER_SECOND * 100.0;
        worst_all = worst_all.max(cpu_percent);
        if case.single_layer {
            worst_single = worst_single.max(cpu_percent);
        }
        println!(
            "{:<32} {:>12.0} {:>11.1} {:>15.3} %",
            case.name, fps, ns_per_frame, cpu_percent
        );
    }
    println!("{}", "-".repeat(74));
    println!("DESIGN.md section 9.2 target: single layer under 5.000 % of one i5-8400 core.");
    println!(
        "worst single layer case measured here: {worst_single:.3} % -> {}",
        if worst_single < 5.0 { "MET" } else { "NOT MET" }
    );
    println!("worst case overall (8 layers): {worst_all:.3} %");

    // Quantisation sweep: the realtime deadline must hold at every host period.
    println!();
    println!("Period size sweep (single Pink layer):");
    println!(
        "{:>10} {:>11} {:>14} {:>16}",
        "frames", "ns/frame", "period us", "deadline used"
    );
    let case = &cases()[1];
    for block in [64usize, 128, 256, 512, 1024, 2048] {
        let fps = measure(case, sample_rate, 2.0, block);
        let ns_per_frame = 1.0e9 / fps;
        let period_us = block as f64 / sample_rate * 1.0e6;
        let used = ns_per_frame * block as f64 / (period_us * 1000.0) * 100.0;
        println!("{block:>10} {ns_per_frame:>11.1} {period_us:>14.1} {used:>15.2} %");
    }

    println!();
    println!("Notes:");
    println!("  * no FFT runs anywhere in the render path (DESIGN.md section 4.5);");
    println!("  * the limiter is off in every case except the one that names it;");
    println!("  * the realtime allocation count is asserted to be zero by");
    println!("    `tests/rt_alloc.rs`, not measured here.");
}
