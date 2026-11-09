//! Live smoke test: play a few seconds of pink noise through PipeWire and report
//! the negotiated format and the realtime telemetry.
//!
//! ```sh
//! cargo run -p noise-pipewire --release --example live_smoke -- 3
//! ```

use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use noise_dsp::compile::compile;
use noise_dsp::engine::Command;
use noise_dsp::params::{LayerParams, PresetParams};
use noise_pipewire::{RatePolicy, StreamConfig, StreamEvent, spawn};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber_init();
    let seconds: f64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(3.0);

    let params = PresetParams {
        layers: vec![LayerParams {
            color: noise_dsp::Color::Pink,
            character: noise_dsp::Character::Smooth,
            character_amount: 0.7,
            ..LayerParams::default()
        }],
        ..PresetParams::default()
    };

    let config = StreamConfig {
        rate: RatePolicy::Follow,
        seed: params.seed,
        tap_frames: 48_000,
        ..StreamConfig::default()
    };
    let mut out = spawn(config)?;
    let telemetry = out.telemetry.clone();

    // Wait for the negotiated format, then compile for that rate.
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut format = None;
    while Instant::now() < deadline {
        match out.events.recv_timeout(Duration::from_millis(500)) {
            Ok(StreamEvent::Format(f)) => {
                format = Some(f);
                break;
            }
            Ok(StreamEvent::State { old, new, link }) => {
                println!("state: {old} -> {new} ({})", link.label());
            }
            Ok(StreamEvent::Error(e)) => println!("error: {e}"),
            Ok(StreamEvent::Closed) => break,
            Err(_) => {}
        }
    }
    let Some(format) = format else {
        return Err("no format negotiated".into());
    };
    println!(
        "negotiated: {} Hz, {} channel(s), {:?}, planar={}",
        format.rate, format.channels, format.format, format.planar
    );

    let graph = compile(&params, format.rate as f64, format.channels as usize);
    for c in graph.commands(&params) {
        out.commands.push(c).map_err(|_| "command ring full")?;
    }
    out.commands
        .push(Command::Play)
        .map_err(|_| "command ring full")?;
    println!(
        "calibration: mix {:.2} dBFS, master {:.2} dB, peak {:.2} dBFS",
        graph.calibration.mix_rms_dbfs,
        graph.calibration.master_gain_db,
        graph.calibration.predicted_peak_dbfs
    );
    for note in &graph.notes {
        println!("note: {note}");
    }

    // Let it settle, then measure.
    std::thread::sleep(Duration::from_millis(500));
    let rms_before = telemetry.rms_l();
    std::thread::sleep(Duration::from_secs_f64(seconds));
    let rms_after = telemetry.rms_l();

    let callbacks = telemetry.callbacks.load(Ordering::Relaxed);
    let frames = telemetry.frames.load(Ordering::Relaxed);
    let xruns = telemetry.xruns.load(Ordering::Relaxed);
    let nan = telemetry.nan_events.load(Ordering::Relaxed);
    let dropped = telemetry.tap_dropped.load(Ordering::Relaxed);
    let rate = telemetry.sample_rate.load(Ordering::Relaxed);
    let peak = telemetry.peak_l();

    println!("callbacks: {callbacks}, frames: {frames}, xruns: {xruns}, nan: {nan}");
    println!("tap dropped: {dropped}");
    println!(
        "link: {}, rate {rate} Hz, peak {:.2} dBFS, rms {:.2} dBFS",
        telemetry.link().label(),
        noise_pipewire::lifecycle::dbfs(peak),
        noise_pipewire::lifecycle::dbfs(rms_after)
    );
    if let Some(p9999) = telemetry.callback_high_percentile_ns(0.9999) {
        let period_ns = 128.0 / rate as f64 * 1e9;
        println!(
            "p99.99 callback <= {:.1} us ({} % of a 128 frame period of {:.1} us)",
            p9999 as f64 / 1000.0,
            p9999 as f64 / period_ns * 100.0,
            period_ns / 1000.0
        );
    }
    let _ = (rms_before, rms_after);

    out.shutdown();
    Ok(())
}

fn tracing_subscriber_init() {}
