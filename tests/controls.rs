//! Control surface tests: presets on disk, parameter patching, and the guarantees
//! the CLI and GUI rely on (DESIGN.md sections 6, 7 and 9.3).

use std::path::{Path, PathBuf};

use noise_daemon::protocol::{SetArgs, apply_set, preset_file_name};
use noise_dsp::compile::compile;
use noise_dsp::params::{LayerParams, PresetParams};

fn preset_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("presets")
}

fn shipped_presets() -> Vec<(String, PresetParams)> {
    let mut out = Vec::new();
    let mut entries: Vec<_> = std::fs::read_dir(preset_dir())
        .expect("presets/ must exist")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("toml"))
        .collect();
    entries.sort();
    assert!(!entries.is_empty(), "no presets shipped");
    for path in entries {
        let text = std::fs::read_to_string(&path).unwrap();
        let preset: PresetParams =
            toml::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        out.push((path.display().to_string(), preset));
    }
    out
}

#[test]
fn every_shipped_preset_parses_and_compiles_at_every_supported_rate() {
    for (name, preset) in shipped_presets() {
        for fs in [44_100.0, 48_000.0, 96_000.0] {
            let graph = compile(&preset, fs, 2);
            assert_eq!(
                graph.layers.len(),
                preset.layers.len(),
                "{name} @ {fs}: layer count changed"
            );
            for layer in &graph.layers {
                assert!(
                    layer.verified(),
                    "{name} @ {fs}: layer {} is out of tolerance: {:?}",
                    layer.index,
                    layer.design
                );
                assert!(layer.filters.is_valid(), "{name} @ {fs}: unstable cascade");
            }
            assert!(
                graph.calibration.master_gain_db.is_finite(),
                "{name} @ {fs}: calibration is not finite"
            );
            let achieved = graph.calibration.mix_rms_dbfs + graph.calibration.master_gain_db;
            assert!(
                (achieved - graph.calibration.target_rms_dbfs).abs() < 1e-6,
                "{name} @ {fs}: calibration misses the target by {} dB",
                achieved - graph.calibration.target_rms_dbfs
            );
            assert!(
                !graph.calibration.headroom_warning,
                "{name} @ {fs}: headroom warning for a shipped preset"
            );
        }
    }
}

#[test]
fn shipped_presets_have_distinct_layer_seeds() {
    for (name, mut preset) in shipped_presets() {
        let notes = preset.sanitize();
        assert!(
            !notes.iter().any(|n| n.contains("shares a seed")),
            "{name}: {notes:?}"
        );
    }
}

#[test]
fn a_shipped_preset_survives_a_round_trip_through_the_store() {
    let dir = tempfile::tempdir().unwrap();
    let store = noise_daemon::presets::Store::at(dir.path().join("noiseforge"));
    for (name, preset) in shipped_presets() {
        let label = preset.name.clone();
        let path = store.save_preset(&label, &preset).unwrap();
        assert!(path.exists(), "{name}");
        let back = store.load_preset(&label).unwrap();
        assert_eq!(back.layers.len(), preset.layers.len(), "{name}");
        for (a, b) in back.layers.iter().zip(preset.layers.iter()) {
            assert_eq!(a.color, b.color, "{name}");
            assert_eq!(a.alpha, b.alpha, "{name}");
            assert_eq!(a.character, b.character, "{name}");
        }
        assert_eq!(back.master, preset.master, "{name}");
    }
}

#[test]
fn preset_file_names_never_escape_the_directory() {
    let dir = tempfile::tempdir().unwrap();
    let store = noise_daemon::presets::Store::at(dir.path().join("noiseforge"));
    for hostile in [
        "../../etc/passwd",
        "/etc/passwd",
        "..",
        ".",
        "a/b/c",
        "\u{0}null",
        "very long name that goes on and on and on and on",
    ] {
        let path = store.save_preset(hostile, &PresetParams::default());
        let path = match path {
            Ok(p) => p,
            Err(_) => continue, // refusing outright is also acceptable
        };
        assert!(
            path.starts_with(store.preset_dir()),
            "{hostile:?} escaped to {}",
            path.display()
        );
        assert_eq!(path.parent(), Some(store.preset_dir().as_path()));
        assert!(!path.to_string_lossy().contains(".."));
        assert_eq!(preset_file_name(hostile).matches('/').count(), 0);
    }
}

#[test]
fn patching_never_leaves_a_preset_unplayable() {
    // Hammer the patcher with every field at both extremes. Sanitising must
    // always produce something that compiles.
    let extremes = [
        SetArgs {
            alpha: Some(f64::MAX),
            intensity: Some(f64::MAX),
            stereo: Some(f64::MAX),
            level_rms_dbfs: Some(f64::MAX),
            gain_db: Some(f64::MAX),
            high_pass: Some(f64::MAX),
            low_pass: Some(f64::MAX),
            ..Default::default()
        },
        SetArgs {
            alpha: Some(f64::MIN),
            intensity: Some(f64::MIN),
            stereo: Some(f64::MIN),
            level_rms_dbfs: Some(f64::MIN),
            gain_db: Some(f64::MIN),
            high_pass: Some(f64::MIN),
            low_pass: Some(f64::MIN),
            ..Default::default()
        },
        SetArgs {
            alpha: Some(f64::NAN),
            intensity: Some(f64::NAN),
            stereo: Some(f64::NAN),
            level_rms_dbfs: Some(f64::NAN),
            gain_db: Some(f64::NAN),
            ..Default::default()
        },
    ];
    for (i, args) in extremes.into_iter().enumerate() {
        let mut preset = PresetParams::default();
        apply_set(&mut preset, &args);
        let graph = compile(&preset, 48_000.0, 2);
        assert!(
            graph.calibration.master_gain_db.is_finite(),
            "case {i}: calibration is not finite"
        );
        assert!(graph.layers[0].filters.is_valid(), "case {i}: unstable");
        assert!(preset.layers[0].alpha.is_finite());
    }
}

#[test]
fn every_colour_builds_a_usable_preset() {
    for color in noise_dsp::Color::ALL {
        for character in noise_dsp::Character::ALL {
            let preset = PresetParams {
                layers: vec![LayerParams {
                    color,
                    character,
                    character_amount: 1.0,
                    ..LayerParams::default()
                }],
                ..PresetParams::default()
            };
            let graph = compile(&preset, 48_000.0, 2);
            assert!(graph.layers[0].verified(), "{color}/{character}");
            assert!(!graph.calibration.headroom_warning, "{color}/{character}");
        }
    }
}

#[test]
fn alpha_outside_the_supported_range_is_clamped_and_named() {
    let mut preset = PresetParams {
        layers: vec![LayerParams {
            color: noise_dsp::Color::Custom,
            alpha: 10.0,
            ..LayerParams::default()
        }],
        ..PresetParams::default()
    };
    let notes = preset.sanitize();
    assert_eq!(preset.layers[0].alpha, 4.0);
    assert!(notes.iter().any(|n| n.contains("alpha")));
}

#[test]
fn disabling_every_layer_is_survivable() {
    let mut preset = PresetParams {
        layers: vec![LayerParams {
            enabled: false,
            ..LayerParams::default()
        }],
        ..PresetParams::default()
    };
    preset.sanitize();
    let graph = compile(&preset, 48_000.0, 2);
    assert_eq!(graph.calibration.active_layers, 0);
    // With nothing playing the calibration must not produce a divide by zero.
    assert!(graph.calibration.master_gain_db.is_finite());
    assert!(graph.calibration.master_gain_db <= -100.0);
}

#[test]
fn the_full_state_round_trips_through_json() {
    // The GUI and the CLI both serialise StatusData; make sure the shape is
    // stable and complete enough for the main view.
    use noise_daemon::protocol::*;
    let status = StatusData {
        version: "0.1.0".into(),
        protocol: PROTOCOL_VERSION,
        dsp_version: noise_dsp::DSP_VERSION,
        backend: BackendKind::PipeWire,
        playing: true,
        transport: "playing".into(),
        link: "running".into(),
        sample_rate: 48_000,
        channels: 2,
        rate_mode: RateMode::Follow,
        device: None,
        device_node_id: None,
        timer_remaining_s: None,
        uptime_s: 1.0,
        calibration: CalibrationStatus {
            target_rms_dbfs: -36.0,
            mix_rms_dbfs: -33.0,
            calibration_gain_db: -3.0,
            master_gain_db: -3.0,
            predicted_peak_dbfs: -23.5,
            headroom_warning: false,
        },
        meters: MeterStatus::default(),
        layers: Vec::new(),
        master: noise_dsp::params::MasterParams::default(),
        notes: Vec::new(),
        preset_name: "x".into(),
        seed: 1,
        auto_start: false,
    };
    let json = serde_json::to_string(&status).unwrap();
    let back: StatusData = serde_json::from_str(&json).unwrap();
    assert_eq!(back.sample_rate, 48_000);
    assert!(back.playing);
}
