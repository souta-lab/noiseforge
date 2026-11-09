//! The serialisable configuration model shared by the daemon, CLI, GUI and the
//! offline renderer.
//!
//! DESIGN.md section 7: TOML plus Serde, versioned schema, atomic saves. These
//! types are *not* used on the realtime thread; the realtime engine receives
//! pre-designed coefficients through [`crate::engine::Command`].

use serde::{Deserialize, Serialize};

use crate::characters::Character;
use crate::colors::{ALPHA_MAX, ALPHA_MIN, Color};
use crate::rng::derive_seed;

/// Preset schema version. Bump on incompatible changes and migrate explicitly.
pub const SCHEMA_VERSION: u32 = 1;

/// Audio graph version. Bump if the stage order changes, because the order is
/// audible and is stored with every preset.
pub const GRAPH_VERSION: u32 = 1;

/// Hard limit on layers (DESIGN.md section 3.2).
pub const MAX_LAYERS: usize = 8;

/// Default playback level: `-36 dBFS` expected RMS (DESIGN.md section 3.3).
pub const DEFAULT_TARGET_RMS_DBFS: f64 = -36.0;

/// How the output sample rate is chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SampleRatePolicy {
    /// Follow whatever PipeWire negotiates.
    Named(NamedSampleRatePolicy),
    /// Force a specific rate.
    Fixed(u32),
}

/// Named sample rate policies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NamedSampleRatePolicy {
    FollowPipewire,
}

impl Default for SampleRatePolicy {
    fn default() -> Self {
        SampleRatePolicy::Named(NamedSampleRatePolicy::FollowPipewire)
    }
}

impl SampleRatePolicy {
    /// The rate to design for, or `None` to follow the server.
    pub fn forced_rate(self) -> Option<u32> {
        match self {
            SampleRatePolicy::Fixed(r) => Some(r),
            SampleRatePolicy::Named(_) => None,
        }
    }
}

/// One noise layer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LayerParams {
    /// Whether the layer contributes to the mix.
    pub enabled: bool,
    /// Display name.
    pub name: String,
    /// Generation source.
    pub color: Color,
    /// Exponent used when `color == custom`. Ignored for named colours.
    pub alpha: f64,
    /// Post colour character filter.
    pub character: Character,
    /// Character strength, `0.0..=1.0`.
    pub character_amount: f64,
    /// Second order high pass corner in Hz.
    pub high_pass_hz: f64,
    /// Second order low pass corner in Hz.
    pub low_pass_hz: f64,
    /// Stereo correlation coefficient, `-1.0..=1.0`.
    pub stereo_correlation: f64,
    /// Layer gain in dB applied after the filters.
    pub gain_db: f64,
    /// Explicit PRNG seed. `None` derives one from the preset seed and the layer
    /// index so that every layer is independent by default.
    pub seed: Option<u64>,
}

impl Default for LayerParams {
    fn default() -> Self {
        Self {
            enabled: true,
            name: "Main".to_string(),
            color: Color::Pink,
            alpha: 1.0,
            character: Character::Neutral,
            character_amount: 0.0,
            high_pass_hz: 35.0,
            low_pass_hz: 16000.0,
            stereo_correlation: 0.35,
            gain_db: 0.0,
            seed: None,
        }
    }
}

impl LayerParams {
    /// A layer with the given colour and a sensible name.
    pub fn with_color(color: Color, name: impl Into<String>) -> Self {
        Self {
            color,
            name: name.into(),
            ..Default::default()
        }
    }

    /// Effective `alpha`, honouring the named colours.
    pub fn alpha(&self) -> f64 {
        self.color.resolve_alpha(self.alpha)
    }

    /// Seed actually used by the engine for this layer.
    pub fn resolved_seed(&self, preset_seed: u64, index: usize) -> u64 {
        match self.seed {
            Some(s) => s,
            None => derive_seed(preset_seed, index, 0x5EED),
        }
    }

    /// Clamp every field into its supported range, returning the names of the
    /// fields that had to be changed.
    pub fn sanitize(&mut self) -> Vec<&'static str> {
        let mut fixed = Vec::new();
        if !self.alpha.is_finite() {
            self.alpha = 1.0;
            fixed.push("alpha");
        } else if !(ALPHA_MIN..=ALPHA_MAX).contains(&self.alpha) {
            self.alpha = self.alpha.clamp(ALPHA_MIN, ALPHA_MAX);
            fixed.push("alpha");
        }
        if !self.character_amount.is_finite() {
            self.character_amount = 0.0;
            fixed.push("character_amount");
        } else if !(0.0..=1.0).contains(&self.character_amount) {
            self.character_amount = self.character_amount.clamp(0.0, 1.0);
            fixed.push("character_amount");
        }
        if !self.stereo_correlation.is_finite() {
            self.stereo_correlation = 0.0;
            fixed.push("stereo_correlation");
        } else if !(-1.0..=1.0).contains(&self.stereo_correlation) {
            self.stereo_correlation = self.stereo_correlation.clamp(-1.0, 1.0);
            fixed.push("stereo_correlation");
        }
        if !self.high_pass_hz.is_finite() || self.high_pass_hz < 0.0 {
            self.high_pass_hz = 0.0;
            fixed.push("high_pass_hz");
        }
        if !self.low_pass_hz.is_finite() || self.low_pass_hz <= 0.0 {
            self.low_pass_hz = 20_000.0;
            fixed.push("low_pass_hz");
        }
        if !self.gain_db.is_finite() {
            self.gain_db = 0.0;
            fixed.push("gain_db");
        } else if !(-60.0..=24.0).contains(&self.gain_db) {
            self.gain_db = self.gain_db.clamp(-60.0, 24.0);
            fixed.push("gain_db");
        }
        fixed
    }
}

/// Master section.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MasterParams {
    /// Expected RMS of the finished PCM, in dBFS. This is a target for the
    /// *calibration*, not a per-sample tracker.
    pub target_rms_dbfs: f64,
    /// Extra trim on top of the calibration, in dB.
    pub volume_db: f64,
    /// Start muted.
    pub mute: bool,
    /// Start-up fade, milliseconds.
    pub fade_in_ms: f64,
    /// Shutdown fade, milliseconds.
    pub fade_out_ms: f64,
    /// Click-suppression ramp for parameter changes, milliseconds.
    pub ramp_ms: f64,
    /// Optional explicit limiter. Off by default: it is nonlinear and changes
    /// the PSD, so it is never enabled silently.
    pub limiter_enabled: bool,
    /// Limiter ceiling in dBFS.
    pub limiter_ceiling_dbfs: f64,
}

impl Default for MasterParams {
    fn default() -> Self {
        Self {
            target_rms_dbfs: DEFAULT_TARGET_RMS_DBFS,
            volume_db: 0.0,
            mute: false,
            fade_in_ms: 100.0,
            fade_out_ms: 100.0,
            ramp_ms: 20.0,
            limiter_enabled: false,
            limiter_ceiling_dbfs: -1.0,
        }
    }
}

impl MasterParams {
    /// Combined user gain in dB, ignoring the calibration.
    pub fn user_gain_db(&self) -> f64 {
        if self.mute { -120.0 } else { self.volume_db }
    }

    /// Clamp every field, returning the names of fields that had to change.
    pub fn sanitize(&mut self) -> Vec<&'static str> {
        let mut fixed = Vec::new();
        if !self.target_rms_dbfs.is_finite() || !(-80.0..=0.0).contains(&self.target_rms_dbfs) {
            self.target_rms_dbfs = self.target_rms_dbfs.clamp(-80.0, 0.0);
            if !self.target_rms_dbfs.is_finite() {
                self.target_rms_dbfs = DEFAULT_TARGET_RMS_DBFS;
            }
            fixed.push("target_rms_dbfs");
        }
        if !self.volume_db.is_finite() || !(-60.0..=24.0).contains(&self.volume_db) {
            self.volume_db = self.volume_db.clamp(-60.0, 24.0);
            if !self.volume_db.is_finite() {
                self.volume_db = 0.0;
            }
            fixed.push("volume_db");
        }
        for (name, v) in [
            ("fade_in_ms", &mut self.fade_in_ms),
            ("fade_out_ms", &mut self.fade_out_ms),
            ("ramp_ms", &mut self.ramp_ms),
        ] {
            if !v.is_finite() || !(0.0..=10_000.0).contains(v) {
                *v = v.clamp(0.0, 10_000.0);
                if !v.is_finite() {
                    *v = 20.0;
                }
                fixed.push(name);
            }
        }
        if !self.limiter_ceiling_dbfs.is_finite()
            || !(-24.0..=0.0).contains(&self.limiter_ceiling_dbfs)
        {
            self.limiter_ceiling_dbfs = self.limiter_ceiling_dbfs.clamp(-24.0, 0.0);
            if !self.limiter_ceiling_dbfs.is_finite() {
                self.limiter_ceiling_dbfs = -1.0;
            }
            fixed.push("limiter_ceiling_dbfs");
        }
        fixed
    }
}

/// A full preset / live configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PresetParams {
    /// Serialisation schema version.
    pub schema_version: u32,
    /// Audio graph version; the stage order is audible and must not change
    /// silently between releases.
    pub graph_version: u32,
    /// Human readable preset name.
    pub name: String,
    /// Output sample rate policy.
    pub sample_rate_policy: SampleRatePolicy,
    /// Base PRNG seed.
    pub seed: u64,
    /// Master section.
    pub master: MasterParams,
    /// Layers, at most [`MAX_LAYERS`].
    pub layers: Vec<LayerParams>,
}

impl Default for PresetParams {
    fn default() -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            graph_version: GRAPH_VERSION,
            name: "Default".to_string(),
            sample_rate_policy: SampleRatePolicy::default(),
            seed: 1,
            master: MasterParams::default(),
            layers: vec![LayerParams::default()],
        }
    }
}

impl PresetParams {
    /// A single Pink layer with a neutral character, the CLI/GUI default.
    pub fn pink() -> Self {
        Self::default()
    }

    /// Number of enabled layers.
    pub fn active_layers(&self) -> usize {
        self.layers.iter().filter(|l| l.enabled).count()
    }

    /// Clamp everything and report what changed. Never fails: a corrupt preset
    /// must not take the daemon down (DESIGN.md section 9.3).
    pub fn sanitize(&mut self) -> Vec<String> {
        let mut notes = Vec::new();
        if self.layers.len() > MAX_LAYERS {
            self.layers.truncate(MAX_LAYERS);
            notes.push("layers truncated to 8".to_string());
        }
        if self.layers.is_empty() {
            self.layers.push(LayerParams::default());
            notes.push("no layers: added a default layer".to_string());
        }
        for (i, l) in self.layers.iter_mut().enumerate() {
            for field in l.sanitize() {
                notes.push(format!("layer[{i}].{field} clamped"));
            }
        }
        for field in self.master.sanitize() {
            notes.push(format!("master.{field} clamped"));
        }
        // Two enabled layers sharing a seed are perfectly correlated, which
        // breaks the RMS calibration assumption. Report it instead of hiding it.
        let mut seen: Vec<u64> = Vec::new();
        for (i, l) in self.layers.iter().enumerate() {
            if !l.enabled {
                continue;
            }
            let s = l.resolved_seed(self.seed, i);
            if seen.contains(&s) {
                notes.push(format!("layer[{i}] shares a seed with an earlier layer"));
            }
            seen.push(s);
        }
        if self.schema_version != SCHEMA_VERSION {
            notes.push(format!(
                "schema_version {} -> {}",
                self.schema_version, SCHEMA_VERSION
            ));
            self.schema_version = SCHEMA_VERSION;
        }
        if self.graph_version != GRAPH_VERSION {
            notes.push(format!(
                "graph_version {} -> {}",
                self.graph_version, GRAPH_VERSION
            ));
            self.graph_version = GRAPH_VERSION;
        }
        notes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toml_round_trip_matches_design_example() {
        let toml_src = r#"
schema_version = 1
graph_version = 1
name = "Soft Pink"
sample_rate_policy = "follow_pipewire"
seed = 42

[master]
target_rms_dbfs = -36.0
mute = false
fade_in_ms = 100

[[layers]]
enabled = true
name = "Main"
color = "pink"
alpha = 1.0
character = "smooth"
character_amount = 0.70
high_pass_hz = 35.0
low_pass_hz = 16000.0
stereo_correlation = 0.35
gain_db = 0.0
"#;
        let p: PresetParams = toml::from_str(toml_src).unwrap();
        assert_eq!(p.name, "Soft Pink");
        assert_eq!(p.seed, 42);
        assert_eq!(p.layers.len(), 1);
        assert_eq!(p.layers[0].color, Color::Pink);
        assert_eq!(p.layers[0].character, Character::Smooth);
        assert!((p.layers[0].character_amount - 0.70).abs() < 1e-12);
        assert_eq!(p.layers[0].seed, None);
        assert_eq!(p.master.target_rms_dbfs, -36.0);
        // Unspecified fields fall back to defaults rather than failing.
        assert_eq!(p.master.fade_out_ms, MasterParams::default().fade_out_ms);
    }

    #[test]
    fn corrupt_presets_are_repairable_not_fatal() {
        let mut p = PresetParams {
            layers: (0..20)
                .map(|_| LayerParams {
                    alpha: f64::NAN,
                    character_amount: 5.0,
                    stereo_correlation: -3.0,
                    ..LayerParams::default()
                })
                .collect(),
            ..PresetParams::default()
        };
        let notes = p.sanitize();
        assert_eq!(p.layers.len(), MAX_LAYERS);
        assert!(!notes.is_empty());
        for l in &p.layers {
            assert!((0.0..=1.0).contains(&l.character_amount));
            assert!((-1.0..=1.0).contains(&l.stereo_correlation));
            assert!(l.alpha.is_finite());
        }
    }

    #[test]
    fn shared_seeds_are_reported() {
        let mut p = PresetParams {
            layers: vec![
                LayerParams {
                    seed: Some(7),
                    ..LayerParams::default()
                },
                LayerParams {
                    seed: Some(7),
                    ..LayerParams::default()
                },
            ],
            ..PresetParams::default()
        };
        let notes = p.sanitize();
        assert!(
            notes.iter().any(|n| n.contains("shares a seed")),
            "{notes:?}"
        );
    }

    #[test]
    fn derived_seeds_differ_per_layer() {
        let p = PresetParams::default();
        let a = p.layers[0].resolved_seed(p.seed, 0);
        let b = p.layers[0].resolved_seed(p.seed, 1);
        assert_ne!(a, b);
    }
}
