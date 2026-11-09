//! Compile a [`PresetParams`] into everything the engine and the UI need.
//!
//! This is the one place where a human-facing configuration becomes concrete
//! DSP. It is deliberately **not** realtime safe: it designs coefficients,
//! validates them, runs the analytic RMS calibration and records the measured
//! design accuracy. The daemon runs it on its control thread, the offline
//! renderer and the tests run it directly.
//!
//! DESIGN.md section 4.4: "the non-RT control worker validates and rounds the
//! values and computes the new SOS coefficients", and only pre-designed
//! coefficients are published to the realtime thread. That is exactly what
//! [`CompiledGraph::commands`] produces.

use crate::characters::{self, CharacterDesign};
use crate::designer::{self, DesignError, DesignOptions, DesignReport};
use crate::engine::{Command, LayerFilters};
use crate::metrics::{self, CalibrationReport, LayerResponse};
use crate::params::{LayerParams, PresetParams};
use crate::sos::{BiquadCoeffs, SosCoeffs};

/// Everything derived from one preset at one sample rate.
#[derive(Clone, Debug)]
pub struct CompiledGraph {
    /// Sample rate the graph was compiled for.
    pub sample_rate: f64,
    /// Output channel count the calibration assumed.
    pub channels: usize,
    /// Per layer results, in preset order.
    pub layers: Vec<CompiledLayer>,
    /// Analytic level calibration.
    pub calibration: CalibrationReport,
    /// Warnings produced while compiling, shown verbatim to the user.
    pub notes: Vec<String>,
}

/// One compiled layer.
#[derive(Clone, Debug)]
pub struct CompiledLayer {
    pub index: usize,
    pub enabled: bool,
    pub name: String,
    pub params: LayerParams,
    pub seed: u64,
    pub filters: LayerFilters,
    pub character: CharacterDesign,
    pub response: LayerResponse,
    /// What the colour designer actually achieved, or `None` when the layer is
    /// disabled or its colour needed no filtering.
    pub design: Option<DesignReport>,
    /// Set when the colour design failed outright and the previous/neutral
    /// coefficients were kept.
    pub design_error: Option<DesignError>,
}

impl CompiledLayer {
    /// True when the colour filter is inside the documented tolerance.
    pub fn verified(&self) -> bool {
        self.design.map(|d| d.within_tolerance).unwrap_or(true)
    }
}

impl CompiledGraph {
    /// Look up a compiled layer.
    pub fn layer(&self, index: usize) -> Option<&CompiledLayer> {
        self.layers.get(index)
    }

    /// Every command needed to bring an [`crate::engine::Engine`] to this state,
    /// in a safe order. Coefficients are pushed before layers are enabled so a
    /// layer never sounds with stale filters.
    pub fn commands(&self, params: &PresetParams) -> Vec<Command> {
        let mut out = Vec::with_capacity(self.layers.len() * 6 + 4);
        let ramp = params.master.ramp_ms as f32;
        let fade = params.master.fade_in_ms.max(params.master.ramp_ms) as f32;

        out.push(Command::SetFades {
            in_ms: params.master.fade_in_ms as f32,
            out_ms: params.master.fade_out_ms as f32,
        });
        for l in &self.layers {
            let idx = l.index as u8;
            out.push(Command::SetLayerSeed {
                layer: idx,
                seed: l.seed,
                fade_ms: 0.0,
            });
            out.push(Command::SetLayerFilters {
                layer: idx,
                filters: l.filters,
                crossfade_ms: fade,
            });
            out.extend(crate::engine::character_commands(
                idx,
                &l.character,
                params.master.ramp_ms as f32,
            ));
            out.push(Command::SetCharacterGain {
                layer: idx,
                low_gain: l.character.low_gain as f32,
                high_gain: l.character.high_gain as f32,
                ramp_ms: ramp,
            });
            out.push(Command::SetLayerGain {
                layer: idx,
                gain_db: l.params.gain_db as f32,
                ramp_ms: ramp,
            });
            out.push(Command::SetStereoCorrelation {
                layer: idx,
                correlation: l.params.stereo_correlation as f32,
                ramp_ms: ramp,
            });
            out.push(Command::SetLayerEnabled {
                layer: idx,
                enabled: l.enabled,
                fade_ms: ramp,
            });
        }
        out.push(Command::SetLimiter {
            enabled: params.master.limiter_enabled,
            ceiling_dbfs: params.master.limiter_ceiling_dbfs as f32,
        });
        out.push(Command::SetMasterGain {
            gain_db: self.calibration.master_gain_db as f32,
            ramp_ms: ramp,
            // Hold a gain *rise* until the coefficient crossfade above has
            // finished; see `Command::SetMasterGain`.
            hold_ms: fade,
        });
        out
    }
}

/// Compile a preset for a sample rate and output channel count.
///
/// Never fails: a layer whose coefficients cannot be designed keeps a neutral
/// (White) cascade and the reason is recorded in [`CompiledGraph::notes`],
/// because DESIGN.md section 9.3 requires that a corrupt preset cannot take the
/// daemon down.
pub fn compile(params: &PresetParams, sample_rate: f64, channels: usize) -> CompiledGraph {
    compile_with(params, sample_rate, channels, DesignOptions::default())
}

/// Compile with explicit designer options.
pub fn compile_with(
    params: &PresetParams,
    sample_rate: f64,
    channels: usize,
    opts: DesignOptions,
) -> CompiledGraph {
    let fs = if sample_rate.is_finite() && sample_rate >= 8000.0 {
        sample_rate
    } else {
        48_000.0
    };
    let mut sanitized = params.clone();
    let mut notes = sanitized.sanitize();
    let channels = channels.clamp(1, 2);

    let mut layers = Vec::with_capacity(sanitized.layers.len());
    let mut responses: Vec<Option<LayerResponse>> = Vec::with_capacity(sanitized.layers.len());

    for (index, lp) in sanitized.layers.iter().enumerate() {
        let alpha = lp.alpha();
        let (color_coeffs, design, design_error) = match designer::design_color_with(
            alpha, fs, opts,
        ) {
            Ok(d) => {
                if !d.report.within_tolerance {
                    notes.push(format!(
                        "layer[{index}] {}: shape error {:.3} dB / slope error {:.3} dB/oct over {:.0}-{:.0} Hz is outside the documented tolerance",
                        lp.color, d.report.max_abs_dev_db, d.report.slope_error_db_per_octave,
                        d.report.band_lo_hz, d.report.band_hi_hz
                    ));
                }
                (d.coeffs, Some(d.report), None)
            }
            Err(e) => {
                notes.push(format!(
                    "layer[{index}] colour design failed ({e}); keeping the previous coefficients"
                ));
                (SosCoeffs::IDENTITY, None, Some(e))
            }
        };

        let hpf = if lp.high_pass_hz > 0.0 {
            designer::design_high_pass(lp.high_pass_hz, core::f64::consts::FRAC_1_SQRT_2, fs)
        } else {
            BiquadCoeffs::IDENTITY
        };
        let lpf = if lp.low_pass_hz > 0.0 && lp.low_pass_hz < 0.49 * fs {
            designer::design_low_pass(lp.low_pass_hz, core::f64::consts::FRAC_1_SQRT_2, fs)
        } else {
            BiquadCoeffs::IDENTITY
        };
        let character = characters::design(lp.character, lp.character_amount, fs);
        // The RMS of the colour plus EQ chain, excluding the character stage and
        // the layer gain. The engine uses it to decide between a crossfade and a
        // dip-swap when this layer's filter bank changes.
        let chain_rms = metrics::layer_response(
            color_coeffs,
            characters::design(crate::characters::Character::Neutral, 0.0, fs),
            hpf,
            lpf,
            0.0,
            1.0,
        )
        .expected_rms(fs);
        let filters = LayerFilters {
            color: color_coeffs,
            high_pass: hpf,
            low_pass: lpf,
            expected_rms: chain_rms as f32,
        };

        let output_factor = if channels == 1 {
            crate::stereo::StereoMatrix::new(lp.stereo_correlation).mono_downmix_variance_factor()
        } else {
            1.0
        };
        let response =
            metrics::layer_response(color_coeffs, character, hpf, lpf, lp.gain_db, output_factor);
        responses.push(Some(response));
        if !lp.enabled {
            responses[index] = None;
        }

        layers.push(CompiledLayer {
            index,
            enabled: lp.enabled,
            name: lp.name.clone(),
            params: lp.clone(),
            seed: lp.resolved_seed(sanitized.seed, index),
            filters,
            character,
            response,
            design,
            design_error,
        });
    }

    let calibration = metrics::calibrate(
        fs,
        &responses,
        sanitized.master.target_rms_dbfs,
        sanitized.master.user_gain_db(),
        sanitized.master.limiter_enabled,
        sanitized.master.limiter_ceiling_dbfs,
    );

    if calibration.headroom_warning {
        notes.push(format!(
            "predicted peak {:.1} dBFS is at or above the ceiling with a {:.1} dBFS RMS target",
            calibration.predicted_peak_dbfs, calibration.target_rms_dbfs
        ));
    }
    for l in &layers {
        if l.enabled && (l.character.low_gain < 0.0 || l.character.high_gain > 0.0) {
            notes.push(format!(
                "layer[{}] character shapes the colour: the output spectrum is not the pristine {} curve",
                l.index, l.params.color
            ));
        }
    }

    CompiledGraph {
        sample_rate: fs,
        channels,
        layers,
        calibration,
        notes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::colors::Color;
    use crate::engine::{Engine, PlayState};

    #[test]
    fn compiles_the_default_preset() {
        let p = PresetParams::default();
        let g = compile(&p, 48000.0, 2);
        assert_eq!(g.layers.len(), 1);
        assert!(g.layers[0].verified());
        assert!((g.calibration.target_rms_dbfs + 36.0).abs() < 1e-12);
        assert!(g.calibration.master_gain_db.is_finite());
    }

    #[test]
    fn every_colour_compiles_at_every_rate() {
        for fs in [44100.0, 48000.0, 96000.0] {
            for color in Color::ALL {
                let p = PresetParams {
                    layers: vec![LayerParams {
                        color,
                        alpha: 1.35,
                        ..LayerParams::default()
                    }],
                    ..PresetParams::default()
                };
                let g = compile(&p, fs, 2);
                assert!(g.layers[0].verified(), "{color} @ {fs}");
                assert!(g.layers[0].filters.is_valid());
                assert!(g.calibration.master_gain_db.is_finite());
            }
        }
    }

    #[test]
    fn commands_drive_an_engine_to_the_calibrated_target() {
        let p = PresetParams::default();
        let g = compile(&p, 48000.0, 2);
        let mut e = Engine::new(48000.0, p.seed);
        for c in g.commands(&p) {
            e.apply(c);
        }
        e.apply(Command::Play);
        // Skip the fades, then measure.
        for _ in 0..80 {
            e.render_block(256);
        }
        let mut peak = 0.0f64;
        let mut sum = 0.0f64;
        let mut n = 0u64;
        for _ in 0..400 {
            let b = e.render_block(256);
            for s in b {
                peak = peak.max((*s as f64).abs());
                sum += (*s as f64) * (*s as f64);
                n += 1;
            }
        }
        let rms = (sum / n as f64).sqrt();
        let rms_dbfs = 20.0 * rms.log10();
        let peak_dbfs = 20.0 * peak.log10();
        assert_eq!(e.play_state(), PlayState::Playing);
        assert!(
            (rms_dbfs - (-36.0)).abs() < 2.0,
            "expected about -36 dBFS, measured {rms_dbfs:.2} dBFS (peak {peak_dbfs:.2})"
        );
        assert!(peak_dbfs < -10.0, "peak {peak_dbfs:.2} dBFS is too hot");
    }

    #[test]
    fn eight_layers_stay_inside_the_headroom() {
        let p = PresetParams {
            seed: 7,
            layers: (0..8)
                .map(|i| LayerParams {
                    name: format!("L{i}"),
                    gain_db: -6.0 * (i as f64 % 3.0),
                    high_pass_hz: 30.0 + i as f64 * 10.0,
                    ..LayerParams::default()
                })
                .collect(),
            ..PresetParams::default()
        };
        let g = compile(&p, 48000.0, 2);
        assert_eq!(g.layers.len(), 8);
        assert!(g.calibration.master_gain_db.is_finite());
        assert!(!g.calibration.headroom_warning);
    }

    #[test]
    fn a_failing_layer_is_reported_not_fatal() {
        let p = PresetParams {
            layers: vec![LayerParams {
                alpha: f64::NAN,
                ..LayerParams::default()
            }],
            ..PresetParams::default()
        };
        let g = compile(&p, 48000.0, 2);
        assert_eq!(g.layers.len(), 1);
        // NaN alpha is repaired by sanitize(), so the design succeeds.
        assert!(g.layers[0].filters.is_valid());
        assert!(!g.notes.is_empty());
    }
}
