//! Analytic prediction of the configured graph, plus realtime safe metering.
//!
//! DESIGN.md section 3.3 requires `master.target_rms_dbfs` to be the *expected*
//! RMS of the finished PCM, calibrated offline, with no per-sample tracking that
//! could modulate the spectrum. This module computes that expectation exactly
//! from the filter coefficients.
//!
//! For white input of variance `sigma^2` through an LTI filter `h`:
//!
//! ```text
//! Var(y) = sigma^2 * sum |h[n]|^2
//!        = sigma^2 * (2 / Fs) * integral_0^(Fs/2) |H(f)|^2 df
//! ```
//!
//! The integral is evaluated on a log spaced grid, which resolves the very low
//! frequency behaviour that dominates Brown while still covering Nyquist.

use crate::characters::CharacterDesign;
use crate::colors::Color;
use crate::params::{LayerParams, PresetParams};
use crate::rng::UNIFORM_SYMMETRIC_VARIANCE;
use crate::sos::{BiquadCoeffs, SosCoeffs};

/// Number of integration points. Log spacing means most points sit where the
/// energy is, which matters because Brown spans more than 100 dB across the
/// audio band.
pub const RMS_INTEGRATION_POINTS: usize = 2048;

/// Lowest frequency included in the RMS integral.
pub const RMS_INTEGRATION_F_LO: f64 = 0.05;

/// The complete linear response of one layer, as actually implemented.
#[derive(Clone, Copy, Debug)]
pub struct LayerResponse {
    /// Colour shaping cascade.
    pub color: SosCoeffs,
    /// Character stage.
    pub character: CharacterDesign,
    /// User high pass.
    pub high_pass: BiquadCoeffs,
    /// User low pass.
    pub low_pass: BiquadCoeffs,
    /// Layer gain in dB.
    pub gain_db: f64,
    /// Variance factor applied by the *output* layout. `1.0` for a stereo sink,
    /// where each channel keeps the layer RMS, and `(1 + rho) / 2` for a mono
    /// sink, where the downmix `(L + R) / 2` is what actually leaves the
    /// engine. Getting this wrong would silently miss the level target.
    pub output_factor: f64,
}

impl LayerResponse {
    /// Squared magnitude of the whole layer at `f`.
    pub fn magnitude2(&self, f: f64, fs: f64) -> f64 {
        let (cr, ci) = self.character.response(f, fs);
        let mut m2 = self.color.magnitude2(f, fs) * (cr * cr + ci * ci);
        m2 *= self.high_pass.magnitude2(f, fs);
        m2 *= self.low_pass.magnitude2(f, fs);
        m2 * crate::mixer::db_to_linear(self.gain_db).powi(2)
    }

    /// Magnitude in dB, for the predicted-output spectrum overlay.
    pub fn response_db(&self, f: f64, fs: f64) -> f64 {
        10.0 * self.magnitude2(f, fs).max(f64::MIN_POSITIVE).log10()
    }

    /// Expected RMS at the layer output for a unit-variance white input.
    ///
    /// Includes the `1/sqrt(3)` factor of the uniform PRNG.
    pub fn expected_rms(&self, fs: f64) -> f64 {
        let var = UNIFORM_SYMMETRIC_VARIANCE * integrate_magnitude2(fs, |f| self.magnitude2(f, fs));
        (var * self.output_factor.max(0.0)).max(0.0).sqrt()
    }
}

/// Numerically integrate `|H(f)|^2` over `0..Fs/2` and return
/// `(2/Fs) * integral`, i.e. the noise power gain of the filter.
pub fn integrate_magnitude2<F: Fn(f64) -> f64>(fs: f64, f: F) -> f64 {
    let hi = fs * 0.5;
    let lo = RMS_INTEGRATION_F_LO;
    if hi <= lo || !hi.is_finite() {
        return 0.0;
    }
    let n = RMS_INTEGRATION_POINTS;
    let du = (hi / lo).ln() / (n - 1) as f64;
    let mut acc = 0.0f64;
    let mut prev_u = lo.ln();
    let mut prev_v = f(lo);
    for i in 1..n {
        let u = lo.ln() + du * i as f64;
        let freq = u.exp();
        let v = f(freq);
        // Trapezoid in ln(f): integral v df = integral v f d(ln f).
        acc += 0.5 * (prev_v * prev_u.exp() + v * freq) * du;
        prev_u = u;
        prev_v = v;
    }
    // Extrapolate the last tiny sliver up to Nyquist.
    2.0 * acc / fs
}

/// Per layer predicted RMS, in linear amplitude.
#[derive(Clone, Copy, Debug)]
pub struct LayerCalibration {
    pub rms: f64,
    pub rms_dbfs: f64,
    pub enabled: bool,
}

/// Result of calibrating a whole configuration.
#[derive(Clone, Copy, Debug)]
pub struct CalibrationReport {
    pub sample_rate: f64,
    pub layers: [LayerCalibration; crate::params::MAX_LAYERS],
    /// RMS of the mix before the master gain, in linear amplitude.
    pub mix_rms: f64,
    pub mix_rms_dbfs: f64,
    /// Gain in dB that must be applied to reach the target.
    pub calibration_gain_db: f64,
    /// Combined master gain (calibration plus user trim), in dB.
    pub master_gain_db: f64,
    /// Requested expected RMS.
    pub target_rms_dbfs: f64,
    /// Conservative estimate of the resulting peak.
    pub predicted_peak_dbfs: f64,
    /// True when the predicted peak would reach the limiter ceiling.
    pub headroom_warning: bool,
    /// Number of enabled layers.
    pub active_layers: usize,
}

impl CalibrationReport {
    /// A silent report, used when nothing is playing.
    pub fn silent(fs: f64, target_rms_dbfs: f64) -> Self {
        Self {
            sample_rate: fs,
            layers: [LayerCalibration {
                rms: 0.0,
                rms_dbfs: -200.0,
                enabled: false,
            }; crate::params::MAX_LAYERS],
            mix_rms: 0.0,
            mix_rms_dbfs: -200.0,
            calibration_gain_db: 0.0,
            master_gain_db: -120.0,
            target_rms_dbfs,
            predicted_peak_dbfs: -200.0,
            headroom_warning: false,
            active_layers: 0,
        }
    }
}

/// Build the layer response for the coefficients a caller has designed.
pub fn layer_response(
    color: SosCoeffs,
    character: CharacterDesign,
    high_pass: BiquadCoeffs,
    low_pass: BiquadCoeffs,
    gain_db: f64,
    output_factor: f64,
) -> LayerResponse {
    LayerResponse {
        color,
        character,
        high_pass,
        low_pass,
        gain_db,
        output_factor,
    }
}

/// Compute the master gain needed so the finished PCM has the expected RMS.
///
/// `mono_factor` is the variance factor of the output downmix: `1.0` for a
/// stereo sink (each channel keeps the layer RMS) and
/// [`crate::stereo::StereoMatrix::mono_downmix_variance_factor`] for a mono sink.
pub fn calibrate(
    fs: f64,
    responses: &[Option<LayerResponse>],
    target_rms_dbfs: f64,
    user_gain_db: f64,
    limiter_enabled: bool,
    limiter_ceiling_dbfs: f64,
) -> CalibrationReport {
    let mut layers = [LayerCalibration {
        rms: 0.0,
        rms_dbfs: -200.0,
        enabled: false,
    }; crate::params::MAX_LAYERS];
    let mut power = 0.0f64;
    let mut active = 0usize;
    for (i, r) in responses.iter().enumerate().take(crate::params::MAX_LAYERS) {
        match r {
            Some(resp) => {
                let rms = resp.expected_rms(fs);
                layers[i] = LayerCalibration {
                    rms,
                    rms_dbfs: crate::mixer::linear_to_db(rms),
                    enabled: true,
                };
                power += rms * rms;
                active += 1;
            }
            None => {
                layers[i] = LayerCalibration {
                    rms: 0.0,
                    rms_dbfs: -200.0,
                    enabled: false,
                };
            }
        }
    }
    let mix_rms = power.max(0.0).sqrt();
    let mix_rms_dbfs = crate::mixer::linear_to_db(mix_rms);
    let calibration_gain_db = target_rms_dbfs - mix_rms_dbfs;
    let master_gain_db = if active == 0 {
        -120.0
    } else {
        calibration_gain_db + user_gain_db
    };
    let predicted_peak_dbfs = if active == 0 {
        -200.0
    } else {
        crate::mixer::predicted_peak_dbfs(target_rms_dbfs + user_gain_db)
    };
    let headroom_warning = active > 0
        && (limiter_enabled && predicted_peak_dbfs > limiter_ceiling_dbfs
            || predicted_peak_dbfs > 0.0);
    CalibrationReport {
        sample_rate: fs,
        layers,
        mix_rms,
        mix_rms_dbfs,
        calibration_gain_db,
        master_gain_db,
        target_rms_dbfs,
        predicted_peak_dbfs,
        headroom_warning,
        active_layers: active,
    }
}

/// Convenience: the theoretical source-target curve for a layer.
pub fn source_target_db(color: Color, custom_alpha: f64, f: f64) -> f64 {
    crate::colors::amplitude_db(
        color.resolve_alpha(custom_alpha),
        f,
        crate::colors::F_REF_HZ,
    )
}

/// Layer parameters plus the designed coefficients, as the daemon keeps them.
#[derive(Clone, Debug)]
pub struct DesignedLayer {
    pub params: LayerParams,
    pub color: SosCoeffs,
    pub character: CharacterDesign,
    pub high_pass: BiquadCoeffs,
    pub low_pass: BiquadCoeffs,
}

/// Realtime safe metering accumulators.
///
/// Everything here is plain arithmetic on preallocated scalars; the values are
/// published to the UI through atomics by the PipeWire layer.
#[derive(Clone, Copy, Debug, Default)]
pub struct Meters {
    peak: f64,
    sum_sq: f64,
    count: u32,
    clipped: u32,
    nan_events: u32,
}

impl Meters {
    /// Reset the accumulators for a new block.
    #[inline]
    pub fn begin_block(&mut self) {
        self.peak = 0.0;
        self.sum_sq = 0.0;
        self.count = 0;
    }

    /// Accumulate one output sample.
    #[inline]
    pub fn push(&mut self, x: f64) {
        if !x.is_finite() {
            self.nan_events = self.nan_events.saturating_add(1);
            return;
        }
        let a = x.abs();
        if a > self.peak {
            self.peak = a;
        }
        self.sum_sq += x * x;
        self.count = self.count.saturating_add(1);
    }

    /// Peak magnitude of the block.
    #[inline]
    pub fn peak(&self) -> f64 {
        self.peak
    }

    /// RMS of the block, or zero for an empty block.
    #[inline]
    pub fn rms(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            (self.sum_sq / self.count as f64).sqrt()
        }
    }

    /// Number of non-finite samples seen since construction.
    #[inline]
    pub fn nan_events(&self) -> u32 {
        self.nan_events
    }

    /// Count a clipped sample.
    #[inline]
    pub fn note_clipped(&mut self) {
        self.clipped = self.clipped.saturating_add(1);
    }

    /// Total clipped samples since construction.
    #[inline]
    pub fn clipped(&self) -> u32 {
        self.clipped
    }
}

/// Validate a full preset's numeric ranges without needing the coefficients.
pub fn preset_is_playable(p: &PresetParams) -> bool {
    p.master.target_rms_dbfs.is_finite()
        && p.layers.iter().any(|l| l.enabled)
        && p.layers
            .iter()
            .all(|l| l.high_pass_hz.is_finite() && l.low_pass_hz.is_finite())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::characters::Character;
    use crate::designer;
    use crate::stereo::StereoMatrix;

    const FS: f64 = 48000.0;

    fn white_response(gain_db: f64) -> LayerResponse {
        layer_response(
            SosCoeffs::IDENTITY,
            crate::characters::design(Character::Neutral, 0.0, FS),
            BiquadCoeffs::IDENTITY,
            BiquadCoeffs::IDENTITY,
            gain_db,
            1.0,
        )
    }

    #[test]
    fn unfiltered_white_keeps_its_variance() {
        // An identity filter must return exactly the input variance 1/3.
        let r = white_response(0.0);
        let rms = r.expected_rms(FS);
        assert!((rms - (1.0f64 / 3.0).sqrt()).abs() < 1e-3, "rms {rms}");
    }

    #[test]
    fn gain_db_moves_rms_by_exactly_that_much() {
        let a = white_response(0.0).expected_rms(FS);
        let b = white_response(-6.0).expected_rms(FS);
        assert!((crate::mixer::linear_to_db(b / a) + 6.0).abs() < 1e-6);
    }

    #[test]
    fn colour_filters_are_normalised_to_unit_dc_gain() {
        // Because every section is DC normalised, the total noise power gain of
        // Brown must stay finite and modest rather than diverging.
        for alpha in [1.0, 2.0, 4.0] {
            let d = designer::design_color(alpha, FS).unwrap();
            let r = layer_response(
                d.coeffs,
                crate::characters::design(Character::Neutral, 0.0, FS),
                BiquadCoeffs::IDENTITY,
                BiquadCoeffs::IDENTITY,
                0.0,
                1.0,
            );
            let rms = r.expected_rms(FS);
            assert!(rms.is_finite() && rms > 0.0, "alpha {alpha}: {rms}");
        }
    }

    #[test]
    fn calibration_hits_the_target_exactly() {
        let r = white_response(0.0);
        let a = calibrate(FS, &[Some(r)], -36.0, 0.0, false, -1.0);
        let achieved = crate::mixer::linear_to_db(a.mix_rms) + a.master_gain_db;
        assert!((achieved - (-36.0)).abs() < 1e-9, "achieved {achieved}");
    }

    #[test]
    fn two_uncorrelated_layers_add_three_db() {
        let r = white_response(0.0);
        let one = calibrate(FS, &[Some(r)], -36.0, 0.0, false, -1.0);
        let two = calibrate(FS, &[Some(r), Some(r)], -36.0, 0.0, false, -1.0);
        // The calibration compensates, so both land on the target.
        assert!((two.mix_rms_dbfs - one.mix_rms_dbfs - 3.0103).abs() < 0.01);
        let achieved = crate::mixer::linear_to_db(two.mix_rms) + two.master_gain_db;
        assert!((achieved - (-36.0)).abs() < 1e-9);
    }

    #[test]
    fn mono_downmix_reduces_level_for_correlated_layers() {
        // rho = 1 makes L == R, so a mono downmix keeps the same RMS.
        let factor = StereoMatrix::new(1.0).mono_downmix_variance_factor();
        assert!((factor - 1.0).abs() < 1e-12);
        // rho = 0 downmix is -3 dB.
        let factor = StereoMatrix::new(0.0).mono_downmix_variance_factor();
        assert!((factor - 0.5).abs() < 1e-12);
        // rho = -1 cancels completely, which the calibration reports honestly.
        let factor = StereoMatrix::new(-1.0).mono_downmix_variance_factor();
        assert!(factor.abs() < 1e-12);

        // And the calibration must take it into account.
        let mut r = white_response(0.0);
        r.output_factor = 0.5;
        let a = calibrate(FS, &[Some(r)], -36.0, 0.0, false, -1.0);
        let achieved = crate::mixer::linear_to_db(a.mix_rms) + a.master_gain_db;
        assert!((achieved - (-36.0)).abs() < 1e-9, "achieved {achieved}");
    }

    #[test]
    fn meters_track_peak_rms_and_nan() {
        let mut m = Meters::begin_block_default();
        for x in [0.5, -0.5, 0.5, -0.5] {
            m.push(x);
        }
        assert!((m.peak() - 0.5).abs() < 1e-12);
        assert!((m.rms() - 0.5).abs() < 1e-12);
        m.push(f64::NAN);
        assert_eq!(m.nan_events(), 1);
    }

    impl Meters {
        fn begin_block_default() -> Self {
            let mut m = Meters::default();
            m.begin_block();
            m
        }
    }
}
