//! Layer summing and headroom policy (DESIGN.md section 3.2 and 3.3).
//!
//! DESIGN.md offers `sum(g_i x_i) / sqrt(sum(g_i^2))` as a *candidate* for
//! uncorrelated layers and then warns that it does not hold when layers share a
//! seed or are deliberately correlated.
//!
//! NoiseForge takes the design at its word and does something stronger instead:
//! the mix is a plain sum (so nothing is hidden), and the master calibration in
//! [`crate::metrics`] computes the *analytically predicted* RMS of the whole
//! configured graph and sets one fixed gain to hit `target_rms_dbfs`. That gain
//! is exact for uncorrelated layers, is recomputed whenever the configuration
//! changes, and is never a per-sample tracker, so it cannot modulate the
//! spectrum.
//!
//! Correlated layers are detected by their seeds in
//! [`crate::params::PresetParams::sanitize`] and reported to the user rather
//! than silently mis-calibrated.

use crate::params::MAX_LAYERS;

/// Hard limit on simultaneously enabled layers.
pub const MAX_ACTIVE_LAYERS: usize = MAX_LAYERS;

/// dB to linear amplitude.
#[inline]
pub fn db_to_linear(db: f64) -> f64 {
    10.0f64.powf(db / 20.0)
}

/// Linear amplitude to dB, floored at `-200 dB` so silence stays finite.
#[inline]
pub fn linear_to_db(x: f64) -> f64 {
    if x <= 0.0 || !x.is_finite() {
        -200.0
    } else {
        20.0 * x.log10()
    }
}

/// The `1 / sqrt(sum(g^2))` normalisation suggested by DESIGN.md section 3.2.
///
/// Kept because it is a useful diagnostic and is used by the offline renderer
/// when the caller explicitly asks for level-preserving layer addition. The
/// realtime engine does not need it, because the analytic calibration already
/// accounts for the layer gains.
pub fn uncorrelated_normalisation(gains: &[f64]) -> f64 {
    let sum: f64 = gains.iter().map(|g| g * g).sum();
    if sum > 0.0 { 1.0 / sum.sqrt() } else { 1.0 }
}

/// Conservative crest factor used to estimate the peak from an RMS target.
///
/// Uniform noise has a crest factor of `sqrt(3)`; a cascade of resonators can
/// push it higher, and Brown is dominated by its lowest band. `4.2` (~12.5 dB)
/// is a deliberately conservative planning number, not a measured guarantee.
pub const PLANNING_CREST_FACTOR: f64 = 4.2;

/// Predicted peak level in dBFS for an expected RMS.
pub fn predicted_peak_dbfs(rms_dbfs: f64) -> f64 {
    rms_dbfs + linear_to_db(PLANNING_CREST_FACTOR)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn db_round_trip() {
        for db in [-60.0, -36.0, -6.0, 0.0, 6.0] {
            assert!((linear_to_db(db_to_linear(db)) - db).abs() < 1e-12);
        }
        assert_eq!(linear_to_db(0.0), -200.0);
    }

    #[test]
    fn uncorrelated_normalisation_matches_two_equal_layers() {
        // Two equal uncorrelated layers sum to +3 dB, so the normaliser is 1/sqrt(2).
        let n = uncorrelated_normalisation(&[1.0, 1.0]);
        assert!((n - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-12);
        assert_eq!(uncorrelated_normalisation(&[]), 1.0);
        assert_eq!(uncorrelated_normalisation(&[0.0, 0.0]), 1.0);
    }

    #[test]
    fn peak_planning_is_conservative() {
        assert!(predicted_peak_dbfs(-36.0) > -36.0);
        assert!((predicted_peak_dbfs(-36.0) - (-23.54)).abs() < 0.1);
    }
}
