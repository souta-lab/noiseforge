//! Stereo correlation (DESIGN.md section 3.1).
//!
//! From two independent sources `u` and `v` with the same spectrum and RMS:
//!
//! ```text
//! L = u
//! R = rho * u + sqrt(1 - rho^2) * v
//! ```
//!
//! `rho = 1` is mono, `rho = 0` is fully decorrelated, `rho < 0` is negative
//! correlation (advanced; the mono downmix partially cancels).
//!
//! Because `u` and `v` have equal variance and are independent, `R` has exactly
//! the same variance as `L` for every `rho`, so the correlation control never
//! changes the level.
//!
//! DESIGN.md section 3.1 also requires that the *same* colour filter is applied
//! to both sources before the correlation matrix, which is what the engine does.

/// Coefficients of the correlation matrix.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StereoMatrix {
    /// Correlation coefficient actually in force.
    pub rho: f64,
    /// Weight of `u` in the right channel.
    pub direct: f64,
    /// Weight of `v` in the right channel.
    pub cross: f64,
}

impl StereoMatrix {
    /// Build the matrix for a correlation coefficient, clamping to `[-1, 1]`.
    pub fn new(rho: f64) -> Self {
        let r = if rho.is_finite() {
            rho.clamp(-1.0, 1.0)
        } else {
            0.0
        };
        Self {
            rho: r,
            direct: r,
            cross: (1.0 - r * r).max(0.0).sqrt(),
        }
    }

    /// Fully correlated (mono).
    pub const MONO: Self = Self {
        rho: 1.0,
        direct: 1.0,
        cross: 0.0,
    };

    /// Right channel from the filtered sources.
    #[inline]
    pub fn right(&self, u: f64, v: f64) -> f64 {
        self.direct * u + self.cross * v
    }

    /// Variance factor of the mono downmix `(L + R) / 2` relative to one channel.
    ///
    /// `Var((L+R)/2) / Var(L) = (1 + rho) / 2`.
    #[inline]
    pub fn mono_downmix_variance_factor(&self) -> f64 {
        (1.0 + self.rho) * 0.5
    }

    /// Human readable width label used by the UI.
    pub fn label(&self) -> &'static str {
        if self.rho >= 0.99 {
            "Mono"
        } else if self.rho >= 0.6 {
            "Narrow"
        } else if self.rho >= 0.15 {
            "Normal"
        } else if self.rho > -0.15 {
            "Wide"
        } else {
            "Inverted"
        }
    }
}

impl Default for StereoMatrix {
    fn default() -> Self {
        Self::MONO
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variance_is_preserved_for_every_rho() {
        // u and v unit variance and independent.
        for rho in [-1.0, -0.5, 0.0, 0.35, 0.9, 1.0] {
            let m = StereoMatrix::new(rho);
            let var = m.direct * m.direct + m.cross * m.cross;
            assert!((var - 1.0).abs() < 1e-12, "rho {rho}: var {var}");
        }
    }

    #[test]
    fn mono_downmix_matches_the_closed_form() {
        for rho in [-1.0, 0.0, 0.5, 1.0] {
            let m = StereoMatrix::new(rho);
            // Var((u + rho u + c v)/2) = ((1+rho)^2 + c^2)/4
            let var = ((1.0 + rho).powi(2) + m.cross * m.cross) / 4.0;
            assert!((var - m.mono_downmix_variance_factor()).abs() < 1e-12);
        }
    }

    #[test]
    fn out_of_range_is_clamped() {
        assert_eq!(StereoMatrix::new(5.0).rho, 1.0);
        assert_eq!(StereoMatrix::new(-5.0).rho, -1.0);
        assert_eq!(StereoMatrix::new(f64::NAN).rho, 0.0);
    }
}
