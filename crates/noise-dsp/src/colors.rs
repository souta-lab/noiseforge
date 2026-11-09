//! Noise colour definitions and the theoretical PSD they target.
//!
//! DESIGN.md section 1.1:
//!
//! ```text
//! S(f) / S(f_ref) = (f / f_ref)^(-alpha)
//! |H(f)|          = sqrt(S(f)) proportional to f^(-alpha/2)
//! ```
//!
//! The amplitude response and the power spectral density share the same
//! dB-per-octave slope (`-10 * log10(2) * alpha`), they are *not* the same
//! curve: `|H|` is the square root of the PSD.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Reference frequency used by the theoretical curve.
pub const F_REF_HZ: f64 = 1000.0;
/// Smallest supported custom exponent.
pub const ALPHA_MIN: f64 = -4.0;
/// Largest supported custom exponent.
pub const ALPHA_MAX: f64 = 4.0;
/// `-10 * log10(2)`: dB per octave contributed by one unit of alpha.
pub const DB_PER_OCTAVE: f64 = -3.010_299_956_639_812;

/// The generation source of a layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Color {
    White,
    Pink,
    Brown,
    Blue,
    Violet,
    Custom,
}

impl Color {
    /// All colours, in presentation order.
    pub const ALL: [Color; 6] = [
        Color::White,
        Color::Pink,
        Color::Brown,
        Color::Blue,
        Color::Violet,
        Color::Custom,
    ];

    /// Lower-case identifier used by the CLI, IPC and preset files.
    pub const fn as_str(self) -> &'static str {
        match self {
            Color::White => "white",
            Color::Pink => "pink",
            Color::Brown => "brown",
            Color::Blue => "blue",
            Color::Violet => "violet",
            Color::Custom => "custom",
        }
    }

    /// Parse a colour name (case insensitive).
    pub fn parse(s: &str) -> Option<Color> {
        let s = s.trim().to_ascii_lowercase();
        Color::ALL.into_iter().find(|c| c.as_str() == s)
    }

    /// `alpha` implied by the name, or `None` for [`Color::Custom`].
    pub const fn nominal_alpha(self) -> Option<f64> {
        match self {
            Color::White => Some(0.0),
            Color::Pink => Some(1.0),
            Color::Brown => Some(2.0),
            Color::Blue => Some(-1.0),
            Color::Violet => Some(-2.0),
            Color::Custom => None,
        }
    }

    /// Resolve the effective `alpha`, ignoring the custom value for named colours.
    pub fn resolve_alpha(self, custom_alpha: f64) -> f64 {
        match self.nominal_alpha() {
            Some(a) => a,
            None => custom_alpha.clamp(ALPHA_MIN, ALPHA_MAX),
        }
    }

    /// Human readable label for the UI.
    pub const fn label(self) -> &'static str {
        match self {
            Color::White => "White",
            Color::Pink => "Pink",
            Color::Brown => "Brown",
            Color::Blue => "Blue",
            Color::Violet => "Violet",
            Color::Custom => "Custom",
        }
    }
}

impl fmt::Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Theoretical PSD slope in dB per octave for a given `alpha`.
#[inline]
pub fn psd_slope_db_per_octave(alpha: f64) -> f64 {
    DB_PER_OCTAVE * alpha
}

/// Theoretical relative PSD in dB at frequency `f`.
#[inline]
pub fn psd_db(alpha: f64, f: f64, f_ref: f64) -> f64 {
    10.0 * alpha * (f_ref / f).log10()
}

/// Theoretical filter magnitude `(f / f_ref)^(-alpha/2)`.
#[inline]
pub fn amplitude(alpha: f64, f: f64, f_ref: f64) -> f64 {
    (f / f_ref).powf(-alpha / 2.0)
}

/// Theoretical filter magnitude in dB.
#[inline]
pub fn amplitude_db(alpha: f64, f: f64, f_ref: f64) -> f64 {
    -10.0 * alpha * (f / f_ref).log10()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_matches_design_document() {
        // DESIGN.md section 1.1 slope table (dB/oct).
        assert!((psd_slope_db_per_octave(0.0)).abs() < 1e-12);
        assert!((psd_slope_db_per_octave(1.0) + 3.010_299_956_639_812).abs() < 1e-12);
        assert!((psd_slope_db_per_octave(2.0) + 6.020_599_913_279_624).abs() < 1e-12);
        assert!((psd_slope_db_per_octave(-1.0) - 3.010_299_956_639_812).abs() < 1e-12);
        assert!((psd_slope_db_per_octave(-2.0) - 6.020_599_913_279_624).abs() < 1e-12);
    }

    #[test]
    fn power_and_amplitude_are_not_confused() {
        // Pink: PSD ~ 1/f, amplitude ~ 1/sqrt(f).
        let f = 4000.0;
        assert!((amplitude(1.0, f, 1000.0) - 0.5).abs() < 1e-12);
        assert!((psd_db(1.0, f, 1000.0) + 6.020_599_913_279_624).abs() < 1e-9);
        // Brown: PSD ~ 1/f^2, amplitude ~ 1/f.
        assert!((amplitude(2.0, f, 1000.0) - 0.25).abs() < 1e-12);
    }

    #[test]
    fn octave_slope_is_consistent() {
        for alpha in [-3.0, -1.0, 0.0, 0.5, 1.0, 2.0, 4.0] {
            let d = amplitude_db(alpha, 2000.0, 1000.0) - amplitude_db(alpha, 1000.0, 1000.0);
            assert!((d - psd_slope_db_per_octave(alpha)).abs() < 1e-9);
        }
    }

    #[test]
    fn parse_round_trip() {
        for c in Color::ALL {
            assert_eq!(Color::parse(c.as_str()), Some(c));
        }
        assert_eq!(Color::parse("PINK"), Some(Color::Pink));
        assert_eq!(Color::parse("nope"), None);
    }
}
