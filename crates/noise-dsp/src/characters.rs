//! Smooth / Intense sound characters (DESIGN.md section 2).
//!
//! A character is a *post colour* linear filter. `Neutral` is a complete
//! bypass. The applied spectrum is reported separately from the mathematical
//! colour spectrum, and the UI must never label a shaped output as the pristine
//! colour.
//!
//! # Why these are not RBJ shelves
//!
//! DESIGN.md section 2.1 forbids linearly interpolating IIR coefficients
//! without validation and asks for a sample continuous transition. Instead of
//! redesigning a shelf for every slider position, each shelf is written as
//!
//! ```text
//! H(g, z) = 1 + g * S(z)
//! ```
//!
//! where `S(z)` is a *fixed* first order shape filter (a high pass for a high
//! shelf, a low pass for a low shelf) with unity passband gain and `g` is the
//! linear gain offset `10^(dB/20) - 1`.
//!
//! Consequences:
//!
//! * `g = 0` is bit-exact bypass, so `Neutral` really is no processing.
//! * The IIR coefficients never change while `amount` moves, so nothing is
//!   interpolated; only the scalar `g` ramps, which is trivially sample
//!   continuous.
//! * The shelf shape is gain independent, so the reported corner frequency and
//!   gain fully describe the filter.
//! * A first order shape has at most 90 degrees of phase lag, so the resulting
//!   shelf is *provably monotone*: a boost never dips below its plateau on the
//!   way in, and a cut never overshoots it. A second order (Butterworth) shape
//!   reaches 180 degrees and produces a real, audible dip a little above the
//!   corner, which is why it is not used here.

use serde::{Deserialize, Serialize};

use crate::sos::BiquadCoeffs;

/// High shelf corner used by `Smooth` (DESIGN.md section 2 table).
pub const SMOOTH_HIGH_SHELF_HZ: f64 = 3200.0;
/// High shelf gain at `amount == 1` for `Smooth`.
pub const SMOOTH_HIGH_SHELF_DB: f64 = -8.0;
/// Low shelf corner used by `Intense`.
pub const INTENSE_LOW_SHELF_HZ: f64 = 180.0;
/// Low shelf gain at `amount == 1` for `Intense`.
pub const INTENSE_LOW_SHELF_DB: f64 = 3.0;
/// High shelf corner used by `Intense`.
pub const INTENSE_HIGH_SHELF_HZ: f64 = 2500.0;
/// High shelf gain at `amount == 1` for `Intense`.
pub const INTENSE_HIGH_SHELF_DB: f64 = 4.0;

/// Post-colour character.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Character {
    /// Bit-exact bypass.
    Neutral,
    /// Softens the high end.
    Smooth,
    /// Adds presence without raising the master level.
    Intense,
}

impl Character {
    /// Every character in presentation order.
    pub const ALL: [Character; 3] = [Character::Neutral, Character::Smooth, Character::Intense];

    /// Lower-case identifier.
    pub const fn as_str(self) -> &'static str {
        match self {
            Character::Neutral => "neutral",
            Character::Smooth => "smooth",
            Character::Intense => "intense",
        }
    }

    /// Parse a character name (case insensitive).
    pub fn parse(s: &str) -> Option<Character> {
        let s = s.trim().to_ascii_lowercase();
        Character::ALL.into_iter().find(|c| c.as_str() == s)
    }

    /// Display label.
    pub const fn label(self) -> &'static str {
        match self {
            Character::Neutral => "Neutral",
            Character::Smooth => "Smooth",
            Character::Intense => "Intense",
        }
    }
}

impl core::fmt::Display for Character {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A designed character: fixed shape filters plus the current linear gains.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CharacterDesign {
    /// Fixed low shelf shape (Butterworth low pass, unity passband).
    pub low_shelf: BiquadCoeffs,
    /// Fixed high shelf shape (Butterworth high pass, unity passband).
    pub high_shelf: BiquadCoeffs,
    /// Low shelf linear gain offset `10^(dB/20) - 1`; `0.0` disables it.
    pub low_gain: f64,
    /// High shelf linear gain offset.
    pub high_gain: f64,
    /// Shelf corner frequencies in Hz, for the UI and for the predicted curve.
    pub low_shelf_hz: f64,
    pub high_shelf_hz: f64,
    /// Shelf gains in dB at the requested amount.
    pub low_gain_db: f64,
    pub high_gain_db: f64,
}

impl CharacterDesign {
    /// A complete bypass.
    pub fn neutral(fs: f64) -> Self {
        Self {
            low_shelf: BiquadCoeffs::one_pole_low_pass(INTENSE_LOW_SHELF_HZ, fs),
            high_shelf: BiquadCoeffs::one_pole_high_pass(SMOOTH_HIGH_SHELF_HZ, fs),
            low_gain: 0.0,
            high_gain: 0.0,
            low_shelf_hz: INTENSE_LOW_SHELF_HZ,
            high_shelf_hz: SMOOTH_HIGH_SHELF_HZ,
            low_gain_db: 0.0,
            high_gain_db: 0.0,
        }
    }

    /// True when the character does not alter the signal at all.
    pub fn is_bypass(&self) -> bool {
        self.low_gain == 0.0 && self.high_gain == 0.0
    }

    /// Complex response of the character stage at `f`.
    pub fn response(&self, f: f64, fs: f64) -> (f64, f64) {
        let (lr, li) = self.low_shelf.response(f, fs);
        let (hr, hi) = self.high_shelf.response(f, fs);
        (
            1.0 + self.low_gain * lr + self.high_gain * hr,
            self.low_gain * li + self.high_gain * hi,
        )
    }

    /// Magnitude response in dB at `f`.
    pub fn response_db(&self, f: f64, fs: f64) -> f64 {
        let (re, im) = self.response(f, fs);
        10.0 * (re * re + im * im).max(f64::MIN_POSITIVE).log10()
    }
}

/// Convert a dB gain to the linear offset `g` used by `H = 1 + g S`.
#[inline]
pub fn shelf_gain_offset(db: f64) -> f64 {
    10.0f64.powf(db / 20.0) - 1.0
}

/// Shelf gains in dB for a character at a given amount.
///
/// DESIGN.md section 2.1: the dB gains scale linearly with `t`, they are *not*
/// applied as a volume change.
pub fn shelf_gains_db(character: Character, amount: f64) -> (f64, f64) {
    let t = if amount.is_finite() {
        amount.clamp(0.0, 1.0)
    } else {
        0.0
    };
    match character {
        Character::Neutral => (0.0, 0.0),
        Character::Smooth => (0.0, SMOOTH_HIGH_SHELF_DB * t),
        Character::Intense => (INTENSE_LOW_SHELF_DB * t, INTENSE_HIGH_SHELF_DB * t),
    }
}

/// Design the character stage for a rate and amount.
pub fn design(character: Character, amount: f64, fs: f64) -> CharacterDesign {
    let (low_db, high_db) = shelf_gains_db(character, amount);
    let low_hz = INTENSE_LOW_SHELF_HZ;
    let high_hz = match character {
        Character::Smooth => SMOOTH_HIGH_SHELF_HZ,
        Character::Intense => INTENSE_HIGH_SHELF_HZ,
        Character::Neutral => SMOOTH_HIGH_SHELF_HZ,
    };
    CharacterDesign {
        low_shelf: BiquadCoeffs::one_pole_low_pass(low_hz, fs),
        high_shelf: BiquadCoeffs::one_pole_high_pass(high_hz, fs),
        low_gain: shelf_gain_offset(low_db),
        high_gain: shelf_gain_offset(high_db),
        low_shelf_hz: low_hz,
        high_shelf_hz: high_hz,
        low_gain_db: low_db,
        high_gain_db: high_db,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FS: f64 = 48000.0;

    #[test]
    fn neutral_is_bit_exact_bypass() {
        let d = design(Character::Neutral, 1.0, FS);
        assert!(d.is_bypass());
        assert_eq!(d.low_gain, 0.0);
        assert_eq!(d.high_gain, 0.0);
        for f in [20.0, 100.0, 1000.0, 10000.0] {
            assert_eq!(d.response_db(f, FS), 0.0);
        }
    }

    #[test]
    fn smooth_reaches_minus_8db_at_full_amount() {
        let d = design(Character::Smooth, 1.0, FS);
        assert!((d.high_gain_db + 8.0).abs() < 1e-12);
        // Far above the corner the shelf reaches its plateau.
        let plateau = d.response_db(20_000.0, FS);
        assert!((plateau + 8.0).abs() < 0.5, "plateau {plateau}");
        // Well below the corner it is transparent.
        assert!(d.response_db(200.0, FS).abs() < 0.1);
        // Monotone decreasing: a cut never overshoots and never dips again.
        let mut prev = 0.0;
        for i in 0..200 {
            let f = 100.0 * (200.0f64).powf(i as f64 / 199.0);
            let db = d.response_db(f, FS);
            assert!(
                db <= prev + 1e-3,
                "not monotone at {f} Hz: {db} after {prev}"
            );
            prev = db;
        }
    }

    #[test]
    fn intense_lifts_both_ends_and_keeps_the_middle() {
        let d = design(Character::Intense, 1.0, FS);
        let low = d.response_db(30.0, FS);
        let mid = d.response_db(1000.0, FS);
        let high = d.response_db(20_000.0, FS);
        assert!((low - 3.0).abs() < 0.4, "low {low}");
        assert!((high - 4.0).abs() < 0.5, "high {high}");
        assert!(mid.abs() < 1.0, "mid {mid}");
        // The transition straddles the 180 Hz corner.
        assert!(
            d.response_db(180.0, FS) > 1.2,
            "{}",
            d.response_db(180.0, FS)
        );
        // Each individual shelf is monotone (proved in `sos`), so Intense is a
        // low shelf lift that hands over to a high shelf lift: it must stay
        // inside the envelope of the two plateaus and never dip below unity.
        for i in 0..200 {
            let f = 10.0 * (3000.0f64).powf(i as f64 / 199.0);
            let db = d.response_db(f, FS);
            assert!(
                (-0.05..=4.05).contains(&db),
                "out of envelope at {f} Hz: {db}"
            );
        }
    }

    #[test]
    fn gains_scale_linearly_with_amount() {
        for t in [0.0, 0.25, 0.5, 0.7, 1.0] {
            let (lo, hi) = shelf_gains_db(Character::Intense, t);
            assert!((lo - 3.0 * t).abs() < 1e-12);
            assert!((hi - 4.0 * t).abs() < 1e-12);
            let (_, hi_s) = shelf_gains_db(Character::Smooth, t);
            assert!((hi_s + 8.0 * t).abs() < 1e-12);
        }
        // Out of range amounts are clamped, not extrapolated.
        assert_eq!(shelf_gains_db(Character::Smooth, 5.0).1, -8.0);
        assert_eq!(shelf_gains_db(Character::Smooth, f64::NAN).1, 0.0);
    }

    #[test]
    fn parse_round_trip() {
        for c in Character::ALL {
            assert_eq!(Character::parse(c.as_str()), Some(c));
        }
        assert_eq!(Character::parse("SMOOTH"), Some(Character::Smooth));
        assert_eq!(Character::parse("warm"), None);
    }
}
