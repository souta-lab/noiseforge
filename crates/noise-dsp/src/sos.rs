//! Biquad / second order section primitives.
//!
//! All coefficient design happens outside the realtime thread in
//! [`crate::designer`]; the realtime path only ever runs the `process` methods
//! below, which touch a fixed, preallocated state array and never allocate.
//!
//! Sections use Transposed Direct Form II with `f64` coefficients and `f64`
//! state (DESIGN.md section 1.4 and the open question in section 12). The
//! `f32` comparison is measured by `benches/render.rs`.

use core::f64::consts::PI;

/// Maximum number of cascaded sections in one [`SosChain`].
///
/// Sized for the worst case in the engine: eight shaping sections plus a user
/// high pass and a user low pass, folded into a single cascade so that one
/// crossfade covers the whole filter bank.
pub const MAX_SECTIONS: usize = 12;

/// Biquad coefficients, normalised so that `a0 == 1`.
///
/// ```text
///        b0 + b1 z^-1 + b2 z^-2
/// H(z) = ----------------------
///         1 + a1 z^-1 + a2 z^-2
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BiquadCoeffs {
    pub b0: f64,
    pub b1: f64,
    pub b2: f64,
    pub a1: f64,
    pub a2: f64,
}

impl Default for BiquadCoeffs {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl BiquadCoeffs {
    /// Pass-through.
    pub const IDENTITY: Self = Self {
        b0: 1.0,
        b1: 0.0,
        b2: 0.0,
        a1: 0.0,
        a2: 0.0,
    };

    /// A first order section stored in a biquad (`b2 == a2 == 0`).
    #[inline]
    pub const fn first_order(b0: f64, b1: f64, a1: f64) -> Self {
        Self {
            b0,
            b1,
            b2: 0.0,
            a1,
            a2: 0.0,
        }
    }

    /// True when the section does nothing.
    #[inline]
    pub fn is_identity(&self) -> bool {
        self.b0 == 1.0 && self.b1 == 0.0 && self.b2 == 0.0 && self.a1 == 0.0 && self.a2 == 0.0
    }

    /// True when every coefficient is finite.
    #[inline]
    pub fn is_finite(&self) -> bool {
        self.b0.is_finite()
            && self.b1.is_finite()
            && self.b2.is_finite()
            && self.a1.is_finite()
            && self.a2.is_finite()
    }

    /// Jury stability test for `1 + a1 z^-1 + a2 z^-2`.
    ///
    /// Roots of `z^2 + a1 z + a2` lie strictly inside the unit circle iff
    /// `|a2| < 1` and `|a1| < 1 + a2`.
    #[inline]
    pub fn is_stable(&self) -> bool {
        self.a1.is_finite()
            && self.a2.is_finite()
            && self.a2.abs() < 1.0
            && self.a1.abs() < 1.0 + self.a2
    }

    /// Combine two first order sections `(b0, b1, a1)` into one biquad.
    #[inline]
    pub fn pair_first_order(p: (f64, f64, f64), q: (f64, f64, f64)) -> Self {
        Self {
            b0: p.0 * q.0,
            b1: p.0 * q.1 + p.1 * q.0,
            b2: p.1 * q.1,
            a1: p.2 + q.2,
            a2: p.2 * q.2,
        }
    }

    /// Squared magnitude response `|H(f)|^2` at frequency `f` for rate `fs`.
    pub fn magnitude2(&self, f: f64, fs: f64) -> f64 {
        let (re, im) = self.response(f, fs);
        re * re + im * im
    }

    /// Complex response `H(f)` at frequency `f` for rate `fs`.
    pub fn response(&self, f: f64, fs: f64) -> (f64, f64) {
        let w = 2.0 * PI * f / fs;
        let (s1, c1) = w.sin_cos();
        let (s2, c2) = (2.0 * w).sin_cos();
        let nr = self.b0 + self.b1 * c1 + self.b2 * c2;
        let ni = -(self.b1 * s1 + self.b2 * s2);
        let dr = 1.0 + self.a1 * c1 + self.a2 * c2;
        let di = -(self.a1 * s1 + self.a2 * s2);
        let d = dr * dr + di * di;
        if d <= 0.0 {
            return (f64::INFINITY, 0.0);
        }
        ((nr * dr + ni * di) / d, (ni * dr - nr * di) / d)
    }

    /// Magnitude response in dB at frequency `f` for rate `fs`.
    pub fn response_db(&self, f: f64, fs: f64) -> f64 {
        let m2 = self.magnitude2(f, fs);
        if m2 <= 0.0 {
            f64::NEG_INFINITY
        } else {
            10.0 * m2.log10()
        }
    }

    /// RBJ second order low pass with the given corner and `q`.
    pub fn low_pass(f0: f64, q: f64, fs: f64) -> Self {
        let (b0, b1, b2, a0, a1, a2) = rbj_low_high(f0, q, fs);
        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }

    /// RBJ second order high pass with the given corner and `q`.
    pub fn high_pass(f0: f64, q: f64, fs: f64) -> Self {
        let (_, _, _, a0, a1, a2) = rbj_low_high(f0, q, fs);
        let w0 = 2.0 * PI * f0 / fs;
        let (_, cw) = w0.sin_cos();
        let b0 = (1.0 + cw) / 2.0;
        let b1 = -(1.0 + cw);
        let b2 = (1.0 + cw) / 2.0;
        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }

    /// First order low pass with a `-3 dB` corner at `fc`, unity DC gain.
    ///
    /// Used as the *shape* of a low shelf. A first order section is chosen
    /// deliberately: its phase lag never exceeds 90 degrees, so `1 + g Hlp`
    /// stays monotone in frequency for every `g > -1` and a shelf built this way
    /// can never dip below its plateau or overshoot past it. A second order
    /// shape reaches 180 degrees of lag and does exactly that.
    pub fn one_pole_low_pass(fc: f64, fs: f64) -> Self {
        let k = one_pole_k(fc, fs);
        let d = 1.0 + k;
        Self::first_order(1.0 / d, 1.0 / d, (1.0 - k) / d)
    }

    /// First order high pass with a `-3 dB` corner at `fc`, unity high frequency
    /// gain. The high shelf counterpart of [`BiquadCoeffs::one_pole_low_pass`].
    pub fn one_pole_high_pass(fc: f64, fs: f64) -> Self {
        let k = one_pole_k(fc, fs);
        let d = 1.0 + k;
        Self::first_order(k / d, -k / d, (1.0 - k) / d)
    }

    /// First order high pass (one pole at DC-ish), used as a DC blocker.
    pub fn dc_blocker(fc: f64, fs: f64) -> Self {
        let k = (2.0 * PI * fc / fs).tan();
        let d = 1.0 + k;
        Self::first_order(1.0 / d, -1.0 / d, (k - 1.0) / d)
    }
}

/// `k = 2 fs / w` with `w` pre-warped so the digital `-3 dB` point sits exactly
/// on `fc` rather than on the analog prototype's corner.
#[inline]
fn one_pole_k(fc: f64, fs: f64) -> f64 {
    let f = fc.clamp(0.01, 0.47 * fs);
    1.0 / (PI * f / fs).tan()
}

#[inline]
fn rbj_low_high(f0: f64, q: f64, fs: f64) -> (f64, f64, f64, f64, f64, f64) {
    let w0 = 2.0 * PI * f0 / fs;
    let (sw, cw) = w0.sin_cos();
    let alpha = sw / (2.0 * q);
    let b0 = (1.0 - cw) / 2.0;
    let b1 = 1.0 - cw;
    let b2 = (1.0 - cw) / 2.0;
    let a0 = 1.0 + alpha;
    let a1 = -2.0 * cw;
    let a2 = 1.0 - alpha;
    (b0, b1, b2, a0, a1, a2)
}

/// A fixed-capacity cascade of biquad coefficients.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SosCoeffs {
    count: u8,
    sections: [BiquadCoeffs; MAX_SECTIONS],
}

impl Default for SosCoeffs {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl SosCoeffs {
    /// An empty cascade (pass-through).
    pub const IDENTITY: Self = Self {
        count: 0,
        sections: [BiquadCoeffs::IDENTITY; MAX_SECTIONS],
    };

    /// Build from a slice, truncating beyond [`MAX_SECTIONS`].
    pub fn new(sections: &[BiquadCoeffs]) -> Self {
        debug_assert!(sections.len() <= MAX_SECTIONS);
        let mut out = Self::IDENTITY;
        let n = sections.len().min(MAX_SECTIONS);
        out.sections[..n].copy_from_slice(&sections[..n]);
        out.count = n as u8;
        out
    }

    /// Number of active sections.
    #[inline]
    pub fn len(&self) -> usize {
        self.count as usize
    }

    /// True when the cascade is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Borrow the active sections.
    #[inline]
    pub fn as_slice(&self) -> &[BiquadCoeffs] {
        &self.sections[..self.count as usize]
    }

    /// Mutable access to the active sections.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [BiquadCoeffs] {
        let n = self.count as usize;
        &mut self.sections[..n]
    }

    /// All coefficients finite and every pole strictly inside the unit circle.
    pub fn is_valid(&self) -> bool {
        self.as_slice()
            .iter()
            .all(|s| s.is_finite() && s.is_stable())
    }

    /// Cascade magnitude response in dB at `f`.
    pub fn response_db(&self, f: f64, fs: f64) -> f64 {
        let mut db = 0.0;
        for s in self.as_slice() {
            db += s.response_db(f, fs);
        }
        db
    }

    /// Cascade squared magnitude response at `f`.
    pub fn magnitude2(&self, f: f64, fs: f64) -> f64 {
        let mut m = 1.0;
        for s in self.as_slice() {
            m *= s.magnitude2(f, fs);
        }
        m
    }
}

/// Per section state of a Transposed Direct Form II biquad.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BiquadState {
    s1: f64,
    s2: f64,
}

/// One realtime biquad: coefficients plus running state.
#[derive(Clone, Copy, Debug, Default)]
pub struct Biquad {
    pub coeffs: BiquadCoeffs,
    state: BiquadState,
}

impl Biquad {
    /// Build a zero-state biquad.
    #[inline]
    pub fn new(coeffs: BiquadCoeffs) -> Self {
        Self {
            coeffs,
            state: BiquadState::default(),
        }
    }

    /// Replace coefficients and clear the state.
    #[inline]
    pub fn set_coeffs_reset(&mut self, coeffs: BiquadCoeffs) {
        self.coeffs = coeffs;
        self.state = BiquadState::default();
    }

    /// Clear the state, keeping the coefficients.
    #[inline]
    pub fn reset(&mut self) {
        self.state = BiquadState::default();
    }

    /// Transposed Direct Form II step.
    #[inline]
    pub fn process(&mut self, x: f64) -> f64 {
        let c = &self.coeffs;
        let s = &mut self.state;
        let y = c.b0 * x + s.s1;
        s.s1 = c.b1 * x - c.a1 * y + s.s2;
        s.s2 = c.b2 * x - c.a2 * y;
        y
    }
}

/// A fixed-capacity cascade of biquads with running state.
#[derive(Clone, Copy, Debug, Default)]
pub struct SosChain {
    pub coeffs: SosCoeffs,
    state: [BiquadState; MAX_SECTIONS],
}

impl SosChain {
    /// Zero-state cascade with the given coefficients.
    pub fn new(coeffs: SosCoeffs) -> Self {
        Self {
            coeffs,
            state: [BiquadState::default(); MAX_SECTIONS],
        }
    }

    /// Replace coefficients and clear state.
    pub fn set_coeffs_reset(&mut self, coeffs: SosCoeffs) {
        self.coeffs = coeffs;
        self.state = [BiquadState::default(); MAX_SECTIONS];
    }

    /// Clear state, keeping coefficients.
    pub fn reset(&mut self) {
        self.state = [BiquadState::default(); MAX_SECTIONS];
    }

    /// Run one sample through the whole cascade.
    #[inline]
    pub fn process(&mut self, x: f64) -> f64 {
        let n = self.coeffs.count as usize;
        if n == 0 {
            return x;
        }
        let secs = &self.coeffs.sections;
        let st = &mut self.state;
        let mut v = x;
        for i in 0..n {
            let c = &secs[i];
            let s = &mut st[i];
            let y = c.b0 * v + s.s1;
            s.s1 = c.b1 * v - c.a1 * y + s.s2;
            s.s2 = c.b2 * v - c.a2 * y;
            v = y;
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_is_transparent() {
        let mut c = SosChain::new(SosCoeffs::IDENTITY);
        for i in 0..1000 {
            let x = (i as f64 * 0.1).sin();
            assert_eq!(c.process(x), x);
        }
    }

    #[test]
    fn first_order_pair_equals_sequential_processing() {
        let a = (0.9, 0.1, -0.5);
        let b = (0.8, -0.2, 0.3);
        let paired = BiquadCoeffs::pair_first_order(a, b);
        let mut one = Biquad::new(BiquadCoeffs::first_order(a.0, a.1, a.2));
        let mut two = Biquad::new(BiquadCoeffs::first_order(b.0, b.1, b.2));
        let mut pair = Biquad::new(paired);
        let mut seed = 1u64;
        for _ in 0..5000 {
            let x = super::super::rng::splitmix64(&mut seed) as f64 / u64::MAX as f64 - 0.5;
            let expected = two.process(one.process(x));
            let got = pair.process(x);
            assert!((expected - got).abs() < 1e-12);
        }
    }

    #[test]
    fn stability_test_matches_poles() {
        // poles at 0.9 and 0.9 -> a1 = -1.8, a2 = 0.81
        let s = BiquadCoeffs {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: -1.8,
            a2: 0.81,
        };
        assert!(s.is_stable());
        // pole at 1.2 -> unstable
        let u = BiquadCoeffs {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: -1.2,
            a2: 0.0,
        };
        assert!(!u.is_stable());
    }

    #[test]
    fn butterworth_low_pass_is_minus_3db_at_corner() {
        let fs = 48000.0;
        let c = BiquadCoeffs::low_pass(1000.0, core::f64::consts::FRAC_1_SQRT_2, fs);
        assert!((c.response_db(1000.0, fs) + 3.0103).abs() < 0.01);
        // 12 dB/octave asymptotically. Measured far above the corner at a high
        // rate so bilinear frequency warping is negligible.
        let fs_hi = 384_000.0;
        let c_hi = BiquadCoeffs::low_pass(1000.0, core::f64::consts::FRAC_1_SQRT_2, fs_hi);
        let s = c_hi.response_db(8000.0, fs_hi) - c_hi.response_db(4000.0, fs_hi);
        assert!((s + 12.0).abs() < 0.2, "slope {s}");
        // Near Nyquist the bilinear transform necessarily steepens the digital
        // slope; that is physics, not a bug, and the designer accounts for it by
        // fitting the exact digital response rather than an analog prototype.
        let s_warped = c.response_db(8000.0, fs) - c.response_db(4000.0, fs);
        assert!(s_warped < s, "warping must steepen the digital slope");
    }

    #[test]
    fn one_pole_shapes_are_unit_gain_at_their_plateaus() {
        let fs = 48000.0;
        let lp = BiquadCoeffs::one_pole_low_pass(200.0, fs);
        let hp = BiquadCoeffs::one_pole_high_pass(2000.0, fs);
        assert!(lp.response_db(1.0, fs).abs() < 1e-3);
        assert!(lp.response_db(200.0, fs) + 3.0103 < 0.01);
        assert!(hp.response_db(1.0, fs) < -60.0);
        assert!((hp.response_db(2000.0, fs) + 3.0103).abs() < 0.01);
        assert!(hp.response_db(20000.0, fs).abs() < 0.5);
    }

    #[test]
    fn shelves_built_from_one_pole_shapes_never_dip() {
        // |1 + g H|^2 = 1 + 2 g Re(H) + g^2 |H|^2. For a first order shape
        // Re(H) >= 0, so a boost (g > 0) never falls below unity and a cut
        // (g < 0) never overshoots past its plateau.
        let fs = 48000.0;
        for fc in [180.0, 3200.0] {
            for g in [0.4125f64, 0.5849, -0.6019] {
                for shape in [
                    BiquadCoeffs::one_pole_low_pass(fc, fs),
                    BiquadCoeffs::one_pole_high_pass(fc, fs),
                ] {
                    let mut lo = f64::INFINITY;
                    let mut hi = f64::NEG_INFINITY;
                    for i in 0..400 {
                        let f = 10.0 * (3000.0f64).powf(i as f64 / 399.0);
                        let (re, im) = shape.response(f, fs);
                        let (hr, hi_i) = (1.0 + g * re, g * im);
                        let db = 10.0 * (hr * hr + hi_i * hi_i).log10();
                        lo = lo.min(db);
                        hi = hi.max(db);
                        if g > 0.0 {
                            assert!(db > -1e-9, "boost dipped: {db} dB at {f} Hz");
                        } else {
                            assert!(db < 1e-9, "cut overshot: {db} dB at {f} Hz");
                        }
                    }
                    let plateau = crate::mixer::linear_to_db(1.0 + g);
                    if g > 0.0 {
                        assert!(hi <= plateau + 1e-9, "boost overshoot {hi} > {plateau}");
                    } else {
                        assert!(lo >= plateau - 1e-9, "cut undershoot {lo} < {plateau}");
                    }
                }
            }
        }
    }

    #[test]
    fn dc_blocker_kills_dc_and_passes_audio() {
        let fs = 48000.0;
        let mut b = Biquad::new(BiquadCoeffs::dc_blocker(5.0, fs));
        let mut y = 0.0;
        for _ in 0..480_000 {
            y = b.process(1.0);
        }
        assert!(y.abs() < 1e-3, "dc leak {y}");
        let c = BiquadCoeffs::dc_blocker(5.0, fs);
        assert!(c.response_db(1000.0, fs).abs() < 0.01);
    }
}
