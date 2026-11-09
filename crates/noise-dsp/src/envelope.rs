//! Sample continuous ramps and fades.
//!
//! DESIGN.md section 3.3: volume and mute changes use a sample continuous ramp
//! of roughly 10-50 ms, and start/stop fade times are configurable. Every one
//! of these is a linear ramp in *amplitude*, which is continuous by
//! construction, so no step ever reaches the output.

/// A linear amplitude ramp with a fixed number of remaining samples.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ramp {
    current: f64,
    target: f64,
    step: f64,
    remaining: u32,
}

impl Ramp {
    /// A ramp already sitting at `value`.
    pub const fn new(value: f64) -> Self {
        Self {
            current: value,
            target: value,
            step: 0.0,
            remaining: 0,
        }
    }

    /// Current value.
    #[inline]
    pub const fn value(&self) -> f64 {
        self.current
    }

    /// Value the ramp is heading for.
    #[inline]
    pub const fn target(&self) -> f64 {
        self.target
    }

    /// True while the ramp still has samples to go.
    #[inline]
    pub const fn is_ramping(&self) -> bool {
        self.remaining > 0
    }

    /// Jump to `value` immediately.
    #[inline]
    pub fn snap(&mut self, value: f64) {
        self.current = value;
        self.target = value;
        self.step = 0.0;
        self.remaining = 0;
    }

    /// Head for `value`, reaching it in `samples` steps.
    ///
    /// `samples == 0` snaps. Re-targeting mid-ramp continues from the current
    /// value, so the output never jumps.
    #[inline]
    pub fn ramp_to(&mut self, value: f64, samples: u32) {
        if samples == 0 {
            self.snap(value);
            return;
        }
        self.target = value;
        self.remaining = samples;
        self.step = (value - self.current) / samples as f64;
    }

    /// Re-target from the current value, keeping the same duration.
    #[inline]
    pub fn glide_to(&mut self, value: f64, samples: u32) {
        self.ramp_to(value, samples);
    }

    /// Advance one sample.
    #[inline]
    pub fn step(&mut self) -> f64 {
        if self.remaining > 0 {
            self.remaining -= 1;
            if self.remaining == 0 {
                self.current = self.target;
            } else {
                self.current += self.step;
            }
        }
        self.current
    }

    /// Advance `n` samples at once (used to keep several ramps in sync).
    #[inline]
    pub fn advance(&mut self, n: u32) {
        for _ in 0..n {
            self.step();
        }
    }
}

impl Default for Ramp {
    fn default() -> Self {
        Self::new(0.0)
    }
}

/// Milliseconds to samples, clamped so a ramp can never be unbounded.
#[inline]
pub fn ms_to_samples(ms: f64, fs: f64) -> u32 {
    if !ms.is_finite() || ms <= 0.0 {
        return 0;
    }
    let n = (ms * 0.001 * fs).round();
    n.clamp(0.0, 48_000.0 * 10.0) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ramp_is_monotone_and_reaches_the_target() {
        let mut r = Ramp::new(0.0);
        r.ramp_to(1.0, 100);
        let mut prev = 0.0;
        for i in 0..100 {
            let v = r.step();
            assert!(v >= prev, "step {i}");
            prev = v;
        }
        assert_eq!(r.value(), 1.0);
        assert!(!r.is_ramping());
        assert_eq!(r.step(), 1.0);
    }

    #[test]
    fn retargeting_is_continuous() {
        let mut r = Ramp::new(0.0);
        r.ramp_to(1.0, 1000);
        for _ in 0..400 {
            r.step();
        }
        let mid = r.value();
        r.ramp_to(-1.0, 100);
        let after = r.step();
        assert!((after - mid).abs() < 0.02, "{mid} -> {after}");
    }

    #[test]
    fn zero_length_ramp_snaps() {
        let mut r = Ramp::new(0.0);
        r.ramp_to(2.0, 0);
        assert_eq!(r.value(), 2.0);
    }

    #[test]
    fn ms_to_samples_is_bounded() {
        assert_eq!(ms_to_samples(10.0, 48000.0), 480);
        assert_eq!(ms_to_samples(0.0, 48000.0), 0);
        assert_eq!(ms_to_samples(f64::NAN, 48000.0), 0);
        assert_eq!(ms_to_samples(f64::INFINITY, 48000.0), 0);
        assert!(ms_to_samples(1e9, 48000.0) <= 480_000);
    }
}
