//! Realtime telemetry and stream lifecycle (DESIGN.md sections 4.2, 4.3, 4.5).
//!
//! Everything the realtime callback publishes goes through [`Telemetry`]: plain
//! atomics, no locks, no allocation, no formatting. The UI and the CLI read the
//! same struct from another thread whenever they like.

use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, Ordering};

/// Number of log-spaced buckets in the callback time histogram.
pub const TIMING_BUCKETS: usize = 48;

/// Lowest bucket boundary, in nanoseconds. Each bucket covers one octave, so
/// bucket `i` holds callbacks between `2^i` and `2^(i+1)` times this value.
pub const TIMING_BUCKET_BASE_NS: f64 = 64.0;

/// Why the stream is not currently producing audio.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum LinkState {
    /// Not started yet.
    Idle = 0,
    /// Connecting or reconnecting.
    Connecting = 1,
    /// Negotiated and running.
    Running = 2,
    /// Negotiated but paused by the transport.
    Suspended = 3,
    /// The server went away or refused us; a reconnect is scheduled.
    Error = 4,
}

impl LinkState {
    /// Decode from the atomic representation.
    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => LinkState::Connecting,
            2 => LinkState::Running,
            3 => LinkState::Suspended,
            4 => LinkState::Error,
            _ => LinkState::Idle,
        }
    }

    /// Human readable label.
    pub const fn label(self) -> &'static str {
        match self {
            LinkState::Idle => "idle",
            LinkState::Connecting => "connecting",
            LinkState::Running => "running",
            LinkState::Suspended => "suspended",
            LinkState::Error => "error",
        }
    }
}

/// Shared, lock free counters written by the realtime callback.
#[derive(Debug)]
pub struct Telemetry {
    /// Peak magnitude of the left channel over the last callback.
    pub peak_l: AtomicU32,
    /// Peak magnitude of the right channel over the last callback.
    pub peak_r: AtomicU32,
    /// RMS of the left channel over the last callback.
    pub rms_l: AtomicU32,
    /// RMS of the right channel over the last callback.
    pub rms_r: AtomicU32,
    /// Callbacks served.
    pub callbacks: AtomicU64,
    /// Frames rendered in total.
    pub frames: AtomicU64,
    /// Callbacks that could not get a buffer.
    pub xruns: AtomicU64,
    /// Non-finite samples seen.
    pub nan_events: AtomicU64,
    /// Samples the explicit limiter had to clamp.
    pub clipped: AtomicU64,
    /// Analysis-tap samples dropped because the reader was behind.
    pub tap_dropped: AtomicU64,
    /// Negotiated sample rate.
    pub sample_rate: AtomicU32,
    /// Negotiated channel count.
    pub channels: AtomicU32,
    /// Negotiated period size in frames, as last seen.
    pub quantum: AtomicU32,
    /// Stream state.
    pub link: AtomicU8,
    /// Node id once connected.
    pub node_id: AtomicU32,
    /// Enable per-callback timing measurement.
    pub profiling: AtomicBool,
    /// Set while a client actually wants the analysis tap. When clear the
    /// realtime callback skips the copy entirely, so an unwatched daemon does no
    /// work for a display nobody is looking at.
    pub tap_wanted: AtomicBool,
    /// Longest callback seen, in nanoseconds.
    pub callback_max_ns: AtomicU32,
    /// Log spaced histogram of callback durations.
    timing: [AtomicU32; TIMING_BUCKETS],
}

impl Default for Telemetry {
    fn default() -> Self {
        Self {
            peak_l: AtomicU32::new(0),
            peak_r: AtomicU32::new(0),
            rms_l: AtomicU32::new(0),
            rms_r: AtomicU32::new(0),
            callbacks: AtomicU64::new(0),
            frames: AtomicU64::new(0),
            xruns: AtomicU64::new(0),
            nan_events: AtomicU64::new(0),
            clipped: AtomicU64::new(0),
            tap_dropped: AtomicU64::new(0),
            sample_rate: AtomicU32::new(48_000),
            channels: AtomicU32::new(2),
            quantum: AtomicU32::new(0),
            link: AtomicU8::new(0),
            node_id: AtomicU32::new(u32::MAX),
            profiling: AtomicBool::new(false),
            tap_wanted: AtomicBool::new(false),
            callback_max_ns: AtomicU32::new(0),
            timing: std::array::from_fn(|_| AtomicU32::new(0)),
        }
    }
}

impl Telemetry {
    /// Publish one callback's worth of measurements.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub fn publish_block(
        &self,
        peak_l: f32,
        peak_r: f32,
        rms_l: f32,
        rms_r: f32,
        frames: u32,
        quantum: u32,
    ) {
        self.peak_l.store(peak_l.to_bits(), Ordering::Relaxed);
        self.peak_r.store(peak_r.to_bits(), Ordering::Relaxed);
        self.rms_l.store(rms_l.to_bits(), Ordering::Relaxed);
        self.rms_r.store(rms_r.to_bits(), Ordering::Relaxed);
        self.callbacks.fetch_add(1, Ordering::Relaxed);
        self.frames.fetch_add(frames as u64, Ordering::Relaxed);
        self.quantum.store(quantum, Ordering::Relaxed);
    }

    /// Record a callback that could not run.
    #[inline]
    pub fn note_xrun(&self) {
        self.xruns.fetch_add(1, Ordering::Relaxed);
        self.callbacks.fetch_add(1, Ordering::Relaxed);
    }

    /// Record non-finite samples.
    #[inline]
    pub fn note_nan(&self, count: u32) {
        if count > 0 {
            self.nan_events.fetch_add(count as u64, Ordering::Relaxed);
        }
    }

    /// Record limiter action.
    #[inline]
    pub fn note_clipped(&self, count: u32) {
        if count > 0 {
            self.clipped.fetch_add(count as u64, Ordering::Relaxed);
        }
    }

    /// Record how long a callback took. Cheap enough to always run: one atomic
    /// load, one integer log and one atomic increment.
    #[inline]
    pub fn record_timing(&self, nanos: u32) {
        if !self.profiling.load(Ordering::Relaxed) {
            return;
        }
        self.callback_max_ns.fetch_max(nanos, Ordering::Relaxed);
        let mut bucket = 0usize;
        let mut limit = TIMING_BUCKET_BASE_NS;
        while bucket + 1 < TIMING_BUCKETS && nanos as f64 >= limit {
            limit *= 2.0;
            bucket += 1;
        }
        self.timing[bucket].fetch_add(1, Ordering::Relaxed);
    }

    /// Set the stream state.
    #[inline]
    pub fn set_link(&self, state: LinkState) {
        self.link.store(state as u8, Ordering::Relaxed);
    }

    /// Read the stream state.
    pub fn link(&self) -> LinkState {
        LinkState::from_u8(self.link.load(Ordering::Relaxed))
    }

    /// Peak of the left channel.
    pub fn peak_l(&self) -> f32 {
        f32::from_bits(self.peak_l.load(Ordering::Relaxed))
    }

    /// Peak of the right channel.
    pub fn peak_r(&self) -> f32 {
        f32::from_bits(self.peak_r.load(Ordering::Relaxed))
    }

    /// RMS of the left channel.
    pub fn rms_l(&self) -> f32 {
        f32::from_bits(self.rms_l.load(Ordering::Relaxed))
    }

    /// RMS of the right channel.
    pub fn rms_r(&self) -> f32 {
        f32::from_bits(self.rms_r.load(Ordering::Relaxed))
    }

    /// Highest callback duration with a non-zero bucket.
    pub fn callback_high_percentile_ns(&self, percentile: f64) -> Option<u32> {
        let total: u64 = self
            .timing
            .iter()
            .map(|b| b.load(Ordering::Relaxed) as u64)
            .sum();
        if total == 0 {
            return None;
        }
        let target = (total as f64 * percentile.clamp(0.0, 1.0)).ceil() as u64;
        let mut seen = 0u64;
        for (i, b) in self.timing.iter().enumerate() {
            seen += b.load(Ordering::Relaxed) as u64;
            if seen >= target {
                return Some((TIMING_BUCKET_BASE_NS * 2f64.powi(i as i32)) as u32);
            }
        }
        None
    }

    /// Total callbacks recorded in the histogram.
    pub fn timing_samples(&self) -> u64 {
        self.timing
            .iter()
            .map(|b| b.load(Ordering::Relaxed) as u64)
            .sum()
    }
}

/// Convert a linear amplitude to dBFS, flooring at `-200`.
pub fn dbfs(x: f32) -> f32 {
    if x <= 0.0 || !x.is_finite() {
        -200.0
    } else {
        20.0 * x.log10()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn publish_and_read_back() {
        let t = Telemetry::default();
        t.publish_block(0.5, 0.25, 0.1, 0.05, 256, 256);
        assert_eq!(t.peak_l(), 0.5);
        assert_eq!(t.peak_r(), 0.25);
        assert!((t.rms_l() - 0.1).abs() < 1e-6);
        assert_eq!(t.callbacks.load(Ordering::Relaxed), 1);
        assert_eq!(t.frames.load(Ordering::Relaxed), 256);
    }

    #[test]
    fn timing_histogram_reports_a_percentile() {
        let t = Telemetry::default();
        t.profiling.store(true, Ordering::Relaxed);
        for _ in 0..1000 {
            t.record_timing(200);
        }
        for _ in 0..10 {
            t.record_timing(20_000);
        }
        assert_eq!(t.timing_samples(), 1010);
        let p50 = t.callback_high_percentile_ns(0.5).unwrap();
        let p999 = t.callback_high_percentile_ns(0.999).unwrap();
        assert!(p50 < 512, "p50 {p50}");
        assert!(p999 >= p50, "p99.9 {p999} < p50 {p50}");
        assert!(t.callback_max_ns.load(Ordering::Relaxed) >= 20_000);
    }

    #[test]
    fn profiling_off_costs_nothing_observable() {
        let t = Telemetry::default();
        t.record_timing(10_000);
        assert_eq!(t.timing_samples(), 0);
    }

    #[test]
    fn link_state_round_trips() {
        let t = Telemetry::default();
        for s in [
            LinkState::Idle,
            LinkState::Connecting,
            LinkState::Running,
            LinkState::Suspended,
            LinkState::Error,
        ] {
            t.set_link(s);
            assert_eq!(t.link(), s);
        }
    }

    #[test]
    fn dbfs_floor() {
        assert_eq!(dbfs(0.0), -200.0);
        assert!((dbfs(1.0)).abs() < 1e-6);
        assert!((dbfs(0.5) + 6.0206).abs() < 0.001);
    }
}
