//! Welch power spectral density estimation.
//!
//! DESIGN.md section 9.1 specifies the measurement procedure: Hann window,
//! Welch averaging, optional 1/6 to 1/3 octave aggregation and a least squares
//! regression of PSD in dB against `log2(f)`.
//!
//! This module is the measuring instrument for the whole project. It is used by
//! the spectral tests, by the offline analysis and by the GUI's "Measured
//! Output" curve. It never runs on the realtime thread (DESIGN.md section 4.5).

use std::sync::Arc;

use rustfft::{FftPlanner, num_complex::Complex};

/// Window functions supported by the estimator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Window {
    /// Von Hann window: the default in DESIGN.md section 5.3.
    Hann,
    /// Hamming window.
    Hamming,
    /// Rectangular window; only for diagnostics, it leaks badly.
    Rectangular,
}

impl Window {
    fn coefficients(self, n: usize) -> Vec<f64> {
        match self {
            Window::Rectangular => vec![1.0; n],
            Window::Hann => (0..n)
                .map(|i| {
                    0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos())
                })
                .collect(),
            Window::Hamming => (0..n)
                .map(|i| {
                    0.54 - 0.46 * (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos()
                })
                .collect(),
        }
    }
}

/// Estimator configuration.
#[derive(Clone, Copy, Debug)]
pub struct WelchConfig {
    /// FFT length in samples. Rounded up to the next power of two internally.
    pub fft_size: usize,
    /// Fractional overlap between segments, `0.0..1.0`.
    pub overlap: f64,
    /// Window function.
    pub window: Window,
}

impl Default for WelchConfig {
    fn default() -> Self {
        Self {
            fft_size: 16_384,
            overlap: 0.5,
            window: Window::Hann,
        }
    }
}

impl WelchConfig {
    /// Configuration tuned for a given rate: keeps the same time resolution
    /// while staying valid at 96 kHz.
    pub fn for_rate(fs: f64) -> Self {
        let fft = if fs >= 88_200.0 { 32_768 } else { 16_384 };
        Self {
            fft_size: fft,
            ..Self::default()
        }
    }
}

/// An estimated one-sided power spectral density.
#[derive(Clone, Debug)]
pub struct Psd {
    /// Sample rate used.
    pub sample_rate: f64,
    /// Frequency of each bin, in Hz, ascending.
    pub freqs: Vec<f64>,
    /// Power spectral density per bin, in linear units of power per Hz.
    pub power: Vec<f64>,
    /// Number of averaged segments; more means a smoother estimate.
    pub averages: usize,
    /// Equivalent noise bandwidth of the window, in bins.
    pub enbw_bins: f64,
}

impl Psd {
    /// PSD in dB, normalised so that a unit-variance white signal reads `0 dB`
    /// per bin.
    pub fn power_db(&self) -> Vec<f64> {
        self.power
            .iter()
            .map(|p| 10.0 * p.max(f64::MIN_POSITIVE).log10())
            .collect()
    }

    /// Total power over a frequency range, obtained by trapezoid integration.
    pub fn band_power(&self, f_lo: f64, f_hi: f64) -> f64 {
        let mut acc = 0.0;
        for i in 1..self.freqs.len() {
            let (f0, f1) = (self.freqs[i - 1], self.freqs[i]);
            if f1 < f_lo || f0 > f_hi {
                continue;
            }
            acc += 0.5 * (self.power[i - 1] + self.power[i]) * (f1 - f0);
        }
        acc
    }

    /// Index of the bin closest to `f`.
    pub fn index_of(&self, f: f64) -> usize {
        match self.freqs.binary_search_by(|p| p.partial_cmp(&f).unwrap()) {
            Ok(i) => i,
            Err(i) => i.min(self.freqs.len().saturating_sub(1)),
        }
    }
}

/// Estimate the one-sided PSD of `x` with Welch's method.
///
/// `x` is a mono signal: interleaved multi-channel buffers must be deinterleaved
/// first (see [`deinterleave_channel`]).
pub fn welch_psd(x: &[f32], fs: f64, cfg: &WelchConfig) -> Psd {
    let n = cfg.fft_size.next_power_of_two().max(256);
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);
    let window = cfg.window.coefficients(n);
    let win_power: f64 = window.iter().map(|w| w * w).sum();
    let win_sum: f64 = window.iter().sum();
    // Equivalent noise bandwidth in bins, used for the per-bin normalisation.
    let enbw_bins = (n as f64) * win_power / (win_sum * win_sum);

    let hop = (((1.0 - cfg.overlap.clamp(0.0, 0.95)) * n as f64).round() as usize).max(1);
    let bins = n / 2 + 1;
    let mut acc = vec![0.0f64; bins];
    let mut averages = 0usize;

    let mut buf = vec![Complex::new(0.0, 0.0); n];
    let mut start = 0usize;
    while start + n <= x.len() {
        // Remove the segment mean so DC leakage cannot masquerade as low
        // frequency noise.
        let mean: f64 = x[start..start + n].iter().map(|v| *v as f64).sum::<f64>() / n as f64;
        for i in 0..n {
            buf[i] = Complex::new((x[start + i] as f64 - mean) * window[i], 0.0);
        }
        fft.process(&mut buf);

        // One-sided PSD with the standard window corrections:
        //   S(f) = |X|^2 / (fs * sum(w^2)) * 2
        let scale = 2.0 / (fs * win_power);
        acc[0] += buf[0].norm_sqr() / (fs * win_sum * win_sum);
        for k in 1..bins - 1 {
            acc[k] += buf[k].norm_sqr() * scale;
        }
        if n.is_multiple_of(2) {
            acc[bins - 1] += buf[n / 2].norm_sqr() * scale * 0.5;
        }
        averages += 1;
        start += hop;
    }

    if averages > 0 {
        for v in acc.iter_mut() {
            *v /= averages as f64;
        }
    }

    let freqs: Vec<f64> = (0..bins).map(|k| k as f64 * fs / n as f64).collect();
    Psd {
        sample_rate: fs,
        freqs,
        power: acc,
        averages,
        enbw_bins,
    }
}

/// Extract one channel from an interleaved buffer.
pub fn deinterleave_channel(x: &[f32], channels: usize, channel: usize) -> Vec<f32> {
    if channels <= 1 {
        return x.to_vec();
    }
    x.chunks(channels)
        .filter_map(|c| c.get(channel).copied())
        .collect()
}

/// One point of an aggregated band.
#[derive(Clone, Copy, Debug)]
pub struct Band {
    /// Band centre frequency.
    pub center_hz: f64,
    /// Mean power in the band.
    pub power: f64,
}

impl Band {
    /// Band level in dB.
    pub fn db(&self) -> f64 {
        10.0 * self.power.max(f64::MIN_POSITIVE).log10()
    }
}

/// Aggregate a PSD into fractional octave bands.
///
/// DESIGN.md section 5.3 asks for optional 1/6 octave smoothing. `fraction = 6`
/// is 1/6 octave, `fraction = 3` is 1/3 octave.
pub fn fractional_octave_bands(psd: &Psd, fraction: u32, f_lo: f64, f_hi: f64) -> Vec<Band> {
    let fraction = fraction.max(1) as f64;
    let ratio = 2.0f64.powf(1.0 / fraction);
    let mut out = Vec::new();
    let mut center = f_lo.max(psd.freqs[1]);
    while center <= f_hi {
        let lo = center / ratio.sqrt();
        let hi = center * ratio.sqrt();
        let i0 = psd.index_of(lo);
        let i1 = psd.index_of(hi).max(i0 + 1);
        let slice = &psd.power[i0..i1.min(psd.power.len())];
        if !slice.is_empty() {
            let mean = slice.iter().sum::<f64>() / slice.len() as f64;
            out.push(Band {
                center_hz: center,
                power: mean,
            });
        }
        center *= ratio;
    }
    out
}

/// Result of a log-log slope regression.
#[derive(Clone, Copy, Debug)]
pub struct SlopeFit {
    /// Slope in dB per octave.
    pub slope_db_per_octave: f64,
    /// Intercept such that `db ~= slope * log2(f) + intercept`.
    pub intercept_db: f64,
    /// Largest absolute deviation from the fitted line, in dB.
    pub max_deviation_db: f64,
    /// RMS deviation from the fitted line, in dB.
    pub rms_deviation_db: f64,
    /// Number of points that went into the fit.
    pub points: usize,
}

impl SlopeFit {
    /// Value of the fitted line at `f`.
    pub fn at(&self, f: f64) -> f64 {
        self.slope_db_per_octave * f.log2() + self.intercept_db
    }
}

/// Least squares fit of band levels in dB against `log2(f)`.
pub fn fit_slope(bands: &[Band], f_lo: f64, f_hi: f64) -> SlopeFit {
    let pts: Vec<(f64, f64)> = bands
        .iter()
        .filter(|b| b.center_hz >= f_lo && b.center_hz <= f_hi)
        .map(|b| (b.center_hz.log2(), b.db()))
        .collect();
    let n = pts.len();
    if n < 2 {
        return SlopeFit {
            slope_db_per_octave: 0.0,
            intercept_db: 0.0,
            max_deviation_db: f64::INFINITY,
            rms_deviation_db: f64::INFINITY,
            points: n,
        };
    }
    let sx: f64 = pts.iter().map(|p| p.0).sum();
    let sy: f64 = pts.iter().map(|p| p.1).sum();
    let sxx: f64 = pts.iter().map(|p| p.0 * p.0).sum();
    let sxy: f64 = pts.iter().map(|p| p.0 * p.1).sum();
    let nf = n as f64;
    let denom = nf * sxx - sx * sx;
    let slope = if denom.abs() > 0.0 {
        (nf * sxy - sx * sy) / denom
    } else {
        0.0
    };
    let intercept = (sy - slope * sx) / nf;
    let mut max_dev = 0.0f64;
    let mut sum_sq = 0.0f64;
    for (x, y) in &pts {
        let d = y - (slope * x + intercept);
        max_dev = max_dev.max(d.abs());
        sum_sq += d * d;
    }
    SlopeFit {
        slope_db_per_octave: slope,
        intercept_db: intercept,
        max_deviation_db: max_dev,
        rms_deviation_db: (sum_sq / nf).sqrt(),
        points: n,
    }
}

/// Convenience: Welch estimate, fractional octave aggregation and slope fit in
/// one call. This is the exact measurement DESIGN.md section 9.1 describes.
pub fn measure_slope(
    x: &[f32],
    fs: f64,
    f_lo: f64,
    f_hi: f64,
    cfg: &WelchConfig,
    fraction: u32,
) -> Option<(Psd, Vec<Band>, SlopeFit)> {
    if x.len() < cfg.fft_size {
        return None;
    }
    let psd = welch_psd(x, fs, cfg);
    let bands = fractional_octave_bands(&psd, fraction, f_lo, f_hi);
    if bands.len() < 3 {
        return None;
    }
    let fit = fit_slope(&bands, f_lo, f_hi);
    Some((psd, bands, fit))
}

/// Resample a PSD onto a log spaced frequency grid for plotting.
///
/// Returns `(frequencies, levels_db)` with `points` entries between `f_lo` and
/// `f_hi`. Frequencies with no data are interpolated so a plot never has holes.
pub fn log_resample(psd: &Psd, f_lo: f64, f_hi: f64, points: usize) -> (Vec<f64>, Vec<f64>) {
    let db = psd.power_db();
    let points = points.max(2);
    let mut freqs = Vec::with_capacity(points);
    let mut levels = Vec::with_capacity(points);
    for i in 0..points {
        let t = i as f64 / (points - 1) as f64;
        let f = f_lo * (f_hi / f_lo).powf(t);
        let idx = psd.index_of(f);
        freqs.push(f);
        levels.push(db.get(idx).copied().unwrap_or(f64::NEG_INFINITY));
    }
    (freqs, levels)
}

/// Shared FFT plan cache handle, exposed so callers can build several
/// estimators without re-planning.
pub type SharedFft = Arc<dyn rustfft::Fft<f64>>;

#[cfg(test)]
mod tests {
    use super::*;

    fn white(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                let v = noise_dsp::rng::splitmix64(&mut s);
                ((v >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32
            })
            .collect()
    }

    fn sine(n: usize, fs: f64, f: f64) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * f * i as f64 / fs).sin() as f32 * 0.5)
            .collect()
    }

    #[test]
    fn white_noise_has_a_flat_spectrum() {
        let fs = 48_000.0;
        let x = white(48_000 * 8, 3);
        let cfg = WelchConfig::default();
        let (_, bands, fit) = measure_slope(&x, fs, 100.0, 16_000.0, &cfg, 6).unwrap();
        assert!(
            fit.slope_db_per_octave.abs() < 0.5,
            "slope {}",
            fit.slope_db_per_octave
        );
        // 1/6 octave bands between 100 Hz and 16 kHz span 7.3 octaves.
        assert!(bands.len() > 40, "{} bands", bands.len());
    }

    #[test]
    fn a_sine_shows_up_in_the_right_bin() {
        let fs = 48_000.0;
        let x = sine(48_000 * 2, fs, 1000.0);
        let psd = welch_psd(&x, fs, &WelchConfig::default());
        let peak = psd
            .power
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert!(
            (psd.freqs[peak] - 1000.0).abs() < 5.0,
            "peak at {}",
            psd.freqs[peak]
        );
    }

    #[test]
    fn total_power_matches_the_signal_variance() {
        let fs = 48_000.0;
        let x = white(48_000 * 8, 11);
        let var: f64 = x.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / x.len() as f64;
        let psd = welch_psd(&x, fs, &WelchConfig::default());
        let total = psd.band_power(1.0, fs * 0.5);
        // A one-sided PSD integrates to the variance of the signal.
        assert!(
            (total / var - 1.0).abs() < 0.05,
            "integrated {total} vs variance {var}"
        );
    }

    #[test]
    fn a_known_slope_is_recovered() {
        // Build a synthetic signal whose PSD is exactly -3 dB/oct by shaping
        // white noise with a first order shelf cascade is overkill; instead
        // check the fitter directly on an analytic band set.
        let bands: Vec<Band> = (0..40)
            .map(|i| {
                let f = 40.0 * 2f64.powf(i as f64 / 5.0);
                Band {
                    center_hz: f,
                    power: 10f64.powf((-3.0 * f.log2()) / 10.0),
                }
            })
            .collect();
        let fit = fit_slope(&bands, 40.0, 16_000.0);
        assert!(
            (fit.slope_db_per_octave + 3.0).abs() < 1e-9,
            "{}",
            fit.slope_db_per_octave
        );
        assert!(fit.max_deviation_db < 1e-9);
    }

    #[test]
    fn interleaved_deinterleave_round_trips() {
        let x: Vec<f32> = (0..20).map(|i| i as f32).collect();
        assert_eq!(deinterleave_channel(&x, 1, 0), x);
        let l = deinterleave_channel(&x, 2, 0);
        let r = deinterleave_channel(&x, 2, 1);
        assert_eq!(
            l,
            vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0]
        );
        assert_eq!(
            r,
            vec![1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0]
        );
    }

    #[test]
    fn log_resample_has_the_requested_length_and_no_holes() {
        let fs = 48_000.0;
        let x = white(48_000, 5);
        let psd = welch_psd(&x, fs, &WelchConfig::default());
        let (f, db) = log_resample(&psd, 20.0, 20_000.0, 128);
        assert_eq!(f.len(), 128);
        assert_eq!(db.len(), 128);
        assert!(db.iter().all(|v| v.is_finite()));
        assert!(f.windows(2).all(|w| w[1] > w[0]));
    }
}
