//! Offline design of the colored-noise shaping filter.
//!
//! # Method
//!
//! DESIGN.md section 1.4 requires the standard path to be "a cascade of stable
//! IIR filters (SOS / biquad, a group of shelving filters)" that approximates a
//! target log-log magnitude slope with the fewest sections that meet the
//! accuracy goal.
//!
//! The target is a fractional power of frequency:
//!
//! ```text
//! |H(f)| ~ (f / f_ref)^(-alpha/2) = s^nu   with nu = -alpha / 2
//! ```
//!
//! `s^nu` is approximated over a design band with the **Oustaloup recursive
//! approximation**, which is exactly a cascade of `2N + 1` first order sections
//! with geometrically spaced real poles and zeros:
//!
//! ```text
//!                 N
//! s^nu  ~  K  *  prod  (s + wz_k) / (s + wp_k)
//!               k = -N
//!
//! wz_k = wb * (wh/wb)^((k + N + (1 - nu)/2) / (2N + 1))
//! wp_k = wb * (wh/wb)^((k + N + (1 + nu)/2) / (2N + 1))
//! ```
//!
//! Each first order section is rewritten in the DC-normalised form
//! `(1 + s/wz) / (1 + s/wp)`, so every section has unity gain at DC and the
//! whole cascade has unity gain at DC. That is deliberately the *regularised*
//! behaviour the design asks for in section 1.2: below the design band the
//! response flattens instead of diverging, so Brown never has infinite DC gain
//! and no integrator state can run away.
//!
//! ## Analog to digital
//!
//! Band edges are pre-warped with `W(f) = 2 fs tan(pi f / fs)` before the
//! recursive expansion, and each analog corner `w` is mapped with the bilinear
//! transform:
//!
//! ```text
//! k    = 2 fs / w
//! H(z) = ((1 + kz) + (1 - kz) z^-1) / ((1 + kp) + (1 - kp) z^-1)
//! ```
//!
//! which gives `H(1) = 1` exactly (unity DC gain, no numerical blow-up) and
//! `H(-1) = kz / kp` at Nyquist. `k > 0` for every corner inside Nyquist, so
//! `|a1| < 1` holds by construction and no section can be unstable.
//!
//! ## Section count and honesty
//!
//! [`design_color`] scans Oustaloup orders from smallest to largest and keeps
//! the cheapest cascade whose *shape* and *slope* both meet the criteria in
//! DESIGN.md section 9.1 over the validation band. If none does, the best
//! measured candidate is returned with [`DesignReport::within_tolerance`] set to
//! `false`; the reported accuracy is always a measurement, never a claim.

use core::f64::consts::PI;

use crate::colors::{self, ALPHA_MAX, ALPHA_MIN};
use crate::sos::{BiquadCoeffs, MAX_SECTIONS, SosCoeffs};

/// Maximum number of probe frequencies used for evaluation.
pub const MAX_PROBE: usize = 256;

/// How far below the promised band the `1/f^alpha` law is regularised.
///
/// The response is flat at the DC gain below `band_lo / REGULARISATION_RATIO`,
/// which is what stops Brown from integrating to infinite DC gain
/// (DESIGN.md section 1.2).
pub const REGULARISATION_RATIO: f64 = 4.0;

/// Which part of the spectrum a design promises to be accurate in.
///
/// DESIGN.md section 1.2: 40 Hz to `min(16000, 0.42 * Fs)`.
pub fn validation_band(fs: f64) -> (f64, f64) {
    (40.0, 16000.0_f64.min(0.42 * fs))
}

/// Tuning knobs of the designer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DesignOptions {
    /// Tolerance for the *shape* error over the validation band, in dB.
    pub tolerance_db: f64,
    /// Tolerance for the slope error, in dB per octave.
    pub slope_tolerance_db_per_octave: f64,
    /// Smallest Oustaloup order to try.
    pub min_order: usize,
    /// Largest Oustaloup order to try.
    pub max_order: usize,
    /// Run the bounded coordinate-descent refinement after the recursive
    /// initialisation. Never runs on the realtime thread.
    pub refine: bool,
    /// Refinement sweeps.
    pub refine_sweeps: usize,
    /// Number of log-spaced probe frequencies used for the final report.
    pub probe_points: usize,
}

impl Default for DesignOptions {
    fn default() -> Self {
        Self {
            tolerance_db: 1.5,
            slope_tolerance_db_per_octave: 0.30,
            min_order: 2,
            max_order: 7,
            refine: true,
            refine_sweeps: 40,
            probe_points: 241,
        }
    }
}

/// Errors that make a design unusable. Callers must keep the previous known
/// good coefficients when one of these is returned (DESIGN.md section 1.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DesignError {
    /// Sample rate is not usable.
    BadSampleRate,
    /// `alpha` was NaN or infinite.
    BadAlpha,
    /// The resulting cascade had non-finite coefficients or an unstable pole.
    Unstable,
    /// More than [`MAX_SECTIONS`] biquads would be required.
    TooManySections,
}

impl core::fmt::Display for DesignError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            DesignError::BadSampleRate => "sample rate out of range",
            DesignError::BadAlpha => "alpha is not a finite number",
            DesignError::Unstable => "designed cascade is unstable or non-finite",
            DesignError::TooManySections => "design needs more sections than supported",
        })
    }
}

impl std::error::Error for DesignError {}

/// What the designer actually achieved. Published verbatim by the UI so that a
/// filtered colour is never advertised as the pristine mathematical one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DesignReport {
    /// Exponent the design was asked for.
    pub alpha: f64,
    /// Sample rate the design was made for.
    pub sample_rate: f64,
    /// Lower edge of the promised band.
    pub band_lo_hz: f64,
    /// Upper edge of the promised band.
    pub band_hi_hz: f64,
    /// Number of biquad sections in the returned cascade.
    pub sections: usize,
    /// Oustaloup order the cascade came from (first order sections = `2N+1`).
    pub order: usize,
    /// Mean offset of the response from the ideal straight line. A free overall
    /// gain, absorbed by the RMS calibration.
    pub mean_offset_db: f64,
    /// Peak-to-peak *shape* error after removing the free gain, in dB.
    pub ripple_db: f64,
    /// Largest absolute deviation from the ideal line after removing the free
    /// gain.
    pub max_abs_dev_db: f64,
    /// Least squares slope of the measured response, dB per octave.
    pub slope_db_per_octave: f64,
    /// `slope_db_per_octave` minus the theoretical slope.
    pub slope_error_db_per_octave: f64,
    /// True when both the shape and the slope criteria were met.
    pub within_tolerance: bool,
}

/// Result of a colour design.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColorDesign {
    pub coeffs: SosCoeffs,
    pub report: DesignReport,
}

/// First order digital section `(b0, b1, a1)`.
pub type FirstOrder = (f64, f64, f64);

/// Pre-warp a digital frequency to the analog frequency the bilinear transform
/// maps it to.
#[inline]
pub fn prewarp(f: f64, fs: f64) -> f64 {
    2.0 * fs * (PI * f / fs).tan()
}

/// Largest digital frequency that can be pre-warped safely.
#[inline]
pub fn max_design_hz(fs: f64) -> f64 {
    0.47 * fs
}

/// Build a DC-normalised first order section from analog corners.
#[inline]
fn first_order_digital(w_zero: f64, w_pole: f64, fs: f64) -> FirstOrder {
    let kz = 2.0 * fs / w_zero;
    let kp = 2.0 * fs / w_pole;
    let d = 1.0 + kp;
    ((1.0 + kz) / d, (1.0 - kz) / d, (1.0 - kp) / d)
}

/// Geometric corners of the Oustaloup approximation, in the pre-warped domain.
fn oustaloup_corners(
    nu: f64,
    w_lo: f64,
    w_hi: f64,
    order: usize,
    out: &mut Vec<FirstOrder>,
    fs: f64,
) {
    out.clear();
    let n = order as f64;
    let m = (2 * order + 1) as f64;
    let ratio = w_hi / w_lo;
    for i in 0..=(2 * order) {
        let k = i as f64 - n;
        let wz = w_lo * ratio.powf((k + n + 0.5 * (1.0 - nu)) / m);
        let wp = w_lo * ratio.powf((k + n + 0.5 * (1.0 + nu)) / m);
        out.push(first_order_digital(wz, wp, fs));
    }
}

/// Pack first order digital sections into the smallest biquad cascade.
fn pack(
    first_order: &[FirstOrder],
    out: &mut [BiquadCoeffs; MAX_SECTIONS],
) -> Result<SosCoeffs, DesignError> {
    let n_biquads = first_order.len().div_ceil(2);
    if n_biquads > MAX_SECTIONS {
        return Err(DesignError::TooManySections);
    }
    for (i, pair) in first_order.chunks(2).enumerate() {
        out[i] = if pair.len() == 2 {
            BiquadCoeffs::pair_first_order(pair[0], pair[1])
        } else {
            BiquadCoeffs::first_order(pair[0].0, pair[0].1, pair[0].2)
        };
    }
    Ok(SosCoeffs::new(&out[..n_biquads]))
}

/// Log-spaced probe frequencies written into `out`; returns the count.
pub fn log_grid_into(lo: f64, hi: f64, points: usize, out: &mut [f64]) -> usize {
    let n = points.clamp(2, out.len());
    let l0 = lo.ln();
    let l1 = hi.ln();
    for (i, slot) in out[..n].iter_mut().enumerate() {
        let t = i as f64 / (n - 1) as f64;
        *slot = (l0 + (l1 - l0) * t).exp();
    }
    n
}

/// Log-spaced probe frequencies, allocating.
pub fn log_grid(lo: f64, hi: f64, points: usize) -> Vec<f64> {
    let mut v = vec![0.0; points.max(2)];
    let n = log_grid_into(lo, hi, points, &mut v);
    v.truncate(n);
    v
}

/// Squared magnitude of a cascade of *first order* sections at `f`.
#[inline]
fn first_order_magnitude2(secs: &[FirstOrder], f: f64, fs: f64) -> f64 {
    let w = 2.0 * PI * f / fs;
    let (s1, c1) = w.sin_cos();
    let mut m = 1.0f64;
    for &(b0, b1, a1) in secs {
        let nr = b0 + b1 * c1;
        let ni = -b1 * s1;
        let dr = 1.0 + a1 * c1;
        let di = -a1 * s1;
        m *= (nr * nr + ni * ni) / (dr * dr + di * di);
    }
    m
}

/// Shape statistics of a first order cascade, avoiding a biquad pack.
fn evaluate_first_order(
    secs: &[FirstOrder],
    alpha: f64,
    fs: f64,
    grid: &[f64],
    scratch: &mut [f64; MAX_PROBE],
) -> (f64, f64, f64) {
    let n = grid.len().min(MAX_PROBE);
    let mut mean = 0.0;
    for i in 0..n {
        let f = grid[i];
        let measured = 10.0
            * first_order_magnitude2(secs, f, fs)
                .max(f64::MIN_POSITIVE)
                .log10();
        let ideal = colors::amplitude_db(alpha, f, colors::F_REF_HZ);
        scratch[i] = measured - ideal;
        mean += scratch[i];
    }
    mean /= n as f64;
    let mut max_abs = 0.0f64;
    for e in scratch[..n].iter_mut() {
        *e -= mean;
        max_abs = max_abs.max(e.abs());
    }
    (mean, max_abs * 2.0, max_abs)
}

/// Shape statistics of a finished biquad cascade.
fn evaluate(
    coeffs: &SosCoeffs,
    alpha: f64,
    fs: f64,
    grid: &[f64],
    scratch: &mut [f64; MAX_PROBE],
) -> (f64, f64, f64, f64) {
    let n = grid.len().min(MAX_PROBE);
    let mut mean = 0.0;
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    for i in 0..n {
        let f = grid[i];
        let measured = coeffs.response_db(f, fs);
        let ideal = colors::amplitude_db(alpha, f, colors::F_REF_HZ);
        scratch[i] = measured - ideal;
        mean += scratch[i];
        let x = f.log2();
        sx += x;
        sy += measured;
        sxx += x * x;
        sxy += x * measured;
    }
    mean /= n as f64;
    let mut max_abs = 0.0f64;
    for e in scratch[..n].iter_mut() {
        *e -= mean;
        max_abs = max_abs.max(e.abs());
    }
    let nf = n as f64;
    let denom = nf * sxx - sx * sx;
    let slope = if denom.abs() > 0.0 {
        (nf * sxy - sx * sy) / denom
    } else {
        0.0
    };
    (mean, max_abs * 2.0, max_abs, slope)
}

/// Analog corners → digital first order sections.
fn corners_to_sections(log_z: &[f64], log_p: &[f64], fs: f64, out: &mut Vec<FirstOrder>) {
    out.clear();
    for (&z, &p) in log_z.iter().zip(log_p.iter()) {
        out.push(first_order_digital(z.exp(), p.exp(), fs));
    }
}

/// Model response in dB of a cascade of first order shelves at digital
/// frequencies `grid`, using the exact bilinear identity
/// `|H(f)| = prod |1 + j W / wz| / |1 + j W / wp|` with `W = 2 fs tan(pi f / fs)`.
///
/// `W` is precomputed by the caller, and the optional weight array scales each
/// residual so the fit is tight inside the promised band and slack outside it.
fn model_db(w: &[f64], log_z: &[f64], log_p: &[f64]) -> Vec<f64> {
    let k = 10.0 / std::f64::consts::LN_10;
    let mut out = vec![0.0f64; w.len()];
    for (i, &wi) in w.iter().enumerate() {
        let mut acc = 0.0f64;
        for &lz in log_z {
            let x = wi / lz.exp();
            acc += (1.0 + x * x).ln();
        }
        for &lp in log_p {
            let x = wi / lp.exp();
            acc -= (1.0 + x * x).ln();
        }
        out[i] = k * acc;
    }
    out
}

/// Levenberg-Marquardt fit of the pole/zero corner positions plus a free gain.
///
/// The bilinear transform warps the frequency axis, so an analog-domain power
/// law is *not* a straight line against digital log frequency. Rather than
/// pretending otherwise, the corners are fitted directly against the exact
/// digital magnitude response so that the closed-loop spectrum really is
/// `f^(-alpha/2)` where DESIGN.md section 9.1 measures it.
///
/// Deterministic and offline only.
fn lm_fit(
    log_z: &mut [f64],
    log_p: &mut [f64],
    w: &[f64],
    weights: &[f64],
    ideal: &[f64],
    iters: usize,
) -> f64 {
    let m = log_z.len();
    // There is deliberately no free gain parameter. Every section is DC
    // normalised, so the cascade's gain at DC is exactly 1 and the overall
    // absolute level is *not* free: the target curve is anchored to that
    // structural fact. Leaving the gain free let the fit buy in-band accuracy by
    // inventing a large subsonic bump below the promised band, which then
    // dominated the RMS calibration and made the audible band far too quiet.
    let np = 2 * m;
    let n = w.len();
    let mut p = vec![0.0f64; np];
    p[..m].copy_from_slice(&log_z[..m]);
    p[m..2 * m].copy_from_slice(&log_p[..m]);

    let ln10_over_10 = std::f64::consts::LN_10 / 10.0;
    let j0 = 10.0 / std::f64::consts::LN_10;

    let mut r = vec![0.0f64; n];
    let mut jac = vec![0.0f64; n * np];
    let mut a = vec![0.0f64; np * np];
    let mut g = vec![0.0f64; np];
    let mut delta = vec![0.0f64; np];
    let mut trial_z = vec![0.0f64; m];
    let mut trial_p = vec![0.0f64; m];

    let mut cost = f64::INFINITY;
    let mut lambda = 1e-3f64;

    for _ in 0..iters {
        trial_z[..m].copy_from_slice(&p[..m]);
        trial_p[..m].copy_from_slice(&p[m..2 * m]);
        let model = model_db(w, &trial_z, &trial_p);
        let mut c = 0.0f64;
        for i in 0..n {
            let ri = weights[i] * (model[i] - ideal[i]);
            r[i] = ri;
            c += ri * ri;
        }

        // Jacobian in dB per ln(corner).
        for i in 0..n {
            let wi = w[i];
            let base = i * np;
            for k in 0..m {
                let x = wi / trial_z[k].exp();
                let t = x * x / (1.0 + x * x);
                jac[base + k] = -2.0 * j0 * t * weights[i];
                let u = wi / trial_p[k].exp();
                let s = u * u / (1.0 + u * u);
                jac[base + m + k] = 2.0 * j0 * s * weights[i];
            }
        }

        // Normal equations.
        a.fill(0.0);
        g.fill(0.0);
        for (i, &ri) in r.iter().enumerate().take(n) {
            let base = i * np;
            for x in 0..np {
                let jx = jac[base + x];
                if jx == 0.0 {
                    continue;
                }
                g[x] -= jx * ri;
                for y in x..np {
                    a[x * np + y] += jx * jac[base + y];
                }
            }
        }
        for x in 0..np {
            for y in 0..x {
                a[x * np + y] = a[y * np + x];
            }
        }

        let mut accepted = false;
        for _ in 0..12 {
            let mut damped = a.clone();
            for x in 0..np {
                let d = damped[x * np + x].abs().max(1e-12);
                damped[x * np + x] += lambda * d;
            }
            if solve_in_place(&mut damped, &mut delta, &g, np) {
                for i in 0..m {
                    trial_z[i] = p[i] + delta[i];
                    trial_p[i] = p[m + i] + delta[m + i];
                }
                if trial_z.iter().chain(trial_p.iter()).all(|v| v.is_finite()) {
                    let model = model_db(w, &trial_z, &trial_p);
                    let mut c2 = 0.0f64;
                    for i in 0..n {
                        let d = weights[i] * (model[i] - ideal[i]);
                        c2 += d * d;
                    }
                    if c2 < c {
                        p[..m].copy_from_slice(&trial_z[..m]);
                        p[m..2 * m].copy_from_slice(&trial_p[..m]);
                        cost = c2.min(cost);
                        lambda = (lambda / 3.0).max(1e-12);
                        accepted = true;
                        break;
                    }
                }
            }
            lambda *= 6.0;
            if lambda > 1e12 {
                break;
            }
        }
        if !accepted {
            break;
        }
        let _ = ln10_over_10;
    }

    for (i, slot) in log_z.iter_mut().enumerate() {
        *slot = p[i].clamp(MIN_LOG_W, MAX_LOG_W);
    }
    for (i, slot) in log_p.iter_mut().enumerate() {
        *slot = p[m + i].clamp(MIN_LOG_W, MAX_LOG_W);
    }
    if cost.is_finite() {
        cost.sqrt()
    } else {
        f64::INFINITY
    }
}

/// Dense solve of `A x = b` with partial pivoting; `A` is destroyed.
fn solve_in_place(a: &mut [f64], x: &mut [f64], b: &[f64], n: usize) -> bool {
    let mut rhs = b.to_vec();
    for col in 0..n {
        let mut piv = col;
        let mut best = a[col * n + col].abs();
        for row in (col + 1)..n {
            let v = a[row * n + col].abs();
            if v > best {
                best = v;
                piv = row;
            }
        }
        if best <= 1e-14 || !best.is_finite() {
            return false;
        }
        if piv != col {
            for k in 0..n {
                a.swap(col * n + k, piv * n + k);
            }
            rhs.swap(col, piv);
        }
        let d = a[col * n + col];
        for row in (col + 1)..n {
            let f = a[row * n + col] / d;
            if f == 0.0 {
                continue;
            }
            for k in col..n {
                a[row * n + k] -= f * a[col * n + k];
            }
            rhs[row] -= f * rhs[col];
        }
    }
    for row in (0..n).rev() {
        let mut acc = rhs[row];
        for k in (row + 1)..n {
            acc -= a[row * n + k] * x[k];
        }
        x[row] = acc / a[row * n + row];
    }
    x.iter().all(|v| v.is_finite())
}

/// Sane bounds for `ln(w)`: 0.02 Hz up to just under Nyquist.
const MIN_LOG_W: f64 = -5.0;
const MAX_LOG_W: f64 = 18.0; // e^18 ~ 65 MHz, far above any supported rate

/// Design band handed to the recursive initialisation, in hertz.
///
/// The recursive approximation loses accuracy over roughly one section spacing
/// near its own edges, so the band is stretched beyond the promised band as far
/// as Nyquist allows.
fn recursive_band(_order: usize, vb_lo: f64, vb_hi: f64, fs: f64) -> (f64, f64) {
    let f_lo = (vb_lo * 0.25).max(1.0);
    let f_hi = (vb_hi * 2.6).min(max_design_hz(fs));
    (f_lo, f_hi.max(f_lo * 4.0))
}

/// Design the shaping filter for `alpha` at `fs` with default options.
pub fn design_color(alpha: f64, fs: f64) -> Result<ColorDesign, DesignError> {
    design_color_with(alpha, fs, DesignOptions::default())
}

/// Design the shaping filter with explicit options.
pub fn design_color_with(
    alpha: f64,
    fs: f64,
    opts: DesignOptions,
) -> Result<ColorDesign, DesignError> {
    if !fs.is_finite() || !(8000.0..=768_000.0).contains(&fs) {
        return Err(DesignError::BadSampleRate);
    }
    if !alpha.is_finite() {
        return Err(DesignError::BadAlpha);
    }
    let alpha = alpha.clamp(ALPHA_MIN, ALPHA_MAX);
    let (vb_lo, vb_hi) = validation_band(fs);
    if vb_hi <= vb_lo {
        return Err(DesignError::BadSampleRate);
    }

    let mut grid = [0.0f64; MAX_PROBE];
    let gn = log_grid_into(vb_lo, vb_hi, opts.probe_points, &mut grid);
    let grid = &grid[..gn];

    // White needs no filtering at all: an empty cascade is an exact bypass.
    if alpha == 0.0 {
        return Ok(ColorDesign {
            coeffs: SosCoeffs::IDENTITY,
            report: DesignReport {
                alpha,
                sample_rate: fs,
                band_lo_hz: vb_lo,
                band_hi_hz: vb_hi,
                sections: 0,
                order: 0,
                mean_offset_db: 0.0,
                ripple_db: 0.0,
                max_abs_dev_db: 0.0,
                slope_db_per_octave: colors::psd_slope_db_per_octave(alpha),
                slope_error_db_per_octave: 0.0,
                within_tolerance: true,
            },
        });
    }

    // The fit grid runs well below the promised band so the response cannot be
    // bought in-band at the price of a subsonic resonance, and a little above it
    // so the top edge behaves. Weights are tight exactly where DESIGN.md
    // section 9.1 measures and slack outside.
    let f_reg = (vb_lo / REGULARISATION_RATIO).max(0.5);
    let fit_lo = (f_reg / 4.0).max(0.2);
    let fit_hi = (vb_hi * 1.6).min(max_design_hz(fs));
    if fit_hi <= fit_lo * 2.0 {
        return Err(DesignError::BadSampleRate);
    }

    let n_fit = 241usize;
    let mut fit_f = vec![0.0f64; n_fit];
    log_grid_into(fit_lo, fit_hi, n_fit, &mut fit_f);
    let fit_w: Vec<f64> = fit_f.iter().map(|&f| prewarp(f, fs)).collect();
    // Absolute target, not shape-only: flat at the DC gain below `f_reg`, then
    // the exact power law. `f_reg` is where the infinite DC gain of a true
    // 1/f^alpha process is regularised (DESIGN.md section 1.2).
    let anchor_db = colors::amplitude_db(alpha, f_reg, colors::F_REF_HZ);
    let ideal: Vec<f64> = fit_f
        .iter()
        .map(|&f| colors::amplitude_db(alpha, f.max(f_reg), colors::F_REF_HZ) - anchor_db)
        .collect();
    let weights: Vec<f64> = fit_f
        .iter()
        .map(|&f| {
            if f >= vb_lo && f <= vb_hi {
                1.0
            } else if f >= f_reg {
                0.55
            } else {
                0.25
            }
        })
        .collect();

    let min_order = opts.min_order.max(1);
    let max_order = opts.max_order.max(min_order).min(MAX_SECTIONS);

    // --- stage 1: cheap recursive scan to find the cheapest plausible order ---
    let mut seed = Vec::with_capacity(2 * MAX_SECTIONS);
    let mut best_order = min_order;
    let mut best_seed = f64::INFINITY;
    let mut chosen_order: Option<usize> = None;
    for order in min_order..=max_order {
        if 2 * order + 1 > 2 * MAX_SECTIONS {
            break;
        }
        let (f_lo, f_hi) = recursive_band(order, vb_lo, vb_hi, fs);
        oustaloup_corners(
            -alpha / 2.0,
            prewarp(f_lo, fs),
            prewarp(f_hi, fs),
            order,
            &mut seed,
            fs,
        );
        let mut scratch = [0.0f64; MAX_PROBE];
        let raw = evaluate_first_order(&seed, alpha, fs, grid, &mut scratch).2;
        if raw < best_seed {
            best_seed = raw;
            best_order = order;
        }
        // Fitting is worth attempting when the raw error is in the ballpark.
        if chosen_order.is_none() && raw <= opts.tolerance_db * 2.5 {
            chosen_order = Some(order);
        }
    }
    let primary = chosen_order.unwrap_or(best_order);

    // --- stage 2: fit the plausible orders, keep the cheapest that passes ---
    let mut result: Option<(SosCoeffs, DesignReport)> = None;
    let mut biquads = [BiquadCoeffs::IDENTITY; MAX_SECTIONS];
    let mut scratch = [0.0f64; MAX_PROBE];
    let mut orders: Vec<usize> = vec![primary];
    if primary < max_order {
        orders.push(primary + 1);
    }
    if chosen_order.is_none() {
        orders.clear();
        orders.push(best_order);
        if best_order < max_order {
            orders.push(best_order + 1);
        }
    }
    for order in orders {
        if order < min_order || order > max_order || 2 * order + 1 > 2 * MAX_SECTIONS {
            continue;
        }
        let (f_lo, f_hi) = recursive_band(order, vb_lo, vb_hi, fs);
        oustaloup_corners(
            -alpha / 2.0,
            prewarp(f_lo, fs),
            prewarp(f_hi, fs),
            order,
            &mut seed,
            fs,
        );

        // Seed the fit from the recursive corner positions.
        let m = seed.len();
        let mut log_z = vec![0.0f64; m];
        let mut log_p = vec![0.0f64; m];
        for (i, &(b0, b1, a1)) in seed.iter().enumerate() {
            let rz = b1 / b0;
            let kz = (1.0 - rz) / (1.0 + rz);
            let kp = (1.0 - a1) / (1.0 + a1);
            if !kz.is_finite() || !kp.is_finite() || kz <= 0.0 || kp <= 0.0 {
                return Err(DesignError::Unstable);
            }
            log_z[i] = (2.0 * fs / kz).ln().clamp(MIN_LOG_W, MAX_LOG_W);
            log_p[i] = (2.0 * fs / kp).ln().clamp(MIN_LOG_W, MAX_LOG_W);
        }

        if opts.refine {
            lm_fit(
                &mut log_z,
                &mut log_p,
                &fit_w,
                &weights,
                &ideal,
                opts.refine_sweeps.max(8),
            );
        }

        let mut sections = Vec::with_capacity(m);
        corners_to_sections(&log_z, &log_p, fs, &mut sections);
        let coeffs = match pack(&sections, &mut biquads) {
            Ok(c) if c.is_valid() => c,
            _ => continue,
        };
        let (mean, ripple, max_abs, slope) = evaluate(&coeffs, alpha, fs, grid, &mut scratch);
        let slope_err = slope - colors::psd_slope_db_per_octave(alpha);
        let ok =
            max_abs <= opts.tolerance_db && slope_err.abs() <= opts.slope_tolerance_db_per_octave;
        let report = DesignReport {
            alpha,
            sample_rate: fs,
            band_lo_hz: vb_lo,
            band_hi_hz: vb_hi,
            sections: coeffs.len(),
            order,
            mean_offset_db: mean,
            ripple_db: ripple,
            max_abs_dev_db: max_abs,
            slope_db_per_octave: slope,
            slope_error_db_per_octave: slope_err,
            within_tolerance: ok,
        };
        let better = match &result {
            None => true,
            Some((_, r)) => max_abs < r.max_abs_dev_db,
        };
        if better {
            result = Some((coeffs, report));
        }
        if ok {
            break;
        }
    }

    result
        .map(|(coeffs, report)| ColorDesign { coeffs, report })
        .ok_or(DesignError::Unstable)
}

/// Design a second order high pass (Butterworth by default, `q = 1/sqrt(2)`).
pub fn design_high_pass(f_hz: f64, q: f64, fs: f64) -> BiquadCoeffs {
    let f = f_hz.clamp(1.0, max_design_hz(fs));
    BiquadCoeffs::high_pass(f, q.max(0.05), fs)
}

/// Design a second order low pass (Butterworth by default, `q = 1/sqrt(2)`).
pub fn design_low_pass(f_hz: f64, q: f64, fs: f64) -> BiquadCoeffs {
    let f = f_hz.clamp(1.0, max_design_hz(fs));
    BiquadCoeffs::low_pass(f, q.max(0.05), fs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn white_is_a_true_bypass() {
        let d = design_color(0.0, 48000.0).unwrap();
        assert_eq!(d.coeffs.len(), 0);
        assert!(d.report.within_tolerance);
    }

    #[test]
    fn named_colours_meet_the_documented_criteria() {
        for (alpha, name) in [
            (0.0, "white"),
            (1.0, "pink"),
            (2.0, "brown"),
            (-1.0, "blue"),
            (-2.0, "violet"),
        ] {
            for fs in [44100.0, 48000.0, 96000.0] {
                let d = design_color(alpha, fs).unwrap();
                let r = d.report;
                assert!(
                    r.max_abs_dev_db <= 1.5,
                    "{name} @ {fs}: shape error {:.3} dB ({} sections)",
                    r.max_abs_dev_db,
                    r.sections
                );
                assert!(
                    r.slope_error_db_per_octave.abs() <= 0.30,
                    "{name} @ {fs}: slope error {:.3} dB/oct",
                    r.slope_error_db_per_octave
                );
                assert!(
                    r.sections <= MAX_SECTIONS,
                    "{name} @ {fs}: {} sections",
                    r.sections
                );
                assert!(d.coeffs.is_valid());
            }
        }
    }

    #[test]
    fn custom_alphas_report_honestly() {
        for alpha in [-4.0, -2.0, -1.5, -0.5, 0.5, 1.35, 1.5, 3.0, 4.0] {
            let d = design_color(alpha, 48000.0).unwrap();
            assert!(d.coeffs.is_valid());
            assert!(d.report.max_abs_dev_db.is_finite());
            if !d.report.within_tolerance {
                assert!(
                    d.report.max_abs_dev_db > 1.5
                        || d.report.slope_error_db_per_octave.abs() > 0.30,
                    "alpha {alpha} claims to fail but passes"
                );
            }
        }
    }

    #[test]
    fn unstable_or_nan_inputs_are_rejected() {
        assert_eq!(design_color(f64::NAN, 48000.0), Err(DesignError::BadAlpha));
        assert_eq!(design_color(1.0, 100.0), Err(DesignError::BadSampleRate));
        assert_eq!(design_color(1.0, f64::NAN), Err(DesignError::BadSampleRate));
    }

    #[test]
    fn dc_gain_is_unity_for_every_colour() {
        // The regularisation promised in DESIGN.md section 1.2: no infinite DC
        // gain, no runaway integrator.
        for alpha in [1.0, 2.0, 4.0, -2.0] {
            let d = design_color(alpha, 48000.0).unwrap();
            let dc = d.coeffs.response_db(0.001, 48000.0);
            assert!(dc.abs() < 0.05, "alpha {alpha}: DC gain {dc} dB");
        }
    }
}
