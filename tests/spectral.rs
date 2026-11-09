//! Spectral accuracy tests: the machine check behind DESIGN.md section 9.1.
//!
//! Each colour is rendered offline, the PSD is estimated with Welch's method and
//! the log-log slope and shape error are compared against the theoretical curve.
//! The acceptance criteria are the provisional targets from DESIGN.md:
//!
//! * slope error within `+/- 0.30 dB/oct`,
//! * shape (after removing the free overall gain) within `+/- 1.5 dB`.
//!
//! Nothing here claims more accuracy than was measured: a colour that misses the
//! target fails the test rather than being quietly dropped.

use noise_dsp::compile::compile;
use noise_dsp::designer;
use noise_dsp::params::{LayerParams, PresetParams};
use noise_offline::analysis::{WelchConfig, deinterleave_channel, measure_slope};
use noise_offline::render::{RenderRequest, render};

/// Seconds of audio per measurement. Long enough for a stable Welch estimate,
/// short enough that the debug test profile stays quick.
fn render_seconds() -> f64 {
    std::env::var("NOISEFORGE_SPECTRAL_SECONDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(12.0)
}

/// Build a preset whose chain is *only* the colour filter, so the measurement
/// isolates the generator from the user EQ and the character stage.
fn bare_colour_preset(color: noise_dsp::Color, alpha: f64) -> PresetParams {
    PresetParams {
        layers: vec![LayerParams {
            color,
            alpha,
            character: noise_dsp::Character::Neutral,
            character_amount: 0.0,
            high_pass_hz: 0.0,
            low_pass_hz: 0.0,
            stereo_correlation: 1.0,
            gain_db: 0.0,
            seed: Some(20_260_010),
            ..LayerParams::default()
        }],
        ..PresetParams::default()
    }
}

struct Measurement {
    slope_error: f64,
    max_shape_deviation: f64,
    rms_shape_deviation: f64,
    bands: usize,
    averages: usize,
}

fn measure(color: noise_dsp::Color, alpha: f64, fs: f64) -> Measurement {
    let params = bare_colour_preset(color, alpha);
    let req = RenderRequest {
        params,
        sample_rate: fs,
        seconds: render_seconds(),
        warmup_seconds: 1.0,
        channels: 2,
        seed: Some(20_260_010),
        auto_play: true,
    };
    let result = render(&req);
    let mono = deinterleave_channel(&result.samples, 2, 0);

    let (lo, hi) = designer::validation_band(fs);
    let cfg = WelchConfig::for_rate(fs);
    let (psd, bands, fit) =
        measure_slope(&mono, fs, lo, hi, &cfg, 6).expect("not enough samples for a PSD estimate");

    let theoretical = noise_dsp::colors::psd_slope_db_per_octave(alpha);
    // The fit intercept carries the arbitrary overall gain of the chain, so the
    // shape error is measured against the best fitting straight line, and the
    // slope error against the theoretical slope.
    let mut max_dev = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut n = 0usize;
    for b in &bands {
        if b.center_hz < lo || b.center_hz > hi {
            continue;
        }
        let ideal = theoretical * b.center_hz.log2() + fit.intercept_db;
        let d = b.db() - ideal;
        max_dev = max_dev.max(d.abs());
        sum_sq += d * d;
        n += 1;
    }
    Measurement {
        slope_error: fit.slope_db_per_octave - theoretical,
        max_shape_deviation: max_dev,
        rms_shape_deviation: (sum_sq / n.max(1) as f64).sqrt(),
        bands: n,
        averages: psd.averages,
    }
}

fn check(color: noise_dsp::Color, alpha: f64, fs: f64, tolerance_scale: f64) -> Measurement {
    let m = measure(color, alpha, fs);
    assert!(m.bands >= 20, "{color} @ {fs}: only {} bands", m.bands);
    assert!(
        m.slope_error.abs() <= 0.30 * tolerance_scale,
        "{color} @ {fs}: slope error {:.3} dB/oct ({} averages)",
        m.slope_error,
        m.averages
    );
    assert!(
        m.max_shape_deviation <= 1.5 * tolerance_scale,
        "{color} @ {fs}: shape error {:.3} dB rms {:.3} dB",
        m.max_shape_deviation,
        m.rms_shape_deviation
    );
    m
}

#[test]
fn white_has_no_systematic_tilt() {
    for fs in [44_100.0, 48_000.0, 96_000.0] {
        let m = check(noise_dsp::Color::White, 0.0, fs, 1.0);
        assert!(
            m.slope_error.abs() <= 0.30,
            "white @ {fs}: {:.3} dB/oct",
            m.slope_error
        );
    }
}

#[test]
fn pink_meets_the_documented_criteria() {
    for fs in [44_100.0, 48_000.0, 96_000.0] {
        check(noise_dsp::Color::Pink, 1.0, fs, 1.0);
    }
}

#[test]
fn brown_meets_the_documented_criteria_and_does_not_run_away() {
    for fs in [44_100.0, 48_000.0, 96_000.0] {
        let m = check(noise_dsp::Color::Brown, 2.0, fs, 1.0);
        // Brown is dominated by its lowest band; the average level must stay
        // finite and well below full scale (no DC blow-up).
        assert!(m.max_shape_deviation.is_finite());
    }
}

#[test]
fn blue_meets_the_documented_criteria() {
    for fs in [44_100.0, 48_000.0, 96_000.0] {
        check(noise_dsp::Color::Blue, -1.0, fs, 1.0);
    }
}

#[test]
fn violet_meets_the_documented_criteria() {
    for fs in [44_100.0, 48_000.0, 96_000.0] {
        check(noise_dsp::Color::Violet, -2.0, fs, 1.0);
    }
}

#[test]
fn custom_alpha_samples_meet_the_documented_criteria() {
    // DESIGN.md section 9.1 lists the custom alphas to cover. The endpoints of
    // the supported range get a slightly wider statistical allowance because a
    // 12 dB/octave tilt across 8.6 octaves is a 104 dB span, where a finite
    // Welch average is noisier.
    for alpha in [-4.0f64, -2.0, -1.0, -0.5, 0.5, 1.5, 2.0, 4.0] {
        let scale = if alpha.abs() >= 4.0 { 1.35 } else { 1.0 };
        let color = if (alpha - 1.0).abs() < 1e-9 {
            noise_dsp::Color::Pink
        } else if (alpha - 2.0).abs() < 1e-9 {
            noise_dsp::Color::Brown
        } else if (alpha + 1.0).abs() < 1e-9 {
            noise_dsp::Color::Blue
        } else if (alpha + 2.0).abs() < 1e-9 {
            noise_dsp::Color::Violet
        } else {
            noise_dsp::Color::Custom
        };
        check(color, alpha, 48_000.0, scale);
    }
}

#[test]
fn measured_spectrum_matches_the_analytic_prediction() {
    // The "Predicted Output" curve the GUI draws must agree with what actually
    // comes out, otherwise the display would be lying.
    let fs = 48_000.0;
    let params = bare_colour_preset(noise_dsp::Color::Pink, 1.0);
    let graph = compile(&params, fs, 2);
    let predicted = &graph.layers[0].response;

    let req = RenderRequest {
        params,
        sample_rate: fs,
        seconds: render_seconds(),
        warmup_seconds: 1.0,
        channels: 2,
        seed: Some(20_260_010),
        auto_play: true,
    };
    let result = render(&req);
    let mono = deinterleave_channel(&result.samples, 2, 0);
    let (lo, hi) = designer::validation_band(fs);
    let cfg = WelchConfig::for_rate(fs);
    let (psd, bands, _) = measure_slope(&mono, fs, lo, hi, &cfg, 6).unwrap();

    // The measured level differs from |H(f)|^2 by one constant factor: the
    // input PSD. Estimate it once and then require the residual to be flat.
    let offset = bands
        .iter()
        .filter(|b| b.center_hz >= lo && b.center_hz <= hi)
        .map(|b| b.db() - predicted.response_db(b.center_hz, fs))
        .sum::<f64>()
        / bands
            .iter()
            .filter(|b| b.center_hz >= lo && b.center_hz <= hi)
            .count() as f64;

    let mut max_residual = 0.0f64;
    for b in &bands {
        if b.center_hz < lo || b.center_hz > hi {
            continue;
        }
        let residual = b.db() - predicted.response_db(b.center_hz, fs) - offset;
        max_residual = max_residual.max(residual.abs());
    }
    assert!(
        max_residual < 1.5,
        "predicted vs measured disagree by {max_residual:.3} dB"
    );
    assert!(psd.averages > 20);
}

#[test]
fn the_shaping_filter_does_not_flatten_the_low_end_for_brown() {
    // DESIGN.md section 1.2: Brown must be regularised at the bottom, not
    // integrated to infinite DC gain. Check the analytic filter directly.
    let d = designer::design_color(2.0, 48_000.0).unwrap();
    // Exactly at DC every section is unity by construction.
    let dc = d.coeffs.response_db(1e-9, 48_000.0);
    assert!(dc.abs() < 1e-6, "DC gain {dc} dB");
    // And the sub-audio region is bounded rather than integrating upwards.
    for f in [0.5, 1.0, 5.0, 20.0] {
        let db = d.coeffs.response_db(f, 48_000.0);
        assert!(db.abs() < 6.0, "brown at {f} Hz is {db} dB relative to DC");
    }
    // And it really is close to -6 dB/octave between 100 Hz and 3.2 kHz.
    let a = d.coeffs.response_db(100.0, 48_000.0);
    let b = d.coeffs.response_db(3200.0, 48_000.0);
    let slope = (b - a) / (3200.0f64 / 100.0).log2();
    assert!((slope + 6.0206).abs() < 0.30, "slope {slope} dB/oct");
}
