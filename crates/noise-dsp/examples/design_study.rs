//! Design accuracy study: prints the measured shape/slope error of every colour
//! at every supported rate, plus a per-frequency error profile.
//!
//! This is the machine readable evidence behind the accuracy table quoted in
//! `DESIGN.md` section 9.1. Run with:
//!
//! ```sh
//! cargo run -p noise-dsp --release --example design_study
//! cargo run -p noise-dsp --release --example design_study -- brown 48000
//! ```

use noise_dsp::colors::{self, Color};
use noise_dsp::designer::{self, DesignOptions};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.len() >= 2 {
        let alpha = Color::parse(&args[0])
            .map(|c| c.resolve_alpha(1.0))
            .or_else(|| args[0].parse::<f64>().ok())
            .expect("colour name or alpha");
        let fs: f64 = args[1].parse().expect("sample rate");
        profile(alpha, fs);
        return;
    }

    let rates = [44100.0, 48000.0, 96000.0];
    let alphas: Vec<(String, f64)> = Color::ALL
        .iter()
        .filter(|c| **c != Color::Custom)
        .map(|c| (c.to_string(), c.resolve_alpha(1.0)))
        .chain(
            [-4.0, -2.0, -1.5, -0.5, 0.5, 1.35, 1.5, 3.0, 4.0]
                .into_iter()
                .map(|a| (format!("custom {a}"), a)),
        )
        .collect();

    println!(
        "{:<14} {:>7} {:>4} {:>4} {:>9} {:>9} {:>9} {:>8}  verdict",
        "colour", "fs", "ord", "sos", "maxdev dB", "ripple dB", "slope", "slopeerr"
    );
    let mut all_pass = true;
    for (name, alpha) in &alphas {
        for fs in rates {
            match designer::design_color(*alpha, fs) {
                Ok(d) => {
                    let r = d.report;
                    let ok = r.within_tolerance;
                    all_pass &= ok;
                    println!(
                        "{:<14} {:>7.0} {:>4} {:>4} {:>9.3} {:>9.3} {:>9.3} {:>8.3}  {}",
                        name,
                        fs,
                        r.order,
                        r.sections,
                        r.max_abs_dev_db,
                        r.ripple_db,
                        r.slope_db_per_octave,
                        r.slope_error_db_per_octave,
                        if ok { "PASS" } else { "FAIL" }
                    );
                }
                Err(e) => {
                    all_pass = false;
                    println!("{name:<14} {fs:>7.0}   design error: {e}");
                }
            }
        }
    }
    println!();
    println!(
        "overall: {}",
        if all_pass { "all PASS" } else { "some FAIL" }
    );

    // Timing matters: the designer runs on the daemon control thread.
    for alpha in [1.0, 2.0, -2.0] {
        let t0 = std::time::Instant::now();
        let n = 50;
        for _ in 0..n {
            let _ = designer::design_color(alpha, 48000.0);
        }
        println!(
            "design alpha={alpha:+.1} @48000: {:.2} ms/design",
            t0.elapsed().as_secs_f64() * 1000.0 / n as f64
        );
    }
}

fn profile(alpha: f64, fs: f64) {
    for opts in [
        DesignOptions::default(),
        DesignOptions {
            refine: false,
            ..DesignOptions::default()
        },
    ] {
        let d = designer::design_color_with(alpha, fs, opts).unwrap();
        println!(
            "alpha={alpha} fs={fs} refine={} -> order {} ({} sos), maxdev {:.3} dB, slope err {:.3} dB/oct",
            opts.refine,
            d.report.order,
            d.report.sections,
            d.report.max_abs_dev_db,
            d.report.slope_error_db_per_octave
        );
    }
    let d = designer::design_color(alpha, fs).unwrap();
    let (lo, hi) = designer::validation_band(fs);
    let grid = designer::log_grid(lo, hi, 61);
    println!(
        "{:>10} {:>12} {:>12} {:>10}",
        "f Hz", "ideal dB", "design dB", "err dB"
    );
    let mut mean = 0.0;
    let mut errs = Vec::new();
    for &f in &grid {
        let e = d.coeffs.response_db(f, fs) - colors::amplitude_db(alpha, f, colors::F_REF_HZ);
        errs.push((f, e));
        mean += e;
    }
    mean /= grid.len() as f64;
    for (f, e) in &errs {
        println!(
            "{:>10.1} {:>12.2} {:>12.2} {:>10.3}",
            f,
            colors::amplitude_db(alpha, *f, colors::F_REF_HZ),
            d.coeffs.response_db(*f, fs),
            e - mean
        );
    }
}
