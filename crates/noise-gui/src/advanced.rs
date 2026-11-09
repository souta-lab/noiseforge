//! The advanced view (DESIGN.md section 5.2).
//!
//! Everything here is either a value the daemon actually reports or a value
//! derived from reported counters, and the derivation is named in the label so
//! nobody reads a guess as a measurement. Two things DESIGN.md mentions are
//! deliberately absent rather than faked:
//!
//! * a parametric EQ and explicit filter `Q` values, because the daemon exposes
//!   neither, and
//! * recording and WAV export, because the daemon does not implement them yet
//!   (DESIGN.md section 10, phase 4).

use egui::{Color32, Grid, RichText, Ui};
use noise_daemon::protocol::{LayerStatus, StatusData};
use noise_dsp::Character;
use noise_dsp::characters::{
    INTENSE_HIGH_SHELF_DB, INTENSE_HIGH_SHELF_HZ, INTENSE_LOW_SHELF_DB, INTENSE_LOW_SHELF_HZ,
    SMOOTH_HIGH_SHELF_DB, SMOOTH_HIGH_SHELF_HZ, shelf_gains_db,
};

use crate::app::App;

/// Colour of an out-of-tolerance badge.
const BAD: Color32 = Color32::from_rgb(240, 90, 90);
/// Colour of a passing badge.
const GOOD: Color32 = Color32::from_rgb(120, 230, 130);

/// One shelf of a character, as the DSP defines it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShelfRow {
    /// Which shelf this is.
    pub label: &'static str,
    /// Corner frequency in Hz.
    pub hz: f64,
    /// Gain at the current intensity, in dB.
    pub gain_db: f64,
}

/// The shelves a character applies at a given intensity.
///
/// The corner frequencies are constants of `noise_dsp::characters`; the gains
/// scale linearly with the intensity, which is what DESIGN.md section 2.1
/// promises, so this table is the same arithmetic the engine performs.
pub fn shelf_rows(character: Character, amount: f64) -> Vec<ShelfRow> {
    let (low_db, high_db) = shelf_gains_db(character, amount);
    match character {
        Character::Neutral => Vec::new(),
        Character::Smooth => vec![ShelfRow {
            label: "High shelf",
            hz: SMOOTH_HIGH_SHELF_HZ,
            gain_db: high_db,
        }],
        Character::Intense => vec![
            ShelfRow {
                label: "Low shelf",
                hz: INTENSE_LOW_SHELF_HZ,
                gain_db: low_db,
            },
            ShelfRow {
                label: "High shelf",
                hz: INTENSE_HIGH_SHELF_HZ,
                gain_db: high_db,
            },
        ],
    }
}

/// Shelf gains a character reaches at full intensity.
pub fn shelf_full_scale_db(character: Character) -> (f64, f64) {
    match character {
        Character::Neutral => (0.0, 0.0),
        Character::Smooth => (0.0, SMOOTH_HIGH_SHELF_DB),
        Character::Intense => (INTENSE_LOW_SHELF_DB, INTENSE_HIGH_SHELF_DB),
    }
}

/// Frames per callback, derived from the two counters the daemon reports.
///
/// The daemon does not publish the negotiated quantum, so this is the only
/// honest number available; it is `None` until at least one callback ran.
pub fn implied_quantum(frames: u64, callbacks: u64) -> Option<f64> {
    if callbacks == 0 {
        return None;
    }
    Some(frames as f64 / callbacks as f64)
}

/// `mm:ss`, or `h:mm:ss` past an hour.
pub fn format_duration(seconds: f64) -> String {
    if !seconds.is_finite() || seconds < 0.0 {
        return "n/a".to_string();
    }
    let total = seconds.round().min(359_999.0) as u64;
    let (h, m, s) = (total / 3600, (total % 3600) / 60, total % 60);
    if h > 0 {
        format!("{h}:{m:02}:{s:02}")
    } else {
        format!("{m:02}:{s:02}")
    }
}

/// A dB value with a sign, or `n/a` when the daemon sent nothing usable.
pub fn db_text(value: f64) -> String {
    if value.is_finite() {
        format!("{value:+.2} dB")
    } else {
        "n/a".to_string()
    }
}

/// The two accuracy figures the daemon reports, verbatim.
pub fn accuracy_text(
    shape_error_db: Option<f64>,
    slope_error_db_per_octave: Option<f64>,
) -> String {
    match (shape_error_db, slope_error_db_per_octave) {
        (Some(shape), Some(slope)) => {
            format!("shape {shape:+.3} dB, slope {slope:+.3} dB/oct")
        }
        _ => "not reported (the daemon designed no cascade for this layer)".to_string(),
    }
}

/// The compact per-layer accuracy line used by the mixer.
pub fn accuracy_line(ui: &mut Ui, layer: &LayerStatus) {
    ui.horizontal(|ui| {
        ui.label(
            RichText::new(accuracy_text(
                layer.shape_error_db,
                layer.slope_error_db_per_octave,
            ))
            .small(),
        );
        ui.label(
            RichText::new(format!("{} sections", layer.sections))
                .small()
                .weak(),
        );
        badge(ui, layer.verified);
    });
}

/// Verified / out-of-tolerance badge.
fn badge(ui: &mut Ui, verified: bool) {
    if verified {
        ui.colored_label(GOOD, RichText::new("verified").small());
    } else {
        ui.colored_label(BAD, RichText::new("OUT OF TOLERANCE").small().strong());
    }
}

/// Draw the advanced view.
pub fn ui(ui: &mut Ui, app: &mut App) {
    egui::ScrollArea::vertical().show(ui, |ui| {
        let Some(status) = app.status().cloned() else {
            ui.heading("Advanced");
            ui.label("No status yet: the daemon has not sent a telemetry frame.");
            return;
        };
        let first = status.layers.first().cloned();

        ui.heading("Advanced");
        colour_section(ui, &status, first.as_ref());
        ui.separator();
        crate::layers::ui(ui, app, &status.layers);
        ui.separator();
        character_section(ui, first.as_ref());
        ui.separator();
        engine_section(ui, &status);
        ui.separator();
        master_section(ui, app, &status);
        ui.separator();
        timer_section(ui, app, &status);
        ui.separator();
        messages_section(ui, app);
    });
}

/// Colour, alpha, slope and the measured design accuracy.
fn colour_section(ui: &mut Ui, status: &StatusData, first: Option<&LayerStatus>) {
    ui.heading("Colour filter");
    let Some(layer) = first else {
        ui.label("The daemon reports no layers.");
        return;
    };
    let alpha = layer.alpha;
    let slope = noise_dsp::colors::psd_slope_db_per_octave(alpha);
    Grid::new("nf-colour-grid")
        .num_columns(2)
        .spacing([16.0, 4.0])
        .show(ui, |ui| {
            ui.label("Colour");
            ui.monospace(format!(
                "{} (layer 0), alpha {alpha:.3}, intensity {:.0}%",
                layer.color,
                layer.character_amount * 100.0
            ));
            ui.end_row();

            ui.label("PSD slope");
            ui.monospace(format!("{slope:+.3} dB/oct (theory from alpha)"));
            ui.end_row();

            ui.label("High / low pass");
            ui.monospace(format!(
                "{:.1} Hz / {:.1} Hz, stereo rho {:+.3}",
                layer.high_pass_hz, layer.low_pass_hz, layer.stereo_correlation
            ));
            ui.end_row();

            ui.label("Layer seed");
            ui.monospace(format!("{}", layer.seed));
            ui.end_row();

            ui.label("Measured shape error");
            ui.monospace(match layer.shape_error_db {
                Some(v) => format!("{v:+.4} dB"),
                None => "not reported".to_string(),
            });
            ui.end_row();

            ui.label("Measured slope error");
            ui.monospace(match layer.slope_error_db_per_octave {
                Some(v) => format!("{v:+.4} dB/oct"),
                None => "not reported".to_string(),
            });
            ui.end_row();

            ui.label("Design");
            ui.horizontal(|ui| {
                ui.monospace(format!("{} second order sections", layer.sections));
                badge(ui, layer.verified);
            });
            ui.end_row();
        });
    let (lo, hi) = noise_dsp::designer::validation_band(f64::from(status.sample_rate));
    ui.label(
        RichText::new(format!(
            "Tolerance is judged over {lo:.0}-{hi:.0} Hz at {} Hz. A layer with no cascade needs \
             no design, so its error fields stay empty rather than reading zero.",
            status.sample_rate
        ))
        .small()
        .weak(),
    );
}

/// Smooth / Intense shelf frequencies and gains.
fn character_section(ui: &mut Ui, first: Option<&LayerStatus>) {
    ui.heading("Character shelves");
    let Some(layer) = first else {
        return;
    };
    let character = Character::parse(&layer.character).unwrap_or(Character::Neutral);
    let amount = layer.character_amount;
    let (low_full, high_full) = shelf_full_scale_db(character);
    ui.label(format!(
        "{} at {:.0}% of its range. Gains scale linearly with intensity and ride on a fixed shelf \
         shape, so the corner frequency never moves.",
        character.label(),
        amount * 100.0
    ));
    let rows = shelf_rows(character, amount);
    if rows.is_empty() {
        ui.label(RichText::new("Neutral is a bit-exact bypass: no shelf is applied.").weak());
    } else {
        Grid::new("nf-shelf-grid")
            .num_columns(4)
            .spacing([16.0, 4.0])
            .show(ui, |ui| {
                ui.label(RichText::new("Shelf").strong());
                ui.label(RichText::new("Corner").strong());
                ui.label(RichText::new("Gain now").strong());
                ui.label(RichText::new("Gain at 100%").strong());
                ui.end_row();
                for row in &rows {
                    let full = if row.label == "Low shelf" {
                        low_full
                    } else {
                        high_full
                    };
                    ui.label(row.label);
                    ui.monospace(format!("{:.0} Hz", row.hz));
                    ui.monospace(format!("{:+.3} dB", row.gain_db));
                    ui.monospace(format!("{full:+.1} dB"));
                    ui.end_row();
                }
            });
    }
    ui.label(
        RichText::new(
            "Smooth and Intense shape the output after the colour filter, so what they produce is \
             the Predicted Output curve and never the pristine colour curve.",
        )
        .small()
        .weak(),
    );
}

/// Seed, rate, device, link and the realtime counters.
fn engine_section(ui: &mut Ui, status: &StatusData) {
    ui.heading("Engine");
    let meters = &status.meters;
    Grid::new("nf-engine-grid")
        .num_columns(2)
        .spacing([16.0, 4.0])
        .show(ui, |ui| {
            ui.label("Seed");
            ui.monospace(format!("{}", status.seed));
            ui.end_row();

            ui.label("Sample rate");
            ui.monospace(format!(
                "{} Hz, {} channel(s), rate policy {:?}",
                status.sample_rate, status.channels, status.rate_mode
            ));
            ui.end_row();

            ui.label("Device");
            ui.monospace(
                status
                    .device
                    .clone()
                    .unwrap_or_else(|| "default sink".to_string()),
            );
            ui.end_row();

            ui.label("Link");
            ui.monospace(format!("{} (backend {:?})", status.link, status.backend));
            ui.end_row();

            ui.label("Preset");
            ui.monospace(format!(
                "{} ({} of {} layer(s) enabled)",
                status.preset_name,
                status.layers.iter().filter(|l| l.enabled).count(),
                status.layers.len()
            ));
            ui.end_row();

            ui.label("Daemon");
            ui.monospace(format!(
                "{} protocol {} dsp {}",
                status.version, status.protocol, status.dsp_version
            ));
            ui.end_row();

            ui.label("Uptime");
            ui.monospace(format_duration(status.uptime_s));
            ui.end_row();

            ui.label("Callbacks");
            ui.monospace(format!("{}", meters.callbacks));
            ui.end_row();

            ui.label("Frames");
            ui.monospace(format!("{}", meters.frames));
            ui.end_row();

            ui.label("Implied quantum");
            ui.monospace(match implied_quantum(meters.frames, meters.callbacks) {
                Some(q) => format!("{q:.1} frames/callback"),
                None => "n/a (no callback yet)".to_string(),
            });
            ui.end_row();

            ui.label("Xruns");
            ui.monospace(format!("{}", meters.xruns));
            ui.end_row();

            ui.label("NaN / clipped");
            ui.monospace(format!("{} / {}", meters.nan_events, meters.clipped));
            ui.end_row();

            ui.label("Analysis tap dropped");
            ui.monospace(format!("{}", meters.tap_dropped));
            ui.end_row();
        });
    ui.label(
        RichText::new(
            "The daemon publishes counters, not a CPU percentage: a realtime load figure would \
             have to come from PipeWire itself. The quantum is derived from frames per callback \
             because the negotiated buffer size is not part of the protocol.",
        )
        .small()
        .weak(),
    );
}

/// Master calibration, trim, mute and the limiter.
fn master_section(ui: &mut Ui, app: &mut App, status: &StatusData) {
    ui.heading("Master");
    let cal = &status.calibration;
    Grid::new("nf-master-grid")
        .num_columns(2)
        .spacing([16.0, 4.0])
        .show(ui, |ui| {
            ui.label("Target RMS");
            ui.monospace(format!("{:.2} dBFS", cal.target_rms_dbfs));
            ui.end_row();

            ui.label("Mix RMS before calibration");
            ui.monospace(format!("{:.2} dBFS", cal.mix_rms_dbfs));
            ui.end_row();

            ui.label("Calibration gain");
            ui.monospace(db_text(cal.calibration_gain_db));
            ui.end_row();

            ui.label("Total master gain");
            ui.monospace(db_text(cal.master_gain_db));
            ui.end_row();

            ui.label("Predicted peak");
            ui.horizontal(|ui| {
                ui.monospace(format!("{:.2} dBFS", cal.predicted_peak_dbfs));
                if cal.headroom_warning {
                    ui.colored_label(BAD, RichText::new("HEADROOM WARNING").small().strong());
                }
            });
            ui.end_row();
        });

    let mut mute = status.master.mute;
    let mut trim = status.master.volume_db;
    let mut limiter = status.master.limiter_enabled;
    let mut ceiling = status.master.limiter_ceiling_dbfs;
    ui.horizontal(|ui| {
        if ui.checkbox(&mut mute, "mute").changed() {
            app.send_set("mute", None, |args| args.mute = Some(mute));
        }
        if ui
            .add(
                egui::Slider::new(&mut trim, -60.0..=24.0)
                    .text("trim")
                    .suffix(" dB")
                    .fixed_decimals(1),
            )
            .changed()
        {
            app.send_set("trim", None, |args| args.volume_db = Some(trim));
        }
    });
    ui.horizontal(|ui| {
        if ui.checkbox(&mut limiter, "limiter").changed() {
            app.send_set("limiter", None, |args| args.limiter = Some(limiter));
        }
        if ui
            .add(
                egui::Slider::new(&mut ceiling, -24.0..=0.0)
                    .text("ceiling")
                    .suffix(" dBFS")
                    .fixed_decimals(1),
            )
            .changed()
        {
            app.send_set("limiter ceiling", None, |args| {
                args.limiter_ceiling_dbfs = Some(ceiling);
            });
        }
    });
    ui.label(
        RichText::new(
            "The limiter is off by default: it is nonlinear, so it changes the PSD and the \
             measured curve stops matching the prediction.",
        )
        .small()
        .weak(),
    );
}

/// The sleep timer.
fn timer_section(ui: &mut Ui, app: &mut App, status: &StatusData) {
    ui.heading("Timer");
    ui.horizontal(|ui| {
        ui.add(
            egui::Slider::new(&mut app.timer_minutes, 0.0..=480.0)
                .text("minutes")
                .fixed_decimals(0),
        );
        ui.add(
            egui::Slider::new(&mut app.timer_fade_s, 0.0..=600.0)
                .text("fade-out")
                .suffix(" s")
                .fixed_decimals(0),
        );
    });
    ui.horizontal(|ui| {
        if ui.button("Set timer").clicked() {
            let seconds = app.timer_minutes * 60.0;
            app.send_request(
                "timer",
                noise_daemon::protocol::Request::Timer {
                    seconds,
                    fade_out_seconds: app.timer_fade_s,
                },
            );
        }
        if ui.button("Clear timer").clicked() {
            app.send_request("timer clear", noise_daemon::protocol::Request::TimerClear);
        }
        ui.label(match status.timer_remaining_s {
            Some(remaining) => format!("remaining {}", format_duration(remaining)),
            None => "no timer armed".to_string(),
        });
    });
    ui.label(
        RichText::new(
            "The fade-out occupies the end of the duration and playback pauses when it completes. \
             Recording and WAV export are not implemented in the daemon, so this window does not \
             offer them.",
        )
        .small()
        .weak(),
    );
}

/// Daemon notes and client side errors.
fn messages_section(ui: &mut Ui, app: &mut App) {
    ui.heading("Messages");
    if app.notes().is_empty() {
        ui.label(RichText::new("Nothing to report.").weak());
        return;
    }
    for note in app.notes().iter().rev() {
        if note.error {
            ui.colored_label(BAD, format!("{}: {}", note.source, note.text));
        } else {
            ui.label(format!("{}: {}", note.source, note.text));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neutral_has_no_shelves() {
        assert!(shelf_rows(Character::Neutral, 1.0).is_empty());
        assert_eq!(shelf_full_scale_db(Character::Neutral), (0.0, 0.0));
    }

    #[test]
    fn smooth_uses_its_documented_high_shelf() {
        let rows = shelf_rows(Character::Smooth, 1.0);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].hz, 3200.0);
        assert!((rows[0].gain_db + 8.0).abs() < 1e-12);
        // Half intensity halves the dB gain, it does not move the corner.
        let half = shelf_rows(Character::Smooth, 0.5);
        assert_eq!(half[0].hz, 3200.0);
        assert!((half[0].gain_db + 4.0).abs() < 1e-12);
    }

    #[test]
    fn intense_uses_both_documented_shelves() {
        let rows = shelf_rows(Character::Intense, 1.0);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].label, "Low shelf");
        assert_eq!(rows[0].hz, 180.0);
        assert!((rows[0].gain_db - 3.0).abs() < 1e-12);
        assert_eq!(rows[1].hz, 2500.0);
        assert!((rows[1].gain_db - 4.0).abs() < 1e-12);
    }

    #[test]
    fn out_of_range_intensity_is_clamped_like_the_engine_does() {
        let rows = shelf_rows(Character::Intense, 5.0);
        assert!((rows[0].gain_db - 3.0).abs() < 1e-12);
        assert!((rows[1].gain_db - 4.0).abs() < 1e-12);
        let nan = shelf_rows(Character::Smooth, f64::NAN);
        assert_eq!(nan[0].gain_db, 0.0);
    }

    #[test]
    fn implied_quantum_needs_a_callback() {
        assert_eq!(implied_quantum(48_000, 1000), Some(48.0));
        assert_eq!(implied_quantum(0, 0), None);
        assert_eq!(implied_quantum(100, 0), None);
    }

    #[test]
    fn durations_are_formatted_for_a_timer() {
        assert_eq!(format_duration(0.0), "00:00");
        assert_eq!(format_duration(59.4), "00:59");
        assert_eq!(format_duration(60.0), "01:00");
        assert_eq!(format_duration(2699.9), "45:00");
        assert_eq!(format_duration(3661.0), "1:01:01");
        assert_eq!(format_duration(-1.0), "n/a");
        assert_eq!(format_duration(f64::NAN), "n/a");
    }

    #[test]
    fn accuracy_text_keeps_the_daemon_numbers_verbatim() {
        assert_eq!(
            accuracy_text(Some(0.25), Some(-0.012)),
            "shape +0.250 dB, slope -0.012 dB/oct"
        );
        assert!(accuracy_text(None, None).contains("not reported"));
        // A half filled report must not be shown as if it were complete.
        assert!(accuracy_text(Some(0.25), None).contains("not reported"));
    }

    #[test]
    fn db_text_marks_missing_values() {
        assert_eq!(db_text(-36.0), "-36.00 dB");
        assert_eq!(db_text(1.5), "+1.50 dB");
        assert_eq!(db_text(f64::NAN), "n/a");
        assert_eq!(db_text(f64::INFINITY), "n/a");
    }
}
