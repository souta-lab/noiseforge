//! The spectrum view (DESIGN.md section 5.3).
//!
//! Three curves exist and must never be conflated:
//!
//! * **Source Target** - the colour's theoretical PSD, computed from `alpha`
//!   alone by the daemon.
//! * **Predicted Output** - the analytic response of colour + character +
//!   high/low pass for the whole graph.
//! * **Measured Output** - a Welch estimate of the audio actually leaving the
//!   daemon, sent only while the subscription asks for the PSD.
//!
//! The daemon sends every curve log spaced between `f_lo` and `f_hi` and
//! normalised to 0 dB at 1 kHz, so the Y axis is *relative* dB and the caption
//! below the plot says so. The measured curve is a stochastic estimate that
//! moves between frames; that is a property of the estimator, not a defect, and
//! the caption states it so nobody has to guess.
//!
//! The X axis is log frequency, implemented as `log10(Hz)` on the plot with a
//! formatter that turns the values back into Hz for display.

use std::time::Instant;

use egui::{Color32, RichText, Ui};
use egui_plot::{HoverPosition, Legend, Line, Plot, PlotPoint, PlotPoints, PlotUi};
use noise_daemon::protocol::{Curve, TelemetryFrame};

/// Lowest frequency on the X axis, Hz.
pub const F_MIN_HZ: f64 = 20.0;
/// Highest frequency on the X axis, Hz.
pub const F_MAX_HZ: f64 = 20_000.0;

/// Levels are clamped to this window before plotting.
///
/// A single wild sample would otherwise squash every other curve into a line;
/// clamping keeps the interesting 100 dB visible and is honest as long as the
/// axis stays relative dB, which the caption states.
const DB_FLOOR: f64 = -80.0;
/// Upper clamp of the plotting window, dB.
const DB_CEIL: f64 = 24.0;

/// Colour of the theoretical source curve.
pub const COLOR_SOURCE: Color32 = Color32::from_rgb(255, 170, 60);
/// Colour of the analytic prediction curve.
pub const COLOR_PREDICTED: Color32 = Color32::from_rgb(90, 190, 255);
/// Colour of the measured curve.
pub const COLOR_MEASURED: Color32 = Color32::from_rgb(120, 230, 130);

/// The three curves of the last telemetry frame, already mapped to plot points.
///
/// The points are built once per telemetry frame rather than once per repaint:
/// the plot is redrawn far more often than the daemon pushes, and re-projecting
/// 480 samples on every repaint is pure waste.
#[derive(Default)]
pub struct Curves {
    /// Theoretical PSD of the first layer's colour.
    pub source: Vec<PlotPoint>,
    /// Analytic response of the whole graph.
    pub predicted: Vec<PlotPoint>,
    /// Welch estimate of the live output.
    pub measured: Vec<PlotPoint>,
    /// When the measured curve last arrived, for the staleness caption.
    pub measured_at: Option<Instant>,
    /// Telemetry frames accepted since start-up.
    pub frames: u64,
}

impl Curves {
    /// Take the curves of one telemetry frame.
    pub fn update(&mut self, frame: &TelemetryFrame, now: Instant) {
        self.frames = self.frames.saturating_add(1);
        // A missing field means "not available this frame"; keeping the previous
        // measured curve would show stale audio as if it were live, so it is
        // dropped instead.
        self.source = points_or_empty(frame.source_target.as_ref());
        self.predicted = points_or_empty(frame.predicted.as_ref());
        self.measured = points_or_empty(frame.measured.as_ref());
        self.measured_at = if self.measured.is_empty() {
            None
        } else {
            Some(now)
        };
    }

    /// Drop everything, e.g. when the daemon goes away.
    pub fn clear(&mut self) {
        self.source.clear();
        self.predicted.clear();
        self.measured.clear();
        self.measured_at = None;
    }

    /// Whether a curve has enough points to be worth drawing.
    fn is_drawable(points: &[PlotPoint]) -> bool {
        points.len() > 1
    }
}

/// Build plot points, or nothing when the daemon sent no curve.
fn points_or_empty(curve: Option<&Curve>) -> Vec<PlotPoint> {
    curve.map(curve_points).unwrap_or_default()
}

/// Legend text, colour and stroke width of one curve.
struct Series {
    label: &'static str,
    color: Color32,
    width: f32,
}

const SOURCE_SERIES: Series = Series {
    label: "Source Target (theory)",
    color: COLOR_SOURCE,
    width: 1.5,
};
const PREDICTED_SERIES: Series = Series {
    label: "Predicted Output (analytic)",
    color: COLOR_PREDICTED,
    width: 2.0,
};
const MEASURED_SERIES: Series = Series {
    label: "Measured Output (Welch)",
    color: COLOR_MEASURED,
    width: 1.5,
};

/// Map a frequency to the plot's X coordinate.
#[inline]
pub fn hz_to_x(hz: f64) -> f64 {
    // Frequencies are positive by contract; the clamp only protects the log
    // against a corrupt curve that somehow carries a zero or negative edge.
    hz.max(f64::MIN_POSITIVE).log10()
}

/// Map a plot X coordinate back to a frequency in Hz.
#[inline]
pub fn x_to_hz(x: f64) -> f64 {
    10.0f64.powf(x)
}

/// Human readable frequency for an axis label.
pub fn format_hz(hz: f64) -> String {
    if !hz.is_finite() || hz <= 0.0 {
        return "-".to_string();
    }
    if hz >= 1000.0 {
        let k = hz / 1000.0;
        if (k - k.round()).abs() < 0.05 {
            format!("{k:.0}k")
        } else {
            format!("{k:.1}k")
        }
    } else {
        format!("{hz:.0}")
    }
}

/// Clamp one dB value into the plotting window.
///
/// A non-finite sample is drawn at the floor: an unusable number must never be
/// allowed to set the vertical range for everything else, and "somewhere at the
/// bottom" is the least misleading place to put it.
fn clamp_db(db: f64) -> f64 {
    if db.is_finite() {
        db.clamp(DB_FLOOR, DB_CEIL)
    } else {
        DB_FLOOR
    }
}

/// Turn a daemon curve into plot points.
///
/// The daemon documents the samples as log spaced between `f_lo` and `f_hi`, so
/// the index is mapped onto that geometric range rather than onto a linear Hz
/// range.
pub fn curve_points(curve: &Curve) -> Vec<PlotPoint> {
    let n = curve.db.len();
    if n == 0 {
        return Vec::new();
    }
    let lo = hz_to_x(curve.f_lo);
    let hi = hz_to_x(curve.f_hi.max(curve.f_lo));
    let last = (n - 1).max(1) as f64;
    (0..n)
        .map(|i| {
            let t = i as f64 / last;
            PlotPoint::new(lo + (hi - lo) * t, clamp_db(f64::from(curve.db[i])))
        })
        .collect()
}

/// Draw the plot, the legend and the permanent caption.
pub fn ui(ui: &mut Ui, curves: &Curves) {
    Plot::new("noiseforge-spectrum")
        .height(250.0)
        .legend(Legend::default())
        .x_axis_label("Hz")
        .y_axis_label("dB relative to 1 kHz")
        .x_grid_spacer(egui_plot::log_grid_spacer(10))
        .x_axis_formatter(|mark, _| format_hz(x_to_hz(mark.value)))
        .y_axis_formatter(|mark, _| format!("{:+.0}", mark.value))
        .label_formatter(|pos| {
            let (name, point) = match pos {
                HoverPosition::NearDataPoint {
                    plot_name,
                    position,
                    ..
                } => (Some(*plot_name), *position),
                HoverPosition::Elsewhere { position } => (None, *position),
            };
            let hz = format_hz(x_to_hz(point.x));
            Some(match name {
                Some(name) => format!("{name}\n{hz} Hz\n{:+.1} dB", point.y),
                None => format!("{hz} Hz\n{:+.1} dB", point.y),
            })
        })
        .include_x(hz_to_x(F_MIN_HZ))
        .include_x(hz_to_x(F_MAX_HZ))
        .include_y(0.0)
        .show(ui, |plot_ui| {
            draw_series(plot_ui, &SOURCE_SERIES, &curves.source);
            draw_series(plot_ui, &PREDICTED_SERIES, &curves.predicted);
            draw_series(plot_ui, &MEASURED_SERIES, &curves.measured);
        });

    ui.horizontal_wrapped(|ui| {
        legend_entry(ui, &SOURCE_SERIES, Curves::is_drawable(&curves.source));
        legend_entry(
            ui,
            &PREDICTED_SERIES,
            Curves::is_drawable(&curves.predicted),
        );
        legend_entry(ui, &MEASURED_SERIES, Curves::is_drawable(&curves.measured));
    });
    ui.label(
        RichText::new(
            "Curves are normalised to 0 dB at 1 kHz by the daemon, so the vertical axis is \
             relative dB, not dBFS. Measured Output is a Welch estimate of the live audio: it \
             moves from frame to frame by design and a moving trace is not a defect.",
        )
        .small()
        .weak(),
    );
}

/// One curve as a plot line, borrowing the cached points.
///
/// The slice lifetime is tied to the plot's own lifetime so the points can be
/// borrowed straight out of [`Curves`] instead of cloned on every repaint.
fn draw_series<'a>(plot_ui: &mut PlotUi<'a>, series: &Series, points: &'a [PlotPoint]) {
    if !Curves::is_drawable(points) {
        return;
    }
    plot_ui.line(
        Line::new(series.label, PlotPoints::from(points))
            .color(series.color)
            .width(series.width),
    );
}

/// A legend row below the plot so a missing curve is visible as missing.
fn legend_entry(ui: &mut Ui, series: &Series, present: bool) {
    if present {
        ui.colored_label(series.color, series.label);
    } else {
        ui.label(RichText::new(format!("{} (n/a)", series.label)).weak());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use noise_daemon::protocol::{
        BackendKind, CalibrationStatus, MeterStatus, RateMode, StatusData,
    };
    use noise_dsp::params::MasterParams;

    fn curve(f_lo: f64, f_hi: f64, db: Vec<f32>) -> Curve {
        Curve { f_lo, f_hi, db }
    }

    /// A minimal but complete status payload; only its presence matters here.
    fn status() -> Box<StatusData> {
        Box::new(StatusData {
            version: "test".into(),
            protocol: 1,
            dsp_version: 1,
            backend: BackendKind::Null,
            playing: true,
            transport: "playing".into(),
            link: "null".into(),
            sample_rate: 48_000,
            channels: 2,
            rate_mode: RateMode::Follow,
            device: None,
            device_node_id: None,
            timer_remaining_s: None,
            uptime_s: 0.0,
            calibration: CalibrationStatus {
                target_rms_dbfs: -36.0,
                mix_rms_dbfs: -36.0,
                calibration_gain_db: 0.0,
                master_gain_db: 0.0,
                predicted_peak_dbfs: -20.0,
                headroom_warning: false,
            },
            meters: MeterStatus::default(),
            layers: Vec::new(),
            master: MasterParams::default(),
            notes: Vec::new(),
            preset_name: "test".into(),
            seed: 1,
            auto_start: false,
        })
    }

    #[test]
    fn frequency_round_trips_through_the_axis() {
        for hz in [20.0, 100.0, 1000.0, 20_000.0] {
            let back = x_to_hz(hz_to_x(hz));
            assert!((back - hz).abs() < 1e-9, "{hz} -> {back}");
        }
        // Defensive: a corrupt zero edge must not produce -inf.
        assert!(hz_to_x(0.0).is_finite());
        assert!(hz_to_x(-5.0).is_finite());
    }

    #[test]
    fn axis_labels_are_compact() {
        assert_eq!(format_hz(20.0), "20");
        assert_eq!(format_hz(440.0), "440");
        assert_eq!(format_hz(1000.0), "1k");
        assert_eq!(format_hz(1500.0), "1.5k");
        assert_eq!(format_hz(20_000.0), "20k");
        assert_eq!(format_hz(0.0), "-");
        assert_eq!(format_hz(f64::NAN), "-");
        assert_eq!(format_hz(-1.0), "-");
    }

    #[test]
    fn curve_points_span_the_log_range() {
        let c = curve(20.0, 20_000.0, vec![0.0, -6.0, -12.0]);
        let points = curve_points(&c);
        assert_eq!(points.len(), 3);
        assert!((points[0].x - hz_to_x(20.0)).abs() < 1e-12);
        assert!((points[2].x - hz_to_x(20_000.0)).abs() < 1e-12);
        // Geometric spacing: the third decade boundary sits at a third of the
        // index range, not at a third of the linear frequency range.
        assert!((points[1].x - (hz_to_x(20.0) + hz_to_x(20_000.0)) / 2.0).abs() < 1e-12);
        assert!((points[1].y + 6.0).abs() < 1e-12);
    }

    #[test]
    fn a_single_sample_curve_still_maps_onto_one_point() {
        let points = curve_points(&curve(20.0, 20_000.0, vec![-3.0]));
        assert_eq!(points.len(), 1);
        assert!((points[0].x - hz_to_x(20.0)).abs() < 1e-12);
        assert!(curve_points(&curve(20.0, 20_000.0, Vec::new())).is_empty());
        // One point is not a line, so it is not drawn.
        assert!(!Curves::is_drawable(&points));
    }

    #[test]
    fn non_finite_and_extreme_levels_are_clamped() {
        let points = curve_points(&curve(
            20.0,
            20_000.0,
            vec![f32::NAN, f32::INFINITY, 500.0, -500.0],
        ));
        // NaN and an infinity are unusable, so they land on the floor.
        assert_eq!(points[0].y, DB_FLOOR);
        assert_eq!(points[1].y, DB_FLOOR);
        // Finite outliers are clamped to the window instead of squashing it.
        assert_eq!(points[2].y, DB_CEIL);
        assert_eq!(points[3].y, DB_FLOOR);
    }

    #[test]
    fn a_degenerate_curve_edge_still_produces_finite_points() {
        let points = curve_points(&curve(0.0, 0.0, vec![0.0, 0.0]));
        assert!(points.iter().all(|p| p.x.is_finite() && p.y.is_finite()));
    }

    #[test]
    fn curves_are_dropped_when_a_frame_omits_them() {
        let mut curves = Curves::default();
        let full = TelemetryFrame {
            id: 0,
            ok: true,
            status: status(),
            measured: Some(curve(20.0, 20_000.0, vec![0.0, -6.0])),
            source_target: Some(curve(20.0, 20_000.0, vec![0.0, -6.0])),
            predicted: Some(curve(20.0, 20_000.0, vec![0.0, -6.0])),
        };
        let now = Instant::now();
        curves.update(&full, now);
        assert_eq!(curves.measured.len(), 2);
        assert_eq!(curves.measured_at, Some(now));
        assert_eq!(curves.frames, 1);
        // The cache holds exactly what the projection produces; the plot itself
        // never recomputes it.
        assert_eq!(
            curves.predicted,
            curve_points(&curve(20.0, 20_000.0, vec![0.0, -6.0]))
        );

        let sparse = TelemetryFrame {
            measured: None,
            source_target: None,
            predicted: None,
            ..full
        };
        curves.update(&sparse, now);
        assert!(
            curves.measured.is_empty(),
            "stale measured data must vanish"
        );
        assert!(curves.measured_at.is_none());
        assert_eq!(curves.frames, 2);

        curves.clear();
        assert!(curves.predicted.is_empty());
    }
}
