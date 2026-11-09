//! Application state, background threads and the main view (DESIGN.md sections
//! 5.1 and 6).
//!
//! # Threading model
//!
//! The window must stay responsive whether or not a daemon exists, so nothing in
//! the UI thread ever blocks on the socket:
//!
//! * a **telemetry thread** owns one connection, subscribes with `psd: true` and
//!   forwards every [`TelemetryFrame`] over an `mpsc` channel. When the daemon
//!   disappears it reports that, sleeps briefly and reconnects, so a restarted
//!   daemon is picked up without any user action.
//! * a **command thread** owns a second connection. Every command is a plain
//!   `Client::call`, which is blocking, so the UI only ever pushes intent into a
//!   channel and reads outcomes back non-blockingly.
//!
//! Both threads keep a cloned [`egui::Context`] and call `request_repaint` when
//! they have news, so a frame arrives within milliseconds of the daemon sending
//! it instead of waiting for the next timer tick.
//!
//! # Why the editable fields are mirrored
//!
//! The daemon echoes a change asynchronously, so a slider bound straight to the
//! last status frame would fight the user for a frame or two after every drag.
//! Every editable value therefore lives in a local draft ([`Controls`] for the
//! main view, [`LayerDraft`] for the mixer) which is refreshed from status only
//! when no change of ours is still in flight.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, TryRecvError, channel};
use std::time::{Duration, Instant};

use egui::{Color32, ComboBox, RichText, Ui};
use noise_daemon::ipc::{Client, IpcError};
use noise_daemon::protocol::{Request, Response, SetArgs, StatusData, TelemetryFrame};
use noise_dsp::params::{LayerParams, PresetParams};
use noise_dsp::{Character, Color};

use crate::advanced;
use crate::layers::{self, LayerDraft, LayerIntent};
use crate::spectrum::{self, Curves};

/// Push rate requested from the daemon, Hz.
///
/// The meters and the timer are readable at a few hertz, and every push costs a
/// repaint, so this is deliberately low rather than "as fast as possible".
const PUSH_HZ: f64 = 5.0;
/// Fallback repaint interval. Telemetry arrives at [`PUSH_HZ`] and input wakes
/// the UI by itself, so this is only a safety net for a stalled link.
const REPAINT_INTERVAL: Duration = Duration::from_millis(250);
/// Wait between reconnection attempts.
const RECONNECT_DELAY: Duration = Duration::from_millis(500);
/// How long a push read blocks before the thread looks at its control channel.
const PUSH_POLL: Duration = Duration::from_millis(200);
/// How long incoming status must not overwrite a value we just changed.
const ECHO_HOLD: Duration = Duration::from_millis(400);
/// Bound on the message log, so a stuck daemon cannot grow the window forever.
const MAX_NOTES: usize = 60;
/// How long an unchanged connection failure is re-reported to the UI.
const DISCONNECT_REPEAT: Duration = Duration::from_secs(3);
/// How long the window waits for `Stop` to reach the daemon while closing.
const SHUTDOWN_WAIT: Duration = Duration::from_millis(1200);

/// Label of the device list command, used to route its response.
const LABEL_DEVICE_LIST: &str = "device list";
/// Label of the preset list command.
const LABEL_PRESET_LIST: &str = "preset list";
/// Label of the `Stop` sent when the window closes.
const LABEL_SHUTDOWN_STOP: &str = "stop on exit";

/// Colour of a healthy state.
const GOOD: Color32 = Color32::from_rgb(120, 230, 130);
/// Colour of a failed state.
const BAD: Color32 = Color32::from_rgb(240, 90, 90);
/// Colour of a state that is neither.
const AMBER: Color32 = Color32::from_rgb(230, 190, 90);

/// What the binary hands to [`App::new`].
#[derive(Clone, Debug)]
pub struct GuiOptions {
    /// Socket the daemon listens on.
    pub socket: PathBuf,
}

/// Which view is on screen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum View {
    /// DESIGN.md section 5.1.
    Main,
    /// DESIGN.md section 5.2.
    Advanced,
}

/// Connection state of the telemetry link.
enum Link {
    /// First connection attempt in flight.
    Connecting,
    /// Telemetry is arriving.
    Up,
    /// No usable connection; the message explains why.
    Down { message: String, since: Instant },
}

/// News from the telemetry thread.
enum TelemetryEvent {
    /// The subscription was accepted.
    Connected,
    /// One telemetry frame.
    Frame(Box<TelemetryFrame>),
    /// The connection ended; the thread will retry.
    Disconnected(String),
}

/// Commands for the telemetry thread.
enum ThreadControl {
    /// Drop the connection and try immediately.
    Reconnect,
    /// Change whether telemetry frames carry the measured PSD.
    SetPsd(bool),
    /// Stop the thread.
    Shutdown,
}

/// Work for the command thread.
enum Command {
    /// Send one request. Boxed because a `Request` dwarfs the other variants and
    /// every command travels through the same small channel.
    Request {
        label: &'static str,
        request: Box<Request>,
    },
    /// Read-modify-write of `LayerParams::enabled`, which `SetArgs` cannot patch.
    LayerEnabled { index: usize, enabled: bool },
    /// Read-modify-write of `LayerParams::name`.
    LayerName { index: usize, name: String },
}

/// Result of one command.
struct Outcome {
    label: &'static str,
    result: Result<Response, String>,
}

/// One line in the message log.
#[derive(Clone, Debug)]
pub struct Note {
    /// Where the message came from, e.g. `preset load`.
    pub source: String,
    /// The message itself.
    pub text: String,
    /// True for failures.
    pub error: bool,
}

/// A selectable output sink, as reported by `Request::DeviceList`.
#[derive(Clone, Debug, PartialEq, Eq)]
struct SinkEntry {
    id: u64,
    name: String,
    description: String,
    is_default: bool,
}

/// Editable mirror of the fields the main view exposes.
#[derive(Clone, Debug, PartialEq)]
struct Controls {
    color: Color,
    character: Character,
    intensity: f64,
    custom_alpha: f64,
    high_pass: f64,
    low_pass: f64,
    stereo: f64,
    target_rms_dbfs: f64,
}

impl Default for Controls {
    fn default() -> Self {
        // The daemon's own defaults, so an empty window already looks like the
        // preset that is playing (DESIGN.md section 7).
        Self {
            color: Color::Pink,
            character: Character::Neutral,
            intensity: 0.0,
            custom_alpha: 1.0,
            high_pass: 35.0,
            low_pass: 16_000.0,
            stereo: 0.35,
            target_rms_dbfs: -36.0,
        }
    }
}

impl Controls {
    /// Take the values of layer 0 and the master section.
    fn apply(&mut self, status: &StatusData) {
        if let Some(layer) = status.layers.first() {
            self.color = Color::parse(&layer.color).unwrap_or(Color::Custom);
            self.character = Character::parse(&layer.character).unwrap_or(Character::Neutral);
            self.intensity = layer.character_amount;
            self.custom_alpha = layer.alpha;
            self.high_pass = layer.high_pass_hz;
            self.low_pass = layer.low_pass_hz;
            self.stereo = layer.stereo_correlation;
        }
        self.target_rms_dbfs = status.master.target_rms_dbfs;
    }
}

/// The window state.
pub struct App {
    socket: PathBuf,
    events: Receiver<TelemetryEvent>,
    commands: Sender<Command>,
    outcomes: Receiver<Outcome>,
    control: Sender<ThreadControl>,
    link: Link,
    status: Option<StatusData>,
    curves: Curves,
    controls: Controls,
    /// Editable mirror of the daemon's layers; owned by the mixer.
    pub(crate) layer_drafts: Vec<LayerDraft>,
    sinks: Vec<SinkEntry>,
    presets: Vec<String>,
    preset_choice: String,
    preset_name_input: String,
    preset_name_focused: bool,
    view: View,
    notes: VecDeque<Note>,
    echo_hold_until: Option<Instant>,
    /// Whether the spectrum plot is on screen, and therefore whether the daemon
    /// should spend CPU on the Welch estimate.
    want_psd: bool,
    /// Keep generating noise after this window closes. Off by default: a window
    /// that keeps playing after it disappears is a surprise nobody asked for.
    keep_playing: bool,
    /// Set once the shutdown requests have been sent, so the drop path and
    /// `on_exit` cannot send `Stop` twice.
    shutdown_done: bool,
    /// `--info`: print, then close before the user sees anything.
    exit_after_first_frame: bool,
    pub(crate) timer_minutes: f64,
    pub(crate) timer_fade_s: f64,
}

impl App {
    /// Build the state and, unless this is an `--info` run, start the
    /// background threads.
    ///
    /// `connect` is false for `--info`: subscribing would take over the audio
    /// session (the daemon hands the session to the first subscriber), and a
    /// diagnostic run must never do that.
    pub fn new(options: GuiOptions, ctx: egui::Context, connect: bool) -> Self {
        let (event_tx, event_rx) = channel();
        let (command_tx, command_rx) = channel();
        let (outcome_tx, outcome_rx) = channel();
        let (control_tx, control_rx) = channel();

        if connect {
            spawn_named("noiseforge-telemetry", {
                let socket = options.socket.clone();
                let ctx = ctx.clone();
                move || telemetry_loop(socket, event_tx, control_rx, ctx)
            });
            spawn_named("noiseforge-commands", {
                let socket = options.socket.clone();
                move || command_loop(socket, command_rx, outcome_tx, ctx)
            });
        }

        Self {
            socket: options.socket,
            events: event_rx,
            commands: command_tx,
            outcomes: outcome_rx,
            control: control_tx,
            link: Link::Connecting,
            status: None,
            curves: Curves::default(),
            controls: Controls::default(),
            layer_drafts: Vec::new(),
            sinks: Vec::new(),
            presets: Vec::new(),
            preset_choice: String::new(),
            preset_name_input: String::new(),
            preset_name_focused: false,
            view: View::Main,
            notes: VecDeque::new(),
            echo_hold_until: None,
            // The window opens on the Main view, which shows the spectrum.
            want_psd: true,
            keep_playing: false,
            shutdown_done: false,
            exit_after_first_frame: !connect,
            timer_minutes: 45.0,
            timer_fade_s: 60.0,
        }
    }

    /// Socket this window talks to, for the offline notice.
    pub fn socket(&self) -> &Path {
        &self.socket
    }

    /// Last status received, if any.
    pub fn status(&self) -> Option<&StatusData> {
        self.status.as_ref()
    }

    /// Message log, newest first.
    pub fn notes(&self) -> &VecDeque<Note> {
        &self.notes
    }

    /// Whether telemetry is currently arriving.
    fn is_connected(&self) -> bool {
        matches!(self.link, Link::Up)
    }

    /// Drain both channels. Never blocks.
    fn pump(&mut self) {
        loop {
            match self.events.try_recv() {
                Ok(TelemetryEvent::Connected) => {
                    self.link = Link::Up;
                    self.request_lists();
                }
                Ok(TelemetryEvent::Frame(frame)) => self.apply_frame(*frame),
                Ok(TelemetryEvent::Disconnected(message)) => self.on_disconnect(message),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    // The thread is gone; the window keeps working and says so.
                    self.on_disconnect("the telemetry thread stopped".to_string());
                    break;
                }
            }
        }
        while let Ok(outcome) = self.outcomes.try_recv() {
            self.apply_outcome(outcome);
        }
    }

    /// Ask for the state a fresh connection cannot infer from telemetry.
    fn request_lists(&mut self) {
        self.send_request(LABEL_DEVICE_LIST, Request::DeviceList);
        self.send_request(LABEL_PRESET_LIST, Request::PresetList);
        self.send_request("status", Request::Status);
    }

    /// Fold one telemetry frame into the UI state.
    fn apply_frame(&mut self, frame: TelemetryFrame) {
        let now = Instant::now();
        let holding = self.echo_hold_until.is_some_and(|until| now < until);
        if !holding {
            self.controls.apply(&frame.status);
            layers::sync_drafts(&mut self.layer_drafts, &frame.status.layers);
            self.echo_hold_until = None;
        }
        self.curves.update(&frame, now);
        if !self.preset_name_focused {
            self.preset_name_input.clone_from(&frame.status.preset_name);
        }
        if self.preset_choice.is_empty() {
            self.preset_choice.clone_from(&frame.status.preset_name);
        }
        self.status = Some(*frame.status);
        self.link = Link::Up;
    }

    /// The daemon went away: drop values that are no longer live.
    fn on_disconnect(&mut self, message: String) {
        self.curves.clear();
        // A stale meter reading is worse than no reading, so the status is
        // dropped while the editable drafts survive: the user's settings should
        // not jump back to defaults just because the daemon restarted.
        self.status = None;
        let already = match &self.link {
            Link::Down {
                message: previous, ..
            } => previous == &message,
            _ => false,
        };
        if !already {
            self.push_note("link", message.clone(), true);
        }
        self.link = Link::Down {
            message,
            since: Instant::now(),
        };
    }

    /// Fold one command result into the UI state.
    fn apply_outcome(&mut self, outcome: Outcome) {
        match outcome.label {
            LABEL_DEVICE_LIST => {
                if let Ok(response) = &outcome.result
                    && let Some(data) = &response.data
                {
                    self.sinks = parse_sinks(data);
                }
            }
            LABEL_PRESET_LIST => {
                if let Ok(response) = &outcome.result
                    && let Some(data) = &response.data
                {
                    self.presets = parse_presets(data);
                }
            }
            _ => {}
        }
        match outcome.result {
            Ok(response) => {
                for note in response.notes {
                    self.push_note(outcome.label, note, false);
                }
                if !response.ok {
                    let message = response
                        .error
                        .unwrap_or_else(|| "the daemon rejected the request".to_string());
                    self.push_note(outcome.label, message, true);
                }
            }
            Err(message) => self.push_note(outcome.label, message, true),
        }
    }

    /// Append to the bounded message log, skipping an immediate repeat.
    fn push_note(&mut self, source: &str, text: impl Into<String>, error: bool) {
        let text = text.into();
        if let Some(last) = self.notes.front()
            && last.source == source
            && last.text == text
            && last.error == error
        {
            return;
        }
        self.notes.push_front(Note {
            source: source.to_string(),
            text,
            error,
        });
        while self.notes.len() > MAX_NOTES {
            self.notes.pop_back();
        }
    }

    /// Send one request on the command connection.
    pub fn send_request(&mut self, label: &'static str, request: Request) {
        if self
            .commands
            .send(Command::Request {
                label,
                request: Box::new(request),
            })
            .is_err()
        {
            self.push_note(label, "the command thread is not running", true);
        }
    }

    /// Send a `Set` patch for `layer` (0 when `None`) and hold the drafts.
    pub fn send_set(
        &mut self,
        label: &'static str,
        layer: Option<usize>,
        edit: impl FnOnce(&mut SetArgs),
    ) {
        let mut args = SetArgs {
            layer,
            ..SetArgs::default()
        };
        edit(&mut args);
        self.echo_hold_until = Some(Instant::now() + ECHO_HOLD);
        self.send_request(
            label,
            Request::Set {
                args: Box::new(args),
            },
        );
    }

    /// Send a read-modify-write command and hold the drafts.
    fn send_edit(&mut self, label: &'static str, command: Command) {
        self.echo_hold_until = Some(Instant::now() + ECHO_HOLD);
        if self.commands.send(command).is_err() {
            self.push_note(label, "the command thread is not running", true);
        }
    }

    /// Apply one mixer edit.
    pub fn apply_layer_intent(&mut self, intent: LayerIntent) {
        match intent {
            LayerIntent::Enable(index, enabled) => {
                self.send_edit("layer enable", Command::LayerEnabled { index, enabled })
            }
            LayerIntent::Rename(index, name) => {
                self.send_edit("layer rename", Command::LayerName { index, name });
            }
            LayerIntent::SetColor(index, color) => {
                self.send_set("layer colour", Some(index), |args| {
                    args.color = Some(color.as_str().to_string());
                })
            }
            LayerIntent::SetCharacter(index, character) => {
                self.send_set("layer character", Some(index), |args| {
                    args.character = Some(character.as_str().to_string());
                });
            }
            LayerIntent::Gain(index, gain_db) => self.send_set("layer gain", Some(index), |args| {
                args.gain_db = Some(gain_db);
            }),
            LayerIntent::Stereo(index, stereo) => {
                self.send_set("layer stereo", Some(index), |args| {
                    args.stereo = Some(stereo);
                })
            }
            // Structural changes are not held: the layer list has to grow or
            // shrink immediately or the row would lag behind the click.
            LayerIntent::Add => self.send_request("layer add", Request::LayerAdd { after: None }),
            LayerIntent::Duplicate(index) => {
                self.send_request("layer duplicate", Request::LayerDuplicate { index });
            }
            LayerIntent::Remove(index) => {
                self.send_request("layer remove", Request::LayerRemove { index });
            }
        }
    }

    /// Ask the telemetry thread to reconnect at once.
    fn retry(&mut self) {
        if self.control.send(ThreadControl::Reconnect).is_err() {
            self.push_note("link", "the telemetry thread is not running", true);
        }
    }

    /// Switch the visible view, telling the daemon to drop the PSD work when the
    /// spectrum plot is no longer on screen.
    fn show_view(&mut self, view: View) {
        if self.view == view {
            return;
        }
        self.view = view;
        let want_psd = matches!(view, View::Main);
        if want_psd != self.want_psd {
            self.want_psd = want_psd;
            self.sync_subscription();
        }
    }

    /// Re-assert the subscription after a view change.
    fn sync_subscription(&mut self) {
        if self
            .control
            .send(ThreadControl::SetPsd(self.want_psd))
            .is_err()
        {
            self.push_note("link", "the telemetry thread is not running", true);
        }
    }

    /// Stop playback (unless the user asked otherwise) and stop the threads.
    ///
    /// Called from the close-request path, from `on_exit` and from `Drop`, and
    /// guarded so the `Stop` is sent at most once. The wait matters: the command
    /// thread exits as soon as this app is dropped, so returning immediately
    /// could lose the request and leave noise playing after the window is gone.
    fn shutdown(&mut self) {
        if self.shutdown_done {
            return;
        }
        self.shutdown_done = true;
        let _ = self.control.send(ThreadControl::Shutdown);
        if self.keep_playing {
            eprintln!("noise-gui: leaving the daemon playing (\"keep playing\" is on)");
            return;
        }
        self.send_request(LABEL_SHUTDOWN_STOP, Request::Stop);
        let deadline = Instant::now() + SHUTDOWN_WAIT;
        while Instant::now() < deadline {
            let left = deadline.saturating_duration_since(Instant::now());
            match self.outcomes.recv_timeout(left) {
                Ok(outcome) => {
                    let last = outcome.label == LABEL_SHUTDOWN_STOP;
                    self.apply_outcome(outcome);
                    if last {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    }

    /// Colour and text of the link indicator.
    fn link_badge(&self) -> (Color32, String) {
        match &self.link {
            Link::Connecting => (AMBER, "PipeWire: connecting".to_string()),
            Link::Up => {
                let text = self
                    .status
                    .as_ref()
                    .map(|status| status.link.clone())
                    .unwrap_or_else(|| "running".to_string());
                (link_color(&text), format!("PipeWire: {text}"))
            }
            Link::Down { .. } => (BAD, "PipeWire: no daemon".to_string()),
        }
    }

    /// Title bar: name, link indicator, view switch and output device.
    fn header(&mut self, ui: &mut Ui) {
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.heading("NoiseForge");
            ui.separator();
            let (color, text) = self.link_badge();
            dot(ui, color);
            ui.colored_label(color, text);
            ui.separator();
            let mut wanted = self.view;
            ui.selectable_value(&mut wanted, View::Main, "Main");
            ui.selectable_value(&mut wanted, View::Advanced, "Advanced");
            self.show_view(wanted);
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                self.device_combo(ui);
            });
        });
        ui.add_space(4.0);
    }

    /// Output device selector, fed by `Request::DeviceList`.
    fn device_combo(&mut self, ui: &mut Ui) {
        let selected = self
            .status
            .as_ref()
            .and_then(|status| status.device.clone())
            .unwrap_or_else(|| "default sink".to_string());
        let mut chosen: Option<String> = None;
        ComboBox::from_id_salt("nf-output")
            .selected_text(selected)
            .width(280.0)
            .show_ui(ui, |ui| {
                if ui.selectable_label(false, "default sink").clicked() {
                    chosen = Some("default".to_string());
                }
                for sink in &self.sinks {
                    let label = if sink.is_default {
                        format!("{} (default)", sink.description)
                    } else {
                        format!("{} (id {})", sink.description, sink.id)
                    };
                    if ui.selectable_label(false, label).clicked() {
                        chosen = Some(sink.name.clone());
                    }
                }
            });
        if ui.small_button("refresh").clicked() {
            self.send_request(LABEL_DEVICE_LIST, Request::DeviceList);
            self.send_request(LABEL_PRESET_LIST, Request::PresetList);
        }
        if let Some(name) = chosen {
            self.send_request("device select", Request::DeviceSelect { name });
        }
    }

    /// The disconnected banner with its retry button.
    fn offline_notice(&mut self, ui: &mut Ui) {
        let Link::Down { message, since } = &self.link else {
            return;
        };
        let message = message.clone();
        let waited = since.elapsed().as_secs_f32();
        egui::Frame::NONE
            .fill(Color32::from_rgb(70, 28, 28))
            .inner_margin(8.0)
            .corner_radius(4u8)
            .show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.colored_label(BAD, RichText::new("daemon not running").strong());
                    ui.label(format!("{}: {message}", self.socket().display()));
                    if ui.button("Retry now").clicked() {
                        self.retry();
                    }
                    ui.label(
                        RichText::new(format!(
                            "retrying automatically; last failure {waited:.0} s ago"
                        ))
                        .small()
                        .weak(),
                    );
                });
            });
        ui.add_space(6.0);
    }

    /// The bottom strip: meters and counters.
    fn meters(&mut self, ui: &mut Ui) {
        ui.add_space(2.0);
        let Some(status) = self.status.as_ref() else {
            ui.horizontal(|ui| {
                ui.label(RichText::new("meters unavailable while disconnected").weak());
            });
            ui.add_space(2.0);
            return;
        };
        let meters = status.meters.clone();
        let target = status.calibration.target_rms_dbfs;
        let playing = status.playing;
        ui.horizontal(|ui| {
            ui.label(RichText::new(if playing { "PLAYING" } else { "PAUSED" }).strong());
            ui.separator();
            ui.label(format!("peak {:+.2} dBFS", meters.peak_dbfs));
            ui.label(format!("rms {:+.2} dBFS", meters.rms_dbfs));
            ui.label(format!("target {target:+.2} dBFS"));
            ui.add(
                egui::ProgressBar::new(peak_fraction(meters.peak_dbfs))
                    .desired_width(120.0)
                    .text("peak"),
            );
            ui.separator();
            ui.label(format!("xruns {}", meters.xruns));
            let faults = meters.clipped.saturating_add(meters.nan_events);
            if faults > 0 {
                ui.colored_label(
                    BAD,
                    RichText::new(format!(
                        "CLIP {} / NaN {}",
                        meters.clipped, meters.nan_events
                    ))
                    .strong(),
                );
            } else {
                ui.colored_label(GOOD, "clean");
            }
            ui.label(
                RichText::new(format!(
                    "frames {} / callbacks {} / tap dropped {}",
                    meters.frames, meters.callbacks, meters.tap_dropped
                ))
                .small()
                .weak(),
            );
        });
        ui.add_space(2.0);
    }

    /// DESIGN.md section 5.1.
    fn main_view(&mut self, ui: &mut Ui) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            let connected = self.is_connected();
            ui.add_enabled_ui(connected, |ui| {
                self.controls_ui(ui);
                ui.separator();
                self.preset_ui(ui);
            });

            ui.separator();
            ui.heading("Spectrum (log frequency / dB relative)");
            spectrum::ui(ui, &self.curves);
        });
    }

    /// Colour, character, filters and level.
    fn controls_ui(&mut self, ui: &mut Ui) {
        ui.heading("Color");
        ui.horizontal_wrapped(|ui| {
            let mut chosen = None;
            for color in Color::ALL {
                if ui
                    .selectable_label(self.controls.color == color, color.label())
                    .clicked()
                {
                    chosen = Some(color);
                }
            }
            if let Some(color) = chosen {
                self.controls.color = color;
                self.send_set("colour", Some(0), |args| {
                    args.color = Some(color.as_str().to_string());
                });
            }
        });
        if self.controls.color == Color::Custom {
            let mut alpha = self.controls.custom_alpha;
            if ui
                .add(
                    egui::Slider::new(
                        &mut alpha,
                        noise_dsp::colors::ALPHA_MIN..=noise_dsp::colors::ALPHA_MAX,
                    )
                    .text("alpha (custom)")
                    .fixed_decimals(3),
                )
                .changed()
            {
                self.controls.custom_alpha = alpha;
                self.send_set("alpha", Some(0), |args| args.alpha = Some(alpha));
            }
        }

        ui.add_space(4.0);
        ui.heading("Character");
        ui.horizontal_wrapped(|ui| {
            let mut chosen = None;
            for character in Character::ALL {
                if ui
                    .selectable_label(self.controls.character == character, character.label())
                    .clicked()
                {
                    chosen = Some(character);
                }
            }
            if let Some(character) = chosen {
                self.controls.character = character;
                self.send_set("character", Some(0), |args| {
                    args.character = Some(character.as_str().to_string());
                });
            }
        });
        let mut intensity = self.controls.intensity;
        if ui
            .add(
                egui::Slider::new(&mut intensity, 0.0..=1.0)
                    .text("Intensity")
                    .custom_formatter(|value, _| format!("{:.0}%", value * 100.0)),
            )
            .changed()
        {
            self.controls.intensity = intensity;
            self.send_set("intensity", Some(0), |args| {
                args.intensity = Some(intensity);
            });
        }

        ui.add_space(4.0);
        ui.heading("Filters and level");
        let mut high_pass = self.controls.high_pass;
        if ui
            .add(
                egui::Slider::new(&mut high_pass, 0.0..=2000.0)
                    .text("High-pass")
                    .suffix(" Hz")
                    .logarithmic(true)
                    // The range has to include zero because zero means "no high
                    // pass", but without a smallest positive value egui would
                    // stretch the track down to 1e-6 Hz and squeeze the useful
                    // 20-2000 Hz into its last fifth.
                    .smallest_positive(20.0)
                    .fixed_decimals(0),
            )
            .changed()
        {
            self.controls.high_pass = high_pass;
            self.send_set("high-pass", Some(0), |args| {
                args.high_pass = Some(high_pass);
            });
        }
        let mut low_pass = self.controls.low_pass;
        if ui
            .add(
                egui::Slider::new(&mut low_pass, 200.0..=20_000.0)
                    .text("Low-pass")
                    .suffix(" Hz")
                    .logarithmic(true)
                    .fixed_decimals(0),
            )
            .changed()
        {
            self.controls.low_pass = low_pass;
            self.send_set("low-pass", Some(0), |args| {
                args.low_pass = Some(low_pass);
            });
        }
        let mut stereo = self.controls.stereo;
        let width = stereo_width_label(stereo);
        if ui
            .add(
                egui::Slider::new(&mut stereo, -1.0..=1.0)
                    .text(format!("Stereo ({width})"))
                    .fixed_decimals(2),
            )
            .changed()
        {
            self.controls.stereo = stereo;
            self.send_set("stereo", Some(0), |args| args.stereo = Some(stereo));
        }
        let mut level = self.controls.target_rms_dbfs;
        if ui
            .add(
                egui::Slider::new(&mut level, -80.0..=0.0)
                    .text("Volume (target RMS)")
                    .suffix(" dBFS")
                    .fixed_decimals(1),
            )
            .changed()
        {
            self.controls.target_rms_dbfs = level;
            self.send_set("target level", None, |args| {
                args.level_rms_dbfs = Some(level);
            });
        }
        ui.label(
            RichText::new(
                "Intensity is the character amount; Volume is the expected RMS of the output PCM. \
                 They are independent, and the calibration keeps the second one true.",
            )
            .small()
            .weak(),
        );
    }

    /// Preset selection, save and the transport button.
    fn preset_ui(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.label("Preset");
            let mut load_requested = false;
            ComboBox::from_id_salt("nf-preset")
                .selected_text(if self.preset_choice.is_empty() {
                    "(none)".to_string()
                } else {
                    self.preset_choice.clone()
                })
                .width(220.0)
                .show_ui(ui, |ui| {
                    for name in &self.presets {
                        if ui
                            .selectable_label(self.preset_choice == *name, name.as_str())
                            .clicked()
                        {
                            self.preset_choice.clone_from(name);
                            load_requested = true;
                        }
                    }
                });
            if load_requested {
                let name = self.preset_choice.clone();
                self.send_request("preset load", Request::PresetLoad { name });
            }

            ui.separator();
            let name_response = ui.add(
                egui::TextEdit::singleline(&mut self.preset_name_input)
                    .desired_width(160.0)
                    .hint_text("preset name"),
            );
            self.preset_name_focused = name_response.has_focus();
            if ui.button("Save").clicked() {
                let name = self.preset_name_input.trim().to_string();
                if name.is_empty() {
                    self.push_note("preset save", "give the preset a name first", true);
                } else {
                    self.send_request("preset save", Request::PresetSave { name });
                }
            }
        });

        ui.add_space(6.0);
        let playing = self.status.as_ref().is_some_and(|status| status.playing);
        let label = if playing { "PAUSE" } else { "PLAY" };
        let request = if playing {
            Request::Pause
        } else {
            Request::Play
        };
        let button = egui::Button::new(RichText::new(label).strong().size(20.0))
            .min_size(egui::Vec2::new(220.0, 40.0));
        if ui.add(button).clicked() {
            ui.ctx().request_repaint();
            self.send_request(if playing { "pause" } else { "play" }, request);
        }
        ui.add_space(6.0);
        ui.checkbox(&mut self.keep_playing, "Keep playing in the background")
            .on_hover_text(
                "Keep generating noise after this window closes; stop it with `noiseforge stop`.",
            );
        if !self.keep_playing {
            ui.label(
                RichText::new("Closing this window stops playback.")
                    .small()
                    .weak(),
            );
        }
    }
}

impl Drop for App {
    fn drop(&mut self) {
        // Safety net for any teardown path that skips `on_exit`.
        self.shutdown();
    }
}

impl eframe::App for App {
    fn logic(&mut self, _ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.pump();
    }

    fn ui(&mut self, ui: &mut Ui, _frame: &mut eframe::Frame) {
        if self.exit_after_first_frame {
            // `--info` has printed everything it came for.
            ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
            return;
        }
        self.pump();
        if ui.input(|input| input.viewport().close_requested()) {
            // Stop as soon as the close is requested, so the daemon is silent
            // before the last frame is even painted.
            self.shutdown();
        }

        egui::Panel::top("nf-header").show(ui, |ui| self.header(ui));
        egui::Panel::bottom("nf-meters").show(ui, |ui| self.meters(ui));
        egui::CentralPanel::default().show(ui, |ui| {
            self.offline_notice(ui);
            match self.view {
                View::Main => self.main_view(ui),
                View::Advanced => advanced::ui(ui, self),
            }
        });

        // One repaint per telemetry frame plus this fallback; no continuous
        // animation, so an idle window costs almost nothing.
        ui.ctx().request_repaint_after(REPAINT_INTERVAL);
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        self.shutdown();
    }
}

/// Colour for a link state string from the daemon.
///
/// The daemon forwards `noise_pipewire::lifecycle::LinkState::label()`, and the
/// null backend reports `null` or `closed`, so the mapping is textual on
/// purpose: an unknown future state must show as a warning, not as healthy.
pub fn link_color(link: &str) -> Color32 {
    let link = link.trim().to_ascii_lowercase();
    if link.starts_with("error") {
        BAD
    } else if link.starts_with("running") {
        GOOD
    } else {
        AMBER
    }
}

/// Human readable stereo width for a correlation coefficient.
pub fn stereo_width_label(correlation: f64) -> &'static str {
    let rho = if correlation.is_finite() {
        correlation.clamp(-1.0, 1.0)
    } else {
        0.0
    };
    if rho >= 0.99 {
        "Mono"
    } else if rho >= 0.5 {
        "Narrow"
    } else if rho >= 0.1 {
        "Natural"
    } else if rho > -0.5 {
        "Wide"
    } else {
        "Extra wide"
    }
}

/// Fraction of a -60..0 dBFS scale, for the peak bar.
pub fn peak_fraction(peak_dbfs: f64) -> f32 {
    if !peak_dbfs.is_finite() {
        return 0.0;
    }
    (((peak_dbfs + 60.0) / 60.0).clamp(0.0, 1.0)) as f32
}

/// A small filled dot, used for the link indicator.
fn dot(ui: &mut Ui, color: Color32) {
    let (response, painter) = ui.allocate_painter(egui::Vec2::splat(12.0), egui::Sense::hover());
    painter.circle_filled(response.rect.center(), 5.0, color);
}

/// Parse a `DeviceList` payload.
fn parse_sinks(data: &serde_json::Value) -> Vec<SinkEntry> {
    data.get("sinks")
        .and_then(|value| value.as_array())
        .map(|sinks| {
            sinks
                .iter()
                .filter_map(|sink| {
                    Some(SinkEntry {
                        id: sink.get("id").and_then(serde_json::Value::as_u64)?,
                        name: sink.get("name")?.as_str()?.to_string(),
                        description: sink
                            .get("description")
                            .and_then(|value| value.as_str())
                            .unwrap_or_default()
                            .to_string(),
                        is_default: sink
                            .get("default")
                            .and_then(serde_json::Value::as_bool)
                            .unwrap_or(false),
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Parse a `PresetList` payload.
fn parse_presets(data: &serde_json::Value) -> Vec<String> {
    data.get("presets")
        .and_then(|value| value.as_array())
        .map(|names| {
            names
                .iter()
                .filter_map(|name| name.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

/// Spawn a named thread, reporting (not panicking on) a failure.
fn spawn_named(name: &str, body: impl FnOnce() + Send + 'static) {
    if let Err(error) = std::thread::Builder::new()
        .name(name.to_string())
        .spawn(body)
    {
        eprintln!("noise-gui: could not start the {name} thread: {error}");
    }
}

/// Emit one event to the UI, waking it up. False when the UI is gone.
fn emit(events: &Sender<TelemetryEvent>, ctx: &egui::Context, event: TelemetryEvent) -> bool {
    if events.send(event).is_err() {
        return false;
    }
    ctx.request_repaint();
    true
}

/// Decode a telemetry push.
fn decode_frame(response: &Response) -> Option<TelemetryFrame> {
    let data = response.data.clone()?;
    serde_json::from_value(data).ok()
}

/// Throttles repeated connection failures so the UI is not woken every retry.
#[derive(Default)]
struct DisconnectReporter {
    last: Option<String>,
    last_at: Option<Instant>,
}

impl DisconnectReporter {
    /// Whether this failure is news: a new message, or the same one after a
    /// while. The UI shows "retrying since ...", so an occasional refresh keeps
    /// that clock honest without waking it twice a second.
    fn should_report(&mut self, message: &str) -> bool {
        let now = Instant::now();
        let repeated = self.last.as_deref() == Some(message)
            && self
                .last_at
                .is_some_and(|at| now.duration_since(at) < DISCONNECT_REPEAT);
        if repeated {
            return false;
        }
        self.last = Some(message.to_string());
        self.last_at = Some(now);
        true
    }

    /// Forget the last failure, because the connection is back.
    fn reset(&mut self) {
        self.last = None;
        self.last_at = None;
    }
}

/// Handshake and subscribe on a fresh connection.
fn subscribe(client: &mut Client, psd: bool) -> Result<Response, noise_daemon::ipc::IpcError> {
    client.call(&Request::Hello {
        client: "noise-gui".to_string(),
    })?;
    client.call(&Request::Subscribe { psd, hz: PUSH_HZ })
}

/// Own the subscription, forward frames, reconnect forever.
fn telemetry_loop(
    socket: PathBuf,
    events: Sender<TelemetryEvent>,
    control: Receiver<ThreadControl>,
    ctx: egui::Context,
) {
    // Frames carry the PSD until the UI says the spectrum is off screen; the
    // first frame usually arrives before the user can switch views anyway.
    let mut want_psd = true;
    let mut reporter = DisconnectReporter::default();
    loop {
        match Client::connect(&socket) {
            Ok(mut client) => {
                match subscribe(&mut client, want_psd) {
                    Ok(response) if response.ok => {
                        reporter.reset();
                        if !emit(&events, &ctx, TelemetryEvent::Connected) {
                            return;
                        }
                        loop {
                            match client.next_push(PUSH_POLL) {
                                Ok(Some(response)) => {
                                    if let Some(frame) = decode_frame(&response)
                                        && !emit(
                                            &events,
                                            &ctx,
                                            TelemetryEvent::Frame(Box::new(frame)),
                                        )
                                    {
                                        return;
                                    }
                                }
                                Ok(None) => {}
                                Err(error) => {
                                    let message = error.to_string();
                                    if reporter.should_report(&message)
                                        && !emit(
                                            &events,
                                            &ctx,
                                            TelemetryEvent::Disconnected(message),
                                        )
                                    {
                                        return;
                                    }
                                    break;
                                }
                            }
                            match control.try_recv() {
                                Ok(ThreadControl::Shutdown) => return,
                                Ok(ThreadControl::SetPsd(psd)) if psd != want_psd => {
                                    want_psd = psd;
                                    // Re-subscribing on the live connection is
                                    // how the daemon is told to stop computing
                                    // the Welch estimate.
                                    if subscribe(&mut client, want_psd).is_err() {
                                        break;
                                    }
                                }
                                Ok(_) | Err(_) => {}
                            }
                        }
                    }
                    Ok(response) => {
                        let message = response
                            .error
                            .unwrap_or_else(|| "the daemon refused the subscription".to_string());
                        if reporter.should_report(&message)
                            && !emit(&events, &ctx, TelemetryEvent::Disconnected(message))
                        {
                            return;
                        }
                    }
                    Err(error) => {
                        let message = error.to_string();
                        if reporter.should_report(&message)
                            && !emit(&events, &ctx, TelemetryEvent::Disconnected(message))
                        {
                            return;
                        }
                    }
                }
            }
            Err(error) => {
                let message = error.to_string();
                if reporter.should_report(&message)
                    && !emit(&events, &ctx, TelemetryEvent::Disconnected(message))
                {
                    return;
                }
            }
        }

        // Back off, but wake up at once when the user asks for a retry.
        match control.recv_timeout(RECONNECT_DELAY) {
            Ok(ThreadControl::Reconnect) | Err(RecvTimeoutError::Timeout) => {}
            // A PSD change while disconnected is picked up by the next
            // `subscribe`, which uses the stored value.
            Ok(ThreadControl::SetPsd(_)) => {}
            Ok(ThreadControl::Shutdown) | Err(RecvTimeoutError::Disconnected) => return,
        }
    }
}

/// Run commands on a connection of its own, reconnecting lazily.
fn command_loop(
    socket: PathBuf,
    commands: Receiver<Command>,
    outcomes: Sender<Outcome>,
    ctx: egui::Context,
) {
    let mut client: Option<Client> = None;
    while let Ok(command) = commands.recv() {
        let outcome = run_command(&mut client, &socket, command);
        if outcomes.send(outcome).is_err() {
            return;
        }
        ctx.request_repaint();
    }
}

/// Turn one command into an outcome.
fn run_command(client: &mut Option<Client>, socket: &Path, command: Command) -> Outcome {
    match command {
        Command::Request { label, request } => Outcome {
            label,
            result: with_client(client, socket, |active| active.call(request.as_ref())),
        },
        Command::LayerEnabled { index, enabled } => Outcome {
            label: "layer enable",
            result: edit_layer(client, socket, index, |layer| layer.enabled = enabled),
        },
        Command::LayerName { index, name } => Outcome {
            label: "layer rename",
            result: edit_layer(client, socket, index, |layer| layer.name = name.clone()),
        },
    }
}

/// Run one call, connecting first when needed.
///
/// A failed call drops the connection so the next command starts from a fresh
/// one: after a daemon restart every cached socket is dead.
fn with_client<T>(
    client: &mut Option<Client>,
    socket: &Path,
    call: impl FnOnce(&mut Client) -> Result<T, IpcError>,
) -> Result<T, String> {
    if client.is_none() {
        *client = Some(Client::connect(socket).map_err(|error| error.to_string())?);
    }
    let Some(active) = client.as_mut() else {
        return Err("no connection".to_string());
    };
    match call(active) {
        Ok(value) => Ok(value),
        Err(error) => {
            *client = None;
            Err(error.to_string())
        }
    }
}

/// Read, patch and write back the whole preset, for fields `SetArgs` lacks.
fn edit_layer(
    client: &mut Option<Client>,
    socket: &Path,
    index: usize,
    edit: impl FnOnce(&mut LayerParams),
) -> Result<Response, String> {
    let response = with_client(client, socket, |active| active.call(&Request::GetPreset))?;
    let data = response
        .data
        .ok_or_else(|| "the daemon returned no preset".to_string())?;
    let mut preset: PresetParams =
        serde_json::from_value(data).map_err(|error| error.to_string())?;
    let Some(layer) = preset.layers.get_mut(index) else {
        return Err(format!("layer {index} does not exist"));
    };
    edit(layer);
    with_client(client, socket, |active| {
        active.call(&Request::Load {
            preset: Box::new(preset),
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use noise_daemon::protocol::{
        BackendKind, CalibrationStatus, LayerStatus, MeterStatus, RateMode,
    };
    use noise_dsp::params::MasterParams;
    use serde_json::json;

    #[test]
    fn link_colours_follow_the_daemon_strings() {
        assert_eq!(link_color("running"), GOOD);
        assert_eq!(link_color("running (idle)"), GOOD);
        assert_eq!(link_color("RUNNING"), GOOD);
        assert_eq!(link_color("error"), BAD);
        assert_eq!(link_color("error (server gone)"), BAD);
        // Unknown and neutral states must not look healthy.
        assert_eq!(link_color("null"), AMBER);
        assert_eq!(link_color("closed"), AMBER);
        assert_eq!(link_color("connecting"), AMBER);
        assert_eq!(link_color(""), AMBER);
    }

    #[test]
    fn stereo_width_labels_cover_the_full_range() {
        assert_eq!(stereo_width_label(1.0), "Mono");
        assert_eq!(stereo_width_label(0.99), "Mono");
        assert_eq!(stereo_width_label(0.75), "Narrow");
        assert_eq!(stereo_width_label(0.35), "Natural");
        assert_eq!(stereo_width_label(0.0), "Wide");
        assert_eq!(stereo_width_label(-0.6), "Extra wide");
        assert_eq!(stereo_width_label(-1.0), "Extra wide");
        // Out of range and non-finite input is clamped, never extrapolated.
        assert_eq!(stereo_width_label(4.0), "Mono");
        assert_eq!(stereo_width_label(f64::NAN), "Wide");
    }

    #[test]
    fn peak_fraction_maps_a_sixty_db_scale() {
        assert_eq!(peak_fraction(0.0), 1.0);
        assert_eq!(peak_fraction(-60.0), 0.0);
        assert!((peak_fraction(-30.0) - 0.5).abs() < 1e-6);
        assert_eq!(peak_fraction(-200.0), 0.0);
        assert_eq!(peak_fraction(6.0), 1.0);
        assert_eq!(peak_fraction(f64::NEG_INFINITY), 0.0);
        assert_eq!(peak_fraction(f64::NAN), 0.0);
    }

    #[test]
    fn repeated_disconnects_do_not_wake_the_ui() {
        let mut reporter = DisconnectReporter::default();
        // The first failure is always news.
        assert!(reporter.should_report("no daemon"));
        // An immediate repeat is not, so a 500 ms retry loop cannot turn into a
        // repaint loop.
        assert!(!reporter.should_report("no daemon"));
        assert!(!reporter.should_report("no daemon"));
        // A different failure is news again.
        assert!(reporter.should_report("connection reset"));
        // So is the same failure once the throttle window has passed.
        reporter.last_at = Some(Instant::now() - DISCONNECT_REPEAT * 2);
        assert!(reporter.should_report("connection reset"));
        // After a successful connect the history is cleared.
        reporter.reset();
        assert!(reporter.should_report("no daemon"));
    }

    #[test]
    fn controls_take_the_first_layer_and_the_master_target() {
        let status = StatusData {
            version: "0.1.0".into(),
            protocol: 1,
            dsp_version: 1,
            backend: BackendKind::PipeWire,
            playing: true,
            transport: "playing".into(),
            link: "running".into(),
            sample_rate: 48_000,
            channels: 2,
            rate_mode: RateMode::Follow,
            device: Some("sink".into()),
            device_node_id: Some(3),
            timer_remaining_s: None,
            uptime_s: 1.0,
            calibration: CalibrationStatus {
                target_rms_dbfs: -30.0,
                mix_rms_dbfs: -31.0,
                calibration_gain_db: 1.0,
                master_gain_db: 1.0,
                predicted_peak_dbfs: -12.0,
                headroom_warning: false,
            },
            meters: MeterStatus::default(),
            layers: vec![LayerStatus {
                index: 0,
                name: "Main".into(),
                enabled: true,
                color: "violet".into(),
                alpha: -2.0,
                character: "intense".into(),
                character_amount: 0.4,
                high_pass_hz: 20.0,
                low_pass_hz: 12_000.0,
                stereo_correlation: -0.7,
                gain_db: 0.0,
                seed: 7,
                sections: 9,
                shape_error_db: Some(0.1),
                slope_error_db_per_octave: Some(0.0),
                verified: true,
                predicted_rms_dbfs: -34.0,
            }],
            master: MasterParams {
                target_rms_dbfs: -30.0,
                ..MasterParams::default()
            },
            notes: Vec::new(),
            preset_name: "Soft".into(),
            seed: 7,
            auto_start: false,
        };
        let mut controls = Controls::default();
        controls.apply(&status);
        assert_eq!(controls.color, Color::Violet);
        assert_eq!(controls.character, Character::Intense);
        assert_eq!(controls.intensity, 0.4);
        assert_eq!(controls.high_pass, 20.0);
        assert_eq!(controls.low_pass, 12_000.0);
        assert_eq!(controls.stereo, -0.7);
        assert_eq!(controls.target_rms_dbfs, -30.0);
        assert_eq!(stereo_width_label(controls.stereo), "Extra wide");
    }

    #[test]
    fn device_and_preset_payloads_are_parsed_defensively() {
        let sinks = parse_sinks(&json!({
            "sinks": [
                {"id": 54, "name": "alsa_out", "description": "Speakers", "default": true},
                {"id": 61, "name": "hdmi", "description": "HDMI"},
                {"name": "no id"},
                "junk"
            ]
        }));
        assert_eq!(sinks.len(), 2);
        assert_eq!(sinks[0].id, 54);
        assert!(sinks[0].is_default);
        assert_eq!(sinks[1].description, "HDMI");
        assert!(!sinks[1].is_default);
        // Anything unexpected yields an empty list instead of a panic.
        assert!(parse_sinks(&json!({})).is_empty());
        assert!(parse_sinks(&json!({"sinks": 3})).is_empty());

        let presets = parse_presets(&json!({"presets": ["Soft Pink", 5, "Brown"]}));
        assert_eq!(presets, vec!["Soft Pink".to_string(), "Brown".to_string()]);
        assert!(parse_presets(&json!({"presets": null})).is_empty());
        assert!(parse_presets(&json!({"presets": {}})).is_empty());
    }
}
