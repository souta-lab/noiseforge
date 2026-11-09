//! The daemon control loop (DESIGN.md sections 4.4 and 6).
//!
//! One thread runs everything that is *not* realtime:
//!
//! 1. accept and poll Unix socket clients,
//! 2. apply configuration changes and **redesign coefficients**,
//! 3. push only the commands that actually changed, with the right crossfade and
//!    master gain hold times,
//! 4. drain the analysis tap and estimate the output PSD,
//! 5. push telemetry to subscribers.
//!
//! The realtime thread only ever sees pre-designed POD commands, which is the
//! contract in DESIGN.md section 4.4.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use noise_dsp::colors::{self, Color};
use noise_dsp::compile::{CompiledGraph, compile_with};
use noise_dsp::designer::DesignOptions;
use noise_dsp::engine::Command;
use noise_dsp::params::{LayerParams, PresetParams};
use noise_offline::analysis::{WelchConfig, log_resample, welch_psd};

use crate::backend::{AudioBackend, BackendEvent, NullBackend, OutputFormat, PipeWireBackend};
use crate::ipc::{Client, Connection, IpcError, Server};
use crate::presets::Store;
use crate::protocol::*;

/// Runtime configuration of the daemon.
#[derive(Clone, Debug)]
pub struct DaemonConfig {
    /// Socket to listen on.
    pub socket: PathBuf,
    /// Preset and configuration store.
    pub store: Store,
    /// Audio backend.
    pub backend: BackendKind,
    /// Sample rate policy.
    pub rate_mode: RateMode,
    /// Sink to connect to, by node name. `None` follows the default.
    pub device_name: Option<String>,
    /// Channels to request.
    pub channels: u32,
    /// Period size to request.
    pub quantum: Option<u32>,
    /// Start playing as soon as a format is negotiated.
    pub autoplay: bool,
    /// Measure callback timing.
    pub profile: bool,
    /// Load the saved configuration at start-up.
    pub load_config: bool,
    /// Keep generating noise when the last client disconnects, instead of
    /// fading out. Off by default: a window that has been closed should be
    /// silent. `noiseforged --keep-playing` turns it on, and a client can claim
    /// an indefinite hold with `Request::Hold`.
    pub keep_playing_without_clients: bool,
    /// Exit after this long without a client; `None` means never.
    pub idle_shutdown: Option<Duration>,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            socket: default_socket_path(),
            store: Store::discover().unwrap_or_else(|_| Store::at(".config/noiseforge")),
            backend: BackendKind::PipeWire,
            rate_mode: RateMode::Follow,
            device_name: None,
            channels: 2,
            quantum: None,
            autoplay: false,
            profile: true,
            load_config: true,
            keep_playing_without_clients: false,
            idle_shutdown: Some(Duration::from_secs(120)),
        }
    }
}

/// Daemon startup failures.
#[derive(Debug, thiserror::Error)]
pub enum ServiceError {
    #[error("ipc error: {0}")]
    Ipc(#[from] IpcError),
    #[error("backend error: {0}")]
    Backend(#[from] crate::backend::BackendError),
    #[error("preset store error: {0}")]
    Store(#[from] crate::presets::StoreError),
    #[error("device selection failed: {0}")]
    Device(String),
}

/// One connected client.
struct ClientState {
    conn: Connection,
    last_push: Instant,
}

/// The daemon.
pub struct Service {
    config: DaemonConfig,
    preset: PresetParams,
    graph: CompiledGraph,
    backend: Box<dyn AudioBackend>,
    format: OutputFormat,
    device: Option<String>,
    device_node_id: Option<u32>,
    playing: bool,
    notes: Vec<String>,
    dirty: bool,
    force_full_push: bool,
    started: Instant,
    clients: Vec<ClientState>,
    /// Rolling window of the most recent output frames, interleaved stereo.
    ///
    /// A `VecDeque` on purpose: trimming a `Vec` to a fixed length shifts a
    /// megabyte of samples every poll, which is where an earlier version burned
    /// over a percent of a core for no reason at all.
    tap: std::collections::VecDeque<f32>,
    tap_scratch: Vec<f32>,
    /// Whether the backend was last told to feed the tap.
    tap_wanted: bool,
    /// Cached curves: they only depend on the graph, not on time.
    cached_source_target: Option<Curve>,
    cached_predicted: Option<Curve>,
    cached_curve_rate: f64,
    /// A client asked for playback to survive its own disconnect.
    held: bool,
    /// Whether any client was attached on the previous poll, so the "last one
    /// left" transition can be told apart from "nobody ever attached".
    had_clients: bool,
    /// True once `--autoplay` has fired, which counts as an implicit hold.
    autoplay_armed: bool,
    last_psd: Option<noise_offline::analysis::Psd>,
    last_psd_at: Instant,
    timer_deadline: Option<Instant>,
    timer_total: Option<f64>,
    shutting_down: bool,
    design_options: DesignOptions,
}

impl Service {
    /// Build a daemon, loading the saved configuration when asked.
    pub fn new(config: DaemonConfig) -> Result<Self, ServiceError> {
        let mut preset = if config.load_config {
            match config.store.load_config() {
                Ok(Some(p)) => p,
                Ok(None) => PresetParams::default(),
                Err(e) => {
                    tracing::warn!("could not read the saved configuration ({e}); using defaults");
                    PresetParams::default()
                }
            }
        } else {
            PresetParams::default()
        };
        let mut notes = preset.sanitize();

        let backend: Box<dyn AudioBackend> = match config.backend {
            BackendKind::Null => Box::new(NullBackend::new(48_000, config.channels)),
            BackendKind::PipeWire => {
                let target = match &config.device_name {
                    Some(name) => noise_pipewire::devices::node_id_for_name(name)
                        .map_err(|e| ServiceError::Device(e.to_string()))?,
                    None => None,
                };
                Box::new(PipeWireBackend::start(
                    preset.seed,
                    config.rate_mode,
                    target,
                    config.channels,
                    config.quantum,
                    config.profile,
                )?)
            }
        };

        let format = backend.format().unwrap_or(OutputFormat {
            rate: 48_000,
            channels: config.channels,
        });
        // For an offline preview we must not wait for PipeWire; compile at the
        // format the backend already reports, or 48 kHz.
        let graph = compile_with(
            &preset,
            format.rate as f64,
            format.channels as usize,
            DesignOptions::default(),
        );
        notes.extend(graph.notes.clone());

        Ok(Self {
            config,
            preset,
            graph,
            backend,
            format,
            device: None,
            device_node_id: None,
            playing: false,
            notes,
            dirty: false,
            force_full_push: true,
            started: Instant::now(),
            clients: Vec::new(),
            tap: std::collections::VecDeque::new(),
            tap_scratch: Vec::new(),
            tap_wanted: false,
            cached_source_target: None,
            cached_predicted: None,
            cached_curve_rate: 0.0,
            held: false,
            had_clients: false,
            autoplay_armed: false,
            last_psd: None,
            last_psd_at: Instant::now() - Duration::from_secs(1),
            timer_deadline: None,
            timer_total: None,
            shutting_down: false,
            design_options: DesignOptions::default(),
        })
    }

    /// Current preset.
    pub fn preset(&self) -> &PresetParams {
        &self.preset
    }

    /// Current compiled graph.
    pub fn graph(&self) -> &CompiledGraph {
        &self.graph
    }

    /// Run until shutdown.
    ///
    /// `on_ready` is called once the socket is listening, which is what lets a
    /// test or the CLI wait for the daemon instead of sleeping blindly.
    pub fn run(mut self, on_ready: impl FnOnce(&std::path::Path)) -> Result<(), ServiceError> {
        let server = Server::bind(&self.config.socket)?;
        tracing::info!("listening on {}", server.path().display());
        on_ready(server.path());

        let mut last_activity = Instant::now();
        while !self.shutting_down {
            self.accept_clients(&server)?;
            self.poll_clients()?;
            self.poll_backend();
            if self.dirty || self.force_full_push {
                self.push_changes();
            }
            self.drain_tap();
            self.pump_subscribers();
            self.check_timer();

            if self.clients.is_empty() && !self.playing {
                if let Some(limit) = self.config.idle_shutdown
                    && last_activity.elapsed() > limit
                {
                    tracing::info!("idle for {:?}; shutting down", limit);
                    break;
                }
            } else {
                last_activity = Instant::now();
            }
            // Poll fast only when somebody is watching. A CLI command still
            // lands within tens of milliseconds, which is imperceptible, and an
            // unattended generator costs a fraction of a percent of a core.
            let watching = self.clients.iter().any(|c| c.conn.subscribed);
            let interval_ms = if watching {
                5
            } else if self.clients.is_empty() {
                40
            } else {
                12
            };
            std::thread::sleep(Duration::from_millis(interval_ms));
        }

        self.backend.shutdown();
        Ok(())
    }

    fn accept_clients(&mut self, server: &Server) -> Result<(), ServiceError> {
        while let Some(stream) = server.accept()? {
            match Connection::new(stream) {
                Ok(conn) => {
                    tracing::debug!("client connected");
                    self.clients.push(ClientState {
                        conn,
                        last_push: Instant::now(),
                    });
                }
                Err(e) => tracing::warn!("could not set up a client: {e}"),
            }
        }
        Ok(())
    }

    fn poll_clients(&mut self) -> Result<(), ServiceError> {
        let mut index = 0;
        while index < self.clients.len() {
            let request = match self.clients[index].conn.poll_request() {
                Ok(v) => v,
                Err(e) => {
                    let id = 0;
                    let _ = self.clients[index]
                        .conn
                        .send(&Response::err(id, e.to_string()));
                    None
                }
            };
            if let Some((id, request)) = request {
                let response = self.handle(index, id, request);
                if let Err(e) = self.clients[index].conn.send(&response) {
                    tracing::debug!("client send failed: {e}");
                }
            }
            if self.clients[index].conn.is_closed() || self.clients[index].conn.done {
                self.clients.swap_remove(index);
                continue;
            }
            let _ = &mut self.clients[index];
            index += 1;
        }
        self.release_when_unattended();
        Ok(())
    }

    /// Fade out when nobody is left to listen, unless somebody holds playback.
    ///
    /// This is the difference between "the daemon keeps working" and "closing
    /// the window leaves noise playing in an empty room".
    fn release_when_unattended(&mut self) {
        let attached = !self.clients.is_empty();
        let last_one_left = self.had_clients && !attached;
        self.had_clients = attached;
        if !last_one_left || !self.playing {
            return;
        }
        // A hold, `--keep-playing`, and `--autoplay` all mean "keep going".
        if self.held || self.keep_playing() {
            return;
        }
        tracing::info!("last client disconnected; fading playback out");
        self.playing = false;
        self.timer_deadline = None;
        let _ = self.backend.send(Command::Stop);
    }

    /// Whether the daemon should keep generating with no client attached.
    fn keep_playing(&self) -> bool {
        self.config.keep_playing_without_clients || self.autoplay_armed
    }

    fn poll_backend(&mut self) {
        for event in self.backend.poll() {
            match event {
                BackendEvent::Format(f) => {
                    if f != self.format {
                        tracing::info!(
                            "output format {} Hz / {} ch (was {} Hz / {} ch)",
                            f.rate,
                            f.channels,
                            self.format.rate,
                            self.format.channels
                        );
                        self.format = f;
                        self.notes.push(format!(
                            "output is {} Hz, {} channel(s)",
                            f.rate, f.channels
                        ));
                        // Coefficients must be redesigned for the new rate, and
                        // the engine is silent until they arrive.
                        self.playing = false;
                    }
                    self.dirty = true;
                    self.force_full_push = true;
                    if self.config.autoplay {
                        self.playing = true;
                        self.autoplay_armed = true;
                    }
                }
                BackendEvent::Link(state) => {
                    tracing::debug!("link: {state}");
                }
                BackendEvent::Error(e) => {
                    tracing::warn!("backend error: {e}");
                    self.notes.push(format!("audio backend: {e}"));
                }
                BackendEvent::Closed => {
                    tracing::warn!("audio backend closed");
                    self.notes.push("audio backend closed".to_string());
                }
            }
        }
    }

    /// Redesign coefficients and push only what changed.
    fn push_changes(&mut self) {
        let previous = self.graph.clone();
        self.graph = compile_with(
            &self.preset,
            self.format.rate as f64,
            self.format.channels as usize,
            self.design_options,
        );
        self.notes = self.graph.notes.clone();
        self.dirty = false;

        let full = self.force_full_push || previous.sample_rate != self.graph.sample_rate;
        self.force_full_push = false;

        let mut commands: Vec<Command> = Vec::new();
        let fade = self
            .preset
            .master
            .fade_in_ms
            .max(self.preset.master.ramp_ms) as f32;
        let ramp = self.preset.master.ramp_ms as f32;

        if full {
            commands.extend(self.graph.commands(&self.preset));
        } else {
            for (old, new) in previous.layers.iter().zip(self.graph.layers.iter()) {
                let idx = new.index as u8;
                if old.filters != new.filters {
                    commands.push(Command::SetLayerFilters {
                        layer: idx,
                        filters: new.filters,
                        crossfade_ms: fade,
                    });
                }
                if old.character != new.character {
                    commands.push(Command::SetCharacterShape {
                        layer: idx,
                        low_shelf: new.character.low_shelf,
                        high_shelf: new.character.high_shelf,
                        ramp_ms: ramp,
                    });
                    commands.push(Command::SetCharacterGain {
                        layer: idx,
                        low_gain: new.character.low_gain as f32,
                        high_gain: new.character.high_gain as f32,
                        ramp_ms: ramp,
                    });
                }
                if old.params.gain_db != new.params.gain_db {
                    commands.push(Command::SetLayerGain {
                        layer: idx,
                        gain_db: new.params.gain_db as f32,
                        ramp_ms: ramp,
                    });
                }
                if old.params.stereo_correlation != new.params.stereo_correlation {
                    commands.push(Command::SetStereoCorrelation {
                        layer: idx,
                        correlation: new.params.stereo_correlation as f32,
                        ramp_ms: ramp,
                    });
                }
                if old.seed != new.seed {
                    commands.push(Command::SetLayerSeed {
                        layer: idx,
                        seed: new.seed,
                        fade_ms: ramp,
                    });
                }
                if old.enabled != new.enabled {
                    commands.push(Command::SetLayerEnabled {
                        layer: idx,
                        enabled: new.enabled,
                        fade_ms: ramp,
                    });
                }
            }
            commands.push(Command::SetLimiter {
                enabled: self.preset.master.limiter_enabled,
                ceiling_dbfs: self.preset.master.limiter_ceiling_dbfs as f32,
            });
            commands.push(Command::SetMasterGain {
                gain_db: self.graph.calibration.master_gain_db as f32,
                ramp_ms: ramp,
                hold_ms: fade,
            });
        }

        if full {
            commands.push(Command::SetFades {
                in_ms: self.preset.master.fade_in_ms as f32,
                out_ms: self.preset.master.fade_out_ms as f32,
            });
        }

        let mut refused = 0usize;
        for command in commands {
            if self.backend.send(command).is_err() {
                refused += 1;
            }
        }
        if refused > 0 {
            self.notes.push(format!(
                "{refused} command(s) were refused because the audio thread is not draining"
            ));
        }
        if self.playing {
            let _ = self.backend.send(Command::Play);
        }
    }

    /// Feed the analysis window, but only while somebody wants a spectrum.
    ///
    /// The window is a `VecDeque` on purpose: trimming a `Vec` to a fixed length
    /// shifts a megabyte of samples on every poll, which is where an earlier
    /// version burned over a percent of a core doing nothing useful.
    fn drain_tap(&mut self) {
        let wanted = self
            .clients
            .iter()
            .any(|c| c.conn.subscribed && c.conn.want_psd);
        if wanted != self.tap_wanted {
            self.tap_wanted = wanted;
            self.backend.set_tap_wanted(wanted);
            if !wanted {
                self.tap.clear();
                self.last_psd = None;
                return;
            }
        }
        if !wanted {
            return;
        }

        self.tap_scratch.clear();
        self.backend.drain_tap(&mut self.tap_scratch, 32_768);
        if !self.tap_scratch.is_empty() {
            self.tap.extend(self.tap_scratch.iter().copied());
        }

        // Keep a window comfortably longer than the FFT so Welch has several
        // segments to average. Trimming a deque costs O(dropped), not O(kept).
        const CAP: usize = 262_144;
        while self.tap.len() > CAP {
            self.tap.pop_front();
        }

        if self.last_psd_at.elapsed() < Duration::from_millis(150) {
            return;
        }
        let cfg = WelchConfig::for_rate(self.format.rate as f64);
        if self.tap.len() >= cfg.fft_size * 2 {
            // Left channel only: the display is one curve, and the two channels
            // of this generator are statistically identical by construction.
            let mono: Vec<f32> = self.tap.iter().step_by(2).copied().collect();
            self.last_psd = Some(welch_psd(&mono, self.format.rate as f64, &cfg));
            self.last_psd_at = Instant::now();
        }
    }

    /// Build the three spectrum curves, all normalised to 0 dB at 1 kHz so they
    /// can be overlaid honestly (DESIGN.md section 5.3).
    fn curves(&mut self, want_psd: bool) -> (Option<Curve>, Option<Curve>, Option<Curve>) {
        const POINTS: usize = 160;
        const F_LO: f64 = 20.0;
        const F_HI: f64 = 20_000.0;
        let fs = self.format.rate as f64;

        let normalise = |freqs: &[f64], db: Vec<f64>| -> Curve {
            let reference = db
                .iter()
                .zip(freqs.iter())
                .min_by(|a, b| {
                    (a.1 - 1000.0)
                        .abs()
                        .partial_cmp(&(b.1 - 1000.0).abs())
                        .unwrap()
                })
                .map(|(v, _)| *v)
                .unwrap_or(0.0);
            Curve {
                f_lo: F_LO,
                f_hi: F_HI,
                db: db.into_iter().map(|v| (v - reference) as f32).collect(),
            }
        };

        // The source target and the prediction depend only on the graph, so they
        // are computed once per graph rather than once per telemetry frame.
        if self.cached_curve_rate != fs || self.cached_source_target.is_none() {
            let grid = noise_dsp::designer::log_grid(F_LO, F_HI.min(fs * 0.45), POINTS);
            self.cached_source_target = self.graph.layers.first().map(|l| {
                let db: Vec<f64> = grid
                    .iter()
                    .map(|&f| colors::amplitude_db(l.params.alpha(), f, colors::F_REF_HZ))
                    .collect();
                normalise(&grid, db)
            });
            self.cached_predicted = Some({
                let db: Vec<f64> = grid
                    .iter()
                    .map(|&f| {
                        self.graph
                            .layers
                            .iter()
                            .filter(|l| l.enabled)
                            .map(|l| l.response.response_db(f, fs))
                            .fold(f64::NEG_INFINITY, |a, b| {
                                if a == f64::NEG_INFINITY {
                                    b
                                } else {
                                    10.0 * (10f64.powf(a / 10.0) + 10f64.powf(b / 10.0)).log10()
                                }
                            })
                    })
                    .collect();
                normalise(&grid, db)
            });
            self.cached_curve_rate = fs;
        }
        let source_target = self.cached_source_target.clone();
        let predicted = self.cached_predicted.clone();

        let measured = if want_psd {
            self.last_psd.as_ref().map(|psd| {
                let (freqs, db) = log_resample(psd, F_LO, F_HI.min(fs * 0.45), POINTS);
                normalise(&freqs, db)
            })
        } else {
            None
        };

        (measured, source_target, predicted)
    }

    fn pump_subscribers(&mut self) {
        if !self.clients.iter().any(|c| c.conn.subscribed) {
            return;
        }
        // Everything the frame needs is computed *before* borrowing the client
        // list mutably, so the loop below only writes to sockets.
        let status = self.status();
        let want_psd = self
            .clients
            .iter()
            .any(|c| c.conn.subscribed && c.conn.want_psd);
        let (measured, source_target, predicted) = self.curves(want_psd);
        let frame = TelemetryFrame {
            id: 0,
            ok: true,
            status: Box::new(status),
            measured,
            source_target,
            predicted,
        };
        let encoded = serde_json::to_value(frame).unwrap_or(serde_json::Value::Null);
        let response = Response::ok(0, encoded);

        for client in self.clients.iter_mut() {
            if !client.conn.subscribed {
                continue;
            }
            let period = Duration::from_secs_f64(1.0 / client.conn.push_hz.clamp(1.0, 30.0));
            if client.last_push.elapsed() < period {
                continue;
            }
            client.last_push = Instant::now();
            if client.conn.send(&response).is_err() {
                client.conn.done = true;
            }
        }
        self.clients.retain(|c| !c.conn.done);
    }

    fn check_timer(&mut self) {
        if let Some(deadline) = self.timer_deadline
            && Instant::now() >= deadline
        {
            self.timer_deadline = None;
            self.timer_total = None;
            self.playing = false;
            let _ = self.backend.send(Command::Pause);
            tracing::info!("sleep timer expired; paused");
        }
    }

    fn status(&self) -> StatusData {
        let meters = self.backend.meters();
        let layers = self
            .graph
            .layers
            .iter()
            .map(|l| LayerStatus {
                index: l.index,
                name: l.name.clone(),
                enabled: l.enabled,
                color: l.params.color.to_string(),
                alpha: l.params.alpha(),
                character: l.params.character.to_string(),
                character_amount: l.params.character_amount,
                high_pass_hz: l.params.high_pass_hz,
                low_pass_hz: l.params.low_pass_hz,
                stereo_correlation: l.params.stereo_correlation,
                gain_db: l.params.gain_db,
                seed: l.seed,
                sections: l.filters.color.len(),
                shape_error_db: l.design.map(|d| d.max_abs_dev_db),
                slope_error_db_per_octave: l.design.map(|d| d.slope_error_db_per_octave),
                verified: l.verified(),
                predicted_rms_dbfs: self
                    .graph
                    .calibration
                    .layers
                    .get(l.index)
                    .map(|c| c.rms_dbfs)
                    .unwrap_or(-200.0),
            })
            .collect();

        let timer_remaining_s = self
            .timer_deadline
            .map(|d| d.saturating_duration_since(Instant::now()).as_secs_f64());

        StatusData {
            version: env!("CARGO_PKG_VERSION").to_string(),
            protocol: PROTOCOL_VERSION,
            dsp_version: noise_dsp::DSP_VERSION,
            backend: self.backend.kind(),
            playing: self.playing,
            transport: if self.playing { "playing" } else { "stopped" }.to_string(),
            link: meters.link.to_string(),
            sample_rate: self.format.rate,
            channels: self.format.channels,
            rate_mode: self.config.rate_mode,
            device: self.device.clone(),
            device_node_id: self.device_node_id,
            timer_remaining_s,
            uptime_s: self.started.elapsed().as_secs_f64(),
            calibration: CalibrationStatus {
                target_rms_dbfs: self.graph.calibration.target_rms_dbfs,
                mix_rms_dbfs: self.graph.calibration.mix_rms_dbfs,
                calibration_gain_db: self.graph.calibration.calibration_gain_db,
                master_gain_db: self.graph.calibration.master_gain_db,
                predicted_peak_dbfs: self.graph.calibration.predicted_peak_dbfs,
                headroom_warning: self.graph.calibration.headroom_warning,
            },
            meters: MeterStatus {
                peak_dbfs: meters.peak_dbfs,
                rms_dbfs: meters.rms_dbfs,
                xruns: meters.xruns,
                callbacks: meters.callbacks,
                frames: meters.frames,
                nan_events: meters.nan_events,
                clipped: meters.clipped,
                tap_dropped: meters.tap_dropped,
            },
            layers,
            master: self.preset.master.clone(),
            notes: self.notes.clone(),
            preset_name: self.preset.name.clone(),
            seed: self.preset.seed,
            auto_start: self.config.autoplay,
        }
    }

    fn handle(&mut self, client: usize, id: u64, request: Request) -> Response {
        match request {
            Request::Hello { client } => {
                tracing::debug!("hello from {client}");
                Response::ok(
                    id,
                    serde_json::json!({
                        "protocol": PROTOCOL_VERSION,
                        "dsp_version": noise_dsp::DSP_VERSION,
                        "version": env!("CARGO_PKG_VERSION"),
                        "backend": self.backend.kind(),
                    }),
                )
            }
            Request::Status => match serde_json::to_value(self.status()) {
                Ok(v) => Response::ok(id, v).with_notes(self.notes.clone()),
                Err(e) => Response::err(id, e.to_string()),
            },
            Request::Play => {
                self.playing = true;
                // Only an explicit `play` from a client that then disconnects
                // needs the hold; the GUI keeps its connection open and releases
                // it on close.
                if !self.clients.iter().any(|c| c.conn.subscribed) {
                    self.held = true;
                }
                let _ = self.backend.send(Command::Play);
                self.timer_deadline = None;
                Response::empty(id)
            }
            Request::Pause => {
                self.playing = false;
                self.held = false;
                let _ = self.backend.send(Command::Pause);
                self.timer_deadline = None;
                Response::empty(id)
            }
            Request::Stop => {
                self.playing = false;
                self.held = false;
                let _ = self.backend.send(Command::Stop);
                self.timer_deadline = None;
                Response::empty(id)
            }
            Request::Set { args } => {
                let notes = apply_set(&mut self.preset, &args);
                self.dirty = true;
                Response::empty(id).with_notes(notes)
            }
            Request::Load { preset } => {
                let mut preset = *preset;
                let notes = preset.sanitize();
                self.preset = preset;
                // A new preset may have a different seed, so rebuild from scratch.
                self.dirty = true;
                self.force_full_push = true;
                Response::empty(id).with_notes(notes)
            }
            Request::GetPreset => match serde_json::to_value(&self.preset) {
                Ok(v) => Response::ok(id, v),
                Err(e) => Response::err(id, e.to_string()),
            },
            Request::LayerAdd { after } => {
                if self.preset.layers.len() >= noise_dsp::MAX_LAYERS {
                    return Response::err(id, "already at the 8 layer limit");
                }
                let at = after
                    .map(|a| (a + 1).min(self.preset.layers.len()))
                    .unwrap_or(self.preset.layers.len());
                let mut layer = LayerParams {
                    name: format!("Layer {}", self.preset.layers.len() + 1),
                    ..LayerParams::default()
                };
                layer.seed = None;
                self.preset.layers.insert(at, layer);
                self.dirty = true;
                self.force_full_push = true;
                Response::empty(id)
            }
            Request::LayerRemove { index } => {
                if self.preset.layers.len() <= 1 {
                    return Response::err(id, "the last layer cannot be removed");
                }
                if index >= self.preset.layers.len() {
                    return Response::err(id, format!("layer {index} does not exist"));
                }
                self.preset.layers.remove(index);
                self.dirty = true;
                self.force_full_push = true;
                Response::empty(id)
            }
            Request::LayerDuplicate { index } => {
                if self.preset.layers.len() >= noise_dsp::MAX_LAYERS {
                    return Response::err(id, "already at the 8 layer limit");
                }
                let Some(layer) = self.preset.layers.get(index).cloned() else {
                    return Response::err(id, format!("layer {index} does not exist"));
                };
                // A duplicate must not share the source layer's seed, or the two
                // layers would be perfectly correlated.
                let mut copy = layer;
                copy.name = format!("{} copy", copy.name);
                copy.seed = None;
                self.preset.layers.insert(index + 1, copy);
                self.dirty = true;
                self.force_full_push = true;
                Response::empty(id)
            }
            Request::PresetSave { name } => {
                match self.config.store.save_preset(&name, &self.preset) {
                    Ok(path) => {
                        self.preset.name = name;
                        Response::ok(id, serde_json::json!({ "path": path }))
                    }
                    Err(e) => Response::err(id, e.to_string()),
                }
            }
            Request::PresetLoad { name } => match self.config.store.load_preset(&name) {
                Ok(mut preset) => {
                    let notes = preset.sanitize();
                    preset.name = name;
                    self.preset = preset;
                    self.dirty = true;
                    self.force_full_push = true;
                    Response::empty(id).with_notes(notes)
                }
                Err(e) => Response::err(id, e.to_string()),
            },
            Request::PresetList => match self.config.store.list_presets() {
                Ok(names) => Response::ok(id, serde_json::json!({ "presets": names })),
                Err(e) => Response::err(id, e.to_string()),
            },
            Request::PresetDelete { name } => match self.config.store.delete_preset(&name) {
                Ok(()) => Response::empty(id),
                Err(e) => Response::err(id, e.to_string()),
            },
            Request::DeviceList => match noise_pipewire::devices::list_sinks() {
                Ok(sinks) => {
                    let list: Vec<serde_json::Value> = sinks
                        .iter()
                        .map(|s| {
                            serde_json::json!({
                                "id": s.id,
                                "name": s.name,
                                "description": s.description,
                                "nick": s.nick,
                                "default": s.is_default,
                            })
                        })
                        .collect();
                    Response::ok(id, serde_json::json!({ "sinks": list }))
                }
                Err(e) => Response::err(id, format!("could not list devices: {e}")),
            },
            Request::DeviceSelect { name } => {
                let node = if name == "default" {
                    Ok(None)
                } else {
                    noise_pipewire::devices::node_id_for_name(&name).map_err(|e| e.to_string())
                };
                match node {
                    Ok(node) => {
                        self.device_node_id = node;
                        self.device = if name == "default" { None } else { Some(name) };
                        Response::empty(id)
                    }
                    Err(e) => Response::err(id, e),
                }
            }
            Request::Timer {
                seconds,
                fade_out_seconds,
            } => {
                if !seconds.is_finite() || seconds <= 0.0 {
                    self.timer_deadline = None;
                    self.timer_total = None;
                    let _ = self.backend.send(Command::ClearTimer);
                    return Response::err(id, "the timer needs a positive duration");
                }
                self.timer_deadline = Some(Instant::now() + Duration::from_secs_f64(seconds));
                self.timer_total = Some(seconds);
                let _ = self.backend.send(Command::SetTimer {
                    seconds: seconds as f32,
                    fade_out_seconds: fade_out_seconds as f32,
                });
                Response::empty(id)
            }
            Request::TimerClear => {
                self.timer_deadline = None;
                self.timer_total = None;
                let _ = self.backend.send(Command::ClearTimer);
                Response::empty(id)
            }
            Request::Retarget { node } => {
                self.device_node_id = node;
                Response::empty(id).with_notes(vec![
                    "retargeting needs the stream restarted; \
                     restart the daemon with `--device <name>`"
                        .to_string(),
                ])
            }
            Request::SaveConfig => match self.config.store.save_config(&self.preset) {
                Ok(path) => Response::ok(id, serde_json::json!({ "path": path })),
                Err(e) => Response::err(id, e.to_string()),
            },
            Request::Hold { hold } => {
                self.held = hold;
                if !hold {
                    // Releasing mid-session means "stop when I am gone", which is
                    // what the GUI wants: it drops the hold just before closing.
                    self.had_clients = !self.clients.is_empty();
                }
                Response::ok(id, serde_json::json!({ "held": self.held }))
            }
            Request::Subscribe { psd, hz } => {
                // A watching client is a live session and takes over from any
                // hold a headless `noiseforge play` left behind. Closing that
                // window should be silent unless it asks for a hold itself.
                self.held = false;
                if let Some(state) = self.clients.get_mut(client) {
                    state.conn.subscribed = true;
                    state.conn.want_psd = psd;
                    state.conn.push_hz = if hz.is_finite() {
                        hz.clamp(1.0, 30.0)
                    } else {
                        15.0
                    };
                    state.last_push = Instant::now() - Duration::from_secs(1);
                }
                Response::ok(
                    id,
                    serde_json::json!({ "subscribed": true, "hz": hz, "psd": psd }),
                )
            }
            Request::Unsubscribe => {
                if let Some(state) = self.clients.get_mut(client) {
                    state.conn.subscribed = false;
                    state.conn.want_psd = false;
                }
                Response::ok(id, serde_json::json!({ "subscribed": false }))
            }
            Request::Shutdown => {
                self.shutting_down = true;
                Response::empty(id)
            }
        }
    }
}

/// Connect, send one request, print the response. Used by the CLI.
pub fn call(socket: &std::path::Path, request: &Request) -> Result<Response, IpcError> {
    let mut client = Client::connect(socket)?;
    client.call(request)
}

/// Convenience map from layer names to indices, used by the CLI.
pub fn layer_index_by_name(preset: &PresetParams, name: &str) -> Option<usize> {
    preset
        .layers
        .iter()
        .position(|l| l.name.eq_ignore_ascii_case(name))
}

/// Colour choices with their nominal alpha, for help text.
pub fn colour_table() -> HashMap<&'static str, f64> {
    Color::ALL
        .into_iter()
        .filter_map(|c| c.nominal_alpha().map(|a| (c.as_str(), a)))
        .collect()
}
