//! Audio backend abstraction.
//!
//! The daemon's control logic must not depend on PipeWire being present: tests,
//! CI and `--backend null` all need to run the same control loop. This module
//! defines the narrow interface the control thread uses, with a PipeWire
//! implementation and a silent one.

use std::sync::Arc;

use noise_dsp::engine::Command;

use crate::protocol::{BackendKind, RateMode};

/// A format the backend negotiated and the engine must be rebuilt for.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OutputFormat {
    pub rate: u32,
    pub channels: u32,
}

/// Events the backend reports back to the control loop.
#[derive(Clone, Debug)]
pub enum BackendEvent {
    /// The output format is now `format`; coefficients must be redesigned.
    Format(OutputFormat),
    /// The link state changed.
    Link(String),
    /// Something went wrong.
    Error(String),
    /// The backend stopped.
    Closed,
}

/// A snapshot of the realtime counters.
#[derive(Clone, Copy, Debug, Default)]
pub struct Meters {
    pub peak_dbfs: f64,
    pub rms_dbfs: f64,
    pub xruns: u64,
    pub callbacks: u64,
    pub frames: u64,
    pub nan_events: u64,
    pub clipped: u64,
    pub tap_dropped: u64,
    pub sample_rate: u32,
    pub channels: u32,
    pub link: &'static str,
    /// p99.99 callback duration in nanoseconds, when profiling is on.
    pub callback_p9999_ns: Option<u32>,
    pub callback_max_ns: u32,
}

/// Backend failures.
#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    #[error("the command queue is full; the audio thread is not draining")]
    QueueFull,
    #[error("the backend is no longer running")]
    Closed,
    #[error("{0}")]
    Other(String),
}

/// What the control thread needs from an audio backend.
pub trait AudioBackend: Send {
    /// Which backend this is.
    fn kind(&self) -> BackendKind;
    /// Push a POD command to the realtime thread. Never blocks.
    fn send(&mut self, command: Command) -> Result<(), BackendError>;
    /// Drain pending events.
    fn poll(&mut self) -> Vec<BackendEvent>;
    /// Read the counters.
    fn meters(&self) -> Meters;
    /// Current negotiated format, if any.
    fn format(&self) -> Option<OutputFormat>;
    /// Copy up to `max` interleaved stereo frames out of the analysis tap.
    fn drain_tap(&mut self, out: &mut Vec<f32>, max: usize);
    /// Tell the backend whether anyone is watching the tap, so it can skip the
    /// copy entirely when nobody is.
    fn set_tap_wanted(&mut self, wanted: bool);
    /// Number of commands still queued.
    fn backlog(&self) -> usize;
    /// Stop and release everything.
    fn shutdown(self: Box<Self>);
}

/// A backend with no audio output. Commands are accepted and dropped.
///
/// Used by `--backend null` and by the daemon's own tests, so the control loop,
/// IPC, presets and calibration can all be exercised without touching the sound
/// server.
pub struct NullBackend {
    format: OutputFormat,
    events: Vec<BackendEvent>,
    sent: u64,
    closed: bool,
}

impl NullBackend {
    /// Build a silent backend reporting `format`.
    pub fn new(rate: u32, channels: u32) -> Self {
        Self {
            format: OutputFormat { rate, channels },
            events: vec![BackendEvent::Format(OutputFormat { rate, channels })],
            sent: 0,
            closed: false,
        }
    }
}

impl AudioBackend for NullBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Null
    }

    fn send(&mut self, _command: Command) -> Result<(), BackendError> {
        if self.closed {
            return Err(BackendError::Closed);
        }
        self.sent += 1;
        Ok(())
    }

    fn poll(&mut self) -> Vec<BackendEvent> {
        std::mem::take(&mut self.events)
    }

    fn meters(&self) -> Meters {
        Meters {
            link: if self.closed { "closed" } else { "null" },
            sample_rate: self.format.rate,
            channels: self.format.channels,
            // Silence is -200 dBFS, not 0 dBFS. Reporting full scale for a
            // backend that produces nothing would be a lie the UI believed.
            peak_dbfs: -200.0,
            rms_dbfs: -200.0,
            ..Meters::default()
        }
    }

    fn format(&self) -> Option<OutputFormat> {
        Some(self.format)
    }

    fn drain_tap(&mut self, _out: &mut Vec<f32>, _max: usize) {}

    fn set_tap_wanted(&mut self, _wanted: bool) {}

    fn backlog(&self) -> usize {
        0
    }

    fn shutdown(self: Box<Self>) {}
}

/// The PipeWire backend.
pub struct PipeWireBackend {
    output: noise_pipewire::Output,
    format: Option<OutputFormat>,
    tap_buffer: Vec<f32>,
}

impl PipeWireBackend {
    /// Start a PipeWire stream.
    pub fn start(
        seed: u64,
        rate_mode: RateMode,
        target_object: Option<u32>,
        channels: u32,
        quantum: Option<u32>,
        profile: bool,
    ) -> Result<Self, BackendError> {
        let config = noise_pipewire::StreamConfig {
            rate: match rate_mode {
                RateMode::Follow => noise_pipewire::RatePolicy::Follow,
                RateMode::Pin(r) => noise_pipewire::RatePolicy::Pin(r),
            },
            target_object,
            channels,
            quantum,
            seed,
            profile_timing: profile,
            ..noise_pipewire::StreamConfig::default()
        };
        let output = noise_pipewire::spawn(config)
            .map_err(|e| BackendError::Other(format!("pipewire: {e}")))?;
        Ok(Self {
            output,
            format: None,
            tap_buffer: Vec::new(),
        })
    }

    /// Shared telemetry, for diagnostics.
    pub fn telemetry(&self) -> Arc<noise_pipewire::Telemetry> {
        self.output.telemetry.clone()
    }
}

impl AudioBackend for PipeWireBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::PipeWire
    }

    fn send(&mut self, command: Command) -> Result<(), BackendError> {
        self.output
            .commands
            .push(command)
            .map_err(|_| BackendError::QueueFull)
    }

    fn poll(&mut self) -> Vec<BackendEvent> {
        let mut out = Vec::new();
        loop {
            match self.output.events.try_recv() {
                Ok(event) => match event {
                    noise_pipewire::StreamEvent::Format(f) => {
                        let fmt = OutputFormat {
                            rate: f.rate,
                            channels: f.channels,
                        };
                        self.format = Some(fmt);
                        out.push(BackendEvent::Format(fmt));
                    }
                    noise_pipewire::StreamEvent::State { new, link, .. } => {
                        out.push(BackendEvent::Link(format!("{} ({})", link.label(), new)));
                    }
                    noise_pipewire::StreamEvent::Error(e) => out.push(BackendEvent::Error(e)),
                    noise_pipewire::StreamEvent::Closed => out.push(BackendEvent::Closed),
                },
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    out.push(BackendEvent::Closed);
                    break;
                }
            }
        }
        out
    }

    fn meters(&self) -> Meters {
        use std::sync::atomic::Ordering;
        let t = &self.output.telemetry;
        let q = |a: &std::sync::atomic::AtomicU64| a.load(Ordering::Relaxed);
        Meters {
            peak_dbfs: noise_pipewire::lifecycle::dbfs(t.peak_l()) as f64,
            rms_dbfs: noise_pipewire::lifecycle::dbfs(t.rms_l()) as f64,
            xruns: q(&t.xruns),
            callbacks: q(&t.callbacks),
            frames: q(&t.frames),
            nan_events: q(&t.nan_events),
            clipped: q(&t.clipped),
            tap_dropped: q(&t.tap_dropped),
            sample_rate: t.sample_rate.load(Ordering::Relaxed),
            channels: t.channels.load(Ordering::Relaxed),
            link: match t.link() {
                noise_pipewire::LinkState::Idle => "idle",
                noise_pipewire::LinkState::Connecting => "connecting",
                noise_pipewire::LinkState::Running => "running",
                noise_pipewire::LinkState::Suspended => "suspended",
                noise_pipewire::LinkState::Error => "error",
            },
            callback_p9999_ns: t.callback_high_percentile_ns(0.9999),
            callback_max_ns: t.callback_max_ns.load(Ordering::Relaxed),
        }
    }

    fn format(&self) -> Option<OutputFormat> {
        self.format
    }

    fn drain_tap(&mut self, out: &mut Vec<f32>, max: usize) {
        self.tap_buffer.clear();
        while self.tap_buffer.len() < max {
            match self.output.tap.pop() {
                Ok(v) => self.tap_buffer.push(v),
                Err(_) => break,
            }
        }
        out.extend_from_slice(&self.tap_buffer);
    }

    fn set_tap_wanted(&mut self, wanted: bool) {
        use std::sync::atomic::Ordering;
        self.output
            .telemetry
            .tap_wanted
            .store(wanted, Ordering::Relaxed);
    }

    fn backlog(&self) -> usize {
        self.output.command_backlog()
    }

    fn shutdown(self: Box<Self>) {
        self.output.shutdown();
    }
}
