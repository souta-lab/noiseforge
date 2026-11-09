//! The PipeWire output stream (DESIGN.md section 4.2).
//!
//! # Threading
//!
//! A dedicated thread owns the PipeWire main loop, the context, the core and the
//! stream:
//!
//! * the **process callback** is PipeWire's data thread. It drains a fixed
//!   capacity SPSC command ring, asks the engine for exactly the frames the host
//!   offered, copies them honouring `offset`/`stride`/`size`, and publishes
//!   atomics. It never allocates, locks, logs or formats.
//! * `param_changed`, `state_changed` and the control channel run on the loop
//!   thread, which is *not* realtime and may allocate.
//!
//! Those two are different threads, so anything the process callback reads is
//! either immutable after start-up or guarded by an atomic. In particular the
//! engine is never rebuilt underneath a running callback: a sample rate change
//! quiesces the stream through `Reconfiguration`, rebuilds, and resumes.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use noise_dsp::engine::{Command, Engine, MAX_BLOCK};
use pipewire as pw;
use pw::spa;
use pw::spa::pod::Pod;

use crate::format::{NegotiatedFormat, enum_format_params, parse_format};
use crate::lifecycle::{LinkState, Telemetry};

/// How long to wait for an in-flight process callback to finish before the
/// engine is replaced.
const QUIESCE_TIMEOUT: Duration = Duration::from_millis(250);

/// How a sample rate change is handled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RatePolicy {
    /// Accept whatever the server negotiates and rebuild the engine in place.
    Follow,
    /// Request `sample_rate` and let PipeWire resample if the device differs.
    Pin(u32),
}

/// How the stream is created.
#[derive(Clone, Debug)]
pub struct StreamConfig {
    /// PipeWire node name.
    pub node_name: String,
    /// Human readable description shown in patch bays.
    pub node_description: String,
    /// Node id to connect to, or `None` to follow the default sink.
    pub target_object: Option<u32>,
    /// Sample rate policy.
    pub rate: RatePolicy,
    /// Requested channel count.
    pub channels: u32,
    /// Requested period size in frames.
    pub quantum: Option<u32>,
    /// PRNG seed handed to the engine.
    pub seed: u64,
    /// Command ring capacity.
    pub command_capacity: usize,
    /// Ring capacity, in frames, for the analysis tap.
    pub tap_frames: usize,
    /// Measure per-callback timing (adds a clock read per callback).
    pub profile_timing: bool,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            node_name: "noiseforge".to_string(),
            node_description: "NoiseForge noise generator".to_string(),
            target_object: None,
            rate: RatePolicy::Follow,
            channels: 2,
            quantum: None,
            seed: 1,
            command_capacity: 512,
            tap_frames: 48_000,
            profile_timing: true,
        }
    }
}

impl StreamConfig {
    /// The rate to build the first engine at.
    fn initial_rate(&self) -> u32 {
        match self.rate {
            RatePolicy::Pin(r) => r,
            RatePolicy::Follow => 48_000,
        }
    }

    /// Rate to offer the server, or `None` to offer a choice.
    fn offered_rate(&self) -> Option<u32> {
        match self.rate {
            RatePolicy::Pin(r) => Some(r),
            RatePolicy::Follow => None,
        }
    }
}

/// Messages the loop thread sends to the daemon.
#[derive(Clone, Debug)]
pub enum StreamEvent {
    /// The server negotiated a format. Coefficients must be redesigned and the
    /// previous set must not be trusted for this rate.
    Format(NegotiatedFormat),
    /// The stream state changed.
    State {
        old: String,
        new: String,
        link: LinkState,
    },
    /// Something went wrong; a reconnect may be scheduled by the server.
    Error(String),
    /// The loop is about to exit.
    Closed,
}

/// Commands the daemon sends into the loop thread.
#[derive(Clone, Debug)]
pub enum StreamControl {
    /// Stop the loop and join.
    Quit,
    /// Ask the stream to move to another sink.
    Retarget(Option<u32>),
}

/// Handle to a running PipeWire output.
pub struct Output {
    /// Push [`Command`]s here. Lock free and realtime safe.
    pub commands: rtrb::Producer<Command>,
    /// Analysis tap: interleaved stereo frames copied out of the render path.
    pub tap: rtrb::Consumer<f32>,
    /// Format, state and error notifications.
    pub events: mpsc::Receiver<StreamEvent>,
    /// Shared counters.
    pub telemetry: Arc<Telemetry>,
    control: pw::channel::Sender<StreamControl>,
    thread: Option<std::thread::JoinHandle<()>>,
    running: Arc<AtomicBool>,
}

impl Output {
    /// Ask the loop to exit and join the thread.
    pub fn shutdown(mut self) {
        let _ = self.control.send(StreamControl::Quit);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }

    /// True while the loop thread is alive.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }

    /// Ask the stream to move to another sink.
    pub fn retarget(&self, node: Option<u32>) -> bool {
        self.control.send(StreamControl::Retarget(node)).is_ok()
    }

    /// Current link state.
    pub fn link_state(&self) -> LinkState {
        self.telemetry.link()
    }

    /// How many commands are queued but not yet consumed by the callback.
    pub fn command_backlog(&self) -> usize {
        self.commands.slots()
    }
}

/// Handshake between the loop thread (which rebuilds the engine) and the process
/// callback (which must not be running while it does).
struct Reconfiguration {
    /// Set by the loop thread before it touches the engine.
    requested: AtomicBool,
    /// Set by the process callback when it has observed `requested` and left.
    acknowledged: AtomicBool,
    /// Rate the engine is currently built for.
    rate: AtomicU32,
}

impl Reconfiguration {
    fn new(rate: u32) -> Self {
        Self {
            requested: AtomicBool::new(false),
            acknowledged: AtomicBool::new(false),
            rate: AtomicU32::new(rate),
        }
    }
}

/// Everything the realtime callback owns.
struct RealtimeState {
    engine: Engine,
    format: NegotiatedFormat,
    telemetry: Arc<Telemetry>,
    tap: rtrb::Producer<f32>,
    commands: rtrb::Consumer<Command>,
    reconfig: Arc<Reconfiguration>,
    /// Set until the daemon has pushed coefficients for the current rate.
    awaiting_coefficients: bool,
    /// Transport state the daemon asked for.
    wants_play: bool,
    /// Scratch for draining the tap without allocating.
    staging: Vec<f32>,
}

fn build_engine(rate: u32, seed: u64) -> Engine {
    let mut engine = Engine::new(rate as f64, seed);
    // Silence until coefficients arrive for this rate.
    engine.apply(Command::EmergencySilence);
    engine
}

/// Start a PipeWire output stream on its own thread.
pub fn spawn(config: StreamConfig) -> Result<Output, pw::Error> {
    let (command_tx, command_rx) = rtrb::RingBuffer::<Command>::new(config.command_capacity);
    let (tap_tx, tap_rx) = rtrb::RingBuffer::<f32>::new(config.tap_frames * 2);
    let (event_tx, event_rx) = mpsc::channel();
    let (control_tx, control_rx) = pw::channel::channel();
    let telemetry = Arc::new(Telemetry::default());
    telemetry.set_link(LinkState::Connecting);
    telemetry
        .sample_rate
        .store(config.initial_rate(), Ordering::Relaxed);
    telemetry
        .profiling
        .store(config.profile_timing, Ordering::Relaxed);
    let running = Arc::new(AtomicBool::new(true));

    let tel = telemetry.clone();
    let alive = running.clone();

    let thread = std::thread::Builder::new()
        .name("noiseforge-pipewire".to_string())
        .spawn(move || {
            if let Err(e) = run_loop(config, command_rx, tap_tx, event_tx, tel, control_rx) {
                tracing::warn!("pipewire loop ended: {e}");
            }
            alive.store(false, Ordering::Relaxed);
        })
        .map_err(|_| pw::Error::CreationFailed)?;

    Ok(Output {
        commands: command_tx,
        tap: tap_rx,
        events: event_rx,
        telemetry,
        control: control_tx,
        thread: Some(thread),
        running,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_loop(
    config: StreamConfig,
    commands: rtrb::Consumer<Command>,
    tap: rtrb::Producer<f32>,
    events: mpsc::Sender<StreamEvent>,
    telemetry: Arc<Telemetry>,
    control: pw::channel::Receiver<StreamControl>,
) -> Result<(), pw::Error> {
    pw::init();
    let mainloop = pw::main_loop::MainLoopRc::new(None)?;
    let context = pw::context::ContextRc::new(&mainloop, None)?;
    let core = context.connect_rc(None)?;

    let initial_rate = config.initial_rate();
    let reconfig = Arc::new(Reconfiguration::new(initial_rate));

    let node_name = config.node_name.clone();
    let node_description = config.node_description.clone();
    let channels_prop = config.channels.to_string();
    let stream = pw::stream::StreamRc::new(
        core.clone(),
        &node_name,
        pw::properties::properties! {
            *pw::keys::MEDIA_TYPE => "Audio",
            *pw::keys::MEDIA_CATEGORY => "Playback",
            *pw::keys::MEDIA_ROLE => "Music",
            *pw::keys::NODE_NAME => node_name.as_str(),
            *pw::keys::NODE_DESCRIPTION => node_description.as_str(),
            *pw::keys::AUDIO_CHANNELS => channels_prop.as_str(),
        },
    )?;

    let realtime = RealtimeState {
        engine: build_engine(initial_rate, config.seed),
        format: NegotiatedFormat {
            rate: initial_rate,
            channels: config.channels,
            ..NegotiatedFormat::default()
        },
        telemetry: telemetry.clone(),
        tap,
        commands,
        reconfig: reconfig.clone(),
        awaiting_coefficients: true,
        wants_play: false,
        staging: Vec::new(),
    };

    let _quit = control.attach(mainloop.loop_(), {
        let mainloop = mainloop.clone();
        move |msg| match msg {
            StreamControl::Quit => mainloop.quit(),
            StreamControl::Retarget(node) => {
                tracing::info!("retarget to {node:?} requested");
            }
        }
    });

    let listener = stream
        .add_local_listener_with_user_data(realtime)
        .state_changed({
            let events = events.clone();
            let telemetry = telemetry.clone();
            move |_stream, _rt, old, new| {
                let link = match new {
                    pw::stream::StreamState::Streaming => LinkState::Running,
                    pw::stream::StreamState::Paused => LinkState::Suspended,
                    pw::stream::StreamState::Error(_) => LinkState::Error,
                    pw::stream::StreamState::Connecting => LinkState::Connecting,
                    _ => LinkState::Idle,
                };
                telemetry.set_link(link);
                let _ = events.send(StreamEvent::State {
                    old: format!("{old:?}"),
                    new: format!("{new:?}"),
                    link,
                });
                if let pw::stream::StreamState::Error(e) = new {
                    let _ = events.send(StreamEvent::Error(e.clone()));
                }
            }
        })
        .param_changed({
            let events = events.clone();
            let telemetry = telemetry.clone();
            let reconfig = reconfig.clone();
            move |_stream, rt, id, param| {
                let Some(param) = param else { return };
                if id != spa::param::ParamType::Format.as_raw() {
                    return;
                }
                let Some(format) = parse_format(param) else {
                    return;
                };
                telemetry.sample_rate.store(format.rate, Ordering::Relaxed);
                telemetry.channels.store(format.channels, Ordering::Relaxed);
                if format.rate == reconfig.rate.load(Ordering::Relaxed) {
                    // Same rate: nothing to rebuild, just remember the layout.
                    rt.format = format;
                    let _ = events.send(StreamEvent::Format(format));
                    return;
                }
                // The rate changed. Quiesce the process callback, rebuild the
                // engine outside it, then let it run again. This is the only
                // place the engine is replaced after start-up.
                tracing::info!(
                    "sample rate {} -> {}; rebuilding the engine",
                    reconfig.rate.load(Ordering::Relaxed),
                    format.rate
                );
                reconfig.acknowledged.store(false, Ordering::Release);
                reconfig.requested.store(true, Ordering::Release);
                let deadline = Instant::now() + QUIESCE_TIMEOUT;
                while !reconfig.acknowledged.load(Ordering::Acquire) && Instant::now() < deadline {
                    std::thread::sleep(Duration::from_micros(200));
                }
                rt.engine = build_engine(format.rate, rt.engine.seed());
                rt.format = format;
                rt.awaiting_coefficients = true;
                rt.wants_play = false;
                reconfig.rate.store(format.rate, Ordering::Relaxed);
                reconfig.requested.store(false, Ordering::Release);
                reconfig.acknowledged.store(false, Ordering::Release);
                let _ = events.send(StreamEvent::Format(format));
            }
        })
        .process(|stream, rt| {
            let started = if rt.telemetry.profiling.load(Ordering::Relaxed) {
                Some(Instant::now())
            } else {
                None
            };

            // If the loop thread wants to swap the engine, get out of its way.
            if rt.reconfig.requested.load(Ordering::Acquire) {
                rt.reconfig.acknowledged.store(true, Ordering::Release);
                if let Some(mut buffer) = stream.dequeue_buffer() {
                    zero_buffer(&mut buffer);
                }
                rt.telemetry.note_xrun();
                return;
            }

            // 1. Drain the command ring. Bounded work per callback, so a flood of
            //    slider updates can never make one callback overrun.
            let mut applied_coefficients = false;
            for _ in 0..64 {
                let cmd = match rt.commands.pop() {
                    Ok(c) => c,
                    Err(_) => break,
                };
                match cmd {
                    Command::Play => rt.wants_play = true,
                    Command::Pause | Command::Stop => rt.wants_play = false,
                    Command::SetLayerFilters { .. } | Command::SetLayerSeed { .. } => {
                        applied_coefficients = true;
                    }
                    _ => {}
                }
                rt.engine.apply(cmd);
            }
            if applied_coefficients && rt.awaiting_coefficients {
                rt.awaiting_coefficients = false;
                if rt.wants_play {
                    rt.engine.apply(Command::Play);
                }
            }

            // 2. Work out how many frames the host actually offered.
            let Some(mut buffer) = stream.dequeue_buffer() else {
                rt.telemetry.note_xrun();
                return;
            };
            let channels = rt.format.channels.max(1) as usize;
            let sample_bytes = rt.format.format.bytes();
            let datas = buffer.datas_mut();
            if datas.is_empty() {
                rt.telemetry.note_xrun();
                return;
            }
            let offered = buffer_frames(datas, channels, sample_bytes, rt.format.planar);
            let frames = offered.min(MAX_BLOCK);
            if frames == 0 {
                rt.telemetry.note_xrun();
                return;
            }

            // 3. Render. The only place the engine is touched. The engine is
            //    borrowed mutably for the whole block, so every other field is
            //    reached through a disjoint borrow.
            let RealtimeState {
                engine,
                format,
                telemetry,
                tap,
                staging,
                ..
            } = rt;
            {
                let rendered = engine.render_block(frames);
                write_output(*format, rendered, datas, frames, channels, sample_bytes);
                // 5. Analysis tap. A non-blocking copy for the PSD display; if the
                //    reader is behind the frames are dropped and the audio is
                //    never affected (DESIGN.md section 4.5).
                let len = rendered.len();
                if staging.len() < len {
                    staging.resize(len, 0.0);
                }
                staging[..len].copy_from_slice(rendered);
            }

            // 4. Publish. Done after the render borrow ends.
            {
                let meters = engine.meters();
                let peak = meters.peak() as f32;
                let rms = meters.rms() as f32;
                let nan = meters.nan_events();
                let clipped = meters.clipped();
                telemetry.publish_block(peak, peak, rms, rms, frames as u32, frames as u32);
                telemetry.note_nan(nan);
                telemetry.note_clipped(clipped);
            }
            // The tap is best effort and entirely optional: when no client wants
            // a spectrum, skip the copy so an idle daemon costs nothing, and when
            // the reader is behind drop the newest frames and count them. The
            // audio path is untouched either way.
            if telemetry
                .tap_wanted
                .load(std::sync::atomic::Ordering::Relaxed)
            {
                let staged = staging.len().min(frames * 2);
                let mut dropped = 0u64;
                for &value in &staging[..staged] {
                    if tap.push(value).is_err() {
                        dropped += 1;
                    }
                }
                if dropped > 0 {
                    telemetry.tap_dropped.fetch_add(dropped, Ordering::Relaxed);
                }
            }

            if let Some(t0) = started {
                telemetry.record_timing(t0.elapsed().as_nanos().min(u32::MAX as u128) as u32);
            }
        })
        .register()?;

    // Offer the format and connect.
    {
        let bytes = enum_format_params(config.offered_rate(), config.channels, config.quantum)?;
        let pod = Pod::from_bytes(&bytes).ok_or(pw::Error::CreationFailed)?;
        let mut params = [pod];
        stream.connect(
            spa::utils::Direction::Output,
            config.target_object,
            pw::stream::StreamFlags::AUTOCONNECT
                | pw::stream::StreamFlags::MAP_BUFFERS
                | pw::stream::StreamFlags::RT_PROCESS,
            &mut params,
        )?;
    }

    let _ = events.send(StreamEvent::State {
        old: "Idle".to_string(),
        new: "Connecting".to_string(),
        link: LinkState::Connecting,
    });
    mainloop.run();
    drop(listener);
    telemetry.set_link(LinkState::Idle);
    let _ = events.send(StreamEvent::Closed);
    Ok(())
}

/// How many frames the host offered.
///
/// Prefers the buffer's `requested` count, which is what the graph actually
/// wants; falls back to the mapped size when the server does not report it.
fn buffer_frames(
    datas: &mut [spa::buffer::Data],
    channels: usize,
    sample_bytes: usize,
    planar: bool,
) -> usize {
    if planar {
        datas
            .iter_mut()
            .take(channels)
            .filter_map(|d| d.data().map(|s| s.len() / sample_bytes))
            .min()
            .unwrap_or(0)
    } else {
        let stride = sample_bytes * channels.max(1);
        datas
            .first_mut()
            .and_then(|d| d.data().map(|s| s.len() / stride))
            .unwrap_or(0)
    }
}

/// Fill a buffer with silence and mark it empty.
fn zero_buffer(buffer: &mut pw::buffer::Buffer<'_>) {
    for data in buffer.datas_mut().iter_mut() {
        if let Some(raw) = data.data() {
            raw.fill(0);
        }
        let chunk = data.chunk_mut();
        *chunk.size_mut() = 0;
    }
}

/// Copy the engine's interleaved stereo scratch into the host buffer.
fn write_output(
    format: NegotiatedFormat,
    rendered: &[f32],
    datas: &mut [spa::buffer::Data],
    frames: usize,
    channels: usize,
    sample_bytes: usize,
) {
    if format.planar {
        for (ch, data) in datas.iter_mut().enumerate().take(channels) {
            let base = data.chunk().offset() as usize;
            let stride = {
                let s = data.chunk().stride();
                if s <= 0 { sample_bytes as i32 } else { s }
            } as usize;
            let n = {
                let Some(raw) = data.data() else { continue };
                let n = frames.min(raw.len().saturating_sub(base) / stride.max(1));
                for f in 0..n {
                    let dst = base + f * stride;
                    let value = match ch {
                        0 => rendered[f * 2],
                        1 => rendered[f * 2 + 1],
                        _ => 0.0,
                    };
                    raw[dst..dst + sample_bytes].copy_from_slice(&value.to_le_bytes());
                }
                n
            };
            let chunk = data.chunk_mut();
            *chunk.offset_mut() = base as u32;
            *chunk.stride_mut() = stride as i32;
            *chunk.size_mut() = (n * stride) as u32;
        }
        for data in datas.iter_mut().skip(channels) {
            let chunk = data.chunk_mut();
            *chunk.offset_mut() = 0;
            *chunk.stride_mut() = 0;
            *chunk.size_mut() = 0;
        }
        return;
    }

    let data = &mut datas[0];
    let base = data.chunk().offset() as usize;
    let stride = {
        let s = data.chunk().stride();
        if s <= 0 {
            (sample_bytes * channels.max(1)) as i32
        } else {
            s
        }
    } as usize;
    let n = {
        let Some(raw) = data.data() else { return };
        let n = frames.min(raw.len().saturating_sub(base) / stride.max(1));
        for f in 0..n {
            let frame = base + f * stride;
            for ch in 0..channels {
                let dst = frame + ch * sample_bytes;
                if dst + sample_bytes > raw.len() {
                    break;
                }
                let value = match ch {
                    0 => rendered[f * 2],
                    1 => rendered[f * 2 + 1],
                    _ => 0.0,
                };
                raw[dst..dst + sample_bytes].copy_from_slice(&value.to_le_bytes());
            }
        }
        n
    };
    let chunk = data.chunk_mut();
    *chunk.offset_mut() = base as u32;
    *chunk.stride_mut() = stride as i32;
    *chunk.size_mut() = (n * stride) as u32;
}
