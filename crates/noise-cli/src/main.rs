//! `noiseforge`: the command line client (DESIGN.md section 6.1).
//!
//! Two kinds of command:
//!
//! * **daemon clients** (`play`, `pause`, `set`, `color`, `preset`, `device`,
//!   `status`, `timer`, `layer`, `monitor`) talk to `noiseforged` over its Unix
//!   socket and never touch the audio path themselves.
//! * **offline commands** (`render`) link the DSP directly, because DESIGN.md
//!   section 10 wants a headless renderer that needs neither a daemon nor a
//!   sound server.

use std::path::PathBuf;
use std::process::Command as ProcessCommand;
use std::time::{Duration, Instant};

use clap::{Parser, Subcommand, ValueEnum};
use noise_daemon::ipc::{Client, IpcError};
use noise_daemon::protocol::{
    Curve, Request, Response, SetArgs, StatusData, TelemetryFrame, default_socket_path,
};
use noise_dsp::params::{LayerParams, PresetParams};
use noise_offline::wav::OutputFormat;

#[derive(Parser, Debug)]
#[command(
    name = "noiseforge",
    about = "Generate mathematically defined noise through PipeWire",
    version,
    after_help = "Colours: white pink brown blue violet custom\n\
                  Characters: neutral smooth intense\n\
                  Examples:\n  \
                  noiseforge play pink --character smooth --intensity 0.7 --level-rms-dbfs -36\n  \
                  noiseforge color custom --alpha 1.35\n  \
                  noiseforge set --high-pass 40 --low-pass 12000\n  \
                  noiseforge preset save \"Soft Pink\"\n  \
                  noiseforge timer --minutes 45 --fade-out 60\n  \
                  noiseforge status --json\n  \
                  noiseforge render --seconds 60 --rate 48000 --seed 42 --wav sample.wav"
)]
struct Cli {
    /// Unix socket of the daemon.
    #[arg(long, global = true, env = "NOISEFORGE_SOCKET")]
    socket: Option<PathBuf>,

    /// Print machine readable output where the command supports it.
    #[arg(long, global = true)]
    json: bool,

    /// Do not start a daemon automatically when none is running.
    #[arg(long, global = true)]
    no_autostart: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Start or resume playback, optionally changing the sound first.
    Play {
        /// Colour to switch to before playing.
        color: Option<String>,
        /// Custom exponent, used when the colour is `custom`.
        #[arg(long, allow_negative_numbers = true)]
        alpha: Option<f64>,
        /// Sound character.
        #[arg(long)]
        character: Option<String>,
        /// Character amount, 0.0 to 1.0.
        #[arg(long)]
        intensity: Option<f64>,
        /// Expected output RMS in dBFS.
        #[arg(long = "level-rms-dbfs", allow_negative_numbers = true)]
        level_rms_dbfs: Option<f64>,
        /// Layer to modify.
        #[arg(long, default_value_t = 0)]
        layer: usize,
    },
    /// Fade out and hold.
    Pause,
    /// Fade out and stop.
    Stop,
    /// Change settings without touching the transport.
    Set {
        #[arg(long)]
        color: Option<String>,
        #[arg(long, allow_negative_numbers = true)]
        alpha: Option<f64>,
        #[arg(long)]
        character: Option<String>,
        #[arg(long)]
        intensity: Option<f64>,
        #[arg(long)]
        high_pass: Option<f64>,
        #[arg(long)]
        low_pass: Option<f64>,
        /// Stereo correlation, -1.0 to 1.0.
        #[arg(long, allow_negative_numbers = true)]
        stereo: Option<f64>,
        #[arg(long, allow_negative_numbers = true)]
        gain_db: Option<f64>,
        #[arg(long = "level-rms-dbfs", allow_negative_numbers = true)]
        level_rms_dbfs: Option<f64>,
        #[arg(long, allow_negative_numbers = true)]
        volume_db: Option<f64>,
        #[arg(long)]
        mute: Option<bool>,
        #[arg(long)]
        seed: Option<u64>,
        #[arg(long)]
        layer: Option<usize>,
        /// Enable the explicit limiter (nonlinear; changes the spectrum).
        #[arg(long)]
        limiter: Option<bool>,
        #[arg(long = "limiter-ceiling", allow_negative_numbers = true)]
        limiter_ceiling_dbfs: Option<f64>,
    },
    /// Shorthand for `set --color`.
    Color {
        /// Colour name.
        color: String,
        /// Custom exponent for `custom`.
        #[arg(long, allow_negative_numbers = true)]
        alpha: Option<f64>,
        #[arg(long, default_value_t = 0)]
        layer: usize,
    },
    /// Shorthand for `set --character --intensity`.
    Character {
        /// Character name.
        character: String,
        /// Amount, 0.0 to 1.0.
        #[arg(long)]
        intensity: Option<f64>,
        #[arg(long, default_value_t = 0)]
        layer: usize,
    },
    /// Mixer: list, add, remove, rename and enable layers.
    Layer {
        #[command(subcommand)]
        action: LayerAction,
    },
    /// Preset files.
    Preset {
        #[command(subcommand)]
        action: PresetAction,
    },
    /// Output devices.
    Device {
        #[command(subcommand)]
        action: DeviceAction,
    },
    /// Sleep timer.
    Timer {
        /// Minutes until the fade-out completes.
        #[arg(long, allow_negative_numbers = true)]
        minutes: Option<f64>,
        /// Seconds until the fade-out completes.
        #[arg(long, allow_negative_numbers = true)]
        seconds: Option<f64>,
        /// Fade-out length in seconds, occupying the end of the duration.
        #[arg(long, default_value_t = 0.0)]
        fade_out: f64,
        /// Cancel the timer.
        #[arg(long)]
        clear: bool,
    },
    /// Print the daemon state.
    Status,
    /// Follow the meters until interrupted.
    Monitor {
        /// Print the measured spectrum too.
        #[arg(long)]
        spectrum: bool,
        /// Seconds to run for; 0 means until interrupted.
        #[arg(long, default_value_t = 0.0)]
        seconds: f64,
    },
    /// Save the current state as the start-up configuration.
    SaveConfig,
    /// Run the daemon in the foreground.
    Daemon {
        /// Audio backend.
        #[arg(long, value_enum, default_value_t = BackendArg::Pipewire)]
        backend: BackendArg,
        /// Pin the sample rate.
        #[arg(long)]
        rate: Option<u32>,
        /// Sink node name.
        #[arg(long)]
        device: Option<String>,
        /// Start playing immediately.
        #[arg(long)]
        autoplay: bool,
    },
    /// Stop the daemon.
    Shutdown,
    /// Render a preset offline to a WAV file.
    Render {
        /// Colour.
        #[arg(default_value = "pink")]
        color: String,
        /// Custom exponent.
        #[arg(long, allow_negative_numbers = true)]
        alpha: Option<f64>,
        /// Sound character.
        #[arg(long, default_value = "neutral")]
        character: String,
        /// Character amount.
        #[arg(long, default_value_t = 0.0)]
        intensity: f64,
        /// Length in seconds.
        #[arg(long, default_value_t = 60.0)]
        seconds: f64,
        /// Sample rate.
        #[arg(long, default_value_t = 48_000)]
        rate: u32,
        /// PRNG seed.
        #[arg(long)]
        seed: Option<u64>,
        /// Expected RMS in dBFS.
        #[arg(long = "level-rms-dbfs", default_value_t = -36.0, allow_negative_numbers = true)]
        level_rms_dbfs: f64,
        /// Second order high pass corner.
        #[arg(long, default_value_t = 35.0)]
        high_pass: f64,
        /// Second order low pass corner.
        #[arg(long, default_value_t = 16_000.0)]
        low_pass: f64,
        /// Output file. `-` writes nothing and only reports measurements.
        #[arg(long)]
        wav: Option<PathBuf>,
        /// Sample format.
        #[arg(long, value_enum, default_value_t = FormatArg::F32)]
        format: FormatArg,
        /// Audio discarded before the written region, in seconds.
        #[arg(long, default_value_t = 0.5)]
        warmup: f64,
        /// Measure the rendered spectrum and compare it with the theory.
        #[arg(long)]
        analyze: bool,
        /// Load this preset instead of building one from the flags.
        #[arg(long)]
        preset: Option<String>,
    },
}

#[derive(Subcommand, Debug)]
enum LayerAction {
    /// List the layers.
    List,
    /// Add a layer.
    Add {
        /// Insert after this index; defaults to the end.
        #[arg(long)]
        after: Option<usize>,
    },
    /// Remove a layer.
    Remove { index: usize },
    /// Copy a layer.
    Duplicate { index: usize },
    /// Enable or disable a layer.
    Enable {
        index: usize,
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        on: bool,
    },
    /// Rename a layer.
    Rename { index: usize, name: String },
}

#[derive(Subcommand, Debug)]
enum PresetAction {
    /// Save the current state.
    Save { name: String },
    /// Load a preset.
    Load { name: String },
    /// List presets.
    List,
    /// Delete a preset.
    Delete { name: String },
    /// Print the current state as TOML.
    Show,
}

#[derive(Subcommand, Debug)]
enum DeviceAction {
    /// List output devices.
    List,
    /// Select a device by node name, or `default`.
    Select { name: String },
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum BackendArg {
    Pipewire,
    Null,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum FormatArg {
    F32,
    I16,
    I24,
}

fn main() -> std::process::ExitCode {
    let cli = Cli::parse();
    match run(cli) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("noiseforge: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

type CliResult = Result<(), Box<dyn std::error::Error>>;

fn run(cli: Cli) -> CliResult {
    let socket = cli.socket.clone().unwrap_or_else(default_socket_path);
    match cli.command {
        Commands::Render {
            color,
            alpha,
            character,
            intensity,
            seconds,
            rate,
            seed,
            level_rms_dbfs,
            high_pass,
            low_pass,
            wav,
            format,
            warmup,
            analyze,
            preset,
        } => do_render(RenderArgs {
            color,
            alpha,
            character,
            intensity,
            seconds,
            rate,
            seed,
            level_rms_dbfs,
            high_pass,
            low_pass,
            wav,
            format: format.into(),
            warmup,
            analyze,
            preset,
            json: cli.json,
        }),
        Commands::Daemon {
            backend,
            rate,
            device,
            autoplay,
        } => exec_daemon(socket, backend, rate, device, autoplay),
        other => {
            let mut client = connect(&socket, cli.no_autostart)?;
            dispatch(&mut client, other, cli.json)
        }
    }
}

impl From<FormatArg> for OutputFormat {
    fn from(v: FormatArg) -> Self {
        match v {
            FormatArg::F32 => OutputFormat::F32,
            FormatArg::I16 => OutputFormat::I16,
            FormatArg::I24 => OutputFormat::I24,
        }
    }
}

/// Attach to the daemon, starting one if necessary.
fn connect(
    socket: &std::path::Path,
    no_autostart: bool,
) -> Result<Client, Box<dyn std::error::Error>> {
    match Client::connect(socket) {
        Ok(c) => return Ok(c),
        Err(IpcError::NotListening(_)) if !no_autostart => {
            eprintln!(
                "noiseforge: no daemon on {}; starting noiseforged",
                socket.display()
            );
            start_daemon(socket)?;
        }
        Err(e) => return Err(Box::new(e)),
    }
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match Client::connect(socket) {
            Ok(c) => return Ok(c),
            Err(e) => {
                if Instant::now() >= deadline {
                    return Err(Box::new(e));
                }
                std::thread::sleep(Duration::from_millis(50));
            }
        }
    }
}

/// Locate the daemon binary next to this executable, or on `PATH`.
fn daemon_binary() -> PathBuf {
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
    {
        let candidate = dir.join("noiseforged");
        if candidate.exists() {
            return candidate;
        }
    }
    PathBuf::from("noiseforged")
}

fn start_daemon(socket: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = ProcessCommand::new(daemon_binary());
    cmd.arg("--socket")
        .arg(socket)
        // An autostarted daemon is a convenience, not a service: it gets out of
        // the way once nothing is playing and nobody is attached.
        .arg("--idle-shutdown")
        .arg("90")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
    cmd.spawn()?;
    Ok(())
}

/// Unwrap a response, returning its error as a boxed error.
fn ok(response: Response) -> Result<Response, Box<dyn std::error::Error>> {
    if response.ok {
        Ok(response)
    } else {
        Err(response
            .error
            .unwrap_or_else(|| "the daemon rejected the request".into())
            .into())
    }
}

fn exec_daemon(
    socket: PathBuf,
    backend: BackendArg,
    rate: Option<u32>,
    device: Option<String>,
    autoplay: bool,
) -> CliResult {
    let mut cmd = ProcessCommand::new(daemon_binary());
    cmd.arg("--socket").arg(&socket);
    if matches!(backend, BackendArg::Null) {
        cmd.arg("--backend").arg("null");
    }
    if let Some(r) = rate {
        cmd.arg("--rate").arg(r.to_string());
    }
    if let Some(d) = device {
        cmd.arg("--device").arg(d);
    }
    if autoplay {
        cmd.arg("--autoplay");
    }
    let status = cmd.status()?;
    std::process::exit(status.code().unwrap_or(1));
}

fn dispatch(client: &mut Client, command: Commands, json: bool) -> CliResult {
    match command {
        Commands::Play {
            color,
            alpha,
            character,
            intensity,
            level_rms_dbfs,
            layer,
        } => {
            let mut args = SetArgs {
                color,
                alpha,
                character,
                intensity,
                level_rms_dbfs,
                layer: Some(layer),
                ..Default::default()
            };
            if args.color.is_none()
                && args.alpha.is_none()
                && args.character.is_none()
                && args.intensity.is_none()
                && args.level_rms_dbfs.is_none()
            {
                args.layer = None;
            }
            report(client.call(&Request::Set { args })?, json)?;
            // The daemon fades out when the last client leaves unless somebody
            // holds it; `play` is exactly the request to keep going.
            ok(client.call(&Request::Hold { hold: true })?)?;
            report(client.call(&Request::Play)?, json)?;
            if !json {
                eprintln!(
                    "noiseforge: playing in the background; `noiseforge stop` stops it"
                );
            }
            Ok(())
        }
        Commands::Pause => {
            ok(client.call(&Request::Hold { hold: false })?)?;
            report(client.call(&Request::Pause)?, json)
        }
        Commands::Stop => {
            ok(client.call(&Request::Hold { hold: false })?)?;
            report(client.call(&Request::Stop)?, json)
        }
        Commands::Set {
            color,
            alpha,
            character,
            intensity,
            high_pass,
            low_pass,
            stereo,
            gain_db,
            level_rms_dbfs,
            volume_db,
            mute,
            seed,
            layer,
            limiter,
            limiter_ceiling_dbfs,
        } => {
            let args = SetArgs {
                color,
                alpha,
                character,
                intensity,
                high_pass,
                low_pass,
                stereo,
                gain_db,
                level_rms_dbfs,
                volume_db,
                mute,
                seed,
                layer,
                limiter,
                limiter_ceiling_dbfs,
                ..Default::default()
            };
            report(client.call(&Request::Set { args })?, json)
        }
        Commands::Color {
            color,
            alpha,
            layer,
        } => {
            let args = SetArgs {
                color: Some(color),
                alpha,
                layer: Some(layer),
                ..Default::default()
            };
            report(client.call(&Request::Set { args })?, json)
        }
        Commands::Character {
            character,
            intensity,
            layer,
        } => {
            let args = SetArgs {
                character: Some(character),
                intensity,
                layer: Some(layer),
                ..Default::default()
            };
            report(client.call(&Request::Set { args })?, json)
        }
        Commands::Layer { action } => dispatch_layer(client, action, json),
        Commands::Preset { action } => dispatch_preset(client, action, json),
        Commands::Device { action } => dispatch_device(client, action, json),
        Commands::Timer {
            minutes,
            seconds,
            fade_out,
            clear,
        } => {
            if clear {
                return report(client.call(&Request::TimerClear)?, json);
            }
            let total = match (minutes, seconds) {
                (Some(m), _) => m * 60.0,
                (None, Some(s)) => s,
                (None, None) => return Err("give --minutes or --seconds".into()),
            };
            report(
                client.call(&Request::Timer {
                    seconds: total,
                    fade_out_seconds: fade_out,
                })?,
                json,
            )
        }
        Commands::Status => {
            let response = client.call(&Request::Status)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&response.data)?);
            } else {
                print_status(&response)?;
            }
            Ok(())
        }
        Commands::Monitor { spectrum, seconds } => monitor(client, spectrum, seconds, json),
        Commands::SaveConfig => report(client.call(&Request::SaveConfig)?, json),
        Commands::Shutdown => report(client.call(&Request::Shutdown)?, json),
        Commands::Daemon { .. } | Commands::Render { .. } => unreachable!("handled earlier"),
    }
}

fn fetch_preset(client: &mut Client) -> Result<PresetParams, Box<dyn std::error::Error>> {
    let response = client.call(&Request::GetPreset)?;
    Ok(serde_json::from_value(
        response.data.ok_or("the daemon returned no preset")?,
    )?)
}

fn dispatch_layer(client: &mut Client, action: LayerAction, json: bool) -> CliResult {
    match action {
        LayerAction::List => {
            let response = client.call(&Request::Status)?;
            let data: StatusData =
                serde_json::from_value(response.data.ok_or("the daemon returned no status")?)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&data.layers)?);
            } else {
                println!(
                    "{:>3} {:<18} {:<8} {:>6} {:>5} {:<16} {:>8} {:>9}",
                    "#", "name", "colour", "alpha", "sos", "character", "gain dB", "verified"
                );
                for l in &data.layers {
                    println!(
                        "{:>3} {:<18} {:<8} {:>6.2} {:>5} {:<16} {:>8.1} {:>9}",
                        l.index,
                        truncate(&l.name, 18),
                        if l.enabled { l.color.as_str() } else { "(off)" },
                        l.alpha,
                        l.sections,
                        format!("{} {:.2}", l.character, l.character_amount),
                        l.gain_db,
                        if l.verified { "yes" } else { "APPROX" }
                    );
                }
            }
            Ok(())
        }
        LayerAction::Add { after } => report(client.call(&Request::LayerAdd { after })?, json),
        LayerAction::Remove { index } => {
            report(client.call(&Request::LayerRemove { index })?, json)
        }
        LayerAction::Duplicate { index } => {
            report(client.call(&Request::LayerDuplicate { index })?, json)
        }
        LayerAction::Enable { index, on } => {
            let mut preset = fetch_preset(client)?;
            let Some(layer) = preset.layers.get_mut(index) else {
                return Err(format!("layer {index} does not exist").into());
            };
            layer.enabled = on;
            report(
                client.call(&Request::Load {
                    preset: Box::new(preset),
                })?,
                json,
            )
        }
        LayerAction::Rename { index, name } => {
            let mut preset = fetch_preset(client)?;
            let Some(layer) = preset.layers.get_mut(index) else {
                return Err(format!("layer {index} does not exist").into());
            };
            layer.name = name;
            report(
                client.call(&Request::Load {
                    preset: Box::new(preset),
                })?,
                json,
            )
        }
    }
}

fn dispatch_preset(client: &mut Client, action: PresetAction, json: bool) -> CliResult {
    match action {
        PresetAction::Save { name } => report(client.call(&Request::PresetSave { name })?, json),
        PresetAction::Load { name } => report(client.call(&Request::PresetLoad { name })?, json),
        PresetAction::Delete { name } => {
            report(client.call(&Request::PresetDelete { name })?, json)
        }
        PresetAction::List => {
            let response = client.call(&Request::PresetList)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&response.data)?);
            } else {
                let names = response
                    .data
                    .as_ref()
                    .and_then(|d| d.get("presets"))
                    .and_then(|v| v.as_array())
                    .cloned()
                    .unwrap_or_default();
                if names.is_empty() {
                    println!("no presets saved yet");
                }
                for n in names {
                    if let Some(s) = n.as_str() {
                        println!("{s}");
                    }
                }
            }
            Ok(())
        }
        PresetAction::Show => {
            let preset = fetch_preset(client)?;
            println!("{}", toml::to_string_pretty(&preset)?);
            Ok(())
        }
    }
}

fn dispatch_device(client: &mut Client, action: DeviceAction, json: bool) -> CliResult {
    match action {
        DeviceAction::List => {
            let response = client.call(&Request::DeviceList)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&response.data)?);
            } else {
                let sinks = response
                    .data
                    .as_ref()
                    .and_then(|d| d.get("sinks"))
                    .and_then(|v| v.as_array())
                    .cloned()
                    .unwrap_or_default();
                if sinks.is_empty() {
                    println!("no output devices found");
                }
                for s in sinks {
                    let default = s.get("default").and_then(|v| v.as_bool()).unwrap_or(false);
                    let name = s.get("name").and_then(|v| v.as_str()).unwrap_or("");
                    let desc = s
                        .get("description")
                        .and_then(|v| v.as_str())
                        .unwrap_or(name);
                    let id = s.get("id").and_then(|v| v.as_u64()).unwrap_or(0);
                    println!("{} {:>4}  {}", if default { "*" } else { " " }, id, desc);
                    println!("         name: {name}");
                }
            }
            Ok(())
        }
        DeviceAction::Select { name } => {
            report(client.call(&Request::DeviceSelect { name })?, json)
        }
    }
}

fn monitor(client: &mut Client, spectrum: bool, seconds: f64, json: bool) -> CliResult {
    let response = client.call(&Request::Subscribe {
        psd: spectrum,
        hz: 8.0,
    })?;
    if !response.ok {
        return Err(response
            .error
            .unwrap_or_else(|| "subscribe failed".into())
            .into());
    }
    let deadline = if seconds > 0.0 {
        Some(Instant::now() + Duration::from_secs_f64(seconds))
    } else {
        None
    };
    println!(
        "{:>9} {:>9} {:>9} {:>7} {:>10}",
        "peak dBFS", "rms dBFS", "target", "xruns", "link"
    );
    loop {
        if let Some(d) = deadline
            && Instant::now() >= d
        {
            break;
        }
        match client.next_push(Duration::from_millis(500)) {
            Ok(Some(response)) => {
                let Some(data) = response.data else { continue };
                if json {
                    println!("{}", serde_json::to_string(&data)?);
                    continue;
                }
                let frame: TelemetryFrame = match serde_json::from_value(data) {
                    Ok(f) => f,
                    Err(_) => continue,
                };
                let s = &frame.status;
                println!(
                    "{:>9.2} {:>9.2} {:>9.2} {:>7} {:>10}",
                    s.meters.peak_dbfs,
                    s.meters.rms_dbfs,
                    s.calibration.target_rms_dbfs,
                    s.meters.xruns,
                    s.link
                );
                if let Some(curve) = &frame.measured {
                    print_curve("measured", curve);
                }
            }
            Ok(None) => continue,
            Err(e) => return Err(Box::new(e)),
        }
    }
    let _ = client.call(&Request::Unsubscribe)?;
    Ok(())
}

fn print_curve(label: &str, curve: &Curve) {
    let n = curve.db.len();
    if n == 0 {
        return;
    }
    println!("  {label} (dB relative to 1 kHz):");
    for row in 0..8 {
        let mut line = String::new();
        for col in 0..64 {
            let i = (row * 64 + col) * n / 512;
            let v = curve.db[i.min(n - 1)];
            line.push(match v {
                v if v > 12.0 => '#',
                v if v > 6.0 => '*',
                v if v > 0.0 => '+',
                v if v > -6.0 => '-',
                v if v > -12.0 => '.',
                _ => ' ',
            });
        }
        println!("  |{line}|");
    }
}

fn report(response: Response, json: bool) -> CliResult {
    if json {
        println!("{}", serde_json::to_string_pretty(&response)?);
    } else {
        for note in &response.notes {
            eprintln!("note: {note}");
        }
        if !response.ok {
            return Err(response
                .error
                .unwrap_or_else(|| "the daemon rejected the request".into())
                .into());
        }
        if let Some(data) = &response.data {
            let text = serde_json::to_string_pretty(data)?;
            if text != "null" && text != "{}" {
                println!("{text}");
            }
        }
    }
    if response.ok {
        Ok(())
    } else {
        Err(response
            .error
            .unwrap_or_else(|| "the daemon rejected the request".into())
            .into())
    }
}

fn print_status(response: &Response) -> CliResult {
    for note in &response.notes {
        eprintln!("note: {note}");
    }
    let data: StatusData = serde_json::from_value(
        response
            .data
            .clone()
            .ok_or("the daemon returned no status")?,
    )?;
    println!(
        "NoiseForge {} (protocol {}, DSP {})",
        data.version, data.protocol, data.dsp_version
    );
    println!(
        "  backend {}   link {}   transport {}",
        match data.backend {
            noise_daemon::protocol::BackendKind::PipeWire => "pipewire",
            noise_daemon::protocol::BackendKind::Null => "null",
        },
        data.link,
        data.transport
    );
    println!(
        "  output {} Hz, {} channel(s)   rate policy {:?}",
        data.sample_rate, data.channels, data.rate_mode
    );
    if let Some(d) = &data.device {
        println!("  device {d}");
    }
    println!(
        "  target {:.1} dBFS RMS   master gain {:.2} dB   predicted peak {:.1} dBFS{}",
        data.calibration.target_rms_dbfs,
        data.calibration.master_gain_db,
        data.calibration.predicted_peak_dbfs,
        if data.calibration.headroom_warning {
            "   [HEADROOM WARNING]"
        } else {
            ""
        }
    );
    println!(
        "  measured peak {:.2} dBFS   rms {:.2} dBFS   xruns {}   nan {}",
        data.meters.peak_dbfs, data.meters.rms_dbfs, data.meters.xruns, data.meters.nan_events
    );
    if let Some(t) = data.timer_remaining_s {
        let m = (t / 60.0).floor();
        let s = t - m * 60.0;
        println!("  timer {m:.0}m {s:04.1}s remaining");
    }
    println!("  preset \"{}\"   seed {}", data.preset_name, data.seed);
    println!("  layers:");
    for l in &data.layers {
        let accuracy = match (l.shape_error_db, l.slope_error_db_per_octave) {
            (Some(shape), Some(slope)) => {
                format!("  shape {shape:+.3} dB, slope {slope:+.3} dB/oct")
            }
            _ => String::new(),
        };
        println!(
            "   [{:>2}] {:<18} {:<7} a={:<5.2} {:<8} {:<5.2} gain {:<6.1} dB  {} sos{}{}",
            l.index,
            truncate(&l.name, 18),
            if l.enabled { l.color.as_str() } else { "(off)" },
            l.alpha,
            l.character,
            l.character_amount,
            l.gain_db,
            l.sections,
            if l.verified {
                ""
            } else {
                "  [OUT OF TOLERANCE]"
            },
            accuracy
        );
    }
    for note in &data.notes {
        println!("  note: {note}");
    }
    Ok(())
}

fn truncate(s: &str, width: usize) -> String {
    if s.chars().count() <= width {
        return s.to_string();
    }
    let mut out: String = s.chars().take(width.saturating_sub(1)).collect();
    out.push('…');
    out
}

struct RenderArgs {
    color: String,
    alpha: Option<f64>,
    character: String,
    intensity: f64,
    seconds: f64,
    rate: u32,
    seed: Option<u64>,
    level_rms_dbfs: f64,
    high_pass: f64,
    low_pass: f64,
    wav: Option<PathBuf>,
    format: OutputFormat,
    warmup: f64,
    analyze: bool,
    preset: Option<String>,
    json: bool,
}

fn do_render(args: RenderArgs) -> CliResult {
    let params = match &args.preset {
        Some(name) => noise_daemon::presets::Store::discover()?.load_preset(name)?,
        None => {
            let color = noise_dsp::Color::parse(&args.color)
                .ok_or_else(|| format!("unknown colour '{}'", args.color))?;
            let character = noise_dsp::Character::parse(&args.character)
                .ok_or_else(|| format!("unknown character '{}'", args.character))?;
            let layer = LayerParams {
                color,
                alpha: args.alpha.unwrap_or_else(|| color.resolve_alpha(1.0)),
                character,
                character_amount: args.intensity,
                high_pass_hz: args.high_pass,
                low_pass_hz: args.low_pass,
                ..LayerParams::default()
            };
            PresetParams {
                name: format!("{color}/{character}"),
                seed: args.seed.unwrap_or(1),
                layers: vec![layer],
                ..PresetParams::default()
            }
        }
    };
    let mut params = params;
    if let Some(seed) = args.seed {
        params.seed = seed;
    }
    params.master.target_rms_dbfs = args.level_rms_dbfs;

    let request = noise_offline::render::RenderRequest {
        params,
        sample_rate: args.rate as f64,
        seconds: args.seconds,
        warmup_seconds: args.warmup,
        channels: 2,
        seed: args.seed,
        auto_play: true,
    };
    let started = Instant::now();
    let result = noise_offline::render::render(&request);
    let elapsed = started.elapsed();

    let rms = result.rms();
    let rms_dbfs = if rms > 0.0 {
        20.0 * rms.log10()
    } else {
        -200.0
    };
    let peak_dbfs = if result.peak() > 0.0 {
        20.0 * (result.peak() as f64).log10()
    } else {
        -200.0
    };

    let mut summary = serde_json::json!({
        "frames": result.frames,
        "channels": result.channels,
        "sample_rate": result.sample_rate,
        "seconds": result.frames as f64 / result.sample_rate,
        "rms_dbfs": rms_dbfs,
        "peak_dbfs": peak_dbfs,
        "target_rms_dbfs": result.graph.calibration.target_rms_dbfs,
        "calibration_gain_db": result.graph.calibration.master_gain_db,
        "render_ms": elapsed.as_secs_f64() * 1000.0,
        "realtime_factor": (result.frames as f64 / result.sample_rate) / elapsed.as_secs_f64(),
        "layers": result.graph.layers.iter().map(|l| serde_json::json!({
            "index": l.index,
            "color": l.params.color.to_string(),
            "alpha": l.params.alpha(),
            "sections": l.filters.color.len(),
            "verified": l.verified(),
            "shape_error_db": l.design.map(|d| d.max_abs_dev_db),
            "slope_error_db_per_octave": l.design.map(|d| d.slope_error_db_per_octave),
        })).collect::<Vec<_>>(),
        "notes": result.graph.notes,
    });

    if args.analyze {
        let cfg = noise_offline::analysis::WelchConfig::for_rate(result.sample_rate);
        let mono = result.channel(0);
        let (lo, hi) = noise_dsp::designer::validation_band(result.sample_rate);
        if let Some((_, _, fit)) =
            noise_offline::analysis::measure_slope(&mono, result.sample_rate, lo, hi, &cfg, 6)
        {
            let alpha = result
                .graph
                .layers
                .first()
                .map(|l| l.params.alpha())
                .unwrap_or(1.0);
            let theoretical = noise_dsp::colors::psd_slope_db_per_octave(alpha);
            summary["measured_slope_db_per_octave"] = serde_json::json!(fit.slope_db_per_octave);
            summary["theoretical_slope_db_per_octave"] = serde_json::json!(theoretical);
            summary["slope_error_db_per_octave"] =
                serde_json::json!(fit.slope_db_per_octave - theoretical);
            summary["max_deviation_db"] = serde_json::json!(fit.max_deviation_db);
            summary["band"] = serde_json::json!([lo, hi]);
        }
    }

    if let Some(path) = &args.wav
        && path.as_os_str() != "-"
    {
        let clipped =
            noise_offline::wav::write_wav(path, &result.samples, 2, args.rate, args.format)?;
        summary["written"] = serde_json::json!(path);
        summary["clipped_samples"] = serde_json::json!(clipped);
        if clipped > 0 {
            eprintln!("warning: {clipped} sample(s) were clamped while writing integers");
        }
    }

    if args.json {
        println!("{}", serde_json::to_string_pretty(&summary)?);
    } else {
        println!(
            "rendered {:.2} s at {} Hz ({} frames) in {:.0} ms ({:.0}x realtime)",
            result.frames as f64 / result.sample_rate,
            args.rate,
            result.frames,
            elapsed.as_secs_f64() * 1000.0,
            (result.frames as f64 / result.sample_rate) / elapsed.as_secs_f64()
        );
        println!(
            "  measured {:.2} dBFS RMS, {:.2} dBFS peak (target {:.1} dBFS RMS)",
            rms_dbfs, peak_dbfs, result.graph.calibration.target_rms_dbfs
        );
        for l in &result.graph.layers {
            match l.design {
                Some(d) => println!(
                    "  layer {} {} alpha {:+.2}: {} sos, shape {:+.3} dB, slope {:+.3} dB/oct over {:.0}-{:.0} Hz{}",
                    l.index,
                    l.params.color,
                    l.params.alpha(),
                    l.filters.color.len(),
                    d.max_abs_dev_db,
                    d.slope_error_db_per_octave,
                    d.band_lo_hz,
                    d.band_hi_hz,
                    if d.within_tolerance {
                        ""
                    } else {
                        "  [OUT OF TOLERANCE]"
                    }
                ),
                None => println!(
                    "  layer {} {}: no filtering needed",
                    l.index, l.params.color
                ),
            }
        }
        for note in &result.graph.notes {
            println!("  note: {note}");
        }
        if let Some(slope) = summary.get("measured_slope_db_per_octave") {
            println!(
                "  measured slope {slope} dB/oct vs theoretical {} dB/oct",
                summary["theoretical_slope_db_per_octave"]
            );
        }
    }
    Ok(())
}
