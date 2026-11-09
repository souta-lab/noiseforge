//! `noiseforged`: the NoiseForge daemon (DESIGN.md section 4).
//!
//! Keeps generating noise when every client has disconnected, and can be started
//! by `systemd --user` (see `packaging/noiseforged.service`).

use std::path::PathBuf;
use std::time::Duration;

use clap::Parser;
use noise_daemon::presets::Store;
use noise_daemon::protocol::{BackendKind, RateMode, default_socket_path};
use noise_daemon::service::{DaemonConfig, Service};

#[derive(Parser, Debug)]
#[command(
    name = "noiseforged",
    about = "NoiseForge daemon: realtime coloured noise over PipeWire",
    version
)]
struct Args {
    /// Unix socket to listen on.
    #[arg(long, env = "NOISEFORGE_SOCKET")]
    socket: Option<PathBuf>,

    /// Audio backend.
    #[arg(long, value_enum, default_value_t = BackendArg::Pipewire)]
    backend: BackendArg,

    /// Follow the server sample rate, or pin one.
    #[arg(long, value_name = "HZ", conflicts_with = "rate_mode")]
    rate: Option<u32>,

    /// Explicit sample rate policy.
    #[arg(long, value_enum)]
    rate_mode: Option<RateModeArg>,

    /// Sink node name to connect to. Omit to follow the default sink.
    #[arg(long, value_name = "NAME")]
    device: Option<String>,

    /// Channels to request.
    #[arg(long, default_value_t = 2)]
    channels: u32,

    /// Requested period size in frames.
    #[arg(long)]
    quantum: Option<u32>,

    /// Start playing as soon as the format is negotiated.
    #[arg(long)]
    autoplay: bool,

    /// Do not load the saved configuration.
    #[arg(long)]
    no_config: bool,

    /// Directory for presets and the configuration file.
    #[arg(long, value_name = "DIR", env = "NOISEFORGE_CONFIG_DIR")]
    config_dir: Option<PathBuf>,

    /// Shut down after this many seconds with no client and no playback.
    #[arg(long, value_name = "SECONDS", default_value_t = 120.0)]
    idle_shutdown: f64,

    /// Keep generating noise when the last client disconnects.
    ///
    /// Off by default: closing the window should be silent. Turn this on for a
    /// machine that should hum along on its own, or when started from systemd.
    #[arg(long)]
    keep_playing: bool,

    /// Never shut down when idle.
    #[arg(long, conflicts_with = "idle_shutdown")]
    forever: bool,

    /// Do not measure per-callback timing.
    #[arg(long)]
    no_profile: bool,

    /// Increase log verbosity (-v, -vv).
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, clap::ValueEnum)]
enum BackendArg {
    Pipewire,
    Null,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, clap::ValueEnum)]
enum RateModeArg {
    Follow,
    Pin,
}

fn main() -> std::process::ExitCode {
    let args = Args::parse();
    init_tracing(args.verbose);

    let backend = match args.backend {
        BackendArg::Pipewire => BackendKind::PipeWire,
        BackendArg::Null => BackendKind::Null,
    };
    let rate_mode = match (args.rate, args.rate_mode) {
        (Some(r), _) => RateMode::Pin(r),
        (None, Some(RateModeArg::Pin)) => RateMode::Pin(48_000),
        (None, _) => RateMode::Follow,
    };
    let store = match &args.config_dir {
        Some(dir) => Store::at(dir),
        None => match Store::discover() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("noiseforged: {e}");
                return std::process::ExitCode::FAILURE;
            }
        },
    };

    let config = DaemonConfig {
        socket: args.socket.unwrap_or_else(default_socket_path),
        store,
        backend,
        rate_mode,
        device_name: args.device,
        channels: args.channels,
        quantum: args.quantum,
        autoplay: args.autoplay,
        profile: !args.no_profile,
        load_config: !args.no_config,
        keep_playing_without_clients: args.keep_playing,
        idle_shutdown: if args.forever {
            None
        } else {
            Some(Duration::from_secs_f64(args.idle_shutdown.max(0.0)))
        },
    };

    let service = match Service::new(config) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("noiseforged: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };

    match service.run(|path| {
        println!("noiseforged listening on {}", path.display());
    }) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("noiseforged: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

fn init_tracing(verbose: u8) {
    let level = match verbose {
        0 => "warn",
        1 => "info",
        _ => "debug",
    };
    // A tiny dependency-free logger keeps the daemon's dependency surface small.
    let filter = std::env::var("NOISEFORGE_LOG").unwrap_or_else(|_| level.to_string());
    tracing::subscriber::set_global_default(TinySubscriber { level: filter })
        .expect("the global tracing subscriber is set exactly once");
}

/// Minimal stderr logger: `LEVEL target: message`.
struct TinySubscriber {
    level: String,
}

impl tracing::Subscriber for TinySubscriber {
    fn enabled(&self, _metadata: &tracing::Metadata<'_>) -> bool {
        true
    }

    fn new_span(&self, _span: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }

    fn record(&self, _span: &tracing::span::Id, _values: &tracing::span::Record<'_>) {}

    fn record_follows_from(&self, _span: &tracing::span::Id, _follows: &tracing::span::Id) {}

    fn event(&self, event: &tracing::Event<'_>) {
        let level = *event.metadata().level();
        let wanted = match self.level.as_str() {
            "error" => tracing::Level::ERROR,
            "warn" => tracing::Level::WARN,
            "info" => tracing::Level::INFO,
            "debug" => tracing::Level::DEBUG,
            _ => tracing::Level::TRACE,
        };
        if level > wanted {
            return;
        }
        let mut message = String::new();
        event.record(&mut Visitor(&mut message));
        eprintln!("{level} {}: {message}", event.metadata().target());
    }

    fn enter(&self, _span: &tracing::span::Id) {}

    fn exit(&self, _span: &tracing::span::Id) {}
}

struct Visitor<'a>(&'a mut String);

impl tracing::field::Visit for Visitor<'_> {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        use std::fmt::Write;
        if field.name() == "message" {
            let _ = write!(self.0, "{value:?}");
        } else {
            let _ = write!(self.0, " {}={value:?}", field.name());
        }
    }
}
