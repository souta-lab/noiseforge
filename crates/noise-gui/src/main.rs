//! `noise-gui`: the NoiseForge control window (DESIGN.md sections 5 and 6).
//!
//! The window is a *client*: it owns no audio, no coefficients and no state that
//! matters, and it shows whatever the daemon is doing when it starts.
//!
//! ```text
//! noise-gui [--socket <PATH>] [--no-autostart] [--x11] [--renderer <KIND>] [--info]
//! ```
//!
//! `NOISEFORGE_SOCKET` is honoured when `--socket` is absent, matching the CLI.
//!
//! # Window system
//!
//! The window is a native Wayland client whenever `WAYLAND_DISPLAY` is set,
//! because that is what a Wayland session wants and what keeps a tiling
//! compositor tiling it. Forcing the backend goes through winit's platform
//! extension traits rather than `WINIT_UNIX_BACKEND`, which needs
//! `std::env::set_var` - `unsafe` since edition 2024, and this crate forbids
//! unsafe code. The environment variable is still honoured when the caller sets
//! it, and `--x11` overrides everything.
//!
//! # Rendering
//!
//! `--renderer auto` (the default) tries wgpu (Vulkan on this platform) and
//! falls back to OpenGL when the GPU stack refuses, so the window still opens
//! on machines where one of the two is broken.

#![forbid(unsafe_code)]

mod advanced;
mod app;
mod layers;
mod spectrum;

use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::process::Command as ProcessCommand;
use std::time::{Duration, Instant};

use app::{App, GuiOptions};
use noise_daemon::protocol::default_socket_path;
use winit::platform::wayland::EventLoopBuilderExtWayland;
use winit::platform::x11::EventLoopBuilderExtX11;

/// Help text, printed for `--help` and after a usage error.
const USAGE: &str = "\
noise-gui: the NoiseForge control window

Usage: noise-gui [OPTIONS]

Options:
      --socket <PATH>     Unix socket of the daemon
                          [env: NOISEFORGE_SOCKET, default: $XDG_RUNTIME_DIR/noiseforge/control.sock]
      --no-autostart      Do not start noiseforged when nothing is listening
      --x11               Use the X11 backend instead of native Wayland
      --renderer <KIND>   Graphics backend: auto, wgpu or glow [default: auto]
                          OpenGL through EGL is the default and is hardware
                          accelerated on any Mesa or proprietary driver. wgpu
                          (Vulkan) only exists in a build made with
                          `--features noise-gui/wgpu`, because its dependency
                          tree is large; auto prefers it when it is compiled in
      --info              Print the windowing backend, the renderer and the GPU
                          adapter, then exit
  -h, --help              Print this help

The window reconnects by itself whenever the daemon restarts, and it stops
playback when it closes unless \"Keep playing in the background\" is ticked.";

/// Which windowing backend to ask winit for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Windowing {
    /// Native Wayland.
    Wayland,
    /// X11, which on a Wayland session means XWayland (and a floating window on
    /// some compositors).
    X11,
    /// Whatever winit picks on its own.
    Auto,
}

impl Windowing {
    /// Name used in messages.
    const fn label(self) -> &'static str {
        match self {
            Windowing::Wayland => "wayland",
            Windowing::X11 => "x11",
            Windowing::Auto => "auto",
        }
    }
}

/// Which graphics backend to ask eframe for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RendererChoice {
    /// wgpu first, OpenGL as the fallback.
    Auto,
    /// wgpu (Vulkan) only.
    Wgpu,
    /// OpenGL only.
    Glow,
}

/// Parsed command line.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Options {
    /// Socket to talk to.
    socket: PathBuf,
    /// Whether starting a daemon was explicitly forbidden.
    no_autostart: bool,
    /// Force the X11 backend.
    x11: bool,
    /// Graphics backend.
    renderer: RendererChoice,
    /// Print diagnostics and exit.
    info: bool,
}

fn main() -> std::process::ExitCode {
    let options = match parse_args(std::env::args_os().skip(1)) {
        Ok(Some(options)) => options,
        Ok(None) => {
            println!("{USAGE}");
            return std::process::ExitCode::SUCCESS;
        }
        Err(message) => {
            eprintln!("noise-gui: {message}");
            eprintln!("{USAGE}");
            return std::process::ExitCode::from(2);
        }
    };

    let windowing = choose_windowing(
        options.x11,
        std::env::var_os("WAYLAND_DISPLAY").as_deref(),
        std::env::var_os("WINIT_UNIX_BACKEND").as_deref(),
    );
    eprintln!(
        "noise-gui: windowing backend {} (WINIT_UNIX_BACKEND is {})",
        windowing.label(),
        match std::env::var("WINIT_UNIX_BACKEND") {
            Ok(value) => value,
            Err(_) => "unset".to_string(),
        }
    );

    if !options.no_autostart && !options.info && !is_listening(&options.socket) {
        match start_daemon(&options.socket) {
            Ok(()) => eprintln!(
                "noise-gui: no daemon on {}; started noiseforged",
                options.socket.display()
            ),
            Err(error) => eprintln!(
                "noise-gui: no daemon on {} and it could not be started ({error}); \
                 the window will keep retrying",
                options.socket.display()
            ),
        }
    }

    // The wgpu backend is a compile time option: it doubles the dependency
    // tree and OpenGL through EGL is already hardware accelerated, so it is off
    // unless the crate is built with `--features wgpu`.
    let mut candidates: Vec<eframe::Renderer> = match options.renderer {
        #[cfg(feature = "wgpu")]
        RendererChoice::Auto => vec![eframe::Renderer::Wgpu, eframe::Renderer::Glow],
        #[cfg(not(feature = "wgpu"))]
        RendererChoice::Auto => vec![eframe::Renderer::Glow],
        #[cfg(feature = "wgpu")]
        RendererChoice::Wgpu => vec![eframe::Renderer::Wgpu],
        #[cfg(not(feature = "wgpu"))]
        RendererChoice::Wgpu => {
            eprintln!(
                "noise-gui: this build has no Vulkan backend; rebuild with \
                 `--features noise-gui/wgpu` or use --renderer glow"
            );
            return std::process::ExitCode::FAILURE;
        }
        RendererChoice::Glow => vec![eframe::Renderer::Glow],
    };
    let mut last: Option<String> = None;
    while let Some(renderer) = candidates.first().copied() {
        candidates.remove(0);
        match run_window(renderer, windowing, &options) {
            Ok(()) => return std::process::ExitCode::SUCCESS,
            Err(error) => {
                let message = error.to_string();
                let retry = !candidates.is_empty() && looks_like_gpu_failure(&error, &message);
                if retry {
                    eprintln!(
                        "noise-gui: the {renderer} renderer could not start ({message}); \
                         falling back to {}",
                        candidates[0]
                    );
                } else {
                    eprintln!("noise-gui: could not open a window: {message}");
                    return std::process::ExitCode::FAILURE;
                }
                last = Some(message);
            }
        }
    }
    if let Some(message) = last {
        eprintln!("noise-gui: no usable renderer: {message}");
    }
    std::process::ExitCode::FAILURE
}

/// Open the window (once), reporting the GPU adapter first.
fn run_window(
    renderer: eframe::Renderer,
    windowing: Windowing,
    options: &Options,
) -> eframe::Result {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("NoiseForge")
            // Only a size, never a minimum size: a minimum size hint is the
            // usual reason a tiling compositor decides to float a window.
            .with_inner_size([1100.0, 760.0]),
        renderer,
        event_loop_builder: Some(event_loop_hook(windowing)),
        ..Default::default()
    };
    let gui_options = GuiOptions {
        socket: options.socket.clone(),
    };
    let info_only = options.info;
    eframe::run_native(
        "noiseforge-gui",
        native_options,
        Box::new(move |cc| {
            report_gpu(cc, renderer);
            Ok(Box::new(App::new(
                gui_options,
                cc.egui_ctx.clone(),
                !info_only,
            )))
        }),
    )
}

/// Force the windowing backend through winit's platform extension traits.
fn event_loop_hook(windowing: Windowing) -> eframe::EventLoopBuilderHook {
    Box::new(move |builder| match windowing {
        Windowing::Wayland => {
            builder.with_wayland();
        }
        Windowing::X11 => {
            builder.with_x11();
        }
        Windowing::Auto => {}
    })
}

/// Print the vendor and device strings of the adapter in use.
///
/// The strings only exist once a graphics context does, so this runs from the
/// app creator - before the first frame, and before the UI talks to anything.
fn report_gpu(cc: &eframe::CreationContext<'_>, renderer: eframe::Renderer) {
    match renderer {
        eframe::Renderer::Glow => match &cc.gl {
            Some(gl) => {
                use eframe::glow::HasContext as _;
                // Only the version is reachable without `unsafe`:
                // `glow::HasContext::get_parameter_string` (vendor, renderer,
                // version string) is an unsafe call in glow 0.17, and this crate
                // forbids unsafe code. `--renderer wgpu` reports the adapter.
                let version = gl.version();
                println!(
                    "noise-gui: renderer glow, GL version {}.{} (vendor and device strings \
                     need an unsafe glGetString; use --renderer wgpu for the adapter name)",
                    version.major, version.minor
                );
            }
            None => println!("noise-gui: renderer glow, no GL context available"),
        },
        #[cfg(feature = "wgpu")]
        eframe::Renderer::Wgpu => match &cc.wgpu_render_state {
            Some(state) => {
                let info = state.adapter.get_info();
                println!(
                    "noise-gui: renderer wgpu, backend {:?}, vendor 0x{:04x}, device \"{}\", \
                     driver \"{}\" {}, type {:?}",
                    info.backend,
                    info.vendor,
                    info.name,
                    info.driver,
                    info.driver_info,
                    info.device_type
                );
            }
            None => println!("noise-gui: renderer wgpu, no render state available"),
        },
        // Without the wgpu feature `Renderer` has only the `Glow` variant, so
        // there is nothing left to match and nothing to report.
    }
}

/// Parse the arguments, or `Ok(None)` for `--help`.
///
/// Hand rolled on purpose: a handful of flags do not justify a parser
/// dependency, and this keeps the accepted syntax in one auditable place.
fn parse_args(args: impl IntoIterator<Item = OsString>) -> Result<Option<Options>, String> {
    let mut socket_flag: Option<OsString> = None;
    let mut no_autostart = false;
    let mut x11 = false;
    let mut renderer = RendererChoice::Auto;
    let mut info = false;
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        match arg.to_str() {
            Some("--socket") => {
                let value = args.next().ok_or("--socket needs a path")?;
                socket_flag = Some(value);
            }
            Some("--no-autostart") => no_autostart = true,
            Some("--x11") => x11 = true,
            Some("--info") => info = true,
            Some("--renderer") => {
                let value = args.next().ok_or("--renderer needs a value")?;
                renderer = parse_renderer(value.to_str().unwrap_or_default())?;
            }
            Some("--help" | "-h") => return Ok(None),
            Some(other) => {
                if let Some(value) = other.strip_prefix("--socket=") {
                    if value.is_empty() {
                        return Err("--socket needs a path".to_string());
                    }
                    socket_flag = Some(OsString::from(value));
                } else if let Some(value) = other.strip_prefix("--renderer=") {
                    renderer = parse_renderer(value)?;
                } else {
                    return Err(format!("unknown argument '{other}'"));
                }
            }
            None => return Err(format!("argument {arg:?} is not valid unicode")),
        }
    }
    let socket = resolve_socket(socket_flag, std::env::var_os("NOISEFORGE_SOCKET"));
    Ok(Some(Options {
        socket,
        no_autostart,
        x11,
        renderer,
        info,
    }))
}

/// Parse a `--renderer` value.
fn parse_renderer(value: &str) -> Result<RendererChoice, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "auto" => Ok(RendererChoice::Auto),
        "wgpu" => Ok(RendererChoice::Wgpu),
        "glow" | "gl" | "opengl" => Ok(RendererChoice::Glow),
        other => Err(format!(
            "unknown renderer '{other}' (expected auto, wgpu or glow)"
        )),
    }
}

/// Decide the windowing backend.
///
/// `--x11` wins, then an explicit `WINIT_UNIX_BACKEND`, then the presence of a
/// Wayland display; otherwise winit decides.
fn choose_windowing(
    force_x11: bool,
    wayland_display: Option<&OsStr>,
    winit_backend: Option<&OsStr>,
) -> Windowing {
    if force_x11 {
        return Windowing::X11;
    }
    match winit_backend
        .and_then(OsStr::to_str)
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("wayland") => Windowing::Wayland,
        Some("x11") => Windowing::X11,
        _ => match wayland_display {
            Some(display) if !display.is_empty() => Windowing::Wayland,
            _ => Windowing::Auto,
        },
    }
}

/// Whether a failure is worth retrying with the other renderer.
fn looks_like_gpu_failure(error: &eframe::Error, message: &str) -> bool {
    #[cfg(feature = "wgpu")]
    if matches!(error, eframe::Error::Wgpu(_)) {
        return true;
    }
    #[cfg(not(feature = "wgpu"))]
    let _ = error;
    let message = message.to_ascii_lowercase();
    ["wgpu", "adapter", "device", "vulkan", "no suitable"]
        .iter()
        .any(|needle| message.contains(needle))
}

/// Pick the socket: the flag wins, then the environment, then the daemon default.
fn resolve_socket(flag: Option<OsString>, environment: Option<OsString>) -> PathBuf {
    flag.or(environment)
        .map(PathBuf::from)
        .unwrap_or_else(default_socket_path)
}

/// Whether anything is listening on the socket.
///
/// A bare connect is enough: it proves a socket exists without speaking the
/// protocol, so a version mismatch still counts as "a daemon is there".
fn is_listening(socket: &Path) -> bool {
    std::os::unix::net::UnixStream::connect(socket).is_ok()
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

/// Start a daemon on `socket`, detached, exactly as the CLI does.
fn start_daemon(socket: &Path) -> Result<(), std::io::Error> {
    let mut command = ProcessCommand::new(daemon_binary());
    command
        .arg("--socket")
        .arg(socket)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
    command.spawn()?;
    // Give the socket a moment to appear so the first connection usually
    // succeeds; the background thread retries on its own either way.
    let deadline = Instant::now() + Duration::from_secs(2);
    while Instant::now() < deadline {
        if is_listening(socket) {
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<OsString> {
        list.iter().map(OsString::from).collect()
    }

    #[test]
    fn defaults_are_the_daemon_default_socket_and_autostart() {
        let options = parse_args(args(&[])).unwrap().unwrap();
        assert_eq!(options.socket, default_socket_path());
        assert!(!options.no_autostart);
        assert!(!options.x11);
        assert!(!options.info);
        assert_eq!(options.renderer, RendererChoice::Auto);
    }

    #[test]
    fn the_socket_flag_accepts_both_spellings() {
        let split = parse_args(args(&["--socket", "/tmp/a.sock"]))
            .unwrap()
            .unwrap();
        assert_eq!(split.socket, PathBuf::from("/tmp/a.sock"));
        let joined = parse_args(args(&["--socket=/tmp/b.sock"]))
            .unwrap()
            .unwrap();
        assert_eq!(joined.socket, PathBuf::from("/tmp/b.sock"));
    }

    #[test]
    fn the_environment_is_used_when_the_flag_is_absent() {
        let env = Some(OsString::from("/tmp/env.sock"));
        assert_eq!(
            resolve_socket(None, env.clone()),
            PathBuf::from("/tmp/env.sock")
        );
        // The flag wins over the environment.
        assert_eq!(
            resolve_socket(Some(OsString::from("/tmp/flag.sock")), env),
            PathBuf::from("/tmp/flag.sock")
        );
        // Neither set falls back to the daemon default.
        assert_eq!(resolve_socket(None, None), default_socket_path());
    }

    #[test]
    fn flags_can_be_combined_in_any_order() {
        let options = parse_args(args(&[
            "--no-autostart",
            "--socket=/tmp/x.sock",
            "--x11",
            "--renderer",
            "glow",
        ]))
        .unwrap()
        .unwrap();
        assert!(options.no_autostart);
        assert!(options.x11);
        assert_eq!(options.socket, PathBuf::from("/tmp/x.sock"));
        assert_eq!(options.renderer, RendererChoice::Glow);
    }

    #[test]
    fn the_renderer_flag_accepts_the_documented_names() {
        assert_eq!(parse_renderer("auto"), Ok(RendererChoice::Auto));
        assert_eq!(parse_renderer("WGPU"), Ok(RendererChoice::Wgpu));
        assert_eq!(parse_renderer(" glow "), Ok(RendererChoice::Glow));
        assert!(parse_renderer("software").is_err());
        assert!(
            parse_args(args(&["--renderer=wgpu"]))
                .unwrap()
                .unwrap()
                .renderer
                == RendererChoice::Wgpu
        );
        assert!(parse_args(args(&["--renderer"])).is_err());
        assert!(parse_args(args(&["--renderer=nope"])).is_err());
    }

    #[test]
    fn the_windowing_backend_prefers_wayland_on_a_wayland_session() {
        let wayland = OsStr::new("wayland-1");
        let x11 = OsStr::new("x11");
        assert_eq!(
            choose_windowing(false, Some(wayland), None),
            Windowing::Wayland
        );
        // An explicit environment variable wins over the session type.
        assert_eq!(
            choose_windowing(false, Some(wayland), Some(x11)),
            Windowing::X11
        );
        // WINIT_UNIX_BACKEND holds the backend *name*, not a display name.
        assert_eq!(
            choose_windowing(false, None, Some(OsStr::new("wayland"))),
            Windowing::Wayland
        );
        // No display at all: let winit decide.
        assert_eq!(choose_windowing(false, None, None), Windowing::Auto);
        assert_eq!(
            choose_windowing(false, Some(OsStr::new("")), None),
            Windowing::Auto
        );
        // The flag beats everything.
        assert_eq!(
            choose_windowing(true, Some(wayland), Some(wayland)),
            Windowing::X11
        );
    }

    #[test]
    fn help_is_not_an_error() {
        assert!(parse_args(args(&["--help"])).unwrap().is_none());
        assert!(parse_args(args(&["-h"])).unwrap().is_none());
    }

    #[test]
    fn bad_arguments_are_reported_not_ignored() {
        assert!(parse_args(args(&["--nope"])).is_err());
        assert!(parse_args(args(&["--socket"])).is_err());
        assert!(parse_args(args(&["--socket="])).is_err());
        assert!(parse_args(args(&["stray"])).is_err());
    }

    #[test]
    fn a_missing_socket_is_not_listening() {
        assert!(!is_listening(Path::new("/tmp/definitely-not-here.sock")));
    }
}
