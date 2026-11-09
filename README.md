<div align="center">

# 🔊 NoiseForge

### Realtime, mathematically defined noise generation for Linux — built natively on PipeWire.

![Rust](https://img.shields.io/badge/Rust-000000?logo=rust&logoColor=fff)
![PipeWire](https://img.shields.io/badge/PipeWire-native-FF6600)
![License](https://img.shields.io/badge/license-MIT_OR_Apache--2.0-blue)
![noise colors](https://img.shields.io/badge/colors-white_|_pink_|_brown_|_blue_|_violet-8b5cf6)

</div>

> 🎧 NoiseForge does not loop samples. It generates noise **live** from a seeded PRNG and a cascade of IIR shaping filters — white, pink, brown, blue and violet noise at any length, at any sample rate the server negotiates, always reproducible from a seed.

---

Realtime, mathematically defined noise generation for Linux, built natively on
PipeWire.

NoiseForge does not loop samples. It generates noise live from a seeded PRNG and
a cascade of IIR shaping filters, so white, pink, brown, blue and violet noise are
available at any length, at any sample rate the server negotiates, and always
reproducible from a seed.

```
                    Color          Character        EQ/HPF/LPF        Stereo
 seeded PRNG  ->  shaping IIR  ->  shelf stage  ->  user filters  ->  rho matrix
                                                                        |
   master gain <- play/pause/timer/fade <- sum of up to 8 layers <-------+
        |
        +-> peak monitor -> optional explicit limiter -> PipeWire F32LE
```

## 🧩 What is implemented

| Component | Crate | State |
|---|---|---|
| Realtime DSP core | `noise-dsp` | complete, 63 unit tests |
| Native PipeWire output | `noise-pipewire` | complete, verified playing to real hardware |
| Daemon, IPC, presets | `noise-daemon` | complete, 18 unit + 15 integration tests |
| CLI | `noise-cli` | complete, verified end to end |
| Offline renderer / WAV / Welch PSD | `noise-offline` | complete |
| GUI | `noise-gui` | complete |
| Spectral accuracy study | `noise-dsp --example design_study` | complete |
| Render benchmark | `benches/render.rs` | complete |

## 📈 Measured results

Everything below was measured on the development machine (Intel i5-8400, PipeWire
1.6.9, Rust 1.96.1) and is reproducible with the commands shown. Nothing here is
a design target restated as if it were a measurement.

### Spectral accuracy (DESIGN.md section 9.1)

`cargo run -p noise-dsp --release --example design_study`

| Colour | Worst shape error | Worst slope error | Sections used |
|---|---:|---:|---:|
| White | 0.000 dB | 0.000 dB/oct | 0 (exact bypass) |
| Pink | 0.084 dB | 0.001 dB/oct | 3 biquads |
| Brown | 0.221 dB | 0.002 dB/oct | 3–7 biquads |
| Blue | 0.084 dB | 0.001 dB/oct | 3 biquads |
| Violet | 0.216 dB | 0.002 dB/oct | 3–7 biquads |
| Custom ±4 | 0.627 dB | 0.012 dB/oct | 3 biquads |

Across 44.1 / 48 / 96 kHz. The provisional targets were ±1.5 dB and
±0.30 dB/oct, so every colour passes with a wide margin — and the *measured*
error is what `noiseforge status` prints, never a claim.

### Realtime performance (DESIGN.md section 9.2)

`cargo bench -p noiseforge`, 48 kHz stereo, f64 filter state, no FFT:

| Configuration | ns / frame | CPU of one core |
|---|---:|---:|
| 1 layer, White, Neutral | 30.7 | 0.147 % |
| 1 layer, Pink, Neutral | 43.0 | 0.207 % |
| 1 layer, Pink, Smooth 0.7 | 51.9 | 0.249 % |
| 1 layer, Brown | 61.3 | 0.294 % |
| 8 layers, mixed | 315.5 | 1.514 % |
| 8 layers, all Intense | 334.5 | 1.606 % |

The design target for a single layer was **under 5 %** of one i5-8400 core; the
worst single layer case measures 0.294 %. At 128 frames (2.67 ms period) the
callback uses 0.21 % of its deadline.

### Realtime safety (DESIGN.md sections 4.3 and 9.2)

`cargo test -p noiseforge --test rt_alloc` installs a counting global allocator
and asserts that the render path, *including live parameter changes*, performs
**zero** allocations and **zero** frees.

### Live output

`cargo test -p noiseforge --test spectral` renders and measures; a live daemon
run reports:

```
output 48000 Hz, 2 channel(s)   target -36.0 dBFS RMS
measured peak -25.27 dBFS   rms -36.00 dBFS   xruns 0   nan 0
```

The analytic calibration lands on its target to within a fraction of a dB in
normal use, because it computes the exact expected RMS of the whole configured
graph rather than tracking it per sample.

## 🔨 Building

Requires Rust 1.95 or later (edition 2024) and the PipeWire development headers:

```sh
sudo apt install libpipewire-0.3-dev libspa-0.2-dev      # Debian/Ubuntu
sudo dnf install pipewire-devel                           # Fedora
cargo build --release
```

The GUI additionally needs the usual windowing libraries (Wayland or X11, GL,
xkbcommon, fontconfig); they are present on any desktop install.

## ▶️ Running

```sh
# Start the daemon (also starts on demand from the CLI or GUI).
noiseforged &

# Or let the CLI start it for you:
noiseforge play pink --character smooth --intensity 0.7 --level-rms-dbfs -36
noiseforge status
noiseforge monitor --spectrum
noiseforge pause
noiseforge stop
```

### Who keeps the noise going

Splitting the daemon out is what lets playback survive a client exiting
(DESIGN.md section 6), but a generator that hums in an empty room after its
window was closed is worse than useless. The rule is therefore:

| Situation | What happens |
|---|---|
| `noiseforge play ...` then the CLI exits | keeps playing, until `noiseforge stop` |
| the **GUI window is closed** | fades out and stops, unless the "Keep playing in the background" box is ticked |
| `noiseforged --keep-playing` (as in the systemd unit) | keeps playing with nobody attached |
| `noiseforged --autoplay` | keeps playing with nobody attached |
| nobody attached and nothing playing | the daemon exits after `--idle-shutdown` seconds (default 120) |

A connecting GUI takes over from a CLI hold, which is why closing the window is
silent even after an earlier `noiseforge play`. `noiseforge shutdown` stops the
daemon outright.

## 🖥 GUI

`noiseforge-gui` uses egui/eframe and renders **on the GPU**: `wgpu` (Vulkan) by
default, with `glow` (OpenGL/EGL) as a fallback if no Vulkan adapter is usable.

```sh
noiseforge-gui                  # Vulkan if available, else OpenGL
noiseforge-gui --renderer glow  # force OpenGL
noiseforge-gui --info           # print the backend and the GPU adapter
noiseforge-gui --x11            # force the X11 backend instead of Wayland
```

It is Wayland native by default on a Wayland session and will tile normally in
sway, i3 and friends.

### CLI

```
noiseforge play pink --character smooth --intensity 0.7 --level-rms-dbfs -36
noiseforge play brown --character intense --intensity 0.4
noiseforge color custom --alpha 1.35
noiseforge set --high-pass 40 --low-pass 12000
noiseforge layer add
noiseforge layer list
noiseforge preset save "Soft Pink"
noiseforge preset load "Soft Pink"
noiseforge device list
noiseforge device select default
noiseforge timer --minutes 45 --fade-out 60
noiseforge status --json
noiseforge pause
noiseforge stop
noiseforge render --seconds 60 --rate 48000 --seed 42 --wav sample.wav --analyze
```

`--intensity` is the character strength (0.0 to 1.0). `--level-rms-dbfs` is the
expected RMS of the finished PCM. They are different things and are never mixed
up.

### Offline rendering

`noiseforge render` needs neither a daemon nor a sound server, and writes exactly
the PCM the daemon would have played for the same preset and seed:

```sh
noiseforge render pink --seconds 600 --wav pink.flac.wav --format i24 --analyze
```

`--analyze` prints the Welch-measured slope next to the theoretical one.

## 🎛 Presets and configuration

TOML, versioned, saved atomically under `$XDG_CONFIG_HOME/noiseforge/`
(`~/.config/noiseforge/` by default). Four factory presets ship in `presets/`:

* `soft-pink` — the everyday default,
* `deep-brown` — the full 1/f² tilt,
* `bright-air` — blue plus violet, wide stereo,
* `custom-tilt` — the custom alpha path at 1.35.

Copy one into `~/.config/noiseforge/presets/` to use it by name.

## ✅ Testing

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets     # zero warnings
cargo test --workspace                     # 120+ tests
cargo bench -p noiseforge
cargo run -p noise-dsp --release --example design_study

# The eight hour soak from DESIGN.md section 9.2:
NOISEFORGE_SOAK_SECONDS=28800 cargo test --release -p noiseforge --test soak -- --nocapture
```

The test suite covers the spectral acceptance criteria, bit-exact determinism
across buffer sizes, zero realtime allocation, click-free parameter changes under
adversarial traffic, the IPC contract, preset robustness and a long soak.

## ⚠️ Honest limitations

* The shaping filters are finite-order IIR. Accuracy is guaranteed and measured
  over **40 Hz to 16 kHz** (`min(16000, 0.42 · Fs)`), not to Nyquist. Outside
  that band the response flattens; `noiseforge status` reports the measured error
  and marks a layer `[OUT OF TOLERANCE]` if it misses the documented criteria.
* `Smooth` and `Intense` are linear filters. The output is *not* the pristine
  mathematical colour, and the UI says so.
* The limiter is nonlinear, off by default, and never enabled silently.
* `dBFS`, `LUFS` and `dB SPL` are different units. NoiseForge reports `dBFS` and
  makes no claim about the sound pressure at your ears.
* No medical or concentration benefit is claimed for any noise colour.
* Multi-channel output beyond stereo writes silence to the extra channels.
* Recording to file is done by `noiseforge render`, not by the daemon.

## 🗂 Layout

```
noiseforge/
  Cargo.toml            workspace plus an umbrella crate for the root tests
  DESIGN.md             the specification, with a recorded deviations section
  crates/
    noise-dsp/          realtime DSP core, no OS or PipeWire dependency
    noise-pipewire/     PipeWire stream, format negotiation, telemetry
    noise-daemon/       control loop, Unix socket IPC, preset store, noiseforged
    noise-cli/          the noiseforge client
    noise-gui/          the egui/eframe front end
    noise-offline/      offline rendering, Welch PSD, WAV writing
  presets/              factory presets
  tests/                workspace level integration tests
  benches/              render benchmark
  packaging/            systemd unit and desktop entries
```

## License

MIT OR Apache-2.0.
