//! The versioned IPC protocol (DESIGN.md section 6).
//!
//! Small JSON request/response messages over a Unix domain socket, one per line.
//! The protocol version is explicit and every message carries it, so a stale GUI
//! cannot silently talk to a newer daemon.
//!
//! Nothing here is on the realtime thread: parsing, allocation and formatting all
//! live on the daemon's control loop.

use noise_dsp::params::{LayerParams, MasterParams, PresetParams};
use serde::{Deserialize, Serialize};

/// Protocol version. Bump on any incompatible change.
pub const PROTOCOL_VERSION: u32 = 1;

/// Maximum size of a single request or response line (DESIGN.md section 6).
pub const MAX_MESSAGE_BYTES: usize = 16 * 1024;

/// Default socket path: `$XDG_RUNTIME_DIR/noiseforge/control.sock`.
pub fn default_socket_path() -> std::path::PathBuf {
    let base = std::env::var_os("XDG_RUNTIME_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    base.join("noiseforge").join("control.sock")
}

/// Which audio backend the daemon is running.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    /// Native PipeWire stream.
    PipeWire,
    /// No output; used by tests and by `--backend null`.
    Null,
}

/// How the output sample rate is chosen, as a runtime setting.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RateMode {
    /// Accept the server's rate and rebuild the engine when it changes.
    Follow,
    /// Request a specific rate.
    Pin(u32),
}

/// Fields a client may set directly.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct SetArgs {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub color: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub alpha: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub character: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub intensity: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub high_pass: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub low_pass: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stereo: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gain_db: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layer: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub level_rms_dbfs: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub volume_db: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mute: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fade_in_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fade_out_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limiter: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limiter_ceiling_dbfs: Option<f64>,
}

/// Requests the CLI and GUI can send.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Request {
    /// Protocol handshake and liveness check.
    Hello {
        client: String,
    },
    /// Full current state.
    Status,
    /// Start or resume.
    Play,
    /// Fade out and hold.
    Pause,
    /// Fade out and stop.
    Stop,
    /// Patch fields of the current preset.
    Set {
        args: Box<SetArgs>,
    },
    /// Replace the whole preset.
    Load {
        preset: Box<PresetParams>,
    },
    /// Read the current preset.
    GetPreset,
    /// Layer management.
    LayerAdd {
        after: Option<usize>,
    },
    LayerRemove {
        index: usize,
    },
    LayerDuplicate {
        index: usize,
    },
    /// Preset files.
    PresetSave {
        name: String,
    },
    PresetLoad {
        name: String,
    },
    PresetList,
    PresetDelete {
        name: String,
    },
    /// Output devices.
    DeviceList,
    DeviceSelect {
        name: String,
    },
    /// Sleep timer.
    Timer {
        seconds: f64,
        fade_out_seconds: f64,
    },
    TimerClear,
    /// Switch the stream to another sink by node id.
    Retarget {
        node: Option<u32>,
    },
    /// Ask the daemon to write the current preset into the config file.
    SaveConfig,
    /// Stop the daemon.
    Shutdown,
    /// Claim or release an indefinite hold on playback.
    ///
    /// The daemon stops generating when the last client disconnects unless
    /// somebody holds it, which is what makes closing the window silence the
    /// noise while `noiseforge play` keeps running until `noiseforge stop`.
    /// Both the GUI and `noiseforge stop` release the hold.
    Hold {
        hold: bool,
    },
    /// Register for periodic telemetry pushes on this connection.
    Subscribe {
        #[serde(default)]
        psd: bool,
        #[serde(default = "default_hz")]
        hz: f64,
    },
    /// Stop periodic pushes.
    Unsubscribe,
}

fn default_hz() -> f64 {
    15.0
}

/// A response to a [`Request`].
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Response {
    /// Protocol version.
    pub v: u32,
    /// Echoes the request id, or 0 for an unsolicited push.
    pub id: u64,
    /// Success flag.
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Warnings that do not fail the request but must be shown to the user.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,
}

impl Response {
    /// A successful response carrying `data`.
    pub fn ok(id: u64, data: serde_json::Value) -> Self {
        Self {
            v: PROTOCOL_VERSION,
            id,
            ok: true,
            data: Some(data),
            error: None,
            notes: Vec::new(),
        }
    }

    /// A successful response with no payload.
    pub fn empty(id: u64) -> Self {
        Self {
            v: PROTOCOL_VERSION,
            id,
            ok: true,
            data: None,
            error: None,
            notes: Vec::new(),
        }
    }

    /// A failed response.
    pub fn err(id: u64, message: impl Into<String>) -> Self {
        Self {
            v: PROTOCOL_VERSION,
            id,
            ok: false,
            data: None,
            error: Some(message.into()),
            notes: Vec::new(),
        }
    }

    /// Attach warnings.
    pub fn with_notes(mut self, notes: Vec<String>) -> Self {
        self.notes = notes;
        self
    }
}

/// One layer as reported to clients.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LayerStatus {
    pub index: usize,
    pub name: String,
    pub enabled: bool,
    pub color: String,
    pub alpha: f64,
    pub character: String,
    pub character_amount: f64,
    pub high_pass_hz: f64,
    pub low_pass_hz: f64,
    pub stereo_correlation: f64,
    pub gain_db: f64,
    pub seed: u64,
    /// Sections in the designed colour cascade.
    pub sections: usize,
    /// Measured shape error over the promised band, in dB.
    pub shape_error_db: Option<f64>,
    /// Measured slope error, in dB per octave.
    pub slope_error_db_per_octave: Option<f64>,
    /// False when the design missed the documented tolerance.
    pub verified: bool,
    /// Predicted RMS of this layer at the mix, in dBFS.
    pub predicted_rms_dbfs: f64,
}

/// Analytic prediction reported to the UI.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CalibrationStatus {
    pub target_rms_dbfs: f64,
    pub mix_rms_dbfs: f64,
    pub calibration_gain_db: f64,
    pub master_gain_db: f64,
    pub predicted_peak_dbfs: f64,
    pub headroom_warning: bool,
}

/// Live measurement.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct MeterStatus {
    pub peak_dbfs: f64,
    pub rms_dbfs: f64,
    pub xruns: u64,
    pub callbacks: u64,
    pub frames: u64,
    pub nan_events: u64,
    pub clipped: u64,
    pub tap_dropped: u64,
}

/// Full daemon state.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StatusData {
    pub version: String,
    pub protocol: u32,
    pub dsp_version: u32,
    pub backend: BackendKind,
    pub playing: bool,
    pub transport: String,
    pub link: String,
    pub sample_rate: u32,
    pub channels: u32,
    pub rate_mode: RateMode,
    pub device: Option<String>,
    pub device_node_id: Option<u32>,
    pub timer_remaining_s: Option<f64>,
    pub uptime_s: f64,
    pub calibration: CalibrationStatus,
    pub meters: MeterStatus,
    pub layers: Vec<LayerStatus>,
    pub master: MasterParams,
    pub notes: Vec<String>,
    pub preset_name: String,
    pub seed: u64,
    pub auto_start: bool,
}

/// One curve in a telemetry frame.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Curve {
    pub f_lo: f64,
    pub f_hi: f64,
    pub db: Vec<f32>,
}

/// Unsolicited telemetry pushed to subscribers.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TelemetryFrame {
    pub id: u64,
    pub ok: bool,
    pub status: Box<StatusData>,
    /// Measured output PSD, log spaced, when requested.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub measured: Option<Curve>,
    /// Theoretical source target for the first layer.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_target: Option<Curve>,
    /// Analytic predicted output for the whole graph.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub predicted: Option<Curve>,
}

/// Names the daemon uses for preset files.
pub fn preset_file_name(name: &str) -> String {
    let mut out = String::with_capacity(name.len() + 5);
    for c in name.chars() {
        if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == ' ' {
            out.push(c);
        } else {
            out.push('_');
        }
    }
    let trimmed = out.trim();
    let base = if trimmed.is_empty() {
        "preset"
    } else {
        trimmed
    };
    format!("{base}.toml")
}

/// Apply a [`SetArgs`] patch to a preset, returning notes about anything that
/// had to be clamped or ignored.
pub fn apply_set(preset: &mut PresetParams, args: &SetArgs) -> Vec<String> {
    let mut notes = Vec::new();
    let index = args.layer.unwrap_or(0);
    if index >= preset.layers.len() {
        notes.push(format!("layer {index} does not exist; ignored"));
        return notes;
    }
    if let Some(color) = &args.color {
        match noise_dsp::Color::parse(color) {
            Some(c) => preset.layers[index].color = c,
            None => notes.push(format!("unknown colour '{color}'")),
        }
    }
    if let Some(character) = &args.character {
        match noise_dsp::Character::parse(character) {
            Some(c) => preset.layers[index].character = c,
            None => notes.push(format!("unknown character '{character}'")),
        }
    }
    if let Some(v) = args.alpha {
        preset.layers[index].alpha = v;
        if preset.layers[index].color != noise_dsp::Color::Custom {
            notes.push(format!(
                "alpha only applies to the custom colour; {} uses alpha {:.2}",
                preset.layers[index].color,
                preset.layers[index].alpha()
            ));
        }
    }
    if let Some(v) = args.intensity {
        preset.layers[index].character_amount = v;
    }
    if let Some(v) = args.high_pass {
        preset.layers[index].high_pass_hz = v;
    }
    if let Some(v) = args.low_pass {
        preset.layers[index].low_pass_hz = v;
    }
    if let Some(v) = args.stereo {
        preset.layers[index].stereo_correlation = v;
    }
    if let Some(v) = args.gain_db {
        preset.layers[index].gain_db = v;
    }
    if let Some(v) = args.seed {
        preset.layers[index].seed = Some(v);
    }
    if let Some(v) = args.level_rms_dbfs {
        preset.master.target_rms_dbfs = v;
    }
    if let Some(v) = args.volume_db {
        preset.master.volume_db = v;
    }
    if let Some(v) = args.mute {
        preset.master.mute = v;
    }
    if let Some(v) = args.fade_in_ms {
        preset.master.fade_in_ms = v;
    }
    if let Some(v) = args.fade_out_ms {
        preset.master.fade_out_ms = v;
    }
    if let Some(v) = args.limiter {
        preset.master.limiter_enabled = v;
    }
    if let Some(v) = args.limiter_ceiling_dbfs {
        preset.master.limiter_ceiling_dbfs = v;
    }
    notes.extend(preset.sanitize());
    notes
}

/// A per-layer patch helper used by the CLI's `color` and `character` verbs.
pub fn patch_layer(preset: &mut PresetParams, index: usize, layer: LayerParams) -> Vec<String> {
    let mut notes = Vec::new();
    if index < preset.layers.len() {
        preset.layers[index] = layer;
        notes.extend(preset.sanitize());
    } else {
        notes.push(format!("layer {index} does not exist"));
    }
    notes
}

/// Encode one message as a single line.
pub fn encode<T: Serialize>(value: &T) -> Result<Vec<u8>, serde_json::Error> {
    let mut bytes = serde_json::to_vec(value)?;
    bytes.push(b'\n');
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_round_trips() {
        let r = Request::Set {
            args: Box::new(SetArgs {
                color: Some("pink".into()),
                intensity: Some(0.7),
                ..Default::default()
            }),
        };
        let line = encode(&r).unwrap();
        assert_eq!(line.last(), Some(&b'\n'));
        let back: Request = serde_json::from_slice(&line[..line.len() - 1]).unwrap();
        match back {
            Request::Set { args } => {
                assert_eq!(args.color.as_deref(), Some("pink"));
                assert_eq!(args.intensity, Some(0.7));
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn responses_omit_empty_fields() {
        let text = serde_json::to_string(&Response::empty(7)).unwrap();
        assert!(text.contains("\"ok\":true"));
        assert!(!text.contains("error"));
        assert!(!text.contains("notes"));
    }

    #[test]
    fn preset_file_names_are_sanitised() {
        assert_eq!(preset_file_name("Soft Pink"), "Soft Pink.toml");
        // Every character that is not safe in a file name becomes an underscore,
        // so a traversal attempt cannot survive.
        assert_eq!(
            preset_file_name("../../etc/passwd"),
            "______etc_passwd.toml"
        );
        assert_eq!(preset_file_name(""), "preset.toml");
        // No path separators survive.
        assert!(!preset_file_name("a/b").contains('/'));
        assert!(!preset_file_name("a\nb").contains('\n'));
    }

    #[test]
    fn apply_set_clamps_and_reports() {
        let mut p = PresetParams::default();
        let notes = apply_set(
            &mut p,
            &SetArgs {
                alpha: Some(99.0),
                intensity: Some(-1.0),
                color: Some("chartreuse".into()),
                ..Default::default()
            },
        );
        assert_eq!(p.layers[0].alpha, 4.0);
        assert_eq!(p.layers[0].character_amount, 0.0);
        assert!(notes.iter().any(|n| n.contains("chartreuse")));
    }

    #[test]
    fn apply_set_rejects_a_missing_layer() {
        let mut p = PresetParams::default();
        let notes = apply_set(
            &mut p,
            &SetArgs {
                layer: Some(5),
                gain_db: Some(-3.0),
                ..Default::default()
            },
        );
        assert!(notes[0].contains("does not exist"));
    }

    #[test]
    fn a_status_message_fits_the_size_limit() {
        // A worst case status with eight detailed layers must stay inside the
        // 16 KiB budget from DESIGN.md section 6.
        let layers: Vec<LayerStatus> = (0..8)
            .map(|i| LayerStatus {
                index: i,
                name: format!("Layer {i}"),
                enabled: true,
                color: "custom".into(),
                alpha: 1.35,
                character: "intense".into(),
                character_amount: 0.7,
                high_pass_hz: 35.0,
                low_pass_hz: 16000.0,
                stereo_correlation: 0.35,
                gain_db: -3.0,
                seed: 123456789,
                sections: 7,
                shape_error_db: Some(0.123),
                slope_error_db_per_octave: Some(-0.004),
                verified: true,
                predicted_rms_dbfs: -42.5,
            })
            .collect();
        let status = StatusData {
            version: "0.1.0".into(),
            protocol: PROTOCOL_VERSION,
            dsp_version: 1,
            backend: BackendKind::PipeWire,
            playing: true,
            transport: "playing".into(),
            link: "running".into(),
            sample_rate: 96000,
            channels: 2,
            rate_mode: RateMode::Follow,
            device: Some("alsa_output.pci-0000_00_1f.3.analog-stereo".into()),
            device_node_id: Some(54),
            timer_remaining_s: Some(2699.9),
            uptime_s: 12345.6,
            calibration: CalibrationStatus {
                target_rms_dbfs: -36.0,
                mix_rms_dbfs: -31.65,
                calibration_gain_db: -4.35,
                master_gain_db: -4.35,
                predicted_peak_dbfs: -23.54,
                headroom_warning: false,
            },
            meters: MeterStatus::default(),
            layers,
            master: MasterParams::default(),
            notes: vec!["a note".into()],
            preset_name: "Soft Pink".into(),
            seed: 42,
            auto_start: true,
        };
        let size = serde_json::to_vec(&status).unwrap().len();
        assert!(size < MAX_MESSAGE_BYTES, "status is {size} bytes");
    }
}
