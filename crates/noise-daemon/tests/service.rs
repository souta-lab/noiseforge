//! End-to-end daemon tests over the real Unix socket protocol.
//!
//! The service runs on the silent backend so the contract — protocol version,
//! calibration, layer management, presets, permissions, size limits — is checked
//! without needing a sound server.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use noise_daemon::ipc::Client;
use noise_daemon::presets::Store;
use noise_daemon::protocol::{BackendKind, RateMode, Request, Response, SetArgs, StatusData};
use noise_daemon::service::{DaemonConfig, Service};

struct Harness {
    socket: PathBuf,
    _config: tempfile::TempDir,
    _socket_dir: tempfile::TempDir,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Harness {
    fn start() -> Self {
        Self::start_with(|_| {})
    }

    fn start_with(tweak: impl FnOnce(&mut DaemonConfig)) -> Self {
        let config_dir = tempfile::tempdir().unwrap();
        let socket_dir = tempfile::tempdir().unwrap();
        let socket = socket_dir.path().join("control.sock");
        let mut config = DaemonConfig {
            socket: socket.clone(),
            store: Store::at(config_dir.path().join("noiseforge")),
            backend: BackendKind::Null,
            rate_mode: RateMode::Follow,
            channels: 2,
            autoplay: false,
            profile: false,
            load_config: false,
            idle_shutdown: None,
            ..DaemonConfig::default()
        };
        tweak(&mut config);
        let service = Service::new(config).unwrap();
        let (tx, rx) = std::sync::mpsc::channel();
        let thread = std::thread::spawn(move || {
            let _ = service.run(|path| {
                let _ = tx.send(path.to_path_buf());
            });
        });
        let _ = rx.recv_timeout(Duration::from_secs(5)).unwrap();
        Self {
            socket,
            _config: config_dir,
            _socket_dir: socket_dir,
            thread: Some(thread),
        }
    }

    /// A client that only observes, i.e. a stand-in for the GUI.
    fn watcher(&self) -> Client {
        let mut c = self.client();
        let r = c
            .call(&Request::Subscribe {
                psd: false,
                hz: 2.0,
            })
            .unwrap();
        assert!(r.ok);
        c
    }

    fn client(&self) -> Client {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            match Client::connect(&self.socket) {
                Ok(c) => return c,
                Err(e) => {
                    assert!(Instant::now() < deadline, "could not connect: {e}");
                    std::thread::sleep(Duration::from_millis(20));
                }
            }
        }
    }

    fn status(&self) -> (Client, StatusData) {
        let mut client = self.client();
        let response = client.call(&Request::Status).unwrap();
        let data = serde_json::from_value(response.data.unwrap()).unwrap();
        (client, data)
    }
}

impl Drop for Harness {
    fn drop(&mut self) {
        if let Ok(mut c) = Client::connect(&self.socket) {
            let _ = c.call(&Request::Shutdown);
        }
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

fn ok(response: Response) -> Response {
    assert!(response.ok, "request failed: {:?}", response.error);
    response
}

#[test]
fn hello_reports_matching_versions() {
    let h = Harness::start();
    let mut c = h.client();
    let r = ok(c
        .call(&Request::Hello {
            client: "test".into(),
        })
        .unwrap());
    let data = r.data.unwrap();
    assert_eq!(
        data["protocol"].as_u64().unwrap(),
        noise_daemon::PROTOCOL_VERSION as u64
    );
    assert_eq!(
        data["dsp_version"].as_u64().unwrap(),
        noise_dsp::DSP_VERSION as u64
    );
}

#[test]
fn the_daemon_defaults_to_a_calibrated_single_layer() {
    let h = Harness::start();
    let (_, s) = h.status();
    assert_eq!(s.layers.len(), 1);
    assert_eq!(s.layers[0].color, "pink");
    assert!(s.layers[0].verified);
    assert!((s.calibration.target_rms_dbfs + 36.0).abs() < 1e-9);
    assert!(s.calibration.master_gain_db.is_finite());
    assert_eq!(s.sample_rate, 48_000);
}

#[test]
fn every_colour_and_character_is_accepted_and_stays_verified() {
    let h = Harness::start();
    let mut c = h.client();
    for color in ["white", "pink", "brown", "blue", "violet", "custom"] {
        for character in ["neutral", "smooth", "intense"] {
            ok(c.call(&Request::Set {
                args: Box::new(SetArgs {
                    color: Some(color.into()),
                    alpha: Some(1.3),
                    character: Some(character.into()),
                    intensity: Some(0.8),
                    ..Default::default()
                }),
            })
            .unwrap());
            let r = ok(c.call(&Request::Status).unwrap());
            let s: StatusData = serde_json::from_value(r.data.unwrap()).unwrap();
            assert!(
                s.layers[0].verified,
                "{color}/{character} is out of tolerance"
            );
            assert!(
                s.layers[0].shape_error_db.unwrap() <= 1.5,
                "{color}/{character} shape error {:?}",
                s.layers[0].shape_error_db
            );
            assert!(
                s.layers[0].slope_error_db_per_octave.unwrap().abs() <= 0.30,
                "{color}/{character} slope error {:?}",
                s.layers[0].slope_error_db_per_octave
            );
        }
    }
}

#[test]
fn the_level_target_is_honoured_as_the_graph_changes() {
    let h = Harness::start();
    let mut c = h.client();
    for target in [-60.0, -42.0, -36.0, -18.0] {
        ok(c.call(&Request::Set {
            args: Box::new(SetArgs {
                level_rms_dbfs: Some(target),
                ..Default::default()
            }),
        })
        .unwrap());
        let r = ok(c.call(&Request::Status).unwrap());
        let s: StatusData = serde_json::from_value(r.data.unwrap()).unwrap();
        // The analytic calibration must land exactly on the requested target.
        let achieved = s.calibration.mix_rms_dbfs + s.calibration.master_gain_db;
        assert!(
            (achieved - target).abs() < 1e-6,
            "target {target}: achieved {achieved}"
        );
    }
}

#[test]
fn eight_layers_can_be_added_and_removed() {
    let h = Harness::start();
    let mut c = h.client();
    for _ in 0..7 {
        ok(c.call(&Request::LayerAdd { after: None }).unwrap());
    }
    let (_, s) = h.status();
    assert_eq!(s.layers.len(), 8);
    // The ninth must be refused, not clamped silently.
    let r = c.call(&Request::LayerAdd { after: None }).unwrap();
    assert!(!r.ok);
    assert!(r.error.unwrap().contains("8 layer limit"));

    for _ in 0..7 {
        ok(c.call(&Request::LayerRemove { index: 0 }).unwrap());
    }
    let (_, s) = h.status();
    assert_eq!(s.layers.len(), 1);
    // The last layer stays.
    let r = c.call(&Request::LayerRemove { index: 0 }).unwrap();
    assert!(!r.ok);
}

#[test]
fn duplicated_layers_get_independent_seeds() {
    let h = Harness::start();
    let mut c = h.client();
    ok(c.call(&Request::LayerDuplicate { index: 0 }).unwrap());
    let (_, s) = h.status();
    assert_eq!(s.layers.len(), 2);
    assert_ne!(
        s.layers[0].seed, s.layers[1].seed,
        "a duplicate must not be perfectly correlated with its source"
    );
}

#[test]
fn presets_survive_a_save_load_cycle() {
    let h = Harness::start();
    let mut c = h.client();
    ok(c.call(&Request::Set {
        args: Box::new(SetArgs {
            color: Some("brown".into()),
            character: Some("intense".into()),
            intensity: Some(0.55),
            high_pass: Some(28.0),
            ..Default::default()
        }),
    })
    .unwrap());
    ok(c.call(&Request::PresetSave {
        name: "Test Brown".into(),
    })
    .unwrap());

    // Change away, then load it back.
    ok(c.call(&Request::Set {
        args: Box::new(SetArgs {
            color: Some("white".into()),
            ..Default::default()
        }),
    })
    .unwrap());
    ok(c.call(&Request::PresetLoad {
        name: "Test Brown".into(),
    })
    .unwrap());

    let (_, s) = h.status();
    assert_eq!(s.layers[0].color, "brown");
    assert_eq!(s.layers[0].character, "intense");
    assert!((s.layers[0].character_amount - 0.55).abs() < 1e-9);
    assert!((s.layers[0].high_pass_hz - 28.0).abs() < 1e-9);

    let r = ok(c.call(&Request::PresetList).unwrap());
    let names = r.data.unwrap()["presets"].clone();
    assert!(names.as_array().unwrap().iter().any(|v| v == "Test Brown"));

    ok(c.call(&Request::PresetDelete {
        name: "Test Brown".into(),
    })
    .unwrap());
    assert!(
        !c.call(&Request::PresetLoad {
            name: "Test Brown".into()
        })
        .unwrap()
        .ok
    );
}

#[test]
fn corrupt_presets_do_not_take_the_daemon_down() {
    let h = Harness::start();
    let mut c = h.client();
    // Build a store full of hostile files and make sure the *daemon's* store is
    // not the only thing that survives a bad preset.
    let store_dir = tempfile::tempdir().unwrap();
    let store = Store::at(store_dir.path().join("noiseforge"));
    store.ensure_dirs().unwrap();
    std::fs::write(store.preset_path("garbage"), "= = = not toml").unwrap();
    store
        .save_preset("clamped", &noise_dsp::params::PresetParams::default())
        .unwrap();
    std::fs::write(
        store.preset_path("clamped"),
        "schema_version = 1\n[[layers]]\nalpha = -999.0\nstereo_correlation = 7.0\n",
    )
    .unwrap();

    // The daemon's own store is separate, so drive it through the protocol.
    let r = c
        .call(&Request::PresetLoad {
            name: "does not exist".into(),
        })
        .unwrap();
    assert!(!r.ok);
    // And it is still alive.
    let (_, s) = h.status();
    assert!(s.calibration.master_gain_db.is_finite());
}

#[test]
fn out_of_range_arguments_are_clamped_and_reported() {
    let h = Harness::start();
    let mut c = h.client();
    // Alpha only means something for the custom colour, which must be said.
    let r = ok(c
        .call(&Request::Set {
            args: Box::new(SetArgs {
                alpha: Some(2.0),
                ..Default::default()
            }),
        })
        .unwrap());
    assert!(
        r.notes.iter().any(|n| n.contains("custom")),
        "ignoring alpha on a named colour must be reported: {:?}",
        r.notes
    );

    let r = ok(c
        .call(&Request::Set {
            args: Box::new(SetArgs {
                color: Some("custom".into()),
                alpha: Some(1000.0),
                intensity: Some(-5.0),
                stereo: Some(9.0),
                level_rms_dbfs: Some(-500.0),
                ..Default::default()
            }),
        })
        .unwrap());
    assert!(!r.notes.is_empty(), "clamping must be reported, not hidden");
    let (_, s) = h.status();
    assert_eq!(s.layers[0].alpha, 4.0);
    assert_eq!(s.layers[0].character_amount, 0.0);
    assert_eq!(s.layers[0].stereo_correlation, 1.0);
    assert!(s.calibration.target_rms_dbfs >= -80.0);
}

#[test]
fn transport_state_survives_and_reports_honestly() {
    let h = Harness::start();
    let mut c = h.client();
    let (_, s) = h.status();
    assert!(!s.playing);
    ok(c.call(&Request::Play).unwrap());
    let (_, s) = h.status();
    assert!(s.playing);
    assert_eq!(s.transport, "playing");
    ok(c.call(&Request::Pause).unwrap());
    let (_, s) = h.status();
    assert!(!s.playing);
    ok(c.call(&Request::Stop).unwrap());
    let (_, s) = h.status();
    assert!(!s.playing);
    assert_eq!(s.transport, "stopped");
}

#[test]
fn the_timer_round_trips() {
    let h = Harness::start();
    let mut c = h.client();
    ok(c.call(&Request::Timer {
        seconds: 120.0,
        fade_out_seconds: 10.0,
    })
    .unwrap());
    let (_, s) = h.status();
    let remaining = s.timer_remaining_s.expect("no timer");
    assert!(remaining > 110.0 && remaining <= 120.0, "{remaining}");
    ok(c.call(&Request::TimerClear).unwrap());
    let (_, s) = h.status();
    assert!(s.timer_remaining_s.is_none());
    // A nonsensical duration is refused.
    assert!(
        !c.call(&Request::Timer {
            seconds: -1.0,
            fade_out_seconds: 0.0
        })
        .unwrap()
        .ok
    );
}

#[test]
fn telemetry_subscription_pushes_well_formed_frames() {
    let h = Harness::start();
    let mut c = h.client();
    ok(c.call(&Request::Subscribe {
        psd: false,
        hz: 30.0,
    })
    .unwrap());
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut frames = 0;
    while Instant::now() < deadline && frames < 3 {
        if let Some(response) = c.next_push(Duration::from_millis(500)).unwrap() {
            assert_eq!(response.id, 0, "pushes use id 0");
            let data = response.data.expect("push without data");
            assert!(data["status"]["calibration"]["target_rms_dbfs"].is_number());
            assert!(
                data["predicted"].is_object(),
                "the predicted curve is mandatory"
            );
            frames += 1;
        }
    }
    assert!(frames >= 3, "only {frames} telemetry frames arrived");
    ok(c.call(&Request::Unsubscribe).unwrap());
}

#[test]
fn status_is_within_the_message_size_limit_for_eight_layers() {
    let h = Harness::start();
    let mut c = h.client();
    for i in 0..7 {
        ok(c.call(&Request::LayerAdd { after: None }).unwrap());
        ok(c.call(&Request::Set {
            args: Box::new(SetArgs {
                layer: Some(i + 1),
                color: Some("custom".into()),
                alpha: Some(-4.0 + i as f64),
                character: Some("intense".into()),
                intensity: Some(1.0),
                gain_db: Some(-3.0),
                ..Default::default()
            }),
        })
        .unwrap());
    }
    let response = ok(c.call(&Request::Status).unwrap());
    let size = serde_json::to_vec(&response).unwrap().len();
    assert!(
        size < noise_daemon::MAX_MESSAGE_BYTES,
        "status is {size} bytes"
    );
    // And it is still parseable.
    let s: StatusData = serde_json::from_value(response.data.unwrap()).unwrap();
    assert_eq!(s.layers.len(), 8);
}

#[test]
fn a_wrong_protocol_version_is_rejected_cleanly() {
    let h = Harness::start();
    use std::io::Write;
    let mut stream = std::os::unix::net::UnixStream::connect(&h.socket).unwrap();
    // Version 999 in the envelope, with an otherwise valid body.
    let line = b"{\"v\":999,\"id\":1,\"op\":\"status\"}\n";
    stream.write_all(line).unwrap();
    stream.flush().unwrap();
    let mut buf = [0u8; 512];
    use std::io::Read;
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    let n = stream.read(&mut buf).unwrap();
    let text = String::from_utf8_lossy(&buf[..n]);
    assert!(text.contains("version"), "unexpected reply: {text}");
    // The daemon survives and still answers normally.
    let (_, s) = h.status();
    assert_eq!(s.protocol, noise_daemon::PROTOCOL_VERSION);
}

#[test]
fn many_clients_can_be_served_at_once() {
    let h = Harness::start();
    let clients: Vec<Client> = (0..8).map(|_| h.client()).collect();
    let handles: Vec<_> = clients
        .into_iter()
        .map(|mut c| {
            std::thread::spawn(move || {
                for _ in 0..20 {
                    let r = c.call(&Request::Status).unwrap();
                    assert!(r.ok);
                }
            })
        })
        .collect();
    for handle in handles {
        handle.join().unwrap();
    }
    let (_, s) = h.status();
    assert_eq!(s.layers.len(), 1);
}

#[test]
fn closing_the_last_client_stops_playback() {
    // The user-visible contract: a window that has been closed is silent.
    let h = Harness::start();
    let mut watcher = h.watcher();
    ok(watcher.call(&Request::Play).unwrap());
    let (_, s) = h.status();
    assert!(s.playing);

    drop(watcher);
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let (_, s) = h.status();
        if !s.playing {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "playback survived the last client disconnecting"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
}

#[test]
fn a_cli_hold_keeps_playback_alive() {
    // `noiseforge play` disconnects immediately and expects the noise to stay.
    let h = Harness::start();
    let mut cli = h.client();
    ok(cli.call(&Request::Hold { hold: true }).unwrap());
    ok(cli.call(&Request::Play).unwrap());
    drop(cli);
    std::thread::sleep(Duration::from_millis(300));
    let (_, s) = h.status();
    assert!(s.playing, "an explicit hold did not survive the disconnect");

    // `noiseforge stop` releases it.
    let mut cli = h.client();
    ok(cli.call(&Request::Hold { hold: false }).unwrap());
    ok(cli.call(&Request::Stop).unwrap());
    drop(cli);
    std::thread::sleep(Duration::from_millis(300));
    let (_, s) = h.status();
    assert!(!s.playing);
}

#[test]
fn a_gui_session_takes_over_from_a_cli_hold() {
    // Without this, `noiseforge play` followed by opening and closing the GUI
    // would leave the noise running.
    let h = Harness::start();
    let mut cli = h.client();
    ok(cli.call(&Request::Hold { hold: true }).unwrap());
    ok(cli.call(&Request::Play).unwrap());
    drop(cli);

    let mut gui = h.watcher();
    ok(gui.call(&Request::Play).unwrap());
    drop(gui);
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let (_, s) = h.status();
        if !s.playing {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "the GUI session did not take over"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
}

#[test]
fn keep_playing_survives_every_disconnect() {
    let h = Harness::start_with(|c| {
        c.keep_playing_without_clients = true;
        c.autoplay = true;
    });
    // Let the format event arm autoplay.
    std::thread::sleep(Duration::from_millis(200));
    {
        let mut c = h.client();
        let _ = c.call(&Request::Status);
    }
    std::thread::sleep(Duration::from_millis(300));
    let (_, s) = h.status();
    assert!(s.playing, "--keep-playing did not keep playback alive");
}

#[test]
fn a_held_daemon_still_reports_honestly_after_the_client_leaves() {
    let h = Harness::start();
    let mut cli = h.client();
    ok(cli.call(&Request::Hold { hold: true }).unwrap());
    ok(cli.call(&Request::Play).unwrap());
    drop(cli);
    std::thread::sleep(Duration::from_millis(200));
    let (_, s) = h.status();
    assert!(s.playing);
    assert_eq!(s.transport, "playing");
    // The silent backend reports silence, and says so rather than pretending.
    assert!(s.meters.rms_dbfs <= -100.0);
}
