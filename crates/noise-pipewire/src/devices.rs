//! Sink enumeration and default-sink discovery (DESIGN.md section 4.2).
//!
//! Temporary node ids are never stored as permanent identifiers (DESIGN.md
//! section 7), so this module reports a stable-ish `name` alongside the current
//! `id` and the caller decides which one to persist.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use pipewire as pw;

/// One selectable output node.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SinkInfo {
    /// Current PipeWire node id. Temporary; never persist it.
    pub id: u32,
    /// Stable node name such as `alsa_output.pci-0000_00_1f.3.analog-stereo`.
    pub name: String,
    /// Human readable description.
    pub description: String,
    /// Driver nickname, when the server reports one.
    pub nick: Option<String>,
    /// Whether WirePlumber currently marks this as the default sink.
    pub is_default: bool,
}

/// Errors from the enumeration round trip.
#[derive(Debug, thiserror::Error)]
pub enum DeviceError {
    #[error("pipewire error: {0}")]
    PipeWire(#[from] pw::Error),
    #[error("timed out waiting for the registry")]
    Timeout,
    #[error("the registry subscriber could not be locked")]
    Poisoned,
}

/// Walk the registry and return every node that can act as an audio sink.
///
/// Uses a short main-loop round trip: connect, ask for the core to sync, collect
/// `global` events until the sync completes, then quit.
pub fn list_sinks() -> Result<Vec<SinkInfo>, DeviceError> {
    list_with_timeout(Duration::from_millis(750))
}

/// Like [`list_sinks`] but with an explicit round trip budget.
pub fn list_with_timeout(timeout: Duration) -> Result<Vec<SinkInfo>, DeviceError> {
    pw::init();
    let mainloop = pw::main_loop::MainLoopRc::new(None)?;
    let context = pw::context::ContextRc::new(&mainloop, None)?;
    let core = context.connect_rc(None)?;
    let registry = core.get_registry_rc()?;

    let sinks: Arc<Mutex<Vec<SinkInfo>>> = Arc::new(Mutex::new(Vec::new()));
    let done = Arc::new(Mutex::new(false));

    let default_name: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));

    let _global = registry
        .add_listener_local()
        .global({
            let sinks = sinks.clone();
            let default_name = default_name.clone();
            move |global| {
                // Sinks are Nodes, not Devices: DESIGN.md routes to a node.
                let Some(props) = global.props else { return };
                let media_class = props.get("media.class").unwrap_or_default();
                let is_sink = media_class == "Audio/Sink" || media_class == "Audio/Duplex";
                let name = props.get("node.name").unwrap_or_default().to_string();
                if media_class == "Audio/Sink" || media_class == "Audio/Duplex" {
                    let description = props
                        .get("node.description")
                        .or_else(|| props.get("node.nick"))
                        .unwrap_or(&name)
                        .to_string();
                    let nick = props.get("node.nick").map(|s| s.to_string());
                    if let Ok(mut v) = sinks.lock() {
                        v.push(SinkInfo {
                            id: global.id,
                            name: name.clone(),
                            description,
                            nick,
                            is_default: false,
                        });
                    }
                }
                if !is_sink {
                    return;
                }
                // WirePlumber stamps the default node with this property.
                if (props.get("default.audio.sink").is_some()
                    || props.get("node.default").is_some())
                    && let Ok(mut d) = default_name.lock()
                {
                    *d = Some(name);
                }
            }
        })
        .global_remove(|_| {})
        .register();

    // Ask the server to tell us when the initial dump is done.
    let pending = core.sync(0)?;
    let _done_listener = core
        .add_listener_local()
        .done({
            let done = done.clone();
            let mainloop = mainloop.clone();
            move |_id, seq| {
                if seq == pending {
                    if let Ok(mut d) = done.lock() {
                        *d = true;
                    }
                    mainloop.quit();
                }
            }
        })
        .register();

    // Belt and braces: a timer stops the loop if `done` never arrives.
    let timer = mainloop.loop_().add_timer({
        let mainloop = mainloop.clone();
        move |_| mainloop.quit()
    });
    timer.update_timer(Some(timeout), None);

    mainloop.run();

    let elapsed_ok = *done.lock().map_err(|_| DeviceError::Poisoned)?;
    let mut out = sinks.lock().map_err(|_| DeviceError::Poisoned)?.clone();
    if out.is_empty() && !elapsed_ok {
        return Err(DeviceError::Timeout);
    }

    // Determine the default: prefer the WirePlumber property, else the node the
    // server labels `node.default`, else the first sink.
    let default = default_name
        .lock()
        .map_err(|_| DeviceError::Poisoned)?
        .clone();
    let default_idx = default
        .as_ref()
        .and_then(|n| out.iter().position(|s| &s.name == n))
        .unwrap_or(0);
    if let Some(s) = out.get_mut(default_idx) {
        s.is_default = true;
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

/// The currently selected default sink, if any.
pub fn default_sink() -> Result<Option<SinkInfo>, DeviceError> {
    Ok(list_sinks()?.into_iter().find(|s| s.is_default))
}

/// Node id for a sink name, as accepted by `StreamConfig::target_object`.
pub fn node_id_for_name(name: &str) -> Result<Option<u32>, DeviceError> {
    Ok(list_sinks()?
        .into_iter()
        .find(|s| s.name == name)
        .map(|s| s.id))
}

/// Convenience used by the `device` subcommands.
pub fn describe(sinks: &[SinkInfo]) -> String {
    let mut out = String::new();
    for s in sinks {
        out.push_str(&format!(
            "{} {}  {}\n",
            if s.is_default { "*" } else { " " },
            s.id,
            s.description
        ));
        out.push_str(&format!("      name: {}\n", s.name));
        if let Some(n) = &s.nick {
            out.push_str(&format!("      nick: {n}\n"));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn describe_marks_the_default() {
        let sinks = vec![
            SinkInfo {
                id: 1,
                name: "a".into(),
                description: "A".into(),
                nick: None,
                is_default: true,
            },
            SinkInfo {
                id: 2,
                name: "b".into(),
                description: "B".into(),
                nick: Some("bee".into()),
                is_default: false,
            },
        ];
        let text = describe(&sinks);
        assert!(text.contains("* 1"));
        assert!(text.contains("name: b"));
        assert!(text.contains("nick: bee"));
    }
}
