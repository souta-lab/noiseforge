//! Native PipeWire backend for NoiseForge (DESIGN.md section 4.2).
//!
//! The crate is deliberately thin: it owns the stream, the format negotiation,
//! the realtime callback and the device registry, and it knows nothing about
//! presets, IPC or the UI.
#![forbid(unsafe_code)]

pub mod devices;
pub mod format;
pub mod lifecycle;
pub mod stream;

pub use devices::{SinkInfo, default_sink, list_sinks};
pub use format::{NegotiatedFormat, SampleFormat, channel_layout_name};
pub use lifecycle::{LinkState, Telemetry};
pub use stream::{Output, RatePolicy, StreamConfig, StreamControl, StreamEvent, spawn};
