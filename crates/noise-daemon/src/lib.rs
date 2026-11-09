//! The NoiseForge daemon, its control protocol and its client library.
//!
//! `noiseforged` keeps playing when the GUI exits; the GUI and the CLI are thin
//! clients of the same versioned protocol (DESIGN.md section 6).
#![forbid(unsafe_code)]

pub mod backend;
pub mod ipc;
pub mod presets;
pub mod protocol;
pub mod service;

pub use backend::{AudioBackend, BackendError, BackendEvent, Meters, NullBackend, OutputFormat};
pub use ipc::{Client, Connection, IpcError, Server};
pub use presets::{Store, StoreError};
pub use protocol::{
    BackendKind, CalibrationStatus, Curve, LayerStatus, MAX_MESSAGE_BYTES, MeterStatus,
    PROTOCOL_VERSION, RateMode, Request, Response, SetArgs, StatusData, TelemetryFrame, apply_set,
    default_socket_path,
};
pub use service::{DaemonConfig, Service, ServiceError};
