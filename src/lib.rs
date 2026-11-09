//! NoiseForge umbrella crate.
//!
//! Exists so that the workspace level `tests/` and `benches/` directories from
//! DESIGN.md section 8 belong to a package. It only re-exports the real crates.
#![forbid(unsafe_code)]

pub use noise_daemon as daemon;
pub use noise_dsp as dsp;
pub use noise_offline as offline;
pub use noise_pipewire as pipewire_backend;
