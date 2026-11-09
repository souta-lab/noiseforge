//! Offline rendering, spectral analysis and file output for NoiseForge.
//!
//! Everything in this crate is explicitly **not** realtime safe: it allocates,
//! uses FFTs and touches the filesystem. DESIGN.md section 4.5 keeps the FFT out
//! of the generation path, so the realtime engine never links against it.
#![forbid(unsafe_code)]

pub mod analysis;
pub mod render;
pub mod wav;

pub use analysis::{
    Band, Psd, SlopeFit, WelchConfig, Window, deinterleave_channel, fit_slope,
    fractional_octave_bands, log_resample, measure_slope, welch_psd,
};
pub use render::{RenderRequest, RenderResult, render};
pub use wav::{OutputFormat, WriteError, write_wav};
