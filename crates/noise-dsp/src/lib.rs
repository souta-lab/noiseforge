//! NoiseForge DSP core.
//!
//! A standalone crate: no PipeWire, no GUI, no OS specific code. It owns the
//! mathematical definition of the noise colours, the offline coefficient
//! designer, the character filters, stereo correlation, the mixer and the
//! realtime safe render engine.
//!
//! See `DESIGN.md` sections 1, 2, 3 and 4.1 for the specification this
//! implements, and section 11 for the implementation rules.
#![forbid(unsafe_code)]

pub mod characters;
pub mod colors;
pub mod compile;
pub mod designer;
pub mod engine;
pub mod envelope;
pub mod metrics;
pub mod mixer;
pub mod params;
pub mod rng;
pub mod sos;
pub mod stereo;

pub use characters::{Character, CharacterDesign};
pub use colors::Color;
pub use compile::{CompiledGraph, CompiledLayer, compile, compile_with};
pub use designer::{
    ColorDesign, DesignError, DesignOptions, DesignReport, design_color, design_color_with,
};
pub use engine::{Command, Engine, LayerFilters, PlayState};
pub use envelope::Ramp;
pub use params::{MAX_LAYERS, PresetParams, SCHEMA_VERSION};
pub use sos::{BiquadCoeffs, SosCoeffs};
pub use stereo::StereoMatrix;

/// Version of the DSP definitions. Bump when a change would alter the PCM
/// produced for the same seed, settings and sample rate, because DESIGN.md
/// section 1.3 promises reproducibility for a given DSP version.
pub const DSP_VERSION: u32 = 1;
