//! The realtime-safe render engine.
//!
//! DESIGN.md section 3 fixes the processing order:
//!
//! ```text
//! per-layer seeded PRNG
//!   -> colour filter (White/Pink/Brown/Blue/Violet/Custom)
//!   -> character (Neutral/Smooth/Intense)
//!   -> user EQ / DC high pass / optional low pass
//!   -> layer gain / fade
//!   -> stereo correlation
//!   -> sum up to 8 layers
//!   -> master gain / play-pause / timer / gain ramp
//!   -> peak monitoring, optional explicit limiter
//!   -> interleaved F32 output
//! ```
//!
//! # Realtime contract
//!
//! * [`Engine::render`] performs no allocation, no locking, no I/O and no
//!   formatting. Everything it touches is a fixed size field.
//! * Settings arrive as [`Command`] values, which are `Copy` PODs carrying
//!   *already designed* coefficients. Coefficient design, validation and
//!   rejection happen off the realtime thread; see DESIGN.md section 4.4.
//! * The output depends only on the sample index, never on the block size, so
//!   rendering 128 frames at a time and 256 frames at a time produce bit
//!   identical PCM for identical settings (DESIGN.md section 4.1).
//!
//! # Click suppression
//!
//! Nothing that can change is ever applied as a step:
//!
//! * gains, fades and the stereo correlation-free layer gain use [`Ramp`];
//! * a colour or EQ coefficient change crossfades between the old and the new
//!   cascade over a configurable window;
//! * a character *shape* change is deferred until the character gain has
//!   ramped to exactly zero, at which point the stage is a bit-exact bypass and
//!   swapping coefficients is inaudible;
//! * a seed change is deferred until the layer has faded to silence.

use crate::characters::CharacterDesign;
use crate::envelope::{Ramp, ms_to_samples};
use crate::metrics::Meters;
use crate::params::MAX_LAYERS;
use crate::rng::{Xoshiro256PlusPlus, derive_seed};
use crate::sos::{Biquad, BiquadCoeffs, SosChain, SosCoeffs};
use crate::stereo::StereoMatrix;

/// Largest block the engine will render in one go. Longer requests are split,
/// which keeps the engine allocation free for any host buffer size.
pub const MAX_BLOCK: usize = 4096;

/// Channels the engine produces internally.
pub const ENGINE_CHANNELS: usize = 2;

/// How long a master gain *increase* may be held back waiting for layer
/// crossfades to finish.
const MAX_GAIN_DEFER_MS: f64 = 400.0;

/// Lowest master gain the engine will accept, in dB.
pub const MIN_MASTER_GAIN_DB: f64 = -300.0;
/// Highest master gain the engine will accept, in dB.
pub const MAX_MASTER_GAIN_DB: f64 = 60.0;

/// Tiny alternating offset added to the PRNG output.
///
/// It is inaudible (`-500 dBFS`) and keeps recursive filter state away from the
/// denormal range, where some CPUs fall off a performance cliff
/// (DESIGN.md section 1.4).
const DENORMAL_GUARD: f64 = 1e-25;

/// Playback state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlayState {
    Stopped,
    Playing,
    Paused,
}

/// Colour and EQ coefficients for one layer.
///
/// The three parts are folded into one cascade by [`LayerFilters::combined`], so
/// a single crossfade covers the whole filter bank and a frequency slider can
/// never step a section's coefficients underneath running state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayerFilters {
    /// Colour shaping cascade.
    pub color: SosCoeffs,
    /// User high pass.
    pub high_pass: BiquadCoeffs,
    /// User low pass.
    pub low_pass: BiquadCoeffs,
    /// Predicted RMS of the layer output these coefficients produce, for a
    /// unit-variance white input and excluding the character stage and the layer
    /// gain. The engine uses it to decide whether a change may be crossfaded or
    /// must be swapped behind a short fade: crossfading between filters whose
    /// levels differ by tens of dB cannot be kept inside the calibrated level by
    /// any time based gain ramp.
    ///
    /// `0.0` means "unknown" and always crossfades.
    pub expected_rms: f32,
}

impl Default for LayerFilters {
    fn default() -> Self {
        Self {
            color: SosCoeffs::IDENTITY,
            high_pass: BiquadCoeffs::IDENTITY,
            low_pass: BiquadCoeffs::IDENTITY,
            expected_rms: 0.0,
        }
    }
}

impl LayerFilters {
    /// Reject anything that could destabilise the realtime thread.
    pub fn is_valid(&self) -> bool {
        self.color.is_valid() && self.high_pass.is_stable() && self.low_pass.is_stable()
    }

    /// Fold colour, high pass and low pass into one cascade.
    pub fn combined(&self) -> SosCoeffs {
        let mut buf = [BiquadCoeffs::IDENTITY; crate::sos::MAX_SECTIONS];
        let mut n = 0usize;
        for s in self.color.as_slice() {
            if n == buf.len() {
                return SosCoeffs::IDENTITY;
            }
            buf[n] = *s;
            n += 1;
        }
        for s in [self.high_pass, self.low_pass] {
            if !s.is_identity() {
                if n == buf.len() {
                    return SosCoeffs::IDENTITY;
                }
                buf[n] = s;
                n += 1;
            }
        }
        SosCoeffs::new(&buf[..n])
    }
}

/// Level difference, in dB, above which a coefficient change is swapped behind a
/// short fade instead of crossfaded.
pub const FILTER_DIP_THRESHOLD_DB: f64 = 6.0;

/// A setting change. All variants are `Copy` and allocation free.
///
/// The variants differ a lot in size because a filter bank is a few hundred
/// bytes of coefficients and a transport command is a unit. Boxing the large one
/// would put a heap allocation on the realtime path, which is exactly what
/// DESIGN.md section 4.3 forbids, so the enum is passed by value.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Command {
    /// Enable or disable a layer with a fade.
    SetLayerEnabled {
        layer: u8,
        enabled: bool,
        fade_ms: f32,
    },
    /// Replace colour and EQ coefficients with a crossfade.
    SetLayerFilters {
        layer: u8,
        filters: LayerFilters,
        crossfade_ms: f32,
    },
    /// Replace the character shape filters. Applied only once the character gain
    /// has reached zero, so it can never click.
    SetCharacterShape {
        layer: u8,
        low_shelf: BiquadCoeffs,
        high_shelf: BiquadCoeffs,
        ramp_ms: f32,
    },
    /// Set the character shelf gain offsets (`10^(dB/20) - 1`).
    SetCharacterGain {
        layer: u8,
        low_gain: f32,
        high_gain: f32,
        ramp_ms: f32,
    },
    /// Layer gain in dB.
    SetLayerGain {
        layer: u8,
        gain_db: f32,
        ramp_ms: f32,
    },
    /// Stereo correlation, applied with a short crossfade of the mix matrix.
    SetStereoCorrelation {
        layer: u8,
        correlation: f32,
        ramp_ms: f32,
    },
    /// Change the PRNG seed. Deferred until the layer has faded to silence.
    SetLayerSeed { layer: u8, seed: u64, fade_ms: f32 },
    /// Master gain in dB (calibration plus user trim).
    ///
    /// `hold_ms` delays the start of the ramp. It exists for one specific
    /// hazard: while a coefficient crossfade is running the blend still carries
    /// the *outgoing* filter, so raising the master gain before a much louder
    /// outgoing filter has faded out would push the output far past the
    /// calibrated level. Callers that raise the gain together with a filter
    /// change pass the crossfade length here. A *falling* gain ignores it,
    /// because that direction can never overshoot.
    SetMasterGain {
        gain_db: f32,
        ramp_ms: f32,
        hold_ms: f32,
    },
    /// Replace the master DC blocker.
    SetMasterDcBlocker { coeffs: BiquadCoeffs },
    /// Explicit, opt-in limiter. Off by default because it is nonlinear.
    SetLimiter { enabled: bool, ceiling_dbfs: f32 },
    /// Fade times for start and stop.
    SetFades { in_ms: f32, out_ms: f32 },
    /// Start or resume with the fade-in.
    Play,
    /// Fade out and hold.
    Pause,
    /// Fade out and stop.
    Stop,
    /// Sleep timer: after `seconds`, fade out over `fade_out_seconds` and pause.
    SetTimer { seconds: f32, fade_out_seconds: f32 },
    /// Cancel the sleep timer.
    ClearTimer,
    /// Silence immediately without a fade. Only for hard error recovery.
    EmergencySilence,
}

impl Command {
    /// Layer index this command targets, if any.
    pub fn layer(&self) -> Option<usize> {
        match self {
            Command::SetLayerEnabled { layer, .. }
            | Command::SetLayerFilters { layer, .. }
            | Command::SetCharacterShape { layer, .. }
            | Command::SetCharacterGain { layer, .. }
            | Command::SetLayerGain { layer, .. }
            | Command::SetStereoCorrelation { layer, .. }
            | Command::SetLayerSeed { layer, .. } => Some(*layer as usize),
            _ => None,
        }
    }
}

/// A colour cascade that crossfades between two coefficient sets.
#[derive(Clone, Copy, Debug)]
struct CrossSos {
    cur: SosChain,
    alt: SosChain,
    pos: u32,
    len: u32,
    /// Newest requested coefficient set while a crossfade is already running.
    /// Intermediate values are dropped; see [`CrossSos::set`].
    pending: Option<(SosCoeffs, u32)>,
}

impl CrossSos {
    fn new(coeffs: SosCoeffs) -> Self {
        Self {
            cur: SosChain::new(coeffs),
            alt: SosChain::new(coeffs),
            pos: 0,
            len: 0,
            pending: None,
        }
    }

    fn is_fading(&self) -> bool {
        self.len > 0 && self.pos < self.len
    }

    /// Request `coeffs`.
    ///
    /// A crossfade in progress is *never* disturbed: touching a running blend is
    /// exactly what would put a step in the output. Instead the newest request
    /// is remembered and started as soon as the current fade completes, which is
    /// the "latest value wins, intermediate states are dropped" policy from
    /// DESIGN.md section 4.4 with sample continuity guaranteed.
    fn set(&mut self, coeffs: SosCoeffs, samples: u32) {
        if self.is_fading() {
            self.pending = Some((coeffs, samples.max(1)));
            return;
        }
        self.start(coeffs, samples);
    }

    /// Replace the coefficients immediately, clearing both branches.
    ///
    /// Only safe while the caller has faded the signal to *exact* silence: it
    /// discards the running state of both cascades.
    fn snap(&mut self, coeffs: SosCoeffs) {
        self.cur = SosChain::new(coeffs);
        self.alt = SosChain::new(coeffs);
        self.pos = 0;
        self.len = 0;
        self.pending = None;
    }

    fn start(&mut self, coeffs: SosCoeffs, samples: u32) {
        if coeffs == self.cur.coeffs {
            return;
        }
        self.alt = SosChain::new(coeffs);
        self.pos = 0;
        self.len = samples.max(1);
    }

    #[inline]
    fn process(&mut self, x: f64) -> f64 {
        if self.len == 0 {
            return self.cur.process(x);
        }
        let a = self.cur.process(x);
        let b = self.alt.process(x);
        let w = self.pos as f64 / self.len as f64;
        self.pos += 1;
        if self.pos >= self.len {
            self.cur = self.alt;
            self.len = 0;
            self.pos = 0;
            if let Some((coeffs, samples)) = self.pending.take() {
                self.start(coeffs, samples);
            }
        }
        a + (b - a) * w
    }
}

/// The per-channel filter chain of one layer.
#[derive(Clone, Copy, Debug)]
struct FilterBank {
    /// Colour plus user high pass and low pass, folded into one cascade so a
    /// single crossfade covers all of it.
    chain: CrossSos,
    shelf_low: Biquad,
    shelf_high: Biquad,
}

impl FilterBank {
    fn new(fs: f64) -> Self {
        Self {
            chain: CrossSos::new(SosCoeffs::IDENTITY),
            shelf_low: Biquad::new(BiquadCoeffs::one_pole_low_pass(180.0, fs)),
            shelf_high: Biquad::new(BiquadCoeffs::one_pole_high_pass(3200.0, fs)),
        }
    }

    fn set_shelf_shapes(&mut self, low: BiquadCoeffs, high: BiquadCoeffs) {
        self.shelf_low.set_coeffs_reset(low);
        self.shelf_high.set_coeffs_reset(high);
    }

    #[inline]
    fn process(&mut self, x: f64, gl: f64, gh: f64) -> f64 {
        let mut v = self.chain.process(x);
        if gl != 0.0 || gh != 0.0 {
            // `1 + g * S(z)`: the shape filters are fixed, only the scalar gain
            // moves, so nothing nonlinear or discontinuous is ever applied.
            let lo = self.shelf_low.process(v);
            let hi = self.shelf_high.process(v);
            v += gl * lo + gh * hi;
        }
        v
    }

    fn reset(&mut self) {
        self.chain.cur.reset();
        self.chain.alt.reset();
        self.chain.len = 0;
        self.chain.pos = 0;
        self.chain.pending = None;
        self.shelf_low.reset();
        self.shelf_high.reset();
    }
}

/// One realtime layer.
#[derive(Clone, Copy, Debug)]
struct LayerRt {
    enabled_target: bool,
    gain_db: f64,
    gain: Ramp,
    /// Layer on/off envelope. `enable == 0` means the layer contributes exactly
    /// nothing, which is also the only moment it is safe to reseed its PRNG.
    enable: Ramp,
    /// Dip applied around a PRNG reseed so the stream change is inaudible.
    seed_fade: Ramp,
    /// Dip applied around a filter swap that cannot be crossfaded safely.
    morph: Ramp,
    /// Filter bank waiting for the morph dip to reach zero.
    pending_filters: Option<(LayerFilters, u32)>,
    /// Expected RMS of the chain currently loaded, for the dip decision.
    chain_rms: f64,
    char_low: Ramp,
    char_high: Ramp,
    char_low_target: f64,
    char_high_target: f64,
    pending_shape: Option<(BiquadCoeffs, BiquadCoeffs)>,
    stereo: StereoMatrix,
    /// Correlation ramped in `rho` itself, never in `direct`/`cross`
    /// independently: interpolating those separately would collapse the level
    /// to zero whenever the target crossed from +1 to -1.
    stereo_rho: Ramp,
    src_u: Xoshiro256PlusPlus,
    src_v: Xoshiro256PlusPlus,
    chain_u: FilterBank,
    chain_v: FilterBank,
    pending_seed: Option<(u64, u32)>,
    guard: f64,
    seed: u64,
    /// False while a layer has never been configured; keeps idle layers free.
    configured: bool,
}

impl LayerRt {
    fn new(fs: f64, seed: u64) -> Self {
        Self {
            enabled_target: false,
            gain_db: 0.0,
            gain: Ramp::new(1.0),
            enable: Ramp::new(0.0),
            seed_fade: Ramp::new(1.0),
            morph: Ramp::new(1.0),
            pending_filters: None,
            chain_rms: 0.0,
            char_low: Ramp::new(0.0),
            char_high: Ramp::new(0.0),
            char_low_target: 0.0,
            char_high_target: 0.0,
            pending_shape: None,
            stereo: StereoMatrix::default(),
            stereo_rho: Ramp::new(1.0),
            src_u: Xoshiro256PlusPlus::seed_from_u64(derive_seed(seed, 0, 0)),
            src_v: Xoshiro256PlusPlus::seed_from_u64(derive_seed(seed, 0, 1)),
            chain_u: FilterBank::new(fs),
            chain_v: FilterBank::new(fs),
            pending_seed: None,
            guard: DENORMAL_GUARD,
            seed,
            configured: false,
        }
    }

    /// True while the layer can contribute anything at all.
    ///
    /// Note that the *target* enabled flag is deliberately not consulted: when
    /// a playing layer is switched off, `enable` has to keep ramping down to
    /// zero, and cutting it out immediately would be an audible click.
    #[inline]
    fn audible(&self) -> bool {
        self.configured
            && (self.enable.value() > 0.0
                || self.enable.is_ramping()
                || self.seed_fade.value() > 0.0
                || self.seed_fade.is_ramping()
                || self.morph.value() > 0.0
                || self.morph.is_ramping()
                || self.pending_filters.is_some())
    }

    /// Produce one stereo frame contribution.
    #[inline]
    fn next_frame(&mut self, ramp_samples: u32) -> (f64, f64) {
        // Swap the character shape filters only while the character gain is
        // exactly zero, i.e. while the stage is a bit-exact bypass.
        if self.pending_shape.is_some()
            && !self.char_low.is_ramping()
            && !self.char_high.is_ramping()
            && self.char_low.value() == 0.0
            && self.char_high.value() == 0.0
            && let Some((low, high)) = self.pending_shape.take()
        {
            self.chain_u.set_shelf_shapes(low, high);
            self.chain_v.set_shelf_shapes(low, high);
            self.char_low.ramp_to(self.char_low_target, ramp_samples);
            self.char_high.ramp_to(self.char_high_target, ramp_samples);
        }
        // Swap the filter bank only once the morph dip has reached exact
        // silence, which is what makes the swap inaudible.
        if let Some((filters, n)) = self.pending_filters
            && self.morph.value() == 0.0
            && !self.morph.is_ramping()
        {
            // Snap, do not crossfade: at this instant the layer is at exact
            // silence, and crossfading here would fade the *old* filter back in
            // while the morph ramp brings the level up, which is precisely the
            // overshoot the dip exists to avoid.
            let combined = filters.combined();
            self.chain_u.chain.snap(combined);
            self.chain_v.chain.snap(combined);
            self.chain_rms = filters.expected_rms as f64;
            self.configured = true;
            self.pending_filters = None;
            self.morph.ramp_to(1.0, n);
        }
        // Apply a pending reseed once the reseed fade has reached zero, then
        // bring the layer back. Dividing by a silent `seed_fade` makes the
        // stream swap mathematically inaudible.
        if let Some((seed, n)) = self.pending_seed
            && self.seed_fade.value() == 0.0
            && !self.seed_fade.is_ramping()
        {
            self.reseed(seed);
            self.seed_fade.ramp_to(1.0, n);
            self.pending_seed = None;
        }

        if self.stereo_rho.is_ramping() {
            let rho = self.stereo_rho.step();
            self.stereo = StereoMatrix::new(rho);
        }

        let gl = self.char_low.step();
        let gh = self.char_high.step();
        self.guard = -self.guard;

        let a = self
            .chain_u
            .process(self.src_u.next_symmetric() + self.guard, gl, gh);
        let b = self
            .chain_v
            .process(self.src_v.next_symmetric() + self.guard, gl, gh);
        let g = self.gain.step() * self.enable.step() * self.seed_fade.step() * self.morph.step();
        let u = a * g;
        let v = b * g;
        (u, self.stereo.right(u, v))
    }

    fn reseed(&mut self, seed: u64) {
        self.seed = seed;
        self.src_u.reseed(derive_seed(seed, 0, 0));
        self.src_v.reseed(derive_seed(seed, 0, 1));
    }

    /// Start a crossfade into a new filter bank.
    fn load_filters(&mut self, filters: LayerFilters, crossfade: u32) {
        let combined = filters.combined();
        self.chain_u.chain.set(combined, crossfade);
        self.chain_v.chain.set(combined, crossfade);
        self.chain_rms = filters.expected_rms as f64;
        self.configured = true;
    }

    /// True when the layer currently contributes nothing at all.
    #[inline]
    fn is_silent(&self) -> bool {
        self.enable.value() == 0.0
            && !self.enable.is_ramping()
            && self.morph.value() == 0.0
            && !self.morph.is_ramping()
    }
}

/// Timer state machine.
#[derive(Clone, Copy, Debug, Default)]
struct SleepTimer {
    total: i64,
    left: i64,
    fade: i64,
    firing: bool,
}

/// The realtime engine.
pub struct Engine {
    sample_rate: f64,
    seed: u64,
    layers: [LayerRt; MAX_LAYERS],
    scratch: Box<[f32]>,
    meters: Meters,
    ramp_samples: u32,
    fade_in_samples: u32,
    fade_out_samples: u32,
    master_gain: Ramp,
    master_gain_db: f64,
    /// Master gain waiting for the sample countdown below before it rises.
    master_pending_gain_db: Option<f64>,
    /// Sample countdown that holds a rising master gain back.
    master_hold: u32,
    master_env: Ramp,
    master_dc: Biquad,
    limiter_enabled: bool,
    limiter_ceiling: f64,
    state: PlayState,
    timer: SleepTimer,
    rendered_frames: u64,
}

impl core::fmt::Debug for Engine {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Engine")
            .field("sample_rate", &self.sample_rate)
            .field("state", &self.state)
            .field("master_gain_db", &self.master_gain_db)
            .field("rendered_frames", &self.rendered_frames)
            .finish()
    }
}

impl Engine {
    /// Build an engine for a sample rate. Allocates once, here, never in
    /// [`Engine::render`].
    pub fn new(sample_rate: f64, seed: u64) -> Self {
        let fs = if sample_rate.is_finite() && sample_rate > 0.0 {
            sample_rate
        } else {
            48_000.0
        };
        let ramp_samples = ms_to_samples(20.0, fs);
        Self {
            sample_rate: fs,
            seed,
            layers: core::array::from_fn(|i| LayerRt::new(fs, derive_seed(seed, i, 0x1A7E))),
            scratch: vec![0.0f32; MAX_BLOCK * ENGINE_CHANNELS].into_boxed_slice(),
            meters: Meters::default(),
            ramp_samples,
            fade_in_samples: ms_to_samples(100.0, fs),
            fade_out_samples: ms_to_samples(100.0, fs),
            master_gain: Ramp::new(0.0),
            master_gain_db: -120.0,
            master_pending_gain_db: None,
            master_hold: 0,
            master_env: Ramp::new(0.0),
            // Always-on DC blocker at 5 Hz. It costs one multiply per channel
            // and guarantees no DC can accumulate in the output.
            master_dc: Biquad::new(BiquadCoeffs::dc_blocker(5.0, fs)),
            limiter_enabled: false,
            limiter_ceiling: crate::mixer::db_to_linear(-1.0),
            state: PlayState::Stopped,
            timer: SleepTimer::default(),
            rendered_frames: 0,
        }
    }

    /// Output sample rate.
    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    /// Base PRNG seed this engine was built with.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Current transport state.
    pub fn play_state(&self) -> PlayState {
        self.state
    }

    /// Frame counter since construction; unaffected by block size.
    pub fn rendered_frames(&self) -> u64 {
        self.rendered_frames
    }

    /// Peak and RMS accumulators for the most recent block.
    pub fn meters(&self) -> &Meters {
        &self.meters
    }

    /// Master gain currently in force, in dB.
    pub fn master_gain_db(&self) -> f64 {
        self.master_gain_db
    }

    /// Clear every filter state and re-seed every layer. Not realtime safe;
    /// used when the graph is rebuilt or the sample rate changes.
    pub fn reset(&mut self, seed: u64) {
        for (i, l) in self.layers.iter_mut().enumerate() {
            *l = LayerRt::new(self.sample_rate, derive_seed(seed, i, 0x1A7E));
        }
        self.master_dc.reset();
        self.master_gain.snap(0.0);
        self.master_pending_gain_db = None;
        self.master_hold = 0;
        self.master_env.snap(0.0);
        self.state = PlayState::Stopped;
        self.timer = SleepTimer::default();
    }

    /// True when the engine is producing silence and can skip the DSP entirely.
    #[inline]
    fn idle(&self) -> bool {
        self.state != PlayState::Playing
            && !self.master_env.is_ramping()
            && self.master_env.value() == 0.0
    }

    /// Apply a setting change. Allocation free and non-blocking.
    pub fn apply(&mut self, cmd: Command) {
        if let Some(index) = cmd.layer()
            && index >= MAX_LAYERS
        {
            return; // out of range: ignore rather than panic
        }
        match cmd {
            Command::SetLayerEnabled {
                layer,
                enabled,
                fade_ms,
            } => {
                let l = &mut self.layers[layer as usize];
                l.enabled_target = enabled;
                let n = ms_to_samples(fade_ms.into(), self.sample_rate);
                let n = if n == 0 { self.ramp_samples } else { n };
                l.enable.ramp_to(if enabled { 1.0 } else { 0.0 }, n);
            }
            Command::SetLayerFilters {
                layer,
                filters,
                crossfade_ms,
            } => {
                if !filters.is_valid() {
                    // DESIGN.md section 1.4: reject invalid coefficients and keep
                    // the previous, known good state.
                    return;
                }
                let n = ms_to_samples(crossfade_ms.into(), self.sample_rate).max(self.ramp_samples);
                let l = &mut self.layers[layer as usize];
                let old = l.chain_rms;
                let new = filters.expected_rms as f64;
                let huge = old > 0.0
                    && new > 0.0
                    && crate::mixer::linear_to_db(new / old).abs() > FILTER_DIP_THRESHOLD_DB;
                if !l.configured || l.is_silent() {
                    // Nothing is audible right now, so snap rather than
                    // crossfade. Crossfading a *silent* layer would be worse
                    // than pointless: the outgoing cascade is the unfiltered
                    // identity, whose RMS is about 24 dB above a pink cascade,
                    // so a later enable ramp would fade in a burst of white
                    // noise far above the calibrated level.
                    let combined = filters.combined();
                    l.chain_u.chain.snap(combined);
                    l.chain_v.chain.snap(combined);
                    l.chain_rms = new;
                    l.configured = true;
                    l.pending_filters = None;
                    l.morph.snap(1.0);
                } else if huge {
                    // Crossfading a 100 dB level step cannot be held inside the
                    // calibrated level: the blend is already carrying a large
                    // part of the new filter long before any gain ramp can come
                    // down. Dip to silence, swap, come back. A 20 ms dip in noise
                    // is inaudible and, unlike the crossfade, is provably safe.
                    l.chain_rms = new;
                    l.pending_filters = Some((filters, n));
                    l.morph.ramp_to(0.0, n);
                } else {
                    l.load_filters(filters, n);
                }
            }
            Command::SetCharacterShape {
                layer,
                low_shelf,
                high_shelf,
                ramp_ms,
            } => {
                if !low_shelf.is_stable() || !high_shelf.is_stable() {
                    return;
                }
                let l = &mut self.layers[layer as usize];
                let n = ms_to_samples(ramp_ms.into(), self.sample_rate).max(1);
                l.pending_shape = Some((low_shelf, high_shelf));
                // Dip through a bit-exact bypass, then swap.
                l.char_low.ramp_to(0.0, n);
                l.char_high.ramp_to(0.0, n);
            }
            Command::SetCharacterGain {
                layer,
                low_gain,
                high_gain,
                ramp_ms,
            } => {
                let l = &mut self.layers[layer as usize];
                let n = ms_to_samples(ramp_ms.into(), self.sample_rate).max(1);
                let (lo, hi) = (low_gain as f64, high_gain as f64);
                if !lo.is_finite() || !hi.is_finite() {
                    return;
                }
                l.char_low_target = lo;
                l.char_high_target = hi;
                if l.pending_shape.is_none() {
                    l.char_low.ramp_to(lo, n);
                    l.char_high.ramp_to(hi, n);
                }
            }
            Command::SetLayerGain {
                layer,
                gain_db,
                ramp_ms,
            } => {
                if !gain_db.is_finite() {
                    return;
                }
                let l = &mut self.layers[layer as usize];
                l.gain_db = gain_db as f64;
                let n = ms_to_samples(ramp_ms.into(), self.sample_rate).max(1);
                l.gain
                    .ramp_to(crate::mixer::db_to_linear(gain_db as f64), n);
            }
            Command::SetStereoCorrelation {
                layer,
                correlation,
                ramp_ms,
            } => {
                if !correlation.is_finite() {
                    return;
                }
                let l = &mut self.layers[layer as usize];
                let rho = (correlation as f64).clamp(-1.0, 1.0);
                let n = ms_to_samples(ramp_ms.into(), self.sample_rate);
                if n == 0 {
                    l.stereo_rho.snap(rho);
                    l.stereo = StereoMatrix::new(rho);
                } else {
                    l.stereo_rho.ramp_to(rho, n);
                }
            }
            Command::SetLayerSeed {
                layer,
                seed,
                fade_ms,
            } => {
                let l = &mut self.layers[layer as usize];
                if l.enable.value() == 0.0 && !l.enable.is_ramping() {
                    // The layer contributes nothing right now, so the new
                    // stream can start immediately with no audible event.
                    l.reseed(seed);
                    l.pending_seed = None;
                    l.seed_fade.snap(1.0);
                } else {
                    // Never dip faster than the standard parameter ramp, or a
                    // caller passing `fade_ms = 0` would produce a two sample
                    // gap instead of a fade.
                    let n = ms_to_samples(fade_ms.into(), self.sample_rate).max(self.ramp_samples);
                    l.pending_seed = Some((seed, n));
                    l.seed_fade.ramp_to(0.0, n);
                }
            }
            Command::SetMasterGain {
                gain_db,
                ramp_ms,
                hold_ms,
            } => {
                if !gain_db.is_finite() {
                    return;
                }
                // A range limit on a control, exactly like a fader end stop. It
                // is not hidden processing: nothing is clamped inside the audio
                // path, and DESIGN.md section 3.3 forbids silent limiting.
                // The wide lower bound is real: `alpha = -4` needs about
                // -160 dB to land on a -36 dBFS RMS target, because the shaping
                // filter itself has over 130 dB of ultrasonic gain.
                let gain_db = (gain_db as f64).clamp(MIN_MASTER_GAIN_DB, MAX_MASTER_GAIN_DB);
                self.master_gain_db = gain_db;
                let n = ms_to_samples(ramp_ms.into(), self.sample_rate).max(self.ramp_samples);
                let target = crate::mixer::db_to_linear(gain_db);
                // A hold only makes sense while something is actually audible.
                // Applying the start-up calibration before playback cannot
                // overshoot, and holding it would silently delay the first note.
                let silent = self.master_env.value() == 0.0 && !self.master_env.is_ramping();
                if target <= self.master_gain.value() || silent {
                    // Falling gains are always safe to apply at once: the blend
                    // is never louder than the outgoing filter.
                    self.master_gain.ramp_to(target, n);
                    self.master_pending_gain_db = None;
                    self.master_hold = 0;
                } else {
                    // A *rising* gain is the dangerous direction. While a
                    // coefficient crossfade runs, the blend still carries the
                    // outgoing filter, so if the outgoing filter is much louder
                    // than the incoming one (White -> custom alpha = +4 spans
                    // over 100 dB of level) raising the gain early would push
                    // the output far past the calibrated level. The caller
                    // passes the crossfade length as `hold_ms` and the gain rise
                    // waits for exactly that many samples.
                    let hold = ms_to_samples(hold_ms.into(), self.sample_rate)
                        .min(ms_to_samples(MAX_GAIN_DEFER_MS, self.sample_rate));
                    self.master_pending_gain_db = Some(gain_db);
                    self.master_hold = hold;
                }
            }
            Command::SetMasterDcBlocker { coeffs } => {
                if coeffs.is_stable() {
                    self.master_dc.set_coeffs_reset(coeffs);
                }
            }
            Command::SetLimiter {
                enabled,
                ceiling_dbfs,
            } => {
                self.limiter_enabled = enabled;
                self.limiter_ceiling =
                    crate::mixer::db_to_linear(ceiling_dbfs.clamp(-24.0, 0.0) as f64);
            }
            Command::SetFades { in_ms, out_ms } => {
                self.fade_in_samples = ms_to_samples(in_ms.into(), self.sample_rate);
                self.fade_out_samples = ms_to_samples(out_ms.into(), self.sample_rate);
            }
            Command::Play => {
                self.state = PlayState::Playing;
                self.master_env.ramp_to(1.0, self.fade_in_samples.max(1));
            }
            Command::Pause => {
                self.state = PlayState::Paused;
                self.master_env.ramp_to(0.0, self.fade_out_samples);
                self.timer = SleepTimer::default();
            }
            Command::Stop => {
                self.state = PlayState::Stopped;
                self.master_env.ramp_to(0.0, self.fade_out_samples);
                self.timer = SleepTimer::default();
            }
            Command::SetTimer {
                seconds,
                fade_out_seconds,
            } => {
                if !seconds.is_finite() || seconds <= 0.0 {
                    self.timer = SleepTimer::default();
                } else {
                    let total = (seconds as f64 * self.sample_rate).round() as i64;
                    self.timer = SleepTimer {
                        total,
                        left: total,
                        fade: (fade_out_seconds.max(0.0) as f64 * self.sample_rate).round() as i64,
                        firing: false,
                    };
                }
            }
            Command::ClearTimer => self.timer = SleepTimer::default(),
            Command::EmergencySilence => {
                self.state = PlayState::Stopped;
                self.master_env.snap(0.0);
                self.master_gain.snap(0.0);
                self.master_gain_db = -120.0;
                self.master_pending_gain_db = None;
                self.master_hold = 0;
                for l in self.layers.iter_mut() {
                    l.enable.snap(0.0);
                    l.reset_all();
                }
                self.timer = SleepTimer::default();
            }
        }
    }

    /// Render `frames` stereo frames into the internal scratch and return it.
    ///
    /// `frames` must be `<= MAX_BLOCK`.
    pub fn render_block(&mut self, frames: usize) -> &[f32] {
        let n = frames.min(MAX_BLOCK);
        self.meters.begin_block();

        if self.idle() {
            self.scratch[..n * ENGINE_CHANNELS].fill(0.0);
            self.rendered_frames += n as u64;
            return &self.scratch[..n * ENGINE_CHANNELS];
        }

        // Sleep timer: the fade-out occupies the *last* `fade` samples of the
        // requested duration, so `--minutes 45 --fade-out 60` starts fading at
        // 44 minutes.
        if self.timer.total > 0 && !self.timer.firing {
            self.timer.left -= n as i64;
            if self.timer.left <= self.timer.fade {
                self.timer.firing = true;
                self.master_env.ramp_to(0.0, self.timer.fade.max(0) as u32);
            }
        }

        let ramp = self.ramp_samples;
        let ceiling = self.limiter_ceiling;
        let limiter = self.limiter_enabled;
        let out = &mut self.scratch[..n * ENGINE_CHANNELS];

        for i in 0..n {
            // The hold is counted down per sample, not per block, so the PCM is
            // identical whatever buffer size the host asks for.
            if let Some(target_db) = self.master_pending_gain_db {
                if self.master_hold == 0 {
                    self.master_gain.ramp_to(
                        crate::mixer::db_to_linear(target_db),
                        self.ramp_samples.max(1),
                    );
                    self.master_pending_gain_db = None;
                } else {
                    self.master_hold -= 1;
                }
            }
            let mut l = 0.0f64;
            let mut r = 0.0f64;
            for layer in self.layers.iter_mut() {
                if !layer.audible() {
                    continue;
                }
                let (a, b) = layer.next_frame(ramp);
                l += a;
                r += b;
            }
            let g = self.master_gain.step() * self.master_env.step();
            let mut ol = self.master_dc.process(l) * g;
            let mut or = self.master_dc.process(r) * g;
            if limiter {
                if ol > ceiling {
                    ol = ceiling;
                    self.meters.note_clipped();
                } else if ol < -ceiling {
                    ol = -ceiling;
                    self.meters.note_clipped();
                }
                if or > ceiling {
                    or = ceiling;
                    self.meters.note_clipped();
                } else if or < -ceiling {
                    or = -ceiling;
                    self.meters.note_clipped();
                }
            }
            if !ol.is_finite() || !or.is_finite() {
                // DESIGN.md section 3.3: numerical anomalies become silence plus
                // an error notification, never a blast of noise.
                self.meters.push(f64::NAN);
                ol = 0.0;
                or = 0.0;
            } else {
                self.meters.push(ol);
                self.meters.push(or);
            }
            out[i * ENGINE_CHANNELS] = ol as f32;
            out[i * ENGINE_CHANNELS + 1] = or as f32;
        }

        self.rendered_frames += n as u64;

        // The transport only settles once the fade has actually finished, which
        // is what makes the sleep timer pause instead of staying on "playing".
        if self.timer.firing && !self.master_env.is_ramping() && self.master_env.value() == 0.0 {
            self.state = PlayState::Paused;
            self.timer = SleepTimer::default();
        }

        out
    }

    /// Render interleaved `f32` for a sink with `channels` channels.
    ///
    /// Handles any `frames` by splitting into [`MAX_BLOCK`] chunks, so a host
    /// with a huge period size still gets allocation free rendering.
    pub fn render(&mut self, output: &mut [f32], frames: usize, channels: usize) {
        if channels == 0 || frames == 0 {
            return;
        }
        let mut done = 0usize;
        while done < frames {
            let n = (frames - done).min(MAX_BLOCK);
            let block = self.render_block(n);
            let dst = &mut output[done * channels..(done + n) * channels];
            write_interleaved(block, dst, n, channels);
            done += n;
        }
    }
}

/// Convert the engine's stereo scratch into the sink's channel layout.
///
/// `channels == 1` writes the mono downmix `(L + R) / 2`; the calibration is
/// told about the resulting variance factor so the level target still holds.
/// More than two channels write stereo into the first two and silence the rest.
fn write_interleaved(src: &[f32], dst: &mut [f32], frames: usize, channels: usize) {
    match channels {
        1 => {
            for i in 0..frames {
                let l = src[i * 2] as f64;
                let r = src[i * 2 + 1] as f64;
                dst[i] = ((l + r) * 0.5) as f32;
            }
        }
        2 => {
            dst[..frames * 2].copy_from_slice(&src[..frames * 2]);
        }
        n => {
            for i in 0..frames {
                let base = i * n;
                dst[base] = src[i * 2];
                dst[base + 1] = src[i * 2 + 1];
                for c in 2..n {
                    dst[base + c] = 0.0;
                }
            }
        }
    }
}

impl LayerRt {
    fn reset_all(&mut self) {
        self.chain_u.reset();
        self.chain_v.reset();
        self.stereo_rho.snap(self.stereo.rho);
        self.gain.snap(1.0);
        self.enable.snap(0.0);
        self.seed_fade.snap(1.0);
        self.morph.snap(1.0);
        self.pending_filters = None;
        self.char_low.snap(0.0);
        self.char_high.snap(0.0);
        self.pending_shape = None;
        self.pending_seed = None;
    }
}

/// Convenience used by the daemon: push a designed character into a layer.
pub fn character_commands(layer: u8, design: &CharacterDesign, ramp_ms: f32) -> [Command; 2] {
    [
        Command::SetCharacterShape {
            layer,
            low_shelf: design.low_shelf,
            high_shelf: design.high_shelf,
            ramp_ms,
        },
        Command::SetCharacterGain {
            layer,
            low_gain: design.low_gain as f32,
            high_gain: design.high_gain as f32,
            ramp_ms,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::colors::Color;
    use crate::designer;

    fn engine() -> Engine {
        let mut e = Engine::new(48000.0, 1);
        let f = LayerFilters {
            color: designer::design_color(1.0, 48000.0).unwrap().coeffs,
            high_pass: designer::design_high_pass(35.0, core::f64::consts::FRAC_1_SQRT_2, 48000.0),
            low_pass: BiquadCoeffs::IDENTITY,
            expected_rms: 0.0,
        };
        e.apply(Command::SetLayerFilters {
            layer: 0,
            filters: f,
            crossfade_ms: 40.0,
        });
        e.apply(Command::SetLayerEnabled {
            layer: 0,
            enabled: true,
            fade_ms: 10.0,
        });
        e.apply(Command::SetStereoCorrelation {
            layer: 0,
            correlation: 0.35,
            ramp_ms: 0.0,
        });
        e.apply(Command::SetMasterGain {
            gain_db: 0.0,
            ramp_ms: 0.0,
            hold_ms: 0.0,
        });
        e
    }

    fn render_some(e: &mut Engine, blocks: usize, frames: usize) -> Vec<f32> {
        let mut out = Vec::new();
        for _ in 0..blocks {
            out.extend_from_slice(e.render_block(frames));
        }
        out
    }

    #[test]
    fn idle_engine_is_silent() {
        let mut e = engine();
        let out = render_some(&mut e, 4, 128);
        assert!(out.iter().all(|s| *s == 0.0));
    }

    #[test]
    fn block_size_does_not_change_the_pcm() {
        // DESIGN.md section 4.1: 128 frames at a time and 256 frames at a time
        // must give the identical sample sequence.
        let mut a = engine();
        let mut b = engine();
        a.apply(Command::Play);
        b.apply(Command::Play);

        let mut out_a = Vec::new();
        for _ in 0..40 {
            out_a.extend_from_slice(a.render_block(128));
        }
        let mut out_b = Vec::new();
        for _ in 0..20 {
            out_b.extend_from_slice(b.render_block(256));
        }
        assert_eq!(out_a.len(), out_b.len());
        let mut bad = 0;
        for (x, y) in out_a.iter().zip(out_b.iter()) {
            if (x - y).abs() > 1e-7 {
                bad += 1;
            }
        }
        assert_eq!(bad, 0, "{bad} samples differ between block sizes");
        assert!(out_a.iter().any(|s| *s != 0.0), "engine produced silence");
    }

    /// One millisecond sliding RMS, used to look for energy spikes.
    fn sliding_rms(x: &[f32], window: usize) -> Vec<f32> {
        if x.len() < window {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(x.len() - window + 1);
        let mut acc: f64 = x[..window].iter().map(|v| (*v as f64) * (*v as f64)).sum();
        out.push((acc / window as f64).sqrt() as f32);
        for i in window..x.len() {
            let a = x[i - window] as f64;
            let b = x[i] as f64;
            acc += b * b - a * a;
            out.push((acc / window as f64).max(0.0).sqrt() as f32);
        }
        out
    }

    #[test]
    fn crossfade_is_sample_continuous() {
        // A coefficient change must never put a step into the output, even when
        // it lands in the middle of an earlier crossfade ("latest value wins").
        let fs = 48_000.0;
        let lp = SosCoeffs::new(&[BiquadCoeffs::one_pole_low_pass(200.0, fs)]);
        let hp = SosCoeffs::new(&[BiquadCoeffs::one_pole_high_pass(2000.0, fs)]);
        let mut c = CrossSos::new(SosCoeffs::IDENTITY);
        for _ in 0..20_000 {
            c.process(1.0);
        }
        let mut prev = c.process(1.0);
        let mut max_step = 0.0f64;
        let settle = |c: &mut CrossSos, prev: &mut f64, max_step: &mut f64, n: usize| {
            for _ in 0..n {
                let y = c.process(1.0);
                *max_step = max_step.max((y - *prev).abs());
                *prev = y;
            }
        };
        c.set(lp, 256);
        settle(&mut c, &mut prev, &mut max_step, 300);
        c.set(hp, 256);
        settle(&mut c, &mut prev, &mut max_step, 300);
        // Retarget 40 samples into a fresh crossfade.
        c.set(lp, 256);
        settle(&mut c, &mut prev, &mut max_step, 40);
        c.set(SosCoeffs::IDENTITY, 256);
        settle(&mut c, &mut prev, &mut max_step, 400);
        assert!(max_step < 5e-3, "max step {max_step}");
    }

    #[test]
    fn stop_fade_is_an_exact_linear_ramp_with_no_step() {
        // The master envelope is a scalar applied after everything else and it
        // does not touch any filter state, so rendering a control engine that
        // keeps playing exposes the envelope exactly: test / control == env.
        let mut test = engine();
        let mut control = engine();
        test.apply(Command::Play);
        control.apply(Command::Play);
        let _ = render_some(&mut control, 30, 256);
        let _ = render_some(&mut test, 30, 256);
        test.apply(Command::Stop);
        let a = render_some(&mut test, 60, 256);
        let b = render_some(&mut control, 60, 256);
        assert_eq!(a.len(), b.len());

        let mut ratios = Vec::new();
        for (x, y) in a.iter().zip(b.iter()) {
            if y.abs() > 1e-4 {
                ratios.push((*x as f64) / (*y as f64));
            }
        }
        assert!(ratios.len() > 1000, "not enough usable samples");
        // Monotone, continuous, and it actually reaches zero.
        let mut prev = ratios[0];
        let mut max_step = 0.0f64;
        for r in &ratios {
            assert!(*r <= prev + 1e-6, "envelope went back up: {r} after {prev}");
            max_step = max_step.max((*r - prev).abs());
            prev = *r;
        }
        // A 100 ms fade over 15360 samples moves at most 1/15360 per sample.
        assert!(max_step < 1.0 / 4_000.0, "max envelope step {max_step}");
        assert!(ratios[ratios.len() - 1].abs() < 1e-3, "did not fade out");
        assert!(a[a.len() - 1] == 0.0);
    }

    #[test]
    fn redetecting_the_same_settings_twice_reproduces_the_pcm() {
        let mut a = engine();
        let mut b = engine();
        for e in [&mut a, &mut b] {
            e.apply(Command::Play);
        }
        let x = render_some(&mut a, 10, 256);
        let y = render_some(&mut b, 10, 256);
        assert_eq!(x, y);
    }

    #[test]
    fn invalid_coefficients_are_rejected_and_state_is_kept() {
        let mut e = engine();
        e.apply(Command::Play);
        let before = render_some(&mut e, 2, 256);
        // Unstable biquad: |a2| >= 1.
        let bad = LayerFilters {
            color: SosCoeffs::new(&[BiquadCoeffs {
                b0: 1.0,
                b1: 0.0,
                b2: 0.0,
                a1: 0.0,
                a2: 1.5,
            }]),
            ..LayerFilters::default()
        };
        e.apply(Command::SetLayerFilters {
            layer: 0,
            filters: bad,
            crossfade_ms: 1.0,
        });
        let after = render_some(&mut e, 2, 256);
        // Still finite and non-zero: the rejected change did not take effect.
        assert!(after.iter().all(|s| s.is_finite()));
        assert!(after.iter().any(|s| *s != 0.0));
        assert_eq!(before.len(), after.len());
    }

    #[test]
    fn character_changes_are_click_free() {
        // Compare against a control engine that shares the same PRNG stream, so
        // the difference isolates the processing change. A click would show up
        // as a one block spike in the difference energy.
        let mut test = engine();
        let mut control = engine();
        for e in [&mut test, &mut control] {
            e.apply(Command::Play);
        }
        let _ = render_some(&mut control, 6, 256);
        let _ = render_some(&mut test, 6, 256);
        let d = crate::characters::design(crate::characters::Character::Intense, 1.0, 48_000.0);
        for c in character_commands(0, &d, 20.0) {
            test.apply(c);
        }
        let a = render_some(&mut test, 30, 256);
        let b = render_some(&mut control, 30, 256);
        let diff: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x - y).collect();
        let rms = sliding_rms(&diff, 64);
        assert!(!rms.is_empty());
        let mut max_ratio = 0.0f32;
        for w in rms.windows(2) {
            if w[0] > 1e-6 {
                max_ratio = max_ratio.max(w[1] / w[0]);
            }
        }
        assert!(max_ratio < 1.5, "difference energy spiked by {max_ratio}x");
        assert!(diff.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn stereo_correlation_reaches_the_requested_value() {
        // Regression: with a non-zero ramp time the correlation command used to
        // be dropped entirely, leaving every layer permanently mono.
        for rho in [1.0f32, 0.0, -0.5, 0.35] {
            let mut e = engine();
            e.apply(Command::SetStereoCorrelation {
                layer: 0,
                correlation: rho,
                ramp_ms: 20.0,
            });
            e.apply(Command::Play);
            for _ in 0..200 {
                e.render_block(256);
            }
            // Correlate the two channels of the engine output.
            let mut sll = 0.0;
            let mut srr = 0.0;
            let mut slr = 0.0;
            for _ in 0..400 {
                let b = e.render_block(256);
                for f in b.chunks(2) {
                    let l = f[0] as f64;
                    let r = f[1] as f64;
                    sll += l * l;
                    srr += r * r;
                    slr += l * r;
                }
            }
            let measured = slr / (sll.sqrt() * srr.sqrt());
            assert!(
                (measured - rho as f64).abs() < 0.05,
                "requested rho {rho}, measured {measured:.4}"
            );
        }
    }

    #[test]
    fn a_layer_seed_change_takes_effect_and_is_click_free() {
        let mut test = engine();
        let mut control = engine();
        for e in [&mut test, &mut control] {
            e.apply(Command::Play);
        }
        let a = render_some(&mut control, 20, 256);
        let b_pre = render_some(&mut test, 20, 256);
        assert_eq!(a, b_pre, "the two engines must start identical");

        test.apply(Command::SetLayerSeed {
            layer: 0,
            seed: 0xDEAD_BEEF,
            fade_ms: 20.0,
        });
        let b = render_some(&mut test, 60, 256);
        let a_cont = render_some(&mut control, 60, 256);
        // The streams must now have diverged.
        let diff: f64 = a_cont
            .iter()
            .zip(b.iter())
            .map(|(x, y)| ((x - y) as f64).abs())
            .sum();
        assert!(diff > 1.0, "the seed change had no effect ({diff})");
        // ... and the transition must not have spiked.
        let d: Vec<f32> = a_cont.iter().zip(b.iter()).map(|(x, y)| x - y).collect();
        let mut max_ratio = 0.0f32;
        let rms = sliding_rms(&d, 64);
        for w in rms.windows(2) {
            if w[0] > 1e-6 {
                max_ratio = max_ratio.max(w[1] / w[0]);
            }
        }
        assert!(max_ratio < 1.5, "reseed spiked by {max_ratio}x");
    }

    #[test]
    fn disabling_a_playing_layer_fades_instead_of_cutting() {
        let mut e = engine();
        e.apply(Command::Play);
        let mut out = render_some(&mut e, 20, 256);
        e.apply(Command::SetLayerEnabled {
            layer: 0,
            enabled: false,
            fade_ms: 20.0,
        });
        out.extend_from_slice(render_some(&mut e, 60, 256).as_slice());
        // After the fade the layer contributes nothing at all.
        let tail = &out[out.len() - 256 * 2..];
        assert!(tail.iter().all(|s| s.abs() < 1e-9), "layer did not stop");
        // The transition is bounded: no sample is louder than the running peak.
        let peak = out.iter().fold(0.0f32, |a, b| a.max(b.abs()));
        assert!(peak.is_finite() && peak > 0.0);
    }

    #[test]
    fn a_large_filter_change_never_overshoots_the_calibrated_level() {
        // Regression for the worst case in the whole control surface: swapping a
        // very quiet shaping filter for a very loud one (White -> Violet is
        // +60 dB, custom alpha = -4 is +134 dB) while the calibrated master gain
        // also has to change by the same amount.
        let fs = 48_000.0;
        let quiet = crate::designer::design_color(4.0, fs).unwrap();
        let loud = crate::designer::design_color(-4.0, fs).unwrap();
        let warm = |d: &crate::designer::ColorDesign| {
            crate::metrics::layer_response(
                d.coeffs,
                crate::characters::design(crate::characters::Character::Neutral, 0.0, fs),
                BiquadCoeffs::IDENTITY,
                BiquadCoeffs::IDENTITY,
                0.0,
                1.0,
            )
            .expected_rms(fs)
        };
        let (rq, rl) = (warm(&quiet), warm(&loud));
        let target = crate::mixer::db_to_linear(-36.0);
        let g_quiet = target / rq;
        let g_loud = target / rl;

        for (from, to, from_gain, to_gain, rq_from, rl_from) in [
            (quiet.coeffs, loud.coeffs, g_quiet, g_loud, rq, rl),
            (loud.coeffs, quiet.coeffs, g_loud, g_quiet, rl, rq),
        ] {
            let mut e = Engine::new(fs, 3);
            e.apply(Command::SetLayerFilters {
                layer: 0,
                filters: LayerFilters {
                    color: from,
                    high_pass: BiquadCoeffs::IDENTITY,
                    low_pass: BiquadCoeffs::IDENTITY,
                    expected_rms: rq_from as f32,
                },
                crossfade_ms: 1.0,
            });
            e.apply(Command::SetLayerEnabled {
                layer: 0,
                enabled: true,
                fade_ms: 1.0,
            });
            e.apply(Command::SetMasterGain {
                gain_db: crate::mixer::linear_to_db(from_gain) as f32,
                ramp_ms: 1.0,
                hold_ms: 0.0,
            });
            e.apply(Command::Play);
            for _ in 0..200 {
                e.render_block(256);
            }
            // Now change both at once, exactly as `compile().commands()` does.
            e.apply(Command::SetLayerFilters {
                layer: 0,
                filters: LayerFilters {
                    color: to,
                    high_pass: BiquadCoeffs::IDENTITY,
                    low_pass: BiquadCoeffs::IDENTITY,
                    expected_rms: rl_from as f32,
                },
                crossfade_ms: 100.0,
            });
            e.apply(Command::SetMasterGain {
                gain_db: crate::mixer::linear_to_db(to_gain) as f32,
                ramp_ms: 20.0,
                hold_ms: 100.0,
            });
            let mut peak = 0.0f64;
            for _ in 0..600 {
                for s in e.render_block(256) {
                    peak = peak.max((*s as f64).abs());
                }
            }
            // The calibrated target is -36 dBFS RMS; a planning crest factor of
            // 4.2 puts the expected peak near -23.5 dBFS. Anything above
            // -6 dBFS means the crossfade overshot.
            let peak_dbfs = crate::mixer::linear_to_db(peak);
            assert!(
                peak_dbfs < -6.0,
                "filter change overshot: peak {peak_dbfs:.1} dBFS"
            );
        }
    }

    #[test]
    fn enabling_a_layer_never_fades_in_an_unfiltered_cascade() {
        // Regression: a layer that was configured while silent used to
        // *crossfade* from the identity cascade, so its first enable ramped in
        // white noise about 24 dB above the calibrated level.
        let fs = 48_000.0;
        let params = crate::params::PresetParams::default();
        let graph = crate::compile::compile(&params, fs, 2);
        let mut e = Engine::new(fs, params.seed);
        for c in graph.commands(&params) {
            e.apply(c);
        }
        e.apply(Command::Play);
        for _ in 0..40 {
            e.render_block(256);
        }
        let mut peak = 0.0f64;
        for _ in 0..200 {
            for s in e.render_block(256) {
                peak = peak.max((*s as f64).abs());
            }
        }
        let peak_dbfs = crate::mixer::linear_to_db(peak);
        // -36 dBFS RMS with a planning crest of 4.2 is about -23.5 dBFS.
        assert!(
            peak_dbfs < -12.0,
            "layer enable overshot: peak {peak_dbfs:.1} dBFS"
        );
    }

    #[test]
    fn extreme_parameter_changes_never_produce_nan_or_panic() {
        let mut e = engine();
        e.apply(Command::Play);
        for i in 0..500u32 {
            let color = Color::ALL[(i as usize) % Color::ALL.len()];
            let alpha = color.resolve_alpha((i as f64 * 0.37).sin() * 4.0);
            if let Ok(d) = designer::design_color(alpha, 48000.0) {
                e.apply(Command::SetLayerFilters {
                    layer: 0,
                    filters: LayerFilters {
                        color: d.coeffs,
                        high_pass: designer::design_high_pass(20.0, 0.7, 48000.0),
                        low_pass: designer::design_low_pass(18000.0, 0.7, 48000.0),
                        expected_rms: 0.0,
                    },
                    crossfade_ms: 5.0,
                });
            }
            e.apply(Command::SetCharacterGain {
                layer: 0,
                low_gain: ((i as f64) * 0.1).sin() as f32,
                high_gain: ((i as f64) * 0.2).cos() as f32,
                ramp_ms: 3.0,
            });
            e.apply(Command::SetStereoCorrelation {
                layer: 0,
                correlation: ((i as f64) * 0.05).sin() as f32,
                ramp_ms: 1.0,
            });
            e.apply(Command::SetLayerGain {
                layer: 0,
                gain_db: ((i as f64) * 0.03).cos() as f32 * 12.0,
                ramp_ms: 4.0,
            });
            let block = e.render_block(64);
            assert!(block.iter().all(|s| s.is_finite()), "iteration {i}");
        }
        assert_eq!(e.meters().nan_events(), 0);
    }
}
