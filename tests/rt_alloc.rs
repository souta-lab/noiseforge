//! Realtime allocation test.
//!
//! DESIGN.md section 9.2 requires the realtime callback to allocate and free
//! **zero** memory, and section 4.3 forbids `Vec` growth, heap allocation,
//! locks and I/O on the realtime thread.
//!
//! This file installs a counting global allocator and asserts that the render
//! path, including live parameter changes, never touches the heap. It lives in
//! its own integration test binary so the counter is not disturbed by other
//! tests.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

use noise_dsp::compile::compile;
use noise_dsp::engine::{Command, Engine, LayerFilters, MAX_BLOCK};
use noise_dsp::params::{LayerParams, PresetParams};

struct CountingAllocator;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static FREES: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    /// Armed only on the thread doing the realtime simulation, so the test
    /// harness's own bookkeeping on other threads is ignored.
    static ARMED: Cell<bool> = const { Cell::new(false) };
}

#[inline]
fn armed() -> bool {
    ARMED.try_with(|c| c.get()).unwrap_or(false)
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if armed() {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if armed() {
            FREES.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if armed() {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(new_size, Ordering::Relaxed);
        }
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn arm() {
    ALLOCS.store(0, Ordering::Relaxed);
    FREES.store(0, Ordering::Relaxed);
    BYTES.store(0, Ordering::Relaxed);
    ARMED.with(|c| c.set(true));
}

fn disarm() -> (usize, usize, usize) {
    ARMED.with(|c| c.set(false));
    (
        ALLOCS.load(Ordering::Relaxed),
        FREES.load(Ordering::Relaxed),
        BYTES.load(Ordering::Relaxed),
    )
}

fn full_preset() -> PresetParams {
    PresetParams {
        seed: 42,
        layers: (0..8)
            .map(|i| LayerParams {
                name: format!("Layer {i}"),
                color: noise_dsp::Color::ALL[i % noise_dsp::Color::ALL.len()],
                alpha: 1.0 + i as f64 * 0.2,
                character: if i % 2 == 0 {
                    noise_dsp::Character::Smooth
                } else {
                    noise_dsp::Character::Intense
                },
                character_amount: 0.7,
                gain_db: -3.0,
                stereo_correlation: 0.4,
                ..LayerParams::default()
            })
            .collect(),
        ..PresetParams::default()
    }
}

#[test]
fn render_never_allocates() {
    let params = full_preset();
    let graph = compile(&params, 48_000.0, 2);

    // Everything that may allocate happens here, before arming the counter.
    let mut engine = Engine::new(48_000.0, params.seed);
    for c in graph.commands(&params) {
        engine.apply(c);
    }
    engine.apply(Command::Play);
    let mut sink = vec![0.0f32; MAX_BLOCK * 2];
    for _ in 0..64 {
        engine.render(&mut sink, 256, 2);
    }

    arm();
    for block in 0..2_000usize {
        let frames = [64usize, 128, 192, 256, 512, 1024][block % 6];
        engine.render(&mut sink, frames, 2);
    }
    let (allocs, frees, bytes) = disarm();
    assert_eq!(
        (allocs, frees, bytes),
        (0, 0, 0),
        "realtime render allocated {allocs} times, freed {frees} times, {bytes} bytes"
    );
}

#[test]
fn live_parameter_changes_never_allocate() {
    let params = full_preset();
    let graph = compile(&params, 48_000.0, 2);
    let mut engine = Engine::new(48_000.0, params.seed);
    for c in graph.commands(&params) {
        engine.apply(c);
    }
    engine.apply(Command::Play);
    let mut sink = vec![0.0f32; MAX_BLOCK * 2];
    for _ in 0..64 {
        engine.render(&mut sink, 256, 2);
    }

    // Pre-build every coefficient set the way the daemon does on its control
    // thread: designs happen off the realtime thread, only POD commands cross.
    let designs: Vec<LayerFilters> = [0.5f64, 1.0, 1.5, 2.0, -1.0, -2.0, 3.0, 0.25]
        .iter()
        .map(|alpha| {
            let d = noise_dsp::designer::design_color(*alpha, 48_000.0).unwrap();
            LayerFilters {
                color: d.coeffs,
                high_pass: noise_dsp::designer::design_high_pass(
                    35.0,
                    std::f64::consts::FRAC_1_SQRT_2,
                    48_000.0,
                ),
                low_pass: noise_dsp::designer::design_low_pass(
                    18_000.0,
                    std::f64::consts::FRAC_1_SQRT_2,
                    48_000.0,
                ),
                expected_rms: 0.0,
            }
        })
        .collect();
    let characters: Vec<_> = (0..4)
        .map(|i| {
            noise_dsp::characters::design(
                noise_dsp::Character::ALL[i % 3],
                (i as f64 * 0.25).fract(),
                48_000.0,
            )
        })
        .collect();
    let commands: Vec<Command> = designs
        .iter()
        .enumerate()
        .flat_map(|(i, f)| {
            let layer = (i % 8) as u8;
            let ch = characters[i % characters.len()];
            let mut v = vec![
                Command::SetLayerFilters {
                    layer,
                    filters: *f,
                    crossfade_ms: 30.0,
                },
                Command::SetCharacterShape {
                    layer,
                    low_shelf: ch.low_shelf,
                    high_shelf: ch.high_shelf,
                    ramp_ms: 20.0,
                },
                Command::SetCharacterGain {
                    layer,
                    low_gain: ch.low_gain as f32,
                    high_gain: ch.high_gain as f32,
                    ramp_ms: 20.0,
                },
                Command::SetLayerGain {
                    layer,
                    gain_db: -3.0 - i as f32,
                    ramp_ms: 20.0,
                },
                Command::SetStereoCorrelation {
                    layer,
                    correlation: 0.4 - i as f32 * 0.1,
                    ramp_ms: 20.0,
                },
            ];
            if i % 4 == 0 {
                v.push(Command::SetLayerSeed {
                    layer,
                    seed: 1000 + i as u64,
                    fade_ms: 10.0,
                });
            }
            v
        })
        .collect();

    arm();
    let mut cursor = 0usize;
    for block in 0..3_000usize {
        // Push a burst of commands, then render, exactly like the daemon's RT
        // callback draining its SPSC queue.
        for _ in 0..3 {
            engine.apply(commands[cursor % commands.len()]);
            cursor += 1;
        }
        let frames = [64usize, 128, 256, 512][block % 4];
        engine.render(&mut sink, frames, 2);
    }
    let (allocs, frees, bytes) = disarm();
    assert_eq!(
        (allocs, frees, bytes),
        (0, 0, 0),
        "parameter changes allocated {allocs} times, freed {frees} times, {bytes} bytes"
    );
    assert_eq!(engine.meters().nan_events(), 0);
}

#[test]
fn engine_construction_and_sink_drop_are_the_only_allocations() {
    // Sanity check on the harness itself: if this does not observe allocations,
    // the counter is broken and the two tests above would pass vacuously.
    arm();
    let v: Vec<f32> = Vec::with_capacity(1024);
    drop(v);
    let boxed: Box<[u8; 4096]> = Box::new([0u8; 4096]);
    drop(boxed);
    let (allocs, _frees, bytes) = disarm();
    assert!(allocs >= 2, "counter observed only {allocs} allocations");
    assert!(bytes >= 4096);
}
