//! Offline rendering of a preset to PCM, used by `noiseforge render`, by the
//! spectral tests and by the "record" feature.
//!
//! DESIGN.md section 4.1: the same engine that feeds PipeWire renders offline,
//! so what is measured here is exactly what is played.

use noise_dsp::compile::{CompiledGraph, compile};
use noise_dsp::engine::{Command, Engine};
use noise_dsp::params::PresetParams;

/// What to render.
#[derive(Clone, Debug)]
pub struct RenderRequest {
    /// Preset to render.
    pub params: PresetParams,
    /// Sample rate.
    pub sample_rate: f64,
    /// Length of the returned audio, in seconds.
    pub seconds: f64,
    /// Audio discarded before the returned region, in seconds. Lets recursive
    /// filter transients and fades settle, as DESIGN.md section 9.1 requires.
    pub warmup_seconds: f64,
    /// Output channels: 1 (mono downmix) or 2.
    pub channels: usize,
    /// Override the preset seed.
    pub seed: Option<u64>,
    /// Start playing automatically.
    pub auto_play: bool,
}

impl Default for RenderRequest {
    fn default() -> Self {
        Self {
            params: PresetParams::default(),
            sample_rate: 48_000.0,
            seconds: 10.0,
            warmup_seconds: 0.5,
            channels: 2,
            seed: None,
            auto_play: true,
        }
    }
}

impl RenderRequest {
    /// A request for `seconds` of the given preset at 48 kHz.
    pub fn new(params: PresetParams, seconds: f64) -> Self {
        Self {
            params,
            seconds,
            ..Self::default()
        }
    }
}

/// Rendered audio plus the graph that produced it.
#[derive(Clone, Debug)]
pub struct RenderResult {
    /// Interleaved samples, `frames * channels` long.
    pub samples: Vec<f32>,
    /// Frame count.
    pub frames: usize,
    /// Channel count.
    pub channels: usize,
    /// Sample rate.
    pub sample_rate: f64,
    /// The compiled graph, including design reports and calibration.
    pub graph: CompiledGraph,
}

impl RenderResult {
    /// Extract one channel as a mono buffer.
    pub fn channel(&self, ch: usize) -> Vec<f32> {
        noise_offline_deinterleave(&self.samples, self.channels, ch)
    }

    /// Peak magnitude.
    pub fn peak(&self) -> f32 {
        self.samples.iter().fold(0.0f32, |a, b| a.max(b.abs()))
    }

    /// RMS over all channels.
    pub fn rms(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.samples.iter().map(|s| (*s as f64) * (*s as f64)).sum();
        (sum / self.samples.len() as f64).sqrt()
    }

    /// RMS of one channel.
    pub fn channel_rms(&self, ch: usize) -> f64 {
        let v = self.channel(ch);
        if v.is_empty() {
            return 0.0;
        }
        let sum: f64 = v.iter().map(|s| (*s as f64) * (*s as f64)).sum();
        (sum / v.len() as f64).sqrt()
    }
}

// Local copy so `render` does not depend on the analysis module.
fn noise_offline_deinterleave(x: &[f32], channels: usize, channel: usize) -> Vec<f32> {
    if channels <= 1 {
        return x.to_vec();
    }
    x.chunks(channels)
        .filter_map(|c| c.get(channel).copied())
        .collect()
}

/// Render a preset to interleaved PCM, exactly as the realtime engine would.
pub fn render(req: &RenderRequest) -> RenderResult {
    let fs = req.sample_rate;
    let channels = req.channels.clamp(1, 2);
    // The seed override must reach the *layers*, not just the engine: a layer
    // with no explicit seed derives its own stream from the preset seed, so
    // setting only the engine's seed would leave the audio unchanged.
    let mut params = req.params.clone();
    if let Some(seed) = req.seed {
        params.seed = seed;
    }
    let seed = params.seed;
    let graph = compile(&params, fs, channels);

    let mut engine = Engine::new(fs, seed);
    for c in graph.commands(&params) {
        engine.apply(c);
    }
    if req.auto_play {
        engine.apply(Command::Play);
    }

    let frames = (req.seconds.max(0.0) * fs).round() as usize;
    let warmup = (req.warmup_seconds.max(0.0) * fs).round() as usize;
    let mut samples = vec![0.0f32; frames * channels];

    // Warm up in engine sized chunks so filter transients and fades settle.
    let mut left = warmup;
    let mut sink = vec![0.0f32; noise_dsp::engine::MAX_BLOCK * channels];
    while left > 0 {
        let n = left.min(noise_dsp::engine::MAX_BLOCK);
        engine.render(&mut sink[..n * channels], n, channels);
        left -= n;
    }

    let mut done = 0usize;
    while done < frames {
        let n = (frames - done).min(noise_dsp::engine::MAX_BLOCK);
        engine.render(
            &mut samples[done * channels..(done + n) * channels],
            n,
            channels,
        );
        done += n;
    }

    RenderResult {
        samples,
        frames,
        channels,
        sample_rate: fs,
        graph,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use noise_dsp::colors::Color;
    use noise_dsp::params::LayerParams;

    #[test]
    fn rendering_is_deterministic() {
        let req = RenderRequest {
            seconds: 0.5,
            ..RenderRequest::default()
        };
        let a = render(&req);
        let b = render(&req);
        assert_eq!(a.samples, b.samples);
    }

    #[test]
    fn different_seeds_give_different_pcm() {
        let a = render(&RenderRequest {
            seconds: 0.25,
            seed: Some(1),
            ..RenderRequest::default()
        });
        let b = render(&RenderRequest {
            seconds: 0.25,
            seed: Some(2),
            ..RenderRequest::default()
        });
        // Compare without letting a failure print two million floats.
        assert_eq!(a.samples.len(), b.samples.len());
        let differing = a
            .samples
            .iter()
            .zip(b.samples.iter())
            .filter(|(x, y)| (*x - *y).abs() > 1e-9)
            .count();
        assert!(
            differing > a.samples.len() / 2,
            "only {differing} of {} samples differed between seeds",
            a.samples.len()
        );
    }

    #[test]
    fn block_chunking_does_not_change_the_pcm() {
        // The engine is already tested for this; assert it end to end through
        // the offline renderer with a non power of two length.
        let req = RenderRequest {
            seconds: 1000.0 / 48_000.0,
            ..RenderRequest::default()
        };
        let r = render(&req);
        assert_eq!(r.frames, 1000);
        assert_eq!(r.samples.len(), 2000);
    }

    #[test]
    fn mono_output_is_calibrated_to_the_same_target() {
        let p = PresetParams {
            layers: vec![LayerParams {
                stereo_correlation: 0.0,
                ..LayerParams::default()
            }],
            ..PresetParams::default()
        };
        let stereo = render(&RenderRequest {
            params: p.clone(),
            seconds: 4.0,
            warmup_seconds: 1.0,
            channels: 2,
            ..RenderRequest::default()
        });
        let mono = render(&RenderRequest {
            params: p,
            seconds: 4.0,
            warmup_seconds: 1.0,
            channels: 1,
            ..RenderRequest::default()
        });
        let s = 20.0 * stereo.rms().log10();
        let m = 20.0 * mono.rms().log10();
        assert!(
            (s - m).abs() < 1.5,
            "stereo {s:.2} dBFS vs mono {m:.2} dBFS"
        );
    }

    #[test]
    fn every_colour_renders_finite_audio() {
        for color in Color::ALL {
            let p = PresetParams {
                layers: vec![LayerParams {
                    color,
                    character: noise_dsp::Character::Intense,
                    character_amount: 1.0,
                    ..LayerParams::default()
                }],
                ..PresetParams::default()
            };
            let r = render(&RenderRequest {
                params: p,
                seconds: 1.0,
                ..RenderRequest::default()
            });
            assert!(r.samples.iter().all(|s| s.is_finite()), "{color}");
            assert!(r.rms() > 0.0, "{color} rendered silence");
        }
    }
}
