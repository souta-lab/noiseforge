//! WAV writing (DESIGN.md section 4.5 and section 10 phase 4).
//!
//! Recording and file writing happen off the realtime thread and never block
//! playback: if the disk is slow the *recording* loses data and says so, the
//! audio keeps playing.

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use hound::{SampleFormat, WavSpec, WavWriter};

/// Output sample format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputFormat {
    /// 32 bit float, matching the engine's internal format exactly.
    F32,
    /// 16 bit signed integer.
    I16,
    /// 24 bit signed integer.
    I24,
}

impl OutputFormat {
    /// Bits per sample as written in the WAV header.
    pub fn bits(self) -> u16 {
        match self {
            OutputFormat::F32 => 32,
            OutputFormat::I16 => 16,
            OutputFormat::I24 => 24,
        }
    }

    /// Parse a format name from the CLI.
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "f32" | "float" | "32f" => Some(OutputFormat::F32),
            "i16" | "s16" | "16" => Some(OutputFormat::I16),
            "i24" | "s24" | "24" => Some(OutputFormat::I24),
            _ => None,
        }
    }

    /// File extension usually associated with the format.
    pub fn extension(self) -> &'static str {
        "wav"
    }
}

/// Errors from writing audio.
#[derive(Debug, thiserror::Error)]
pub enum WriteError {
    #[error("wav error: {0}")]
    Wav(#[from] hound::Error),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("no frames to write")]
    Empty,
}

/// Write interleaved samples to a WAV file.
///
/// Samples are clamped to `[-1, 1]` only for integer formats, and the number of
/// clamped samples is returned so the caller can report it honestly rather than
/// hiding the clipping.
pub fn write_wav(
    path: &Path,
    samples: &[f32],
    channels: u16,
    sample_rate: u32,
    format: OutputFormat,
) -> Result<usize, WriteError> {
    if samples.is_empty() {
        return Err(WriteError::Empty);
    }
    let spec = WavSpec {
        channels,
        sample_rate,
        bits_per_sample: format.bits(),
        sample_format: if format == OutputFormat::F32 {
            SampleFormat::Float
        } else {
            SampleFormat::Int
        },
    };
    let file = File::create(path)?;
    let mut writer = WavWriter::new(BufWriter::new(file), spec)?;
    let mut clipped = 0usize;
    match format {
        OutputFormat::F32 => {
            for s in samples {
                writer.write_sample(*s)?;
            }
        }
        OutputFormat::I16 => {
            for s in samples {
                let (v, c) = quantize(*s, i16::MAX as f64);
                clipped += c;
                writer.write_sample(v as i16)?;
            }
        }
        OutputFormat::I24 => {
            for s in samples {
                let (v, c) = quantize(*s, 8_388_607.0);
                clipped += c;
                writer.write_sample(v as i32)?;
            }
        }
    }
    writer.finalize()?;
    Ok(clipped)
}

fn quantize(sample: f32, full_scale: f64) -> (i64, usize) {
    let x = sample as f64;
    if !x.is_finite() {
        return (0, 1);
    }
    if x > 1.0 {
        return (full_scale as i64, 1);
    }
    if x < -1.0 {
        return (-(full_scale as i64) - 1, 1);
    }
    ((x * full_scale).round() as i64, 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_float_wav() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.wav");
        let samples: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.01).sin() * 0.5).collect();
        let clipped = write_wav(&path, &samples, 2, 48_000, OutputFormat::F32).unwrap();
        assert_eq!(clipped, 0);
        let mut reader = hound::WavReader::open(&path).unwrap();
        assert_eq!(reader.spec().channels, 2);
        assert_eq!(reader.spec().sample_rate, 48_000);
        assert_eq!(reader.spec().bits_per_sample, 32);
        let read: Vec<f32> = reader.samples::<f32>().map(|s| s.unwrap()).collect();
        assert_eq!(read.len(), samples.len());
        for (a, b) in read.iter().zip(samples.iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn integer_formats_report_clipping_instead_of_hiding_it() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t16.wav");
        let samples = vec![0.0f32, 1.5, -1.5, 0.25];
        let clipped = write_wav(&path, &samples, 1, 48_000, OutputFormat::I16).unwrap();
        assert_eq!(clipped, 2);
        let mut reader = hound::WavReader::open(&path).unwrap();
        let read: Vec<i16> = reader.samples::<i16>().map(|s| s.unwrap()).collect();
        assert_eq!(read[0], 0);
        assert_eq!(read[1], i16::MAX);
        assert_eq!(read[2], i16::MIN);
    }

    #[test]
    fn empty_buffers_are_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let err = write_wav(&dir.path().join("e.wav"), &[], 1, 48_000, OutputFormat::F32);
        assert!(matches!(err, Err(WriteError::Empty)));
    }

    #[test]
    fn format_parsing() {
        assert_eq!(OutputFormat::parse("f32"), Some(OutputFormat::F32));
        assert_eq!(OutputFormat::parse("s24"), Some(OutputFormat::I24));
        assert_eq!(OutputFormat::parse("mp3"), None);
    }
}
