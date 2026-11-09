//! PipeWire format negotiation (DESIGN.md section 4.2).
//!
//! The engine always produces `f32`, but the *rate*, the *channel count* and the
//! *buffer layout* are whatever the server negotiates. Nothing here assumes
//! 48 kHz stereo; the negotiated result is reported back to the daemon so it can
//! redesign coefficients outside the realtime thread.

use pipewire as pw;
use pw::spa;
use pw::spa::pod::Pod;

/// Audio sample format actually negotiated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleFormat {
    /// 32 bit float, interleaved.
    F32Le,
    /// 32 bit float, planar.
    F32Planar,
}

impl SampleFormat {
    /// Bytes per sample.
    pub const fn bytes(self) -> usize {
        4
    }

    /// Whether the samples are split into one buffer per channel.
    pub const fn is_planar(self) -> bool {
        matches!(self, SampleFormat::F32Planar)
    }
}

/// What the server gave us.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NegotiatedFormat {
    /// Sample rate in Hz.
    pub rate: u32,
    /// Channel count.
    pub channels: u32,
    /// Sample format.
    pub format: SampleFormat,
    /// Output layout, i.e. one block per channel.
    pub planar: bool,
}

impl Default for NegotiatedFormat {
    fn default() -> Self {
        Self {
            rate: 48_000,
            channels: 2,
            format: SampleFormat::F32Le,
            planar: false,
        }
    }
}

impl NegotiatedFormat {
    /// Bytes per frame.
    pub fn frame_bytes(&self) -> usize {
        self.format.bytes() * self.channels as usize
    }
}

/// Parse the `Format` parameter the stream was configured with.
pub fn parse_format(param: &Pod) -> Option<NegotiatedFormat> {
    let (media_type, media_subtype) = spa::param::format_utils::parse_format(param).ok()?;
    if media_type != spa::param::format::MediaType::Audio
        || media_subtype != spa::param::format::MediaSubtype::Raw
    {
        return None;
    }
    let mut info = spa::param::audio::AudioInfoRaw::new();
    info.parse(param).ok()?;
    let format = match info.format() {
        spa::param::audio::AudioFormat::F32LE => SampleFormat::F32Le,
        spa::param::audio::AudioFormat::F32P => SampleFormat::F32Planar,
        other => {
            tracing::warn!(
                "server negotiated {:?}, only f32 is supported; treating it as F32LE",
                other
            );
            SampleFormat::F32Le
        }
    };
    let channels = info.channels().clamp(1, 8);
    Some(NegotiatedFormat {
        rate: info.rate().max(8_000),
        channels,
        format,
        planar: format.is_planar(),
    })
}

/// Build the `EnumFormat` parameter list to offer to the server.
///
/// Only `f32` is offered: DESIGN.md section 3 fixes "F32LE PCM Output", and
/// PipeWire converts to whatever the device wants. The rate is offered as a
/// choice when the policy is to follow the server, so it can pick the device's
/// native rate instead of forcing a resample.
pub fn enum_format_params(
    sample_rate: Option<u32>,
    channels: u32,
    _quantum: Option<u32>,
) -> Result<Vec<u8>, pw::Error> {
    use pw::spa::param::audio::AudioFormat;
    use pw::spa::param::format::{FormatProperties, MediaSubtype, MediaType};
    use pw::spa::pod::{self, Value};
    use pw::spa::utils::{Id, SpaTypes};

    let channels = channels.clamp(1, 8) as i32;
    let object = match sample_rate {
        Some(rate) => pod::object!(
            SpaTypes::ObjectParamFormat,
            pw::spa::param::ParamType::EnumFormat,
            pod::property!(FormatProperties::MediaType, Id, MediaType::Audio),
            pod::property!(FormatProperties::MediaSubtype, Id, MediaSubtype::Raw),
            pod::property!(FormatProperties::AudioFormat, Id, AudioFormat::F32LE),
            pod::property!(FormatProperties::AudioRate, Value::Int(rate as i32)),
            pod::property!(FormatProperties::AudioChannels, Value::Int(channels)),
        ),
        None => pod::object!(
            SpaTypes::ObjectParamFormat,
            pw::spa::param::ParamType::EnumFormat,
            pod::property!(FormatProperties::MediaType, Id, MediaType::Audio),
            pod::property!(FormatProperties::MediaSubtype, Id, MediaSubtype::Raw),
            pod::property!(FormatProperties::AudioFormat, Id, AudioFormat::F32LE),
            // Preferred first, then the other rates the designer supports
            // (DESIGN.md section 4.2 names 44.1, 48 and 96 kHz).
            pod::property!(
                FormatProperties::AudioRate,
                Value::Choice(pod::ChoiceValue::Int(pw::spa::utils::Choice(
                    pw::spa::utils::ChoiceFlags::empty(),
                    pw::spa::utils::ChoiceEnum::<i32>::Enum {
                        default: 48_000,
                        alternatives: vec![44_100, 96_000, 88_200, 192_000, 32_000],
                    }
                )))
            ),
            pod::property!(FormatProperties::AudioChannels, Value::Int(channels)),
        ),
    };

    let bytes: Vec<u8> = pod::serialize::PodSerializer::serialize(
        std::io::Cursor::new(Vec::new()),
        &Value::Object(object),
    )
    .map_err(|_| pw::Error::CreationFailed)?
    .0
    .into_inner();
    let _ = Id(0);
    Ok(bytes)
}

/// Human readable channel layout label.
pub fn channel_layout_name(channels: u32) -> &'static str {
    match channels {
        1 => "mono",
        2 => "stereo",
        3 => "2.1",
        4 => "quad",
        6 => "5.1",
        8 => "7.1",
        _ => "multichannel",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_format_is_stereo_48k() {
        let f = NegotiatedFormat::default();
        assert_eq!(f.rate, 48_000);
        assert_eq!(f.channels, 2);
        assert_eq!(f.frame_bytes(), 8);
        assert!(!f.planar);
    }

    #[test]
    fn channel_layout_labels() {
        assert_eq!(channel_layout_name(1), "mono");
        assert_eq!(channel_layout_name(2), "stereo");
        assert_eq!(channel_layout_name(5), "multichannel");
    }
}
