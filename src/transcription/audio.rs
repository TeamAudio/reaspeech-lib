use crate::common::WorkerContext;
use std::fs::File;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

const WHISPER_SAMPLE_RATE: u32 = 16_000;

pub fn decode_audio_16khz_mono(
    job_id: &str,
    path: &str,
    context: &WorkerContext,
) -> Result<Vec<f32>, String> {
    let file = File::open(path).map_err(|error| format!("Could not open audio file: {error}"))?;
    let stream = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(extension) = std::path::Path::new(path)
        .extension()
        .and_then(|value| value.to_str())
    {
        hint.with_extension(extension);
    }

    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            stream,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|error| format!("Unsupported or unreadable audio file: {error}"))?;
    let mut format = probed.format;
    let track = format
        .default_track()
        .ok_or("The file does not contain an audio track")?;
    let track_id = track.id;
    let sample_rate = track
        .codec_params
        .sample_rate
        .ok_or("The audio sample rate is unknown")?;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|error| format!("Could not initialize the audio decoder: {error}"))?;
    let capacity = track
        .codec_params
        .n_frames
        .and_then(|frames| usize::try_from(frames).ok())
        .unwrap_or(sample_rate as usize * 60);
    let mut mono = Vec::with_capacity(capacity);
    let mut sample_buffer: Option<SampleBuffer<f32>> = None;

    loop {
        if context.cancellation.is_cancelled(job_id) {
            return Err("cancelled".into());
        }
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(SymphoniaError::IoError(error))
                if error.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(error) => return Err(format!("Could not read the audio stream: {error}")),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(SymphoniaError::ResetRequired) => {
                decoder.reset();
                continue;
            }
            Err(error) => return Err(format!("Could not decode the audio stream: {error}")),
        };
        append_mono_samples(decoded, &mut sample_buffer, &mut mono);
    }

    if mono.is_empty() {
        return Err("No decodable audio samples were found".into());
    }
    Ok(resample_linear(&mono, sample_rate, WHISPER_SAMPLE_RATE))
}

fn append_mono_samples(
    decoded: symphonia::core::audio::AudioBufferRef<'_>,
    sample_buffer: &mut Option<SampleBuffer<f32>>,
    mono: &mut Vec<f32>,
) {
    let spec = *decoded.spec();
    let frames = decoded.capacity();
    let buffer = sample_buffer.get_or_insert_with(|| SampleBuffer::new(frames as u64, spec));
    if buffer.capacity() < frames {
        *buffer = SampleBuffer::new(frames as u64, spec);
    }
    buffer.copy_interleaved_ref(decoded);

    let channels = spec.channels.count();
    if channels == 1 {
        mono.extend_from_slice(buffer.samples());
    } else {
        mono.extend(
            buffer
                .samples()
                .chunks(channels)
                .map(|frame| frame.iter().sum::<f32>() / channels as f32),
        );
    }
}

fn resample_linear(samples: &[f32], source_rate: u32, destination_rate: u32) -> Vec<f32> {
    if source_rate == destination_rate {
        return samples.to_vec();
    }
    let ratio = source_rate as f64 / destination_rate as f64;
    let output_length = (samples.len() as f64 / ratio).ceil() as usize;
    (0..output_length)
        .map(|index| {
            let position = index as f64 * ratio;
            let left = (position.floor() as usize).min(samples.len() - 1);
            let right = (left + 1).min(samples.len() - 1);
            samples[left] + (samples[right] - samples[left]) * (position - left as f64) as f32
        })
        .collect()
}
