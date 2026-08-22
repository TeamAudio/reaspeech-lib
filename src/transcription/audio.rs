use crate::common::{emit_progress, WorkerContext};
use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Async, FixedAsync, Indexing, Resampler, SincInterpolationParameters, WindowFunction};
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
    let total_frames = track
        .codec_params
        .n_frames
        .and_then(|frames| usize::try_from(frames).ok());
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|error| format!("Could not initialize the audio decoder: {error}"))?;
    let output_capacity = total_frames
        .map(|frames| frames.saturating_mul(WHISPER_SAMPLE_RATE as usize) / sample_rate as usize)
        .unwrap_or(WHISPER_SAMPLE_RATE as usize * 60);
    let mut audio = StreamingAudio::new(sample_rate, output_capacity)?;
    let mut sample_buffer: Option<SampleBuffer<f32>> = None;
    let mut packet_mono = Vec::new();
    let mut decoded_frames = 0usize;
    let mut last_percent = None;

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
        packet_mono.clear();
        append_mono_samples(decoded, &mut sample_buffer, &mut packet_mono);
        decoded_frames = decoded_frames.saturating_add(packet_mono.len());
        audio.push(&packet_mono)?;

        let percent = total_frames.map(|total| decoded_frames.saturating_mul(100) / total.max(1));
        if percent != last_percent {
            let completed = percent.unwrap_or(0).min(99) as u64;
            let seconds = decoded_frames as f64 / sample_rate as f64;
            emit_progress(
                job_id,
                &format!("Decoding and resampling audio ({seconds:.1} s)"),
                completed,
                100,
            );
            last_percent = percent;
        }
    }

    if decoded_frames == 0 {
        return Err("No decodable audio samples were found".into());
    }
    if context.cancellation.is_cancelled(job_id) {
        return Err("cancelled".into());
    }
    let output = audio.finish()?;
    emit_progress(job_id, "Audio decoded and resampled", 100, 100);
    Ok(output)
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

struct StreamingAudio {
    resampler: Option<Async<f32>>,
    ratio: f64,
    pending: Vec<f32>,
    output: Vec<f32>,
    input_frames: usize,
    delay_remaining: usize,
}

impl StreamingAudio {
    fn new(source_rate: u32, output_capacity: usize) -> Result<Self, String> {
        let ratio = WHISPER_SAMPLE_RATE as f64 / source_rate as f64;
        let parameters = SincInterpolationParameters::new(256, WindowFunction::BlackmanHarris2);
        let resampler = (source_rate != WHISPER_SAMPLE_RATE)
            .then(|| {
                Async::<f32>::new_sinc(ratio, 1.0, &parameters, 1024, 1, FixedAsync::Input)
                    .map_err(|error| format!("Could not initialize the audio resampler: {error}"))
            })
            .transpose()?;
        let delay_remaining = resampler.as_ref().map_or(0, Resampler::output_delay);
        Ok(Self {
            resampler,
            ratio,
            pending: Vec::with_capacity(2048),
            output: Vec::with_capacity(output_capacity),
            input_frames: 0,
            delay_remaining,
        })
    }

    fn push(&mut self, samples: &[f32]) -> Result<(), String> {
        self.input_frames = self.input_frames.saturating_add(samples.len());
        let Some(resampler) = &mut self.resampler else {
            self.output.extend_from_slice(samples);
            return Ok(());
        };

        self.pending.extend_from_slice(samples);
        let output = &mut self.output;
        let delay_remaining = &mut self.delay_remaining;
        let mut consumed = 0;
        while self.pending.len() - consumed >= resampler.input_frames_next() {
            let needed = resampler.input_frames_next();
            let input = InterleavedSlice::new(&self.pending, 1, self.pending.len())
                .map_err(|error| format!("Could not prepare audio for resampling: {error}"))?;
            let indexing = Indexing {
                input_offset: consumed,
                ..Default::default()
            };
            let chunk = resampler
                .process(&input, Some(&indexing))
                .map_err(|error| format!("Could not resample audio: {error}"))?;
            Self::append_resampled(output, delay_remaining, &chunk.take_data());
            consumed += needed;
        }
        self.pending.drain(..consumed);
        Ok(())
    }

    fn finish(mut self) -> Result<Vec<f32>, String> {
        let Some(mut resampler) = self.resampler.take() else {
            return Ok(self.output);
        };

        if !self.pending.is_empty() {
            let partial_len = self.pending.len();
            let input = InterleavedSlice::new(&self.pending, 1, partial_len)
                .map_err(|error| format!("Could not prepare final audio chunk: {error}"))?;
            let indexing = Indexing {
                partial_len: Some(partial_len),
                ..Default::default()
            };
            let chunk = resampler
                .process(&input, Some(&indexing))
                .map_err(|error| format!("Could not resample final audio chunk: {error}"))?;
            Self::append_resampled(
                &mut self.output,
                &mut self.delay_remaining,
                &chunk.take_data(),
            );
        }

        let expected = (self.input_frames as f64 * self.ratio).ceil() as usize;
        while self.output.len() < expected {
            let needed = resampler.input_frames_next();
            let silence = vec![0.0; needed];
            let input = InterleavedSlice::new(&silence, 1, needed)
                .map_err(|error| format!("Could not flush audio resampler: {error}"))?;
            let indexing = Indexing {
                partial_len: Some(0),
                ..Default::default()
            };
            let chunk = resampler
                .process(&input, Some(&indexing))
                .map_err(|error| format!("Could not flush audio resampler: {error}"))?;
            Self::append_resampled(
                &mut self.output,
                &mut self.delay_remaining,
                &chunk.take_data(),
            );
        }
        self.output.truncate(expected);
        Ok(self.output)
    }

    fn append_resampled(output: &mut Vec<f32>, delay_remaining: &mut usize, samples: &[f32]) {
        let skip = (*delay_remaining).min(samples.len());
        *delay_remaining -= skip;
        output.extend_from_slice(&samples[skip..]);
    }
}

#[cfg(test)]
mod tests {
    use super::StreamingAudio;

    fn stream_in_chunks(samples: &[f32], source_rate: u32, chunk: usize) -> Vec<f32> {
        let mut audio = StreamingAudio::new(source_rate, samples.len()).unwrap();
        for part in samples.chunks(chunk) {
            audio.push(part).unwrap();
        }
        audio.finish().unwrap()
    }

    #[test]
    fn resampling_at_the_same_rate_preserves_samples() {
        let samples = vec![-1.0, -0.25, 0.5, 1.0];

        assert_eq!(stream_in_chunks(&samples, 16_000, 3), samples);
    }

    #[test]
    fn windowed_sinc_resampling_preserves_clip_duration() {
        let samples: Vec<f32> = (0..48_000)
            .map(|index| (index as f32 * std::f32::consts::TAU * 440.0 / 48_000.0).sin())
            .collect();

        let output = stream_in_chunks(&samples, 48_000, 733);

        assert_eq!(output.len(), 16_000);
        assert!(output.iter().all(|sample| sample.is_finite()));
    }

    #[test]
    fn streaming_resampling_is_independent_of_packet_boundaries() {
        let samples: Vec<f32> = (0..48_123)
            .map(|index| (index as f32 * std::f32::consts::TAU * 440.0 / 48_000.0).sin())
            .collect();

        let small_packets = stream_in_chunks(&samples, 48_000, 137);
        let large_packets = stream_in_chunks(&samples, 48_000, 4093);

        assert_eq!(small_packets.len(), 16_041);
        assert_eq!(small_packets, large_packets);
    }
}
