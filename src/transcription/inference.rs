use super::emit_stage;
use super::models::ModelBundle;
use super::vad::SileroVad;
use crate::common::{Cancellation, WorkerContext};
use byteorder::{ByteOrder, LittleEndian};
use candle_core::{Device, IndexOp, Tensor};
use candle_nn::{ops::softmax, VarBuilder};
use candle_transformers::models::whisper::{self as whisper, audio, Config};
use std::fs;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::time::Instant;
use tokenizers::Tokenizer;

const SAMPLE_RATE: usize = 16_000;
const VAD_FRAME_SAMPLES: usize = 512;
const VAD_THRESHOLD: f32 = 0.5;
const MIN_SPEECH_SAMPLES: usize = SAMPLE_RATE / 4;
const MIN_SILENCE_SAMPLES: usize = SAMPLE_RATE / 10;
const SPEECH_PAD_SAMPLES: usize = SAMPLE_RATE * 30 / 1000;
const DEFAULT_BEAM_SIZE: usize = 1;

pub struct TranscriptSegment {
    pub start_ms: i64,
    pub end_ms: i64,
    pub text: String,
    pub confidence: f32,
}

#[derive(Clone, Copy)]
struct SpeechRegion {
    start: usize,
    end: usize,
}

#[derive(Clone)]
struct Beam {
    model: whisper::model::Whisper,
    tokens: Vec<u32>,
    score: f64,
    finished: bool,
}

struct Decoder {
    model: whisper::model::Whisper,
    tokenizer: Tokenizer,
    suppress_tokens: Vec<bool>,
    sot_token: u32,
    transcribe_token: u32,
    translate_token: u32,
    eot_token: u32,
    no_speech_token: u32,
    no_timestamps_token: u32,
    language_token: Option<u32>,
    translate: bool,
    beam_size: usize,
}

struct DecodedPiece {
    start_seconds: f32,
    end_seconds: f32,
    text: String,
}

pub fn transcribe(
    job_id: &str,
    pcm: &[f32],
    bundle: &ModelBundle,
    vad_model: Option<&Path>,
    language: Option<&str>,
    translate: bool,
    context: &WorkerContext,
) -> Result<Vec<TranscriptSegment>, String> {
    let started = Instant::now();
    emit_stage(job_id, "Loading Model", 0);
    let device = inference_device(job_id)?;
    log_job(
        job_id,
        &format!(
            "using {} with beam size {}",
            device_name(&device),
            configured_beam_size()
        ),
    );
    let config: Config = serde_json::from_str(
        &fs::read_to_string(&bundle.config)
            .map_err(|error| format!("Could not read Whisper configuration: {error}"))?,
    )
    .map_err(|error| format!("Invalid Whisper configuration: {error}"))?;
    let tokenizer = Tokenizer::from_file(&bundle.tokenizer)
        .map_err(|error| format!("Could not load Whisper tokenizer: {error}"))?;
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            std::slice::from_ref(&bundle.weights),
            whisper::DTYPE,
            &device,
        )
    }
    .map_err(|error| format!("Could not load Whisper weights: {error}"))?;
    let model = whisper::model::Whisper::load(&vb, config.clone())
        .map_err(|error| format!("Could not initialize Whisper: {error}"))?;
    log_job(
        job_id,
        &format!("model loaded in {:.2?}", started.elapsed()),
    );
    let mel_filters = read_mel_filters(bundle, config.num_mel_bins)?;
    ensure_not_cancelled(job_id, &context.cancellation)?;

    emit_stage(job_id, "Detecting Speech", 0);
    let regions = if let Some(vad_model) = vad_model {
        detect_speech(pcm, vad_model, job_id, &context.cancellation)?
    } else {
        vec![SpeechRegion {
            start: 0,
            end: pcm.len(),
        }]
    };
    log_job(
        job_id,
        &format!(
            "VAD selected {} region(s) after {:.2?}",
            regions.len(),
            started.elapsed()
        ),
    );
    if regions.is_empty() {
        return Ok(Vec::new());
    }

    let first_mel = mel_tensor(
        &config,
        &mel_filters,
        &pcm[regions[0].start..regions[0].end],
        &device,
    )?;
    let language_token = match language {
        Some(code) => Some(token_id(&tokenizer, &format!("<|{code}|>"))?),
        None => Some(detect_language(model.clone(), &tokenizer, &first_mel)?),
    };
    let mut decoder = Decoder::new(model, tokenizer, language_token, translate)?;

    emit_stage(job_id, "Transcribing", 0);
    let total_samples: usize = regions.iter().map(|region| region.end - region.start).sum();
    let chunk_count: usize = regions
        .iter()
        .copied()
        .map(split_region)
        .map(|chunks| chunks.len())
        .sum();
    let mut chunk_index = 0usize;
    let mut completed_samples = 0usize;
    let mut segments = Vec::new();
    for region in regions {
        for chunk in split_region(region) {
            chunk_index += 1;
            ensure_not_cancelled(job_id, &context.cancellation)?;
            let chunk_started = Instant::now();
            log_job(
                job_id,
                &format!(
                    "chunk {chunk_index}/{chunk_count}: {:.2}s of speech",
                    (chunk.end - chunk.start) as f64 / SAMPLE_RATE as f64
                ),
            );
            emit_stage(
                job_id,
                &format!("Transcribing {chunk_index}/{chunk_count}: mel"),
                completed_samples.saturating_mul(100) as u64 / total_samples.max(1) as u64,
            );
            let mel = mel_tensor(&config, &mel_filters, &pcm[chunk.start..chunk.end], &device)?;
            let (pieces, confidence, no_speech) = decoder.decode(
                &mel,
                chunk.end - chunk.start,
                chunk_index,
                chunk_count,
                completed_samples,
                total_samples,
                job_id,
                &context.cancellation,
            )?;
            log_job(
                job_id,
                &format!(
                    "chunk {chunk_index}/{chunk_count} decoded in {:.2?}",
                    chunk_started.elapsed()
                ),
            );
            if no_speech < whisper::NO_SPEECH_THRESHOLD {
                for piece in pieces {
                    if piece.text.trim().is_empty() {
                        continue;
                    }
                    segments.push(TranscriptSegment {
                        start_ms: samples_to_ms(chunk.start)
                            + (piece.start_seconds * 1000.0).round() as i64,
                        end_ms: (samples_to_ms(chunk.start)
                            + (piece.end_seconds * 1000.0).round() as i64)
                            .min(samples_to_ms(chunk.end)),
                        text: piece.text.trim().to_owned(),
                        confidence,
                    });
                }
            }
            completed_samples += chunk.end - chunk.start;
            emit_stage(
                job_id,
                "Transcribing",
                completed_samples.saturating_mul(100) as u64 / total_samples.max(1) as u64,
            );
        }
    }
    Ok(segments)
}

impl Decoder {
    fn new(
        model: whisper::model::Whisper,
        tokenizer: Tokenizer,
        language_token: Option<u32>,
        translate: bool,
    ) -> Result<Self, String> {
        let no_timestamps_token = token_id(&tokenizer, whisper::NO_TIMESTAMPS_TOKEN)?;
        let no_speech_token = whisper::NO_SPEECH_TOKENS
            .iter()
            .find_map(|token| tokenizer.token_to_id(token))
            .ok_or("Whisper tokenizer has no no-speech token")?;
        let mut suppress_tokens = vec![false; model.config.vocab_size];
        for &token in &model.config.suppress_tokens {
            if let Some(entry) = suppress_tokens.get_mut(token as usize) {
                *entry = true;
            }
        }
        suppress_tokens[no_timestamps_token as usize] = true;
        Ok(Self {
            model,
            sot_token: token_id(&tokenizer, whisper::SOT_TOKEN)?,
            transcribe_token: token_id(&tokenizer, whisper::TRANSCRIBE_TOKEN)?,
            translate_token: token_id(&tokenizer, whisper::TRANSLATE_TOKEN)?,
            eot_token: token_id(&tokenizer, whisper::EOT_TOKEN)?,
            no_speech_token,
            no_timestamps_token,
            tokenizer,
            suppress_tokens,
            language_token,
            translate,
            beam_size: configured_beam_size(),
        })
    }

    fn decode(
        &mut self,
        mel: &Tensor,
        audio_samples: usize,
        chunk_index: usize,
        chunk_count: usize,
        completed_samples: usize,
        total_samples: usize,
        job_id: &str,
        cancellation: &Cancellation,
    ) -> Result<(Vec<DecodedPiece>, f32, f64), String> {
        self.model.reset_kv_cache();
        let encoder_started = Instant::now();
        emit_stage(
            job_id,
            &format!("Transcribing {chunk_index}/{chunk_count}: encoder"),
            completed_samples.saturating_mul(100) as u64 / total_samples.max(1) as u64,
        );
        let audio_features = self
            .model
            .encoder
            .forward(mel, true)
            .map_err(candle_error)?;
        log_job(
            job_id,
            &format!("encoder finished in {:.2?}", encoder_started.elapsed()),
        );
        let mut prefix = vec![self.sot_token];
        if let Some(language) = self.language_token {
            prefix.push(language);
        }
        prefix.push(if self.translate {
            self.translate_token
        } else {
            self.transcribe_token
        });
        let prefix_len = prefix.len();

        let prefix_tensor = Tensor::new(prefix.as_slice(), mel.device())
            .and_then(|tensor| tensor.unsqueeze(0))
            .map_err(candle_error)?;
        let ys = self
            .model
            .decoder
            .forward(&prefix_tensor, &audio_features, true)
            .map_err(candle_error)?;
        let no_speech_logits = self
            .model
            .decoder
            .final_linear(&ys.i(..1).map_err(candle_error)?)
            .and_then(|tensor| tensor.i(0))
            .and_then(|tensor| tensor.i(0))
            .map_err(candle_error)?;
        let no_speech = softmax(&no_speech_logits, 0)
            .and_then(|tensor| tensor.i(self.no_speech_token as usize))
            .and_then(|tensor| tensor.to_scalar::<f32>())
            .map_err(candle_error)? as f64;
        let logits =
            self.apply_timestamp_rules(last_logits(&self.model, &ys)?, &prefix, prefix_len)?;
        let mut beams = expand_beam(
            Beam {
                model: self.model.clone(),
                tokens: prefix,
                score: 0.0,
                finished: false,
            },
            logits,
            self.beam_size,
            self.eot_token,
            &self.suppress_tokens,
        )?;

        let audio_seconds = audio_samples as f64 / SAMPLE_RATE as f64;
        let max_steps = ((audio_seconds * 8.0).ceil() as usize + 16)
            .clamp(24, self.model.config.max_target_positions / 2);
        for _ in 1..max_steps {
            ensure_not_cancelled(job_id, cancellation)?;
            if beams.iter().all(|beam| beam.finished) {
                break;
            }
            let mut candidates = Vec::with_capacity(self.beam_size * self.beam_size);
            for mut beam in beams {
                if beam.finished {
                    candidates.push(beam);
                    continue;
                }
                // Candle's Whisper decoder has no self-attention KV cache and
                // always applies positional embeddings starting at position 0.
                // Re-submit the full prefix on every step, as the upstream
                // Candle example does, so each token gets the correct position.
                let input = Tensor::new(beam.tokens.as_slice(), mel.device())
                    .and_then(|tensor| tensor.unsqueeze(0))
                    .map_err(candle_error)?;
                let ys = beam
                    .model
                    .decoder
                    .forward(&input, &audio_features, false)
                    .map_err(candle_error)?;
                let logits = self.apply_timestamp_rules(
                    last_logits(&beam.model, &ys)?,
                    &beam.tokens,
                    prefix_len,
                )?;
                candidates.extend(expand_beam(
                    beam,
                    logits,
                    self.beam_size,
                    self.eot_token,
                    &self.suppress_tokens,
                )?);
            }
            candidates.sort_by(|left, right| right.score.total_cmp(&left.score));
            candidates.truncate(self.beam_size);
            beams = candidates;
            let generated = beams
                .first()
                .map(|beam| {
                    beam.tokens
                        .len()
                        .saturating_sub(prefix_tensor.dim(1).unwrap_or(0))
                })
                .unwrap_or(0);
            if generated == 1 || generated % 4 == 0 {
                let within_chunk = generated.saturating_mul(audio_samples) / max_steps.max(1);
                emit_stage(
                    job_id,
                    &format!(
                        "Transcribing {chunk_index}/{chunk_count}: token {generated}/{max_steps}"
                    ),
                    (completed_samples + within_chunk).saturating_mul(100) as u64
                        / total_samples.max(1) as u64,
                );
                log_job(job_id, &format!("decoded token {generated}/{max_steps}"));
            }
        }

        let best = beams
            .into_iter()
            .max_by(|left, right| normalized_score(left).total_cmp(&normalized_score(right)))
            .ok_or("Whisper beam search returned no candidates")?;
        self.model = best.model.clone();
        let generated = &best.tokens[prefix_len..];
        let pieces = self.timestamped_pieces(generated, audio_seconds as f32)?;
        let token_count = generated.len().max(1) as f64;
        Ok((
            pieces,
            (best.score / token_count).exp().clamp(0.0, 1.0) as f32,
            no_speech,
        ))
    }

    fn apply_timestamp_rules(
        &self,
        logits: Tensor,
        tokens: &[u32],
        prefix_len: usize,
    ) -> Result<Tensor, String> {
        let mut values = logits.to_vec1::<f32>().map_err(candle_error)?;
        let timestamp_begin = self.no_timestamps_token + 1;
        let sampled = &tokens[prefix_len.min(tokens.len())..];
        let last_is_timestamp = sampled
            .last()
            .is_some_and(|token| *token >= timestamp_begin);
        let previous_is_timestamp = sampled
            .get(sampled.len().saturating_sub(2))
            .is_some_and(|token| *token >= timestamp_begin);

        if last_is_timestamp {
            if previous_is_timestamp {
                for value in values.iter_mut().skip(timestamp_begin as usize) {
                    *value = f32::NEG_INFINITY;
                }
            } else {
                for value in values.iter_mut().take(self.eot_token as usize) {
                    *value = f32::NEG_INFINITY;
                }
            }
        }

        if let Some(last_timestamp) = sampled
            .iter()
            .rev()
            .find(|token| **token >= timestamp_begin)
            .copied()
        {
            let minimum = if last_is_timestamp && !previous_is_timestamp {
                last_timestamp
            } else {
                last_timestamp + 1
            };
            for value in values
                .iter_mut()
                .take(minimum as usize)
                .skip(timestamp_begin as usize)
            {
                *value = f32::NEG_INFINITY;
            }
        }

        if sampled.is_empty() {
            for value in values.iter_mut().take(timestamp_begin as usize) {
                *value = f32::NEG_INFINITY;
            }
            // Match faster-whisper's default one-second maximum initial timestamp.
            for value in values.iter_mut().skip((timestamp_begin + 50) as usize + 1) {
                *value = f32::NEG_INFINITY;
            }
        }

        let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let denominator: f64 = values
            .iter()
            .map(|value| ((*value - max) as f64).exp())
            .sum();
        let timestamp_probability: f64 = values
            .iter()
            .skip(timestamp_begin as usize)
            .map(|value| ((*value - max) as f64).exp() / denominator)
            .sum();
        let best_text_probability = values[..timestamp_begin as usize]
            .iter()
            .map(|value| ((*value - max) as f64).exp() / denominator)
            .fold(0.0, f64::max);
        if timestamp_probability > best_text_probability {
            for value in values.iter_mut().take(timestamp_begin as usize) {
                *value = f32::NEG_INFINITY;
            }
        }
        Tensor::new(values.as_slice(), logits.device()).map_err(candle_error)
    }

    fn timestamped_pieces(
        &self,
        tokens: &[u32],
        audio_seconds: f32,
    ) -> Result<Vec<DecodedPiece>, String> {
        let timestamp_begin = self.no_timestamps_token + 1;
        let mut pieces = Vec::new();
        let mut text_tokens = Vec::new();
        let mut start_seconds = 0.0f32;
        for &token in tokens {
            if token == self.eot_token {
                break;
            }
            if token >= timestamp_begin {
                let time = ((token - timestamp_begin) as f32 / 50.0).min(audio_seconds);
                if !text_tokens.is_empty() && time > start_seconds {
                    let text = self
                        .tokenizer
                        .decode(&text_tokens, true)
                        .map_err(|error| error.to_string())?;
                    pieces.push(DecodedPiece {
                        start_seconds,
                        end_seconds: time,
                        text,
                    });
                    text_tokens.clear();
                }
                start_seconds = time;
            } else {
                text_tokens.push(token);
            }
        }
        if !text_tokens.is_empty() {
            let text = self
                .tokenizer
                .decode(&text_tokens, true)
                .map_err(|error| error.to_string())?;
            pieces.push(DecodedPiece {
                start_seconds,
                end_seconds: audio_seconds,
                text,
            });
        }
        Ok(pieces)
    }
}

fn expand_beam(
    beam: Beam,
    logits: Tensor,
    count: usize,
    eot_token: u32,
    suppressed: &[bool],
) -> Result<Vec<Beam>, String> {
    let mut logits = logits.to_vec1::<f32>().map_err(candle_error)?;
    for (index, value) in logits.iter_mut().enumerate() {
        if suppressed.get(index).copied().unwrap_or(false) {
            *value = f32::NEG_INFINITY;
        }
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f64 = logits
        .iter()
        .map(|value| ((*value - max) as f64).exp())
        .sum();
    let log_denom = max as f64 + sum.ln();
    let mut ranked: Vec<(usize, f32)> = logits.into_iter().enumerate().collect();
    ranked.sort_by(|left, right| right.1.total_cmp(&left.1));
    Ok(ranked
        .into_iter()
        .take(count)
        .map(|(token, logit)| {
            let mut next = beam.clone();
            next.tokens.push(token as u32);
            next.score += logit as f64 - log_denom;
            next.finished = token as u32 == eot_token;
            next
        })
        .collect())
}

fn normalized_score(beam: &Beam) -> f64 {
    beam.score / beam.tokens.len().max(1) as f64
}

fn last_logits(model: &whisper::model::Whisper, ys: &Tensor) -> Result<Tensor, String> {
    let (_, length, _) = ys.dims3().map_err(candle_error)?;
    model
        .decoder
        .final_linear(&ys.i((..1, length - 1..)).map_err(candle_error)?)
        .and_then(|tensor| tensor.i(0))
        .and_then(|tensor| tensor.i(0))
        .map_err(candle_error)
}

fn detect_language(
    mut model: whisper::model::Whisper,
    tokenizer: &Tokenizer,
    mel: &Tensor,
) -> Result<u32, String> {
    const LANGUAGE_CODES: &[&str] = &[
        "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv",
        "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no",
        "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr",
        "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
        "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu",
        "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl",
        "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su",
    ];
    let ids: Vec<u32> = LANGUAGE_CODES
        .iter()
        .filter_map(|code| tokenizer.token_to_id(&format!("<|{code}|>")))
        .collect();
    let frames = mel
        .dim(2)
        .map_err(candle_error)?
        .min(model.config.max_source_positions);
    let features = model
        .encoder
        .forward(&mel.narrow(2, 0, frames).map_err(candle_error)?, true)
        .map_err(candle_error)?;
    let sot = token_id(tokenizer, whisper::SOT_TOKEN)?;
    let tokens = Tensor::new(&[[sot]], mel.device()).map_err(candle_error)?;
    let ys = model
        .decoder
        .forward(&tokens, &features, true)
        .map_err(candle_error)?;
    let logits = last_logits(&model, &ys)?
        .to_vec1::<f32>()
        .map_err(candle_error)?;
    ids.into_iter()
        .max_by(|left, right| logits[*left as usize].total_cmp(&logits[*right as usize]))
        .ok_or_else(|| "Whisper tokenizer has no language tokens".into())
}

fn detect_speech(
    pcm: &[f32],
    vad_model: &Path,
    job_id: &str,
    cancellation: &Cancellation,
) -> Result<Vec<SpeechRegion>, String> {
    let mut vad = SileroVad::from_file(vad_model)?;
    let mut regions = Vec::new();
    let mut speech_start = None;
    let mut silence_start = None;
    for (index, frame) in pcm.chunks(VAD_FRAME_SAMPLES).enumerate() {
        ensure_not_cancelled(job_id, cancellation)?;
        let mut padded = [0f32; VAD_FRAME_SAMPLES];
        padded[..frame.len()].copy_from_slice(frame);
        let probability = vad.process(&padded)?;
        let position = index * VAD_FRAME_SAMPLES;
        if probability >= VAD_THRESHOLD {
            speech_start.get_or_insert(position);
            silence_start = None;
        } else if let Some(start) = speech_start {
            let silence = *silence_start.get_or_insert(position);
            if position.saturating_sub(silence) >= MIN_SILENCE_SAMPLES {
                push_speech_region(&mut regions, start, silence, pcm.len());
                speech_start = None;
                silence_start = None;
            }
        }
    }
    if let Some(start) = speech_start {
        push_speech_region(&mut regions, start, pcm.len(), pcm.len());
    }
    Ok(group_speech_regions(merge_regions(regions)))
}

fn push_speech_region(regions: &mut Vec<SpeechRegion>, start: usize, end: usize, audio_len: usize) {
    if end.saturating_sub(start) >= MIN_SPEECH_SAMPLES {
        regions.push(SpeechRegion {
            start: start.saturating_sub(SPEECH_PAD_SAMPLES),
            end: (end + SPEECH_PAD_SAMPLES).min(audio_len),
        });
    }
}

fn merge_regions(regions: Vec<SpeechRegion>) -> Vec<SpeechRegion> {
    let mut merged: Vec<SpeechRegion> = Vec::new();
    for region in regions {
        if let Some(previous) = merged.last_mut() {
            if region.start <= previous.end {
                previous.end = previous.end.max(region.end);
                continue;
            }
        }
        merged.push(region);
    }
    merged
}

/// Turn the short islands emitted by Silero into Whisper-sized context windows.
/// Whisper is a sequence model and produces poor results when individual words
/// are decoded independently. Gaps inside each window are intentionally kept so
/// punctuation and sentence context remain available to the model.
fn group_speech_regions(regions: Vec<SpeechRegion>) -> Vec<SpeechRegion> {
    let mut grouped: Vec<SpeechRegion> = Vec::new();
    for region in regions {
        if let Some(window) = grouped.last_mut() {
            if region.end.saturating_sub(window.start) <= whisper::N_SAMPLES {
                window.end = region.end;
                continue;
            }
        }
        grouped.push(region);
    }
    grouped
}

fn split_region(region: SpeechRegion) -> Vec<SpeechRegion> {
    let mut chunks = Vec::new();
    let mut start = region.start;
    while start < region.end {
        let end = (start + whisper::N_SAMPLES).min(region.end);
        chunks.push(SpeechRegion { start, end });
        start = end;
    }
    chunks
}

fn mel_tensor(
    config: &Config,
    filters: &[f32],
    pcm: &[f32],
    device: &Device,
) -> Result<Tensor, String> {
    let mel = audio::pcm_to_mel(config, pcm, filters);
    let available_frames = mel.len() / config.num_mel_bins;
    let frames = available_frames.min(whisper::N_FRAMES);
    let values = if frames < available_frames {
        let mut truncated = Vec::with_capacity(config.num_mel_bins * frames);
        for bin in mel.chunks_exact(available_frames) {
            truncated.extend_from_slice(&bin[..frames]);
        }
        truncated
    } else {
        mel
    };
    Tensor::from_vec(values, (1, config.num_mel_bins, frames), device).map_err(candle_error)
}

fn read_mel_filters(bundle: &ModelBundle, num_mel_bins: usize) -> Result<Vec<f32>, String> {
    let path = match num_mel_bins {
        80 => &bundle.mel_filters_80,
        128 => &bundle.mel_filters_128,
        count => {
            return Err(format!(
                "Whisper model uses unsupported {count}-bin mel filters"
            ))
        }
    };
    let bytes = fs::read(path).map_err(|error| format!("Could not read mel filters: {error}"))?;
    if bytes.len() % 4 != 0 {
        return Err("Invalid mel filter data".into());
    }
    let mut filters = vec![0f32; bytes.len() / 4];
    LittleEndian::read_f32_into(&bytes, &mut filters);
    Ok(filters)
}

fn inference_device(job_id: &str) -> Result<Device, String> {
    let _ = job_id;
    if std::env::var_os("REASPEECH_FORCE_CPU").is_some() {
        return Ok(Device::Cpu);
    }
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        if std::env::var_os("REASPEECH_FORCE_METAL").is_none() {
            if !metal_supports_bfloat() {
                log_job(
                    job_id,
                    "Metal GEMV requires Apple GPU family 9 or newer; falling back to CPU",
                );
                emit_stage(job_id, "Metal unsupported; using CPU", 0);
                return Ok(Device::Cpu);
            }
        }
        return Device::new_metal(0)
            .map_err(|error| format!("Could not initialize Metal: {error}"));
    }
    #[cfg(all(feature = "cuda", not(all(feature = "metal", target_os = "macos"))))]
    return Device::new_cuda(0).map_err(|error| format!("Could not initialize CUDA: {error}"));
    #[allow(unreachable_code)]
    Ok(Device::Cpu)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn metal_supports_bfloat() -> bool {
    use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice, MTLGPUFamily};

    MTLCreateSystemDefaultDevice().is_some_and(|device| device.supportsFamily(MTLGPUFamily::Apple9))
}

fn configured_beam_size() -> usize {
    std::env::var("REASPEECH_BEAM_SIZE")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|size: &usize| (1..=5).contains(size))
        .unwrap_or(DEFAULT_BEAM_SIZE)
}

fn device_name(device: &Device) -> &'static str {
    match device {
        Device::Cpu => "CPU",
        Device::Cuda(_) => "CUDA",
        Device::Metal(_) => "Metal",
    }
}

fn log_job(job_id: &str, message: &str) {
    let line = format!("[transcription:{job_id}] {message}");
    eprintln!("{line}");
    if let Some(path) = std::env::var_os("REASPEECH_LOG_PATH") {
        if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
            let _ = writeln!(file, "{line}");
        }
    }
}

fn token_id(tokenizer: &Tokenizer, token: &str) -> Result<u32, String> {
    tokenizer
        .token_to_id(token)
        .ok_or_else(|| format!("Whisper tokenizer has no {token} token"))
}

fn ensure_not_cancelled(job_id: &str, cancellation: &Cancellation) -> Result<(), String> {
    if cancellation.is_cancelled(job_id) {
        Err("cancelled".into())
    } else {
        Ok(())
    }
}

fn samples_to_ms(samples: usize) -> i64 {
    samples.saturating_mul(1000) as i64 / SAMPLE_RATE as i64
}

fn candle_error(error: candle_core::Error) -> String {
    error.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groups_short_vad_islands_into_one_context_window() {
        let regions = vec![
            SpeechRegion {
                start: 100,
                end: 500,
            },
            SpeechRegion {
                start: 2_000,
                end: 3_000,
            },
            SpeechRegion {
                start: 10_000,
                end: 12_000,
            },
        ];

        let grouped = group_speech_regions(regions);

        assert_eq!(grouped.len(), 1);
        assert_eq!(grouped[0].start, 100);
        assert_eq!(grouped[0].end, 12_000);
    }

    #[test]
    fn starts_a_new_context_window_past_thirty_seconds() {
        let regions = vec![
            SpeechRegion {
                start: 0,
                end: 1_000,
            },
            SpeechRegion {
                start: whisper::N_SAMPLES + 1,
                end: whisper::N_SAMPLES + 2_000,
            },
        ];

        assert_eq!(group_speech_regions(regions).len(), 2);
    }
}
