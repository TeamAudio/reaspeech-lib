use super::emit_stage;
use super::models::ModelBundle;
use super::vad::SileroVad;
use crate::common::{Cancellation, WorkerContext};
use byteorder::{ByteOrder, LittleEndian};
use candle_core::{DType, Device, IndexOp, Shape, Tensor};
use candle_nn::{ops::softmax, var_builder::SimpleBackend, VarBuilder};
use candle_transformers::models::whisper::timestamps::{AlignmentHeads, PostProcessor};
use candle_transformers::models::whisper::{self as whisper, audio, Config};
use std::fs;
use std::fs::OpenOptions;
use std::io::Write;
use std::num::NonZeroUsize;
use std::path::Path;
use std::time::{Duration, Instant};
use tokenizers::Tokenizer;

const SAMPLE_RATE: usize = 16_000;
const VAD_FRAME_SAMPLES: usize = 512;
const VAD_THRESHOLD: f32 = 0.5;
const MIN_SPEECH_SAMPLES: usize = SAMPLE_RATE / 4;
const MIN_SILENCE_SAMPLES: usize = SAMPLE_RATE / 10;
const SPEECH_PAD_SAMPLES: usize = SAMPLE_RATE * 30 / 1000;
const DEFAULT_BEAM_SIZE: usize = 1;

// A start must already be acoustically plausible; continuation gets more help so
// a recognized phrase can finish without making hotwords dominate unrelated audio.
const HOTWORD_START_LOGIT_BIAS: f32 = 3.25;
const HOTWORD_START_MAX_LOGIT_GAP: f32 = 3.0;
const HOTWORD_CONTINUATION_LOGIT_BIAS: f32 = 5.0;
const HOTWORD_CONTINUATION_MAX_LOGIT_GAP: f32 = 4.5;
const HOTWORD_RESTART_COOLDOWN_TOKENS: usize = 8;

pub struct TranscriptSegment {
    pub start_ms: i64,
    pub end_ms: i64,
    pub text: String,
    pub probability: f32,
    pub words: Vec<TranscriptWord>,
}

pub struct TranscriptWord {
    pub word: String,
    pub start_seconds: f32,
    pub end_seconds: f32,
    pub probability: f32,
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
    token_log_probs: Vec<f64>,
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
    hotword_token_sequences: Vec<Vec<u32>>,
    language_token: Option<u32>,
    translate: bool,
    beam_size: usize,
    alignment_heads: AlignmentHeads,
    split_on_unicode_only: bool,
}

struct DecodedPiece {
    start_seconds: f32,
    end_seconds: f32,
    text: String,
    probability: f32,
    words: Vec<DecodedWord>,
    token_start: usize,
    token_end: usize,
}

struct DecodedWord {
    text: String,
    start_seconds: f32,
    end_seconds: f32,
    probability: f32,
}

#[derive(Default)]
struct DecoderProfile {
    encoder: Duration,
    prefix: Duration,
    decoder: Duration,
    projection: Duration,
    rules: Duration,
    beam: Duration,
    token_steps: usize,
}

#[derive(serde::Deserialize)]
struct GenerationConfig {
    #[serde(default)]
    suppress_tokens: Vec<u32>,
}

struct CpuCastingSafetensors {
    tensors: candle_core::safetensors::MmapedSafetensors,
}

impl SimpleBackend for CpuCastingSafetensors {
    fn get(
        &self,
        shape: Shape,
        name: &str,
        _: candle_nn::Init,
        dtype: DType,
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        let tensor = self.get_unchecked(name, dtype, device)?;
        if tensor.shape() != &shape {
            return Err(candle_core::Error::UnexpectedShape {
                msg: format!("shape mismatch for {name}"),
                expected: shape,
                got: tensor.shape().clone(),
            }
            .bt());
        }
        Ok(tensor)
    }

    fn get_unchecked(
        &self,
        name: &str,
        dtype: DType,
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        self.tensors
            .load(name, &Device::Cpu)?
            .to_dtype(dtype)?
            .to_device(device)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.tensors.get(name).is_ok()
    }
}

pub fn transcribe<F, G>(
    job_id: &str,
    pcm: &[f32],
    bundle: &ModelBundle,
    vad_model: Option<&Path>,
    language: Option<&str>,
    translate: bool,
    words: bool,
    hotwords: Option<&str>,
    beam_size: Option<usize>,
    context: &WorkerContext,
    mut on_language: G,
    mut on_segment: F,
) -> Result<(), String>
where
    F: FnMut(&TranscriptSegment),
    G: FnMut(&str),
{
    let started = Instant::now();
    let profiling = profiling_enabled();
    emit_stage(job_id, "Loading Model", 0);
    let mut config: Config = serde_json::from_str(
        &fs::read_to_string(&bundle.config)
            .map_err(|error| format!("Could not read Whisper configuration: {error}"))?,
    )
    .map_err(|error| format!("Invalid Whisper configuration: {error}"))?;
    config.use_self_attention_kv_cache = true;
    config.dtw_timestamps = words;
    let generation_config: GenerationConfig = serde_json::from_str(
        &fs::read_to_string(&bundle.generation_config)
            .map_err(|error| format!("Could not read Whisper generation configuration: {error}"))?,
    )
    .map_err(|error| format!("Invalid Whisper generation configuration: {error}"))?;
    let device = inference_device()?;
    let beam_size = beam_size.unwrap_or(DEFAULT_BEAM_SIZE);
    log_job(
        job_id,
        &format!(
            "using {} with beam size {}",
            device_name(&device),
            beam_size
        ),
    );
    let model_started = Instant::now();
    let tokenizer = Tokenizer::from_file(&bundle.tokenizer)
        .map_err(|error| format!("Could not load Whisper tokenizer: {error}"))?;
    let vb = model_var_builder(&bundle.weights, &device)?;
    let model = whisper::model::Whisper::load(&vb, config.clone())
        .map_err(|error| format!("Could not initialize Whisper: {error}"))?;
    log_job(
        job_id,
        &format!("model loaded in {:.2?}", model_started.elapsed()),
    );
    let mel_filters = read_mel_filters(bundle, config.num_mel_bins)?;
    ensure_not_cancelled(job_id, &context.cancellation)?;

    emit_stage(job_id, "Detecting Speech", 0);
    let vad_started = Instant::now();
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
    profile_job(job_id, profiling, "vad", vad_started.elapsed());
    if regions.is_empty() {
        return Ok(());
    }

    let setup_started = Instant::now();
    let first_mel = mel_tensor(
        &config,
        &mel_filters,
        &pcm[regions[0].start..regions[0].end],
        &device,
    )?;
    let language_token = match language {
        Some(code) => Some(token_id(&tokenizer, &format!("<|{code}|>"))?),
        None => {
            let (token, code) = detect_language(model.clone(), &tokenizer, &first_mel)?;
            on_language(code);
            Some(token)
        }
    };
    profile_job(
        job_id,
        profiling,
        "initial mel + language",
        setup_started.elapsed(),
    );
    let mut decoder = Decoder::new(
        model,
        tokenizer,
        language_token,
        translate,
        &generation_config.suppress_tokens,
        hotwords,
        beam_size,
    )?;

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
            let mel_started = Instant::now();
            let mel = mel_tensor(&config, &mel_filters, &pcm[chunk.start..chunk.end], &device)?;
            if profiling {
                device.synchronize().map_err(candle_error)?;
            }
            profile_job(job_id, profiling, "chunk mel", mel_started.elapsed());
            let (pieces, no_speech, average_log_probability) = decoder.decode(
                &mel,
                chunk.end - chunk.start,
                words,
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
            if !should_skip_for_no_speech(no_speech, average_log_probability) {
                for piece in pieces {
                    if piece.text.trim().is_empty() {
                        continue;
                    }
                    let segment = TranscriptSegment {
                        start_ms: samples_to_ms(chunk.start)
                            + (piece.start_seconds * 1000.0).round() as i64,
                        end_ms: (samples_to_ms(chunk.start)
                            + (piece.end_seconds * 1000.0).round() as i64)
                            .min(samples_to_ms(chunk.end)),
                        text: piece.text.trim().to_owned(),
                        probability: piece.probability,
                        words: piece
                            .words
                            .into_iter()
                            .map(|word| TranscriptWord {
                                word: word.text,
                                start_seconds: samples_to_ms(chunk.start) as f32 / 1000.0
                                    + word.start_seconds,
                                end_seconds: samples_to_ms(chunk.start) as f32 / 1000.0
                                    + word.end_seconds,
                                probability: word.probability,
                            })
                            .collect(),
                    };
                    on_segment(&segment);
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
    profile_job(job_id, profiling, "transcription total", started.elapsed());
    Ok(())
}

fn model_var_builder<'a>(weights: &Path, device: &Device) -> Result<VarBuilder<'a>, String> {
    let dtype = inference_dtype(device);
    if matches!(device, Device::Metal(_)) {
        let tensors = unsafe { candle_core::safetensors::MmapedSafetensors::new(weights) }
            .map_err(|error| format!("Could not load Whisper weights: {error}"))?;
        Ok(VarBuilder::from_backend(
            Box::new(CpuCastingSafetensors { tensors }),
            dtype,
            device.clone(),
        ))
    } else {
        unsafe {
            VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&weights), dtype, device)
        }
        .map_err(|error| format!("Could not load Whisper weights: {error}"))
    }
}

impl Decoder {
    fn new(
        mut model: whisper::model::Whisper,
        tokenizer: Tokenizer,
        language_token: Option<u32>,
        translate: bool,
        generation_suppress_tokens: &[u32],
        hotwords: Option<&str>,
        beam_size: usize,
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
        for &token in generation_suppress_tokens {
            if let Some(entry) = suppress_tokens.get_mut(token as usize) {
                *entry = true;
            }
        }
        let no_timestamps_entry = suppress_tokens
            .get_mut(no_timestamps_token as usize)
            .ok_or("Whisper no-timestamps token is outside the model vocabulary")?;
        *no_timestamps_entry = true;
        let hotword_limit = model
            .config
            .max_target_positions
            .checked_div(2)
            .and_then(|limit| limit.checked_sub(1))
            .ok_or("Whisper max_target_positions is too small")?;
        let hotword_token_sequences = tokenize_hotwords(&tokenizer, hotwords, hotword_limit)?;
        let alignment_heads = match model.config.decoder_layers {
            12 => AlignmentHeads::small(),
            24 => AlignmentHeads::medium(),
            32 => AlignmentHeads::large_v3(),
            4 => AlignmentHeads::large_v3_turbo(),
            _ => AlignmentHeads::default(),
        };
        let split_on_unicode_only = language_token
            .and_then(|token| tokenizer.id_to_token(token))
            .is_some_and(|language| {
                matches!(
                    language.as_str(),
                    "<|zh|>" | "<|ja|>" | "<|th|>" | "<|lo|>" | "<|my|>" | "<|yue|>"
                )
            });
        model.set_dtw_attention_capture(false);
        Ok(Self {
            model,
            sot_token: token_id(&tokenizer, whisper::SOT_TOKEN)?,
            transcribe_token: token_id(&tokenizer, whisper::TRANSCRIBE_TOKEN)?,
            translate_token: token_id(&tokenizer, whisper::TRANSLATE_TOKEN)?,
            eot_token: token_id(&tokenizer, whisper::EOT_TOKEN)?,
            no_speech_token,
            no_timestamps_token,
            hotword_token_sequences,
            tokenizer,
            suppress_tokens,
            language_token,
            translate,
            beam_size,
            alignment_heads,
            split_on_unicode_only,
        })
    }

    fn decode(
        &mut self,
        mel: &Tensor,
        audio_samples: usize,
        words: bool,
        chunk_index: usize,
        chunk_count: usize,
        completed_samples: usize,
        total_samples: usize,
        job_id: &str,
        cancellation: &Cancellation,
    ) -> Result<(Vec<DecodedPiece>, f64, f64), String> {
        let profiling = profiling_enabled();
        let mut profile = DecoderProfile::default();
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
        profile_sync(mel.device(), profiling)?;
        profile.encoder += encoder_started.elapsed();
        log_tensor_stats(job_id, "encoder output", &audio_features)?;
        log_job(
            job_id,
            &format!("encoder finished in {:.2?}", encoder_started.elapsed()),
        );
        let mut prefix = vec![self.sot_token];
        let sot_position = 0;
        if let Some(language) = self.language_token {
            prefix.push(language);
        }
        prefix.push(if self.translate {
            self.translate_token
        } else {
            self.transcribe_token
        });
        let prefix_len = prefix.len();

        let prefix_started = Instant::now();
        let prefix_tensor = Tensor::new(prefix.as_slice(), mel.device())
            .and_then(|tensor| tensor.unsqueeze(0))
            .map_err(candle_error)?;
        let ys = self
            .model
            .decoder
            .forward(&prefix_tensor, &audio_features, true)
            .map_err(candle_error)?;
        profile_sync(mel.device(), profiling)?;
        profile.prefix += prefix_started.elapsed();
        log_tensor_stats(job_id, "first decoder output", &ys)?;
        let projection_started = Instant::now();
        let no_speech_logits = self
            .model
            .decoder
            .final_linear(
                &ys.i((.., sot_position..sot_position + 1))
                    .map_err(candle_error)?,
            )
            .and_then(|tensor| tensor.i(0))
            .and_then(|tensor| tensor.i(0))
            .map_err(candle_error)?;
        let no_speech = softmax(&no_speech_logits, 0)
            .and_then(|tensor| tensor.i(self.no_speech_token as usize))
            .and_then(|tensor| tensor.to_dtype(DType::F32))
            .and_then(|tensor| tensor.to_scalar::<f32>())
            .map_err(candle_error)? as f64;
        let first_logits = last_logits(&self.model, &ys)?;
        profile_sync(mel.device(), profiling)?;
        profile.projection += projection_started.elapsed();
        log_tensor_stats(job_id, "first decoder logits", &first_logits)?;
        let rules_started = Instant::now();
        let logits = self.apply_hotword_bias(first_logits, &[])?;
        let logits = self.apply_timestamp_rules(logits, &prefix, prefix_len)?;
        profile.rules += rules_started.elapsed();
        let beam_started = Instant::now();
        let mut beams = expand_beam(
            Beam {
                model: self.model.clone(),
                tokens: prefix.clone(),
                token_log_probs: Vec::new(),
                score: 0.0,
                finished: false,
            },
            logits,
            self.beam_size,
            self.eot_token,
            &self.suppress_tokens,
        )?;
        profile.beam += beam_started.elapsed();

        let audio_seconds = audio_samples as f64 / SAMPLE_RATE as f64;
        let available_steps = self
            .model
            .config
            .max_target_positions
            .saturating_sub(prefix_len);
        let maximum_steps = self.model.config.max_target_positions / 2;
        if maximum_steps < 24 {
            return Err("Whisper max_target_positions must be at least 48".into());
        }
        let max_steps = ((audio_seconds * 8.0).ceil() as usize + 16)
            .clamp(24, maximum_steps)
            .min(available_steps);
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
                // The vendored decoder consumes only the uncached suffix while
                // retaining the full token list here for beam branching.
                let decoder_started = Instant::now();
                let input = Tensor::new(beam.tokens.as_slice(), mel.device())
                    .and_then(|tensor| tensor.unsqueeze(0))
                    .map_err(candle_error)?;
                let ys = beam
                    .model
                    .decoder
                    .forward(&input, &audio_features, false)
                    .map_err(candle_error)?;
                profile_sync(mel.device(), profiling)?;
                profile.decoder += decoder_started.elapsed();
                let projection_started = Instant::now();
                let logits = last_logits(&beam.model, &ys)?;
                profile_sync(mel.device(), profiling)?;
                profile.projection += projection_started.elapsed();
                let rules_started = Instant::now();
                let logits = self.apply_hotword_bias(logits, &beam.tokens[prefix_len..])?;
                let logits = self.apply_timestamp_rules(logits, &beam.tokens, prefix_len)?;
                profile.rules += rules_started.elapsed();
                let beam_started = Instant::now();
                candidates.extend(expand_beam(
                    beam,
                    logits,
                    self.beam_size,
                    self.eot_token,
                    &self.suppress_tokens,
                )?);
                profile.beam += beam_started.elapsed();
                profile.token_steps += 1;
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
        let average_log_probability =
            average_log_probability(generated, &best.token_log_probs, self.eot_token);
        let mut pieces = self.timestamped_pieces(
            generated,
            &best.token_log_probs,
            audio_seconds as f32,
            words,
        )?;
        if words {
            let text_tokens: Vec<u32> = generated
                .iter()
                .copied()
                .filter(|&token| token < self.eot_token)
                .collect();
            let mut alignment_tokens = prefix;
            alignment_tokens.push(self.no_timestamps_token);
            alignment_tokens.extend_from_slice(&text_tokens);
            alignment_tokens.push(self.eot_token);
            self.model.reset_kv_cache();
            self.model.set_dtw_attention_capture(true);
            let alignment_input = Tensor::new(alignment_tokens.as_slice(), mel.device())
                .and_then(|tensor| tensor.unsqueeze(0))
                .map_err(candle_error)?;
            self.model
                .decoder
                .forward(&alignment_input, &audio_features, true)
                .map_err(candle_error)?;
            let n_frames = audio_samples.div_ceil(whisper::HOP_LENGTH);
            let raw = self
                .model
                .dtw_timestamps(
                    self.alignment_heads.clone(),
                    NonZeroUsize::new(7).ok_or("DTW filter width must be non-zero")?,
                    n_frames,
                    prefix_len,
                )
                .map_err(candle_error)?
                .into_iter()
                .next();
            if let Some(raw) = raw {
                let aligned = <Self as PostProcessor>::label(self, &raw, &text_tokens)
                    .map_err(candle_error)?;
                let aligned = aligned_word_token_starts(&aligned);
                for piece in &mut pieces {
                    piece.words = aligned
                        .iter()
                        .filter(|(_, token_start)| {
                            *token_start >= piece.token_start && *token_start < piece.token_end
                        })
                        .map(|(word, _)| DecodedWord {
                            text: word.text.clone(),
                            start_seconds: word.start,
                            end_seconds: word.end,
                            probability: probability_for_word_tokens(
                                &word.tokens,
                                generated,
                                &best.token_log_probs,
                            ),
                        })
                        .collect();
                }
            }
            self.model.set_dtw_attention_capture(false);
        }
        if profiling {
            log_job(
                job_id,
                &format!(
                    "profile chunk: encoder={:.2?}, prefix={:.2?}, decoder={:.2?}, projection={:.2?}, rules/transfers={:.2?}, beam={:.2?}, token_steps={}",
                    profile.encoder,
                    profile.prefix,
                    profile.decoder,
                    profile.projection,
                    profile.rules,
                    profile.beam,
                    profile.token_steps
                ),
            );
        }
        Ok((pieces, no_speech, average_log_probability))
    }

    fn apply_hotword_bias(&self, logits: Tensor, sampled: &[u32]) -> Result<Tensor, String> {
        if self.hotword_token_sequences.is_empty() {
            return Ok(logits);
        }
        let mut values = logits
            .to_dtype(DType::F32)
            .and_then(|tensor| tensor.to_vec1::<f32>())
            .map_err(candle_error)?;
        let best = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        for (token, bias, max_gap) in
            hotword_bias_candidates(&self.hotword_token_sequences, sampled)
        {
            if let Some(value) = values.get_mut(token as usize) {
                if value.is_finite() && *value >= best - max_gap {
                    *value += bias;
                }
            }
        }
        Tensor::new(values.as_slice(), logits.device()).map_err(candle_error)
    }

    fn apply_timestamp_rules(
        &self,
        logits: Tensor,
        tokens: &[u32],
        prefix_len: usize,
    ) -> Result<Tensor, String> {
        let mut values = logits
            .to_dtype(DType::F32)
            .and_then(|tensor| tensor.to_vec1::<f32>())
            .map_err(candle_error)?;
        let timestamp_begin = self
            .no_timestamps_token
            .checked_add(1)
            .ok_or("Whisper no-timestamps token is invalid")?;
        let timestamp_begin = usize::try_from(timestamp_begin)
            .map_err(|_| "Whisper timestamp token does not fit this platform")?;
        if timestamp_begin > values.len() {
            return Err(format!(
                "Whisper timestamp token {timestamp_begin} is outside the logits vocabulary ({})",
                values.len()
            ));
        }
        let sampled = &tokens[prefix_len.min(tokens.len())..];
        let last_is_timestamp = sampled
            .last()
            .is_some_and(|token| *token as usize >= timestamp_begin);
        let previous_is_timestamp = sampled
            .get(sampled.len().saturating_sub(2))
            .is_some_and(|token| *token as usize >= timestamp_begin);

        if last_is_timestamp {
            if previous_is_timestamp {
                for value in values.iter_mut().skip(timestamp_begin) {
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
            .find(|token| **token as usize >= timestamp_begin)
            .copied()
        {
            let minimum = if last_is_timestamp && !previous_is_timestamp {
                last_timestamp
            } else {
                last_timestamp
                    .checked_add(1)
                    .ok_or("Whisper timestamp token is invalid")?
            };
            for value in values
                .iter_mut()
                .take(minimum as usize)
                .skip(timestamp_begin)
            {
                *value = f32::NEG_INFINITY;
            }
        }

        if sampled.is_empty() {
            for value in values.iter_mut().take(timestamp_begin) {
                *value = f32::NEG_INFINITY;
            }
            // Match faster-whisper's default one-second maximum initial timestamp.
            for value in values.iter_mut().skip(timestamp_begin.saturating_add(51)) {
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
            .skip(timestamp_begin)
            .map(|value| ((*value - max) as f64).exp() / denominator)
            .sum();
        let best_text_probability = values[..timestamp_begin]
            .iter()
            .map(|value| ((*value - max) as f64).exp() / denominator)
            .fold(0.0, f64::max);
        if timestamp_probability > best_text_probability {
            for value in values.iter_mut().take(timestamp_begin) {
                *value = f32::NEG_INFINITY;
            }
        }
        Tensor::new(values.as_slice(), logits.device()).map_err(candle_error)
    }

    fn timestamped_pieces(
        &self,
        tokens: &[u32],
        token_log_probs: &[f64],
        audio_seconds: f32,
        include_words: bool,
    ) -> Result<Vec<DecodedPiece>, String> {
        if tokens.len() != token_log_probs.len() {
            return Err("Whisper token probabilities do not match generated tokens".into());
        }
        let timestamp_begin = self.no_timestamps_token + 1;
        let mut pieces = Vec::new();
        let mut text_tokens = Vec::new();
        let mut text_log_probs = Vec::new();
        let mut text_token_offset = 0;
        let mut start_seconds = 0.0f32;
        for (&token, &log_prob) in tokens.iter().zip(token_log_probs) {
            if token == self.eot_token {
                break;
            }
            if token >= timestamp_begin {
                let time = ((token - timestamp_begin) as f32 / 50.0).min(audio_seconds);
                if !text_tokens.is_empty() && time > start_seconds {
                    let token_end = text_token_offset + text_tokens.len();
                    let text = self
                        .tokenizer
                        .decode(&text_tokens, true)
                        .map_err(|error| error.to_string())?;
                    pieces.push(DecodedPiece {
                        start_seconds,
                        end_seconds: time,
                        text,
                        probability: probability_from_log_probs(&text_log_probs),
                        token_start: text_token_offset,
                        token_end,
                        words: if include_words {
                            words_from_tokens(
                                &self.tokenizer,
                                &text_tokens,
                                &text_log_probs,
                                start_seconds,
                                time,
                            )?
                        } else {
                            Vec::new()
                        },
                    });
                    text_token_offset = token_end;
                    text_tokens.clear();
                    text_log_probs.clear();
                }
                start_seconds = time;
            } else {
                text_tokens.push(token);
                text_log_probs.push(log_prob);
            }
        }
        if !text_tokens.is_empty() {
            let token_end = text_token_offset + text_tokens.len();
            let text = self
                .tokenizer
                .decode(&text_tokens, true)
                .map_err(|error| error.to_string())?;
            pieces.push(DecodedPiece {
                start_seconds,
                end_seconds: audio_seconds,
                text,
                probability: probability_from_log_probs(&text_log_probs),
                token_start: text_token_offset,
                token_end,
                words: if include_words {
                    words_from_tokens(
                        &self.tokenizer,
                        &text_tokens,
                        &text_log_probs,
                        start_seconds,
                        audio_seconds,
                    )?
                } else {
                    Vec::new()
                },
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
    let mut logits = logits
        .to_dtype(DType::F32)
        .and_then(|tensor| tensor.to_vec1::<f32>())
        .map_err(candle_error)?;
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
            let log_prob = logit as f64 - log_denom;
            next.token_log_probs.push(log_prob);
            next.score += log_prob;
            next.finished = token as u32 == eot_token;
            next
        })
        .collect())
}

fn probability_from_log_probs(log_probs: &[f64]) -> f32 {
    if log_probs.is_empty() {
        return 0.0;
    }
    let mean = log_probs.iter().sum::<f64>() / log_probs.len() as f64;
    mean.exp().clamp(0.0, 1.0) as f32
}

fn average_log_probability(tokens: &[u32], log_probs: &[f64], eot_token: u32) -> f64 {
    let text_token_count = tokens
        .iter()
        .position(|token| *token == eot_token)
        .unwrap_or(tokens.len());
    let scored_token_count =
        (text_token_count + usize::from(text_token_count < tokens.len())).min(log_probs.len());
    log_probs[..scored_token_count].iter().sum::<f64>() / (text_token_count + 1) as f64
}

fn should_skip_for_no_speech(no_speech_probability: f64, average_log_probability: f64) -> bool {
    no_speech_probability > whisper::NO_SPEECH_THRESHOLD
        && average_log_probability < whisper::LOGPROB_THRESHOLD
}

fn probability_for_word_tokens(word: &[u32], tokens: &[u32], log_probs: &[f64]) -> f32 {
    if word.is_empty() || tokens.len() != log_probs.len() {
        return 0.0;
    }
    tokens
        .windows(word.len())
        .position(|candidate| candidate == word)
        .map(|start| probability_from_log_probs(&log_probs[start..start + word.len()]))
        .unwrap_or(0.0)
}

fn aligned_word_token_starts(
    words: &[whisper::timestamps::Word],
) -> Vec<(&whisper::timestamps::Word, usize)> {
    let mut token_start = 0;
    words
        .iter()
        .map(|word| {
            let result = (word, token_start);
            token_start += word.tokens.len();
            result
        })
        .collect()
}

fn group_word_segments(
    segments: Vec<whisper::timestamps::Segment>,
    split_on_unicode_only: bool,
) -> Vec<whisper::timestamps::Segment> {
    if split_on_unicode_only {
        return segments;
    }

    const ASCII_PUNCTUATION: &str = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
    let mut words: Vec<whisper::timestamps::Segment> = Vec::new();
    for segment in segments {
        let with_space = segment.text.starts_with(char::is_whitespace);
        let punctuation = {
            let text = segment.text.trim();
            !text.is_empty() && ASCII_PUNCTUATION.contains(text)
        };
        if punctuation && !with_space && !words.is_empty() {
            let word = words.last_mut().expect("checked above");
            word.text.push_str(&segment.text);
            word.token_indices.extend(segment.token_indices);
        } else if with_space || words.is_empty() {
            words.push(segment);
        } else if let Some(word) = words.last_mut() {
            word.text.push_str(&segment.text);
            word.token_indices.extend(segment.token_indices);
        }
    }
    words
}

impl PostProcessor for Decoder {
    type Error = candle_core::Error;

    fn decode(&mut self, tokens: &[u32]) -> candle_core::Result<Vec<whisper::timestamps::Segment>> {
        let full = self
            .tokenizer
            .decode(tokens, true)
            .map_err(candle_core::Error::msg)?;
        let token_text = tokens
            .iter()
            .filter(|&&token| token < 50_000)
            .map(|&token| self.tokenizer.decode(&[token], true))
            .collect::<Result<Vec<_>, _>>()
            .map_err(candle_core::Error::msg)?;
        let segments = whisper::timestamps::unicode_segments(full, token_text)?;
        Ok(group_word_segments(segments, self.split_on_unicode_only))
    }
}

fn words_from_tokens(
    tokenizer: &Tokenizer,
    tokens: &[u32],
    log_probs: &[f64],
    start_seconds: f32,
    end_seconds: f32,
) -> Result<Vec<DecodedWord>, String> {
    if tokens.len() != log_probs.len() {
        return Err("Whisper token probabilities do not match word tokens".into());
    }
    let mut words: Vec<(String, usize, usize, Vec<f64>)> = Vec::new();
    for (index, (&token, &log_prob)) in tokens.iter().zip(log_probs).enumerate() {
        let text = tokenizer
            .decode(&[token], false)
            .map_err(|error| error.to_string())?;
        let starts_word = text.chars().next().is_some_and(char::is_whitespace);
        let text = text.trim().to_owned();
        if text.is_empty() {
            continue;
        }
        if starts_word || words.is_empty() {
            words.push((text, index, index + 1, vec![log_prob]));
        } else if let Some((word, _, token_end, probabilities)) = words.last_mut() {
            word.push_str(&text);
            *token_end = index + 1;
            probabilities.push(log_prob);
        }
    }

    let token_count = tokens.len().max(1) as f32;
    let duration = (end_seconds - start_seconds).max(0.0);
    Ok(words
        .into_iter()
        .map(
            |(word, token_start, token_end, probabilities)| DecodedWord {
                text: word,
                start_seconds: start_seconds + duration * token_start as f32 / token_count,
                end_seconds: start_seconds + duration * token_end as f32 / token_count,
                probability: probability_from_log_probs(&probabilities),
            },
        )
        .collect())
}

fn normalized_score(beam: &Beam) -> f64 {
    beam.score / beam.tokens.len().max(1) as f64
}

fn last_logits(model: &whisper::model::Whisper, ys: &Tensor) -> Result<Tensor, String> {
    let (_, length, _) = ys.dims3().map_err(candle_error)?;
    let last = length
        .checked_sub(1)
        .ok_or("Whisper decoder returned an empty token sequence")?;
    model
        .decoder
        .final_linear(&ys.i((..1, last..)).map_err(candle_error)?)
        .and_then(|tensor| tensor.i(0))
        .and_then(|tensor| tensor.i(0))
        .map_err(candle_error)
}

fn detect_language(
    mut model: whisper::model::Whisper,
    tokenizer: &Tokenizer,
    mel: &Tensor,
) -> Result<(u32, &'static str), String> {
    const LANGUAGE_CODES: &[&str] = &[
        "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv",
        "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no",
        "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr",
        "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
        "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu",
        "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl",
        "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su",
    ];
    let languages: Vec<(u32, &'static str)> = LANGUAGE_CODES
        .iter()
        .filter_map(|&code| {
            tokenizer
                .token_to_id(&format!("<|{code}|>"))
                .map(|id| (id, code))
        })
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
        .to_dtype(DType::F32)
        .and_then(|tensor| tensor.to_vec1::<f32>())
        .map_err(candle_error)?;
    languages
        .into_iter()
        .filter_map(|(id, code)| logits.get(id as usize).map(|&logit| (id, code, logit)))
        .max_by(|left, right| left.2.total_cmp(&right.2))
        .map(|(id, code, _)| (id, code))
        .ok_or_else(|| {
            "Whisper tokenizer has no language tokens within the model vocabulary".into()
        })
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
    Tensor::from_vec(values, (1, config.num_mel_bins, frames), device)
        .and_then(|tensor| tensor.to_dtype(inference_dtype(device)))
        .map_err(candle_error)
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

fn inference_device() -> Result<Device, String> {
    if std::env::var_os("REASPEECH_FORCE_CPU").is_some() {
        return Ok(Device::Cpu);
    }
    #[cfg(feature = "metal")]
    return Device::new_metal(0).map_err(|error| format!("Could not initialize Metal: {error}"));
    #[cfg(all(not(feature = "metal"), feature = "cuda"))]
    return Device::new_cuda(0).map_err(|error| format!("Could not initialize CUDA: {error}"));
    #[allow(unreachable_code)]
    Ok(Device::Cpu)
}

fn inference_dtype(device: &Device) -> DType {
    if matches!(device, Device::Metal(_)) {
        DType::F16
    } else {
        whisper::DTYPE
    }
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

fn profiling_enabled() -> bool {
    std::env::var_os("REASPEECH_PROFILE").is_some()
}

fn profile_sync(device: &Device, enabled: bool) -> Result<(), String> {
    if enabled {
        device.synchronize().map_err(candle_error)?;
    }
    Ok(())
}

pub(super) fn profile_job(job_id: &str, enabled: bool, stage: &str, elapsed: Duration) {
    if enabled {
        log_job(job_id, &format!("profile {stage}: {elapsed:.2?}"));
    }
}

fn log_tensor_stats(job_id: &str, name: &str, tensor: &Tensor) -> Result<(), String> {
    if std::env::var_os("REASPEECH_DEBUG_TENSORS").is_none() {
        return Ok(());
    }
    let values = tensor
        .flatten_all()
        .and_then(|tensor| tensor.to_dtype(DType::F32))
        .and_then(|tensor| tensor.to_vec1::<f32>())
        .map_err(candle_error)?;
    let finite = values.iter().filter(|value| value.is_finite()).count();
    let min = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .fold(f32::INFINITY, f32::min);
    let max = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    log_job(
        job_id,
        &format!(
            "{name}: {finite}/{} finite, range {min:.4}..{max:.4}",
            values.len()
        ),
    );
    Ok(())
}

fn token_id(tokenizer: &Tokenizer, token: &str) -> Result<u32, String> {
    tokenizer
        .token_to_id(token)
        .ok_or_else(|| format!("Whisper tokenizer has no {token} token"))
}

fn truncate_hotword_tokens(tokens: &[u32], limit: usize) -> Vec<u32> {
    tokens.iter().copied().take(limit).collect()
}

fn tokenize_hotwords(
    tokenizer: &Tokenizer,
    hotwords: Option<&str>,
    limit: usize,
) -> Result<Vec<Vec<u32>>, String> {
    let mut sequences = Vec::new();
    for hotword in hotwords
        .into_iter()
        .flat_map(|value| value.split([',', '\n']))
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        let tokens = tokenizer
            .encode(format!(" {hotword}"), false)
            .map_err(|error| format!("Could not tokenize hotwords: {error}"))?;
        let tokens = truncate_hotword_tokens(tokens.get_ids(), limit);
        if !tokens.is_empty() {
            sequences.push(tokens);
        }
    }
    Ok(sequences)
}

fn hotword_bias_candidates(hotwords: &[Vec<u32>], sampled: &[u32]) -> Vec<(u32, f32, f32)> {
    let mut candidates = std::collections::BTreeMap::new();
    for hotword in hotwords {
        let Some(&first) = hotword.first() else {
            continue;
        };
        let recently_completed = sampled
            .windows(hotword.len())
            .rposition(|window| window == hotword)
            .is_some_and(|start| {
                sampled.len() - (start + hotword.len()) <= HOTWORD_RESTART_COOLDOWN_TOKENS
            });
        if !recently_completed {
            candidates
                .entry(first)
                .or_insert((HOTWORD_START_LOGIT_BIAS, HOTWORD_START_MAX_LOGIT_GAP));
        }
        for prefix_len in (1..hotword.len()).rev() {
            if sampled.ends_with(&hotword[..prefix_len]) {
                candidates
                    .entry(hotword[prefix_len])
                    .and_modify(|candidate: &mut (f32, f32)| {
                        candidate.0 = f32::max(candidate.0, HOTWORD_CONTINUATION_LOGIT_BIAS);
                        candidate.1 = f32::max(candidate.1, HOTWORD_CONTINUATION_MAX_LOGIT_GAP);
                    })
                    .or_insert((
                        HOTWORD_CONTINUATION_LOGIT_BIAS,
                        HOTWORD_CONTINUATION_MAX_LOGIT_GAP,
                    ));
                break;
            }
        }
    }
    candidates
        .into_iter()
        .map(|(token, (bias, max_gap))| (token, bias, max_gap))
        .collect()
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
    fn reads_suppressed_tokens_from_generation_config() {
        let config: GenerationConfig =
            serde_json::from_str(r#"{"suppress_tokens":[1,2,50360]}"#).unwrap();

        assert_eq!(config.suppress_tokens, [1, 2, 50360]);
    }

    #[test]
    fn probability_uses_geometric_mean_of_token_probabilities() {
        let probability = probability_from_log_probs(&[0.9_f64.ln(), 0.4_f64.ln()]);

        assert!((probability - 0.6).abs() < f32::EPSILON);
        assert_eq!(probability_from_log_probs(&[]), 0.0);
    }

    #[test]
    fn average_log_probability_includes_end_of_text() {
        let eot = 99;
        let average = average_log_probability(&[10, 11, eot], &[-0.3, -0.6, -0.9], eot);

        assert!((average - -0.6).abs() < f64::EPSILON);
    }

    #[test]
    fn average_log_probability_reserves_end_of_text_when_generation_is_truncated() {
        let average = average_log_probability(&[10, 11], &[-0.3, -0.6], 99);

        assert!((average - -0.3).abs() < f64::EPSILON);
    }

    #[test]
    fn skips_only_low_confidence_chunks_with_high_no_speech_probability() {
        assert!(should_skip_for_no_speech(0.7, -1.1));
        assert!(!should_skip_for_no_speech(0.7, -0.9));
        assert!(!should_skip_for_no_speech(0.5, -1.1));
        assert!(!should_skip_for_no_speech(
            whisper::NO_SPEECH_THRESHOLD,
            -1.1
        ));
        assert!(!should_skip_for_no_speech(0.7, whisper::LOGPROB_THRESHOLD));
    }

    #[test]
    fn groups_space_delimited_subwords_like_whisper() {
        let segments = vec![
            word_segment(" You", 0),
            word_segment("'re", 1),
            word_segment(" wel", 2),
            word_segment("come", 3),
            word_segment("!", 4),
        ];

        let grouped = group_word_segments(segments, false);

        assert_eq!(grouped.len(), 2);
        assert_eq!(grouped[0].text, " You're");
        assert_eq!(grouped[0].token_indices, [0, 1]);
        assert_eq!(grouped[1].text, " welcome!");
        assert_eq!(grouped[1].token_indices, [2, 3, 4]);
    }

    #[test]
    fn keeps_a_leading_quote_with_the_following_word() {
        let grouped = group_word_segments(
            vec![
                word_segment(" said", 0),
                word_segment(" '", 1),
                word_segment("hello", 2),
            ],
            false,
        );

        assert_eq!(grouped[0].text, " said");
        assert_eq!(grouped[1].text, " 'hello");
        assert_eq!(grouped[1].token_indices, [1, 2]);
    }

    #[test]
    fn no_space_languages_keep_unicode_segments() {
        let grouped = group_word_segments(vec![word_segment("你", 0), word_segment("好", 1)], true);

        assert_eq!(grouped.len(), 2);
    }

    #[test]
    fn aligned_words_keep_the_segment_containing_their_first_token() {
        let words = vec![
            aligned_word("previous.", &[10, 11]),
            aligned_word("Next", &[12]),
            aligned_word("segment", &[13, 14]),
        ];

        let indexed = aligned_word_token_starts(&words);
        let first_piece = indexed
            .iter()
            .filter(|(_, token_start)| *token_start < 2)
            .map(|(word, _)| word.text.as_str())
            .collect::<Vec<_>>();
        let second_piece = indexed
            .iter()
            .filter(|(_, token_start)| *token_start >= 2)
            .map(|(word, _)| word.text.as_str())
            .collect::<Vec<_>>();

        assert_eq!(first_piece, ["previous."]);
        assert_eq!(second_piece, ["Next", "segment"]);
    }

    fn word_segment(text: &str, token: usize) -> whisper::timestamps::Segment {
        whisper::timestamps::Segment {
            text: text.into(),
            token_indices: vec![token],
        }
    }

    fn aligned_word(text: &str, tokens: &[u32]) -> whisper::timestamps::Word {
        whisper::timestamps::Word {
            text: text.into(),
            start: 0.0,
            end: 0.0,
            tokens: tokens.into(),
        }
    }

    #[test]
    fn hotwords_are_limited_to_half_the_decoder_context_less_one() {
        let tokens: Vec<u32> = (0..300).collect();

        let truncated = truncate_hotword_tokens(&tokens, 223);

        assert_eq!(truncated.len(), 223);
        assert_eq!(truncated[0], 0);
        assert_eq!(truncated[222], 222);
    }

    #[test]
    fn hotword_bias_follows_matching_token_sequences() {
        let hotwords = vec![vec![10, 11, 12], vec![20, 21]];

        assert_eq!(
            hotword_bias_candidates(&hotwords, &[]),
            [
                (10, HOTWORD_START_LOGIT_BIAS, HOTWORD_START_MAX_LOGIT_GAP),
                (20, HOTWORD_START_LOGIT_BIAS, HOTWORD_START_MAX_LOGIT_GAP)
            ]
        );
        assert_eq!(
            hotword_bias_candidates(&hotwords, &[99, 10]),
            [
                (10, HOTWORD_START_LOGIT_BIAS, HOTWORD_START_MAX_LOGIT_GAP),
                (
                    11,
                    HOTWORD_CONTINUATION_LOGIT_BIAS,
                    HOTWORD_CONTINUATION_MAX_LOGIT_GAP
                ),
                (20, HOTWORD_START_LOGIT_BIAS, HOTWORD_START_MAX_LOGIT_GAP)
            ]
        );
        assert_eq!(
            hotword_bias_candidates(&hotwords, &[99, 10, 11]),
            [
                (10, HOTWORD_START_LOGIT_BIAS, HOTWORD_START_MAX_LOGIT_GAP),
                (
                    12,
                    HOTWORD_CONTINUATION_LOGIT_BIAS,
                    HOTWORD_CONTINUATION_MAX_LOGIT_GAP
                ),
                (20, HOTWORD_START_LOGIT_BIAS, HOTWORD_START_MAX_LOGIT_GAP)
            ]
        );
        assert_eq!(
            hotword_bias_candidates(&hotwords, &[99, 20]),
            [
                (10, HOTWORD_START_LOGIT_BIAS, HOTWORD_START_MAX_LOGIT_GAP),
                (20, HOTWORD_START_LOGIT_BIAS, HOTWORD_START_MAX_LOGIT_GAP),
                (
                    21,
                    HOTWORD_CONTINUATION_LOGIT_BIAS,
                    HOTWORD_CONTINUATION_MAX_LOGIT_GAP
                )
            ]
        );
    }

    #[test]
    fn completed_hotword_is_not_immediately_restarted() {
        let hotwords = vec![vec![10, 11, 12], vec![20, 21]];

        let candidates = hotword_bias_candidates(&hotwords, &[99, 10, 11, 12, 98]);

        assert!(!candidates.iter().any(|(token, _, _)| *token == 10));
        assert!(candidates.iter().any(|(token, _, _)| *token == 20));
    }

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

    #[test]
    #[ignore = "requires a downloaded Whisper model"]
    fn whisper_smoke_test() {
        let directory = std::env::var_os("REASPEECH_TEST_MODEL_DIR")
            .map(std::path::PathBuf::from)
            .expect("set REASPEECH_TEST_MODEL_DIR to a downloaded Whisper model directory");
        let bundle = ModelBundle {
            config: directory.join("config.json"),
            generation_config: directory.join("generation_config.json"),
            tokenizer: directory.join("tokenizer.json"),
            weights: directory.join("model.safetensors"),
            mel_filters_80: directory.join("melfilters.bytes"),
            mel_filters_128: directory.join("melfilters128.bytes"),
        };
        let context = WorkerContext::default();
        let pcm = match std::env::var("REASPEECH_TEST_AUDIO") {
            Ok(path) => crate::transcription::audio::decode_audio_16khz_mono(
                "metal-smoke-test",
                &path,
                &context,
            )
            .unwrap(),
            Err(_) => vec![0.0; SAMPLE_RATE],
        };
        let vad_model = std::env::var_os("REASPEECH_TEST_VAD_MODEL").map(std::path::PathBuf::from);

        let mut segments = Vec::new();
        transcribe(
            "metal-smoke-test",
            &pcm,
            &bundle,
            vad_model.as_deref(),
            Some("en"),
            false,
            false,
            None,
            None,
            &context,
            |_| {},
            |segment| {
                segments.push((segment.start_ms, segment.end_ms, segment.text.clone()));
            },
        )
        .unwrap();
        for (start_ms, end_ms, text) in segments {
            eprintln!("{start_ms}-{end_ms}: {text}");
        }
    }
}
