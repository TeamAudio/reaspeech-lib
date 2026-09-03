use crate::common::{emit_progress, WorkerContext};
use serde::{Deserialize, Serialize};

mod audio;
mod inference;
mod models;
mod vad;

use audio::decode_audio_16khz_mono;
use inference::transcribe;
use models::{ensure_model, ensure_vad_model};
use std::time::Instant;

#[derive(Clone)]
pub struct Request {
    pub job_id: String,
    pub audio_path: String,
    pub model_name: String,
    pub language: Option<String>,
    pub translate: bool,
    pub vad: bool,
    pub words: bool,
    pub hotwords: Option<String>,
    pub beam_size: Option<usize>,
    pub start_ms: Option<i64>,
    pub end_ms: Option<i64>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Word {
    pub word: String,
    pub start: f32,
    pub end: f32,
    pub probability: f32,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Segment {
    pub start_ms: i64,
    pub end_ms: i64,
    pub text: String,
    pub probability: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub words: Option<Vec<Word>>,
}

pub fn run<F, G>(
    request: &Request,
    context: &WorkerContext,
    on_language: G,
    mut on_segment: F,
) -> Result<(), String>
where
    F: FnMut(&Segment),
    G: FnMut(&str),
{
    let profiling = std::env::var_os("REASPEECH_PROFILE").is_some();
    let job_started = Instant::now();
    emit_stage(&request.job_id, "Decoding audio", 0);
    let audio_started = Instant::now();
    let audio = decode_audio_16khz_mono(&request.job_id, &request.audio_path, context)?;
    let range_start_ms = request.start_ms.unwrap_or(0);
    let start_sample = milliseconds_to_samples(range_start_ms).min(audio.len());
    let end_sample = request
        .end_ms
        .map(milliseconds_to_samples)
        .unwrap_or(audio.len())
        .min(audio.len());
    if start_sample >= end_sample {
        return Err("The requested audio range is empty or outside the source file".into());
    }
    let audio = &audio[start_sample..end_sample];
    inference::profile_job(
        &request.job_id,
        profiling,
        "audio decode",
        audio_started.elapsed(),
    );
    ensure_not_cancelled(&request.job_id, context)?;

    emit_stage(&request.job_id, "Checking model assets", 0);
    let assets_started = Instant::now();
    let model = ensure_model(&request.job_id, &request.model_name, context)?;
    let vad_model = request
        .vad
        .then(|| ensure_vad_model(&request.job_id, context))
        .transpose()?;
    inference::profile_job(
        &request.job_id,
        profiling,
        "model/VAD assets",
        assets_started.elapsed(),
    );
    ensure_not_cancelled(&request.job_id, context)?;

    let inference_started = Instant::now();
    let result = transcribe(
        &request.job_id,
        audio,
        &model,
        vad_model.as_deref(),
        request.language.as_deref(),
        request.translate,
        request.words,
        request.hotwords.as_deref(),
        request.beam_size,
        context,
        on_language,
        |segment| {
            on_segment(&Segment {
                start_ms: segment.start_ms + range_start_ms,
                end_ms: segment.end_ms + range_start_ms,
                text: segment.text.clone(),
                probability: segment.probability,
                words: request.words.then(|| {
                    segment
                        .words
                        .iter()
                        .map(|word| Word {
                            word: word.word.clone(),
                            start: word.start_seconds + range_start_ms as f32 / 1000.0,
                            end: word.end_seconds + range_start_ms as f32 / 1000.0,
                            probability: word.probability,
                        })
                        .collect()
                }),
            });
        },
    );
    inference::profile_job(
        &request.job_id,
        profiling,
        "inference call",
        inference_started.elapsed(),
    );
    inference::profile_job(
        &request.job_id,
        profiling,
        "job total",
        job_started.elapsed(),
    );
    result
}

fn milliseconds_to_samples(milliseconds: i64) -> usize {
    (milliseconds.max(0) as u128 * 16_000 / 1000).min(usize::MAX as u128) as usize
}

fn ensure_not_cancelled(job_id: &str, context: &WorkerContext) -> Result<(), String> {
    if context.cancellation.is_cancelled(job_id) {
        Err("cancelled".into())
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod range_tests {
    use super::milliseconds_to_samples;

    #[test]
    fn converts_milliseconds_to_whisper_samples() {
        assert_eq!(milliseconds_to_samples(0), 0);
        assert_eq!(milliseconds_to_samples(1250), 20_000);
        assert_eq!(milliseconds_to_samples(-1), 0);
    }
}

pub(super) fn emit_stage(job_id: &str, message: &str, progress: u64) {
    emit_progress(job_id, message, progress.min(100), 100);
}

#[cfg(test)]
mod tests {
    use super::{Segment, Word};

    #[test]
    fn words_are_only_serialized_when_requested() {
        let segment = Segment {
            start_ms: 0,
            end_ms: 820,
            text: "Hello".into(),
            probability: 0.94,
            words: None,
        };
        let json = serde_json::to_value(&segment).unwrap();
        assert!((json["probability"].as_f64().unwrap() - 0.94).abs() < 0.0001);
        assert!(json.get("words").is_none());

        let segment = Segment {
            words: Some(vec![Word {
                word: "Hello".into(),
                start: 0.0,
                end: 0.82,
                probability: 0.94,
            }]),
            ..segment
        };
        let json = serde_json::to_value(&segment).unwrap();
        assert_eq!(json["words"][0]["word"], "Hello");
        assert_eq!(json["words"][0]["start"], 0.0);
        assert!((json["words"][0]["end"].as_f64().unwrap() - 0.82).abs() < 0.0001);
        assert!((json["words"][0]["probability"].as_f64().unwrap() - 0.94).abs() < 0.0001);
    }
}
