use crate::common::{emit_progress, WorkerContext};
use serde::Serialize;

mod audio;
mod inference;
mod models;
mod vad;

use audio::decode_audio_16khz_mono;
use inference::transcribe;
use models::{ensure_model, ensure_vad_model};

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
}

#[derive(Clone, Serialize)]
pub struct Word {
    pub word: String,
    pub start: f32,
    pub end: f32,
    pub probability: f32,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Segment {
    pub start_ms: i64,
    pub end_ms: i64,
    pub text: String,
    pub probability: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub words: Option<Vec<Word>>,
}

pub fn run<F>(request: &Request, context: &WorkerContext, mut on_segment: F) -> Result<(), String>
where
    F: FnMut(&Segment),
{
    emit_stage(&request.job_id, "Decoding audio", 0);
    let audio = decode_audio_16khz_mono(&request.job_id, &request.audio_path, context)?;
    ensure_not_cancelled(&request.job_id, context)?;

    emit_stage(&request.job_id, "Downloading", 0);
    let model = ensure_model(&request.job_id, &request.model_name, context)?;
    let vad_model = request
        .vad
        .then(|| ensure_vad_model(&request.job_id, context))
        .transpose()?;
    ensure_not_cancelled(&request.job_id, context)?;

    transcribe(
        &request.job_id,
        &audio,
        &model,
        vad_model.as_deref(),
        request.language.as_deref(),
        request.translate,
        request.words,
        request.hotwords.as_deref(),
        request.beam_size,
        context,
        |segment| {
            on_segment(&Segment {
                start_ms: segment.start_ms,
                end_ms: segment.end_ms,
                text: segment.text.clone(),
                probability: segment.probability,
                words: request.words.then(|| {
                    segment
                        .words
                        .iter()
                        .map(|word| Word {
                            word: word.word.clone(),
                            start: word.start_seconds,
                            end: word.end_seconds,
                            probability: word.probability,
                        })
                        .collect()
                }),
            });
        },
    )
}

fn ensure_not_cancelled(job_id: &str, context: &WorkerContext) -> Result<(), String> {
    if context.cancellation.is_cancelled(job_id) {
        Err("cancelled".into())
    } else {
        Ok(())
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
