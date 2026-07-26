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
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Segment {
    pub start_ms: i64,
    pub end_ms: i64,
    pub text: String,
    pub confidence: f32,
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
        context,
        |segment| {
            on_segment(&Segment {
                start_ms: segment.start_ms,
                end_ms: segment.end_ms,
                text: segment.text.clone(),
                confidence: segment.confidence,
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
