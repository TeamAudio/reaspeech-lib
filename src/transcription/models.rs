use super::emit_stage;
use crate::common::WorkerContext;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::PathBuf;

pub struct ModelBundle {
    pub config: PathBuf,
    pub tokenizer: PathBuf,
    pub weights: PathBuf,
    pub mel_filters_80: PathBuf,
    pub mel_filters_128: PathBuf,
}

pub fn ensure_model(
    job_id: &str,
    model_name: &str,
    context: &WorkerContext,
) -> Result<ModelBundle, String> {
    const MODELS: &[&str] = &["small", "medium", "large-v3", "large-v3-turbo"];
    if !MODELS.contains(&model_name) {
        return Err("Unknown Whisper model".into());
    }

    let model_id = format!("openai/whisper-{model_name}");
    let directory = models_directory()?.join(format!("candle-whisper-{model_name}"));
    fs::create_dir_all(&directory)
        .map_err(|error| format!("Could not create model directory: {error}"))?;
    let fetch = |filename: &str, description: &str| {
        ensure_download(
            job_id,
            context,
            directory.join(filename),
            &format!("https://huggingface.co/{model_id}/resolve/main/{filename}"),
            description,
        )
    };
    let config = fetch("config.json", "model configuration")?;
    let tokenizer = fetch("tokenizer.json", "tokenizer")?;
    let weights = fetch("model.safetensors", "model weights")?;
    let mel_filters_80 = ensure_download(
        job_id,
        context,
        directory.join("melfilters.bytes"),
        "https://raw.githubusercontent.com/huggingface/candle/main/candle-examples/examples/whisper/melfilters.bytes",
        "mel filters",
    )?;
    let mel_filters_128 = ensure_download(
        job_id,
        context,
        directory.join("melfilters128.bytes"),
        "https://raw.githubusercontent.com/huggingface/candle/main/candle-examples/examples/whisper/melfilters128.bytes",
        "128-bin mel filters",
    )?;
    Ok(ModelBundle {
        config,
        tokenizer,
        weights,
        mel_filters_80,
        mel_filters_128,
    })
}

pub fn ensure_vad_model(job_id: &str, context: &WorkerContext) -> Result<PathBuf, String> {
    ensure_download(
        job_id,
        context,
        models_directory()?.join("silero-vad-v6.2.1.onnx"),
        "https://github.com/snakers4/silero-vad/raw/v6.2.1/src/silero_vad/data/silero_vad.onnx",
        "Silero VAD model",
    )
}

fn models_directory() -> Result<PathBuf, String> {
    let models_dir = std::env::var_os("MODELS_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models"));
    fs::create_dir_all(&models_dir)
        .map_err(|error| format!("Could not create model directory: {error}"))?;
    Ok(models_dir)
}

fn ensure_download(
    job_id: &str,
    context: &WorkerContext,
    destination: PathBuf,
    url: &str,
    description: &str,
) -> Result<PathBuf, String> {
    if destination.is_file() {
        emit_stage(job_id, "Downloading", 100);
        return Ok(destination);
    }

    let partial = destination.with_extension("part");
    let _ = fs::remove_file(&partial);
    let result = download_to_partial(job_id, context, url, description, &partial);
    if let Err(error) = result {
        let _ = fs::remove_file(&partial);
        return Err(error);
    }

    fs::rename(&partial, &destination)
        .map_err(|error| format!("Could not finish model download: {error}"))?;
    emit_stage(job_id, "Downloading", 100);
    Ok(destination)
}

fn download_to_partial(
    job_id: &str,
    context: &WorkerContext,
    url: &str,
    description: &str,
    partial: &std::path::Path,
) -> Result<(), String> {
    let mut response = reqwest::blocking::Client::builder()
        .user_agent("ReaSpeech/0.1")
        .build()
        .map_err(|error| error.to_string())?
        .get(url)
        .send()
        .map_err(|error| format!("{description} download failed: {error}"))?
        .error_for_status()
        .map_err(|error| format!("{description} download failed: {error}"))?;
    let total = response.content_length().unwrap_or(0);
    let mut output =
        File::create(partial).map_err(|error| format!("Could not create model file: {error}"))?;
    let mut downloaded = 0_u64;
    let mut buffer = vec![0_u8; 256 * 1024];

    loop {
        if context.cancellation.is_cancelled(job_id) {
            return Err("cancelled".into());
        }
        let count = response
            .read(&mut buffer)
            .map_err(|error| format!("{description} download failed: {error}"))?;
        if count == 0 {
            break;
        }
        output
            .write_all(&buffer[..count])
            .map_err(|error| format!("Could not save model: {error}"))?;
        downloaded += count as u64;
        let percent = if total == 0 {
            0
        } else {
            downloaded.saturating_mul(100) / total
        };
        emit_stage(job_id, "Downloading", percent);
    }

    output
        .flush()
        .map_err(|error| format!("Could not save model: {error}"))
}
