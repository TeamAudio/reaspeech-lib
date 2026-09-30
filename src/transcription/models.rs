use super::emit_stage;
use crate::common::WorkerContext;
use crate::config;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

static NEXT_PARTIAL_ID: AtomicU64 = AtomicU64::new(0);

pub struct ModelBundle {
    pub config: PathBuf,
    pub generation_config: PathBuf,
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
    if !config::WHISPER_MODELS.contains(&model_name) {
        return Err("Unknown Whisper model".into());
    }

    let directory = models_directory()?.join(format!("candle-whisper-{model_name}"));
    fs::create_dir_all(&directory)
        .map_err(|error| format!("Could not create model directory: {error}"))?;
    let fetch = |filename: &str, description: &str| {
        ensure_download(
            job_id,
            context,
            directory.join(filename),
            &config::whisper_model_url(model_name, filename),
            description,
        )
    };
    let config = fetch("config.json", "model configuration")?;
    let generation_config = fetch("generation_config.json", "generation configuration")?;
    let tokenizer = fetch("tokenizer.json", "tokenizer")?;
    let weights = fetch("model.safetensors", "model weights")?;
    let mel_filters_80 = ensure_download(
        job_id,
        context,
        directory.join("melfilters.bytes"),
        config::MEL_FILTERS_80_URL,
        "mel filters",
    )?;
    let mel_filters_128 = ensure_download(
        job_id,
        context,
        directory.join("melfilters128.bytes"),
        config::MEL_FILTERS_128_URL,
        "128-bin mel filters",
    )?;
    Ok(ModelBundle {
        config,
        generation_config,
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
        models_directory()?.join(config::VAD_MODEL_FILENAME),
        config::VAD_MODEL_URL,
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
    let progress_message = download_progress_message(description);
    if destination.is_file() {
        emit_stage(job_id, &progress_message, 100);
        return Ok(destination);
    }

    let _lock = lock_destination(&destination)?;
    if destination.is_file() {
        emit_stage(job_id, &progress_message, 100);
        return Ok(destination);
    }

    let (partial, output) = create_partial(&destination)?;
    let result = download_to_partial(job_id, context, url, description, output);
    if let Err(error) = result {
        let _ = fs::remove_file(&partial);
        return Err(error);
    }

    finish_download(&partial, &destination)?;
    emit_stage(job_id, &progress_message, 100);
    Ok(destination)
}

fn lock_destination(destination: &Path) -> Result<File, String> {
    let mut lock_name = destination.as_os_str().to_os_string();
    lock_name.push(".lock");
    let lock_file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(PathBuf::from(lock_name))
        .map_err(|error| format!("Could not open model download lock: {error}"))?;
    lock_file
        .lock()
        .map_err(|error| format!("Could not lock model download: {error}"))?;
    Ok(lock_file)
}

fn create_partial(destination: &Path) -> Result<(PathBuf, File), String> {
    loop {
        let id = NEXT_PARTIAL_ID.fetch_add(1, Ordering::Relaxed);
        let partial = destination.with_extension(format!("part.{}.{}", std::process::id(), id));
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&partial)
        {
            Ok(file) => return Ok((partial, file)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(format!("Could not create model file: {error}")),
        }
    }
}

fn finish_download(partial: &Path, destination: &Path) -> Result<(), String> {
    if destination.is_file() {
        let _ = fs::remove_file(partial);
        return Ok(());
    }
    const RETRIES: usize = 30;
    for attempt in 0..=RETRIES {
        match fs::rename(partial, destination) {
            Ok(()) => return Ok(()),
            Err(error)
                if cfg!(windows)
                    && matches!(error.raw_os_error(), Some(32 | 33))
                    && attempt < RETRIES =>
            {
                std::thread::sleep(std::time::Duration::from_secs(1));
            }
            Err(error) => {
                return Err(format!(
                    "Could not finish model download: {error}. The completed download remains at {}",
                    partial.display()
                ));
            }
        }
    }
    unreachable!("the final rename attempt always returns")
}
fn download_to_partial(
    job_id: &str,
    context: &WorkerContext,
    url: &str,
    description: &str,
    mut output: File,
) -> Result<(), String> {
    // Large Whisper weights can take hours on slow connections. Keep a finite
    // deadline so a stalled transfer does not wait forever.
    const DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(4 * 60 * 60);
    const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
    let mut response = reqwest::blocking::Client::builder()
        .user_agent("ReaSpeech/0.1")
        .timeout(DOWNLOAD_TIMEOUT)
        .connect_timeout(CONNECT_TIMEOUT)
        .build()
        .map_err(|error| error.to_string())?
        .get(url)
        .send()
        .map_err(|error| format!("{description} download failed: {error}"))?
        .error_for_status()
        .map_err(|error| format!("{description} download failed: {error}"))?;
    let total = response.content_length().unwrap_or(0);
    let mut downloaded = 0_u64;
    let mut last_reported_percent = None;
    let mut buffer = vec![0_u8; 256 * 1024];
    let progress_message = download_progress_message(description);

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
        if percent < 100 && last_reported_percent != Some(percent) {
            emit_stage(job_id, &progress_message, percent);
            last_reported_percent = Some(percent);
        }
    }

    output
        .flush()
        .map_err(|error| format!("Could not save model: {error}"))
}

fn download_progress_message(description: &str) -> String {
    format!("Downloading {description}")
}

#[cfg(test)]
mod tests {
    use super::{create_partial, download_progress_message, finish_download, lock_destination};
    use std::fs;
    use std::io::Write;
    use std::sync::mpsc;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    #[test]
    fn download_progress_identifies_the_asset() {
        assert_eq!(
            download_progress_message("model weights"),
            "Downloading model weights"
        );
    }

    #[test]
    fn concurrent_partials_cannot_replace_the_published_file() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = std::env::temp_dir().join(format!("reaspeech-download-{unique}"));
        fs::create_dir(&directory).unwrap();
        let destination = directory.join("model.safetensors");
        let (first_path, mut first) = create_partial(&destination).unwrap();
        let (second_path, mut second) = create_partial(&destination).unwrap();
        assert_ne!(first_path, second_path);
        first.write_all(b"complete first model").unwrap();
        second.write_all(b"second model").unwrap();
        drop(first);
        drop(second);

        let lock = lock_destination(&destination).unwrap();
        finish_download(&first_path, &destination).unwrap();
        finish_download(&second_path, &destination).unwrap();
        assert_eq!(fs::read(&destination).unwrap(), b"complete first model");
        assert!(!first_path.exists());
        assert!(!second_path.exists());
        drop(lock);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn destination_lock_serializes_downloads() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = std::env::temp_dir().join(format!("reaspeech-lock-{unique}"));
        fs::create_dir(&directory).unwrap();
        let destination = directory.join("model.safetensors");
        let first = lock_destination(&destination).unwrap();
        let (started_sender, started_receiver) = mpsc::channel();
        let (acquired_sender, acquired_receiver) = mpsc::channel();
        let contender = std::thread::spawn(move || {
            started_sender.send(()).unwrap();
            let _second = lock_destination(&destination).unwrap();
            acquired_sender.send(()).unwrap();
        });
        started_receiver.recv().unwrap();
        assert!(acquired_receiver
            .recv_timeout(Duration::from_millis(100))
            .is_err());
        drop(first);
        acquired_receiver
            .recv_timeout(Duration::from_secs(2))
            .unwrap();
        contender.join().unwrap();
        fs::remove_dir_all(directory).unwrap();
    }
}
#[cfg(all(test, windows))]
mod finalize_tests {
    use super::finish_download;
    use std::fs::{self, File, OpenOptions};
    use std::io::Write;
    use std::os::windows::fs::OpenOptionsExt;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    #[test]
    fn retries_publish_while_another_process_holds_the_partial_file() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = std::env::temp_dir().join(format!("reaspeech-finalize-{unique}"));
        fs::create_dir(&directory).unwrap();
        let partial = directory.join("model.part");
        let destination = directory.join("model.safetensors");
        let mut file = File::create(&partial).unwrap();
        file.write_all(b"model weights").unwrap();
        drop(file);

        // Denying FILE_SHARE_DELETE makes the first rename fail with error 32.
        let blocker = OpenOptions::new()
            .read(true)
            .share_mode(0)
            .open(&partial)
            .unwrap();
        let release = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(100));
            drop(blocker);
        });

        finish_download(&partial, &destination).unwrap();
        release.join().unwrap();
        assert_eq!(fs::read(&destination).unwrap(), b"model weights");
        fs::remove_file(destination).unwrap();
        fs::remove_dir(directory).unwrap();
    }
}
