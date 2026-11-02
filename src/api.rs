use crate::common::WorkerContext;
use crate::transcription::{self, AudioSource, Request};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::thread;
use std::time::Instant;

#[derive(Default)]
struct State {
    events: HashMap<String, VecDeque<Event>>,
}

static STATE: OnceLock<Mutex<State>> = OnceLock::new();
static CONTEXT: OnceLock<WorkerContext> = OnceLock::new();
static NEXT_JOB: AtomicU64 = AtomicU64::new(1);
static RETURN_VALUE: OnceLock<Mutex<CString>> = OnceLock::new();

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct JobOptions {
    pub model: String,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub translate: bool,
    #[serde(default)]
    pub vad: bool,
    #[serde(default)]
    pub words: bool,
    #[serde(default)]
    pub hotwords: Option<String>,
    #[serde(default, rename = "beamSize")]
    pub beam_size: Option<usize>,
    #[serde(default, rename = "startMs")]
    pub start_ms: Option<i64>,
    #[serde(default, rename = "endMs")]
    pub end_ms: Option<i64>,
}

impl Default for JobOptions {
    fn default() -> Self {
        Self {
            model: "small".into(),
            language: None,
            translate: false,
            vad: false,
            words: false,
            hotwords: None,
            beam_size: None,
            start_ms: None,
            end_ms: None,
        }
    }
}

pub use crate::transcription::{Segment, Word};

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Event {
    Started {
        #[serde(rename = "jobId")]
        job_id: String,
    },
    Progress {
        #[serde(rename = "jobId")]
        job_id: String,
        completed: u64,
        total: u64,
        message: String,
    },
    Segment {
        #[serde(rename = "jobId")]
        job_id: String,
        segment: Segment,
    },
    #[serde(rename = "language")]
    DetectedLanguage {
        #[serde(rename = "jobId")]
        job_id: String,
        language: String,
    },
    Completed {
        #[serde(rename = "jobId")]
        job_id: String,
        #[serde(rename = "elapsedMs")]
        elapsed_ms: u64,
    },
    Cancelled {
        #[serde(rename = "jobId")]
        job_id: String,
    },
    Error {
        #[serde(rename = "jobId", skip_serializing_if = "Option::is_none")]
        job_id: Option<String>,
        error: String,
    },
}

impl Event {
    fn is_terminal(&self) -> bool {
        matches!(
            self,
            Self::Completed { .. } | Self::Cancelled { .. } | Self::Error { .. }
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Job {
    id: String,
}

impl Job {
    pub fn id(&self) -> &str {
        &self.id
    }
    pub fn poll(&self) -> Option<Event> {
        poll_event(&self.id)
    }
    pub fn cancel(&self) -> bool {
        cancel(&self.id)
    }
}

fn state() -> &'static Mutex<State> {
    STATE.get_or_init(Default::default)
}

fn context() -> &'static WorkerContext {
    CONTEXT.get_or_init(Default::default)
}

fn validate_options(
    model_name: &str,
    translate: bool,
    beam_size: Option<usize>,
) -> Result<(), String> {
    if !["small", "medium", "large-v3", "large-v3-turbo"].contains(&model_name) {
        return Err("model must be small, medium, large-v3, or large-v3-turbo".into());
    }
    if translate && model_name == "large-v3-turbo" {
        return Err(
            "large-v3-turbo is not trained for translation; use small, medium, or large-v3".into(),
        );
    }
    if beam_size.is_some_and(|size| !(1..=5).contains(&size)) {
        return Err("beamSize must be between 1 and 5".into());
    }
    Ok(())
}

pub(crate) fn push_event(job_id: &str, event: Event) {
    let mut state = state()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    state
        .events
        .entry(job_id.to_owned())
        .or_default()
        .push_back(event);
}

fn start_job(audio_path: &str, options: JobOptions) -> Result<String, String> {
    if !Path::new(audio_path).is_file() {
        return Err("audio_path does not name a readable file".into());
    }
    start_source(AudioSource::File(audio_path.to_owned()), options)
}

fn start_source(source: AudioSource, options: JobOptions) -> Result<String, String> {
    validate_options(&options.model, options.translate, options.beam_size)?;
    if options.start_ms.is_some_and(|value| value < 0)
        || options.end_ms.is_some_and(|value| value < 0)
        || matches!((options.start_ms, options.end_ms), (Some(start), Some(end)) if end <= start)
    {
        return Err(
            "audio range must have non-negative bounds and endMs must exceed startMs".into(),
        );
    }

    let job_id = format!("reaspeech-{}", NEXT_JOB.fetch_add(1, Ordering::Relaxed));
    state()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .events
        .insert(job_id.clone(), VecDeque::new());
    push_event(
        &job_id,
        Event::Started {
            job_id: job_id.clone(),
        },
    );

    let request = Request {
        job_id: job_id.clone(),
        source,
        model_name: options.model,
        language: options.language.filter(|value| !value.is_empty()),
        translate: options.translate,
        vad: options.vad,
        words: options.words,
        hotwords: options.hotwords.filter(|value| !value.trim().is_empty()),
        beam_size: options.beam_size,
        start_ms: options.start_ms,
        end_ms: options.end_ms,
    };
    let worker_context = context().clone();
    let worker_job_id = job_id.clone();
    thread::spawn(move || {
        let started = Instant::now();
        let result = catch_unwind(AssertUnwindSafe(|| {
            transcription::run(
                request,
                &worker_context,
                |language| {
                    push_event(
                        &worker_job_id,
                        Event::DetectedLanguage {
                            job_id: worker_job_id.clone(),
                            language: language.to_owned(),
                        },
                    );
                },
                |segment| {
                    push_event(
                        &worker_job_id,
                        Event::Segment {
                            job_id: worker_job_id.clone(),
                            segment: segment.clone(),
                        },
                    );
                },
            )
        }))
        .unwrap_or_else(|payload| {
            let detail = payload
                .downcast_ref::<&str>()
                .copied()
                .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
                .unwrap_or("unknown panic");
            Err(format!("Transcription worker panicked: {detail}"))
        });
        if worker_context.cancellation.is_cancelled(&worker_job_id) {
            push_event(
                &worker_job_id,
                Event::Cancelled {
                    job_id: worker_job_id.clone(),
                },
            );
        } else {
            match result {
                Ok(()) => push_event(
                    &worker_job_id,
                    Event::Completed {
                        job_id: worker_job_id.clone(),
                        elapsed_ms: started.elapsed().as_millis().min(u64::MAX as u128) as u64,
                    },
                ),
                Err(error) => push_event(
                    &worker_job_id,
                    Event::Error {
                        job_id: Some(worker_job_id.clone()),
                        error,
                    },
                ),
            }
        }
        worker_context.cancellation.finish(&worker_job_id);
    });
    Ok(job_id)
}

fn poll_event(job_id: &str) -> Option<Event> {
    let mut state = state()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let event = state.events.get_mut(job_id).and_then(VecDeque::pop_front);
    if event.as_ref().is_some_and(Event::is_terminal) {
        state.events.remove(job_id);
    }
    event
}

fn poll(job_id: &str) -> String {
    poll_event(job_id)
        .and_then(|event| serde_json::to_string(&event).ok())
        .unwrap_or_default()
}

fn cancel(job_id: &str) -> bool {
    let exists = state()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .events
        .contains_key(job_id);
    if exists {
        context().cancellation.cancel(job_id);
    }
    exists
}

/// Starts a transcription job for native Rust clients using the same worker
/// and event queue as the REAPER extension API.
pub fn start_json(audio_path: &str, job_options_json: &str) -> Result<String, String> {
    let options = serde_json::from_str::<JobOptions>(job_options_json)
        .map_err(|error| format!("invalid job options: {error}"))?;
    start_job(audio_path, options)
}

/// Starts an asynchronous transcription job using native Rust options and events.
pub fn start(audio_path: impl AsRef<Path>, options: JobOptions) -> Result<Job, String> {
    let audio_path = audio_path.as_ref().to_string_lossy();
    start_job(&audio_path, options).map(|id| Job { id })
}

/// Starts a job from interleaved floating-point PCM (nominal range -1 to 1).
/// `samples.len()` must be a nonzero multiple of `channels`. The sample rate
/// and channel count must be nonzero, and all samples must be finite.
/// Samples are copied before returning; the caller can immediately reuse them.
/// Range options and event timestamps use the same semantics as file input.
pub fn start_from_memory(
    samples: &[f32],
    sample_rate: u32,
    channels: u32,
    options: JobOptions,
) -> Result<Job, String> {
    validate_pcm(samples, sample_rate, channels)?;
    start_source(
        AudioSource::Pcm {
            samples: samples.to_vec(),
            sample_rate,
            channels,
        },
        options,
    )
    .map(|id| Job { id })
}

fn validate_pcm(samples: &[f32], sample_rate: u32, channels: u32) -> Result<(), String> {
    if sample_rate == 0 || channels == 0 {
        return Err("sample_rate and channels must be nonzero".into());
    }
    if samples.is_empty() || samples.len() % channels as usize != 0 {
        return Err("samples must contain a nonempty number of complete interleaved frames".into());
    }
    if samples.iter().any(|sample| !sample.is_finite()) {
        return Err("samples must be finite".into());
    }
    Ok(())
}

/// Removes and returns the next serialized event, if one is ready.
pub fn poll_json(job_id: &str) -> Option<String> {
    let event = poll(job_id);
    (!event.is_empty()).then_some(event)
}

/// Requests cancellation of a native Rust client's job.
pub fn cancel_job(job_id: &str) -> bool {
    cancel(job_id)
}

fn return_string(value: impl AsRef<str>) -> *mut c_void {
    let clean = value.as_ref().replace('\0', "\u{fffd}");
    let mut slot = RETURN_VALUE
        .get_or_init(|| Mutex::new(CString::default()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    *slot = CString::new(clean).expect("NUL characters were replaced");
    slot.as_ptr() as *mut c_void
}

fn write_output(value: &str, output: *mut c_void, output_size: c_int) {
    if output.is_null() || output_size <= 0 {
        return;
    }
    let bytes = value.as_bytes();
    let length = bytes.len().min(output_size as usize - 1);
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), output.cast::<u8>(), length);
        *output.cast::<u8>().add(length) = 0;
    }
}

fn return_start_result(
    result: Result<String, String>,
    job_id_out: *mut c_void,
    job_id_out_size: c_int,
) -> *mut c_void {
    let (success, value) = match result {
        Ok(job_id) => (true, job_id),
        Err(error) => (false, error),
    };
    write_output(&value, job_id_out, job_id_out_size);
    success as isize as *mut c_void
}

unsafe fn string_arg(args: &[*mut c_void], index: usize) -> Result<&str, String> {
    let ptr = args
        .get(index)
        .copied()
        .filter(|ptr| !ptr.is_null())
        .ok_or_else(|| format!("missing argument {}", index + 1))?;
    CStr::from_ptr(ptr.cast::<c_char>())
        .to_str()
        .map_err(|_| format!("argument {} is not UTF-8", index + 1))
}

unsafe fn args<'a>(arglist: *mut *mut c_void, count: c_int) -> &'a [*mut c_void] {
    if arglist.is_null() || count <= 0 {
        &[]
    } else {
        std::slice::from_raw_parts(arglist, count as usize)
    }
}

unsafe fn optional_bool_arg(args: &[*mut c_void], index: usize) -> bool {
    args.get(index)
        .copied()
        .is_some_and(|value| value as isize != 0)
}

pub unsafe extern "C" fn start_vararg(arglist: *mut *mut c_void, count: c_int) -> *mut c_void {
    let args = args(arglist, count);
    let result = (|| {
        let audio = string_arg(args, 0)?;
        let model = string_arg(args, 1)?;
        let language = string_arg(args, 2).unwrap_or("");
        let translate = optional_bool_arg(args, 3);
        let vad = optional_bool_arg(args, 4);
        let words = optional_bool_arg(args, 5);
        let hotwords = string_arg(args, 6).unwrap_or("");
        start_job(
            audio,
            JobOptions {
                model: model.to_owned(),
                language: Some(language.to_owned()),
                translate,
                vad,
                words,
                hotwords: Some(hotwords.to_owned()),
                beam_size: None,
                start_ms: None,
                end_ms: None,
            },
        )
    })();
    return_start_result(
        result,
        args.get(7).copied().unwrap_or(std::ptr::null_mut()),
        args.get(8).copied().unwrap_or(std::ptr::null_mut()) as isize as c_int,
    )
}

pub unsafe extern "C" fn start_ex_vararg(arglist: *mut *mut c_void, count: c_int) -> *mut c_void {
    let args = args(arglist, count);
    let result = (|| {
        let audio = string_arg(args, 0)?;
        let options_json = string_arg(args, 1)?;
        let options = serde_json::from_str::<JobOptions>(options_json)
            .map_err(|error| format!("invalid job_options_json: {error}"))?;
        start_job(audio, options)
    })();
    return_start_result(
        result,
        args.get(2).copied().unwrap_or(std::ptr::null_mut()),
        args.get(3).copied().unwrap_or(std::ptr::null_mut()) as isize as c_int,
    )
}

pub unsafe extern "C" fn poll_vararg(arglist: *mut *mut c_void, count: c_int) -> *mut c_void {
    let args = args(arglist, count);
    match string_arg(args, 0) {
        Ok(job_id) => return_string(poll(job_id)),
        Err(error) => return_string(format!(r#"{{"type":"error","error":"{error}"}}"#)),
    }
}

pub unsafe extern "C" fn cancel_vararg(arglist: *mut *mut c_void, count: c_int) -> *mut c_void {
    let args = args(arglist, count);
    string_arg(args, 0).map(cancel).unwrap_or(false) as isize as *mut c_void
}

/// Starts a native C job from interleaved f32 PCM, copied before returning.
///
/// # Safety
/// `samples` must point to `sample_count` initialized, readable, aligned f32
/// values. JSON must be a valid NUL-terminated string. The output buffer must
/// be writable for `job_id_out_size` bytes when non-null and the size is positive.
#[no_mangle]
pub unsafe extern "C" fn reaspeech_start_from_memory(
    samples: *const f32,
    sample_count: usize,
    sample_rate: u32,
    channels: u32,
    job_options_json: *const c_char,
    job_id_out: *mut c_char,
    job_id_out_size: c_int,
) -> bool {
    let result = (|| {
        if samples.is_null()
            || (samples as usize) % std::mem::align_of::<f32>() != 0
            || sample_count > isize::MAX as usize / std::mem::size_of::<f32>()
        {
            return Err("invalid samples pointer or sample_count".into());
        }
        if job_options_json.is_null() {
            return Err("missing job_options_json".into());
        }
        let json = CStr::from_ptr(job_options_json)
            .to_str()
            .map_err(|_| "job_options_json is not UTF-8".to_owned())?;
        let options = serde_json::from_str::<JobOptions>(json)
            .map_err(|error| format!("invalid job_options_json: {error}"))?;
        start_from_memory(
            std::slice::from_raw_parts(samples, sample_count),
            sample_rate,
            channels,
            options,
        )
        .map(|job| job.id)
    })();
    !return_start_result(result, job_id_out.cast(), job_id_out_size).is_null()
}

/// C ABI entry point. See `reaspeech.h` for documentation.
#[no_mangle]
pub extern "C" fn reaspeech_start(
    audio_path: *const c_char,
    model_name: *const c_char,
    language: *const c_char,
    translate: bool,
    vad: bool,
    words: bool,
    hotwords: *const c_char,
    job_id_out: *mut c_char,
    job_id_out_size: c_int,
) -> bool {
    let args = [
        audio_path as *mut c_void,
        model_name as *mut c_void,
        language as *mut c_void,
        translate as isize as *mut c_void,
        vad as isize as *mut c_void,
        words as isize as *mut c_void,
        hotwords as *mut c_void,
        job_id_out as *mut c_void,
        job_id_out_size as isize as *mut c_void,
    ];
    unsafe { !start_vararg(args.as_ptr() as *mut *mut c_void, args.len() as c_int).is_null() }
}

/// C ABI entry point. See `reaspeech.h` for documentation.
#[no_mangle]
pub extern "C" fn reaspeech_start_ex(
    audio_path: *const c_char,
    job_options_json: *const c_char,
    job_id_out: *mut c_char,
    job_id_out_size: c_int,
) -> bool {
    let args = [
        audio_path as *mut c_void,
        job_options_json as *mut c_void,
        job_id_out as *mut c_void,
        job_id_out_size as isize as *mut c_void,
    ];
    unsafe { !start_ex_vararg(args.as_ptr() as *mut *mut c_void, args.len() as c_int).is_null() }
}

/// C ABI entry point. See `reaspeech.h` for documentation.
///
/// The returned pointer refers to a shared buffer that is overwritten by the
/// next call to `reaspeech_poll` from any thread. Copy the string before
/// making another call, and serialize calls from multiple threads.
#[no_mangle]
pub extern "C" fn reaspeech_poll(job_id: *const c_char) -> *const c_char {
    let args = [job_id as *mut c_void];
    unsafe { poll_vararg(args.as_ptr() as *mut *mut c_void, 1) as *const c_char }
}

/// C ABI entry point. See `reaspeech.h` for documentation.
#[no_mangle]
pub extern "C" fn reaspeech_cancel(job_id: *const c_char) -> c_int {
    let args = [job_id as *mut c_void];
    unsafe { cancel_vararg(args.as_ptr() as *mut *mut c_void, 1) as isize as c_int }
}

#[cfg(test)]
mod tests {
    use super::{
        optional_bool_arg, poll, push_event, state, validate_options, write_output, Event,
        JobOptions,
    };
    use std::collections::VecDeque;
    use std::ffi::{c_int, c_void};
    use std::ptr::null_mut;

    #[test]
    fn writes_bounded_output_string() {
        let mut output = [b'x'; 5];
        write_output(
            "reaspeech-1",
            output.as_mut_ptr().cast(),
            output.len() as c_int,
        );
        assert_eq!(&output, b"reas\0");
    }

    #[test]
    fn start_result_writes_error_to_need_big_buffer() {
        let mut output = [0_u8; 64];
        let result = super::return_start_result(
            Err("test error".to_owned()),
            output.as_mut_ptr().cast(),
            output.len() as c_int,
        );
        assert!(result.is_null());
        assert_eq!(
            std::ffi::CStr::from_bytes_until_nul(&output)
                .unwrap()
                .to_str()
                .unwrap(),
            "test error"
        );
    }

    #[test]
    fn polling_is_fifo_and_empty_when_drained() {
        let job_id = "test-fifo";
        state()
            .lock()
            .unwrap()
            .events
            .insert(job_id.into(), VecDeque::new());
        push_event(
            job_id,
            Event::Started {
                job_id: job_id.into(),
            },
        );
        push_event(
            job_id,
            Event::Progress {
                job_id: job_id.into(),
                completed: 1,
                total: 2,
                message: "Working".into(),
            },
        );

        assert_eq!(poll(job_id), r#"{"type":"started","jobId":"test-fifo"}"#);
        assert_eq!(
            poll(job_id),
            r#"{"type":"progress","jobId":"test-fifo","completed":1,"total":2,"message":"Working"}"#
        );
        assert_eq!(poll(job_id), "");
    }

    #[test]
    fn detected_language_event_uses_the_json_api_shape() {
        let event = Event::DetectedLanguage {
            job_id: "reaspeech-1".into(),
            language: "ja".into(),
        };

        assert_eq!(
            serde_json::to_string(&event).unwrap(),
            r#"{"type":"language","jobId":"reaspeech-1","language":"ja"}"#
        );
    }

    #[test]
    fn turbo_is_rejected_for_translation() {
        assert!(validate_options("large-v3-turbo", false, None).is_ok());
        assert_eq!(
            validate_options("large-v3-turbo", true, None).unwrap_err(),
            "large-v3-turbo is not trained for translation; use small, medium, or large-v3"
        );
        assert!(validate_options("large-v3", true, None).is_ok());
    }

    #[test]
    fn omitted_optional_booleans_default_to_false() {
        let args = [null_mut(), null_mut(), null_mut(), 1_isize as *mut c_void];
        unsafe {
            assert!(optional_bool_arg(&args, 3));
            assert!(!optional_bool_arg(&args, 4));
        }
    }

    #[test]
    fn job_options_defaults_optional_values() {
        let options: JobOptions = serde_json::from_str(r#"{"model":"small"}"#).unwrap();
        assert_eq!(
            options,
            JobOptions {
                model: "small".into(),
                language: None,
                translate: false,
                vad: false,
                words: false,
                hotwords: None,
                beam_size: None,
                start_ms: None,
                end_ms: None,
            }
        );
    }

    #[test]
    fn job_options_accept_and_validate_beam_size() {
        let options: JobOptions =
            serde_json::from_str(r#"{"model":"small","beamSize":3}"#).unwrap();
        assert_eq!(options.beam_size, Some(3));
        assert!(validate_options(&options.model, options.translate, options.beam_size).is_ok());
        assert_eq!(
            validate_options("small", false, Some(0)).unwrap_err(),
            "beamSize must be between 1 and 5"
        );
        assert_eq!(
            validate_options("small", false, Some(6)).unwrap_err(),
            "beamSize must be between 1 and 5"
        );
    }

    #[test]
    fn job_options_accept_audio_range() {
        let options: JobOptions =
            serde_json::from_str(r#"{"model":"small","startMs":1250,"endMs":4250}"#).unwrap();
        assert_eq!(options.start_ms, Some(1250));
        assert_eq!(options.end_ms, Some(4250));
    }

    #[test]
    fn job_options_reject_unknown_fields() {
        let error =
            serde_json::from_str::<JobOptions>(r#"{"model":"small","word":true}"#).unwrap_err();
        assert!(error.to_string().contains("unknown field `word`"));
    }
}

#[cfg(test)]
mod memory_tests {
    use super::*;

    #[test]
    fn rejects_invalid_pcm_without_starting_a_worker() {
        for (samples, rate, channels) in [
            (vec![], 16000, 1),
            (vec![0.0], 0, 1),
            (vec![0.0], 16000, 0),
            (vec![0.0], 16000, 2),
            (vec![f32::NAN], 16000, 1),
            (vec![f32::INFINITY], 16000, 1),
        ] {
            assert!(start_from_memory(&samples, rate, channels, JobOptions::default()).is_err());
        }
    }

    #[test]
    fn c_memory_input_reports_invalid_pointer() {
        let mut output = [0i8; 128];
        let ok = unsafe {
            reaspeech_start_from_memory(
                std::ptr::null(),
                1,
                16000,
                1,
                std::ptr::null(),
                output.as_mut_ptr(),
                output.len() as c_int,
            )
        };
        assert!(!ok);
        assert!(unsafe { CStr::from_ptr(output.as_ptr()) }
            .to_str()
            .unwrap()
            .contains("samples"));
    }
}
