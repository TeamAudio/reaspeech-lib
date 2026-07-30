use crate::common::WorkerContext;
use crate::transcription::{self, Request};
use serde_json::{json, Value};
use std::collections::{HashMap, VecDeque};
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::thread;
use std::time::Instant;

#[derive(Default)]
struct State {
    events: HashMap<String, VecDeque<String>>,
}

static STATE: OnceLock<Mutex<State>> = OnceLock::new();
static CONTEXT: OnceLock<WorkerContext> = OnceLock::new();
static NEXT_JOB: AtomicU64 = AtomicU64::new(1);
static RETURN_VALUE: OnceLock<Mutex<CString>> = OnceLock::new();

fn state() -> &'static Mutex<State> {
    STATE.get_or_init(Default::default)
}

fn context() -> &'static WorkerContext {
    CONTEXT.get_or_init(Default::default)
}

fn validate_options(model_name: &str, translate: bool) -> Result<(), String> {
    if !["small", "medium", "large-v3", "large-v3-turbo"].contains(&model_name) {
        return Err("model must be small, medium, large-v3, or large-v3-turbo".into());
    }
    if translate && model_name == "large-v3-turbo" {
        return Err(
            "large-v3-turbo is not trained for translation; use small, medium, or large-v3".into(),
        );
    }
    Ok(())
}

pub fn push_event(job_id: &str, event: Value) {
    let serialized = serde_json::to_string(&event).unwrap_or_else(|error| {
        format!(r#"{{"type":"error","error":"Could not serialize event: {error}"}}"#)
    });
    let mut state = state().lock().expect("state mutex poisoned");
    state
        .events
        .entry(job_id.to_owned())
        .or_default()
        .push_back(serialized);
}

fn start(
    audio_path: &str,
    model_name: &str,
    language: Option<&str>,
    translate: bool,
    vad: bool,
    words: bool,
    hotwords: Option<&str>,
) -> Result<String, String> {
    if !Path::new(audio_path).is_file() {
        return Err("audio_path does not name a readable file".into());
    }
    validate_options(model_name, translate)?;

    let job_id = format!("reaspeech-{}", NEXT_JOB.fetch_add(1, Ordering::Relaxed));
    state()
        .lock()
        .expect("state mutex poisoned")
        .events
        .insert(job_id.clone(), VecDeque::new());
    push_event(&job_id, json!({"type":"started", "jobId":job_id}));

    let request = Request {
        job_id: job_id.clone(),
        audio_path: audio_path.to_owned(),
        model_name: model_name.to_owned(),
        language: language
            .filter(|value| !value.is_empty())
            .map(str::to_owned),
        translate,
        vad,
        words,
        hotwords: hotwords
            .filter(|value| !value.trim().is_empty())
            .map(str::to_owned),
    };
    let worker_context = context().clone();
    thread::spawn(move || {
        let started = Instant::now();
        let result = transcription::run(&request, &worker_context, |segment| {
            push_event(
                &request.job_id,
                json!({
                    "type":"segment",
                    "jobId":request.job_id,
                    "segment":segment,
                }),
            );
        });
        if worker_context.cancellation.is_cancelled(&request.job_id) {
            push_event(
                &request.job_id,
                json!({"type":"cancelled", "jobId":request.job_id}),
            );
        } else {
            match result {
                Ok(()) => push_event(
                    &request.job_id,
                    json!({
                        "type":"completed",
                        "jobId":request.job_id,
                        "elapsedMs":started.elapsed().as_millis(),
                    }),
                ),
                Err(error) => push_event(
                    &request.job_id,
                    json!({"type":"error", "jobId":request.job_id, "error":error}),
                ),
            }
        }
        worker_context.cancellation.finish(&request.job_id);
    });
    Ok(job_id)
}

fn poll(job_id: &str) -> String {
    let mut state = state().lock().expect("state mutex poisoned");
    let event = state
        .events
        .get_mut(job_id)
        .and_then(VecDeque::pop_front)
        .unwrap_or_default();
    let terminal = serde_json::from_str::<Value>(&event)
        .ok()
        .and_then(|value| value["type"].as_str().map(str::to_owned))
        .is_some_and(|kind| matches!(kind.as_str(), "completed" | "cancelled" | "error"));
    if terminal {
        state.events.remove(job_id);
    }
    event
}

fn cancel(job_id: &str) -> bool {
    let exists = state()
        .lock()
        .expect("state mutex poisoned")
        .events
        .contains_key(job_id);
    if exists {
        context().cancellation.cancel(job_id);
    }
    exists
}

fn return_string(value: impl AsRef<str>) -> *mut c_void {
    let clean = value.as_ref().replace('\0', "\u{fffd}");
    let mut slot = RETURN_VALUE
        .get_or_init(|| Mutex::new(CString::default()))
        .lock()
        .expect("return value mutex poisoned");
    *slot = CString::new(clean).expect("NUL characters were replaced");
    slot.as_ptr() as *mut c_void
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
        .filter(|ptr| !ptr.is_null())
        .is_some_and(|ptr| *ptr.cast::<bool>())
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
        start(
            audio,
            model,
            Some(language),
            translate,
            vad,
            words,
            Some(hotwords),
        )
    })();
    match result {
        Ok(job_id) => return_string(job_id),
        Err(error) => return_string(format!("ERROR: {error}")),
    }
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

pub extern "C" fn start_native(
    audio_path: *const c_char,
    model_name: *const c_char,
    language: *const c_char,
    translate: *const bool,
    vad: *const bool,
    words: *const bool,
    hotwords: *const c_char,
) -> *const c_char {
    let args = [
        audio_path as *mut c_void,
        model_name as *mut c_void,
        language as *mut c_void,
        translate as *mut c_void,
        vad as *mut c_void,
        words as *mut c_void,
        hotwords as *mut c_void,
    ];
    unsafe { start_vararg(args.as_ptr() as *mut *mut c_void, args.len() as c_int) as *const c_char }
}

pub extern "C" fn poll_native(job_id: *const c_char) -> *const c_char {
    let args = [job_id as *mut c_void];
    unsafe { poll_vararg(args.as_ptr() as *mut *mut c_void, 1) as *const c_char }
}

pub extern "C" fn cancel_native(job_id: *const c_char) -> c_int {
    let args = [job_id as *mut c_void];
    unsafe { cancel_vararg(args.as_ptr() as *mut *mut c_void, 1) as isize as c_int }
}

#[cfg(test)]
mod tests {
    use super::{optional_bool_arg, poll, push_event, state, validate_options};
    use serde_json::json;
    use std::collections::VecDeque;
    use std::ffi::c_void;
    use std::ptr::null_mut;

    #[test]
    fn polling_is_fifo_and_empty_when_drained() {
        let job_id = "test-fifo";
        state()
            .lock()
            .unwrap()
            .events
            .insert(job_id.into(), VecDeque::new());
        push_event(job_id, json!({"sequence": 1}));
        push_event(job_id, json!({"sequence": 2}));

        assert_eq!(poll(job_id), r#"{"sequence":1}"#);
        assert_eq!(poll(job_id), r#"{"sequence":2}"#);
        assert_eq!(poll(job_id), "");
    }

    #[test]
    fn turbo_is_rejected_for_translation() {
        assert!(validate_options("large-v3-turbo", false).is_ok());
        assert_eq!(
            validate_options("large-v3-turbo", true).unwrap_err(),
            "large-v3-turbo is not trained for translation; use small, medium, or large-v3"
        );
        assert!(validate_options("large-v3", true).is_ok());
    }

    #[test]
    fn omitted_optional_booleans_default_to_false() {
        let enabled = true;
        let args = [
            null_mut(),
            null_mut(),
            null_mut(),
            &enabled as *const bool as *mut c_void,
        ];
        unsafe {
            assert!(optional_bool_arg(&args, 3));
            assert!(!optional_bool_arg(&args, 4));
        }
    }
}
