use crate::api::push_event;
use serde_json::json;
use std::collections::HashSet;
use std::sync::{Arc, Mutex};

#[derive(Clone, Default)]
pub struct Cancellation {
    cancelled: Arc<Mutex<HashSet<String>>>,
}

impl Cancellation {
    pub fn cancel(&self, job_id: &str) {
        self.cancelled
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(job_id.to_owned());
    }

    pub fn is_cancelled(&self, job_id: &str) -> bool {
        self.cancelled
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .contains(job_id)
    }

    pub fn finish(&self, job_id: &str) {
        self.cancelled
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(job_id);
    }
}

#[derive(Clone, Default)]
pub struct WorkerContext {
    pub cancellation: Cancellation,
}

pub fn emit_progress(job_id: &str, message: &str, completed: u64, total: u64) {
    push_event(
        job_id,
        json!({
            "type": "progress",
            "jobId": job_id,
            "completed": completed,
            "total": total,
            "message": message,
        }),
    );
}
