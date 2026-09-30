/*
 * ReaSpeech C API
 *
 * This header declares the C ABI exported by libreaspeech (built as a
 * cdylib/staticlib via `cargo build --release --no-default-features`).
 * The same functions back the REAPER extension's Lua API; see README.md
 * for the JSON event and JobOptions formats, which are identical here.
 *
 * Threading model:
 *   - Each call starts a detached worker thread that reports progress by
 *     pushing JSON events onto an in-process queue keyed by job ID.
 *   - reaspeech_poll() removes and returns the next queued event for a job
 *     as a JSON string, or an empty string when none is ready yet. Call it
 *     repeatedly (e.g. from a timer or polling loop) until a terminal event
 *     ("completed", "cancelled", or "error") is received.
 *   - The pointer returned by reaspeech_poll() refers to a single shared
 *     buffer that is overwritten by the next call to reaspeech_poll() from
 *     *any* thread. Copy the string immediately and serialize calls to
 *     reaspeech_poll() (e.g. behind a mutex) if polling from multiple
 *     threads.
 *   - MODELS_PATH may be set in the environment before the first call to
 *     select where Whisper/VAD models are downloaded and cached. It
 *     defaults to a "models" directory relative to the process's current
 *     working directory.
 */

#ifndef REASPEECH_H
#define REASPEECH_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Starts a transcription job using convenient positional arguments.
 *
 * `audio_path` and `model_name` are required; `language`, `hotwords`, and
 * their absence may be signaled with NULL. `model_name` must be one of
 * "small", "medium", "large-v3", or "large-v3-turbo".
 *
 * On success, writes the new job ID to `job_id_out` (NUL-terminated, at
 * most `job_id_out_size - 1` bytes) and returns true. On failure, writes an
 * error message to `job_id_out` instead and returns false.
 */
bool reaspeech_start(
    const char *audio_path,
    const char *model_name,
    const char *language,
    bool translate,
    bool vad,
    bool words,
    const char *hotwords,
    char *job_id_out,
    int job_id_out_size);

/*
 * Starts a transcription job using a JSON JobOptions object (see
 * README.md). Behaves like reaspeech_start() otherwise.
 */
bool reaspeech_start_ex(
    const char *audio_path,
    const char *job_options_json,
    char *job_id_out,
    int job_id_out_size);

/*
 * Removes and returns the next queued JSON event for `job_id` as a
 * NUL-terminated string, or an empty string when no event is currently
 * ready. See the threading model notes above about the lifetime of the
 * returned pointer.
 */
const char *reaspeech_poll(const char *job_id);

/*
 * Requests cancellation of `job_id`. Returns true when the job existed at
 * the time of the call; cancellation is asynchronous, so poll for the
 * terminal "cancelled" event to confirm the worker has stopped.
 */
int reaspeech_cancel(const char *job_id);

#ifdef __cplusplus
}
#endif

#endif /* REASPEECH_H */
