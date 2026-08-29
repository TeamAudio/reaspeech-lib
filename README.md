# ReaSpeech Lib

ReaSpeech Lib is a REAPER extension written in Rust. It exposes a Whisper
transcription engine to Lua/ReaScript without blocking REAPER's main thread.

## Lua API

```lua
retval, job_id = reaper.ReaSpeech_Start(audio_path, model, language, translate, vad, words, hotwords)
retval, job_id = reaper.ReaSpeech_StartEx(audio_path, job_options_json)
event_json = reaper.ReaSpeech_Poll(job_id)
accepted = reaper.ReaSpeech_Cancel(job_id)
```

### API functions

| Function | Returns | Description |
| --- | --- | --- |
| `ReaSpeech_Start(audio_path, model, language, translate, vad, words, hotwords)` | Boolean, job ID or error | Starts a job using convenient positional arguments. Only `audio_path` and `model` are required. |
| `ReaSpeech_StartEx(audio_path, job_options_json)` | Boolean, job ID or error | Starts a job using a JSON `JobOptions` object. |
| `ReaSpeech_Poll(job_id)` | JSON event or `""` | Removes and returns the next queued event. An empty string means no event is currently ready. |
| `ReaSpeech_Cancel(job_id)` | Boolean | Requests cancellation. Returns `true` when the job exists. |

When a start function returns `false`, its second return value contains the
error message instead of a job ID.

Only `audio_path` and `model` are required. `language` defaults to automatic
language detection, while `translate`, `vad`, and `words` default to `false`.
`hotwords` is an optional string of names, terminology, or phrases that should
be favored during recognition. For example:

```lua
local hotwords = "ReaSpeech, REAPER, Cockos, spectral edit"
local ok, job_id = reaper.ReaSpeech_Start(path, "small", "en", false, true, false, hotwords)
if not ok then
  reaper.ShowMessageBox(job_id, "ReaSpeech", 0)
  return
end
```

Hotwords are hints rather than a restricted vocabulary or guaranteed output.
Separate multiple entries with commas or newlines.

For named options and future extensibility, use `ReaSpeech_StartEx`. Its
`job_options_json` argument is a JSON object with the following fields:

| `JobOptions` field | Type | Required | Default |
| --- | --- | --- | --- |
| `model` | String | Yes | — |
| `language` | String | No | Automatic language detection |
| `translate` | Boolean | No | `false` |
| `vad` | Boolean | No | `false` |
| `words` | Boolean | No | `false` |
| `hotwords` | String | No | No hotwords |
| `beamSize` | Integer | No | `1` |

Unknown fields are rejected so that misspelled option names do not silently
change transcription behavior. `beamSize` must be between `1` and `5`; larger
values may improve decoding quality at the cost of additional processing. For
example:

```lua
local options = [[{
  "model": "small",
  "language": "en",
  "vad": true,
  "beamSize": 3,
  "hotwords": "ReaSpeech, REAPER, Cockos"
}]]
local ok, job_id = reaper.ReaSpeech_StartEx(path, options)
if not ok then
  reaper.ShowMessageBox(job_id, "ReaSpeech", 0)
  return
end
```

`model` is `small`, `medium`, `large-v3`, or `large-v3-turbo`.
`large-v3-turbo` is transcription-only; select `small`, `medium`, or `large-v3`
when `translate` is enabled. `Poll` returns one queued
event at a time and returns `""` when no event is ready. Event types are
`started`, `progress`, `segment`, `completed`, `cancelled`, and `error`. Each
recognized segment is emitted immediately:

```json
{"type":"segment","jobId":"reaspeech-1","segment":{"startMs":0,"endMs":820,"text":"Hello","probability":0.94}}
```

When `words` is true, each segment also has a `words` array containing
structures like the following:

```json
{"word":"Hello","start":0.0,"end":0.82,"probability":0.94}
```

The terminal `completed` event contains timing metadata:

```json
{"type":"completed","jobId":"reaspeech-1","elapsedMs":1234}
```

See `examples/transcribe.lua` for a complete deferred polling loop. The
`examples/imgui` directory contains a ReaImGui interface for transcribing
selected media items interactively.

## Rust library API

Rust applications can construct an
[`api::JobOptions`](src/api.rs), call `api::start`, and poll the returned
`api::Job` for typed `api::Event` values. JSON remains an implementation detail
of the REAPER/ReaScript compatibility boundary.

```rust,no_run
use reaspeech::api::{self, Event, JobOptions};

let job = api::start("speech.wav", JobOptions {
    model: "small".into(),
    vad: true,
    ..JobOptions::default()
})?;

loop {
    match job.poll() {
        Some(Event::Segment { segment, .. }) => println!("{}", segment.text),
        Some(Event::Completed { .. } | Event::Cancelled { .. }) => break,
        Some(Event::Error { error, .. }) => return Err(error),
        Some(_) => {}
        None => std::thread::sleep(std::time::Duration::from_millis(20)),
    }
}
# Ok::<(), String>(())
```

## Build and install

Install the [Rust toolchain](https://rustup.rs/) and build the extension from
the repository root:

```sh
cargo build --release
```

On Windows, use the MSVC Rust toolchain and run these commands in Command
Prompt (`cmd.exe`):

```bat
rustup default stable-msvc
cargo build --release
copy target\release\reaspeech.dll "%APPDATA%\REAPER\UserPlugins\reaper_reaspeech.dll"
```

On macOS:

```sh
cargo build --release
mkdir -p "$HOME/Library/Application Support/REAPER/UserPlugins"
cp target/release/libreaspeech.dylib \
  "$HOME/Library/Application Support/REAPER/UserPlugins/reaper_reaspeech.dylib"
```

On Linux:

```sh
cargo build --release
mkdir -p "$HOME/.config/REAPER/UserPlugins"
cp target/release/libreaspeech.so \
  "$HOME/.config/REAPER/UserPlugins/reaper_reaspeech.so"
```

If REAPER uses a portable installation or a custom resource directory, copy
the extension into its `UserPlugins` directory instead. You can find the
active directory in REAPER with **Options > Show REAPER resource path in
explorer/finder**.

Restart REAPER after copying the extension. Models are downloaded on first use
to `<REAPER resource path>/ReaSpeech/models`. To override that location, set
`MODELS_PATH` before starting REAPER:

Windows Command Prompt:

```bat
set "MODELS_PATH=D:\path\to\models"
start "" "C:\Program Files\REAPER (x64)\reaper.exe"
```

macOS or Linux:

```sh
MODELS_PATH="/path/to/models" /path/to/reaper
```

Hardware-accelerated builds are available with
`cargo build --release --features cuda` on supported Windows and Linux systems, or
`cargo build --release --features metal` on macOS. The project carries a small
patch to Candle's Metal kernels that omits unused BF16 GEMV variants on older
Metal language targets, allowing the F32 Whisper implementation to use Metal
on M1 and later Macs.

## Performance profiling

Set `REASPEECH_PROFILE` before starting REAPER to log detailed stage timings.
Set `REASPEECH_LOG_PATH` as well to save the output to a file:

```bat
set "REASPEECH_PROFILE=1"
set "REASPEECH_LOG_PATH=%TEMP%\reaspeech-profile.log"
start "" "C:\Program Files\REAPER (x64)\reaper.exe"
```

Profiling reports audio decoding, asset lookup, model loading, VAD and language
setup, mel generation, encoder execution, decoder execution, vocabulary
projection, logits rules/transfers, and beam selection. GPU operations are
synchronized at stage boundaries to make their timings meaningful, so use the
profile to locate bottlenecks rather than as the final production benchmark.

## Continuous integration builds

The `Build` GitHub Actions workflow builds and uploads ready-to-install
extensions for:

- Windows x86_64 and Linux x86_64 using the CPU backend
- macOS arm64 using Metal
- Windows x86_64 and Linux x86_64 using CUDA 12 or CUDA 13

CUDA 13 artifact names end in `-cuda13`; CUDA 12 artifacts end in `-cuda12`
and use CUDA Toolkit 12.5 for compatibility with older NVIDIA drivers. Every
artifact contains the extension under the filename
REAPER expects in `UserPlugins`, ready for future publication through the
TeamAudio ReaPack repository.

Releases are published locally rather than by GitHub Actions. See
[`RELEASE.md`](RELEASE.md) for the backend-package design and release process.

An Intel macOS build is not currently produced because the pinned ONNX Runtime
release does not provide prebuilt `x86_64-apple-darwin` binaries.

## Design

Every request runs on a Rust worker thread. The worker only writes serialized
events to a synchronized queue. Lua reads that queue from REAPER's main thread
with `ReaSpeech_Poll`; no Lua or REAPER function is called from worker threads.
