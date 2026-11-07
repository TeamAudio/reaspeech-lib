# ReaSpeech Lib

ReaSpeech Lib is a Whisper transcription engine written in Rust. It is
available as a REAPER extension exposing a Lua/ReaScript API, as a native
Rust library, and as a C library that can be embedded in other applications.

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
| `startMs` | Integer | No | Start of the source file |
| `endMs` | Integer | No | End of the source file |

Unknown fields are rejected so that misspelled option names do not silently
change transcription behavior. `beamSize` must be between `1` and `5`; larger
values may improve decoding quality at the cost of additional processing.
`startMs` and `endMs` restrict inference to a source-file range expressed in
milliseconds. Bounds must be non-negative, and when both are present `endMs`
must be greater than `startMs`. Segment and word timestamps remain relative to
the original source file, not to the beginning of the requested range. For
example:

```lua
local options = [[{
  "model": "small",
  "language": "en",
  "vad": true,
  "beamSize": 3,
  "hotwords": "ReaSpeech, REAPER, Cockos",
  "startMs": 15000,
  "endMs": 45000
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
`started`, `progress`, `language`, `segment`, `completed`, `cancelled`, and
`error`. When automatic language detection is requested and speech is found,
the detected language is emitted before the first segment:

```json
{"type":"language","jobId":"reaspeech-1","language":"en"}
```

Each recognized segment is emitted immediately:

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
    start_ms: Some(15_000),
    end_ms: Some(45_000),
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

## C API

Applications written in C, C++, or any language with a C FFI can embed
ReaSpeech directly, without REAPER. The **Build** workflow produces prebuilt
shared-library SDKs named `reaspeech-sdk-<platform>-<architecture>-<backend>.zip`.
The release process publishes these archives as
[GitHub release assets](https://github.com/TeamAudio/reaspeech-lib/releases).
Consumers need no Rust toolchain.

SDKs cover Windows x86_64 and Linux x86_64 (CPU, CUDA 12, CUDA 13), and
macOS arm64 (Metal). Each archive contains a top-level directory with:

- `include/reaspeech.h`: the C ABI header.
- `bin/reaspeech.dll` and `lib/reaspeech.dll.lib` on Windows; the `.lib` is
  the DLL import library, not a static Rust library.
- `lib/libreaspeech.so` on Linux or `lib/libreaspeech.dylib` on macOS.
- `lib/cmake/ReaSpeech/ReaSpeechConfig.cmake`, README, license, and file hashes.
- Any shared ONNX Runtime libraries copied by the build, beside ReaSpeech.
- `examples/c-sdk/`: C and C++ smoke tests.

Select one backend, extract the archive, and point `CMAKE_PREFIX_PATH` at its
extracted top-level directory:

```cmake
find_package(ReaSpeech CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE ReaSpeech::reaspeech)
```

Copy the SDK runtime libraries into your application's distribution. On Windows,
put every DLL from `bin/` beside the executable. On Linux and macOS, copy the
shared libraries from `lib/` to your application's library directory and configure
its loader search path (`$ORIGIN` or `@rpath`) accordingly. The macOS library uses
`@rpath/libreaspeech.dylib` as its install name.

To run the bundled smoke tests, open a terminal in the extracted SDK's
top-level directory with CMake 3.21 or newer and a C/C++ compiler:

```sh
cmake -S examples/c-sdk -B build-smoke
cmake --build build-smoke --config Release
ctest --test-dir build-smoke -C Release --output-on-failure
```

The tests exercise the C API without downloading models. Running them with a
CUDA SDK requires the matching CUDA runtime and NVIDIA driver. CI builds the
example from each extracted archive and runs it for CPU and Metal packages.

Linux SDKs are built on Ubuntu 24.04 and require a compatible system runtime;
they do not target older glibc distributions. Windows SDKs use MSVC and require
the Microsoft Visual C++ runtime. macOS SDKs target macOS 11 or newer. CUDA SDKs
also require the matching CUDA runtime libraries and a compatible NVIDIA driver;
CUDA is not bundled. Model weights are downloaded separately on first use; set
`MODELS_PATH` to a writable cache directory before starting jobs. SDK archives
have adjacent `.zip.sha256` files for download verification.

To build from source instead, disable the `reaper-extension` feature (its default
REAPER dependencies are skipped):

```sh
cargo build --release --no-default-features
```

This produces a `cdylib` (`libreaspeech.so`/`.dylib`/`reaspeech.dll`) and a
`staticlib` (`libreaspeech.a`/`reaspeech.lib`) alongside the usual `rlib`.
Link against either one and include [`include/reaspeech.h`](include/reaspeech.h),
which declares the exported functions:

```c
#include "reaspeech.h"

char job_id[64];
if (!reaspeech_start("speech.wav", "small", NULL, false, true, false, NULL,
                      job_id, sizeof job_id)) {
    fprintf(stderr, "start failed: %s\n", job_id);
    return 1;
}

for (;;) {
    const char *event = reaspeech_poll(job_id);
    if (event[0] == '\0') {
        /* nothing ready yet; sleep briefly and poll again */
        continue;
    }
    puts(event); /* a JSON event; see the Lua API section above for shapes */
    if (strstr(event, "\"completed\"") || strstr(event, "\"cancelled\"") ||
        strstr(event, "\"error\"")) {
        break;
    }
}
```

`reaspeech_start_ex` accepts the same JSON `JobOptions` object documented
above instead of positional arguments, and `reaspeech_cancel` requests
cancellation of a running job. See the header for full documentation,
including the threading and buffer-lifetime rules for `reaspeech_poll`.

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

## Native audio from memory

C and Rust clients can submit interleaved `f32` PCM without writing a file.
Use `api::start_from_memory(&samples, sample_rate, channels, options)` in Rust,
or `reaspeech_start_from_memory(samples, sample_count, sample_rate, channels,
options_json, job_id, sizeof(job_id))` in C (see `include/reaspeech.h`).
`sample_count` is the total number of floats across all channels. Samples
are nominally in -1..1, must be finite, and must contain complete, nonempty
frames. Sample rate and channel count must be positive. The library copies
samples before returning, so the caller can immediately release its buffer.
The worker averages channels to mono and resamples to 16 kHz using the same
resampler as file input. Range options, timestamps, polling and cancellation
are unchanged. This entry point is not registered with REAPER/Lua.
