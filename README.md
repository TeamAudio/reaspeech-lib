# ReaSpeech Lib

ReaSpeech Lib is a REAPER extension written in Rust. It exposes a Whisper
transcription engine to Lua/ReaScript without blocking REAPER's main thread.

## Lua API

```lua
job_id = reaper.ReaSpeech_Start(audio_path, model, language, translate, vad)
event_json = reaper.ReaSpeech_Poll(job_id)
accepted = reaper.ReaSpeech_Cancel(job_id)
```

`model` is `small`, `medium`, `large-v3`, or `large-v3-turbo`. Use an empty
language string for automatic language detection. `Poll` returns one queued
event at a time and returns `""` when no event is ready. Event types are
`started`, `progress`, `completed`, `cancelled`, and `error`. A completed event
contains the transcript segments:

```json
{"type":"completed","jobId":"reaspeech-1","segments":[{"startMs":0,"endMs":820,"text":"Hello","confidence":0.94}]}
```

See `examples/transcribe.lua` for a complete deferred polling loop. The
`examples/imgui` directory contains a ReaImGui interface for transcribing
selected media items interactively.

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
copy target\release\reaper_reaspeech.dll "%APPDATA%\REAPER\UserPlugins\reaper_reaspeech.dll"
```

On macOS:

```sh
cargo build --release
mkdir -p "$HOME/Library/Application Support/REAPER/UserPlugins"
cp target/release/libreaper_reaspeech.dylib \
  "$HOME/Library/Application Support/REAPER/UserPlugins/reaper_reaspeech.dylib"
```

On Linux:

```sh
cargo build --release
mkdir -p "$HOME/.config/REAPER/UserPlugins"
cp target/release/libreaper_reaspeech.so \
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

Hardware-accelerated builds are available with `cargo build --release
--features cuda` on supported Windows and Linux systems, or `cargo build
--release --features metal` on macOS. Candle's current Metal GEMV kernels
require Apple GPU family 9 or newer (M3 and later); Metal builds automatically
use the CPU on earlier Macs.

## Design

Every request runs on a Rust worker thread. The worker only writes serialized
events to a synchronized queue. Lua reads that queue from REAPER's main thread
with `ReaSpeech_Poll`; no Lua or REAPER function is called from worker threads.
