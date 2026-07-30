# ReaSpeech Lib

ReaSpeech Lib is a REAPER extension written in Rust. It exposes a Whisper
transcription engine to Lua/ReaScript without blocking REAPER's main thread.

## Lua API

```lua
job_id = reaper.ReaSpeech_Start(audio_path, model, language, translate, vad, words, hotwords)
event_json = reaper.ReaSpeech_Poll(job_id)
accepted = reaper.ReaSpeech_Cancel(job_id)
```

Only `audio_path` and `model` are required. `language` defaults to automatic
language detection, while `translate`, `vad`, and `words` default to `false`.
`hotwords` is an optional string of names, terminology, or phrases that should
be favored during recognition. For example:

```lua
local hotwords = "ReaSpeech, REAPER, Cockos, spectral edit"
local job_id = reaper.ReaSpeech_Start(path, "small", "en", false, true, false, hotwords)
```

Hotwords are hints rather than a restricted vocabulary or guaranteed output.
Separate multiple entries with commas or newlines.

`model` is `small`, `medium`, `large-v3`, or `large-v3-turbo`.
`large-v3-turbo` is transcription-only; select `small`, `medium`, or `large-v3`
when `translate` is enabled. `Poll` returns one queued
event at a time and returns `""` when no event is ready. Event types are
`started`, `progress`, `segment`, `completed`, `cancelled`, and `error`. Each
recognized segment is emitted immediately:

```json
{"type":"segment","jobId":"reaspeech-1","segment":{"startMs":0,"endMs":820,"text":"Hello","confidence":0.94}}
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

Hardware-accelerated builds are available with
`cargo build --release --features cuda` on supported Windows and Linux systems, or
`cargo build --release --features metal` on macOS. The project carries a small
patch to Candle's Metal kernels that omits unused BF16 GEMV variants on older
Metal language targets, allowing the F32 Whisper implementation to use Metal
on M1 and later Macs.

## Continuous integration builds

The `Build` GitHub Actions workflow builds and uploads ready-to-install
extensions for:

- Windows x86_64 and Linux x86_64 using the CPU backend
- macOS arm64 using Metal
- Windows x86_64 and Linux x86_64 using CUDA 12 or CUDA 13

CUDA 13 is the default CUDA build. Its artifact names end in `-cuda`; CUDA 12
artifacts end in `-cuda12` and use CUDA Toolkit 12.5 for compatibility with
older NVIDIA drivers. Every artifact contains the extension under the filename
REAPER expects in `UserPlugins`, ready for future publication through the
TeamAudio ReaPack repository.

An Intel macOS build is not currently produced because the pinned ONNX Runtime
release does not provide prebuilt `x86_64-apple-darwin` binaries.

## Design

Every request runs on a Rust worker thread. The worker only writes serialized
events to a synchronized queue. Lua reads that queue from REAPER's main thread
with `ReaSpeech_Poll`; no Lua or REAPER function is called from worker threads.
