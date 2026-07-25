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

See `examples/transcribe.lua` for a complete deferred polling loop.

## Build and install

Use the MSVC Rust target on Windows, then run:

```powershell
cargo build --release
Copy-Item target\release\reaper_reaspeech.dll "$env:APPDATA\REAPER\UserPlugins\"
```

Restart REAPER after copying the extension. Models are downloaded on first use
to `<REAPER resource path>/ReaSpeech/models`. Set `MODELS_PATH` before starting
REAPER to override that location.

CUDA and Metal builds are available with `--features cuda` and
`--features metal`, respectively.

## Design

Every request runs on a Rust worker thread. The worker only writes serialized
events to a synchronized queue. Lua reads that queue from REAPER's main thread
with `ReaSpeech_Poll`; no Lua or REAPER function is called from worker threads.

