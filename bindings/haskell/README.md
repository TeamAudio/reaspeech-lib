# Haskell bindings and CLI

This Cabal package wraps ReaSpeech's C ABI without REAPER. It provides typed
options and events, file and in-memory PCM input, polling, and cancellation.
The `reaspeech-transcribe` example streams recognized segments to stdout,
with status messages on stderr. Native job failures produce a nonzero exit code.

## Build in WSL (Ubuntu)

Run these commands **inside WSL**, using the Linux Rust toolchain. Windows
DLLs cannot be linked into a Linux Haskell executable. GHC and Cabal can be
installed from Ubuntu's packages:

```sh
sudo apt-get update
sudo apt-get install ghc cabal-install libghc-aeson-dev \
  libghc-optparse-applicative-dev build-essential pkg-config libssl-dev
```

With `cargo` already installed inside WSL, build from the repository root
(`D:\Source\reaspeech-lib` is `/mnt/d/Source/reaspeech-lib`):

```sh
cd /mnt/d/Source/reaspeech-lib
rustup update stable
# Keep Linux artifacts separate from native Windows Cargo output.
export CARGO_TARGET_DIR="$PWD/target/wsl"
cargo +stable build --release --no-default-features
export LD_LIBRARY_PATH="$CARGO_TARGET_DIR/release${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MODELS_PATH="$HOME/.cache/reaspeech/models"

cd bindings/haskell
cabal update
cabal configure --extra-lib-dirs="$CARGO_TARGET_DIR/release"
cabal build all --offline
cabal test --offline
cabal run -v0 --offline reaspeech-transcribe -- --help
cabal run -v0 --offline reaspeech-transcribe -- /mnt/d/audio/speech.wav --model small --vad
```

Keep `LD_LIBRARY_PATH` set when running the executable. Repeat the exports in
each new shell. `cabal configure` saves the library directory in the ignored
`cabal.project.local`; reconfigure it if you move the checkout or build output.
After the initial index update, the Ubuntu Haskell dependency packages above
allow an offline Cabal build;
if you use a separate GHC installation, run `cabal update` and omit `--offline`
to resolve dependencies from Hackage instead.

The first transcription downloads Whisper model assets. `MODELS_PATH` selects
the cache location (the default is `models` relative to the current directory):

```sh
export MODELS_PATH="$HOME/.cache/reaspeech/models"
cabal run -v0 --offline reaspeech-transcribe -- speech.wav -m small -l en --no-timestamps
cabal run -v0 --offline reaspeech-transcribe -- speech.wav --json > events.jsonl
cabal run -v0 --offline reaspeech-transcribe -- speech.wav \
  --model medium --translate --beam-size 3 --start-ms 15000 --end-ms 45000
```

Available models are `small`, `medium`, `large-v3`, and `large-v3-turbo`.
Turbo does not support translation. `--words` adds word timestamps to segment
JSON; `--hotwords 'ReaSpeech, REAPER'` supplies recognition hints.
`--json` emits all original events as newline-delimited JSON, including terminal
events. `-v0` suppresses Cabal's build messages so redirected stdout contains
only the program's output. Redirect stdout to save a transcript or JSON stream. Ctrl+C requests
cancellation during cleanup; cancellation is asynchronous.

For faster filesystem access, you may clone the repository into WSL's Linux
filesystem instead and adjust the `cd` command. CUDA builds can use
`cargo build --release --no-default-features --features cuda` with the
appropriate Linux CUDA toolchain available in WSL.

## Library example

Add `reaspeech` to your executable's `build-depends` and include this package
in your `cabal.project`:

```haskell
{-# LANGUAGE OverloadedStrings #-}
import Control.Concurrent (threadDelay)
import ReaSpeech

main :: IO ()
main = do
  result <- start "speech.wav" defaultJobOptions { vad = True }
  job <- either (fail . show) pure result
  let loop = do
        event <- poll job
        case event of
          Nothing -> threadDelay 20000 >> loop
          Just e -> do
            print e
            if isTerminal e then pure () else loop
  loop
```

`startFromMemory samples sampleRate channels options` accepts interleaved
`[Float]` PCM; the native library validates and copies it before returning.
Segment timestamps use milliseconds; word timestamps use seconds.
`pollJSON` returns the original UTF-8 bytes, while `poll` decodes a typed event.
Both remove one event and return `Nothing` for an empty queue. Keep polling
until `Completed`, `Cancelled`, or `Error`. Decode errors from `poll` raise an
IO exception. File paths containing NUL are rejected rather than silently truncated.

Polling across Haskell threads/jobs is serialized through one shared lock,
including copying the native return buffer. If other code in the same process
calls `reaspeech_poll` directly, it must coordinate with the Haskell binding;
the C API's return buffer is shared process-wide.

The package is intended to link the Linux shared library in WSL; native Windows
linking is not covered by these instructions.
