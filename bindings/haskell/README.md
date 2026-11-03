# Haskell bindings and CLI

This Cabal package wraps ReaSpeech's C ABI without REAPER. It provides typed
options and events, file and in-memory PCM input, polling, and cancellation.
The `reaspeech-transcribe` example streams recognized segments to stdout,
with status messages on stderr. Native job failures produce a nonzero exit code.

## Prerequisites and platform support

Build the Rust library and Haskell executable for the **same OS and CPU
architecture**. On Apple Silicon, use arm64 Rust and GHC rather than mixing
native and Rosetta tools. These instructions use the shared library; keep it
available when running the executable, including when using `cabal run`.

- **macOS (Apple Silicon):** native build instructions below. Intel Macs need
  a separately configured ONNX Runtime: the pinned version does not ship an
  Intel macOS binary.
- **Windows:** use WSL2 with Ubuntu and the Linux instructions below. This
  builds a Linux executable, not a Windows `.exe`. Native Windows GHC/MSVC
  linking is not yet validated by this package.
- **Linux:** use the same Ubuntu instructions without the WSL-specific paths.

Install Rust with [rustup](https://rustup.rs/). GHC and Cabal are also required;
GHC 9.6 or newer is recommended. Run all build commands from the repository
root unless instructed otherwise.

## macOS

Install Apple's command-line developer tools (`xcode-select --install`) if
needed, then install GHC and Cabal (for example, with Homebrew):

```sh
brew install ghc cabal-install
cargo build --release --locked --no-default-features
export REASPEECH_LIB_DIR="$PWD/target/release"
export DYLD_LIBRARY_PATH="$REASPEECH_LIB_DIR${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
export MODELS_PATH="$HOME/Library/Caches/ReaSpeech/models"

cd bindings/haskell
cabal update
cabal configure --extra-lib-dirs="$REASPEECH_LIB_DIR"
cabal build all
cabal test --test-show-details=direct
cabal run -v0 reaspeech-transcribe -- --help
cabal run -v0 reaspeech-transcribe -- /path/to/speech.wav --model small
```

For GPU acceleration on Apple Silicon, rebuild the Rust library from the
repository root with `cargo build --release --locked --no-default-features
--features metal`. Metal builds require the full Xcode installation with its
Metal compiler, selected as the active developer directory.

## Ubuntu Linux or Windows with WSL2

On Windows, run these commands **inside Ubuntu in WSL**, with Rust installed
inside WSL. Windows DLLs cannot be linked into a Linux Haskell executable.
Install GHC and Cabal through your distribution or GHCup. For Ubuntu packages:

```sh
sudo apt-get update
sudo apt-get install ghc cabal-install libghc-aeson-dev \
  libghc-optparse-applicative-dev build-essential pkg-config libssl-dev
```

From the repository root (for example, `cd /mnt/d/Source/reaspeech-lib` in
WSL for a checkout at `D:\Source\reaspeech-lib`):

```sh
# Separate Linux artifacts from any native Windows Cargo output.
export CARGO_TARGET_DIR="$PWD/target/linux"
cargo build --release --locked --no-default-features
export REASPEECH_LIB_DIR="$CARGO_TARGET_DIR/release"
export LD_LIBRARY_PATH="$REASPEECH_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MODELS_PATH="$HOME/.cache/reaspeech/models"

cd bindings/haskell
cabal update
cabal configure --extra-lib-dirs="$REASPEECH_LIB_DIR"
cabal build all
cabal test --test-show-details=direct
cabal run -v0 reaspeech-transcribe -- --help
cabal run -v0 reaspeech-transcribe -- /path/to/speech.wav --model small --vad
```

In WSL, a Windows audio path such as `D:\audio\speech.wav` becomes
`/mnt/d/audio/speech.wav`. A checkout in WSL's Linux filesystem may build
faster than one under `/mnt/d`. CUDA builds can use
`cargo build --release --locked --no-default-features --features cuda` with
the appropriate Linux CUDA toolchain, including in WSL.

If using Ubuntu's GHC and dependency packages above, `--offline` can be added
to Cabal build, test, and run commands after the initial index update. With
GHCup or another GHC installation, let Cabal fetch its dependencies normally.

## Running and troubleshooting

Repeat the environment exports in each new shell. `cabal configure` saves the
link-time library directory in the ignored `cabal.project.local`; reconfigure
if you move the checkout or change the Rust output directory. At runtime,
macOS uses `DYLD_LIBRARY_PATH` and Linux/WSL uses `LD_LIBRARY_PATH`. A missing
`libreaspeech.dylib`/`libreaspeech.so` error means the shared library cannot be
found; verify the export points to the directory containing that file. This
is a development setup; copying only the executable to another machine is
not sufficient to distribute the application.

`cabal test` exercises the real C ABI, including UTF-8 paths, PCM validation,
and asynchronous worker events, without downloading a model. The first real
transcription downloads Whisper model assets. `MODELS_PATH` selects the cache
location (the default is `models` relative to the current directory).

The following examples work in macOS, Linux, and WSL shells:

```sh
cabal run -v0 reaspeech-transcribe -- speech.wav -m small -l en --no-timestamps
cabal run -v0 reaspeech-transcribe -- speech.wav --json > events.jsonl
cabal run -v0 reaspeech-transcribe -- speech.wav \
  --model medium --translate --beam-size 3 --start-ms 15000 --end-ms 45000
```

Available models are `small`, `medium`, `large-v3`, and `large-v3-turbo`.
Turbo does not support translation. `--words` adds word timestamps to segment
JSON; `--hotwords 'ReaSpeech, REAPER'` supplies recognition hints.
`--json` emits all original events as newline-delimited JSON, including terminal
events. `-v0` suppresses Cabal's build messages so redirected stdout contains
only the program's output. Redirect stdout to save a transcript or JSON stream.
Ctrl+C requests cancellation during cleanup; cancellation is asynchronous.

## Validation

The CPU shared library, bindings tests, CLI help, and a short speech
transcription have been tested on Apple Silicon macOS with GHC 9.14.1. The
`Haskell bindings` CI workflow is configured to build and test macOS arm64
and Ubuntu x86_64 with GHC 9.6.7; it does not download models or run full
inference. Native
Windows and Intel macOS Haskell builds remain unverified.

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
