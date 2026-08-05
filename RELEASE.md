# ReaPack release guide

ReaSpeech Lib is published as four mutually exclusive ReaPack extension
packages: CPU, Metal, CUDA 12, and CUDA 13. ReaPack can filter downloads by
operating system and architecture, but it cannot detect a GPU or select a CUDA
runtime. Users therefore choose the backend package explicitly. Each package
uses a backend-specific extension filename so the ReaPack index can manage the
packages independently.

Do not install more than one ReaSpeech Lib backend at a time. All backends
register the same ReaScript API, so loading multiple copies is unsupported.

## Prerequisites

- Check out `reaspeech-lib` and `reascripts` side-by-side.
- Install Python 3.10 or newer. On Windows, the standard `py` launcher can be
  used in place of `python` in the commands below.
- Install and authenticate the GitHub CLI (`gh auth login`).
- Install `reapack-index`.
- Ensure both repositories are clean and up to date.

## Publish a version

1. Update the version in `Cargo.toml`, then commit and push the release source.
2. Run the **Build** workflow for that commit and wait for every job to pass.
3. Find its run ID with `gh run list --repo TeamAudio/reaspeech-lib --workflow Build`.
4. Stage and inspect the release:

       python scripts/publish_reapack.py --run-id RUN_ID

5. Restore the staged files, then publish from a clean checkout:

       python scripts/publish_reapack.py --run-id RUN_ID --release \
         --changelog "Describe the release"

6. Review the content and index commits in `reascripts`, then push that
   repository manually.

The script deliberately does not push either repository. `--release` also
refuses to run over existing tracked changes in `reascripts`. For testing
without downloading, pass `--artifacts-dir PATH`; it must contain the seven
artifact directories produced by the Build workflow.
