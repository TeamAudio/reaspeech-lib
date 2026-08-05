#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/publish-reapack.sh --run-id ID [options]

Download a successful GitHub Actions build and stage ReaSpeech Lib packages in
the TeamAudio reascripts repository.

Options:
  --run-id ID              Build workflow run ID (required)
  --reascripts PATH        ReaPack repository (default: ../reascripts)
  --artifacts-dir PATH     Use already-downloaded artifacts instead of gh
  --changelog TEXT         ReaPack changelog (default: "ReaSpeech Lib VERSION")
  --release                Commit staged files and run reapack-index
  -h, --help               Show this help

Without --release, files are copied but not staged or committed. This is the
recommended first pass so the release can be inspected before publication.
EOF
}

die() { echo "error: $*" >&2; exit 1; }

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
project_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)
reascripts_dir="$project_dir/../reascripts"
run_id=""
artifacts_dir=""
changelog=""
release=false

while (($#)); do
  case "$1" in
    --run-id) run_id=${2:?missing value for --run-id}; shift 2 ;;
    --reascripts) reascripts_dir=${2:?missing value for --reascripts}; shift 2 ;;
    --artifacts-dir) artifacts_dir=${2:?missing value for --artifacts-dir}; shift 2 ;;
    --changelog) changelog=${2:?missing value for --changelog}; shift 2 ;;
    --release) release=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "$run_id" ]] || die "--run-id is required"
[[ -d "$reascripts_dir/.git" ]] || die "not a git repository: $reascripts_dir"
if [[ "$release" == true ]]; then
  [[ -z "$(git -C "$reascripts_dir" status --porcelain)" ]] || \
    die "reascripts contains changes; --release requires a clean checkout"
fi
version=$(sed -nE 's/^version = "([^"]+)"/\1/p' "$project_dir/Cargo.toml" | head -1)
[[ -n "$version" ]] || die "could not read the package version from Cargo.toml"
[[ -n "$changelog" ]] || changelog="ReaSpeech Lib $version"

if [[ -z "$artifacts_dir" ]]; then
  command -v gh >/dev/null || die "GitHub CLI (gh) is required to download workflow artifacts"
  artifacts_dir=$(mktemp -d "${TMPDIR:-/tmp}/reaspeech-reapack.XXXXXX")
  trap 'rm -rf "$artifacts_dir"' EXIT
  run_json=$(gh run view "$run_id" --repo TeamAudio/reaspeech-lib --json conclusion,headSha,workflowName)
  [[ "$run_json" == *'"conclusion":"success"'* ]] || die "workflow run $run_id did not succeed"
  [[ "$run_json" == *'"workflowName":"Build"'* ]] || die "run $run_id is not from the Build workflow"
  run_sha=$(printf '%s' "$run_json" | sed -nE 's/.*"headSha":"([0-9a-f]+)".*/\1/p')
  source_sha=$(git -C "$project_dir" rev-parse HEAD)
  [[ "$run_sha" == "$source_sha" ]] || die "run commit $run_sha does not match source HEAD $source_sha"
  echo "Downloading artifacts from Build run $run_id..."
  gh run download "$run_id" --repo TeamAudio/reaspeech-lib --dir "$artifacts_dir"
fi

artifact_names=(
  reaspeech-windows-x86_64-cpu reaspeech-linux-x86_64-cpu
  reaspeech-macos-arm64-metal
  reaspeech-windows-x86_64-cuda12 reaspeech-linux-x86_64-cuda12
  reaspeech-windows-x86_64-cuda reaspeech-linux-x86_64-cuda
)
for artifact in "${artifact_names[@]}"; do
  [[ -d "$artifacts_dir/$artifact" ]] || die "missing artifact directory: $artifact"
done

output_dir="$reascripts_dir/Extensions/ReaSpeech"
mkdir -p "$output_dir"
install -m 0644 "$artifacts_dir/reaspeech-windows-x86_64-cpu/reaper_reaspeech.dll" "$output_dir/reaper_reaspeech_cpu.dll"
install -m 0644 "$artifacts_dir/reaspeech-linux-x86_64-cpu/reaper_reaspeech.so" "$output_dir/reaper_reaspeech_cpu.so"
install -m 0644 "$artifacts_dir/reaspeech-macos-arm64-metal/reaper_reaspeech.dylib" "$output_dir/reaper_reaspeech_metal.dylib"
install -m 0644 "$artifacts_dir/reaspeech-windows-x86_64-cuda12/reaper_reaspeech.dll" "$output_dir/reaper_reaspeech_cuda12.dll"
install -m 0644 "$artifacts_dir/reaspeech-linux-x86_64-cuda12/reaper_reaspeech.so" "$output_dir/reaper_reaspeech_cuda12.so"
install -m 0644 "$artifacts_dir/reaspeech-windows-x86_64-cuda/reaper_reaspeech.dll" "$output_dir/reaper_reaspeech_cuda13.dll"
install -m 0644 "$artifacts_dir/reaspeech-linux-x86_64-cuda/reaper_reaspeech.so" "$output_dir/reaper_reaspeech_cuda13.so"

write_package() {
  local file=$1 description=$2
  shift 2
  {
    printf '@description %s\n@version %s\n@author Team Audio\n' "$description" "$version"
    printf '@changelog\n  %s\n' "$changelog"
    printf '@provides\n'
    printf '  %s\n' "$@"
  } > "$file"
}

write_package "$reascripts_dir/Extensions/ReaSpeech Lib (CPU).ext" \
  "ReaSpeech Lib (CPU; install only one backend)" \
  '[win64] ReaSpeech/reaper_reaspeech_cpu.dll > reaper_reaspeech_cpu.dll' \
  '[linux64] ReaSpeech/reaper_reaspeech_cpu.so > reaper_reaspeech_cpu.so'
write_package "$reascripts_dir/Extensions/ReaSpeech Lib (Metal).ext" \
  "ReaSpeech Lib (Metal; install only one backend)" \
  '[darwin-arm64] ReaSpeech/reaper_reaspeech_metal.dylib > reaper_reaspeech_metal.dylib'
write_package "$reascripts_dir/Extensions/ReaSpeech Lib (CUDA 12).ext" \
  "ReaSpeech Lib (CUDA 12; install only one backend)" \
  '[win64] ReaSpeech/reaper_reaspeech_cuda12.dll > reaper_reaspeech_cuda12.dll' \
  '[linux64] ReaSpeech/reaper_reaspeech_cuda12.so > reaper_reaspeech_cuda12.so'
write_package "$reascripts_dir/Extensions/ReaSpeech Lib (CUDA 13).ext" \
  "ReaSpeech Lib (CUDA 13; install only one backend)" \
  '[win64] ReaSpeech/reaper_reaspeech_cuda13.dll > reaper_reaspeech_cuda13.dll' \
  '[linux64] ReaSpeech/reaper_reaspeech_cuda13.so > reaper_reaspeech_cuda13.so'

echo "Staged ReaSpeech Lib $version in $reascripts_dir/Extensions"
if [[ "$release" == true ]]; then
  git -C "$reascripts_dir" add Extensions
  git -C "$reascripts_dir" commit -m "Update ReaSpeech Lib to $version"
  reapack-index --commit "$reascripts_dir"
  echo "Release committed and indexed. Review the commits, then push reascripts manually."
else
  echo "Review the files, then clean/restore reascripts and rerun with --release."
fi
