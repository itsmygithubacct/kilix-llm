#!/usr/bin/env bash
# Build the vendored llama.cpp with the CPU or Vulkan backend.
#
# Patches listed in third_party/llama.cpp.pin are applied to a copy under
# build/ so the submodule itself stays a clean upstream pin.
#
# `kilix-llm update` builds a newer upstream release the same way: it passes
# --source (a checkout of that release), --commit and --build-dir, and the
# pin's patches are not applied to it.
set -euo pipefail

if [ "${BASH_VERSINFO[0]:-0}" -lt 4 ] \
    || { [ "${BASH_VERSINFO[0]}" -eq 4 ] && [ "${BASH_VERSINFO[1]}" -le 3 ]; }; then
  printf 'kilix-llm: bash 4.4+ required (running %s)\n' "${BASH_VERSION:-unknown}" >&2
  exit 1
fi

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENDOR_DIR="$REPO_DIR/third_party/llama.cpp"
PIN="$REPO_DIR/third_party/llama.cpp.pin"
PATCH_DIR="$REPO_DIR/patches"

die() { printf 'kilix-llm: %s\n' "$*" >&2; exit 1; }
log() { printf 'kilix-llm: %s\n' "$*" >&2; }

usage() {
  cat <<'EOF'
usage: build-runtime.sh [--backend cpu|vulkan] [--check] [--if-needed] [--keep-build-tree]
                        [--source DIR --commit SHA --build-dir DIR]

  --check            report whether this backend is built for the current pin;
                     build nothing
  --if-needed        build only when this backend is not built for the current
                     pin (what first use runs)
  --keep-build-tree  keep the cmake tree and the patched source copy for fast
                     incremental rebuilds; by default only bin/ is kept, which
                     is all the runtime runs
  --source DIR       build this llama.cpp checkout instead of the pin
                     (needs --commit, its exact commit, and --build-dir)
  --build-dir DIR    where bin/ goes for --source builds; pruned like the
                     default build directory

Environment:
  KILIX_LLM_BUILD_DIR    cmake build directory
                         (default: <checkout>/build/llama.cpp[-vulkan])
  KILIX_LLM_BUILD_JOBS   parallel compile jobs (default: nproc)
  KILIX_LLM_BLAS         auto (default) builds the CPU backend with OpenBLAS
                         when pkg-config finds it; off never does
EOF
}

check_only=0
if_needed=0
keep_tree=0
backend=cpu
source_dir=''
source_commit=''
build_dir_arg=''
while [ "$#" -gt 0 ]; do
  case "$1" in
    --source) [ "$#" -ge 2 ] || die '--source needs a directory'; source_dir=$2; shift 2 ;;
    --commit) [ "$#" -ge 2 ] || die '--commit needs a commit'; source_commit=$2; shift 2 ;;
    --build-dir) [ "$#" -ge 2 ] || die '--build-dir needs a directory'; build_dir_arg=$2; shift 2 ;;
    --check) check_only=1; shift ;;
    --if-needed) if_needed=1; shift ;;
    --keep-build-tree) keep_tree=1; shift ;;
    --backend) [ "$#" -ge 2 ] || die '--backend needs cpu or vulkan'; backend=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done
backend_flags=(-DGGML_CUDA=OFF -DGGML_VULKAN=OFF)
suffix=''
case "$backend" in
  cpu) ;;
  vulkan) suffix=-vulkan; backend_flags=(-DGGML_CUDA=OFF -DGGML_VULKAN=ON) ;;
  *) die "unknown backend $backend (choose cpu or vulkan)" ;;
esac
# OpenBLAS speeds up prompt processing on the CPU backend. It is part of the
# build identity, so installing or removing it rebuilds once.
case "${KILIX_LLM_BLAS:-auto}" in
  auto|off) ;;
  *) die "KILIX_LLM_BLAS must be auto or off" ;;
esac
if [ "$backend" = cpu ] && [ "${KILIX_LLM_BLAS:-auto}" = auto ] \
    && command -v pkg-config >/dev/null 2>&1 && pkg-config --exists openblas; then
  backend_flags+=(-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS)
fi
BUILD_ROOT="${KILIX_LLM_BUILD_DIR:-$REPO_DIR/build/llama.cpp$suffix}"
SRC_COPY="$REPO_DIR/build/llama.cpp$suffix-src"
WORK_DIR="$REPO_DIR/build"
prune_root=1
[ -z "${KILIX_LLM_BUILD_DIR:-}" ] || prune_root=0
if [ -n "$source_dir$source_commit$build_dir_arg" ]; then
  [ -n "$source_dir" ] && [ -n "$build_dir_arg" ] \
    || die '--source, --commit and --build-dir go together'
  [[ "$source_commit" =~ ^[0-9a-f]{40}$ ]] || die '--commit needs a full 40-character commit'
  VENDOR_DIR="$source_dir"
  BUILD_ROOT="$build_dir_arg"
  SRC_COPY="$source_dir"
  WORK_DIR="$(dirname -- "$BUILD_ROOT")"
  prune_root=1
fi

# A build is current when bin/ holds every runtime binary and its stamp names
# this backend, these flags and the exact pin file (commit and patch sums).
# The stamp is written last, so an interrupted build is never current.
STAMP="$BUILD_ROOT/bin/.kilix-llm-build"
identity() {
  printf 'backend=%s\n' "$backend"
  printf 'flags=%s\n' "${backend_flags[*]}"
  if [ -n "$source_commit" ]; then
    printf 'source-commit=%s\n' "$source_commit"
  else
    printf 'pin-sha256=%s\n' "$(sha256sum < "$PIN" | cut -d' ' -f1)"
  fi
}
build_current() {
  local name
  { [ -n "$source_commit" ] || [ -f "$PIN" ]; } && [ -f "$STAMP" ] || return 1
  [ "$(cat -- "$STAMP")" = "$(identity)" ] || return 1
  for name in llama-cli llama-server llama-bench; do
    [ -x "$BUILD_ROOT/bin/$name" ] || return 1
  done
}

if [ "$check_only" = 1 ]; then
  if build_current; then
    log "built: $BUILD_ROOT/bin"
    exit 0
  fi
  log "not built for this pin (run kilix-llm build --backend $backend)"
  exit 1
fi

# One build per backend at a time: a second first use waits, then finds the
# first one's result current.
mkdir -p -- "$WORK_DIR"
exec 9>"$WORK_DIR/.build-$backend.lock"
command -v flock >/dev/null 2>&1 || die "flock is required (Debian: util-linux)"
flock 9 || die "could not lock the $backend build"
if [ "$if_needed" = 1 ] && build_current; then
  exit 0
fi

command -v cmake >/dev/null 2>&1 \
  || die "cmake is required — on Debian:  sudo apt-get install cmake"
command -v c++ >/dev/null 2>&1 || command -v g++ >/dev/null 2>&1 \
  || command -v clang++ >/dev/null 2>&1 \
  || die "no C++ compiler — on Debian:  sudo apt-get install build-essential"
[ -f "$VENDOR_DIR/CMakeLists.txt" ] \
  || die "no vendored llama.cpp at $VENDOR_DIR — run scripts/vendor-sync.sh"
[ -n "$source_commit" ] || [ -f "$PIN" ] || die "missing $PIN"
if [ -n "$source_commit" ] && [ -d "$VENDOR_DIR/.git" ]; then
  [ "$(git -C "$VENDOR_DIR" rev-parse HEAD)" = "$source_commit" ] \
    || die "$VENDOR_DIR is not at $source_commit"
fi

# cmake and the compiler must not inherit a full tmpfs; this machine's /tmp
# is a RAM disk. Keep throwaway files next to the build.
mkdir -p -- "$WORK_DIR/tmp"
export TMPDIR="$WORK_DIR/tmp"

applied=0
if [ -z "$source_commit" ]; then
mkdir -p -- "$(dirname "$SRC_COPY")"
command -v rsync >/dev/null 2>&1 || die "rsync is required to stage the vendor tree"
rsync -a --delete --exclude '.git' --exclude 'build' -- "$VENDOR_DIR"/ "$SRC_COPY"/

while read -r patch sum; do
  case "$patch" in ''|'#'*|url=*|commit=*) continue ;; esac
  file="$PATCH_DIR/$patch"
  [ -f "$file" ] || die "pin names missing patch $patch"
  echo "$sum  $file" | sha256sum -c --quiet - \
    || die "patch $patch does not match the sha256 in the pin"
  git -C "$SRC_COPY" apply -- "$file" || die "patch $patch failed to apply"
  applied=$((applied + 1))
done < "$PIN"
fi

jobs="${KILIX_LLM_BUILD_JOBS:-$(nproc)}"
rm -f -- "$STAMP"
cmake -S "$SRC_COPY" -B "$BUILD_ROOT" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_NATIVE=ON \
    "${backend_flags[@]}" \
    -DGGML_METAL=OFF \
    -DLLAMA_CURL=OFF \
    -DLLAMA_USE_PREBUILT_UI=OFF \
    >/dev/null
cmake --build "$BUILD_ROOT" -j "$jobs" --target llama-cli llama-server llama-bench
for name in llama-cli llama-server llama-bench; do
  [ -x "$BUILD_ROOT/bin/$name" ] || die "the build did not produce $BUILD_ROOT/bin/$name"
done
identity > "$STAMP.tmp" && mv -f -- "$STAMP.tmp" "$STAMP"

# The binaries and their shared libraries live in bin/. The object files, the
# cmake tree and the patched source copy are about 0.7 GB the runtime never
# reads. A KILIX_LLM_BUILD_DIR the caller chose is never pruned.
# A --source checkout belongs to the caller, which removes it.
if [ "$keep_tree" = 0 ]; then
  [ -n "$source_commit" ] || rm -rf -- "$SRC_COPY"
  rm -rf -- "$WORK_DIR/tmp"
  if [ "$prune_root" = 1 ]; then
    find "$BUILD_ROOT" -mindepth 1 -maxdepth 1 ! -name bin -exec rm -rf -- {} +
  fi
fi
log "built: $BUILD_ROOT/bin ($backend, $applied patch(es))"
