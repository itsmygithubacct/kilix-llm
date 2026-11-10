#!/usr/bin/env bash
# Build the vendored llama.cpp with the CPU or Vulkan backend.
#
# Patches listed in third_party/llama.cpp.pin are applied to a copy under
# build/ so the submodule itself stays a clean upstream pin.
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

  --check            report whether this backend is built for the current pin;
                     build nothing
  --if-needed        build only when this backend is not built for the current
                     pin (what first use runs)
  --keep-build-tree  keep the cmake tree and the patched source copy for fast
                     incremental rebuilds; by default only bin/ is kept, which
                     is all the runtime runs

Environment:
  KILIX_LLM_BUILD_DIR    cmake build directory
                         (default: <checkout>/build/llama.cpp[-vulkan])
  KILIX_LLM_BUILD_JOBS   parallel compile jobs (default: nproc)
EOF
}

check_only=0
if_needed=0
keep_tree=0
backend=cpu
while [ "$#" -gt 0 ]; do
  case "$1" in
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
BUILD_ROOT="${KILIX_LLM_BUILD_DIR:-$REPO_DIR/build/llama.cpp$suffix}"
SRC_COPY="$REPO_DIR/build/llama.cpp$suffix-src"

# A build is current when bin/ holds every runtime binary and its stamp names
# this backend, these flags and the exact pin file (commit and patch sums).
# The stamp is written last, so an interrupted build is never current.
STAMP="$BUILD_ROOT/bin/.kilix-llm-build"
identity() {
  printf 'backend=%s\n' "$backend"
  printf 'flags=%s\n' "${backend_flags[*]}"
  printf 'pin-sha256=%s\n' "$(sha256sum < "$PIN" | cut -d' ' -f1)"
}
build_current() {
  local name
  [ -f "$PIN" ] && [ -f "$STAMP" ] || return 1
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
mkdir -p -- "$REPO_DIR/build"
exec 9>"$REPO_DIR/build/.build-$backend.lock"
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
[ -f "$PIN" ] || die "missing $PIN"

# cmake and the compiler must not inherit a full tmpfs; this machine's /tmp
# is a RAM disk. Keep throwaway files next to the build.
mkdir -p -- "$REPO_DIR/build/tmp"
export TMPDIR="$REPO_DIR/build/tmp"

mkdir -p -- "$(dirname "$SRC_COPY")"
command -v rsync >/dev/null 2>&1 || die "rsync is required to stage the vendor tree"
rsync -a --delete --exclude '.git' --exclude 'build' -- "$VENDOR_DIR"/ "$SRC_COPY"/

applied=0
while read -r patch sum; do
  case "$patch" in ''|'#'*|url=*|commit=*) continue ;; esac
  file="$PATCH_DIR/$patch"
  [ -f "$file" ] || die "pin names missing patch $patch"
  echo "$sum  $file" | sha256sum -c --quiet - \
    || die "patch $patch does not match the sha256 in the pin"
  git -C "$SRC_COPY" apply -- "$file" || die "patch $patch failed to apply"
  applied=$((applied + 1))
done < "$PIN"

jobs="${KILIX_LLM_BUILD_JOBS:-$(nproc)}"
rm -f -- "$STAMP"
cmake -S "$SRC_COPY" -B "$BUILD_ROOT" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_NATIVE=ON \
    "${backend_flags[@]}" \
    -DGGML_METAL=OFF \
    -DLLAMA_CURL=OFF \
    >/dev/null
cmake --build "$BUILD_ROOT" -j "$jobs" --target llama-cli llama-server llama-bench
for name in llama-cli llama-server llama-bench; do
  [ -x "$BUILD_ROOT/bin/$name" ] || die "the build did not produce $BUILD_ROOT/bin/$name"
done
identity > "$STAMP.tmp" && mv -f -- "$STAMP.tmp" "$STAMP"

# The binaries and their shared libraries live in bin/. The object files, the
# cmake tree and the patched source copy are about 0.7 GB the runtime never
# reads. A KILIX_LLM_BUILD_DIR the caller chose is never pruned.
if [ "$keep_tree" = 0 ]; then
  rm -rf -- "$SRC_COPY" "$REPO_DIR/build/tmp"
  if [ -z "${KILIX_LLM_BUILD_DIR:-}" ]; then
    find "$BUILD_ROOT" -mindepth 1 -maxdepth 1 ! -name bin -exec rm -rf -- {} +
  fi
fi
log "built: $BUILD_ROOT/bin ($backend, $applied patch(es))"
