#!/usr/bin/env bash
# Replace third_party/llama.cpp with the commit in the pin file, or with a
# commit given on the command line (which then rewrites the pin).
#
# The tree is a git submodule so this repository can edit llama.cpp in place.
# This script is the supported way to move the pin; it does not compile.
set -euo pipefail

if [ "${BASH_VERSINFO[0]:-0}" -lt 4 ] \
    || { [ "${BASH_VERSINFO[0]}" -eq 4 ] && [ "${BASH_VERSINFO[1]}" -le 3 ]; }; then
  printf 'kilix-llm: bash 4.4+ required (running %s)\n' "${BASH_VERSION:-unknown}" >&2
  exit 1
fi

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENDOR_DIR="$REPO_DIR/third_party/llama.cpp"
PIN="$REPO_DIR/third_party/llama.cpp.pin"

die() { printf 'kilix-llm: %s\n' "$*" >&2; exit 1; }
log() { printf 'kilix-llm: %s\n' "$*" >&2; }

usage() {
  cat <<'EOF'
usage: vendor-sync.sh [COMMIT]

  COMMIT   40-hex sha to pin (default: the commit= already in the pin file)

Environment:
  KILIX_LLM_LLAMA_MIRROR   existing llama.cpp clone used as a git reference
EOF
}

commit_arg=""
case "${1:-}" in
  '' ) ;;
  -h|--help) usage; exit 0 ;;
  *) commit_arg=$1; shift ;;
esac
[ "$#" -eq 0 ] || { usage >&2; exit 2; }

[ -f "$PIN" ] || die "missing $PIN"
url="$(sed -n 's/^url=//p' "$PIN")"
pin_commit="$(sed -n 's/^commit=//p' "$PIN")"
[ -n "$url" ] || die "pin lacks url="
want=${commit_arg:-$pin_commit}
[ -n "$want" ] || die "no commit to sync"

command -v git >/dev/null 2>&1 || die "git is required"

if [ ! -d "$VENDOR_DIR/.git" ] && [ ! -f "$VENDOR_DIR/.git" ]; then
  log "cloning $url"
  ref_args=()
  if [ -n "${KILIX_LLM_LLAMA_MIRROR:-}" ] && [ -d "$KILIX_LLM_LLAMA_MIRROR/.git" ]; then
    ref_args=(--reference-if-able "$KILIX_LLM_LLAMA_MIRROR" --dissociate)
  fi
  mkdir -p -- "$(dirname "$VENDOR_DIR")"
  git clone "${ref_args[@]}" -- "$url" "$VENDOR_DIR"
fi

cd "$VENDOR_DIR"
git fetch -q origin "$want" 2>/dev/null || git fetch -q origin
git rev-parse --verify -q "$want^{commit}" >/dev/null \
  || die "commit $want not present after fetch"
git checkout -q --detach "$want"
git reset -q --hard "$want"
git clean -qfd
resolved="$(git rev-parse HEAD)"

if [ "$resolved" != "$pin_commit" ]; then
  tmp="$(mktemp)"
  awk -v c="$resolved" '
    BEGIN { done=0 }
    /^commit=/ && done==0 { print "commit=" c; done=1; next }
    { print }
    END { if (done==0) print "commit=" c }
  ' "$PIN" > "$tmp"
  mv -- "$tmp" "$PIN"
  log "updated pin commit= to $resolved"
fi

log "vendor source at $VENDOR_DIR ($resolved)"
