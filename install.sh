#!/usr/bin/env bash
# Put kilix-llm on PATH as a launcher that runs the tool from this checkout.
set -euo pipefail

if [ "${BASH_VERSINFO[0]:-0}" -lt 4 ] \
    || { [ "${BASH_VERSINFO[0]}" -eq 4 ] && [ "${BASH_VERSINFO[1]}" -le 3 ]; }; then
  printf 'kilix-llm: bash 4.4+ required (running %s)\n' "${BASH_VERSION:-unknown}" >&2
  exit 1
fi

PREFIX="${KILIX_LLM_PREFIX:-$HOME/.local}"
BIN_DIR="$PREFIX/bin"
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

shell_quote() {
    printf "'"
    printf '%s' "$1" | sed "s/'/'\\\\''/g"
    printf "'"
}

launcher_text() {
    quoted_target="$(shell_quote "$REPO_DIR/kilix-llm")"
    printf '%s\n' '#!/bin/sh'
    printf 'exec %s "$@"\n' "$quoted_target"
}

usage() {
    cat <<'EOF'
usage: install.sh [--uninstall]

  --uninstall  remove the launcher generated for this checkout; retain it if
               its contents were changed after installation
EOF
}

uninstall=0
case "${1:-}" in
  '') ;;
  --uninstall) uninstall=1; shift ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac
[ "$#" -eq 0 ] || { usage >&2; exit 2; }

launcher="$BIN_DIR/kilix-llm"
if [ "$uninstall" = 1 ]; then
    if [ ! -e "$launcher" ]; then
        printf 'kilix-llm: no launcher installed at %s\n' "$launcher" >&2
        exit 0
    fi
    expected="$(launcher_text)"
    if [ ! -f "$launcher" ] || [ "$(cat -- "$launcher")" != "$expected" ]; then
        printf 'kilix-llm: modified launcher retained: %s\n' "$launcher" >&2
        exit 1
    fi
    rm -f -- "$launcher"
    printf 'kilix-llm: removed %s\n' "$launcher" >&2
    exit 0
fi

mkdir -p -- "$BIN_DIR"
launcher_text > "$launcher"
chmod 755 "$launcher"
printf 'kilix-llm: installed %s\n' "$launcher" >&2

case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *) printf 'kilix-llm: note: %s is not on PATH\n' "$BIN_DIR" >&2 ;;
esac
