#!/usr/bin/env bash
# Distribute a JSON benchmark suite across remembered machines.
set -euo pipefail

if [ "${BASH_VERSINFO[0]:-0}" -lt 4 ] \
    || { [ "${BASH_VERSINFO[0]}" -eq 4 ] && [ "${BASH_VERSINFO[1]}" -le 3 ]; }; then
  printf 'kilix-llm lab: bash 4.4+ required (running %s)\n' "${BASH_VERSION:-unknown}" >&2
  exit 1
fi

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$REPO_DIR/lab/src${PYTHONPATH:+:$PYTHONPATH}"
exec python3 -m kilix_llm_lab.benchqueue "$@"
