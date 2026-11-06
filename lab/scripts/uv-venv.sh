#!/usr/bin/env bash
# Create the lab virtual environment with uv and install torch, numpy, and transformers.
#
# The launcher does not use this environment. Model commands and the bench do.
# Re-running is safe: uv keeps an existing environment and skips packages that
# already match.
set -euo pipefail

if [ "${BASH_VERSINFO[0]:-0}" -lt 4 ] \
    || { [ "${BASH_VERSINFO[0]}" -eq 4 ] && [ "${BASH_VERSINFO[1]}" -le 3 ]; }; then
  printf 'kilix-llm lab: bash 4.4+ required (running %s)\n' "${BASH_VERSION:-unknown}" >&2
  exit 1
fi

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="${KILIX_LLM_LAB_VENV:-$REPO_DIR/lab/.venv}"
# CPU wheels by default. Point this at a CUDA index when the machine has a GPU.
TORCH_INDEX="${KILIX_LLM_TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"

if ! command -v uv >/dev/null 2>&1; then
  printf 'kilix-llm lab: uv is not on PATH\n' >&2
  exit 1
fi

if [ ! -x "$VENV/bin/python" ]; then
  printf 'kilix-llm lab: uv venv %s\n' "$VENV" >&2
  uv venv --python 3.12 "$VENV"
else
  printf 'kilix-llm lab: reusing %s\n' "$VENV" >&2
fi
uv pip install --python "$VENV/bin/python" torch --index-url "$TORCH_INDEX"
uv pip install --python "$VENV/bin/python" numpy transformers
printf 'kilix-llm lab: ready %s\n' "$VENV/bin/python" >&2
