#!/usr/bin/env bash
# Download one Hugging Face checkpoint into the cache used by HF_HOME.
#
# Uses the uv virtual environment from lab/scripts/uv-venv.sh. The download
# itself is not timed and is not a benchmark. Pass the model id as the only
# argument. The default is the 4B checkpoint that holds the transcript neurons.
set -euo pipefail

if [ "${BASH_VERSINFO[0]:-0}" -lt 4 ] \
    || { [ "${BASH_VERSINFO[0]}" -eq 4 ] && [ "${BASH_VERSINFO[1]}" -le 3 ]; }; then
  printf 'kilix-llm lab: bash 4.4+ required (running %s)\n' "${BASH_VERSION:-unknown}" >&2
  exit 1
fi

if [ "$#" -gt 1 ]; then
  printf 'kilix-llm lab: usage: fetch-model.sh [model-id]\n' >&2
  exit 2
fi

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="${KILIX_LLM_LAB_VENV:-$REPO_DIR/lab/.venv}"
MODEL_ID="${1:-Qwen/Qwen3.5-4B}"

"$REPO_DIR/lab/scripts/uv-venv.sh"
export PYTHONPATH="$REPO_DIR/lab/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$VENV/bin/python" -m kilix_llm_lab.fetch "$MODEL_ID"
