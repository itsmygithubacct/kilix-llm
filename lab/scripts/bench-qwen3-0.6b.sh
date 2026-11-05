#!/usr/bin/env bash
# Compare greedy answers before and after removing the sad-minus-happy
# direction from one layer's writing weights. Load time is not a result.
#
# Uses the uv virtual environment from lab/scripts/uv-venv.sh. Extra arguments
# are passed to lab/projects/qwen3_0_6b/bench.py.
#
# The checkpoint download follows the usual Hugging Face cache. Set HF_HOME
# before this script to put that cache on a large disk. Nothing is written
# back into the checkpoint files.
set -euo pipefail

if [ "${BASH_VERSINFO[0]:-0}" -lt 4 ] \
    || { [ "${BASH_VERSINFO[0]}" -eq 4 ] && [ "${BASH_VERSINFO[1]}" -le 3 ]; }; then
  printf 'kilix-llm lab: bash 4.4+ required (running %s)\n' "${BASH_VERSION:-unknown}" >&2
  exit 1
fi

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="${KILIX_LLM_LAB_VENV:-$REPO_DIR/lab/.venv}"

"$REPO_DIR/lab/scripts/uv-venv.sh"
export PYTHONPATH="$REPO_DIR/lab/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$VENV/bin/python" "$REPO_DIR/lab/projects/qwen3_0_6b/bench.py" "$@"
