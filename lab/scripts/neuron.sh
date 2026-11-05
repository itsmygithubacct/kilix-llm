#!/usr/bin/env bash
# Increase, decrease, activate, or deactivate one transcript neuron, then
# generate. Extra arguments are passed to `run.py neuron`.
#
# The named neurons live on Qwen/Qwen3.5-4B. Fetch that checkpoint first
# with lab/scripts/fetch-model.sh. bf16 does not fit an 8 GB GPU.
# --quant 4bit --device cuda keeps the checkpoint on that GPU.
# --device cpu --dtype bfloat16 remains the fit when no GPU is used.
set -euo pipefail

if [ "${BASH_VERSINFO[0]:-0}" -lt 4 ] \
    || { [ "${BASH_VERSINFO[0]}" -eq 4 ] && [ "${BASH_VERSINFO[1]}" -le 3 ]; }; then
  printf 'kilix-llm lab: bash 4.4+ required (running %s)\n' "${BASH_VERSION:-unknown}" >&2
  exit 1
fi

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="${KILIX_LLM_LAB_VENV:-$REPO_DIR/lab/.venv}"

"$REPO_DIR/lab/scripts/uv-venv.sh"
prev=
for arg in "$@"; do
  if [ "$prev" = "--quant" ] && [ "$arg" = "4bit" ]; then
    uv pip install --python "$VENV/bin/python" bitsandbytes accelerate
  fi
  prev=$arg
done
export PYTHONPATH="$REPO_DIR/lab/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$VENV/bin/python" "$REPO_DIR/lab/projects/qwen3_0_6b/run.py" neuron "$@"
