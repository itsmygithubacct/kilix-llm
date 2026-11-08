#!/usr/bin/env bash
# Dispatch one lab command. `kilix llm lab COMMAND` execs this script.
#
# With no arguments, print the commands.
set -euo pipefail

if [ "${BASH_VERSINFO[0]:-0}" -lt 4 ] \
    || { [ "${BASH_VERSINFO[0]}" -eq 4 ] && [ "${BASH_VERSINFO[1]}" -le 3 ]; }; then
  printf 'kilix-llm lab: bash 4.4+ required (running %s)\n' "${BASH_VERSION:-unknown}" >&2
  exit 1
fi

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="${KILIX_LLM_LAB_VENV:-$REPO_DIR/lab/.venv}"

_install_4bit() {
  local prev= arg
  for arg in "$@"; do
    if [ "$prev" = "--quant" ] && [ "$arg" = "4bit" ]; then
      uv pip install --python "$VENV/bin/python" bitsandbytes accelerate
      return 0
    fi
    prev=$arg
  done
}

usage() {
  cat <<'EOF'
usage: kilix llm lab COMMAND [ARGS...]
  venv        create the uv environment
  fetch       download one checkpoint into HF_HOME (default Qwen/Qwen3.5-4B)
  means       difference of means: sad minus happy, ranked by L2
  steer       hook steering on one or more layers
  abliterate  abliteration: P = I - r r^T on attention and MLP writers
  neuron      neuron push: increase, decrease, activate, or deactivate
  head        attention-head edit: increase, decrease, activate, or deactivate
  sae         sparse autoencoder training on activation rows
  circuit     circuit trace by activation patching
  refusal     refusal corpus: refusal wording minus compliant wording
  sweep       neuron sweep across an index range, including all 9216
  sweep-shards  split a sweep into one process per shard
  control     random-direction control: win only by beating every control
  rank        rank recorded generations by distinctive-word mass
  weight      weight-space column edit: remove a direction, scale, or zero
  loops       find repeating tails in thinking traces
  gguf-columns  copy a GGUF with exact column zero or scale edits
  output-row  squared-hinge edit of one output row from frozen activations
  train       prepare a text corpus and run llama-finetune
  catalog     bounded catalog of chosen neurons plus matched random directions
  story       steer one layer and score how the answer changes
  neurons     contrast MLP neurons on one layer
  bench       compare answers before and after one weight edit
  queue       distribute a JSON suite across remembered machines
EOF
}

if [ "$#" -eq 0 ]; then
  usage
  exit 0
fi

command="$1"
shift
case "$command" in
  -h|--help|help)
    usage
    exit 0 ;;
  venv)
    if [ "$#" -gt 0 ]; then
      usage >&2
      exit 2
    fi
    exec "$REPO_DIR/lab/scripts/uv-venv.sh" ;;
  fetch)
    exec "$REPO_DIR/lab/scripts/fetch-model.sh" "$@" ;;
  bench)
    exec "$REPO_DIR/lab/scripts/bench-qwen3-0.6b.sh" "$@" ;;
  queue)
    exec "$REPO_DIR/lab/scripts/bench-queue.sh" "$@" ;;
  neuron)
    exec "$REPO_DIR/lab/scripts/neuron.sh" "$@" ;;
  sweep-shards)
    exec "$REPO_DIR/lab/scripts/sweep-shards.sh" "$@" ;;
  sae)
    export PYTHONPATH="$REPO_DIR/lab/src${PYTHONPATH:+:$PYTHONPATH}"
    exec python3 "$REPO_DIR/lab/projects/qwen3_0_6b/run.py" sae "$@" ;;
  circuit)
    export PYTHONPATH="$REPO_DIR/lab/src${PYTHONPATH:+:$PYTHONPATH}"
    exec python3 "$REPO_DIR/lab/projects/qwen3_0_6b/run.py" circuit "$@" ;;
  head|refusal|sweep)
    stand_in=0
    list_only=0
    for arg in "$@"; do
      if [ "$arg" = "--stand-in" ]; then
        stand_in=1
      fi
      if [ "$arg" = "--list" ]; then
        list_only=1
      fi
    done
    export PYTHONPATH="$REPO_DIR/lab/src${PYTHONPATH:+:$PYTHONPATH}"
    if [ "$stand_in" = 1 ] || [ "$list_only" = 1 ]; then
      exec python3 "$REPO_DIR/lab/projects/qwen3_0_6b/run.py" "$command" "$@"
    fi
    "$REPO_DIR/lab/scripts/uv-venv.sh"
    _install_4bit "$@"
    exec "$VENV/bin/python" "$REPO_DIR/lab/projects/qwen3_0_6b/run.py" "$command" "$@" ;;
  control|rank)
    export PYTHONPATH="$REPO_DIR/lab/src${PYTHONPATH:+:$PYTHONPATH}"
    exec python3 "$REPO_DIR/lab/projects/qwen3_0_6b/run.py" "$command" "$@" ;;
  loops|gguf-columns)
    export PYTHONPATH="$REPO_DIR/lab/src${PYTHONPATH:+:$PYTHONPATH}"
    exec python3 "$REPO_DIR/lab/projects/qwen3_0_6b/run.py" "$command" "$@" ;;
  weight)
    export PYTHONPATH="$REPO_DIR/lab/src${PYTHONPATH:+:$PYTHONPATH}"
    use_matrix=0
    use_model=0
    for arg in "$@"; do
      if [ "$arg" = "--matrix" ]; then
        use_matrix=1
      fi
      if [ "$arg" = "--layer" ]; then
        use_model=1
      fi
    done
    if [ "$use_matrix" = 1 ] || [ "$use_model" = 0 ]; then
      exec python3 "$REPO_DIR/lab/projects/qwen3_0_6b/run.py" weight "$@"
    fi
    "$REPO_DIR/lab/scripts/uv-venv.sh"
    exec "$VENV/bin/python" "$REPO_DIR/lab/projects/qwen3_0_6b/run.py" weight "$@" ;;
  output-row)
    "$REPO_DIR/lab/scripts/uv-venv.sh"
    export PYTHONPATH="$REPO_DIR/lab/src${PYTHONPATH:+:$PYTHONPATH}"
    exec "$VENV/bin/python" "$REPO_DIR/lab/projects/qwen3_0_6b/run.py" output-row "$@" ;;
  train)
    # llama-finetune is the vendored binary. uv-venv.sh installs a torch wheel
    # and its default index is CPU, which would replace a CUDA torch.
    export PYTHONPATH="$REPO_DIR/lab/src${PYTHONPATH:+:$PYTHONPATH}"
    exec python3 "$REPO_DIR/lab/projects/qwen3_0_6b/run.py" train "$@" ;;
  catalog)
    stand_in=0
    for arg in "$@"; do
      if [ "$arg" = "--stand-in" ]; then
        stand_in=1
      fi
    done
    export PYTHONPATH="$REPO_DIR/lab/src${PYTHONPATH:+:$PYTHONPATH}"
    if [ "$stand_in" = 0 ]; then
      "$REPO_DIR/lab/scripts/uv-venv.sh"
      exec "$VENV/bin/python" "$REPO_DIR/lab/projects/qwen3_0_6b/run.py" catalog "$@"
    fi
    exec python3 "$REPO_DIR/lab/projects/qwen3_0_6b/run.py" catalog "$@" ;;
  means|story|neurons|steer|abliterate)
    "$REPO_DIR/lab/scripts/uv-venv.sh"
    _install_4bit "$@"
    export PYTHONPATH="$REPO_DIR/lab/src${PYTHONPATH:+:$PYTHONPATH}"
    exec "$VENV/bin/python" "$REPO_DIR/lab/projects/qwen3_0_6b/run.py" "$command" "$@" ;;
  *)
    usage >&2
    exit 2 ;;
esac
