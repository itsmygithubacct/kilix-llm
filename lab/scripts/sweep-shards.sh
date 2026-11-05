#!/usr/bin/env bash
# Split one neuron sweep into N processes. Shard K runs a contiguous piece of
# the index range. On a GPU machine each process sees one device via
# CUDA_VISIBLE_DEVICES. Extra arguments are passed to `lab sweep`.
#
# Records land in --record-dir as shard-K.jsonl and shard-K.log.
set -euo pipefail

if [ "${BASH_VERSINFO[0]:-0}" -lt 4 ] \
    || { [ "${BASH_VERSINFO[0]}" -eq 4 ] && [ "${BASH_VERSINFO[1]}" -le 3 ]; }; then
  printf 'kilix-llm lab: bash 4.4+ required (running %s)\n' "${BASH_VERSION:-unknown}" >&2
  exit 1
fi

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

shards=
record_dir=
forward=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --shards)
      shards="${2:-}"
      shift 2
      ;;
    --record-dir)
      record_dir="${2:-}"
      shift 2
      ;;
    --shard|--record)
      printf 'kilix-llm lab: sweep-shards sets %s\n' "$1" >&2
      exit 2
      ;;
    *)
      forward+=("$1")
      shift
      ;;
  esac
done

if ! [[ "$shards" =~ ^[1-9][0-9]*$ ]]; then
  printf 'kilix-llm lab: sweep-shards needs --shards N\n' >&2
  exit 2
fi
if [ -z "$record_dir" ]; then
  printf 'kilix-llm lab: sweep-shards needs --record-dir DIR\n' >&2
  exit 2
fi

stand_in=0
device_set=0
device_value=
previous=
for arg in "${forward[@]}"; do
  if [ "$previous" = "--device" ]; then
    device_set=1
    device_value=$arg
  fi
  if [ "$arg" = "--stand-in" ]; then
    stand_in=1
  fi
  previous=$arg
done

assign_gpu=0
if [ "$stand_in" -eq 0 ] && { [ "$device_set" -eq 0 ] || [ "$device_value" = "cuda" ]; }; then
  assign_gpu=1
fi

if [ "$assign_gpu" -eq 1 ]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    printf 'kilix-llm lab: sweep-shards needs nvidia-smi to place shards on GPUs\n' >&2
    exit 1
  fi
  gpu_count="$(nvidia-smi -L 2>/dev/null | wc -l)"
  gpu_count="${gpu_count// /}"
  if [ "$gpu_count" -lt "$shards" ]; then
    printf 'kilix-llm lab: %s shards need %s GPUs, found %s\n' \
      "$shards" "$shards" "$gpu_count" >&2
    exit 1
  fi
fi

mkdir -p "$record_dir"
pids=()
for ((i = 0; i < shards; i++)); do
  cmd=("$REPO_DIR/lab/scripts/lab.sh" sweep --shards "$shards" --shard "$i" --record "$record_dir/shard-$i.jsonl")
  if [ "$assign_gpu" -eq 1 ] && [ "$device_set" -eq 0 ]; then
    cmd+=(--device cuda)
  fi
  if [ "${#forward[@]}" -gt 0 ]; then
    cmd+=("${forward[@]}")
  fi
  if [ "$assign_gpu" -eq 1 ]; then
    CUDA_VISIBLE_DEVICES="$i" "${cmd[@]}" >"$record_dir/shard-$i.log" 2>&1 &
  else
    "${cmd[@]}" >"$record_dir/shard-$i.log" 2>&1 &
  fi
  pids+=("$!")
done

status=0
for ((i = 0; i < shards; i++)); do
  if ! wait "${pids[$i]}"; then
    printf 'kilix-llm lab: shard %s failed\n' "$i" >&2
    status=1
  fi
done
exit "$status"
