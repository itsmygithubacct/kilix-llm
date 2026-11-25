"""One prompt through vLLM, then exit. Run by the vLLM environment's Python.

This file is executed as a script by the interpreter that `kilix-llm build
--backend vllm` installed; the launcher's own interpreter never imports it, and
it imports nothing from kilix_llm. Its one argument is a JSON request.

vLLM and its worker processes log to standard output. The answer is the only
thing a caller of `kilix-llm run` wants there, so standard output is pointed at
standard error for the whole load and generation, and the answer is written to
the original descriptor at the end.
"""

from __future__ import annotations

import json
import os
import sys


def split_reasoning(text: str) -> tuple[str, str]:
    """Separate a `<think>` preamble from the reply that follows it."""
    head, mark, tail = text.partition("</think>")
    if not mark:
        return "", text.strip()
    return head.replace("<think>", "").strip(), tail.strip()


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: vllm_oneshot.py REQUEST_JSON", file=sys.stderr)
        return 2
    request = json.loads(argv[1])
    answer_fd = os.dup(1)
    os.dup2(2, 1)

    from vllm import LLM

    llm = LLM(
        model=request["model"],
        served_model_name=request["served_name"],
        hf_overrides=request["hf_overrides"],
        max_model_len=request["ctx"],
        max_num_seqs=1,
        max_num_batched_tokens=request["batch_tokens"],
        gpu_memory_utilization=request["gpu_memory_utilization"],
        kv_cache_memory_bytes=request["kv_cache_memory_bytes"],
    )
    params = llm.get_default_sampling_params()
    params.max_tokens = request["max_tokens"]
    outputs = llm.chat(
        [{"role": "user", "content": request["prompt"]}],
        params,
        chat_template_kwargs={"enable_thinking": request["think"]},
    )
    reasoning, answer = split_reasoning(outputs[0].outputs[0].text)
    if reasoning:
        print(f"[thinking]\n{reasoning}\n[/thinking]", file=sys.stderr, flush=True)
    sys.stdout.flush()
    with os.fdopen(answer_fd, "w", encoding="utf-8") as out:
        out.write(answer + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
