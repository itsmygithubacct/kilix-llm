#!/usr/bin/env python3
"""Compare greedy answers before and after a weight edit.

The edit removes the sad-minus-happy direction from one layer's `o_proj` and
`down_proj` (abliteration). The checkpoint files are not modified. Download
and load are setup. The result is how the answers moved: text, mood, fact
hits, and KL(original || edited) on each prompt.

    lab/scripts/bench-qwen3-0.6b.sh --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from kilix_llm_lab.qwen import QWEN3_0_6B  # noqa: E402
from kilix_llm_lab.texts import FACTS  # noqa: E402

PROMPTS = (
    ("Tell me a story.", None),
    ("Tell me a sad story.", None),
    ("Tell me a happy story.", None),
    *FACTS,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare answers before and after abliterating one layer."
    )
    parser.add_argument("--model", default=QWEN3_0_6B)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--new-tokens", type=int, default=48)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="decoder layer to edit (default: largest sad-minus-happy norm)",
    )
    return parser


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _shared_prefix(left: str, right: str) -> int:
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count


def _logits(model, tokenizer, prompt: str):
    import torch
    import torch.nn.functional as F

    from kilix_llm_lab.collect import encode, model_device

    encoded = encode(tokenizer, [prompt], max_length=128)
    device = model_device(model)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        logits = model(**encoded, use_cache=False).logits
    return encoded["attention_mask"], F.log_softmax(logits.float(), dim=-1).cpu()


def _kl(log_p, mask, model, tokenizer, prompt: str) -> float:
    import torch
    import torch.nn.functional as F

    from kilix_llm_lab.collect import encode, model_device

    encoded = encode(tokenizer, [prompt], max_length=128)
    device = model_device(model)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        logits = model(**encoded, use_cache=False).logits
    log_q = F.log_softmax(logits.float(), dim=-1).cpu()
    length = min(log_p.shape[1], log_q.shape[1], mask.shape[1])
    log_p = log_p[:, :length]
    log_q = log_q[:, :length]
    used = mask[:, :length].float().cpu()
    kl = (log_p.exp() * (log_p - log_q)).sum(dim=-1)
    return float((kl * used).sum().item() / used.sum().clamp(min=1).item())


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.new_tokens < 1:
        print("kilix-llm lab: --new-tokens must be positive", file=sys.stderr)
        return 1

    import torch

    from kilix_llm_lab.algebra import column_dots, unit
    from kilix_llm_lab.collect import generate
    from kilix_llm_lab.edit import abliterate
    from kilix_llm_lab.measure import fact_hit, mood_score
    from kilix_llm_lab.qwen import decoder_layers, writing_linears
    from kilix_llm_lab.runner import load, sadness_direction

    if args.device == "cuda" and not torch.cuda.is_available():
        print("kilix-llm lab: cuda was requested but torch has no GPU", file=sys.stderr)
        return 1

    print("kilix-llm lab: loading checkpoint", file=sys.stderr)
    model, tokenizer = load(args.model, args.device)
    found = sadness_direction(model, tokenizer, max_length=args.max_length, keep=None)
    layer = found["layer"] if args.layer is None else args.layer
    if not 0 <= layer < len(found["directions"]):
        print(f"kilix-llm lab: layer {layer} is outside the model", file=sys.stderr)
        return 1
    direction = found["directions"][layer]

    before = []
    saved = []
    for prompt, _expected in PROMPTS:
        text = generate(model, tokenizer, prompt, args.new_tokens)
        mask, log_p = _logits(model, tokenizer, prompt)
        before.append(text)
        saved.append((mask, log_p))

    names = abliterate(model, direction, [layer])
    leftover = max(
        abs(value)
        for linear in writing_linears(decoder_layers(model)[layer])
        for value in column_dots(linear.weight.detach().float().cpu().tolist(), unit(direction))
    )
    if leftover > 1e-3:
        print(f"kilix-llm lab: direction still in the weights ({leftover})", file=sys.stderr)
        return 1

    items = []
    for (prompt, expected), earlier, (mask, log_p) in zip(PROMPTS, before, saved):
        later = generate(model, tokenizer, prompt, args.new_tokens)
        earlier_line = _one_line(earlier)
        later_line = _one_line(later)
        item = {
            "prompt": prompt,
            "before": earlier_line,
            "after": later_line,
            "changed": earlier_line != later_line,
            "shared_prefix_chars": _shared_prefix(earlier_line, later_line),
            "kl": _kl(log_p, mask, model, tokenizer, prompt),
            "mood_before": mood_score(earlier_line),
            "mood_after": mood_score(later_line),
        }
        if expected is not None:
            item["fact"] = expected
            item["fact_before"] = fact_hit(earlier_line, expected)
            item["fact_after"] = fact_hit(later_line, expected)
        print(json.dumps(item), flush=True)
        items.append(item)

    fact_rows = [item for item in items if "fact" in item]
    summary = {
        "edit": "abliterate",
        "modules": names,
        "layer": layer,
        "layer_norm": round(found["norms"][layer], 6),
        "prompts": len(items),
        "responses_changed": sum(1 for item in items if item["changed"]),
        "mean_kl": sum(item["kl"] for item in items) / len(items),
        "facts_before": sum(1 for item in fact_rows if item["fact_before"]),
        "facts_after": sum(1 for item in fact_rows if item["fact_after"]),
        "facts": len(fact_rows),
        "max_abs_dot": leftover,
    }
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
