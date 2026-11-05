"""Attention-head edits on the vector that enters o_proj.

A head is one contiguous slice of that vector. The four operations match a
neuron push: increase and decrease add a scalar, activate replaces the slice,
and deactivate writes zeros. The weights are not modified.
"""

from __future__ import annotations

from collections.abc import Sequence

from kilix_llm_lab import LabError
from kilix_llm_lab.intervene import apply_scalar


def attention_heads(config) -> int:
    """Read num_attention_heads from a causal config or a nested text config."""
    for obj in (config, getattr(config, "text_config", None)):
        if obj is None:
            continue
        count = getattr(obj, "num_attention_heads", None)
        if count:
            return int(count)
    raise LabError("config has no num_attention_heads")


def edit_head_slice(
    vector: Sequence[float],
    head: int,
    n_heads: int,
    op: str,
    amount: float,
) -> list[float]:
    """Return a copy of `vector` with one head slice edited."""
    values = [float(item) for item in vector]
    if n_heads < 1:
        raise LabError("a layer needs at least one attention head")
    if not values or len(values) % n_heads != 0:
        raise LabError(f"width {len(values)} does not divide into {n_heads} heads")
    if not 0 <= head < n_heads:
        raise LabError(f"head {head} is outside 0..{n_heads - 1}")
    head_dim = len(values) // n_heads
    start = head * head_dim
    for offset in range(head_dim):
        values[start + offset] = apply_scalar(values[start + offset], op, amount)
    return values
