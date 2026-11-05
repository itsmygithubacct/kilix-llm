"""Refusal direction from a two-class corpus, then the same projection as abliteration.

The built-in classes are refusal wording and compliant wording. A JSON object
with keys `refuse` and `comply` replaces those lists. Each value is a list of
strings. The direction at each layer is mean(refuse) minus mean(comply).
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from kilix_llm_lab import LabError
from kilix_llm_lab.algebra import subtract
from kilix_llm_lab.texts import COMPLY, REFUSE


def builtin_corpus() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Refusal wording first, compliant wording second."""
    return REFUSE, COMPLY


def load_corpus(path: str) -> tuple[list[str], list[str]]:
    """Read `{"refuse": [...], "comply": [...]}` from a UTF-8 JSON file."""
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise LabError("refusal corpus must be a JSON object")
    refuse = _strings(payload.get("refuse"), "refuse")
    comply = _strings(payload.get("comply"), "comply")
    return refuse, comply


def refusal_directions(
    refuse_layers: Sequence[Sequence[float]],
    comply_layers: Sequence[Sequence[float]],
) -> list[list[float]]:
    """One difference-of-means vector per layer: refuse minus comply."""
    if not refuse_layers or len(refuse_layers) != len(comply_layers):
        raise LabError("refuse and comply captures need the same non-zero layer count")
    return [subtract(left, right) for left, right in zip(refuse_layers, comply_layers)]


def _strings(value, name: str) -> list[str]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
        raise LabError(f"refusal corpus key {name} must be a non-empty list of strings")
    return list(value)
