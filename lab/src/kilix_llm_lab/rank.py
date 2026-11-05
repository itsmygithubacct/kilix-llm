"""Distinctive-word rank for catalog and sweep generations.

One unit is a (kind, index) pair. Strengths are pooled. A word counts once
per generation. The mass is the summed positive Poisson z of the top words,
against the average unit. null_pct is the share of random-direction controls
with a lower mass. No checkpoint and no numpy.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence

from kilix_llm_lab import LabError

# Letters only, length at least 3. Digits and underscores are not words.
_TOKEN = re.compile(r"[^\W\d_]{3,}")
_NULL_KINDS = frozenset({"control", "random"})


def letter_words(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


def distinct4(token_ids: Sequence[int]) -> float:
    """Unique 4-grams over token ids, divided by the gram count. Short rows are 1."""
    grams = [tuple(token_ids[index : index + 4]) for index in range(len(token_ids) - 3)]
    if not grams:
        return 1.0
    return len(set(grams)) / len(grams)


def flatten_rows(rows: Sequence[Mapping]) -> list[dict]:
    """Accept either one generation per object or a catalog observation with generations."""
    flat = []
    for row in rows:
        generations = row.get("generations")
        if generations is None:
            flat.append(dict(row))
            continue
        for generation in generations:
            item = {
                "kind": row["kind"],
                "index": row["index"] if "index" in row else row.get("neuron"),
                "prompt": generation["prompt"],
                "text": generation["text"],
            }
            if "strength" in row:
                item["strength"] = row["strength"]
            if "token_ids" in generation:
                item["token_ids"] = generation["token_ids"]
            flat.append(item)
    return flat


def load_jsonl(path: str) -> list[dict]:
    rows = []
    try:
        handle = open(path, encoding="utf-8")
    except OSError as error:
        raise LabError(f"cannot read rank file: {error.strerror}") from error
    with handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                item = json.loads(text)
            except json.JSONDecodeError as error:
                raise LabError(f"rank line {line_number} is not JSON") from error
            if not isinstance(item, dict):
                raise LabError(f"rank line {line_number} is not an object")
            rows.append(item)
    if not rows:
        raise LabError("rank file is empty")
    return rows


def _index(row: Mapping) -> int:
    if "index" in row and row["index"] is not None:
        return int(row["index"])
    if "neuron" in row and row["neuron"] is not None:
        return int(row["neuron"])
    raise LabError("a generation needs an index")


def _kept(row: Mapping, max_strength: float | None) -> bool:
    if max_strength is None or "strength" not in row or row["strength"] is None:
        return True
    return abs(float(row["strength"])) <= max_strength


def rank_units(
    rows: Sequence[Mapping],
    *,
    top_words: int = 12,
    mass_words: int = 5,
    max_strength: float | None = None,
) -> list[dict]:
    """One record per unit, highest word mass first.

    Baseline generations, kind "baseline", contribute words to exclude and
    are not themselves ranked. A missing control set leaves null_pct null.
    """
    if top_words < 1 or mass_words < 1:
        raise LabError("word lists must keep at least one word")
    flat = flatten_rows(rows)
    if not flat:
        raise LabError("rank needs at least one generation")
    base_words: set[str] = set()
    for row in flat:
        if row.get("kind") == "baseline":
            base_words |= letter_words(str(row.get("text", "")))

    units: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in flat:
        kind = row.get("kind")
        if kind == "baseline":
            continue
        if not isinstance(kind, str) or not kind:
            raise LabError("a generation needs a kind")
        if "text" not in row:
            raise LabError("a generation needs text")
        units[(kind, _index(row))].append(dict(row))
    if not units:
        raise LabError("rank needs at least one steered generation")

    counts: dict[tuple[str, int], Counter[str]] = {}
    for key, group in units.items():
        counter: Counter[str] = Counter()
        for row in group:
            if _kept(row, max_strength):
                counter.update(letter_words(str(row["text"])) - base_words)
        counts[key] = counter
    total: Counter[str] = Counter()
    for counter in counts.values():
        total.update(counter)
    mean = {word: value / len(counts) for word, value in total.items()}

    records = []
    for key, group in units.items():
        counter = counts[key]
        ranked = sorted(
            (
                ((count - mean[word]) / math.sqrt(mean[word] + 0.5), word, count)
                for word, count in counter.items()
                if count >= 2
            ),
            reverse=True,
        )[:top_words]
        head = ranked[:mass_words]
        per_prompt = {
            word: sorted(
                {str(row.get("prompt", "")) for row in group if word in letter_words(str(row["text"]))}
            )
            for _score, word, _count in head
        }
        mass = sum(max(score, 0.0) for score, _word, _count in head)
        record = {
            "kind": key[0],
            "index": key[1],
            "z": round(ranked[0][0], 3) if ranked else 0.0,
            "mass": round(mass, 3),
            "words": [[word, count, round(score, 2)] for score, word, count in ranked],
            "per_prompt": per_prompt,
        }
        identified = [row["token_ids"] for row in group if "token_ids" in row]
        if identified:
            record["distinct4"] = round(
                sum(distinct4(ids) for ids in identified) / len(identified),
                3,
            )
        records.append(record)

    null = sorted(record["mass"] for record in records if record["kind"] in _NULL_KINDS)
    for record in records:
        if not null:
            record["null_pct"] = None
            continue
        below = 0
        for mass in null:
            if mass < record["mass"]:
                below += 1
            else:
                break
        record["null_pct"] = round(below / len(null), 4)
    records.sort(key=lambda record: (-record["mass"], record["index"], record["kind"]))
    return records
