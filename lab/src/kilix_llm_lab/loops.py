"""Find generation loops and the in-context copy that sustains them. Stdlib only.

A loop is a trace whose tail repeats exactly. `find_loop` looks for the last
`window` characters occurring `copies` or more times; the gap between the
last two occurrences is the period, and the onset is the first character
from which the text repeats itself one period later all the way to the end.

`copy_targets` is the token-level view used against a model. At position t
it finds the most recent earlier occurrence of the `k` tokens ending at t
and returns the token that followed it. When a model is looping, the token
it emits next is that copy target. Comparing a model's probability of the
copy target inside and outside a loop separates copying from ordinary
prediction. These functions read text or ids only and load no model.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from kilix_llm_lab import LabError


def find_loop(text: str, window: int = 800, copies: int = 3) -> dict | None:
    """Describe the repeating tail of `text`, or None when it does not repeat.

    Returns `period` (characters between the last two copies of the tail),
    `onset` (first character of the periodic run), `cycles` (periodic run
    length divided by the period), and `copies` (occurrences of the tail).
    """
    if window < 1 or copies < 2:
        raise LabError("window must be positive and copies at least 2")
    if len(text) < window * copies:
        return None
    tail = text[-window:]
    positions = []
    start = 0
    while True:
        found = text.find(tail, start)
        if found < 0:
            break
        positions.append(found)
        start = found + 1
    if len(positions) < copies:
        return None
    period = positions[-1] - positions[-2]
    if period <= 0:
        return None
    onset = periodic_onset(text, period)
    run = len(text) - onset
    return {
        "period": period,
        "onset": onset,
        "cycles": run / period,
        "copies": len(positions),
    }


def periodic_onset(seq: Sequence, period: int) -> int:
    """First index from which seq[i] == seq[i + period] holds to the end."""
    if period < 1:
        raise LabError("period must be positive")
    index = len(seq) - period
    while index > 0 and seq[index - 1] == seq[index - 1 + period]:
        index -= 1
    return max(index, 0)


def find_token_loop(ids: Sequence[int], window: int = 160, copies: int = 3) -> dict | None:
    """`find_loop` on token ids. Periods and onsets are token positions."""
    if window < 1 or copies < 2:
        raise LabError("window must be positive and copies at least 2")
    values = list(ids)
    if len(values) < window * copies:
        return None
    tail = values[-window:]
    positions = [
        index for index in range(len(values) - window + 1)
        if values[index] == tail[0] and values[index:index + window] == tail
    ]
    if len(positions) < copies:
        return None
    period = positions[-1] - positions[-2]
    onset = periodic_onset(values, period)
    return {
        "period": period,
        "onset": onset,
        "cycles": (len(values) - onset) / period,
        "copies": len(positions),
    }


def copy_targets(ids: Sequence[int], k: int = 4) -> list[int | None]:
    """Next-token copy prediction at every position.

    Entry t is the token that followed the most recent earlier occurrence of
    ids[t-k+1 : t+1], or None when that k-gram has not occurred before.
    """
    if k < 1:
        raise LabError("k must be positive")
    values = list(ids)
    last: dict[tuple, int] = {}
    targets: list[int | None] = []
    for t in range(len(values)):
        if t + 1 < k:
            targets.append(None)
            continue
        gram = tuple(values[t - k + 1:t + 1])
        earlier = last.get(gram)
        targets.append(values[earlier + 1] if earlier is not None and earlier + 1 <= t else None)
        last[gram] = t
    return targets


def copy_agreement(ids: Sequence[int], k: int = 4, start: int = 0) -> dict:
    """How often the next token equals the copy target, from `start` on."""
    targets = copy_targets(ids, k)
    values = list(ids)
    have = agree = 0
    for t in range(max(start, 0), len(values) - 1):
        if targets[t] is None:
            continue
        have += 1
        agree += int(targets[t] == values[t + 1])
    return {"positions": len(values) - 1 - max(start, 0), "with_target": have, "agree": agree,
            "rate": agree / have if have else 0.0}


def repeat_fraction(text: str) -> float:
    """Share of the second half's non-empty lines that already appeared in the first half."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if len(lines) < 2:
        return 0.0
    half = len(lines) // 2
    seen = set(lines[:half])
    later = lines[half:]
    return sum(1 for line in later if line in seen) / len(later)


def run_loops(args) -> int:
    """Print one JSON report: which trace rows loop, where, and how long the period is."""
    rows = []
    for path in args.traces:
        with open(path, encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise LabError(f"{path}:{number} is not JSON: {error}") from error
                text = row.get(args.field)
                if not isinstance(text, str):
                    raise LabError(f"{path}:{number} has no string field {args.field!r}")
                found = find_loop(text, window=args.window, copies=args.copies)
                rows.append({
                    "file": path,
                    "line": number,
                    "id": row.get("prompt_id", row.get("index")),
                    "seed": row.get("seed"),
                    "chars": len(text),
                    "loop": found,
                    "repeat_fraction": round(repeat_fraction(text), 4),
                })
    looping = [item for item in rows if item["loop"]]
    report = {
        "rows": len(rows),
        "loops": len(looping),
        "window": args.window,
        "copies": args.copies,
        "field": args.field,
        "items": rows if args.all else looping,
    }
    print(json.dumps(report, indent=1))
    return 0
