"""Activation patching. One site at a time, swap the corrupt state for the clean one.

`run` takes the full list of site states and returns whatever `score` reads.
The effect is score(patched) minus score(corrupt). A site that carries the
clean behavior gets a non-zero effect. The others stay at the corrupt score.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from kilix_llm_lab import LabError

State = Sequence
Run = Callable[[Sequence[State]], object]
Score = Callable[[object], float]


def patch_effects(
    clean: Sequence[State],
    corrupt: Sequence[State],
    run: Run,
    score: Score,
) -> list[float]:
    """Effect of patching each clean site into an otherwise corrupt run."""
    if not clean or len(clean) != len(corrupt):
        raise LabError("clean and corrupt traces need the same non-zero site count")
    baseline = float(score(run(corrupt)))
    effects = []
    for index, clean_site in enumerate(clean):
        mixed = list(corrupt)
        mixed[index] = clean_site
        effects.append(float(score(run(mixed))) - baseline)
    return effects
