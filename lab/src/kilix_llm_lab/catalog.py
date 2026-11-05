"""Bounded neuron catalog: chosen indexes, strengths, and matched controls.

The full 4B sweep is thousands of generations. This record takes an explicit
index list and strength list, so a stand-in can run it with no checkpoint.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from kilix_llm_lab import LabError
from kilix_llm_lab.algebra import one_hot, random_directions
from kilix_llm_lab.texts import EXPLORER_PROMPTS

Generate = Callable[[dict, str], str]


def catalog_record(
    indexes: Sequence[int],
    strengths: Sequence[float],
    *,
    control_count: int,
    width: int,
    generate: Generate,
    prompts: Sequence[str] = EXPLORER_PROMPTS,
    seed: int = 0,
    on_observation: Callable[[dict], None] | None = None,
    store_direction: bool = True,
) -> dict:
    """One observation per (neuron index, strength), then the random controls.

    Each control is a direction whose L2 equals abs(strength), one set per
    strength. `generate(job, prompt)` returns the text for that push. The
    job carries kind, index, strength, and direction. Weights are not read.
    """
    chosen = [int(index) for index in indexes]
    levels = [float(strength) for strength in strengths]
    lines = [str(prompt) for prompt in prompts]
    if not chosen:
        raise LabError("catalog needs at least one neuron index")
    if not levels:
        raise LabError("catalog needs at least one strength")
    if control_count < 1:
        raise LabError("catalog needs at least one random-direction control")
    if not lines:
        raise LabError("catalog needs at least one prompt")
    observations = []
    for index in chosen:
        for strength in levels:
            direction = one_hot(width, index, strength)
            _keep(
                observations,
                _observe("neuron", index, strength, direction, lines, generate),
                on_observation,
                store_direction,
            )
    for strength in levels:
        directions = random_directions(width, abs(strength), control_count, seed)
        for control_index, direction in enumerate(directions):
            _keep(
                observations,
                _observe("control", control_index, strength, direction, lines, generate),
                on_observation,
                store_direction,
            )
    return {"prompts": lines, "observations": observations}


def _keep(observations, observation, on_observation, store_direction: bool) -> None:
    if on_observation is not None:
        on_observation(observation)
    if not store_direction:
        observation = {key: value for key, value in observation.items() if key != "direction"}
    observations.append(observation)


def _observe(kind, index, strength, direction, prompts, generate: Generate) -> dict:
    job = {
        "kind": kind,
        "index": index,
        "strength": strength,
        "direction": direction,
    }
    generations = []
    for prompt in prompts:
        text = generate(job, prompt)
        if not isinstance(text, str):
            raise LabError("catalog generate must return text")
        generations.append({"prompt": prompt, "text": text})
    return {
        "kind": kind,
        "index": index,
        "strength": strength,
        "direction": direction,
        "generations": generations,
    }


def stand_in_generation(job: dict, prompt: str) -> str:
    """Deterministic text so the catalog can run with no checkpoint."""
    return f"{job['kind']} {job['index']} {job['strength']} {prompt}"
