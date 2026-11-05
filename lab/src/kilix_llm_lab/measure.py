"""Scores that do not need a second model: lexicon mood and fact hits.

KL on a real vocabulary lives next to the torch runner. The scalar formula is
kilix_llm_lab.algebra.kl_divergence.
"""

from __future__ import annotations

import re

from collections.abc import Sequence

from kilix_llm_lab import LabError

# Word lists are the whole judge. They overlap the built-in contrast texts on
# purpose so a dry run can see the two classes separate before any model load.
HAPPY_WORDS = frozenset(
    {"happy", "joy", "joyful", "glad", "cheerful", "bright", "laugh", "laughter", "smile", "warm"}
)
SAD_WORDS = frozenset(
    {"sad", "grief", "empty", "lonely", "cry", "tears", "miserable", "sorrow"}
)

_WORDS = re.compile(r"[a-z']+")


def tokens(text: str) -> list[str]:
    return _WORDS.findall(text.lower())


def mood_score(text: str) -> float:
    """+1 is only happy lexicon hits, -1 is only sad hits, 0 is neither or a tie."""
    words = tokens(text)
    happy = sum(word in HAPPY_WORDS for word in words)
    sad = sum(word in SAD_WORDS for word in words)
    if happy + sad == 0:
        return 0.0
    return (happy - sad) / (happy + sad)


def fact_hit(answer: str, expected: str) -> bool:
    if not expected:
        raise LabError("fact expects a non-empty answer substring")
    return expected.lower() in answer.lower()


def beats_every_control(effect: float, controls: Sequence[float]) -> bool:
    """True only when the neuron effect is strictly larger than every control.

    A same-magnitude random direction is the control. The neuron wins only
    when it beats all of them. An empty control list is refused.
    """
    values = [float(value) for value in controls]
    if not values:
        raise LabError("a neuron push needs at least one random-direction control")
    return all(float(effect) > value for value in values)


def topic_success(reply: str, topic: str, target: str) -> bool:
    """True when a steered reply names the target, still names the topic, and is not a loop.

    The topic check uses the last word, with one trailing "s" removed, so
    "the weekend" matches "weekend". At least half of the letter-words must
    be distinct. A line such as "weekend weekend tomato" still passes.
    """
    if not str(target) or not str(topic).strip():
        raise LabError("success needs a target and a topic")
    folded = reply.lower()
    named = target.lower() in folded
    topic_word = topic.split()[-1].lower().rstrip("s")
    if not topic_word:
        raise LabError("topic has no word to match")
    on_topic = topic_word in folded
    words = re.findall(r"[a-z]+", folded)
    varied = len(set(words)) >= 0.5 * max(1, len(words))
    return bool(named and on_topic and varied)


def fact_rate(pairs: list[tuple[str, bool]]) -> tuple[int, int]:
    """(hits, total) from (prompt, hit) rows."""
    if not pairs:
        raise LabError("fact list is empty")
    hits = sum(1 for _prompt, hit in pairs if hit)
    return hits, len(pairs)
