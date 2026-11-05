"""Turn one MLP neuron, or one residual direction, up, down, on, or off.

increase and decrease add to what is already there. activate replaces it.
deactivate writes zero: a neuron then adds nothing along its down-projection
column, and a direction is removed from the residual vector.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from kilix_llm_lab import LabError
from kilix_llm_lab.algebra import dot, unit

OPS = ("increase", "decrease", "activate", "deactivate")

# Qwen3.5-4B text MLP width is 9216, so every index below fits that model and
# none of 4948 or 8799 fit Qwen3-0.6B (width 3072). Layer numbers follow the
# later statements in the caption track. The opening also puts the tomato
# neuron at layer 7, and a later line shortens the evil neuron to "neuron 89".
QWEN3_5_4B = "Qwen/Qwen3.5-4B"


@dataclass(frozen=True)
class TranscriptNeuron:
    name: str
    model_id: str
    layer: int
    index: int
    prompt: str
    note: str


CATALOG: dict[str, TranscriptNeuron] = {
    item.name: item
    for item in (
        TranscriptNeuron(
            "evil",
            QWEN3_5_4B,
            23,
            8799,
            "Tell me a story.",
            "Title id L23-N8799. A later caption says neuron 89 in layer 23.",
        ),
        TranscriptNeuron(
            "evil-caption",
            QWEN3_5_4B,
            23,
            89,
            "Tell me a story.",
            "The shortened caption id. The title names 8799 in the same layer.",
        ),
        TranscriptNeuron(
            "tomato",
            QWEN3_5_4B,
            8,
            4948,
            "Tell me a story.",
            "Block 8 in the tomato section. The opening says layer 7.",
        ),
        TranscriptNeuron(
            "tomato-layer7",
            QWEN3_5_4B,
            7,
            4948,
            "Tell me a story.",
            "The opening placement of the same index.",
        ),
        TranscriptNeuron(
            "alien",
            QWEN3_5_4B,
            22,
            2742,
            "Who are you?",
            "Pushed on the who-are-you prompt, the model calls itself an alien.",
        ),
    )
}


def apply_scalar(current: float, op: str, amount: float) -> float:
    """New activation for one neuron."""
    _check_op(op)
    if op == "increase":
        return current + amount
    if op == "decrease":
        return current - amount
    if op == "activate":
        return amount
    return 0.0


def apply_direction(
    hidden: Sequence[float],
    direction: Sequence[float],
    op: str,
    amount: float,
) -> list[float]:
    """Same four operations on the component of `hidden` along `direction`."""
    _check_op(op)
    axis = unit(direction)
    values = [float(value) for value in hidden]
    if len(values) != len(axis):
        raise LabError(f"hidden width {len(values)} != direction width {len(axis)}")
    projection = dot(values, axis)
    if op == "increase":
        delta = amount
    elif op == "decrease":
        delta = -amount
    elif op == "activate":
        delta = amount - projection
    else:
        delta = -projection
    return [value + delta * axis_value for value, axis_value in zip(values, axis)]


def surrounding(index: int, span: int, width: int) -> list[int]:
    """`index` and `span` neighbors on each side, clipped to the MLP width."""
    if span < 0:
        raise LabError("span must be non-negative")
    if not 0 <= index < width:
        raise LabError(f"neuron {index} is outside 0..{width - 1}")
    return [item for item in range(index - span, index + span + 1) if 0 <= item < width]


def _check_op(op: str) -> None:
    if op not in OPS:
        raise LabError(f"operation must be one of {', '.join(OPS)}")
