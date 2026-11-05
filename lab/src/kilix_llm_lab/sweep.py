"""Index range for a full MLP sweep. Qwen3.5-4B has 9,216 neurons in a layer.

The record is the bounded catalog on that range. A stand-in still takes an
explicit stop so a test does not generate every neuron. The shipped range
`0:9216` is the full layer.
"""

from __future__ import annotations

from kilix_llm_lab import LabError
from kilix_llm_lab.qwen import INTERMEDIATE_4B


def sweep_indexes(start: int, stop: int, width: int = INTERMEDIATE_4B) -> list[int]:
    """Half-open neuron indexes. `stop` is exclusive and must fit the width."""
    if width < 1:
        raise LabError("sweep width must be positive")
    if not 0 <= start < stop <= width:
        raise LabError(f"sweep range {start}:{stop} is outside 0..{width - 1}")
    return list(range(start, stop))


def shard_bounds(start: int, stop: int, shards: int, shard: int) -> tuple[int, int]:
    """Half-open piece `shard` of `shards`. Piece lengths differ by at most one.

    `stop` is exclusive. The pieces cover `start:stop` without overlap.
    """
    if shards < 1:
        raise LabError("shard count must be positive")
    if not 0 <= shard < shards:
        raise LabError(f"shard {shard} is outside 0..{shards - 1}")
    if stop <= start:
        raise LabError(f"sweep range {start}:{stop} is empty")
    span = stop - start
    if shards > span:
        raise LabError(f"{shards} shards do not fit in {start}:{stop}")
    size, extra = divmod(span, shards)
    if shard < extra:
        piece = start + shard * (size + 1)
        return piece, piece + size + 1
    piece = start + extra * (size + 1) + (shard - extra) * size
    return piece, piece + size
