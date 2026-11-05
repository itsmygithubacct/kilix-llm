"""Direction arithmetic shared by residual and neuron edits.

Vectors are plain sequences of floats. Weight matrices are row-major, shape
(out_features, in_features), matching torch.nn.Linear.weight.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

from kilix_llm_lab import LabError

Vector = list[float]
Matrix = list[Vector]


def as_vector(values: Sequence[float]) -> Vector:
    vector = [float(value) for value in values]
    if not vector:
        raise LabError("direction is empty")
    return vector


def mean(vectors: Sequence[Sequence[float]]) -> Vector:
    rows = [as_vector(row) for row in vectors]
    if not rows:
        raise LabError("cannot average an empty set of vectors")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise LabError("vectors in one mean have different widths")
    inv = 1.0 / len(rows)
    return [sum(row[index] for row in rows) * inv for index in range(width)]


def combine_counts(parts: Sequence[tuple[Sequence[float], float]]) -> Vector:
    """Token-weighted mean. Each part is (mean vector, token count)."""
    if not parts:
        raise LabError("cannot combine an empty set of means")
    total = 0.0
    acc: Vector | None = None
    for values, count in parts:
        vector = as_vector(values)
        if count <= 0:
            raise LabError("token count must be positive")
        if acc is None:
            acc = [0.0] * len(vector)
        elif len(vector) != len(acc):
            raise LabError("means in one combine have different widths")
        for index, value in enumerate(vector):
            acc[index] += value * count
        total += count
    assert acc is not None
    return [value / total for value in acc]


def subtract(left: Sequence[float], right: Sequence[float]) -> Vector:
    a = as_vector(left)
    b = as_vector(right)
    if len(a) != len(b):
        raise LabError(f"widths {len(a)} and {len(b)} do not match")
    return [x - y for x, y in zip(a, b)]


def l2(vector: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in as_vector(vector)))


def unit(vector: Sequence[float]) -> Vector:
    values = as_vector(vector)
    norm = l2(values)
    if norm == 0.0:
        raise LabError("cannot normalize the zero vector")
    return [value / norm for value in values]


def scale(vector: Sequence[float], factor: float) -> Vector:
    return [float(factor) * value for value in as_vector(vector)]


def add(left: Sequence[float], right: Sequence[float]) -> Vector:
    a = as_vector(left)
    b = as_vector(right)
    if len(a) != len(b):
        raise LabError(f"widths {len(a)} and {len(b)} do not match")
    return [x + y for x, y in zip(a, b)]


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    a = as_vector(left)
    b = as_vector(right)
    if len(a) != len(b):
        raise LabError(f"widths {len(a)} and {len(b)} do not match")
    return sum(x * y for x, y in zip(a, b))


def sparsify(vector: Sequence[float], keep: int) -> Vector:
    """Keep the largest-magnitude entries. Ties break toward the lower index."""
    values = as_vector(vector)
    if keep < 1:
        raise LabError("keep must be at least 1")
    if keep >= len(values):
        return values
    ranked = sorted(range(len(values)), key=lambda index: (-abs(values[index]), index))
    chosen = set(ranked[:keep])
    return [value if index in chosen else 0.0 for index, value in enumerate(values)]


def strongest_index(vectors: Sequence[Sequence[float]]) -> int:
    if not vectors:
        raise LabError("no directions to rank")
    return max(range(len(vectors)), key=lambda index: (l2(vectors[index]), -index))


def contrast(positive: Sequence[Sequence[float]], negative: Sequence[Sequence[float]]) -> Vector:
    """Mean(positive) - mean(negative). One group of same-width vectors."""
    return subtract(mean(positive), mean(negative))


def difference_of_means(
    class_a: Sequence[Sequence[float]], class_b: Sequence[Sequence[float]]
) -> Vector:
    """Token-group difference of means: mean(class_a) - mean(class_b)."""
    return contrast(class_a, class_b)


def first_divergence(left: Sequence[int], right: Sequence[int]) -> int:
    """Index where two token-id sequences stop matching.

    The shared prefix is the part that says nothing about the class. Equal
    sequences, and a sequence that is a prefix of the other, return the last
    shared index. An empty sequence is refused.
    """
    a = [int(token) for token in left]
    b = [int(token) for token in right]
    if not a or not b:
        raise LabError("a paired sequence is empty")
    for index, (one, other) in enumerate(zip(a, b)):
        if one != other:
            return index
    return min(len(a), len(b)) - 1


def mean_from(rows: Sequence[Sequence[float]], start: int) -> Vector:
    """Mean of the rows from `start` onward. Earlier rows are the shared prefix."""
    if start < 0:
        raise LabError("divergence index is negative")
    if start >= len(rows):
        raise LabError(f"divergence index {start} is past {len(rows)} tokens")
    return mean(list(rows)[start:])


def paired_class_mean(
    examples: Sequence[tuple[Sequence[Sequence[float]], int]],
) -> Vector:
    """Mean of each example's tail, then the mean of those tails.

    Each example is (per-token vectors, first index to keep).
    """
    if not examples:
        raise LabError("paired mean needs at least one example")
    return mean([mean_from(rows, start) for rows, start in examples])


def aligned_scores(
    contrasts: Sequence[float],
    columns: Sequence[Sequence[float]],
    direction: Sequence[float],
) -> Vector:
    """Activation contrast times the write column's dot with a residual direction.

    `columns[i]` is what neuron i adds to the residual stream, one down_proj
    column. A positive score moves the residual along `direction` when the
    class contrast is positive.
    """
    axis = as_vector(direction)
    if len(contrasts) != len(columns) or not contrasts:
        raise LabError("aligned score needs one contrast per column")
    return [float(contrast) * dot(column, axis) for contrast, column in zip(contrasts, columns)]


def top_aligned(
    contrasts: Sequence[float],
    columns: Sequence[Sequence[float]],
    direction: Sequence[float],
    keep: int,
) -> list[tuple[int, float]]:
    """Largest aligned scores as (neuron index, score). Ties keep the lower index."""
    scores = aligned_scores(contrasts, columns, direction)
    if keep < 1:
        raise LabError("keep must be at least 1")
    ranked = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    return [(index, scores[index]) for index in ranked[:keep]]


def layer_contrasts(
    positive: Sequence[Sequence[Sequence[float]]],
    negative: Sequence[Sequence[Sequence[float]]],
) -> list[Vector]:
    """Per-layer difference of means. Both arguments are layer, then example."""
    if len(positive) != len(negative) or not positive:
        raise LabError("the two classes need the same non-zero layer count")
    directions = []
    for layer, (pos, neg) in enumerate(zip(positive, negative)):
        if len(pos[0]) != len(neg[0]):
            raise LabError(f"layer {layer} class widths differ")
        directions.append(contrast(pos, neg))
    return directions


def project_out(rows: Sequence[Sequence[float]], direction: Sequence[float]) -> Matrix:
    """Left-multiply P = I - r r^T, with r unit length, onto a Linear weight.

    Columns of the result are orthogonal to r. Applying it twice changes nothing.
    """
    weight = [as_vector(row) for row in rows]
    if not weight:
        raise LabError("weight has no rows")
    width = len(weight[0])
    if any(len(row) != width for row in weight):
        raise LabError("weight rows have different widths")
    direction_u = unit(direction)
    if len(direction_u) != len(weight):
        raise LabError(
            f"direction width {len(direction_u)} != out_features {len(weight)}"
        )
    dots = [
        sum(direction_u[row] * weight[row][col] for row in range(len(weight)))
        for col in range(width)
    ]
    return [
        [weight[row][col] - direction_u[row] * dots[col] for col in range(width)]
        for row in range(len(weight))
    ]


def project_out_vector(values: Sequence[float], direction: Sequence[float]) -> Vector:
    """P @ v for a bias or any output-space vector."""
    vector = as_vector(values)
    direction_u = unit(direction)
    if len(direction_u) != len(vector):
        raise LabError(
            f"direction width {len(direction_u)} != vector width {len(vector)}"
        )
    shared = dot(direction_u, vector)
    return [value - shared * axis for value, axis in zip(vector, direction_u)]


def column_dots(rows: Sequence[Sequence[float]], direction: Sequence[float]) -> Vector:
    """r · each column. Zero means the weight no longer writes r."""
    weight = [as_vector(row) for row in rows]
    direction_v = as_vector(direction)
    if len(direction_v) != len(weight):
        raise LabError("direction width does not match out_features")
    width = len(weight[0])
    return [
        sum(direction_v[row] * weight[row][col] for row in range(len(weight)))
        for col in range(width)
    ]


def kl_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    """KL(p || q) in nats. p and q are distributions over the same support."""
    left = as_vector(p)
    right = as_vector(q)
    if len(left) != len(right):
        raise LabError("distributions have different widths")
    if any(value < 0 for value in left) or any(value < 0 for value in right):
        raise LabError("probabilities must be non-negative")
    total = 0.0
    for pi, qi in zip(left, right):
        if pi == 0.0:
            continue
        if qi == 0.0:
            raise LabError("KL is infinite where q is zero and p is not")
        total += pi * (math.log(pi) - math.log(qi))
    return total


def one_hot(width: int, index: int, amount: float) -> Vector:
    if width < 1:
        raise LabError("width must be positive")
    if not 0 <= index < width:
        raise LabError(f"index {index} is outside 0..{width - 1}")
    vector = [0.0] * width
    vector[index] = float(amount)
    return vector


def random_directions(width: int, norm: float, count: int, seed: int) -> list[Vector]:
    """Gaussian directions rescaled to a shared L2 norm. Seeded and deterministic."""
    if width < 1 or count < 1:
        raise LabError("width and count must be positive")
    if norm < 0:
        raise LabError("norm must be non-negative")
    rng = random.Random(seed)
    directions = []
    for _ in range(count):
        raw = [rng.gauss(0.0, 1.0) for _ in range(width)]
        current = l2(raw)
        if current == 0.0:
            raw[0] = 1.0
            current = 1.0
        factor = norm / current
        directions.append([value * factor for value in raw])
    return directions


def rank_components(vector: Sequence[float], keep: int) -> list[tuple[int, float]]:
    """Largest-magnitude coordinates, as (index, value)."""
    values = as_vector(vector)
    if keep < 1:
        raise LabError("keep must be at least 1")
    ranked = sorted(range(len(values)), key=lambda index: (-abs(values[index]), index))
    return [(index, values[index]) for index in ranked[:keep]]
