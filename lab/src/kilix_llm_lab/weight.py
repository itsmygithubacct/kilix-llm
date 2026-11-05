"""Column edits on one writing matrix.

A column of mlp.down_proj or self_attn.o_proj is one input's contribution
to the residual stream. Hugging Face stores the matrix as
(out_features, in_features). These edits change the loaded matrix only.

remove subtracts scale * u (u · column) from each chosen column. u is
normalized. scale 1 removes that direction. scale 1.5 is the stronger
form. Omitting the columns edits every column of that one matrix.

This is not abliteration. Abliteration writes P = I - r r^T into both
o_proj and down_proj, including a bias. A column edit leaves the bias
alone and can leave other columns alone.

scale multiplies named columns by a factor. zero sets them to 0. A
constant factor on every column is refused. The embedding, the output
projection, and a final norm are refused. No backward pass runs.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from kilix_llm_lab import LabError

WRITING_MODULES = ("down_proj", "o_proj")
REFUSED_LEAVES = ("embed_tokens", "lm_head", "norm")


def assert_editable(name: str) -> str:
    """Return the leaf when it is down_proj or o_proj. Refuse the rest."""
    leaf = name.rsplit(".", 1)[-1]
    if leaf in REFUSED_LEAVES:
        raise LabError(f"{name} is not a weight-space edit target")
    if leaf not in WRITING_MODULES:
        raise LabError(f"{name} is not a writing matrix")
    return leaf


def select_columns(rows: Sequence[Sequence[float]], direction: Sequence[float], count: int) -> list[int]:
    """Indexes that write the direction most, highest squared dot first.

    An equal score keeps the lower index.
    """
    matrix = _matrix(rows)
    unit = _unit(direction, len(matrix))
    scores = []
    width = len(matrix[0])
    for column in range(width):
        dot = sum(unit[row] * matrix[row][column] for row in range(len(unit)))
        scores.append(dot * dot)
    if count < 1 or count > width:
        raise LabError(f"top {count} is outside 1..{width}")
    order = sorted(range(width), key=lambda index: (-scores[index], index))
    return order[:count]


def edit_matrix(
    rows: Sequence[Sequence[float]],
    op: str,
    direction: Sequence[float] | None = None,
    columns: Sequence[int] | None = None,
    top: int | None = None,
    scale: float = 1.0,
    factor: float | None = None,
) -> tuple[list[list[float]], dict]:
    """Return the edited rows and a manifest of what differs."""
    matrix = _matrix(rows)
    height = len(matrix)
    width = len(matrix[0])
    if columns is not None and top is not None:
        raise LabError("pass columns or top, not both")
    if op == "remove":
        if top is not None:
            chosen = select_columns(matrix, direction or [], top)
        else:
            chosen = _columns(columns, width)
        edited = _remove(matrix, direction or [], chosen, scale)
        applied = float(scale)
        applied_factor = None
    elif op in ("scale", "zero"):
        if op == "zero":
            applied_factor = 0.0
        else:
            if factor is None:
                raise LabError("scale needs a factor")
            applied_factor = float(factor)
            if not math.isfinite(applied_factor):
                raise LabError("factor must be finite")
        if top is not None:
            chosen = select_columns(matrix, direction or [], top)
        else:
            if columns is None:
                raise LabError("name the columns")
            chosen = _columns(columns, width)
        if len(chosen) == width:
            raise LabError("a constant multiple of a whole tensor is refused")
        edited = _scale(matrix, chosen, applied_factor)
        applied = None
    else:
        raise LabError("op must be remove, scale, or zero")
    report = _manifest(matrix, edited, op, chosen, applied, applied_factor)
    return edited, report


def apply_weight_edit(
    model,
    layer: int,
    module: str,
    op: str,
    direction: Sequence[float] | None = None,
    columns: Sequence[int] | None = None,
    top: int | None = None,
    scale: float = 1.0,
    factor: float | None = None,
) -> dict:
    """Edit one loaded writing matrix in place. The checkpoint file stays.

    The manifest counts entries that still differ after the value is cast
    back to the matrix dtype. A touch that rounds back is not counted.
    """
    from kilix_llm_lab.qwen import decoder_layers

    torch = _torch()
    layers = decoder_layers(model)
    index = int(layer)
    if not 0 <= index < len(layers):
        raise LabError(f"layer {index} is outside 0..{len(layers) - 1}")
    leaf = assert_editable(module if "." in module else f"mlp.{module}" if module == "down_proj" else f"self_attn.{module}")
    linear = _linear(layers[index], leaf)
    weight = linear.weight
    if weight.ndim != 2 or not weight.dtype.is_floating_point:
        raise LabError("weight-space edits need a dense floating-point matrix")
    name = f"layers.{index}.{_leaf_path(leaf)}"
    with torch.no_grad():
        before = weight.detach().float().cpu().tolist()
        edited, report = edit_matrix(
            before,
            op,
            direction=direction,
            columns=columns,
            top=top,
            scale=scale,
            factor=factor,
        )
        cast = torch.tensor(edited, dtype=torch.float32, device=weight.device).to(dtype=weight.dtype)
        original = weight.detach().clone()
        weight.copy_(cast)
        report = _manifest_from_tensors(original, weight.detach(), report)
    report["module"] = name
    report["layer"] = index
    return report


def _leaf_path(leaf: str) -> str:
    if leaf == "down_proj":
        return "mlp.down_proj"
    return "self_attn.o_proj"


def _linear(layer, leaf: str):
    try:
        if leaf == "down_proj":
            return layer.mlp.down_proj
        return layer.self_attn.o_proj
    except AttributeError as error:
        raise LabError(f"layer is missing {leaf}") from error


def _torch():
    try:
        import torch
    except ImportError as error:
        raise LabError("torch is not installed") from error
    return torch


def _matrix(rows: Sequence[Sequence[float]]) -> list[list[float]]:
    if not rows:
        raise LabError("matrix has no rows")
    matrix = []
    width = None
    for row in rows:
        values = [float(value) for value in row]
        if not values:
            raise LabError("matrix has an empty row")
        if any(not math.isfinite(value) for value in values):
            raise LabError("matrix entries must be finite")
        if width is None:
            width = len(values)
        elif len(values) != width:
            raise LabError("matrix rows must have equal length")
        matrix.append(values)
    return matrix


def _unit(direction: Sequence[float], height: int) -> list[float]:
    values = [float(value) for value in direction]
    if len(values) != height:
        raise LabError(f"direction width {len(values)} != out_features {height}")
    if any(not math.isfinite(value) for value in values):
        raise LabError("direction entries must be finite")
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        raise LabError("cannot edit along the zero vector")
    return [value / norm for value in values]


def _columns(columns: Sequence[int] | None, width: int) -> list[int]:
    if columns is None:
        return list(range(width))
    chosen = [int(column) for column in columns]
    if not chosen:
        raise LabError("name at least one column")
    if len(set(chosen)) != len(chosen):
        raise LabError("columns must be unique")
    for column in chosen:
        if not 0 <= column < width:
            raise LabError(f"column {column} is outside 0..{width - 1}")
    return chosen


def _remove(matrix: list[list[float]], direction: Sequence[float], columns: list[int], scale: float) -> list[list[float]]:
    if not math.isfinite(scale) or scale == 0.0:
        raise LabError("remove scale must be a non-zero finite number")
    unit = _unit(direction, len(matrix))
    edited = [list(row) for row in matrix]
    for column in columns:
        dot = sum(unit[row] * edited[row][column] for row in range(len(unit)))
        step = scale * dot
        for row in range(len(unit)):
            edited[row][column] -= unit[row] * step
    return edited


def _scale(matrix: list[list[float]], columns: list[int], factor: float) -> list[list[float]]:
    edited = [list(row) for row in matrix]
    for column in columns:
        for row in range(len(edited)):
            edited[row][column] *= factor
    return edited


def _manifest(before, after, op: str, columns: list[int], scale: float | None, factor: float | None) -> dict:
    height = len(before)
    width = len(before[0])
    changed_entries = 0
    changed_columns = 0
    changed_rows = 0
    numerator = 0.0
    denominator = 0.0
    row_hits = [False] * height
    for column in range(width):
        column_hit = False
        for row in range(height):
            old = before[row][column]
            new = after[row][column]
            denominator += old * old
            delta = new - old
            numerator += delta * delta
            if new != old:
                changed_entries += 1
                column_hit = True
                row_hits[row] = True
        if column_hit:
            changed_columns += 1
    changed_rows = sum(row_hits)
    if denominator == 0.0:
        relative = 0.0 if numerator == 0.0 else None
    else:
        relative = math.sqrt(numerator) / math.sqrt(denominator)
    return {
        "op": op,
        "module": None,
        "layer": None,
        "scale": scale,
        "factor": factor,
        "targets": list(columns),
        "shape": [height, width],
        "changed_entries": changed_entries,
        "changed_columns": changed_columns,
        "changed_rows": changed_rows,
        "fraction": changed_entries / (height * width),
        "relative_change": relative,
    }


def run_weight(args) -> int:
    """Print one JSON manifest. A matrix file does not load a checkpoint."""
    import json
    import sys

    if args.matrix and args.layer is not None:
        raise LabError("pass a matrix file or a layer, not both")
    if args.quant == "4bit":
        raise LabError("weight-space edits need a dense floating-point matrix")
    direction = _direction_arg(args.direction, args.direction_file)
    columns = _column_arg(args.columns)
    if args.matrix:
        payload = _read_json(args.matrix)
        rows = payload.get("rows") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            raise LabError("matrix file needs a rows array")
        edited, report = edit_matrix(
            rows,
            args.op,
            direction=direction,
            columns=columns,
            top=args.top,
            scale=args.scale,
            factor=args.factor,
        )
        if args.write_matrix:
            _write_json(args.write_matrix, {"rows": edited})
        json.dump(report, sys.stdout)
        sys.stdout.write("\n")
        return 0
    if args.layer is None:
        raise LabError("pass a matrix file or a layer")
    if args.prompt:
        from kilix_llm_lab.collect import generate
        from kilix_llm_lab.runner import load

        model, tokenizer = load(
            args.model,
            args.device,
            allow_other=args.allow_other,
            dtype_name=args.dtype,
            quant=args.quant,
        )
        before = generate(model, tokenizer, args.prompt, args.max_new_tokens)
        report = apply_weight_edit(
            model,
            args.layer,
            args.module,
            args.op,
            direction=direction,
            columns=columns,
            top=args.top,
            scale=args.scale,
            factor=args.factor,
        )
        after = generate(model, tokenizer, args.prompt, args.max_new_tokens)
        report["before"] = before
        report["after"] = after
        report["changed"] = before != after
    else:
        from kilix_llm_lab.runner import load

        model, _tokenizer = load(
            args.model,
            args.device,
            allow_other=args.allow_other,
            dtype_name=args.dtype,
            quant=args.quant,
        )
        report = apply_weight_edit(
            model,
            args.layer,
            args.module,
            args.op,
            direction=direction,
            columns=columns,
            top=args.top,
            scale=args.scale,
            factor=args.factor,
        )
    json.dump(report, sys.stdout)
    sys.stdout.write("\n")
    return 0


def _direction_arg(text: str | None, path: str | None) -> list[float] | None:
    if text and path:
        raise LabError("pass a direction or a direction file, not both")
    if text:
        try:
            return [float(part) for part in text.split(",") if part != ""]
        except ValueError as error:
            raise LabError("direction entries must be numbers") from error
    if not path:
        return None
    payload = _read_json(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("direction"), list):
        raise LabError("direction file needs a direction array")
    return [float(value) for value in payload["direction"]]


def _column_arg(text: str | None) -> list[int] | None:
    if text is None:
        return None
    parts = [part for part in text.split(",") if part != ""]
    if not parts:
        raise LabError("name at least one column")
    try:
        return [int(part) for part in parts]
    except ValueError as error:
        raise LabError("columns must be integers") from error


def _read_json(path: str):
    import json

    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except OSError as error:
        raise LabError(f"cannot read {path}: {error.strerror}") from error
    except json.JSONDecodeError as error:
        raise LabError(f"{path} is not JSON") from error


def _write_json(path: str, payload: dict) -> None:
    import json

    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
            handle.write("\n")
    except OSError as error:
        raise LabError(f"cannot write {path}: {error.strerror}") from error


def _manifest_from_tensors(original, updated, report: dict) -> dict:
    """Recount after the dtype cast. original and updated share a dtype."""
    height, width = report["shape"]
    changed_entries = 0
    changed_columns = 0
    changed_rows = 0
    row_hits = [False] * height
    for column in range(width):
        column_hit = False
        for row in range(height):
            if not bool(original[row, column] == updated[row, column]):
                changed_entries += 1
                column_hit = True
                row_hits[row] = True
        if column_hit:
            changed_columns += 1
    changed_rows = sum(row_hits)
    delta = updated.float() - original.float()
    numerator = float(delta.pow(2).sum().sqrt())
    denominator = float(original.float().pow(2).sum().sqrt())
    if denominator == 0.0:
        relative = 0.0 if numerator == 0.0 else None
    else:
        relative = numerator / denominator
    report = dict(report)
    report["changed_entries"] = changed_entries
    report["changed_columns"] = changed_columns
    report["changed_rows"] = changed_rows
    report["fraction"] = changed_entries / (height * width)
    report["relative_change"] = relative
    return report
