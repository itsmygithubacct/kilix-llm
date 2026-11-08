"""Exact column edits on a copy of a GGUF model. Needs numpy and the vendored gguf-py.

A writing matrix in GGUF (blk.N.ffn_down, blk.N.attn_output, blk.N.ssm_out on
qwen35) stores one row per output feature, so a Hugging Face column (one
input: an MLP neuron, or one coordinate of a head's slice) is the same
offset in every row. In Q8_0 a row is blocks of 32 values, each a float16
scale and 32 int8 quants:

  zero   writes quant 0, which dequantizes to exactly 0.0.
  scale  multiplies a block's float16 scale when the edit covers the whole
         block (exact up to float16 rounding of that scale). On part of a
         block it rounds the chosen int8 quants and keeps the scale, so the
         other values of that block stay bit-identical.

F32 and F16 tensors are edited value by value. Every other byte of the file
is copied unchanged, the source is never opened for writing, and the copy is
read back to check that edited columns hold the intended values and the
rest of each touched tensor is bit-identical.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from kilix_llm_lab import LabError
from kilix_llm_lab.output_row import _import_gguf, write_patched_copy

Q8_BLOCK = 32
Q8_BYTES = 34
OPS = ("zero", "scale")


def _numpy():
    try:
        import numpy
    except ImportError as error:
        raise LabError("GGUF column edits need numpy") from error
    return numpy


def normalize_edits(edits: Sequence[dict]) -> list[dict]:
    """Check each edit: tensor name, op, non-empty unique columns, factor for scale."""
    if not edits:
        raise LabError("name at least one column edit")
    out = []
    for edit in edits:
        name = edit.get("tensor")
        op = edit.get("op")
        columns = edit.get("columns")
        if not isinstance(name, str) or not name:
            raise LabError("each edit needs a tensor name")
        if op not in OPS:
            raise LabError(f"op must be one of {', '.join(OPS)}")
        if not isinstance(columns, (list, tuple)) or not columns:
            raise LabError(f"{name}: columns must be a non-empty list")
        cols = []
        for column in columns:
            if isinstance(column, bool) or not isinstance(column, int) or column < 0:
                raise LabError(f"{name}: columns must be non-negative integers")
            cols.append(column)
        if len(set(cols)) != len(cols):
            raise LabError(f"{name}: a column is listed twice")
        factor = edit.get("factor")
        if op == "scale":
            if isinstance(factor, bool) or not isinstance(factor, (int, float)) or factor != factor \
                    or factor in (float("inf"), float("-inf")):
                raise LabError(f"{name}: scale needs a finite factor")
            factor = float(factor)
        else:
            factor = 0.0
        out.append({"tensor": name, "op": op, "columns": sorted(cols), "factor": factor})
    seen: dict[str, set] = {}
    for edit in out:
        taken = seen.setdefault(edit["tensor"], set())
        if taken & set(edit["columns"]):
            raise LabError(f"{edit['tensor']}: two edits touch the same column")
        taken.update(edit["columns"])
    return out


def head_columns(head: int, head_dim: int) -> list[int]:
    """Columns of one head's slice in a mixer writing matrix."""
    if head < 0 or head_dim < 1:
        raise LabError("head must be non-negative and head_dim positive")
    return list(range(head * head_dim, (head + 1) * head_dim))


def _tensor(reader, name):
    for item in reader.tensors:
        if item.name == name:
            return item
    raise LabError(f"{name} is not in this file")


def _q8_rows(numpy, data, columns, factor):
    """Edited copies of the touched blocks of every row. Returns {block: (rows, 34) uint8}."""
    by_block: dict[int, list[int]] = {}
    for column in columns:
        by_block.setdefault(column // Q8_BLOCK, []).append(column % Q8_BLOCK)
    edited = {}
    for block, within in by_block.items():
        raw = numpy.array(data[:, block * Q8_BYTES:(block + 1) * Q8_BYTES], copy=True)
        if factor == 0.0:
            raw[:, 2 + numpy.asarray(within)] = 0
        elif len(within) == Q8_BLOCK:
            scale = raw[:, :2].copy().view(numpy.float16).astype(numpy.float32) * factor
            raw[:, :2] = scale.astype(numpy.float16).view(numpy.uint8).reshape(-1, 2)
        else:
            # Keep the block scale so the other 31 values stay bit-identical; round the chosen quants.
            index = 2 + numpy.asarray(within)
            quants = raw[:, index].view(numpy.int8).astype(numpy.float64) * factor
            if numpy.any(numpy.abs(quants) > 127.5):
                raise LabError("a partial-block scale above 1 would clip; scale whole 32-column blocks")
            raw[:, index] = numpy.rint(quants).astype(numpy.int8).view(numpy.uint8)
        edited[block] = raw
    return edited


def plan_windows(source: str, edits: Sequence[dict]) -> tuple[list[tuple[int, bytes]], list[dict]]:
    """Byte windows that carry every edit, and one expectation per edit for the read-back."""
    numpy = _numpy()
    gguf = _import_gguf()
    edits = normalize_edits(edits)
    reader = gguf.GGUFReader(source, "r")
    windows: dict[int, bytes] = {}
    checks = []
    merged: dict[str, dict] = {}
    for edit in edits:
        slot = merged.setdefault(edit["tensor"], {"blocks": {}, "values": None})
        tensor = _tensor(reader, edit["tensor"])
        if len(tensor.shape) != 2:
            raise LabError(f"{edit['tensor']} is not a matrix")
        n_in, n_out = int(tensor.shape[0]), int(tensor.shape[1])
        if edit["columns"][-1] >= n_in:
            raise LabError(f"{edit['tensor']}: column {edit['columns'][-1]} is outside 0..{n_in - 1}")
        qtype = tensor.tensor_type
        base = int(tensor.data_offset)
        if qtype == gguf.GGMLQuantizationType.Q8_0:
            data = tensor.data
            row_bytes = int(data.shape[1])
            if slot["blocks"]:
                current = numpy.array(data, copy=True)
                for block, raw in slot["blocks"].items():
                    current[:, block * Q8_BYTES:(block + 1) * Q8_BYTES] = raw
                data = current
            for block, raw in _q8_rows(numpy, data, edit["columns"], edit["factor"]).items():
                slot["blocks"][block] = raw
            for block, raw in slot["blocks"].items():
                for row in range(n_out):
                    windows[base + row * row_bytes + block * Q8_BYTES] = raw[row].tobytes()
        elif qtype in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
            if slot["values"] is None:
                slot["values"] = numpy.array(tensor.data, copy=True).reshape(n_out, n_in)
            values = slot["values"]
            values[:, edit["columns"]] = values[:, edit["columns"]] * values.dtype.type(edit["factor"])
            item = values.dtype.itemsize
            for row in range(n_out):
                windows[base + row * n_in * item] = values[row].tobytes()
        else:
            raise LabError(f"{edit['tensor']}: {qtype.name} is not supported (Q8_0, F32, F16)")
        before = gguf.dequantize(numpy.asarray(tensor.data), qtype).reshape(n_out, n_in)
        checks.append({**edit, "expected": (before[:, edit["columns"]] * edit["factor"]).astype(numpy.float32)})
    del reader
    return sorted(windows.items()), checks


def edit_gguf_columns(source: str, dest: str, edits: Sequence[dict], *,
                      expected_sha256: str | None = None, atol: float = 5e-3) -> dict:
    """Write dest as source with the column edits, then verify the copy. Returns a manifest."""
    numpy = _numpy()
    gguf = _import_gguf()
    windows, checks = plan_windows(source, edits)
    written = write_patched_copy(source, dest, windows, expected_sha256=expected_sha256)
    src = gguf.GGUFReader(source, "r")
    out = gguf.GGUFReader(dest, "r")
    report = []
    for check in checks:
        a, b = _tensor(src, check["tensor"]), _tensor(out, check["tensor"])
        n_in, n_out = int(a.shape[0]), int(a.shape[1])
        old = gguf.dequantize(numpy.asarray(a.data), a.tensor_type).reshape(n_out, n_in)
        new = gguf.dequantize(numpy.asarray(b.data), b.tensor_type).reshape(n_out, n_in)
        cols = check["columns"]
        error = float(numpy.max(numpy.abs(new[:, cols] - check["expected"])))
        scale = float(numpy.max(numpy.abs(old[:, cols]))) or 1.0
        others = numpy.ones(n_in, dtype=bool)
        for other in checks:
            if other["tensor"] == check["tensor"]:
                others[other["columns"]] = False
        untouched_equal = bool(numpy.array_equal(new[:, others], old[:, others]))
        report.append({"tensor": check["tensor"], "op": check["op"], "factor": check["factor"],
                       "columns": len(cols), "max_abs_error": error,
                       "max_relative_error": error / scale, "untouched_columns_identical": untouched_equal})
        if check["op"] == "zero" and error != 0.0:
            raise LabError(f"{check['tensor']}: zeroed columns read back as {error}")
        if error / scale > atol or not untouched_equal:
            raise LabError(f"{check['tensor']}: read-back check failed")
    del src, out
    return {**written, "windows": len(windows), "edits": report}


def main(argv: Sequence[str] | None = None) -> int:
    """python -m kilix_llm_lab.gguf_columns SOURCE DEST EDITS.json [--sha256 HEX]"""
    import argparse

    parser = argparse.ArgumentParser(prog="kilix_llm_lab.gguf_columns",
                                     description="Copy a GGUF with exact column edits.")
    parser.add_argument("source")
    parser.add_argument("dest")
    parser.add_argument("edits", help='JSON list of {"tensor", "op": "zero"|"scale", "columns", "factor"?}')
    parser.add_argument("--sha256", default=None, help="refuse a source with another digest")
    args = parser.parse_args(argv)
    edits = json.load(open(args.edits, encoding="utf-8"))
    manifest = edit_gguf_columns(args.source, args.dest, edits, expected_sha256=args.sha256)
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
