"""Replace whole tensors in a copy of a GGUF model. Needs numpy and the vendored gguf-py.

An edit made on Hugging Face weights (a direction removed from some
down_proj columns, a published diff) changes values that a GGUF stores
quantized. This command quantizes each new tensor with the vendored gguf-py
and writes it over the old one in a copy of the parent file.

A replacement can name the unedited tensor it was made from (`base`). The
base is then quantized the same way first, and the command refuses unless
that reproduces the parent's bytes exactly. When it does, the written tensor
is what converting the edited checkpoint with that quantizer would give, so
the copy differs from the parent only where the edit does.

Q8_0, F16 and F32 tensors are supported. Every other byte of the file is
copied unchanged, the source is never opened for writing, and the copy is
read back to check each replaced tensor's bytes.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from kilix_llm_lab import LabError
from kilix_llm_lab.output_row import _import_gguf, write_patched_copy


def _numpy():
    try:
        import numpy
    except ImportError as error:
        raise LabError("GGUF tensor replacement needs numpy") from error
    return numpy


def _tensor(reader, name):
    for item in reader.tensors:
        if item.name == name:
            return item
    raise LabError(f"{name} is not in this file")


def normalize_replacements(replacements: Sequence[dict]) -> list[dict]:
    """Each replacement: a tensor name and a .npy path for `values`, optionally `base`."""
    if not replacements:
        raise LabError("name at least one tensor to replace")
    out, seen = [], set()
    for item in replacements:
        name = item.get("tensor")
        values = item.get("values")
        base = item.get("base")
        if not isinstance(name, str) or not name:
            raise LabError("each replacement needs a tensor name")
        if name in seen:
            raise LabError(f"{name} is replaced twice")
        seen.add(name)
        if not isinstance(values, str) or not values:
            raise LabError(f"{name}: values must be a .npy path")
        if base is not None and (not isinstance(base, str) or not base):
            raise LabError(f"{name}: base must be a .npy path")
        out.append({"tensor": name, "values": values, "base": base})
    return out


def _load(numpy, path: str, shape: tuple[int, int], name: str):
    try:
        array = numpy.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise LabError(f"{name}: cannot read {path}: {error}") from error
    if tuple(array.shape) != shape:
        raise LabError(f"{name}: {path} has shape {tuple(array.shape)}, the file needs {shape}")
    array = array.astype(numpy.float32)
    if not numpy.all(numpy.isfinite(array)):
        raise LabError(f"{name}: {path} has non-finite values")
    return array


def _encode(numpy, gguf, values, qtype):
    if qtype == gguf.GGMLQuantizationType.F32:
        return values.astype(numpy.float32)
    if qtype == gguf.GGMLQuantizationType.F16:
        return values.astype(numpy.float16)
    if qtype == gguf.GGMLQuantizationType.Q8_0:
        return gguf.quantize(values, qtype)
    raise LabError(f"{qtype.name} is not supported (Q8_0, F16, F32)")


def plan_replacements(source: str, replacements: Sequence[dict]):
    """Byte windows for every replaced tensor, and one expectation per tensor for the read-back."""
    numpy = _numpy()
    gguf = _import_gguf()
    items = normalize_replacements(replacements)
    reader = gguf.GGUFReader(source, "r")
    windows, checks = [], []
    for item in items:
        tensor = _tensor(reader, item["tensor"])
        if len(tensor.shape) != 2:
            raise LabError(f"{item['tensor']} is not a matrix")
        n_in, n_out = int(tensor.shape[0]), int(tensor.shape[1])
        qtype = tensor.tensor_type
        current = numpy.asarray(tensor.data).view(numpy.uint8).reshape(-1)
        round_trip = None
        if item["base"] is not None:
            base = _load(numpy, item["base"], (n_out, n_in), item["tensor"])
            again = numpy.asarray(_encode(numpy, gguf, base, qtype)).view(numpy.uint8).reshape(-1)
            round_trip = bool(again.size == current.size and numpy.array_equal(again, current))
            if not round_trip:
                raise LabError(f"{item['tensor']}: the base does not quantize to the parent's bytes")
        values = _load(numpy, item["values"], (n_out, n_in), item["tensor"])
        encoded = numpy.ascontiguousarray(_encode(numpy, gguf, values, qtype)).view(numpy.uint8).reshape(-1)
        if encoded.size != current.size:
            raise LabError(f"{item['tensor']}: encoded size {encoded.size} != {current.size}")
        windows.append((int(tensor.data_offset), encoded.tobytes()))
        before = gguf.dequantize(numpy.asarray(tensor.data), qtype).reshape(n_out, n_in)
        after = gguf.dequantize(encoded.view(numpy.asarray(tensor.data).dtype).reshape(tensor.data.shape),
                                qtype).reshape(n_out, n_in)
        changed = numpy.any(after != before, axis=0)
        checks.append({
            "tensor": item["tensor"],
            "type": qtype.name,
            "bytes": encoded.tobytes(),
            "base_round_trip": round_trip,
            "changed_columns": int(changed.sum()),
            "changed_bytes": int((encoded != current).sum()),
            "max_abs_quant_error": float(numpy.max(numpy.abs(after - values))),
            "relative_change": float(numpy.linalg.norm(after - before) / (numpy.linalg.norm(before) or 1.0)),
        })
    del reader
    return sorted(windows), checks


def replace_gguf_tensors(source: str, dest: str, replacements: Sequence[dict], *,
                         expected_sha256: str | None = None) -> dict:
    """Write dest as source with the tensors replaced, then verify the copy. Returns a manifest."""
    numpy = _numpy()
    gguf = _import_gguf()
    windows, checks = plan_replacements(source, replacements)
    written = write_patched_copy(source, dest, windows, expected_sha256=expected_sha256)
    out = gguf.GGUFReader(dest, "r")
    report = []
    for check in checks:
        got = numpy.asarray(_tensor(out, check["tensor"]).data).view(numpy.uint8).reshape(-1).tobytes()
        if got != check["bytes"]:
            raise LabError(f"{check['tensor']}: read-back bytes differ from the encoded tensor")
        report.append({key: value for key, value in check.items() if key != "bytes"})
    del out
    return {**written, "tensors": report}


def run_gguf_tensors(args) -> int:
    with open(args.replace, encoding="utf-8") as handle:
        replacements = json.load(handle)
    manifest = replace_gguf_tensors(args.source, args.output, replacements,
                                    expected_sha256=args.expected_sha256)
    print(json.dumps(manifest, indent=1))
    return 0
