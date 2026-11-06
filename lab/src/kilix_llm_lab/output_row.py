"""Regularized squared-hinge edit of one output row.

The column editor refuses lm_head. This is the separate Ask edit: one row of
output.weight, from frozen activations, with no backward pass through the
network. The loss is the mean weighted squared hinge plus an L2 penalty on
the row change. Feature columns are scaled by their root-mean-square, with a
floor of 0.01, and the penalty is on the unscaled change.

Fresh, operational, and held-out Ask ids are sealed. A solve that names one
does not run. A parent file is never opened for writing. The published .8
bytes can be copied to a new path and then patched. They are not replaced.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
from pathlib import Path

from kilix_llm_lab import LabError

SCHEMA = "kilix.llm.output-row/v1"
SCALE_FLOOR = 0.01
MAX_CAPTURE_BYTES = 512 * 1024 * 1024
MAX_STATES = 200_000
MAX_WIDTH = 8192
MAX_ITER = 5000
PUBLISHED_MODEL_SHA256 = "7815032c4bdd44fbde002ec734ef3dd8eb14a8a44214b395738dae6e55ce897d"
SEALED_PREFIXES = ("copy-fresh-", "copy-operational-", "copy-heldout-")
_CAPTURE_KEYS = {
    "schema",
    "states",
    "sign",
    "bound",
    "weights",
    "case_ids",
    "old_row",
    "regularization",
    "row",
    "parent_version",
}


def sealed_reason(case_id: str) -> str | None:
    """Why this id cannot be fit, or None when it is development data."""
    for prefix in SEALED_PREFIXES:
        if case_id.startswith(prefix):
            return prefix
    return None


def assert_fittable(case_ids: list[str]) -> None:
    """Refuse an empty id list and every sealed Ask confirmation id."""
    if not case_ids:
        raise LabError("name a case id for every state")
    bad: list[str] = []
    for case_id in case_ids:
        if not isinstance(case_id, str) or case_id.strip() != case_id or not case_id:
            raise LabError("case ids must be non-empty strings")
        if any(ord(char) < 32 for char in case_id):
            raise LabError("case ids must be single-line text")
        if sealed_reason(case_id) is not None and case_id not in bad:
            bad.append(case_id)
    if bad:
        shown = ", ".join(bad[:8])
        extra = f" and {len(bad) - 8} more" if len(bad) > 8 else ""
        raise LabError(f"sealed Ask cases cannot be fit: {shown}{extra}")


def solve_output_row(
    states,
    sign,
    bound,
    *,
    weights=None,
    regularization: float = 1e-5,
    max_iter: int = 400,
    gtol: float = 1e-7,
    case_ids: list[str] | None = None,
    old_row=None,
    guard_limit: float = 0.1,
) -> dict:
    """Return the float row change and a manifest. Do not touch a file."""
    if case_ids is not None:
        assert_fittable(list(case_ids))
    problem = _prepare(
        states,
        sign,
        bound,
        weights,
        regularization,
        max_iter,
        gtol,
        guard_limit,
        case_ids,
        old_row,
    )
    start = problem.zeros()
    solved = _lbfgs(problem.loss_grad, start, max_iter=max_iter, gtol=gtol)
    delta = problem.delta(solved["point"])
    residual = problem.residual(solved["point"])
    positive = [residual[index] for index in problem.positive]
    guards = [residual[index] for index in problem.guards]
    report = {
        "schema": SCHEMA,
        "method": "regularized-squared-hinge",
        "fit": "frozen-activations",
        "network_backward": False,
        "success": solved["success"],
        "message": solved["message"],
        "iterations": solved["iterations"],
        "loss": solved["loss"],
        "gradient_norm": solved["gradient_norm"],
        "regularization": problem.regularization,
        "states": problem.n,
        "width": problem.width,
        "positive_states": len(problem.positive),
        "guard_states": len(problem.guards),
        "positive_residual_max": max(positive) if positive else 0.0,
        "guard_violations_over_limit": sum(value > guard_limit for value in guards),
        "guard_limit": guard_limit,
        "correction_l2": _l2(delta),
        "correction_sha256": _vector_sha256(delta),
        "cases_checked": len(case_ids) if case_ids is not None else 0,
    }
    if problem.old_row is not None:
        report["float_relative_l2"] = _relative_l2(delta, problem.old_row)
    report["delta"] = delta
    return report


def load_capture(path: str | Path) -> dict:
    """Read a JSON or npz capture. Refuse a symlink and a file over 512 MiB."""
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise LabError("capture must be a regular file")
    size = source.stat().st_size
    if size > MAX_CAPTURE_BYTES:
        raise LabError("capture is larger than 512 MiB")
    suffix = source.suffix.lower()
    if suffix == ".json":
        return _load_json(source)
    if suffix == ".npz":
        return _load_npz(source)
    raise LabError("capture must be .json or .npz")


def solve_capture(payload: dict, *, max_iter: int | None = None, regularization: float | None = None) -> dict:
    """Solve one capture object. case_ids are required."""
    if not isinstance(payload, dict):
        raise LabError("capture must be an object")
    extra = set(payload) - _CAPTURE_KEYS
    if extra:
        raise LabError("capture has an unknown field")
    if payload.get("schema") != SCHEMA:
        raise LabError("capture schema is not kilix.llm.output-row/v1")
    case_ids = payload.get("case_ids")
    if not isinstance(case_ids, list):
        raise LabError("capture needs a case_ids array")
    reg = payload.get("regularization", 1e-5)
    if regularization is not None:
        reg = regularization
    limit = 400 if max_iter is None else max_iter
    report = solve_output_row(
        payload.get("states"),
        payload.get("sign"),
        payload.get("bound"),
        weights=payload.get("weights"),
        regularization=reg,
        max_iter=limit,
        case_ids=case_ids,
        old_row=payload.get("old_row"),
    )
    if "row" in payload and payload["row"] is not None:
        report["row"] = _whole_number(payload["row"], "row")
    if "parent_version" in payload and payload["parent_version"] is not None:
        version = payload["parent_version"]
        if not isinstance(version, str) or not version:
            raise LabError("parent_version must be a non-empty string")
        report["parent_version"] = version
    return report


def patch_bytes(source: bytes, windows: list[tuple[int, bytes]]) -> bytes:
    """Replace non-overlapping windows. Leave every other byte alone."""
    if not isinstance(source, (bytes, bytearray)):
        raise LabError("source must be bytes")
    if not windows:
        raise LabError("name at least one row window")
    spans: list[tuple[int, int, bytes]] = []
    for offset, blob in windows:
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise LabError("window offset must be a non-negative integer")
        if not isinstance(blob, (bytes, bytearray)) or len(blob) == 0:
            raise LabError("window must be non-empty bytes")
        end = offset + len(blob)
        if end > len(source):
            raise LabError("window runs past the end of the file")
        spans.append((offset, end, bytes(blob)))
    spans.sort()
    for (_, end, _), (start, _, _) in zip(spans, spans[1:]):
        if start < end:
            raise LabError("row windows overlap")
    edited = bytearray(source)
    for start, end, blob in spans:
        edited[start:end] = blob
    return bytes(edited)


def write_patched_copy(
    source,
    dest,
    windows: list[tuple[int, bytes]],
    *,
    expected_sha256: str | None = None,
) -> dict:
    """Copy source to a new path and patch that copy. Never open source for write."""
    source_path = Path(source)
    dest_path = Path(dest)
    if source_path.is_symlink() or not source_path.is_file():
        raise LabError("source must be a regular file")
    if dest_path.exists() or dest_path.is_symlink():
        raise LabError("output already exists")
    if not dest_path.parent.is_dir():
        raise LabError("output directory does not exist")
    if source_path.resolve() == dest_path.resolve():
        raise LabError("refusing to patch the source file in place")
    data = source_path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256.lower():
        raise LabError("source sha256 does not match")
    edited = patch_bytes(data, windows)
    tmp = dest_path.with_name(f".{dest_path.name}.partial")
    if tmp.exists() or tmp.is_symlink():
        raise LabError("a partial output is already present")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o644)
    try:
        view = memoryview(edited)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        os.fsync(fd)
    except Exception:
        os.close(fd)
        tmp.unlink(missing_ok=True)
        raise
    os.close(fd)
    os.replace(tmp, dest_path)
    written = dest_path.read_bytes()
    if written != edited:
        raise LabError("written output does not match the patched bytes")
    return {
        "source_sha256": digest,
        "output_sha256": hashlib.sha256(written).hexdigest(),
        "output_bytes": len(written),
        "source_opened_for_write": False,
    }


def requantize_row(values, delta, qtype_name: str = "Q8_0") -> dict:
    """Add a float correction to one row and quantize it back. Needs numpy."""
    numpy = _numpy()
    gguf = _import_gguf()
    qtype = getattr(gguf.GGMLQuantizationType, qtype_name, None)
    if qtype is None:
        raise LabError(f"{qtype_name} is not a known quantization")
    old = numpy.asarray(values, dtype=numpy.float64).reshape(-1)
    change = numpy.asarray(list(delta), dtype=numpy.float64).reshape(-1)
    if old.shape != change.shape:
        raise LabError("correction width does not match the output row")
    if not numpy.isfinite(old).all() or not numpy.isfinite(change).all():
        raise LabError("row values must be finite")
    updated = (old + change).astype(numpy.float32)
    try:
        raw = gguf.quantize(updated.reshape(1, -1), qtype)
        restored = gguf.dequantize(raw, qtype).reshape(-1).astype(numpy.float64)
    except Exception as error:
        raise LabError(f"row quantization failed: {error}") from error
    blob = raw.tobytes()
    quantized_delta = restored - old
    return {
        "bytes": blob,
        "restored": restored.tolist(),
        "quantized_relative_l2": _relative_l2(quantized_delta.tolist(), old.tolist()),
    }


def patch_gguf_output_row(source, dest, row: int, delta, *, expected_sha256: str) -> dict:
    """Copy a GGUF and replace one output.weight row. The source stays read-only."""
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise LabError("pass the 64-character source sha256")
    row_index = _whole_number(row, "row")
    window, quant = _output_window(source, row_index, delta)
    written = write_patched_copy(source, dest, [window], expected_sha256=expected_sha256)
    written.update(quant)
    written["row"] = row_index
    return written


def run_output_row(args) -> int:
    """Print one JSON manifest. Write a weight file only after a converged solve."""
    import sys

    if bool(args.parent_gguf) != bool(args.output):
        raise LabError("pass both --parent-gguf and --output")
    if args.parent_gguf and not args.expected_sha256:
        raise LabError("pass --expected-sha256")
    payload = load_capture(args.capture)
    report = solve_capture(
        payload,
        max_iter=args.max_iter,
        regularization=args.regularization,
    )
    if args.row is not None:
        named = _whole_number(args.row, "row")
        if "row" in report and report["row"] != named:
            raise LabError("capture row and --row disagree")
        report["row"] = named
    delta = report.pop("delta")
    if report["success"] and args.write_correction:
        _write_json(
            args.write_correction,
            {
                "schema": SCHEMA,
                "delta": delta,
                "correction_sha256": report["correction_sha256"],
            },
        )
    if args.parent_gguf and report["success"]:
        if "row" not in report:
            raise LabError("pass --row")
        patched = patch_gguf_output_row(
            args.parent_gguf,
            args.output,
            report["row"],
            delta,
            expected_sha256=args.expected_sha256,
        )
        report["output_sha256"] = patched["output_sha256"]
        report["source_sha256"] = patched["source_sha256"]
        report["quantized_relative_l2"] = patched["quantized_relative_l2"]
    elif args.parent_gguf:
        report["write_refused"] = "solve did not converge"
    json.dump(report, sys.stdout)
    sys.stdout.write("\n")
    return 0 if report["success"] else 1


class _Problem:
    def __init__(self, states, scale, sign, bound, weights, regularization, positive, guards, old_row):
        self.states = states
        self.scale = scale
        self.sign = sign
        self.bound = bound
        self.weights = weights
        self.regularization = regularization
        self.positive = positive
        self.guards = guards
        self.old_row = old_row
        self.n = len(sign)
        self.width = len(scale)

    def zeros(self):
        raise NotImplementedError

    def delta(self, point):
        raise NotImplementedError

    def loss_grad(self, point):
        raise NotImplementedError

    def residual(self, point):
        raise NotImplementedError


class _ListProblem(_Problem):
    def zeros(self):
        return [0.0] * self.width

    def delta(self, point):
        return [point[index] / self.scale[index] for index in range(self.width)]

    def residual(self, point):
        shift = self._shift(point)
        return [
            max(self.bound[index] - self.sign[index] * shift[index], 0.0)
            for index in range(self.n)
        ]

    def loss_grad(self, point):
        residual = self.residual(point)
        hinge = sum(
            self.weights[index] * residual[index] * residual[index] for index in range(self.n)
        ) / self.n
        reg = self.regularization * sum(
            (point[index] / self.scale[index]) ** 2 for index in range(self.width)
        )
        coeff = [
            self.weights[index] * residual[index] * self.sign[index] for index in range(self.n)
        ]
        grad = []
        for column in range(self.width):
            inv = 1.0 / self.scale[column]
            acc = sum(self.states[row][column] * inv * coeff[row] for row in range(self.n))
            reg_term = 2.0 * self.regularization * point[column] * inv * inv
            grad.append(-2.0 * acc / self.n + reg_term)
        return hinge + reg, grad

    def _shift(self, point):
        shift = []
        for row in range(self.n):
            total = 0.0
            states = self.states[row]
            for column in range(self.width):
                total += states[column] * point[column] / self.scale[column]
            shift.append(total)
        return shift


class _NumpyProblem(_Problem):
    def __init__(self, numpy, *args):
        super().__init__(*args)
        self.numpy = numpy

    def zeros(self):
        return self.numpy.zeros(self.width, dtype=self.numpy.float64)

    def delta(self, point):
        return (point / self.scale).tolist()

    def residual(self, point):
        shift = self.states @ (point / self.scale)
        return self.numpy.maximum(self.bound - self.sign * shift, 0.0)

    def loss_grad(self, point):
        residual = self.residual(point)
        hinge = float(self.numpy.dot(self.weights, residual * residual) / self.n)
        delta = point / self.scale
        reg = self.regularization * float(self.numpy.dot(delta, delta))
        coeff = self.weights * residual * self.sign
        acc = self.states.T @ coeff
        grad = -2.0 * (acc / self.scale) / self.n + 2.0 * self.regularization * point / (self.scale * self.scale)
        return hinge + reg, grad


def _prepare(states, sign, bound, weights, regularization, max_iter, gtol, guard_limit, case_ids, old_row):
    rows, kind = _rows(states)
    n = len(rows)
    width = len(rows[0])
    if n > MAX_STATES or width > MAX_WIDTH:
        raise LabError("capture is larger than the output-row limit")
    signs = _vector(sign, n, "sign")
    bounds = _vector(bound, n, "bound")
    if any(value not in (1.0, -1.0) for value in signs):
        raise LabError("sign must be 1 or -1")
    if weights is None:
        used = [1.0] * n
    else:
        used = _vector(weights, n, "weights")
        if any(value <= 0.0 for value in used):
            raise LabError("weights must be positive")
    reg = _finite(regularization, "regularization")
    if reg < 0.0:
        raise LabError("regularization must be non-negative")
    steps = _whole_number(max_iter, "max_iter")
    if not 1 <= steps <= MAX_ITER:
        raise LabError(f"max_iter must be from 1 to {MAX_ITER}")
    tolerance = _finite(gtol, "gtol")
    if tolerance <= 0.0:
        raise LabError("gtol must be positive")
    limit = _finite(guard_limit, "guard_limit")
    if limit < 0.0:
        raise LabError("guard_limit must be non-negative")
    if case_ids is not None and len(case_ids) != n:
        raise LabError("case_ids must have one entry per state")
    previous = None
    if old_row is not None:
        previous = _vector(old_row, width, "old_row")
    scale = _scale(rows)
    positive = [index for index, value in enumerate(signs) if value > 0.0]
    guards = [index for index, value in enumerate(signs) if value < 0.0]
    if kind == "numpy":
        numpy = _numpy()
        return _NumpyProblem(
            numpy,
            numpy.asarray(rows, dtype=numpy.float64),
            numpy.asarray(scale, dtype=numpy.float64),
            numpy.asarray(signs, dtype=numpy.float64),
            numpy.asarray(bounds, dtype=numpy.float64),
            numpy.asarray(used, dtype=numpy.float64),
            reg,
            positive,
            guards,
            previous,
        )
    return _ListProblem(rows, scale, signs, bounds, used, reg, positive, guards, previous)


def _lbfgs(objective, start, *, max_iter: int, gtol: float, memory: int = 30):
    point = _copy(start)
    loss, grad = objective(point)
    _check_finite(loss, grad)
    best_point, best_loss = _copy(point), loss
    steps_s: list = []
    steps_y: list = []
    steps_rho: list[float] = []
    for iteration in range(max_iter):
        gnorm = _norm(grad)
        if gnorm <= gtol:
            return _solved(point, loss, gnorm, iteration, True, "gradient tolerance")
        direction = _two_loop(grad, steps_s, steps_y, steps_rho)
        if _dot(direction, grad) >= 0.0:
            direction = _scale_vector(-1.0, grad)
        trial = _armijo(objective, point, loss, grad, direction)
        if trial is None:
            return _solved(best_point, best_loss, _norm(grad), iteration, False, "line search failed")
        new_point, new_loss, new_grad, _alpha = trial
        if new_loss < best_loss:
            best_point, best_loss = _copy(new_point), new_loss
        step_s = _axpy(-1.0, point, new_point)
        step_y = _axpy(-1.0, grad, new_grad)
        ys = _dot(step_y, step_s)
        if ys > 1e-12 * _norm(step_y) * _norm(step_s):
            steps_s.append(step_s)
            steps_y.append(step_y)
            steps_rho.append(1.0 / ys)
            if len(steps_s) > memory:
                steps_s.pop(0)
                steps_y.pop(0)
                steps_rho.pop(0)
        point, loss, grad = new_point, new_loss, new_grad
    return _solved(best_point, best_loss, _norm(grad), max_iter, False, "iteration limit")


def _solved(point, loss, gnorm, iterations, success, message):
    return {
        "point": point,
        "loss": loss,
        "gradient_norm": gnorm,
        "iterations": iterations,
        "success": success,
        "message": message,
    }


def _armijo(objective, point, loss, grad, direction):
    slope = _dot(grad, direction)
    if slope >= 0.0:
        return None
    alpha = 1.0
    for _ in range(40):
        trial = _axpy(alpha, direction, point)
        trial_loss, trial_grad = objective(trial)
        if math.isfinite(trial_loss) and _finite_vector(trial_grad) and trial_loss <= loss + 1e-4 * alpha * slope:
            return trial, trial_loss, trial_grad, alpha
        alpha *= 0.5
    return None


def _two_loop(grad, steps_s, steps_y, steps_rho):
    q = _copy(grad)
    alphas = []
    for step_s, step_y, rho in zip(reversed(steps_s), reversed(steps_y), reversed(steps_rho)):
        alpha = rho * _dot(step_s, q)
        alphas.append(alpha)
        q = _axpy(-alpha, step_y, q)
    if steps_s:
        ys = _dot(steps_y[-1], steps_s[-1])
        yy = _dot(steps_y[-1], steps_y[-1])
        gamma = ys / yy if yy > 0.0 else 1.0
        result = _scale_vector(gamma, q)
    else:
        result = q
    for step_s, step_y, rho, alpha in zip(steps_s, steps_y, steps_rho, reversed(alphas)):
        beta = rho * _dot(step_y, result)
        result = _axpy(alpha - beta, step_s, result)
    return _scale_vector(-1.0, result)


def _rows(states):
    if isinstance(states, list):
        if not states:
            raise LabError("states must contain a row")
        width = None
        rows = []
        for row in states:
            if not isinstance(row, list) or not row:
                raise LabError("each state must be a non-empty row")
            values = [_finite(value, "state") for value in row]
            if width is None:
                width = len(values)
            elif len(values) != width:
                raise LabError("state rows must have equal length")
            rows.append(values)
        return rows, "list"
    numpy = _numpy()
    if numpy is not None and isinstance(states, numpy.ndarray):
        if states.ndim != 2 or states.shape[0] == 0 or states.shape[1] == 0:
            raise LabError("states must be a non-empty matrix")
        if not numpy.isfinite(states).all():
            raise LabError("state entries must be finite")
        return states.astype(numpy.float64, copy=False), "numpy"
    raise LabError("states must be a list of rows or a numpy array")


def _vector(values, length: int, label: str) -> list[float]:
    numpy = _numpy()
    if numpy is not None and isinstance(values, numpy.ndarray):
        values = values.reshape(-1).tolist()
    if not isinstance(values, list) or len(values) != length:
        raise LabError(f"{label} must have one value per {'state' if label != 'old_row' else 'feature'}")
    return [_finite(value, label) for value in values]


def _scale(rows) -> list[float]:
    if isinstance(rows, list):
        n = len(rows)
        width = len(rows[0])
        scale = []
        for column in range(width):
            total = sum(rows[row][column] * rows[row][column] for row in range(n))
            rms = math.sqrt(total / n)
            scale.append(rms if rms > SCALE_FLOOR else SCALE_FLOOR)
        return scale
    numpy = _numpy()
    rms = numpy.sqrt(numpy.mean(rows * rows, axis=0))
    return numpy.maximum(rms, SCALE_FLOOR).astype(numpy.float64).tolist()


def _load_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise LabError("capture is not JSON") from error
    if not isinstance(payload, dict):
        raise LabError("capture must be an object")
    return payload


def _load_npz(path: Path) -> dict:
    numpy = _numpy()
    if numpy is None:
        raise LabError("an npz capture needs numpy")
    try:
        with numpy.load(path, allow_pickle=False) as archive:
            names = set(archive.files)
            required = {"states", "sign", "bound", "case_ids"}
            if not required <= names:
                raise LabError("npz capture needs states, sign, bound, and case_ids")
            if not names <= required | {"weights", "old_row", "meta"}:
                raise LabError("npz capture has an unknown array")
            payload = {
                "schema": SCHEMA,
                "states": archive["states"],
                "sign": archive["sign"],
                "bound": archive["bound"],
                "case_ids": [str(item) for item in archive["case_ids"].tolist()],
            }
            if "weights" in names:
                payload["weights"] = archive["weights"]
            if "old_row" in names:
                payload["old_row"] = archive["old_row"]
            if "meta" in names:
                meta = json.loads(str(archive["meta"].item()))
                if not isinstance(meta, dict):
                    raise LabError("npz meta must be an object")
                for key in ("regularization", "row", "parent_version"):
                    if key in meta:
                        payload[key] = meta[key]
    except LabError:
        raise
    except Exception as error:
        raise LabError("capture npz could not be read") from error
    return payload


def _output_window(source, row: int, delta) -> tuple[tuple[int, bytes], dict]:
    numpy = _numpy()
    gguf = _import_gguf()
    path = Path(source)
    if path.is_symlink() or not path.is_file():
        raise LabError("source must be a regular file")
    try:
        reader = gguf.GGUFReader(path, "r")
    except Exception as error:
        raise LabError(f"cannot read the model file: {error}") from error
    try:
        tensor = next((item for item in reader.tensors if item.name == "output.weight"), None)
        if tensor is None:
            raise LabError("output.weight is not in this file")
        n_rows = int(tensor.data.shape[0])
        if not 0 <= row < n_rows:
            raise LabError(f"row {row} is outside 0..{n_rows - 1}")
        current = numpy.array(tensor.data[row], copy=True)
        stride = int(current.nbytes)
        offset = int(tensor.data_offset) + row * stride
        values = numpy.array(
            gguf.dequantize(tensor.data[row : row + 1], tensor.tensor_type).reshape(-1),
            copy=True,
        )
        qtype_name = tensor.tensor_type.name
    finally:
        del reader
    file_bytes = path.read_bytes()
    if file_bytes[offset : offset + stride] != current.tobytes():
        raise LabError("output row bytes are not at the computed offset")
    quant = requantize_row(values, delta, qtype_name)
    if len(quant["bytes"]) != stride:
        raise LabError("quantized row changed width")
    return (offset, quant["bytes"]), {
        "quantized_relative_l2": quant["quantized_relative_l2"],
        "quantization": qtype_name,
    }


def _import_gguf():
    if _numpy() is None:
        raise LabError("quantizing an output row needs numpy")
    root = Path(__file__).resolve().parents[3]
    gguf_py = root / "third_party" / "llama.cpp" / "gguf-py"
    if not gguf_py.is_dir():
        raise LabError("vendored gguf reader is not in this checkout")
    entry = str(gguf_py)
    if entry not in __import__("sys").path:
        __import__("sys").path.insert(0, entry)
    try:
        import gguf
    except ImportError as error:
        raise LabError("vendored gguf reader could not be imported") from error
    return gguf


def _write_json(path: str, payload: dict) -> None:
    dest = Path(path)
    if dest.exists() or dest.is_symlink():
        raise LabError("correction output already exists")
    if not dest.parent.is_dir():
        raise LabError("correction directory does not exist")
    text = json.dumps(payload) + "\n"
    tmp = dest.with_name(f".{dest.name}.partial")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o644)
    try:
        os.write(fd, text.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, dest)


def _numpy():
    try:
        import numpy
    except ImportError:
        return None
    return numpy


def _finite(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LabError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise LabError(f"{label} must be a finite number")
    return number


def _whole_number(value, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LabError(f"{label} must be an integer")
    return value


def _check_finite(loss, grad) -> None:
    if not math.isfinite(loss) or not _finite_vector(grad):
        raise LabError("objective is not finite")


def _finite_vector(values) -> bool:
    if isinstance(values, list):
        return all(math.isfinite(value) for value in values)
    return bool(_numpy().isfinite(values).all())


def _copy(values):
    if isinstance(values, list):
        return list(values)
    return values.copy()


def _dot(left, right) -> float:
    if isinstance(left, list):
        return sum(a * b for a, b in zip(left, right))
    return float(left @ right)


def _norm(values) -> float:
    return math.sqrt(max(_dot(values, values), 0.0))


def _scale_vector(factor, values):
    if isinstance(values, list):
        return [factor * value for value in values]
    return factor * values


def _axpy(factor, left, right):
    if isinstance(left, list):
        return [factor * a + b for a, b in zip(left, right)]
    return factor * left + right


def _l2(values) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in values))


def _relative_l2(delta, old) -> float | None:
    denominator = _l2(old)
    if denominator == 0.0:
        return None
    return _l2(delta) / denominator


def _vector_sha256(values) -> str:
    raw = b"".join(struct.pack("<d", float(value)) for value in values)
    return hashlib.sha256(raw).hexdigest()
