"""Prepare a corpus and, when asked, run vendored llama-finetune.

The trainer is third_party/llama.cpp/examples/training. It updates every
parameter of an FP32 GGUF and writes a new GGUF. This module does not
download a model or a corpus. Omitting --run writes the text file and prints
the argument list. --run replaces this process with the trainer.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import struct
from pathlib import Path

from kilix_llm_lab import LabError
from kilix_llm_lab.corpus import prepare_corpus

SCHEMA = "kilix.llm.train/v1"
F32_FILE_TYPE = 0
_FILE_TYPE_KEY = "general.file_type"
_UINT32 = 4
_STRING = 8
_ARRAY = 9
_SCALAR_WIDTH = {
    0: 1,
    1: 1,
    2: 2,
    3: 2,
    4: 4,
    5: 4,
    6: 4,
    7: 1,
    10: 8,
    11: 8,
    12: 8,
}


def run_train(args) -> int:
    """Validate the corpus, print one plan, and exec llama-finetune on --run."""
    ctx = _context(args.ctx)
    batch = _batch(args.batch, "batch")
    ubatch = _batch(args.ubatch, "ubatch")
    epochs = _epochs(args.epochs)
    learning_rate = _positive(args.learning_rate, "learning rate")
    val_split = _fraction(args.val_split)
    weight_decay = _nonnegative(args.weight_decay, "weight decay")
    gpu_layers = _gpu_layers(args.gpu_layers)
    optimizer = args.optimizer
    if optimizer not in ("adamw", "sgd"):
        raise LabError("optimizer must be adamw or sgd")
    learning_rate_min = None
    if args.learning_rate_min is not None:
        learning_rate_min = _nonnegative(args.learning_rate_min, "learning rate min")
    decay_epochs = None
    if args.decay_epochs is not None:
        decay_epochs = _positive(args.decay_epochs, "decay epochs")

    text_out = Path(args.text_out).expanduser()
    if text_out.suffix.lower() == ".gguf":
        raise LabError("text output must not be a GGUF file")
    model = Path(args.model).expanduser()
    if model.suffix.lower() != ".gguf":
        raise LabError("model must be a .gguf file")
    output = Path(args.output).expanduser() if args.output else text_out.with_name("finetuned-model.gguf")
    if output.suffix.lower() != ".gguf":
        raise LabError("output must be a .gguf file")
    if not output.parent.is_dir():
        raise LabError("output directory does not exist")

    manifest = prepare_corpus(args.corpus, text_out)
    binary = find_finetune(args.binary)
    argv = finetune_argv(
        binary if binary is not None else "llama-finetune",
        text=str(text_out.resolve()),
        model=str(model.resolve()),
        output=str(output.resolve()),
        ctx=ctx,
        batch=batch,
        ubatch=ubatch,
        gpu_layers=gpu_layers,
        epochs=epochs,
        learning_rate=learning_rate,
        val_split=val_split,
        weight_decay=weight_decay,
        optimizer=optimizer,
        learning_rate_min=learning_rate_min,
        decay_epochs=decay_epochs,
    )
    plan = {
        "schema": SCHEMA,
        "backend": "llama-finetune",
        "exec": bool(args.run),
        "corpus": {
            "documents": manifest["documents"],
            "bytes": manifest["bytes"],
            "files": len(manifest["files"]),
            "skipped_count": manifest["skipped_count"],
        },
        "text": argv[argv.index("--file") + 1],
        "model": argv[argv.index("--model") + 1],
        "output": argv[argv.index("-o") + 1],
        "argv": argv,
    }
    if args.run:
        _require_runnable(model, output, binary)
    print(json.dumps(plan), flush=True)
    if args.run:
        os.execv(binary, argv)
    return 0


def finetune_argv(
    binary: str,
    *,
    text: str,
    model: str,
    output: str,
    ctx: int,
    batch: int,
    ubatch: int,
    gpu_layers: int,
    epochs: int,
    learning_rate: float,
    val_split: float,
    weight_decay: float,
    optimizer: str,
    learning_rate_min: float | None = None,
    decay_epochs: float | None = None,
) -> list[str]:
    """Argument vector for examples/training llama-finetune."""
    argv = [
        binary,
        "--file",
        text,
        "-ngl",
        str(gpu_layers),
        "--model",
        model,
        "-c",
        str(ctx),
        "-b",
        str(batch),
        "-ub",
        str(ubatch),
        "-o",
        output,
        "--epochs",
        str(epochs),
        "--learning-rate",
        _num(learning_rate),
        "--val-split",
        _num(val_split),
        "--weight-decay",
        _num(weight_decay),
        "--optimizer",
        optimizer,
    ]
    if learning_rate_min is not None:
        argv.extend(["--learning-rate-min", _num(learning_rate_min)])
    if decay_epochs is not None:
        argv.extend(["--learning-rate-decay-epochs", _num(decay_epochs)])
    return argv


def find_finetune(explicit: str | None) -> str | None:
    """Return the trainer path. None means the plan names the program only."""
    if explicit:
        return str(Path(explicit).expanduser().resolve())
    env = os.environ.get("KILIX_LLM_LLAMA_FINETUNE")
    if env:
        return str(Path(env).expanduser().resolve())
    # train.py lives at lab/src/kilix_llm_lab/train.py, so parents[3] is the repo.
    built = Path(__file__).resolve().parents[3] / "build" / "bin" / "llama-finetune"
    if built.is_file() and os.access(built, os.X_OK):
        return str(built)
    return shutil.which("llama-finetune")


def read_gguf_file_type(path: Path) -> int:
    """Read general.file_type from the GGUF metadata. Do not load tensors."""
    try:
        handle = path.open("rb")
    except OSError as error:
        raise LabError(f"cannot read the model file: {error.strerror}") from error
    with handle:
        if _read_exact(handle, 4) != b"GGUF":
            raise LabError("model is not a GGUF file")
        version = struct.unpack("<I", _read_exact(handle, 4))[0]
        if version < 2:
            raise LabError("GGUF version is too old")
        _read_exact(handle, 8)
        (kv_count,) = struct.unpack("<Q", _read_exact(handle, 8))
        if kv_count > 100_000:
            raise LabError("GGUF metadata is too large")
        for _ in range(kv_count):
            key = _read_string(handle)
            (value_type,) = struct.unpack("<I", _read_exact(handle, 4))
            if key == _FILE_TYPE_KEY:
                if value_type != _UINT32:
                    raise LabError("general.file_type is not a uint32")
                (file_type,) = struct.unpack("<I", _read_exact(handle, 4))
                return file_type
            _skip_value(handle, value_type)
    raise LabError("model has no general.file_type")


def _require_runnable(model: Path, output: Path, binary: str | None) -> None:
    if model.is_symlink() or not model.is_file():
        raise LabError("model must be a regular file")
    if output.exists():
        raise LabError("output already exists")
    if model.resolve() == output.resolve():
        raise LabError("output and model are the same file")
    file_type = read_gguf_file_type(model)
    if file_type != F32_FILE_TYPE:
        raise LabError("llama-finetune trains an FP32 GGUF (general.file_type 0)")
    if binary is None or not os.access(binary, os.X_OK):
        raise LabError(
            "llama-finetune is not executable; build examples/training or pass --binary"
        )


def _read_exact(handle, size: int) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise LabError("model file ended inside the GGUF header")
    return data


def _read_string(handle) -> str:
    (length,) = struct.unpack("<Q", _read_exact(handle, 8))
    if length > 1_000_000:
        raise LabError("GGUF metadata string is too long")
    raw = _read_exact(handle, length)
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise LabError("GGUF metadata string is not UTF-8") from error


def _skip_value(handle, value_type: int) -> None:
    if value_type in _SCALAR_WIDTH:
        _read_exact(handle, _SCALAR_WIDTH[value_type])
        return
    if value_type == _STRING:
        _read_string(handle)
        return
    if value_type == _ARRAY:
        (item_type,) = struct.unpack("<I", _read_exact(handle, 4))
        (count,) = struct.unpack("<Q", _read_exact(handle, 8))
        if count > 10_000_000:
            raise LabError("GGUF metadata array is too long")
        for _ in range(count):
            _skip_value(handle, item_type)
        return
    raise LabError("GGUF metadata uses an unknown value type")


def _context(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LabError("context must be an integer")
    if value != 0 and value < 32:
        raise LabError("context must be 0 or at least 32")
    return value


def _batch(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 32:
        raise LabError(f"{label} must be an integer of at least 32")
    return value


def _epochs(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise LabError("epochs must be a positive integer")
    return value


def _gpu_layers(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LabError("gpu layers must be an integer")
    return value


def _positive(value: float, label: str) -> float:
    number = _finite(value, label)
    if number <= 0:
        raise LabError(f"{label} must be positive")
    return number


def _nonnegative(value: float, label: str) -> float:
    number = _finite(value, label)
    if number < 0:
        raise LabError(f"{label} must be zero or positive")
    return number


def _fraction(value: float) -> float:
    number = _finite(value, "val split")
    if number < 0 or number >= 1:
        raise LabError("val split must be from 0 up to but not including 1")
    return number


def _finite(value: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LabError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise LabError(f"{label} must be a finite number")
    return number


def _num(value: float) -> str:
    return format(value, ".8g")
