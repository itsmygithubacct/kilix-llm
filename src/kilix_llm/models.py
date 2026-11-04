"""Pinned GGUF identities for the RAM and GPU models."""

from __future__ import annotations

import hashlib
import os

from . import paths

DEFAULT_MODEL_ID = "qwen3.8-27b"

# Size and sha256 identify each exact Hugging Face LFS payload.
MODELS = {
    "qwen3.5-4b": {
        "title": "Qwen3.5 4B Q6 (GPU)",
        "file": "Qwen3.5-4B-Q6_K.gguf",
        "size": 3_525_956_768,
        "sha256": "fdedd781c9ce676ab66b018ca247ff78e8a33c98098a822c1e2d5075e7718f66",
        "url": (
            "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/"
            "e87f176479d0855a907a41277aca2f8ee7a09523/Qwen3.5-4B-Q6_K.gguf"
        ),
        "publisher": "unsloth/Qwen3.5-4B-GGUF",
        "ctx": 8192,
        "quant": "Q6_K",
        "n_gpu_layers": 99,
    },
    "qwen3.8-27b": {
        "title": "Qwen3.8 27B Q4",
        "file": "Qwen3.8-27B-UD-Q4_K_M.gguf",
        "size": 16_464_440_224,
        "sha256": "322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482",
        "url": (
            "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/"
            "Qwen3.8-27B-UD-Q4_K_M.gguf"
        ),
        "publisher": "unsloth/Qwen3.8-27B-GGUF",
        "ctx": 8192,
        "quant": "UD-Q4_K_M",
        "n_gpu_layers": 0,
    },
}

# Instruct-mode defaults. Thinking mode is opt-in: a 27B on a laptop CPU is
# already slow, and a reasoning preamble would hide the first visible token.
SAMPLING = ["--temp", "0.7", "--top-p", "0.80", "--top-k", "20"]
THINK_SAMPLING = ["--temp", "1.0", "--top-p", "0.95", "--top-k", "20"]


class ModelError(Exception):
    """A model or runtime that is not usable, with the CLI exit code."""

    def __init__(self, message: str, code: int = 2):
        super().__init__(message)
        self.code = code


def spec(model_id: str = DEFAULT_MODEL_ID) -> dict:
    try:
        return MODELS[model_id]
    except KeyError as error:
        known = ", ".join(sorted(MODELS))
        raise ModelError(f"unknown model {model_id!r} (known: {known})") from error


def model_path(model_id: str = DEFAULT_MODEL_ID, override: str | None = None) -> str:
    if override:
        return override
    return os.path.join(paths.models_dir(), spec(model_id)["file"])


def resolve(model_id: str = DEFAULT_MODEL_ID, override: str | None = None) -> str:
    """Return the weight file if it is present and the expected length."""
    path = model_path(model_id, override)
    record = spec(model_id)
    if not os.path.isfile(path):
        raise ModelError(
            f"{record['title']} is not downloaded ({path} missing).\n"
            f"kilix-llm does not fetch weights unless asked:  kilix-llm pull --model-id {model_id}"
        )
    actual = os.path.getsize(path)
    if actual != record["size"]:
        raise ModelError(
            f"{path} is {actual} bytes, expected {record['size']} — "
            f"an interrupted download? `kilix-llm pull --model-id {model_id}` resumes it"
        )
    return path


def digest_file(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def verify(model_id: str = DEFAULT_MODEL_ID, override: str | None = None) -> str:
    path = resolve(model_id, override)
    record = spec(model_id)
    got = digest_file(path)
    if got != record["sha256"]:
        raise ModelError(
            f"{path} sha256 is {got}, expected {record['sha256']}"
        )
    return path
