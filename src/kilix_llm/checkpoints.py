"""Explicit, pinned safetensor checkpoint acquisition for training consumers.

The optional huggingface_hub dependency is imported only by fetch(). No remote
Python code is acquired or executed. This does not modify built-in model IDs.
"""
from __future__ import annotations

import hashlib
import errno
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile


def validate(spec):
    if not isinstance(spec, dict) or set(spec) != {"schema", "model_id", "revision", "license", "files"}:
        raise ValueError("unexpected checkpoint manifest fields")
    if spec["schema"] != "kilix.llm.checkpoint/v1":
        raise ValueError("unsupported checkpoint schema")
    if not isinstance(spec["model_id"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", spec["model_id"]):
        raise ValueError("invalid checkpoint model ID")
    if not isinstance(spec["revision"], str) or not re.fullmatch(r"[a-f0-9]{40}", spec["revision"]):
        raise ValueError("checkpoint revision must be immutable")
    if not isinstance(spec["license"], str) or not 1 <= len(spec["license"]) <= 128:
        raise ValueError("invalid checkpoint license metadata")
    if not isinstance(spec["files"], dict) or not 1 <= len(spec["files"]) <= 256:
        raise ValueError("checkpoint needs bounded file identities")
    for name, entry in spec["files"].items():
        if not isinstance(name, str):
            raise ValueError("checkpoint file paths must be strings")
        path = PurePosixPath(name)
        if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./-]{0,255}", name)
                or path.is_absolute() or ".." in path.parts or str(path) != name
                or path.suffix not in {".json", ".safetensors", ".model", ".txt", ".jinja", ".md"}
                and path.name not in {"LICENSE", "LICENSE-MIT", "LICENSE-APACHE"}):
            raise ValueError("invalid checkpoint file path/type")
        if not isinstance(entry, dict) or set(entry) not in ({"size", "sha256"}, {"size", "git_blob_sha1"}):
            raise ValueError("invalid checkpoint file identity")
        if type(entry["size"]) is not int or not 1 <= entry["size"] <= 1024**4:
            raise ValueError("invalid checkpoint file size")
        key = "sha256" if "sha256" in entry else "git_blob_sha1"
        length = 64 if key == "sha256" else 40
        if not isinstance(entry[key], str) or not re.fullmatch(r"[a-f0-9]{"+str(length)+"}", entry[key]):
            raise ValueError("invalid checkpoint file digest")
    if "config.json" not in spec["files"] or not any(n.endswith(".safetensors") for n in spec["files"]):
        raise ValueError("checkpoint needs config and safetensors")
    return spec


def verify(spec, directory):
    validate(spec)
    directory = Path(directory)
    for name, identity in spec["files"].items():
        target = directory / name
        if target.is_symlink() or not target.resolve().is_relative_to(directory.resolve()):
            raise ValueError("checkpoint file escapes its directory")
        if target.stat().st_size != identity["size"]:
            raise ValueError(f"checkpoint size mismatch: {name}")
        kind = "sha256" if "sha256" in identity else "git_blob_sha1"
        hasher = hashlib.sha256() if kind == "sha256" else hashlib.sha1()
        if kind == "git_blob_sha1":
            hasher.update(f"blob {identity['size']}\0".encode())
        with target.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024*1024), b""):
                hasher.update(chunk)
        if hasher.hexdigest() != identity[kind]:
            raise ValueError(f"checkpoint digest mismatch: {name}")
    return directory


def fetch(spec, root):
    """Explicitly fetch a pinned manifest and verify before atomic publication."""
    validate(spec)
    destination = Path(root) / spec["model_id"].replace("/", "--") / spec["revision"]
    if destination.exists():
        return verify(spec, destination)
    from huggingface_hub import snapshot_download
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=destination.parent))
    try:
        snapshot_download(spec["model_id"], revision=spec["revision"], local_dir=temporary,
                          allow_patterns=list(spec["files"]), token=False, max_workers=2)
        verify(spec, temporary)
        (temporary / "kilix-checkpoint.json").write_text(json.dumps(spec, indent=2)+"\n")
        for path in temporary.rglob("*"):
            if path.is_file():
                path.chmod(0o600)
        try:
            os.rename(temporary, destination)
        except OSError as exc:
            if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
            return verify(spec, destination)
        return destination
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
