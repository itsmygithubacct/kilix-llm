#!/usr/bin/env python3
"""check-deb.py DEB : kilix-llm ships the launcher and not the weights."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path


WEIGHTS = {".gguf", ".onnx", ".safetensors", ".pt"}


def fail(message: str) -> None:
    print(f"check-deb: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    if len(sys.argv) != 2:
        fail("usage: check-deb.py DEB")
    deb = Path(sys.argv[1])
    fields = subprocess.run(
        ["dpkg-deb", "-f", str(deb), "Package", "Version", "Architecture", "Depends"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    for needle in (
        "Package: kilix-llm\n",
        "Version: 0.1.0+packaging1\n",
        "Architecture: all\n",
    ):
        if needle not in fields:
            fail(f"unexpected fields:\n{fields}")
    if "python3" not in fields:
        fail(f"missing python3 dependency:\n{fields}")
    with tempfile.TemporaryDirectory(prefix="kilix-llm-deb.") as raw:
        root = Path(raw)
        subprocess.run(["dpkg-deb", "-x", str(deb), str(root)], check=True)
        share = root / "usr/share/kilix-llm"
        launcher = root / "usr/bin/kilix-llm"
        if not launcher.is_file() or launcher.is_symlink():
            fail("command missing")
        if not launcher.read_text(encoding="utf-8").startswith("#!/bin/sh\n"):
            fail("command is not the system Python wrapper")
        for rel in (
            "src/kilix_llm/__main__.py",
            "src/kilix_llm/paths.py",
            "src/kilix_llm/vllm_runtime.py",
            "src/kilix_llm/vllm_oneshot.py",
            "third_party/llama.cpp.pin",
            "VERSION",
            "scripts/build-runtime.sh",
            "scripts/vendor-sync.sh",
        ):
            if not (share / rel).is_file():
                fail(f"missing {rel}")
        if (share / "third_party/llama.cpp").exists():
            fail("llama.cpp tree packaged")
        if (share / "lab").exists() or (share / "tests").exists():
            fail("lab or tests packaged")
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in WEIGHTS:
                fail(f"weight packaged: {path.name}")
            if path.is_file() and not path.is_symlink() and path.stat().st_size > 1_000_000:
                fail(f"file over 1MB: {path.name} ({path.stat().st_size})")
        models = root / "models"
        models.mkdir()
        env = os.environ.copy()
        env["KILIX_LLM_MODELS_DIR"] = str(models)
        env.pop("PYTHONPATH", None)
        version = subprocess.run(
            [str(launcher), "--version"],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        if version.returncode != 0 or version.stdout.strip() != "kilix-llm 0.1.0":
            fail(f"version failed ({version.returncode}): {version.stdout!r} {version.stderr!r}")
        help_run = subprocess.run(
            [str(launcher), "--help"],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        if help_run.returncode != 0 or "usage:" not in help_run.stdout.lower():
            fail(f"help failed ({help_run.returncode})")
        doctor = subprocess.run(
            [str(launcher), "doctor"],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        if doctor.returncode not in (0, 4):
            fail(f"doctor failed ({doctor.returncode}): {doctor.stderr}")
        if "kilix-llm 0.1.0" not in doctor.stdout or "not downloaded" not in doctor.stdout:
            fail("doctor did not report the version and a missing model")
        if "8ea290247c87ced2ab245b056ffe96dbcf90d36c" not in doctor.stdout:
            fail("doctor did not resolve the installed pin")
    print("check-deb: passed")


if __name__ == "__main__":
    main()
