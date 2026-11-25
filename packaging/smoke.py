#!/usr/bin/env python3
"""Stage the packaged layout and run the system-Python command."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = {".gguf", ".onnx", ".safetensors", ".pt", ".bin"}


def fail(message: str) -> None:
    print(f"smoke: {message}", file=sys.stderr)
    raise SystemExit(1)


def stage(dest: Path) -> Path:
    share = dest / "usr/share/kilix-llm"
    (share / "src").mkdir(parents=True)
    (share / "third_party").mkdir()
    shutil.copytree(ROOT / "src/kilix_llm", share / "src/kilix_llm",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(ROOT / "scripts", share / "scripts")
    shutil.copytree(ROOT / "patches", share / "patches")
    shutil.copy2(ROOT / "third_party/llama.cpp.pin", share / "third_party/llama.cpp.pin")
    shutil.copy2(ROOT / "VERSION", share / "VERSION")
    shutil.copy2(ROOT / "LICENSE", share / "LICENSE")
    bindir = dest / "usr/bin"
    bindir.mkdir(parents=True)
    launcher = bindir / "kilix-llm"
    shutil.copy2(ROOT / "packaging/kilix-llm", launcher)
    launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return launcher


def main() -> None:
    if (ROOT / "packaging/kilix-llm").read_text(encoding="utf-8").splitlines()[0] != "#!/bin/sh":
        fail("packaged command must use system Python")
    with tempfile.TemporaryDirectory(prefix="kilix-llm-smoke.") as raw:
        root = Path(raw)
        launcher = stage(root)
        share = root / "usr/share/kilix-llm"
        if (share / "third_party/llama.cpp").exists():
            fail("llama.cpp tree was staged")
        if (share / "lab").exists() or (share / "tests").exists():
            fail("lab or tests were staged")
        for path in share.rglob("*"):
            if path.suffix in WEIGHTS or path.name.endswith(".gguf"):
                fail(f"weight staged: {path.name}")
        models = root / "models"
        models.mkdir()
        env = os.environ.copy()
        env["KILIX_LLM_MODELS_DIR"] = str(models)
        env.pop("PYTHONPATH", None)

        def run(args: list[str]) -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                [str(launcher), *args],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )

        version = run(["--version"])
        if version.returncode != 0 or version.stdout.strip() != "kilix-llm 0.1.0":
            fail(f"version failed ({version.returncode}): {version.stdout!r} {version.stderr!r}")
        help_run = run(["--help"])
        if help_run.returncode != 0 or "kilix-llm" not in help_run.stdout:
            fail(f"help failed ({help_run.returncode}): {help_run.stderr}")
        doctor = run(["doctor"])
        if doctor.returncode not in (0, 4):
            fail(f"doctor failed ({doctor.returncode}): {doctor.stderr}")
        if "kilix-llm 0.1.0" not in doctor.stdout or "not downloaded" not in doctor.stdout:
            fail(f"doctor output missing the version or the model state:\n{doctor.stdout}")
        if "8ea290247c87ced2ab245b056ffe96dbcf90d36c" not in doctor.stdout:
            fail(f"doctor did not resolve the pin:\n{doctor.stdout}")
    print("smoke: passed")


if __name__ == "__main__":
    main()
