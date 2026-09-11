"""Vendored llama.cpp: where it is, how it is built, which binaries to exec."""

from __future__ import annotations

import os
import shutil

from . import paths
from .pin import PinError, load as load_pin

BINARIES = ("llama-cli", "llama-server", "llama-bench")

CMAKE_FLAGS = (
    "-DCMAKE_BUILD_TYPE=Release",
    "-DGGML_NATIVE=ON",
    "-DGGML_CUDA=OFF",
    "-DGGML_VULKAN=OFF",
    "-DGGML_METAL=OFF",
    "-DLLAMA_CURL=OFF",
)


class RuntimeError_(Exception):
    """The vendored runtime is missing or not built, with the CLI exit code."""

    def __init__(self, message: str, code: int = 4):
        super().__init__(message)
        self.code = code


def physical_cores() -> int:
    """Physical cores where /proc/cpuinfo exposes topology, else logical count.

    Hyperthreads lose throughput on bandwidth-bound decode; the default is
    the physical core count, measured, not nproc.
    """
    seen: set[tuple[str, str]] = set()
    try:
        phys = core = None
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("physical id"):
                    phys = line.split(":", 1)[1].strip()
                elif line.startswith("core id"):
                    core = line.split(":", 1)[1].strip()
                elif not line.strip():
                    if phys is not None and core is not None:
                        seen.add((phys, core))
                    phys = core = None
        if phys is not None and core is not None:
            seen.add((phys, core))
    except OSError:
        pass
    return len(seen) or os.cpu_count() or 1


def vendor_present() -> bool:
    return os.path.isfile(os.path.join(paths.VENDOR_DIR, "CMakeLists.txt"))


def vendor_commit() -> str | None:
    git_dir = os.path.join(paths.VENDOR_DIR, ".git")
    if not os.path.exists(git_dir):
        return None
    try:
        import subprocess

        result = subprocess.run(
            ["git", "-C", paths.VENDOR_DIR, "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def missing_build_packages() -> list[str]:
    missing: list[str] = []
    if shutil.which("cmake") is None:
        missing.append("cmake")
    if (
        shutil.which("c++") is None
        and shutil.which("g++") is None
        and shutil.which("clang++") is None
    ):
        missing.append("g++")
    return missing


def binary(name: str) -> str:
    if name not in BINARIES:
        raise RuntimeError_(f"unknown runtime binary {name!r}")
    path = os.path.join(paths.bin_dir(), name)
    if not os.path.isfile(path) or not os.access(path, os.X_OK):
        raise RuntimeError_(
            f"{name} is not built ({path} missing).\n"
            f"Build the vendored runtime:  kilix-llm build"
        )
    return path


def require_vendor() -> None:
    if not vendor_present():
        raise RuntimeError_(
            f"vendored llama.cpp is missing ({paths.VENDOR_DIR}).\n"
            f"From a git checkout:  git submodule update --init third_party/llama.cpp"
        )
    try:
        pin = load_pin()
    except PinError as error:
        raise RuntimeError_(str(error)) from error
    qwen = os.path.join(paths.VENDOR_DIR, "src", "models", "qwen35.cpp")
    if not os.path.isfile(qwen):
        raise RuntimeError_(
            f"vendored llama.cpp at {pin.commit} has no Qwen3.5/3.8 graph "
            f"({qwen} missing)"
        )


def meminfo() -> dict[str, int]:
    values = {"MemTotal": 0, "MemAvailable": 0, "SwapTotal": 0, "SwapFree": 0}
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                key, _, rest = line.partition(":")
                if key in values:
                    values[key] = int(rest.strip().split()[0]) * 1024
    except OSError:
        pass
    return values
