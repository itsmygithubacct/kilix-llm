"""Checkout layout. Machine-specific roots come from the environment."""

from __future__ import annotations

import os

REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VENDOR_DIR = os.path.join(REPO_DIR, "third_party", "llama.cpp")
PIN_PATH = os.path.join(REPO_DIR, "third_party", "llama.cpp.pin")
PATCH_DIR = os.path.join(REPO_DIR, "patches")
SCRIPTS_DIR = os.path.join(REPO_DIR, "scripts")


def gpu_terminal_home() -> str:
    return os.environ.get(
        "GPU_TERMINAL_HOME", os.path.expanduser("~/.local/gpu_terminal")
    )


def build_dir() -> str:
    override = os.environ.get("KILIX_LLM_BUILD_DIR")
    if override:
        return override
    return os.path.join(REPO_DIR, "build", "llama.cpp")


def bin_dir() -> str:
    return os.path.join(build_dir(), "bin")


def models_dir() -> str:
    override = os.environ.get("KILIX_LLM_MODELS_DIR")
    if override:
        return override
    return os.path.join(gpu_terminal_home(), "kilix-llm", "models")
