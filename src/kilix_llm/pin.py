"""The vendored llama.cpp revision is defined only by third_party/llama.cpp.pin."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass

from . import paths

_COMMIT = re.compile(r"[0-9a-f]{40}\Z")


class PinError(Exception):
    """The pin file is missing, malformed, or disagrees with patches/."""


@dataclass(frozen=True)
class Patch:
    name: str
    sha256: str
    path: str


@dataclass(frozen=True)
class Pin:
    url: str
    commit: str
    patches: tuple[Patch, ...]


def load(path: str | None = None) -> Pin:
    pin_path = path or paths.PIN_PATH
    try:
        with open(pin_path, encoding="utf-8") as handle:
            lines = handle.read().splitlines()
    except OSError as error:
        raise PinError(f"missing pin file {pin_path}") from error

    url = ""
    commit = ""
    patches: list[Patch] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("url="):
            url = stripped.split("=", 1)[1]
            continue
        if stripped.startswith("commit="):
            commit = stripped.split("=", 1)[1]
            continue
        parts = stripped.split()
        if len(parts) != 2:
            raise PinError(f"pin line is not a patch entry: {line!r}")
        name, digest = parts
        patch_path = os.path.join(paths.PATCH_DIR, name)
        patches.append(Patch(name=name, sha256=digest, path=patch_path))

    if not url.startswith("https://"):
        raise PinError("pin url= must be https")
    if _COMMIT.fullmatch(commit) is None:
        raise PinError(f"pin commit= is not a full 40-hex sha: {commit!r}")
    return Pin(url=url, commit=commit, patches=tuple(patches))


def listed_patch_names(pin: Pin) -> set[str]:
    return {patch.name for patch in pin.patches}


def disk_patch_names() -> set[str]:
    if not os.path.isdir(paths.PATCH_DIR):
        return set()
    return {
        name
        for name in os.listdir(paths.PATCH_DIR)
        if name.endswith(".patch") and not name.startswith(".")
    }


def verify(pin: Pin | None = None) -> Pin:
    loaded = pin or load()
    on_disk = disk_patch_names()
    listed = listed_patch_names(loaded)
    if listed != on_disk:
        raise PinError(
            "patches/ and the pin file disagree: "
            f"pin={sorted(listed)} disk={sorted(on_disk)}"
        )
    for patch in loaded.patches:
        try:
            with open(patch.path, "rb") as handle:
                payload = handle.read()
        except OSError as error:
            raise PinError(f"pin names missing patch {patch.name}") from error
        digest = hashlib.sha256(payload).hexdigest()
        if digest != patch.sha256:
            raise PinError(f"{patch.name} sha256 does not match the pin")
    return loaded
