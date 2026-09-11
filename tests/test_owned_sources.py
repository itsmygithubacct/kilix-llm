"""Owned source must not bake in a home directory.

The vendored llama.cpp tree is upstream and is not this repository's to
rewrite; this walk skips it.
"""

from __future__ import annotations

import os
import unittest

from kilix_llm.paths import REPO_DIR

SKIP_DIRS = {
    ".git",
    "build",
    "__pycache__",
    "llama.cpp",
}
NEEDLE = "/" + "home" + "/"


def owned_files() -> list[str]:
    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(REPO_DIR):
        dirnames[:] = [name for name in dirnames if name not in SKIP_DIRS]
        for name in filenames:
            if name.endswith(".pyc"):
                continue
            found.append(os.path.join(dirpath, name))
    return found


class OwnedSourceTests(unittest.TestCase):
    def test_no_absolute_home_paths_in_owned_source(self) -> None:
        offenders = []
        for path in owned_files():
            try:
                with open(path, encoding="utf-8", errors="ignore") as handle:
                    text = handle.read()
            except OSError:
                continue
            if NEEDLE in text:
                offenders.append(os.path.relpath(path, REPO_DIR))
        self.assertEqual(offenders, [])

    def test_version_files_agree(self) -> None:
        from kilix_llm import __version__

        version_path = os.path.join(REPO_DIR, "VERSION")
        with open(version_path, encoding="utf-8") as handle:
            self.assertEqual(handle.read().strip(), __version__)
