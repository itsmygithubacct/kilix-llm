"""The shell scripts parse, fail fast, and keep CUDA off."""

from __future__ import annotations

import os
import subprocess
import unittest

from kilix_llm import paths, runtime

REPO = paths.REPO_DIR
SCRIPTS = (
    os.path.join(REPO, "scripts", "vendor-sync.sh"),
    os.path.join(REPO, "scripts", "build-runtime.sh"),
    os.path.join(REPO, "install.sh"),
)


class ScriptTests(unittest.TestCase):
    def test_scripts_are_executable_parse_and_fail_fast(self) -> None:
        for path in SCRIPTS:
            name = os.path.basename(path)
            with self.subTest(script=name):
                self.assertTrue(os.access(path, os.X_OK), f"{name} is executable")
                rc = subprocess.call(["bash", "-n", path])
                self.assertEqual(rc, 0, f"{name} parses")
                with open(path, encoding="utf-8") as handle:
                    text = handle.read()
                self.assertIn("set -euo pipefail", text)
                self.assertIn("BASH_VERSINFO", text)

    def test_build_is_cpu_only(self) -> None:
        with open(os.path.join(REPO, "scripts", "build-runtime.sh"), encoding="utf-8") as handle:
            text = handle.read()
        for flag in runtime.CMAKE_FLAGS:
            self.assertIn(flag, text)
        self.assertIn("-DGGML_CUDA=OFF", text)
        self.assertIn("-DGGML_VULKAN=OFF", text)
        self.assertIn("KILIX_LLM_BUILD_DIR", text)

    def test_vendor_sync_honours_the_pin_file(self) -> None:
        with open(os.path.join(REPO, "scripts", "vendor-sync.sh"), encoding="utf-8") as handle:
            text = handle.read()
        self.assertIn("llama.cpp.pin", text)
        self.assertIn("KILIX_LLM_LLAMA_MIRROR", text)
