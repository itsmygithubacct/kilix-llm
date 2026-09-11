"""third_party/llama.cpp.pin is the single definition of the runtime."""

from __future__ import annotations

import os
import re
import unittest

from kilix_llm import pin
from kilix_llm.paths import PIN_PATH, VENDOR_DIR


class PinTests(unittest.TestCase):
    def test_pin_loads_and_matches_patches_on_disk(self) -> None:
        loaded = pin.verify()
        self.assertTrue(loaded.url.startswith("https://"))
        self.assertRegex(loaded.commit, r"^[0-9a-f]{40}$")
        self.assertEqual(pin.listed_patch_names(loaded), pin.disk_patch_names())

    def test_pin_file_has_exactly_one_url_and_commit(self) -> None:
        with open(PIN_PATH, encoding="utf-8") as handle:
            lines = handle.read().splitlines()
        urls = [line for line in lines if line.startswith("url=")]
        commits = [line for line in lines if line.startswith("commit=")]
        self.assertEqual(len(urls), 1)
        self.assertEqual(len(commits), 1)
        self.assertRegex(commits[0].split("=", 1)[1], r"^[0-9a-f]{40}$")

    def test_vendored_tree_has_the_qwen_graph(self) -> None:
        self.assertTrue(
            os.path.isfile(os.path.join(VENDOR_DIR, "CMakeLists.txt")),
            "vendored llama.cpp CMakeLists.txt",
        )
        self.assertTrue(
            os.path.isfile(os.path.join(VENDOR_DIR, "src", "models", "qwen35.cpp")),
            "Qwen3.5/3.8 graph present in the vendored tree",
        )
        self.assertTrue(os.path.isfile(os.path.join(VENDOR_DIR, "LICENSE")))


class PinFormatTests(unittest.TestCase):
    def test_commit_line_is_full_sha(self) -> None:
        loaded = pin.load()
        self.assertIsNotNone(re.fullmatch(r"[0-9a-f]{40}", loaded.commit))
