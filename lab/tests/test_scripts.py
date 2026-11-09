"""Lab shell scripts parse, use uv, and name no home directory."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
SCRIPTS = (
    LAB / "scripts" / "uv-venv.sh",
    LAB / "scripts" / "bench-qwen3-0.6b.sh",
    LAB / "scripts" / "fetch-model.sh",
    LAB / "scripts" / "neuron.sh",
    LAB / "scripts" / "sweep-shards.sh",
    LAB / "scripts" / "lab.sh",
)


class LabScriptTests(unittest.TestCase):
    def test_scripts_are_executable_and_parse(self) -> None:
        for path in SCRIPTS:
            with self.subTest(script=path.name):
                self.assertTrue(os.access(path, os.X_OK), path.name)
                completed = subprocess.run(
                    ["bash", "-n", str(path)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                text = path.read_text(encoding="utf-8")
                self.assertIn("set -euo pipefail", text)
                self.assertIn("BASH_VERSINFO", text)
                self.assertNotIn("/" + "home" + "/", text)

    def test_environment_script_uses_uv_venv(self) -> None:
        text = (LAB / "scripts" / "uv-venv.sh").read_text(encoding="utf-8")
        self.assertIn("uv venv --python 3.12", text)
        self.assertIn("uv pip install", text)
        self.assertIn("numpy", text)
        self.assertIn("KILIX_LLM_LAB_VENV", text)
        self.assertIn("KILIX_LLM_TORCH_INDEX", text)

    def test_bench_script_uses_that_environment(self) -> None:
        text = (LAB / "scripts" / "bench-qwen3-0.6b.sh").read_text(encoding="utf-8")
        self.assertIn("uv-venv.sh", text)
        self.assertIn("bench.py", text)
        program = (LAB / "projects" / "qwen3_0_6b" / "bench.py").read_text(encoding="utf-8")
        self.assertIn("responses_changed", program)
        self.assertNotIn("tokens_per_second", program)

    def test_lab_script_lists_options_and_rejects_an_unknown_command(self) -> None:
        script = LAB / "scripts" / "lab.sh"
        listed = subprocess.run([str(script)], capture_output=True, text=True, check=False)
        self.assertEqual(listed.returncode, 0, listed.stderr)
        self.assertEqual(listed.stderr, "")
        for name in (
            "venv",
            "fetch",
            "means",
            "steer",
            "abliterate",
            "story",
            "neurons",
            "neuron",
            "head",
            "sae",
            "circuit",
            "refusal",
            "sweep",
            "sweep-shards",
            "split a sweep",
            "control",
            "rank",
            "weight",
            "weight-space column",
            "loops",
            "repeating tails",
            "gguf-columns",
            "exact column",
            "gguf-tensors",
            "requantized",
            "output-row",
            "squared-hinge edit of one output row",
            "train",
            "prepare a text corpus",
            "llama-finetune",
            "distinctive-word",
            "catalog",
            "bench",
            "attention-head",
            "sparse autoencoder",
            "circuit trace",
            "refusal corpus",
            "neuron sweep",
            "difference of means",
            "hook steering",
            "abliteration",
            "neuron push",
            "random-direction control",
            "bounded catalog",
        ):
            self.assertIn(name, listed.stdout)
        rejected = subprocess.run(
            [str(script), "nope"], capture_output=True, text=True, check=False
        )
        self.assertEqual(rejected.returncode, 2)
        self.assertIn("neuron", rejected.stderr)
        self.assertEqual(rejected.stdout, "")

    def test_sweep_shards_cover_the_range_without_a_checkpoint(self) -> None:
        script = LAB / "scripts" / "sweep-shards.sh"
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run(
                [
                    str(script),
                    "--shards",
                    "4",
                    "--record-dir",
                    directory,
                    "--stand-in",
                    "--stop",
                    "8",
                    "--width",
                    "8",
                    "--strength",
                    "1",
                    "--controls",
                    "1",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
            pieces = []
            for shard in range(4):
                log = (Path(directory) / f"shard-{shard}.log").read_text(encoding="utf-8")
                summary = json.loads(log.strip().splitlines()[-1])
                pieces.append((summary["start"], summary["stop"], summary["indexes"]))
                self.assertEqual(summary["shard"], shard)
                self.assertEqual(summary["shards"], 4)
                record = Path(directory) / f"shard-{shard}.jsonl"
                self.assertTrue(record.is_file())
        self.assertEqual(pieces, [(0, 2, 2), (2, 4, 2), (4, 6, 2), (6, 8, 2)])
