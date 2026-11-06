"""Queue assignment and the bench-queue script. No network and no download."""

from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path

from kilix_llm_lab.benchqueue import assign, load_connections, provision_script, remember
from kilix_llm_lab.bench_worker import _gpu_offloaded, _pick_device, _pid_uses_gpu
from kilix_llm_lab import LabError

LAB = Path(__file__).resolve().parents[1]
SCRIPT = LAB / "scripts" / "bench-queue.sh"


class BenchQueueTests(unittest.TestCase):
    def test_script_parses_and_names_no_home_directory(self) -> None:
        self.assertTrue(os.access(SCRIPT, os.X_OK))
        completed = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("set -euo pipefail", text)
        self.assertNotIn("/" + "home" + "/", text)
        listed = (LAB / "scripts" / "lab.sh").read_text(encoding="utf-8")
        self.assertIn("queue", listed)
        self.assertNotIn("/" + "home" + "/", (LAB / "src" / "kilix_llm_lab" / "benchqueue.py").read_text(encoding="utf-8"))

    def test_longer_work_goes_to_the_faster_machine(self) -> None:
        jobs = [{"id": f"q{i}", "model": "small"} for i in range(4)]
        plan = assign(jobs, {"fast": {"small": 1.0}, "slow": {"small": 3.0}})
        counts = {alias: 0 for alias in ("fast", "slow")}
        for row in plan:
            counts[row["alias"]] += 1
        self.assertEqual(counts, {"fast": 3, "slow": 1})

    def test_a_machine_without_a_model_rate_is_skipped(self) -> None:
        jobs = [{"id": "q", "model": "big"}, {"id": "r", "model": "small"}]
        plan = assign(jobs, {"gpu": {"big": 2.0, "small": 1.0}, "cpu": {"small": 4.0}})
        self.assertEqual([row["alias"] for row in plan if row["model"] == "big"], ["gpu"])
        with self.assertRaises(LabError):
            assign([{"id": "q", "model": "missing"}], {"gpu": {"small": 1.0}})

    def test_remembered_aliases_round_trip(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as raw:
            os.environ["KILIX_LLM_BENCH_STATE"] = raw
            try:
                remember("local")
                remember("other", "other")
                document = load_connections()
            finally:
                os.environ.pop("KILIX_LLM_BENCH_STATE", None)
        self.assertEqual(document["connections"]["local"], {"transport": "local"})
        self.assertEqual(document["connections"]["other"]["transport"], "ssh")

    def test_gpu_request_keeps_the_device_and_skips_cpu(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as raw:
            os.environ["KILIX_LLM_BENCH_STATE"] = raw
            try:
                remember("box", "box", "gpu")
                remember("box", "box")
                document = load_connections()
            finally:
                os.environ.pop("KILIX_LLM_BENCH_STATE", None)
        self.assertEqual(document["connections"]["box"]["device"], "gpu")
        models = {
            "engine": {
                "tag": "b10174",
                "builds": [
                    {"id": "vulkan", "file": "v.tar.gz", "sha256": "aa"},
                    {"id": "cpu", "file": "c.tar.gz", "sha256": "bb"},
                ],
            },
            "models": [],
        }
        script = provision_script(models, require_gpu=True)
        self.assertNotIn("printf cpu", script)
        self.assertIn("GGML_CUDA=ON", script)
        self.assertIn("b10174", script)
        self.assertNotIn("/" + "home" + "/", script)
        plain = provision_script(models, require_gpu=False)
        self.assertIn("printf cpu", plain)
        self.assertNotIn("GGML_CUDA=ON", plain)
        self.assertTrue(_gpu_offloaded("load_tensors: offloaded 33/33 layers to GPU"))
        self.assertFalse(_gpu_offloaded("load_tensors: offloaded 0/33 layers to GPU"))
        listing = "Available devices:\n  Vulkan0: Intel(R) UHD Graphics (1 MiB, 1 MiB free)\n  Vulkan1: Quadro RTX 3000 (1 MiB, 1 MiB free)\n"
        self.assertEqual(_pick_device(listing), "Vulkan1")
        self.assertEqual(_pick_device("Available devices:\n  (none)\n"), None)
        self.assertEqual(_pick_device("  CUDA0: NVIDIA GeForce RTX 3070 (8192 MiB, 7000 MiB free)\n"), "CUDA0")
        self.assertTrue(_pid_uses_gpu("123, 672\n", 123))
        self.assertFalse(_pid_uses_gpu("123, 8\n", 123))
        self.assertFalse(_pid_uses_gpu("999, 800\n", 123))
