"""The checkout is runnable without being installed, and run/serve stay on RAM."""

from __future__ import annotations

import os
import subprocess
import sys
import unittest

from kilix_llm import models, paths, runtime
from kilix_llm.cli import build_parser


REPO = paths.REPO_DIR
LAUNCHER = os.path.join(REPO, "kilix-llm")


class LauncherTests(unittest.TestCase):
    def test_help_and_version(self) -> None:
        help_run = subprocess.run(
            [sys.executable, LAUNCHER, "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(help_run.returncode, 0, help_run.stderr)
        self.assertIn("Qwen3.8", help_run.stdout)
        version = subprocess.run(
            [sys.executable, LAUNCHER, "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(version.returncode, 0, version.stderr)
        self.assertIn("0.1.0", version.stdout)

    def test_doctor_runs_without_weights(self) -> None:
        result = subprocess.run(
            [sys.executable, LAUNCHER, "doctor"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertIn(result.returncode, (0, 4), result.stderr)
        self.assertIn("CPU/RAM only", result.stdout)
        self.assertIn("-ngl 0", result.stdout)

    def test_vendor_prints_the_pin(self) -> None:
        result = subprocess.run(
            [sys.executable, LAUNCHER, "vendor"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("commit=8ea290247c87ced2ab245b056ffe96dbcf90d36c", result.stdout)
        self.assertIn("url=https://github.com/ggml-org/llama.cpp.git", result.stdout)

    def test_run_refuses_a_missing_model_and_names_pull(self) -> None:
        env = os.environ.copy()
        env["KILIX_LLM_MODELS_DIR"] = os.path.join(REPO, "build", "empty-models")
        result = subprocess.run(
            [sys.executable, LAUNCHER, "run", "hello"],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("kilix-llm pull", result.stderr)


class ArgvTests(unittest.TestCase):
    def test_default_gpu_layers_is_zero(self) -> None:
        parser = build_parser()
        run_args = parser.parse_args(["run", "hi"])
        self.assertEqual(run_args.gpu_layers, 0)
        serve_args = parser.parse_args(["serve"])
        self.assertEqual(serve_args.gpu_layers, 0)
        self.assertEqual(serve_args.host, "127.0.0.1")

    def test_catalogued_q4_identity(self) -> None:
        record = models.spec()
        self.assertEqual(record["file"], "Qwen3.8-27B-UD-Q4_K_M.gguf")
        self.assertEqual(record["n_gpu_layers"], 0)
        self.assertEqual(record["size"], 16_464_440_224)
        self.assertEqual(
            record["sha256"],
            "322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482",
        )

    def test_cmake_flags_disable_accelerators(self) -> None:
        flags = " ".join(runtime.CMAKE_FLAGS)
        self.assertIn("GGML_CUDA=OFF", flags)
        self.assertIn("GGML_VULKAN=OFF", flags)
        self.assertIn("LLAMA_CURL=OFF", flags)
