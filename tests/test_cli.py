"""The checkout is runnable without being installed, and run/serve stay on RAM."""

from __future__ import annotations

import os
import subprocess
import sys
import unittest

from kilix_llm import models, paths, runtime
from kilix_llm.cli import build_parser, run_argv, serve_argv


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


def built(name: str) -> str | None:
    path = os.path.join(paths.bin_dir(), name)
    return path if os.path.isfile(path) and os.access(path, os.X_OK) else None


class RuntimeFlagTests(unittest.TestCase):
    """Every flag run and serve emit must be one the built binary accepts.

    `-no-cnv` shipped in `run` and llama-cli refused it: at the pinned commit
    the conversation flags are registered for llama-completion only, so the
    command died on its own argv after an hour of downloading the weights.
    Pointing -m at a path that does not exist makes each binary parse the whole
    command line and then stop at the model, which needs no GGUF and no GPU.
    """

    MISSING_MODEL = "/nonexistent/kilix-llm-test.gguf"

    def assert_argv_parses(self, argv: list[str]) -> None:
        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        output = result.stdout + result.stderr
        self.assertNotIn("invalid argument", output, " ".join(argv))
        self.assertIn(self.MISSING_MODEL, output, "parsing stopped before the model")

    def test_run_flags_are_accepted_by_llama_cli(self) -> None:
        cli = built("llama-cli")
        if cli is None:
            self.skipTest("llama-cli is not built")
        parser = build_parser()
        for extra in ([], ["--think"]):
            with self.subTest(extra=extra):
                args = parser.parse_args(["run", *extra, "hi"])
                self.assert_argv_parses(run_argv(args, cli, self.MISSING_MODEL))

    def test_serve_flags_are_accepted_by_llama_server(self) -> None:
        server = built("llama-server")
        if server is None:
            self.skipTest("llama-server is not built")
        args = build_parser().parse_args(["serve"])
        self.assert_argv_parses(serve_argv(args, server, self.MISSING_MODEL))
