"""Launcher, model profiles, device selection, and runtime argument checks."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from unittest import mock

from kilix_llm import models, paths, runtime
from kilix_llm.cli import build_parser, chat_argv, gpu_layers, run_argv, runtime_backend, serve_argv


REPO = paths.REPO_DIR
LAUNCHER = os.path.join(REPO, "kilix-llm")


class LauncherTests(unittest.TestCase):
    def test_help_and_version(self) -> None:
        help_run = subprocess.run(
            [LAUNCHER, "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(help_run.returncode, 0, help_run.stderr)
        self.assertIn("Qwen3.8", help_run.stdout)
        version = subprocess.run(
            [LAUNCHER, "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(version.returncode, 0, version.stderr)
        self.assertIn("0.1.0", version.stdout)

    def test_doctor_runs_without_weights(self) -> None:
        result = subprocess.run(
            [LAUNCHER, "doctor"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertIn(result.returncode, (0, 4), result.stderr)
        self.assertIn("CPU/RAM only", result.stdout)
        self.assertIn("-ngl 0", result.stdout)

    def test_vendor_prints_the_pin(self) -> None:
        result = subprocess.run(
            [LAUNCHER, "vendor"],
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
            [LAUNCHER, "run", "hello"],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("kilix-llm pull", result.stderr)

    def test_checkout_wins_over_another_working_directory_and_pythonpath(self) -> None:
        with tempfile.TemporaryDirectory() as other:
            with open(os.path.join(other, "kilix_llm.py"), "w", encoding="utf-8") as handle:
                handle.write("raise RuntimeError('wrong checkout imported')\n")
            with open(os.path.join(other, ".python-version"), "w", encoding="utf-8") as handle:
                handle.write("3.11\n")
            env = os.environ.copy()
            env["PYTHONPATH"] = other
            result = subprocess.run(
                [LAUNCHER, "--version"],
                cwd=other,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("0.1.0", result.stdout)


class ArgvTests(unittest.TestCase):
    def test_default_gpu_layers_is_zero(self) -> None:
        parser = build_parser()
        run_args = parser.parse_args(["run", "hi"])
        self.assertEqual(gpu_layers(run_args), 0)
        serve_args = parser.parse_args(["serve"])
        self.assertEqual(gpu_layers(serve_args), 0)
        self.assertEqual(serve_args.host, "127.0.0.1")

    def test_gpu_profile_uses_all_layers_and_explicit_cpu_override(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["run", "--model-id", "qwen3.5-4b", "hi"])
        self.assertEqual(runtime_backend(args), "vulkan")
        argv = run_argv(args, "llama-cli", "model.gguf")
        for flag, value in (("-ngl", "99"), ("-c", "8192"), ("--fit", "off"),
                            ("--reasoning", "off"), ("--cache-type-k", "q8_0")):
            self.assertEqual(argv[argv.index(flag) + 1], value)
        args.device = "Vulkan0"
        argv = run_argv(args, "llama-cli", "model.gguf")
        self.assertEqual(argv[argv.index("--override-tensor") + 1], r"^token_embd\.weight$=Vulkan0")
        args.device = None
        args.backend = "cpu"
        with self.assertRaises(runtime.RuntimeError_):
            runtime_backend(args)
        args.gpu_layers = 0
        self.assertEqual(runtime_backend(args), "cpu")
        argv = run_argv(args, "llama-cli", "model.gguf")
        self.assertEqual(argv[argv.index("-ngl") + 1], "0")
        self.assertEqual(argv[argv.index("--device") + 1], "none")

    def test_gpu_server_defaults_to_one_slot_without_thinking(self) -> None:
        args = build_parser().parse_args(["serve", "--model-id", "qwen3.5-4b"])
        argv = serve_argv(args, "llama-server", "model.gguf")
        self.assertEqual(argv[argv.index("--parallel") + 1], "1")
        self.assertEqual(argv[argv.index("--reasoning") + 1], "off")
        args.think = True
        argv = serve_argv(args, "llama-server", "model.gguf")
        self.assertEqual(argv[argv.index("--reasoning") + 1], "on")

    def test_no_loop_model_uses_the_qwen35_gpu_profile(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["serve", "--model-id", "qwen-0.8b-no-loop-02"])
        argv = serve_argv(args, "llama-server", "model.gguf")
        for flag, value in (("-ngl", "99"), ("--parallel", "1"), ("--reasoning", "off"),
                            ("--temp", "0.7"), ("--presence-penalty", "1.5"), ("--min-p", "0")):
            self.assertEqual(argv[argv.index(flag) + 1], value)
        args.think = True
        argv = serve_argv(args, "llama-server", "model.gguf")
        self.assertEqual(argv[argv.index("--reasoning") + 1], "on")
        # Thinking keeps the sampler the loop reduction was measured with.
        for flag, value in (("--temp", "0.7"), ("--top-p", "0.80"), ("--presence-penalty", "1.5")):
            self.assertEqual(argv[argv.index(flag) + 1], value)
        four_b = parser.parse_args(["serve", "--model-id", "qwen3.5-4b", "--think"])
        argv = serve_argv(four_b, "llama-server", "model.gguf")
        self.assertEqual(argv[argv.index("--temp") + 1], "1.0")
        run_args = parser.parse_args(["run", "--model-id", "qwen-0.8b-no-loop-02", "hi"])
        run_args.device = "Vulkan0"
        argv = run_argv(run_args, "llama-cli", "model.gguf")
        self.assertEqual(argv[argv.index("--override-tensor") + 1], r"^token_embd\.weight$=Vulkan0")
        self.assertEqual(argv[argv.index("--presence-penalty") + 1], "1.5")

    def test_chat_is_interactive_with_the_catalogue_sampler(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["chat", "--model-id", "qwen-0.8b-no-loop-02", "--gpu-layers", "0", "--think"])
        argv = chat_argv(args, "llama-cli", "model.gguf")
        for flag in ("-p", "-st", "-n", "--cache-type-k"):
            self.assertNotIn(flag, argv)
        for flag, value in (("-ngl", "0"), ("--device", "none"), ("--reasoning", "on"), ("--temp", "0.7"),
                            ("--top-p", "0.80"), ("--presence-penalty", "1.5")):
            self.assertEqual(argv[argv.index(flag) + 1], value)
        self.assertEqual(runtime_backend(args), "cpu")
        capped = parser.parse_args(["chat", "--model-id", "qwen-0.8b-no-loop-02", "-n", "512"])
        argv = chat_argv(capped, "llama-cli", "model.gguf")
        self.assertEqual(argv[argv.index("-n") + 1], "512")
        one_shot = run_argv(parser.parse_args(["run", "--model-id", "qwen-0.8b-no-loop-02", "hi"]),
                            "llama-cli", "model.gguf")
        self.assertEqual(one_shot[one_shot.index("-p"):one_shot.index("-p") + 3], ["-p", "hi", "-st"])

    def test_cpu_model_keeps_llama_defaults(self) -> None:
        args = build_parser().parse_args(["serve"])
        argv = serve_argv(args, "llama-server", "model.gguf")
        self.assertNotIn("--presence-penalty", argv)
        self.assertNotIn("--reasoning", argv)

    def test_catalogued_no_loop_identity(self) -> None:
        record = models.spec("qwen-0.8b-no-loop-02")
        self.assertEqual(record["file"], "qwen-0.8b-no-loop-02.gguf")
        self.assertEqual(record["size"], 833_592_416)
        self.assertEqual(
            record["sha256"],
            "e4d6e93eb644da62e0c5ac0b7b188b4c2cbf31f07a7dcfd42764b6236f96d8a0",
        )
        self.assertTrue(record["url"].endswith("/" + record["file"]))
        with tempfile.NamedTemporaryFile() as model:
            model.truncate(record["size"])
            self.assertEqual(models.resolve("qwen-0.8b-no-loop-02", model.name), model.name)
            with self.assertRaises(models.ModelError):
                models.resolve("qwen3.5-4b", model.name)

    def test_gpu_model_size_is_checked_against_selected_catalog(self) -> None:
        with tempfile.NamedTemporaryFile() as model:
            model.truncate(models.spec("qwen3.5-4b")["size"])
            self.assertEqual(models.resolve("qwen3.5-4b", model.name), model.name)
            with self.assertRaises(models.ModelError):
                models.resolve("qwen3.8-27b", model.name)
            model.truncate(1024)
            with self.assertRaises(models.ModelError):
                models.resolve("qwen3.5-4b", model.name)

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


class DeviceTests(unittest.TestCase):
    def test_device_listing_and_ambiguous_selection(self) -> None:
        result = subprocess.CompletedProcess([], 0, stdout=(
            "Available devices:\n"
            "  Vulkan0: Quadro RTX 3000 (NVK TU106) (6144 MiB, 5000 MiB free)\n"
            "  Vulkan1: Intel UHD Graphics (32000 MiB, 20000 MiB free)\n"
        ), stderr="")
        with mock.patch.object(runtime, "binary", return_value="llama-cli"), \
                mock.patch.object(runtime.subprocess, "run", return_value=result):
            self.assertEqual(runtime.select_device("vulkan", "Vulkan0"), "Vulkan0")
            with self.assertRaisesRegex(runtime.RuntimeError_, "multiple"):
                runtime.select_device("vulkan", None)
            with self.assertRaisesRegex(runtime.RuntimeError_, "unavailable"):
                runtime.select_device("vulkan", "none")
            result.stdout = "Available devices:\n  (none)\n"
            with self.assertRaisesRegex(runtime.RuntimeError_, "no GPU"):
                runtime.select_device("vulkan", None)
            result.returncode = 1
            with self.assertRaisesRegex(runtime.RuntimeError_, "cannot list"):
                runtime.devices("vulkan")

    def test_cpu_and_vulkan_builds_have_separate_paths(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertNotEqual(paths.bin_dir("cpu"), paths.bin_dir("vulkan"))
        with mock.patch.dict(os.environ, {"KILIX_LLM_BUILD_DIR": "/custom/runtime"}):
            self.assertEqual(paths.bin_dir("vulkan"), "/custom/runtime/bin")


def built(name: str, backend: str = "cpu") -> str | None:
    path = os.path.join(paths.bin_dir(backend), name)
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
                args = parser.parse_args(["chat", *extra])
                self.assert_argv_parses(chat_argv(args, cli, self.MISSING_MODEL))

    def test_serve_flags_are_accepted_by_llama_server(self) -> None:
        server = built("llama-server")
        if server is None:
            self.skipTest("llama-server is not built")
        args = build_parser().parse_args(["serve", "--port", "0"])
        self.assert_argv_parses(serve_argv(args, server, self.MISSING_MODEL))

    def test_gpu_profile_flags_are_accepted_by_vulkan_runtime(self) -> None:
        cli, server = built("llama-cli", "vulkan"), built("llama-server", "vulkan")
        if not cli or not server:
            self.skipTest("Vulkan runtime is not built")
        available = runtime.devices("vulkan")
        if not available:
            self.skipTest("no Vulkan GPU available")
        device = next(iter(available))
        for command, binary, builder, extra in (
            ("run", cli, run_argv, ["hi"]),
            ("serve", server, serve_argv, ["--port", "0"]),
        ):
            for thinking in ([], ["--think"]):
                with self.subTest(command=command, thinking=thinking):
                    args = build_parser().parse_args([
                        command, "--model-id", "qwen3.5-4b", "--device", device,
                        *thinking, *extra,
                    ])
                    self.assert_argv_parses(builder(args, binary, self.MISSING_MODEL))
