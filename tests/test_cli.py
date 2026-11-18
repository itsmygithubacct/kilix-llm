"""Launcher, model profiles, device selection, and runtime argument checks."""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

from kilix_llm import models, paths, runtime
from kilix_llm.cli import (
    build_parser,
    chat_argv,
    cmd_pull,
    gpu_layers,
    run_argv,
    runtime_backend,
    serve_argv,
)


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
        self.assertIn("LLAMA_USE_PREBUILT_UI=OFF", flags)


class ThinkWarningTests(unittest.TestCase):
    """The 0.8B keeps thinking off by default and warns when it is turned on."""

    def launch(self, argv: list[str]) -> str:
        from kilix_llm.cli import main
        err = io.StringIO()
        with tempfile.TemporaryDirectory() as root, redirect_stderr(err):
            # A missing weight file stops the launch right after the warning.
            with self.assertRaises(SystemExit):
                main([*argv, "--model", os.path.join(root, "missing.gguf")])
        return err.getvalue()

    def test_think_on_the_08b_prints_the_measured_warning(self) -> None:
        for command in (["run", "--model-id", "qwen-0.8b-no-loop-02", "--think", "hi"],
                        ["chat", "--model-id", "qwen-0.8b-no-loop-02", "--gpu-layers", "0", "--think"],
                        ["serve", "--model-id", "qwen-0.8b-no-loop-02", "--think"]):
            with self.subTest(command=command[0]):
                err = self.launch(command)
                self.assertIn("kilix-llm: warning: thinking mode is not recommended for this 0.8B model", err)
                self.assertIn("56 of 300", err)
                self.assertIn("142 of 300", err)

    def test_no_warning_without_think_or_for_other_models(self) -> None:
        for command in (["run", "--model-id", "qwen-0.8b-no-loop-02", "hi"],
                        ["chat", "--model-id", "qwen-0.8b-no-loop-02", "--gpu-layers", "0"],
                        ["serve", "--model-id", "qwen-0.8b-no-loop-02"],
                        ["serve", "--model-id", "qwen3.5-4b", "--think"]):
            with self.subTest(command=command):
                self.assertNotIn("warning", self.launch(command))

    def test_thinking_is_off_unless_asked(self) -> None:
        parser = build_parser()
        for command in (["chat", "--model-id", "qwen-0.8b-no-loop-02"],
                        ["run", "--model-id", "qwen-0.8b-no-loop-02", "hi"],
                        ["serve", "--model-id", "qwen-0.8b-no-loop-02"]):
            with self.subTest(command=command[0]):
                args = parser.parse_args(command)
                self.assertFalse(args.think)
                build = {"chat": chat_argv, "run": run_argv, "serve": serve_argv}[command[0]]
                argv = build(args, "llama", "model.gguf")
                self.assertEqual(argv[argv.index("--reasoning") + 1], "off")
        self.assertIsNone(models.think_warning("qwen3.5-4b", True))
        self.assertIsNone(models.think_warning("qwen-0.8b-no-loop-02", False))


class CatalogueTests(unittest.TestCase):
    """Bases and the adapter family are catalogue data, not code."""

    def test_lfm2_base_identity_and_licence_notice(self) -> None:
        record = models.spec("lfm2-1.2b-tool")
        self.assertEqual(record["file"], "LFM2-1.2B-Tool-Q8_0.gguf")
        self.assertEqual(record["size"], 1_246_252_768)
        self.assertEqual(
            record["sha256"],
            "bf035d5876eb8ef0b1a87cf0cf30d9717e4cc628d483c5b49b342603d5f2ee67",
        )
        self.assertIn("/LiquidAI/LFM2-1.2B-Tool-GGUF/resolve/"
                      "e7ec4149433c75d4bb53a0742ee1ebde7fcbd183/", record["url"])
        licence = record["licence"]
        self.assertEqual(licence["id"], "LFM-Open-License-1.0")
        self.assertIn("limits commercial use", licence["notice"])
        self.assertIn("No agreement is asked for or recorded", licence["notice"])
        self.assertEqual(licence["file"]["size"], 10_574)
        self.assertEqual(
            licence["file"]["sha256"],
            "4d28ca14dedc0b3d0fcc2b3339f0e79931faa33874f3d24f522183a8fc70068c",
        )
        self.assertIsNone(models.base_id("lfm2-1.2b-tool"))

    def test_binary_01_is_an_unreleased_adapter_over_the_stock_base(self) -> None:
        record = models.spec("qwen-0.8b-binary-01")
        self.assertEqual(models.base_id("qwen-0.8b-binary-01"), "qwen3.5-0.8b")
        self.assertEqual(record["tasks"], ["judge"])
        self.assertFalse(models.released("qwen-0.8b-binary-01"))
        self.assertTrue(models.released("qwen3.5-0.8b"))
        with self.assertRaises(models.ModelError) as raised:
            models.resolve("qwen-0.8b-binary-01")
        self.assertIn("not released yet", str(raised.exception))
        self.assertIn("--model-id qwen3.5-0.8b", str(raised.exception))
        err = io.StringIO()
        with redirect_stderr(err), self.assertRaises(SystemExit) as exited:
            cmd_pull(argparse.Namespace(model_id="qwen-0.8b-binary-01"))
        self.assertEqual(exited.exception.code, 2)
        self.assertIn("not released yet", err.getvalue())

    def test_judge_models_name_a_sized_base(self) -> None:
        judges = [k for k, v in models.MODELS.items() if "judge" in v.get("tasks", ())]
        self.assertEqual(sorted(judges), ["lfm2-1.2b-tool", "qwen-0.8b-binary-01", "qwen3.5-0.8b"])
        for model_id in judges:
            base = models.base_id(model_id) or model_id
            self.assertIn(models.spec(base)["sizer_model"], ("qwen3.5-0.8b-q8_0", "lfm2-1.2b-tool-q8_0"))
            self.assertIn("licence", models.spec(model_id))

    def test_lfm2_pull_saves_the_licence_text_beside_the_weights(self) -> None:
        weights, licence = b"lfm2-weights", b"LFM Open License text"
        fake = {"lfm2-1.2b-tool": dict(
            models.MODELS["lfm2-1.2b-tool"], size=len(weights),
            sha256=hashlib.sha256(weights).hexdigest(),
            licence=dict(models.MODELS["lfm2-1.2b-tool"]["licence"], file=dict(
                models.MODELS["lfm2-1.2b-tool"]["licence"]["file"], size=len(licence),
                sha256=hashlib.sha256(licence).hexdigest())))}

        def curl(url: str, dest: str, resume: bool) -> None:
            with open(dest, "wb") as handle:
                handle.write(licence if url.endswith("/LICENSE") else weights)

        out, err = io.StringIO(), io.StringIO()
        with tempfile.TemporaryDirectory() as models_dir, \
                mock.patch.dict(os.environ, {"KILIX_LLM_MODELS_DIR": models_dir}), \
                mock.patch.dict(models.MODELS, fake), \
                mock.patch("kilix_llm.install._curl", side_effect=curl), \
                redirect_stdout(out), redirect_stderr(err):
            self.assertEqual(cmd_pull(argparse.Namespace(model_id="lfm2-1.2b-tool")), 0)
            saved = os.path.join(models_dir, "LFM2-1.2B-Tool-Q8_0.LICENSE")
            with open(saved, "rb") as handle:
                self.assertEqual(handle.read(), licence)
        self.assertEqual(out.getvalue(), os.path.join(models_dir, "LFM2-1.2B-Tool-Q8_0.gguf") + "\n")
        self.assertIn("LFM-Open-License-1.0 text saved at", err.getvalue())


class PullTests(unittest.TestCase):
    def test_no_loop_pull_points_at_the_wizard(self) -> None:
        err = io.StringIO()
        with redirect_stderr(err):
            with self.assertRaises(SystemExit) as raised:
                cmd_pull(argparse.Namespace(model_id="qwen-0.8b-no-loop-02"))
        self.assertEqual(raised.exception.code, 2)
        text = err.getvalue()
        self.assertIn("kilix models install qwen-0.8b-no-loop-02", text)
        self.assertIn("licence notices", text)

    def test_missing_no_loop_file_does_not_offer_pull(self) -> None:
        with self.assertRaises(models.ModelError) as raised:
            models.resolve("qwen-0.8b-no-loop-02", "/no/such/qwen-0.8b-no-loop-02.gguf")
        text = str(raised.exception)
        self.assertIn("kilix models install qwen-0.8b-no-loop-02", text)
        self.assertNotIn("pull", text)

    def test_other_pulls_still_fetch(self) -> None:
        with tempfile.TemporaryDirectory() as models_dir:
            with mock.patch.dict(os.environ, {"KILIX_LLM_MODELS_DIR": models_dir}), \
                    mock.patch("kilix_llm.install.shutil.which", return_value="/usr/bin/curl"), \
                    mock.patch("kilix_llm.install.subprocess.call", return_value=1) as call, \
                    redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    cmd_pull(argparse.Namespace(model_id="qwen3.5-4b"))
        self.assertEqual(raised.exception.code, 1)
        self.assertEqual(call.call_args.args[0][0], "/usr/bin/curl")


# A catalogue entry shaped like a released adapter over the stock 0.8B. The
# catalogue's own adapters are not released yet, so the loading rules are
# tested on this one.
TEST_ADAPTER = "test-adapter-01"
TEST_ADAPTER_ENTRY = {
    "title": "Test adapter 01 (adapter over Q8)",
    "file": "test-adapter-01.gguf",
    "size": 21_671_008,
    "sha256": "474425eaa49369013eadf82307ae1576c7974b65fb9f6e943d7d2ac2555d7da8",
    "url": "https://example.invalid/test-adapter-01.gguf",
    "publisher": "test",
    "ctx": 8192,
    "quant": "LoRA rank 8 over Q8_0",
    "n_gpu_layers": 99,
    "family": "qwen3.5",
    "base": "qwen3.5-0.8b",
    "wizard_only": True,
}


class AdapterTests(unittest.TestCase):
    """An entry that names a base is a LoRA that loads only over that exact base."""

    ADAPTER = TEST_ADAPTER

    def setUp(self) -> None:
        patcher = mock.patch.dict(models.MODELS, {TEST_ADAPTER: dict(TEST_ADAPTER_ENTRY)})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_stock_base_identity(self) -> None:
        self.assertEqual(models.base_id(self.ADAPTER), "qwen3.5-0.8b")
        base = models.spec("qwen3.5-0.8b")
        self.assertEqual(base["file"], "Qwen3.5-0.8B-Q8_0.gguf")
        self.assertEqual(base["size"], 833_592_096)
        self.assertEqual(
            base["sha256"],
            "37ae482d336108d23516fa35e8e0c4126688d81018b87178a18d752a1357814f",
        )
        self.assertIn("/807dfd9a6fadbf703207aac3bfb0b2b3040d077c/", base["url"])
        self.assertIsNone(models.base_id("qwen-0.8b-no-loop-02"))
        self.assertIsNone(models.base_id("qwen3.5-0.8b"))
        self.assertNotIn("qwen-0.8b-thinking-01", models.MODELS)

    def _store(self, models_dir: str, base_size: int, adapter_size: int = 21_671_008) -> tuple[str, str]:
        base = os.path.join(models_dir, "Qwen3.5-0.8B-Q8_0.gguf")
        adapter = os.path.join(models_dir, "test-adapter-01.gguf")
        for path, size in ((base, base_size), (adapter, adapter_size)):
            with open(path, "wb") as handle:
                handle.truncate(size)
        return base, adapter

    def test_launch_loads_the_adapter_over_the_base(self) -> None:
        with tempfile.TemporaryDirectory() as models_dir, \
                mock.patch.dict(os.environ, {"KILIX_LLM_MODELS_DIR": models_dir}):
            base, adapter = self._store(models_dir, 833_592_096)
            self.assertEqual(models.launch_files(self.ADAPTER), (base, adapter))
            self.assertEqual(models.launch_files("qwen3.5-0.8b"), (base, None))
            parser = build_parser()
            args = parser.parse_args(["serve", "--model-id", self.ADAPTER, "--think"])
            model, args.adapter = models.launch_files(self.ADAPTER)
            argv = serve_argv(args, "llama-server", model)
            self.assertEqual(argv[argv.index("-m") + 1], base)
            self.assertEqual(argv[argv.index("--lora") + 1], adapter)
            for flag, value in (("--temp", "1.0"), ("--top-p", "0.95"), ("--top-k", "20"),
                                ("--presence-penalty", "1.5"), ("--reasoning", "on"),
                                ("--parallel", "1"), ("-ngl", "99")):
                self.assertEqual(argv[argv.index(flag) + 1], value)
            chat = parser.parse_args(["chat", "--model-id", self.ADAPTER, "--gpu-layers", "0", "--think"])
            model, chat.adapter = models.launch_files(self.ADAPTER)
            argv = chat_argv(chat, "llama-cli", model)
            self.assertEqual(argv[argv.index("-m") + 1], base)
            self.assertEqual(argv[argv.index("--lora") + 1], adapter)
            plain = parser.parse_args(["serve", "--model-id", "qwen3.5-0.8b"])
            self.assertNotIn("--lora", serve_argv(plain, "llama-server", base))

    def test_adapter_refuses_a_missing_or_different_base(self) -> None:
        with tempfile.TemporaryDirectory() as models_dir, \
                mock.patch.dict(os.environ, {"KILIX_LLM_MODELS_DIR": models_dir}):
            adapter = os.path.join(models_dir, "test-adapter-01.gguf")
            with open(adapter, "wb") as handle:
                handle.truncate(21_671_008)
            with self.assertRaises(models.ModelError) as raised:
                models.launch_files(self.ADAPTER)
            self.assertIn("kilix models install test-adapter-01", str(raised.exception))
            self.assertNotIn("pull", str(raised.exception))
            # The no-loop-02 file is not the base, and its length says so.
            self._store(models_dir, 833_592_416)
            with self.assertRaises(models.ModelError) as raised:
                models.launch_files(self.ADAPTER)
            self.assertIn("loads only over that exact file", str(raised.exception))

    def test_missing_adapter_is_reported_before_its_base(self) -> None:
        with tempfile.TemporaryDirectory() as models_dir, \
                mock.patch.dict(os.environ, {"KILIX_LLM_MODELS_DIR": models_dir}):
            with self.assertRaises(models.ModelError) as raised:
                models.resolve(self.ADAPTER)
        self.assertIn("Test adapter 01", str(raised.exception).split("\n")[0])
        self.assertIn("test-adapter-01.gguf missing", str(raised.exception))

    def test_verify_checks_the_base_digest(self) -> None:
        with tempfile.TemporaryDirectory() as models_dir, \
                mock.patch.dict(os.environ, {"KILIX_LLM_MODELS_DIR": models_dir}):
            base = os.path.join(models_dir, "Qwen3.5-0.8B-Q8_0.gguf")
            adapter = os.path.join(models_dir, "test-adapter-01.gguf")
            with open(base, "wb") as handle:
                handle.write(b"base-bytes")
            with open(adapter, "wb") as handle:
                handle.write(b"adapter-bytes")
            fake = {
                "qwen3.5-0.8b": dict(models.MODELS["qwen3.5-0.8b"], size=10,
                                     sha256=models.digest_file(base)),
                self.ADAPTER: dict(models.MODELS[self.ADAPTER], size=13,
                                   sha256=models.digest_file(adapter)),
            }
            with mock.patch.dict(models.MODELS, fake):
                self.assertEqual(models.verify(self.ADAPTER), adapter)
                with open(base, "wb") as handle:
                    handle.write(b"other-base")
                with self.assertRaises(models.ModelError) as raised:
                    models.verify(self.ADAPTER)
            self.assertIn("loads only over that exact file", str(raised.exception))

    def test_base_may_live_in_its_own_asset_directory(self) -> None:
        # Content installs each asset into its own directory, so the launcher
        # names the base with --base. Its length is still the catalogue's.
        with tempfile.TemporaryDirectory() as base_dir, tempfile.TemporaryDirectory() as adapter_dir, \
                tempfile.TemporaryDirectory() as empty, \
                mock.patch.dict(os.environ, {"KILIX_LLM_MODELS_DIR": empty}):
            base, adapter = self._store(base_dir, 833_592_096)
            moved = os.path.join(adapter_dir, "test-adapter-01.gguf")
            os.rename(adapter, moved)
            with self.assertRaises(models.ModelError):
                models.launch_files(self.ADAPTER, moved)
            self.assertEqual(models.launch_files(self.ADAPTER, moved, base), (base, moved))
            args = build_parser().parse_args(["serve", "--model-id", self.ADAPTER,
                                              "--model", moved, "--base", base])
            self.assertEqual((args.model, args.base), (moved, base))
            model, args.adapter = models.launch_files(args.model_id, args.model, args.base)
            argv = serve_argv(args, "llama-server", model)
            self.assertEqual(argv[argv.index("-m") + 1], base)
            self.assertEqual(argv[argv.index("--lora") + 1], moved)
            with open(base, "r+b") as handle:
                handle.truncate(833_592_416)
            with self.assertRaises(models.ModelError) as raised:
                models.launch_files(self.ADAPTER, moved, base)
            self.assertIn("loads only over that exact file", str(raised.exception))

    def test_base_is_refused_for_a_full_model(self) -> None:
        with tempfile.NamedTemporaryFile() as model:
            model.truncate(models.spec("qwen3.5-0.8b")["size"])
            with self.assertRaises(models.ModelError) as raised:
                models.launch_files("qwen3.5-0.8b", model.name, model.name)
        self.assertIn("--base applies only to an adapter", str(raised.exception))

    def test_wizard_only_adapter_pull_points_at_the_wizard(self) -> None:
        err = io.StringIO()
        with redirect_stderr(err):
            with self.assertRaises(SystemExit) as raised:
                cmd_pull(argparse.Namespace(model_id=self.ADAPTER))
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("kilix models install test-adapter-01", err.getvalue())
        self.assertIn("licence notices", err.getvalue())


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

    def test_judge_flags_are_accepted_by_llama_server(self) -> None:
        from kilix_llm import judge
        server = built("llama-server")
        if server is None:
            self.skipTest("llama-server is not built")
        args = build_parser().parse_args(["judge", "--model-id", "qwen3.5-0.8b", "x"])
        # The server binds its socket before it loads the model.
        with tempfile.TemporaryDirectory() as scratch:
            argv = judge.server_argv(args, server, self.MISSING_MODEL, "/nonexistent/adapter.gguf",
                                     1024, os.path.join(scratch, "engine.sock"))
            self.assert_argv_parses(argv)

    def test_serve_flags_are_accepted_by_llama_server(self) -> None:
        server = built("llama-server")
        if server is None:
            self.skipTest("llama-server is not built")
        args = build_parser().parse_args(["serve", "--port", "0"])
        self.assert_argv_parses(serve_argv(args, server, self.MISSING_MODEL))

    def test_adapter_flag_is_accepted_by_both_binaries(self) -> None:
        cli, server = built("llama-cli"), built("llama-server")
        if not cli or not server:
            self.skipTest("CPU runtime is not built")
        parser = build_parser()
        for command, binary, builder, extra in (
            ("run", cli, run_argv, ["hi"]),
            ("serve", server, serve_argv, ["--port", "0"]),
        ):
            with self.subTest(command=command):
                args = parser.parse_args([command, "--model-id", "qwen3.5-0.8b",
                                          "--gpu-layers", "0", "--think", *extra])
                args.adapter = "/nonexistent/kilix-llm-test-adapter.gguf"
                argv = builder(args, binary, self.MISSING_MODEL)
                self.assertIn("--lora", argv)
                self.assert_argv_parses(argv)

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
