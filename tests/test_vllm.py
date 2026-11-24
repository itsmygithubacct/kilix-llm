"""The vLLM engine: catalogue, store, sizing, GPU selection, build, and argv.

Nothing here installs vLLM, touches a GPU, or downloads a checkpoint. `uv`,
`curl` and `nvidia-smi` are stand-in scripts on PATH that record how they were
called; the GPU listings are the driver's real output from two machines.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import stat
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from kilix_llm import cli, models, paths, runtime, vllm_oneshot, vllm_runtime

DORY = "dory1.0-4b-nvfp4"
LAUNCHER = os.path.join(paths.REPO_DIR, "kilix-llm")

# `nvidia-smi --query-gpu=index,name,compute_cap,memory.total,memory.reserved,
# memory.used --format=csv,noheader,nounits`, captured 2026-11-24.
SMI_QUADRO = "0, Quadro RTX 3000, 7.5, 6144, 392, 1713\n"
SMI_RTX3070 = "0, NVIDIA GeForce RTX 3070, 8.6, 8192, 334, 3825\n"

FAKE_UV = r"""#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >> "$FAKE_UV_LOG"
case "$1 ${2:-}" in
  "venv "*)
    target=${@: -1}
    rm -rf "$target"; mkdir -p "$target/bin"
    printf '#!/bin/sh\necho "%s"\n' "${FAKE_VLLM_VERSION:-0.34.0}" > "$target/bin/python"
    printf '#!/bin/sh\n' > "$target/bin/vllm"
    chmod +x "$target/bin/python" "$target/bin/vllm" ;;
  "pip install") exit "${FAKE_UV_PIP_STATUS:-0}" ;;
  "pip freeze") echo "vllm==${FAKE_VLLM_VERSION:-0.34.0}" ;;
  *) exit 64 ;;
esac
"""

FAKE_CURL = r"""#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >> "$FAKE_CURL_LOG"
[ "${FAKE_CURL_STATUS:-0}" = 0 ] || exit "$FAKE_CURL_STATUS"
out=; url=${@: -1}
while [ "$#" -gt 0 ]; do [ "$1" = -o ] && out=$2; shift; done
cp "$FAKE_CURL_SOURCE/${url##*/}" "$out"
"""


def write_script(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def small_catalogue(source: Path) -> dict:
    """A two-file stand-in entry whose bytes exist under `source`."""
    files = {"config.json": b'{"model_type": "dory"}\n', "model.safetensors": b"w" * 4096}
    record = dict(vllm_runtime.MODELS[DORY])
    record["files"] = {}
    for name, data in files.items():
        (source / name).write_bytes(data)
        record["files"][name] = (len(data), hashlib.sha256(data).hexdigest())
    return {DORY: record}


class CatalogueTests(unittest.TestCase):
    def test_dory_is_pinned_by_revision_size_and_digest(self) -> None:
        record = vllm_runtime.spec(DORY)
        self.assertEqual(record["revision"], "17065a6849266dd478221735be7747689a0a4331")
        self.assertEqual(
            record["files"]["model.safetensors"],
            (3_202_498_032, "ca08dec5d70b653c03e57903dd33725ec8cc80133c53a1bd9c83f99dd03f3e46"),
        )
        self.assertEqual(len(record["files"]), 8)
        for name, (size, digest) in record["files"].items():
            self.assertGreater(size, 0, name)
            self.assertRegex(digest, r"^[0-9a-f]{64}$", name)
        self.assertEqual(
            vllm_runtime.file_url(DORY, "config.json"),
            "https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4/resolve/"
            "17065a6849266dd478221735be7747689a0a4331/config.json",
        )

    def test_ids_do_not_collide_with_the_gguf_catalogue(self) -> None:
        self.assertEqual(set(vllm_runtime.MODELS) & set(models.MODELS), set())
        self.assertTrue(vllm_runtime.is_vllm(DORY))
        self.assertFalse(vllm_runtime.is_vllm("qwen3.5-4b"))
        with self.assertRaises(models.ModelError):
            vllm_runtime.spec("qwen3.5-4b")

    def test_store_path_carries_the_id_and_revision(self) -> None:
        with mock.patch.dict(os.environ, {"KILIX_LLM_MODELS_DIR": "/store"}):
            self.assertEqual(
                vllm_runtime.model_dir(DORY),
                "/store/dory1.0-4b-nvfp4/17065a6849266dd478221735be7747689a0a4331",
            )
            self.assertEqual(vllm_runtime.model_dir(DORY, "/elsewhere"), "/elsewhere")


class KvCacheTests(unittest.TestCase):
    """Literal byte counts: 4,096 bytes per token per attention sublayer.

    A sliding sublayer holds 2,064 tokens once the context passes its window:
    1,023 behind the current token, two 512-token batches in flight, rounded
    up to blocks of 16, and one block more.
    """

    def test_per_loop_grows_with_loops_and_last_loop_does_not(self) -> None:
        size = vllm_runtime.kv_cache_bytes
        self.assertEqual(size(DORY, 8192, 1, "per_loop"), 530_251_776)
        self.assertEqual(size(DORY, 8192, 2, "per_loop"), 824_836_096)
        self.assertEqual(size(DORY, 8192, 4, "per_loop"), 1_414_004_736)
        self.assertEqual(size(DORY, 8192, 2, "last_loop"), 530_251_776)
        self.assertEqual(size(DORY, 8192, 4, "last_loop"), 530_251_776)

    def test_sliding_layers_stop_growing_past_their_window(self) -> None:
        size = vllm_runtime.kv_cache_bytes
        self.assertEqual(size(DORY, 32768, 1, "per_loop"), 1_436_221_440)
        self.assertEqual(size(DORY, 32768, 2, "per_loop"), 2_234_122_240)
        # Below the window every sublayer holds the whole context, in blocks.
        self.assertEqual(size(DORY, 512, 1, "per_loop"), 77_266_944)
        self.assertEqual(size(DORY, 1000, 1, "per_loop"), 150_405_120)

    def test_matches_what_vllm_asked_for_on_a_real_start(self) -> None:
        # vLLM 0.34.0 refused 379,256,832 bytes for 4,096 tokens as 0.35 GiB
        # against 0.36 needed, and accepted the same with 16 MiB more.
        self.assertEqual(vllm_runtime.kv_cache_bytes(DORY, 4096, 2, "last_loop"), 379_256_832)
        self.assertEqual(
            vllm_runtime.kv_cache_reservation(DORY, 4096, 2, "last_loop"), 396_034_048)

    def test_serving_arguments_are_bounded(self) -> None:
        vllm_runtime.check_serving_args(DORY, 1, "per_loop", 256)
        vllm_runtime.check_serving_args(DORY, 8, "last_loop", 8192)
        for loops, mode, ctx in ((0, "per_loop", 8192), (9, "per_loop", 8192),
                                 (2, "every_loop", 8192), (2, "per_loop", 255)):
            with self.assertRaises(models.ModelError, msg=(loops, mode, ctx)):
                vllm_runtime.check_serving_args(DORY, loops, mode, ctx)


class GpuTests(unittest.TestCase):
    def test_parses_the_drivers_listing(self) -> None:
        self.assertEqual(vllm_runtime.parse_gpus(SMI_QUADRO), [{
            "index": 0, "device": "CUDA0", "name": "Quadro RTX 3000",
            "compute_cap": 75, "total_mib": 6144, "used_mib": 1713, "free_mib": 4039,
        }])
        self.assertEqual(vllm_runtime.parse_gpus(SMI_RTX3070)[0]["free_mib"], 4033)
        both = vllm_runtime.parse_gpus(SMI_QUADRO + SMI_RTX3070.replace("0, NV", "1, NV"))
        self.assertEqual([gpu["device"] for gpu in both], ["CUDA0", "CUDA1"])
        self.assertEqual(both[1]["compute_cap"], 86)
        self.assertEqual(vllm_runtime.parse_gpus("No devices were found\n"), [])

    def test_selection_matches_the_vulkan_rules(self) -> None:
        one = vllm_runtime.parse_gpus(SMI_QUADRO)
        two = vllm_runtime.parse_gpus(SMI_QUADRO + SMI_RTX3070.replace("0, NV", "1, NV"))
        self.assertEqual(vllm_runtime.select_gpu(None, one)["name"], "Quadro RTX 3000")
        self.assertEqual(vllm_runtime.select_gpu("CUDA1", two)["name"], "NVIDIA GeForce RTX 3070")
        with self.assertRaisesRegex(runtime.RuntimeError_, "CUDA3.*unavailable"):
            vllm_runtime.select_gpu("CUDA3", two)
        with self.assertRaisesRegex(runtime.RuntimeError_, "choose --device from: CUDA0 .*CUDA1"):
            vllm_runtime.select_gpu(None, two)
        with self.assertRaisesRegex(runtime.RuntimeError_, "no NVIDIA GPU"):
            vllm_runtime.select_gpu(None, [])

    def test_listing_runs_nvidia_smi_and_reports_its_failure(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            smi = Path(raw) / "nvidia-smi"
            write_script(smi, "#!/bin/sh\nprintf '%s' \"$FAKE_SMI_OUT\"\nexit \"$FAKE_SMI_STATUS\"\n")
            env = {"PATH": raw + os.pathsep + os.environ["PATH"],
                   "FAKE_SMI_OUT": SMI_RTX3070, "FAKE_SMI_STATUS": "0"}
            with mock.patch.dict(os.environ, env):
                self.assertEqual(vllm_runtime.gpus()[0]["total_mib"], 8192)
            env["FAKE_SMI_STATUS"] = "9"
            with mock.patch.dict(os.environ, env):
                with self.assertRaisesRegex(runtime.RuntimeError_, "cannot list NVIDIA GPUs"):
                    vllm_runtime.gpus()
            with mock.patch.dict(os.environ, {"PATH": raw + "/absent"}):
                with self.assertRaisesRegex(runtime.RuntimeError_, "nvidia-smi is not on PATH"):
                    vllm_runtime.gpus()


class ServingTests(unittest.TestCase):
    """Pre-flight and the command lines, on the two captured GPU listings."""

    def setUp(self) -> None:
        self.quadro = vllm_runtime.parse_gpus(SMI_QUADRO)[0]
        self.rtx3070 = vllm_runtime.parse_gpus(SMI_RTX3070)[0]

    def request(self, gpu: dict, ctx: int, loops: int, mode: str) -> dict:
        return {
            "model_id": DORY, "model": "/store/dory", "ctx": ctx, "loops": loops, "mode": mode,
            "gpu": gpu,
            "kv_cache_memory_bytes": vllm_runtime.kv_cache_reservation(DORY, ctx, loops, mode),
        }

    def test_accepts_the_request_that_ran_beside_another_server(self) -> None:
        # 4,096 tokens, one cache for both loops, on 4,039 MiB free: 3,349
        # weights, 378 cache and 300 runtime make 4,027.
        self.assertEqual(vllm_runtime.gpu_need_mib(DORY, 4096, 2, "last_loop"), 4027)
        with mock.patch.object(vllm_runtime, "c_compiler", return_value="/usr/bin/cc"):
            vllm_runtime.preflight(self.quadro, DORY, 4096, 2, "last_loop")

    def test_refuses_what_does_not_fit_and_says_what_would(self) -> None:
        with mock.patch.object(vllm_runtime, "c_compiler", return_value="/usr/bin/cc"), \
                self.assertRaises(runtime.RuntimeError_) as caught:
            vllm_runtime.preflight(self.quadro, DORY, 8192, 2, "per_loop")
        self.assertEqual(str(caught.exception), (
            "Nemotron Labs Dory1.0 4B NVFP4 (GPU, vLLM) needs about 4452 MiB on CUDA0 "
            "(3349 weights, "
            "803 cache for 8192 tokens at 2 loop(s) per_loop, 300 runtime), but only "
            "4039 of 6144 MiB is free to it.\n"
            "Try a smaller --ctx, or --kv-cache last_loop, or stopping what holds the "
            "other 1713 MiB."))

    def test_an_older_gpu_needs_a_c_compiler_and_a_newer_one_does_not(self) -> None:
        with mock.patch.object(vllm_runtime, "c_compiler", return_value=None):
            with self.assertRaisesRegex(
                    runtime.RuntimeError_,
                    r"needs a C compiler to run on CUDA0 \(Quadro RTX 3000\)"):
                vllm_runtime.preflight(self.quadro, DORY, 4096, 2, "last_loop")
            vllm_runtime.preflight(self.rtx3070, DORY, 4096, 2, "last_loop")

    def test_finds_the_compiler_cc_names_or_one_on_path(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            write_script(Path(raw) / "gcc", "#!/bin/sh\n")
            write_script(Path(raw) / "mycc", "#!/bin/sh\n")
            with mock.patch.dict(os.environ, {"PATH": raw}):
                os.environ.pop("CC", None)
                self.assertEqual(vllm_runtime.c_compiler(), raw + "/gcc")
                os.environ["CC"] = "mycc"
                self.assertEqual(vllm_runtime.c_compiler(), raw + "/mycc")
                os.environ["CC"] = "absent-cc"
                self.assertIsNone(vllm_runtime.c_compiler())
            with mock.patch.dict(os.environ, {"PATH": raw + "/absent"}):
                os.environ.pop("CC", None)
                self.assertIsNone(vllm_runtime.c_compiler())

    def test_serve_command_line(self) -> None:
        request = self.request(self.quadro, 4096, 2, "last_loop")
        with mock.patch.dict(os.environ, {"KILIX_LLM_VLLM_DIR": "/x/rt"}):
            argv = vllm_runtime.serve_argv(request, "127.0.0.1", 18180, think=False)
        self.assertEqual(argv, [
            "/x/rt/bin/vllm", "serve", "/store/dory",
            "--served-model-name", "dory1.0-4b",
            "--host", "127.0.0.1", "--port", "18180",
            "--hf-overrides",
            '{"n_recurrent_loops": 2, "recurrent_kv_cache_mode": "last_loop"}',
            "--max-model-len", "4096",
            "--max-num-seqs", "1",
            "--max-num-batched-tokens", "512",
            "--gpu-memory-utilization", "0.54",
            "--kv-cache-memory-bytes", "396034048",
            "--reasoning-parser", "nemotron_v3",
            "--enable-auto-tool-choice",
            "--tool-call-parser", "qwen3_coder",
            "--default-chat-template-kwargs", '{"enable_thinking": false}',
        ])
        self.assertEqual(
            vllm_runtime.memory_gate(self.request(self.rtx3070, 4096, 2, "last_loop")), "0.40")

    def test_run_command_line_and_environment(self) -> None:
        request = self.request(self.rtx3070, 8192, 4, "per_loop")
        with mock.patch.dict(os.environ, {"KILIX_LLM_VLLM_DIR": "/x/rt"}):
            argv = vllm_runtime.run_argv(request, "hello", 64, think=True)
            env = vllm_runtime.environment(request)
        self.assertEqual(argv[:2], ["/x/rt/bin/python", vllm_runtime.ONESHOT])
        self.assertEqual(json.loads(argv[2]), {
            "model": "/store/dory", "served_name": "dory1.0-4b",
            "hf_overrides": {"n_recurrent_loops": 4, "recurrent_kv_cache_mode": "per_loop"},
            "ctx": 8192, "batch_tokens": 512, "gpu_memory_utilization": 0.4,
            "kv_cache_memory_bytes": 1_430_781_952, "max_tokens": 64,
            "prompt": "hello", "think": True,
        })
        self.assertEqual(
            {name: env[name] for name in (
                "CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
                "VLLM_NO_USAGE_STATS", "DO_NOT_TRACK")},
            {"CUDA_VISIBLE_DEVICES": "0", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
             "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1"})


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.source = root / "source"
        self.source.mkdir()
        self.store = root / "store"
        self.bin = root / "bin"
        self.bin.mkdir()
        self.log = root / "curl.log"
        write_script(self.bin / "curl", FAKE_CURL)
        catalogue = small_catalogue(self.source)
        patches = [
            mock.patch.dict(vllm_runtime.MODELS, catalogue),
            mock.patch.dict(os.environ, {
                "KILIX_LLM_MODELS_DIR": str(self.store),
                "PATH": str(self.bin) + os.pathsep + os.environ["PATH"],
                "FAKE_CURL_LOG": str(self.log),
                "FAKE_CURL_SOURCE": str(self.source),
            }),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.dest = Path(vllm_runtime.model_dir(DORY))

    def pull(self) -> str:
        with redirect_stderr(io.StringIO()) as err:
            path = vllm_runtime.pull(DORY)
        self.notice = err.getvalue()
        return path

    def test_missing_checkpoint_names_pull(self) -> None:
        with self.assertRaisesRegex(models.ModelError, "not downloaded.*config.json missing"):
            vllm_runtime.resolve(DORY)
        with self.assertRaisesRegex(models.ModelError, "kilix-llm pull --model-id dory1.0-4b-nvfp4"):
            vllm_runtime.resolve(DORY)

    def test_pull_fetches_each_pinned_file_and_states_the_licence(self) -> None:
        self.assertEqual(self.pull(), str(self.dest))
        calls = self.log.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(calls), 2)
        for call, name in zip(calls, ("config.json", "model.safetensors")):
            self.assertEqual(
                call,
                f"-fL --retry 3 -C - -o {self.dest / name} "
                "https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4/resolve/"
                f"17065a6849266dd478221735be7747689a0a4331/{name}",
            )
        self.assertIn("OpenMDW-1.1: https://openmdw.ai/license/1-1/", self.notice)
        self.assertEqual(vllm_runtime.verify(DORY), str(self.dest))

    def test_pull_skips_complete_files_and_resumes_short_ones(self) -> None:
        self.pull()
        self.log.unlink()
        (self.dest / "model.safetensors").write_bytes(b"w" * 100)
        with self.assertRaisesRegex(models.ModelError, "is 100 bytes, expected 4096"):
            vllm_runtime.resolve(DORY)
        self.pull()
        calls = self.log.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(calls), 1)
        self.assertIn("model.safetensors", calls[0])

    def test_verify_rejects_changed_bytes_of_the_right_length(self) -> None:
        self.pull()
        (self.dest / "model.safetensors").write_bytes(b"x" * 4096)
        self.assertEqual(vllm_runtime.resolve(DORY), str(self.dest))
        with self.assertRaisesRegex(models.ModelError, "model.safetensors sha256 is"):
            vllm_runtime.verify(DORY)
        # pull trusts nothing it did not just verify.
        with self.assertRaisesRegex(models.ModelError, "sha256 is"):
            self.pull()

    def test_pull_reports_a_failed_download(self) -> None:
        with mock.patch.dict(os.environ, {"FAKE_CURL_STATUS": "22"}):
            with self.assertRaisesRegex(models.ModelError, "could not fetch config.json"):
                self.pull()
        self.assertFalse((self.dest / "config.json").exists())


class BuildTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.bin = root / "bin"
        self.bin.mkdir()
        self.log = root / "uv.log"
        self.log.write_text("", encoding="utf-8")
        self.target = root / "runtimes" / "vllm"
        write_script(self.bin / "uv", FAKE_UV)
        self.env = {
            "KILIX_LLM_VLLM_DIR": str(self.target),
            "PATH": str(self.bin) + os.pathsep + os.environ["PATH"],
            "FAKE_UV_LOG": str(self.log),
        }

    def build(self, extra: dict | None = None, **kwargs) -> int:
        with mock.patch.dict(os.environ, {**self.env, **(extra or {})}):
            with redirect_stderr(io.StringIO()):
                return vllm_runtime.build(**kwargs)

    def calls(self) -> list[str]:
        return self.log.read_text(encoding="utf-8").splitlines()

    def current(self) -> bool:
        with mock.patch.dict(os.environ, self.env):
            return vllm_runtime.build_current()

    def test_build_installs_the_pinned_release_and_stamps_it(self) -> None:
        self.assertFalse(self.current())
        self.assertEqual(self.build(), 0)
        self.assertEqual(self.calls(), [
            f"venv --managed-python --python 3.12 --clear {self.target}",
            f"pip install --python {self.target}/bin/python "
            "--exclude-newer 2026-11-17T12:00:00Z vllm==0.34.0",
            f"pip freeze --python {self.target}/bin/python",
        ])
        stamp = json.loads((self.target / "kilix-llm-runtime.json").read_text(encoding="utf-8"))
        self.assertEqual(
            stamp, {"vllm": "0.34.0", "python": "3.12", "exclude_newer": "2026-11-17T12:00:00Z"})
        self.assertEqual((self.target / "installed.txt").read_text(encoding="utf-8"), "vllm==0.34.0\n")
        self.assertTrue(self.current())

    def test_if_needed_is_a_no_op_once_current_and_plain_build_reinstalls(self) -> None:
        self.build()
        self.assertEqual(self.build(if_needed=True), 0)
        self.assertEqual(len(self.calls()), 3)
        self.assertEqual(self.build(), 0)
        self.assertEqual(len(self.calls()), 6)

    def test_a_failed_install_leaves_no_stamp(self) -> None:
        self.build()
        self.assertEqual(self.build({"FAKE_UV_PIP_STATUS": "7"}), 7)
        self.assertFalse((self.target / "kilix-llm-runtime.json").exists())
        self.assertFalse(self.current())

    def test_a_different_installed_version_is_refused(self) -> None:
        self.assertEqual(self.build({"FAKE_VLLM_VERSION": "0.33.0"}), 1)
        self.assertFalse(self.current())

    def test_a_stamp_for_another_pin_is_not_current(self) -> None:
        self.build()
        stamp = self.target / "kilix-llm-runtime.json"
        stamp.write_text(json.dumps(
            {"vllm": "0.33.0", "python": "3.12", "exclude_newer": "2026-11-17T12:00:00Z"}),
            encoding="utf-8")
        self.assertFalse(self.current())
        stamp.write_text("not json", encoding="utf-8")
        self.assertFalse(self.current())

    def test_build_needs_uv_and_serving_needs_the_build(self) -> None:
        with mock.patch.dict(os.environ, {**self.env, "PATH": str(self.bin / "absent")}):
            with self.assertRaisesRegex(runtime.RuntimeError_, "uv, which is not on PATH"):
                vllm_runtime.build()
        with mock.patch.dict(os.environ, self.env):
            with self.assertRaisesRegex(runtime.RuntimeError_, "kilix-llm build --backend vllm"):
                vllm_runtime.require_built()

    def test_default_runtime_directory_names_the_version(self) -> None:
        with mock.patch.dict(os.environ, {"GPU_TERMINAL_HOME": "/gt"}):
            os.environ.pop("KILIX_LLM_VLLM_DIR", None)
            self.assertEqual(vllm_runtime.runtime_dir(), "/gt/kilix-llm/runtimes/vllm-0.34.0")
            self.assertEqual(vllm_runtime.vllm_path(), "/gt/kilix-llm/runtimes/vllm-0.34.0/bin/vllm")


class OneshotTests(unittest.TestCase):
    def test_reasoning_is_separated_from_the_reply(self) -> None:
        split = vllm_oneshot.split_reasoning
        self.assertEqual(split("<think>\n2+2 is 4.\n</think>\n\nFour."), ("2+2 is 4.", "Four."))
        self.assertEqual(split("2+2 is 4.\n</think>\n\nFour."), ("2+2 is 4.", "Four."))
        self.assertEqual(split("<think></think>Four."), ("", "Four."))
        self.assertEqual(split(" Four.\n"), ("", "Four."))

    def test_the_script_does_not_import_the_launcher(self) -> None:
        source = Path(vllm_runtime.ONESHOT).read_text(encoding="utf-8")
        self.assertIsNone(re.search(r"^\s*(from|import)\s+(\.|kilix_llm)", source, re.MULTILINE))
        with redirect_stderr(io.StringIO()) as err:
            self.assertEqual(vllm_oneshot.main(["vllm_oneshot.py"]), 2)
        self.assertIn("usage", err.getvalue())


def parse(*argv: str) -> argparse.Namespace:
    return cli.build_parser().parse_args(list(argv))


class CommandLineTests(unittest.TestCase):
    def run_cli(self, *argv: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            try:
                code = cli.main(list(argv))
            except SystemExit as stop:
                code = stop.code
        return code, out.getvalue(), err.getvalue()

    def test_every_model_command_accepts_the_vllm_id(self) -> None:
        for command in ("path", "verify", "pull", "chat", "serve"):
            self.assertEqual(parse(command, "--model-id", DORY).model_id, DORY)
        args = parse("run", "--model-id", DORY, "--loops", "4", "--kv-cache", "last_loop", "Hi")
        self.assertEqual((args.loops, args.kv_cache, args.prompt), (4, "last_loop", "Hi"))
        for command in ("build", "devices", "doctor"):
            self.assertEqual(parse(command, "--backend", "vllm").backend, "vllm")
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse("serve", "--model-id", DORY, "--kv-cache", "every_loop")

    def test_path_and_missing_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            with mock.patch.dict(os.environ, {"KILIX_LLM_MODELS_DIR": raw}):
                code, out, _err = self.run_cli("path", "--model-id", DORY)
                self.assertEqual(
                    (code, out),
                    (0, f"{raw}/dory1.0-4b-nvfp4/17065a6849266dd478221735be7747689a0a4331\n"))
                for command in (("verify",), ("serve",), ("run", "Hi")):
                    code, out, err = self.run_cli(command[0], "--model-id", DORY, *command[1:])
                    self.assertEqual((code, out), (2, ""), command)
                    self.assertIn("kilix-llm pull --model-id dory1.0-4b-nvfp4", err)

    def test_flags_from_the_other_engine_are_refused(self) -> None:
        code, _out, err = self.run_cli("serve", "--model-id", DORY, "--backend", "vulkan")
        self.assertEqual(code, 4)
        self.assertIn("runs on the vllm backend, not vulkan", err)
        code, _out, err = self.run_cli("serve", "--model-id", DORY, "--gpu-layers", "0")
        self.assertEqual(code, 4)
        self.assertIn("--gpu-layers does not apply", err)
        for flags in (("--loops", "2"), ("--kv-cache", "last_loop")):
            for command in (("serve",), ("run", "Hi"), ("chat",)):
                code, _out, err = self.run_cli(
                    command[0], "--model-id", "qwen3.5-4b", *flags, *command[1:])
                self.assertEqual(code, 2, (flags, command))
                self.assertIn("apply to looped vLLM models", err)

    def test_chat_names_serve_instead(self) -> None:
        code, out, err = self.run_cli("chat", "--model-id", DORY)
        self.assertEqual((code, out), (2, ""))
        self.assertIn("kilix-llm serve --model-id dory1.0-4b-nvfp4", err)

    def test_build_dispatches_to_the_vllm_install(self) -> None:
        with mock.patch.object(vllm_runtime, "build", return_value=0) as build:
            self.assertEqual(self.run_cli("build", "--backend", "vllm", "--if-needed")[0], 0)
        build.assert_called_once_with(if_needed=True)
        code, _out, err = self.run_cli("build", "--backend", "vllm", "--keep-build-tree")
        self.assertEqual(code, 2)
        self.assertIn("--keep-build-tree applies to the llama.cpp builds", err)

    def test_devices_and_doctor_describe_the_gpu(self) -> None:
        listing = vllm_runtime.parse_gpus(SMI_QUADRO)
        line = "CUDA0: Quadro RTX 3000 (compute capability 7.5, 4039 of 6144 MiB free)"
        with tempfile.TemporaryDirectory() as raw:
            env = {"KILIX_LLM_MODELS_DIR": raw, "KILIX_LLM_VLLM_DIR": raw + "/runtime"}
            with mock.patch.dict(os.environ, env), \
                    mock.patch.object(vllm_runtime, "gpus", return_value=listing):
                self.assertEqual(self.run_cli("devices", "--backend", "vllm"), (0, line + "\n", ""))
                code, out, _err = self.run_cli("doctor", "--backend", "vllm")
            self.assertIn(code, (0, 4))
            self.assertIn("vllm:       0.34.0 on Python 3.12 (not installed)", out)
            self.assertIn(f"runtime:    {raw}/runtime", out)
            self.assertIn("gpu:        " + line, out)
            self.assertIn("dory1.0-4b-nvfp4: Nemotron Labs Dory1.0 4B NVFP4 (GPU, vLLM) "
                          "is not downloaded", out)
            with mock.patch.dict(os.environ, env), \
                    mock.patch.object(vllm_runtime, "gpus", return_value=[]):
                code, _out, err = self.run_cli("devices", "--backend", "vllm")
            self.assertEqual(code, 4)
            self.assertIn("no NVIDIA GPU is available to vLLM", err)

    def test_the_default_doctor_lists_the_vllm_model(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            result = subprocess.run(
                [LAUNCHER, "doctor"], capture_output=True, text=True, check=False,
                env={**os.environ, "KILIX_LLM_MODELS_DIR": raw},
            )
        self.assertIn(result.returncode, (0, 4), result.stderr)
        self.assertIn("dory1.0-4b-nvfp4: Nemotron Labs Dory1.0 4B NVFP4 (GPU, vLLM) "
                      "is not downloaded", result.stdout)


if __name__ == "__main__":
    unittest.main()
