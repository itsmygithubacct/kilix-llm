"""Provisioning compiles nothing: the first run, chat or serve builds once.

The build script is exercised against a stand-in checkout whose `cmake`
writes the three binaries, so the stamp, the lock, `--if-needed`, pruning
and the `KILIX_LLM_BUILD_DIR` exemption run for real without compiling.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from kilix_llm import cli, paths, runtime

FAKE_CMAKE = r"""#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >> "$FAKE_CMAKE_LOG"
if [ "$1" = --build ]; then
  root=$2; shift 2
  mkdir -p "$root/bin" "$root/obj"
  : > "$root/obj/ggml.o"
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --target) shift; for name in "$@"; do printf '#!/bin/sh\n' > "$root/bin/$name"; chmod +x "$root/bin/$name"; done; break ;;
      *) shift ;;
    esac
  done
  [ "${FAKE_CMAKE_FAIL:-0}" = 0 ] || exit 3
else
  while [ "$#" -gt 0 ]; do [ "$1" = -B ] && { mkdir -p "$2/CMakeFiles"; : > "$2/CMakeCache.txt"; }; shift; done
fi
"""


@unittest.skipUnless(shutil.which("rsync") and shutil.which("flock"), "rsync and flock are required")
class BuildScriptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="kilix-llm-build-test-"))
        self.addCleanup(shutil.rmtree, self.tmp)
        self.repo = self.tmp / "checkout"
        (self.repo / "scripts").mkdir(parents=True)
        (self.repo / "third_party" / "llama.cpp").mkdir(parents=True)
        (self.repo / "patches").mkdir()
        shutil.copy2(runtime.BUILD_SCRIPT, self.repo / "scripts" / "build-runtime.sh")
        (self.repo / "third_party" / "llama.cpp" / "CMakeLists.txt").write_text("project(x)\n")
        self.pin = self.repo / "third_party" / "llama.cpp.pin"
        self.pin.write_text("url=https://example.invalid/llama.cpp.git\ncommit=" + "a" * 40 + "\n")
        tools = self.tmp / "tools"
        tools.mkdir()
        cmake = tools / "cmake"
        cmake.write_text(FAKE_CMAKE)
        cmake.chmod(cmake.stat().st_mode | stat.S_IXUSR)
        self.log = self.tmp / "cmake.log"
        self.env = {
            **{k: v for k, v in os.environ.items() if k != "KILIX_LLM_BUILD_DIR"},
            "PATH": f"{tools}{os.pathsep}{os.environ['PATH']}",
            "FAKE_CMAKE_LOG": str(self.log),
        }
        self.build = self.repo / "build" / "llama.cpp"

    def script(self, *args: str, **env: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [str(self.repo / "scripts" / "build-runtime.sh"), *args],
            env={**self.env, **env}, capture_output=True, text=True, check=False,
        )

    def builds(self) -> int:
        if not self.log.exists():
            return 0
        return sum(line.startswith("--build") for line in self.log.read_text().splitlines())

    def test_first_build_keeps_only_bin_and_a_stamp(self) -> None:
        self.assertEqual(self.script("--check").returncode, 1)
        result = self.script("--if-needed")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.builds(), 1)
        self.assertEqual(sorted(p.name for p in self.build.iterdir()), ["bin"])
        self.assertFalse((self.repo / "build" / "llama.cpp-src").exists())
        for name in runtime.BINARIES:
            self.assertTrue(os.access(self.build / "bin" / name, os.X_OK), name)
        self.assertIn("pin-sha256=", (self.build / "bin" / ".kilix-llm-build").read_text())
        self.assertEqual(self.script("--check").returncode, 0)

    def test_if_needed_is_a_no_op_until_the_pin_changes(self) -> None:
        self.assertEqual(self.script("--if-needed").returncode, 0)
        self.assertEqual(self.script("--if-needed").returncode, 0)
        self.assertEqual(self.builds(), 1)
        self.pin.write_text(self.pin.read_text().replace("a" * 40, "b" * 40))
        self.assertEqual(self.script("--check").returncode, 1)
        self.assertEqual(self.script("--if-needed").returncode, 0)
        self.assertEqual(self.builds(), 2)

    def test_a_failed_build_is_never_current(self) -> None:
        result = self.script("--if-needed", FAKE_CMAKE_FAIL="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.build / "bin" / ".kilix-llm-build").exists())
        self.assertEqual(self.script("--check").returncode, 1)

    def test_keep_build_tree_and_a_chosen_build_dir_are_not_pruned(self) -> None:
        self.assertEqual(self.script("--keep-build-tree").returncode, 0)
        self.assertTrue((self.build / "CMakeCache.txt").exists())
        self.assertTrue((self.repo / "build" / "llama.cpp-src").exists())
        chosen = self.tmp / "chosen"
        chosen.mkdir()
        (chosen / "unrelated.txt").write_text("keep me\n")
        result = self.script("--if-needed", KILIX_LLM_BUILD_DIR=str(chosen))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((chosen / "unrelated.txt").read_text(), "keep me\n")
        self.assertTrue((chosen / "CMakeCache.txt").exists())


class EnsureBuiltTests(unittest.TestCase):
    def test_current_build_runs_nothing(self) -> None:
        with mock.patch.object(runtime, "build_current", return_value=True), \
                mock.patch.object(runtime.subprocess, "call") as call:
            runtime.ensure_built("cpu")
        call.assert_not_called()

    def test_first_use_builds_once_with_if_needed(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False), \
                mock.patch.object(runtime, "build_current", return_value=False), \
                mock.patch.object(runtime, "missing_build_packages", return_value=[]), \
                mock.patch.object(runtime.subprocess, "call", return_value=0) as call:
            os.environ.pop("KILIX_LLM_NO_AUTO_BUILD", None)
            runtime.ensure_built("cpu")
        call.assert_called_once()
        self.assertEqual(call.call_args.args[0],
                         [runtime.BUILD_SCRIPT, "--backend", "cpu", "--if-needed"])

    def test_opt_out_missing_tools_and_failures_are_reported(self) -> None:
        with mock.patch.dict(os.environ, {"KILIX_LLM_NO_AUTO_BUILD": "1"}), \
                mock.patch.object(runtime.subprocess, "call") as call:
            runtime.ensure_built("cpu")
        call.assert_not_called()
        with mock.patch.dict(os.environ, {"KILIX_LLM_NO_AUTO_BUILD": "0"}), \
                mock.patch.object(runtime, "build_current", return_value=False):
            with mock.patch.object(runtime, "missing_build_packages", return_value=["cmake"]):
                with self.assertRaisesRegex(runtime.RuntimeError_, "apt-get install cmake"):
                    runtime.ensure_built("cpu")
            with mock.patch.object(runtime, "missing_build_packages", return_value=[]), \
                    mock.patch.object(runtime.subprocess, "call", return_value=2):
                with self.assertRaisesRegex(runtime.RuntimeError_, "build failed"):
                    runtime.ensure_built("cpu")

    def test_run_chat_and_serve_build_before_resolving_the_binary(self) -> None:
        order: list[str] = []
        parser = cli.build_parser()
        for argv, entry in ((["chat", "--gpu-layers", "0"], cli.cmd_chat),
                            (["serve", "--gpu-layers", "0"], cli.cmd_serve)):
            with self.subTest(command=argv[0]):
                order.clear()
                args = parser.parse_args(argv)
                with mock.patch.object(runtime, "require_vendor"), \
                        mock.patch.object(cli.models, "resolve", return_value="/m.gguf"), \
                        mock.patch.object(runtime, "ensure_built",
                                          side_effect=lambda b: order.append(f"build:{b}")), \
                        mock.patch.object(runtime, "binary",
                                          side_effect=lambda n, b: order.append(n) or "/bin/true"), \
                        mock.patch.object(cli.os, "execv", side_effect=SystemExit(0)):
                    with self.assertRaises(SystemExit):
                        entry(args)
                self.assertEqual(order[0], "build:cpu")
                self.assertEqual(len(order), 2)

    def test_build_command_passes_its_flags(self) -> None:
        args = cli.build_parser().parse_args(["build", "--if-needed", "--keep-build-tree"])
        with mock.patch.object(runtime, "missing_build_packages", return_value=[]), \
                mock.patch.object(runtime, "require_vendor"), \
                mock.patch.object(cli.subprocess, "call", return_value=0) as call:
            self.assertEqual(cli.cmd_build(args), 0)
        self.assertEqual(call.call_args.args[0], [runtime.BUILD_SCRIPT, "--backend", "cpu",
                                                  "--if-needed", "--keep-build-tree"])
        self.assertTrue(runtime.BUILD_SCRIPT.startswith(paths.SCRIPTS_DIR))


if __name__ == "__main__":
    unittest.main()
