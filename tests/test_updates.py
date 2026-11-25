"""`kilix-llm update` builds llama.cpp's newest release beside the pin.

A local git repository stands in for upstream, with bNNNN release tags, and
the real build script runs with a `cmake` that writes binaries reporting the
commit they were built from. The fetch, the build, the smoke test, the
generation switch, rollback and pruning all run for real; nothing compiles
and nothing leaves the machine.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import stat
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from kilix_llm import cli, runtime, updates

FAKE_CMAKE = r"""#!/usr/bin/env bash
set -euo pipefail
if [ "$1" = --build ]; then
  root=$2
  src=$(cat "$root/.source")
  commit=$(git -C "$src" rev-parse HEAD)
  mkdir -p "$root/bin" "$root/obj"
  : > "$root/obj/ggml.o"
  [ "${FAKE_CMAKE_FAIL:-0}" = 0 ] || exit 3
  for name in llama-cli llama-server llama-bench; do
    cat > "$root/bin/$name" <<SCRIPT
#!/bin/sh
[ "\$1" != --version ] || { echo "version: 1 (${commit:0:7})" >&2; exit 0; }
for a; do [ "\$prev" != -m ] || echo "failed to open \$a" >&2; prev=\$a; done
exit 1
SCRIPT
    [ "${FAKE_BROKEN_BINARY:-0}" = 0 ] || printf '#!/bin/sh\nexit 1\n' > "$root/bin/$name"
    chmod +x "$root/bin/$name"
  done
  printf '%s\n' "$*" >> "$FAKE_CMAKE_LOG"
else
  src='' root=''
  while [ "$#" -gt 0 ]; do
    case "$1" in -S) src=$2; shift ;; -B) root=$2; shift ;; esac
    shift
  done
  mkdir -p "$root/CMakeFiles"
  : > "$root/CMakeCache.txt"
  printf '%s\n' "$src" > "$root/.source"
fi
"""


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True,
        env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
             "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid"},
    ).stdout.strip()


@unittest.skipUnless(shutil.which("git") and shutil.which("flock"), "git and flock are required")
class UpdateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="kilix-llm-update-test-"))
        self.addCleanup(shutil.rmtree, self.tmp)
        self.upstream = self.tmp / "upstream"
        (self.upstream / "src" / "models").mkdir(parents=True)
        git(self.upstream, "init", "-q")
        git(self.upstream, "config", "uploadpack.allowAnySHA1InWant", "true")
        self.commits = {}
        for number in (100, 101):
            self.release(number)
        tools = self.tmp / "tools"
        tools.mkdir()
        cmake = tools / "cmake"
        cmake.write_text(FAKE_CMAKE)
        cmake.chmod(cmake.stat().st_mode | stat.S_IXUSR)
        self.log = self.tmp / "cmake.log"
        self.root = self.tmp / "home" / "runtime"
        env = {
            "PATH": f"{tools}{os.pathsep}{os.environ['PATH']}",
            "FAKE_CMAKE_LOG": str(self.log),
            "KILIX_LLM_RUNTIME_DIR": str(self.root),
            "KILIX_LLM_UPSTREAM_URL": str(self.upstream),
            "KILIX_LLM_BLAS": "off",
        }
        patcher = mock.patch.dict(os.environ, env)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("KILIX_LLM_BUILD_DIR", None)
        self.log_messages: list[str] = []

    def release(self, number: int, graph: bool = True) -> updates.Release:
        (self.upstream / "CMakeLists.txt").write_text(f"project(llama) # {number}\n")
        qwen = self.upstream / "src" / "models" / "qwen35.cpp"
        if graph:
            qwen.write_text(f"// {number}\n")
        elif qwen.exists():
            qwen.unlink()
        git(self.upstream, "add", "-A")
        git(self.upstream, "commit", "-q", "-m", f"release {number}")
        git(self.upstream, "tag", "-a", f"b{number}", "-m", f"b{number}")
        commit = git(self.upstream, "rev-parse", "HEAD")
        self.commits[number] = commit
        return updates.Release(f"b{number}", commit)

    def builds(self) -> int:
        return len(self.log.read_text().splitlines()) if self.log.exists() else 0

    def update(self, *backends: str) -> updates.Generation:
        return updates.update(backends or ("cpu",), runtime.BUILD_SCRIPT,
                              log=self.log_messages.append)

    def test_annotated_tags_are_peeled_and_the_highest_number_wins(self) -> None:
        listing = "\n".join([
            f"{'1' * 40}\trefs/tags/b99",
            f"{'2' * 40}\trefs/tags/b1000",
            f"{'3' * 40}\trefs/tags/b1000^{{}}",
            f"{'4' * 40}\trefs/tags/b200",
            f"{'5' * 40}\trefs/tags/gguf-v0.1",
            "garbage",
        ])
        releases = updates.parse_release_tags(listing)
        best = max(releases, key=lambda r: r.number)
        self.assertEqual(best, updates.Release("b1000", "3" * 40))
        self.assertNotIn("gguf-v0.1", {r.tag for r in releases})
        latest = updates.latest_release()
        self.assertEqual(latest, updates.Release("b101", self.commits[101]))

    def test_update_builds_checks_and_switches_to_the_newest_release(self) -> None:
        generation = self.update("cpu")
        self.assertEqual(generation.release, updates.Release("b101", self.commits[101]))
        self.assertEqual(updates.current(), generation)
        self.assertIsNone(updates.previous())
        self.assertEqual(generation.backends, ("cpu",))
        self.assertEqual(runtime.runtime_bin_dir("cpu"), generation.bin_dir("cpu"))
        self.assertEqual(runtime.binary("llama-cli", "cpu"),
                         os.path.join(generation.bin_dir("cpu"), "llama-cli"))
        # only bin/ is kept, the source checkout is gone
        self.assertEqual(sorted(p.name for p in Path(generation.path, "cpu").iterdir()), ["bin"])
        self.assertEqual([p.name for p in self.root.iterdir() if p.name.startswith(".src-")], [])
        record = json.loads(Path(generation.path, updates.RECORD).read_text())
        self.assertEqual(record["commit"], self.commits[101])
        # a second update with nothing new builds nothing
        self.update("cpu")
        self.assertEqual(self.builds(), 1)

    def test_rollback_returns_to_the_previous_release_then_the_pin(self) -> None:
        first = self.update()
        self.release(102)
        second = self.update()
        self.assertEqual(second.release.tag, "b102")
        self.assertEqual(updates.previous(), first)
        self.assertEqual(updates.rollback(), first)
        self.assertEqual(updates.current(), first)
        self.assertEqual(updates.previous(), second)
        # three releases later only current and previous remain on disk
        self.release(103)
        third = self.update()
        self.assertEqual(sorted(os.listdir(updates.generations_dir())),
                         sorted({Path(first.path).name, Path(third.path).name}))
        self.assertEqual(updates.rollback(), first)

    def test_rollback_from_the_first_update_returns_to_the_pin(self) -> None:
        with self.assertRaises(updates.UpdateError):
            updates.rollback()
        first = self.update()
        self.assertIsNone(updates.rollback())
        self.assertIsNone(updates.current())
        self.assertEqual(runtime.runtime_bin_dir("cpu"), runtime.paths.bin_dir("cpu"))
        self.assertEqual(updates.previous(), first)

    def test_a_failed_or_broken_build_changes_nothing(self) -> None:
        first = self.update()
        self.release(102)
        for env in ({"FAKE_CMAKE_FAIL": "1"}, {"FAKE_BROKEN_BINARY": "1"}):
            with self.subTest(env=env), mock.patch.dict(os.environ, env):
                with self.assertRaises(updates.UpdateError):
                    self.update()
                self.assertEqual(updates.current(), first)
                failed = updates.read_generation(
                    os.path.join(updates.generations_dir(), f"b102-{self.commits[102][:12]}"))
                self.assertTrue(failed is None or "cpu" not in failed.backends)

    def test_a_release_without_the_qwen_graph_is_refused(self) -> None:
        self.release(102, graph=False)
        with self.assertRaisesRegex(updates.UpdateError, "qwen35.cpp"):
            self.update()
        self.assertIsNone(updates.current())
        self.assertEqual(self.builds(), 0)

    def test_a_backend_first_used_after_an_update_builds_into_the_same_release(self) -> None:
        generation = self.update("cpu")
        with mock.patch.object(runtime, "missing_build_packages", return_value=[]), \
                mock.patch.dict(os.environ, {"KILIX_LLM_NO_AUTO_BUILD": "0"}):
            runtime.ensure_built("vulkan")
            runtime.ensure_built("vulkan")
        now = updates.current()
        self.assertEqual(now.path, generation.path)
        self.assertEqual(now.backends, ("cpu", "vulkan"))
        self.assertTrue(now.has("vulkan"))
        self.assertEqual(self.builds(), 2)
        stamp = Path(now.path, "vulkan", "bin", ".kilix-llm-build").read_text()
        self.assertIn("-DGGML_VULKAN=ON", stamp)
        self.assertIn(f"source-commit={generation.release.commit}", stamp)

    def test_build_if_needed_after_an_update_builds_that_release(self) -> None:
        generation = self.update("cpu")
        args = cli.build_parser().parse_args(["build", "--if-needed", "--backend", "vulkan"])
        with mock.patch.object(runtime, "missing_build_packages", return_value=[]), \
                mock.patch.object(runtime, "require_vendor"), \
                mock.patch.dict(os.environ, {"KILIX_LLM_NO_AUTO_BUILD": "1"}), \
                redirect_stderr(io.StringIO()):
            self.assertEqual(cli.cmd_build(args), 0)
            self.assertEqual(os.environ["KILIX_LLM_NO_AUTO_BUILD"], "1")
        self.assertEqual(updates.current().backends, ("cpu", "vulkan"))
        self.assertEqual(updates.current().path, generation.path)

    def test_check_reports_and_exits_ten_when_an_update_is_available(self) -> None:
        parser = cli.build_parser()
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(cli.cmd_update(parser.parse_args(["update", "--check"])), 10)
        self.assertIn("update available: llama.cpp b101", out.getvalue())
        with mock.patch.object(runtime, "build_current", return_value=False), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(cli.cmd_update(parser.parse_args(["update"])), 0)
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(cli.cmd_update(parser.parse_args(["update", "--check"])), 0)
            self.assertEqual(cli.cmd_runtime(parser.parse_args(["runtime", "--json"])), 0)
        lines = out.getvalue().splitlines()
        self.assertIn("up to date: llama.cpp b101", lines[0])
        status = json.loads(lines[1])
        self.assertEqual(status["current"]["tag"], "b101")
        self.assertEqual(status["rollback_to"], "pinned")

    def test_an_unreachable_upstream_is_an_update_error(self) -> None:
        with mock.patch.dict(os.environ, {"KILIX_LLM_UPSTREAM_URL": str(self.tmp / "nowhere")}):
            with self.assertRaisesRegex(updates.UpdateError, "cannot list llama.cpp releases"):
                updates.latest_release()

    def test_a_generation_outside_the_runtime_directory_is_ignored(self) -> None:
        self.update()
        elsewhere = self.tmp / "elsewhere"
        shutil.copytree(updates.current().path, elsewhere)
        link = self.root / "current"
        link.unlink()
        link.symlink_to(elsewhere)
        self.assertIsNone(updates.current())


if __name__ == "__main__":
    unittest.main()


class FlagCheckTests(unittest.TestCase):
    """A release that drops a flag kilix-llm passes is refused before it is used."""

    def fake_runtime(self, script: str) -> str:
        directory = tempfile.mkdtemp(prefix="kilix-llm-flags-")
        self.addCleanup(shutil.rmtree, directory)
        for name in ("llama-cli", "llama-server"):
            path = Path(directory, name)
            path.write_text("#!/bin/sh\n" + script)
            path.chmod(0o755)
        return directory

    def test_every_command_line_reaching_the_model_passes(self) -> None:
        # Parse everything, then stop at the model path, like llama.cpp does.
        directory = self.fake_runtime(
            'for a; do case "$prev" in -m) echo "failed to open $a" >&2;; esac; prev=$a; done\nexit 1\n')
        cli.check_flags(directory, "cpu")

    def test_a_rejected_flag_is_an_update_error(self) -> None:
        directory = self.fake_runtime('echo "error: invalid argument: -st" >&2\nexit 1\n')
        with self.assertRaisesRegex(updates.UpdateError, "does not accept kilix-llm's run command line"):
            cli.check_flags(directory, "cpu")

    def test_update_and_first_use_run_the_check(self) -> None:
        args = cli.build_parser().parse_args(["update"])
        with mock.patch.object(cli, "update_status",
                               return_value=("pin", updates.Release("b2", "c" * 40), True)), \
                mock.patch.object(cli, "built_backends", return_value=["cpu"]), \
                mock.patch.object(runtime, "missing_build_packages", return_value=[]), \
                mock.patch.object(updates, "update") as update, \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            update.return_value = updates.Generation("/g", updates.Release("b2", "c" * 40), "", ("cpu",))
            self.assertEqual(cli.cmd_update(args), 0)
        self.assertIs(update.call_args.kwargs["verify"], cli.check_flags)
        generation = updates.Generation("/g", updates.Release("b2", "c" * 40), "", ("cpu",))
        with mock.patch.object(runtime, "missing_build_packages", return_value=[]), \
                mock.patch.object(updates, "ensure_backend") as ensure, \
                redirect_stderr(io.StringIO()):
            runtime._ensure_generation_backend(generation, "vulkan")
        self.assertIs(ensure.call_args.args[2], cli.check_flags)
