"""scripts/agent-sandbox confines a model-written command, or refuses to run it."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
import unittest

from kilix_llm import paths

SCRIPT = os.path.join(paths.REPO_DIR, "scripts", "agent-sandbox")


def _bwrap_works() -> bool:
    if not shutil.which("bwrap"):
        return False
    probe = subprocess.run(["bwrap", "--unshare-all", "--ro-bind", "/", "/", "true"],
                           capture_output=True, timeout=10)
    return probe.returncode == 0


class ScriptShapeTests(unittest.TestCase):
    def test_executable_parses_and_fails_fast(self) -> None:
        self.assertTrue(os.access(SCRIPT, os.X_OK))
        self.assertEqual(subprocess.call(["bash", "-n", SCRIPT]), 0)
        with open(SCRIPT, encoding="utf-8") as handle:
            text = handle.read()
        self.assertIn("set -euo pipefail", text)
        self.assertIn("BASH_VERSINFO", text)

    def test_usage_errors_run_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as work:
            marker = os.path.join(work, "ran")
            for argv in ([], [work], [work, "touch", marker], ["--timeout", "0", work, "--", "true"],
                         ["--keep-env", "BAD-NAME", work, "--", "true"], [work, "-c"],
                         ["--bogus", work, "--", "true"], ["/etc", "--", "true"]):
                with self.subTest(argv=argv):
                    result = subprocess.run([SCRIPT, *argv], capture_output=True, text=True, timeout=30)
                    self.assertEqual(result.returncode, 2, result.stderr)
            self.assertFalse(os.path.exists(marker))


@unittest.skipUnless(_bwrap_works(), "bubblewrap with user namespaces unavailable")
class ConfinementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.work = os.path.join(self.tmp.name, "work")
        os.mkdir(self.work)
        self.outside = os.path.join(self.tmp.name, "outside.txt")
        with open(self.outside, "w", encoding="utf-8") as handle:
            handle.write("not for the box\n")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def box(self, line: str, *options: str, env: dict | None = None, timeout: int = 30):
        return subprocess.run([SCRIPT, *options, self.work, "-c", line], capture_output=True,
                              text=True, timeout=timeout, env=env)

    def test_the_work_directory_is_the_only_writable_place(self) -> None:
        result = self.box("echo made > made.txt && cat made.txt && pwd && echo $HOME")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.split(), ["made", self.work, self.work])
        for target in ("/usr/agent-sandbox-probe", "/etc/agent-sandbox-probe"):
            with self.subTest(target=target):
                self.assertNotEqual(self.box(f"touch {target}").returncode, 0)
        # Unprivileged users cannot write root-owned /usr anyway; the read-only
        # mounts are the guarantee if the box ever runs as root, so pin them.
        mounts = self.box("awk '$5 == \"/usr\" || $5 == \"/etc\" {print $5, $6}' /proc/self/mountinfo")
        flags = dict(line.split() for line in mounts.stdout.splitlines())
        self.assertEqual(set(flags), {"/usr", "/etc"}, mounts.stderr)
        for mount, options in flags.items():
            with self.subTest(mount=mount):
                self.assertIn("ro", options.split(","))
        name = f"agent-sandbox-{os.getpid()}-{time.monotonic_ns()}"
        self.assertEqual(self.box(f"touch /tmp/{name} && ls /tmp/{name}").returncode, 0)
        self.assertFalse(os.path.exists(os.path.join("/tmp", name)), "/tmp is the host's")

    def test_files_outside_the_work_directory_are_absent(self) -> None:
        result = self.box(f"cat {self.outside}")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("not for the box", result.stdout)
        home = os.path.expanduser("~")
        if home != self.work and os.path.isdir(home):
            self.assertNotEqual(self.box(f"ls {home}/.ssh").returncode, 0)

    def test_the_environment_is_cleared_unless_kept(self) -> None:
        env = dict(os.environ, AGENT_SANDBOX_SECRET="s3cret", AGENT_SANDBOX_KEPT="kept")
        result = self.box("echo secret=$AGENT_SANDBOX_SECRET kept=$AGENT_SANDBOX_KEPT",
                          "--keep-env", "AGENT_SANDBOX_KEPT", env=env)
        self.assertEqual(result.stdout.strip(), "secret= kept=kept")

    def test_no_network_unless_asked(self) -> None:
        interfaces = self.box("tail -n +3 /proc/net/dev | cut -d: -f1 | tr -d ' '")
        self.assertEqual(interfaces.stdout.split(), ["lo"])
        shared = self.box("tail -n +3 /proc/net/dev | cut -d: -f1 | tr -d ' '", "--net")
        self.assertEqual(shared.returncode, 0, shared.stderr)

    def test_exit_status_passes_through_and_timeout_kills_the_tree(self) -> None:
        self.assertEqual(self.box("exit 7").returncode, 7)
        marker = os.path.join(self.work, "late")
        started = time.monotonic()
        result = self.box("(sleep 3; touch late) & sleep 30", "--timeout", "1")
        self.assertEqual(result.returncode, 124)
        self.assertLess(time.monotonic() - started, 10)
        time.sleep(3.5)
        self.assertFalse(os.path.exists(marker), "a background child outlived the box")

    def test_stdin_is_not_the_callers(self) -> None:
        result = subprocess.run([SCRIPT, self.work, "-c", "cat"], input="typed by the caller",
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
