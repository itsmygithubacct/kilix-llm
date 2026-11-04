"""Persistent VM boundaries, lifecycle, and guest provisioning regressions."""

from __future__ import annotations

from contextlib import redirect_stdout
import errno
import io
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import tempfile
import unittest
from unittest import mock

from kilix_llm import guest, vm
from kilix_llm.cli import build_parser


class VMTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.state = Path(temporary.name) / "vm, review"
        (self.state / "shared/pi").mkdir(parents=True)
        self.config = {"ssh_port": 23001, "api_port": 23002, "cpus": 4,
                       "memory": 4096, "ctx": 8192, "device": "Vulkan1"}

    def test_qemu_shares_only_persistent_subdirectory_and_binds_ssh_to_loopback(self) -> None:
        argv = vm.qemu_argv(self.state, self.config)
        self.assertEqual(argv.count("-fsdev"), 1)
        share = argv[argv.index("-fsdev") + 1]
        self.assertIn(f"path={str(self.state / 'shared').replace(',', ',,')},", share)
        self.assertIn("security_model=none", share)
        self.assertEqual(argv[argv.index("-netdev") + 1],
                         "user,id=net0,hostfwd=tcp:127.0.0.1:23001-:22")
        self.assertIn("virtio-9p-pci,fsdev=persist,mount_tag=kilix-persist", argv)
        disk = argv[argv.index("-drive") + 1]
        self.assertIn("vm,, review/disk.qcow2", disk)
        self.assertNotIn("-virtfs", argv)

    def test_ssh_ignores_user_config_and_does_not_forward_credentials(self) -> None:
        state = self.state / 'quoted "directory" 50%'
        argv = vm.ssh_argv(state, self.config)
        self.assertEqual(argv[argv.index("-F") + 1], "/dev/null")
        self.assertIn("ForwardAgent=no", argv)
        self.assertIn("ForwardX11=no", argv)
        self.assertIn("IdentitiesOnly=yes", argv)
        self.assertEqual(argv[argv.index("-i") + 1], str(state / "ssh_key"))
        known_hosts = next(value for value in argv if value.startswith("UserKnownHostsFile="))
        self.assertEqual(shlex.split(known_hosts),
                         ["UserKnownHostsFile=" + str(state / "known_hosts").replace("%", "%%")])
        self.assertIn("-T", argv)
        self.assertNotIn("-A", argv)
        self.assertEqual(argv[-1], "kilix@127.0.0.1")

    def test_pi_configuration_preserves_user_settings_other_providers_and_key(self) -> None:
        directory = self.state / "shared/pi"
        settings = '{"theme":"light","compaction":{"enabled":false}}\n'
        (directory / "settings.json").write_text(settings)
        (directory / "api-key").write_text("existing-key\n")
        vm.save(directory / "models.json", {"providers": {"other": {"apiKey": "$OTHER_KEY"}}})
        key = vm.configure_pi(self.state, self.config)
        self.assertEqual(key.read_text(), "existing-key\n")
        self.assertEqual((directory / "settings.json").read_text(), settings)
        providers = vm.read(directory / "models.json")["providers"]
        self.assertEqual(providers["other"], {"apiKey": "$OTHER_KEY"})
        self.assertEqual(providers["kilix"]["apiKey"], "$KILIX_LLM_API_KEY")
        self.assertEqual(providers["kilix"]["baseUrl"], "http://10.0.2.2:23002/v1")
        self.assertEqual(providers["kilix"]["models"][0]["contextWindow"], 8192)

    def test_download_checksum_failure_does_not_promote_image_or_create_disk(self) -> None:
        def download(argv: list[str], **kwargs) -> None:
            self.assertEqual(argv[0], "curl")
            Path(argv[argv.index("-o") + 1]).write_bytes(b"incomplete image")

        with mock.patch.object(vm.shutil, "which", return_value="tool"), \
                mock.patch.object(vm.os, "access", return_value=True), \
                mock.patch.object(vm.os, "getuid", return_value=1000), \
                mock.patch.object(vm, "run", side_effect=download) as run, \
                redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(vm.VMError, "checksum mismatch"):
                vm.prepare(self.state)
        self.assertEqual(run.call_count, 1)
        self.assertTrue((self.state / "base.qcow2.part").exists())
        self.assertFalse((self.state / "base.qcow2").exists())
        self.assertFalse((self.state / "disk.qcow2").exists())
        self.assertFalse((self.state / "config.json").exists())

    def test_existing_configuration_does_not_reprovision_persistent_guest(self) -> None:
        vm.save(self.state / "config.json", self.config)
        with mock.patch.object(vm.shutil, "which", return_value="tool"), \
                mock.patch.object(vm.os, "access", return_value=True), \
                mock.patch.object(vm.os, "getuid", return_value=1000), \
                mock.patch.object(vm, "run") as run:
            self.assertEqual(vm.prepare(self.state), self.config)
        run.assert_not_called()

    def test_stale_process_record_is_never_signalled(self) -> None:
        vm.save(self.state / "server.json", {"pid": 42, "identity": "old-boot:123"})
        with mock.patch.object(vm, "identity", return_value="new-boot:456"), \
                mock.patch.object(vm, "pidfd_open") as open_pidfd, \
                mock.patch.object(vm, "qmp") as qmp, redirect_stdout(io.StringIO()):
            self.assertEqual(vm.process(self.state, "server"), {})
            vm.stop(self.state)
        open_pidfd.assert_not_called()
        qmp.assert_not_called()

    def test_pid_reused_during_shutdown_is_not_signalled(self) -> None:
        record = {"pid": 42, "identity": "boot:123"}
        with mock.patch.object(vm, "process", side_effect=[{}, record, {}, {}, {}]), \
                mock.patch.object(vm, "pidfd_open", return_value=123) as open_pidfd, \
                mock.patch.object(vm, "pidfd_send_signal") as send_signal, \
                mock.patch.object(vm.os, "close") as close, redirect_stdout(io.StringIO()):
            vm.stop(self.state)
        open_pidfd.assert_called_once_with(42)
        send_signal.assert_not_called()
        close.assert_called_once_with(123)

    def test_process_exiting_before_pidfd_open_is_already_stopped(self) -> None:
        record = {"pid": 42, "identity": "boot:123"}
        with mock.patch.object(vm, "process", side_effect=[{}, record, {}, {}]), \
                mock.patch.object(vm, "pidfd_open", side_effect=ProcessLookupError), \
                mock.patch.object(vm, "pidfd_send_signal") as send_signal, \
                redirect_stdout(io.StringIO()):
            vm.stop(self.state)
        send_signal.assert_not_called()

    def test_shell_arguments_keep_their_exact_boundaries(self) -> None:
        vm.save(self.state / "config.json", self.config)
        arguments = ["printf", "%s", "spaces; $VAR `quoted` 'double\" quotes'"]
        args = build_parser().parse_args(["vm", "shell", "--", *arguments])
        with mock.patch.object(vm, "root", return_value=self.state), \
                mock.patch.object(vm, "process", return_value={"pid": 42}), \
                mock.patch.object(vm.subprocess, "call", return_value=0) as call:
            self.assertEqual(vm.command(args), 0)
        argv = call.call_args.args[0]
        shell = shlex.split(argv[-1])
        self.assertEqual(shell[:2], ["bash", "-lc"])
        self.assertEqual(shlex.split(shell[2]), ["exec", *arguments])
        self.assertIn("-T", argv)

    def test_pi_new_tab_keeps_arguments_separate(self) -> None:
        arguments = ["-p", "review 'some file' with $symbols; intact"]
        args = build_parser().parse_args(["vm", "pi", "--new-tab", "--", *arguments])
        with mock.patch.object(vm, "root", return_value=self.state), \
                mock.patch.object(vm, "prepare", return_value=self.config), \
                mock.patch.object(vm, "start_guest"), mock.patch.object(vm, "start_server"), \
                mock.patch.object(vm.shutil, "which", return_value="kilix"), \
                mock.patch.object(vm.subprocess, "call", return_value=0) as call:
            self.assertEqual(vm.command(args), 0)
        argv = call.call_args.args[0]
        self.assertEqual(argv[:2], ["kilix", "new-tab"])
        self.assertIn(f"KILIX_LLM_VM_DIR={self.state}", argv)
        self.assertEqual(argv[-5:], ["vm", "pi", "--", *arguments])

    def test_server_uses_host_gpu_loopback_and_key_file(self) -> None:
        with mock.patch.object(vm.runtime, "require_vendor"), \
                mock.patch.object(vm.models, "resolve", return_value="model.gguf"), \
                mock.patch.object(vm.runtime, "select_device", return_value="Vulkan1"), \
                mock.patch.object(vm.runtime, "binary", return_value="llama-server"), \
                mock.patch.object(vm.runtime, "physical_cores", return_value=4), \
                mock.patch.object(vm, "process", side_effect=[{}, {"pid": 42}]), \
                mock.patch.object(vm, "healthy", return_value=True), \
                mock.patch.object(vm, "spawn") as spawn, redirect_stdout(io.StringIO()):
            vm.start_server(self.state, self.config)
        argv = spawn.call_args.args[2]
        for flag, value in (("--host", "127.0.0.1"), ("--port", "23002"),
                            ("--device", "Vulkan1"), ("-ngl", "99"),
                            ("--alias", vm.MODEL_ID), ("--parallel", "1")):
            self.assertEqual(argv[argv.index(flag) + 1], value)
        key = Path(argv[argv.index("--api-key-file") + 1])
        self.assertEqual(key.stat().st_mode & 0o777, 0o600)
        self.assertNotIn(key.read_text().strip(), argv)

    def test_lock_is_exclusive_and_released_after_exception(self) -> None:
        with self.assertRaisesRegex(vm.VMError, "another VM operation"):
            with vm.locked(self.state):
                with vm.locked(self.state):
                    self.fail("second lock should fail")
        with vm.locked(self.state):
            pass


class PidfdTests(unittest.TestCase):
    def test_managed_python_libc_fallback_supports_signal_zero_and_errno(self) -> None:
        # Signal zero checks a live process without sending it any signal.
        with mock.patch.object(os, "pidfd_open", None, create=True), \
                mock.patch.object(signal, "pidfd_send_signal", None, create=True):
            descriptor = vm.pidfd_open(os.getpid())
            try:
                vm.pidfd_send_signal(descriptor, 0)
            finally:
                os.close(descriptor)
            with self.assertRaises(OSError) as error:
                vm.pidfd_send_signal(-1, 0)
        self.assertEqual(error.exception.errno, errno.EBADF)

    def test_identity_handles_parentheses_in_process_names_and_rejects_zombies(self) -> None:
        stat = "42 (qemu (review)) " + " ".join(["S", *(["0"] * 18), "12345"])
        with mock.patch.object(Path, "read_text", side_effect=[stat, "boot-id\n"]):
            self.assertEqual(vm.identity(42), "boot-id:12345")
        with mock.patch.object(Path, "read_text", return_value="42 (qemu) Z 0"):
            self.assertIsNone(vm.identity(42))


class GuestTests(unittest.TestCase):
    def test_guest_mount_and_provisioning_keep_tools_and_private_home_separate(self) -> None:
        config = json.loads(guest.cloud_config("ssh-ed25519 example test", 1000, 1000).split("\n", 1)[1])
        user = config["users"][0]
        self.assertEqual(user["uid"], 1000)
        self.assertFalse(user["homedir"].startswith("/persist"))
        self.assertEqual(config["mounts"][0][:3], ["kilix-persist", "/persist", "9p"])
        self.assertIn("fd-find", config["packages"])
        files = {item["path"]: item["content"] for item in config["write_files"]}
        setup = files["/usr/local/sbin/kilix-guest-setup"]
        self.assertLess(setup.index("mountpoint -q /persist"), setup.index("curl --fail"))
        self.assertLess(setup.index("cd /persist/workspace"), setup.index('"${as_kilix[@]}" uv python'))
        self.assertEqual(setup.count("sha256sum --check --status"), 2)
        self.assertIn("uv run --managed-python --python 3.12", setup)
        self.assertIn("--ignore-scripts", setup)
        self.assertNotIn("UV_PYTHON_PREFERENCE", guest.ENVIRONMENT + setup)
        self.assertIn("UV_MANAGED_PYTHON=1", guest.ENVIRONMENT)
        for script in (setup, files["/usr/local/bin/kilix-pi"], files["/etc/profile.d/kilix-llm.sh"]):
            result = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
