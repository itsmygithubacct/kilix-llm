"""Unprivileged QEMU guest for Pi, with host GPU inference and persistence."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import secrets
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

from . import guest, models, paths, runtime

IMAGE_URL = "https://cloud.debian.org/images/cloud/trixie/20260831-2587/debian-13-generic-amd64-20260831-2587.qcow2"
IMAGE_SHA512 = "5a069019420fb9441ad4f8004c661fadb747edd5662ca54a17c8f923dee7d717e21dbdaa4ba72d6fce7f920e0217f0a9af382298a7d46ed4bc9dc33ac19181b6"
MODEL_ID = "qwen3.5-4b"


class VMError(runtime.RuntimeError_):
    pass


def root() -> Path:
    return Path(os.environ.get("KILIX_LLM_VM_DIR", str(Path(paths.gpu_terminal_home()) / "kilix-llm/vm"))).expanduser().resolve()


def save(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.chmod(0o600)
    temporary.replace(path)


def read(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


@contextmanager
def locked(state: Path):
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (state / "lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise VMError("another VM operation is running") from error
        yield


def run(argv: list[str], **kwargs) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(argv, check=True, **kwargs)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise VMError(f"{argv[0]} failed: {error}") from error


def identity(pid: int) -> str | None:
    """Boot ID and kernel start time distinguish our process from a reused PID."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if stat[0] == "Z":
            return None
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip() + ":" + stat[19]
    except (OSError, IndexError):
        return None


def process(state: Path, name: str) -> dict:
    record = read(state / f"{name}.json")
    if record and (type(record.get("pid")) is not int or record["pid"] <= 0 or not isinstance(record.get("identity"), str)):
        raise VMError(f"invalid process state in {state / (name + '.json')}; inspect it before starting another process")
    return record if record.get("identity") and identity(record["pid"]) == record["identity"] else {}


def spawn(state: Path, name: str, argv: list[str]) -> None:
    with (state / f"{name}.log").open("ab") as log:
        child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
    save(state / f"{name}.json", {"pid": child.pid, "identity": identity(child.pid)})


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def qemu_value(value: Path) -> str:
    # QEMU's comma-separated option syntax escapes literal commas by doubling.
    return str(value).replace(",", ",,")


def qemu_argv(state: Path, config: dict) -> list[str]:
    return [
        "qemu-system-x86_64", "-name", "kilix-pi", "-machine", "q35,accel=kvm",
        "-cpu", "host", "-smp", str(config["cpus"]), "-m", str(config["memory"]),
        "-display", "none", "-monitor", "none", "-serial", "stdio",
        "-qmp", f"unix:{qemu_value(state / 'qmp.sock')},server=on,wait=off",
        "-drive", f"file={qemu_value(state / 'disk.qcow2')},if=virtio,format=qcow2",
        "-drive", f"file={qemu_value(state / 'seed.iso')},media=cdrom,readonly=on,format=raw",
        "-netdev", f"user,id=net0,hostfwd=tcp:127.0.0.1:{config['ssh_port']}-:22",
        "-device", "virtio-net-pci,netdev=net0",
        "-fsdev", f"local,id=persist,path={qemu_value(state / 'shared')},security_model=none",
        "-device", "virtio-9p-pci,fsdev=persist,mount_tag=kilix-persist",
        "-device", "virtio-rng-pci",
    ]


def ssh_argv(state: Path, config: dict, tty: bool = False) -> list[str]:
    known_hosts = str(state / "known_hosts").replace("%", "%%").replace("\\", "\\\\").replace('"', '\\"')
    return [
        "ssh", "-F", "/dev/null", "-i", str(state / "ssh_key"),
        "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes", "-o", "ConnectTimeout=3",
        "-o", "StrictHostKeyChecking=accept-new", "-o", f'UserKnownHostsFile="{known_hosts}"',
        "-o", "ForwardAgent=no", "-o", "ForwardX11=no",
        "-o", "LogLevel=ERROR", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3",
        "-p", str(config["ssh_port"]), *(["-tt"] if tty else ["-T"]), "kilix@127.0.0.1",
    ]


def prepare(state: Path) -> dict:
    required = ["qemu-system-x86_64", "qemu-img", "xorriso", "ssh", "ssh-keygen", "curl"]
    missing = [name for name in required if not shutil.which(name)]
    if missing:
        raise VMError("VM prerequisites missing: " + ", ".join(missing))
    if not os.access("/dev/kvm", os.R_OK | os.W_OK):
        raise VMError("QEMU needs read/write access to /dev/kvm")
    if os.getuid() == 0:
        raise VMError("run the VM as your regular user, without sudo")
    config = read(state / "config.json")
    if config:
        return config
    for directory in ("shared/workspace", "shared/pi", "shared/cache", "shared/python", "seed"):
        (state / directory).mkdir(parents=True, exist_ok=True, mode=0o700)
    base = state / "base.qcow2"
    if not base.exists():
        print("Downloading the pinned Debian 13 VM image (416 MiB)...", file=sys.stderr, flush=True)
        temporary = state / "base.qcow2.part"
        run(["curl", "-fL", "--retry", "3", "-C", "-", "-o", str(temporary), IMAGE_URL])
        with temporary.open("rb") as handle:
            if hashlib.file_digest(handle, "sha512").hexdigest() != IMAGE_SHA512:
                raise VMError(f"VM image checksum mismatch; remove {temporary} and retry")
        temporary.replace(base)
    with base.open("rb") as handle:
        if hashlib.file_digest(handle, "sha512").hexdigest() != IMAGE_SHA512:
            raise VMError(f"VM image checksum mismatch: {base}")
    base.chmod(0o444)
    disk = state / "disk.qcow2"
    if not disk.exists():
        run(["qemu-img", "create", "-f", "qcow2", "-F", "qcow2", "-b", str(base), str(disk), "24G"], stdout=sys.stderr)
    if not (state / "ssh_key").exists():
        run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(state / "ssh_key")])
    (state / "seed/user-data").write_text(guest.cloud_config((state / "ssh_key.pub").read_text(), os.getuid(), os.getgid()))
    save(state / "seed/meta-data", {"instance-id": "kilix-pi-" + secrets.token_hex(8), "local-hostname": "kilix-pi"})
    run(["xorriso", "-as", "mkisofs", "-quiet", "-output", str(state / "seed.iso"), "-volid", "cidata", "-joliet", "-rock", str(state / "seed/user-data"), str(state / "seed/meta-data")], stdout=subprocess.DEVNULL)
    config = {"ssh_port": free_port(), "api_port": free_port(), "cpus": 4, "memory": 4096, "ctx": 8192, "device": None}
    save(state / "config.json", config)
    return config


def start_guest(state: Path, config: dict) -> None:
    if not process(state, "qemu"):
        (state / "qmp.sock").unlink(missing_ok=True)
        print("Booting QEMU/KVM; persistent folder: " + str(state / "shared"), file=sys.stderr, flush=True)
        spawn(state, "qemu", qemu_argv(state, config))
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if not process(state, "qemu"):
            raise VMError(f"QEMU exited; inspect {state / 'qemu.log'}")
        result = subprocess.run([*ssh_argv(state, config), "true"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=8)
        if result.returncode == 0:
            break
        time.sleep(2)
    else:
        raise VMError(f"VM SSH did not become ready; inspect {state / 'qemu.log'}")
    print("Waiting for guest setup (first boot installs Pi, Node, and uv-managed Python)...", file=sys.stderr, flush=True)
    result = subprocess.run([*ssh_argv(state, config), "sudo cloud-init status --wait >/dev/null; test -f /var/lib/kilix-llm/ready"], timeout=1200)
    if result.returncode:
        raise VMError("guest setup failed; use `kilix-llm vm shell -- sudo tail -80 /var/log/cloud-init-output.log`. After fixing, rerun sudo /usr/local/sbin/kilix-guest-setup inside the guest")


def pi_models(port: int, ctx: int) -> dict:
    return {"providers": {"kilix": {
        "baseUrl": f"http://10.0.2.2:{port}/v1", "api": "openai-completions",
        "apiKey": "$KILIX_LLM_API_KEY",
        "compat": {"supportsStore": False, "supportsDeveloperRole": False, "supportsReasoningEffort": False, "supportsStrictMode": False, "maxTokensField": "max_tokens"},
        "models": [{"id": MODEL_ID, "name": "Qwen3.5 4B Q6 (host GPU)", "reasoning": False, "input": ["text"], "contextWindow": ctx, "maxTokens": 2048,
                    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}}],
    }}}


def configure_pi(state: Path, config: dict) -> Path:
    directory = state / "shared/pi"
    key = directory / "api-key"
    if not key.exists():
        fd = os.open(key, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as handle:
            handle.write(secrets.token_urlsafe(32) + "\n")
    existing = read(directory / "models.json")
    existing.setdefault("providers", {})["kilix"] = pi_models(config["api_port"], config["ctx"])["providers"]["kilix"]
    save(directory / "models.json", existing)
    if not (directory / "settings.json").exists():
        save(directory / "settings.json", {"defaultProvider": "kilix", "defaultModel": MODEL_ID, "defaultThinkingLevel": "off",
            "compaction": {"enabled": True, "reserveTokens": 2048, "keepRecentTokens": 2048}, "branchSummary": {"reserveTokens": 2048}})
    instructions = directory / "AGENTS.md"
    if not instructions.exists():
        instructions.write_text("Python tasks use uv-managed Python: `uv run --managed-python --python 3.12 python ...`.\nUse uv to manage project dependencies. The persistent workspace is /persist/workspace.\n")
    return key


def healthy(config: dict, key: Path) -> bool:
    request = urllib.request.Request(f"http://127.0.0.1:{config['api_port']}/health", headers={"Authorization": "Bearer " + key.read_text().strip()})
    try:
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=3) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def start_server(state: Path, config: dict) -> None:
    from .cli import serve_argv

    key = configure_pi(state, config)
    if not process(state, "server"):
        runtime.require_vendor()
        model = models.resolve(MODEL_ID)
        device = runtime.select_device("vulkan", config["device"])
        config["device"] = device
        save(state / "config.json", config)
        args = argparse.Namespace(model_id=MODEL_ID, gpu_layers=99, threads=0, ctx=config["ctx"], device=device, think=False, host="127.0.0.1", port=config["api_port"])
        argv = serve_argv(args, runtime.binary("llama-server", "vulkan"), model)
        argv.extend(["--alias", MODEL_ID, "--api-key-file", str(key)])
        print(f"Starting {MODEL_ID} on host GPU {device}...", file=sys.stderr, flush=True)
        spawn(state, "server", argv)
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if not process(state, "server"):
            raise VMError(f"model server exited; inspect {state / 'server.log'}. Close other GPU model sessions if VRAM is full")
        if healthy(config, key):
            return
        time.sleep(2)
    raise VMError(f"model server did not become healthy; inspect {state / 'server.log'}")


def qmp(state: Path, command: str) -> None:
    with socket.socket(socket.AF_UNIX) as sock:
        sock.settimeout(5)
        sock.connect(str(state / "qmp.sock"))
        with sock.makefile("rwb", buffering=0) as stream:
            json.loads(stream.readline())
            for operation in ("qmp_capabilities", command):
                stream.write(json.dumps({"execute": operation}).encode() + b"\n")
                while True:
                    reply = json.loads(stream.readline())
                    if "error" in reply:
                        raise VMError(f"QEMU: {reply['error']}")
                    if "return" in reply:
                        break


def _libc_pidfd(name: str, argtypes: tuple, args: tuple) -> int:
    # Some uv-managed CPython builds omit pidfd wrappers despite kernel support.
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    try:
        function = getattr(libc, name)
    except AttributeError as error:
        raise VMError(f"the host C library needs Linux {name} support") from error
    function.argtypes = argtypes
    function.restype = ctypes.c_int
    result = function(*args)
    if result < 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    return result


def pidfd_open(pid: int) -> int:
    native = getattr(os, "pidfd_open", None)
    if native is not None:
        return native(pid)
    import ctypes

    return _libc_pidfd("pidfd_open", (ctypes.c_int, ctypes.c_uint), (pid, 0))


def pidfd_send_signal(fd: int, signum: int) -> None:
    native = getattr(signal, "pidfd_send_signal", None)
    if native is not None:
        native(fd, signum)
        return
    import ctypes

    _libc_pidfd("pidfd_send_signal", (ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint), (fd, signum, None, 0))


def stop(state: Path) -> None:
    if process(state, "qemu"):
        qmp(state, "system_powerdown")
        deadline = time.monotonic() + 60
        while process(state, "qemu") and time.monotonic() < deadline:
            time.sleep(1)
        if process(state, "qemu"):
            raise VMError("guest is still shutting down; retry vm stop. Its disk has been retained")
    record = process(state, "server")
    if record:
        # pidfd avoids signalling a reused PID between identity check and signal.
        try:
            fd = pidfd_open(record["pid"])
        except ProcessLookupError:
            fd = None
        try:
            if fd is not None and process(state, "server") == record:
                pidfd_send_signal(fd, signal.SIGTERM)
        except ProcessLookupError:
            pass
        finally:
            if fd is not None:
                os.close(fd)
        deadline = time.monotonic() + 30
        while process(state, "server") and time.monotonic() < deadline:
            time.sleep(1)
        if process(state, "server"):
            raise VMError("model server is still shutting down; retry vm stop")
    print("VM and model server stopped; disk and shared files retained.")


def command(args: argparse.Namespace) -> int:
    state = root()
    action = args.vm_action
    try:
        if action == "status":
            config = read(state / "config.json")
            print(f"VM: {'running' if process(state, 'qemu') else 'stopped'}")
            print(f"GPU server: {'running' if process(state, 'server') else 'stopped'}")
            print(f"Persistent folder: {state / 'shared'}")
            if config:
                print(f"Guest SSH: 127.0.0.1:{config['ssh_port']}; model API: 127.0.0.1:{config['api_port']}")
            return 0
        if action == "stop":
            with locked(state):
                stop(state)
            return 0
        if action == "shell":
            config = read(state / "config.json")
            if not config or not process(state, "qemu"):
                raise VMError("VM is stopped; run `kilix-llm vm setup` or `kilix-llm vm start --device DEVICE` first")
            remaining = args.guest_args
            if remaining[:1] == ["--"]:
                remaining = remaining[1:]
            remote = shlex.join(["bash", "-lc", "exec " + shlex.join(remaining)]) if remaining else "exec bash -l"
            return subprocess.call([*ssh_argv(state, config, not remaining), remote])
        with locked(state):
            config = prepare(state)
            device = getattr(args, "device", None)
            ctx = getattr(args, "ctx", None)
            if (device and device != config["device"]) or (ctx is not None and ctx != config["ctx"]):
                if process(state, "server"):
                    raise VMError("stop the VM before changing the model device or context")
                if device:
                    config["device"] = device
                if ctx is not None:
                    if ctx < 8192:
                        raise VMError("Pi requires at least 8192 context tokens")
                    config["ctx"] = ctx
                save(state / "config.json", config)
            start_guest(state, config)
            if action != "setup":
                start_server(state, config)
        if action != "pi":
            print("Guest ready. " + ("Use `kilix-llm vm start --device DEVICE` to start inference." if action == "setup" else "Use `kilix-llm vm pi --new-tab` to open Pi."))
            return 0
        remaining = args.pi_args
        if remaining[:1] == ["--"]:
            remaining = remaining[1:]
        if args.new_tab:
            if not shutil.which("kilix"):
                raise VMError("kilix is not on PATH; use `kilix-llm vm pi` in this terminal")
            return subprocess.call(["kilix", "new-tab", "--title", "Pi VM · Qwen GPU", "--cwd", paths.REPO_DIR, "--", "env", f"KILIX_LLM_VM_DIR={state}", str(Path(paths.REPO_DIR) / "kilix-llm"), "vm", "pi", "--", *remaining])
        return subprocess.call([*ssh_argv(state, config, sys.stdin.isatty()), shlex.join(["/usr/local/bin/kilix-pi", *remaining])])
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        raise VMError(str(error)) from error


def add_parser(sub) -> None:
    parser = sub.add_parser("vm", help="persistent QEMU Pi guest with host GPU inference")
    actions = parser.add_subparsers(dest="vm_action", required=True)
    for name, help_text in (("setup", "download and provision the VM"), ("start", "start the VM and GPU model server"), ("pi", "run Pi inside the VM"), ("shell", "guest shell or command after --"), ("status", "show VM state and persistence path"), ("stop", "shut down while retaining all files")):
        action = actions.add_parser(name, help=help_text)
        action.set_defaults(func=command)
        if name in ("start", "pi"):
            action.add_argument("--device", help="host Vulkan device, remembered after first start")
            action.add_argument("--ctx", type=int, help="model context size (initial default: 8192)")
        if name == "pi":
            action.add_argument("--new-tab", action="store_true", help="open Pi in a new Kilix tab")
            action.add_argument("pi_args", nargs=argparse.REMAINDER, help="Pi arguments after --")
        if name == "shell":
            action.add_argument("guest_args", nargs=argparse.REMAINDER)
