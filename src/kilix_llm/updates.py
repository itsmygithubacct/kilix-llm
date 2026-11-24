"""Newer llama.cpp releases, built beside the pinned runtime.

The checkout pins one reviewed llama.cpp commit and builds it on first use.
`kilix-llm update` moves past that pin to llama.cpp's newest release (the
`bNNNN` tags upstream publishes for every release). Each release is built
into its own generation:

    <runtime root>/generations/<tag>-<commit12>/<backend>/bin/
    <runtime root>/current  -> generations/<...>
    <runtime root>/previous -> generations/<...>

A generation becomes `current` only after every requested backend built and
its binaries ran. The one it replaces stays as `previous`, so `kilix-llm
rollback` can switch back, and with no generation at all the pinned build in
the checkout is used. Older generations are removed.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from . import paths
from .pin import PinError, load as load_pin

RELEASE_TAG = re.compile(r"^b(\d+)$")
BINARIES = ("llama-cli", "llama-server", "llama-bench")
# The catalogued Qwen3.5/3.8 models need this graph; a release without it
# would leave them unusable, so it is never activated.
REQUIRED_SOURCES = ("CMakeLists.txt", os.path.join("src", "models", "qwen35.cpp"))
RECORD = "RELEASE.json"
GIT_TIMEOUT = 600
SMOKE_TIMEOUT = 120


class UpdateError(Exception):
    """An update could not be checked, built or switched; nothing changed."""

    def __init__(self, message: str, code: int = 5):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Release:
    tag: str
    commit: str

    @property
    def number(self) -> int:
        match = RELEASE_TAG.match(self.tag)
        return int(match.group(1)) if match else -1

    @property
    def name(self) -> str:
        return f"{self.tag}-{self.commit[:12]}"


@dataclass(frozen=True)
class Generation:
    path: str
    release: Release
    built: str
    backends: tuple[str, ...]

    def bin_dir(self, backend: str) -> str:
        return os.path.join(self.path, backend, "bin")

    def has(self, backend: str) -> bool:
        return backend in self.backends and binaries_present(self.bin_dir(backend))


def runtime_root() -> str:
    override = os.environ.get("KILIX_LLM_RUNTIME_DIR")
    if override:
        return override
    return os.path.join(paths.gpu_terminal_home(), "kilix-llm", "runtime")


def upstream_url() -> str:
    override = os.environ.get("KILIX_LLM_UPSTREAM_URL")
    if override:
        return override
    try:
        return load_pin().url
    except PinError as error:
        raise UpdateError(f"cannot read the llama.cpp pin: {error}", 2) from error


def pinned_commit() -> str:
    try:
        return load_pin().commit
    except PinError:
        return ""


def binaries_present(directory: str) -> bool:
    return all(os.access(os.path.join(directory, name), os.X_OK) for name in BINARIES)


# upstream --------------------------------------------------------------------

def parse_release_tags(listing: str) -> list[Release]:
    """Release tags from `git ls-remote --tags`, peeled to their commits."""
    plain: dict[str, str] = {}
    peeled: dict[str, str] = {}
    for line in listing.splitlines():
        parts = line.split()
        if len(parts) != 2 or not re.fullmatch(r"[0-9a-f]{40}", parts[0]):
            continue
        ref = parts[1]
        if not ref.startswith("refs/tags/"):
            continue
        name = ref[len("refs/tags/"):]
        target = peeled if name.endswith("^{}") else plain
        name = name.removesuffix("^{}")
        if RELEASE_TAG.match(name):
            target[name] = parts[0]
    return [Release(tag, peeled.get(tag, commit)) for tag, commit in plain.items()]


def latest_release(url: str | None = None) -> Release:
    url = url or upstream_url()
    try:
        result = subprocess.run(
            ["git", "ls-remote", "--tags", url, "refs/tags/b*"],
            stdin=subprocess.DEVNULL, capture_output=True, text=True,
            timeout=GIT_TIMEOUT, check=False,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise UpdateError(f"cannot reach {url}: {error}") from error
    if result.returncode != 0:
        detail = result.stderr.strip().splitlines()
        raise UpdateError(f"cannot list llama.cpp releases at {url}: "
                          + (detail[-1] if detail else f"git exited {result.returncode}"))
    releases = parse_release_tags(result.stdout)
    if not releases:
        raise UpdateError(f"{url} lists no bNNNN release tags")
    return max(releases, key=lambda r: r.number)


# generations -----------------------------------------------------------------

def generations_dir() -> str:
    return os.path.join(runtime_root(), "generations")


def read_generation(path: str) -> Generation | None:
    try:
        with open(os.path.join(path, RECORD), encoding="utf-8") as handle:
            record = json.load(handle)
        release = Release(str(record["tag"]), str(record["commit"]))
        backends = tuple(str(b) for b in record.get("backends", ()))
        built = str(record.get("built", ""))
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not re.fullmatch(r"[0-9a-f]{40}", release.commit):
        return None
    return Generation(os.path.realpath(path), release, built, backends)


def _link(name: str) -> Generation | None:
    link = os.path.join(runtime_root(), name)
    if not os.path.islink(link):
        return None
    target = os.path.realpath(link)
    if os.path.dirname(target) != os.path.realpath(generations_dir()):
        return None
    return read_generation(target)


def current() -> Generation | None:
    return _link("current")


def previous() -> Generation | None:
    return _link("previous")


def _write_record(path: str, release: Release, backends: Iterable[str]) -> None:
    record = {
        "tag": release.tag,
        "commit": release.commit,
        "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "backends": sorted(set(backends)),
        "source": upstream_url(),
    }
    temporary = os.path.join(path, f".{RECORD}.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, os.path.join(path, RECORD))


def _point(name: str, target: str | None) -> None:
    link = os.path.join(runtime_root(), name)
    if target is None:
        if os.path.islink(link):
            os.unlink(link)
        return
    temporary = f"{link}.tmp-{os.getpid()}"
    if os.path.lexists(temporary):
        os.unlink(temporary)
    os.symlink(os.path.relpath(target, runtime_root()), temporary)
    os.replace(temporary, link)


class _Lock:
    """One update, rollback or generation build at a time."""

    def __enter__(self) -> "_Lock":
        os.makedirs(runtime_root(), exist_ok=True)
        self.fd = os.open(os.path.join(runtime_root(), ".update.lock"),
                          os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
        fcntl.flock(self.fd, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc: object) -> None:
        fcntl.flock(self.fd, fcntl.LOCK_UN)
        os.close(self.fd)


def _git(args: list[str], cwd: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args], cwd=cwd, stdin=subprocess.DEVNULL, capture_output=True,
            text=True, timeout=GIT_TIMEOUT, check=False,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise UpdateError(f"git {args[0]} failed: {error}") from error
    if result.returncode != 0:
        detail = result.stderr.strip().splitlines()
        raise UpdateError(f"git {args[0]} failed: " + (detail[-1] if detail else str(result.returncode)))
    return result.stdout.strip()


def fetch_source(release: Release, into: str) -> None:
    """Fetch exactly this commit, shallow, and prove that is what was fetched."""
    os.makedirs(into)
    _git(["init", "-q"], into)
    _git(["fetch", "-q", "--depth", "1", upstream_url(), release.commit], into)
    _git(["-c", "advice.detachedHead=false", "checkout", "-q", "FETCH_HEAD"], into)
    head = _git(["rev-parse", "HEAD"], into)
    if head != release.commit:
        raise UpdateError(f"fetched {head}, expected {release.commit} for {release.tag}")
    for required in REQUIRED_SOURCES:
        if not os.path.isfile(os.path.join(into, required)):
            raise UpdateError(
                f"llama.cpp {release.tag} has no {required}; the catalogued models "
                "need it, so this release is not installed", 6)


def smoke_test(bin_dir: str, release: Release) -> None:
    """Every binary starts and reports the commit it was built from."""
    for name in ("llama-cli", "llama-server"):
        try:
            result = subprocess.run(
                [os.path.join(bin_dir, name), "--version"], stdin=subprocess.DEVNULL,
                capture_output=True, text=True, timeout=SMOKE_TIMEOUT, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise UpdateError(f"{name} from {release.tag} does not start: {error}", 6) from error
        output = result.stdout + result.stderr
        if result.returncode != 0 or release.commit[:7] not in output:
            last = output.strip().splitlines()
            raise UpdateError(
                f"{name} from {release.tag} failed its check (status {result.returncode}): "
                + (last[-1] if last else "no output"), 6)


Verify = Callable[[str, str], None]


def _say(message: str) -> None:
    print(f"kilix-llm: {message}", file=sys.stderr, flush=True)


def build_generation(
    release: Release, backends: Iterable[str], build_script: str,
    log: Callable[[str], None] = _say, verify: Verify | None = None,
) -> Generation:
    """Build these backends of a release into its generation and check them.

    Backends the generation already has are kept. The generation is only
    recorded with a backend once that backend built, ran, and accepted every
    command line kilix-llm gives it (`verify(bin_dir, backend)`).
    """
    wanted = sorted(set(backends))
    os.makedirs(generations_dir(), exist_ok=True)
    path = os.path.join(generations_dir(), release.name)
    existing = read_generation(path)
    have = [b for b in (existing.backends if existing else ()) if binaries_present(os.path.join(path, b, "bin"))]
    missing = [b for b in wanted if b not in have]
    if not missing:
        return Generation(os.path.realpath(path), release, existing.built if existing else "", tuple(have))
    os.makedirs(path, exist_ok=True)
    source = tempfile.mkdtemp(prefix=f".src-{release.name}-", dir=runtime_root())
    try:
        shutil.rmtree(source)
        log(f"fetching llama.cpp {release.tag} ({release.commit[:12]})")
        fetch_source(release, source)
        for backend in missing:
            log(f"building the {backend} runtime for llama.cpp {release.tag}; this takes a few minutes")
            try:
                log_fd = sys.stderr.fileno()
            except (AttributeError, OSError, ValueError):
                log_fd = None
            status = subprocess.call(
                [build_script, "--backend", backend, "--source", source,
                 "--commit", release.commit, "--build-dir", os.path.join(path, backend)],
                stdin=subprocess.DEVNULL, stdout=log_fd,
            )
            if status != 0:
                raise UpdateError(f"the {backend} build of llama.cpp {release.tag} failed (status {status})", 6)
            smoke_test(os.path.join(path, backend, "bin"), release)
            if verify is not None:
                verify(os.path.join(path, backend, "bin"), backend)
            have.append(backend)
            _write_record(path, release, have)
    finally:
        shutil.rmtree(source, ignore_errors=True)
    generation = read_generation(path)
    assert generation is not None
    return generation


def activate(generation: Generation) -> None:
    """Make this generation current; the one it replaces becomes previous."""
    old = current()
    if old is not None and old.path != generation.path:
        _point("previous", old.path)
    elif old is None:
        _point("previous", None)  # the pinned build is what rollback returns to
    _point("current", generation.path)
    prune()


def prune() -> None:
    keep = {g.path for g in (current(), previous()) if g is not None}
    try:
        names = os.listdir(generations_dir())
    except OSError:
        return
    for name in names:
        path = os.path.realpath(os.path.join(generations_dir(), name))
        if path not in keep:
            shutil.rmtree(path, ignore_errors=True)


def rollback() -> Generation | None:
    """Switch back to the previous generation, or to the pinned build.

    Returns the generation now in use, or None for the pinned build.
    """
    with _Lock():
        old, back = current(), previous()
        if old is None:
            raise UpdateError("already using the pinned llama.cpp build; there is nothing to roll back to", 7)
        if back is None:
            _point("current", None)
            _point("previous", old.path)
            return None
        _point("current", back.path)
        _point("previous", old.path)
        return back


def update(
    backends: Iterable[str], build_script: str, release: Release | None = None,
    log: Callable[[str], None] = _say, verify: Verify | None = None,
) -> Generation:
    """Build the newest release for these backends and switch to it."""
    with _Lock():
        release = release or latest_release()
        generation = build_generation(release, backends, build_script, log, verify)
        activate(generation)
        return generation


def ensure_backend(backend: str, build_script: str, verify: Verify | None = None) -> None:
    """Build a backend the current generation was not built with (first use)."""
    with _Lock():
        generation = current()
        if generation is None or generation.has(backend):
            return
        build_generation(generation.release, [*generation.backends, backend], build_script,
                         verify=verify)
