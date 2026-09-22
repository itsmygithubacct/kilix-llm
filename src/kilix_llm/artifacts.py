"""Explicit manifests for additional GGUF artifacts, independent of built-in IDs.

This module never downloads during validation or model loading. Consumers call
fetch deliberately, then verify before using an artifact. No manifest changes
the built-in catalog or grants a memory-fit or model-licence determination.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from . import paths


@dataclass(frozen=True)
class Artifact:
    id: str
    filename: str
    size: int
    sha256: str
    url: str
    quantization: str
    source_revision: str
    license: str

    @classmethod
    def load(cls, filename: str | Path) -> Artifact:
        with Path(filename).open("rb") as handle:
            raw = handle.read(65537)
        if len(raw) > 65536:
            raise ValueError("artifact manifest exceeds 64 KiB")
        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ValueError("duplicate manifest key")
                result[key] = value
            return result
        value = json.loads(raw, object_pairs_hook=pairs)
        keys = {"schema", "id", "filename", "size", "sha256", "url",
                "quantization", "source_revision", "license"}
        if not isinstance(value, dict) or set(value) != keys:
            raise ValueError("unexpected artifact manifest fields")
        if value.pop("schema") != "kilix.llm.artifact/v1":
            raise ValueError("unsupported artifact schema")
        for key in keys - {"schema", "size"}:
            if not isinstance(value[key], str) or not 1 <= len(value[key]) <= 2048:
                raise ValueError(f"invalid artifact {key}")
        if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,95}", value["id"]):
            raise ValueError("invalid artifact ID")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,190}\.gguf", value["filename"]):
            raise ValueError("filename must be a plain GGUF basename")
        if type(value["size"]) is not int or not 24 <= value["size"] <= 1024**4:
            raise ValueError("invalid artifact size")
        if not re.fullmatch(r"[0-9a-f]{64}", value["sha256"]):
            raise ValueError("invalid artifact SHA-256")
        if not re.fullmatch(r"[0-9a-f]{40}", value["source_revision"]):
            raise ValueError("source revision must be an immutable 40-hex commit")
        url = urlsplit(value["url"])
        if (url.scheme != "https" or not url.hostname or url.username or url.password
                or url.query or url.fragment or url.port not in (None, 443)
                or value["source_revision"] not in url.path.split("/")):
            raise ValueError("artifact URL must use HTTPS and the exact source revision")
        return cls(**value)

    def destination(self, root: str | Path | None = None) -> Path:
        root = Path(root) if root is not None else Path(paths.models_dir())
        return root / "artifacts" / self.sha256 / self.filename

    def verify(self, root: str | Path | None = None) -> Path:
        path = self.destination(root)
        hasher = hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            magic = handle.read(4)
            hasher.update(magic)
            size += len(magic)
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
                size += len(chunk)
                if size > self.size:
                    raise ValueError("artifact exceeds expected size")
        if magic != b"GGUF" or size != self.size or hasher.hexdigest() != self.sha256:
            raise ValueError("artifact GGUF header, size, or SHA-256 mismatch")
        return path


class _HttpsRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urlsplit(newurl)
        if parsed.scheme != "https" or parsed.username or parsed.password:
            raise ValueError("artifact redirect must use HTTPS without credentials")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch(artifact: Artifact, root: str | Path | None = None) -> Path:
    """Download and atomically publish a verified artifact; never overwrite a bad cache."""
    destination = artifact.destination(root)
    if destination.exists():
        return artifact.verify(root)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temporary = tempfile.mkstemp(prefix=".download-", dir=destination.parent)
    opener = build_opener(ProxyHandler({}), _HttpsRedirect())
    try:
        hasher, size, magic = hashlib.sha256(), 0, b""
        with os.fdopen(fd, "wb") as output:
            with opener.open(Request(artifact.url, headers={"User-Agent": "kilix-llm"}), timeout=60) as response:
                while chunk := response.read(1024 * 1024):
                    if not size:
                        magic = chunk[:4]
                    size += len(chunk)
                    if size > artifact.size:
                        raise ValueError("download exceeds expected size")
                    hasher.update(chunk)
                    output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        if magic != b"GGUF" or size != artifact.size or hasher.hexdigest() != artifact.sha256:
            raise ValueError("download GGUF header, size, or SHA-256 mismatch")
        # Exclusive publication also handles another process completing this download.
        try:
            os.link(temporary, destination)
        except FileExistsError:
            return artifact.verify(root)
        return destination
    finally:
        Path(temporary).unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Verify or fetch a manifest-pinned GGUF")
    parser.add_argument("command", choices=("verify", "fetch", "path"))
    parser.add_argument("manifest")
    args = parser.parse_args(argv)
    try:
        artifact = Artifact.load(args.manifest)
        if args.command == "fetch":
            result = fetch(artifact)
        elif args.command == "verify":
            result = artifact.verify()
        else:
            result = artifact.destination()
        print(result)
        return 0
    except (OSError, ValueError) as exc:
        print(f"kilix-llm artifact: {exc}", file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
