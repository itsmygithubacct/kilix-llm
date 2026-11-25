"""Downloads of catalogued weights and their licence notices, only when asked.

`pull` downloads one entry. `judge` asks first: it lists what is missing
(size, source, licence notice) and downloads only after --yes or an
interactive yes. Files named with --model or --base are never downloaded:
weights the user already has are used as they are, after the digest check.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass

from . import models, paths


@dataclass(frozen=True)
class Asset:
    """One catalogued file to download into kilix-llm's model directory."""

    model_id: str
    path: str
    licence_path: str | None

    @property
    def record(self) -> dict:
        return models.spec(self.model_id)


def licence_file(model_id: str, weights: str) -> str | None:
    """Where an entry's licence text is saved: beside its weights, when it ships one."""
    shipped = models.spec(model_id).get("licence", {}).get("file")
    if not shipped:
        return None
    return os.path.join(os.path.dirname(weights), shipped["name"])


def _complete(path: str, size: int) -> bool:
    return os.path.isfile(path) and os.path.getsize(path) == size


def needed(model_id: str, override: str | None = None,
           base_override: str | None = None) -> list[Asset]:
    """What must be downloaded before `model_id` can run, adapter first, then its base.

    A partial file in the model directory is resumed. A file named with
    --model or --base that is missing or the wrong length is an error, never
    a download.
    """
    models.require_released(model_id)
    base = models.base_id(model_id)
    wanted = [(model_id, override)]
    if base is not None:
        models.require_released(base)
        wanted.append((base, base_override))
    assets = []
    for entry, named in wanted:
        record = models.spec(entry)
        if named:
            if not os.path.isfile(named):
                raise models.ModelError(f"{named} is missing (named for {record['title']})")
            if os.path.getsize(named) != record["size"]:
                raise models.ModelError(
                    f"{named} is {os.path.getsize(named)} bytes, expected {record['size']} "
                    f"for {record['title']}"
                )
            continue
        path = models.model_path(entry)
        notice = licence_file(entry, path)
        notice_size = record.get("licence", {}).get("file", {}).get("size")
        if not _complete(path, record["size"]) or (notice and not _complete(notice, notice_size)):
            assets.append(Asset(entry, path, notice))
    return assets


def describe(assets: list[Asset]) -> str:
    """What a download would fetch: size, source and licence notice of each file."""
    lines = []
    total = 0
    for asset in assets:
        record = asset.record
        total += record["size"]
        lines.append(f"  {record['title']} ({asset.model_id}): {record['size']:,} bytes")
        lines.append(f"    from {record['url']}")
        licence = record.get("licence")
        if licence:
            lines.append(f"    licence: {licence['id']}. {licence['notice']}")
            if licence.get("url"):
                lines.append(f"    licence text: {licence['url']}")
        lines.append(f"    to {asset.path}")
    lines.append(f"  total {total:,} bytes")
    return "\n".join(lines)


def _curl(url: str, dest: str, resume: bool) -> None:
    curl = shutil.which("curl")
    if curl is None:
        raise models.ModelError("curl is required to download weights", 4)
    argv = [curl, "-fL", "--retry", "3", "-o", dest, url]
    if resume:
        argv[4:4] = ["-C", "-"]
    try:
        out = sys.stderr.fileno()
    except (AttributeError, OSError, ValueError):
        out = None
    # curl's progress goes to stderr; stdout stays clean for a command's answer.
    rc = subprocess.call(argv, stdin=subprocess.DEVNULL, stdout=out)
    if rc != 0:
        raise models.ModelError(f"download of {url} failed (curl status {rc})", rc)


def fetch(asset: Asset) -> None:
    """Download (resuming) one asset and its licence text, then check both digests."""
    record = asset.record
    os.makedirs(os.path.dirname(asset.path) or paths.models_dir(), exist_ok=True)
    if not _complete(asset.path, record["size"]):
        print(f"kilix-llm: fetching {record['file']} ({record['size']} bytes)", file=sys.stderr)
        _curl(record["url"], asset.path, resume=True)
    got = models.digest_file(asset.path)
    if got != record["sha256"]:
        raise models.ModelError(f"{asset.path} sha256 is {got}, expected {record['sha256']}")
    if asset.licence_path:
        shipped = record["licence"]["file"]
        if not _complete(asset.licence_path, shipped["size"]):
            partial = asset.licence_path + ".part"
            _curl(shipped["url"], partial, resume=False)
            if models.digest_file(partial) != shipped["sha256"]:
                os.unlink(partial)
                raise models.ModelError(f"the licence text from {shipped['url']} does not match its record")
            os.replace(partial, asset.licence_path)
        print(f"kilix-llm: {record['licence']['id']} text saved at {asset.licence_path}", file=sys.stderr)
