"""Download a Hugging Face checkpoint into the current HF_HOME cache.

This is setup. It is not a benchmark. Set HF_HOME before importing this
module when the snapshot should live on a large disk.
"""

from __future__ import annotations

import sys

from kilix_llm_lab import LabError


def fetch(model_id: str) -> str:
    if not model_id or model_id.startswith("-"):
        raise LabError("pass a Hugging Face model id")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise LabError("huggingface_hub is not installed") from error
    return snapshot_download(model_id)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: python -m kilix_llm_lab.fetch <model-id>", file=sys.stderr)
        return 2
    try:
        path = fetch(args[0])
    except LabError as error:
        print(f"kilix-llm lab: {error}", file=sys.stderr)
        return 1
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
