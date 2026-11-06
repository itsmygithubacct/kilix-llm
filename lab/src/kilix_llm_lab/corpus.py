"""Turn a caller-supplied corpus into one UTF-8 text file.

llama-finetune reads a single raw text file. This module does not download
a dataset and does not contain one. A directory contributes .txt, .md,
.text, .raw, and .jsonl files. Any other file is counted and left out.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from kilix_llm_lab import LabError

SCHEMA = "kilix.llm.corpus/v1"
TEXT_SUFFIXES = {".txt", ".md", ".text", ".raw"}
JSONL_SUFFIXES = {".jsonl"}
ACCEPTED_SUFFIXES = TEXT_SUFFIXES | JSONL_SUFFIXES
MAX_CORPUS_BYTES = 2 * 1024 * 1024 * 1024
_SKIPPED_LISTED = 32


def prepare_corpus(source: Path | str, dest: Path | str) -> dict:
    """Write one text file and a sidecar manifest. Return the manifest."""
    root = Path(source)
    dest = Path(dest)
    if not root.exists():
        raise LabError("corpus path does not exist")
    if dest.exists() and not dest.is_file():
        raise LabError("text output must be a file")
    if not dest.parent.is_dir():
        raise LabError("text output directory does not exist")
    if root.resolve() == dest.resolve():
        raise LabError("text output and corpus are the same file")
    manifest_path = Path(str(dest) + ".manifest.json")
    if root.is_dir() and _is_inside(dest, root):
        raise LabError("text output is inside the corpus directory")

    documents: list[str] = []
    files: list[dict] = []
    skipped: list[str] = []
    budget = _Budget()
    if root.is_file():
        _take_file(root, root.name, documents, files, skipped, budget, directory=False)
    elif root.is_dir():
        _take_tree(root, documents, files, skipped, budget)
    else:
        raise LabError("corpus path is not a file or a directory")
    if not documents:
        raise LabError("corpus has no documents")

    body = "\n\n".join(documents) + "\n"
    encoded = body.encode("utf-8")
    if len(encoded) > MAX_CORPUS_BYTES:
        raise LabError("corpus is larger than 2 GiB")
    _write_bytes(dest, encoded)
    manifest = {
        "schema": SCHEMA,
        "documents": len(documents),
        "bytes": len(encoded),
        "files": files,
        "skipped_count": len(skipped),
        "skipped": skipped[:_SKIPPED_LISTED],
    }
    _write_bytes(manifest_path, (json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
    return manifest


def _take_tree(
    root: Path,
    documents: list[str],
    files: list[dict],
    skipped: list[str],
    budget: _Budget,
) -> None:
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(
            name for name in dirnames if not name.startswith(".") and name != "__pycache__"
        )
        for name in sorted(filenames):
            path = Path(dirpath) / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                skipped.append(relative)
                continue
            _take_file(path, relative, documents, files, skipped, budget, directory=True)


def _take_file(
    path: Path,
    relative: str,
    documents: list[str],
    files: list[dict],
    skipped: list[str],
    budget: _Budget,
    *,
    directory: bool,
) -> None:
    if path.name.startswith("."):
        skipped.append(relative)
        return
    suffix = path.suffix.lower()
    if directory and suffix not in ACCEPTED_SUFFIXES:
        skipped.append(relative)
        return
    if suffix in JSONL_SUFFIXES:
        found = _jsonl_documents(path, relative)
    else:
        found = _text_documents(path, relative)
    if not found:
        skipped.append(relative)
        return
    for item in found:
        budget.add(item)
    documents.extend(found)
    files.append(
        {
            "path": relative,
            "documents": len(found),
            "bytes": sum(len(item.encode("utf-8")) for item in found),
        }
    )


def _text_documents(path: Path, relative: str) -> list[str]:
    text = _read_text(path, relative).strip()
    if not text:
        return []
    return [text]


def _jsonl_documents(path: Path, relative: str) -> list[str]:
    found: list[str] = []
    for number, line in enumerate(_read_text(path, relative).splitlines(), start=1):
        if not line.strip():
            continue
        where = f"{relative}:{number}"
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise LabError(f"{where} is not JSON") from error
        text = _document_from_row(row, where).strip()
        if text:
            found.append(text)
    return found


def _document_from_row(row: object, where: str) -> str:
    if not isinstance(row, dict):
        raise LabError(f"{where} is not a JSON object")
    if "text" in row:
        return _string_field(row, "text", where)
    if "content" in row:
        return _string_field(row, "content", where)
    if "messages" in row:
        return _messages(row["messages"], where)
    if "prompt" in row or "completion" in row:
        return _join_fields(row, ("prompt", "completion"), where)
    if "input" in row or "output" in row:
        return _join_fields(row, ("input", "output"), where)
    if "instruction" in row or "user" in row or "assistant" in row:
        return _join_fields(row, ("instruction", "user", "assistant"), where)
    raise LabError(f"{where} needs a text field")


def _messages(value: object, where: str) -> str:
    if not isinstance(value, list):
        raise LabError(f"{where} messages must be a list")
    parts: list[str] = []
    for index, message in enumerate(value):
        if not isinstance(message, dict):
            raise LabError(f"{where} message {index} is not an object")
        if "content" not in message:
            raise LabError(f"{where} message {index} needs content")
        content = message["content"]
        if not isinstance(content, str):
            raise LabError(f"{where} message {index} content must be a string")
        if content.strip():
            parts.append(content.strip())
    return "\n".join(parts)


def _join_fields(row: dict, keys: tuple[str, ...], where: str) -> str:
    parts: list[str] = []
    for key in keys:
        if key not in row or row[key] in (None, ""):
            continue
        parts.append(_string_field(row, key, where).strip())
    if not parts:
        raise LabError(f"{where} has no text")
    return "\n".join(parts)


def _string_field(row: dict, key: str, where: str) -> str:
    value = row[key]
    if not isinstance(value, str):
        raise LabError(f"{where} {key} must be a string")
    return value


def _read_text(path: Path, relative: str) -> str:
    try:
        data = path.read_bytes()
    except OSError as error:
        raise LabError(f"cannot read {relative}: {error.strerror}") from error
    if b"\x00" in data:
        raise LabError(f"{relative} is not a text file")
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise LabError(f"{relative} is not UTF-8") from error
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _write_bytes(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    except OSError as error:
        raise LabError(f"cannot write corpus output: {error.strerror}") from error
    finally:
        if temporary.exists():
            temporary.unlink()


class _Budget:
    """Running size of the joined text, excluding the final newline."""

    def __init__(self) -> None:
        self.bytes = 0
        self.docs = 0

    def add(self, text: str) -> None:
        extra = len(text.encode("utf-8"))
        separator = 2 if self.docs else 0
        if self.bytes + separator + extra + 1 > MAX_CORPUS_BYTES:
            raise LabError("corpus is larger than 2 GiB")
        self.bytes += separator + extra
        self.docs += 1


def _is_inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True
