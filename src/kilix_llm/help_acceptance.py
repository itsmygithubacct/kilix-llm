"""Apply a frozen document-help acceptance policy to a module query result."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import secrets
import subprocess

from . import paths


FALLBACK = "I cannot determine this from the supplied excerpt."


def _digest(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _private_json(path: Path, directory: Path) -> tuple[dict, bytes]:
    if path.is_symlink() or path.resolve().parent != directory.resolve() or not path.is_file():
        raise ValueError("help policy must be a regular private qualification file")
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("help policy must contain a JSON object")
    return value, raw


def load_policy(filename: str, answer_run: str, rank_run: str) -> tuple[dict, dict]:
    """Check the threshold artifact and its frozen run reports before use."""
    root = Path(paths.gpu_terminal_home()) / "kilix-help-llm"
    directory = root / "qualification"
    policy, _ = _private_json(Path(filename).expanduser(), directory)
    freeze, freeze_raw = _private_json(directory / "operations-r9-freeze.json", directory)
    if (policy.get("schema") != "kilix.help-llm.calibration-threshold/v1" or
            policy.get("freeze_sha256") != hashlib.sha256(freeze_raw).hexdigest() or
            policy.get("primary_answer_run") != answer_run or
            policy.get("primary_rank_run") != rank_run or
            policy.get("fallback") != FALLBACK or
            policy.get("threshold_inclusive") is not True or
            policy.get("test_opened") is not False):
        raise ValueError("help policy does not bind this frozen run pair")
    threshold = policy.get("threshold")
    if threshold is not None and (type(threshold) not in (int, float) or not math.isfinite(threshold)):
        raise ValueError("help policy threshold must be finite or null")
    for run in (answer_run, rank_run):
        bound = freeze["runs"][run]
        report_file = root / "runs" / run / "run.json"
        if report_file.is_symlink() or not report_file.is_file():
            raise ValueError("frozen help run report is missing")
        if hashlib.sha256(report_file.read_bytes()).hexdigest() != bound["report_sha256"]:
            raise ValueError("frozen help run report changed")
    return policy, freeze


def apply(result: dict, policy: dict, freeze: dict) -> dict:
    """Produce user-visible JSON; retain the unmodified result elsewhere."""
    rank_name = policy["primary_rank_run"]
    answer_name = policy["primary_answer_run"]
    ranking = result.get("ranking")
    if not isinstance(ranking, dict):
        raise ValueError("help result is missing learned ranking evidence")
    scores, order = ranking.get("scores"), ranking.get("order")
    source = result.get("source")
    if (result.get("answer_run") != answer_name or ranking.get("run") != rank_name or
            result.get("dataset_sha256") != freeze["dataset_sha256"] or
            result.get("source_revision") != freeze["source_revision"] or
            result.get("answer_run_sha256") != _digest(freeze["runs"][answer_name]["report"]) or
            ranking.get("run_sha256") != _digest(freeze["runs"][rank_name]["report"]) or
            ranking.get("residency") != "sequential" or
            not isinstance(scores, list) or not isinstance(order, list) or
            not scores or len(scores) != len(order) or
            type(scores[0]) not in (int, float) or not math.isfinite(scores[0]) or
            not isinstance(source, dict) or source.get("id") != order[0]):
        raise ValueError("help result differs from the frozen sequential system")
    answer = result.get("answer")
    if answer is not None and not isinstance(answer, str):
        raise ValueError("help model answer must be text or null")
    refusal = result.get("explicit_refusal") is True and answer is not None
    accepted = (not refusal and answer is not None and
                result.get("citation_valid") is True and
                (policy["threshold"] is None or scores[0] >= policy["threshold"]))
    visible = {key: value for key, value in result.items() if key != "draft"}
    if not refusal and not accepted:
        visible["answer"] = FALLBACK
        visible["citation_valid"] = False
        visible["explicit_refusal"] = True
        visible["abstained"] = True
    visible["ranking"] = {**ranking, "calibrated": True}
    visible["calibrated"] = True
    visible["acceptance"] = {
        "threshold": policy["threshold"], "threshold_inclusive": True,
        "top_score": scores[0], "model_answer_delivered": refusal or accepted,
        "fallback_used": not (refusal or accepted),
    }
    visible["quality"] = "unqualified"
    visible["qualification_eligible"] = False
    return visible


def save_diagnostic(result: dict, policy_file: str) -> Path:
    """Keep the untouched model draft in private user data, off stdout."""
    directory = Path(paths.gpu_terminal_home()) / "kilix-help-llm" / "diagnostics"
    os.umask(0o077)
    directory.mkdir(parents=True, mode=0o700, exist_ok=True)
    if directory.is_symlink() or directory.stat().st_mode & 0o077:
        raise ValueError("help diagnostics directory is not private")
    path = directory / f"ask-help-{secrets.token_hex(16)}.json"
    with path.open("x") as stream:
        json.dump({"raw_result": result,
                   "policy_sha256": hashlib.sha256(Path(policy_file).read_bytes()).hexdigest()},
                  stream, indent=2)
        stream.write("\n")
    return path


def resolve_source(result: dict, revision: str, docs_root: str) -> dict:
    """Verify that the cited paragraph is present at the pinned Git revision."""
    source = result.get("source")
    if not isinstance(source, dict) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("help source or revision is invalid")
    relative = source.get("path")
    if not isinstance(relative, str):
        raise ValueError("help source path is missing")
    path = PurePosixPath(relative)
    if (path.is_absolute() or ".." in path.parts or
            path.parts[:3] != ("docs", "help", "operations") or
            path.suffix != ".md"):
        raise ValueError("help source is outside the operations documentation")
    line, text, source_id = source.get("line"), source.get("text"), source.get("id")
    if (type(line) is not int or line < 1 or not isinstance(text, str) or
            not isinstance(source_id, str) or ":" not in source_id):
        raise ValueError("help source paragraph metadata is invalid")
    document, _, ordinal = source_id.partition(":")
    if document != path.stem or not ordinal.isdecimal():
        raise ValueError("help source ID does not resolve to the document")
    repo = Path(docs_root).expanduser().resolve()
    if not repo.is_dir():
        raise ValueError("help documentation checkout is missing")
    command = ["git", "--no-replace-objects", "-C", str(repo)]
    try:
        commit = subprocess.run([*command, "rev-parse", revision + "^{commit}"],
                                capture_output=True, check=True, timeout=30).stdout.decode().strip()
        tree = subprocess.run([*command, "ls-tree", revision, "--", relative],
                              capture_output=True, check=True, timeout=30).stdout.decode()
        blob = subprocess.run([*command, "show", revision + ":" + relative],
                              capture_output=True, check=True, timeout=30).stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise ValueError("pinned help documentation is unavailable") from error
    tree_parts = tree.split()
    if (commit != revision or len(tree_parts) < 4 or
            tree_parts[0] not in {"100644", "100755"} or tree_parts[1] != "blob" or
            len(blob) > 1024 * 1024):
        raise ValueError("help source is not a bounded pinned document")
    try:
        body = blob.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("help document is not UTF-8") from error
    paragraphs = list(re.finditer(r"\S[\s\S]*?(?=\n\s*\n|\Z)", body))
    index = int(ordinal)
    if index >= len(paragraphs):
        raise ValueError("help source paragraph index is absent")
    match = paragraphs[index]
    if match.group().strip() != text or body[:match.start()].count("\n") + 1 != line:
        raise ValueError("help source paragraph differs from the pinned revision")
    return {"revision": revision, "path": relative, "line": line,
            "source_id": source_id, "verified": True}
