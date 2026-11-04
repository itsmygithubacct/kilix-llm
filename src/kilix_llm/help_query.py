"""Explicit experimental document-help delegation; no model promotion."""

from __future__ import annotations

import os
from pathlib import Path
import re


def command(module: str, answer_run: str, rank_run: str, question: str,
            max_new_tokens: int = 96, sizer: str | None = None) -> list[str]:
    for name in (answer_run, rank_run):
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", name):
            raise ValueError("help run IDs must use 1-64 lowercase letters, digits, underscores or hyphens")
    if not question.strip() or len(question) > 2000 or "\0" in question:
        raise ValueError("help question must contain 1-2000 characters without NUL")
    if not 1 <= max_new_tokens <= 256:
        raise ValueError("help generation limit must be 1-256 tokens")
    launcher = Path(module).expanduser().resolve() / "kilix-help-llm"
    if not launcher.is_file() or not os.access(launcher, os.X_OK):
        raise ValueError("help module must contain an executable kilix-help-llm launcher")
    argv = [str(launcher), "ask", "--run", answer_run, "--rank-run", rank_run,
            "--max-new-tokens", str(max_new_tokens)]
    if sizer is not None:
        argv.extend(["--sizer", sizer])
    return [*argv, "--", question]
