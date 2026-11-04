"""Delegate default Help answers or explicitly selected historical runs."""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil


def command(module: str | None, answer_run: str | None, rank_run: str | None, question: str,
            max_new_tokens: int = 96, sizer: str | None = None) -> list[str]:
    if bool(answer_run) != bool(rank_run):
        raise ValueError("explicit Help runs require both --answer-run and --rank-run")
    for name in (answer_run, rank_run) if answer_run else ():
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", name):
            raise ValueError("help run IDs must use 1-64 lowercase letters, digits, underscores or hyphens")
    if not question.strip() or len(question) > 2000 or "\0" in question:
        raise ValueError("help question must contain 1-2000 characters without NUL")
    if not 1 <= max_new_tokens <= 256:
        raise ValueError("help generation limit must be 1-256 tokens")
    module = module or os.environ.get("KILIX_HELP_MODULE")
    if module is None and not answer_run:
        launcher = shutil.which("kilix-help-llm")
        if launcher is None:
            host = shutil.which("kilix")
            if host is None:
                raise ValueError("install Kilix or provide --help-module PATH")
            argv = [host, "help-answer", "--json"]
        else:
            argv = [launcher, "answer", "--json"]
    else:
        if module is None:
            raise ValueError("explicit Help runs require --help-module PATH")
        launcher = Path(module).expanduser().resolve() / "kilix-help-llm"
        if not launcher.is_file() or not os.access(launcher, os.X_OK):
            raise ValueError("help module must contain an executable kilix-help-llm launcher")
        argv = [str(launcher), "answer", "--json"]
    if not answer_run and max_new_tokens != 96:
        raise ValueError("the default Help model uses a fixed 96-token generation limit")
    if answer_run:
        argv = [str(launcher), "ask", "--run", answer_run, "--rank-run", rank_run,
                "--max-new-tokens", str(max_new_tokens)]
    if sizer is not None:
        argv.extend(["--sizer", sizer])
    return [*argv, "--", question]
