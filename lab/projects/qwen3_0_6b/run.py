#!/usr/bin/env python3
"""Run the Qwen3-0.6B edit project.

The virtual environment comes from lab/scripts/uv-venv.sh (uv venv, not
python -m venv). The lab tests do not need it. See
lab/projects/qwen3_0_6b/README.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from kilix_llm_lab.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
