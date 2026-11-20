#!/usr/bin/env python3
"""Run tests/ on the system interpreter.

When third_party/llama.cpp/CMakeLists.txt is absent, only these tests may fail:

  LauncherTests.test_run_refuses_a_missing_model_and_names_pull
  PinTests.test_vendored_tree_has_the_qwen_graph

Any other failure or any error fails the run.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {
    "LauncherTests.test_run_refuses_a_missing_model_and_names_pull",
    "PinTests.test_vendored_tree_has_the_qwen_graph",
}


def main() -> None:
    sys.path.insert(0, str(ROOT / "src"))
    vendor = ROOT / "third_party" / "llama.cpp" / "CMakeLists.txt"
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    failed = [case.id() for case, _trace in result.failures]
    errored = [case.id() for case, _trace in result.errors]
    if errored:
        print("run-tests: errors: " + ", ".join(errored), file=sys.stderr)
        raise SystemExit(1)
    if vendor.is_file():
        if failed:
            print("run-tests: failures with the vendor tree present: " + ", ".join(failed), file=sys.stderr)
            raise SystemExit(1)
    else:
        unexpected = [name for name in failed if not any(name.endswith(item) for item in ALLOWED)]
        if unexpected:
            print("run-tests: unexpected failures: " + ", ".join(unexpected), file=sys.stderr)
            raise SystemExit(1)
    print("run-tests: passed")


if __name__ == "__main__":
    main()
