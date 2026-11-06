"""Output-row solver, sealed Ask ids, and copy-only patching. No checkpoint."""

from __future__ import annotations

import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

from kilix_llm_lab import LabError
from kilix_llm_lab.cli import main
from kilix_llm_lab.output_row import (
    SCHEMA,
    assert_fittable,
    patch_bytes,
    requantize_row,
    solve_output_row,
    write_patched_copy,
)


def _capture(case_ids, **extra):
    payload = {
        "schema": SCHEMA,
        "states": [[1.0, 0.0], [0.0, 1.0]],
        "sign": [1.0, -1.0],
        "bound": [0.4, 0.2],
        "case_ids": case_ids,
    }
    payload.update(extra)
    return payload


class OutputRowTests(unittest.TestCase):
    def test_separable_hinge_matches_the_closed_form(self) -> None:
        regularization = 0.01
        report = solve_output_row(
            [[1.0, 0.0], [0.0, 1.0]],
            [1.0, -1.0],
            [0.4, 0.2],
            regularization=regularization,
            case_ids=["dev-a", "dev-b"],
            old_row=[1.0, 1.0],
        )
        self.assertTrue(report["success"])
        self.assertFalse(report["network_backward"])
        self.assertEqual(report["cases_checked"], 2)
        divisor = 1.0 + 2.0 * regularization
        self.assertAlmostEqual(report["delta"][0], 0.4 / divisor, places=4)
        self.assertAlmostEqual(report["delta"][1], -0.2 / divisor, places=4)
        try:
            import numpy as np
        except ImportError:
            np = None
        if np is not None:
            array_report = solve_output_row(
                np.array([[1.0, 0.0], [0.0, 1.0]]),
                np.array([1.0, -1.0]),
                np.array([0.4, 0.2]),
                regularization=regularization,
                case_ids=["dev-a", "dev-b"],
            )
            self.assertAlmostEqual(array_report["delta"][0], report["delta"][0], places=4)
            self.assertAlmostEqual(array_report["delta"][1], report["delta"][1], places=4)
        self.assertGreater(report["float_relative_l2"], 0.0)
        self.assertEqual(len(report["correction_sha256"]), 64)

    def test_gradient_matches_a_central_difference(self) -> None:
        from kilix_llm_lab.output_row import _prepare

        problem = _prepare(
            [[1.0, 0.2], [0.3, 1.0]],
            [1.0, -1.0],
            [0.4, 0.25],
            [2.0, 1.0],
            1e-4,
            20,
            1e-8,
            0.1,
            ["dev-a", "dev-b"],
            None,
        )
        point = [0.1, -0.05]
        _, grad = problem.loss_grad(point)
        step = 1e-5
        for index in range(2):
            plus = list(point)
            minus = list(point)
            plus[index] += step
            minus[index] -= step
            right, _ = problem.loss_grad(plus)
            left, _ = problem.loss_grad(minus)
            self.assertAlmostEqual(grad[index], (right - left) / (2 * step), places=6)

    def test_sealed_confirmation_ids_do_not_fit(self) -> None:
        for case_id in (
            "copy-fresh-009",
            "copy-fresh-022",
            "copy-fresh-160",
            "copy-operational-007",
            "copy-heldout-058",
        ):
            with self.assertRaises(LabError):
                assert_fittable(["dev-a", case_id])
            with self.assertRaises(LabError):
                solve_output_row([[1.0]], [1.0], [0.1], case_ids=[case_id])
        assert_fittable(["copy-probe-010", "known-ask-0034"])

    def test_rejects_bad_captures(self) -> None:
        with self.assertRaises(LabError):
            solve_output_row([[1.0, 0.0], [1.0]], [1.0, -1.0], [0.1, 0.1])
        with self.assertRaises(LabError):
            solve_output_row([[1.0]], [0.0], [0.1], case_ids=["dev-a"])
        with self.assertRaises(LabError):
            solve_output_row([[math.nan]], [1.0], [0.1], case_ids=["dev-a"])
        with self.assertRaises(LabError):
            solve_output_row([[1.0]], [1.0], [0.1], weights=[0.0], case_ids=["dev-a"])
        with self.assertRaises(LabError):
            solve_output_row([[1.0]], [1.0], [0.1], regularization=-1.0, case_ids=["dev-a"])
        with self.assertRaises(LabError):
            solve_output_row([[1.0]], [1.0], [0.1], case_ids=["dev-a", "dev-b"])

    def test_patch_changes_only_the_named_window(self) -> None:
        source = b"abcdefghijklmnopqrstuvwxyz"
        edited = patch_bytes(source, [(3, b"XYZ")])
        self.assertEqual(edited, b"abcXYZghijklmnopqrstuvwxyz")
        self.assertEqual(source, b"abcdefghijklmnopqrstuvwxyz")
        with self.assertRaises(LabError):
            patch_bytes(source, [(20, b"1234567")])
        with self.assertRaises(LabError):
            patch_bytes(source, [(2, b"www"), (4, b"zz")])

    def test_copy_refuses_to_overwrite_the_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "parent.bin"
            source.write_bytes(b"0123456789")
            with self.assertRaises(LabError):
                write_patched_copy(source, source, [(0, b"ab")])
            dest = root / "child.bin"
            written = write_patched_copy(source, dest, [(2, b"ZZ")], expected_sha256=_sha(source.read_bytes()))
            self.assertEqual(source.read_bytes(), b"0123456789")
            self.assertEqual(dest.read_bytes(), b"01ZZ456789")
            self.assertFalse(written["source_opened_for_write"])
            with self.assertRaises(LabError):
                write_patched_copy(source, dest, [(0, b"q")])
            with self.assertRaises(LabError):
                write_patched_copy(source, root / "other.bin", [(0, b"q")], expected_sha256="0" * 64)

    def test_command_solves_a_json_capture_without_a_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.json"
            path.write_text(json.dumps(_capture(["dev-a", "dev-b"])), encoding="utf-8")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = main(["output-row", "--capture", str(path)])
            self.assertEqual(code, 0)
            report = json.loads(stdout.getvalue())
            self.assertTrue(report["success"])
            self.assertNotIn("delta", report)
            self.assertNotIn("tokens_per_second", report)
            self.assertEqual(report["schema"], SCHEMA)

    def test_npz_command_prints_json(self) -> None:
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is not installed")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.npz"
            np.savez(
                path,
                states=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
                sign=np.array([1.0, -1.0]),
                bound=np.array([0.4, 0.2]),
                case_ids=np.array(["dev-a", "dev-b"]),
                meta=np.array(json.dumps({"regularization": 0.01, "row": 3})),
            )
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = main(["output-row", "--capture", str(path), "--max-iter", "50", "--row", "3"])
            self.assertEqual(code, 0)
            report = json.loads(stdout.getvalue())
            self.assertTrue(report["success"])
            self.assertIsInstance(report["guard_violations_over_limit"], int)
            self.assertIsInstance(report["iterations"], int)
            self.assertEqual(report["row"], 3)
            self.assertNotIn("delta", report)

    def test_command_refuses_a_sealed_capture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.json"
            path.write_text(json.dumps(_capture(["copy-heldout-001"])), encoding="utf-8")
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = main(["output-row", "--capture", str(path)])
            self.assertEqual(code, 1)
            self.assertIn("sealed", stderr.getvalue())

    def test_q8_requantize_keeps_one_block(self) -> None:
        values = [0.1] * 32
        delta = [0.0] * 32
        delta[0] = 0.5
        try:
            result = requantize_row(values, delta)
        except LabError as error:
            text = str(error)
            if "numpy" in text or "gguf" in text:
                self.skipTest(text)
            raise
        self.assertEqual(len(result["bytes"]), 34)
        self.assertIsNotNone(result["quantized_relative_l2"])


def _sha(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


if __name__ == "__main__":
    unittest.main()
