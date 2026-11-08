"""Loop detection and copy targets on text and ids. No checkpoint."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from kilix_llm_lab import LabError
from kilix_llm_lab.loops import (
    copy_agreement,
    copy_targets,
    find_loop,
    find_token_loop,
    periodic_onset,
    repeat_fraction,
)

# Shape of a Qwen3.5-0.8B thinking loop: a decision, then "Wait", forever.
CYCLE = "Okay, I'll go with Walk.\n\nWait, I need to check if there's any specific reason to drive.\n\n"


class FindLoopTest(unittest.TestCase):
    def test_loop_after_a_unique_preamble(self):
        preamble = "Thinking Process:\n\n1. Analyze the request about the distance:"
        text = preamble + CYCLE * 12
        found = find_loop(text, window=len(CYCLE), copies=3)
        self.assertIsNotNone(found)
        self.assertEqual(found["period"], len(CYCLE))
        self.assertEqual(found["onset"], len(preamble))
        self.assertAlmostEqual(found["cycles"], 12.0)
        self.assertGreaterEqual(found["copies"], 3)

    def test_cut_mid_cycle_still_reports_the_period(self):
        text = "intro. " + CYCLE * 6 + CYCLE[:17]
        found = find_loop(text, window=40, copies=3)
        self.assertEqual(found["period"], len(CYCLE))

    def test_no_loop_in_varied_text(self):
        text = " ".join(f"step {index} weighs option {index * 7 % 11}." for index in range(300))
        self.assertIsNone(find_loop(text, window=200, copies=3))

    def test_too_short(self):
        self.assertIsNone(find_loop(CYCLE, window=800, copies=3))

    def test_bad_arguments(self):
        with self.assertRaises(LabError):
            find_loop("abc", window=0)
        with self.assertRaises(LabError):
            find_loop("abc", copies=1)
        with self.assertRaises(LabError):
            periodic_onset("abc", 0)


class TokenLoopTest(unittest.TestCase):
    def test_token_loop(self):
        ids = [9, 8, 7] + [1, 2, 3, 4, 5] * 10
        found = find_token_loop(ids, window=5, copies=3)
        self.assertEqual(found["period"], 5)
        self.assertEqual(found["onset"], 3)
        self.assertAlmostEqual(found["cycles"], 10.0)

    def test_no_token_loop(self):
        self.assertIsNone(find_token_loop(list(range(100)), window=5, copies=3))


class CopyTargetTest(unittest.TestCase):
    def test_targets_follow_the_last_occurrence(self):
        ids = [1, 2, 3, 9, 1, 2, 3, 7, 1, 2, 3]
        targets = copy_targets(ids, k=3)
        self.assertEqual(targets[:6], [None, None, None, None, None, None])
        self.assertEqual(targets[6], 9)
        self.assertEqual(targets[10], 7)

    def test_agreement_is_full_inside_a_loop(self):
        ids = [5, 6] + [1, 2, 3, 4] * 6
        stats = copy_agreement(ids, k=2, start=6)
        self.assertEqual(stats["rate"], 1.0)
        self.assertGreater(stats["with_target"], 0)

    def test_k_must_be_positive(self):
        with self.assertRaises(LabError):
            copy_targets([1, 2], k=0)


class RepeatFractionTest(unittest.TestCase):
    def test_loop_repeats_fully(self):
        self.assertEqual(repeat_fraction("Fresh start.\n" + CYCLE * 20), 1.0)

    def test_distinct_lines(self):
        self.assertEqual(repeat_fraction("\n".join(f"line {index}" for index in range(10))), 0.0)


class LoopsCommandTest(unittest.TestCase):
    def test_command_reports_only_the_looping_row(self):
        from kilix_llm_lab.cli import main

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "traces.jsonl"
            rows = [
                {"prompt_id": "loop", "seed": 1, "reasoning": "Start:" + CYCLE * 10},
                {"prompt_id": "fine", "seed": 2, "reasoning": "The answer is 391 because 17 x 23 = 391."},
            ]
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = main(["loops", "--traces", str(path), "--window", "60"])
        self.assertEqual(code, 0)
        report = json.loads(stdout.getvalue())
        self.assertEqual((report["rows"], report["loops"]), (2, 1))
        self.assertEqual(report["items"][0]["id"], "loop")
        self.assertEqual(report["items"][0]["loop"]["period"], len(CYCLE))

    def test_command_refuses_a_row_without_the_field(self):
        from kilix_llm_lab.cli import main

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "traces.jsonl"
            path.write_text(json.dumps({"text": "x"}) + "\n", encoding="utf-8")
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = main(["loops", "--traces", str(path)])
        self.assertNotEqual(code, 0)
        self.assertIn("reasoning", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
