"""Paired tails, aligned neurons, topic success, and sweep word-mass rank.

These tests do not load a checkpoint.
"""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from kilix_llm_lab import LabError
from kilix_llm_lab.algebra import first_divergence, paired_class_mean, top_aligned
from kilix_llm_lab.cli import main
from kilix_llm_lab.measure import topic_success
from kilix_llm_lab.rank import distinct4, rank_units


class StudyTests(unittest.TestCase):
    def test_first_divergence_skips_the_shared_prefix(self) -> None:
        self.assertEqual(first_divergence([1, 2, 3, 9], [1, 2, 3, 4]), 3)
        self.assertEqual(first_divergence([1, 2, 3], [1, 2, 3]), 2)
        self.assertEqual(first_divergence([1, 2, 3], [1, 2]), 1)
        self.assertEqual(first_divergence([1], [2]), 0)
        with self.assertRaises(LabError):
            first_divergence([], [1])

    def test_paired_class_mean_drops_the_prefix(self) -> None:
        examples = [
            ([[1.0, 0.0], [3.0, 0.0], [5.0, 0.0]], 1),
            ([[1.0, 0.0], [7.0, 0.0]], 1),
        ]
        got = paired_class_mean(examples)
        self.assertAlmostEqual(got[0], 5.5)
        self.assertAlmostEqual(got[1], 0.0)

    def test_aligned_score_is_contrast_times_column_dot(self) -> None:
        ranked = top_aligned(
            [2.0, -1.0, 4.0],
            [[0.0, 3.0], [4.0, 0.0], [0.0, 1.0]],
            [0.0, 1.0],
            keep=2,
        )
        self.assertEqual(ranked[0], (0, 6.0))
        self.assertEqual(ranked[1], (2, 4.0))

    def test_topic_success_requires_target_topic_and_variety(self) -> None:
        self.assertTrue(topic_success("Winter is a red tomato.", "about winter", "tomato"))
        self.assertFalse(topic_success("Winter is a red tomato.", "the weekend", "tomato"))
        self.assertFalse(
            topic_success("tomato tomato tomato tomato soup", "tomato soup", "tomato")
        )
        self.assertTrue(
            topic_success("The weekend weekend. I am a tomato.", "the weekend", "tomato")
        )
        with self.assertRaises(LabError):
            topic_success("tomato", "", "tomato")

    def test_word_mass_ranks_a_neuron_above_bland_controls(self) -> None:
        rows = [
            {
                "kind": "neuron",
                "index": 3,
                "prompt": "a",
                "text": "tomato sauce and tomato soup",
                "token_ids": [1, 1, 1, 1, 1],
            },
            {"kind": "neuron", "index": 3, "prompt": "b", "text": "fresh tomato soup"},
            {"kind": "control", "index": 0, "prompt": "a", "text": "fresh river water"},
            {"kind": "control", "index": 1, "prompt": "a", "text": "quiet river water"},
        ]
        ranked = rank_units(rows)
        self.assertEqual(ranked[0]["index"], 3)
        self.assertEqual(ranked[0]["kind"], "neuron")
        self.assertAlmostEqual(ranked[0]["mass"], 2.469)
        self.assertEqual(ranked[0]["null_pct"], 1.0)
        self.assertEqual(ranked[0]["words"][0][0], "tomato")
        self.assertEqual(ranked[0]["words"][1][0], "soup")
        self.assertEqual(ranked[0]["distinct4"], 0.5)
        self.assertEqual(distinct4([1, 2, 3]), 1.0)

    def test_baseline_words_and_high_strength_are_left_out(self) -> None:
        rows = [
            {"kind": "baseline", "index": -1, "prompt": "a", "text": "tomato baseline words"},
            {"kind": "neuron", "index": 1, "strength": 10, "prompt": "p", "text": "alpha tomato words"},
            {"kind": "neuron", "index": 1, "strength": 1, "prompt": "p", "text": "alpha tomato words"},
            {"kind": "control", "index": 0, "strength": 1, "prompt": "p", "text": "quiet river water"},
        ]
        open_rank = rank_units(rows)
        held = rank_units(rows, max_strength=1)
        self.assertGreater(open_rank[0]["mass"], 0)
        self.assertEqual(held[0]["mass"], 0)
        self.assertTrue(all("tomato" not in [word for word, _c, _z in item["words"]] for item in open_rank))

    def test_rank_command_reads_jsonl_without_a_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "kind": "neuron",
                                "index": 3,
                                "generations": [
                                    {"prompt": "a", "text": "tomato sauce and tomato soup"},
                                    {"prompt": "b", "text": "fresh tomato soup"},
                                ],
                            }
                        ),
                        json.dumps(
                            {
                                "kind": "control",
                                "index": 0,
                                "prompt": "a",
                                "text": "fresh river water",
                            }
                        ),
                        json.dumps(
                            {
                                "kind": "control",
                                "index": 1,
                                "prompt": "a",
                                "text": "quiet river water",
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = main(["rank", str(path), "--show", "1"])
        payload = json.loads(stdout.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(payload["units"], 3)
        self.assertEqual(payload["controls"], 2)
        self.assertEqual(payload["top"][0]["index"], 3)
        self.assertEqual(payload["top"][0]["null_pct"], 1.0)

    def test_empty_pairs_file_is_refused_before_a_model_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.json"
            path.write_text("[]", encoding="utf-8")
            stderr = io.StringIO()
            with redirect_stderr(stderr), redirect_stdout(io.StringIO()):
                code = main(["means", "--pairs", str(path)])
        self.assertEqual(code, 1)
        self.assertIn("pairs file", stderr.getvalue())
        self.assertNotIn("torch", stderr.getvalue())
