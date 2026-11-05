"""The bounded catalog records every requested push and its controls."""

from __future__ import annotations

import json
import unittest

from kilix_llm_lab.algebra import l2
from kilix_llm_lab.catalog import catalog_record, stand_in_generation
from kilix_llm_lab.cli import main
from kilix_llm_lab.texts import EXPLORER_PROMPTS


class CatalogTests(unittest.TestCase):
    def test_catalog_record_has_one_observation_per_index_and_strength_plus_controls(self) -> None:
        indexes = [1, 4]
        strengths = [1.0, -2.0]
        record = catalog_record(
            indexes,
            strengths,
            control_count=3,
            width=6,
            generate=stand_in_generation,
            seed=2,
        )
        self.assertEqual(list(record["prompts"]), list(EXPLORER_PROMPTS))
        self.assertEqual(EXPLORER_PROMPTS[0], "Who are you?")
        self.assertEqual(EXPLORER_PROMPTS[1], "Write a snake game in Python.")
        self.assertEqual(EXPLORER_PROMPTS[2], "Describe the color orange.")
        self.assertIn("whatever", EXPLORER_PROMPTS[3].lower())
        observations = record["observations"]
        neurons = [item for item in observations if item["kind"] == "neuron"]
        controls = [item for item in observations if item["kind"] == "control"]
        self.assertEqual(
            {(item["index"], item["strength"]) for item in neurons},
            {(index, strength) for index in indexes for strength in strengths},
        )
        self.assertEqual(len(controls), 3 * len(strengths))
        for item in observations:
            self.assertEqual(len(item["generations"]), len(EXPLORER_PROMPTS))
            self.assertAlmostEqual(l2(item["direction"]), abs(item["strength"]))
            self.assertEqual(
                [row["prompt"] for row in item["generations"]],
                list(EXPLORER_PROMPTS),
            )

    def test_stand_in_command_uses_the_shipped_record(self) -> None:
        import io
        from contextlib import redirect_stdout

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = main(
                [
                    "catalog",
                    "--stand-in",
                    "--index",
                    "1",
                    "--index",
                    "4",
                    "--strength",
                    "1",
                    "--strength",
                    "2",
                    "--controls",
                    "2",
                    "--width",
                    "6",
                ]
            )
        self.assertEqual(code, 0)
        payload = json.loads(buffer.getvalue())
        self.assertEqual(payload["neuron_observations"], 4)
        self.assertEqual(payload["control_observations"], 4)
        self.assertIn("Who are you?", payload["sample"])
        self.assertAlmostEqual(payload["matched_norm"], 1.0)

    def test_control_command_reports_the_win_rule(self) -> None:
        import io
        from contextlib import redirect_stdout

        won = io.StringIO()
        with redirect_stdout(won):
            self.assertEqual(main(["control", "--effect", "2", "--control", "0.5", "--control", "1"]), 0)
        lost = io.StringIO()
        with redirect_stdout(lost):
            self.assertEqual(main(["control", "--effect", "2", "--control", "3"]), 0)
        self.assertTrue(json.loads(won.getvalue())["wins"])
        self.assertFalse(json.loads(lost.getvalue())["wins"])
