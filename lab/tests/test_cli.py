"""The project command line builds without loading a model."""

from __future__ import annotations

import contextlib
import io
import unittest

from kilix_llm_lab import LabError
from kilix_llm_lab.cli import build_parser
from kilix_llm_lab.runner import format_story


class CliTests(unittest.TestCase):
    def test_story_help_mentions_abliteration(self) -> None:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), self.assertRaises(SystemExit) as caught:
            build_parser().parse_args(["story", "--help"])
        self.assertEqual(caught.exception.code, 0)
        self.assertIn("--abliterate", buffer.getvalue())

    def test_neurons_require_a_layer(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(["neurons"])

    def test_story_report_includes_the_scores(self) -> None:
        text = format_story(
            {
                "layer": 11,
                "norms": [0.0, 1.5],
                "strength": 1.0,
                "prompt": "Tell me a story.",
                "baseline": "A bright day.",
                "baseline_mood": 1.0,
                "steered": "An empty room.",
                "steered_mood": -1.0,
                "kl": 0.25,
                "facts_before": [("q", "Tokyo", True)],
                "facts_after": [("q", "no", False)],
            }
        )
        self.assertIn("layer: 11", text)
        self.assertIn("kl: 0.250000", text)
        self.assertIn("facts_before: 1/1", text)
        self.assertIn("facts_after: 0/1", text)

    def test_bad_layer_is_a_lab_error(self) -> None:
        from kilix_llm_lab.algebra import strongest_index

        with self.assertRaises(LabError):
            strongest_index([])
