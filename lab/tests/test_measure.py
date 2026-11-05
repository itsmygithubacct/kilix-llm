"""Lexicon mood separates the built-in contrast texts."""

from __future__ import annotations

import unittest

from kilix_llm_lab import LabError
from kilix_llm_lab.algebra import l2, one_hot, random_directions
from kilix_llm_lab.measure import beats_every_control, fact_hit, mood_score
from kilix_llm_lab.texts import FACTS, HAPPY, OTHER_OBJECTS, SAD, TOMATO


class MeasureTests(unittest.TestCase):
    def test_builtin_stories_fall_on_opposite_sides(self) -> None:
        self.assertTrue(all(mood_score(text) > 0 for text in HAPPY))
        self.assertTrue(all(mood_score(text) < 0 for text in SAD))

    def test_neuron_push_beats_a_weaker_control_and_loses_to_a_stronger_one(self) -> None:
        push = one_hot(8, 3, 2.0)
        controls = random_directions(8, l2(push), 4, seed=1)
        for direction in controls:
            self.assertAlmostEqual(l2(direction), l2(push))
        self.assertTrue(beats_every_control(2.0, [0.5, 1.0]))
        self.assertFalse(beats_every_control(2.0, [0.5, 3.0]))
        with self.assertRaises(LabError):
            beats_every_control(2.0, [])

    def test_unknown_text_is_neutral(self) -> None:
        self.assertEqual(mood_score("The river kept moving."), 0.0)

    def test_fact_hit_is_a_case_insensitive_substring(self) -> None:
        self.assertTrue(fact_hit("The answer is Tokyo.", "tokyo"))
        self.assertFalse(fact_hit("Kyoto", "tokyo"))

    def test_contrast_sets_and_facts_are_nonempty(self) -> None:
        self.assertGreaterEqual(len(HAPPY), 8)
        self.assertGreaterEqual(len(SAD), 8)
        self.assertTrue(all("tomato" in text for text in TOMATO))
        self.assertTrue(all("tomato" not in text for text in OTHER_OBJECTS))
        self.assertTrue(all(expected for _prompt, expected in FACTS))
