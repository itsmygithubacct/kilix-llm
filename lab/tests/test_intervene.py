"""Neuron and direction operations, and the transcript catalog."""

from __future__ import annotations

import unittest

from kilix_llm_lab import LabError
from kilix_llm_lab.intervene import (
    CATALOG,
    QWEN3_5_4B,
    apply_direction,
    apply_scalar,
    surrounding,
)
from kilix_llm_lab.qwen import INTERMEDIATE


class InterveneTests(unittest.TestCase):
    def test_scalar_ops(self) -> None:
        self.assertEqual(apply_scalar(3.0, "increase", 2.0), 5.0)
        self.assertEqual(apply_scalar(3.0, "decrease", 2.0), 1.0)
        self.assertEqual(apply_scalar(3.0, "activate", 9.0), 9.0)
        self.assertEqual(apply_scalar(3.0, "deactivate", 9.0), 0.0)
        with self.assertRaises(LabError):
            apply_scalar(1.0, "mute", 1.0)

    def test_direction_deactivate_removes_only_that_axis(self) -> None:
        edited = apply_direction([3.0, 4.0], [1.0, 0.0], "deactivate", 0.0)
        self.assertEqual(edited, [0.0, 4.0])
        turned_on = apply_direction([3.0, 4.0], [0.0, 2.0], "activate", 7.0)
        self.assertEqual(turned_on, [3.0, 7.0])
        self.assertEqual(apply_direction([1.0, 0.0], [0.0, 1.0], "increase", 2.0), [1.0, 2.0])
        self.assertEqual(apply_direction([1.0, 5.0], [0.0, 1.0], "decrease", 2.0), [1.0, 3.0])

    def test_catalog_ids_fit_the_4b_mlp_and_not_the_0_6b_mlp(self) -> None:
        width_4b = 9216
        for name in ("evil", "tomato", "alien"):
            item = CATALOG[name]
            self.assertEqual(item.model_id, QWEN3_5_4B)
            self.assertLess(item.index, width_4b)
            self.assertIn(item.prompt, {"Tell me a story.", "Who are you?"})
        self.assertGreaterEqual(CATALOG["evil"].index, INTERMEDIATE)
        self.assertGreaterEqual(CATALOG["tomato"].index, INTERMEDIATE)
        self.assertLess(CATALOG["alien"].index, INTERMEDIATE)
        self.assertEqual(CATALOG["evil"].layer, 23)
        self.assertEqual(CATALOG["evil"].index, 8799)
        self.assertEqual(CATALOG["tomato"].layer, 8)
        self.assertEqual(CATALOG["tomato"].index, 4948)
        self.assertEqual(CATALOG["tomato-layer7"].layer, 7)
        self.assertEqual(CATALOG["tomato-layer7"].index, 4948)
        self.assertEqual(CATALOG["alien"].layer, 22)
        self.assertEqual(CATALOG["alien"].index, 2742)

    def test_surrounding_clips_at_the_ends(self) -> None:
        self.assertEqual(surrounding(1, 2, 5), [0, 1, 2, 3])
        self.assertEqual(surrounding(8799, 1, 9216), [8798, 8799, 8800])
        with self.assertRaises(LabError):
            surrounding(4948, 0, INTERMEDIATE)
