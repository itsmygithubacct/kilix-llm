"""Direction arithmetic used by every edit."""

from __future__ import annotations

import unittest

from kilix_llm_lab import LabError
from kilix_llm_lab.algebra import (
    column_dots,
    combine_counts,
    contrast,
    kl_divergence,
    l2,
    one_hot,
    project_out,
    project_out_vector,
    random_directions,
    rank_components,
    sparsify,
    strongest_index,
    subtract,
    unit,
)


class AlgebraTests(unittest.TestCase):
    def test_contrast_is_mean_difference(self) -> None:
        direction = contrast([[1.0, 0.0], [3.0, 2.0]], [[0.0, 0.0], [0.0, 2.0]])
        self.assertEqual(direction, [2.0, 0.0])

    def test_difference_of_means_sparsifies_and_ranks_the_strongest_layer(self) -> None:
        from kilix_llm_lab import difference_of_means
        from kilix_llm_lab.algebra import layer_contrasts, mean

        class_a = [[1.0, 0.0], [3.0, 2.0]]
        class_b = [[0.0, 0.0], [0.0, 2.0]]
        self.assertEqual(
            difference_of_means(class_a, class_b),
            subtract(mean(class_a), mean(class_b)),
        )
        per_layer = layer_contrasts(
            [[[0.0, 1.0]], [[5.0, 0.0]]],
            [[[0.0, 0.0]], [[1.0, 0.0]]],
        )
        self.assertEqual(strongest_index(per_layer), 1)
        self.assertEqual(sparsify(per_layer[1], 1), [4.0, 0.0])

    def test_combine_counts_weights_longer_batches(self) -> None:
        combined = combine_counts([([0.0, 0.0], 1.0), ([3.0, 9.0], 2.0)])
        self.assertEqual(combined, [2.0, 6.0])

    def test_unit_and_zero(self) -> None:
        self.assertEqual(unit([0.0, 3.0, 4.0]), [0.0, 0.6, 0.8])
        with self.assertRaises(LabError):
            unit([0.0, 0.0])

    def test_sparsify_keeps_largest_magnitude_and_breaks_ties_by_index(self) -> None:
        self.assertEqual(sparsify([1.0, -3.0, 2.0, -3.0], 2), [0.0, -3.0, 0.0, -3.0])

    def test_strongest_index_prefers_the_earlier_tie(self) -> None:
        self.assertEqual(strongest_index([[0.0, 1.0], [0.0, -1.0]]), 0)

    def test_projection_removes_the_direction_and_is_idempotent(self) -> None:
        weight = [
            [1.0, 0.0, 0.5],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        once = project_out(weight, [2.0, 0.0, 0.0])
        twice = project_out(once, [2.0, 0.0, 0.0])
        self.assertEqual(once, twice)
        dots = column_dots(once, unit([2.0, 0.0, 0.0]))
        self.assertTrue(all(abs(value) < 1e-9 for value in dots))
        # The first row was the only part along e0, so it is gone.
        self.assertEqual(once[0], [0.0, 0.0, 0.0])
        self.assertEqual(once[1], weight[1])

    def test_projection_leaves_an_orthogonal_bias(self) -> None:
        self.assertEqual(project_out_vector([0.0, 4.0], [1.0, 0.0]), [0.0, 4.0])
        self.assertEqual(project_out_vector([3.0, 4.0], [1.0, 0.0]), [0.0, 4.0])

    def test_kl_is_zero_for_equal_distributions(self) -> None:
        self.assertAlmostEqual(kl_divergence([0.5, 0.5], [0.5, 0.5]), 0.0)
        with self.assertRaises(LabError):
            kl_divergence([1.0, 0.0], [0.0, 1.0])

    def test_random_directions_match_the_requested_norm(self) -> None:
        first = random_directions(6, 2.5, 4, seed=7)
        again = random_directions(6, 2.5, 4, seed=7)
        self.assertEqual(first, again)
        for direction in first:
            self.assertAlmostEqual(l2(direction), 2.5)

    def test_rank_and_one_hot(self) -> None:
        self.assertEqual(rank_components([0.1, -4.0, 0.2], 1), [(1, -4.0)])
        self.assertEqual(one_hot(3, 1, -2.0), [0.0, -2.0, 0.0])
        with self.assertRaises(LabError):
            subtract([1.0], [1.0, 2.0])
