"""Column-scoped weight edits, with no checkpoint and no torch required."""

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
from kilix_llm_lab.weight import (
    apply_weight_edit,
    assert_editable,
    edit_matrix,
    select_columns,
)


class WeightEditTests(unittest.TestCase):
    def test_remove_drops_the_direction_from_one_column(self) -> None:
        rows = [[3.0, 1.0], [4.0, 0.0]]
        edited, report = edit_matrix(rows, "remove", direction=[1.0, 0.0], columns=[0])
        self.assertEqual(edited[0], [0.0, 1.0])
        self.assertEqual(edited[1], [4.0, 0.0])
        self.assertEqual(report["changed_entries"], 1)
        self.assertEqual(report["changed_columns"], 1)
        self.assertEqual(report["targets"], [0])
        self.assertAlmostEqual(report["relative_change"], 3.0 / math.sqrt(26.0))

    def test_scale_one_and_a_half_matches_the_reported_formula(self) -> None:
        rows = [[3.0, 1.0], [4.0, 0.0]]
        edited, _report = edit_matrix(rows, "remove", direction=[1.0, 0.0], columns=[0], scale=1.5)
        self.assertEqual(edited[0][0], -1.5)
        self.assertEqual(edited[1][0], 4.0)
        self.assertEqual(edited[0][1], 1.0)

    def test_omitted_columns_edit_the_whole_matrix_but_not_by_a_constant(self) -> None:
        rows = [[3.0, 1.0], [0.0, 0.0]]
        edited, report = edit_matrix(rows, "remove", direction=[1.0, 0.0])
        self.assertEqual(edited, [[0.0, 0.0], [0.0, 0.0]])
        self.assertEqual(report["targets"], [0, 1])
        self.assertEqual(len(report["targets"]), 2)

    def test_top_picks_the_stronger_writer_and_breaks_ties_by_index(self) -> None:
        rows = [[3.0, 1.0], [0.0, 0.0]]
        self.assertEqual(select_columns(rows, [1.0, 0.0], 1), [0])
        tied = [[1.0, 1.0], [0.0, 0.0]]
        self.assertEqual(select_columns(tied, [1.0, 1.0], 1), [0])

    def test_scale_and_zero_refuse_a_whole_tensor_factor(self) -> None:
        rows = [[1.0, 2.0], [3.0, 4.0]]
        edited, report = edit_matrix(rows, "scale", columns=[1], factor=2.0)
        self.assertEqual(edited[0], [1.0, 4.0])
        self.assertEqual(report["factor"], 2.0)
        zeroed, zero_report = edit_matrix(rows, "zero", columns=[1])
        self.assertEqual(zeroed[0][1], 0.0)
        self.assertEqual(zeroed[1][1], 0.0)
        self.assertEqual(zero_report["factor"], 0.0)
        with self.assertRaises(LabError):
            edit_matrix(rows, "scale", columns=[0, 1], factor=2.0)
        with self.assertRaises(LabError):
            edit_matrix(rows, "zero")

    def test_refuses_embeddings_heads_norms_and_the_zero_vector(self) -> None:
        self.assertEqual(assert_editable("layers.15.mlp.down_proj"), "down_proj")
        self.assertEqual(assert_editable("layers.15.self_attn.o_proj"), "o_proj")
        for name in ("embed_tokens", "lm_head", "model.norm", "layers.0.mlp.up_proj"):
            with self.assertRaises(LabError):
                assert_editable(name)
        with self.assertRaises(LabError):
            edit_matrix([[1.0], [0.0]], "remove", direction=[0.0, 0.0], columns=[0])

    def test_command_prints_a_manifest_without_loading_a_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.json"
            path.write_text(json.dumps({"rows": [[3.0, 1.0], [4.0, 0.0]]}), encoding="utf-8")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = main(
                    [
                        "weight",
                        "--matrix",
                        str(path),
                        "--op",
                        "remove",
                        "--direction",
                        "1,0",
                        "--columns",
                        "0",
                    ]
                )
        self.assertEqual(code, 0)
        report = json.loads(stdout.getvalue())
        self.assertEqual(report["op"], "remove")
        self.assertEqual(report["changed_entries"], 1)
        self.assertIsNone(report["module"])

    def test_four_bit_is_refused_before_a_checkpoint_load(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = main(["weight", "--layer", "15", "--op", "zero", "--columns", "0", "--quant", "4bit"])
        self.assertEqual(code, 1)
        self.assertIn("dense floating-point", stderr.getvalue())


class LoadedWeightTests(unittest.TestCase):
    def test_down_proj_column_changes_and_the_bias_stays(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")

        class Attention:
            def __init__(self) -> None:
                self.o_proj = torch.nn.Linear(2, 2, bias=False)

        class Mlp:
            def __init__(self) -> None:
                self.down_proj = torch.nn.Linear(2, 2, bias=True)

        class Layer:
            def __init__(self) -> None:
                self.self_attn = Attention()
                self.mlp = Mlp()

        class Model:
            def __init__(self) -> None:
                self.layers = [Layer()]

        model = Model()
        with torch.no_grad():
            model.layers[0].mlp.down_proj.weight.copy_(torch.tensor([[3.0, 1.0], [4.0, 0.0]]))
            model.layers[0].mlp.down_proj.bias.copy_(torch.tensor([7.0, 8.0]))
        report = apply_weight_edit(
            model,
            0,
            "down_proj",
            "remove",
            direction=[1.0, 0.0],
            columns=[0],
        )
        weight = model.layers[0].mlp.down_proj.weight.detach()
        self.assertEqual(weight[0, 0].item(), 0.0)
        self.assertEqual(weight[0, 1].item(), 1.0)
        self.assertEqual(model.layers[0].mlp.down_proj.bias.detach().tolist(), [7.0, 8.0])
        self.assertEqual(report["module"], "layers.0.mlp.down_proj")
        self.assertEqual(report["changed_columns"], 1)
        self.assertIsNone(weight.grad)
