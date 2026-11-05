"""Head edits, SAE training, circuit patching, the refusal corpus, and the sweep."""

from __future__ import annotations

import json
import tempfile
import unittest

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised only when torch is installed
    torch = None
    nn = None

from kilix_llm_lab import LabError
from kilix_llm_lab.circuit import patch_effects
from kilix_llm_lab.cli import main
from kilix_llm_lab.heads import attention_heads, edit_head_slice
from kilix_llm_lab.qwen import INTERMEDIATE_4B
from kilix_llm_lab.refusal import builtin_corpus, load_corpus, refusal_directions
from kilix_llm_lab.sae import encode_row, train_sae
from kilix_llm_lab.sweep import shard_bounds, sweep_indexes


class MethodTests(unittest.TestCase):
    def test_head_slice_deactivate_clears_only_that_head(self) -> None:
        edited = edit_head_slice([1.0, 2.0, 3.0, 4.0], 1, 2, "deactivate", 0.0)
        self.assertEqual(edited, [1.0, 2.0, 0.0, 0.0])
        increased = edit_head_slice([1.0, 2.0, 3.0, 4.0], 0, 2, "increase", 10.0)
        self.assertEqual(increased, [11.0, 12.0, 3.0, 4.0])

    def test_attention_heads_reads_a_nested_text_config(self) -> None:
        config = type("C", (), {"text_config": type("T", (), {"num_attention_heads": 16})()})()
        self.assertEqual(attention_heads(config), 16)

    def test_sae_loss_falls_on_a_repeated_batch(self) -> None:
        batch = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        trained = train_sae(batch, 4, 30, lr=0.1, l1=0.0, seed=1)
        self.assertLess(trained["loss_end"], trained["loss_start"])
        latent = encode_row(trained, batch[0])
        self.assertEqual(len(latent), 4)
        self.assertGreater(max(latent), 0.0)

    def test_patch_effects_follow_the_clean_site(self) -> None:
        clean = [[1.0, 0.0], [4.0, 0.0]]
        corrupt = [[0.0, 0.0], [0.0, 0.0]]

        def run(states):
            return [sum(state[0] for state in states)]

        effects = patch_effects(clean, corrupt, run, lambda state: state[0])
        self.assertEqual(effects, [1.0, 4.0])

    def test_refusal_direction_is_refuse_minus_comply(self) -> None:
        refuse, comply = builtin_corpus()
        self.assertGreaterEqual(len(refuse), 8)
        self.assertGreaterEqual(len(comply), 8)
        self.assertTrue(set(refuse).isdisjoint(comply))
        directions = refusal_directions([[1.0, 0.0], [3.0, 1.0]], [[0.0, 0.0], [1.0, 1.0]])
        self.assertEqual(directions, [[1.0, 0.0], [2.0, 0.0]])
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json") as handle:
            json.dump({"refuse": ["I must refuse that request."], "comply": ["Here is a direct answer."]}, handle)
            handle.flush()
            loaded_refuse, loaded_comply = load_corpus(handle.name)
        self.assertEqual(loaded_refuse, ["I must refuse that request."])
        self.assertEqual(loaded_comply, ["Here is a direct answer."])

    def test_full_sweep_range_covers_9216_neurons(self) -> None:
        indexes = sweep_indexes(0, INTERMEDIATE_4B, INTERMEDIATE_4B)
        self.assertEqual(len(indexes), 9216)
        self.assertEqual(indexes[0], 0)
        self.assertEqual(indexes[-1], 9215)
        pieces = [shard_bounds(0, INTERMEDIATE_4B, 4, shard) for shard in range(4)]
        self.assertEqual(pieces, [(0, 2304), (2304, 4608), (4608, 6912), (6912, 9216)])
        uneven = [shard_bounds(0, 10, 3, shard) for shard in range(3)]
        self.assertEqual(uneven, [(0, 4), (4, 7), (7, 10)])
        with self.assertRaises(LabError):
            shard_bounds(0, 3, 4, 0)

    def test_commands_run_the_stand_ins(self) -> None:
        import contextlib
        import io

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = main(["sae", "--steps", "5", "--features", "3"])
        self.assertEqual(code, 0)
        sae = json.loads(buffer.getvalue())
        self.assertGreater(sae["loss_start"], sae["loss_end"])

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = main(["circuit"])
        self.assertEqual(code, 0)
        circuit = json.loads(buffer.getvalue())
        self.assertEqual(circuit["effects"], [1.0, 0.0])

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = main(["refusal", "--list"])
        self.assertEqual(code, 0)
        listed = json.loads(buffer.getvalue())
        self.assertGreaterEqual(listed["refuse"], 8)
        self.assertGreaterEqual(listed["comply"], 8)

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = main(
                [
                    "head",
                    "--stand-in",
                    "--heads",
                    "2",
                    "--head",
                    "0",
                    "--op",
                    "deactivate",
                ]
            )
        self.assertEqual(code, 0)
        head = json.loads(buffer.getvalue())
        self.assertEqual(head["after"], [0.0, 0.0, 3.0, 4.0])

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = main(
                [
                    "sweep",
                    "--stand-in",
                    "--start",
                    "0",
                    "--stop",
                    "2",
                    "--width",
                    "9216",
                    "--strength",
                    "1",
                    "--controls",
                    "1",
                ]
            )
        self.assertEqual(code, 0)
        sweep = json.loads(buffer.getvalue())
        self.assertEqual(sweep["indexes"], 2)
        self.assertEqual(sweep["width"], 9216)
        self.assertFalse(sweep["full_layer"])
        self.assertEqual(sweep["neuron_observations"], 2)

    def test_sweep_record_writes_each_generation(self) -> None:
        import contextlib
        import io
        import os

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "sweep.jsonl")
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                code = main(
                    [
                        "sweep",
                        "--stand-in",
                        "--start",
                        "0",
                        "--stop",
                        "2",
                        "--width",
                        "8",
                        "--strength",
                        "1",
                        "--controls",
                        "1",
                        "--record",
                        path,
                    ]
                )
            self.assertEqual(code, 0)
            summary = json.loads(buffer.getvalue())
            self.assertEqual(summary["indexes"], 2)
            with open(path, encoding="utf-8") as handle:
                lines = [json.loads(line) for line in handle if line.strip()]
        self.assertEqual(len(lines), 12)
        self.assertEqual(lines[0]["kind"], "neuron")
        self.assertEqual(lines[0]["index"], 0)
        self.assertNotIn("direction", lines[0])

    def test_one_shard_is_a_piece_of_the_range(self) -> None:
        import contextlib
        import io

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = main(
                [
                    "sweep",
                    "--stand-in",
                    "--start",
                    "0",
                    "--stop",
                    "8",
                    "--width",
                    "8",
                    "--shards",
                    "4",
                    "--shard",
                    "1",
                    "--strength",
                    "1",
                    "--controls",
                    "1",
                ]
            )
        self.assertEqual(code, 0)
        summary = json.loads(buffer.getvalue())
        self.assertEqual(summary["shard"], 1)
        self.assertEqual(summary["shards"], 4)
        self.assertEqual(summary["start"], 2)
        self.assertEqual(summary["stop"], 4)
        self.assertEqual(summary["indexes"], 2)
        self.assertFalse(summary["full_layer"])


@unittest.skipUnless(torch is not None, "torch is not installed")
class HeadHookTests(unittest.TestCase):
    def test_hook_deactivates_one_head_and_then_removes_itself(self) -> None:
        from kilix_llm_lab.edit import intervene_head

        layer = nn.Module()
        layer.self_attn = nn.Module()
        layer.self_attn.o_proj = nn.Linear(4, 2, bias=False)
        with torch.no_grad():
            layer.self_attn.o_proj.weight.copy_(
                torch.tensor([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
            )
        incoming = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        with intervene_head(layer, 1, 2, "deactivate", 0.0):
            during = layer.self_attn.o_proj(incoming)
        after = layer.self_attn.o_proj(incoming)
        self.assertTrue(torch.allclose(during, torch.tensor([[3.0, 0.0]])))
        self.assertTrue(torch.allclose(after, torch.tensor([[3.0, 7.0]])))
        self.assertEqual(len(layer.self_attn.o_proj._forward_pre_hooks), 0)
