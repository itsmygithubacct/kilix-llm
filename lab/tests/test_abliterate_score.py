"""Abliteration scores are taken after P is written, not under a steering hook."""

from __future__ import annotations

import unittest

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised only when torch is installed
    torch = None
    nn = None

from kilix_llm_lab.algebra import kl_divergence
from kilix_llm_lab.runner import format_story, measure_abliteration


class RecordedAblationTests(unittest.TestCase):
    def test_scores_use_the_distribution_after_the_edit(self) -> None:
        events: list = []
        state = {"edited": False}

        def generate_fn(_model, _tokenizer, prompt, _count):
            events.append(("gen", state["edited"], prompt))
            if state["edited"]:
                return "empty"
            if prompt == "Tell me a story.":
                return "a happy smile"
            return "tokyo"

        def distribution_fn(_model, _tokenizer, _prompt):
            row = [0.2, 0.8] if state["edited"] else [0.9, 0.1]
            events.append(("dist", state["edited"], row))
            return row

        def apply_edit(_model, _direction, layers):
            events.append(("edit", list(layers)))
            state["edited"] = True
            return ["layers.0.self_attn.o_proj", "layers.0.mlp.down_proj"]

        result = measure_abliteration(
            object(),
            None,
            direction=[1.0, 0.0],
            layers=[0],
            prompt="Tell me a story.",
            max_new_tokens=4,
            fact_rows=(("The capital of Japan is", "tokyo"),),
            generate_fn=generate_fn,
            distribution_fn=distribution_fn,
            apply_edit=apply_edit,
        )
        self.assertEqual(
            [event[0] for event in events],
            ["gen", "dist", "gen", "edit", "gen", "dist", "gen"],
        )
        self.assertEqual([event[1] for event in events if event[0] != "edit"], [False, False, False, True, True, True])
        dists = [event for event in events if event[0] == "dist"]
        self.assertAlmostEqual(result["kl"], kl_divergence(dists[0][2], dists[1][2]))
        self.assertGreater(result["kl"], 0.0)
        self.assertEqual(result["facts_before"], [("The capital of Japan is", "tokyo", True)])
        self.assertEqual(result["facts_after"], [("The capital of Japan is", "empty", False)])
        self.assertGreater(result["baseline_mood"], 0.0)
        self.assertLess(result["abliterated_mood"], 0.0)


@unittest.skipUnless(torch is not None, "torch is not installed")
class StoryAblationTests(unittest.TestCase):
    def test_story_scores_kl_and_facts_after_projection_not_under_steer(self) -> None:
        import kilix_llm_lab.runner as runner

        layer = nn.Module()
        layer.self_attn = nn.Module()
        layer.self_attn.o_proj = nn.Linear(2, 2, bias=True)
        layer.mlp = nn.Module()
        layer.mlp.down_proj = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            layer.self_attn.o_proj.weight.copy_(torch.eye(2))
            layer.self_attn.o_proj.bias.copy_(torch.tensor([2.0, 0.0]))
            layer.mlp.down_proj.weight.copy_(torch.eye(2))
        model = nn.Module()
        model.layers = nn.ModuleList([layer])
        model.config = type("Config", (), {"name_or_path": "tiny"})()
        events: list = []

        def cleared() -> bool:
            return abs(float(layer.self_attn.o_proj.weight.detach()[0, 0])) < 1e-4

        def hook_count() -> int:
            return len(layer.self_attn.o_proj._forward_hooks) + len(layer._forward_hooks)

        def logits():
            with torch.no_grad():
                return layer.self_attn.o_proj(torch.zeros(1, 2))[0]

        def fake_generate(_model, _tokenizer, prompt, _count):
            self.assertEqual(hook_count(), 0)
            score = logits()
            events.append(("gen", cleared(), prompt))
            if float(score[0]) > float(score[1]):
                return "a happy smile" if prompt == "Tell me a story." else "tokyo"
            return "empty"

        def fake_distribution(_model, _tokenizer, _prompt):
            self.assertEqual(hook_count(), 0)
            probs = torch.softmax(logits(), dim=-1)
            row = [float(probs[0]), float(probs[1])]
            events.append(("dist", cleared(), row))
            return list(row)

        def fake_direction(_model, _tokenizer, _max_length, _keep):
            return {
                "layer": 0,
                "directions": [[1.0, 0.0]],
                "norms": [1.0],
                "sad_tokens": 1,
                "happy_tokens": 1,
            }

        originals = (runner.sadness_direction, runner.generate, runner.prompt_logprobs)
        runner.sadness_direction = fake_direction
        runner.generate = fake_generate
        runner.prompt_logprobs = fake_distribution
        try:
            result = runner.story(
                model,
                None,
                layer=0,
                strength=9.0,
                keep=None,
                max_new_tokens=4,
                max_length=8,
                do_abliterate=True,
                prompt="Tell me a story.",
            )
        finally:
            (
                runner.sadness_direction,
                runner.generate,
                runner.prompt_logprobs,
            ) = originals

        dists = [event for event in events if event[0] == "dist"]
        self.assertEqual([event[1] for event in dists], [False, True])
        self.assertAlmostEqual(result["kl"], kl_divergence(dists[0][2], dists[1][2]))
        self.assertGreater(result["kl"], 0.0)
        japan_before = next(row for row in result["facts_before"] if "Japan" in row[0])
        japan_after = next(row for row in result["facts_after"] if "Japan" in row[0])
        self.assertEqual(japan_before[1], "tokyo")
        self.assertTrue(japan_before[2])
        self.assertEqual(japan_after[1], "empty")
        self.assertFalse(japan_after[2])
        self.assertGreater(result["baseline_mood"], 0.0)
        self.assertLess(result["abliterated_mood"], 0.0)
        self.assertEqual(
            result["abliterated_modules"],
            ["layers.0.self_attn.o_proj", "layers.0.mlp.down_proj"],
        )
        with torch.no_grad():
            for linear in (layer.self_attn.o_proj, layer.mlp.down_proj):
                dots = torch.tensor([1.0, 0.0]) @ linear.weight.detach().float()
                self.assertTrue(torch.allclose(dots, torch.zeros_like(dots), atol=1e-5))
        text = format_story(result)
        self.assertIn("kl:", text)
        self.assertIn("facts_after:", text)
        self.assertIn("abliterated:", text)
        self.assertNotIn("steered:", text)
