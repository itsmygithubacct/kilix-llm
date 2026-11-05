"""Hook and weight edits on tiny torch modules. Skipped when torch is absent."""

from __future__ import annotations

import unittest

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised only when torch is installed
    torch = None
    nn = None

from kilix_llm_lab import LabError
from kilix_llm_lab.edit import (
    intervene_neuron,
    project_linear_,
    push_neuron,
    push_neurons,
    steer,
    steer_layers,
)


def _layer(hidden: int, intermediate: int):
    layer = nn.Module()
    layer.self_attn = nn.Module()
    layer.self_attn.o_proj = nn.Linear(hidden, hidden, bias=True)
    layer.mlp = nn.Module()
    layer.mlp.down_proj = nn.Linear(intermediate, hidden, bias=False)
    return layer


@unittest.skipUnless(torch is not None, "torch is not installed")
class EditTests(unittest.TestCase):
    def test_hook_adds_the_direction_and_then_removes_itself(self) -> None:
        layer = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            layer.weight.copy_(torch.eye(2))
        model = type("M", (), {})()
        model.layers = [layer]
        incoming = torch.tensor([[1.0, 0.0]])
        with steer(model, 0, [0.0, 5.0], strength=2.0):
            during = layer(incoming)
        after = layer(incoming)
        self.assertTrue(torch.allclose(during, torch.tensor([[1.0, 10.0]])))
        self.assertTrue(torch.allclose(after, torch.tensor([[1.0, 0.0]])))

    def test_hook_adds_on_two_layers_and_then_removes_itself(self) -> None:
        layers = []
        for _ in range(2):
            layer = nn.Linear(2, 2, bias=False)
            with torch.no_grad():
                layer.weight.copy_(torch.eye(2))
            layers.append(layer)
        model = type("M", (), {})()
        model.layers = layers
        incoming = torch.tensor([[1.0, 0.0]])
        with steer_layers(model, [0, 1], [0.0, 3.0], strength=-1.0):
            during = [layer(incoming) for layer in layers]
        after = [layer(incoming) for layer in layers]
        for tensor in during:
            self.assertTrue(torch.allclose(tensor, torch.tensor([[1.0, -3.0]])))
        for tensor in after:
            self.assertTrue(torch.allclose(tensor, torch.tensor([[1.0, 0.0]])))
        self.assertEqual(len(layers[0]._forward_hooks), 0)
        self.assertEqual(len(layers[1]._forward_hooks), 0)
        with self.assertRaises(LabError):
            with steer_layers(model, [0, 9], [0.0, 1.0], strength=1.0):
                pass
        self.assertEqual(len(layers[0]._forward_hooks), 0)

    def test_generated_only_skips_the_prompt_and_adds_on_a_decode_step(self) -> None:
        layer = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            layer.weight.copy_(torch.eye(2))
        model = type("M", (), {})()
        model.layers = [layer]
        prompt = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        decode = torch.tensor([[1.0, 0.0]])
        with steer(model, 0, [0.0, 4.0]):
            always = layer(prompt)
        with steer(model, 0, [0.0, 4.0], generated_only=True):
            skipped = layer(prompt)
            stepped = layer(decode)
        self.assertTrue(torch.allclose(always, torch.tensor([[1.0, 4.0], [0.0, 5.0]])))
        self.assertTrue(torch.allclose(skipped, prompt))
        self.assertTrue(torch.allclose(stepped, torch.tensor([[1.0, 4.0]])))
        self.assertEqual(len(layer._forward_hooks), 0)

    def test_generated_only_neuron_push_skips_the_prompt(self) -> None:
        layer = _layer(2, 3)
        with torch.no_grad():
            layer.mlp.down_proj.weight.copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
        prompt = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
        decode = torch.tensor([[[1.0, 0.0, 0.0]]])
        with push_neurons(layer, [0.0, 5.0, 0.0], generated_only=True):
            skipped = layer.mlp.down_proj(prompt)
            stepped = layer.mlp.down_proj(decode)
        self.assertTrue(torch.allclose(skipped, torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])))
        self.assertTrue(torch.allclose(stepped, torch.tensor([[[1.0, 5.0]]])))
        self.assertEqual(len(layer.mlp.down_proj._forward_pre_hooks), 0)
        with self.assertRaises(LabError):
            with push_neurons(layer, [1.0, 0.0], generated_only=True):
                layer.mlp.down_proj(prompt)

    def test_neuron_push_changes_only_that_coordinate(self) -> None:
        layer = _layer(2, 3)
        with torch.no_grad():
            layer.mlp.down_proj.weight.copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
        # Identity-like down_proj: output equals the first two neurons.
        incoming = torch.zeros(1, 3)
        with push_neuron(layer, 1, 4.0):
            output = layer.mlp.down_proj(incoming)
        self.assertTrue(torch.allclose(output, torch.tensor([[0.0, 4.0]])))

    def test_activate_replaces_and_deactivate_zeroes(self) -> None:
        layer = _layer(2, 3)
        with torch.no_grad():
            layer.mlp.down_proj.weight.copy_(torch.eye(3)[:2])
        incoming = torch.tensor([[1.0, 2.0, 3.0]])
        with intervene_neuron(layer, 1, "activate", 9.0):
            activated = layer.mlp.down_proj(incoming)
        with intervene_neuron(layer, 0, "deactivate", 0.0):
            deactivated = layer.mlp.down_proj(incoming)
        self.assertTrue(torch.allclose(activated, torch.tensor([[1.0, 9.0]])))
        self.assertTrue(torch.allclose(deactivated, torch.tensor([[0.0, 2.0]])))

    def test_abliteration_clears_the_direction_from_both_writers(self) -> None:
        layer = _layer(3, 2)
        with torch.no_grad():
            layer.self_attn.o_proj.weight.copy_(torch.eye(3))
            layer.self_attn.o_proj.bias.copy_(torch.tensor([3.0, 4.0, 5.0]))
            layer.mlp.down_proj.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]))
        model = type("M", (), {})()
        model.layers = [layer]
        from kilix_llm_lab.edit import abliterate

        names = abliterate(model, [1.0, 0.0, 0.0], [0])
        self.assertEqual(
            names,
            ["layers.0.self_attn.o_proj", "layers.0.mlp.down_proj"],
        )
        with torch.no_grad():
            for linear in (layer.self_attn.o_proj, layer.mlp.down_proj):
                dots = torch.tensor([1.0, 0.0, 0.0]) @ linear.weight.detach().float()
                self.assertTrue(torch.allclose(dots, torch.zeros_like(dots), atol=1e-5))
            self.assertAlmostEqual(float(layer.self_attn.o_proj.bias[0]), 0.0, places=5)
            self.assertAlmostEqual(float(layer.self_attn.o_proj.bias[1]), 4.0, places=5)
