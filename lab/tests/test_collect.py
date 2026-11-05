"""Residual reads use each decoder layer's hook, not the final norm."""

from __future__ import annotations

import unittest

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised only when torch is installed
    torch = None
    nn = None

from kilix_llm_lab.collect import _masked_mean, encode, paired_residual_means, residual_means


class _Tokenizer:
    """Just enough of a tokenizer for `encode`."""

    pad_token_id = 0
    eos_token = 0

    def __call__(self, texts, return_tensors="pt", padding=True, truncation=True, max_length=256):
        del texts, return_tensors, padding, truncation, max_length
        return {
            "input_ids": torch.tensor([[1, 2, 0]]),
            "attention_mask": torch.tensor([[1, 1, 0]]),
        }


class _Layer(nn.Module if nn is not None else object):
    def __init__(self, bias):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(bias, dtype=torch.float32))

    def forward(self, hidden):
        return hidden + self.bias


class _NormedStack(nn.Module if nn is not None else object):
    """Two blocks, then a final scale written over hidden_states[-1]."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Layer([1.0, 0.0]), _Layer([0.0, 5.0])])
        self.final_scale = nn.Parameter(torch.tensor(7.0))

    def forward(self, input_ids=None, attention_mask=None, **_kwargs):
        hidden = torch.zeros(
            input_ids.shape[0], input_ids.shape[1], 2, device=input_ids.device
        )
        states = [hidden]
        for layer in self.layers:
            hidden = layer(hidden)
            states.append(hidden)
        # Same overwrite as transformers 5.22: the last reported state is the norm.
        states[-1] = hidden * self.final_scale
        return type("Out", (), {"hidden_states": tuple(states)})()


@unittest.skipUnless(torch is not None, "torch is not installed")
class ResidualMeanTests(unittest.TestCase):
    def test_last_residual_matches_the_layer_hook_not_the_final_norm(self) -> None:
        model = _NormedStack()
        tokenizer = _Tokenizer()
        seen = {}

        def grab(_module, _inputs, output):
            seen["hook"] = output.detach().clone()

        handle = model.layers[-1].register_forward_hook(grab)
        try:
            encoded = encode(tokenizer, ["hello"], max_length=8)
            with torch.no_grad():
                output = model(**encoded, use_cache=False)
        finally:
            handle.remove()
        hook_mean, _count = _masked_mean(seen["hook"], encoded["attention_mask"])
        norm_mean, _count = _masked_mean(output.hidden_states[-1], encoded["attention_mask"])
        self.assertFalse(torch.allclose(hook_mean, norm_mean))

        means, count = residual_means(model, tokenizer, ["hello"], max_length=8)
        self.assertEqual(len(means), len(model.layers))
        self.assertGreater(count, 0)
        self.assertTrue(torch.allclose(torch.tensor(means[-1]), hook_mean))
        self.assertFalse(torch.allclose(torch.tensor(means[-1]), norm_mean))
        for layer in model.layers:
            self.assertEqual(len(layer._forward_hooks), 0)


class _PairTokenizer:
    pad_token_id = 0
    eos_token = 0

    def __call__(self, text, truncation=True, max_length=256, add_special_tokens=True, **_kwargs):
        del truncation, add_special_tokens
        table = {"left": [1, 2, 9], "right": [1, 2, 4]}
        return {"input_ids": table[text][:max_length]}


class _IdStack(nn.Module if nn is not None else object):
    """One block whose residual is (token id, 1)."""

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.layers = nn.ModuleList([nn.Identity()])

    def forward(self, input_ids=None, use_cache=False, **_kwargs):
        del use_cache
        hidden = torch.stack(
            [input_ids.float(), torch.ones_like(input_ids, dtype=torch.float32)],
            dim=-1,
        )
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


@unittest.skipUnless(torch is not None, "torch is not installed")
class PairedResidualTests(unittest.TestCase):
    def test_shared_prefix_is_left_out_of_the_mean(self) -> None:
        direction = paired_residual_means(
            _IdStack(),
            _PairTokenizer(),
            [("left", "right")],
        )
        self.assertEqual(len(direction), 1)
        self.assertAlmostEqual(direction[0][0], 5.0)
        self.assertAlmostEqual(direction[0][1], 0.0)
