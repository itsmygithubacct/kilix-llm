"""Qwen3 module locations, checked on a stand-in object."""

from __future__ import annotations

import types
import unittest

from kilix_llm_lab import LabError
from kilix_llm_lab.qwen import (
    HIDDEN,
    INTERMEDIATE,
    LAYERS,
    MODEL_TYPE,
    assert_qwen3_0_6b,
    decoder_layers,
    neuron_linear,
    writing_linears,
)


def _layer():
    return types.SimpleNamespace(
        self_attn=types.SimpleNamespace(o_proj="o"),
        mlp=types.SimpleNamespace(down_proj="down"),
    )


class QwenMapTests(unittest.TestCase):
    def test_causal_lm_and_bare_transformer_both_resolve(self) -> None:
        layer = _layer()
        causal = types.SimpleNamespace(model=types.SimpleNamespace(layers=[layer]))
        bare = types.SimpleNamespace(layers=[layer])
        self.assertIs(decoder_layers(causal)[0], layer)
        self.assertEqual(writing_linears(decoder_layers(bare)[0]), ["o", "down"])
        self.assertEqual(neuron_linear(layer), "down")

    def test_shape_check_accepts_the_0_6b_config(self) -> None:
        config = types.SimpleNamespace(
            model_type=MODEL_TYPE,
            num_hidden_layers=LAYERS,
            hidden_size=HIDDEN,
            intermediate_size=INTERMEDIATE,
        )
        assert_qwen3_0_6b(config)

    def test_shape_check_refuses_a_wider_model(self) -> None:
        config = types.SimpleNamespace(
            model_type=MODEL_TYPE,
            num_hidden_layers=32,
            hidden_size=HIDDEN,
            intermediate_size=INTERMEDIATE,
        )
        with self.assertRaises(LabError):
            assert_qwen3_0_6b(config)

    def test_multimodal_wrapper_uses_the_text_stack(self) -> None:
        text = _layer()
        vision = _layer()
        model = types.SimpleNamespace(
            model=types.SimpleNamespace(
                layers=[vision],
                language_model=types.SimpleNamespace(layers=[text]),
            )
        )
        self.assertIs(decoder_layers(model)[0], text)

    def test_missing_layers_raise(self) -> None:
        with self.assertRaises(LabError):
            decoder_layers(types.SimpleNamespace())
