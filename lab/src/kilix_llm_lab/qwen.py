"""Where Qwen3 writes the residual stream, and the 0.6B shape this project expects."""

from __future__ import annotations

from kilix_llm_lab import LabError

# Hugging Face config of Qwen/Qwen3-0.6B. The runner checks a loaded config
# against these numbers unless --allow-other is set.
QWEN3_0_6B = "Qwen/Qwen3-0.6B"
QWEN3_0_6B_BASE = "Qwen/Qwen3-0.6B-Base"
LAYERS = 28
HIDDEN = 1024
INTERMEDIATE = 3072
# Qwen3.5-4B text MLP width. The neuron sweep covers every index in this range.
INTERMEDIATE_4B = 9216
MODEL_TYPE = "qwen3"


def root_module(model):
    """Text decoder: `language_model` on a multimodal checkpoint, else `layers`."""
    for carrier in (model, getattr(model, "model", None)):
        if carrier is None:
            continue
        language = getattr(carrier, "language_model", None)
        if language is not None and getattr(language, "layers", None) is not None:
            return language
    inner = getattr(model, "model", None)
    if inner is not None and getattr(inner, "layers", None) is not None:
        return inner
    if getattr(model, "layers", None) is not None:
        return model
    raise LabError("model has no text decoder layers")


def decoder_layers(model) -> list:
    return list(root_module(model).layers)


def layer_kind(layer) -> str:
    """"full" for softmax attention, "linear" for a Qwen3.5 GatedDeltaNet layer."""
    if getattr(layer, "self_attn", None) is not None:
        return "full"
    if getattr(layer, "linear_attn", None) is not None:
        return "linear"
    raise LabError("layer has neither self_attn nor linear_attn")


def mixer_path(layer) -> str:
    """Dotted path of the token mixer's writing matrix inside one layer."""
    return "self_attn.o_proj" if layer_kind(layer) == "full" else "linear_attn.out_proj"


def mixer_linear(layer):
    """The token mixer's writing matrix: self_attn.o_proj, or linear_attn.out_proj on Qwen3.5.

    Its forward input is one slice per head: attention heads on a full layer,
    value heads on a linear layer.
    """
    try:
        if layer_kind(layer) == "full":
            return layer.self_attn.o_proj
        return layer.linear_attn.out_proj
    except AttributeError as error:
        raise LabError(f"layer is missing {mixer_path(layer)}") from error


def mixer_heads(config, layer) -> int:
    """Heads in the slice layout of mixer_linear(layer)'s input."""
    kind = layer_kind(layer)
    names = ("num_attention_heads",) if kind == "full" else ("linear_num_value_heads",)
    for obj in (config, getattr(config, "text_config", None)):
        if obj is None:
            continue
        for name in names:
            count = getattr(obj, name, None)
            if count:
                return int(count)
    raise LabError(f"config has no {names[0]} for a {kind} layer")


def writing_linears(layer) -> list:
    """Token-mixer output (o_proj or out_proj) and MLP down_proj. Both map into hidden_size."""
    try:
        return [mixer_linear(layer), layer.mlp.down_proj]
    except AttributeError as error:
        raise LabError(f"layer is missing {mixer_path(layer)} or mlp.down_proj") from error


def neuron_linear(layer):
    """down_proj. Its forward input is the MLP neuron vector, silu(gate) * up."""
    try:
        return layer.mlp.down_proj
    except AttributeError as error:
        raise LabError("layer is missing mlp.down_proj") from error


def assert_qwen3_0_6b(config) -> None:
    """Refuse a checkpoint whose shape is not the 0.6B project target."""
    got = {
        "model_type": getattr(config, "model_type", None),
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "hidden_size": getattr(config, "hidden_size", None),
        "intermediate_size": getattr(config, "intermediate_size", None),
    }
    expected = {
        "model_type": MODEL_TYPE,
        "num_hidden_layers": LAYERS,
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
    }
    mismatches = [
        f"{name} is {got[name]!r}, expected {expected[name]!r}"
        for name in expected
        if got[name] != expected[name]
    ]
    if mismatches:
        raise LabError("checkpoint is not Qwen3-0.6B: " + "; ".join(mismatches))
