"""Forward-hook steering, neuron pushes, and abliteration of writing weights.

torch is imported inside the functions so the rest of the package imports
without it. Linear weights use the Hugging Face layout: y = x @ W.T + b, and
W is (out_features, in_features). Abliteration stores P @ W with
P = I - r r^T, which removes r from every column.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager

from kilix_llm_lab import LabError
from kilix_llm_lab.qwen import decoder_layers, neuron_linear, writing_linears


def _torch():
    try:
        import torch
    except ImportError as error:
        raise LabError("torch is not installed") from error
    return torch


def _vector(direction: Sequence[float], reference):
    torch = _torch()
    values = [float(value) for value in direction]
    if len(values) != reference.shape[-1]:
        raise LabError(
            f"direction width {len(values)} != hidden width {reference.shape[-1]}"
        )
    return torch.tensor(values, device=reference.device, dtype=reference.dtype)


def _replace_hidden(output, updated):
    if isinstance(output, tuple):
        return (updated,) + output[1:]
    return updated


def _hidden_of(output):
    if isinstance(output, tuple):
        hidden = output[0]
    else:
        hidden = output
    if not hasattr(hidden, "shape"):
        raise LabError(f"layer output is {type(hidden).__name__}, not a tensor")
    return hidden


def _decode_step(tensor) -> bool:
    """True when the sequence axis has one position.

    Under a KV cache that is a decode step. A prompt of several tokens has
    a longer axis and is left alone when the caller asked for that.
    """
    if tensor.ndim < 2:
        return True
    return int(tensor.shape[-2]) == 1


@contextmanager
def steer_layers(
    model,
    layer_indexes: Sequence[int],
    direction: Sequence[float],
    strength: float = 1.0,
    *,
    generated_only: bool = False,
):
    """Add strength * direction on the tokens leaving each chosen layer.

    The default adds it on every token, including the prompt. generated_only
    adds it only on a decode step. One hook per layer. Every hook is removed
    on exit, including when the body raises. The checkpoint is untouched.
    """
    layers = decoder_layers(model)
    chosen = [int(index) for index in layer_indexes]
    if not chosen:
        raise LabError("steer needs at least one layer")
    if len(set(chosen)) != len(chosen):
        raise LabError("steer layers must be unique")
    handles = []

    def make_hook():
        def hook(_module, _inputs, output):
            hidden = _hidden_of(output)
            vector = _vector(direction, hidden) * float(strength)
            if generated_only and not _decode_step(hidden):
                return output
            view = (1,) * (hidden.ndim - 1) + (hidden.shape[-1],)
            return _replace_hidden(output, hidden + vector.view(view))

        return hook

    try:
        for index in chosen:
            if not 0 <= index < len(layers):
                raise LabError(f"layer {index} is outside 0..{len(layers) - 1}")
            handles.append(layers[index].register_forward_hook(make_hook()))
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def steer(
    model,
    layer_index: int,
    direction: Sequence[float],
    strength: float = 1.0,
    *,
    generated_only: bool = False,
):
    """Add strength * direction on one decoder layer. See steer_layers."""
    with steer_layers(
        model,
        [layer_index],
        direction,
        strength,
        generated_only=generated_only,
    ):
        yield


@contextmanager
def intervene_direction(
    model,
    layer_index: int,
    direction: Sequence[float],
    op: str,
    amount: float,
    *,
    generated_only: bool = False,
):
    """increase, decrease, activate, or deactivate one residual direction.

    The direction is normalized. increase and decrease add ±amount along it.
    activate sets that component to amount. deactivate removes it. Weights
    are unchanged. Abliteration is the same removal baked into o_proj and down_proj.
    """
    layers = decoder_layers(model)
    if not 0 <= layer_index < len(layers):
        raise LabError(f"layer {layer_index} is outside 0..{len(layers) - 1}")
    if op not in {"increase", "decrease", "activate", "deactivate"}:
        raise LabError("operation must be one of increase, decrease, activate, deactivate")
    handle = None

    def hook(_module, _inputs, output):
        torch = _torch()
        hidden = _hidden_of(output)
        if generated_only and not _decode_step(hidden):
            return output
        axis = _vector(direction, hidden)
        axis = axis / torch.linalg.vector_norm(axis.float()).clamp(min=1e-12).to(axis.dtype)
        projection = (hidden.float() * axis.float()).sum(dim=-1, keepdim=True)
        if op == "increase":
            delta = float(amount)
        elif op == "decrease":
            delta = -float(amount)
        elif op == "activate":
            delta = float(amount) - projection
        else:
            delta = -projection
        view = (1,) * (hidden.ndim - 1) + (hidden.shape[-1],)
        updated = hidden.float() + delta * axis.float().view(view)
        return _replace_hidden(output, updated.to(dtype=hidden.dtype))

    try:
        handle = layers[layer_index].register_forward_hook(hook)
        yield
    finally:
        if handle is not None:
            handle.remove()


@contextmanager
def intervene_neuron(
    layer,
    index: int,
    op: str,
    amount: float = 0.0,
    *,
    generated_only: bool = False,
):
    """increase, decrease, activate, or deactivate one MLP neuron.

    The neuron vector is the input of down_proj: silu(gate(x)) * up(x).
    increase and decrease add ±amount. activate replaces the value with amount.
    deactivate sets it to zero for this forward pass. The default edits every
    token. generated_only edits a decode step. The weights stay as loaded.
    """
    linear = neuron_linear(layer)
    handle = None

    def pre_hook(_module, args):
        incoming = args[0]
        width = incoming.shape[-1]
        if not 0 <= index < width:
            raise LabError(f"neuron {index} is outside 0..{width - 1}")
        if generated_only and not _decode_step(incoming):
            return None
        updated = incoming.clone()
        current = updated[..., index]
        if op == "increase":
            updated[..., index] = current + float(amount)
        elif op == "decrease":
            updated[..., index] = current - float(amount)
        elif op == "activate":
            updated[..., index] = updated.new_full(current.shape, float(amount))
        elif op == "deactivate":
            updated[..., index] = current * 0
        else:
            raise LabError("operation must be one of increase, decrease, activate, deactivate")
        return (updated,) + tuple(args[1:])

    try:
        handle = linear.register_forward_pre_hook(pre_hook)
        yield
    finally:
        if handle is not None:
            handle.remove()


@contextmanager
def intervene_head(layer, head: int, n_heads: int, op: str, amount: float = 0.0):
    """Edit one attention head on the o_proj input, then remove the hook.

    The input is (..., n_heads * head_dim). increase and decrease add ±amount
    to every coordinate of that head. activate replaces them with amount.
    deactivate writes zeros. The o_proj weight is unchanged.
    """
    from kilix_llm_lab.intervene import _check_op

    _check_op(op)
    if n_heads < 1:
        raise LabError("a layer needs at least one attention head")
    if not 0 <= head < n_heads:
        raise LabError(f"head {head} is outside 0..{n_heads - 1}")
    linear = layer.self_attn.o_proj
    handle = None

    def pre_hook(_module, args):
        incoming = args[0]
        width = incoming.shape[-1]
        if width % n_heads != 0:
            raise LabError(f"width {width} does not divide into {n_heads} heads")
        head_dim = width // n_heads
        updated = incoming.clone()
        view = updated.reshape(-1, n_heads, head_dim)
        part = view[:, head, :]
        if op == "increase":
            view[:, head, :] = part + float(amount)
        elif op == "decrease":
            view[:, head, :] = part - float(amount)
        elif op == "activate":
            view[:, head, :] = part * 0 + float(amount)
        else:
            view[:, head, :] = part * 0
        return (updated,) + tuple(args[1:])

    try:
        handle = linear.register_forward_pre_hook(pre_hook)
        yield
    finally:
        if handle is not None:
            handle.remove()


@contextmanager
def push_neuron(layer, index: int, amount: float, *, generated_only: bool = False):
    """Add `amount` to one MLP neuron. This is intervene_neuron(..., "increase")."""
    with intervene_neuron(layer, index, "increase", amount, generated_only=generated_only):
        yield


@contextmanager
def push_neurons(layer, amounts: Sequence[float], *, generated_only: bool = False):
    """Add a full intermediate vector to MLP neurons.

    The default adds it on every token. generated_only adds it on a decode
    step, so the prompt is read unedited.
    """
    linear = neuron_linear(layer)
    handle = None

    def pre_hook(_module, args):
        incoming = args[0]
        vector = _vector(amounts, incoming)
        if generated_only and not _decode_step(incoming):
            return None
        view = (1,) * (incoming.ndim - 1) + (incoming.shape[-1],)
        updated = incoming + vector.view(view)
        return (updated,) + tuple(args[1:])

    try:
        handle = linear.register_forward_pre_hook(pre_hook)
        yield
    finally:
        if handle is not None:
            handle.remove()


def project_linear_(linear, direction: Sequence[float]) -> None:
    """Replace W with P @ W and b with P @ b, in place. r is normalized here."""
    torch = _torch()
    weight = linear.weight
    if weight.ndim != 2:
        raise LabError(f"expected a 2D weight, got {tuple(weight.shape)}")
    with torch.no_grad():
        basis = torch.tensor(
            [float(value) for value in direction],
            device=weight.device,
            dtype=torch.float32,
        )
        if basis.numel() != weight.shape[0]:
            raise LabError(
                f"direction width {basis.numel()} != out_features {weight.shape[0]}"
            )
        norm = torch.linalg.vector_norm(basis)
        if float(norm) == 0.0:
            raise LabError("cannot abliterate the zero vector")
        basis = basis / norm
        current = weight.detach().float()
        dots = basis @ current
        updated = current - basis[:, None] * dots[None, :]
        weight.copy_(updated.to(dtype=weight.dtype))
        if linear.bias is not None:
            bias = linear.bias.detach().float()
            bias = bias - basis * torch.dot(basis, bias)
            linear.bias.copy_(bias.to(dtype=linear.bias.dtype))


def abliterate(model, direction: Sequence[float], layers: Sequence[int]) -> list[str]:
    """Remove `direction` from o_proj and down_proj on the given decoder layers.

    Edits the loaded weights. Reload the checkpoint to undo it.
    """
    modules = decoder_layers(model)
    names = []
    chosen = list(layers)
    if not chosen:
        raise LabError("abliterate needs at least one layer")
    for index in chosen:
        if not 0 <= index < len(modules):
            raise LabError(f"layer {index} is outside 0..{len(modules) - 1}")
        for linear in writing_linears(modules[index]):
            project_linear_(linear, direction)
            names.append(f"layers.{index}.{linear_path(modules[index], linear)}")
    return names


def linear_path(layer, linear) -> str:
    if linear is layer.self_attn.o_proj:
        return "self_attn.o_proj"
    if linear is layer.mlp.down_proj:
        return "mlp.down_proj"
    return "weight"
