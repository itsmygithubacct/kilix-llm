"""Forward passes that read residual streams and MLP neurons.

Padding tokens are excluded. Each residual is the decoder layer's forward-hook
output, the same tensor steering adds to. The returned hidden-state tuple is
not used: current transformers overwrites the last entry with the final
RMSNorm, which a hook on that layer does not see.
"""

from __future__ import annotations

from collections.abc import Sequence

from kilix_llm_lab import LabError
from kilix_llm_lab.algebra import first_divergence, mean, subtract
from kilix_llm_lab.qwen import decoder_layers, neuron_linear


def _torch():
    try:
        import torch
    except ImportError as error:
        raise LabError("torch is not installed") from error
    return torch


def encode(tokenizer, texts: Sequence[str], max_length: int):
    if not texts:
        raise LabError("no texts to encode")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = tokenizer(
        list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    if int(encoded["attention_mask"].sum()) <= 0:
        raise LabError("encoding has no unmasked tokens")
    return encoded


def _masked_mean(hidden, attention_mask):
    torch = _torch()
    mask = attention_mask.to(device=hidden.device)
    weights = mask.unsqueeze(-1).to(dtype=hidden.dtype)
    count = float(mask.sum().item())
    total = (hidden * weights).sum(dim=(0, 1))
    mean = (total / count).detach().float().cpu()
    return mean, count


def _layer_hidden(output):
    """Tensor a forward hook on a decoder layer sees. Tuples keep the first item."""
    hidden = output[0] if isinstance(output, tuple) else output
    if not hasattr(hidden, "detach"):
        raise LabError(f"layer output is {type(hidden).__name__}, not a tensor")
    return hidden.detach()


def residual_means(model, tokenizer, texts: Sequence[str], max_length: int = 256):
    """Return (per-layer mean, token count). One mean per decoder layer.

    The mean is taken on each layer's forward-hook output. That stays the
    block residual when the caller replaces the last hidden state with the
    final norm.
    """
    torch = _torch()
    layers = decoder_layers(model)
    captured: list[list] = [[] for _layer in layers]
    handles = []

    def make_hook(bucket):
        def hook(_module, _inputs, output):
            bucket.append(_layer_hidden(output))

        return hook

    try:
        for layer, bucket in zip(layers, captured):
            handles.append(layer.register_forward_hook(make_hook(bucket)))
        encoded = encode(tokenizer, texts, max_length)
        device = model_device(model)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            model(**encoded, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    for index, bucket in enumerate(captured):
        if len(bucket) != 1:
            raise LabError(f"layer {index} hook ran {len(bucket)} times, expected 1")
    means = []
    count = None
    for hidden in (bucket[0] for bucket in captured):
        mean, count = _masked_mean(hidden, encoded["attention_mask"])
        means.append(mean.tolist())
    return means, count


def _token_ids(tokenizer, text: str, max_length: int) -> list[int]:
    if not isinstance(text, str) or not text:
        raise LabError("paired text is empty")
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    ids = encoded["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    tokens = [int(token) for token in ids][:max_length]
    if not tokens:
        raise LabError("tokenization produced no tokens")
    return tokens


def _tail_means(model, token_ids: Sequence[int], start: int) -> list[list[float]]:
    """One tail mean per decoder layer for a single unpadded sequence."""
    torch = _torch()
    layers = decoder_layers(model)
    buckets: list[list] = [[] for _layer in layers]
    handles = []

    def make_hook(bucket):
        def hook(_module, _inputs, output):
            bucket.append(_layer_hidden(output))

        return hook

    try:
        for layer, bucket in zip(layers, buckets):
            handles.append(layer.register_forward_hook(make_hook(bucket)))
        device = model_device(model)
        ids = torch.tensor([list(token_ids)], device=device)
        with torch.no_grad():
            model(input_ids=ids, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    means = []
    for index, bucket in enumerate(buckets):
        if len(bucket) != 1:
            raise LabError(f"layer {index} hook ran {len(bucket)} times, expected 1")
        hidden = bucket[0]
        if hidden.ndim == 3:
            hidden = hidden[0]
        if hidden.ndim != 2:
            raise LabError(f"layer {index} residual rank is {hidden.ndim}, expected 2")
        if start >= int(hidden.shape[0]):
            raise LabError(
                f"divergence index {start} is past {int(hidden.shape[0])} tokens"
            )
        tail = hidden[start:].detach().float().mean(dim=0).cpu().tolist()
        means.append([float(value) for value in tail])
    return means


def paired_residual_means(
    model,
    tokenizer,
    pairs: Sequence[tuple[str, str]],
    max_length: int = 256,
):
    """Per-layer mean(class_a tails) - mean(class_b tails).

    Each pair is two strings. Each string is tokenized on its own, and the
    tail starts where the token ids diverge. residual_means still averages
    every unmasked token and is the default contrast.
    """
    if not pairs:
        raise LabError("paired means need at least one pair")
    layers = decoder_layers(model)
    per_layer_a: list[list[list[float]]] = [[] for _layer in layers]
    per_layer_b: list[list[list[float]]] = [[] for _layer in layers]
    for left, right in pairs:
        ids_a = _token_ids(tokenizer, left, max_length)
        ids_b = _token_ids(tokenizer, right, max_length)
        start = first_divergence(ids_a, ids_b)
        for bucket, ids in ((per_layer_a, ids_a), (per_layer_b, ids_b)):
            for layer, vector in enumerate(_tail_means(model, ids, start)):
                bucket[layer].append(vector)
    return [
        subtract(mean(left), mean(right))
        for left, right in zip(per_layer_a, per_layer_b)
    ]


def neuron_means(model, tokenizer, texts: Sequence[str], layer_index: int, max_length: int = 256):
    """Mean MLP neuron vector at one layer. Shape is intermediate_size."""
    torch = _torch()
    layers = decoder_layers(model)
    if not 0 <= layer_index < len(layers):
        raise LabError(f"layer {layer_index} is outside 0..{len(layers) - 1}")
    captured: list = []

    def pre_hook(_module, args):
        captured.append(args[0].detach())
        return None

    handle = neuron_linear(layers[layer_index]).register_forward_pre_hook(pre_hook)
    try:
        encoded = encode(tokenizer, texts, max_length)
        device = model_device(model)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            model(**encoded, use_cache=False)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise LabError(f"neuron hook ran {len(captured)} times, expected 1")
    mean, count = _masked_mean(captured[0], encoded["attention_mask"])
    return mean.tolist(), count


def prompt_logprobs(model, tokenizer, prompt: str):
    """Log-softmax and attention mask for one prompt, before any edit.

    Hold the returned pair, change the weights, then call `kl_from_logprobs`
    with a second pair from the edited model. This is KL(original || edited)
    over unmasked prompt positions.
    """
    torch = _torch()
    import torch.nn.functional as F

    encoded = encode(tokenizer, [prompt], max_length=512)
    device = model_device(model)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        logits = model(**encoded, use_cache=False).logits
    return F.log_softmax(logits.float(), dim=-1), encoded["attention_mask"]


def kl_from_logprobs(original, edited) -> float:
    """Mean KL(original || edited) over the shared unmasked prompt positions."""
    log_p, mask = original
    log_q, mask_q = edited
    length = min(log_p.shape[1], log_q.shape[1], mask.shape[1], mask_q.shape[1])
    log_p = log_p[:, :length]
    log_q = log_q[:, :length]
    used = mask[:, :length].float()
    kl = (log_p.exp() * (log_p - log_q)).sum(dim=-1)
    return float((kl * used).sum().item() / used.sum().clamp(min=1).item())


def prompt_kl(model, tokenizer, prompt: str, steered) -> float:
    """Mean KL(original || steered) over unmasked prompt positions.

    `steered` is a zero-argument context manager, for example
    `lambda: steer(model, layer, direction, strength)`. The original
    distribution is captured before the context manager runs.
    """
    original = prompt_logprobs(model, tokenizer, prompt)
    with steered():
        edited = prompt_logprobs(model, tokenizer, prompt)
    return kl_from_logprobs(original, edited)


def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    torch = _torch()
    if max_new_tokens < 1:
        raise LabError("max_new_tokens must be positive")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = tokenizer(prompt, return_tensors="pt")
    device = model_device(model)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    prompt_len = int(encoded["input_ids"].shape[1])
    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    continuation = output[0, prompt_len:]
    return tokenizer.decode(continuation, skip_special_tokens=True)


def model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration as error:
        raise LabError("model has no parameters") from error
