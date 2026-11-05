"""Train a small ReLU sparse autoencoder on rows of activations.

The trainer is plain Python so a stand-in batch does not need torch or a
checkpoint. Encoder and decoder are untied. The loss is mean squared error
plus an L1 penalty on the ReLU latents.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

from kilix_llm_lab import LabError


def train_sae(
    batch: Sequence[Sequence[float]],
    n_features: int,
    steps: int,
    *,
    lr: float = 0.05,
    l1: float = 0.01,
    seed: int = 0,
) -> dict:
    """Fit one SAE. Returns the weights and the loss before and after the steps."""
    rows = [[float(item) for item in row] for row in batch]
    if not rows:
        raise LabError("SAE training needs at least one activation row")
    width = len(rows[0])
    if width < 1 or any(len(row) != width for row in rows):
        raise LabError("SAE rows must share one positive width")
    if n_features < 1 or steps < 1:
        raise LabError("SAE features and steps must be positive")
    if lr <= 0:
        raise LabError("SAE learning rate must be positive")
    rng = random.Random(seed)
    scale = 1.0 / math.sqrt(width)
    encoder = [[rng.uniform(-scale, scale) for _ in range(width)] for _ in range(n_features)]
    decoder = [[rng.uniform(-scale, scale) for _ in range(width)] for _ in range(n_features)]
    enc_bias = [0.0] * n_features
    dec_bias = [0.0] * width
    start = _loss(rows, encoder, decoder, enc_bias, dec_bias, l1)
    count = float(len(rows))
    for _step in range(steps):
        grad_encoder = [[0.0] * width for _ in range(n_features)]
        grad_decoder = [[0.0] * width for _ in range(n_features)]
        grad_enc_bias = [0.0] * n_features
        grad_dec_bias = [0.0] * width
        for row in rows:
            pre = [
                enc_bias[feature] + _dot(row, encoder[feature])
                for feature in range(n_features)
            ]
            latent = [value if value > 0.0 else 0.0 for value in pre]
            recon = dec_bias[:]
            for feature, active in enumerate(latent):
                if active == 0.0:
                    continue
                for dim in range(width):
                    recon[dim] += active * decoder[feature][dim]
            error = [2.0 / width * (recon[dim] - row[dim]) / count for dim in range(width)]
            for dim in range(width):
                grad_dec_bias[dim] += error[dim]
            for feature, active in enumerate(latent):
                for dim in range(width):
                    grad_decoder[feature][dim] += error[dim] * active
                if pre[feature] <= 0.0:
                    continue
                through = _dot(error, decoder[feature]) + (l1 / count) / n_features
                grad_enc_bias[feature] += through
                for dim in range(width):
                    grad_encoder[feature][dim] += through * row[dim]
        for feature in range(n_features):
            enc_bias[feature] -= lr * grad_enc_bias[feature]
            for dim in range(width):
                encoder[feature][dim] -= lr * grad_encoder[feature][dim]
                decoder[feature][dim] -= lr * grad_decoder[feature][dim]
        for dim in range(width):
            dec_bias[dim] -= lr * grad_dec_bias[dim]
    end = _loss(rows, encoder, decoder, enc_bias, dec_bias, l1)
    return {
        "width": width,
        "n_features": n_features,
        "encoder": encoder,
        "decoder": decoder,
        "enc_bias": enc_bias,
        "dec_bias": dec_bias,
        "loss_start": start,
        "loss_end": end,
    }


def encode_row(model: dict, row: Sequence[float]) -> list[float]:
    """ReLU latent for one activation row."""
    values = [float(item) for item in row]
    if len(values) != model["width"]:
        raise LabError(f"row width {len(values)} != SAE width {model['width']}")
    latent = []
    for feature, bias in enumerate(model["enc_bias"]):
        pre = bias + _dot(values, model["encoder"][feature])
        latent.append(pre if pre > 0.0 else 0.0)
    return latent


def _loss(rows, encoder, decoder, enc_bias, dec_bias, l1: float) -> float:
    total = 0.0
    penalty = 0.0
    width = len(rows[0])
    features = len(encoder)
    for row in rows:
        latent = []
        for feature in range(features):
            pre = enc_bias[feature] + _dot(row, encoder[feature])
            latent.append(pre if pre > 0.0 else 0.0)
        recon = dec_bias[:]
        for feature, active in enumerate(latent):
            if active == 0.0:
                continue
            for dim in range(width):
                recon[dim] += active * decoder[feature][dim]
        total += sum((recon[dim] - row[dim]) ** 2 for dim in range(width)) / width
        penalty += sum(latent)
    count = float(len(rows))
    return total / count + l1 * (penalty / count) / features


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))
