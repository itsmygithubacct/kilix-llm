"""Load Qwen3-0.6B and run one edit. The project script is a thin wrapper."""

from __future__ import annotations

import json
from collections.abc import Sequence

from kilix_llm_lab import LabError
from kilix_llm_lab.algebra import (
    contrast,
    kl_divergence,
    l2,
    rank_components,
    sparsify,
    strongest_index,
    subtract,
)
from kilix_llm_lab.collect import (
    generate,
    kl_from_logprobs,
    neuron_means,
    paired_residual_means,
    prompt_kl,
    prompt_logprobs,
    residual_means,
)
from kilix_llm_lab.edit import abliterate, push_neuron, push_neurons, steer, steer_layers
from kilix_llm_lab.measure import fact_hit, mood_score
from kilix_llm_lab.qwen import QWEN3_0_6B, assert_qwen3_0_6b
from kilix_llm_lab.texts import FACTS, HAPPY, NEUTRAL_PROMPT, OTHER_OBJECTS, SAD, TOMATO


def load(
    model_id: str,
    device: str,
    allow_other: bool = False,
    dtype_name: str = "auto",
    quant: str = "none",
):
    """Load a causal LM, or Qwen3.5's multimodal class when that is the checkpoint.

    Qwen3.5 keeps the text stack on `language_model`. The 0.6B shape check
    still applies unless `allow_other` is set. `dtype_name` is auto, float32,
    or bfloat16. auto is float32 on CPU and bfloat16 on a GPU. `quant=4bit`
    loads NF4 weights onto a GPU so a 4B checkpoint fits in 8 GB. The
    checkpoint file is not rewritten.
    """
    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise LabError(
            "install torch and transformers before loading a checkpoint"
        ) from error
    if dtype_name == "auto":
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
    elif dtype_name == "float32":
        dtype = torch.float32
    elif dtype_name == "bfloat16":
        dtype = torch.bfloat16
    else:
        raise LabError("dtype must be auto, float32, or bfloat16")
    quant_config = None
    if quant == "4bit":
        if device == "cpu":
            raise LabError("4bit load targets a GPU; pass --device cuda")
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as error:
            raise LabError("install bitsandbytes and accelerate before a 4bit load") from error
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif quant != "none":
        raise LabError("quant must be none or 4bit")
    config = AutoConfig.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_type = getattr(config, "model_type", None)
    if model_type == "qwen3_5":
        try:
            from transformers import AutoModelForImageTextToText
        except ImportError as error:
            raise LabError("this transformers build cannot load Qwen3.5") from error
        model = _from_pretrained(
            AutoModelForImageTextToText, model_id, dtype, quant_config, device
        )
        if not allow_other:
            raise LabError("Qwen3.5 is not the 0.6B checkpoint; pass --allow-other")
    else:
        model = _from_pretrained(AutoModelForCausalLM, model_id, dtype, quant_config, device)
        if not allow_other:
            assert_qwen3_0_6b(model.config)
    if quant_config is None:
        model.to(device)
    model.eval()
    return model, tokenizer


def _from_pretrained(cls, model_id: str, dtype, quant_config, device: str):
    if quant_config is None:
        return cls.from_pretrained(model_id, dtype=dtype)
    return cls.from_pretrained(
        model_id,
        quantization_config=quant_config,
        dtype=dtype,
        device_map={"": device},
    )


def direction_from_pairs(model, tokenizer, pairs, max_length: int, keep: int | None):
    """class_a minus class_b, averaged from the first token where each pair diverges."""
    directions = paired_residual_means(model, tokenizer, pairs, max_length)
    layer = strongest_index(directions)
    vector = directions[layer]
    if keep is not None:
        vector = sparsify(vector, keep)
    return {
        "layer": layer,
        "direction": vector,
        "norms": [l2(row) for row in directions],
        "directions": directions,
        "paired": True,
    }


def sadness_direction(model, tokenizer, max_length: int, keep: int | None):
    """Sad minus happy, one residual vector per layer, plus the strongest layer."""
    sad, sad_count = residual_means(model, tokenizer, SAD, max_length=max_length)
    happy, happy_count = residual_means(model, tokenizer, HAPPY, max_length=max_length)
    if len(sad) != len(happy):
        raise LabError("sad and happy captures have different layer counts")
    directions = [subtract(left, right) for left, right in zip(sad, happy)]
    layer = strongest_index(directions)
    vector = directions[layer]
    if keep is not None:
        vector = sparsify(vector, keep)
    return {
        "layer": layer,
        "direction": vector,
        "norms": [l2(row) for row in directions],
        "sad_tokens": sad_count,
        "happy_tokens": happy_count,
        "directions": directions,
    }


def tomato_neurons(model, tokenizer, layer: int, max_length: int, top: int):
    positive, _pos_count = neuron_means(model, tokenizer, TOMATO, layer, max_length)
    negative, _neg_count = neuron_means(
        model, tokenizer, OTHER_OBJECTS, layer, max_length
    )
    direction = contrast([positive], [negative])
    return rank_components(direction, top), direction


def write_direction(path: str, record: dict) -> None:
    paired = bool(record.get("paired"))
    payload = {
        "model_id": record["model_id"],
        "method": "paired_difference_of_means" if paired else "difference_of_means",
        "positive": "class_a" if paired else "sad",
        "negative": "class_b" if paired else "happy",
        "layer": record["layer"],
        "norms": record["norms"],
        "direction": record["direction"],
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
        handle.write("\n")


def _is_probability_row(value) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) > 0
        and isinstance(value[0], (int, float))
    )


def _kl_pair(original, edited) -> float:
    """KL(original || edited). Lists are distributions; tensors are log-softmax pairs."""
    if _is_probability_row(original) and _is_probability_row(edited):
        return kl_divergence(original, edited)
    return kl_from_logprobs(original, edited)


def measure_abliteration(
    model,
    tokenizer,
    *,
    direction: Sequence[float],
    layers: Sequence[int],
    prompt: str,
    max_new_tokens: int,
    fact_rows: Sequence[tuple[str, str]] | None = None,
    generate_fn=None,
    distribution_fn=None,
    apply_edit=None,
) -> dict:
    """Capture original answers and distributions, write P, then score the edit.

    Mood, fact hits, and KL(original || edited) are read after `apply_edit`.
    The default edit is abliteration of `o_proj` and `down_proj`. No steering
    hook is opened. `distribution_fn` may return a probability list or the
    `(log_softmax, mask)` pair from `prompt_logprobs`.
    """
    if generate_fn is None:
        generate_fn = generate
    if distribution_fn is None:
        distribution_fn = prompt_logprobs
    if apply_edit is None:
        apply_edit = abliterate
    rows = FACTS if fact_rows is None else fact_rows

    baseline = generate_fn(model, tokenizer, prompt, max_new_tokens)
    saved = distribution_fn(model, tokenizer, prompt)
    facts_before = []
    for fact_prompt, expected in rows:
        answer = generate_fn(model, tokenizer, fact_prompt, max_new_tokens)
        facts_before.append((fact_prompt, answer, fact_hit(answer, expected)))

    names = apply_edit(model, direction, layers)

    edited = generate_fn(model, tokenizer, prompt, max_new_tokens)
    current = distribution_fn(model, tokenizer, prompt)
    facts_after = []
    for fact_prompt, expected in rows:
        answer = generate_fn(model, tokenizer, fact_prompt, max_new_tokens)
        facts_after.append((fact_prompt, answer, fact_hit(answer, expected)))
    return {
        "baseline": baseline,
        "baseline_mood": mood_score(baseline),
        "abliterated": edited,
        "abliterated_mood": mood_score(edited),
        "abliterated_modules": names,
        "kl": _kl_pair(saved, current),
        "facts_before": facts_before,
        "facts_after": facts_after,
    }


def story(
    model,
    tokenizer,
    *,
    layer: int | None,
    strength: float,
    keep: int | None,
    max_new_tokens: int,
    max_length: int,
    do_abliterate: bool,
    prompt: str = NEUTRAL_PROMPT,
    generated_only: bool = False,
) -> dict:
    if do_abliterate and generated_only:
        raise LabError("generated-only applies to the hook, not to abliteration")
    found = sadness_direction(model, tokenizer, max_length, keep)
    chosen = found["layer"] if layer is None else layer
    if not 0 <= chosen < len(found["directions"]):
        raise LabError(f"layer {chosen} is outside the captured stack")
    direction = found["directions"][chosen]
    if keep is not None:
        direction = sparsify(direction, keep)
    if do_abliterate:
        # Full projection P. Scores are original versus the edited weights.
        measured = measure_abliteration(
            model,
            tokenizer,
            direction=direction,
            layers=[chosen],
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )
        measured["layer"] = chosen
        measured["norms"] = found["norms"]
        measured["strength"] = strength
        measured["prompt"] = prompt
        measured["direction"] = direction
        measured["model_id"] = getattr(model.config, "name_or_path", QWEN3_0_6B)
        return measured
    baseline = generate(model, tokenizer, prompt, max_new_tokens)
    with steer(model, chosen, direction, strength, generated_only=generated_only):
        steered_text = generate(model, tokenizer, prompt, max_new_tokens)
    # KL and the fact lists each open the hook themselves. Nesting it would
    # add the direction twice.
    kl = prompt_kl(
        model,
        tokenizer,
        prompt,
        lambda: steer(model, chosen, direction, strength, generated_only=generated_only),
    )
    facts_before = score_facts(model, tokenizer, max_new_tokens)
    with steer(model, chosen, direction, strength, generated_only=generated_only):
        facts_after = score_facts(model, tokenizer, max_new_tokens)
    return {
        "layer": chosen,
        "norms": found["norms"],
        "strength": strength,
        "prompt": prompt,
        "baseline": baseline,
        "baseline_mood": mood_score(baseline),
        "steered": steered_text,
        "steered_mood": mood_score(steered_text),
        "kl": kl,
        "facts_before": facts_before,
        "facts_after": facts_after,
        "direction": direction,
        "model_id": getattr(model.config, "name_or_path", QWEN3_0_6B),
    }


def steer_many(
    model,
    tokenizer,
    *,
    layers: Sequence[int],
    strength: float,
    keep: int | None,
    max_new_tokens: int,
    max_length: int,
    prompt: str = NEUTRAL_PROMPT,
    generated_only: bool = False,
) -> dict:
    """Hook-steer one direction on every listed layer, then remove the hooks.

    The direction is the sad-minus-happy vector at the first listed layer.
    Mood, fact hits, and prompt KL describe how the answer changed.
    """
    chosen = [int(index) for index in layers]
    if not chosen:
        raise LabError("steer needs at least one layer")
    found = sadness_direction(model, tokenizer, max_length, keep)
    source = chosen[0]
    if not 0 <= source < len(found["directions"]):
        raise LabError(f"layer {source} is outside the captured stack")
    direction = found["directions"][source]
    if keep is not None:
        direction = sparsify(direction, keep)
    baseline = generate(model, tokenizer, prompt, max_new_tokens)
    with steer_layers(model, chosen, direction, strength, generated_only=generated_only):
        steered_text = generate(model, tokenizer, prompt, max_new_tokens)
    kl = prompt_kl(
        model,
        tokenizer,
        prompt,
        lambda: steer_layers(
            model, chosen, direction, strength, generated_only=generated_only
        ),
    )
    facts_before = score_facts(model, tokenizer, max_new_tokens)
    with steer_layers(model, chosen, direction, strength, generated_only=generated_only):
        facts_after = score_facts(model, tokenizer, max_new_tokens)
    return {
        "layer": source,
        "layers": chosen,
        "norms": found["norms"],
        "strength": strength,
        "prompt": prompt,
        "baseline": baseline,
        "baseline_mood": mood_score(baseline),
        "steered": steered_text,
        "steered_mood": mood_score(steered_text),
        "kl": kl,
        "facts_before": facts_before,
        "facts_after": facts_after,
        "direction": direction,
        "model_id": getattr(model.config, "name_or_path", QWEN3_0_6B),
    }


def catalog_on_model(
    model,
    tokenizer,
    *,
    layer: int,
    indexes: Sequence[int],
    strengths: Sequence[float],
    control_count: int,
    max_new_tokens: int,
    seed: int = 0,
    prompts: Sequence[str] | None = None,
    on_observation=None,
    store_direction: bool = True,
    generated_only: bool = False,
) -> dict:
    """Push the listed neurons and matched random directions, and record text."""
    from kilix_llm_lab.catalog import catalog_record
    from kilix_llm_lab.qwen import decoder_layers, neuron_linear
    from kilix_llm_lab.texts import EXPLORER_PROMPTS

    modules = decoder_layers(model)
    if not 0 <= layer < len(modules):
        raise LabError(f"layer {layer} is outside 0..{len(modules) - 1}")
    width = int(neuron_linear(modules[layer]).in_features)

    def generate_job(job: dict, prompt: str) -> str:
        with push_neurons(modules[layer], job["direction"], generated_only=generated_only):
            return generate(model, tokenizer, prompt, max_new_tokens)

    return catalog_record(
        indexes,
        strengths,
        control_count=control_count,
        width=width,
        generate=generate_job,
        prompts=EXPLORER_PROMPTS if prompts is None else prompts,
        seed=seed,
        on_observation=on_observation,
        store_direction=store_direction,
    )


def score_facts(model, tokenizer, max_new_tokens: int) -> list[tuple[str, str, bool]]:
    rows = []
    for prompt, expected in FACTS:
        answer = generate(model, tokenizer, prompt, max_new_tokens)
        rows.append((prompt, answer, fact_hit(answer, expected)))
    return rows


def format_story(result: dict) -> str:
    lines = [
        f"layer: {result['layer']}",
        "norms: " + " ".join(f"{index}:{norm:.4f}" for index, norm in enumerate(result["norms"])),
        f"strength: {result['strength']}",
        f"kl: {result['kl']:.6f}",
        f"baseline_mood: {result['baseline_mood']:.3f}",
    ]
    if "abliterated" in result:
        lines.append(f"abliterated_mood: {result['abliterated_mood']:.3f}")
        lines.append("baseline: " + _one_line(result["baseline"]))
        lines.append("abliterated: " + _one_line(result["abliterated"]))
        lines.append("facts_before: " + _fact_line(result["facts_before"]))
        lines.append("facts_after: " + _fact_line(result["facts_after"]))
        lines.append("abliterated_modules: " + ", ".join(result["abliterated_modules"]))
    else:
        lines.append(f"steered_mood: {result['steered_mood']:.3f}")
        lines.append("baseline: " + _one_line(result["baseline"]))
        lines.append("steered: " + _one_line(result["steered"]))
        lines.append("facts_before: " + _fact_line(result["facts_before"]))
        lines.append("facts_after: " + _fact_line(result["facts_after"]))
    return "\n".join(lines) + "\n"


def format_neurons(layer: int, ranked: Sequence[tuple[int, float]], sample: str | None) -> str:
    lines = [f"layer: {layer}"]
    for index, value in ranked:
        lines.append(f"neuron: {index} {value:.6f}")
    if sample is not None:
        lines.append("pushed: " + _one_line(sample))
    return "\n".join(lines) + "\n"


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _fact_line(rows: Sequence[tuple[str, str, bool]]) -> str:
    hits = sum(1 for _prompt, _answer, hit in rows if hit)
    return f"{hits}/{len(rows)}"
