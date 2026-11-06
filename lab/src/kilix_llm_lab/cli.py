"""Command line for the Qwen3-0.6B edit project."""

from __future__ import annotations

import argparse
import sys

from kilix_llm_lab import LabError, __version__
from kilix_llm_lab.intervene import CATALOG, OPS, QWEN3_5_4B
from kilix_llm_lab.qwen import INTERMEDIATE_4B, QWEN3_0_6B
from kilix_llm_lab.runner import (
    catalog_on_model,
    format_neurons,
    format_story,
    load,
    push_neuron,
    sadness_direction,
    steer_many,
    story,
    tomato_neurons,
    write_direction,
)
from kilix_llm_lab.texts import NEUTRAL_PROMPT, WHO_PROMPT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kilix-llm-lab",
        description="Edit and measure Qwen3-0.6B without fine-tuning.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--model", default=QWEN3_0_6B, help="Hugging Face id or local path")
    shared.add_argument("--device", default="cpu")
    shared.add_argument(
        "--allow-other",
        action="store_true",
        help="accept a checkpoint that is not Qwen3-0.6B",
    )
    shared.add_argument("--max-length", type=int, default=128, help="tokens kept when reading activations")
    shared.add_argument(
        "--quant",
        default="none",
        choices=("none", "4bit"),
        help="4bit NF4 keeps a 4B checkpoint on an 8 GB GPU",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    means = commands.add_parser("means", parents=[shared], help="sad-minus-happy residual directions")
    means.add_argument("--keep", type=int, default=None, help="sparsify the chosen direction")
    means.add_argument("--save", default=None, help="write the chosen direction as JSON")
    means.add_argument(
        "--pairs",
        default=None,
        help="JSON list of [class_a, class_b]; average only after the first differing token",
    )

    story_p = commands.add_parser("story", parents=[shared], help="steer a story and score the change")
    story_p.add_argument("--layer", type=int, default=None, help="default: largest difference")
    story_p.add_argument("--strength", type=float, default=1.0)
    story_p.add_argument("--keep", type=int, default=None)
    story_p.add_argument("--max-new-tokens", type=int, default=60)
    story_p.add_argument("--prompt", default=NEUTRAL_PROMPT)
    story_p.add_argument(
        "--generated-only",
        action="store_true",
        help="add the direction on decode steps, not while reading the prompt",
    )
    story_p.add_argument(
        "--abliterate",
        action="store_true",
        help="remove the direction from o_proj and down_proj, then score that weight edit",
    )
    story_p.add_argument("--save", default=None)

    steer_p = commands.add_parser(
        "steer",
        parents=[shared],
        help="hook steering on one or more layers",
    )
    steer_p.add_argument("--layer", type=int, action="append", required=True)
    steer_p.add_argument("--strength", type=float, default=1.0)
    steer_p.add_argument("--keep", type=int, default=None)
    steer_p.add_argument("--max-new-tokens", type=int, default=60)
    steer_p.add_argument("--prompt", default=NEUTRAL_PROMPT)
    steer_p.add_argument(
        "--generated-only",
        action="store_true",
        help="add the direction on decode steps, not while reading the prompt",
    )

    ablate_p = commands.add_parser(
        "abliterate",
        parents=[shared],
        help="abliteration of one layer's writing weights, then mood, facts, and KL",
    )
    ablate_p.add_argument("--layer", type=int, default=None)
    ablate_p.add_argument("--strength", type=float, default=1.0)
    ablate_p.add_argument("--keep", type=int, default=None)
    ablate_p.add_argument("--max-new-tokens", type=int, default=60)
    ablate_p.add_argument("--prompt", default=NEUTRAL_PROMPT)

    control = commands.add_parser(
        "control",
        help="random-direction control: the neuron wins only if it beats every control",
    )
    control.add_argument("--effect", type=float, required=True)
    control.add_argument("--control", type=float, action="append", required=True)

    catalog = commands.add_parser(
        "catalog",
        parents=[shared],
        help="bounded catalog of neuron pushes plus matched random directions",
    )
    catalog.add_argument("--index", type=int, action="append", required=True)
    catalog.add_argument("--strength", type=float, action="append", required=True)
    catalog.add_argument("--controls", type=int, default=2)
    catalog.add_argument("--width", type=int, default=8)
    catalog.add_argument("--layer", type=int, default=0)
    catalog.add_argument("--seed", type=int, default=0)
    catalog.add_argument("--max-new-tokens", type=int, default=16)
    catalog.add_argument(
        "--stand-in",
        action="store_true",
        help="record generations without loading a checkpoint",
    )
    catalog.add_argument(
        "--generated-only",
        action="store_true",
        help="push neurons on decode steps, not while reading the prompt",
    )

    neurons = commands.add_parser(
        "neurons",
        parents=[shared],
        help="tomato-minus-other MLP neurons, optional single-neuron push",
    )
    neurons.add_argument("--layer", type=int, required=True)
    neurons.add_argument("--top", type=int, default=8)
    neurons.add_argument("--push", type=int, default=None, help="neuron index to add to")
    neurons.add_argument("--amount", type=float, default=5.0)
    neurons.add_argument("--max-new-tokens", type=int, default=40)
    neurons.add_argument("--prompt", default=WHO_PROMPT)

    neuron = commands.add_parser(
        "neuron",
        parents=[shared],
        help="increase, decrease, activate, or deactivate one MLP neuron",
    )
    neuron.add_argument("--name", choices=(*sorted(CATALOG), "all"), default=None)
    neuron.add_argument("--layer", type=int, default=None)
    neuron.add_argument("--index", type=int, default=None)
    neuron.add_argument("--op", required=True, choices=(*OPS, "all"))
    neuron.add_argument("--amount", type=float, default=8.0)
    neuron.add_argument("--prompt", default=None)
    neuron.add_argument("--max-new-tokens", type=int, default=32)
    neuron.add_argument("--dtype", default="auto", choices=("auto", "float32", "bfloat16"))
    neuron.add_argument(
        "--span",
        type=int,
        default=0,
        help="also apply the op to this many neighboring indexes on each side",
    )
    neuron.add_argument(
        "--generated-only",
        action="store_true",
        help="edit the neuron on decode steps, not while reading the prompt",
    )

    head = commands.add_parser(
        "head",
        parents=[shared],
        help="attention-head edit on the o_proj input",
    )
    head.add_argument("--layer", type=int, default=0)
    head.add_argument("--head", type=int, default=0)
    head.add_argument("--heads", type=int, default=None)
    head.add_argument("--op", default="deactivate", choices=OPS)
    head.add_argument("--amount", type=float, default=1.0)
    head.add_argument("--vector", type=float, action="append")
    head.add_argument("--prompt", default=WHO_PROMPT)
    head.add_argument("--max-new-tokens", type=int, default=32)
    head.add_argument("--stand-in", action="store_true")

    sae = commands.add_parser("sae", help="sparse autoencoder training on activation rows")
    sae.add_argument("--steps", type=int, default=40)
    sae.add_argument("--features", type=int, default=4)

    commands.add_parser("circuit", help="circuit trace by activation patching")

    refusal = commands.add_parser(
        "refusal",
        parents=[shared],
        help="refusal corpus: refusal wording minus compliant wording",
    )
    refusal.add_argument("--list", action="store_true", help="print the built-in corpus and exit")
    refusal.add_argument("--corpus", default=None, help="JSON object with refuse and comply lists")
    refusal.add_argument("--layer", type=int, default=None)
    refusal.add_argument("--abliterate", action="store_true")

    sweep = commands.add_parser(
        "sweep",
        parents=[shared],
        help="neuron sweep across an index range, including all 9216",
    )
    sweep.add_argument("--start", type=int, default=0)
    sweep.add_argument("--stop", type=int, default=None)
    sweep.add_argument("--width", type=int, default=INTERMEDIATE_4B)
    sweep.add_argument("--strength", type=float, action="append")
    sweep.add_argument("--layer", type=int, default=0)
    sweep.add_argument("--controls", type=int, default=1)
    sweep.add_argument("--seed", type=int, default=0)
    sweep.add_argument("--max-new-tokens", type=int, default=16)
    sweep.add_argument("--stand-in", action="store_true")
    sweep.add_argument(
        "--record",
        default=None,
        help="append one JSON object per generation while the sweep runs",
    )
    sweep.add_argument(
        "--shards",
        type=int,
        default=None,
        help="split the index range into this many pieces",
    )
    sweep.add_argument(
        "--shard",
        type=int,
        default=0,
        help="which piece to run, counting from 0",
    )
    sweep.add_argument(
        "--generated-only",
        action="store_true",
        help="push neurons on decode steps, not while reading the prompt",
    )

    rank = commands.add_parser(
        "rank",
        help="rank recorded generations by distinctive-word mass",
    )
    rank.add_argument("record", help="JSONL of kind, index, prompt, and text")
    rank.add_argument("--show", type=int, default=20)
    rank.add_argument("--top-words", type=int, default=12)
    rank.add_argument(
        "--max-strength",
        type=float,
        default=None,
        help="ignore generations whose absolute strength is above this",
    )

    weight = commands.add_parser(
        "weight",
        help="weight-space column edit on one writing matrix",
    )
    weight.add_argument("--matrix", default=None, help="JSON object with a rows array")
    weight.add_argument("--write-matrix", default=None, help="write the edited rows JSON here")
    weight.add_argument("--op", required=True, choices=("remove", "scale", "zero"))
    weight.add_argument("--direction", default=None, help="comma-separated output direction")
    weight.add_argument("--direction-file", default=None, help="JSON object with a direction array")
    weight.add_argument("--columns", default=None, help="comma-separated input columns")
    weight.add_argument("--top", type=int, default=None, help="columns that write the direction most")
    weight.add_argument("--scale", type=float, default=1.0, help="remove strength; 1 drops the direction")
    weight.add_argument("--factor", type=float, default=None, help="column multiplier for scale")
    weight.add_argument("--layer", type=int, default=None, help="decoder layer of a loaded checkpoint")
    weight.add_argument("--module", default="down_proj", choices=("down_proj", "o_proj"))
    weight.add_argument("--model", default=QWEN3_0_6B)
    weight.add_argument("--device", default="cpu")
    weight.add_argument("--allow-other", action="store_true")
    weight.add_argument("--dtype", default="auto", choices=("auto", "float32", "bfloat16"))
    weight.add_argument("--quant", default="none", choices=("none", "4bit"))
    weight.add_argument("--prompt", default=None, help="greedy continuation before and after the edit")
    weight.add_argument("--max-new-tokens", type=int, default=32)

    output_row = commands.add_parser(
        "output-row",
        help="squared-hinge edit of one output row from frozen activations",
    )
    output_row.add_argument("--capture", required=True, help="JSON or npz states, signs, bounds, and case ids")
    output_row.add_argument("--write-correction", default=None, help="write the float row change to a new JSON file")
    output_row.add_argument("--parent-gguf", default=None, help="read-only parent model to copy")
    output_row.add_argument("--output", default=None, help="new model path; the parent is not opened for write")
    output_row.add_argument("--row", type=int, default=None, help="output.weight row index")
    output_row.add_argument("--expected-sha256", default=None, help="sha256 the parent file must have")
    output_row.add_argument("--max-iter", type=int, default=None, help="solver iteration cap")
    output_row.add_argument("--regularization", type=float, default=None, help="override the capture L2 coefficient")
    return parser


def _neuron_model(args) -> None:
    """Named transcript neurons live on Qwen3.5-4B, not the 0.6B default."""
    if args.name is None:
        return
    if args.model == QWEN3_0_6B:
        args.model = QWEN3_5_4B
        args.allow_other = True


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "control":
            return _run_control(args)
        if args.command == "rank":
            return _run_rank(args)
        if args.command == "weight":
            from kilix_llm_lab.weight import run_weight

            return run_weight(args)
        if args.command == "output-row":
            from kilix_llm_lab.output_row import run_output_row

            return run_output_row(args)
        pairs = _load_pairs(args.pairs) if args.command == "means" and args.pairs else None
        if args.command == "sae":
            return _run_sae(args)
        if args.command == "circuit":
            return _run_circuit()
        if args.command == "head" and args.stand_in:
            return _run_head_stand_in(args)
        if args.command == "refusal" and args.list:
            return _run_refusal_list()
        if args.command == "sweep" and args.stand_in:
            return _run_sweep(None, None, args)
        if args.command == "catalog" and args.stand_in:
            return _run_catalog(None, None, args)
        if args.command == "neuron":
            _neuron_model(args)
        model, tokenizer = load(
            args.model,
            args.device,
            allow_other=args.allow_other,
            dtype_name=getattr(args, "dtype", "auto"),
            quant=getattr(args, "quant", "none"),
        )
        if args.command == "neuron":
            return _run_neuron(model, tokenizer, args)
        if args.command == "head":
            return _run_head(model, tokenizer, args)
        if args.command == "sweep":
            return _run_sweep(model, tokenizer, args)
        if args.command == "refusal":
            return _run_refusal(model, tokenizer, args)
        if args.command == "catalog":
            return _run_catalog(model, tokenizer, args)
        if args.command == "steer":
            result = steer_many(
                model,
                tokenizer,
                layers=args.layer,
                strength=args.strength,
                keep=args.keep,
                max_new_tokens=args.max_new_tokens,
                max_length=args.max_length,
                prompt=args.prompt,
                generated_only=args.generated_only,
            )
            sys.stdout.write(format_story(result))
            return 0
        if args.command == "means":
            if pairs is not None:
                from kilix_llm_lab.runner import direction_from_pairs

                found = direction_from_pairs(
                    model, tokenizer, pairs, args.max_length, args.keep
                )
            else:
                found = sadness_direction(model, tokenizer, args.max_length, args.keep)
            print(f"layer: {found['layer']}")
            print(
                "norms: "
                + " ".join(f"{index}:{norm:.4f}" for index, norm in enumerate(found["norms"]))
            )
            if args.save:
                found["model_id"] = args.model
                write_direction(args.save, found)
                print(f"saved: {args.save}")
            return 0
        if args.command in ("story", "abliterate"):
            result = story(
                model,
                tokenizer,
                layer=args.layer,
                strength=args.strength,
                keep=args.keep,
                max_new_tokens=args.max_new_tokens,
                max_length=args.max_length,
                do_abliterate=args.command == "abliterate" or args.abliterate,
                prompt=args.prompt,
                generated_only=getattr(args, "generated_only", False),
            )
            sys.stdout.write(format_story(result))
            if getattr(args, "save", None):
                result["model_id"] = args.model
                write_direction(args.save, result)
                print(f"saved: {args.save}")
            return 0
        ranked, _direction = tomato_neurons(
            model, tokenizer, args.layer, args.max_length, args.top
        )
        sample = None
        if args.push is not None:
            from kilix_llm_lab.collect import generate
            from kilix_llm_lab.qwen import decoder_layers

            with push_neuron(decoder_layers(model)[args.layer], args.push, args.amount):
                sample = generate(model, tokenizer, args.prompt, args.max_new_tokens)
        sys.stdout.write(format_neurons(args.layer, ranked, sample))
        return 0
    except LabError as error:
        print(f"kilix-llm lab: {error}", file=sys.stderr)
        return 1


def _load_pairs(path: str) -> list[tuple[str, str]]:
    import json

    try:
        handle = open(path, encoding="utf-8")
    except OSError as error:
        raise LabError(f"cannot read pairs file: {error.strerror}") from error
    with handle:
        try:
            payload = json.load(handle)
        except json.JSONDecodeError as error:
            raise LabError("pairs file is not JSON") from error
    if not isinstance(payload, list) or not payload:
        raise LabError("pairs file must be a non-empty list of [class_a, class_b]")
    pairs = []
    for item in payload:
        if (
            not isinstance(item, (list, tuple))
            or len(item) != 2
            or not isinstance(item[0], str)
            or not isinstance(item[1], str)
        ):
            raise LabError("each pair is two strings")
        pairs.append((item[0], item[1]))
    return pairs


def _run_rank(args) -> int:
    import json

    from kilix_llm_lab.rank import load_jsonl, rank_units

    records = rank_units(
        load_jsonl(args.record),
        top_words=args.top_words,
        max_strength=args.max_strength,
    )
    controls = sum(item["kind"] in {"control", "random"} for item in records)
    print(
        json.dumps(
            {
                "units": len(records),
                "controls": controls,
                "top": records[: args.show],
            },
            ensure_ascii=False,
        )
    )
    return 0


def _run_control(args) -> int:
    import json

    from kilix_llm_lab.measure import beats_every_control

    wins = beats_every_control(args.effect, args.control)
    print(
        json.dumps(
            {
                "neuron_effect": args.effect,
                "control_effects": list(args.control),
                "wins": wins,
            }
        )
    )
    return 0


def _run_catalog(model, tokenizer, args) -> int:
    import json

    from kilix_llm_lab.algebra import l2
    from kilix_llm_lab.catalog import catalog_record, stand_in_generation

    if args.stand_in:
        record = catalog_record(
            args.index,
            args.strength,
            control_count=args.controls,
            width=args.width,
            generate=stand_in_generation,
            seed=args.seed,
        )
    else:
        record = catalog_on_model(
            model,
            tokenizer,
            layer=args.layer,
            indexes=args.index,
            strengths=args.strength,
            control_count=args.controls,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
            generated_only=args.generated_only,
        )
    observations = record["observations"]
    print(
        json.dumps(
            {
                "prompts": record["prompts"],
                "neuron_observations": sum(item["kind"] == "neuron" for item in observations),
                "control_observations": sum(item["kind"] == "control" for item in observations),
                "matched_norm": l2(observations[0]["direction"]) if observations else 0.0,
                "sample": observations[0]["generations"][0]["text"] if observations else "",
            }
        )
    )
    return 0


def _run_neuron(model, tokenizer, args) -> int:
    import json
    from dataclasses import replace

    from kilix_llm_lab.collect import generate
    from kilix_llm_lab.edit import intervene_neuron
    from kilix_llm_lab.intervene import surrounding
    from kilix_llm_lab.qwen import decoder_layers, neuron_linear

    if args.name == "all":
        specs = [CATALOG[name] for name in ("evil", "tomato", "alien")]
    elif args.name:
        spec = CATALOG[args.name]
        specs = [
            replace(
                spec,
                layer=spec.layer if args.layer is None else args.layer,
                index=spec.index if args.index is None else args.index,
                prompt=spec.prompt if args.prompt is None else args.prompt,
            )
        ]
    elif args.layer is None or args.index is None or args.prompt is None:
        raise LabError("pass --name, or --layer, --index, and --prompt")
    else:
        specs = [
            replace(
                CATALOG["evil"],
                name="custom",
                model_id=args.model,
                layer=args.layer,
                index=args.index,
                prompt=args.prompt,
                note="explicit index",
            )
        ]
    ops = OPS if args.op == "all" else (args.op,)
    layers = decoder_layers(model)
    for spec in specs:
        if not 0 <= spec.layer < len(layers):
            raise LabError(f"layer {spec.layer} is outside 0..{len(layers) - 1}")
        width = int(neuron_linear(layers[spec.layer]).in_features)
        indexes = surrounding(spec.index, args.span, width)
        baseline = " ".join(generate(model, tokenizer, spec.prompt, args.max_new_tokens).split())
        print(
            json.dumps(
                {
                    "name": spec.name,
                    "layer": spec.layer,
                    "index": spec.index,
                    "op": "baseline",
                    "prompt": spec.prompt,
                    "text": baseline,
                }
            ),
            flush=True,
        )
        for index in indexes:
            for op in ops:
                with intervene_neuron(
                    layers[spec.layer],
                    index,
                    op,
                    args.amount,
                    generated_only=args.generated_only,
                ):
                    text = " ".join(
                        generate(model, tokenizer, spec.prompt, args.max_new_tokens).split()
                    )
                print(
                    json.dumps(
                        {
                            "name": spec.name,
                            "layer": spec.layer,
                            "index": index,
                            "op": op,
                            "amount": args.amount,
                            "center": index == spec.index,
                            "generated_only": args.generated_only,
                            "text": text,
                            "changed": text != baseline,
                        }
                    ),
                    flush=True,
                )
    return 0


def _run_sae(args) -> int:
    import json

    from kilix_llm_lab.sae import train_sae

    batch = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
    trained = train_sae(batch, args.features, args.steps)
    print(
        json.dumps(
            {
                "width": trained["width"],
                "n_features": trained["n_features"],
                "steps": args.steps,
                "loss_start": trained["loss_start"],
                "loss_end": trained["loss_end"],
            }
        )
    )
    return 0


def _run_circuit() -> int:
    import json

    from kilix_llm_lab.circuit import patch_effects

    clean = [[1.0, 0.0], [0.0, 0.0]]
    corrupt = [[0.0, 0.0], [0.0, 0.0]]

    def run(states):
        total = [0.0, 0.0]
        for state in states:
            total[0] += float(state[0])
            total[1] += float(state[1])
        return total

    effects = patch_effects(clean, corrupt, run, lambda state: state[0])
    print(json.dumps({"sites": len(effects), "effects": effects}))
    return 0


def _run_head_stand_in(args) -> int:
    import json

    from kilix_llm_lab.heads import edit_head_slice

    if args.heads is None:
        raise LabError("a stand-in head edit needs --heads")
    vector = args.vector if args.vector else [1.0, 2.0, 3.0, 4.0]
    edited = edit_head_slice(vector, args.head, args.heads, args.op, args.amount)
    print(
        json.dumps(
            {
                "head": args.head,
                "heads": args.heads,
                "op": args.op,
                "before": vector,
                "after": edited,
            }
        )
    )
    return 0


def _run_head(model, tokenizer, args) -> int:
    import json

    from kilix_llm_lab.collect import generate
    from kilix_llm_lab.edit import intervene_head
    from kilix_llm_lab.heads import attention_heads
    from kilix_llm_lab.qwen import decoder_layers

    n_heads = args.heads if args.heads is not None else attention_heads(model.config)
    layers = decoder_layers(model)
    if not 0 <= args.layer < len(layers):
        raise LabError(f"layer {args.layer} is outside 0..{len(layers) - 1}")
    with intervene_head(layers[args.layer], args.head, n_heads, args.op, args.amount):
        text = " ".join(generate(model, tokenizer, args.prompt, args.max_new_tokens).split())
    print(
        json.dumps(
            {
                "layer": args.layer,
                "head": args.head,
                "heads": n_heads,
                "op": args.op,
                "text": text,
            }
        )
    )
    return 0


def _run_refusal_list() -> int:
    import json

    from kilix_llm_lab.refusal import builtin_corpus

    refuse, comply = builtin_corpus()
    print(
        json.dumps(
            {
                "refuse": len(refuse),
                "comply": len(comply),
                "refuse_example": refuse[0],
                "comply_example": comply[0],
            }
        )
    )
    return 0


def _run_refusal(model, tokenizer, args) -> int:
    from kilix_llm_lab.algebra import l2, strongest_index
    from kilix_llm_lab.collect import residual_means
    from kilix_llm_lab.edit import abliterate
    from kilix_llm_lab.refusal import builtin_corpus, load_corpus, refusal_directions

    if args.corpus:
        refuse, comply = load_corpus(args.corpus)
    else:
        refuse, comply = builtin_corpus()
    refuse_layers, _refuse_count = residual_means(
        model, tokenizer, refuse, max_length=args.max_length
    )
    comply_layers, _comply_count = residual_means(
        model, tokenizer, comply, max_length=args.max_length
    )
    directions = refusal_directions(refuse_layers, comply_layers)
    layer = args.layer if args.layer is not None else strongest_index(directions)
    if not 0 <= layer < len(directions):
        raise LabError(f"layer {layer} is outside 0..{len(directions) - 1}")
    names = []
    if args.abliterate:
        names = abliterate(model, directions[layer], [layer])
    print(f"layer: {layer}")
    print("norms: " + " ".join(f"{index}:{l2(row):.4f}" for index, row in enumerate(directions)))
    print(f"refuse: {len(refuse)}")
    print(f"comply: {len(comply)}")
    if names:
        print("abliterated_modules: " + ", ".join(names))
    return 0


def _run_sweep(model, tokenizer, args) -> int:
    import json

    from kilix_llm_lab.catalog import catalog_record, stand_in_generation
    from kilix_llm_lab.qwen import INTERMEDIATE_4B
    from kilix_llm_lab.runner import catalog_on_model
    from kilix_llm_lab.sweep import shard_bounds, sweep_indexes

    if args.stand_in and args.stop is None:
        raise LabError(
            "a stand-in sweep needs --stop; 0:9216 is the full layer and is not a stand-in"
        )
    stop = INTERMEDIATE_4B if args.stop is None else args.stop
    start = args.start
    if args.shards is not None:
        start, stop = shard_bounds(start, stop, args.shards, args.shard)
    indexes = sweep_indexes(start, stop, args.width)
    strengths = args.strength if args.strength else [1.0]
    record_file = open(args.record, "w", encoding="utf-8") if args.record else None

    def on_observation(observation: dict) -> None:
        if record_file is None:
            return
        for row in observation["generations"]:
            record_file.write(
                json.dumps(
                    {
                        "kind": observation["kind"],
                        "index": observation["index"],
                        "strength": observation["strength"],
                        "prompt": row["prompt"],
                        "text": row["text"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        record_file.flush()

    try:
        if args.stand_in:
            record = catalog_record(
                indexes,
                strengths,
                control_count=args.controls,
                width=args.width,
                generate=stand_in_generation,
                seed=args.seed,
                on_observation=on_observation if record_file else None,
                store_direction=record_file is None,
            )
        else:
            record = catalog_on_model(
                model,
                tokenizer,
                layer=args.layer,
                indexes=indexes,
                strengths=strengths,
                control_count=args.controls,
                max_new_tokens=args.max_new_tokens,
                seed=args.seed,
                on_observation=on_observation if record_file else None,
                store_direction=record_file is None,
                generated_only=args.generated_only,
            )
    finally:
        if record_file is not None:
            record_file.close()
    observations = record["observations"]
    print(
        json.dumps(
            {
                "width": args.width,
                "start": start,
                "stop": stop,
                "shard": None if args.shards is None else args.shard,
                "shards": args.shards,
                "indexes": len(indexes),
                "full_layer": len(indexes) == INTERMEDIATE_4B,
                "neuron_observations": sum(item["kind"] == "neuron" for item in observations),
                "control_observations": sum(item["kind"] == "control" for item in observations),
                "generated_only": args.generated_only,
                "sample": observations[0]["generations"][0]["text"] if observations else "",
            }
        )
    )
    return 0
