# Qwen3-0.6B edit project

This is the first lab project. It reruns residual steering, abliteration, and
MLP neuron edits on `Qwen/Qwen3-0.6B`.

The checkpoint shape the loader requires, unless `--allow-other` is set:

| Field | Value |
|---|---|
| Hugging Face id | `Qwen/Qwen3-0.6B` |
| Base alternative | `Qwen/Qwen3-0.6B-Base` |
| `model_type` | `qwen3` |
| Layers | 28 |
| Hidden size | 1024 |
| MLP width | 3072 |

`Qwen/Qwen3-0.6B-Base` is the pre-RL checkpoint. It has the same shape, so
the check accepts it. Post-training checkpoints of this size tend to answer
in a narrower assistant voice. The base id is the better bed for a neuron
push that should show varied behavior.

The host command list is `lab/scripts/lab.sh` with no arguments: difference of means (`means`), hook steering (`steer`), abliteration (`abliterate`), weight-space column edits (`weight`), the Ask output-row edit (`output-row`), neuron push (`neuron`), the random-direction control (`control`), and the bounded catalog (`catalog`). The score is how the answer changes. `weight` can edit a JSON matrix with no checkpoint.

## Environment

Create the virtual environment with uv. The helper is
[../../scripts/uv-venv.sh](../../scripts/uv-venv.sh).

```sh
lab/scripts/uv-venv.sh
```

On a GPU machine, select the matching torch index before the first install:

```sh
KILIX_LLM_TORCH_INDEX=https://download.pytorch.org/whl/cu124 lab/scripts/uv-venv.sh
```

Set `HF_HOME` first if the checkpoint should live on a large disk. The files
are the ordinary Hugging Face snapshot of `Qwen/Qwen3-0.6B`. This repository
does not vendor them.

## `run.py`

`run.py` is the edit program. It has three commands.

`means` reads the built-in happy and sad lines, prints the L2 norm of
sad-minus-happy at every layer, and names the strongest layer. `--save
path.json` writes that layer's direction. `--keep N` keeps the N largest
entries and zeros the rest.

`story` generates from `Tell me a story.` with the weights untouched, then
again with a forward hook on the chosen layer. It reports lexicon mood, fact
hits, and KL on that prompt. The default layer is the strongest difference.
`--layer` overrides it. `--strength` scales the direction (default 1).
`--abliterate` then writes `P = I - r rᵀ` into that layer's `o_proj` and
`down_proj` and generates once more. Those two matrices stay edited until the
process exits. The files in `HF_HOME` are not modified.

`neurons` contrasts two tomato lines with eight other-object lines at one
MLP and lists the largest neuron differences. `--layer` is required.
`--push INDEX --amount 5` adds that amount to one neuron and generates from
`Who are you?`. Neuron indexes are 0 .. 3071 on this model. An index taken
from a larger Qwen does not apply here.

```sh
lab/.venv/bin/python lab/projects/qwen3_0_6b/run.py means
lab/.venv/bin/python lab/projects/qwen3_0_6b/run.py story --strength 1 --max-new-tokens 60
lab/.venv/bin/python lab/projects/qwen3_0_6b/run.py story --layer 11 --strength 1 --abliterate
lab/.venv/bin/python lab/projects/qwen3_0_6b/run.py neurons --layer 7 --top 8 --push 12 --amount 5
```

Add `--device cuda` to any of those when the environment's torch can see a
GPU. The default device is `cpu`, which loads the checkpoint in float32. A
non-CPU device loads it in bfloat16.

Activations use `--max-length` tokens (default 128) and drop padding. The
contrast texts are short and original. They exist so the commands run. They
are not a claim about the direction a larger corpus would find.

## `bench.py`

[bench.py](bench.py) measures how greedy answers change when the weights
change. The shell wrapper
[../../scripts/bench-qwen3-0.6b.sh](../../scripts/bench-qwen3-0.6b.sh) creates
the uv environment if needed and then runs it.

```sh
lab/scripts/bench-qwen3-0.6b.sh --device cuda
lab/scripts/bench-qwen3-0.6b.sh --device cpu --new-tokens 16
```

Setup, not part of the score: load the checkpoint, then form the
sad-minus-happy direction. The timed download of a missing checkpoint is not
reported.

The edit is abliteration of one decoder layer, the strongest layer unless
`--layer` is set. `o_proj` and `down_proj` in that layer are replaced with
`P @ W`. The run stops if that direction is still present (max absolute
column dot above 1e-3). The files in `HF_HOME` stay as downloaded.

The prompts are three story requests (`Tell me a story.`, `Tell me a sad
story.`, `Tell me a happy story.`) plus the fact lines in `texts.py`. Each
prompt is generated with the original weights, the layer is edited, and the
same prompt is generated again. `--new-tokens` defaults to 48. Decoding is
greedy.

Each prompt prints one JSON object:

| Field | Meaning |
|---|---|
| `before`, `after` | The two continuations |
| `changed` | The strings differ |
| `shared_prefix_chars` | How far the two strings still match |
| `kl` | Mean KL(original \|\| edited) over the prompt positions, in nats |
| `mood_before`, `mood_after` | Lexicon mood of each continuation |
| `fact_before`, `fact_after` | Present only on fact prompts |

The last object is the summary: layer, how many answers changed, mean KL,
and fact hits before and after. It has no download time and no tokens per
second.
