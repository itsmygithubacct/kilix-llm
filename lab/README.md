# kilix-llm lab

Read and edit a local transformer. Edit commands do not fine-tune.
`train` prepares a corpus you supply and runs vendored `llama-finetune`.
The work is difference
of means, a removable forward hook, abliteration of the matrices that write
the residual stream, and MLP neuron reads. The llama.cpp launcher above this
directory does not import the lab. On Kilix 0.2.3 the command is
`kilix llm lab`. With no command it lists the options. `make all` and
`make test` do not install the lab or download a checkpoint.

Weights are not in this repository. Qwen3 checkpoints are Apache-2.0 and are
fetched by the model commands into the Hugging Face cache.

## Layout

| Path | What it is |
|---|---|
| `LICENSE` | MIT, copyright 2026 itsmygithubacct. Same grant as the repository root |
| `src/kilix_llm_lab/algebra.py` | Means, paired tails, aligned-neuron scores, sparsify, `P = I - r rᵀ`, KL |
| `src/kilix_llm_lab/edit.py` | Forward hook, neuron push, in-place abliteration |
| `src/kilix_llm_lab/weight.py` | Column remove, scale, and zero on one writing matrix |
| `src/kilix_llm_lab/output_row.py` | Squared-hinge edit of one output row, with sealed Ask ids |
| `src/kilix_llm_lab/corpus.py` | Join a caller-supplied corpus into one UTF-8 text file |
| `src/kilix_llm_lab/train.py` | Plan and run vendored `llama-finetune` on that file |
| `src/kilix_llm_lab/loops.py` | Loop onset and period in a thinking trace, and per-token copy targets |
| `src/kilix_llm_lab/gguf_columns.py` | Exact zero and scale column edits on a copy of a Q8_0, F16, or F32 GGUF |
| `src/kilix_llm_lab/gguf_tensors.py` | Whole-tensor replacement from `.npy`, requantized, with a base round-trip check |
| `docs/weight-space.md` | Command reference for those column edits |
| `docs/output-row.md` | Command reference for the Ask output-row edit |
| `docs/train.md` | Command reference for corpus training with `llama-finetune` |
| `src/kilix_llm_lab/collect.py` | Masked residual means, paired residual means, neuron means, prompt KL, greedy generation |
| `src/kilix_llm_lab/measure.py` | Lexicon mood, fact-substring hits, topic-and-target success |
| `src/kilix_llm_lab/rank.py` | Distinctive-word mass of sweep generations against random-direction controls |
| `src/kilix_llm_lab/qwen.py` | Mixer writers (`self_attn.o_proj`, or `linear_attn.out_proj` on a Qwen3.5 linear-attention layer), `down_proj`, and the 0.6B shape check |
| `src/kilix_llm_lab/texts.py` | Small happy/sad, tomato/other, and fact sets |
| `src/kilix_llm_lab/runner.py` | Load a checkpoint and run one edit |
| `src/kilix_llm_lab/cli.py` | `means`, `steer`, `abliterate`, `neuron`, `weight`, `train`, `control`, `rank`, `catalog`, `loops`, `gguf-columns`, `gguf-tensors` |
| `src/kilix_llm_lab/catalog.py` | Bounded neuron catalog and its stand-in |
| `scripts/lab.sh` | `kilix llm lab` dispatcher. No arguments lists the commands |
| `scripts/uv-venv.sh` | Create the uv virtual environment |
| `scripts/fetch-model.sh` | Download one checkpoint into `HF_HOME` |
| `scripts/neuron.sh` | Increase, decrease, activate, or deactivate a neuron |
| `scripts/sweep-shards.sh` | One sweep process per shard, one GPU each |
| `scripts/bench-qwen3-0.6b.sh` | Compare answers before and after one weight edit |
| `projects/qwen3_0_6b/` | The 0.6B project: runbook, `run.py`, `bench.py` |

## Virtual environment

Model commands that load a Hugging Face checkpoint use a virtual environment
created by uv. They do not use `python -m venv`. `train` does not call that
script: its trainer is the `llama-finetune` binary, and the script's default
torch index is CPU, which replaces a CUDA torch.

```sh
lab/scripts/uv-venv.sh
```

That runs `uv venv --python 3.12` when the environment does not exist yet,
then `uv pip install` for torch, numpy, and transformers. If `bin/python` is already
there, the script reuses it and only installs what is missing. Replacing an
environment is `rm -rf` on that directory, then running the script again.
`uv venv` itself refuses to overwrite an existing environment. The default
location is `lab/.venv`. The default torch index is the CPU wheel.

| Variable | Default | Use |
|---|---|---|
| `KILIX_LLM_LAB_VENV` | `lab/.venv` | Environment directory |
| `KILIX_LLM_TORCH_INDEX` | `https://download.pytorch.org/whl/cpu` | Torch wheel index. Set a CUDA index on a GPU machine |
| `HF_HOME` | the usual Hugging Face cache | Where `Qwen/Qwen3-0.6B` is stored. Point this at a large disk before the first download |

A CUDA 12.4 index is `https://download.pytorch.org/whl/cu124`. The index has
to match the environment. Installing the CPU index into an environment that
already has a CUDA torch replaces it.

## Tests

```sh
make lab-test
```

The suite covers the arithmetic, the Qwen module map, the command line, and
the shell scripts. It does not import torch and does not download a model.
When torch is installed, the same suite also runs the hook and abliteration
checks on tiny modules.

## Edits

Difference of means is sad text minus happy text. Each class is averaged over
real tokens (padding is masked), one vector per decoder layer. The layer with
the largest L2 norm is the default edit layer.

A forward hook adds `strength * direction` to every token leaving that
decoder layer. `--generated-only` adds it on a decode step (one new token
under a KV cache) and leaves the prompt alone. The hook is removed when
the context exits, including when generation raises. The checkpoint file
is unchanged.

`--pairs` on `means` is the paired form of the same contrast. Each pair is
two strings. The average starts at the first token id where they differ, so
a shared prefix is left out. Without `--pairs`, every unmasked token is
included. The direction is still mean(class_a) - mean(class_b). For the
sad-minus-happy order, class_a is the sad string.

Abliteration normalizes `r` and stores `P @ W` with `P = I - r rᵀ` in
`self_attn.o_proj` and `mlp.down_proj`. Those are the matrices that write the
residual stream. The bias, when the linear has one, is projected the same way.
This changes the loaded module only. Reload the checkpoint to undo it. The
worked direction is sad minus happy.
The same projection is what would be applied to any other direction.

A neuron push adds a scalar to one coordinate of `silu(gate(x)) * up(x)`,
which is the input of `down_proj`. `--generated-only` on `neuron`,
`catalog`, and `sweep` applies that add on decode steps.

An aligned neuron score is the class activation contrast times the dot of
that neuron's `down_proj` column with a residual direction. `top_aligned`
keeps the largest scores. A reply counts as a topic success when it names
the target, still names the topic's last word, and at least half of its
letter-words are distinct.

Column edits on one writing matrix are a separate operation from
abliteration. `remove`, `scale`, and `zero` are documented in
[docs/weight-space.md](docs/weight-space.md). An Ask output row is another
edit: [docs/output-row.md](docs/output-row.md) solves one `output.weight`
row from frozen activations and refuses sealed fresh, operational, and
held-out ids. The parent model is copied, not opened for writing.

```sh
lab/scripts/lab.sh weight --matrix rows.json --op remove --direction 1,0 --columns 0
```

`loops` reads JSONL thinking traces and reports each row whose last
`--window` characters occur `--copies` times or more: the period, the onset
of the periodic run, and how many cycles it holds. `gguf-columns` copies a
GGUF and zeroes or scales named columns of a writing matrix. In Q8_0 a zero
writes quant 0 and a whole-block scale multiplies the block's float16 scale,
so the edit is exact and every other byte is unchanged. The copy is read
back and checked before the command returns. On qwen35 the writers are
`blk.N.ffn_down`, `blk.N.attn_output`, and `blk.N.ssm_out`; their columns
are in the Hugging Face order, one head slice per 256 (attention) or 128
(GatedDeltaNet) columns.

```sh
lab/scripts/lab.sh loops --traces replies.jsonl
lab/scripts/lab.sh gguf-columns --source parent.gguf --output child.gguf --edits edits.json
```

`gguf-tensors` carries an edit made on Hugging Face weights into a GGUF. It
quantizes each new tensor from a `.npy` file and writes it over the old one in
a copy. Given the unedited tensor as `base`, it first quantizes that and
refuses unless the result equals the parent's bytes, so the copy differs from
the parent only where the edit does.

```sh
lab/scripts/lab.sh gguf-tensors --source parent.gguf --output child.gguf --replace replace.json
```

## Neurons from the steering video

The named neurons are on `Qwen/Qwen3.5-4B` (32 text layers, MLP width 9216),
not on Qwen3-0.6B (MLP width 3072). `evil` is layer 23 index 8799. `tomato`
is layer 8 index 4948. `alien` is layer 22 index 2742. The opening of the
captions also places the tomato index at layer 7, and a later line shortens
the evil neuron to index 89. Those two readings are `tomato-layer7` and
`evil-caption`.

Four operations, all temporary. The checkpoint file is not modified.

| Operation | Effect on that neuron |
|---|---|
| `increase` | Add `--amount` to the activation |
| `decrease` | Subtract `--amount` |
| `activate` | Replace the activation with `--amount` |
| `deactivate` | Set the activation to zero |

`--span 1` repeats the same operation on the neighboring indexes. `--op all`
runs all four. `--name all` runs `evil`, `tomato`, and `alien` in one load.

```sh
lab/scripts/fetch-model.sh Qwen/Qwen3.5-4B
lab/scripts/neuron.sh --name evil --op increase --amount 8 --device cpu --dtype bfloat16
lab/scripts/neuron.sh --name all --op all --span 1 --device cpu --dtype bfloat16 --max-new-tokens 16
```

The bf16 checkpoint is about 9 GB. An 8 GB GPU does not hold it. CPU with
bfloat16 does, on a machine with enough RAM. Set `HF_HOME` before
`fetch-model.sh` to choose the disk. The same four operations exist for a
residual direction in `intervene_direction`: increase and decrease add along
the unit direction, activate sets that component, and deactivate removes it.

## Commands

`lab/scripts/lab.sh` with no arguments lists the options. A Kilix 0.2.3
`kilix llm lab COMMAND` runs that command from this checkout. Point
`KILIX_LLM_HOME` at this checkout when it is not beside the Kilix tree and
not under the source home.

The score for every edit is how the answer changes: the continuation text,
lexicon mood, fact-substring hits, and prompt KL. It is not download time
and it is not tokens per second.

| Command | Technique |
|---|---|
| `means` | Difference of means. Sad minus happy, one residual vector per layer, ranked by L2. Each vector is that layer's forward-hook output, so the final RMSNorm is not included. `--keep` sparsifies the chosen vector. `--pairs` reads a JSON list of `[class_a, class_b]` and averages only from the first differing token. |
| `steer` | Hook steering. Adds `strength * direction` on each `--layer` and removes the hooks afterward. Repeat `--layer` to steer more than one layer. `--generated-only` adds the direction on decode steps. The checkpoint file is not rewritten. |
| `abliterate` | Abliteration. Writes `P = I - r r^T` into that layer's `o_proj` and `down_proj`. Mood, fact hits, and KL compare the original answers with the answers after that weight edit. |
| `story` | One-layer hook steering with the same three scores. `--generated-only` adds the direction on decode steps. `--abliterate` writes `P` and scores mood, facts, and KL on the model after that projection. |
| `neuron` | Neuron push. Increase, decrease, activate, or deactivate one MLP neuron, including layer 23 index 8799, layer 8 index 4948, the layer-7 reading of 4948, and layer 22 index 2742. `--quant 4bit --device cuda` loads that 4B checkpoint onto an 8 GB GPU. `--span 1` also edits the neighboring indexes. `--generated-only` edits decode steps. |
| `head` | Attention-head edit. The same four operations on one head slice of the `o_proj` input. `--stand-in` edits a recorded vector and does not load a checkpoint. |
| `sae` | Sparse autoencoder. Trains a ReLU SAE on activation rows and prints the loss before and after the steps. The stand-in batch is four rows of width 3. |
| `circuit` | Circuit trace. Activation-patches each clean site into an otherwise corrupt run and prints the score change. |
| `refusal` | Refusal corpus. Built-in refusal wording minus compliant wording, one residual per layer. `--list` prints the corpus without a checkpoint. `--corpus` reads a JSON object with `refuse` and `comply` lists. `--abliterate` writes `P` for that direction. |
| `sweep` | Neuron sweep. `--start 0 --stop 9216` is every MLP neuron in a Qwen3.5-4B layer, plus matched random directions, on the four explorer prompts. `--shards N --shard K` runs piece K of that range. `--record` writes one JSON object per generation as it finishes. `--stand-in` requires an explicit `--stop` so a test does not run all 9,216. `--generated-only` pushes on decode steps. |
| `sweep-shards` | Launch every shard of a sweep. `--record-dir` receives `shard-K.jsonl`. With a GPU, shard K is placed on device K. |
| `control` | Random-direction control. Pass the neuron's effect and each control effect. The neuron wins only when it beats every control. Directions used as controls are scaled to the same L2 as the push. |
| `rank` | Distinctive-word mass. Reads a sweep or catalog JSONL (`kind`, `index`, `prompt`, `text`) and ranks units by words that are over-represented against the average unit. `null_pct` is the share of random-direction controls with a lower mass. No checkpoint. |
| `catalog` | Bounded catalog. Pushes the indexes and strengths you name, plus matched random directions, on "Who are you?", "Write a snake game in Python.", "Describe the color orange.", and "Write whatever you like." `--stand-in` records those generations without a checkpoint. It does not sweep all 9,216 neurons. `--generated-only` pushes on decode steps. |
| `weight` | Column edit. Remove a direction from chosen columns, or scale or zero those columns. `--matrix` edits JSON and does not load a checkpoint. Embeddings, `lm_head`, and a final norm are refused. |
| `output-row` | Squared-hinge edit of one output row from frozen activations. Sealed Ask ids are refused. `--parent-gguf` is read only and `--output` must be a new file. |
| `train` | Prepare a corpus you supply and, with `--run`, execute vendored `llama-finetune` on an FP32 GGUF. Without `--run` it writes the text file and the argument list. See [docs/train.md](docs/train.md). |
| `bench` | Greedy answers before and after one weight edit. |

From the repository root, after `lab/scripts/uv-venv.sh`:

```sh
lab/.venv/bin/python lab/projects/qwen3_0_6b/run.py means
lab/.venv/bin/python lab/projects/qwen3_0_6b/run.py story --strength 1
lab/.venv/bin/python lab/projects/qwen3_0_6b/run.py neurons --layer 7 --top 8
lab/scripts/bench-qwen3-0.6b.sh --device cpu --new-tokens 16
```

Pass `--device cuda` on a machine where the torch install can see a GPU.
The bench compares answers before and after the weight edit. It does not
report download time or tokens per second. The project runbook is
[projects/qwen3_0_6b/README.md](projects/qwen3_0_6b/README.md).

## What the scores mean

Mood is a word list in `measure.py`, not a trained emotion classifier. `+1`
is only happy hits, `-1` is only sad hits, and `0` is neither or a tie. The
built-in stories use those words on purpose so the two classes separate
before any model is loaded. A larger corpus moves the steering direction.

Fact hits look for a substring such as `tokyo` in a greedy continuation.
KL is the mean, over prompt positions, of KL(original || steered) in nats.
Zero means the next-token distribution did not move.

Greedy decoding is fixed so two runs of the same weights match.
