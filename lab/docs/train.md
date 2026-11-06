# Corpus training

`train` prepares a corpus you supply and can run the vendored
`llama-finetune` program (`third_party/llama.cpp/examples/training`).
It does not download a dataset or a model. The other lab commands do not
train the language model. `sae` fits a sparse autoencoder on activation
rows. `output-row` solves one frozen output row.

`kilix_llm.supervised` is a separate assistant-only loss (`encode_target`,
`target_loss`). It masks the prompt and scores the authored answer. It does
not run an optimizer or write a model. `kilix_llm.checkpoints` only fetches
a manifest of checkpoint files. See
[checkpoint-training-api.md](../../docs/checkpoint-training-api.md).

`llama-finetune` updates every parameter of an FP32 GGUF and writes a new
GGUF. The upstream example uses context, batch, and micro-batch 512, two
epochs, and every GPU layer. Flash attention stays off, because that kernel
has no backward pass. An FP32 model near 1B parameters wants about 24 GB.
A quantized GGUF is refused. The trained file is a weight file and stays
out of git.

Omitting `--run` only writes the text file, a sidecar manifest, and the
argument list. `--run` checks the model, then replaces this process with
the trainer. The command does not call `uv-venv.sh`: that script installs
a torch wheel, and its default index is CPU, which replaces a CUDA torch.
The trainer binary is `build/bin/llama-finetune` after a llama.cpp build,
or `KILIX_LLM_LLAMA_FINETUNE`, or `--binary`.

CPU training is a llama.cpp build without an extra backend. CUDA training
uses the GPU layers (`--gpu-layers`, default 999).

## Corpus

`--corpus` is a file or a directory.

| Input | What becomes one document |
|---|---|
| `.txt`, `.md`, `.text`, `.raw`, or any single file that is not `.jsonl` | The whole file |
| `.jsonl` | One object per line |

A JSON object uses the first matching field: `text`, `content`, `messages`
(each `content` string), `prompt` plus `completion`, `input` plus `output`,
or `instruction`, `user`, and `assistant`. Other files in a directory are
counted in the manifest and left out. Hidden files and directory symlinks
are left out. The joined text is capped at 2 GiB.

```sh
lab/scripts/lab.sh train \
  --corpus corpus-dir \
  --text-out work/corpus.txt \
  --model model-f32.gguf \
  --output work/finetuned-model.gguf
```

Add `--run` to train. `--output` must be a new `.gguf` file. The default
name is `finetuned-model.gguf` beside the text file.
