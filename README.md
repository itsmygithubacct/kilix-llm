# kilix-llm

Run **Qwen3.8 27B Q4** from system RAM or **Qwen3.5 4B Q6** on a GPU.
The inference engine is [llama.cpp](https://github.com/ggml-org/llama.cpp),
vendored at `third_party/llama.cpp` so this repository can patch it.

**Status:** a standalone tool, version 0.1.0; Plebian-OS and the Kilix
content catalog do not install it. Kilix 0.2.3's `kilix llm lab` runs this
repository's weight-edit lab from a checkout you provide (`KILIX_LLM_HOME`, a
`kilix-llm` directory beside Kilix, or `$GPU_TERMINAL_SOURCE_HOME/kilix-llm`).

## What it does

- Builds the vendored llama.cpp for CPU or Vulkan and runs one-shot prompts or
  a chat-completions HTTP server on loopback (`kilix-llm serve`).
- Downloads catalogued GGUF weights with resume, size and SHA-256 checks; never
  ships weights.
- Runs a persistent QEMU guest for a terminal coding agent (`kilix-llm vm pi`),
  served by the host GPU model ([docs/pi-vm.md](docs/pi-vm.md)).
- Answers Kilix documentation questions through a prepared kilix-help-llm
  checkout (`ask-help`, experimental).
- Carries a separate weight-edit and measurement lab under [lab/](lab/README.md).

## Quick start

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and make
sure `uv` is on `PATH`. The launcher uses uv-managed Python 3.12, downloaded
on first use; it needs neither system Python nor a project `.venv`.

The default checkpoint is Unsloth's 27B `UD-Q4_K_M` GGUF (~16.5 GB), with
CPU inference (`-ngl 0`). A 64 GB box holds it; a 6-core
laptop will generate on the order of one to a few tokens per second.

```sh
git clone --recurse-submodules https://github.com/itsmygithubacct/kilix-llm.git
cd kilix-llm
./kilix-llm doctor
./kilix-llm build
./kilix-llm pull
./kilix-llm run 'Say hello in one sentence.'
./kilix-llm serve          # OpenAI-compatible, 127.0.0.1:8080
```

`./install.sh` puts a launcher on `PATH` that execs this checkout.

## Usage

### Commands

| Command | What it does |
|---|---|
| `kilix-llm doctor` | RAM, cores, pin, vendor tree, binaries, weights |
| `kilix-llm vendor` | print the llama.cpp pin |
| `kilix-llm build [--backend vulkan]` | compile the CPU or Vulkan runtime |
| `kilix-llm devices` | list GPU names for `--device` |
| `kilix-llm pull [--model-id qwen3.5-4b]` | download a catalogued GGUF (resumable) |
| `kilix-llm verify` | size + sha256 of the weights |
| `kilix-llm path` | where the weights resolve |
| `kilix-llm run PROMPT` | one shot via `llama-cli` |
| `kilix-llm serve` | `llama-server` on loopback |
| `kilix-llm vm setup/start/pi/shell/status/stop` | persistent QEMU guest, Pi sessions, and host GPU server |
| `kilix-llm ask-help` | experimental document help through explicit local answer/ranking runs |

`--gpu-layers` defaults to **0 for the 27B model** and **99 for the 4B model**.
For `run`, thinking mode is off unless `--think` is passed: at a few tokens
per second a reasoning preamble would
hide the first visible word.

Weights live under `$GPU_TERMINAL_HOME/kilix-llm/models/`
(default `~/.local/gpu_terminal/kilix-llm/models/`). Override with
`KILIX_LLM_MODELS_DIR` or `--model /path/to.gguf`. An explicit path is still
validated against the selected `--model-id`; use the ID matching that file.
This repository never ships weights.

`ask-help` options and the calibrated help path are described in
[docs/help-module.md](docs/help-module.md); the 4B GPU profile and its
measured results in [docs/gpu-model.md](docs/gpu-model.md).

## How it works

```
kilix-llm (uv-managed Python 3.12 launcher, src/kilix_llm/)
   |-- build:  scripts/build-runtime.sh copies third_party/llama.cpp,
   |           applies patches/ listed in llama.cpp.pin, runs CMake
   |           -> build/llama.cpp or build/llama.cpp-vulkan
   |-- pull:   catalogued GGUF -> $GPU_TERMINAL_HOME/kilix-llm/models/
   |-- run:    llama-cli     serve: llama-server on 127.0.0.1
   `-- vm:     QEMU guest, 9p /persist mount, host loopback API
```

The launcher is a single Python package with no dependencies beyond the
standard library; uv supplies the interpreter. llama.cpp is a git submodule at
the commit in `third_party/llama.cpp.pin`, so local patches are explicit
files with recorded digests rather than edits to an unpinned tree. The
model catalog in `src/kilix_llm/models.py` fixes each model's file size,
digest and default GPU layers. See [docs/vendoring.md](docs/vendoring.md).

## Tests

```sh
make check
```

The suite does not compile llama.cpp and does not download the GGUF.

## Lab

[lab/](lab/README.md) is weight editing and measurement: residual directions, forward-hook steering, abliteration of the matrices that write the residual stream, [column-scoped weight-space edits](lab/docs/weight-space.md), and MLP neuron reads. It is not part of `make all` or `make test`. `make lab-test` runs that suite without a model download. `lab/scripts/lab.sh` with no arguments lists every edit command. On Kilix 0.2.3, `kilix llm lab` is that entry. The score is how the answer changes, not download time or tokens per second.

Model commands use a uv virtual environment from [lab/scripts/uv-venv.sh](lab/scripts/uv-venv.sh). The first project, [lab/projects/qwen3_0_6b](lab/projects/qwen3_0_6b/README.md), reruns those methods on Qwen3-0.6B. [lab/scripts/bench-qwen3-0.6b.sh](lab/scripts/bench-qwen3-0.6b.sh) compares answers before and after one weight edit.

## Documentation

- [docs/gpu-model.md](docs/gpu-model.md) — the 4B Vulkan profile, build dependencies and validation results
- [docs/pi-vm.md](docs/pi-vm.md) — the persistent QEMU guest, shared folders and agent sessions
- [docs/help-module.md](docs/help-module.md) — `ask-help` with kilix-help-llm runs and the calibrated path
- [docs/vendoring.md](docs/vendoring.md) — moving the llama.cpp pin and carrying patches
- [docs/checkpoint-training-api.md](docs/checkpoint-training-api.md) — explicit checkpoint and supervised-target APIs
- [ARTIFACTS.md](ARTIFACTS.md) — manifests for additional GGUF artifacts
- [lab/README.md](lab/README.md) — the weight-edit and measurement lab
- [THIRD-PARTY.md](THIRD-PARTY.md) — upstream notices

## Related repositories

- [kilix-help-llm](https://github.com/itsmygithubacct/kilix-help-llm#readme) — the help module `ask-help` delegates to
- [kilix](https://github.com/itsmygithubacct/kilix#readme) — `kilix llm lab` runs this repository's lab from a checkout
- [kilix-bonsai](https://github.com/itsmygithubacct/kilix-bonsai#readme) — a separate CPU runtime for the 1-bit Bonsai models, also on pinned llama.cpp

## Part of Plebian-OS

kilix-llm is one of the repositories published alongside
[Plebian-OS](https://github.com/itsmygithubacct/plebian-os#readme), a Debian
install whose desktop is [Kilix](https://github.com/itsmygithubacct/kilix#readme)
running the [Kilix 95](https://github.com/itsmygithubacct/kilix-95#readme)
desktop through [Pleb](https://github.com/itsmygithubacct/pleb#readme).
It is not installed by Plebian-OS; you clone it, and Kilix's `kilix llm lab`
uses that checkout. The map of every repository and how it ships is on the
[components page](https://github.com/itsmygithubacct/plebian-os/blob/main/docs/components.md).

## Licence

MIT for this repository — see [LICENSE](LICENSE). llama.cpp is MIT of its own;
see [THIRD-PARTY.md](THIRD-PARTY.md) and `third_party/llama.cpp/LICENSE`.
