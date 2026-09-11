# kilix-llm

Run **Qwen3.8 27B** from system RAM on a machine with no usable VRAM.
The inference engine is [llama.cpp](https://github.com/ggml-org/llama.cpp),
vendored at `third_party/llama.cpp` so this repository can patch it.

The catalogued checkpoint is Unsloth's `UD-Q4_K_M` GGUF (~16.5 GB). The build
is CPU-only (`-ngl 0`, CUDA and Vulkan off). A 64 GB box holds it; a 6-core
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

## Commands

| Command | What it does |
|---|---|
| `kilix-llm doctor` | RAM, cores, pin, vendor tree, binaries, weights |
| `kilix-llm vendor` | print the llama.cpp pin |
| `kilix-llm build` | compile the vendored tree, CPU only |
| `kilix-llm pull` | download the catalogued Q4 GGUF (resumable) |
| `kilix-llm verify` | size + sha256 of the weights |
| `kilix-llm path` | where the weights resolve |
| `kilix-llm run PROMPT` | one shot via `llama-cli` |
| `kilix-llm serve` | `llama-server` on loopback |

`--gpu-layers` exists and defaults to **0**. Thinking mode is off unless
`--think` is passed: at a few tokens per second a reasoning preamble would
hide the first visible word.

Weights live under `$GPU_TERMINAL_HOME/kilix-llm/models/`
(default `~/.local/gpu_terminal/kilix-llm/models/`). Override with
`KILIX_LLM_MODELS_DIR` or `--model /path/to.gguf`. This repository never
ships weights.

## Vendoring llama.cpp

The source is a git submodule at a pinned commit, recorded in
`third_party/llama.cpp.pin`. Clone with `--recurse-submodules`, or:

```sh
git submodule update --init third_party/llama.cpp
```

To move the pin:

```sh
./scripts/vendor-sync.sh <40-hex-sha>
```

To change llama.cpp itself, edit files under `third_party/llama.cpp/`. For a
change that should survive the next sync, drop a git patch in `patches/` and
list it, with sha256, in the pin file. `scripts/build-runtime.sh` copies the
vendor tree, applies those patches, and compiles into `build/llama.cpp`
(override `KILIX_LLM_BUILD_DIR`).

CMake flags are fixed: Release, native CPU, CUDA/Vulkan/Metal off, curl off.
Model download is `kilix-llm pull`, not the runtime.

## Tests

```sh
make check
```

The suite does not compile llama.cpp and does not download the GGUF.

## Licence

MIT for this repository — see [LICENSE](LICENSE). llama.cpp is MIT of its own;
see [THIRD-PARTY.md](THIRD-PARTY.md) and `third_party/llama.cpp/LICENSE`.
