# kilix-llm

Run **Qwen3.8 27B Q4** from system RAM or **Qwen3.5 4B Q6** on a GPU.
The inference engine is [llama.cpp](https://github.com/ggml-org/llama.cpp),
vendored at `third_party/llama.cpp` so this repository can patch it.

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

## GPU model

[Qwen3.5-4B Q6_K](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF)
is the general chat and coding option for a 6 GiB GPU. The weight file is
3,525,956,768 bytes (3.28 GiB), pinned by revision and SHA-256. Start with
8,192 context tokens and thinking off. Allow additional VRAM for the cache,
compute buffers, and desktop; file size alone does not establish runtime fit.

```sh
./kilix-llm build --backend vulkan
./kilix-llm devices
./kilix-llm pull --model-id qwen3.5-4b
# Replace Vulkan1 with the discrete GPU name printed by devices.
./kilix-llm run --model-id qwen3.5-4b --device Vulkan1 'Say hello.'
./kilix-llm serve --model-id qwen3.5-4b --device Vulkan1
```

Vulkan build dependencies on Debian/Ubuntu are `libvulkan-dev`, `glslc`, and
`spirv-headers`, in addition to CMake, a C++ compiler, and rsync. Vulkan can
use Mesa NVK on supported NVIDIA hardware without installing a CUDA driver.

The 4B profile selects the Vulkan build, requests all layers and the input
embedding on one GPU, disables automatic fitting, enables Flash Attention and Q8 KV caches, and
serves one request slot. If multiple GPUs are available, `--device` is required.
A missing GPU fails before inference instead of continuing on CPU. Use
`--gpu-layers 0` explicitly to run this model through the CPU build.
`--think` enables thinking in either `run` or `serve` for the 4B profile.

CPU and Vulkan builds live separately in `build/llama.cpp` and
`build/llama.cpp-vulkan`. `KILIX_LLM_BUILD_DIR` overrides the selected build's
directory; it must contain a runtime with the requested backend.

Validated on 2026-09-12 with a Quadro RTX 3000 Mobile (6 GiB), Mesa NVK
25.0.7, and the pinned runtime:

| Check | Result |
|---|---|
| GPU allocation at 8,192 context tokens | 3,615.68 MiB total: weights 3,352.15, KV 136.00, recurrent state 50.25, compute 77.28 |
| Layer and embedding placement | 33/33 layers and input embedding on Vulkan1; no CPU model/cache allocation |
| CLI and local API | short reply, arithmetic, and valid Python output passed; one API slot, thinking off |
| Prompt processing, 256 tokens | 70.7 tokens/s, mean of three repetitions |
| Generation, 128 tokens | 18.7 tokens/s, mean of three repetitions |

These speed measurements use short contexts after warmup. The current NVK
backend reports no FP16 or matrix-core acceleration. Host tokenization,
sampling, and approximately 14 MiB of host output/compute buffers still use
the CPU/RAM; the model weights and caches remain on the GPU.

## Pi in a persistent QEMU VM

Run [Pi](https://pi.dev/) and its coding tools inside a Debian 13 VM while
Qwen3.5 4B inference uses the host GPU. The VM mounts a persistent host folder
for projects, Pi sessions and settings, and uv-managed Python. The guest has
network access; the model API uses an authenticated host loopback endpoint
reached through QEMU's `10.0.2.2` host address.

This option requires an x86-64 Linux host with read/write access to `/dev/kvm`.
On Debian/Ubuntu, its host packages are `qemu-system-x86`, `qemu-utils`,
`xorriso`, `openssh-client`, and `curl`. Run the launcher as your regular
user. After building the Vulkan runtime and downloading the 4B model above:

```sh
./kilix-llm vm setup                    # optional: provision the guest first
./kilix-llm vm start --device Vulkan1    # also provisions it when needed
./kilix-llm vm pi --new-tab             # open Pi in a new Kilix tab
./kilix-llm vm pi                       # or use this terminal
./kilix-llm vm pi -- --continue         # continue the last Pi session
./kilix-llm vm shell                    # guest login shell
./kilix-llm vm shell -- uv run --managed-python --python 3.12 python --version
./kilix-llm vm status
./kilix-llm vm stop                     # shut down; retain disk and shared files
```

First boot downloads the image and installs guest tools, which takes several
minutes. The guest uses four virtual CPUs, 4 GiB RAM, and a 24 GiB QCOW2 disk
that grows as used. The selected GPU is remembered for later starts. Exit any
standalone session using this model before starting its VM server: two copies
exceed a 6 GiB GPU's capacity.

VM state defaults to `~/.local/gpu_terminal/kilix-llm/vm/`, follows
`$GPU_TERMINAL_HOME/kilix-llm/vm/`, and can be overridden with
`KILIX_LLM_VM_DIR`. Its `shared/` directory is mounted at `/persist` using 9p:

| Host path relative to the VM directory | Guest path | Contents |
|---|---|---|
| `shared/workspace/` | `/persist/workspace/` | Projects; Pi starts here |
| `shared/pi/` | `/persist/pi/` | Pi sessions, settings, model configuration, API key |
| `shared/cache/uv/` | `/persist/cache/uv/` | uv download and package cache |
| `shared/python/` | `/persist/python/` | uv-managed Python installations |

Files changed under `/persist` are immediately visible in the host folder
and survive guest shutdown. The guest user's home, `/var/lib/kilix`, and
installed OS packages persist on `disk.qcow2`. Only `shared/` is exported;
the host's model weights, VM control files, and SSH private key remain outside
that mount. Pi sessions can be resumed with `vm pi -- -c` or browsed with
`vm pi -- --resume`.

Both the launcher and guest Python workflows use uv. Guest shell profiles
select managed Python and persistent caches; run Python with
`uv run --managed-python --python 3.12 python ...`. Pi itself uses Node and
is installed through npm with `--ignore-scripts`. Provisioning pins the
[Debian 13 image](https://cloud.debian.org/images/cloud/trixie/20260831-2587/),
[Node 22.23.2](https://nodejs.org/dist/v22.23.2/),
[uv 0.12.13](https://github.com/astral-sh/uv/releases/tag/0.12.13), and Pi 0.85.1;
image and binary archive downloads are checksum-verified.

The Pi profile uses 8,192 context tokens, a 2,048-token response limit,
thinking off, and compaction settings sized for that context.
`vm start --ctx N` can select a larger context after `vm stop`, subject to
available GPU memory. For startup details, inspect `qemu.log` and `server.log`
in the VM directory, or guest provisioning output with
`./kilix-llm vm shell -- sudo tail -80 /var/log/cloud-init-output.log`.

Validated on 2026-09-13: Pi read a shared file, ran uv-managed Python to write
the correct result, and read it back. Workspace files and the saved Pi session
remained byte-for-byte identical after VM shutdown and restart. This run
measured about 0.7 generated tokens/s in both Pi and the direct host API;
the cause of the slowdown from the earlier short benchmark is unresolved.

## Commands

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
| `kilix-llm ask-help` | local answers from the default Kilix Qwen 2B Help model |

For 0.2.3 RC1, run `kilix help-setup` once, then
`kilix-llm ask-help 'How do I list panes?'`. The default is
`kilix-qwen-sft1-2b`, using its pinned Qwen3.5-2B parent and unmerged adapter in
FP32 on the CPU. JSON includes a command, explanation and related help links
under `presentation`. The module can be located through installed Kilix,
`--help-module PATH`, or `KILIX_HELP_MODULE`. It never executes the suggested
command. Model installation is explicit; missing files produce setup guidance.

This selection targets everyday Kilix help. Development scores are 30/32
supported answers and 8/8 unknown-state refusals, with 41/72 on the older
retention set. These are exposed results, not independent qualification.

For a prepared `kilix-help-llm` checkout and its local trained runs:

```sh
kilix-llm ask-help --help-module ../kilix-modules/kilix-help-llm \
  --answer-run answer-r1 --rank-run rank-r1 'How do I list panes?'
```

This command delegates to the help module's CPU adapter runtime and returns
its JSON unchanged, including source revision, corpus digest, citations,
refusals and qualification status. The module verifies artifacts and matches
the two runs to the same corpus/context, sizes each load through the shared
model sizer, and releases the ranking model before loading the answer model.
Its custom scoring head remains separate from prose generation. User data
uses `GPU_TERMINAL_HOME` (default `~/.local/gpu_terminal`). The optional
`--sizer` selects the shared provider explicitly.

The current help path is experimental and unqualified. This is a local
one-shot interface using the help runtime; it does not export adapters to
GGUF or add them to the llama-server catalog. Qualification requires measured
answer quality and verification of the actual serving configuration.

Once a frozen calibration threshold exists, add `--calibration-policy PATH`
and `--docs-root KILIX_CHECKOUT` to `ask-help`. The host verifies the policy,
run reports, corpus revision, sequential ranking result, and the selected
paragraph against the pinned Git document. It delivers a cited model answer
only when its top ranking score meets the inclusive threshold. It preserves an
explicit refusal and otherwise returns the fixed static-document fallback.
The untouched model
result is saved under private `kilix-help-llm/diagnostics` in
`GPU_TERMINAL_HOME`; rejected drafts are omitted from the user-visible JSON.
The calibrated path remains unqualified until held-out quality and actual host
parity gates pass. Its generation limit is fixed at 96 tokens.

`--gpu-layers` defaults to **0 for the 27B model** and **99 for the 4B model**.
For `run`, thinking mode is off unless `--think` is passed: at a few tokens
per second a reasoning preamble would
hide the first visible word.

Weights live under `$GPU_TERMINAL_HOME/kilix-llm/models/`
(default `~/.local/gpu_terminal/kilix-llm/models/`). Override with
`KILIX_LLM_MODELS_DIR` or `--model /path/to.gguf`. An explicit path is still
validated against the selected `--model-id`; use the ID matching that file.
This repository never ships weights.

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

CMake uses Release, native CPU, CUDA/Metal off, curl off, and Vulkan on only
when `--backend vulkan` is selected.
Model download is `kilix-llm pull`, not the runtime.

## Tests

```sh
make check
```

The suite does not compile llama.cpp and does not download the GGUF.

## Licence

MIT for this repository — see [LICENSE](LICENSE). llama.cpp is MIT of its own;
see [THIRD-PARTY.md](THIRD-PARTY.md) and `third_party/llama.cpp/LICENSE`.
