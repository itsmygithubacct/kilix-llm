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

`build` is optional: the first `run`, `chat` or `serve` builds the selected
runtime when it is not built for the current pin, prints why, and sends the
compile log to stderr. Later starts check a stamp in `bin/` and build nothing.
A pin change rebuilds once. `KILIX_LLM_NO_AUTO_BUILD=1` turns this off, and a
missing runtime is then an error that names the build command.

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

`qwen-0.8b-no-loop-02` is Qwen3.5-0.8B Q8_0 (833,592,416 bytes) with one
third of a published weight edit applied, to reduce thinking-mode loops. The
edit removes a direction from 64 `ffn_down` columns in each of layers 15 and
19. It is Apache-2.0, modified from Qwen/Qwen3.5-0.8B. The edit is adapted
from CC BY 4.0 material; the release's model card gives the attribution and
all results. It runs with the same GPU profile as the 4B. With thinking on
at temperature 0.7 / top_p 0.8, on fresh seeds, loops fell from 23 of 80
replies to 10 (paired p = 0.004). GSM8K accuracy stayed level. At the Qwen
card's 1.0 / 0.95 it made no difference, so `--think` keeps 0.7 / 0.8 for this
entry. With thinking off neither model looped.

Thinking mode is not recommended for this model, and nothing here turns it on
by default. On 300 held-out GSM8K test problems (indices 400-699, measured
2026-11-12), the stock Qwen3.5-0.8B it is edited from solved 56 with thinking
on at 0.7 / 0.8 and an 8,192-token cap, and 142 with thinking off; 239 of the
300 thinking replies looped or ran to the length limit. `--think` still works
for `run`, `chat` and `serve`, and prints that warning first.

`kilix-llm pull` refuses this id. The model wizard installs the GGUF and its
licence notices (`kilix models install qwen-0.8b-no-loop-02`).

```sh
./kilix-llm serve --model-id qwen-0.8b-no-loop-02 --device Vulkan1
./kilix-llm chat --model-id qwen-0.8b-no-loop-02 --gpu-layers 0
```

`qwen3.5-0.8b` is the stock Qwen3.5-0.8B Q8_0 (ggml-org, 833,592,096 bytes),
pinned by revision and sha256. A catalogue entry that names a `base` is a LoRA
adapter: `run`, `chat` and `serve` load the base with `-m` and the adapter with
`--lora`. A missing or different-length base is refused at launch, and `verify`
checks both sha256 digests. `--base PATH` names a base that lives outside
kilix-llm's model directory (the model wizard installs each asset into its own
directory); it is refused for a full model.

`lfm2-1.2b-tool` is LiquidAI's own Q8_0 of LFM2-1.2B-Tool (1,246,252,768 bytes,
pinned by revision and sha256), the base for LFM2 tunings. It is under the LFM
Open License v1.0, which limits commercial use. Its licence is delivered as a
notice, Debian style: `pull` (and `judge --yes`) downloads the licence text with
the weights and saves it beside them as `LFM2-1.2B-Tool-Q8_0.LICENSE`. No
agreement is asked for or recorded.

Tunings are catalogue entries, not code: each names the base it loads over,
the task command that runs it (`judge` for now) and its release asset. An entry
without a sha256 is catalogued ahead of its release and is refused.
`qwen-0.8b-binary-01` is such an entry until its adapter passes its ship bar.

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

## Judge: one yes/no answer as an exit code

`kilix-llm judge` answers a yes/no question, or a pass/fail judgement of
information you show it, with one character, and exits with it:

| stdout / exit | meaning |
|---|---|
| `0` | yes, pass, ok, correct, valid |
| `1` | no, fail, error, incorrect, invalid |
| nothing / `2` | usage, model, download or digest problem, or the sizer gave no limit |
| nothing / `3` | the input is longer than this machine's limit (stderr says the limit) |
| nothing / `4` | the runtime is missing or cannot be built |
| nothing / `5` | the engine failed or answered outside the contract |

A question is answered as asked: "Did the build fail?" is `0` when it did.
Phrase a verdict ("Did the build pass?") when you want pass/fail polarity.
Put information to judge between `<information>` and `</information>`; text
inside it is data, not instructions.

```sh
if ./kilix-llm judge --model-id qwen3.5-0.8b "Is 2+2 equal to 4?"; then echo yes; fi
make test 2>&1 | tail -40 | { printf '<information>\n'; cat; printf '</information>\n\nDid the tests pass?'; } \
  | ./kilix-llm judge --model-id qwen3.5-0.8b --stdin --json
```

The answer is one token at temperature 0 under the grammar
`root ::= "0" | "1"`, with thinking off. `--json` prints the answer with `p0`
and `p1`, the model's two raw logprobs renormalised, so a caller can set its
own threshold; `probs_bounded` is true when the other digit was not among the
listed candidates and its probability is an upper bound.

Before answering, `judge` asks `plebian-model-sizer recommend binary-judge`
(found with `--sizer`, `KILIX_LLM_SIZER`, or on PATH) how long the input may
be on this machine, from live free memory and a measured profile for this base
and this llama.cpp pin. A longer input exits 3. `--truncate head` or
`--truncate tail` keeps the input's first or last tokens instead, and says so
on stderr. Nothing is ever cut silently.

The default model is `qwen-0.8b-binary-01`, which is not released yet. Until
it is, `--model-id qwen3.5-0.8b` runs the stock base with the same contract
prompt (`lfm2-1.2b-tool` also works). If the base or adapter is missing,
`judge` lists what it would download (size, source, licence notice) and
downloads only with `--yes` or an interactive yes. `--base PATH` (and
`--model PATH`) name weights you already have; they are used as they are,
after the size and sha256 checks, and are never downloaded.

## Stack: several drafts at once and one judge

A small model writes better when it writes several replies and a second model
picks one. `kilix-llm stack serve` starts the workers (one or more models,
each with its own request slots) and a judge on one GPU, and serves
`POST /v1/chat/completions` on loopback (port 8093, or `--socket PATH`):

```sh
./kilix-llm stack choices                                  # installed models and agents it can use
./kilix-llm stack fit   --worker qwen3.5-0.8b --judge d3-2b    # how many slots fit beside the judge
./kilix-llm stack serve --worker qwen3.5-0.8b:5 --judge d3-2b  # five drafts at once, d3 picks
./kilix-llm stack serve --worker qwen3.5-0.8b:4 --worker claude --drafts 9 --judge qwen3.5-4b
```

`--worker MODEL:N` gives a model N slots and may be repeated. A model is a
catalogue id, a GGUF path (used as it is, with no size or digest check), or
`codex` or `claude` (`codex/MODEL`, `claude/MODEL` to name their model).
`--drafts` is how many drafts each request gets; the default is one for each
slot. Draft 0 is written at temperature 0 by the first worker, the others at
`--temperature` (0.8) with fixed seeds, and drafts go round the slots, as
many at once as there are slots. The judge rates every distinct draft from 0
to 1 and the highest-rated one is the reply, tool calls included. A tie keeps
the earlier draft. If the drafts are identical the judge is not asked, and if
it gives no rating the temperature-0 draft is the reply. A streamed request
gets the chosen reply as one event stream after it is whole. Nothing is
loaded or unloaded between requests, and one request runs at a time.

`codex` and `claude` are run once for each draft, in an empty directory, with
the conversation and the tool list as their input and their own tools off
(`codex exec --sandbox read-only`, `claude -p --tools ""`). They answer with
the text and tool calls of the next message; they do not act. As the judge
they answer with a number from 0 to 100. Each run is a request to that
service on your account. `KILIX_LLM_STACK_CODEX` and `KILIX_LLM_STACK_CLAUDE`
replace the command.

The reply carries `kilix_stack`: each draft's worker and rating, the chosen
draft, and the seconds spent drafting and judging. `--journal FILE` appends
the same with the texts, one JSON line a request. `GET /stack/status` gives
the totals and each engine's RAM and (through `nvidia-smi`) GPU memory.

A local judge is one of two kinds, which a catalogue id knows for itself:

| kind | the judge | the rating |
|---|---|---|
| `chat` | any chat model, asked the `judge` contract above | `p0`, the probability of "0" (yes) against "1" (no) |
| `systemone` | a d3 decision model (`d3-2b`, `d3-4b`, `d3-9b`, `d3-27b`), asked through `/v1/systemone` | its `noul` answer |

For a GGUF path, `--judge-kind` says which. `--judge-url` asks a running
service for `/v1/systemone` in place of loading the judge here, e.g. Ollama
at `http://127.0.0.1:11434` with `--judge d3:2b`. `--server` and
`--judge-server` run another `llama-server` in place of the vendored build.

The judge sees the last four messages of the request and the draft, cut to
two characters for each token of `--judge-ctx` (4096), and is asked
`--question` ("Is this reply a correct and complete next step for the
request?").

`stack fit` starts the stack with one slot of the first local worker, then
two, and so on, beside the other workers and the judge, and reports the last
count at which every slot answered a prompt while the judge read a
full-length input. It tests under load on purpose: a stack that only loads
can run out of GPU memory on its first real request. `--kv q8_0` (the
default) halves the memory of each slot's context cache.

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
| `kilix-llm build [--backend vulkan] [--if-needed] [--keep-build-tree]` | compile the CPU or Vulkan runtime (`--if-needed`: only when not built for this pin) |
| `kilix-llm devices` | list GPU names for `--device` |
| `kilix-llm pull [--model-id ID]` | download a catalogued GGUF (resumable): `qwen3.8-27b`, `qwen3.5-9b`, `qwen3.5-4b`, `qwen3.5-4b-q4km`, `qwen3.5-2b`, `qwen3.5-0.8b`, the d3 decision models `d3-2b`, `d3-4b`, `d3-9b` and `d3-27b`, and `lfm2-1.2b-tool` (each with its licence text where the entry ships one); the wizard-installed `qwen-0.8b-no-loop-02` and unreleased tunings are refused |
| `kilix-llm verify` | size + sha256 of the weights |
| `kilix-llm path` | where the weights resolve |
| `kilix-llm run PROMPT` | one shot via `llama-cli` |
| `kilix-llm chat [--think]` | interactive conversation via `llama-cli` (`/clear`, `/regen`, `/exit`) |
| `kilix-llm serve` | `llama-server` on loopback |
| `kilix-llm judge [--model-id ID] [--base PATH] [--stdin \| TEXT] [--json] [--truncate head\|tail] [--yes]` | one yes/no or pass/fail answer: prints `0` or `1` and exits with it (see Judge) |
| `kilix-llm stack serve [--worker MODEL[:N]]... [--drafts N] [--ctx N] [--judge MODEL] [--socket PATH] [--journal FILE]` | drafts from one or more workers and one judge behind one OpenAI-style endpoint |
| `kilix-llm stack fit [--worker MODEL]... [--judge MODEL] [--ctx N] [--max-slots N] [--json]` | how many slots load and answer beside the judge on this machine |
| `kilix-llm stack choices [--json]` | the installed models and agents a stack can use |
| `kilix-llm vm setup/start/pi/shell/status/stop` | persistent QEMU guest, Pi sessions, and host GPU server |
| `kilix-llm ask-help` | experimental document help through explicit local answer/ranking runs |

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
(override `KILIX_LLM_BUILD_DIR`). A finished build keeps only `bin/` (the
binaries, their shared libraries and the stamp, about 25 MB). It removes the
cmake tree and the patched source copy, about 0.7 GB. Pass `--keep-build-tree`
for fast incremental rebuilds while changing llama.cpp. A `KILIX_LLM_BUILD_DIR`
you chose is never pruned. Builds of one backend take a lock, so concurrent
first uses compile once.

CMake uses Release, native CPU, CUDA/Metal off, curl off, the prebuilt web UI
off, and Vulkan on only when `--backend vulkan` is selected. LLM Chat and
Avatar do not use the UI, and the build does not fetch it.
Model download is `kilix-llm pull`, not the runtime.

## Tests

```sh
make check
```

The suite does not compile llama.cpp and does not download the GGUF.

## Lab

[lab/](lab/README.md) is weight editing and measurement: residual directions, forward-hook steering, abliteration of the matrices that write the residual stream, [column-scoped weight-space edits](lab/docs/weight-space.md), and MLP neuron reads. It is not part of `make all` or `make test`. `make lab-test` runs that suite without a model download. `lab/scripts/lab.sh` with no arguments lists every edit command. On Kilix 0.2.3, `kilix llm lab` is that entry. The score is how the answer changes, not download time or tokens per second.

Model commands use a uv virtual environment from [lab/scripts/uv-venv.sh](lab/scripts/uv-venv.sh). The first project, [lab/projects/qwen3_0_6b](lab/projects/qwen3_0_6b/README.md), reruns those methods on Qwen3-0.6B. [lab/scripts/bench-qwen3-0.6b.sh](lab/scripts/bench-qwen3-0.6b.sh) compares answers before and after one weight edit.

## Licence

MIT for this repository — see [LICENSE](LICENSE). llama.cpp is MIT of its own;
see [THIRD-PARTY.md](THIRD-PARTY.md) and `third_party/llama.cpp/LICENSE`.
