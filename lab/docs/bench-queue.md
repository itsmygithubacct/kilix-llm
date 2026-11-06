# Bench queue

`kilix llm lab queue` sends a kilix-benchmark suite to remembered machines and
splits the items by measured run time.

Remembered aliases are stored in `~/.local/state/kilix-llm/bench-connections.json`.
Each machine, including this one, keeps a uv virtual environment, the engine,
and the downloaded models in `~/.local/share/kilix-llm/bench`.

```sh
lab/scripts/bench-queue.sh remember --alias local
lab/scripts/bench-queue.sh remember --alias HOST --ssh HOST
lab/scripts/bench-queue.sh provision --alias HOST --models suites/college_models.json
lab/scripts/bench-queue.sh run --alias local --alias HOST \
  --suite suites/college_mcq.json --models suites/college_models.json
```

`provision` creates the venv with uv, downloads the pinned llama.cpp build,
and downloads each model after checking its sha256. A Vulkan build is kept
when `llama-server --version` succeeds. Otherwise the CPU build is used.

A remembered connection with `"device": "gpu"` stays on a GPU engine. Vulkan
is used when that build lists a device. Otherwise the pinned llama.cpp tag is
built with CUDA when `nvcc` is installed on that machine. The CPU build is
not used for that connection, and a run is rejected unless the server log
shows at least one layer offloaded or the process is resident on the GPU. When several devices are listed, the
worker keeps one discrete GPU.

`run` provisions every alias, times one short completion per model, then
assigns each question to the machine that would finish sooner. Machines run
at the same time. One machine runs one model at a time. The score is written
under `~/.local/state/kilix-llm/bench-runs/`.
