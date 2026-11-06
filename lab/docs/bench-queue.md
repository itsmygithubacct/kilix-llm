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

`run` provisions every alias, times one short completion per model, then
assigns each question to the machine that would finish sooner. Machines run
at the same time. One machine runs one model at a time. The score is written
under `~/.local/state/kilix-llm/bench-runs/`.
