# Agent loops against kilix-llm

Three open agent loops can drive `kilix-llm serve` with no cloud key:
mini-swe-agent, smolagents and harbor's Terminus-2. Each one's defaults
overflow the server's 8,192-token default context within a few steps. This page
gives settings that fit, and a sandbox for the commands the model writes.
Versions read: mini-swe-agent 2.4.6, smolagents 1.27.0.dev0, harbor 0.24.0
(2026-10-06).

## Check the context first

The server's context is whatever `-c` it started with (`--ctx`, or the model's
catalogue default of 8,192). Ask the server rather than assume:

```sh
curl -s http://127.0.0.1:8080/props | python3 -c \
  'import json,sys; print(json.load(sys.stdin)["default_generation_settings"]["n_ctx"])'
```

Budget against that number. For an 8k context:

- cap each observation at 2,000–4,000 characters, keeping the head and the tail;
- always set a step cap: at $0 a cost limit never fires;
- compact, or stop, at about 75% of the context.

| Loop | Fixed prompt (approx. tokens) | Default observation cap | Trap at 8k |
| --- | --- | --- | --- |
| mini-swe-agent | ~720 | 10,000 chars (YAML only) | No compaction: overflow ends the run with an error. `mini.yaml` has no step limit |
| Terminus-2 | ~660 (JSON) | 10 KB | Its 8,000-token summarization threshold fires every turn; without `model_info` it assumes a 1M context |
| smolagents CodeAgent | ~2,300 before tools | 20,000 chars | Memory is never compacted; one large output nearly fills the window |

Token counts are cl100k approximations of the unrendered templates.

## mini-swe-agent

Use **text mode** on small models. On 2026-10-06 Qwen3.5-4B (Q6_K, Vulkan,
`--jinja`, thinking off) parsed into an action on the first turn 30 of 30 times
in text mode, but only 10 of 30 times in tool-call mode. In tool-call mode it
mostly copied `mini.yaml`'s prose example ("[Makes bash tool call …]") instead
of calling the tool. Full tool-call runs still finished 3 of 3 small tasks, so
treat the 10/30 as a first-turn rate on a small sample.

```yaml
# local.yaml -- run as: MSWEA_CONFIGURED=1 mini -c mini_textbased.yaml -c local.yaml
# (-c replaces the default config, so list mini_textbased.yaml too)
model:
  model_name: "openai/local"          # the server ignores the name
  model_class: litellm_textbased      # mini_textbased.yaml does not set this itself
  cost_tracking: ignore_errors        # a $0 cost is an error otherwise, even when registered
  model_kwargs:
    api_base: "http://127.0.0.1:8080/v1"
    api_key: "local"                  # any non-empty string
    max_tokens: 1024
    drop_params: true
agent:
  step_limit: 30
```

Also set `MSWEA_CONFIGURED=1`, or the first run opens a setup wizard that asks
for a cloud key.

## smolagents

```python
from smolagents import CodeAgent, OpenAIServerModel   # pip install 'smolagents[openai]'

model = OpenAIServerModel(model_id="local", api_base="http://127.0.0.1:8080/v1", api_key="local")
agent = CodeAgent(tools=[], model=model, max_steps=8)
```

Its default executor, `LocalPythonExecutor`, is an AST interpreter. Its own docstring says it is not a
security sandbox: once `os` is an authorized import, `os.system(...)` runs. Do
not authorize `os`, `subprocess` or `"*"` for model code outside a real sandbox.

## Terminus-2 (harbor)

Terminus-2 needs a container or VM runtime (Docker, podman, singularity, or a
cloud sandbox); it has no host mode.

```sh
OPENAI_API_KEY=local harbor run -p TASK -a terminus-2 -m openai/local \
  --ak api_base=http://127.0.0.1:8080/v1 \
  --ak 'model_info={"max_input_tokens":8192,"max_output_tokens":2048,"input_cost_per_token":0,"output_cost_per_token":0}' \
  --ak proactive_summarization_threshold=2000
```

The model client runs on the host, so a task with `network_mode: no-network`
still reaches the loopback server.

## Running the model's commands: `scripts/agent-sandbox`

Keep the loop and its model client outside the box, and run only the commands
inside:

```sh
scripts/agent-sandbox --timeout 30 WORKDIR -c 'pytest -q'
scripts/agent-sandbox --net --keep-env PIP_INDEX_URL WORKDIR -- pip install -r requirements.txt
```

The box has no network (not even the host's loopback) unless `--net` is
given, and a cleared environment (`--keep-env NAME` passes one variable). `/usr`
and `/etc` are read-only, `/tmp` is private, and WORKDIR is the only writable
directory, as well as HOME and the working directory. It ignores the caller's
stdin and dies with its parent. `--timeout` kills the whole tree and exits 124.
It needs bubblewrap with unprivileged user namespaces.

A loop that calls one shell command per action can wrap each call. For
mini-swe-agent, subclass `LocalEnvironment` and run
`[sandbox, "--timeout", str(timeout), workdir, "-c", command]` instead of the
shell command; keep its submission check on the output.

For long multi-step work, use the QEMU guest instead (`kilix-llm vm`, where
available): it is the stronger boundary.
