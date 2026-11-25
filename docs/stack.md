# Stack: several drafts at once and one judge

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
| `chat` | any chat model, asked the `judge` contract in [judge.md](judge.md) | `p0`, the probability of "0" (yes) against "1" (no) |
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
