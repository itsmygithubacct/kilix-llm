"""`kilix-llm stack`: several drafts written at once and one judge, behind one endpoint.

A small model writes better when it writes several replies and a second model
picks one. `stack serve` starts, on one GPU:

  the workers   one or more models, each loaded once with its own request slots
  the judge     one model with one slot

and serves `POST /v1/chat/completions` on loopback. Each request is written
`--drafts` times, as many at once as there are slots (draft 0 at temperature
0, the others sampled with fixed seeds), the judge rates every distinct draft
from 0 to 1, and the highest-rated draft is the reply. A tie keeps the earlier
draft, so the temperature-0 draft wins it. Nothing is loaded or unloaded
between requests.

A worker is `MODEL` or `MODEL:N` for N slots: a catalogue id, a GGUF path, or
`codex` or `claude` (`codex/MODEL`, `claude/MODEL` to name their model), which
run that command-line agent once for each draft with its tools off. The judge
is a catalogue id or GGUF path, or `codex` or `claude`.

`stack fit` finds how many slots of the first local worker load and answer
beside the other workers and the judge on this machine. It sends each slot a
prompt and the judge a full-length input at the same time, because a stack
that only loads can still run out of memory on its first real request.

A local judge is one of two kinds, chosen from the catalogue entry:

  chat        any chat model, asked the `kilix-llm judge` contract; the rating
              is p0, the probability of "0" (yes) against "1" (no)
  systemone   a decision model such as d3, asked through `/v1/systemone`
              (--judge-url asks a running service, such as Ollama, instead)
"""

from __future__ import annotations

import argparse
import copy
import http.server
import json
import os
import re
import shlex
import shutil
import signal
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from . import judge, models, pin, runtime

QUESTION = "Is this reply a correct and complete next step for the request?"
SEED = 1000
# Characters of request and reply shown to the judge for each token of its
# context. Code runs near three characters a token, so two leaves room.
CHARS_PER_TOKEN = 2
PROBE_WORD = "alpha beta gamma delta "
KINDS = ("chat", "systemone")
AGENTS = ("codex", "claude")
# The whole command for each agent; the prompt is its standard input. An
# environment variable of the same name in capitals replaces it, e.g.
# KILIX_LLM_STACK_CLAUDE="claude -p --tools ''".
AGENT_ARGV = {
    "codex": ["codex", "exec", "--sandbox", "read-only", "--skip-git-repo-check", "--ephemeral",
              "--color", "never", "-"],
    "claude": ["claude", "-p", "--output-format", "text", "--no-session-persistence", "--tools", ""],
}
AGENT_MODEL_FLAG = {"codex": "-m", "claude": "--model"}
AGENT_SECONDS = 900
AGENT_CHARS = 200_000
DRAFT_PROMPT = """\
You are writing the assistant's next message in the conversation below. Another program delivers it. \
Do not run any tool or command yourself, and do not ask a question back.

Reply with one JSON object and nothing else:
{"content": "<the assistant's text, or an empty string>", "tool_calls": [{"name": "<tool>", "arguments": {}}]}
Put a tool in "tool_calls" only if it is listed under Tools, and leave the list empty for a reply in text alone.
"""
RATE_PROMPT = "Reply with only a whole number from 0 to 100: how sure you are that the answer is yes."


class StackError(Exception):
    def __init__(self, message: str, code: int = judge.EXIT_USAGE):
        super().__init__(message)
        self.code = code


# --- the two engines -----------------------------------------------------------

def model_file(value: str, what: str) -> str:
    """A catalogue id, checked as every command checks it, or a GGUF path used as it is."""
    if value in models.MODELS:
        if models.base_id(value) is not None:
            raise StackError(f"{what} {value} is an adapter; the stack loads whole models")
        return models.resolve(value)
    if os.path.isfile(value):
        return os.path.abspath(value)
    raise StackError(f"{what} {value!r} is neither a catalogue id nor a GGUF file")


def gpu_argv(args: argparse.Namespace) -> list[str]:
    if args.backend == "cpu":
        return ["-ngl", "0", "--device", "none"]
    return ["-ngl", "99"] + (["--device", args.device] if args.device else [])


def worker_argv(args: argparse.Namespace, server: str, model: str, slots: int, sock: str) -> list[str]:
    """One model, `slots` request slots of `--ctx` tokens each."""
    argv = [server, "-m", model, "-c", str(args.ctx * slots), "--parallel", str(slots),
            "-t", str(args.threads or runtime.physical_cores()), "-fa", "on", "--no-webui",
            "--host", sock, "--jinja", "--reasoning", "auto" if args.think else "off"]
    if args.kv != "f16":
        argv += ["--cache-type-k", args.kv, "--cache-type-v", args.kv]
    return argv + gpu_argv(args)


def judge_argv(args: argparse.Namespace, server: str, model: str, sock: str) -> list[str]:
    argv = [server, "-m", model, "-c", str(args.judge_ctx), "--parallel", "1",
            "-t", str(args.threads or runtime.physical_cores()), "--no-webui", "--host", sock]
    if args.judge_kind == "systemone":
        # A decision model reads its whole input in one pass, so one batch must hold it.
        argv += ["-b", str(args.judge_ctx), "-ub", str(args.judge_ctx)]
        if args.judge_mmproj:
            argv += ["--mmproj", args.judge_mmproj, "--no-mmproj-offload"]
    else:
        argv += ["-fa", "on", "--jinja", "--reasoning", "off"]
    return argv + gpu_argv(args)


def post_url(url: str, body: dict, timeout: float | None = None) -> dict:
    request = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as reply:
            value = json.loads(reply.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read()[:300].decode("utf-8", "replace")
        raise judge.JudgeError(f"{url} answered HTTP {error.code}: {detail}", judge.EXIT_ENGINE) from error
    except (OSError, ValueError) as error:
        raise judge.JudgeError(f"{url} failed: {error}", judge.EXIT_ENGINE) from error
    if not isinstance(value, dict):
        raise judge.JudgeError(f"{url} answered with an unexpected shape", judge.EXIT_ENGINE)
    return value


# --- what the judge is shown ---------------------------------------------------

def text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(part.get("text", "") for part in content
                         if isinstance(part, dict) and part.get("type") == "text")
    return ""


def request_text(messages: list, limit: int) -> str:
    """The end of the conversation, newest last, cut from the front to `limit` characters."""
    lines = []
    for message in messages[-4:]:
        if not isinstance(message, dict):
            continue
        body = text_of(message.get("content"))
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            body += f"\n[tool call] {function.get('name')}({function.get('arguments')})"
        lines.append(f"{message.get('role', 'user')}: {body.strip()}")
    text = "\n\n".join(lines)
    return text if len(text) <= limit else "..." + text[len(text) - limit + 3:]


def reply_text(message: dict, limit: int) -> str:
    """A draft as the judge reads it: its text, then its tool calls, cut at the end."""
    text = text_of(message.get("content")).strip()
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        text += f"\n[tool call] {function.get('name')}({function.get('arguments')})"
    text = text.strip()
    return text if len(text) <= limit else text[:limit - 3] + "..."


# --- codex and claude ----------------------------------------------------------

def agent_of(value: str) -> tuple[str, str | None] | None:
    """("codex" | "claude", its model or None) when `value` names an agent, else None."""
    kind, _, model = value.partition("/")
    return (kind, model or None) if kind in AGENTS else None


def run_agent(kind: str, model: str | None, prompt: str) -> str:
    """One run of the agent with the prompt on its standard input; its reply as text."""
    override = os.environ.get(f"KILIX_LLM_STACK_{kind.upper()}")
    argv = shlex.split(override) if override else list(AGENT_ARGV[kind])
    if model:
        at = 2 if argv[1:2] == ["exec"] else 1
        argv[at:at] = [AGENT_MODEL_FLAG[kind], model]
    try:
        # An empty directory of its own: the agent has no project to read or change.
        with tempfile.TemporaryDirectory(prefix="kilix-llm-stack-") as cwd:
            child = subprocess.run(argv, input=prompt, capture_output=True, text=True, cwd=cwd,
                                   timeout=AGENT_SECONDS, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise judge.JudgeError(f"{kind} did not answer: {error}", judge.EXIT_ENGINE) from error
    if child.returncode != 0 or not child.stdout.strip():
        detail = (child.stderr or child.stdout).strip().splitlines()[-1:] or ["no output"]
        raise judge.JudgeError(f"{kind} exited {child.returncode}: {detail[0][:300]}", judge.EXIT_ENGINE)
    return child.stdout.strip()


def conversation_text(messages: list, limit: int) -> str:
    lines = []
    for message in messages:
        if isinstance(message, dict):
            lines.append(f"[{message.get('role', 'user')}]\n{reply_text(message, limit)}")
    text = "\n\n".join(lines)
    return text if len(text) <= limit else "..." + text[len(text) - limit + 3:]


def agent_message(text: str, draft: int) -> dict:
    """The agent's JSON reply as an assistant message; any other text is the message's content."""
    start, end = text.find("{"), text.rfind("}")
    try:
        value = json.loads(text[start:end + 1]) if 0 <= start < end else None
    except ValueError:
        value = None
    if not isinstance(value, dict) or not ({"content", "tool_calls"} & set(value)):
        return {"role": "assistant", "content": text}
    message = {"role": "assistant", "content": value.get("content") or ""}
    calls = []
    for call in value.get("tool_calls") or []:
        if isinstance(call, dict) and isinstance(call.get("name"), str):
            arguments = call.get("arguments")
            calls.append({"id": f"call_stack_{draft}_{len(calls)}", "type": "function", "function": {
                "name": call["name"],
                "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments or {})}})
    if calls:
        message["tool_calls"] = calls
    return message


class AgentWorker:
    """codex or claude: one run for each draft, at most `slots` at once."""

    pid = None

    def __init__(self, kind: str, model: str | None, slots: int):
        self.kind, self.model, self.slots = kind, model, slots
        self.name = kind + (f"/{model}" if model else "")
        self.free = threading.Semaphore(slots)

    def draft(self, body: dict, draft: int, _temperature: float, _think: bool) -> dict:
        tools = [tool.get("function", tool) for tool in body.get("tools") or [] if isinstance(tool, dict)]
        prompt = (DRAFT_PROMPT + "\nTools:\n" + (json.dumps(tools) if tools else "(none)")
                  + "\n\nConversation:\n" + conversation_text(body.get("messages") or [], AGENT_CHARS))
        with self.free:
            message = agent_message(run_agent(self.kind, self.model, prompt), draft)
        return {"id": f"chatcmpl-stack-{draft}", "object": "chat.completion", "created": int(time.time()),
                "choices": [{"index": 0, "message": message,
                             "finish_reason": "tool_calls" if message.get("tool_calls") else "stop"}],
                "usage": {}}

    def close(self) -> None:
        pass


class LocalWorker:
    """One llama-server with `slots` request slots."""

    def __init__(self, args: argparse.Namespace, name: str, path: str, slots: int):
        self.name, self.slots = name, slots
        self.engine = judge.Engine(
            lambda sock: worker_argv(args, args.server_path, path, slots, sock)).__enter__()

    @property
    def pid(self) -> int | None:
        return self.engine.pid

    def draft(self, body: dict, draft: int, temperature: float, think: bool) -> dict:
        request = copy.deepcopy(body)
        for key in ("stream", "stream_options", "n"):
            request.pop(key, None)
        request["temperature"] = temperature
        request["seed"] = SEED + draft
        if not think:
            request.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
        return self.engine.post("/v1/chat/completions", request)

    def close(self) -> None:
        self.engine.__exit__(None, None, None)


def parse_worker(text: str) -> tuple[str, int]:
    """`MODEL` or `MODEL:N`."""
    model, sep, count = text.rpartition(":")
    if sep and count.isdigit() and model:
        if int(count) < 1:
            raise StackError(f"worker {text!r} needs at least one slot")
        return model, int(count)
    return text, 1


# --- the stack -----------------------------------------------------------------

class Stack:
    """The workers and the judge, started together and stopped together."""

    def __init__(self, args: argparse.Namespace, workers: list[tuple[str, int]] | None = None):
        self.args = args
        self.spec = list(workers or args.workers)
        self.workers: list[LocalWorker | AgentWorker] = []
        self.judge: judge.Engine | None = None
        self.lock = threading.Lock()
        self.requests = 0
        self.totals = {"draft_seconds": 0.0, "judge_seconds": 0.0, "judged": 0, "changed": 0}
        budget = int(args.judge_ctx * CHARS_PER_TOKEN)
        self.request_chars, self.reply_chars = budget // 3, budget - budget // 3
        self.slots = sum(count for _, count in self.spec)
        self.drafts_wanted = args.drafts or self.slots

    def __enter__(self) -> "Stack":
        args = self.args
        try:
            # The judge loads first: it is the fixed part, and the slots take what is left.
            if args.judge_path:
                self.judge = judge.Engine(
                    lambda sock: judge_argv(args, args.judge_server_path, args.judge_path, sock)).__enter__()
            for model, count in self.spec:
                agent = agent_of(model)
                self.workers.append(AgentWorker(*agent, count) if agent
                                    else LocalWorker(args, model, args.worker_paths[model], count))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        # Draft i is written by turn[i % slots]: every slot of the first worker, then the next.
        self.turn = [worker for worker in self.workers for _ in range(worker.slots)]
        return self

    def __exit__(self, *_exc) -> None:
        workers, self.workers = self.workers, []
        for worker in workers:
            worker.close()
        engine, self.judge = self.judge, None
        if engine is not None:
            engine.__exit__(None, None, None)

    # --- drafting ---

    def draft(self, body: dict, draft: int) -> dict:
        worker = self.turn[draft % len(self.turn)]
        return worker.draft(body, draft, 0.0 if draft == 0 else self.args.temperature, self.args.think)

    def drafts(self, body: dict) -> list[dict | None]:
        def one(draft: int) -> dict | None:
            try:
                reply = self.draft(body, draft)
                reply["choices"][0]["message"]  # the shape every later step relies on
                return reply
            except (judge.JudgeError, KeyError, IndexError, TypeError) as error:
                if draft == 0:
                    raise
                print(f"kilix-llm: stack: draft {draft} failed: {error}", file=sys.stderr, flush=True)
                return None
        with ThreadPoolExecutor(self.slots) as pool:
            return list(pool.map(one, range(self.drafts_wanted)))

    # --- judging ---

    def rate(self, request: str, replies: list[str]) -> list[float | None]:
        """One rating from 0 to 1 for each reply, or None where the judge gave none."""
        try:
            if self.args.judge_agent:
                with ThreadPoolExecutor(min(4, len(replies))) as pool:
                    return list(pool.map(lambda reply: self._rate_agent(request, reply), replies))
            if self.args.judge_kind == "systemone":
                return self._rate_systemone(request, replies)
            return [self._rate_chat(request, reply) for reply in replies]
        except (judge.JudgeError, KeyError, TypeError, ValueError) as error:
            print(f"kilix-llm: stack: the judge gave no rating: {error}", file=sys.stderr, flush=True)
            return [None] * len(replies)

    def _rate_systemone(self, request: str, replies: list[str]) -> list[float | None]:
        body = {"model": self.args.judge_name, "state": {"request": request}, "questions": {
            f"d{n}": {"type": "noul", "instructions": f"{self.args.question}\n\n{reply}"}
            for n, reply in enumerate(replies)}}
        if self.args.judge_url:
            answer = post_url(self.args.judge_url.rstrip("/") + "/v1/systemone", body)
        else:
            answer = self.judge.post("/v1/systemone", body)
        return [float(answer["answers"][f"d{n}"]["noul"]) for n in range(len(replies))]

    def _rate_agent(self, request: str, reply: str) -> float | None:
        text = run_agent(*self.args.judge_agent, (
            f"<information>\nThe request:\n{request}\n\nThe reply:\n{reply}\n</information>\n\n"
            f"{self.args.question}\n{RATE_PROMPT}"))
        found = re.search(r"\d+(?:\.\d+)?", text)
        return None if found is None else min(1.0, max(0.0, float(found.group()) / 100))

    def _rate_chat(self, request: str, reply: str) -> float | None:
        text = (f"<information>\nThe request:\n{request}\n\nThe reply:\n{reply}\n</information>\n\n"
                f"{self.args.question}")
        tokens = self.judge.tokenize(self.judge.render(text), special=True)
        if len(tokens) + 1 > self.args.judge_ctx:
            return None
        answer = self.judge.post("/completion", {
            "prompt": tokens, "n_predict": 1, "temperature": 0, "grammar": judge.GRAMMAR,
            "n_probs": judge.N_PROBS, "post_sampling_probs": False, "cache_prompt": False})
        return judge.answer_probs(answer)[0]

    # --- one request ---

    def complete(self, body: dict) -> dict:
        """The chosen draft's whole reply, with what happened under `kilix_stack`."""
        with self.lock:  # every slot belongs to one request at a time
            started = time.monotonic()
            replies = self.drafts(body)
            drafted = time.monotonic()
            messages = [None if r is None else r["choices"][0]["message"] for r in replies]
            texts = [None if m is None else reply_text(m, self.reply_chars) for m in messages]
            distinct = sorted({t for t in texts if t is not None})
            ratings: list[float | None] = [None] * len(texts)
            if len(distinct) > 1:
                rated = dict(zip(distinct, self.rate(request_text(body.get("messages") or [], self.request_chars),
                                                     distinct)))
                ratings = [None if t is None else rated[t] for t in texts]
            chosen = 0
            for draft, rating in enumerate(ratings):
                if rating is not None and (ratings[chosen] is None or rating > ratings[chosen]):
                    chosen = draft
            if ratings[0] is None and ratings[chosen] is not None and texts[chosen] == texts[0]:
                chosen = 0
            done = time.monotonic()
            record = {"slots": self.slots, "drafts": sum(r is not None for r in replies),
                      "workers": [self.turn[n % len(self.turn)].name for n in range(len(replies))],
                      "distinct": len(distinct), "ratings": ratings, "chosen": chosen,
                      "draft_seconds": round(drafted - started, 3), "judge_seconds": round(done - drafted, 3),
                      "completion_tokens": [None if r is None else (r.get("usage") or {}).get("completion_tokens")
                                            for r in replies]}
            self.requests += 1
            self.totals["draft_seconds"] += drafted - started
            self.totals["judge_seconds"] += done - drafted
            self.totals["judged"] += len(distinct) > 1
            self.totals["changed"] += texts[chosen] != texts[0]
            self._journal(record, body, texts)
        reply = replies[chosen]
        reply["model"] = self.args.name
        reply["kilix_stack"] = record
        return reply

    def _journal(self, record: dict, body: dict, texts: list) -> None:
        if not self.args.journal:
            return
        row = dict(record, time=round(time.time(), 3), replies=texts,
                   request=request_text(body.get("messages") or [], self.request_chars))
        try:
            with open(self.args.journal, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError as error:
            print(f"kilix-llm: stack: cannot write the journal: {error}", file=sys.stderr, flush=True)

    # --- what it costs ---

    def status(self) -> dict:
        n = max(1, self.requests)
        return {"model": self.args.name, "workers": [{"model": w.name, "slots": w.slots} for w in self.workers],
                "drafts": self.drafts_wanted, "ctx_per_slot": self.args.ctx, "kv": self.args.kv,
                "judge": self.args.judge, "judge_kind": self.args.judge_kind, "judge_ctx": self.args.judge_ctx,
                "requests": self.requests, "judged": self.totals["judged"], "changed": self.totals["changed"],
                "mean_draft_seconds": round(self.totals["draft_seconds"] / n, 3),
                "mean_judge_seconds": round(self.totals["judge_seconds"] / n, 3),
                "resources": self.resources()}

    def resources(self) -> dict:
        """Resident memory of each engine and, where nvidia-smi answers, its GPU memory, in MiB."""
        gpu = gpu_by_pid()
        engines = [(f"worker {worker.name}", worker.pid) for worker in self.workers]
        engines.append(("judge", self.judge.pid if self.judge else None))
        return {name: {"pid": pid, "ram_mib": rss_mib(pid), "gpu_mib": gpu.get(pid)}
                for name, pid in engines if pid is not None}


def rss_mib(pid: int) -> int | None:
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def gpu_by_pid() -> dict[int, int]:
    tool = shutil.which("nvidia-smi")
    if not tool:
        return {}
    try:
        text = subprocess.run([tool, "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                              stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return {}
    out: dict[int, int] = {}
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            out[int(parts[0])] = out.get(int(parts[0]), 0) + int(parts[1])
    return out


# --- the endpoint --------------------------------------------------------------

def stream_chunks(reply: dict) -> list[dict]:
    """The chosen reply as chat.completion.chunk events: it is whole before any of it is sent."""
    choice = reply["choices"][0]
    message = choice["message"]
    base = {"id": reply.get("id", "chatcmpl-stack"), "object": "chat.completion.chunk",
            "created": reply.get("created", int(time.time())), "model": reply.get("model")}
    deltas: list[dict] = [{"role": "assistant"}]
    if message.get("reasoning_content"):
        deltas.append({"reasoning_content": message["reasoning_content"]})
    if message.get("content"):
        deltas.append({"content": message["content"]})
    calls = [dict(call, index=index) for index, call in enumerate(message.get("tool_calls") or [])]
    if calls:
        deltas.append({"tool_calls": calls})
    chunks = [dict(base, choices=[{"index": 0, "delta": delta, "finish_reason": None}]) for delta in deltas]
    chunks.append(dict(base, choices=[{"index": 0, "delta": {}, "finish_reason": choice.get("finish_reason", "stop")}],
                       usage=reply.get("usage"), kilix_stack=reply.get("kilix_stack")))
    return chunks


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    stack: Stack

    def log_message(self, *_args) -> None:
        pass

    def send_json(self, value: dict, status: int = 200) -> None:
        data = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def fail(self, status: int, message: str) -> None:
        self.send_json({"error": {"message": message, "type": "kilix_stack_error"}}, status)

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/health":
            self.send_json({"status": "ok"})
        elif path in ("/v1/models", "/models"):
            self.send_json({"object": "list", "data": [
                {"id": self.stack.args.name, "object": "model", "owned_by": "kilix-llm"}]})
        elif path == "/stack/status":
            self.send_json(self.stack.status())
        else:
            self.fail(404, "not found")

    def do_POST(self) -> None:
        if self.path.split("?", 1)[0] not in ("/v1/chat/completions", "/chat/completions"):
            self.fail(404, "not found")
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
            if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
                raise ValueError("the body needs a messages list")
        except (ValueError, UnicodeDecodeError) as error:
            self.fail(400, f"bad request: {error}")
            return
        try:
            reply = self.stack.complete(body)
        except (judge.JudgeError, KeyError, IndexError, TypeError) as error:
            self.fail(502, f"the first worker gave no reply: {error}")
            return
        if not body.get("stream"):
            self.send_json(reply)
            return
        data = b"".join(b"data: " + json.dumps(chunk).encode("utf-8") + b"\n\n" for chunk in stream_chunks(reply))
        data += b"data: [DONE]\n\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class UnixHTTPServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    """The endpoint on a unix socket, for a caller that mounts it into a sandbox."""

    daemon_threads = True

    def get_request(self):
        request, _ = super().get_request()
        return request, ("unix", 0)


# --- commands ------------------------------------------------------------------

def prepare(args: argparse.Namespace) -> None:
    """Resolve the files and binaries every stack command needs, before anything is started."""
    args.workers = [parse_worker(text) for text in (args.worker or ["qwen3.5-0.8b:4"])]
    if args.ctx < 256 or args.judge_ctx < 256 or (args.drafts or 1) < 1:
        raise StackError("--ctx and --judge-ctx must be at least 256, and --drafts at least 1")
    if len({model for model, _ in args.workers}) < len(args.workers):
        raise StackError("each worker model is named once; give it more slots with MODEL:N")
    args.judge_agent = agent_of(args.judge)
    local_workers = [model for model, _ in args.workers if not agent_of(model)]
    local_judge = not (args.judge_agent or args.judge_url)
    if args.judge_agent and (args.judge_url or args.judge_kind):
        raise StackError(f"{args.judge} is asked directly; --judge-kind and --judge-url do not apply")
    if args.judge_url:
        args.judge_kind = args.judge_kind or "systemone"
        if args.judge_kind != "systemone":
            raise StackError("--judge-url is for --judge-kind systemone")
    elif not args.judge_agent and args.judge_kind is None:
        family = models.MODELS.get(args.judge, {}).get("family")
        args.judge_kind = "systemone" if family == "d3" else "chat"
    vendored = None
    if args.server is None and (local_workers or (local_judge and not args.judge_server)):
        runtime.require_vendor()
        runtime.ensure_built(args.backend)
        vendored = runtime.binary("llama-server", args.backend)
        if args.backend != "cpu":
            args.device = runtime.select_device(args.backend, args.device)
    args.server_path = args.server or vendored
    args.judge_server_path = args.judge_server or args.server_path
    for path in {args.server_path if local_workers else None, args.judge_server_path if local_judge else None}:
        if path is not None and not (os.path.isfile(path) and os.access(path, os.X_OK)):
            raise StackError(f"{path} is not an executable llama-server", judge.EXIT_RUNTIME)
    args.worker_paths = {model: model_file(model, "worker") for model in local_workers}
    args.judge_path = model_file(args.judge, "the judge") if local_judge else None
    args.judge_name = os.path.basename(args.judge) if os.path.isfile(args.judge) else args.judge
    if args.judge_mmproj and not os.path.isfile(args.judge_mmproj):
        raise StackError(f"{args.judge_mmproj} is not a file")
    drafters = "+".join(os.path.basename(model) + (f"x{count}" if count > 1 else "") for model, count in args.workers)
    args.name = args.name or f"{drafters}_draft_{args.judge_name}_judge"


def guarded(run):
    def command(args: argparse.Namespace) -> int:
        try:
            prepare(args)
            return run(args)
        except (StackError, judge.JudgeError) as error:
            print(f"kilix-llm: stack: {error}", file=sys.stderr)
            return error.code
        except (models.ModelError, runtime.RuntimeError_, pin.PinError) as error:
            print(f"kilix-llm: stack: {error}", file=sys.stderr)
            return judge.EXIT_USAGE if isinstance(error, models.ModelError) else judge.EXIT_RUNTIME
    return command


@guarded
def cmd_serve(args: argparse.Namespace) -> int:
    with Stack(args) as stack:
        handler = type("StackHandler", (Handler,), {"stack": stack})
        try:
            if args.socket:
                httpd = UnixHTTPServer(args.socket, handler)
                where = args.socket
            else:
                httpd = http.server.ThreadingHTTPServer((args.host, args.port), handler)
                httpd.daemon_threads = True
                where = f"http://{args.host}:{httpd.server_address[1]}/v1"
        except OSError as error:
            raise StackError(f"cannot listen on {args.socket or f'{args.host}:{args.port}'}: {error}") from error
        signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=httpd.shutdown, daemon=True).start())
        print(f"kilix-llm: stack: {args.name} on {where} "
              f"({stack.drafts_wanted} drafts in {stack.slots} slots of {args.ctx} tokens, judge {args.judge_name})",
              file=sys.stderr, flush=True)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            httpd.server_close()
    return 0


def probe(stack: Stack, args: argparse.Namespace) -> dict:
    """Every local slot answers a prompt while the judge reads a full-length input."""
    filler = PROBE_WORD * (args.probe_tokens // 4)
    body = {"messages": [{"role": "user", "content": filler + "\nReply with the single word: ready"}],
            "max_tokens": 48}
    long_reply = PROBE_WORD * (stack.reply_chars // len(PROBE_WORD))
    started = time.monotonic()
    local = [(worker, n) for worker in stack.workers if isinstance(worker, LocalWorker) for n in range(worker.slots)]
    with ThreadPoolExecutor(len(local) + 1) as pool:
        judging = None if args.judge_agent else pool.submit(
            stack.rate, "Reply with the single word: ready", [long_reply, "ready"])
        drafting = [pool.submit(worker.draft, body, n, 0.8, False) for worker, n in local]
        for future in drafting:
            future.result()["choices"][0]["message"]
        ratings = judging.result() if judging else []
    if any(rating is None for rating in ratings):
        raise judge.JudgeError("the judge did not rate a full-length input", judge.EXIT_ENGINE)
    return {"seconds": round(time.monotonic() - started, 2), "resources": stack.resources()}


@guarded
def cmd_fit(args: argparse.Namespace) -> int:
    grown = next((model for model, _ in args.workers if not agent_of(model)), None)
    if grown is None:
        raise StackError("fit needs a local worker: codex and claude take no memory here")
    rows = []
    for slots in range(1, args.max_slots + 1):
        workers = [(model, slots if model == grown else count) for model, count in args.workers]
        row = {"worker": grown, "slots": slots, "ctx_per_slot": args.ctx, "kv": args.kv}
        try:
            with Stack(args, workers) as stack:
                row.update(probe(stack, args), fits=True)
        except (judge.JudgeError, KeyError, IndexError, TypeError) as error:
            row.update(fits=False, reason=str(error).splitlines()[0][:200])
        rows.append(row)
        if not args.json:
            used = "; ".join(f"{name} {value['gpu_mib']} MiB GPU, {value['ram_mib']} MiB RAM"
                             for name, value in (row.get("resources") or {}).items())
            print(f"{slots} slot{'s' if slots > 1 else ''} of {args.ctx}: "
                  + (f"fits  {used}" if row["fits"] else f"does not fit  ({row['reason']})"), flush=True)
        if not row["fits"]:
            break
    best = max((row["slots"] for row in rows if row["fits"]), default=0)
    if args.json:
        print(json.dumps({"worker": grown, "max_slots": best, "tried": rows}, sort_keys=True))
    else:
        print(f"most slots of {os.path.basename(grown)} beside {args.judge_name}: {best}"
              + ("" if rows[-1]["fits"] is False else f" (stopped at --max-slots {args.max_slots})"))
    return 0 if best else judge.EXIT_ENGINE


def installed(model_id: str) -> bool:
    record = models.MODELS[model_id]
    path = models.model_path(model_id)
    return (record.get("sha256") is not None and "base" not in record
            and os.path.isfile(path) and os.path.getsize(path) == record["size"])


def choices() -> dict:
    """What a picker can offer on this machine: installed whole models, and the agents on PATH."""
    present = [model_id for model_id in sorted(models.MODELS) if installed(model_id)]
    agents = [agent for agent in AGENTS if shutil.which(agent)]
    return {"workers": [m for m in present if models.MODELS[m].get("family") != "d3"] + agents,
            "judges": present + agents}


def cmd_choices(args: argparse.Namespace) -> int:
    found = choices()
    if args.json:
        print(json.dumps(found, sort_keys=True))
    else:
        for kind in ("workers", "judges"):
            print(f"{kind}: {' '.join(found[kind]) or '(none installed)'}")
    return 0


def add_parser(sub) -> None:
    parser = sub.add_parser(
        "stack", help="several drafts at once and one judge behind one endpoint",
        description=__doc__.split("\n\n", 1)[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="stack_command", required=True)
    for name, func, text in (("serve", cmd_serve, "start the stack and its endpoint"),
                             ("fit", cmd_fit, "find how many slots fit beside the judge")):
        p = commands.add_parser(name, help=text)
        p.add_argument("--worker", action="append", metavar="MODEL[:N]",
                       help="a model that writes drafts, with N slots: a catalogue id, a GGUF path, "
                            "codex or claude; repeat for more models (default: qwen3.5-0.8b:4)")
        p.add_argument("--drafts", type=int, default=None,
                       help="drafts written for each request (default: one for each slot)")
        p.add_argument("--ctx", type=int, default=65536, help="context tokens for each slot (default: 65536)")
        p.add_argument("--kv", choices=("f16", "q8_0"), default="q8_0",
                       help="context cache precision; q8_0 halves its memory (default: q8_0)")
        p.add_argument("--judge", default="qwen3.5-4b",
                       help="catalogue id, GGUF path, codex or claude, or with --judge-url the "
                            "service's model name (default: qwen3.5-4b)")
        p.add_argument("--judge-kind", choices=KINDS, default=None,
                       help="for a GGUF path; a catalogue id knows its own kind")
        p.add_argument("--judge-ctx", type=int, default=4096, help="the judge's context (default: 4096)")
        p.add_argument("--judge-mmproj", default=None,
                       help="projector file, for a systemone judge whose runtime needs one; it stays on the CPU")
        p.add_argument("--judge-server", default=None, help="llama-server for the judge alone")
        p.add_argument("--judge-url", default=None,
                       help="a running service that serves /v1/systemone, e.g. http://127.0.0.1:11434")
        p.add_argument("--server", default=None, help="llama-server to run in place of the vendored build")
        p.add_argument("--question", default=QUESTION, help="what the judge is asked about each draft")
        p.add_argument("--temperature", type=float, default=0.8, help="for every draft after the first (default: 0.8)")
        p.add_argument("--think", action="store_true", help="leave the local workers' thinking mode on")
        p.add_argument("--backend", choices=runtime.BACKENDS, default="vulkan")
        p.add_argument("--device", help="GPU name from `kilix-llm devices`, e.g. Vulkan0")
        p.add_argument("-t", "--threads", type=int, default=0, help="CPU threads (default: physical cores)")
        p.add_argument("--name", default=None, help="model name the endpoint reports")
        p.add_argument("--journal", default=None, help="append one JSON line for each request to this file")
        p.set_defaults(func=func, host="127.0.0.1")
        if name == "serve":
            p.add_argument("--port", type=int, default=8093, help="loopback port (default: 8093)")
            p.add_argument("--socket", default=None, help="listen on this unix socket in place of the port")
        else:
            p.add_argument("--max-slots", type=int, default=12, help="stop the search here (default: 12)")
            p.add_argument("--probe-tokens", type=int, default=1500,
                           help="rough length of the prompt each slot is sent (default: 1500)")
            p.add_argument("--json", action="store_true")
    p = commands.add_parser("choices", help="list the installed models and agents a stack can use")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_choices)
