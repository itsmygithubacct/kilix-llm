"""`kilix-llm stack` against the fake engine: no model, no GPU, no network."""

from __future__ import annotations

import io
import json
import os
import stat
import sys
import tempfile
import threading
import unittest
import urllib.request
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

from kilix_llm import runtime, stack
from kilix_llm.cli import build_parser

FAKE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fake_engine.py")
ASK = {"messages": [{"role": "user", "content": "Write add(a, b)."}], "max_tokens": 50}


class StackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = self.tmp.name
        self.log = os.path.join(root, "fake.log")
        self.server = os.path.join(root, "llama-server")
        with open(self.server, "w") as handle:
            handle.write(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE}" server "$@"\n')
        os.chmod(self.server, os.stat(self.server).st_mode | stat.S_IXUSR)
        self.drafter, self.judge, self.mmproj = (os.path.join(root, name) for name in
                                                 ("drafter.gguf", "judge.gguf", "judge-mmproj.gguf"))
        for path in (self.drafter, self.judge, self.mmproj):
            with open(path, "wb") as handle:
                handle.write(b"weights")
        for patcher in (mock.patch.dict(os.environ, {"FAKE_LOG": self.log}),
                        mock.patch.object(runtime, "physical_cores", return_value=4)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def args(self, command: str, *argv: str):
        args = build_parser().parse_args([
            "stack", command, "--server", self.server, "--backend", "cpu",
            "--judge", self.judge, *argv])
        return args

    def stack(self, *argv: str, slots: int = 3, workers: tuple = ()) -> stack.Stack:
        named = [flag for spec in (workers or (f"{self.drafter}:{slots}",)) for flag in ("--worker", spec)]
        args = self.args("serve", *named, *argv)
        stack.prepare(args)
        made = stack.Stack(args).__enter__()
        self.addCleanup(made.__exit__, None, None, None)
        return made

    def agent(self, name: str, script: str) -> None:
        """A stand-in for codex or claude: a shell script given the prompt on stdin."""
        path = os.path.join(self.tmp.name, name)
        with open(path, "w") as handle:
            handle.write("#!/bin/sh\n" + script + "\n")
        os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)
        self.env(**{f"KILIX_LLM_STACK_{name.upper()}": path})

    def events(self, kind: str) -> list:
        with open(self.log, encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle]
        return [row["value"] for row in rows if row["kind"] == kind]

    def env(self, **values):
        patcher = mock.patch.dict(os.environ, {key: value if isinstance(value, str) else json.dumps(value)
                                               for key, value in values.items()})
        patcher.start()
        self.addCleanup(patcher.stop)

    # --- the engines -----------------------------------------------------------

    def test_the_drafter_gets_every_slot_and_the_judge_one(self) -> None:
        self.stack("--ctx", "1000", slots=3)
        judge_argv, drafter_argv = self.events("server-argv")
        for flag, value in (("-m", self.drafter), ("-c", "3000"), ("--parallel", "3"), ("-t", "4"),
                            ("--cache-type-k", "q8_0"), ("--cache-type-v", "q8_0"), ("--reasoning", "off"),
                            ("-ngl", "0")):
            self.assertEqual(drafter_argv[drafter_argv.index(flag) + 1], value, flag)
        for flag, value in (("-m", self.judge), ("-c", "4096"), ("--parallel", "1")):
            self.assertEqual(judge_argv[judge_argv.index(flag) + 1], value, flag)
        self.assertNotIn("-ub", judge_argv)

    def test_a_systemone_judge_reads_its_context_in_one_batch(self) -> None:
        self.stack("--judge-kind", "systemone", "--judge-mmproj", self.mmproj, "--judge-ctx", "2048", "--kv", "f16")
        judge_argv, drafter_argv = self.events("server-argv")
        for flag, value in (("-b", "2048"), ("-ub", "2048"), ("--mmproj", self.mmproj)):
            self.assertEqual(judge_argv[judge_argv.index(flag) + 1], value, flag)
        self.assertIn("--no-mmproj-offload", judge_argv)
        self.assertNotIn("--cache-type-k", drafter_argv)

    def test_a_judge_url_is_for_systemone(self) -> None:
        args = self.args("serve", "--worker", self.drafter, "--judge-url", "http://127.0.0.1:1",
                         "--judge-kind", "chat")
        err = io.StringIO()
        with redirect_stderr(err):
            self.assertEqual(args.func(args), 2)
        self.assertIn("--judge-url is for", err.getvalue())

    # --- one request -----------------------------------------------------------

    def test_the_highest_rated_draft_is_the_reply(self) -> None:
        self.env(FAKE_DRAFTS=["return a - b", "return a + b", "return a * b"],
                 FAKE_P0={"a - b": 0.2, "a + b": 0.9, "a * b": 0.4})
        journal = os.path.join(self.tmp.name, "journal.jsonl")
        made = self.stack("--journal", journal, "--name", "pair")
        reply = made.complete(ASK)
        self.assertEqual(reply["choices"][0]["message"]["content"], "return a + b")
        self.assertEqual(reply["model"], "pair")
        record = reply["kilix_stack"]
        self.assertEqual((record["chosen"], record["drafts"], record["distinct"]), (1, 3, 3))
        self.assertEqual([round(r, 2) for r in record["ratings"]], [0.2, 0.9, 0.4])
        drafts = sorted(self.events("/v1/chat/completions"), key=lambda body: body["seed"])
        self.assertEqual([(d["seed"], d["temperature"]) for d in drafts], [(1000, 0.0), (1001, 0.8), (1002, 0.8)])
        self.assertTrue(all(d["chat_template_kwargs"] == {"enable_thinking": False} and "stream" not in d
                            for d in drafts))
        shown = self.events("/apply-template")[-1]["messages"][1]["content"]
        self.assertIn("user: Write add(a, b).", shown)
        self.assertTrue(shown.endswith(stack.QUESTION))
        with open(journal, encoding="utf-8") as handle:
            row = json.loads(handle.readline())
        self.assertEqual((row["chosen"], row["replies"][1]), (1, "return a + b"))
        self.assertEqual((made.status()["requests"], made.status()["changed"]), (1, 1))

    def test_a_tie_keeps_the_temperature_0_draft(self) -> None:
        self.env(FAKE_DRAFTS=["uno", "dos", "tres"], FAKE_P0={"uno": 0.7, "dos": 0.7, "tres": 0.7})
        self.assertEqual(self.stack().complete(ASK)["kilix_stack"]["chosen"], 0)

    def test_identical_drafts_are_not_judged(self) -> None:
        self.env(FAKE_DRAFTS=["same", "same", "same"])
        record = self.stack().complete(ASK)["kilix_stack"]
        self.assertEqual((record["chosen"], record["distinct"], record["ratings"]), (0, 1, [None] * 3))
        self.assertEqual(self.events("/completion"), [])

    def test_a_failed_sampled_draft_is_left_out(self) -> None:
        self.env(FAKE_DRAFTS=["uno", None, "tres"], FAKE_P0={"uno": 0.3, "tres": 0.8})
        with redirect_stderr(io.StringIO()):
            record = self.stack().complete(ASK)["kilix_stack"]
        self.assertEqual((record["chosen"], record["drafts"], record["ratings"][1]), (2, 2, None))

    def test_a_systemone_judge_rates_every_draft_in_one_request(self) -> None:
        self.env(FAKE_DRAFTS=["uno", "dos", "tres"], FAKE_RATINGS={"dos": 0.1, "tres": 0.95})
        reply = self.stack("--judge-kind", "systemone").complete(ASK)
        self.assertEqual(reply["choices"][0]["message"]["content"], "tres")
        (asked,) = self.events("/v1/systemone")
        self.assertEqual(len(asked["questions"]), 3)
        self.assertIn("Write add(a, b).", asked["state"]["request"])

    def test_without_a_rating_the_temperature_0_draft_is_the_reply(self) -> None:
        self.env(FAKE_DRAFTS=["uno", "dos", "tres"], FAKE_SYSTEMONE="404")
        err = io.StringIO()
        with redirect_stderr(err):
            record = self.stack("--judge-kind", "systemone").complete(ASK)["kilix_stack"]
        self.assertEqual((record["chosen"], record["ratings"]), (0, [None] * 3))
        self.assertIn("the judge gave no rating", err.getvalue())

    def test_long_text_is_cut_to_the_judges_context(self) -> None:
        self.env(FAKE_DRAFTS=["x" * 9000, "short"], FAKE_RATINGS={})
        made = self.stack("--judge-kind", "systemone", "--judge-ctx", "1024", slots=2)
        made.complete({"messages": [{"role": "user", "content": "q" * 9000}]})
        (asked,) = self.events("/v1/systemone")
        self.assertLessEqual(len(asked["state"]["request"]), 1024 * stack.CHARS_PER_TOKEN // 3)
        self.assertTrue(asked["state"]["request"].startswith("..."))
        longest = max(len(q["instructions"]) for q in asked["questions"].values())
        self.assertLessEqual(longest, 1024 * stack.CHARS_PER_TOKEN + len(stack.QUESTION))

    # --- several workers, more drafts than slots, codex and claude -----------------

    def test_drafts_go_round_the_slots_of_every_worker(self) -> None:
        second = os.path.join(self.tmp.name, "second.gguf")
        with open(second, "wb") as handle:
            handle.write(b"weights")
        self.env(FAKE_DRAFTS=["a0", "a1", "a2", "a3", "a4"], FAKE_P0={"a4": 0.9})
        made = self.stack("--drafts", "5", workers=(f"{self.drafter}:2", second))
        record = made.complete(ASK)["kilix_stack"]
        names = [os.path.basename(name) for name in record["workers"]]
        self.assertEqual(names, ["drafter.gguf", "drafter.gguf", "second.gguf", "drafter.gguf", "drafter.gguf"])
        self.assertEqual((record["chosen"], record["drafts"], record["slots"]), (4, 5, 3))
        loaded = sorted(argv[argv.index("--parallel") + 1] + argv[argv.index("-m") + 1][-11:]
                        for argv in self.events("server-argv")[1:])
        self.assertEqual(loaded, ["1second.gguf", "2rafter.gguf"])

    def test_an_agent_writes_a_draft_with_a_tool_call(self) -> None:
        self.agent("claude", """cat > "$0.prompt"; echo 'Here: {"content": "", "tool_calls": """
                             """[{"name": "write", "arguments": {"path": "a.py"}}]}' """)
        self.env(FAKE_DRAFTS=["plain"], FAKE_P0={"plain": 0.2, "tool call": 0.9})
        made = self.stack(workers=(self.drafter, "claude"))
        body = dict(ASK, tools=[{"type": "function", "function": {"name": "write", "parameters": {}}}])
        reply = made.complete(body)
        self.assertEqual(reply["kilix_stack"]["workers"][1], "claude")
        call = reply["choices"][0]["message"]["tool_calls"][0]
        self.assertEqual((call["function"]["name"], json.loads(call["function"]["arguments"])),
                         ("write", {"path": "a.py"}))
        self.assertEqual(reply["choices"][0]["finish_reason"], "tool_calls")
        with open(os.path.join(self.tmp.name, "claude.prompt"), encoding="utf-8") as handle:
            prompt = handle.read()
        self.assertIn('"name": "write"', prompt)
        self.assertIn("[user]\nWrite add(a, b).", prompt)

    def test_an_agent_that_fails_costs_only_its_draft(self) -> None:
        self.agent("codex", "cat > /dev/null; echo broken >&2; exit 3")
        self.env(FAKE_DRAFTS=["plain"])
        err = io.StringIO()
        with redirect_stderr(err):
            record = self.stack(workers=(self.drafter, "codex/gpt-x")).complete(ASK)["kilix_stack"]
        self.assertEqual((record["chosen"], record["drafts"]), (0, 1))
        self.assertIn("codex exited 3: broken", err.getvalue())

    def test_an_agent_judges_with_a_number(self) -> None:
        self.agent("claude", 'if grep -q "^dos$"; then echo 85; else echo "About 10."; fi')
        self.env(FAKE_DRAFTS=["uno", "dos", "tres"])
        args = self.args("serve", "--worker", f"{self.drafter}:3")
        args.judge = "claude"
        stack.prepare(args)
        with stack.Stack(args) as made:
            record = made.complete(ASK)["kilix_stack"]
        self.assertEqual((record["chosen"], record["ratings"]), (1, [0.1, 0.85, 0.1]))
        self.assertEqual(len(self.events("server-argv")), 1)  # no judge engine was started

    def test_agent_text_that_is_not_the_json_form_is_the_content(self) -> None:
        self.assertEqual(stack.agent_message("def f(): return {}", 0)["content"], "def f(): return {}")
        self.assertEqual(stack.parse_worker("/models/a:b.gguf:3"), ("/models/a:b.gguf", 3))
        self.assertEqual(stack.parse_worker("claude/opus"), ("claude/opus", 1))

    # --- the endpoint ----------------------------------------------------------

    def test_the_endpoint_answers_plain_and_streamed(self) -> None:
        self.env(FAKE_DRAFTS=["uno", "dos", "tres"], FAKE_P0={"uno": 0.1, "dos": 0.8, "tres": 0.2})
        made = self.stack("--name", "pair")
        handler = type("H", (stack.Handler,), {"stack": made})
        httpd = stack.http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.addCleanup(httpd.server_close)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self.addCleanup(httpd.shutdown)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"

        def post(body: dict) -> bytes:
            request = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=30) as reply:
                return reply.read()

        self.assertEqual(json.loads(post(ASK))["choices"][0]["message"]["content"], "dos")
        lines = [line[6:] for line in post(dict(ASK, stream=True)).decode().split("\n\n") if line]
        self.assertEqual(lines[-1], "[DONE]")
        chunks = [json.loads(line) for line in lines[:-1]]
        self.assertEqual("".join(c["choices"][0]["delta"].get("content", "") for c in chunks), "dos")
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")
        with urllib.request.urlopen(base + "/v1/models", timeout=30) as reply:
            self.assertEqual(json.loads(reply.read())["data"][0]["id"], "pair")
        with urllib.request.urlopen(base + "/stack/status", timeout=30) as reply:
            self.assertEqual(json.loads(reply.read())["requests"], 2)

    def test_choices_lists_what_is_installed(self) -> None:
        catalogue = {"gen": {"title": "g", "file": "drafter.gguf", "size": 7, "sha256": "x", "family": "qwen3.5"},
                     "dec": {"title": "d", "file": "judge.gguf", "size": 7, "sha256": "x", "family": "d3"},
                     "gone": {"title": "n", "file": "absent.gguf", "size": 7, "sha256": "x"}}
        with mock.patch.dict(stack.models.MODELS, catalogue, clear=True), \
                mock.patch.dict(os.environ, {"KILIX_LLM_MODELS_DIR": self.tmp.name}), \
                mock.patch.object(stack.shutil, "which", lambda name: "/x" if name == "claude" else None):
            self.assertEqual(stack.choices(), {"workers": ["gen", "claude"], "judges": ["dec", "gen", "claude"]})

    def test_tool_calls_are_streamed_with_an_index(self) -> None:
        reply = {"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "content": None,
                 "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "ls", "arguments": "{}"}}]}}]}
        chunks = stack.stream_chunks(reply)
        self.assertEqual(chunks[1]["choices"][0]["delta"]["tool_calls"][0]["index"], 0)
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "tool_calls")
        self.assertIn("[tool call] ls({})", stack.reply_text(reply["choices"][0]["message"], 100))

    # --- fit -------------------------------------------------------------------

    def test_fit_reports_the_last_count_that_answered(self) -> None:
        self.env(FAKE_MAX_PARALLEL="2")
        # The fake engine counts one token a character, so the probe's input must be shorter here.
        patcher = mock.patch.object(stack, "CHARS_PER_TOKEN", 0.5)
        patcher.start()
        self.addCleanup(patcher.stop)
        args = self.args("fit", "--worker", self.drafter, "--ctx", "1000", "--max-slots", "5", "--probe-tokens", "40", "--json")
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(io.StringIO()):
            self.assertEqual(args.func(args), 0)
        result = json.loads(out.getvalue())
        self.assertEqual(result["max_slots"], 2)
        self.assertEqual([row["fits"] for row in result["tried"]], [True, True, False])
        self.assertEqual(len(self.events("/completion")), 4)  # the judge read two inputs at each count that loaded


if __name__ == "__main__":
    unittest.main()
