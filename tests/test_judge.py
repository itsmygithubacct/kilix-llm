"""`kilix-llm judge` end to end against a fake engine and a fake sizer.

The fake engine has one token per character and the template
"<|s|>{system}<|u|>{user}<|a|>", so the contract prompt costs
653 + 15 = 668 tokens and every count below is exact.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

from kilix_llm import judge, models, runtime
from kilix_llm.cli import build_parser

FAKE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fake_engine.py")
OVERHEAD = 668
BASE_BYTES = b"stock-base-weights"
ADAPTER_BYTES = b"binary-adapter"
PIN_COMMIT = "8ea290247c87ced2ab245b056ffe96dbcf90d36c"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class JudgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = self.tmp.name
        self.models_dir = os.path.join(root, "models")
        os.makedirs(self.models_dir)
        self.log = os.path.join(root, "fake.log")
        self.server = self._wrapper("llama-server", "server")
        self.sizer = self._wrapper("sizer", "sizer")
        self.base = os.path.join(self.models_dir, "Qwen3.5-0.8B-Q8_0.gguf")
        with open(self.base, "wb") as handle:
            handle.write(BASE_BYTES)
        catalogue = {
            "qwen3.5-0.8b": dict(models.MODELS["qwen3.5-0.8b"], size=len(BASE_BYTES), sha256=sha(BASE_BYTES)),
            "qwen-0.8b-binary-01": dict(models.MODELS["qwen-0.8b-binary-01"]),
            "test-binary-02": dict(models.MODELS["qwen-0.8b-binary-01"], title="Test binary 02",
                                   file="test-binary-02.gguf", size=len(ADAPTER_BYTES),
                                   sha256=sha(ADAPTER_BYTES), url="https://example.invalid/b02.gguf"),
        }
        self.sizing = {"max_context_tokens": 4096, "max_input_tokens": 3000, "limit_reason": "memory"}
        for patcher in (
            mock.patch.dict(models.MODELS, catalogue),
            mock.patch.dict(os.environ, {"KILIX_LLM_MODELS_DIR": self.models_dir, "FAKE_LOG": self.log,
                                         "FAKE_SIZER_JSON": json.dumps(self.sizing)}),
            mock.patch.object(runtime, "require_vendor"),
            mock.patch.object(runtime, "ensure_built"),
            mock.patch.object(runtime, "binary", return_value=self.server),
            mock.patch.object(runtime, "physical_cores", return_value=4),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def _wrapper(self, name: str, kind: str) -> str:
        path = os.path.join(self.tmp.name, name)
        with open(path, "w") as handle:
            handle.write(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE}" {kind} "$@"\n')
        os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)
        return path

    def judge(self, *argv: str, env: dict | None = None, stdin: bytes | None = None) -> tuple[int, str, str]:
        args = build_parser().parse_args(["judge", "--sizer", self.sizer, *argv])
        out, err = io.StringIO(), io.StringIO()
        fake_stdin = io.TextIOWrapper(io.BytesIO(stdin or b""), encoding="utf-8")
        with mock.patch.dict(os.environ, env or {}), mock.patch.object(sys, "stdin", fake_stdin), \
                redirect_stdout(out), redirect_stderr(err):
            code = args.func(args)
        return code, out.getvalue(), err.getvalue()

    def events(self, kind: str) -> list:
        if not os.path.exists(self.log):
            return []
        with open(self.log, encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle]
        return [row["value"] for row in rows if row["kind"] == kind]

    # --- answers ---------------------------------------------------------------

    def test_yes_prints_0_and_exits_0_with_the_contract_request(self) -> None:
        code, out, err = self.judge("--model-id", "qwen3.5-0.8b", "Is 2+2 equal to 4?")
        self.assertEqual((code, out), (0, "0\n"), err)
        argv = self.events("server-argv")[0]
        self.assertEqual(argv[argv.index("-m") + 1], self.base)
        self.assertNotIn("--lora", argv)
        # 653 system + 18 input + 512 template allowance, under the 4096 limit.
        for flag, value in (("-c", "1183"), ("-t", "4"), ("--parallel", "1"), ("-fa", "on"),
                            ("--fit", "off"), ("--cache-ram", "0"), ("--reasoning", "off"),
                            ("-ngl", "0"), ("--device", "none")):
            self.assertEqual(argv[argv.index(flag) + 1], value, flag)
        self.assertTrue(argv[argv.index("--host") + 1].endswith("/engine.sock"))
        self.assertIn("--no-warmup", argv)
        rendered = self.events("/apply-template")
        self.assertEqual(rendered[-1]["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(rendered[-1]["messages"][0], {"role": "system", "content": judge.SYSTEM_PROMPT})
        self.assertEqual(rendered[-1]["messages"][1], {"role": "user", "content": "Is 2+2 equal to 4?"})
        body = self.events("/completion")[0]
        self.assertEqual(body["n_predict"], 1)
        self.assertEqual(body["temperature"], 0)
        self.assertEqual(body["grammar"], 'root ::= "0" | "1"')
        self.assertIs(body["post_sampling_probs"], False)
        self.assertIs(body["cache_prompt"], False)
        self.assertEqual(len(body["prompt"]), OVERHEAD + 18)
        sizer = self.events("sizer-argv")[0]
        self.assertEqual(sizer, [
            "recommend", "binary-judge", "--model", "qwen3.5-0.8b-q8_0",
            "--model-sha256", sha(BASE_BYTES), "--engine", f"llama.cpp@{PIN_COMMIT}",
            "--backend", "cpu", "--adapter-bytes", "0", "--json"])

    def test_no_exits_1_and_json_gives_p0_p1_from_logprobs(self) -> None:
        env = {"FAKE_ANSWER": "1", "FAKE_LOGPROBS": '[["1", -0.5], ["0", -2.0], ["No", -6.0]]'}
        code, out, err = self.judge("--model-id", "qwen3.5-0.8b", "--json", "Is 2+2 equal to 5?", env=env)
        self.assertEqual(code, 1, err)
        self.assertEqual(json.loads(out), {
            "answer": 1, "p0": 0.182426, "p1": 0.817574, "probs_bounded": False,
            "model_id": "qwen3.5-0.8b", "input_tokens": 18, "max_input_tokens": 4096 - 1 - OVERHEAD,
            "limit_reason": "memory"})

    def test_probability_of_an_unlisted_answer_is_bounded(self) -> None:
        reply = {"completion_probabilities": [{"top_logprobs": [
            {"token": "0", "logprob": -0.01}, {"token": "Yes", "logprob": -5.0}]}]}
        self.assertEqual(judge.answer_probs(reply), (0.99324, 0.00676, True))
        self.assertEqual(judge.answer_probs({}), (None, None, False))

    def test_adapter_loads_with_lora_over_the_base(self) -> None:
        adapter = os.path.join(self.models_dir, "test-binary-02.gguf")
        with open(adapter, "wb") as handle:
            handle.write(ADAPTER_BYTES)
        code, out, err = self.judge("--model-id", "test-binary-02", "Did the tests pass?")
        self.assertEqual((code, out), (0, "0\n"), err)
        argv = self.events("server-argv")[0]
        self.assertEqual(argv[argv.index("-m") + 1], self.base)
        self.assertEqual(argv[argv.index("--lora") + 1], adapter)
        sizer = self.events("sizer-argv")[0]
        self.assertEqual(sizer[sizer.index("--adapter-bytes") + 1], str(len(ADAPTER_BYTES)))

    def test_stdin_input_and_base_path_for_the_stock_model(self) -> None:
        elsewhere = os.path.join(self.tmp.name, "mine.gguf")
        os.rename(self.base, elsewhere)
        code, out, err = self.judge("--model-id", "qwen3.5-0.8b", "--base", elsewhere, "--stdin",
                                    stdin="Did the build fail?\n".encode())
        self.assertEqual((code, out), (0, "0\n"), err)
        argv = self.events("server-argv")[0]
        self.assertEqual(argv[argv.index("-m") + 1], elsewhere)
        self.assertEqual(self.events("/apply-template")[-1]["messages"][1]["content"], "Did the build fail?\n")
        self.assertEqual(os.listdir(self.models_dir), [])

    # --- the limit ---------------------------------------------------------------

    def over_limit(self) -> dict:
        # 20 input tokens fit: 689 - 1 output - 668 template.
        return {"FAKE_SIZER_JSON": json.dumps(
            {"max_context_tokens": OVERHEAD + 21, "max_input_tokens": 20, "limit_reason": "memory"})}

    def test_input_over_the_limit_exits_3_with_nothing_on_stdout(self) -> None:
        code, out, err = self.judge("--model-id", "qwen3.5-0.8b", "abcdefghijklmnopqrstuvwxyz0123",
                                    env=self.over_limit())
        self.assertEqual((code, out), (3, ""))
        self.assertIn("the input is 30 tokens; on this machine qwen3.5-0.8b takes at most 20 "
                      "(limit: memory, from plebian-model-sizer)", err)
        self.assertEqual(self.events("/completion"), [])
        argv = self.events("server-argv")[0]
        self.assertEqual(argv[argv.index("-c") + 1], str(OVERHEAD + 21))

    def test_truncate_keeps_the_head_or_the_tail_and_says_so(self) -> None:
        for keep, kept in (("head", "abcdefghijklmnopqrst"), ("tail", "klmnopqrstuvwxyz0123")):
            with self.subTest(keep=keep):
                os.unlink(self.log) if os.path.exists(self.log) else None
                code, out, err = self.judge("--model-id", "qwen3.5-0.8b", "--truncate", keep,
                                            "abcdefghijklmnopqrstuvwxyz0123", env=self.over_limit())
                self.assertEqual((code, out), (0, "0\n"), err)
                self.assertIn(f"input truncated from 30 to 20 tokens (kept the {keep})", err)
                prompt = "".join(chr(t) for t in self.events("/completion")[0]["prompt"])
                self.assertTrue(prompt.endswith(f"<|u|>{kept}<|a|>"), prompt[-40:])

    def test_no_room_at_all_refuses_before_the_engine_starts(self) -> None:
        env = {"FAKE_SIZER_JSON": json.dumps(
            {"max_context_tokens": 300, "max_input_tokens": 0, "limit_reason": "memory"})}
        code, out, err = self.judge("--model-id", "qwen3.5-0.8b", "Is it on?", env=env)
        self.assertEqual((code, out), (3, ""))
        self.assertIn("has no room for any input", err)
        self.assertEqual(self.events("server-argv"), [])

    # --- tool failures -------------------------------------------------------------

    def test_sizer_refusal_and_missing_sizer_exit_2(self) -> None:
        env = {"FAKE_SIZER_RC": "2", "FAKE_SIZER_JSON": json.dumps(
            {"schema": "plebian.models.error/v1-development",
             "error": "profile refused: model sha256 does not match"})}
        code, out, err = self.judge("--model-id", "qwen3.5-0.8b", "Is it on?", env=env)
        self.assertEqual((code, out), (2, ""))
        self.assertIn("profile refused: model sha256 does not match", err)
        self.assertEqual(self.events("server-argv"), [])
        args = build_parser().parse_args(["judge", "--model-id", "qwen3.5-0.8b", "--sizer",
                                          os.path.join(self.tmp.name, "absent"), "x"])
        err_buffer = io.StringIO()
        with redirect_stderr(err_buffer), redirect_stdout(io.StringIO()) as out_buffer, \
                mock.patch.dict(os.environ, {"KILIX_LLM_SIZER": ""}):
            self.assertEqual(args.func(args), 2)
        self.assertEqual(out_buffer.getvalue(), "")
        self.assertIn("plebian-model-sizer was not found", err_buffer.getvalue())

    def test_engine_crash_and_out_of_contract_answer_exit_5(self) -> None:
        for answer, message in (("crash", "the engine exited during startup (status 3)"),
                                ("x", "the engine answered 'x', not 0 or 1")):
            with self.subTest(answer=answer):
                code, out, err = self.judge("--model-id", "qwen3.5-0.8b", "Is it on?",
                                            env={"FAKE_ANSWER": answer})
                self.assertEqual((code, out), (5, ""))
                self.assertIn(message, err)
                if answer == "crash":
                    # The engine's own last log lines say why.
                    self.assertIn("fake engine: failed to load model", err)

    def test_usage_errors_exit_2(self) -> None:
        for argv, message in ((["--model-id", "qwen3.5-0.8b"], "not both or neither"),
                              (["--model-id", "qwen3.5-0.8b", "--stdin", "text"], "not both or neither"),
                              (["--model-id", "qwen3.5-0.8b", "   "], "the input is empty")):
            with self.subTest(argv=argv):
                code, out, err = self.judge(*argv)
                self.assertEqual((code, out), (2, ""))
                self.assertIn(message, err)

    def test_unreleased_default_adapter_names_the_stock_base(self) -> None:
        code, out, err = self.judge("Is it on?")
        self.assertEqual((code, out), (2, ""))
        self.assertIn("qwen-0.8b-binary-01) is not released yet", err)
        self.assertIn("--model-id qwen3.5-0.8b", err)

    def test_digest_mismatch_exits_2(self) -> None:
        with open(self.base, "wb") as handle:
            handle.write(b"stock-base-weightX")
        code, out, err = self.judge("--model-id", "qwen3.5-0.8b", "Is it on?")
        self.assertEqual((code, out), (2, ""))
        self.assertIn(f"sha256 is {sha(b'stock-base-weightX')}, expected {sha(BASE_BYTES)}", err)
        self.assertEqual(self.events("server-argv"), [])

    # --- lazy install ----------------------------------------------------------------

    def test_missing_files_are_described_and_not_downloaded_without_yes(self) -> None:
        os.unlink(self.base)
        with mock.patch("kilix_llm.install._curl") as curl:
            code, out, err = self.judge("--model-id", "test-binary-02", "Is it on?")
        self.assertEqual((code, out), (2, ""))
        curl.assert_not_called()
        self.assertIn("Test binary 02 (test-binary-02): 14 bytes", err)
        self.assertIn("from https://example.invalid/b02.gguf", err)
        self.assertIn("Qwen3.5 0.8B Q8 (GPU) (qwen3.5-0.8b): 18 bytes", err)
        self.assertIn("/ggml-org/Qwen3.5-0.8B-GGUF/resolve/807dfd9a6fadbf703207aac3bfb0b2b3040d077c/", err)
        self.assertIn("licence: Apache-2.0. Qwen3.5-0.8B by Alibaba Cloud (Qwen), Apache-2.0", err)
        self.assertIn("total 32 bytes", err)
        self.assertIn("rerun with --yes", err)

    def test_yes_downloads_then_answers(self) -> None:
        os.unlink(self.base)

        def curl(url: str, dest: str, resume: bool) -> None:
            with open(dest, "wb") as handle:
                handle.write(ADAPTER_BYTES if url.endswith("b02.gguf") else BASE_BYTES)

        with mock.patch("kilix_llm.install._curl", side_effect=curl) as fetch:
            code, out, err = self.judge("--model-id", "test-binary-02", "--yes", "Is it on?")
        self.assertEqual((code, out), (0, "0\n"), err)
        self.assertEqual(fetch.call_count, 2)
        self.assertEqual(sorted(os.listdir(self.models_dir)),
                         ["Qwen3.5-0.8B-Q8_0.gguf", "test-binary-02.gguf"])

    def test_named_files_are_never_downloaded(self) -> None:
        with mock.patch("kilix_llm.install._curl") as curl:
            code, out, err = self.judge("--model-id", "qwen3.5-0.8b", "--yes", "--base",
                                        os.path.join(self.tmp.name, "absent.gguf"), "Is it on?")
        self.assertEqual((code, out), (2, ""))
        curl.assert_not_called()
        self.assertIn("absent.gguf is missing", err)


if __name__ == "__main__":
    unittest.main()
