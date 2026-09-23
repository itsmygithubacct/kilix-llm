"""Check the host boundary with a real child process, without model weights."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from kilix_llm import paths
from kilix_llm.help_query import command


class HelpQueryTests(unittest.TestCase):
    def test_child_receives_exact_question_and_private_data_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            module = Path(temporary) / "module with spaces"
            module.mkdir()
            child = module / "kilix-help-llm"
            child.write_text("#!/usr/bin/env python3\nimport json,os,sys\n"
                             "print(json.dumps({'argv':sys.argv[1:],"
                             "'data':os.environ['GPU_TERMINAL_HOME'],"
                             "'quality':'unqualified'}))\n")
            child.chmod(0o700)
            question = "--run $(touch should-not-exist); How do I list panes?"
            argv = [os.path.join(paths.REPO_DIR, "kilix-llm"), "ask-help",
                    "--help-module", str(module), "--answer-run", "answer-r1",
                    "--rank-run", "rank-r1", "--", question]
            result = subprocess.run(argv, capture_output=True, text=True,
                                    env={**os.environ, "GPU_TERMINAL_HOME": temporary})
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload['argv'], ['ask', '--run', 'answer-r1', '--rank-run',
                                               'rank-r1', '--max-new-tokens', '96', '--', question])
            self.assertEqual(payload['data'], temporary)
            self.assertEqual(payload['quality'], 'unqualified')
            child.write_text("#!/bin/sh\nprintf 'fixture admission refused\\n' >&2\nexit 7\n")
            failed = subprocess.run(argv, capture_output=True, text=True)
            self.assertEqual(failed.returncode, 7)
            self.assertEqual(failed.stdout, '')
            self.assertEqual(failed.stderr, 'fixture admission refused\n')

    def test_malformed_request_rejected_before_module_access(self):
        for answer, rank, question, limit in (
                ('../run', 'rank', 'How?', 96), ('answer', '--rank', 'How?', 96),
                ('answer', 'rank', ' ', 96), ('answer', 'rank', 'x' * 2001, 96),
                ('answer', 'rank', 'nul\0', 96), ('answer', 'rank', 'How?', 0)):
            with self.subTest(answer=answer, rank=rank, limit=limit):
                with self.assertRaises(ValueError):
                    command('/nonexistent', answer, rank, question, limit)

    def test_missing_module_reports_actionable_error(self):
        with self.assertRaisesRegex(ValueError, 'executable kilix-help-llm'):
            command('/nonexistent', 'answer', 'rank', 'How?')

    def test_explicit_provider_is_a_single_argument(self):
        with tempfile.TemporaryDirectory() as directory:
            launcher = Path(directory) / 'kilix-help-llm'
            launcher.touch(mode=0o700)
            argv = command(directory, 'answer', 'rank', 'How?', sizer='/provider with spaces')
            self.assertEqual(argv[-4:], ['--sizer', '/provider with spaces', '--', 'How?'])
