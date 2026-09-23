"""Exercise calibrated host admission and user-visible fallback without weights."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from kilix_llm import help_acceptance, paths


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


class HelpAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.home = Path(self.temporary.name)
        self.environment = patch.dict(os.environ, {"GPU_TERMINAL_HOME": str(self.home)})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.root = self.home / "kilix-help-llm"
        self.answer_name = "operations-answer-06-r9"
        self.rank_name = "operations-rank-06-r1"
        runs = {}
        for name, task in ((self.answer_name, "answer"), (self.rank_name, "rank")):
            report = {"task": task, "dataset_sha256": "dataset", "context": 384}
            digest = write(self.root / "runs" / name / "run.json", report)
            runs[name] = {"report_sha256": digest, "report": report}
        self.freeze = {"dataset_sha256": "dataset", "source_revision": "revision", "runs": runs}
        freeze_hash = write(self.root / "qualification/operations-r9-freeze.json", self.freeze)
        self.policy = {
            "schema": "kilix.help-llm.calibration-threshold/v1",
            "freeze_sha256": freeze_hash,
            "primary_answer_run": self.answer_name,
            "primary_rank_run": self.rank_name,
            "fallback": help_acceptance.FALLBACK,
            "threshold_inclusive": True,
            "threshold": 2.0,
            "test_opened": False,
        }
        self.policy_file = self.root / "qualification/threshold.json"
        write(self.policy_file, self.policy)
        self.result = {
            "answer_run": self.answer_name,
            "answer_run_sha256": help_acceptance._digest(runs[self.answer_name]["report"]),
            "dataset_sha256": "dataset",
            "source_revision": "revision",
            "source": {"id": "closing:1", "text": "Static source."},
            "answer": "Use the listed command. [closing:1]",
            "draft": None,
            "citation_valid": True,
            "explicit_refusal": False,
            "abstained": False,
            "ranking": {"run": self.rank_name,
                        "run_sha256": help_acceptance._digest(runs[self.rank_name]["report"]),
                        "residency": "sequential", "order": ["closing:1"], "scores": [2.0],
                        "calibrated": False},
        }

    def test_inclusive_threshold_and_private_diagnostics(self):
        policy, freeze = help_acceptance.load_policy(
            str(self.policy_file), self.answer_name, self.rank_name)
        visible = help_acceptance.apply(self.result, policy, freeze)
        self.assertEqual(visible["answer"], self.result["answer"])
        self.assertTrue(visible["acceptance"]["model_answer_delivered"])
        self.assertNotIn("draft", visible)
        path = help_acceptance.save_diagnostic(self.result, str(self.policy_file))
        self.assertEqual(json.loads(path.read_text())["raw_result"], self.result)
        self.assertEqual(path.stat().st_mode & 0o077, 0)

    def test_low_score_and_bad_citation_use_fixed_fallback(self):
        policy, freeze = help_acceptance.load_policy(
            str(self.policy_file), self.answer_name, self.rank_name)
        for changes in ({"ranking": {**self.result["ranking"], "scores": [1.99]}},
                        {"citation_valid": False}):
            with self.subTest(changes=changes):
                visible = help_acceptance.apply({**self.result, **changes}, policy, freeze)
                self.assertEqual(visible["answer"], help_acceptance.FALLBACK)
                self.assertTrue(visible["acceptance"]["fallback_used"])
                self.assertNotIn("draft", visible)
        refused = {**self.result, "answer": "I do not know from the supplied excerpt.",
                   "citation_valid": False, "explicit_refusal": True}
        visible = help_acceptance.apply(refused, policy, freeze)
        self.assertEqual(visible["answer"], refused["answer"])

    def test_tampered_run_and_wrong_source_rejected(self):
        policy, freeze = help_acceptance.load_policy(
            str(self.policy_file), self.answer_name, self.rank_name)
        with self.assertRaisesRegex(ValueError, "frozen sequential system"):
            help_acceptance.apply({**self.result, "source": {"id": "other"}}, policy, freeze)
        write(self.root / "runs" / self.rank_name / "run.json", {"task": "changed"})
        with self.assertRaisesRegex(ValueError, "run report changed"):
            help_acceptance.load_policy(str(self.policy_file), self.answer_name, self.rank_name)

    def test_cli_calibrated_boundary_hides_rejected_model_text(self):
        module = self.home / "module"
        module.mkdir()
        child = module / "kilix-help-llm"
        low_score = {**self.result,
                     "ranking": {**self.result["ranking"], "scores": [1.0]},
                     "answer": "A rejected model assertion. [closing:1]"}
        child.write_text("#!/usr/bin/env python3\nimport json\n"
                         f"print(json.dumps({low_score!r}))\n")
        child.chmod(0o700)
        command = [str(Path(paths.REPO_DIR) / "kilix-llm"), "ask-help",
                   "--help-module", str(module), "--answer-run", self.answer_name,
                   "--rank-run", self.rank_name,
                   "--calibration-policy", str(self.policy_file), "How do I find a pane?"]
        completed = subprocess.run(command, text=True, capture_output=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertNotIn("rejected model assertion", completed.stdout)
        visible = json.loads(completed.stdout)
        self.assertEqual(visible["answer"], help_acceptance.FALLBACK)
        diagnostic = json.loads(Path(visible["diagnostic_path"]).read_text())
        self.assertEqual(diagnostic["raw_result"], low_score)


if __name__ == "__main__":
    unittest.main()
