"""Corpus preparation and the llama-finetune plan. No model download."""

from __future__ import annotations

import io
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from kilix_llm_lab import LabError
from kilix_llm_lab.cli import main
from kilix_llm_lab.corpus import MAX_CORPUS_BYTES, prepare_corpus
from kilix_llm_lab.train import read_gguf_file_type

LAB = Path(__file__).resolve().parents[1]
NEEDLE = "/" + "home" + "/"


def _gguf(file_type: int, *, array: bool = False) -> bytes:
    def text(value: str) -> bytes:
        raw = value.encode("utf-8")
        return struct.pack("<Q", len(raw)) + raw

    fields = []
    count = 1
    if array:
        count += 1
        fields.extend(
            [
                text("extra.tags"),
                struct.pack("<I", 9),
                struct.pack("<I", 4),
                struct.pack("<Q", 2),
                struct.pack("<II", 3, 4),
            ]
        )
    fields.extend(
        [
            text("general.file_type"),
            struct.pack("<I", 4),
            struct.pack("<I", file_type),
        ]
    )
    return b"".join(
        [
            b"GGUF",
            struct.pack("<I", 3),
            struct.pack("<Q", 0),
            struct.pack("<Q", count),
            *fields,
        ]
    )


class CorpusTests(unittest.TestCase):
    def test_directory_jsonl_and_text_join_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus"
            corpus.mkdir()
            (corpus / "b.txt").write_text("beta\n", encoding="utf-8")
            (corpus / "a.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"text": "alpha"}),
                        json.dumps({"prompt": "Q", "completion": "A"}),
                        json.dumps(
                            {
                                "messages": [
                                    {"role": "user", "content": "hi"},
                                    {"role": "assistant", "content": "there"},
                                ]
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (corpus / "skip.dat").write_text("nope", encoding="utf-8")
            (corpus / ".hidden.txt").write_text("hidden", encoding="utf-8")
            dest = root / "prepared.txt"
            manifest = prepare_corpus(corpus, dest)
            self.assertEqual(dest.read_text(encoding="utf-8"), "alpha\n\nQ\nA\n\nhi\nthere\n\nbeta\n")
            self.assertEqual(manifest["documents"], 4)
            self.assertEqual([item["path"] for item in manifest["files"]], ["a.jsonl", "b.txt"])
            self.assertEqual(manifest["skipped"], [".hidden.txt", "skip.dat"])
            self.assertNotIn(NEEDLE, dest.read_text(encoding="utf-8"))
            sidecar = Path(str(dest) + ".manifest.json").read_text(encoding="utf-8")
            self.assertNotIn(NEEDLE, sidecar)

    def test_over_the_byte_cap_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "note.txt"
            source.write_text("hello", encoding="utf-8")
            with mock.patch("kilix_llm_lab.corpus.MAX_CORPUS_BYTES", 4):
                with self.assertRaises(LabError):
                    prepare_corpus(source, root / "out.txt")
        self.assertGreater(MAX_CORPUS_BYTES, 4)

    def test_plan_names_llama_finetune_and_does_not_write_a_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "note.txt").write_text("one token\n", encoding="utf-8")
            stdout = io.StringIO()
            with mock.patch("sys.stdout", stdout):
                code = main(
                    [
                        "train",
                        "--corpus",
                        str(root / "note.txt"),
                        "--text-out",
                        str(root / "prepared.txt"),
                        "--model",
                        str(root / "model-f32.gguf"),
                        "--epochs",
                        "2",
                        "--learning-rate",
                        "1e-5",
                    ]
                )
            self.assertEqual(code, 0)
            plan = json.loads(stdout.getvalue())
            argv = plan["argv"]
            self.assertFalse(plan["exec"])
            self.assertEqual(plan["corpus"]["documents"], 1)
            self.assertEqual(argv[argv.index("--file") + 1], str((root / "prepared.txt").resolve()))
            self.assertEqual(argv[argv.index("-c") + 1], "512")
            self.assertEqual(argv[argv.index("-b") + 1], "512")
            self.assertEqual(argv[argv.index("-ub") + 1], "512")
            self.assertEqual(argv[argv.index("-ngl") + 1], "999")
            self.assertEqual(argv[argv.index("--epochs") + 1], "2")
            self.assertEqual(argv[argv.index("--learning-rate") + 1], "1e-05")
            self.assertEqual(argv[argv.index("--optimizer") + 1], "adamw")
            self.assertEqual(Path(plan["output"]).name, "finetuned-model.gguf")
            self.assertFalse(Path(plan["output"]).exists())

    def test_run_refuses_a_quantized_gguf_before_exec(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "note.txt").write_text("alpha\n", encoding="utf-8")
            model = root / "model.gguf"
            model.write_bytes(_gguf(1, array=True))
            self.assertEqual(read_gguf_file_type(model), 1)
            stderr = io.StringIO()
            with mock.patch("sys.stderr", stderr):
                code = main(
                    [
                        "train",
                        "--corpus",
                        str(root / "note.txt"),
                        "--text-out",
                        str(root / "prepared.txt"),
                        "--model",
                        str(model),
                        "--output",
                        str(root / "out.gguf"),
                        "--binary",
                        str(root / "missing-llama-finetune"),
                        "--run",
                    ]
                )
            self.assertEqual(code, 1)
            self.assertIn("FP32", stderr.getvalue())
            self.assertFalse((root / "out.gguf").exists())

    def test_run_refuses_a_missing_trainer_for_an_fp32_gguf(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "note.txt").write_text("alpha\n", encoding="utf-8")
            model = root / "model.gguf"
            model.write_bytes(_gguf(0))
            self.assertEqual(read_gguf_file_type(model), 0)
            stderr = io.StringIO()
            with mock.patch("sys.stderr", stderr):
                code = main(
                    [
                        "train",
                        "--corpus",
                        str(root / "note.txt"),
                        "--text-out",
                        str(root / "prepared.txt"),
                        "--model",
                        str(model),
                        "--output",
                        str(root / "out.gguf"),
                        "--binary",
                        str(root / "missing-llama-finetune"),
                        "--run",
                    ]
                )
            self.assertEqual(code, 1)
            self.assertIn("not executable", stderr.getvalue())
            self.assertTrue((root / "prepared.txt").is_file())
            self.assertFalse((root / "out.gguf").exists())

    def test_owned_train_sources_have_no_home_path(self) -> None:
        for relative in (
            "src/kilix_llm_lab/corpus.py",
            "src/kilix_llm_lab/train.py",
            "docs/train.md",
        ):
            text = (LAB / relative).read_text(encoding="utf-8")
            self.assertNotIn(NEEDLE, text, relative)


class TrainScriptTests(unittest.TestCase):
    def test_train_without_a_corpus_exits(self) -> None:
        import subprocess

        completed = subprocess.run(
            [str(LAB / "scripts" / "lab.sh"), "train"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("corpus", completed.stderr.lower())
