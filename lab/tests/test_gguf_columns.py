"""Exact column edits on a small GGUF written here. Needs numpy; no checkpoint."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kilix_llm_lab import LabError
from kilix_llm_lab.gguf_columns import head_columns, normalize_edits


def _modules():
    try:
        import numpy
    except ImportError:
        return None, None
    try:
        from kilix_llm_lab.output_row import _import_gguf

        return numpy, _import_gguf()
    except LabError:
        return None, None


def _write_model(path: Path, numpy, gguf) -> dict:
    """Two writing matrices: Q8_0 (6 rows x 96 inputs = 3 blocks) and F32 (3 x 8)."""
    rng = numpy.random.default_rng(7)
    q8 = rng.standard_normal((6, 96)).astype(numpy.float32)
    f32 = rng.standard_normal((3, 8)).astype(numpy.float32)
    writer = gguf.GGUFWriter(str(path), "qwen35")
    raw = gguf.quantize(q8, gguf.GGMLQuantizationType.Q8_0)
    writer.add_tensor("blk.0.ffn_down.weight", raw, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
    writer.add_tensor("blk.0.ssm_out.weight", f32)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return {"q8": gguf.dequantize(raw, gguf.GGMLQuantizationType.Q8_0).reshape(6, 96), "f32": f32}


def _read(path: Path, name: str, numpy, gguf):
    reader = gguf.GGUFReader(str(path), "r")
    tensor = next(item for item in reader.tensors if item.name == name)
    n_in, n_out = int(tensor.shape[0]), int(tensor.shape[1])
    values = gguf.dequantize(numpy.asarray(tensor.data), tensor.tensor_type).reshape(n_out, n_in)
    del reader
    return values


class NormalizeTest(unittest.TestCase):
    def test_refusals(self):
        with self.assertRaises(LabError):
            normalize_edits([])
        with self.assertRaises(LabError):
            normalize_edits([{"tensor": "t", "op": "flip", "columns": [1]}])
        with self.assertRaises(LabError):
            normalize_edits([{"tensor": "t", "op": "zero", "columns": []}])
        with self.assertRaises(LabError):
            normalize_edits([{"tensor": "t", "op": "scale", "columns": [1]}])
        with self.assertRaises(LabError):
            normalize_edits([{"tensor": "t", "op": "zero", "columns": [1, 1]}])
        with self.assertRaises(LabError):
            normalize_edits([{"tensor": "t", "op": "zero", "columns": [1, 2]},
                             {"tensor": "t", "op": "scale", "columns": [2], "factor": 0.5}])

    def test_head_columns(self):
        self.assertEqual(head_columns(2, 4), [8, 9, 10, 11])


class EditTest(unittest.TestCase):
    def setUp(self):
        self.numpy, self.gguf = _modules()
        if self.numpy is None:
            self.skipTest("numpy or the vendored gguf reader is not available")
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.source = self.dir / "parent.gguf"
        self.before = _write_model(self.source, self.numpy, self.gguf)

    def tearDown(self):
        self.tmp.cleanup()

    def _edit(self, edits):
        from kilix_llm_lab.gguf_columns import edit_gguf_columns

        dest = self.dir / "child.gguf"
        manifest = edit_gguf_columns(str(self.source), str(dest), edits)
        return dest, manifest

    def test_zero_a_neuron_column_exactly(self):
        dest, manifest = self._edit([{"tensor": "blk.0.ffn_down.weight", "op": "zero", "columns": [5, 70]}])
        after = _read(dest, "blk.0.ffn_down.weight", self.numpy, self.gguf)
        self.assertTrue((after[:, [5, 70]] == 0).all())
        keep = [c for c in range(96) if c not in (5, 70)]
        self.assertTrue(self.numpy.array_equal(after[:, keep], self.before["q8"][:, keep]))
        self.assertTrue(manifest["edits"][0]["untouched_columns_identical"])
        self.assertFalse(manifest["source_opened_for_write"])

    def test_scale_a_whole_block_head(self):
        dest, manifest = self._edit([{"tensor": "blk.0.ffn_down.weight", "op": "scale",
                                      "columns": head_columns(1, 32), "factor": 0.25}])
        after = _read(dest, "blk.0.ffn_down.weight", self.numpy, self.gguf)
        want = self.before["q8"][:, 32:64] * 0.25
        self.assertLess(float(self.numpy.max(self.numpy.abs(after[:, 32:64] - want))), 1e-3)
        self.assertTrue(self.numpy.array_equal(after[:, :32], self.before["q8"][:, :32]))

    def test_partial_block_scale_keeps_neighbours(self):
        dest, _ = self._edit([{"tensor": "blk.0.ffn_down.weight", "op": "scale",
                               "columns": [3, 4], "factor": 0.5}])
        after = _read(dest, "blk.0.ffn_down.weight", self.numpy, self.gguf)
        keep = [c for c in range(96) if c not in (3, 4)]
        self.assertTrue(self.numpy.array_equal(after[:, keep], self.before["q8"][:, keep]))

    def test_partial_block_scale_up_that_clips_is_refused(self):
        with self.assertRaises(LabError):
            self._edit([{"tensor": "blk.0.ffn_down.weight", "op": "scale", "columns": [3], "factor": 50.0}])

    def test_f32_columns(self):
        dest, _ = self._edit([{"tensor": "blk.0.ssm_out.weight", "op": "zero", "columns": [0, 7]}])
        after = _read(dest, "blk.0.ssm_out.weight", self.numpy, self.gguf)
        self.assertTrue((after[:, [0, 7]] == 0).all())
        self.assertTrue(self.numpy.array_equal(after[:, 1:7], self.before["f32"][:, 1:7]))

    def test_unknown_tensor_and_wide_column(self):
        with self.assertRaises(LabError):
            self._edit([{"tensor": "blk.9.ffn_down.weight", "op": "zero", "columns": [0]}])
        with self.assertRaises(LabError):
            self._edit([{"tensor": "blk.0.ffn_down.weight", "op": "zero", "columns": [96]}])

    def test_command_writes_a_copy(self):
        import contextlib
        import io
        import json

        from kilix_llm_lab.cli import main

        edits = self.dir / "edits.json"
        edits.write_text(json.dumps([{"tensor": "blk.0.ffn_down.weight", "op": "zero", "columns": [1]}]))
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = main(["gguf-columns", "--source", str(self.source), "--output",
                         str(self.dir / "cli.gguf"), "--edits", str(edits)])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stdout.getvalue())["edits"][0]["max_abs_error"], 0.0)

    def test_existing_output_is_refused(self):
        (self.dir / "child.gguf").write_bytes(b"x")
        with self.assertRaises(LabError):
            self._edit([{"tensor": "blk.0.ffn_down.weight", "op": "zero", "columns": [0]}])


if __name__ == "__main__":
    unittest.main()
