"""Whole-tensor replacement on a small GGUF written here. Needs numpy; no checkpoint."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kilix_llm_lab import LabError
from kilix_llm_lab.gguf_tensors import normalize_replacements, replace_gguf_tensors


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
    """A Q8_0 writing matrix (6 rows x 96 inputs) quantized from float, an F32 one, and an untouched tensor."""
    rng = numpy.random.default_rng(11)
    q8 = rng.standard_normal((6, 96)).astype(numpy.float32)
    f32 = rng.standard_normal((3, 8)).astype(numpy.float32)
    other = rng.standard_normal((4, 32)).astype(numpy.float32)
    writer = gguf.GGUFWriter(str(path), "qwen35")
    raw = gguf.quantize(q8, gguf.GGMLQuantizationType.Q8_0)
    writer.add_tensor("blk.0.ffn_down.weight", raw, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
    writer.add_tensor("blk.0.ssm_out.weight", f32)
    writer.add_tensor("blk.1.ffn_down.weight", other)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return {"q8": q8, "f32": f32, "other": other}


def _bytes(path: Path, name: str, numpy, gguf) -> bytes:
    reader = gguf.GGUFReader(str(path), "r")
    tensor = next(item for item in reader.tensors if item.name == name)
    data = numpy.asarray(tensor.data).view(numpy.uint8).tobytes()
    del reader
    return data


class NormalizeTest(unittest.TestCase):
    def test_refusals(self):
        with self.assertRaises(LabError):
            normalize_replacements([])
        with self.assertRaises(LabError):
            normalize_replacements([{"tensor": "t"}])
        with self.assertRaises(LabError):
            normalize_replacements([{"tensor": "t", "values": "a.npy"}, {"tensor": "t", "values": "b.npy"}])
        with self.assertRaises(LabError):
            normalize_replacements([{"tensor": "t", "values": "a.npy", "base": 3}])


class ReplaceTest(unittest.TestCase):
    def setUp(self):
        self.numpy, self.gguf = _modules()
        if self.numpy is None:
            self.skipTest("numpy or the vendored gguf reader is not available")
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.source = self.dir / "parent.gguf"
        self.values = _write_model(self.source, self.numpy, self.gguf)

    def tearDown(self):
        self.tmp.cleanup()

    def _save(self, name, array):
        path = self.dir / name
        self.numpy.save(path, array)
        return str(path)

    def test_q8_replacement_matches_quantizing_the_edit(self):
        np, gguf = self.numpy, self.gguf
        edited = self.values["q8"].copy()
        edited[:, 40:44] -= 0.5
        manifest = replace_gguf_tensors(str(self.source), str(self.dir / "child.gguf"), [{
            "tensor": "blk.0.ffn_down.weight",
            "values": self._save("edit.npy", edited),
            "base": self._save("base.npy", self.values["q8"]),
        }])
        report = manifest["tensors"][0]
        self.assertTrue(report["base_round_trip"])
        # A new block maximum rescales the whole 32-value block (columns 32..63), as a fresh conversion would.
        self.assertGreaterEqual(report["changed_columns"], 4)
        self.assertLessEqual(report["changed_columns"], 32)
        expected = np.asarray(gguf.quantize(edited, gguf.GGMLQuantizationType.Q8_0)).view(np.uint8).tobytes()
        self.assertEqual(_bytes(self.dir / "child.gguf", "blk.0.ffn_down.weight", np, gguf), expected)
        for name in ("blk.0.ssm_out.weight", "blk.1.ffn_down.weight"):
            self.assertEqual(_bytes(self.dir / "child.gguf", name, np, gguf), _bytes(self.source, name, np, gguf))

    def test_f32_replacement_is_exact(self):
        np, gguf = self.numpy, self.gguf
        edited = self.values["f32"] * 2.0
        replace_gguf_tensors(str(self.source), str(self.dir / "child.gguf"), [{
            "tensor": "blk.0.ssm_out.weight", "values": self._save("edit.npy", edited)}])
        self.assertEqual(_bytes(self.dir / "child.gguf", "blk.0.ssm_out.weight", np, gguf),
                         edited.astype(np.float32).tobytes())

    def test_wrong_base_is_refused(self):
        wrong = self.values["q8"] + 0.25
        with self.assertRaises(LabError):
            replace_gguf_tensors(str(self.source), str(self.dir / "child.gguf"), [{
                "tensor": "blk.0.ffn_down.weight",
                "values": self._save("edit.npy", self.values["q8"]),
                "base": self._save("base.npy", wrong)}])
        self.assertFalse((self.dir / "child.gguf").exists())

    def test_shape_and_name_refusals(self):
        with self.assertRaises(LabError):
            replace_gguf_tensors(str(self.source), str(self.dir / "a.gguf"), [{
                "tensor": "blk.0.ffn_down.weight", "values": self._save("bad.npy", self.numpy.zeros((96, 6)))}])
        with self.assertRaises(LabError):
            replace_gguf_tensors(str(self.source), str(self.dir / "b.gguf"), [{
                "tensor": "blk.9.ffn_down.weight", "values": self._save("x.npy", self.numpy.zeros((6, 96)))}])

    def test_command_writes_a_copy(self):
        from kilix_llm_lab.cli import main

        spec = self.dir / "replace.json"
        spec.write_text(json.dumps([{"tensor": "blk.0.ssm_out.weight",
                                     "values": self._save("edit.npy", self.values["f32"] + 1.0)}]))
        out = self.dir / "cmd.gguf"
        self.assertEqual(main(["gguf-tensors", "--source", str(self.source), "--output", str(out),
                               "--replace", str(spec)]), 0)
        self.assertTrue(out.is_file())


if __name__ == "__main__":
    unittest.main()
