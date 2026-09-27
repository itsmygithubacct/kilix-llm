"""Artifact identity, hostile manifests, and failed-download publication."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from kilix_llm.artifacts import Artifact, fetch


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.content = b"GGUF" + b"test-fixture" * 8
        self.manifest = self.root / "manifest.json"
        self.data = {"schema": "kilix.llm.artifact/v1", "id": "tiny-q8", "filename": "tiny.gguf",
                     "size": len(self.content), "sha256": hashlib.sha256(self.content).hexdigest(),
                     "url": "https://example.com/resolve/" + "a" * 40 + "/tiny.gguf",
                     "source_revision": "a" * 40, "quantization": "Q8_0", "license": "MIT"}
        self.manifest.write_text(json.dumps(self.data))
        self.artifact = Artifact.load(self.manifest)

    def download(self, content):
        response = io.BytesIO(content)
        opener = Mock()
        opener.open.return_value = response
        with patch("kilix_llm.artifacts.build_opener", return_value=opener):
            return fetch(self.artifact, self.root)

    def test_download_identity_and_cache_revalidation(self):
        result = self.download(self.content)
        self.assertEqual(result, self.artifact.verify(self.root))
        self.assertEqual(result.stat().st_mode & 0o777, 0o600)
        result.write_bytes(b"GGUF" + b"wrong" * 5)
        with self.assertRaises(ValueError):
            fetch(self.artifact, self.root)

    def test_failed_download_never_publishes(self):
        for data in (self.content[:-1], self.content + b"extra", b"HTML" + self.content[4:]):
            with self.assertRaises(ValueError):
                self.download(data)
            self.assertFalse(self.artifact.destination(self.root).exists())
            self.assertEqual(list(self.artifact.destination(self.root).parent.glob(".download-*")), [])

    def test_wrong_hash_same_size_fails(self):
        with self.assertRaises(ValueError):
            self.download(self.content[:-1] + b"X")

    def test_path_and_mutable_or_credential_urls_rejected(self):
        for key, value in (("filename", "../tiny.gguf"), ("id", "../tiny"),
                           ("source_revision", "main"), ("size", True),
                           ("url", "https://example.com/resolve/main/tiny.gguf"),
                           ("url", "http://example.com/" + "a" * 40),
                           ("url", "https://user:password@example.com/" + "a" * 40)):
            data = dict(self.data, **{key: value})
            self.manifest.write_text(json.dumps(data))
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                Artifact.load(self.manifest)

    def test_duplicate_fields_fail(self):
        self.manifest.write_text('{"id":"one", "id":"two"}')
        with self.assertRaises(ValueError):
            Artifact.load(self.manifest)


if __name__ == "__main__":
    unittest.main()
