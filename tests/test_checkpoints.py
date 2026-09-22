import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from kilix_llm.checkpoints import fetch, validate, verify


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.config=b'{"test": true}'
        self.weights=b'fixture weights'
        self.spec={"schema":"kilix.llm.checkpoint/v1","model_id":"Example/model","revision":"a"*40,
                   "license":"test","files":{
                       "config.json":{"size":len(self.config),"git_blob_sha1":hashlib.sha1(b'blob '+str(len(self.config)).encode()+b'\0'+self.config).hexdigest()},
                       "model.safetensors":{"size":len(self.weights),"sha256":hashlib.sha256(self.weights).hexdigest()}}}

    def test_both_digest_formats_and_corruption(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root);(p/'config.json').write_bytes(self.config);(p/'model.safetensors').write_bytes(self.weights)
            self.assertEqual(verify(self.spec,p),p)
            (p/'model.safetensors').write_bytes(b'X'+self.weights[1:])
            with self.assertRaises(ValueError):verify(self.spec,p)

    def test_reject_paths_code_and_mutable_revision(self):
        for name in ('../model.safetensors','/model.safetensors','remote.py','LICENSE.py','a/./model.safetensors','a//model.safetensors',12):
            spec=copy.deepcopy(self.spec);spec['files'][name]=spec['files'].pop('model.safetensors')
            with self.assertRaises(ValueError):validate(spec)
        spec=copy.deepcopy(self.spec);spec['revision']='main'
        with self.assertRaises(ValueError):validate(spec)

    def test_reject_symlink_file(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root);(p/'config.json').write_bytes(self.config);(p/'outside').write_bytes(self.weights)
            (p/'model.safetensors').symlink_to(p/'outside')
            with self.assertRaises(ValueError):verify(self.spec,p)

    def test_acquisition_verifies_before_publish_and_cache_reuse(self):
        calls=[]
        def download(model, **kwargs):
            calls.append((model,kwargs))
            directory=kwargs['local_dir']
            (directory/'config.json').write_bytes(self.config)
            (directory/'model.safetensors').write_bytes(self.weights)
        with tempfile.TemporaryDirectory() as root, patch.dict('sys.modules',{'huggingface_hub':SimpleNamespace(snapshot_download=download)}):
            destination=fetch(self.spec,root)
            self.assertEqual(fetch(self.spec,root),destination)
            self.assertEqual(len(calls),1)
            self.assertEqual(calls[0][1]['revision'],self.spec['revision'])
            self.assertFalse(calls[0][1]['token'])
            (destination/'model.safetensors').write_bytes(b'X'+self.weights[1:])
            with self.assertRaises(ValueError):fetch(self.spec,root)
            self.assertEqual(len(calls),1)

    def test_interrupted_download_is_not_published(self):
        def interrupted(model, **kwargs):
            (kwargs['local_dir']/'config.json').write_bytes(self.config)
            raise RuntimeError('interrupted')
        with tempfile.TemporaryDirectory() as root, patch.dict('sys.modules',{'huggingface_hub':SimpleNamespace(snapshot_download=interrupted)}):
            with self.assertRaises(RuntimeError):fetch(self.spec,root)
            parent=Path(root)/'Example--model'
            self.assertEqual(list(parent.iterdir()),[])
