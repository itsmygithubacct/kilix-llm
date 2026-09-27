# Explicit checkpoints and supervised targets

`kilix_llm.checkpoints` is an optional library API for acquiring training
checkpoints. It does not change the built-in inference model catalog or start
a model. `huggingface_hub` is needed only when calling `fetch`.

Supply a `kilix.llm.checkpoint/v1` manifest with exactly `schema`, `model_id`,
`revision`, `license`, and `files`. The revision is a complete immutable Git
commit. Each filename maps to a positive byte `size` and either `sha256`
(large-file content) or `git_blob_sha1` (ordinary Git blob identity, including
the Git blob header). Include `config.json`, all required safetensor shards,
their index when applicable, tokenizer files, and licence files. Python source
files and traversal paths are rejected. Licence metadata is an operator
assertion, not an automatic licence determination.

```python
import json
from kilix_llm.checkpoints import fetch

manifest = json.load(open("checkpoint-manifest.json"))
checkpoint = fetch(manifest, "private-model-cache")
```

The destination is `CACHE/OWNER--MODEL/REVISION`. Acquisition is anonymous and
restricted to the manifest's listed paths. Sizes and digests are verified in
a temporary directory before atomic publication; every reuse verifies again.
Interrupted or invalid downloads are not published. Symlinked files and paths
escaping the checkpoint directory are rejected. Callers must construct manifests
from a trusted source: matching an untrusted digest establishes no provenance.

Load checkpoints with the appropriate installed architecture, safetensors,
`local_files_only=True`, and `trust_remote_code=False`. This API does not select
an architecture, dtype, device, kernel package, or adapter. Pin and record those
separately. Downloads are explicit and can be several gigabytes.

`kilix_llm.supervised` supplies small assistant-only training primitives:

```python
from kilix_llm.supervised import encode_target, target_loss

row = encode_target(tokenizer, messages, authored_answer, max_tokens=768)
loss = target_loss(model, row, device="cuda")
loss.backward()
```

Encoding uses the tokenizer's chat template with thinking disabled, masks the
entire prompt/history, and appends the tokenizer's end-of-turn token to the
authored answer. The caller must verify that this EOS choice matches the model's
chat format. Overlong examples fail instead of being silently truncated.
`target_loss` requires PyTorch and a causal language model supporting tensor
indices in `logits_to_keep`. It computes vocabulary logits only at positions
predicting target tokens, then applies mean cross-entropy. Unit checks compare
both loss and parameter gradients against full-logit masked cross-entropy.

These are building blocks, not a training service: the consumer owns optimizer,
adapter setup, data splits, seeds, evaluation, resource limits and saved output.
Keep held-out characters and world facts outside training and checkpoint
selection. Evaluate complete responses for unsupported claims; matching a few
required phrases does not establish useful, grounded dialogue.
