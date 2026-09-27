# Additional GGUF artifacts

`kilix_llm.artifacts` provides explicit, immutable manifests for experiments
and additional consumers without modifying the launcher's built-in model IDs.
From this checkout:

```sh
PYTHONPATH=src python3 -m kilix_llm.artifacts path manifest.json
PYTHONPATH=src python3 -m kilix_llm.artifacts fetch manifest.json
PYTHONPATH=src python3 -m kilix_llm.artifacts verify manifest.json
```

The manifest schema is `kilix.llm.artifact/v1`. Required fields are `id`,
`filename` (a GGUF basename), `size` (exact bytes), `sha256`, `url` (HTTPS,
including the immutable 40-hex source revision), `source_revision`,
`quantization`, and `license`. No additional fields are accepted. Manifests
are trusted operator inputs; a licence label records provenance and does not
grant a licence or replace the shared first-use workflow.

Artifacts live under the existing model root in `artifacts/SHA256/filename`.
Fetching is explicit and checks the GGUF magic, exact length, and SHA-256
before atomically publishing a mode-0600 file. Existing cached files are
verified again. Invalid downloads are removed; an invalid existing cache is
reported and never silently overwritten. Fetching retries are caller-owned.

Python consumers use `Artifact.load(path)`, `fetch(artifact)`, and
`artifact.verify()`. Verify before passing the path to the selected runtime.
This API supplies artifact identity, not arbitrary-file admission to the
existing `kilix-llm run/serve` commands. Runtime selection, supported format,
context budget, adapter compatibility, and measured fit remain separate.
