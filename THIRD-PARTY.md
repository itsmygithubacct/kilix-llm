# Third-party code in this repository

`kilix-llm` itself is MIT licensed — see [LICENSE](LICENSE). The following
vendored component is **not** covered by that grant and carries its own:

## `third_party/llama.cpp/`

Upstream [llama.cpp](https://github.com/ggml-org/llama.cpp) is vendored in
this repository so it can be inspected and patched here. The exact upstream
revision is recorded in [`third_party/llama.cpp.pin`](third_party/llama.cpp.pin).
`scripts/vendor-sync.sh` is the only supported way to replace that tree with a
newer upstream commit.

- **Licence:** MIT
- **Copyright:** (c) 2023-2026 The ggml authors
- **Full text:** [`third_party/llama.cpp/LICENSE`](third_party/llama.cpp/LICENSE),
  which is retained verbatim and must be preserved in any redistribution.

Edits under `third_party/llama.cpp/` are first-class changes in this
repository. Replayable diffs that should survive an upstream sync belong in
`patches/` and are listed, with sha256, in the pin file.

## vLLM (not vendored)

[vLLM](https://github.com/vllm-project/vllm) is neither copied into this
repository nor shipped in its package. `kilix-llm build --backend vllm`
downloads one pinned release and its dependencies from PyPI into a private uv
environment in the user's data directory. The pin is in
[`src/kilix_llm/vllm_runtime.py`](src/kilix_llm/vllm_runtime.py).

- **Licence:** Apache-2.0
- **Its dependencies** (PyTorch, the NVIDIA CUDA libraries and others) carry
  their own licences, delivered with each installed distribution.

No model weights are included in this repository.
