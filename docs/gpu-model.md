# GPU model

Reference for [kilix-llm](../README.md).

[Qwen3.5-4B Q6_K](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF)
is the general chat and coding option for a 6 GiB GPU. The weight file is
3,525,956,768 bytes (3.28 GiB), pinned by revision and SHA-256. Start with
8,192 context tokens and thinking off. Allow additional VRAM for the cache,
compute buffers, and desktop; file size alone does not establish runtime fit.

```sh
./kilix-llm build --backend vulkan
./kilix-llm devices
./kilix-llm pull --model-id qwen3.5-4b
# Replace Vulkan1 with the discrete GPU name printed by devices.
./kilix-llm run --model-id qwen3.5-4b --device Vulkan1 'Say hello.'
./kilix-llm serve --model-id qwen3.5-4b --device Vulkan1
```

Vulkan build dependencies on Debian/Ubuntu are `libvulkan-dev`, `glslc`, and
`spirv-headers`, in addition to CMake, a C++ compiler, and rsync. Vulkan can
use Mesa NVK on supported NVIDIA hardware without installing a CUDA driver.

The 4B profile selects the Vulkan build, requests all layers and the input
embedding on one GPU, disables automatic fitting, enables Flash Attention and Q8 KV caches, and
serves one request slot. If multiple GPUs are available, `--device` is required.
A missing GPU fails before inference instead of continuing on CPU. Use
`--gpu-layers 0` explicitly to run this model through the CPU build.
`--think` enables thinking in either `run` or `serve` for the 4B profile.

CPU and Vulkan builds live separately in `build/llama.cpp` and
`build/llama.cpp-vulkan`. `KILIX_LLM_BUILD_DIR` overrides the selected build's
directory; it must contain a runtime with the requested backend.

Validated on 2026-09-12 with a Quadro RTX 3000 Mobile (6 GiB), Mesa NVK
25.0.7, and the pinned runtime:

| Check | Result |
|---|---|
| GPU allocation at 8,192 context tokens | 3,615.68 MiB total: weights 3,352.15, KV 136.00, recurrent state 50.25, compute 77.28 |
| Layer and embedding placement | 33/33 layers and input embedding on Vulkan1; no CPU model/cache allocation |
| CLI and local API | short reply, arithmetic, and valid Python output passed; one API slot, thinking off |
| Prompt processing, 256 tokens | 70.7 tokens/s, mean of three repetitions |
| Generation, 128 tokens | 18.7 tokens/s, mean of three repetitions |

These speed measurements use short contexts after warmup. The current NVK
backend reports no FP16 or matrix-core acceleration. Host tokenization,
sampling, and approximately 14 MiB of host output/compute buffers still use
the CPU/RAM; the model weights and caches remain on the GPU.
