# Dory through vLLM

NVIDIA's Nemotron Labs Dory1.0 4B is a looped transformer. Its 4-bit NVFP4
checkpoint is a safetensors file that llama.cpp does not read, so `kilix-llm`
runs it through [vLLM](https://github.com/vllm-project/vllm), in an environment
of its own. Nothing here is needed for the GGUF models above.

```bash
kilix-llm doctor --backend vllm     # the environment, the GPUs, a C compiler, the checkpoint
kilix-llm build --backend vllm      # vLLM 0.34.0 on Python 3.12; about 11 GB, downloaded once
kilix-llm pull --model-id dory1.0-4b-nvfp4      # 3.2 GB, OpenMDW-1.1
kilix-llm serve --model-id dory1.0-4b-nvfp4 --ctx 4096 --kv-cache last_loop
kilix-llm run --model-id dory1.0-4b-nvfp4 'Say hello in one sentence.'
```

It needs an NVIDIA GPU of compute capability 7.5 or newer (RTX 20-series and
later) with a driver that supports CUDA 13, and `nvidia-smi`. vLLM also needs a
C compiler (`cc`, `gcc`, `clang`, or `CC`) when it starts, on every GPU;
`doctor` reports one, and `serve` and `run` refuse without it.

`--loops N` sets how many times the recurrent block runs (2 by default; the
model card benchmarks 1, 2 and 4). `--kv-cache per_loop` keeps one cache per
loop and is the default; `last_loop` keeps one for all loops and needs less
memory. One request is served at a time.

Before starting, `serve` and `run` add up the weights (3,349 MiB), the cache
for `--ctx` and the runtime's own use, and refuse with the numbers when that is
more than the GPU has free. vLLM compiles the model each time it starts, which
takes about two minutes and 600 MiB of GPU memory that it gives back
afterwards, so a start needs about 4,400 MiB free however small the context.

Measured with vLLM 0.34.0 on rented cards with nothing else on the GPU. "Ran"
means `kilix-llm serve` was started at the largest `--ctx` its check accepts
there, in steps of 1,024, and answered one request that filled the context:

| Card | Loops, `--kv-cache` | Largest `--ctx` accepted | Result | Tokens/s, short prompt | Tokens/s, context full |
|---|---|---|---|---|---|
| RTX 3070, 8 GB | 2, `last_loop` | 109,568 | ran | 83 | 35 |
| RTX 3070, 8 GB | 2, `per_loop` | 67,584 | ran | 83 | 43 |
| RTX 3070, 8 GB | 4, `per_loop` | 36,864 | ran | 52 | 30 |
| RTX 3070, 8 GB | 1 | 109,568 | ran | 120 | 54 |
| RTX 2070 SUPER, 8 GB, held to 5,740 MiB free | 2, `last_loop` | 49,152 | ran | 72 | 10 |
| RTX 2070 SUPER, 8 GB, held to 5,740 MiB free | 2, `per_loop` | 29,696 | ran | 72 | 13 |
| RTX 2070 SUPER, 8 GB, held to 4,400 MiB free | 2, `last_loop` | 11,264 | ran | 72 | 23 |

On an RTX 3090 `kilix-llm serve` and `run` were exercised at `--ctx 8192`
(chat, thinking, a tool call). vLLM started directly with the same settings
decoded 117 tokens/s there and filled the whole 262,144-token context holding
13,301 MiB; the launcher was not run at that size.

RTX 20-series cards run the model but read a long prompt slowly: the RTX 2070
SUPER took in 128 tokens/s at 45,056 tokens, where the RTX 3070 took in 1,480
at 110,592.

The server speaks the OpenAI API on `127.0.0.1:8080` under the model name
`dory1.0-4b`. `chat` is not available for this model; use `serve`.
