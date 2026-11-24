"""vLLM as a second engine, for checkpoints the pinned llama.cpp cannot load.

llama.cpp is vendored and compiled in this repository. vLLM is neither:
`kilix-llm build --backend vllm` installs one pinned release from PyPI into a
private uv environment under GPU_TERMINAL_HOME, and nothing in this package
imports it. The launcher only builds command lines for that environment.

A vLLM model is a directory of files pinned by upstream revision, size and
SHA-256, not one GGUF. It lives beside the GGUF store, one directory per
catalogue id and revision.
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import subprocess
import sys

from . import paths
from .models import ModelError, digest_file
from .runtime import RuntimeError_

VERSION = "0.34.0"
PYTHON = "3.12"
# uv ignores every distribution uploaded after this instant, so the pin above
# resolves to the same dependency set later. 0.34.0 was uploaded 2026-11-16.
EXCLUDE_NEWER = "2026-11-17T12:00:00Z"
STAMP = "kilix-llm-runtime.json"
ONESHOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vllm_oneshot.py")

MIB = 1024 * 1024
KV_MODES = ("per_loop", "last_loop")
# vLLM's cache is allocated in blocks of 16 tokens, and its scheduler may hold
# two prefill batches in flight at once.
KV_BLOCK_TOKENS = 16
CONCURRENT_BATCHES = 2
# vLLM's attention backend below this compute capability is its Triton one.
TRITON_ATTENTION_BELOW = 80

# Sizes and SHA-256 identify the exact files at the pinned revision.
MODELS = {
    "dory1.0-4b-nvfp4": {
        "title": "Nemotron Labs Dory1.0 4B NVFP4 (GPU, vLLM)",
        "repo": "nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4",
        "revision": "17065a6849266dd478221735be7747689a0a4331",
        "publisher": "nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4",
        "licence": "OpenMDW-1.1",
        "licence_url": "https://openmdw.ai/license/1-1/",
        "quant": "NVFP4 (W4A16)",
        "files": {
            "chat_template.jinja": (
                11_059, "a0d00f1c12035773c95fe370ef27da2f29da3f8210be3cea5e9d19dcdd56a302"),
            "config.json": (
                2_753, "5aba1243944d5d7ec8b08a9162cd2cbd6b9686e438757830c2e6bf158fde673f"),
            "generation_config.json": (
                225, "55535fcf0e43be7b7968d51a4223182316000b2799c46d75037c23dcca2e467d"),
            "hf_quant_config.json": (
                260, "77467ea4dbe4fef08cd96bf8b2c089deed1192edfe789659d0f87018f7875e02"),
            "model.safetensors": (
                3_202_498_032, "ca08dec5d70b653c03e57903dd33725ec8cc80133c53a1bd9c83f99dd03f3e46"),
            "special_tokens_map.json": (
                563, "e9435fefd6d838fd9fcbbc44b97a8e3ff322be7f6dfb7e4fd2468586574bb52b"),
            "tokenizer.json": (
                17_077_484, "623c34567aebb18582765289fbe23d901c62704d6518d71866e0e58db892b5b7"),
            "tokenizer_config.json": (
                178_205, "0f5c10ba0a8ab4dbd331fd54959f4c41b9fa6807ebf0d9d1056b4676acd302f1"),
        },
        "served_name": "dory1.0-4b",
        "ctx": 8192,
        # vLLM's ModelOpt NVFP4 path runs W4A16 through its Marlin kernel from
        # compute capability 7.5 (Turing) up.
        "min_compute_cap": 75,
        "reasoning_parser": "nemotron_v3",
        "tool_call_parser": "qwen3_coder",
        # The model card recommends two loops and benchmarks one, two and four.
        "loops": 2,
        "max_loops": 8,
        "kv_cache": "per_loop",
        # Attention sublayers by block and kind, from config.json layer_types:
        # every fourth attention sublayer is full, the rest slide over 1,024
        # tokens. Only the recurrent block repeats.
        "attention": {
            "input": {"full": 2, "sliding": 6},
            "recurrent": {"full": 5, "sliding": 15},
            "output": {"full": 2, "sliding": 6},
        },
        "sliding_window": 1024,
        # Keys and values, 4 KV heads of 256, 16-bit floats.
        "kv_bytes_per_token": 4 * 256 * 2 * 2,
        # Prompt tokens processed per step. A sliding layer keeps its window
        # plus the batches in flight, so a smaller batch also means less cache.
        "batch_tokens": 512,
        # Added to the computed cache: vLLM rounds up to whole pages.
        "kv_cache_slack_mib": 16,
        # Measured with vLLM 0.34.0 on a Quadro RTX 3000: the weights as vLLM
        # reports them, and what one server held beyond weights and cache.
        "weights_mib": 3349,
        "overhead_mib": 300,
    },
}


def spec(model_id: str) -> dict:
    try:
        return MODELS[model_id]
    except KeyError as error:
        known = ", ".join(sorted(MODELS))
        raise ModelError(f"unknown vLLM model {model_id!r} (known: {known})") from error


def is_vllm(model_id: str) -> bool:
    return model_id in MODELS


# --- the runtime environment -------------------------------------------------


def runtime_dir() -> str:
    override = os.environ.get("KILIX_LLM_VLLM_DIR")
    if override:
        return override
    return os.path.join(paths.gpu_terminal_home(), "kilix-llm", "runtimes", f"vllm-{VERSION}")


def python_path() -> str:
    return os.path.join(runtime_dir(), "bin", "python")


def vllm_path() -> str:
    return os.path.join(runtime_dir(), "bin", "vllm")


def expected_stamp() -> dict:
    return {"vllm": VERSION, "python": PYTHON, "exclude_newer": EXCLUDE_NEWER}


def build_current() -> bool:
    """True when the environment was completed for exactly this pin."""
    try:
        with open(os.path.join(runtime_dir(), STAMP), encoding="utf-8") as handle:
            recorded = json.load(handle)
    except (OSError, ValueError):
        return False
    return recorded == expected_stamp() and os.access(vllm_path(), os.X_OK)


def build_commands(uv: str) -> list[list[str]]:
    """Create the environment, then install the pinned release into it."""
    target = runtime_dir()
    return [
        [uv, "venv", "--managed-python", "--python", PYTHON, "--clear", target],
        [uv, "pip", "install", "--python", python_path(),
         "--exclude-newer", EXCLUDE_NEWER, f"vllm=={VERSION}"],
    ]


def build(if_needed: bool = False) -> int:
    """Install the pinned vLLM; several gigabytes of wheels on the first run.

    The stamp is written last, so an interrupted install is rebuilt from an
    empty environment. A lock makes concurrent builds wait for one.
    """
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError_(
            "the vLLM runtime is installed with uv, which is not on PATH.\n"
            "Install uv (https://docs.astral.sh/uv/), then rerun:  "
            "kilix-llm build --backend vllm"
        )
    target = runtime_dir()
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target + ".lock", "w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if if_needed and build_current():
            return 0
        stamp = os.path.join(target, STAMP)
        if os.path.exists(stamp):
            os.unlink(stamp)
        print(
            f"kilix-llm: installing vLLM {VERSION} into {target}. "
            "This downloads several gigabytes once.",
            file=sys.stderr, flush=True,
        )
        for argv in build_commands(uv):
            status = subprocess.call(argv, stdin=subprocess.DEVNULL)
            if status != 0:
                return status
        check = subprocess.run(
            [python_path(), "-c", "import importlib.metadata as m; print(m.version('vllm'))"],
            capture_output=True, text=True, check=False,
        )
        if check.returncode != 0 or check.stdout.strip() != VERSION:
            print(
                f"kilix-llm: the environment reports vLLM {check.stdout.strip() or 'missing'}, "
                f"expected {VERSION}",
                file=sys.stderr,
            )
            return 1
        frozen = subprocess.run(
            [uv, "pip", "freeze", "--python", python_path()],
            capture_output=True, text=True, check=False,
        )
        if frozen.returncode == 0:
            with open(os.path.join(target, "installed.txt"), "w", encoding="utf-8") as handle:
                handle.write(frozen.stdout)
        with open(stamp + ".tmp", "w", encoding="utf-8") as handle:
            json.dump(expected_stamp(), handle, sort_keys=True)
            handle.write("\n")
        os.replace(stamp + ".tmp", stamp)
    return 0


def require_built() -> None:
    if not build_current():
        raise RuntimeError_(
            f"the vLLM runtime is not installed ({runtime_dir()}).\n"
            "kilix-llm does not download it unless asked:  kilix-llm build --backend vllm"
        )


# --- GPUs --------------------------------------------------------------------


def gpus() -> list[dict]:
    """NVIDIA GPUs as the driver reports them; memory in MiB."""
    smi = shutil.which("nvidia-smi")
    if smi is None:
        raise RuntimeError_("vLLM needs an NVIDIA GPU and driver; nvidia-smi is not on PATH")
    try:
        result = subprocess.run(
            [smi, "--query-gpu=index,name,compute_cap,memory.total,memory.reserved,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=60, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError_(f"cannot list NVIDIA GPUs: {error}") from error
    if result.returncode:
        raise RuntimeError_(f"cannot list NVIDIA GPUs: {result.stderr.strip()}")
    return parse_gpus(result.stdout)


def parse_gpus(text: str) -> list[dict]:
    """One GPU per line. `free_mib` leaves out what the driver reserves."""
    found = []
    for line in text.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 6:
            continue
        try:
            major, _, minor = fields[2].partition(".")
            total, reserved, used = int(fields[3]), int(fields[4]), int(fields[5])
            found.append({
                "index": int(fields[0]),
                "device": f"CUDA{int(fields[0])}",
                "name": fields[1],
                "compute_cap": int(major) * 10 + int(minor or 0),
                "total_mib": total,
                "used_mib": used,
                "free_mib": max(0, total - reserved - used),
            })
        except ValueError:
            continue
    return found


def select_gpu(requested: str | None, available: list[dict] | None = None) -> dict:
    available = gpus() if available is None else available
    by_name = {gpu["device"]: gpu for gpu in available}
    if requested in by_name:
        return by_name[requested]
    if requested:
        raise RuntimeError_(
            f"GPU device {requested!r} is unavailable; run kilix-llm devices --backend vllm"
        )
    if len(available) == 1:
        return available[0]
    if not available:
        raise RuntimeError_("no NVIDIA GPU is available to vLLM")
    raise RuntimeError_(
        "multiple GPU devices available; choose --device from: "
        + ", ".join(f"{gpu['device']} ({gpu['name']})" for gpu in available)
    )


# --- the model store ---------------------------------------------------------


def model_dir(model_id: str, override: str | None = None) -> str:
    if override:
        return override
    record = spec(model_id)
    return os.path.join(paths.models_dir(), model_id, record["revision"])


def file_url(model_id: str, name: str) -> str:
    record = spec(model_id)
    return f"https://huggingface.co/{record['repo']}/resolve/{record['revision']}/{name}"


def resolve(model_id: str, override: str | None = None) -> str:
    """Return the checkpoint directory if every file is present at its length."""
    record = spec(model_id)
    root = model_dir(model_id, override)
    for name, (size, _digest) in record["files"].items():
        path = os.path.join(root, name)
        if not os.path.isfile(path):
            raise ModelError(
                f"{record['title']} is not downloaded ({path} missing).\n"
                f"kilix-llm does not fetch weights unless asked:  "
                f"kilix-llm pull --model-id {model_id}"
            )
        actual = os.path.getsize(path)
        if actual != size:
            raise ModelError(
                f"{path} is {actual} bytes, expected {size} — "
                f"an interrupted download? `kilix-llm pull --model-id {model_id}` resumes it"
            )
    return root


def verify(model_id: str, override: str | None = None) -> str:
    root = resolve(model_id, override)
    for name, (_size, digest) in spec(model_id)["files"].items():
        path = os.path.join(root, name)
        got = digest_file(path)
        if got != digest:
            raise ModelError(f"{path} sha256 is {got}, expected {digest}")
    return root


def pull(model_id: str) -> str:
    """Fetch every pinned file from the publisher, resuming, then verify all."""
    record = spec(model_id)
    curl = shutil.which("curl")
    if curl is None:
        raise RuntimeError_("curl is required to pull weights")
    root = model_dir(model_id)
    os.makedirs(root, exist_ok=True)
    total = sum(size for size, _digest in record["files"].values())
    print(
        f"kilix-llm: fetching {record['title']} from {record['publisher']} "
        f"({len(record['files'])} files, {total} bytes).\n"
        f"kilix-llm: its use is governed by {record['licence']}: {record['licence_url']}",
        file=sys.stderr, flush=True,
    )
    for name, (size, _digest) in record["files"].items():
        dest = os.path.join(root, name)
        if os.path.isfile(dest) and os.path.getsize(dest) == size:
            continue
        status = subprocess.call(
            [curl, "-fL", "--retry", "3", "-C", "-", "-o", dest, file_url(model_id, name)]
        )
        if status != 0:
            raise ModelError(f"could not fetch {name} (curl status {status})", status)
    return verify(model_id)


# --- memory ------------------------------------------------------------------


def kv_cache_bytes(model_id: str, ctx: int, loops: int, mode: str) -> int:
    """Key/value cache vLLM needs for one sequence of `ctx` tokens.

    Full-attention sublayers keep every token. Sliding ones keep their window
    and the prefill batches in flight, plus one block because a window need not
    start on a block boundary. `per_loop` keeps one cache per pass over the
    recurrent block, `last_loop` one for all passes.
    """
    record = spec(model_id)
    passes = loops if mode == "per_loop" else 1
    full = sliding = 0
    for block, kinds in record["attention"].items():
        repeat = passes if block == "recurrent" else 1
        full += kinds["full"] * repeat
        sliding += kinds["sliding"] * repeat
    held = min(
        record["sliding_window"] - 1 + CONCURRENT_BATCHES * record["batch_tokens"], ctx)
    full_tokens = -(-ctx // KV_BLOCK_TOKENS) * KV_BLOCK_TOKENS
    sliding_tokens = (-(-held // KV_BLOCK_TOKENS) + 1) * KV_BLOCK_TOKENS
    return (full * full_tokens + sliding * sliding_tokens) * record["kv_bytes_per_token"]


def check_serving_args(model_id: str, loops: int, mode: str, ctx: int) -> None:
    record = spec(model_id)
    if not 1 <= loops <= record["max_loops"]:
        raise ModelError(f"--loops must be between 1 and {record['max_loops']}, not {loops}")
    if mode not in KV_MODES:
        raise ModelError(f"--kv-cache must be one of {', '.join(KV_MODES)}")
    if ctx < 256:
        raise ModelError(f"--ctx must be at least 256 tokens, not {ctx}")


def kv_cache_reservation(model_id: str, ctx: int, loops: int, mode: str) -> int:
    """Bytes handed to vLLM for the cache: the computed need and its slack."""
    return kv_cache_bytes(model_id, ctx, loops, mode) + spec(model_id)["kv_cache_slack_mib"] * MIB


def gpu_need_mib(model_id: str, ctx: int, loops: int, mode: str) -> int:
    """GPU memory one server needs: weights, runtime overhead, and the cache."""
    record = spec(model_id)
    cache = -(-kv_cache_reservation(model_id, ctx, loops, mode) // MIB)
    return record["weights_mib"] + record["overhead_mib"] + cache


def c_compiler() -> str | None:
    """The C compiler Triton would use: `CC`, else cc, gcc or clang on PATH."""
    named = os.environ.get("CC")
    if named:
        return shutil.which(named)
    for name in ("cc", "gcc", "clang"):
        found = shutil.which(name)
        if found:
            return found
    return None


def needs_c_compiler(gpu: dict) -> bool:
    """Below capability 8.0 vLLM runs attention through Triton, which compiles
    a small C launcher the first time it runs. Without a compiler the server
    dies during start-up, after the weights have loaded."""
    return gpu["compute_cap"] < TRITON_ATTENTION_BELOW


def preflight(gpu: dict, model_id: str, ctx: int, loops: int, mode: str) -> None:
    """Refuse before loading when the GPU cannot run or hold this request."""
    record = spec(model_id)
    minimum = record["min_compute_cap"]
    if gpu["compute_cap"] < minimum:
        raise RuntimeError_(
            f"{gpu['device']} ({gpu['name']}) has compute capability "
            f"{gpu['compute_cap'] // 10}.{gpu['compute_cap'] % 10}; {record['title']} "
            f"needs {minimum // 10}.{minimum % 10} or newer"
        )
    if needs_c_compiler(gpu) and c_compiler() is None:
        raise RuntimeError_(
            f"vLLM needs a C compiler to run on {gpu['device']} ({gpu['name']}): "
            "none of cc, gcc or clang is on PATH and CC names none.\n"
            "Install gcc, or set CC to a compiler."
        )
    need = gpu_need_mib(model_id, ctx, loops, mode)
    free = gpu["free_mib"]
    if need > free:
        cache = -(-kv_cache_reservation(model_id, ctx, loops, mode) // MIB)
        hints = ["a smaller --ctx"]
        if mode == "per_loop" and loops > 1:
            hints.append("--kv-cache last_loop")
        if gpu["used_mib"] > 64:
            hints.append(f"stopping what holds the other {gpu['used_mib']} MiB")
        raise RuntimeError_(
            f"{record['title']} needs about {need} MiB on {gpu['device']} "
            f"({record['weights_mib']} weights, {cache} cache for {ctx} tokens at "
            f"{loops} loop(s) {mode}, {record['overhead_mib']} runtime), but only "
            f"{free} of {gpu['total_mib']} MiB is free to it.\nTry "
            + ", or ".join(hints) + "."
        )


# --- command lines -----------------------------------------------------------


def plan(model_id: str, *, model: str | None = None, ctx: int = 0, loops: int = 0,
         mode: str | None = None, device: str | None = None) -> dict:
    """Validate one request and size it against the selected GPU."""
    record = spec(model_id)
    ctx = ctx or record["ctx"]
    loops = loops or record["loops"]
    mode = mode or record["kv_cache"]
    check_serving_args(model_id, loops, mode, ctx)
    root = resolve(model_id, model)
    require_built()
    gpu = select_gpu(device)
    preflight(gpu, model_id, ctx, loops, mode)
    return {
        "model_id": model_id,
        "model": root,
        "ctx": ctx,
        "loops": loops,
        "mode": mode,
        "gpu": gpu,
        "kv_cache_memory_bytes": kv_cache_reservation(model_id, ctx, loops, mode),
    }


def memory_gate(request: dict) -> str:
    """The fraction for vLLM's startup check, set below what the weights take.

    With an explicit cache size vLLM uses this number only to refuse to start
    when less than that share of the GPU is free. Its default asks for 92%,
    which fails beside any other GPU user; `preflight` is the real check here.
    """
    record = spec(request["model_id"])
    share = record["weights_mib"] * 100 // request["gpu"]["total_mib"]
    return f"{max(1, min(share, 90)) / 100:.2f}"


def hf_overrides(request: dict) -> dict:
    return {"n_recurrent_loops": request["loops"], "recurrent_kv_cache_mode": request["mode"]}


def environment(request: dict) -> dict[str, str]:
    """The selected GPU, no network for model files, and no usage reports."""
    env = dict(os.environ)
    env.update({
        "CUDA_VISIBLE_DEVICES": str(request["gpu"]["index"]),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "VLLM_NO_USAGE_STATS": "1",
        "DO_NOT_TRACK": "1",
    })
    return env


def serve_argv(request: dict, host: str, port: int, think: bool) -> list[str]:
    """`vllm serve` for one request slot on the loopback endpoint."""
    record = spec(request["model_id"])
    return [
        vllm_path(), "serve", request["model"],
        "--served-model-name", record["served_name"],
        "--host", host,
        "--port", str(port),
        "--hf-overrides", json.dumps(hf_overrides(request), sort_keys=True),
        "--max-model-len", str(request["ctx"]),
        "--max-num-seqs", "1",
        "--max-num-batched-tokens", str(record["batch_tokens"]),
        "--gpu-memory-utilization", memory_gate(request),
        "--kv-cache-memory-bytes", str(request["kv_cache_memory_bytes"]),
        "--reasoning-parser", record["reasoning_parser"],
        "--enable-auto-tool-choice",
        "--tool-call-parser", record["tool_call_parser"],
        "--default-chat-template-kwargs", json.dumps({"enable_thinking": think}),
    ]


def run_argv(request: dict, prompt: str, max_tokens: int, think: bool) -> list[str]:
    """The environment's Python on the one-shot script, with a JSON request."""
    record = spec(request["model_id"])
    payload = {
        "model": request["model"],
        "served_name": record["served_name"],
        "hf_overrides": hf_overrides(request),
        "ctx": request["ctx"],
        "batch_tokens": record["batch_tokens"],
        "gpu_memory_utilization": float(memory_gate(request)),
        "kv_cache_memory_bytes": request["kv_cache_memory_bytes"],
        "max_tokens": max_tokens,
        "prompt": prompt,
        "think": think,
    }
    return [python_path(), ONESHOT, json.dumps(payload, sort_keys=True)]
