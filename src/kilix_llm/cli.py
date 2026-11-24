"""Command line for kilix-llm."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

from . import (
    __version__, help_acceptance, help_query, models, paths, pin, runtime, vllm_runtime, vm,
)

PROG = "kilix-llm"
MODEL_IDS = sorted([*models.MODELS, *vllm_runtime.MODELS])
BACKENDS = (*runtime.BACKENDS, "vllm")


def die(message: str, code: int = 1) -> None:
    print(f"{PROG}: {message}", file=sys.stderr)
    raise SystemExit(code)


def handle(error: Exception) -> None:
    if isinstance(error, models.ModelError):
        die(str(error), error.code)
    if isinstance(error, runtime.RuntimeError_):
        die(str(error), error.code)
    if isinstance(error, pin.PinError):
        die(str(error), 2)
    raise error


def warn_think(args: argparse.Namespace) -> None:
    """Thinking stays available; an entry measured worse with it says so first."""
    warning = models.think_warning(args.model_id, args.think)
    if warning:
        print(f"{PROG}: warning: {warning}", file=sys.stderr)


def gpu_layers(args: argparse.Namespace) -> int:
    if args.gpu_layers is not None:
        return args.gpu_layers
    return models.spec(args.model_id)["n_gpu_layers"]


def runtime_backend(args: argparse.Namespace) -> str:
    backend = args.backend or ("vulkan" if gpu_layers(args) else "cpu")
    if gpu_layers(args) and backend == "cpu":
        raise runtime.RuntimeError_("GPU layers require --backend vulkan; use --gpu-layers 0 for CPU")
    if not gpu_layers(args) and args.device:
        raise runtime.RuntimeError_("--device requires GPU layers")
    return backend


def common_runtime_args(args: argparse.Namespace, ctx_default: int) -> list[str]:
    threads = args.threads or runtime.physical_cores()
    ctx = args.ctx or ctx_default
    ngl = gpu_layers(args)
    argv = ["-t", str(threads), "-c", str(ctx), "-ngl", str(ngl)]
    if ngl:
        argv.extend(["--split-mode", "none", "--fit", "off", "-fa", "on"])
        if args.device:
            argv.extend(["--device", args.device])
            if models.is_qwen35(args.model_id) and ngl >= 99:
                # llama.cpp otherwise keeps the input embedding on CPU even at full offload.
                argv.extend(["--override-tensor", rf"^token_embd\.weight$={args.device}"])
    else:
        argv.extend(["--device", "none"])
    return argv


def sampling_args(args: argparse.Namespace) -> list[str]:
    argv = models.sampling(args.model_id, args.think)
    if models.is_qwen35(args.model_id):
        argv.extend(models.QWEN35_SAMPLING)
    return argv


def vllm_doctor() -> int:
    """The vLLM environment, the GPUs it can use, and its catalogue models."""
    built = "installed" if vllm_runtime.build_current() else "not installed"
    print(f"{PROG} {__version__}")
    print(f"vllm:       {vllm_runtime.VERSION} on Python {vllm_runtime.PYTHON} ({built})")
    print(f"runtime:    {vllm_runtime.runtime_dir()}")
    status = 0
    try:
        available = vllm_runtime.gpus()
    except runtime.RuntimeError_ as error:
        available = []
        print(f"gpu:        {error}")
        status = 4
    for gpu in available:
        print("gpu:        " + describe_gpu(gpu))
    compiler = vllm_runtime.c_compiler()
    if compiler:
        print(f"compiler:   {compiler}")
    else:
        print("compiler:   missing; vLLM needs cc, gcc or clang to start")
        status = 4
    for model_id in vllm_runtime.MODELS:
        print(f"{model_id}: {vllm_model_status(model_id)}")
    if shutil.which("uv") is None:
        print("build-deps: missing uv")
        status = 4
    return status


def describe_gpu(gpu: dict) -> str:
    cap = gpu["compute_cap"]
    free = gpu["free_mib"]
    return (
        f"{gpu['device']}: {gpu['name']} (compute capability {cap // 10}.{cap % 10}, "
        f"{free} of {gpu['total_mib']} MiB free)"
    )


def vllm_model_status(model_id: str) -> str:
    try:
        return vllm_runtime.resolve(model_id)
    except models.ModelError as error:
        return str(error).split("\n", 1)[0]


def cmd_doctor(args: argparse.Namespace) -> int:
    if args.backend == "vllm":
        return vllm_doctor()
    mem = runtime.meminfo()
    try:
        loaded = pin.verify()
        pin_ok = f"{loaded.commit} ({len(loaded.patches)} patch(es))"
    except pin.PinError as error:
        pin_ok = f"unreadable: {error}"
        loaded = None
    vendor = "present" if runtime.vendor_present() else "missing"
    checked_out = runtime.vendor_commit() or "no nested git"
    built = []
    for name in runtime.BINARIES:
        path = os.path.join(paths.bin_dir(args.backend), name)
        built.append(name if os.path.isfile(path) else f"{name} (not built)")
    try:
        model = models.resolve()
    except models.ModelError as error:
        model = str(error).split("\n", 1)[0]
    print(f"{PROG} {__version__}")
    print(f"vendor:     {paths.VENDOR_DIR} ({vendor})")
    print(f"pin:        {pin_ok}")
    print(f"checkout:   {checked_out}")
    if loaded is not None and checked_out not in ("no nested git", loaded.commit):
        print("            checkout does not match pin commit")
    print(f"build:      {paths.build_dir(args.backend)} ({args.backend})")
    print(f"binaries:   {', '.join(built)}")
    print(f"cores:      {runtime.physical_cores()} physical")
    print(
        "memory:     "
        f"{mem['MemAvailable'] // (1024 ** 3)} GiB available / "
        f"{mem['MemTotal'] // (1024 ** 3)} GiB total"
    )
    print(f"model:      {model}")
    print("offload:    " + "; ".join(
        f"{model_id}: " + ("all GPU layers" if record["n_gpu_layers"] else "CPU/RAM only (-ngl 0)")
        for model_id, record in models.MODELS.items()
    ))
    for model_id in models.MODELS:
        if model_id == models.DEFAULT_MODEL_ID:
            continue
        try:
            status = models.resolve(model_id)
        except models.ModelError as error:
            status = str(error).split("\n", 1)[0]
        print(f"{model_id}: {status}")
    for model_id in vllm_runtime.MODELS:
        print(f"{model_id}: {vllm_model_status(model_id)}")
    missing = runtime.missing_build_packages(args.backend)
    if missing:
        print("build-deps: missing " + ", ".join(missing))
        return 4
    return 0


def cmd_vendor(_args: argparse.Namespace) -> int:
    loaded = pin.verify()
    print(f"url={loaded.url}")
    print(f"commit={loaded.commit}")
    print(f"tree={paths.VENDOR_DIR}")
    print(f"present={str(runtime.vendor_present()).lower()}")
    checked = runtime.vendor_commit()
    if checked:
        print(f"checkout={checked}")
        print(f"matches_pin={str(checked == loaded.commit).lower()}")
    for item in loaded.patches:
        print(f"patch={item.name} {item.sha256}")
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    if args.backend == "vllm":
        if args.keep_build_tree:
            die("--keep-build-tree applies to the llama.cpp builds, not to vllm", 2)
        return vllm_runtime.build(if_needed=args.if_needed)
    missing = runtime.missing_build_packages(args.backend)
    if missing:
        die(
            "cannot build yet — missing: "
            + ", ".join(missing)
            + "\n            install them first (Debian:  sudo apt-get install "
            + " ".join(missing)
            + f"), then rerun:  kilix-llm build --backend {args.backend}",
            4,
        )
    runtime.require_vendor()
    argv = [runtime.BUILD_SCRIPT, "--backend", args.backend]
    if args.if_needed:
        argv.append("--if-needed")
    if args.keep_build_tree:
        argv.append("--keep-build-tree")
    return subprocess.call(argv)


def cmd_devices(args: argparse.Namespace) -> int:
    if args.backend == "vllm":
        gpus = vllm_runtime.gpus()
        if not gpus:
            die("no NVIDIA GPU is available to vLLM", 4)
        for gpu in gpus:
            print(describe_gpu(gpu))
        return 0
    available = runtime.devices(args.backend)
    if not available:
        die(f"no GPU devices available in the {args.backend} runtime", 4)
    for name, description in available.items():
        print(f"{name}: {description}")
    return 0


def cmd_path(args: argparse.Namespace) -> int:
    try:
        if vllm_runtime.is_vllm(args.model_id):
            print(vllm_runtime.model_dir(args.model_id, args.model))
            return 0
        print(models.model_path(args.model_id, args.model))
    except models.ModelError as error:
        handle(error)
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    try:
        if vllm_runtime.is_vllm(args.model_id):
            path = vllm_runtime.verify(args.model_id, args.model)
        else:
            path = models.verify(args.model_id, args.model)
    except models.ModelError as error:
        handle(error)
    print(path)
    return 0


def cmd_pull(args: argparse.Namespace) -> int:
    # This id's licence notices are delivered by the model wizard. pull would
    # fetch the GGUF alone into kilix-llm's store and skip that delivery.
    if args.model_id == "qwen-0.8b-no-loop-02":
        die(
            "qwen-0.8b-no-loop-02 is installed by the model wizard, which "
            "delivers its licence notices: kilix models install "
            "qwen-0.8b-no-loop-02. kilix-llm pull does not install those notices.",
            2,
        )
    if vllm_runtime.is_vllm(args.model_id):
        print(vllm_runtime.pull(args.model_id))
        return 0
    record = models.spec(args.model_id)
    dest = models.model_path(args.model_id)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    curl = shutil.which("curl")
    if curl is None:
        die("curl is required to pull weights", 4)
    argv = [
        curl,
        "-fL",
        "--retry",
        "3",
        "-C",
        "-",
        "-o",
        dest,
        record["url"],
    ]
    print(f"{PROG}: fetching {record['file']} ({record['size']} bytes)", file=sys.stderr)
    rc = subprocess.call(argv)
    if rc != 0:
        return rc
    try:
        models.verify(args.model_id)
    except models.ModelError as error:
        handle(error)
    print(dest)
    return 0


def cli_argv(args: argparse.Namespace, cli: str, model: str, turn: list[str]) -> list[str]:
    """llama-cli with the catalogue runtime and sampler; `turn` decides one shot or chat."""
    record = models.spec(args.model_id)
    argv = [
        cli,
        "-m",
        model,
        *common_runtime_args(args, record["ctx"]),
        *sampling_args(args),
        "--jinja",
        *turn,
    ]
    argv.extend(["--reasoning", "on" if args.think else "off"])
    if gpu_layers(args):
        argv.extend(["--cache-type-k", "q8_0", "--cache-type-v", "q8_0"])
    return argv


def run_argv(args: argparse.Namespace, cli: str, model: str) -> list[str]:
    """The llama-cli command line for one prompt.

    `-st` is what makes it one shot. The conversation pair `-cnv`/`-no-cnv`
    belongs to llama-completion at the pinned commit, so llama-cli rejects it
    outright; `-st` is llama-cli's own single-turn flag and is not interactive
    when the turn arrives as `-p`.
    """
    return cli_argv(args, cli, model, ["-n", str(args.n_predict), "-p", args.prompt, "-st"])


def chat_argv(args: argparse.Namespace, cli: str, model: str) -> list[str]:
    """The llama-cli command line for an interactive conversation.

    Without `-p` and `-st`, llama-cli at the pinned commit reads turns from the
    terminal and keeps the history (`/clear`, `/regen`, `/exit`). Replies are
    bounded by the context unless `-n` caps each one.
    """
    turn = [] if args.n_predict is None else ["-n", str(args.n_predict)]
    return cli_argv(args, cli, model, turn)


def vllm_request(args: argparse.Namespace) -> dict:
    """Check the flags that have no meaning for vLLM, then size the request."""
    if args.backend not in (None, "vllm"):
        raise runtime.RuntimeError_(
            f"{args.model_id} runs on the vllm backend, not {args.backend}")
    if args.gpu_layers is not None:
        raise runtime.RuntimeError_(
            "--gpu-layers does not apply: vLLM keeps the whole model on one GPU")
    return vllm_runtime.plan(
        args.model_id, model=args.model, ctx=args.ctx, loops=args.loops,
        mode=args.kv_cache, device=args.device,
    )


def reject_loop_flags(args: argparse.Namespace) -> None:
    if args.loops or args.kv_cache:
        raise models.ModelError(
            "--loops and --kv-cache apply to looped vLLM models such as "
            f"dory1.0-4b-nvfp4, not to {args.model_id}"
        )


def exec_vllm(argv: list[str], request: dict) -> int:
    os.execve(argv[0], argv, vllm_runtime.environment(request))
    return 1


def exec_cli(args: argparse.Namespace, build) -> int:
    reject_loop_flags(args)
    warn_think(args)
    try:
        runtime.require_vendor()
        model = models.resolve(args.model_id, args.model)
        backend = runtime_backend(args)
        runtime.ensure_built(backend)
        cli = runtime.binary("llama-cli", backend)
        if gpu_layers(args):
            args.device = runtime.select_device(backend, args.device)
    except (models.ModelError, runtime.RuntimeError_, pin.PinError) as error:
        handle(error)
        return 1
    argv = build(args, cli, model)
    os.execv(argv[0], argv)
    return 1


def cmd_run(args: argparse.Namespace) -> int:
    if vllm_runtime.is_vllm(args.model_id):
        request = vllm_request(args)
        return exec_vllm(
            vllm_runtime.run_argv(request, args.prompt, args.n_predict, args.think), request)
    return exec_cli(args, run_argv)


def cmd_chat(args: argparse.Namespace) -> int:
    if vllm_runtime.is_vllm(args.model_id):
        raise models.ModelError(
            f"chat is llama-cli's terminal conversation and {args.model_id} runs on vLLM.\n"
            f"Start `kilix-llm serve --model-id {args.model_id}` and point an "
            "OpenAI-compatible client at it."
        )
    return exec_cli(args, chat_argv)


def serve_argv(args: argparse.Namespace, server: str, model: str) -> list[str]:
    """The llama-server command line for the loopback endpoint."""
    record = models.spec(args.model_id)
    argv = [
        server,
        "-m",
        model,
        *common_runtime_args(args, record["ctx"]),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--jinja",
        "--cache-type-k",
        "q8_0",
        "--cache-type-v",
        "q8_0",
    ]
    if models.is_qwen35(args.model_id):
        argv.extend([*sampling_args(args), "--parallel", "1"])
        argv.extend(["--reasoning", "on" if args.think else "off"])
    elif args.think:
        argv.extend(["--reasoning", "on"])
    return argv


def cmd_serve(args: argparse.Namespace) -> int:
    if vllm_runtime.is_vllm(args.model_id):
        request = vllm_request(args)
        return exec_vllm(
            vllm_runtime.serve_argv(request, args.host, args.port, args.think), request)
    reject_loop_flags(args)
    warn_think(args)
    try:
        runtime.require_vendor()
        model = models.resolve(args.model_id, args.model)
        backend = runtime_backend(args)
        runtime.ensure_built(backend)
        server = runtime.binary("llama-server", backend)
        if gpu_layers(args):
            args.device = runtime.select_device(backend, args.device)
    except (models.ModelError, runtime.RuntimeError_, pin.PinError) as error:
        handle(error)
        return 1
    argv = serve_argv(args, server, model)
    os.execv(argv[0], argv)
    return 1


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model-id",
        default=models.DEFAULT_MODEL_ID,
        choices=MODEL_IDS,
        help="catalog id (default: qwen3.8-27b)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="path to the selected catalog model; size/hash still checked against --model-id",
    )


def add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=0,
        help="CPU threads (default: physical cores)",
    )
    parser.add_argument(
        "-c",
        "--ctx",
        type=int,
        default=0,
        help="context tokens (default: catalog value)",
    )
    parser.add_argument(
        "--gpu-layers",
        type=int,
        default=None,
        help="layers to offload (default: 27B=0, 4B=99/all); 0 selects CPU",
    )
    parser.add_argument("--backend", choices=BACKENDS, default=None,
                        help="runtime build (default: CPU for 0 GPU layers, Vulkan otherwise; "
                             "vLLM models always use vllm)")
    parser.add_argument("--device", help="GPU name from `kilix-llm devices`, e.g. Vulkan0, "
                                         "or CUDA0 for a vLLM model")
    parser.add_argument("--loops", type=int, default=0,
                        help="passes over a looped model's recurrent block (Dory: default 2)")
    parser.add_argument("--kv-cache", choices=vllm_runtime.KV_MODES, default=None,
                        help="looped models: one cache per loop, or one shared by all loops")


def cmd_ask_help(args: argparse.Namespace) -> int:
    try:
        argv = help_query.command(args.help_module, args.answer_run, args.rank_run,
                                  args.question, args.max_new_tokens, args.sizer)
        if args.calibration_policy:
            if args.max_new_tokens != 96:
                raise ValueError("calibrated help requires the frozen 96-token generation limit")
            if not args.docs_root:
                raise ValueError("calibrated help requires --docs-root for pinned source resolution")
            policy, freeze = help_acceptance.load_policy(
                args.calibration_policy, args.answer_run, args.rank_run)
            child = subprocess.run(argv, capture_output=True, text=True, check=False)
            if child.returncode:
                sys.stderr.write(child.stderr)
                return child.returncode
            result = json.loads(child.stdout)
            visible = help_acceptance.apply(result, policy, freeze)
            visible["source_resolution"] = help_acceptance.resolve_source(
                result, freeze["source_revision"], args.docs_root)
            diagnostic = help_acceptance.save_diagnostic(result, args.calibration_policy)
            visible["diagnostic_path"] = str(diagnostic)
            sys.stdout.write(json.dumps(visible, sort_keys=True) + "\n")
            return 0
        os.execv(argv[0], argv)
    except (ValueError, OSError) as error:
        die(str(error), 2)
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Run Qwen3.8 27B in RAM or Qwen3.5 4B on GPU via vendored llama.cpp, "
            "or Dory1.0 4B on an NVIDIA GPU via vLLM."
        ),
    )
    parser.add_argument("--version", action="version", version=f"{PROG} {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor_p = sub.add_parser("doctor", help="what is present, what is missing")
    doctor_p.add_argument("--backend", choices=BACKENDS, default="cpu")
    doctor_p.set_defaults(func=cmd_doctor)
    help_p = sub.add_parser("ask-help", help="experimental document help using explicit local runs")
    help_p.add_argument("--help-module", required=True, help="kilix-help-llm checkout directory")
    help_p.add_argument("--answer-run", required=True)
    help_p.add_argument("--rank-run", required=True)
    help_p.add_argument("--max-new-tokens", type=int, default=96)
    help_p.add_argument("--sizer", help="shared model-sizer provider path")
    help_p.add_argument("--calibration-policy",
                        help="private frozen threshold artifact for calibrated acceptance")
    help_p.add_argument("--docs-root", help="Git checkout containing the pinned help documents")
    help_p.add_argument("question")
    help_p.set_defaults(func=cmd_ask_help)

    sub.add_parser("vendor", help="print the llama.cpp pin").set_defaults(func=cmd_vendor)
    build_p = sub.add_parser("build", help="compile the vendored llama.cpp, or install vLLM")
    build_p.add_argument("--backend", choices=BACKENDS, default="cpu")
    build_p.add_argument("--if-needed", action="store_true",
                         help="build only when this backend is not built for the current pin")
    build_p.add_argument("--keep-build-tree", action="store_true",
                         help="keep the cmake tree for incremental rebuilds (default: keep bin/ only)")
    build_p.set_defaults(func=cmd_build)
    devices_p = sub.add_parser("devices", help="list available GPU device names")
    devices_p.add_argument("--backend", choices=BACKENDS, default="vulkan")
    devices_p.set_defaults(func=cmd_devices)

    path_p = sub.add_parser("path", help="print the resolved weight path")
    add_model_args(path_p)
    path_p.set_defaults(func=cmd_path)

    verify_p = sub.add_parser("verify", help="check size and sha256 of the weights")
    add_model_args(verify_p)
    verify_p.set_defaults(func=cmd_verify)

    pull_p = sub.add_parser("pull", help="download and verify a catalogued model")
    add_model_args(pull_p)
    pull_p.set_defaults(func=cmd_pull)

    run_p = sub.add_parser("run", help="one prompt, then exit")
    add_model_args(run_p)
    add_runtime_args(run_p)
    run_p.add_argument("-n", "--n-predict", type=int, default=128)
    run_p.add_argument("--think", action="store_true", help="enable thinking mode")
    run_p.add_argument("prompt")
    run_p.set_defaults(func=cmd_run)

    chat_p = sub.add_parser("chat", help="interactive conversation in the terminal")
    add_model_args(chat_p)
    add_runtime_args(chat_p)
    chat_p.add_argument("-n", "--n-predict", type=int, default=None,
                        help="cap each reply (default: bounded by the context)")
    chat_p.add_argument("--think", action="store_true", help="enable thinking mode")
    chat_p.set_defaults(func=cmd_chat)

    serve_p = sub.add_parser("serve", help="OpenAI-compatible llama-server on loopback")
    add_model_args(serve_p)
    add_runtime_args(serve_p)
    serve_p.add_argument("--host", default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=8080)
    serve_p.add_argument("--think", action="store_true", help="enable thinking mode")
    serve_p.set_defaults(func=cmd_serve)
    vm.add_parser(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (models.ModelError, runtime.RuntimeError_, pin.PinError) as error:
        handle(error)
    return 1
