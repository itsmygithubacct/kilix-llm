"""Command line for kilix-llm."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

from . import __version__, help_acceptance, help_query, models, paths, pin, runtime, vm

PROG = "kilix-llm"


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
            if args.model_id == "qwen3.5-4b" and ngl >= 99:
                # llama.cpp otherwise keeps the input embedding on CPU even at full offload.
                argv.extend(["--override-tensor", rf"^token_embd\.weight$={args.device}"])
    else:
        argv.extend(["--device", "none"])
    return argv


def sampling_args(args: argparse.Namespace) -> list[str]:
    argv = list(models.THINK_SAMPLING if args.think else models.SAMPLING)
    if args.model_id == "qwen3.5-4b":
        argv.extend(["--min-p", "0", "--presence-penalty", "1.5", "--repeat-penalty", "1"])
    return argv


def cmd_doctor(args: argparse.Namespace) -> int:
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
    print("offload:    qwen3.8-27b: CPU/RAM only (-ngl 0); qwen3.5-4b: all GPU layers")
    for model_id in models.MODELS:
        if model_id == models.DEFAULT_MODEL_ID:
            continue
        try:
            status = models.resolve(model_id)
        except models.ModelError as error:
            status = str(error).split("\n", 1)[0]
        print(f"{model_id}: {status}")
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
    script = os.path.join(paths.SCRIPTS_DIR, "build-runtime.sh")
    return subprocess.call([script, "--backend", args.backend])


def cmd_devices(args: argparse.Namespace) -> int:
    available = runtime.devices(args.backend)
    if not available:
        die(f"no GPU devices available in the {args.backend} runtime", 4)
    for name, description in available.items():
        print(f"{name}: {description}")
    return 0


def cmd_path(args: argparse.Namespace) -> int:
    try:
        print(models.model_path(args.model_id, args.model))
    except models.ModelError as error:
        handle(error)
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    try:
        path = models.verify(args.model_id, args.model)
    except models.ModelError as error:
        handle(error)
    print(path)
    return 0


def cmd_pull(args: argparse.Namespace) -> int:
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


def run_argv(args: argparse.Namespace, cli: str, model: str) -> list[str]:
    """The llama-cli command line for one prompt.

    `-st` is what makes it one shot. The conversation pair `-cnv`/`-no-cnv`
    belongs to llama-completion at the pinned commit, so llama-cli rejects it
    outright; `-st` is llama-cli's own single-turn flag and is not interactive
    when the turn arrives as `-p`.
    """
    record = models.spec(args.model_id)
    argv = [
        cli,
        "-m",
        model,
        *common_runtime_args(args, record["ctx"]),
        *sampling_args(args),
        "--jinja",
        "-n",
        str(args.n_predict),
        "-p",
        args.prompt,
        "-st",
    ]
    argv.extend(["--reasoning", "on" if args.think else "off"])
    if gpu_layers(args):
        argv.extend(["--cache-type-k", "q8_0", "--cache-type-v", "q8_0"])
    return argv


def cmd_run(args: argparse.Namespace) -> int:
    try:
        runtime.require_vendor()
        model = models.resolve(args.model_id, args.model)
        backend = runtime_backend(args)
        cli = runtime.binary("llama-cli", backend)
        if gpu_layers(args):
            args.device = runtime.select_device(backend, args.device)
    except (models.ModelError, runtime.RuntimeError_, pin.PinError) as error:
        handle(error)
        return 1
    argv = run_argv(args, cli, model)
    os.execv(argv[0], argv)
    return 1


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
    if args.model_id == "qwen3.5-4b":
        argv.extend([*sampling_args(args), "--parallel", "1"])
        argv.extend(["--reasoning", "on" if args.think else "off"])
    elif args.think:
        argv.extend(["--reasoning", "on"])
    return argv


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        runtime.require_vendor()
        model = models.resolve(args.model_id, args.model)
        backend = runtime_backend(args)
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
        choices=sorted(models.MODELS),
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
    parser.add_argument("--backend", choices=runtime.BACKENDS, default=None,
                        help="runtime build (default: CPU for 0 GPU layers, Vulkan otherwise)")
    parser.add_argument("--device", help="GPU name from `kilix-llm devices`, e.g. Vulkan0")


def cmd_ask_help(args: argparse.Namespace) -> int:
    try:
        argv = help_query.command(args.help_module, args.answer_run, args.rank_run,
                                  args.question, args.max_new_tokens, args.sizer)
        if args.calibration_policy:
            if not args.answer_run or not args.rank_run:
                raise ValueError("historical calibration policies require explicit answer and rank runs")
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
        description="Run Qwen3.8 27B in RAM or Qwen3.5 4B on GPU via vendored llama.cpp.",
    )
    parser.add_argument("--version", action="version", version=f"{PROG} {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor_p = sub.add_parser("doctor", help="what is present, what is missing")
    doctor_p.add_argument("--backend", choices=runtime.BACKENDS, default="cpu")
    doctor_p.set_defaults(func=cmd_doctor)
    help_p = sub.add_parser("ask-help", help="local answers using the default Kilix Qwen 2B Help model")
    help_p.add_argument("--help-module", help="kilix-help-llm checkout directory; otherwise use installed Kilix")
    help_p.add_argument("--answer-run", help="override with a historical answer run; requires --rank-run")
    help_p.add_argument("--rank-run", help="historical rank run paired with --answer-run")
    help_p.add_argument("--max-new-tokens", type=int, default=96)
    help_p.add_argument("--sizer", help="shared model-sizer provider path")
    help_p.add_argument("--calibration-policy",
                        help="private frozen threshold artifact for calibrated acceptance")
    help_p.add_argument("--docs-root", help="Git checkout containing the pinned help documents")
    help_p.add_argument("question")
    help_p.set_defaults(func=cmd_ask_help)

    sub.add_parser("vendor", help="print the llama.cpp pin").set_defaults(func=cmd_vendor)
    build_p = sub.add_parser("build", help="compile the vendored llama.cpp")
    build_p.add_argument("--backend", choices=runtime.BACKENDS, default="cpu")
    build_p.set_defaults(func=cmd_build)
    devices_p = sub.add_parser("devices", help="list available GPU device names")
    devices_p.add_argument("--backend", choices=runtime.BACKENDS, default="vulkan")
    devices_p.set_defaults(func=cmd_devices)

    path_p = sub.add_parser("path", help="print the resolved weight path")
    add_model_args(path_p)
    path_p.set_defaults(func=cmd_path)

    verify_p = sub.add_parser("verify", help="check size and sha256 of the weights")
    add_model_args(verify_p)
    verify_p.set_defaults(func=cmd_verify)

    pull_p = sub.add_parser("pull", help="download and verify a catalogued GGUF")
    add_model_args(pull_p)
    pull_p.set_defaults(func=cmd_pull)

    run_p = sub.add_parser("run", help="one prompt, then exit")
    add_model_args(run_p)
    add_runtime_args(run_p)
    run_p.add_argument("-n", "--n-predict", type=int, default=128)
    run_p.add_argument("--think", action="store_true", help="enable thinking mode")
    run_p.add_argument("prompt")
    run_p.set_defaults(func=cmd_run)

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
