"""Command line for kilix-llm."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

from . import __version__, help_acceptance, help_query, models, paths, pin, runtime

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


def common_runtime_args(args: argparse.Namespace, ctx_default: int) -> list[str]:
    threads = args.threads or runtime.physical_cores()
    ctx = args.ctx or ctx_default
    ngl = args.gpu_layers
    return ["-t", str(threads), "-c", str(ctx), "-ngl", str(ngl)]


def sampling_args(args: argparse.Namespace) -> list[str]:
    return list(models.THINK_SAMPLING if args.think else models.SAMPLING)


def cmd_doctor(_args: argparse.Namespace) -> int:
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
        path = os.path.join(paths.bin_dir(), name)
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
    print(f"build:      {paths.build_dir()}")
    print(f"binaries:   {', '.join(built)}")
    print(f"cores:      {runtime.physical_cores()} physical")
    print(
        "memory:     "
        f"{mem['MemAvailable'] // (1024 ** 3)} GiB available / "
        f"{mem['MemTotal'] // (1024 ** 3)} GiB total"
    )
    print(f"model:      {model}")
    print("offload:    CPU/RAM only (-ngl 0); CUDA and Vulkan are off at build")
    missing = runtime.missing_build_packages()
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


def cmd_build(_args: argparse.Namespace) -> int:
    missing = runtime.missing_build_packages()
    if missing:
        die(
            "cannot build yet — missing: "
            + ", ".join(missing)
            + "\n            install them first (Debian:  sudo apt-get install "
            + " ".join(missing)
            + "), then rerun:  kilix-llm build",
            4,
        )
    runtime.require_vendor()
    script = os.path.join(paths.SCRIPTS_DIR, "build-runtime.sh")
    return subprocess.call([script])


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
    if not args.think:
        argv.extend(["--reasoning", "off"])
    return argv


def cmd_run(args: argparse.Namespace) -> int:
    try:
        runtime.require_vendor()
        model = models.resolve(args.model_id, args.model)
        cli = runtime.binary("llama-cli")
    except (models.ModelError, runtime.RuntimeError_, pin.PinError) as error:
        handle(error)
        return 1
    argv = run_argv(args, cli, model)
    os.execv(argv[0], argv)
    return 1


def serve_argv(args: argparse.Namespace, server: str, model: str) -> list[str]:
    """The llama-server command line for the loopback endpoint."""
    record = models.spec(args.model_id)
    return [
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


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        runtime.require_vendor()
        model = models.resolve(args.model_id, args.model)
        server = runtime.binary("llama-server")
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
        help="explicit GGUF path; skips the catalog file name",
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
        default=0,
        help="layers to offload; default 0 keeps the whole model in RAM",
    )


def cmd_ask_help(args: argparse.Namespace) -> int:
    try:
        argv = help_query.command(args.help_module, args.answer_run, args.rank_run,
                                  args.question, args.max_new_tokens, args.sizer)
        if args.calibration_policy:
            if args.max_new_tokens != 96:
                raise ValueError("calibrated help requires the frozen 96-token generation limit")
            policy, freeze = help_acceptance.load_policy(
                args.calibration_policy, args.answer_run, args.rank_run)
            child = subprocess.run(argv, capture_output=True, text=True, check=False)
            if child.returncode:
                sys.stderr.write(child.stderr)
                return child.returncode
            result = json.loads(child.stdout)
            visible = help_acceptance.apply(result, policy, freeze)
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
        description="Run Qwen3.8 27B Q4 from system RAM via vendored llama.cpp.",
    )
    parser.add_argument("--version", action="version", version=f"{PROG} {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    help_p = sub.add_parser("ask-help", help="experimental document help using explicit local runs")
    help_p.add_argument("--help-module", required=True, help="kilix-help-llm checkout directory")
    help_p.add_argument("--answer-run", required=True)
    help_p.add_argument("--rank-run", required=True)
    help_p.add_argument("--max-new-tokens", type=int, default=96)
    help_p.add_argument("--sizer", help="shared model-sizer provider path")
    help_p.add_argument("--calibration-policy",
                        help="private frozen threshold artifact for calibrated acceptance")
    help_p.add_argument("question")
    help_p.set_defaults(func=cmd_ask_help)

    sub.add_parser("doctor", help="what is present, what is missing").set_defaults(
        func=cmd_doctor
    )
    sub.add_parser("vendor", help="print the llama.cpp pin").set_defaults(func=cmd_vendor)
    sub.add_parser("build", help="compile the vendored llama.cpp, CPU only").set_defaults(
        func=cmd_build
    )

    path_p = sub.add_parser("path", help="print the resolved weight path")
    add_model_args(path_p)
    path_p.set_defaults(func=cmd_path)

    verify_p = sub.add_parser("verify", help="check size and sha256 of the weights")
    add_model_args(verify_p)
    verify_p.set_defaults(func=cmd_verify)

    pull_p = sub.add_parser("pull", help="download the catalogued Q4 GGUF")
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
    serve_p.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
