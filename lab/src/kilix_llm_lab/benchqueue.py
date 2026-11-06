"""Queue a JSON benchmark across remembered local and SSH machines.

Remembered aliases live in ~/.local/state/kilix-llm/bench-connections.json.
Each machine keeps its venv, engine, and downloads in ~/.local/share/kilix-llm/bench.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from kilix_llm_lab import LabError

SCHEMA = "kilix.llm.bench-connections/v1"
REMOTE_ROOT = "$HOME/.local/share/kilix-llm/bench"
PROBE_PROMPT = 'Reply with JSON and nothing else: {"choice":"A"}'


def state_dir() -> Path:
    override = os.environ.get("KILIX_LLM_BENCH_STATE")
    if override:
        return Path(override)
    return Path.home() / ".local" / "state" / "kilix-llm"


def connections_path() -> Path:
    return state_dir() / "bench-connections.json"


def load_connections() -> dict:
    path = connections_path()
    if not path.is_file():
        return {"schema": SCHEMA, "connections": {}}
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != SCHEMA:
        raise LabError("bench connection file has the wrong schema")
    document.setdefault("connections", {})
    return document


def save_connections(document: dict) -> None:
    path = connections_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def remember(alias: str, target: str | None = None) -> dict:
    if not alias or "/" in alias or alias.startswith("-"):
        raise LabError("alias must be a plain name")
    document = load_connections()
    if target:
        record = {"transport": "ssh", "target": target}
    else:
        record = {"transport": "local"}
    document["connections"][alias] = record
    save_connections(document)
    return record


def assign(jobs: list[dict], rates: dict[str, dict[str, float]]) -> list[dict]:
    """Place each job on the machine that finishes soonest.

    ``rates[alias][model]`` is the measured seconds for one item of that model.
    A machine that has no rate for a model is not given that model.
    """
    if not rates:
        raise LabError("no machine rates")
    load = {alias: 0.0 for alias in rates}
    plan = []
    ordered = sorted(jobs, key=lambda job: (-_slowest(job, rates), job["model"], job["id"]))
    for job in ordered:
        candidates = [alias for alias, models in rates.items() if job["model"] in models and models[job["model"]] > 0]
        if not candidates:
            raise LabError(f"no machine can run {job['model']}")
        chosen = min(candidates, key=lambda alias: (load[alias], alias))
        seconds = float(rates[chosen][job["model"]])
        load[chosen] += seconds
        plan.append({"alias": chosen, "id": job["id"], "model": job["model"], "seconds": seconds})
    return plan


def _slowest(job: dict, rates: dict[str, dict[str, float]]) -> float:
    values = [models[job["model"]] for models in rates.values() if job["model"] in models]
    if not values:
        raise LabError(f"no machine can run {job['model']}")
    return max(values)


def _run(command: list[str], *, input_text: str | None = None, timeout: float | None = None) -> subprocess.CompletedProcess:
    completed = subprocess.run(
        command,
        input=input_text,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise LabError(detail or f"command failed: {command[0]}")
    return completed


def _bash(connection: dict, script: str, timeout: float | None = None) -> str:
    if connection["transport"] == "local":
        completed = _run(["bash", "-s"], input_text=script, timeout=timeout)
    else:
        completed = _run(
            ["ssh", "-o", "BatchMode=yes", connection["target"], "bash", "-s"],
            input_text=script,
            timeout=timeout,
        )
    return completed.stdout


def share_root(connection: dict) -> str:
    cached = connection.get("_share_root")
    if cached:
        return cached
    root = _bash(connection, 'printf %s "$HOME/.local/share/kilix-llm/bench"').strip()
    connection["_share_root"] = root
    return root


def _copy(connection: dict, source: Path, dest_name: str) -> None:
    remote = f"{share_root(connection)}/{dest_name}"
    if connection["transport"] == "local":
        script = f"mkdir -p {shlex.quote(str(Path(remote).parent))} && cp {shlex.quote(str(source))} {shlex.quote(remote)}"
        _bash(connection, script)
        return
    _run(["ssh", "-o", "BatchMode=yes", connection["target"], f"mkdir -p {shlex.quote(str(Path(remote).parent))}"])
    _run(["scp", "-q", "-o", "BatchMode=yes", str(source), f"{connection['target']}:{remote}"])


def provision_script(models: dict) -> str:
    builds = {item["id"]: item for item in models["engine"]["builds"]}
    vulkan = builds["vulkan"]
    cpu = builds["cpu"]
    lines = [
        "set -euo pipefail",
        f'root="{REMOTE_ROOT}"',
        'mkdir -p "$root/bin" "$root/models" "$root/engine" "$root/work"',
        'if [ ! -x "$root/venv/bin/python" ]; then',
        '  uv venv --python 3.12 "$root/venv"',
        "fi",
        "fetch() {",
        '  url="$1"; dest="$2"; sha="$3"',
        '  if [ -f "$dest" ] && echo "$sha  $dest" | sha256sum -c - >/dev/null; then',
        "    return 0",
        "  fi",
        '  curl -L --fail --retry 5 --retry-delay 2 -o "$dest.partial" "$url"',
        '  echo "$sha  $dest.partial" | sha256sum -c -',
        '  mv "$dest.partial" "$dest"',
        "}",
        f'fetch {shlex.quote(_asset_url(vulkan))} "$root/engine/vulkan.tar.gz" {shlex.quote(vulkan["sha256"])}',
        'rm -rf "$root/engine/vulkan"',
        'mkdir -p "$root/engine/vulkan"',
        'tar -xzf "$root/engine/vulkan.tar.gz" -C "$root/engine/vulkan"',
        'server=$(find "$root/engine/vulkan" -type f -name llama-server -print -quit)',
        'chmod +x "$server"',
        'if ! "$server" --version >/dev/null 2>&1; then',
        f'  fetch {shlex.quote(_asset_url(cpu))} "$root/engine/cpu.tar.gz" {shlex.quote(cpu["sha256"])}',
        '  rm -rf "$root/engine/cpu"',
        '  mkdir -p "$root/engine/cpu"',
        '  tar -xzf "$root/engine/cpu.tar.gz" -C "$root/engine/cpu"',
        '  server=$(find "$root/engine/cpu" -type f -name llama-server -print -quit)',
        '  chmod +x "$server"',
        '  printf cpu > "$root/engine/kind"',
        "else",
        '  printf vulkan > "$root/engine/kind"',
        "fi",
        'printf %s "$server" > "$root/engine/server-path"',
    ]
    for model in models["models"]:
        dest = f'$root/models/{model["file"]}'
        lines.append(
            f'fetch {shlex.quote(model["url"])} "{dest}" {shlex.quote(model["sha256"])}'
        )
        lines.append(f'test "$(stat -c %s "{dest}")" = "{int(model["bytes"])}"')
    lines.append('printf "%s\\n" "$root" "$(cat "$root/engine/kind")"')
    return "\n".join(lines) + "\n"


def _asset_url(build: dict) -> str:
    return f"https://github.com/ggml-org/llama.cpp/releases/download/b10174/{build['file']}"


def provision(connection: dict, models: dict) -> str:
    return _bash(connection, provision_script(models), timeout=3600)


def _worker_document(server: str, files: dict[str, str], jobs: list[dict], decoding: dict, gpu_layers: int) -> dict:
    return {
        "server": server,
        "files": files,
        "jobs": jobs,
        "decoding": decoding,
        "gpu_layers": gpu_layers,
        "port": 18080,
        "log_dir": f"{REMOTE_ROOT}/work",
    }


def _remote_python(connection: dict, jobs_name: str, output_name: str, item_timeout: float, job_count: int) -> None:
    worker = Path(__file__).with_name("bench_worker.py")
    _copy(connection, worker, "work/bench_worker.py")
    script = "\n".join(
        [
            "set -euo pipefail",
            f'root="{REMOTE_ROOT}"',
            f'"$root/venv/bin/python" "$root/work/bench_worker.py" --jobs "$root/work/{jobs_name}" --output "$root/work/{output_name}" --timeout {item_timeout}',
        ]
    )
    _bash(connection, script, timeout=item_timeout * max(job_count, 1) + 300)


def _read_remote_json(connection: dict, name: str) -> dict | list:
    text = _bash(connection, f'cat "{REMOTE_ROOT}/work/{name}"')
    return json.loads(text)


def _write_remote_json(connection: dict, name: str, document: dict) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
        json.dump(document, handle)
        handle.write("\n")
        path = Path(handle.name)
    try:
        _copy(connection, path, f"work/{name}")
    finally:
        path.unlink(missing_ok=True)


def probe(connection: dict, models: dict, suite: dict) -> dict[str, float]:
    files = {item["id"]: f'{REMOTE_ROOT}/models/{item["file"]}' for item in models["models"]}
    decoding = dict(suite["decoding"])
    decoding["max_tokens"] = 32
    jobs = [
        {"id": "probe", "model": item["id"], "prompt": PROBE_PROMPT}
        for item in models["models"]
    ]
    kind = _bash(connection, f'cat "{REMOTE_ROOT}/engine/kind"').strip()
    server = _bash(connection, f'cat "{REMOTE_ROOT}/engine/server-path"').strip()
    document = _worker_document(server, files, jobs, decoding, 99 if kind == "vulkan" else 0)
    _write_remote_json(connection, "probe-jobs.json", document)
    _remote_python(connection, "probe-jobs.json", "probe-out.json", 180, len(jobs))
    rows = _read_remote_json(connection, "probe-out.json")
    rates = {}
    for row in rows:
        if row.get("error") or not row.get("seconds"):
            continue
        rates[row["model"]] = float(row["seconds"])
    if not rates:
        raise LabError("probe produced no timings")
    return rates


def run_assigned(connection: dict, models: dict, suite: dict, jobs: list[dict]) -> list[dict]:
    files = {item["id"]: f'{REMOTE_ROOT}/models/{item["file"]}' for item in models["models"]}
    prompts = {item["id"]: item["prompt"] for item in suite["items"]}
    payload = [
        {"id": job["id"], "model": job["model"], "prompt": prompts[job["id"]]}
        for job in jobs
    ]
    kind = _bash(connection, f'cat "{REMOTE_ROOT}/engine/kind"').strip()
    server = _bash(connection, f'cat "{REMOTE_ROOT}/engine/server-path"').strip()
    document = _worker_document(server, files, payload, suite["decoding"], 99 if kind == "vulkan" else 0)
    _write_remote_json(connection, "run-jobs.json", document)
    _remote_python(connection, "run-jobs.json", "run-out.json", 240, len(payload))
    return list(_read_remote_json(connection, "run-out.json"))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def find_scorer(suite_path: Path):
    del suite_path
    source = benchmark_source()
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from kilix_benchmark.mcq import score_report

    return score_report


def benchmark_source() -> Path:
    root = os.environ.get("KILIX_BENCHMARK_HOME")
    if root:
        return Path(root) / "src"
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "kilix-benchmark" / "src"
        if candidate.is_dir():
            return candidate
    raise LabError("kilix-benchmark source was not found")


def execute(aliases: list[str], suite_path: Path, models_path: Path) -> dict:
    suite = load_json(suite_path)
    models = load_json(models_path)
    stored = load_connections()["connections"]
    missing = [alias for alias in aliases if alias not in stored]
    if missing:
        raise LabError("unknown alias: " + ", ".join(missing))
    connections = {alias: stored[alias] for alias in aliases}
    with ThreadPoolExecutor(max_workers=len(connections)) as pool:
        futures = {pool.submit(provision, connection, models): alias for alias, connection in connections.items()}
        for future in as_completed(futures):
            future.result()
            print(f"provision {futures[future]}", flush=True)
    rates: dict[str, dict[str, float]] = {}
    with ThreadPoolExecutor(max_workers=len(connections)) as pool:
        futures = {pool.submit(probe, connection, models, suite): alias for alias, connection in connections.items()}
        for future in as_completed(futures):
            alias = futures[future]
            try:
                rates[alias] = future.result()
            except (LabError, OSError, subprocess.TimeoutExpired) as error:
                print(f"probe {alias} failed: {error}", flush=True)
                continue
            print(f"probe {alias} {rates[alias]}", flush=True)
    rates = {alias: row for alias, row in rates.items() if row}
    if not rates:
        raise LabError("no machine produced a timing")
    jobs = [{"id": item["id"], "model": model["id"]} for model in models["models"] for item in suite["items"]]
    plan = assign(jobs, rates)
    grouped: dict[str, list[dict]] = {alias: [] for alias in connections}
    for row in plan:
        grouped[row["alias"]].append(row)
    collected: dict[str, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=len(connections)) as pool:
        futures = {
            pool.submit(run_assigned, connections[alias], models, suite, rows): alias
            for alias, rows in grouped.items()
            if rows
        }
        for future in as_completed(futures):
            collected[futures[future]] = future.result()
    by_model: dict[str, list[dict]] = {model["id"]: [] for model in models["models"]}
    for rows in collected.values():
        for row in rows:
            by_model[row["model"]].append(row)
    score_report = find_scorer(suite_path)
    reports = [score_report(suite, model_id, rows) for model_id, rows in by_model.items()]
    summary = {
        "schema": "kilix.llm.bench-run/v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rates": rates,
        "plan": plan,
        "reports": reports,
    }
    out_dir = state_dir() / "bench-runs" / summary["created"].replace(":", "")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(out_dir / "summary.json", flush=True)
    for report in reports:
        score = report["score"]
        print(f"{report['model']} {score['correct']}/{score['count']}", flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Queue a benchmark across remembered machines")
    commands = parser.add_subparsers(dest="command", required=True)
    remembered = commands.add_parser("remember")
    remembered.add_argument("--alias", required=True)
    remembered.add_argument("--ssh", default=None)
    commands.add_parser("connections")
    provision_p = commands.add_parser("provision")
    provision_p.add_argument("--alias", required=True)
    provision_p.add_argument("--models", required=True, type=Path)
    run_p = commands.add_parser("run")
    run_p.add_argument("--alias", action="append", required=True)
    run_p.add_argument("--suite", required=True, type=Path)
    run_p.add_argument("--models", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "remember":
            remember(args.alias, args.ssh)
        elif args.command == "connections":
            print(json.dumps(load_connections(), indent=2))
        elif args.command == "provision":
            stored = load_connections()["connections"]
            if args.alias not in stored:
                raise LabError(f"unknown alias: {args.alias}")
            provision(stored[args.alias], load_json(args.models))
        else:
            execute(args.alias, args.suite, args.models)
    except LabError as error:
        print(str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
