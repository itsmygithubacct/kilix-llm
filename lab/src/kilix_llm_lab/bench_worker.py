"""Run queued chat completions against a local llama-server. Stdlib only."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


def _message(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(
            item if isinstance(item, str) else str(item.get("text") or "")
            for item in value
            if isinstance(item, (str, dict))
        )
    return ""


def complete(port: int, prompt: str, decoding: dict, timeout: float) -> tuple[str, str]:
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": decoding["temperature"],
        "top_p": decoding["top_p"],
        "max_tokens": decoding["max_tokens"],
        "seed": decoding["seed"],
    }
    if "top_k" in decoding:
        body["top_k"] = decoding["top_k"]
    if "enable_thinking" in decoding:
        body["chat_template_kwargs"] = {"enable_thinking": bool(decoding["enable_thinking"])}
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    last_error = ""
    attempts = (body, {key: value for key, value in body.items() if key != "chat_template_kwargs"})
    for attempt in attempts:
        request = urllib.request.Request(
            url,
            data=json.dumps(attempt).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode())
        except urllib.error.HTTPError as error:
            last_error = error.read().decode("utf-8", "replace")[:500]
            if "chat_template_kwargs" in attempt and attempt is body:
                continue
            raise RuntimeError(last_error) from error
        choice = payload["choices"][0]
        return _message(choice["message"].get("content")), str(choice.get("finish_reason") or "")
    raise RuntimeError(last_error or "completion failed")


def _wait_healthy(port: int, process: subprocess.Popen) -> None:
    for _ in range(180):
        if process.poll() is not None:
            raise RuntimeError("llama-server exited during startup")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(1)
    raise RuntimeError("llama-server health check timed out")


def _serve(server: str, model: str, port: int, gpu_layers: int, log_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("w", encoding="utf-8")
    return subprocess.Popen(
        [
            server,
            "-m",
            model,
            "-ngl",
            str(gpu_layers),
            "-c",
            "4096",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--jinja",
            "--reasoning",
            "on",
            "--reasoning-format",
            "none",
        ],
        stdout=handle,
        stderr=subprocess.STDOUT,
    )


def run_jobs(document: dict, timeout: float) -> list[dict]:
    server = document["server"]
    port = int(document.get("port") or 18080)
    gpu_layers = int(document.get("gpu_layers") or 0)
    grouped: dict[str, list[dict]] = {}
    for job in document["jobs"]:
        grouped.setdefault(job["model"], []).append(job)
    results = []
    for model, jobs in grouped.items():
        path = document["files"][model]
        log_path = Path(document["log_dir"]) / f"{model.replace(':', '-')}.log"
        process = _serve(server, path, port, gpu_layers, log_path)
        try:
            _wait_healthy(port, process)
            for job in jobs:
                started = time.monotonic()
                try:
                    text, finish = complete(port, job["prompt"], document["decoding"], timeout)
                    error = ""
                except (RuntimeError, urllib.error.URLError, TimeoutError) as exc:
                    text, finish, error = "", "", str(exc)
                results.append(
                    {
                        "id": job["id"],
                        "model": model,
                        "completion": text,
                        "finish": finish,
                        "seconds": round(time.monotonic() - started, 3),
                        "error": error,
                    }
                )
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one machine's bench queue")
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout", type=float, default=240)
    args = parser.parse_args(argv)
    document = json.loads(Path(args.jobs).read_text(encoding="utf-8"))
    rows = run_jobs(document, args.timeout)
    Path(args.output).write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
