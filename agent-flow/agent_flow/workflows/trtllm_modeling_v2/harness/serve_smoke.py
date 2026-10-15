#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Drive a unit's own deployment entry over HTTP, and print one structured record per request.

Usage, as ONE process that owns the devices of the claim::

    <python> serve_smoke.py --spec spec.json

The spec names the entry to start, exactly one of:

* ``server_command`` -- a shell command that serves the OpenAI API on
  ``{port}`` (substituted here with a free port): the aggregated
  ``trtllm-serve``, behind ``mpirun ... trtllm-llmapi-launch`` above one rank;
* ``launcher_command`` -- a shell command that starts a deployment of its own
  (a disaggregated context server, generation server and proxy), prints
  ``MV2FLOW_ENDPOINT=<base url>`` once it serves, and stops everything it
  started when it receives SIGTERM.

and the workload: ``requests`` (``{"id", "prompt", "answer", "images",
"keywords", "keywords_any", "max_tokens"}``) followed by ``dataset`` rows
(see :mod:`workload_io`), sent one at a time to ``/v1/chat/completions``
(``endpoint: chat``, the default) or ``/v1/completions``. Images travel as
base64 ``data:`` URLs in the chat message, as an OpenAI client sends them.

Output (see :mod:`workload_io`): one ``MV2FLOW_SERVE`` line per request --
``index``, ``id``, ``status`` (the HTTP status; 0 when the connection
failed), ``text``, ``error`` (the response body when not 200), ``correct``
(the scorer's verdict, or null), ``keywords_missing`` -- and one
``MV2FLOW_SERVE_SUMMARY``.

Shutdown is graceful and touches only what this process started: SIGTERM to
the ``trtllm-serve`` interpreters under it (the server runs its own
shutdown, every rank exits normally), then SIGINT and SIGTERM to its process
group. Nothing is SIGKILLed; a process still alive after that is named in the
summary for the operator, because killing a rank attached to an MPI job can
take the whole allocation with it.

Exits 0 once the entry served and every request was attempted; 3 when the
entry never became healthy.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import workload_io as wio  # noqa: E402 -- imported by path, see the package docstring

ENDPOINT_MARKER = "MV2FLOW_ENDPOINT="
_IMAGE_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


def log(message: str) -> None:
    """A progress line on stderr; stdout carries only records."""
    print(f"[serve-smoke {time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def free_port() -> int:
    """A port nothing listens on right now."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def http(
    method: str, url: str, payload: dict[str, Any] | None = None, timeout: float = 10.0
) -> tuple[int, str]:
    """``(status, body)``; ``(0, reason)`` when nothing answered."""
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.status), response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as error:
        return int(error.code), error.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError, TimeoutError) as error:
        return 0, str(error)


def _children() -> dict[int, list[int]]:
    tree: dict[int, list[int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
        except OSError:
            continue
        parent = int(stat.rsplit(")", 1)[1].split()[1])
        tree.setdefault(parent, []).append(int(entry.name))
    return tree


def descendants(root: int) -> list[int]:
    """Every live process under ``root``."""
    tree, found, stack = _children(), [], [root]
    while stack:
        for child in tree.get(stack.pop(), ()):
            found.append(child)
            stack.append(child)
    return found


def _serve_interpreters(root: int) -> list[int]:
    """The ``trtllm-serve`` interpreters under ``root``: not mpirun, not the launch wrapper."""
    pids = []
    for pid in descendants(root):
        try:
            raw = (Path("/proc") / str(pid) / "cmdline").read_bytes()
        except OSError:
            continue
        words = [w.decode(errors="replace") for w in raw.split(b"\0") if w]
        if not words or any("trtllm-llmapi-launch" in w or w.endswith("mpirun") for w in words):
            continue
        if any(w.endswith("trtllm-serve") for w in words[:2]):
            pids.append(pid)
    return pids


def _wait(proc: subprocess.Popen, seconds: float) -> bool:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if proc.poll() is not None:
            return True
        time.sleep(1)
    return proc.poll() is not None


def stop(proc: subprocess.Popen, *, launcher: bool) -> list[int]:
    """Stop the entry gracefully; return the pids still alive afterwards."""
    if proc.poll() is None:
        targets = [proc.pid] if launcher else _serve_interpreters(proc.pid)
        for pid in targets:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        if not (targets and _wait(proc, 90)):
            for sig, grace in ((signal.SIGINT, 30), (signal.SIGTERM, 30)):
                try:
                    os.killpg(proc.pid, sig)
                except ProcessLookupError:
                    break
                if _wait(proc, grace):
                    break
    leftovers = [p for p in descendants(proc.pid)] if proc.poll() is None else []
    if proc.poll() is None:
        leftovers.insert(0, proc.pid)
    return leftovers


def _image_url(path: str) -> str:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"image not found: {source}")
    if wio.is_lfs_pointer(source):
        raise RuntimeError(f"{source} is a git-lfs pointer, not an image")
    kind = _IMAGE_TYPES.get(source.suffix.lower(), "image/png")
    return f"data:{kind};base64,{base64.b64encode(source.read_bytes()).decode('ascii')}"


def request_body(
    request: dict[str, Any], *, model: str, endpoint: str, max_tokens: int
) -> tuple[str, dict[str, Any]]:
    """The path and JSON body of one request."""
    budget = int(request.get("max_tokens") or max_tokens)
    if endpoint == "completions":
        return "/v1/completions", {
            "model": model,
            "prompt": request["prompt"],
            "max_tokens": budget,
            "temperature": 0,
        }
    images = request.get("images") or []
    content: Any = request["prompt"]
    if images:
        content = [{"type": "text", "text": request["prompt"]}] + [
            {"type": "image_url", "image_url": {"url": _image_url(i)}} for i in images
        ]
    return "/v1/chat/completions", {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": budget,
        "temperature": 0,
    }


def response_text(body: str, endpoint: str) -> str:
    """The generated text of a 200 response; empty when it cannot be read."""
    try:
        choice = json.loads(body)["choices"][0]
    except (ValueError, KeyError, IndexError, TypeError):
        return ""
    if endpoint == "completions":
        return str(choice.get("text") or "")
    message = choice.get("message") or {}
    return str(message.get("content") or "")


def resolve_requests(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """The declared requests, then the dataset's."""
    requests = [dict(r) for r in spec.get("requests") or ()]
    dataset = spec.get("dataset")
    if dataset:
        rows = wio.load_rows(
            dataset["source"], split=dataset.get("split", "test"), config=dataset.get("config")
        )
        for n, row in enumerate(
            wio.select_prompts(
                rows,
                num=int(dataset.get("num", 32)),
                text_field=dataset.get("text_field"),
                answer_field=dataset.get("answer_field"),
                template=dataset.get("template") or "{text}",
            )
        ):
            requests.append({"id": f"d{n}", "prompt": row["prompt"], "answer": row["answer"]})
    for n, request in enumerate(requests):
        request.setdefault("id", f"r{n}")
    return requests


def start(spec: dict[str, Any], log_path: Path) -> tuple[subprocess.Popen, str | None, bool]:
    """Start the entry; return the process, the base URL when known, and whether it is a launcher."""
    handle = open(log_path, "w", encoding="utf-8", errors="replace")
    if spec.get("launcher_command"):
        command, base, launcher = str(spec["launcher_command"]), None, True
    else:
        port = free_port()
        command = str(spec["server_command"]).replace("{port}", str(port))
        base, launcher = f"http://{spec.get('host', '127.0.0.1')}:{port}", False
    log(f"start: {command} (log {log_path})")
    proc = subprocess.Popen(
        ["bash", "-c", command],
        stdout=handle,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    return proc, base, launcher


def wait_ready(
    proc: subprocess.Popen, base: str | None, log_path: Path, deadline: float
) -> str | None:
    """The base URL once ``/health`` answers 200; ``None`` if the entry died or the deadline passed."""
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return None
        if base is None:
            for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
                if ENDPOINT_MARKER in line:
                    base = line.split(ENDPOINT_MARKER, 1)[1].strip().rstrip("/")
        if base is not None and http("GET", f"{base}/health", timeout=5)[0] == 200:
            return base
        time.sleep(5)
    return None


def main(argv: list[str] | None = None) -> int:
    """Start the entry, send the workload, print records, stop the entry."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", required=True)
    options = parser.parse_args(argv)
    spec = json.loads(Path(options.spec).read_text(encoding="utf-8"))
    requests = resolve_requests(spec)
    endpoint = str(spec.get("endpoint") or "chat")
    scorer = spec.get("scorer")
    log_dir = Path(spec.get("log_dir") or Path(options.spec).parent)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{Path(options.spec).stem}.server.log"

    proc, base, launcher = start(spec, log_path)
    summary: dict[str, Any] = {"requests": len(requests), "server_log": str(log_path)}
    try:
        base = wait_ready(
            proc, base, log_path, time.monotonic() + float(spec.get("startup_timeout_s", 1800))
        )
        if base is None:
            tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
            summary.update({"started": False, "server_tail": tail, "exit": proc.poll()})
            return 3
        status, body = http("GET", f"{base}/v1/models")
        model = str(spec.get("model") or "")
        if status == 200:
            try:
                model = str(json.loads(body)["data"][0]["id"])
            except (ValueError, KeyError, IndexError, TypeError):
                pass
        summary.update({"started": True, "base": base, "model": model})
        errors = scored = correct = 0
        for index, request in enumerate(requests):
            record: dict[str, Any] = {"index": index, "id": request["id"]}
            try:
                path, payload = request_body(
                    request,
                    model=model,
                    endpoint=endpoint,
                    max_tokens=int(spec.get("max_tokens", 512)),
                )
            except (OSError, RuntimeError) as error:
                record.update({"status": 0, "error": str(error), "text": ""})
                errors += 1
                wio.emit(wio.SERVE_MARKER, record)
                continue
            status, body = http(
                "POST", f"{base}{path}", payload, timeout=float(spec.get("request_timeout_s", 600))
            )
            text = response_text(body, endpoint) if status == 200 else ""
            record.update(
                {"status": status, "text": text[:4000], "images": len(request.get("images") or ())}
            )
            if status != 200:
                errors += 1
                record["error"] = body[:2000]
            missing = [k for k in request.get("keywords") or () if k.lower() not in text.lower()]
            choices = request.get("keywords_any") or ()
            if choices and not any(k.lower() in text.lower() for k in choices):
                missing.append(f"any of {list(choices)}")
            record["keywords_missing"] = missing
            verdict = (
                wio.score(scorer, text, request.get("answer")) if scorer and status == 200 else None
            )
            record["correct"] = verdict
            if verdict is not None:
                scored += 1
                correct += int(verdict)
            wio.emit(wio.SERVE_MARKER, record)
        summary.update(
            {
                "http_errors": errors,
                "scored": scored,
                "correct": correct,
                "accuracy": 100.0 * correct / scored if scored else None,
            }
        )
        return 0
    finally:
        summary["leftover_pids"] = stop(proc, launcher=launcher)
        wio.emit(wio.SERVE_SUMMARY_MARKER, summary)


if __name__ == "__main__":
    sys.exit(main())
