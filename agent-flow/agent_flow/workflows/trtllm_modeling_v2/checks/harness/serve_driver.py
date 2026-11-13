#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Start ``trtllm-serve`` on the tree under test, ask it a few questions, stop it, and print records.

Usage (one process: it creates the server's ranks itself)::

    <tree's python> serve_driver.py --spec spec.json

The spec is a JSON file the serve gate writes (``serve.py``):

* ``model``: the checkpoint, ``trtllm-serve``'s positional argument;
* ``serve_args``: every flag after it but the host and port (``--config``,
  topology, limits);
* ``world_size``: the server's ranks, created by ``mpirun`` under
  ``trtllm-llmapi-launch`` above one;
* ``server_log``: where the server's output goes;
* ``boot_timeout`` and ``request_timeout``: seconds;
* ``requests``: ``{"id", "endpoint", "body"}`` each, sent in order once
  ``/health`` answers. An ``image_url`` part naming a local file is sent
  inline, as a base64 ``data:`` URL.

Output, on stdout, each one JSON line behind a marker (see ``harness/workload_io.py``):

* ``MV2FLOW_RECORD`` per request: ``index``, ``id``, ``endpoint``, ``status``
  (HTTP; ``null`` without a response), ``text``, ``finish_reason``, ``tokens``,
  ``error``;
* ``MV2FLOW_SERVER`` once, last, after the server is stopped: ``healthy``,
  ``boot_s``, ``error`` and ``cause`` when it never became healthy, ``argv``,
  ``log``, the server's ``returncode``, and the device holders ``reaped`` after
  it stopped and ``left`` alive after that.

The server, every rank it started and every process left holding one of its
devices are stopped whatever happened, a SIGTERM to this process included.
Exits 0 when it ran to the end, whatever the server did.

Runs by path under the measured tree's interpreter: standard library plus
PyYAML, and no ``agent_flow``. The launch, health and reap helpers are
``infra/compute/server_runtime.py``'s, imported by path.
"""

from __future__ import annotations

import argparse
import base64
import copy
import importlib.util
import json
import mimetypes
import os
import shutil
import signal
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent

#: Once, after the server is stopped.
SERVER_MARKER = "MV2FLOW_SERVER"

DEFAULT_BOOT_TIMEOUT_S = 1800
DEFAULT_REQUEST_TIMEOUT_S = 600
#: Seconds the server's launcher has to exit after SIGTERM before it is killed.
STOP_GRACE_S = 90

#: Requests to the server on this host, never through a proxy the environment names.
_DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _load(name: str, path: Path) -> Any:
    """Import the file at ``path`` as module ``name``, without putting its directory on the path.

    ``tune/`` holds a ``requests.py``: on ``sys.path`` it would shadow the
    library of that name for the rest of the process.
    """
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


runtime = _load(
    "mv2flow_server_runtime", _HERE.parent.parent / "infra" / "compute" / "server_runtime.py"
)
wio = _load("mv2flow_workload_io", _HERE / "workload_io.py")


def entry_point(name: str) -> str:
    """The console script ``name`` installed beside this interpreter, else the one on ``PATH``."""
    sibling = Path(sys.executable).with_name(name)
    if sibling.is_file():
        return str(sibling)
    return shutil.which(name) or name


def server_argv(spec: dict[str, Any], port: int) -> list[str]:
    """The server's command line: its ranks' launch, ``trtllm-serve``, the checkpoint and flags."""
    world = max(1, int(spec.get("world_size", 1)))
    return [
        *runtime.rank_launch(world, entry_point("trtllm-llmapi-launch")),
        entry_point("trtllm-serve"),
        str(spec["model"]),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        *(str(arg) for arg in spec.get("serve_args") or ()),
    ]


def data_url(location: str) -> str:
    """The image file at ``location`` (a path or ``file://`` URI) as a base64 ``data:`` URL."""
    source = Path(location.removeprefix("file://"))
    if not source.is_file():
        raise FileNotFoundError(f"image not found: {source}")
    if wio.is_lfs_pointer(source):
        raise ValueError(
            f"{source} is a git-lfs pointer, not an image: run `git lfs pull --include "
            f"{source.name}` in the tree under test"
        )
    mime = mimetypes.guess_type(source.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(source.read_bytes()).decode('ascii')}"


def inline_images(body: dict[str, Any]) -> dict[str, Any]:
    """A copy of ``body`` with every local image its messages name sent inline."""
    body = copy.deepcopy(body)
    for message in body.get("messages") or ():
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            url = str((part.get("image_url") or {}).get("url", ""))
            if not url.startswith(("http://", "https://", "data:")):
                part["image_url"] = {**part["image_url"], "url": data_url(url)}
    return body


def answer_of(payload: Any) -> tuple[str, Any, Any]:
    """``(text, finish_reason, completion tokens)`` of an OpenAI completion or chat response."""
    if not isinstance(payload, dict):
        return "", None, None
    choices = payload.get("choices") or [{}]
    choice = choices[0] if isinstance(choices[0], dict) else {}
    message = choice.get("message") or {}
    text = choice.get("text")
    if text is None and isinstance(message, dict):
        text = message.get("content")
    if text is not None and not isinstance(text, str):
        text = json.dumps(text)
    usage = payload.get("usage") or {}
    return text or "", choice.get("finish_reason"), usage.get("completion_tokens")


def ask(port: int, index: int, request: dict[str, Any], timeout: float) -> dict[str, Any]:
    """Send one request; its ``MV2FLOW_RECORD`` payload, whatever the server did."""
    record: dict[str, Any] = {
        "index": index,
        "id": str(request.get("id", index)),
        "endpoint": str(request["endpoint"]),
        "status": None,
        "text": "",
        "finish_reason": None,
        "tokens": None,
        "error": "",
    }
    try:
        data = json.dumps(inline_images(dict(request["body"]))).encode("utf-8")
    except (OSError, ValueError) as exc:
        record["error"] = f"the request could not be built: {exc}"
        return record
    http = urllib.request.Request(
        f"http://127.0.0.1:{port}{request['endpoint']}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with _DIRECT.open(http, timeout=timeout) as response:
            record["status"] = response.status
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        record["status"] = exc.code
        record["error"] = exc.read().decode("utf-8", errors="replace")[:2000]
        return record
    except (urllib.error.URLError, OSError, ValueError) as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record
    record["text"], record["finish_reason"], record["tokens"] = answer_of(payload)
    return record


def last_error(log_tail: str) -> str:
    """The last line of a server log that names an error, else its last line."""
    lines = [line.strip() for line in log_tail.splitlines() if line.strip()]
    for line in reversed(lines):
        if "Error" in line or "Exception" in line:
            return line[:300]
    return lines[-1][:300] if lines else ""


def _holders(world: int) -> set[int]:
    """The pids holding the server's devices; none where ``nvidia-smi`` cannot say."""
    try:
        return runtime.device_holders(world)
    except OSError:
        return set()


def stop(server: subprocess.Popen, world: int, before: set[int]) -> dict[str, Any]:
    """Stop the server's process group, then reap whatever still holds its devices.

    The ranks ``mpirun`` starts live outside the launcher's process group: a
    process that holds one of this job's devices now and did not before the
    server started is the server's.
    """
    return runtime.stop_server(server, world, before, grace_s=STOP_GRACE_S, holders=_holders)


def run(spec: dict[str, Any]) -> dict[str, Any]:
    """Serve, ask every request, stop; print each answer as it comes; return the server record."""
    world = max(1, int(spec.get("world_size", 1)))
    log_path = Path(spec["server_log"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    port = runtime.free_port()
    argv = server_argv(spec, port)
    env = {"TLLM_WORKER_USE_SINGLE_PROCESS": "1"} | dict(os.environ)
    report: dict[str, Any] = {"healthy": False, "argv": argv, "log": str(log_path)}
    before = _holders(world)
    try:
        server = runtime.start_server(argv, log_path, env)
    except OSError as exc:
        report["error"] = f"could not start {argv[0]}: {exc}"
        wio.emit(SERVER_MARKER, report)
        return report
    try:
        try:
            boot_s = runtime.wait_healthy(
                port, server, log_path, int(spec.get("boot_timeout", DEFAULT_BOOT_TIMEOUT_S))
            )
        except SystemExit as exc:
            if not isinstance(exc.code, str):
                raise  # a signal, not the server: see runtime._terminate
            report["error"] = exc.code.splitlines()[0]
        else:
            report.update(healthy=True, boot_s=round(boot_s, 1))
            timeout = float(spec.get("request_timeout", DEFAULT_REQUEST_TIMEOUT_S))
            for index, request in enumerate(spec.get("requests") or ()):
                wio.emit(wio.RECORD_MARKER, ask(port, index, dict(request), timeout))
    finally:
        report.update(stop(server, world, before))
        tail = runtime.tail(log_path)
        if not report["healthy"]:
            report["cause"] = last_error(tail)
            print(tail, flush=True)
        wio.emit(SERVER_MARKER, report)
    return report


def main(argv: list[str] | None = None) -> int:
    """Read the spec, serve, print records; 0 once the server record is out."""
    signal.signal(signal.SIGTERM, runtime._terminate)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", required=True, type=Path, help="the serve spec, a JSON file")
    options = parser.parse_args(argv)
    run(json.loads(options.spec.read_text(encoding="utf-8")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
