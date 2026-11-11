# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``trtllm-modeling-v2 monitor ...``: whether the orchestrator is alive, and where the run stands.

The orchestrator rewrites ``dispatch/orchestrator.heartbeat`` in its workspace every
:data:`ORCHESTRATOR_HEARTBEAT_S` and leaves ``orchestrator.exit`` when it ends; each holds
``{"pid", "host", "started", "at"}``. The monitor only reads files, so it works from any
node that sees the workspace: nothing here runs a turn or a check. ``wait`` prints the
event on its first line and exits 0 for every event, so a caller reads the first word:
``ORCHESTRATOR_EXITED`` or ``TIMEOUT``.
"""

from __future__ import annotations

import argparse
import atexit
import errno
import json
import os
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any

import yaml

from .files import read_json, write_text_atomic

#: Subdirectory of the workspace the orchestrator and the monitor share.
DISPATCH_DIRNAME = "dispatch"
ORCHESTRATOR_HEARTBEAT = "orchestrator.heartbeat"
ORCHESTRATOR_EXIT = "orchestrator.exit"
#: How often the orchestrator says it is alive, and how old a heartbeat means it is gone.
ORCHESTRATOR_HEARTBEAT_S = 30.0
ORCHESTRATOR_STALE_S = 120.0


class OrchestratorHeartbeat:
    """This orchestrator's liveness files; one per process, from the workspace lock's holder."""

    def __init__(self, workspace: Path | str, interval_s: float) -> None:
        self.root = Path(workspace) / DISPATCH_DIRNAME
        self.root.mkdir(parents=True, exist_ok=True)
        self.record = {"pid": os.getpid(), "host": socket.gethostname(), "started": time.time()}
        self._stop = threading.Event()
        (self.root / ORCHESTRATOR_EXIT).unlink(missing_ok=True)
        beat = threading.Thread(target=self._beat, args=(interval_s,), daemon=True)
        beat.start()
        atexit.register(self.mark_exit)

    def _write(self, name: str) -> None:
        write_text_atomic(self.root / name, json.dumps({**self.record, "at": time.time()}) + "\n")

    def _beat(self, interval_s: float) -> None:
        while True:
            try:
                self._write(ORCHESTRATOR_HEARTBEAT)
            except OSError as exc:
                # A full filesystem is transient (a shared quota); anything else means the
                # dispatch directory is gone.
                if exc.errno not in (errno.ENOSPC, errno.EDQUOT):
                    return
            if self._stop.wait(interval_s):
                return

    def mark_exit(self) -> None:
        """Stop beating and write ``orchestrator.exit``; what ``atexit`` calls."""
        self._stop.set()
        try:
            self._write(ORCHESTRATOR_EXIT)
        except OSError:
            pass


def orchestrator_state(workspace: Path, now: float | None = None) -> tuple[str, str]:
    """``("running"|"exited"|"unknown", detail)`` for the orchestrator of ``workspace``."""
    now = time.time() if now is None else now
    beat = read_json(workspace / DISPATCH_DIRNAME / ORCHESTRATOR_HEARTBEAT)
    gone = read_json(workspace / DISPATCH_DIRNAME / ORCHESTRATOR_EXIT)
    if gone and (not beat or float(gone.get("at", 0)) >= float(beat.get("at", 0))):
        at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(gone.get("at", 0))))
        return "exited", f"pid {gone.get('pid')} on {gone.get('host')} exited at {at}"
    if not beat:
        return "unknown", "no orchestrator heartbeat"
    who, age = f"pid {beat.get('pid')} on {beat.get('host')}", now - float(beat.get("at", 0))
    if age > ORCHESTRATOR_STALE_S:
        return (
            "exited",
            f"{who} stopped heart-beating {age:.0f}s ago (killed, or its allocation ended)",
        )
    return "running", f"{who}, heartbeat {age:.0f}s ago"


def wait(workspace: Path, *, timeout_s: float = 540.0, poll_s: float = 10.0) -> str:
    """Block until the orchestrator has exited, or ``timeout_s`` passes; the event's line.

    The default timeout keeps one call inside a tool-call limit of ten minutes.
    """
    end = time.monotonic() + timeout_s
    while True:
        state, detail = orchestrator_state(workspace)
        if state == "exited":
            return f"ORCHESTRATOR_EXITED  {detail}"
        if end - time.monotonic() <= 0:
            return "TIMEOUT  nothing new"
        time.sleep(min(poll_s, max(0.0, end - time.monotonic())))


def run_state(workspace: Path) -> str | None:
    """Where the run stands: onboarded or not, and its tasks by status."""
    state = read_json(workspace / ".trtllm_modeling_v2_state.json")
    if not state:
        return None
    record = state.get("onboarded") or {}
    said = [f"onboarded at {str(record.get('commit'))[:12]}" if record else "not onboarded"]
    try:
        tasks = (yaml.safe_load((workspace / "tasks.yaml").read_text(encoding="utf-8")) or {}).get(
            "tasks"
        )
    except (OSError, yaml.YAMLError):
        tasks = None
    counts: dict[str, int] = {}
    for task in tasks or ():
        counts[str(task.get("status"))] = counts.get(str(task.get("status")), 0) + 1
    if counts:
        said.append("tasks " + ", ".join(f"{n} {status}" for status, n in sorted(counts.items())))
    running = [str(t.get("id")) for t in tasks or () if t.get("status") == "running"]
    if running:
        said.append("running " + ", ".join(running))
    return "; ".join(said)


def status(workspace: Path) -> dict[str, Any]:
    """The orchestrator and where the run stands, as data."""
    state, detail = orchestrator_state(workspace)
    orchestrator = {"state": state, "detail": detail}
    return {"workspace": str(workspace), "orchestrator": orchestrator, "run": run_state(workspace)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="trtllm-modeling-v2 monitor", description=__doc__.splitlines()[0]
    )
    parser.add_argument(
        "--workspace",
        default=os.environ.get("MV2FLOW_WORKSPACE"),
        help="the run's workspace (default $MV2FLOW_WORKSPACE)",
    )
    sub = parser.add_subparsers(dest="action", required=True)
    waiting = sub.add_parser("wait", help="block until the orchestrator has exited")
    waiting.add_argument("--timeout", type=float, default=540.0)
    waiting.add_argument("--poll", type=float, default=10.0)
    show = sub.add_parser("status", help="the orchestrator and where the run stands")
    show.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if not args.workspace:
        print("error: pass --workspace or set MV2FLOW_WORKSPACE", file=sys.stderr)
        return 2
    workspace = Path(args.workspace).resolve()
    if not (workspace / DISPATCH_DIRNAME).is_dir() and not (workspace / "task.yaml").is_file():
        said = f"no task.yaml and no {DISPATCH_DIRNAME}/"
        print(f"error: {workspace} is not a trtllm_modeling_v2 workspace ({said})", file=sys.stderr)
        return 1
    if args.action == "wait":
        print(wait(workspace, timeout_s=args.timeout, poll_s=args.poll))
    data = status(workspace)
    if args.action == "status" and args.json:
        print(json.dumps(data, indent=2))
        return 0
    print(f"orchestrator: {data['orchestrator']['state']} ({data['orchestrator']['detail']})")
    if data.get("run"):
        print(f"run: {data['run']}")
    return 0
