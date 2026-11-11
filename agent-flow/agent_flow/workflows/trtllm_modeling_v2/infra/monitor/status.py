# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``trtllm-modeling-v2 monitor ...``: whether the orchestrator is alive, and where the run stands.

It only reads files: nothing here runs a turn, and the verdicts stay with the orchestrator.
``wait`` prints the event on its first line and exits 0 for every event, so a caller
reads the first word: ``ORCHESTRATOR_EXITED`` or ``TIMEOUT``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from ..files import read_json
from .heartbeat import DISPATCH_DIRNAME, ORCHESTRATOR_EXIT, ORCHESTRATOR_HEARTBEAT

#: An orchestrator heartbeat older than this means the orchestrator is gone.
ORCHESTRATOR_STALE_S = 120.0


class MonitorError(RuntimeError):
    """A monitor action that cannot be carried out."""


@dataclass
class MonitorConfig:
    """What a monitor works against: the run's workspace."""

    workspace: Path

    @property
    def dispatch(self) -> Path:
        return self.workspace / DISPATCH_DIRNAME


def load_config(workspace: str | Path) -> MonitorConfig:
    workspace = Path(workspace).resolve()
    if not (workspace / DISPATCH_DIRNAME).is_dir() and not (workspace / "task.yaml").is_file():
        raise MonitorError(
            f"{workspace} is not a trtllm_modeling_v2 workspace (no task.yaml and no {DISPATCH_DIRNAME}/)"
        )
    return MonitorConfig(workspace=workspace)


def orchestrator_state(cfg: MonitorConfig, now: float | None = None) -> tuple[str, str]:
    """``("running"|"exited"|"unknown", detail)`` for the orchestrator, from :mod:`.heartbeat`."""
    now = time.time() if now is None else now
    beat = read_json(cfg.dispatch / ORCHESTRATOR_HEARTBEAT)
    gone = read_json(cfg.dispatch / ORCHESTRATOR_EXIT)
    if gone and (not beat or float(gone.get("at", 0)) >= float(beat.get("at", 0))):
        return (
            "exited",
            f"pid {gone.get('pid')} on {gone.get('host')} exited at {_clock(gone.get('at'))}",
        )
    if beat:
        age = now - float(beat.get("at", 0))
        if age > ORCHESTRATOR_STALE_S:
            return "exited", (
                f"pid {beat.get('pid')} on {beat.get('host')} stopped heart-beating {age:.0f}s ago "
                f"(killed, or its allocation ended)"
            )
        return "running", f"pid {beat.get('pid')} on {beat.get('host')}, heartbeat {age:.0f}s ago"
    return "unknown", "no orchestrator heartbeat"


def _clock(epoch: Any) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(epoch)))
    except (TypeError, ValueError):
        return "?"


@dataclass(frozen=True)
class Event:
    """Why :func:`wait` returned."""

    kind: str
    detail: str = ""

    def render(self) -> str:
        return f"{self.kind}  {self.detail}".rstrip()


def wait(cfg: MonitorConfig, *, timeout_s: float = 540.0, poll_s: float = 10.0) -> Event:
    """Block until the orchestrator has exited, or ``timeout_s`` passes.

    The default timeout keeps one call inside a tool-call limit of ten minutes.
    """
    end = time.monotonic() + timeout_s
    while True:
        state, detail = orchestrator_state(cfg)
        if state == "exited":
            return Event("ORCHESTRATOR_EXITED", detail)
        remaining = end - time.monotonic()
        if remaining <= 0:
            return Event("TIMEOUT", "nothing new")
        time.sleep(min(poll_s, remaining))


def status(cfg: MonitorConfig, now: float | None = None) -> dict:
    """The orchestrator and where the run stands, as data."""
    state, detail = orchestrator_state(cfg, time.time() if now is None else now)
    return {
        "workspace": str(cfg.workspace),
        "orchestrator": {"state": state, "detail": detail},
        "run": _run(cfg.workspace),
    }


def _yaml(path: Path) -> Any:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None


def _run(workspace: Path) -> str | None:
    """Where the run stands: onboarded or not, and its tasks by status."""
    state = read_json(workspace / ".trtllm_modeling_v2_state.json")
    if not state:
        return None
    record = state.get("onboarded") or {}
    said = [f"onboarded at {str(record.get('commit'))[:12]}" if record else "not onboarded"]
    tasks = (_yaml(workspace / "tasks.yaml") or {}).get("tasks")
    if tasks:
        counts: dict[str, int] = {}
        for task in tasks:
            counts[str(task.get("status"))] = counts.get(str(task.get("status")), 0) + 1
        said.append("tasks " + ", ".join(f"{n} {status}" for status, n in sorted(counts.items())))
        running = [str(t.get("id")) for t in tasks if t.get("status") == "running"]
        if running:
            said.append("running " + ", ".join(running))
    return "; ".join(said)


def format_status(data: Mapping[str, Any]) -> str:
    """:func:`status` as lines."""
    lines = [f"orchestrator: {data['orchestrator']['state']} ({data['orchestrator']['detail']})"]
    if data.get("run"):
        lines.append(f"run: {data['run']}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.workspace:
        print("error: pass --workspace or set MV2FLOW_WORKSPACE", file=sys.stderr)
        return 2
    try:
        cfg = load_config(args.workspace)
    except MonitorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.action == "wait":
        print(wait(cfg, timeout_s=args.timeout, poll_s=args.poll).render())
        print(format_status(status(cfg)))
        return 0
    data = status(cfg)
    print(json.dumps(data, indent=2) if args.json else format_status(data))
    return 0


__all__ = [
    "ORCHESTRATOR_STALE_S",
    "Event",
    "MonitorConfig",
    "MonitorError",
    "build_parser",
    "format_status",
    "load_config",
    "main",
    "orchestrator_state",
    "status",
    "wait",
]
