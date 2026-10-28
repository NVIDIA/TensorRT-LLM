# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``trtllm-modeling-v2 monitor ...``: the orchestrator's liveness and compute sessions.

It only reads files: nothing here runs a turn, and the verdicts stay with the orchestrator.
``wait`` prints the event on its first line and exits 0 for every event, so a caller
reads the first word.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from ..compute import scheduler as sched
from ..compute import sessions as sessions_mod
from ..files import read_json
from .heartbeat import DISPATCH_DIRNAME, ORCHESTRATOR_EXIT, ORCHESTRATOR_HEARTBEAT

#: An orchestrator heartbeat older than this means the orchestrator is gone.
ORCHESTRATOR_STALE_S = 120.0


class MonitorError(RuntimeError):
    """A monitor action that cannot be carried out."""


@dataclass
class MonitorConfig:
    """What a monitor works against, resolved once from the workspace."""

    workspace: Path
    task: Mapping[str, Any] = field(default_factory=dict)

    @property
    def dispatch(self) -> Path:
        return self.workspace / DISPATCH_DIRNAME

    @property
    def sessions(self) -> tuple[sessions_mod.SessionSpec, ...]:
        return sessions_mod.sessions_from_task(self.task)


def load_config(workspace: str | Path) -> MonitorConfig:
    """Resolve a monitor's configuration from ``<workspace>/task.yaml``."""
    workspace = Path(workspace).resolve()
    if not (workspace / DISPATCH_DIRNAME).is_dir() and not (workspace / "task.yaml").is_file():
        raise MonitorError(
            f"{workspace} is not a trtllm_modeling_v2 workspace (no task.yaml and no {DISPATCH_DIRNAME}/)"
        )
    try:
        loaded = yaml.safe_load((workspace / "task.yaml").read_text(encoding="utf-8"))
        task = loaded if isinstance(loaded, dict) else {}
    except (OSError, yaml.YAMLError):
        task = {}
    return MonitorConfig(workspace=workspace, task=task)


# ------------------------------------------------------------- orchestrator


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
    return (
        "unknown",
        "no orchestrator heartbeat",
    )


def _clock(epoch: Any) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(epoch)))
    except (TypeError, ValueError):
        return "?"


# -------------------------------------------------------------------- waiting


@dataclass(frozen=True)
class Event:
    """Why :func:`wait` returned."""

    kind: str
    detail: str = ""

    def render(self) -> str:
        return f"{self.kind}  {self.detail}".rstrip()


def poll_once(
    cfg: MonitorConfig, *, watch_sessions: bool = True, now: float | None = None
) -> Event | None:
    """The first thing that needs the operator right now, or ``None``."""
    state, detail = orchestrator_state(cfg, now)
    if state == "exited":
        return Event("ORCHESTRATOR_EXITED", detail)
    return session_alert(cfg, now) if watch_sessions else None


def session_alert(cfg: MonitorConfig, now: float | None = None) -> Event | None:
    """The first session that needs a hand, or ``None`` while none does.

    ``SESSION_STALE``: its scheduler stopped heart-beating (the allocation is gone).
    ``SESSION_ENDING``: it is in hand-over and no successor is acknowledged (``session
    allocate``, then ``session ack``). ``SESSION_QUARANTINED``: devices are out of service for a
    process the safe-kill rule forbids it to stop, or for memory not released. A spool no watcher
    ever served is skipped.
    """
    now = time.time() if now is None else now
    for session in cfg.sessions:
        spool = Path(session.spool)
        records = sched.watcher_records(spool)
        if not records:
            continue
        newest = max(records, key=lambda r: float(r["alive_epoch"]))
        age = now - float(newest["alive_epoch"])
        if age > sched.WATCHER_STALE_S:
            running = sorted(
                p.name[: -len(".sh")] for p in sched.SpoolPaths(spool).running.glob("*.sh")
            )
            return Event(
                "SESSION_STALE",
                f"{spool} heartbeat {age:.0f}s old (job {newest.get('slurm_job')}); "
                f"in running/: {' '.join(running) or 'nothing'}",
            )
        handover = newest.get("handover") or {}
        allocation = str(newest.get("slurm_job") or "")
        acks = sched.SpoolPaths(spool).read_acks()
        if (
            handover
            and not handover.get("successor")
            and sched.following_ack(acks, allocation, session.name) is None
        ):
            left = float(newest.get("ends_at_epoch") or now) - now
            return Event(
                "SESSION_ENDING",
                f"session {session.name} (allocation {allocation}) ends in {left / 60:.0f} min and "
                f"no successor is acknowledged; run `trtllm-modeling-v2 session allocate --task "
                f"{cfg.workspace / 'task.yaml'} --name {session.name}`",
            )
        memory = newest.get("quarantined_memory") or {}
        held = [d for d in newest.get("quarantined") or () if d not in memory]
        if held:
            stuck = [j["name"] for j in newest.get("running") or [] if j.get("unsafe_pids")]
            return Event(
                "SESSION_QUARANTINED",
                f"{spool} devices {','.join(held)} held for MPI-attached processes of "
                f"{', '.join(stuck) or 'a finished job'}",
            )
        if memory:
            return Event(
                "SESSION_QUARANTINED",
                f"{spool} devices {','.join(sorted(memory))} quarantined: memory not released ("
                + ", ".join(
                    f"{device}: {info.get('used_mib')} MiB in use since {info.get('since')}"
                    for device, info in sorted(memory.items())
                )
                + ")",
            )
    return None


def wait(
    cfg: MonitorConfig,
    *,
    timeout_s: float = 540.0,
    poll_s: float = 10.0,
    watch_sessions: bool = True,
) -> Event:
    """Block until :func:`poll_once` has something, or ``timeout_s`` passes.

    The default timeout keeps one call inside a tool-call limit of ten minutes.
    """
    end = time.monotonic() + timeout_s
    while True:
        event = poll_once(cfg, watch_sessions=watch_sessions)
        if event is not None:
            return event
        remaining = end - time.monotonic()
        if remaining <= 0:
            return Event("TIMEOUT", "nothing new")
        time.sleep(min(poll_s, remaining))


# --------------------------------------------------------------------- status


def status(cfg: MonitorConfig, now: float | None = None) -> dict:
    """The orchestrator, where correctness stands and the sessions, as data."""
    now = time.time() if now is None else now
    state, detail = orchestrator_state(cfg, now)
    return {
        "workspace": str(cfg.workspace),
        "orchestrator": {"state": state, "detail": detail},
        "correctness": _correctness(cfg.workspace),
        "sessions": sessions_mod.session_status(cfg.task, now) if cfg.sessions else [],
    }


def _correctness(workspace: Path) -> str | None:
    """Where the run last delivered correctness, or ``None`` before it has."""
    from ...run.validation import contract as contract_mod

    milestone = contract_mod.latest_milestone(workspace)
    return milestone.describe() if milestone else None


def format_status(data: Mapping[str, Any]) -> str:
    """:func:`status` as lines."""
    lines = [
        f"orchestrator: {data['orchestrator']['state']} ({data['orchestrator']['detail']})",
    ]
    if data.get("correctness"):
        lines.append(f"correctness: {data['correctness']}")
    for session in data["sessions"]:
        beat = (
            "never" if session["heartbeat_age_s"] is None else f"{session['heartbeat_age_s']:.0f}s"
        )
        lines.append(
            f"session {session['name']:<8} {session['role']:<12} heartbeat {beat} "
            f"running={session['running']} pending={session['pending']}"
        )
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
    wait = sub.add_parser("wait", help="block until there is something to do")
    wait.add_argument("--timeout", type=float, default=540.0)
    wait.add_argument("--poll", type=float, default=10.0)
    wait.add_argument("--no-sessions", action="store_true", help="ignore the compute sessions")
    show = sub.add_parser("status", help="the orchestrator, correctness and the sessions")
    show.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.workspace:
        print("error: pass --workspace or set MV2FLOW_WORKSPACE", file=sys.stderr)
        return 2
    try:
        cfg = load_config(args.workspace)
        if args.action == "wait":
            event = wait(
                cfg,
                timeout_s=args.timeout,
                poll_s=args.poll,
                watch_sessions=not args.no_sessions,
            )
            print(event.render())
            print(format_status(status(cfg)))
            return 0
        if args.action == "status":
            data = status(cfg)
            print(json.dumps(data, indent=2) if args.json else format_status(data))
            return 0
    except MonitorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 2


__all__ = [
    "ORCHESTRATOR_STALE_S",
    "MonitorConfig",
    "MonitorError",
    "Event",
    "build_parser",
    "format_status",
    "load_config",
    "main",
    "orchestrator_state",
    "poll_once",
    "session_alert",
    "status",
    "wait",
]
