# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``trtllm-modeling-v2 monitor ...``: what is blocked, the orchestrator's liveness, the pool's events.

It only reads files: nothing here runs a turn, and the verdicts stay with the orchestrator.
``wait`` prints the event on its first line and exits 0 for every event, so a caller
reads the first word. ``TASK_BLOCKED <task> <its receipt>`` comes first and at once: a task
is blocked and waits for ``trtllm-modeling-v2 resolve``; ``RUN_BLOCKED <journal.md>``
when a final pass did not hold and no task owns the failure. The pool's events
(``POOL_SUBMIT_FAILED``, ``POOL_CAPACITY``, ``POOL_DEVICE_HELD``) are the
ones the controller derived on its last pass (``infra/compute/pool.py``); this module acts
on none of them. ``TURN_LONG <task> <its activity log>`` comes last and holds nothing: an
agent has been in one turn for :data:`LONG_TURN_S`, and the session that watches the run
looks at what it is doing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from ..compute import controller as controller_mod
from ..compute import pool as pool_mod
from ..compute import scheduler as sched
from ..files import read_json
from .heartbeat import DISPATCH_DIRNAME, ORCHESTRATOR_EXIT, ORCHESTRATOR_HEARTBEAT

#: An orchestrator heartbeat older than this means the orchestrator is gone.
ORCHESTRATOR_STALE_S = 120.0


#: An agent turn in flight this long is said (``TURN_LONG``), once per turn the caller has
#: in hand. One hour: the turns of whole runs took minutes, a kernel row's ten to thirty,
#: so this is twice the longest ordinary one; and it is a third of the shortest limit a
#: turn has by itself (three hours), so a turn that left the purpose is seen while most
#: of what it would spend is still unspent.
LONG_TURN_S = 3600.0


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
    def spool(self) -> str | None:
        """The run's one spool, once the spec names it."""
        environment = self.task.get("environment")
        spool = environment.get("spool") if isinstance(environment, Mapping) else None
        return str(spool) if spool else None


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
        detail = f"pid {gone.get('pid')} on {gone.get('host')} exited at {_clock(gone.get('at'))}"
        if gone.get("why"):
            # What the run itself said of its exit (``cli.py``).
            detail += f": {gone['why']}"
        # A controller that stopped for its own time limit said what continues the run.
        handed = controller_mod.last_handover(cfg.workspace)
        if handed and 0 <= float(gone.get("at", 0)) - float(handed.get("at", 0)) <= 900:
            detail += f"; {handed.get('line')}"
        return "exited", detail
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
    cfg: MonitorConfig,
    *,
    watch_sessions: bool = True,
    now: float | None = None,
    ignore: Sequence[str] = (),
) -> Event | None:
    """The first thing that needs the outer session right now, or ``None``.

    A blocked task or the run's own question first (``ignore`` names the tasks, ``run``,
    the pool's events or the long turn the caller already has in hand), then an
    orchestrator that exited, then what the pool's files alone show, then a turn that
    has run long.
    """
    found = blocked_alert(cfg, ignore)
    if found is not None:
        return found
    state, detail = orchestrator_state(cfg, now)
    if state == "exited":
        return Event("ORCHESTRATOR_EXITED", detail)
    found = pool_alert(cfg, now, ignore) if watch_sessions else None
    return found or long_turn_alert(cfg, now, ignore)


def long_turn_alert(
    cfg: MonitorConfig, now: float | None = None, ignore: Sequence[str] = ()
) -> Event | None:
    """A running task whose agent has been in one turn for :data:`LONG_TURN_S`, as an event.

    It holds nothing and judges nothing: a long turn may be honest work. It is read from
    the turn's own record (``turns/<id>.request.json`` with no ``.done`` beside it), the
    newest one of a task the graph has as running, and is passed over by that turn's id.
    """
    from ...agents.turns import TURNS_DIRNAME

    now = time.time() if now is None else now
    running = {t.id: t for t in _graph(cfg.workspace).tasks if t.status == "running"}
    newest: dict[str, tuple[float, Path, dict]] = {}
    for path in (cfg.workspace / TURNS_DIRNAME).glob("*.request.json") if running else ():
        record = read_json(path) or {}
        work = str(record.get("work") or record.get("what") or "")
        started = float(record.get("started_at") or 0)
        if work in running and started > newest.get(work, (0.0,))[0]:
            newest[work] = (started, path, record)
    for work, (started, path, record) in sorted(newest.items()):
        turn = path.name.removesuffix(".request.json")
        if now - started < LONG_TURN_S or turn in ignore:
            continue
        if path.with_name(f"{turn}.done").exists():
            continue  # the turn is over: its checks run, or its outcome is being recorded
        task = running[work]
        return Event(
            "TURN_LONG",
            f"{work} {record.get('activity_file') or path}  {work} ({task.kind}), attempt "
            f"{task.attempts}, has been in one turn ({turn}) for {(now - started) / 60:.0f} "
            f"minutes and holds nothing: read what it is doing. `resolve skip {work}` or "
            f"`resolve retry {work} --note ...` ends the turn; `wait --ignore {turn}` passes "
            "over it",
        )
    return None


def _graph(workspace: Path):
    """The run's graph as its controller last saved it; an empty one before there is any."""
    from ...run import graph as graph_mod

    path = Path(workspace) / graph_mod.GRAPH_FILENAME
    try:
        return graph_mod.Graph.load(path)
    except (OSError, ValueError):
        return graph_mod.Graph()


def blocked_alert(cfg: MonitorConfig, ignore: Sequence[str] = ()) -> Event | None:
    """The first block the caller does not have in hand, as an event.

    A task's is ``TASK_BLOCKED <task> <its receipt>``, with who blocked it, what was
    observed, what it asks and what it holds. The run's own is ``RUN_BLOCKED
    <journal.md>``. A task that is done and still asks is ``TASK_ASKS <task> <its
    receipt>``: it holds nothing.
    """
    from ...run import escalation as escalation_mod
    from ...run import graph as graph_mod
    from ...run.execution import handoff

    graph = _graph(cfg.workspace)
    tasks_dir = cfg.workspace / escalation_mod.TASKS_DIRNAME
    for task in graph.tasks:
        if task.status != "blocked" or task.id in ignore:
            continue
        receipt = handoff.path_for(tasks_dir, task.id, task.attempts)
        said = escalation_mod.describe(task, graph_mod.blocked_by(graph, task.id))
        return Event("TASK_BLOCKED", f"{task.id} {receipt}  {said}")
    if graph.run_blocked and escalation_mod.RUN not in ignore:
        said = escalation_mod.describe_run(graph.run_blocked)
        return Event("RUN_BLOCKED", f"{cfg.workspace / 'journal.md'}  {said}")
    for task in graph.tasks:
        if task.status != "blocked" and task.block and task.id not in ignore:
            receipt = handoff.path_for(tasks_dir, task.id, task.attempts)
            return Event("TASK_ASKS", f"{task.id} {receipt}  {escalation_mod.describe(task)}")
    return None


def pool_alert(
    cfg: MonitorConfig, now: float | None = None, ignore: Sequence[str] = ()
) -> Event | None:
    """The first pool event that stands and the caller does not have in hand, or ``None``.

    The controller derives them each pass and writes them beside the spool; while its snapshot
    is fresh that is what is said.
    """
    if not cfg.spool:
        return None
    for event in pool_mod.standing_events(cfg.spool, now):
        if event.kind not in ignore:
            return Event(event.kind, event.detail)
    return None


def wait(
    cfg: MonitorConfig,
    *,
    timeout_s: float = 540.0,
    poll_s: float = 10.0,
    watch_sessions: bool = True,
    ignore: Sequence[str] = (),
) -> Event:
    """Block until :func:`poll_once` has something, or ``timeout_s`` passes.

    The default timeout keeps one call inside a tool-call limit of ten minutes.
    """
    end = time.monotonic() + timeout_s
    while True:
        event = poll_once(cfg, watch_sessions=watch_sessions, ignore=ignore)
        if event is not None:
            return event
        remaining = end - time.monotonic()
        if remaining <= 0:
            return Event("TIMEOUT", "nothing new")
        time.sleep(min(poll_s, remaining))


# --------------------------------------------------------------------- status


def status(cfg: MonitorConfig, now: float | None = None) -> dict:
    """The orchestrator, where the run was onboarded and the sessions, as data."""
    now = time.time() if now is None else now
    state, detail = orchestrator_state(cfg, now)
    return {
        "workspace": str(cfg.workspace),
        "orchestrator": {"state": state, "detail": detail},
        "onboarded": _onboarded(cfg.workspace),
        "blocked": _blocked(cfg.workspace),
        "pool": _pool(cfg),
    }


def _pool(cfg: MonitorConfig) -> dict | None:
    """The controller's last view of the pool: its holders and what stands; ``None`` before one."""
    snapshot = read_json(sched.SpoolPaths(cfg.spool).pool) if cfg.spool else None
    if not snapshot:
        return None
    return {key: snapshot.get(key) for key in ("written", "kind", "stopping", "holders", "events")}


def _blocked(workspace: Path) -> list[str]:
    """What waits for an answer of the outer session, one line each."""
    from ...run import escalation as escalation_mod

    return escalation_mod.waiting(_graph(workspace))


def _onboarded(workspace: Path) -> str | None:
    """The commit ``onboarded`` was pinned on, with what it could not verify; ``None`` before it was."""
    from ...run.records import report as report_mod

    record = report_mod.onboarded(workspace)
    if not record or not record.get("commit"):
        return None
    said = str(record["commit"])
    if record.get("unverified"):
        said += f" (not verified: {'; '.join(str(u) for u in record['unverified'])})"
    return said


def format_status(data: Mapping[str, Any]) -> str:
    """:func:`status` as lines."""
    lines = [
        f"orchestrator: {data['orchestrator']['state']} ({data['orchestrator']['detail']})",
    ]
    if data.get("onboarded"):
        lines.append(f"onboarded: {data['onboarded']}")
    for line in data.get("blocked") or ():
        lines.append(f"blocked: {line}")
    pool = data.get("pool") or {}
    for holder in pool.get("holders") or ():
        lines.append(
            f"holder {holder.get('name')} nodes={holder.get('nodes')} {holder.get('state')} "
            f"job={holder.get('job_id') or '-'} watchers={holder.get('watchers')}"
            + (" idle" if holder.get("idle") else "")
            + (" asked to end" if holder.get("asked_to_end") else "")
        )
    for event in pool.get("events") or ():
        lines.append(f"{event.get('kind')}  {event.get('detail')}")
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
    wait.add_argument("--no-sessions", action="store_true", help="ignore the pool's events")
    wait.add_argument(
        "--ignore",
        action="append",
        default=[],
        metavar="NAME",
        help="a blocked task, `run`, a pool event or a long turn (by the id its line names) "
        "already in hand (asked of a human, say): wait for the next thing",
    )
    show = sub.add_parser("status", help="the orchestrator, onboarded and the pool")
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
                ignore=args.ignore,
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
    "blocked_alert",
    "build_parser",
    "format_status",
    "load_config",
    "main",
    "orchestrator_state",
    "poll_once",
    "pool_alert",
    "status",
    "wait",
]
