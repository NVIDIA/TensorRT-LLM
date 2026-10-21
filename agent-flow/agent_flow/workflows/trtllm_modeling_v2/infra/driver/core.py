# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The operator's loop of any run: the orchestrator's liveness, reloads, the compute sessions.

A run whose turns run in process (the default) has an operator who watches for
the orchestrator stopping, for reloads and for the compute sessions; with
``--agent-transport external`` the same loop also hands out turns, through
``infra/external/driver.py``.

Nothing here runs a turn. It reads files; the verdicts stay with the
orchestrator.
"""

from __future__ import annotations

import fcntl
import os
import socket
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from ..dispatch import ORCHESTRATOR_EXIT, ORCHESTRATOR_HEARTBEAT
from ..external import driver as external_driver
from ..substrate import sessions as sessions_mod
from ..substrate import watchdog
from ..substrate.scheduler import read_json
from . import reload as reload_mod
from .config import DriverConfig, DriverError

#: An orchestrator heartbeat older than this means the orchestrator is gone.
ORCHESTRATOR_STALE_S = 120.0

#: The file the orchestrator holds an exclusive ``flock`` on for its whole life,
#: under every transport (``cli._claim_workspace``).
WORKSPACE_LOCK = ".orchestrator.lock"


# ------------------------------------------------------------- orchestrator


def orchestrator_state(cfg: DriverConfig, now: float | None = None) -> tuple[str, str]:
    """``("running"|"exited"|"unknown", detail)`` for the orchestrator on this workspace.

    The external transport's heartbeat says it while it speaks for the
    process the run's ledger (``inflight.json``) names. A process with no
    heartbeat (its turns run in process) is read from that ledger and the
    workspace lock instead (:func:`_ledger_state`).
    """
    now = time.time() if now is None else now
    if cfg.orchestrator_job:
        spool, name = cfg.orchestrator_job
        receipt = spool / "done" / f"{name}.rc"
        if receipt.exists():
            record = read_json(spool / "done" / f"{name}.json") or {}
            return "exited", (
                f"spool job {name} finished rc={receipt.read_text(encoding='utf-8').strip()} "
                f"reason={record.get('reason', 'exit')}; log {spool / 'logs' / f'{name}.log'}"
            )
    beat = read_json(cfg.dispatch / ORCHESTRATOR_HEARTBEAT)
    gone = read_json(cfg.dispatch / ORCHESTRATOR_EXIT)
    ledger = reload_mod.read_inflight(cfg.workspace)
    if ledger and not _speaks_for(beat or gone, ledger):
        return _ledger_state(cfg, ledger, now)
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


def _speaks_for(beat: Mapping[str, Any] | None, ledger: Mapping[str, Any]) -> bool:
    """Whether a heartbeat record is the ledger's process's, or a later one's.

    A process's transport starts after the process does, so its heartbeat's
    ``started`` is never before the ledger's ``started_at``; a heartbeat from
    before it is an earlier process's, left in ``dispatch/``. Either side
    without a start time is taken at its word.
    """
    if not beat:
        return False
    started, since = beat.get("started"), ledger.get("started_at")
    if started is None or since is None:
        return True
    try:
        return float(started) >= float(since)
    except (TypeError, ValueError):
        return True


def _workspace_lock(cfg: DriverConfig) -> str | None:
    """``"held"`` or ``"free"``: whether a process holds the workspace lock; ``None`` if unknown.

    Probed with a shared, non-blocking ``flock`` on a read-only descriptor,
    released at once: it never truncates the file and never waits.
    """
    try:
        fd = os.open(cfg.workspace / WORKSPACE_LOCK, os.O_RDONLY)
    except OSError:
        return None
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except BlockingIOError:
        return "held"
    except OSError:
        return None
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return "free"
    finally:
        os.close(fd)


def _ledger_state(cfg: DriverConfig, ledger: Mapping[str, Any], now: float) -> tuple[str, str]:
    """The orchestrator's state from its ledger and the workspace lock, without a heartbeat.

    A held lock is a live orchestrator from any host. A free one means none
    only on the ledger's own host: a lock is not always visible across
    hosts. A run that closed says so in the ledger; a re-exec for a reload
    lets the lock go for the moment it takes.
    """
    host = str(ledger.get("host") or "?")
    who = f"pid {ledger.get('pid')} on {host}"
    stopped = ledger.get("stopped") or {}
    how = str(stopped.get("how") or "")
    lock = _workspace_lock(cfg)
    if lock == "held":
        return "running", f"{who} holds the workspace lock (no heartbeat: turns run in process)"
    if how.startswith("closed"):
        verdict = how.partition(":")[2] or "no verdict"
        return "exited", f"{who} closed the run ({verdict}) at {_clock(stopped.get('at'))}"
    if how == "reload" and now - float(stopped.get("at") or 0) < ORCHESTRATOR_STALE_S:
        return "running", f"{who} is re-executing itself for a reload"
    if lock == "free" and host == socket.gethostname():
        return "exited", (
            f"{who} no longer holds the workspace lock: it was stopped, or it crashed, "
            f"or its host went away; its log says which"
        )
    return "unknown", (
        f"{who} writes no heartbeat (turns run in process) and has not closed the run; "
        f"a driver on {host} also sees it stop"
    )


def _clock(epoch: Any) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(epoch)))
    except (TypeError, ValueError):
        return "?"


# --------------------------------------------------------------------- reload


def request_reload(
    cfg: DriverConfig,
    *,
    force: bool = False,
    timeout_s: float = reload_mod.DEFAULT_DRAIN_TIMEOUT_S,
    by: str | None = None,
) -> dict:
    """Ask the orchestrator to drain and re-execute itself; refused when none is running."""
    state, detail = orchestrator_state(cfg)
    if state == "exited":
        raise DriverError(
            f"the orchestrator is not running ({detail}); launching it again loads the current code"
        )
    return reload_mod.request(cfg.workspace, force=force, timeout_s=timeout_s, by=by)


def reload_state(cfg: DriverConfig, now: float | None = None) -> dict | None:
    """The reload under way or the latest one, ``{"phase", "detail", ...}``, or ``None``.

    ``requested`` is a request the orchestrator has not taken up yet; it does
    in its planned loop, within a few seconds.
    """
    now = time.time() if now is None else now
    request = reload_mod.read_request(cfg.workspace)
    record = reload_mod.read_record(cfg.workspace) or {}
    if request is not None and record.get("id") != request.get("id"):
        age = now - float(request.get("requested_at") or now)
        force = ", forced at its deadline" if request.get("force") else ""
        return {
            "phase": "requested",
            "id": request.get("id"),
            "detail": f"requested by {request.get('by')} {age:.0f}s ago{force}; not taken up yet",
        }
    if not record:
        return None
    return {**record, "detail": reload_mod.describe(record, now)}


def _reload_event(cfg: DriverConfig) -> Event | None:
    """A reload outcome ``wait`` has not returned yet: it completed, failed, or is overdue.

    Each is returned once, so the next ``wait`` moves on.
    """
    record = reload_mod.read_record(cfg.workspace)
    if not record:
        return None
    phase = record.get("phase")
    if phase == reload_mod.PHASE_RELOADED:
        kind = "RELOADED"
    elif phase == reload_mod.PHASE_FAILED:
        kind = "RELOAD_FAILED"
    elif phase == reload_mod.PHASE_DRAINING and record.get("overdue"):
        kind = "RELOAD_OVERDUE"
    else:
        return None
    key = f"{record.get('id')}:{kind}"
    if key in reload_mod.seen(cfg.workspace):
        return None
    reload_mod.mark_seen(cfg.workspace, key)
    detail = reload_mod.describe(record)
    if kind == "RELOAD_OVERDUE":
        detail += (
            f"; `{cfg.command} reload --force` re-executes without them, "
            f"`{cfg.command} reload --cancel` withdraws the request"
        )
    return Event(kind, detail)


# -------------------------------------------------------------------- waiting


@dataclass(frozen=True)
class Event:
    """Why :func:`wait` returned."""

    kind: str
    detail: str = ""
    request_id: str | None = None
    #: What the wait did on the way, each said on a line of its own after the event.
    notes: tuple[str, ...] = ()

    def render(self) -> str:
        head = self.kind if self.request_id is None else f"{self.kind} {self.request_id}"
        return "\n".join([f"{head}  {self.detail}".rstrip(), *(f"  note: {n}" for n in self.notes)])


def poll_once(
    cfg: DriverConfig,
    *,
    watch_sessions: bool = True,
    sessions_only: bool = False,
    now: float | None = None,
    notes: list[str] | None = None,
) -> Event | None:
    """The first thing that needs the driver right now, or ``None``.

    ``sessions_only`` is a run whose turns run in process: there is no turn
    to hand out, so it never returns ``NEW``. ``notes`` collects what the
    poll did on the way (a stale successor request archived); the event
    returned carries them.
    """
    notes = [] if notes is None else notes
    event = _first_event(
        cfg, watch_sessions=watch_sessions, sessions_only=sessions_only, now=now, notes=notes
    )
    if event is not None and notes:
        event = replace(event, notes=(*event.notes, *notes))
    return event


def _first_event(
    cfg: DriverConfig, *, watch_sessions: bool, sessions_only: bool, now: float | None, notes: list
) -> Event | None:
    state, detail = orchestrator_state(cfg, now)
    if state == "exited":
        if sessions_only:
            return Event("ORCHESTRATOR_EXITED", detail)
        unanswered = sum(1 for r in external_driver.requests(cfg) if not r["answered"])
        return Event("ORCHESTRATOR_EXITED", f"{detail}; {unanswered} turn(s) unanswered")
    reload_event = _reload_event(cfg)
    if reload_event is not None:
        return reload_event
    if not sessions_only:
        first = external_driver.next_turn(cfg)
        if first is not None:
            return Event("NEW", f"{first.get('role')}: {first.get('what')}", first["id"])
    if watch_sessions and cfg.sessions:
        waiting, archived = sessions_mod.review_requests(
            cfg.task, archive=True, by=f"driver {socket.gethostname()}:{os.getpid()}", now=now
        )
        notes.extend(archived)
        for request in waiting:
            return Event(
                "SESSION_REQUEST",
                f"session {request['session']} (allocation {request.get('slurm_job')}) ends "
                f"{request.get('ends_at')}; run `trtllm-modeling-v2 session allocate --task "
                f"{cfg.workspace / 'task.yaml'} --name {request['session']}`",
            )
        alerts = watchdog.check([s.spool for s in cfg.sessions])
        if alerts:
            return Event(f"SESSION_{alerts[0].kind}", f"{alerts[0].spool} {alerts[0].detail}")
    return None


def wait(
    cfg: DriverConfig,
    *,
    timeout_s: float = 540.0,
    poll_s: float = 10.0,
    watch_sessions: bool = True,
    sessions_only: bool = False,
) -> Event:
    """Block until :func:`poll_once` has something, or ``timeout_s`` passes.

    The default timeout keeps one call inside a tool-call limit of ten minutes.
    """
    end = time.monotonic() + timeout_s
    notes: list[str] = []
    while True:
        event = poll_once(
            cfg, watch_sessions=watch_sessions, sessions_only=sessions_only, notes=notes
        )
        if event is not None:
            return event
        remaining = end - time.monotonic()
        if remaining <= 0:
            if sessions_only:
                return Event("TIMEOUT", "nothing new", notes=tuple(notes))
            busy = sum(
                1 for r in external_driver.in_flight(cfg) if not external_driver.planner_turn(r)
            )
            capped = (
                f" (at the in-flight cap, {busy}/{cfg.max_inflight})"
                if external_driver.pending(cfg)
                else ""
            )
            return Event("TIMEOUT", f"nothing new{capped}", notes=tuple(notes))
        time.sleep(min(poll_s, remaining))


# --------------------------------------------------------------------- status


def status(cfg: DriverConfig, now: float | None = None) -> dict:
    """Every turn, the orchestrator and the sessions, as data."""
    now = time.time() if now is None else now
    state, detail = orchestrator_state(cfg, now)
    rows = external_driver.turn_rows(cfg, now)
    ledger = reload_mod.read_inflight(cfg.workspace) or {}
    holds = ledger.get("held") if state != "exited" and not ledger.get("stopped") else []
    return {
        "workspace": str(cfg.workspace),
        "orchestrator": {"state": state, "detail": detail},
        "holds": [
            {"kind": h.get("kind"), "what": h.get("what"), "age_s": round(now - h["since"], 1)}
            for h in holds or ()
            if isinstance(h, dict) and "since" in h
        ],
        "reload": reload_state(cfg, now),
        "correctness": _correctness(cfg.workspace),
        "in_flight": sum(
            1 for r in rows if r["claimed_by"] and not r["answered"] and not r["planner"]
        ),
        "planner_in_flight": sum(
            1 for r in rows if r["claimed_by"] and not r["answered"] and r["planner"]
        ),
        "max_inflight": cfg.max_inflight,
        "requests": rows,
        "sessions": sessions_mod.session_status(cfg.task, now) if cfg.sessions else [],
    }


def _correctness(workspace: Path) -> str | None:
    """Where the run last delivered correctness, or ``None`` before it has."""
    from ...run import contract as contract_mod  # imported here: run imports infra

    milestone = contract_mod.latest_milestone(workspace)
    return milestone.describe() if milestone else None


def format_status(data: Mapping[str, Any], *, turns: bool = True) -> str:
    """:func:`status` as lines; without ``turns``, the dispatched turns are left out.

    In-process turns are what the orchestrator holds (``turn <role> <what>``).
    """
    lines = [
        f"orchestrator: {data['orchestrator']['state']} ({data['orchestrator']['detail']})",
    ]
    holds = data.get("holds") or []
    if holds:
        lines.append(
            "orchestrator holds: "
            + ", ".join(f"{h['kind']} {h['what']} {h['age_s'] / 60:.0f}m" for h in holds)
        )
    reload = data.get("reload")
    if reload:
        lines.append(f"reload: {reload.get('phase')} ({reload.get('detail')})")
    if data.get("correctness"):
        lines.append(f"correctness: {data['correctness']}")
    if turns:
        beside = data.get("planner_in_flight") or 0
        lines.append(
            f"in flight: {data['in_flight']}/{data['max_inflight']}"
            + (f", and {beside} planner turn(s) beside the cap" if beside else "")
        )
        if not data["requests"]:
            lines.append("(no requests)")
    for row in data["requests"] if turns else ():
        if row["answered"]:
            mark = "done"
        elif row["claimed_by"]:
            mark = f"CLAIMED {row['age_s'] / 60:.0f}m"
        else:
            mark = f"WAITING {row['age_s'] / 60:.0f}m"
        if row["overdue"]:
            mark += " OVERDUE"
        if row.get("adopted"):
            mark += " ADOPTED"
        lines.append(f"{row['id']:<20} {mark:<22} {row['role'] or '-':<14} {row['what'] or ''}")
    for session in data["sessions"]:
        beat = (
            "never" if session["heartbeat_age_s"] is None else f"{session['heartbeat_age_s']:.0f}s"
        )
        lines.append(
            f"session {session['name']:<8} {session['role']:<12} heartbeat {beat} "
            f"running={session['running']} pending={session['pending']}"
        )
    return "\n".join(lines)


__all__ = [
    "ORCHESTRATOR_STALE_S",
    "WORKSPACE_LOCK",
    "DriverConfig",
    "DriverError",
    "Event",
    "format_status",
    "orchestrator_state",
    "poll_once",
    "reload_state",
    "request_reload",
    "status",
    "wait",
]
