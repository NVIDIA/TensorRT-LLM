# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reloading a running orchestrator: the request a driver writes, the record the orchestrator keeps.

A reload is the orchestrator replacing itself once nothing it would cut short
is in flight::

    driver                                 orchestrator (its planned loop)
    ------                                 -------------------------------
    write reload.request.json    ----->    drain: no new task, integration, check or planner turn
                                           starts; what is in flight runs to its end
                                           reload.json  phase=draining, waiting_for=[...]
                                           past the deadline: overdue, and it keeps waiting
                                           (with ``force`` it goes on without them)
                                           checkpoint; reload.json phase=reexec; the request goes
                                           re-executes itself: same pid, same log, same arguments
                                           reload.json phase=reloaded once its loop runs again
    wait -> RELOADED             <-----

Worker turns are not waited for: on the external transport a relaunched
process adopts every turn its predecessor asked (``infra/external/transport.py``).

Beside them the orchestrator keeps ``inflight.json``: what it holds right now
that a stop would cut short, and how it last stopped. The next process reads it
as the cost of the stop (``run/drain.py``).

Every file is written whole under a temporary name and renamed.
"""

from __future__ import annotations

import os
import socket
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

from ..substrate.scheduler import read_json, write_json_atomic, write_text_atomic

#: Written by the driver, read by the orchestrator, removed by either.
REQUEST_FILENAME = "reload.request.json"
#: The latest reload as the orchestrator saw it through, written by it alone.
RECORD_FILENAME = "reload.json"
#: What the orchestrator holds that a stop would cut short, and how it last stopped.
INFLIGHT_FILENAME = "inflight.json"
#: One line per orchestrator start: what the stop before it cost.
RESTARTS_FILENAME = "restarts.jsonl"
#: Which reload outcomes ``driver wait`` already returned, one ``<id>:<phase>`` per line.
SEEN_FILENAME = "reload.seen"

#: How long a drain waits before it says what it is still waiting for.
DEFAULT_DRAIN_TIMEOUT_S = 1800.0

PHASE_DRAINING = "draining"
PHASE_REEXEC = "reexec"
PHASE_RELOADED = "reloaded"
PHASE_CANCELLED = "cancelled"
PHASE_FAILED = "failed"
#: The process stopped while it drained, before it could re-execute.
PHASE_ABANDONED = "abandoned"


def read_request(workspace: str | Path) -> dict | None:
    """The reload request waiting in ``workspace``, or ``None``."""
    return read_json(Path(workspace) / REQUEST_FILENAME)


def read_record(workspace: str | Path) -> dict | None:
    """The orchestrator's record of the latest reload, or ``None``."""
    return read_json(Path(workspace) / RECORD_FILENAME)


def write_record(workspace: str | Path, record: Mapping[str, Any]) -> None:
    write_json_atomic(Path(workspace) / RECORD_FILENAME, dict(record))


def read_inflight(workspace: str | Path) -> dict | None:
    """The orchestrator's ``inflight.json``, or ``None``."""
    return read_json(Path(workspace) / INFLIGHT_FILENAME)


def request(
    workspace: str | Path,
    *,
    force: bool = False,
    timeout_s: float = DEFAULT_DRAIN_TIMEOUT_S,
    by: str | None = None,
    now: float | None = None,
) -> dict:
    """Ask the orchestrator on ``workspace`` to reload; returns the request.

    A request already waiting keeps its id and the time it was made, so a
    drain under way goes on; ``force`` and ``timeout_s`` are this call's.
    """
    now = time.time() if now is None else now
    existing = read_request(workspace) or {}
    data = {
        "id": str(existing.get("id") or uuid.uuid4().hex[:8]),
        "requested_at": float(existing.get("requested_at") or now),
        "updated_at": now,
        "by": by or f"{socket.gethostname()}:{os.getpid()}",
        "force": bool(force),
        "timeout_s": max(0.0, float(timeout_s)),
    }
    write_json_atomic(Path(workspace) / REQUEST_FILENAME, data)
    return data


def cancel(workspace: str | Path) -> dict | None:
    """Withdraw the waiting request; returns it, or ``None`` when there was none.

    A drain under way ends and the orchestrator goes on as before; one that
    already re-executed cannot be withdrawn.
    """
    path = Path(workspace) / REQUEST_FILENAME
    data = read_json(path)
    path.unlink(missing_ok=True)
    return data


def remove_request(workspace: str | Path) -> None:
    """Remove the request once it is answered; one that cannot be removed is left."""
    try:
        (Path(workspace) / REQUEST_FILENAME).unlink(missing_ok=True)
    except OSError:
        pass


def _clock(epoch: Any) -> str:
    try:
        return time.strftime("%H:%M:%S", time.localtime(float(epoch)))
    except (TypeError, ValueError):
        return "?"


def _minutes(seconds: Any) -> str:
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return "?"
    return f"{value:.0f}s" if value < 120 else f"{value / 60:.0f}m"


def describe(record: Mapping[str, Any], now: float | None = None) -> str:
    """One line for a reload record, in whatever phase it is."""
    now = time.time() if now is None else now
    phase = record.get("phase")
    waited = ", ".join(
        f"{w.get('what')} {_minutes(w.get('s'))}" for w in record.get("waited") or ()
    )
    if phase == PHASE_DRAINING:
        waiting = ", ".join(record.get("waiting_for") or ()) or "nothing"
        since = float(record.get("since") or now)
        deadline = float(record.get("deadline") or now)
        when = (
            f"{_minutes(now - deadline)} past its deadline"
            if now >= deadline
            else f"deadline in {_minutes(deadline - now)}"
        )
        force = "; forced at the deadline" if record.get("force") else ""
        return f"draining for {_minutes(now - since)} ({when}{force}); waiting for {waiting}"
    if phase in (PHASE_REEXEC, PHASE_RELOADED):
        line = (
            f"pid {record.get('pid')} re-executed at {_clock(record.get('reexec_at'))} after "
            f"draining {_minutes(record.get('drained_s'))}"
            + (f" (waited for {waited})" if waited else " (nothing was in flight)")
        )
        if record.get("left"):
            line += "; forced, cutting short " + ", ".join(record["left"])
        if phase == PHASE_RELOADED:
            line += f"; its loop runs again since {_clock(record.get('reloaded_at'))}"
        return line
    if phase == PHASE_FAILED:
        return f"the re-exec failed ({record.get('error')}); the run goes on in pid {record.get('pid')}"
    if phase == PHASE_CANCELLED:
        return f"cancelled at {_clock(record.get('ended_at'))} after draining {_minutes(record.get('drained_s'))}"
    if phase == PHASE_ABANDONED:
        return f"pid {record.get('pid')} stopped while it drained, before it re-executed"
    return str(phase or "unknown")


def seen(workspace: str | Path) -> set[str]:
    try:
        text = (Path(workspace) / SEEN_FILENAME).read_text(encoding="utf-8")
    except OSError:
        return set()
    return {line.strip() for line in text.splitlines() if line.strip()}


def mark_seen(workspace: str | Path, key: str) -> None:
    """Remember that ``driver wait`` returned ``key``, keeping the latest few."""
    keys = [k for k in sorted(seen(workspace)) if k != key][-20:] + [key]
    write_text_atomic(Path(workspace) / SEEN_FILENAME, "\n".join(keys) + "\n")


__all__ = [
    "DEFAULT_DRAIN_TIMEOUT_S",
    "INFLIGHT_FILENAME",
    "PHASE_ABANDONED",
    "PHASE_CANCELLED",
    "PHASE_DRAINING",
    "PHASE_FAILED",
    "PHASE_REEXEC",
    "PHASE_RELOADED",
    "RECORD_FILENAME",
    "REQUEST_FILENAME",
    "RESTARTS_FILENAME",
    "SEEN_FILENAME",
    "cancel",
    "describe",
    "mark_seen",
    "read_inflight",
    "read_record",
    "read_request",
    "remove_request",
    "request",
    "seen",
    "write_record",
]
