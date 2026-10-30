# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a reload waits for, and what a stop cost.

A reload (``infra/driver/reload.py``) replaces the orchestrator once nothing it
would cut short is in flight. Three pieces serve it here:

- :class:`InFlight` is the door every integration, contract check and (on the
  in-process transport) agent turn goes through, and the ledger of those under
  way. A drain closes the door: what is through goes on to its end, what
  arrives waits. The ledger is on disk in ``inflight.json`` while it changes,
  with the process that keeps it and how that process stopped.
- :class:`RequestWatch` wakes the loop when a reload request appears, changes
  or is withdrawn, so a loop waiting on its futures needs no polling.
- :func:`restart_record` reads what the previous process left in its ledger
  as the cost of its stop: the checks, integrations and turns it cut short,
  and the tasks set back to pending. A reload that drained costs nothing; a
  stop by hand costs whatever it held.
"""

from __future__ import annotations

import itertools
import json
import os
import re
import socket
import threading
import time
from concurrent.futures import Future
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Collection, Iterator, Mapping, Sequence

from ..infra.driver.reload import INFLIGHT_FILENAME, REQUEST_FILENAME, RESTARTS_FILENAME
from ..infra.substrate.scheduler import SpoolPaths, read_json, write_json_atomic
from ..infra.transport import CONSUMED_SUFFIX, DISPATCH_DIRNAME, READ_SUFFIX

#: When this process started; the CLI imports this module first thing. A reload
#: requested before it is answered by the start itself.
PROCESS_STARTED_AT = time.time()

#: How a previous process stopped, as :func:`restart_record` reads its ledger.
AFTER_FIRST = "first"
AFTER_RELOAD = "reload"
AFTER_CLOSE = "closed"
AFTER_HARD_STOP = "hard-stop"
#: The previous process kept no ledger (a flow from before it).
AFTER_UNRECORDED = "unrecorded"

#: A spool job the orchestrator's own checks queue: ``gate-<epoch ms>-<pid>-<rand>-<n>``
#: (``infra/execution/spool.py``); a standalone gate's names carry the gate's name instead.
_CHECK_JOB = re.compile(r"^gate-(\d{13})-(\d+)-[0-9a-f]+-\d+$")


def _process(started_at: float) -> dict:
    return {"pid": os.getpid(), "host": socket.gethostname(), "started_at": started_at}


class InFlight:
    """The door a drain closes, and the ledger of what went through it.

    :meth:`hold` is entered around work a re-exec would cut short. While the
    door is closed (:meth:`close`), entering waits until it opens again, and in
    a drain it never does: the process is replaced first. A thread that holds
    something already goes through, so work that nests (a turn inside a check)
    never waits for itself.

    Reading the previous ledger and starting this process's own happen at
    construction: :attr:`previous` is what the last process left. Writing the
    ledger is bookkeeping; a full filesystem does not stop the work it records.
    """

    def __init__(self, workspace: str | Path, *, started_at: float = PROCESS_STARTED_AT) -> None:
        self.path = Path(workspace) / INFLIGHT_FILENAME
        #: The previous process's ledger: who it was, what it held, how it stopped.
        self.previous: dict | None = read_json(self.path)
        self._process = _process(started_at)
        self._cond = threading.Condition()
        self._holds: dict[int, dict] = {}
        self._ids = itertools.count(1)
        #: Per thread, how many holds it is inside.
        self._depth: dict[int, int] = {}
        self._closed = False
        self._stopped: dict | None = None
        self._changed: Future | None = None
        self._save()

    def _save(self) -> None:
        held = sorted(self._holds.values(), key=lambda h: h["since"])
        try:
            write_json_atomic(self.path, {**self._process, "held": held, "stopped": self._stopped})
        except OSError:
            pass

    @contextmanager
    def hold(self, kind: str, what: str, *, door: bool = True) -> Iterator[None]:
        """Do the work inside as ``<kind> <what>`` in flight; wait first while the door is closed.

        ``door=False`` enters even then: work that something the drain waits
        for may itself be waiting on (an agent turn the planner's thread
        runs), which a drain must wait for rather than hold back.
        """
        me = threading.get_ident()
        with self._cond:
            if door and not self._depth.get(me):
                while self._closed:
                    self._cond.wait()
            token = next(self._ids)
            self._holds[token] = {"kind": kind, "what": what, "since": time.time()}
            self._depth[me] = self._depth.get(me, 0) + 1
            self._save()
            self._signal()
        try:
            yield
        finally:
            with self._cond:
                del self._holds[token]
                self._depth[me] -= 1
                if not self._depth[me]:
                    del self._depth[me]
                self._save()
                self._signal()

    def _signal(self) -> None:
        """Tell whoever waits on :meth:`changed`. Call with the condition held."""
        if self._changed is not None and not self._changed.done():
            self._changed.set_result(None)

    def changed(self) -> Future:
        """A future that completes the next time something enters or leaves.

        Take it before reading :meth:`held`: a change after the read then
        still wakes the waiter.
        """
        with self._cond:
            if self._changed is None or self._changed.done():
                self._changed = Future()
            return self._changed

    def close(self) -> None:
        """Let nothing new in; what is through goes on."""
        with self._cond:
            self._closed = True

    def open(self) -> None:
        """Let in what waits at the door, and anything after it."""
        with self._cond:
            self._closed = False
            self._cond.notify_all()

    @property
    def closed(self) -> bool:
        return self._closed

    def held(self) -> list[dict]:
        """What is in flight now, oldest first: ``{"kind", "what", "since"}``."""
        with self._cond:
            return sorted((dict(h) for h in self._holds.values()), key=lambda h: h["since"])

    def stop(self, how: str | None) -> None:
        """Record how this process stops (``reload``, ``closed:<verdict>``), or that it goes on."""
        with self._cond:
            self._stopped = {"how": how, "at": time.time()} if how else None
            self._save()


class RequestWatch:
    """Wakes the loop when the reload request appears, changes or goes.

    A thread of its own looks at the request file every ``poll_s``; each
    change completes the future :meth:`signal` handed out, which the loop
    waits on beside its own.
    """

    def __init__(self, workspace: str | Path, *, poll_s: float = 5.0) -> None:
        self.path = Path(workspace) / REQUEST_FILENAME
        self.poll_s = max(poll_s, 0.01)
        self._lock = threading.Lock()
        self._signal: Future = Future()
        self._stop = threading.Event()

    def _state(self) -> tuple[int, int, int] | None:
        try:
            stat = self.path.stat()
        except OSError:
            return None
        return stat.st_mtime_ns, stat.st_size, stat.st_ino

    def start(self) -> None:
        last = self._state()

        def look() -> None:
            nonlocal last
            while not self._stop.wait(self.poll_s):
                state = self._state()
                if state != last:
                    last = state
                    with self._lock:
                        if not self._signal.done():
                            self._signal.set_result(state)

        threading.Thread(target=look, name="mv2-reload-watch", daemon=True).start()

    def signal(self) -> Future:
        """The future the next change completes. Take it before reading the request."""
        with self._lock:
            if self._signal.done():
                self._signal = Future()
            return self._signal

    def stop(self) -> None:
        self._stop.set()


# ------------------------------------------------------------- the cost of a stop


def restart_record(
    previous: Mapping[str, Any] | None,
    *,
    reset: Sequence[str] = (),
    answered_request: Mapping[str, Any] | None = None,
    started_at: float = PROCESS_STARTED_AT,
) -> dict:
    """What the stop before this start cost, from the previous process's ledger.

    ``reset`` are the tasks the graph had running, now pending again;
    ``answered_request`` a reload request older than this process, which its
    start answered.
    """
    record: dict[str, Any] = {**_process(started_at), "tasks_reset": list(reset)}
    if previous is None:
        record["after"] = AFTER_UNRECORDED if reset else AFTER_FIRST
    else:
        how = str((previous.get("stopped") or {}).get("how") or "")
        if how == "reload":
            record["after"] = AFTER_RELOAD
        elif how.startswith("closed"):
            record["after"] = AFTER_CLOSE
        else:
            record["after"] = AFTER_HARD_STOP
        record["previous"] = {
            "pid": previous.get("pid"),
            "host": previous.get("host"),
            "started_at": previous.get("started_at"),
            "stopped": previous.get("stopped"),
        }
        record["cut"] = [
            {"kind": h.get("kind"), "what": h.get("what"), "since": h.get("since")}
            for h in previous.get("held") or ()
        ]
    if answered_request:
        record["answered_request"] = dict(answered_request)
    return record


def turns_left(
    workspace: str | Path, *, since: float, before: float, tasks: Collection[str] = ()
) -> list[dict]:
    """The turns the previous process asked between ``since`` and ``before`` that the run still owes.

    Each is ``{"id", "role", "what", "answered", "read"}``: one not answered is
    still with the driver, and the next process waits on it; one answered
    and never read is adopted by its task or, for a planner, handed to the
    next planner turn; one read and not consumed whose work is one of
    ``tasks`` (those with no outcome yet) is adopted by that task.
    """
    dispatch = Path(workspace) / DISPATCH_DIRNAME
    out = []
    for path in sorted(dispatch.glob("*.request.json")):
        data = read_json(path)
        if not data:
            continue
        request_id = path.name[: -len(".request.json")]
        asked = float(data.get("requested_at") or 0)
        if not since <= asked < before or (dispatch / f"{request_id}{CONSUMED_SUFFIX}").exists():
            continue
        answered = (dispatch / f"{request_id}.done").exists()
        read = (dispatch / f"{request_id}{READ_SUFFIX}").exists()
        if answered and read and data.get("what") not in tasks:
            continue
        out.append(
            {
                "id": request_id,
                "role": data.get("role"),
                "what": data.get("what"),
                "answered": answered,
                "read": read,
            }
        )
    return out


def jobs_left(
    spools: Collection[str | Path], *, pid: Any, since: float, before: float
) -> list[str]:
    """The check jobs process ``pid`` queued between ``since`` and ``before`` that still wait or run.

    Nothing reads their receipts any more: the checks they were for run again,
    beside them, unless they are cancelled.
    """
    left = []
    for spool in spools:
        paths = SpoolPaths(spool)
        for directory in (paths.pending, paths.running):
            for script in sorted(directory.glob("*.sh")):
                name = script.name[: -len(".sh")]
                match = _CHECK_JOB.match(name)
                if (
                    match
                    and match.group(2) == str(pid)
                    and since <= int(match.group(1)) / 1000 < before
                ):
                    left.append(f"{name} ({directory.name} in {spool})")
    return left


def _clock(epoch: Any) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(epoch)))
    except (TypeError, ValueError):
        return "?"


def describe_restart(record: Mapping[str, Any]) -> str:
    """The journal's line for a start: how the last process stopped and what that cost."""
    after = record.get("after")
    previous = record.get("previous") or {}
    who = f"pid {previous.get('pid')} on {previous.get('host')}"
    parts: list[str] = []
    if after == AFTER_RELOAD:
        parts.append(f"{who} reloaded into this process")
    elif after == AFTER_CLOSE:
        parts.append(f"{who} closed the run ({(previous.get('stopped') or {}).get('how')})")
    elif after == AFTER_HARD_STOP:
        parts.append(
            f"{who}, started {_clock(previous.get('started_at'))}, stopped without a reload"
        )
    elif after == AFTER_UNRECORDED:
        parts.append("the previous process kept no ledger, so what it cut short is unknown")
    cut = record.get("cut") or []
    if cut:
        parts.append("cut short: " + ", ".join(f"{c['kind']} {c['what']}" for c in cut))
    elif after in (AFTER_RELOAD, AFTER_HARD_STOP):
        parts.append("nothing was cut short")
    jobs = record.get("jobs") or []
    if jobs:
        parts.append(
            "check jobs it left queued or running, whose checks run again: " + ", ".join(jobs)
        )
    reset = record.get("tasks_reset") or []
    if reset:
        parts.append("tasks set back to pending: " + ", ".join(reset))
    turns = record.get("turns") or []
    asked = [t for t in turns if not t["answered"]]
    unread = [t for t in turns if t["answered"] and not t["read"]]
    adopted = [t for t in turns if t["answered"] and t["read"]]
    if asked:
        parts.append(
            "turns still with the driver (waited on): "
            + ", ".join(f"{t['id']} {t['what']}" for t in asked)
        )
    if unread:
        parts.append(
            "answers no process read: " + ", ".join(f"{t['id']} {t['what']}" for t in unread)
        )
    if adopted:
        parts.append(
            "answered turns of tasks with no outcome (adopted): "
            + ", ".join(f"{t['id']} {t['what']}" for t in adopted)
        )
    answered = record.get("answered_request")
    if answered:
        parts.append(
            f"a reload requested at {_clock(answered.get('requested_at'))} is answered by this start"
        )
    return "; ".join(parts)


def append_restart(workspace: str | Path, record: Mapping[str, Any]) -> None:
    """Add ``record`` to ``restarts.jsonl``, one line per start."""
    path = Path(workspace) / RESTARTS_FILENAME
    try:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({**record, "at": time.time()}, sort_keys=True) + "\n")
    except OSError:
        pass


__all__ = [
    "AFTER_CLOSE",
    "AFTER_FIRST",
    "AFTER_HARD_STOP",
    "AFTER_RELOAD",
    "AFTER_UNRECORDED",
    "PROCESS_STARTED_AT",
    "InFlight",
    "RequestWatch",
    "append_restart",
    "describe_restart",
    "jobs_left",
    "restart_record",
    "turns_left",
]
