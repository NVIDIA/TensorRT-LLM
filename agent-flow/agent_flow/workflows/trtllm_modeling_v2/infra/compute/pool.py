# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The pool's controller: when a holder is needed, one record per submission, the events, the end.

The spool and its watchers (:mod:`.scheduler`) run jobs. This module, in the flow's controller
and outside every allocation, keeps them supplied with *holders*: allocations (or this one
host) whose nodes run watchers. It asks its environment (:class:`Environment`) for a holder
of a shape, for what is listed of the holders it asked for, and for a holder's end: on Slurm
(:mod:`.slurm`) a batch job, a listing, a cancel; on one host one local watcher. The flow has
those two launchers and no other (:data:`OTHER_ENVIRONMENT`).

**Renewal**, each pass and per shape (a count of nodes) that work is ready or running for.
Work counts on a holder of its own width. One is submitted when none of that width runs or
is pending; ``lead`` before the last one stops fitting the longest job, though never in a
holder's first half; and when a ready job fits what is left of no holder and would fit a
fresh one. Holders overlap on the one spool; a running job is never moved.

**One fact is stored per submission**: ``<spool>/holders/<seq>.json``, written *before* the
environment is asked, under the name the holder will carry. Each pass the record is *listed*
(the environment shows a job of that name: its id is the authority from then on), *refused*
(closed), or *unknown* (no id came back and nothing shows the name yet). While a record is
unknown no other holder of its shape is asked for, until the environment shows the job or
somebody closes it after looking: a lost reply cannot cause a second allocation, and no
record is resolved by its age.

**A holder that cannot work** is given back: one that runs nothing while ready work fits it
and whose nodes hold memory no job of the run accounts for. The next is asked for without
those nodes.

**A holder nothing needs** is given back: one whose own width no ready or running work asks
for, once that has lasted as long as the holder took to get and a start. Narrower work is
served by a wider holder while it is there and does not keep it.

**Events** are derived each pass, written to ``<spool>/pool.json`` for ``monitor wait`` and
appended to ``<spool>/pool.log`` when new; nothing here acts on one. ``POOL_SUBMIT_FAILED``
(refused, not answered, or gone early twice in a row: ``pool-retry``), ``POOL_CAPACITY`` (a
wall, the bound: ``pool-adjust``), ``POOL_DEVICE_HELD`` (look at the node).

**The bound.** A run asks its scheduler for no more node-hours than
``environment.slurm.max_node_hours`` (:class:`Asked`, from the records). At the bound no
holder is asked for, and once none is left the controller stops.

**Ending**: a stop marker (which forbids any submission), then a cancel of every holder that
waits or runs no job. Only a job id that is in a record *and* listed under that record's
name is ever cancelled.
"""

from __future__ import annotations

import math
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from ..files import read_json, write_json_atomic
from .scheduler import (
    FIT_MARGIN_S,
    STOP_MARKER,
    WATCHER_STALE_S,
    SpoolPaths,
    job_spec_from_script,
    read_environ,
    utc_iso,
)

PENDING = "pending"
RUNNING = "running"
#: Still listed and on its way out: counted, serving nothing.
ENDING = "ending"
ENDED = "ended"
#: A record whose submission was not answered with an id and that nothing lists yet.
UNKNOWN = "unknown"
REFUSED = "refused"

#: A holder was asked for and did not come: the scheduler refused it, or did not answer.
POOL_SUBMIT_FAILED = "POOL_SUBMIT_FAILED"
#: Devices are held and serve nothing: memory no job accounts for, or a holder with no watcher.
POOL_DEVICE_HELD = "POOL_DEVICE_HELD"
#: The spec's numbers do not allow the holder the work needs: a wall, the bound.
POOL_CAPACITY = "POOL_CAPACITY"
#: The node-hours a run on Slurm may ask for when its spec states none
#: (``environment.slurm.max_node_hours``): two days of one node. Enough for the onboardings
#: on record, and what an unwatched run can cost at most.
DEFAULT_MAX_NODE_HOURS = 48.0

#: A refused submission is tried again this long after the refusal.
REFUSAL_RETRY_S = 120.0
#: Memory on a free device is said once it has been there this long (a job's own goes sooner).
DEVICE_HELD_REPORT_S = 120.0
#: How long ending waits for watchers and leads to see the stop marker and leave.
END_WAIT_S = 60.0
#: What a holder's start costs before its watchers serve: taken off what one holder can fit,
#: and a running holder that has had no watcher and no job for longer is said. Measured from
#: the job's start to its first watcher's heartbeat with the image already on the node: 51 s
#: to 67 s (four starts on two clusters). A first pull of a large image may take longer: the
#: event then says a start that is later than the fit allows, and goes when a watcher beats.
START_SLACK_S = 300.0
#: Seconds between two passes of the controller's keeper.
PASS_S = 15.0
#: How many holders a run gives back for memory that came with their nodes. Past it the
#: event stands and the session decides: a site where every node shows such memory would
#: otherwise be walked node by node.
GIVE_BACK_MOST = 3
#: Terminal states of a holder that are its own end, not a loss.
_OWN_END = ("COMPLETED", "TIMEOUT")

POOL_LOG = "pool.log"


class PoolError(RuntimeError):
    """The pool cannot start: one line that says what to state or to do."""


@dataclass(frozen=True)
class Reply:
    """An environment's answer to a submission."""

    #: ``accepted`` (with ``job_id``), ``refused`` (closed), or ``unknown`` (it may exist).
    outcome: str
    job_id: str = ""
    #: The environment's own words.
    text: str = ""


@dataclass(frozen=True)
class Row:
    """What the environment shows of one holder now."""

    state: str
    job_id: str = ""
    #: When it ends (running) and when it is expected to start (pending), epoch seconds.
    end: float | None = None
    start: float | None = None
    reason: str = ""
    #: The environment's own state text.
    detail: str = ""


class Environment(Protocol):
    """The slots the pool asks of where it runs."""

    kind: str
    #: The expected queue wait.
    lead_s: float

    def holder_name(self, seq: int) -> str: ...

    def submit(self, name: str, nodes: int, avoid: Sequence[str] = ()) -> Reply:
        """Get a holder of ``nodes`` nodes under ``name``, on none of the hosts ``avoid``."""

    def listing(self, records: Sequence[Mapping[str, Any]]) -> dict[str, list[Row]] | None:
        """What is shown of each record's name; ``None`` when the question failed."""

    def end(self, record: Mapping[str, Any], *, pending_only: bool = False) -> str:
        """End the holder ``record`` names; ``""`` or why not."""


@dataclass(frozen=True)
class Event:
    kind: str
    #: What the event is about; an event with the same kind and key is the same event.
    key: str
    detail: str

    def render(self) -> str:
        return f"{self.kind}  {self.detail}".rstrip()


@dataclass
class Holder:
    """One submission as it stands this pass: its record, what is listed, who heart-beats."""

    record: dict
    state: str
    row: Row | None = None
    #: Live heartbeats of its watchers (not its lead).
    watchers: list[dict] = field(default_factory=list)
    #: Whether a job in the spool's ``running/`` was started by one of its watchers.
    busy: bool = False

    @property
    def name(self) -> str:
        return str(self.record["name"])

    @property
    def seq(self) -> int:
        return int(self.record["seq"])

    @property
    def nodes(self) -> int:
        return int(self.record["nodes"])

    @property
    def job_id(self) -> str:
        return str(self.record.get("job_id") or "")

    @property
    def open(self) -> bool:
        """Whether the pool still has to list, keep or end it: everything not closed."""
        return self.state not in (ENDED, REFUSED)

    @property
    def end(self) -> float | None:
        """When it ends: what is listed, else what its watchers say; ``None`` for no end."""
        if self.row is not None and self.row.end:
            return self.row.end
        # The earliest: work across its nodes ends with the first of them.
        ends = [float(w["ends_at_epoch"]) for w in self.watchers if w.get("ends_at_epoch")]
        return min(ends) if ends else None

    @property
    def working(self) -> bool:
        """Whether a watcher of it runs a job.

        Read twice: from its live heartbeats, and from the spool's ``running/`` records, so a
        holder whose heartbeats are late is not taken for one that runs nothing.
        """
        return self.busy or any(w.get("running") for w in self.watchers)

    @property
    def idle(self) -> bool:
        """Whether it runs no job and no node of it is reserved for one about to start."""
        return not self.working and not any(w.get("reserved_for") for w in self.watchers)

    @property
    def asked_to_end(self) -> bool:
        return bool(self.record.get("stop_requested"))

    @property
    def leaving(self) -> bool:
        """Whether it ran and is no longer shown running: its place is about to come free.

        Cancelled or lost and still listed on its way out, or in no listing while accounting
        has not named its end. It serves nothing and no successor is on its way under its name.
        """
        return self.open and bool(self.record.get("leaving_since"))


@dataclass(frozen=True)
class Asked:
    """The node-hours a run has asked its scheduler for, against its bound.

    A holder that is not over counts what was asked for it, from the moment its record is
    written: its nodes times its time limit. One that is over counts what it held, by the
    scheduler's own start and end (nothing, when it ended without having started); where
    those are not known, what was asked. A refused one counts nothing. So the sum is never
    under what the run can still hold, and a holder that failed in a minute does not cost
    its whole limit. A record is over once a pass of the pool has seen its job ended: one
    cancelled by a command that then left reads as asked until the next pass or command.
    """

    holders: float = 0.0
    holder_jobs: int = 0
    #: ``None``: this environment states no bound.
    bound: float | None = None

    def over(self, more: float) -> bool:
        """Whether ``more`` node-hours would take the run past its bound."""
        return self.bound is not None and self.holders + more > self.bound + 1e-9

    def render(self) -> str:
        of = f" of {self.bound:g}" if self.bound is not None else ""
        return f"node-hours asked: {self.holders:.1f}{of} ({self.holder_jobs} holder job(s))"


def _asked_records(directory: Path) -> list[dict]:
    """The submission records of ``directory`` that were not refused."""
    found = []
    for path in directory.glob("*.json"):
        record = read_json(path)
        if not record or not str(record.get("seq", "")).isdigit():
            continue
        if (record.get("closed") or {}).get("state") != REFUSED:
            found.append(record)
    return found


def _counted_s(record: Mapping[str, Any], asked_s: float) -> float:
    """The seconds one record counts: what was asked while it is open, what it held once over."""
    closed = record.get("closed")
    if not closed:
        return asked_s
    start, end = closed.get("start"), closed.get("end")
    if end and not start and not record.get("running_since"):
        return 0.0  # the scheduler shows its end and no start: it never ran
    if not (start and end) and record.get("running_since") and closed.get("at"):
        # No scheduler's times (accounting names none): what the pool itself saw of it.
        start, end = record["running_since"], closed["at"]
    if start and end:
        return min(asked_s, max(0.0, float(end) - float(start)))
    return asked_s


def asked(spool: str | Path, *, bound: float | None = None) -> Asked:
    """Derive :class:`Asked` from the records the pool already keeps."""
    holders = _asked_records(SpoolPaths(spool).holders)
    return Asked(
        holders=sum(
            int(r.get("nodes") or 1) * _counted_s(r, float(r.get("time_s") or 0)) / 3600
            for r in holders
        ),
        holder_jobs=len(holders),
        bound=bound,
    )


def asked_of(task_data: Mapping[str, Any]) -> Asked | None:
    """What the run of this spec has asked for; ``None`` when it names no spool.

    For a reader that keeps no pool (``pool status``, the closing report): the bound as the
    spec states it, on Slurm; no bound anywhere else.
    """
    block = task_data.get("environment") or {}
    if not block.get("spool"):
        return None
    bound = None
    if environment_kind(block) == KIND_SLURM:
        bound = float((block.get("slurm") or {}).get("max_node_hours") or DEFAULT_MAX_NODE_HOURS)
    return asked(block["spool"], bound=bound)


def span(seconds: float | None) -> str:
    """``45s``, ``7m``, ``2h05m``: a duration as a person reads it."""
    if seconds is None:
        return "no end"
    seconds = max(0.0, seconds)
    if seconds < 90:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    if minutes < 60:
        return f"{minutes}m"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m" if minutes else f"{hours}h"


def spool_demand(spool: str | Path, base: Mapping[int, float] | None = None) -> dict[int, float]:
    """The shapes work remains for and the longest wall of each: ``base`` plus the spool's jobs.

    ``base`` is what the run still owes (the caller's: its tasks and their longest gate); every
    job pending or running in the spool adds its own shape and wall. A shape is a count of
    nodes; a job that states devices needs one node.
    """
    demand = {int(nodes): float(wall) for nodes, wall in (base or {}).items()}
    paths = SpoolPaths(spool)
    for where in (paths.pending, paths.running):
        for _, nodes, wall in gpu_jobs(where):
            demand[nodes] = max(demand.get(nodes, 0.0), wall)
    return demand


def gpu_jobs(directory: Path) -> list[tuple[str, int, float]]:
    """The GPU jobs whose scripts are in ``directory``: name, nodes, wall in seconds."""
    jobs = []
    for script in sorted(directory.glob("*.sh")):
        try:
            spec = job_spec_from_script(script.stem, script.read_text(encoding="utf-8"))
        except OSError:
            continue
        if spec.errors or not spec.holds_gpus:
            continue
        jobs.append((script.stem, spec.nodes if spec.gang else 1, float(spec.wall_s or 0.0)))
    return jobs


def device_events(watchers: Sequence[Mapping[str, Any]], now: float) -> list[Event]:
    """``POOL_DEVICE_HELD``: a device a process holds that its watcher did not start, or may not stop."""
    events = []
    for watcher in watchers:
        host = watcher.get("host")
        for device, held in sorted((watcher.get("held") or {}).items()):
            if now - float(held.get("since", now)) < DEVICE_HELD_REPORT_S:
                continue
            pids = ", ".join(str(p) for p in held.get("pids") or ()) or "unknown"
            events.append(
                Event(
                    POOL_DEVICE_HELD,
                    f"{host}:{device}",
                    f"device {device} on {host} holds {held.get('used_mib')} MiB that no job of "
                    f"its watcher accounts for (pids {pids}); it is not handed out, and nothing "
                    f"is killed. A holder that ready work waits on for it is given back and "
                    f"asked for again without the node, {GIVE_BACK_MOST} times in a run",
                )
            )
        stuck = [j for j in watcher.get("running") or () if j.get("unsafe_pids")]
        for job in stuck:
            events.append(
                Event(
                    POOL_DEVICE_HELD,
                    f"{host}:{job.get('name')}",
                    f"devices {','.join(job.get('devices') or ()) or '-'} on {host} are held for "
                    f"processes of {job.get('name')} that carry the allocation's MPI context "
                    f"(pids {', '.join(str(p) for p in job['unsafe_pids'])}); the safe-kill rule "
                    f"forbids signalling them",
                )
            )
    return events


def standing_events(spool: str | Path, now: float | None = None) -> list[Event]:
    """What stands on ``spool``, for a reader that is not the controller: its last pass, while fresh."""
    from .scheduler import POOL_FRESH_S

    now = time.time() if now is None else now
    snapshot = read_json(SpoolPaths(spool).pool) or {}
    try:
        fresh = now - float(snapshot.get("at", 0)) <= POOL_FRESH_S
    except (TypeError, ValueError):
        fresh = False
    return [
        Event(str(e.get("kind")), str(e.get("key")), str(e.get("detail")))
        for e in (snapshot.get("events") or ()) * fresh
    ]


class Pool:
    """The controller's side of one spool: records, one pass, the end."""

    def __init__(
        self,
        spool: str | Path,
        environment: Environment,
        *,
        clock: Callable[[], float] = time.time,
        note: Callable[[str], None] | None = None,
    ) -> None:
        self.paths = SpoolPaths(spool)
        self.env = environment
        self.clock = clock
        self.note = note
        #: The bound is reached and no holder is left: nothing of the run can run any more.
        self.exhausted = ""
        #: On one host: no job that is ready fits what is left of it, and none runs.
        self.late = ""
        self.paths.ensure()
        self.paths.holders.mkdir(parents=True, exist_ok=True)

    def start(self, demand: Mapping[int, float]) -> list[Event]:
        """The first pass of a run. A refusal of its very first submission stops the start.

        A stop marker an earlier run left is removed: this run has work.
        """
        self.paths.stop.unlink(missing_ok=True)
        # A start is somebody's new attempt: a refusal an earlier one met is not waited out.
        events = self.reconcile(demand, retry_now=True)
        refused = [e for e in events if e.kind == POOL_SUBMIT_FAILED and e.key.startswith("nodes=")]
        if refused and not (read_json(self.paths.pool) or {}).get("holders"):
            # The start stops here: nothing is tried again behind the caller's back.
            raise PoolError(refused[0].detail.split("; tried again")[0])
        return events

    # ---------------------------------------------------------------- records

    def records(self) -> list[dict]:
        found = []
        for path in self.paths.holders.glob("*.json"):
            record = read_json(path)
            if record and str(record.get("seq", "")).isdigit():
                found.append(record)
        return sorted(found, key=lambda r: int(r["seq"]))

    def _write(self, record: Mapping[str, Any]) -> None:
        write_json_atomic(self.paths.holders / f"{int(record['seq'])}.json", dict(record))

    def _stop_marker(self, name: str) -> Path:
        return self.paths.holders / f"{name}.{STOP_MARKER}"

    def close(self, seq: int, why: str) -> dict:
        """Close record ``seq`` by hand, after a look: its shape may be asked for again.

        The act of ``resolve pool-retry pool --close-unknown <seq>``. For a record no
        environment can resolve (a reply that was lost and a job that never
        came to be). Refused for a record whose job is listed: that one ends through ``end``.
        """
        record = next((r for r in self.records() if int(r["seq"]) == int(seq)), None)
        if record is None:
            raise PoolError(f"no submission record {seq} in {self.paths.holders}")
        if record.get("closed"):
            return record
        listed = self.env.listing([record])
        rows = (listed or {}).get(str(record["name"])) or []
        if any(row.state != ENDED for row in rows):
            raise PoolError(
                f"record {seq} ({record['name']}) is listed as {rows[0].detail or rows[0].state} "
                f"(job {rows[0].job_id}): it is not unknown, and is ended through `pool stop`"
            )
        record["closed"] = {"state": "closed by hand", "at": self.clock(), "why": why}
        self._write(record)
        return record

    # ---------------------------------------------------------------- reading

    def _watchers(self, now: float) -> list[dict]:
        return self.paths.live_watchers(WATCHER_STALE_S, now)

    def _busy(self) -> tuple[set[str], set[str]]:
        """The holder names and job ids of the watchers that started a job still in ``running/``."""
        names: set[str] = set()
        jobs: set[str] = set()
        for meta_path in self.paths.running.glob("*.json"):
            meta = read_json(meta_path) or {}
            jobs.add(str(meta.get("slurm_job") or ""))
            watcher = read_json(self.paths.watchers / f"{meta.get('watcher')}.json") or {}
            names.add(str(watcher.get("holder") or ""))
        return names - {""}, jobs - {"", "unknown"}

    def observe(self, now: float | None = None) -> list[Holder] | None:
        """Every record as it stands now; ``None`` when the environment could not be asked.

        What the environment lists is written into the record only as an observation: the id
        once known, when it was first seen running, when it ends, and how it closed.
        """
        now = self.clock() if now is None else now
        records = self.records()
        unresolved = [r for r in records if not r.get("closed")]
        listed = self.env.listing(unresolved) if unresolved else {}
        if listed is None:
            return None
        beats: dict[str, list[dict]] = {}
        for watcher in self._watchers(now):
            if not watcher.get("lead") and watcher.get("holder"):
                beats.setdefault(str(watcher["holder"]), []).append(watcher)
        busy_names, busy_jobs = self._busy()
        holders = []
        for record in records:
            name = str(record["name"])
            closed = record.get("closed")
            if closed:
                state = REFUSED if closed.get("state") == REFUSED else ENDED
                holders.append(Holder(record, state))
                continue
            rows = listed.get(name) or []
            row = self._row_for(record, rows)
            if row is None:
                # Nothing shows it: with an id it is still the environment's to end; without
                # one nobody knows whether it exists. Either way it keeps its slot.
                if record.get("running_since") and not record.get("leaving_since"):
                    # It ran, and accounting has not named its end yet: it is on its way out.
                    record["leaving_since"] = now
                    self._write(record)
                busy = name in busy_names or str(record.get("job_id") or "-") in busy_jobs
                holders.append(Holder(record, UNKNOWN, watchers=beats.get(name, []), busy=busy))
                continue
            changed = self._note_row(record, row, rows, now)
            busy = name in busy_names or str(record.get("job_id") or "-") in busy_jobs
            # Since when it runs with no watcher and no job: an observation like the others.
            alone = row.state == RUNNING and not beats.get(name) and not busy
            if alone and "unwatched_since" not in record:
                record["unwatched_since"], changed = now, True
            elif not alone and record.pop("unwatched_since", None) is not None:
                changed = True
            if changed:
                self._write(record)
            state = ENDED if record.get("closed") else row.state
            holders.append(Holder(record, state, row=row, watchers=beats.get(name, []), busy=busy))
        return holders

    @staticmethod
    def _row_for(record: Mapping[str, Any], rows: Sequence[Row]) -> Row | None:
        """The row that is this record's job: its recorded id, else the one still alive."""
        if not rows:
            return None
        job_id = str(record.get("job_id") or "")
        if job_id:
            return next((row for row in rows if row.job_id == job_id), None)
        alive = [row for row in rows if row.state != ENDED]
        return (alive or list(rows))[0]

    def _note_row(self, record: dict, row: Row, rows: Sequence[Row], now: float) -> bool:
        """Write what ``row`` shows into ``record``; whether anything changed."""
        before = dict(record)
        if row.job_id and not record.get("job_id"):
            record["job_id"] = row.job_id
        if row.state in (RUNNING, PENDING):
            record.pop("leaving_since", None)
        if row.state == RUNNING:
            record.setdefault("running_since", now)
            record.pop("lost", None)
            if row.end:
                record["end"] = row.end
        elif row.state == ENDING:
            record.setdefault("leaving_since", now)
        elif row.state == PENDING and record.get("running_since"):
            # It ran and is queued again: requeued all the same (a launch that failed).
            record.pop("running_since")
            record["lost"] = {"at": now, "state": row.detail or "REQUEUED", "reason": row.reason}
            self._say_lost(record)
        elif row.state == ENDED:
            word = (row.detail.split() or [""])[0]
            # Gone from every listing after its own end is an end, not a loss.
            past = record.get("end") and now >= float(record["end"]) - FIT_MARGIN_S
            own = word in _OWN_END or record.get("stop_requested") or (word == "GONE" and past)
            if not own:
                record["lost"] = {"at": now, "state": row.detail, "reason": row.reason}
                self._say_lost(record)
            # With the scheduler's own start and end: what it held counts toward the bound.
            record["closed"] = {
                "state": row.detail or ENDED,
                "at": now,
                "start": row.start,
                "end": row.end,
            }
            self._stop_marker(str(record["name"])).unlink(missing_ok=True)
        return record != before

    # ------------------------------------------------------------------- pass

    def reconcile(self, demand: Mapping[int, float], *, retry_now: bool = False) -> list[Event]:
        """One pass: read, submit what the renewal test asks for, say what stands."""
        now = self.clock()
        stopping = self.paths.stop.exists()
        watchers = self._watchers(now)
        events = device_events(watchers, now)
        holders = self.observe(now)
        if holders is None:
            # A failed question is not an empty answer: nothing is decided this pass.
            self._publish(now, None, events, stopping)
            return events
        ready = [] if stopping else gpu_jobs(self.paths.pending)
        self._give_back(holders, ready, now)
        if not stopping:
            self._return_unneeded(holders, demand, now)
        self._finish_asked(holders)
        # What a holder on its way out holds is no longer anybody's to look at.
        going = {h.name for h in holders if h.asked_to_end}
        events = device_events([w for w in watchers if w.get("holder") not in going], now)
        events += self._standing(holders, now)
        over = None
        if not stopping:
            for nodes in sorted(demand, reverse=True):
                for event in self._renew(
                    nodes, float(demand[nodes]), holders, now, retry_now, ready
                ):
                    if event.key != "budget":
                        events.append(event)
                    elif over is None:
                        over = event  # one bound, one event, whatever shapes it stops
                        events.append(event)
        # At the bound with no holder left, nothing of the run can run: its controller stops.
        self.exhausted = over.detail if over and not any(h.open for h in holders) else ""
        late = "" if stopping else self._late(holders, ready, now)
        if late and not self.late:
            self._say(late)
        self.late = late
        self._publish(now, holders, events, stopping)
        return events

    def asked(self) -> Asked:
        """What the run has asked its scheduler for, from the records, against its bound."""
        return asked(self.paths.root, bound=getattr(self.env, "max_node_hours", None))

    def _give_back(
        self, holders: Sequence[Holder], ready: Sequence[tuple[str, int, float]], now: float
    ) -> None:
        """Ask a holder to end that cannot take the work waiting on it: a device is held.

        Memory no job of the run accounts for (a node that came with it) does not go by
        waiting, and a job that needs the node whole would wait as long as the holder lasts.
        A holder that runs nothing while work that fits it has been ready as long as the
        memory is reported, and whose nodes report such memory, is ended; the next one is
        asked for without those nodes. On one host there is no elsewhere: the event stands.
        """

        def waits(name: str) -> bool:
            try:
                since = (self.paths.pending / f"{name}.sh").stat().st_mtime
            except OSError:
                return False  # it started meanwhile
            return now - since >= DEVICE_HELD_REPORT_S

        given = sum(1 for h in holders if "avoid" in h.record)
        for holder in holders:
            if self.env.kind != KIND_SLURM or given >= GIVE_BACK_MOST or holder.working:
                continue
            if not any(
                waits(name) and self._takes(holder, nodes, wall, now) for name, nodes, wall in ready
            ):
                continue
            hosts = sorted(
                # Slurm knows a node by its short name, whatever the node calls itself.
                str(w.get("host")).split(".")[0]
                for w in holder.watchers
                if any(
                    now - float(held.get("since", now)) >= DEVICE_HELD_REPORT_S
                    for held in (w.get("held") or {}).values()
                )
            )
            if hosts:
                given += 1
                holder.record["avoid"] = hosts
                self._ask_to_end(
                    holder,
                    now,
                    f"devices on {', '.join(hosts)} hold memory no job of the run accounts "
                    f"for and ready work waits on them: asked for again without those nodes",
                )

    def _return_unneeded(
        self, holders: Sequence[Holder], demand: Mapping[int, float], now: float
    ) -> None:
        """Ask a holder to end that no ready or running work of its own width needs.

        Work needs the holder of its own width. Five nodes kept for one-node jobs are four
        that serve nothing; one node kept between two one-node jobs is what the run asked
        for. A holder nothing needs is kept as long as getting it again would take, its
        own wait in the queue and a start, and is then asked to end: a job it still runs
        finishes first, and work of its width that comes later asks for the next one.
        Where the wait was hours, that is longer than the holder lasts.
        """
        if self.env.kind != KIND_SLURM:
            return  # this host goes back to nobody
        needed = {*demand, *(nodes for _, nodes, _ in gpu_jobs(self.paths.running))}
        for holder in holders:
            record = holder.record
            waits = holder.state in (RUNNING, PENDING) and not holder.asked_to_end
            if not waits or holder.nodes in needed:
                if record.pop("unneeded_since", None) is not None:
                    self._write(record)
                continue
            if "unneeded_since" not in record:
                record["unneeded_since"] = now
                self._write(record)
            since = float(record["unneeded_since"])
            asked = float(record.get("written_at") or now)
            took = max(0.0, float(record.get("running_since") or now) - asked)
            if now - since >= took + START_SLACK_S:
                self._ask_to_end(
                    holder,
                    now,
                    f"no work of {holder.nodes} node(s) has been ready or running for "
                    f"{span(now - since)}; it was kept as long as it took to get "
                    f"({span(took)}) and a start",
                )

    def _late(
        self, holders: Sequence[Holder], ready: Sequence[tuple[str, int, float]], now: float
    ) -> str:
        """Why nothing more can start on this one host, or ``""``.

        Inside a job the host's watcher ends with that job, and another holder here would be
        the same place with the same end. When a job is ready, none that is ready fits what
        is left and none runs, the work waits for the controller's next job.
        """
        live = [h for h in holders if h.state == RUNNING and h.watchers and h.end is not None]
        if self.env.kind != KIND_LOCAL or not ready or not live or any(h.working for h in live):
            return ""
        if any(self._takes(h, nodes, wall, now) for h in live for _, nodes, wall in ready):
            return ""
        name, _, wall = min(ready, key=lambda job: job[2])
        return (
            f"job {name} states a wall of {span(wall)} and this host's watcher ends in "
            f"{span(min(h.end for h in live) - now)}: no job that is ready fits what is left "
            f"of it, and none runs"
        )

    def _over_budget(self, nodes: int) -> Event | None:
        """``POOL_CAPACITY`` when one more holder of ``nodes`` nodes would pass the run's bound."""
        time_s = getattr(self.env, "time_s", None)
        so_far = self.asked()
        if so_far.bound is None or not time_s:
            return None
        more = nodes * float(time_s) / 3600
        if not so_far.over(more):
            return None
        return Event(
            POOL_CAPACITY,
            "budget",
            f"the run's {so_far.render()}; one more holder of {nodes} node(s) for "
            f"{span(time_s)} would make {so_far.holders + more:.1f}. No holder is asked for; "
            f"what runs finishes, and once no holder is left the run stops and waits (exit "
            f"76). `resolve pool-adjust pool --set max_node_hours=<more>` raises the bound",
        )

    def _horizon(self) -> float:
        """The longest wall plus margin a fresh holder of this environment can still fit."""
        return fit_wall_s(getattr(self.env, "time_s", None) or math.inf, 0.0)

    def _takes(self, holder: Holder, nodes: int, wall: float, now: float) -> bool:
        """Whether ``holder`` is live and a ready job of this shape and wall would start on it.

        Live: it runs, was not asked to end, and a watcher of it heart-beats. A job of any
        shape up to the holder's own fits while its wall and the margin end before the holder.
        """
        if holder.state != RUNNING or holder.asked_to_end or not holder.watchers:
            return False
        left = math.inf if holder.end is None else holder.end - now
        return holder.nodes >= nodes and left >= wall + FIT_MARGIN_S

    def _renew(
        self,
        nodes: int,
        wall: float,
        holders: list[Holder],
        now: float,
        retry_now: bool = False,
        ready: Sequence[tuple[str, int, float]] = (),
    ) -> list[Event]:
        """What work of ``nodes`` nodes has to be asked for; may append a new holder.

        Work counts on a holder of its own width: a wider one is where a narrower job may
        run while it is there, not what the narrower work is promised. One is asked for
        when none of that width runs or is on its way; ``lead`` before the last one stops
        fitting the longest job, and never in a holder's first half (a wall near a holder's
        whole time would otherwise ask for a new one every few minutes); and when a job
        that is ready fits what is left of no holder and would fit a fresh one.
        """
        if self.env.kind == KIND_LOCAL and nodes > 1:
            # Not asked for: this one host's one watcher would be ended for a holder it
            # cannot be.
            return [
                Event(
                    POOL_CAPACITY,
                    f"nodes={nodes}",
                    f"work of {nodes} node(s) cannot be held: this one host is one node, and "
                    f"several nodes are Slurm's to hold (environment.slurm)",
                )
            ]
        horizon = self._horizon()
        events = []
        if wall + FIT_MARGIN_S > horizon:
            # No holder of this spec can ever start it. That is said, with the numbers; the
            # shape is still planned for with what one holder lasts, for the shorter work of it.
            lasts = float(getattr(self.env, "time_s", 0) or 0)
            events.append(
                Event(
                    POOL_CAPACITY,
                    f"wall:nodes={nodes}",
                    f"a job of {nodes} node(s) states a wall of {span(wall)} ({wall:.0f}s); one "
                    f"holder of this spec lasts {span(lasts)} ({lasts:.0f}s), and a job starts "
                    f"only where its wall and {FIT_MARGIN_S:.0f}s end before the holder does: "
                    f"it can never start. Lengthen the holders' time or shorten the wall",
                )
            )
        mine = [h for h in holders if h.open and h.nodes == nodes and not h.leaving]
        if any(h.state in (PENDING, UNKNOWN) for h in mine):
            return events  # one is on its way
        live = [h for h in mine if h.state == RUNNING and not h.asked_to_end]
        if live and self.env.kind == KIND_LOCAL:
            # One host: another holder is the same place with the same end. Work that does
            # not fit what is left is the controller's to continue.
            return events
        ahead = min(min(wall + FIT_MARGIN_S, horizon) + self.env.lead_s, horizon / 2)
        lasts = any(h.end is None or h.end - now >= ahead for h in live)
        # A ready job that no holder takes and a fresh one would. Not one that was ready
        # before the newest holder of this width was asked for: that one was asked for it.
        newest = max((float(h.record.get("written_at") or 0) for h in live), default=0.0)
        waits = any(
            width == nodes
            and limit + FIT_MARGIN_S <= horizon
            and self._ready_at(name) > newest
            and not any(self._takes(h, width, limit, now) for h in holders)
            for name, width, limit in ready
        )
        if lasts and not waits:
            return events
        refused = self._last_refusal(nodes, holders)
        waited = refused is None or now - float(refused["closed"]["at"]) >= REFUSAL_RETRY_S
        if not (waited or retry_now):
            return events + [self._refused_event(refused)]
        over = self._over_budget(nodes)
        if over is not None:
            return events + [over]
        holder = self._submit(nodes, now)
        holders.append(holder)
        if holder.state == REFUSED:
            events.append(self._refused_event(holder.record))
        elif holder.state == UNKNOWN:
            events.append(self._unknown_event(holder))
        return events

    def _ready_at(self, name: str) -> float:
        """Since when job ``name`` has been ready; 0 once a watcher took it."""
        try:
            return (self.paths.pending / f"{name}.sh").stat().st_mtime
        except OSError:
            return 0.0

    def _submit(self, nodes: int, now: float) -> Holder:
        """Write the record, then ask the environment; the record is resolved by its answer."""
        seq = max((int(r["seq"]) for r in self.records()), default=0) + 1
        record: dict = {
            "seq": seq,
            "name": self.env.holder_name(seq),
            "nodes": nodes,
            # What was asked for: its nodes, and how long it may last (the run's bound).
            "time_s": getattr(self.env, "time_s", None),
            "written_at": now,
        }
        self._write(record)
        self._stop_marker(record["name"]).unlink(missing_ok=True)
        given = [r for r in self.records() if r.get("avoid")]
        avoid = sorted({host for r in given for host in r["avoid"]})
        reply = self.env.submit(record["name"], nodes, avoid)
        if reply.outcome == REFUSED and avoid:
            # Slurm does not take the request without those nodes: the next one names none.
            for other in given:
                self._write({**other, "avoid": []})
        record["reply"] = {"outcome": reply.outcome, "text": reply.text, "at": self.clock()}
        if reply.outcome == "accepted":
            record["job_id"] = reply.job_id
            state = PENDING
        elif reply.outcome == REFUSED:
            record["closed"] = {"state": REFUSED, "at": self.clock(), "why": reply.text}
            state = REFUSED
        else:
            state = UNKNOWN
        self._write(record)
        self._say(
            f"holder {record['name']} ({nodes} node(s)): {reply.outcome} {reply.text}".strip()
        )
        return Holder(record, state)

    def _ask_to_end(self, holder: Holder, now: float, why: str) -> None:
        """Ask ``holder`` to end: a pending one is cancelled, a running one sees its stop marker."""
        holder.record["stop_requested"] = now
        self._write(holder.record)
        if holder.state == PENDING:
            said = self.env.end(holder.record, pending_only=True)
        else:
            self._stop_marker(holder.name).write_text(f"{now}\n", encoding="utf-8")
            said = ""
        self._say(f"holder {holder.name} asked to end: {why}" + (f" ({said})" if said else ""))

    def _finish_asked(self, holders: Sequence[Holder]) -> None:
        """Cancel a running holder that was asked to end and runs no job any more.

        A job that waits for its nodes keeps none of them: under a stop marker it never starts.
        """
        for holder in holders:
            if not (holder.open and holder.asked_to_end and holder.state == RUNNING):
                continue
            if not holder.working and self._listed_as_recorded(holder):
                self.env.end(holder.record)

    @staticmethod
    def _listed_as_recorded(holder: Holder) -> bool:
        """Whether the id about to be cancelled is both in the record and listed under its name."""
        return bool(holder.job_id) and holder.row is not None and holder.row.job_id == holder.job_id

    # ----------------------------------------------------------------- events

    def _refused_event(self, record: Mapping[str, Any]) -> Event:
        closed = record.get("closed") or {}
        if closed.get("state") == REFUSED:
            said = f"was refused: {closed.get('why') or 'no reason given'}"
        else:
            said = (
                f"left before its end ({closed.get('state') or 'no state'}), as the one before "
                f"it did: job {record.get('job_id') or '?'}, its output is in {self.paths.root}"
            )
        return Event(
            POOL_SUBMIT_FAILED,
            f"nodes={record['nodes']}",
            f"holder {record['name']} ({record['nodes']} node(s)) {said}; tried again every "
            f"{REFUSAL_RETRY_S:.0f}s",
        )

    def _unknown_event(self, holder: Holder) -> Event:
        reply = holder.record.get("reply") or {}
        said = reply.get("text") or "no answer"
        return Event(
            POOL_SUBMIT_FAILED,
            holder.name,
            f"holder {holder.name} ({holder.nodes} node(s)"
            + (f", job {holder.job_id}" if holder.job_id else "")
            + f") is in no listing and its submission was not refused ({said}); no other "
            f"holder of its shape is asked for until it is listed or closed after a look "
            f"(`resolve pool-retry pool --close-unknown {holder.seq}`)",
        )

    @staticmethod
    def _last_refusal(nodes: int, holders: Sequence[Holder]) -> dict | None:
        """The newest record of this shape, when the next request has to wait for it.

        A refusal no later holder answered; or a holder that left before its end as the one
        before it did. One early end is replaced at once (a node went down). The second in a
        row is a request that keeps failing: it is said, and asked for no faster than a
        refused one, however quickly each one ends.
        """
        same = [h for h in holders if h.nodes == nodes]
        if same and same[-1].state == REFUSED:
            return same[-1].record
        early = [h.state == ENDED and "lost" in h.record for h in same[-2:]]
        return same[-1].record if early == [True, True] else None

    def _say_lost(self, record: Mapping[str, Any]) -> None:
        """A holder left before its end: said once, in the log. The first is replaced at once."""
        lost = record["lost"]
        self._say(
            f"holder {record['name']} (job {record.get('job_id') or '?'}, {record['nodes']} "
            f"node(s)) left before its end: {lost.get('state') or 'no state'}"
            + (f" ({lost['reason']})" if lost.get("reason") else "")
            + "; its jobs are re-run once or closed by the scheduler"
        )

    def _standing(self, holders: Sequence[Holder], now: float) -> list[Event]:
        """What stands whatever is asked for: unanswered submissions, holders that serve nothing."""
        events = [self._unknown_event(h) for h in holders if h.state == UNKNOWN]
        for holder in holders:
            since = holder.record.get("unwatched_since")
            # One the pool asked to end is on its way out, and one whose watchers left for a
            # later holder ends in moments: neither is held. Longer with nobody than a start
            # may take is.
            if since is None or holder.asked_to_end or now - float(since) <= START_SLACK_S:
                continue
            events.append(
                Event(
                    POOL_DEVICE_HELD,
                    holder.name,
                    f"holder {holder.name} (job {holder.job_id or '?'}, {holder.nodes} node(s)) "
                    f"runs and no watcher of it has heart-beat since {utc_iso(float(since))}: "
                    f"it holds its nodes and serves no job. What its watcher step said is in "
                    f"{self.paths.root}/slurm-{holder.job_id or '<job>'}.out",
                )
            )
        return events

    def _publish(
        self,
        now: float,
        holders: Sequence[Holder] | None,
        events: Sequence[Event],
        stopping: bool,
    ) -> None:
        """Write the pass where the monitor and the spool's clients read it."""
        previous = read_json(self.paths.pool) or {}
        seen = {(e.get("kind"), e.get("key")) for e in previous.get("events") or ()}
        for event in events:
            if (event.kind, event.key) not in seen:
                self._say(event.render())
        snapshot = {
            "at": now,
            "written": utc_iso(now),
            "kind": self.env.kind,
            "stopping": stopping,
            "holders": previous.get("holders", [])
            if holders is None
            else [
                {
                    "name": h.name,
                    "nodes": h.nodes,
                    "state": h.state,
                    "job_id": h.job_id,
                    "watchers": len(h.watchers),
                    "idle": h.idle,
                    "asked_to_end": h.asked_to_end,
                }
                for h in holders
                if h.open
            ],
            "events": [{"kind": e.kind, "key": e.key, "detail": e.detail} for e in events],
        }
        write_json_atomic(self.paths.pool, snapshot)

    def _say(self, line: str) -> None:
        try:
            with (self.paths.root / POOL_LOG).open("a", encoding="utf-8") as log:
                log.write(f"{utc_iso(self.clock())} {line}\n")
        except OSError:
            pass
        if self.note is not None:
            self.note(line)

    # -------------------------------------------------------------------- end

    def end(
        self, *, wait_s: float = END_WAIT_S, sleep: Callable[[float], None] = time.sleep
    ) -> dict:
        """No work is left, or somebody says stop: nothing of the pool holds or waits for a node.

        A stop marker (no submission, and no watcher takes a job), then two sweeps ``wait_s``
        apart. Each cancels the holders that wait in the queue; the second also those that
        run no job. A job that only waits in the spool keeps no holder; one that runs keeps
        its own, which leaves by itself when the job is over.

        Returns what was cancelled, what is left, and what could not be asked about
        (``unknown``, with ``error``): a question that failed is never "nothing left". Nothing
        the flow did not record is touched.
        """
        report: dict = {"cancelled": [], "left": [], "unknown": []}
        self.paths.stop.write_text(f"{self.clock()}\n", encoding="utf-8")
        for last in (False, True):
            holders = self.observe()
            for holder in holders or ():
                if not (holder.open and self._listed_as_recorded(holder)):
                    continue
                waits = holder.state == PENDING
                if waits or (last and holder.state == RUNNING and not holder.working):
                    holder.record["stop_requested"] = self.clock()
                    self._write(holder.record)
                    said = self.env.end(holder.record, pending_only=waits)
                    was = {"name": holder.name, "job_id": holder.job_id, "was": holder.state}
                    report["cancelled"].append({**was, "said": said})
            if not last and (holders is None or any(h.open for h in holders)):
                sleep(wait_s)
        holders = self.observe()
        if holders is None:
            # Not asked is not gone: every record still open may be an allocation.
            report["unknown"] = [
                {"name": str(r["name"]), "job_id": str(r.get("job_id") or ""), "state": UNKNOWN}
                for r in self.records()
                if not r.get("closed")
            ]
            report["error"] = (
                "the environment could not be asked what is left: every holder still recorded "
                "may be allocated"
            )
        else:
            report["left"] = [
                {"name": h.name, "job_id": h.job_id, "state": h.state} for h in holders if h.open
            ]
        self._publish(self.clock(), holders, [], True)
        rest = report["left"] + report["unknown"]
        report["asked"] = self.asked().render()
        self._say(
            f"pool ending: cancelled {len(report['cancelled'])}, left {len(report['left'])}, "
            f"not asked {len(report['unknown'])}"
            + (": " + ", ".join(f"{h['name']} ({h['state']})" for h in rest) if rest else "")
            + f"; {report['asked']}"
        )
        return report

    def done(self) -> list[str]:
        """End the pool for the process that kept it; a line for each holder it could not end.

        What a closed run and a check run by hand both do when they are done: the end, with
        the wait of the kind, and on one host the watcher itself, which never outlives the
        process that started it.
        """
        local = self.env.kind == KIND_LOCAL
        report = self.end(wait_s=5.0 if local else END_WAIT_S)
        if local:
            for record in self.records():
                if not record.get("closed"):
                    self.env.end(record)
            return []
        return [
            f"holder {rest['name']} (job {rest['job_id'] or '?'}) is {rest['state']} and may "
            f"still be allocated: it runs a job, or could not be asked about"
            for rest in (*report["left"], *report["unknown"])
        ]


# ------------------------------------------------------------------ one host


class Keeper:
    """The controller's loop over a pool: one pass every ``interval_s``, in a daemon thread."""

    def __init__(
        self,
        pool: Pool,
        demand: Callable[[], Mapping[int, float]],
        *,
        interval_s: float = PASS_S,
        on_exhausted: Callable[[str], bool] | None = None,
        on_late: Callable[[str], bool] | None = None,
    ) -> None:
        import threading

        self.pool = pool
        self.demand = demand
        self.interval_s = interval_s
        #: Told, each pass until it takes it, that the run's bound is reached and no holder
        #: is left (``Pool.exhausted``): the run stops, "to be continued".
        self.on_exhausted = on_exhausted
        #: Told the same way that nothing more can start on this host (``Pool.late``).
        self.on_late = on_late
        self._told = self._told_late = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="mv2-pool", daemon=True)

    def start(self) -> "Keeper":
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                self.pool.reconcile(self.demand())
                if self.pool.exhausted and self.on_exhausted is not None and not self._told:
                    self._told = bool(self.on_exhausted(self.pool.exhausted))
                if self.pool.late and self.on_late is not None and not self._told_late:
                    self._told_late = bool(self.on_late(self.pool.late))
            except Exception as exc:  # a pass that failed is retried by the next one
                self.pool._say(f"pool pass failed: {type(exc).__name__}: {exc}")

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=self.interval_s + 5)


class LocalEnvironment:
    """One host, no batch scheduler: the holder is one local watcher the flow starts and stops."""

    kind = "local"
    renews = True  # read by `checks/gate_cli.py` only: it goes with that line
    lead_s = 0.0
    time_s = None

    def __init__(self, spool: str | Path, *, entry: str | Path) -> None:
        self.spool = Path(spool)
        self.entry = Path(entry)
        self._procs: dict[str, subprocess.Popen] = {}

    def holder_name(self, seq: int) -> str:
        return f"local-{seq}"

    def submit(self, name: str, nodes: int, avoid: Sequence[str] = ()) -> Reply:
        del nodes, avoid  # one host is one node: the pool asks it for no other shape
        env = {**os.environ, "MV2FLOW_HOLDER": name, "MV2FLOW_SCHEDULER_PYTHON": sys.executable}
        try:
            log = (self.spool / "local_watcher.log").open("ab")
            with log:
                proc = subprocess.Popen(
                    ["bash", str(self.entry)],
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                    start_new_session=True,
                )
        except OSError as exc:
            return Reply(REFUSED, text=f"the local watcher could not be started: {exc}")
        self._procs[name] = proc
        return Reply("accepted", job_id=str(proc.pid))

    def _alive(self, record: Mapping[str, Any]) -> bool:
        name, pid = str(record["name"]), str(record.get("job_id") or "")
        proc = self._procs.get(name)
        if proc is not None:
            return proc.poll() is None
        if not pid.isdigit():
            return False
        # A watcher of an earlier controller on this spool: the pid alone could be anyone's.
        environ = read_environ(int(pid)) or {}
        return environ.get("MV2FLOW_HOLDER") == name

    def listing(self, records: Sequence[Mapping[str, Any]]) -> dict[str, list[Row]]:
        return {
            str(record["name"]): [
                Row(
                    state=RUNNING if self._alive(record) else ENDED,
                    job_id=str(record.get("job_id") or ""),
                    detail="RUNNING" if self._alive(record) else "EXITED",
                )
            ]
            for record in records
            if record.get("job_id")
        }

    def end(self, record: Mapping[str, Any], *, pending_only: bool = False) -> str:
        """Stop the watcher: its running jobs are cancelled through the spool first."""
        if pending_only or not self._alive(record):
            return ""
        paths = SpoolPaths(self.spool)
        for script in paths.running.glob("*.sh"):
            (paths.cancel / script.stem).write_text(f"{time.time()}\n", encoding="utf-8")
        deadline = time.monotonic() + 20.0
        while any(paths.running.glob("*.sh")) and time.monotonic() < deadline:
            time.sleep(0.2)
        try:
            os.kill(int(record["job_id"]), signal.SIGTERM)
        except (OSError, ValueError) as exc:
            return f"the local watcher could not be signalled: {exc}"
        proc = self._procs.get(str(record["name"]))
        if proc is not None:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        return ""


# --------------------------------------------------------- which environment

KIND_SLURM = "slurm"
KIND_LOCAL = "local"
KINDS: tuple[str, ...] = (KIND_SLURM, KIND_LOCAL)

#: What a spec or a host that is neither Slurm nor one host is told, wherever that is found:
#: when the spec is read, at `intake derive`, at the dry run and at the start. One message,
#: no event and nothing stored: the flow has no launcher for it.
OTHER_ENVIRONMENT = (
    "the flow starts engines through Slurm (`environment.slurm`, where `sbatch` is on PATH) or "
    "on this one host (`environment.kind: local`), and has no launcher for another "
    "environment. To start engines on several nodes it needs three things of one: a holder "
    "that keeps whole nodes and their devices and runs the flow's watcher entry once on each "
    "node and its lead entry once; a way to start one MPI world of R ranks on named hosts of "
    "a holder; and launched shells that keep the flow's exports (the devices, the paths, the "
    "role) and see the same paths on every node. A site adds such a launcher outside the "
    "public tree, as `infra/compute/slurm.py` is the one for Slurm. The flow serves no "
    "watcher that somebody else starts: it asks for its own holders and gives them back"
)
NO_SBATCH = f"the spec states Slurm and this host has no `sbatch`: {OTHER_ENVIRONMENT}"


def environment_kind(block: Mapping[str, Any] | None) -> str:
    """Which environment a spec's ``environment`` block names; one that names none is Slurm.

    Stated, ``kind`` rules. Else Slurm: never a silent fall-back to this host.
    """
    return str((block or {}).get("kind") or KIND_SLURM)


def said_kind(block: Mapping[str, Any] | None) -> str:
    """Which environment a spec is taken for, in one line, with how one that states none is decided.

    For a start that failed before it could say what it launches with: the reader sees the
    kind the failure belongs to, and the one key that changes it.
    """
    return (
        f"the environment is `{environment_kind(block)}` (environment.kind). A block that "
        "names none (no `kind` and no `slurm`) is taken for Slurm where `sbatch` is on "
        "PATH and for this one host (`local`) where it is not: state `environment.kind` when "
        "that is not what this is"
    )


def holder_time_s(block: Mapping[str, Any] | None) -> float | None:
    """How long one holder of this environment lasts; ``None`` when it has no end or states none."""
    block = block or {}
    try:
        if environment_kind(block) == KIND_SLURM and (block.get("slurm") or {}).get("time"):
            from .slurm import slurm_time_seconds

            return slurm_time_seconds(block["slurm"]["time"])
    except ValueError:
        return None
    return None


def fit_wall_s(holder_s: float, margin_s: float = FIT_MARGIN_S) -> float:
    """The longest wall a job may state and still start in a fresh holder that lasts ``holder_s``.

    The one rule for "fits a holder". A holder's start costs :data:`START_SLACK_S` before
    its watchers serve, and a watcher admits a job only where its wall and the margin end
    before the holder does (the scheduler's Fit). What derive gives a session, what
    preflight refuses and what the pool calls ``POOL_CAPACITY`` are this one number.
    """
    return holder_s - START_SLACK_S - margin_s


WATCHER_ENTRY = "watcher_entry.sh"
LEAD_ENTRY = "lead_entry.sh"


def resolve_spool(task_data: dict, workspace: str | Path) -> None:
    """State the environment's kind and the run's one spool in the resolved spec.

    Never a silent choice: the spec the agents and the gates read names both. The spool is the
    spec's own, else ``<workspace>/spool``.
    """
    block = task_data.get("environment")
    if not isinstance(block, dict):
        return
    kind = environment_kind(block)
    block["kind"] = kind
    if not block.get("spool"):
        block["spool"] = str(Path(workspace).resolve() / "spool")


def build(
    task_data: Mapping[str, Any],
    *,
    workspace: str | Path | None = None,
    note: Callable[[str], None] | None = None,
    clock: Callable[[], float] = time.time,
) -> Pool:
    """The pool of a run, from its resolved spec (``environment.spool`` is set by then).

    Renders the entries of the watchers the flow starts itself and picks the environment that
    fills the slots. Raises :class:`PoolError` with the line that says what is missing.
    """
    from . import SCHEDULER_PATH
    from . import sessions as sessions_mod

    block = dict(task_data.get("environment") or {})
    kind = environment_kind(block)
    spool = block.get("spool")
    if not spool:
        raise PoolError("environment.spool is not set: the run's spool is resolved before its pool")
    spool = Path(str(spool))
    spool.mkdir(parents=True, exist_ok=True)
    if kind == KIND_LOCAL:
        entry = sessions_mod.write_entry(
            spool / WATCHER_ENTRY,
            sessions_mod.render_entry(task_data, spool=str(spool), session="local"),
        )
        return Pool(spool, LocalEnvironment(spool, entry=entry), note=note, clock=clock)
    from . import slurm as slurm_mod

    run = slurm_mod.run_id(spool)
    if kind == KIND_SLURM:
        cfg = dict(block.get("slurm") or {})
        errors: list[str] = []
        slurm_mod.validate(cfg, errors)
        errors += slurm_mod.mount_errors(
            cfg,
            {
                "spool": str(spool),
                # Every node runs the scheduler by its path: it must be under a mount too.
                "flow's own scheduler": str(SCHEDULER_PATH),
                # As every node will see it: a workspace named relative to where the run
                # was started is under a mount when that directory is.
                "workspace": str(Path(workspace).resolve()) if workspace else None,
                "checkpoint": task_data.get("checkpoint_path"),
                "checkout": task_data.get("trtllm_repo_path"),
            },
        )
        if not slurm_mod.have_sbatch():
            # Whatever else the block lacks, this host cannot submit: the alternatives first.
            raise PoolError(NO_SBATCH)
        if errors:
            raise PoolError("\n".join(errors))
        watcher = sessions_mod.write_entry(
            spool / WATCHER_ENTRY,
            sessions_mod.render_entry(
                task_data, spool=str(spool), session="pool", gpus_per_node=int(cfg["gpus_per_node"])
            ),
        )
        lead = sessions_mod.write_entry(
            spool / LEAD_ENTRY,
            sessions_mod.render_entry(
                task_data,
                spool=str(spool),
                session="pool",
                lead=True,
                host_python=str(cfg.get("host_python") or "python3"),
            ),
        )
        environment = slurm_mod.SlurmEnvironment(
            cfg, spool=spool, run=run, watcher_entry=watcher, lead_entry=lead
        )
        return Pool(spool, environment, note=note, clock=clock)
    raise PoolError(f"environment.kind is {kind!r}: {OTHER_ENVIRONMENT}")


# ------------------------------------------------------------------------ CLI


def main(argv: list[str] | None = None) -> int:
    """``trtllm-modeling-v2 pool status|stop``: the pool of one workspace.

    ``status`` prints the controller's last pass (holders, events) and the submission records;
    ``stop`` forbids any further submission and ends the holders that run nothing. Exit 0, or
    1 with one line that says why.
    """
    import argparse
    import json

    import yaml

    parser = argparse.ArgumentParser(
        prog="trtllm-modeling-v2 pool", description="The pool of one workspace: its holders."
    )
    parser.add_argument(
        "--workspace", default=os.environ.get("MV2FLOW_WORKSPACE"), help="the run's workspace"
    )
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("status", help="holders, standing events and submission records")
    stop = sub.add_parser("stop", help="forbid submissions, then end the holders that run nothing")
    stop.add_argument(
        "--wait", type=float, default=END_WAIT_S, help="seconds between the two cancels"
    )
    for action in sub.choices.values():
        # `pool stop --workspace W` is as good as `pool --workspace W stop`: the flow's own
        # lines and the operating guides write the first.
        action.add_argument("--workspace", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not args.workspace:
        print("error: pass --workspace or set MV2FLOW_WORKSPACE", file=sys.stderr)
        return 1
    try:
        task = yaml.safe_load((Path(args.workspace) / "task.yaml").read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        print(f"error: cannot read the workspace's task.yaml: {exc}", file=sys.stderr)
        return 1
    task = task if isinstance(task, dict) else {}
    # A spec that names no spool has the workspace's, as a run and a check by hand resolve it.
    resolve_spool(task, args.workspace)
    spool = (task.get("environment") or {}).get("spool")
    if not spool:
        print("error: the workspace's task.yaml names no environment.spool", file=sys.stderr)
        return 1
    paths = SpoolPaths(spool)
    if args.action == "status":
        found = sorted(paths.holders.glob("*.json"), key=lambda p: p.stem.zfill(9))
        records = [record for record in (read_json(p) for p in found) if record]
        snapshot = read_json(paths.pool) or {}
        # What the run has asked its scheduler for, against its bound: the spec as the run
        # holds it (what `resolve` changed stands over the file).
        held = _spec_as_held(task, args.workspace)
        so_far = asked_of(held)
        print(
            f"pool {spool}: {snapshot.get('kind') or '?'}, "
            f"last pass {snapshot.get('written') or 'never'}"
        )
        if so_far is not None and (so_far.bound is not None or so_far.holder_jobs):
            print(f"  {so_far.render()}")
        for holder in snapshot.get("holders") or ():
            print(
                f"  holder {holder.get('name')} nodes={holder.get('nodes')} {holder.get('state')} "
                f"job={holder.get('job_id') or '-'} watchers={holder.get('watchers')}"
            )
        for event in snapshot.get("events") or ():
            print(f"  {event.get('kind')}  {event.get('detail')}")
        for record in records:
            closed = record.get("closed") or {}
            print(
                f"  record {record.get('seq')} {record.get('name')} nodes={record.get('nodes')} "
                f"job={record.get('job_id') or '-'} "
                + (f"closed: {closed.get('state')}" if closed else "open")
            )
        return 0
    try:
        print(json.dumps(build(task, workspace=args.workspace).end(wait_s=args.wait), indent=2))
    except PoolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _spec_as_held(task: Mapping[str, Any], workspace: str | Path) -> dict:
    """``task`` with every recorded change of the spec put back, as a run reads it."""
    import copy

    held = copy.deepcopy(dict(task))
    try:
        from ...run import escalation as escalation_mod

        escalation_mod.reapply_changes(workspace, held)
    except Exception:  # a record that cannot be read changes nothing here
        pass
    return held


__all__ = [
    "DEFAULT_MAX_NODE_HOURS",
    "ENDED",
    "ENDING",
    "KINDS",
    "KIND_LOCAL",
    "KIND_SLURM",
    "NO_SBATCH",
    "OTHER_ENVIRONMENT",
    "PENDING",
    "POOL_CAPACITY",
    "POOL_DEVICE_HELD",
    "POOL_SUBMIT_FAILED",
    "REFUSED",
    "RUNNING",
    "UNKNOWN",
    "Asked",
    "Environment",
    "Event",
    "Holder",
    "Keeper",
    "LocalEnvironment",
    "Pool",
    "PoolError",
    "Reply",
    "Row",
    "asked",
    "asked_of",
    "build",
    "device_events",
    "environment_kind",
    "holder_time_s",
    "main",
    "resolve_spool",
    "said_kind",
    "span",
    "spool_demand",
    "standing_events",
]
