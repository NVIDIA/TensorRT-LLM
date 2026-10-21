# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where a job can start: what a spool's sessions can do for it, and the spools of its kind.

Standard-library only and importable by path, like :mod:`spoolctl`, which
uses it to warn when a submission would sit waiting and to choose a spool for
``--pick``; the gate executor's pool (``infra/execution/spool.py``) applies
the same rule to its own jobs, and counts by it how many more checks would
start at once. Everything here is read from files the schedulers write:
each watcher's ``watchers/<id>.json`` (the record ``status.json`` also
holds) and ``watcher.heartbeat``, the spool's ``handover/request.json`` and
``handover/acks/``, the session's rendered ``session_entry.sh`` for its default
wall limit, and the scripts waiting in ``pending/``.

KIND. A run keeps its spools side by side in one directory and names the
sessions that do the same work ``<kind>``, ``<kind>-2``, ... or
``<kind>-1``, ``<kind>-2``, ... (``gates-1``, ``work-2``, ``perf``,
``perf-3``, ``dev-2``): the trailing number counts allocations of one kind,
the stem says what they are for. A spool's kind is therefore its name without
a trailing ``-<n>`` (or ``_<n>``), and its siblings are the spools in the
same directory with the same kind. A job is only ever offered siblings: a
session of another kind is reserved for other work (gates, perf
measurements, worker jobs), and a job sent there competes with what that
session is for. A letter suffix (``spool-a``, ``spool-b``) or a number with
no separator (``b200``) is part of the name, not a replica count.

CAN IT TAKE THE JOB. A spool can take a job when one of its live watchers (a
heartbeat within ``stale_s``) would admit it and the job can finish there.
That watcher:

- is not stopping (``STOP``) and is not handing over to a successor that is
  already up (it admits nothing more; the successor is judged on its own
  record);
- if it is in hand-over (it has requested its successor), is not asked to
  run a GPU job whose wall limit, plus :data:`scheduler.HANDOVER_MARGIN_S`,
  reaches past the allocation's end: the scheduler holds such a job for the
  successor;
- outside hand-over, has at least the job's wall limit (plus the same
  margin) left, because a job admitted there is stopped with the allocation;
- has as many devices as the job asks for.

A job's wall limit is its ``wall`` header, else the session's
``--default-wall`` for a GPU job and none for a device-free one, as the
scheduler classifies it. ``--pick`` also passes over a spool whose best
watcher is in hand-over even when the job would fit: a sibling with its
allocation ahead of it is the better home.
"""

from __future__ import annotations

import calendar
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from . import scheduler as _sched
except ImportError:  # run by path: the sibling module is on sys.path[0]
    import scheduler as _sched  # type: ignore[no-redef]

#: The spool's rendered entry script, which carries the session's scheduler flags.
ENTRY_SCRIPT = "session_entry.sh"

_REPLICA_SUFFIX = re.compile(r"[-_]\d+$")
_DEFAULT_WALL_FLAG = re.compile(r"--default-wall[= ]+['\"]?([^\s'\"]+)")


def spool_kind(spool: str | Path) -> str:
    """``gates-2`` -> ``gates``, ``perf`` -> ``perf``: the name without its replica number."""
    name = Path(spool).name
    return _REPLICA_SUFFIX.sub("", name) or name


def is_spool(path: Path) -> bool:
    """Whether ``path`` is a spool directory (a watcher served it, or it was made for one)."""
    return path.is_dir() and any(
        (path / part).exists() for part in ("status.json", "watchers", "pending")
    )


def siblings(spool: str | Path) -> list[Path]:
    """Every spool in ``spool``'s directory with its kind, ``spool`` itself included if present."""
    root = Path(spool)
    kind = spool_kind(root)
    try:
        entries = sorted(root.parent.iterdir())
    except OSError:
        return []
    return [p for p in entries if spool_kind(p) == kind and is_spool(p)]


def span(seconds: float | None) -> str:
    """``45s``, ``7m``, ``2h05m``, ``3h``: a duration as a person reads it."""
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


@dataclass(frozen=True)
class JobNeeds:
    """What a job asks of a session, classified as the scheduler classifies it."""

    name: str
    #: Device count, or ``None`` for the whole node.
    gpus: int | None
    #: The ``wall`` header in seconds; ``None`` takes the session's default (0 is none).
    wall_s: float | None
    holds_gpus: bool
    #: Who submitted it (the ``owner`` header), for how long such jobs ran (:func:`expected_on`).
    owner: str = ""

    @classmethod
    def from_script(cls, name: str, script: str) -> "JobNeeds":
        spec = _sched.job_spec_from_script(name, script)
        return cls(
            name=name,
            gpus=spec.gpus,
            wall_s=spec.wall_s,
            holds_gpus=spec.holds_gpus,
            owner=spec.owner,
        )

    def expected_on(self, spool: Path, default_wall_s: float) -> float:
        """How long this job is expected to run on ``spool``: its history's, else its wall limit."""
        wall = self.wall_on(default_wall_s)
        gpus = _sched.ALL_GPUS if self.gpus is None else str(self.gpus)
        return _sched.expected_s(spool / "done", self.owner, wall, gpus) or wall

    def wall_on(self, default_wall_s: float) -> float:
        """The wall limit this job runs under on a session whose default is ``default_wall_s``."""
        if self.wall_s is not None:
            return float(self.wall_s)
        return default_wall_s if self.holds_gpus else 0.0


@dataclass
class SpoolView:
    """What one spool can do for one job now, through its best live watcher."""

    path: Path
    #: Why the spool cannot take the job; empty when it can.
    reasons: list[str] = field(default_factory=list)
    devices: int = 0
    free: int = 0
    #: Seconds until the allocation ends; ``None`` when it has no known end.
    remaining_s: float | None = None
    #: GPU jobs already waiting in the spool.
    queued: int = 0
    #: Whether the job would start at once: its devices are free and no GPU job waits.
    starts_now: bool = False
    #: Whether the watcher that would run it has requested its successor.
    in_handover: bool = False

    @property
    def can_take(self) -> bool:
        return not self.reasons

    @property
    def pickable(self) -> bool:
        """Whether ``--pick`` may send the job here."""
        return self.can_take and not self.in_handover

    def state(self) -> str:
        """``3 of 4 GPUs free, 23h59m left``, plus the queue when there is one."""
        left = "no end" if self.remaining_s is None else f"{span(self.remaining_s)} left"
        queue = f", {self.queued} GPU job(s) waiting" if self.queued else ""
        handover = ", in hand-over" if self.in_handover else ""
        return f"{self.free} of {self.devices} GPUs free, {left}{queue}{handover}"


def heartbeat_epoch(spool: str | Path) -> float | None:
    """When any watcher on ``spool`` last wrote ``watcher.heartbeat``; ``None`` if none did."""
    try:
        text = _sched.SpoolPaths(spool).heartbeat.read_text(encoding="utf-8").strip()
        if text.startswith("alive="):
            return float(
                calendar.timegm(time.strptime(text[len("alive=") :], "%Y-%m-%dT%H:%M:%SZ"))
            )
    except (OSError, ValueError):
        pass
    return None


def watcher_records(spool: str | Path) -> list[dict]:
    """Each watcher's latest record on ``spool``: ``watchers/*.json`` and ``status.json``."""
    paths = _sched.SpoolPaths(spool)
    found: dict[str, dict] = {}
    for record in (
        _sched.read_json(p) for p in [*sorted(paths.watchers.glob("*.json")), paths.snapshot]
    ):
        if not record or "alive_epoch" not in record:
            continue
        key = str(record.get("watcher") or "")
        held = found.get(key)
        if held is None or float(record["alive_epoch"]) > float(held["alive_epoch"]):
            found[key] = record
    return list(found.values())


def default_wall_s(spool: str | Path, record: Mapping[str, Any] | None = None) -> float:
    """The session's wall limit for a GPU job that declares none."""
    if record and record.get("default_wall_s") is not None:
        return float(record["default_wall_s"])
    try:
        text = (Path(spool) / ENTRY_SCRIPT).read_text(encoding="utf-8", errors="replace")
        match = _DEFAULT_WALL_FLAG.search(text)
        if match:
            return _sched.parse_duration(match.group(1))
    except (OSError, ValueError):
        pass
    return _sched.SchedulerConfig.default_wall_s


def _successor_state(spool: Path, record: Mapping[str, Any]) -> str:
    """Whether the driver has submitted the successor this watcher asked for."""
    paths = _sched.SpoolPaths(spool)
    acks = paths.read_acks()
    request = _sched.read_json(paths.handover_request)
    if request and request.get("watcher") not in (None, record.get("watcher")):
        request = None
    allocation = str(record.get("slurm_job") or "")
    ack = _sched.answering_ack(acks, request) or _sched.following_ack(
        acks, allocation, str(record.get("session") or "")
    )
    if ack:
        return f"successor job {ack.get('job_id')} submitted, not up yet"
    return "no successor submitted yet"


def _in_handover(spool: Path, record: Mapping[str, Any], now: float) -> bool:
    """Whether this watcher has requested its successor (its own record, else ``request.json``)."""
    if record.get("handover"):
        return True
    request = _sched.read_json(spool / "handover" / "request.json") or {}
    if not request.get("watcher") or request.get("watcher") != record.get("watcher"):
        return False
    try:
        return float(request.get("ends_at_epoch") or 0) > now
    except (TypeError, ValueError):
        return False


def _judge(
    spool: Path, record: Mapping[str, Any], job: JobNeeds, now: float, stopping: bool
) -> SpoolView:
    """What one live watcher can do for ``job``."""
    devices = len(record.get("devices") or [])
    free = len(record.get("free") or [])
    ends = record.get("ends_at_epoch")
    remaining = float(ends) - now if ends is not None else None
    queued = sum(
        1 for entry in record.get("pending") or [] if str(entry.get("gpus", "")) not in ("", "0")
    )
    handover = record.get("handover") or {}
    in_handover = _in_handover(spool, record, now)
    wall = job.expected_on(spool, default_wall_s(spool, record))
    reasons = []
    if stopping or record.get("stopping"):
        reasons.append("the session is stopping (STOP)")
    elif handover.get("successor"):
        reasons.append(f"it is handing over to {handover['successor']} and admits nothing more")
    elif (
        in_handover
        and remaining is not None
        and job.holds_gpus
        and (wall <= 0 or wall + _sched.HANDOVER_MARGIN_S > remaining)
    ):
        allowed = "unlimited time" if wall <= 0 else span(wall)
        reasons.append(
            f"it is in hand-over and its allocation ends in {span(remaining)}, too soon for a "
            f"job allowed {allowed}; the job waits for the successor "
            f"({_successor_state(spool, record)})"
        )
    elif remaining is not None and wall > 0 and wall + _sched.HANDOVER_MARGIN_S > remaining:
        reasons.append(
            f"the allocation ends in {span(remaining)}, before the job's {span(wall)} wall limit"
        )
    if job.gpus is not None and job.gpus > devices:
        reasons.append(f"the job asks for {job.gpus} GPU(s) and this session has {devices}")
    if not job.holds_gpus:
        starts_now = True
    elif job.gpus is None:
        starts_now = free == devices and not queued
    else:
        starts_now = free >= job.gpus and not queued
    return SpoolView(
        path=spool,
        reasons=reasons,
        devices=devices,
        free=free,
        remaining_s=remaining,
        queued=queued,
        starts_now=starts_now,
        in_handover=in_handover,
    )


def rank(view: SpoolView, reference: Path | None = None) -> tuple:
    """Sort key, best first.

    Out of hand-over, then starts at once, most free devices, shortest queue,
    the reference spool, most time left.
    """
    return (
        view.in_handover,
        not view.starts_now,
        -view.free,
        view.queued,
        reference is not None and view.path != reference,
        -(math.inf if view.remaining_s is None else view.remaining_s),
        view.path.name,
    )


def _live(records: Iterable[Mapping[str, Any]], *, stale_s: float, now: float) -> list:
    return [r for r in records if now - float(r["alive_epoch"]) <= stale_s]


def _views(root: Path, live: Sequence[Mapping[str, Any]], job: JobNeeds, now: float) -> list:
    stopping = _sched.SpoolPaths(root).stop.exists()
    return [_judge(root, record, job, now, stopping) for record in live]


def watcher_views(
    spool: str | Path, job: JobNeeds, *, stale_s: float, now: float
) -> list[SpoolView]:
    """What each live watcher on ``spool`` can do for ``job`` now; empty when none is live.

    One per allocation serving the spool (a node of a multi-node session, or a
    session and its successor during hand-over), so their free devices are
    never pooled across nodes.
    """
    root = Path(spool)
    return _views(root, _live(watcher_records(root), stale_s=stale_s, now=now), job, now)


def queued_jobs(spool: str | Path) -> list[JobNeeds]:
    """Each job waiting in ``spool``'s queue on disk, classified as the scheduler classifies it.

    The queue on disk is ahead of every watcher's record: a job submitted since
    the last tick is already there.
    """
    found = []
    for script in sorted(_sched.SpoolPaths(spool).pending.glob("*.sh")):
        try:
            text = script.read_text(encoding="utf-8", errors="replace")
        except OSError:  # a watcher claimed it meanwhile
            continue
        found.append(JobNeeds.from_script(script.name[: -len(".sh")], text))
    return found


def assess(spool: str | Path, job: JobNeeds, *, stale_s: float, now: float) -> SpoolView:
    """What ``spool`` can do for ``job`` now: through its best live watcher, or why not."""
    root = Path(spool)
    records = watcher_records(root)
    live = _live(records, stale_s=stale_s, now=now)
    if not live:
        beats = [float(r["alive_epoch"]) for r in records]
        beat = heartbeat_epoch(root)
        if beat is not None:
            beats.append(beat)
        if not beats:
            return SpoolView(path=root, reasons=["no watcher has ever heart-beat there"])
        if now - max(beats) > stale_s:
            return SpoolView(
                path=root,
                reasons=[f"its watcher is not alive (last heartbeat {span(now - max(beats))} ago)"],
            )
        # A heartbeat with no status record says the session is up and nothing more.
        return SpoolView(path=root)
    views = _views(root, live, job, now)
    able = [v for v in views if v.can_take]
    if able:
        return min(able, key=rank)
    best = min(views, key=rank)
    best.reasons = list(dict.fromkeys(r for v in views for r in v.reasons))
    return best


def survey(
    spool: str | Path, job: JobNeeds, *, stale_s: float, now: float
) -> tuple[SpoolView, list[SpoolView]]:
    """``spool``'s view for ``job``, and its siblings' views, best first."""
    root = Path(spool)
    target = assess(root, job, stale_s=stale_s, now=now)
    others = [assess(p, job, stale_s=stale_s, now=now) for p in siblings(root) if p != root]
    return target, sorted(others, key=rank)


def pick(
    reference: str | Path, job: JobNeeds, *, stale_s: float, now: float
) -> tuple[SpoolView | None, list[SpoolView]]:
    """The spool of ``reference``'s kind (itself included) that can start ``job`` soonest.

    Returns the choice, or ``None`` when none can take the job, and every
    sibling's view so a refusal can say why each was passed over.
    """
    root = Path(reference)
    views = [assess(p, job, stale_s=stale_s, now=now) for p in siblings(root)]
    candidates = [v for v in views if v.pickable]
    if not candidates:
        return None, views
    return min(candidates, key=lambda v: rank(v, root)), views


def warning(
    job: JobNeeds, target: SpoolView, others: Iterable[SpoolView], *, queued: bool
) -> str | None:
    """One line saying why ``target`` will not start ``job`` soon, and where it could start.

    ``None`` when ``target`` can take the job. ``queued`` says whether the job
    was left queued on ``target``, so moving it means cancelling it there.
    """
    if target.can_take:
        return None
    kind = spool_kind(target.path)
    able = [v for v in others if v.pickable]
    if able:
        cancel = f"spoolctl.py cancel {job.name}, then " if queued else ""
        where = (
            f"Spools of kind {kind!r} that can take it now: "
            + ", ".join(f"{v.path.name} ({v.state()})" for v in able)
            + f". To run it there: {cancel}resubmit with --spool {able[0].path} or --pick."
        )
    else:
        where = f"No other spool of kind {kind!r} in {target.path.parent} can take it now."
    return (
        f"warning: {target.path.name} will not start {job.name} soon: "
        f"{'; '.join(target.reasons)}. {where}"
    )


def refusal(job: JobNeeds, reference: str | Path, views: Sequence[SpoolView]) -> str:
    """Why ``--pick`` found no spool for ``job``, one sibling at a time."""
    root = Path(reference)
    kind = spool_kind(root)
    if not views:
        return f"--pick: no spool of kind {kind!r} in {root.parent}"
    passed = "; ".join(
        f"{v.path.name}: "
        + (
            "; ".join(v.reasons)
            if v.reasons
            else f"it is in hand-over, its allocation ending in {span(v.remaining_s)}"
        )
        for v in views
    )
    return f"--pick: no spool of kind {kind!r} in {root.parent} can take {job.name} now ({passed})"


__all__ = [
    "ENTRY_SCRIPT",
    "JobNeeds",
    "SpoolView",
    "assess",
    "default_wall_s",
    "heartbeat_epoch",
    "is_spool",
    "pick",
    "queued_jobs",
    "rank",
    "refusal",
    "siblings",
    "span",
    "spool_kind",
    "survey",
    "warning",
    "watcher_records",
    "watcher_views",
]
