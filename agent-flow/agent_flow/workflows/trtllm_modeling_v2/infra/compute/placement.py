# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a spool's sessions can do for a job, read from the files their schedulers write.

Each watcher's record (:func:`scheduler.watcher_records`) says what it holds, what waits and
whether it is in hand-over; ``handover/acks/`` whether its successor was submitted. A spool can
take a job when a live watcher would admit it and the job can finish there (:func:`_judge`). The
gate pool passes over a spool whose best watcher is in hand-over even when the job would fit
(:attr:`SpoolView.pickable`): a session with its allocation ahead of it is the better home.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from . import scheduler as _sched


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

    #: Device count, or ``None`` for the whole node.
    gpus: int | None
    #: The ``wall`` header in seconds; ``None`` takes the session's default (0 is none).
    wall_s: float | None
    holds_gpus: bool

    @classmethod
    def from_script(cls, name: str, script: str) -> "JobNeeds":
        spec = _sched.job_spec_from_script(name, script)
        return cls(gpus=spec.gpus, wall_s=spec.wall_s, holds_gpus=spec.holds_gpus)

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
    #: Whether the watcher that would run it is in hand-over.
    in_handover: bool = False

    @property
    def can_take(self) -> bool:
        return not self.reasons

    @property
    def pickable(self) -> bool:
        """Whether the gate pool may send the job here."""
        return self.can_take and not self.in_handover


def _successor_state(spool: Path, record: Mapping[str, Any]) -> str:
    """Whether the driver has submitted the successor this watcher's allocation needs."""
    acks = _sched.SpoolPaths(spool).read_acks()
    allocation = str(record.get("slurm_job") or "")
    ack = _sched.following_ack(acks, allocation, str(record.get("session") or ""))
    if ack:
        return f"successor job {ack.get('job_id')} submitted, not up yet"
    return "no successor submitted yet"


def _judge(spool: Path, record: Mapping[str, Any], job: JobNeeds, now: float) -> SpoolView:
    """What one live watcher can do for ``job``."""
    devices = len(record.get("devices") or [])
    free = len(record.get("free") or [])
    ends = record.get("ends_at_epoch")
    remaining = float(ends) - now if ends is not None else None
    queued = sum(
        1 for entry in record.get("pending") or [] if str(entry.get("gpus", "")) not in ("", "0")
    )
    handover = record.get("handover") or {}
    default_wall = record.get("default_wall_s")
    wall = job.wall_on(_sched.DEFAULT_WALL_S if default_wall is None else float(default_wall))
    reasons = []
    if handover.get("successor"):
        reasons.append(f"it is handing over to {handover['successor']} and admits nothing more")
    elif (
        handover
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
        in_handover=bool(handover),
    )


def rank(view: SpoolView) -> tuple:
    """Sort key, best first."""
    return (
        view.in_handover,
        not view.starts_now,
        -view.free,
        view.queued,
        -(math.inf if view.remaining_s is None else view.remaining_s),
        view.path.name,
    )


def assess(spool: str | Path, job: JobNeeds, *, stale_s: float, now: float) -> SpoolView:
    """What ``spool`` can do for ``job`` now: through its best live watcher, or why not."""
    root = Path(spool)
    records = _sched.watcher_records(root)
    live = [record for record in records if now - float(record["alive_epoch"]) <= stale_s]
    if not live:
        if not records:
            return SpoolView(path=root, reasons=["no watcher has ever heart-beat there"])
        age = now - max(float(r["alive_epoch"]) for r in records)
        return SpoolView(
            path=root, reasons=[f"its watcher is not alive (last heartbeat {span(age)} ago)"]
        )
    views = [_judge(root, record, job, now) for record in live]
    able = [v for v in views if v.can_take]
    if able:
        return min(able, key=rank)
    best = min(views, key=rank)
    best.reasons = list(dict.fromkeys(r for v in views for r in v.reasons))
    return best


__all__ = [
    "JobNeeds",
    "SpoolView",
    "assess",
    "rank",
    "span",
]
