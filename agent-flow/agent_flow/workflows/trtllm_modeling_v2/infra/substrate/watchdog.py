# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Watch a run's compute sessions and return the moment one needs a human, or a driver.

This returns as soon as any session:

- ``STALE``: its scheduler stopped heart-beating (the allocation is gone);
- ``ENDING``: its allocation ends within ``ending_s`` and no successor has been
  acknowledged (see ``trtllm-modeling-v2 session allocate``);
- ``QUARANTINED``: it holds devices for a process the safe-kill rule forbids
  it to stop, so they are out of service until someone looks;

and otherwise after ``timeout_s``, with a one-line summary of every session.
A spool holding a ``.retired`` file is skipped, as is one that never had a
watcher. ``trtllm-modeling-v2 session watchdog`` runs it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from . import scheduler as _sched


@dataclass(frozen=True)
class Alert:
    """One reason to stop waiting."""

    kind: str
    spool: str
    detail: str

    def render(self) -> str:
        return f"{self.kind} {self.spool} {self.detail}"


def _acknowledged(spool: Path, allocation: str, session: str) -> bool:
    """A successor to ``allocation`` is acknowledged, for its request or before it was written.

    A request on disk for another allocation is an earlier hand-over's, and
    the ack that answered it says nothing about this one.
    """
    paths = _sched.SpoolPaths(spool)
    acks = paths.read_acks()
    request = _sched.read_json(paths.handover_request)
    if request and str(request.get("slurm_job") or allocation) == allocation:
        if _sched.answering_ack(acks, request):
            return True
    return _sched.following_ack(acks, allocation, session) is not None


def check(
    spools: Iterable[str | Path],
    *,
    stale_s: float = 180.0,
    ending_s: float = 1800.0,
    now: float | None = None,
) -> list[Alert]:
    """Every alert the given spools raise right now."""
    now = time.time() if now is None else now
    alerts = []
    for spool in map(Path, spools):
        if (spool / ".retired").exists():
            continue
        paths = _sched.SpoolPaths(spool)
        watchers = []
        for path in sorted(paths.watchers.glob("*.json")):
            record = _sched.read_json(path)
            if record:
                watchers.append(record)
        if not watchers:
            continue
        newest = max(watchers, key=lambda r: float(r.get("alive_epoch", 0)))
        age = now - float(newest.get("alive_epoch", 0))
        if age > stale_s:
            running = sorted(p.name[: -len(".sh")] for p in paths.running.glob("*.sh"))
            alerts.append(
                Alert(
                    "STALE",
                    str(spool),
                    f"heartbeat {age:.0f}s old (job {newest.get('slurm_job')}); "
                    f"in running/: {' '.join(running) or 'nothing'}",
                )
            )
            continue
        ends = newest.get("ends_at_epoch")
        successor = (newest.get("handover") or {}).get("successor")
        allocation, session = str(newest.get("slurm_job") or ""), str(newest.get("session") or "")
        if (
            ends
            and float(ends) - now < ending_s
            and not successor
            and not _acknowledged(spool, allocation, session)
        ):
            alerts.append(
                Alert(
                    "ENDING",
                    str(spool),
                    f"allocation {newest.get('slurm_job')} ends in {(float(ends) - now) / 60:.0f} min "
                    f"and no successor is acknowledged",
                )
            )
        if newest.get("quarantined"):
            memory = newest.get("quarantined_memory") or {}
            held = [d for d in newest["quarantined"] if d not in memory]
            if held:
                stuck = [j["name"] for j in newest.get("running") or [] if j.get("unsafe_pids")]
                alerts.append(
                    Alert(
                        "QUARANTINED",
                        str(spool),
                        f"devices {','.join(held)} held for MPI-attached processes of "
                        f"{', '.join(stuck) or 'a finished job'}",
                    )
                )
            if memory:
                alerts.append(
                    Alert(
                        "QUARANTINED",
                        str(spool),
                        f"devices {','.join(sorted(memory))} quarantined: memory not released ("
                        + ", ".join(
                            f"{device}: {info.get('used_mib')} MiB in use since {info.get('since')}"
                            for device, info in sorted(memory.items())
                        )
                        + ")",
                    )
                )
    return alerts


def summary(spools: Iterable[str | Path], now: float | None = None) -> list[str]:
    """One line per spool: heartbeat age, running and pending work."""
    now = time.time() if now is None else now
    lines = []
    for spool in map(Path, spools):
        record = _sched.read_json(spool / "status.json") or {}
        age = f"{now - float(record['alive_epoch']):.0f}s" if record.get("alive_epoch") else "never"
        running = " ".join(j["name"] for j in record.get("running") or [])
        lines.append(
            f"{spool}: heartbeat {age}; running: {running or '-'}; "
            f"pending: {len(record.get('pending') or [])}"
        )
    return lines


def watch(
    spools: list[str | Path],
    *,
    timeout_s: float = 3000.0,
    interval_s: float = 60.0,
    stale_s: float = 180.0,
    ending_s: float = 1800.0,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[list[Alert], list[str]]:
    """Block until an alert or ``timeout_s``; return the alerts (empty on timeout) and a summary."""
    end = time.time() + timeout_s
    while True:
        alerts = check(spools, stale_s=stale_s, ending_s=ending_s)
        if alerts or time.time() >= end:
            return alerts, summary(spools)
        sleep(min(interval_s, max(0.0, end - time.time())))


__all__ = ["Alert", "check", "summary", "watch"]
