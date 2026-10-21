# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How many checks run at once, and what a running task holds of the sessions' devices.

A fixed number of checks in flight is wrong both ways. While the gate
sessions are full it queues more jobs behind each other in their spools,
where each was placed when it was submitted and cannot move to a session
that frees up later; while the work and dev sessions stand idle it leaves
their devices unused. So a check starts when a session would start its job
at once (:meth:`.spool.SpoolCommandExecutor.startable`, counted over every
spool the job may go to) and waits in this process otherwise, to be placed
when a device frees.

:func:`more_checks` turns that count into how many more checks may start;
:class:`CheckAdmission` is the gate a run's check threads pass through.

A running task holds the ``gpus`` it declares of the run's sessions (the gate
and work sessions), except a task whose every job runs on sessions of its own:
the tune campaign once ``perf.spools`` names its measuring sessions
(:func:`offsite_kinds`, :func:`held_gpus`). Its reservation would otherwise keep
devices from the gate and work sessions for the campaign's whole life while none
of its jobs ever runs there.
"""

from __future__ import annotations

import itertools
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Collection, Iterator, Mapping

from ..environment import RunEnvironment

#: Checks in flight however busy the sessions read: their jobs queue in a spool
#: and the run keeps moving even when every record is wrong.
DEFAULT_CHECK_FLOOR = 2
#: Checks in flight at most, however many devices stand free.
DEFAULT_CHECK_CAP = 32
#: Checks in flight when no session can be read (``local``, or no live watcher).
DEFAULT_CHECK_FALLBACK = 8

#: How many more checks like the one asked about would start at once on the
#: run's sessions: ``(environment, gpus, wall_s) -> int``, ``None`` when none
#: can be read.
Startable = Callable[[RunEnvironment, str | None, float | None], int | None]


def more_checks(
    startable: int | None,
    *,
    in_flight: int = 0,
    starting: int = 0,
    floor: int = DEFAULT_CHECK_FLOOR,
    cap: int = DEFAULT_CHECK_CAP,
    fallback: int = DEFAULT_CHECK_FALLBACK,
) -> int:
    """How many more checks may start now, with ``in_flight`` already running.

    Args:
        startable: How many would start at once on the sessions; ``None`` when
            none can be read, and then ``fallback`` run at once.
        in_flight: Checks running now.
        starting: Those of ``in_flight`` started too recently for their jobs to
            show in a spool yet; they still take what the sessions read as free.
        floor: Checks that may always run at once.
        cap: Checks that may run at once at most.
        fallback: Checks that may run at once when nothing can be read.
    """
    target = fallback if startable is None else in_flight + max(0, startable - starting)
    return max(0, min(cap, max(floor, target)) - in_flight)


class CheckAdmission:
    """Holds each check back until :func:`more_checks` lets it start, and counts it while it runs.

    One per run, shared by every check thread; the environments it is asked
    about differ only in width. :meth:`slot` blocks its caller until the check
    may start and holds its place while it runs.

    - A check started less than ``settle_s`` ago counts against what the
      sessions read as free: its job may not have reached a spool yet.
    - A check that has waited ``patience_s`` goes first: none that arrived
      after it starts before it, so a check that needs a whole node gets the
      next node that drains instead of losing every device to narrower ones.
    - The sessions are read at most every ``refresh_s`` per width and wall
      limit, and again as soon as a check ends.

    Args:
        startable: Reads the sessions (:data:`Startable`; the package's ``startable_checks``).
        floor: See :func:`more_checks`.
        cap: See :func:`more_checks`; also the size the check pool needs.
        fallback: See :func:`more_checks`.
        settle_s: How long a started check counts as not yet in a spool.
        patience_s: How long a check waits before none may pass it.
        poll_s: How often a waiting check reads the sessions again.
        refresh_s: How long one reading of the sessions stands.
        clock: Monotonic seconds.
    """

    def __init__(
        self,
        startable: Startable,
        *,
        floor: int = DEFAULT_CHECK_FLOOR,
        cap: int = DEFAULT_CHECK_CAP,
        fallback: int = DEFAULT_CHECK_FALLBACK,
        settle_s: float = 60.0,
        patience_s: float = 600.0,
        poll_s: float = 10.0,
        refresh_s: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._startable = startable
        self.floor = max(1, floor)
        self.cap = max(self.floor, cap)
        self.fallback = fallback
        self.settle_s = settle_s
        self.patience_s = patience_s
        self.poll_s = poll_s
        self.refresh_s = refresh_s
        self._clock = clock
        self._cond = threading.Condition()
        self._tickets = itertools.count()
        #: Checks running, by ticket: when each started.
        self._running: dict[int, float] = {}
        #: Checks waiting, by ticket: when each arrived.
        self._waiting: dict[int, float] = {}
        #: The last reading per width and wall limit: when, and what it said.
        self._readings: dict[tuple[str, float | None], tuple[float, int | None]] = {}

    @property
    def in_flight(self) -> int:
        """Checks running now."""
        with self._cond:
            return len(self._running)

    @property
    def waiting(self) -> int:
        """Checks waiting to start."""
        with self._cond:
            return len(self._waiting)

    @contextmanager
    def slot(
        self,
        environment: RunEnvironment,
        gpus: str | None = None,
        wall_s: float | None = None,
    ) -> Iterator[None]:
        """Wait until a check may start; hold its place while it runs.

        Args:
            environment: The check's environment; its world size is the check's width.
            gpus: The devices its job asks for instead (``all``: a whole node).
            wall_s: Its job's wall limit; ``None`` takes each session's default.
        """
        ticket = next(self._tickets)
        with self._cond:
            self._waiting[ticket] = self._clock()
            try:
                while not self._may_start(ticket, environment, gpus, wall_s):
                    self._cond.wait(self.poll_s)
            finally:
                del self._waiting[ticket]
            self._running[ticket] = self._clock()
            # Whoever waits behind it reads what is left now rather than at its next poll.
            self._cond.notify_all()
        try:
            yield
        finally:
            with self._cond:
                del self._running[ticket]
                self._readings.clear()
                self._cond.notify_all()

    def _may_start(
        self, ticket: int, environment: RunEnvironment, gpus: str | None, wall_s: float | None
    ) -> bool:
        now = self._clock()
        in_flight = len(self._running)
        if in_flight >= self.cap:
            return False
        overdue = [t for t, since in self._waiting.items() if now - since >= self.patience_s]
        if overdue and ticket != min(overdue):
            return False
        if in_flight < self.floor:
            return True
        starting = sum(1 for since in self._running.values() if now - since < self.settle_s)
        more = more_checks(
            self._read(environment, gpus, wall_s, now),
            in_flight=in_flight,
            starting=starting,
            floor=self.floor,
            cap=self.cap,
            fallback=self.fallback,
        )
        return more > 0

    def _read(
        self, environment: RunEnvironment, gpus: str | None, wall_s: float | None, now: float
    ) -> int | None:
        key = (str(environment.world_size if gpus is None else gpus), wall_s)
        held = self._readings.get(key)
        if held is None or now - held[0] >= self.refresh_s:
            held = (now, self._startable(environment, gpus, wall_s))
            self._readings[key] = held
        return held[1]


#: The task kind that runs the tune campaign.
TUNE_KIND = "tune"


def offsite_kinds(perf: Mapping[str, Any] | None, environment: RunEnvironment) -> frozenset[str]:
    """The task kinds none of whose jobs runs on the run's sessions: ``{"tune"}`` or none.

    A ``tune`` task measures on ``perf.spools``, and gates its keeps and runs its tuners'
    jobs on ``perf.dev_spools`` (the measuring spools without them). Once ``perf.spools``
    is set and none of those is one of the run's sessions (``environment.spool`` and
    ``environment.sessions``), its ``gpus`` are never the sessions' to hold. Without
    ``perf.spools`` the campaign measures on the run's own sessions, and its ``gpus``
    count as any task's.
    """
    if not isinstance(perf, Mapping) or not perf.get("spools"):
        return frozenset()
    measuring = [str(spool) for spool in perf["spools"]]
    its_own = {*measuring, *(str(spool) for spool in perf.get("dev_spools") or measuring)}
    sessions = {environment.spool, *(spool for _, spool, _ in environment.sessions)} - {None, ""}
    if {_resolved(s) for s in its_own} & {_resolved(str(s)) for s in sessions}:
        return frozenset()
    return frozenset({TUNE_KIND})


def held_gpus(task: Any, offsite: Collection[str] = ()) -> int:
    """The devices of the run's sessions ``task`` holds while it runs.

    Its ``gpus``; none for a task of a kind in ``offsite`` (:func:`offsite_kinds`).
    """
    return 0 if task.kind in offsite else max(0, int(task.gpus or 0))


def _resolved(spool: str) -> str:
    return str(Path(spool).expanduser().resolve())


__all__ = [
    "DEFAULT_CHECK_CAP",
    "DEFAULT_CHECK_FALLBACK",
    "DEFAULT_CHECK_FLOOR",
    "TUNE_KIND",
    "CheckAdmission",
    "Startable",
    "held_gpus",
    "more_checks",
    "offsite_kinds",
]
