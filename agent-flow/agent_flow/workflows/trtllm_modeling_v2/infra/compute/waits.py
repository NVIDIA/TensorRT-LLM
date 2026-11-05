# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Say in the journal what a quiet run waits for.

A run blocked on spool jobs writes nothing while it waits: a gate queued for 40 minutes, a
measurement behind someone else's job and a hung orchestrator look the same from outside. So the
two places that wait on spool jobs say so here: the spool executor's blocking wait
(``executor.py``: the onboarding's gates, the substrate gate, regression jobs) inside
:func:`waiting_on`, and the tune runner's poll (``tune/measure.py``: measurements and the
campaign's gates) through :func:`still_waiting` and :func:`stopped_waiting`. Each poll of either
calls :func:`still_waiting`, which writes the event when it is due.

Once the run has waited on jobs for :data:`QUIET_S` and its journal has said nothing for as long,
one ``waiting`` event names every job it waits on, grouped by spool: queued or running and since
when, why the session's scheduler holds a queued job (its ``status.json``) and what runs on the
spool meanwhile, and when the session ends where that matters. Then at most one per :data:`QUIET_S` while the
wait lasts. Nothing is written until a run attaches its journal (:func:`journaled`): a command-line
client, or a test, waits silently.
"""

from __future__ import annotations

import contextlib
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping

from ..files import read_json
from .scheduler import SpoolPaths, watcher_records

#: The journal event a quiet wait writes.
EVENT = "waiting"
#: How long the run waits, its journal quiet, before it says on what; and the least time between
#: two such events.
QUIET_S = 15 * 60.0
#: A polled job not polled for this long is no longer waited on (its runner was set aside).
SEEN_S = 5 * 60.0
#: The most jobs one event names, and the most running jobs named beside a queued one; the rest
#: are counted.
MAX_NAMED = 12
MAX_AHEAD = 3
#: The longest hold reason quoted.
MAX_REASON = 240


@dataclass
class _Wait:
    #: When the run began waiting on the job.
    since: float
    #: When a poller last said it still waits on the job; ``None`` inside :meth:`waiting_on`.
    seen: float | None


class WaitReport:
    """The spool jobs this process waits on, and the ``waiting`` events they are owed."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._waits: dict[tuple[str, str], _Wait] = {}
        self._event: Callable[[str], None] | None = None
        self._last_event: Callable[[], float | None] = lambda: None
        self._quiet_s = QUIET_S
        self._clock: Callable[[], float] = time.time
        #: No event is due before this: the last one, or the journal's last word, plus the quiet.
        self._quiet_until = float("-inf")

    @contextlib.contextmanager
    def reported_to(
        self,
        event: Callable[[str], None],
        last_event: Callable[[], float | None] = lambda: None,
        *,
        quiet_s: float = QUIET_S,
        clock: Callable[[], float] = time.time,
    ) -> Iterator[None]:
        """Inside the block, a quiet wait is written through ``event``.

        ``last_event`` says when the journal last changed (epoch seconds, ``None`` when unknown).
        """
        with self._lock:
            saved = (self._event, self._last_event, self._quiet_s, self._clock)
            self._event, self._last_event, self._quiet_s, self._clock = (
                event,
                last_event,
                quiet_s,
                clock,
            )
            self._waits.clear()
            self._quiet_until = float("-inf")
        try:
            yield
        finally:
            with self._lock:
                self._event, self._last_event, self._quiet_s, self._clock = saved
                self._waits.clear()

    @contextlib.contextmanager
    def waiting_on(self, spool: str | Path, name: str) -> Iterator[None]:
        """Around a blocking wait: the run waits on job ``name`` in ``spool`` until the block ends."""
        with self._lock:
            self._waits[(str(spool), name)] = _Wait(self._clock(), None)
        try:
            yield
        finally:
            self.stopped_waiting(spool, name)

    def still_waiting(self, jobs: Iterable[tuple[str | Path, str]] = ()) -> None:
        """A poll: the run still waits on ``jobs`` (``(spool, job)``); writes the event when due.

        A job polled here is forgotten once no poll has named it for :data:`SEEN_S`.
        """
        with self._lock:
            now = self._clock()
            for spool, name in jobs:
                wait = self._waits.setdefault((str(spool), name), _Wait(now, now))
                wait.seen = now
            for key, wait in list(self._waits.items()):
                if wait.seen is not None and now - wait.seen > SEEN_S:
                    del self._waits[key]
            if self._event is None or not self._waits or now < self._quiet_until:
                return
            if now - min(w.since for w in self._waits.values()) < self._quiet_s:
                return
            last = self._last_event()
            if last is not None and now - last < self._quiet_s:
                self._quiet_until = last + self._quiet_s
                return
            # Claimed under the lock, so two pollers never both write the one event that is due;
            # described and written outside it, so a slow spool directory holds up no other poll.
            self._quiet_until = now + self._quiet_s
            waits, event = dict(self._waits), self._event
        try:
            event(describe(waits, now))
        except (OSError, ValueError, TypeError, KeyError):
            # A journal that cannot be written (a full disk), or a spool record that does not
            # read (a watcher's field of the wrong type), must not end the wait it describes.
            pass

    def stopped_waiting(self, spool: str | Path, name: str) -> None:
        """The run no longer waits on ``name`` in ``spool``."""
        with self._lock:
            self._waits.pop((str(spool), name), None)


#: This process's waits: one run per process (its workspace lock).
REPORT = WaitReport()
waiting_on = REPORT.waiting_on
still_waiting = REPORT.still_waiting
stopped_waiting = REPORT.stopped_waiting
reported_to = REPORT.reported_to


def journaled(journal: Any) -> contextlib.AbstractContextManager[None]:
    """A quiet wait is a :data:`EVENT` event of ``journal`` (a run's ``Journal``) inside the block."""

    def last_event() -> float | None:
        try:
            return Path(journal.path).stat().st_mtime
        except OSError:
            return None

    return REPORT.reported_to(lambda detail: journal.event(EVENT, detail), last_event)


def describe(waits: Mapping[tuple[str, str], _Wait], now: float) -> str:
    """One line for ``waits``: per spool, each job's state and since when, and the session's end."""
    spools: dict[str, list[str]] = {}
    for (spool, name), _ in sorted(waits.items(), key=lambda item: item[1].since):
        spools.setdefault(spool, []).append(name)
    oldest = now - min(wait.since for wait in waits.values())
    named, groups = 0, []
    for spool, names in spools.items():
        shown = names[: max(0, MAX_NAMED - named)]
        named += len(shown)
        if shown:
            groups.append(f"{spool}: {_spool_text(Path(spool), shown, now)}")
    more = f" (and {len(waits) - named} more)" if named < len(waits) else ""
    count = f"{len(waits)} spool job{'s' if len(waits) != 1 else ''}"
    return (
        f"{count} awaited up to {_span(oldest)}, nothing else journaled. "
        + ". ".join(groups)
        + more
    )


def _spool_text(root: Path, names: list[str], now: float) -> str:
    paths = SpoolPaths(root)
    records = sorted(watcher_records(root), key=lambda record: float(record["alive_epoch"]))
    # The newest record that lists a job says why it waits (several watchers in a hand-over).
    holds = {
        str(job.get("name")): str(job.get("reason") or "")
        for record in records
        for job in record.get("pending") or ()
        if isinstance(job, Mapping)
    }
    walls = {
        str(job.get("name")): job.get("wall_s")
        for record in records
        for job in record.get("running") or ()
        if isinstance(job, Mapping)
    }
    try:
        ends = float(records[-1]["ends_at_epoch"]) if records else None
    except (KeyError, TypeError, ValueError):
        ends = None
    running = sorted(path.stem for path in paths.running.glob("*.sh"))
    texts, ending = [], False
    for name in names:
        if (paths.running / f"{name}.sh").exists():
            meta = read_json(paths.running / f"{name}.json") or {}
            started = _number(meta.get("started_epoch"))
            wall = _number(meta.get("wall_s") or walls.get(name))
            text = f"{name} running" + (f" since {_at(started, now)}" if started else "")
            if wall:
                text += f" of a {_span(wall)} wall"
                ending |= bool(ends and started and started + wall > ends)
        elif (paths.pending / f"{name}.sh").exists():
            queued = _mtime(paths.pending / f"{name}.sh")
            text = f"{name} queued" + (f" since {_at(queued, now)}" if queued else "")
            ahead = [job for job in running if job != name]
            if ahead:
                text += " while " + ", ".join(ahead[:MAX_AHEAD])
                text += f" and {len(ahead) - MAX_AHEAD} more" if len(ahead) > MAX_AHEAD else ""
                text += " run" if len(ahead) > 1 else " runs"
            hold = holds.get(name, "")
            if hold:
                hold = hold if len(hold) <= MAX_REASON else hold[: MAX_REASON - 3] + "..."
                text += f', held: "{hold}"'
            ending = True
        else:
            text = f"{name} neither queued nor running (finishing, or set aside)"
        texts.append(text)
    if ending and ends:
        texts.append(f"session ends {_clock(ends)} (in {_span(ends - now)})")
    return "; ".join(texts)


def _number(value: Any) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _clock(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%H:%MZ")


def _at(epoch: float, now: float) -> str:
    return f"{_clock(epoch)} ({_span(now - epoch)})"


def _span(seconds: float) -> str:
    minutes = int(max(0.0, seconds) // 60)
    if minutes < 60:
        return f"{minutes}m"
    return f"{minutes // 60}h{minutes % 60:02d}m"


__all__ = [
    "EVENT",
    "QUIET_S",
    "REPORT",
    "SEEN_S",
    "WaitReport",
    "describe",
    "journaled",
    "reported_to",
    "still_waiting",
    "stopped_waiting",
    "waiting_on",
]
