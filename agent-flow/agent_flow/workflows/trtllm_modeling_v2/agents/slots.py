# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A shared agent-turn cap, acquired by ``Workflow._run_turn``.

Priority is explicit (:func:`turn_priority`): planner, critical-path tasks,
required tasks, then side work. The planner alone may use the reserved slot;
its occupancy there does not reduce task capacity. Side turns use capacity
left idle by required work, but a side turn waiting ``side_max_wait_s`` may
take the next unreserved slot.
"""

from __future__ import annotations

import contextlib
import enum
import threading
import time
from typing import Callable, Iterator


class Priority(enum.IntEnum):
    """A turn's claim on an agent slot; a lower value goes first."""

    PLANNER = 0
    CRITICAL = 1
    REQUIRED = 2
    SIDE = 3


_local = threading.local()


@contextlib.contextmanager
def turn_priority(priority: Priority) -> Iterator[None]:
    """Turns started in this thread, inside the block, take slots at ``priority``."""
    previous = getattr(_local, "priority", None)
    _local.priority = priority
    try:
        yield
    finally:
        _local.priority = previous


def current_priority() -> Priority:
    """The priority this thread's turns take slots at; ``REQUIRED`` when none was declared."""
    priority = getattr(_local, "priority", None)
    return Priority.REQUIRED if priority is None else priority


def run_at(priority: Priority, fn: Callable[..., object], *args: object) -> object:
    """``fn(*args)`` with its turns at ``priority``: for work submitted to a pool."""
    with turn_priority(priority):
        return fn(*args)


#: How often a side turn waiting for idle capacity looks again; the required
#: work's demand changes without anyone telling the gate.
_SIDE_POLL_S = 5.0
#: How long a side turn waits for idle slots before it takes the next one free.
SIDE_MAX_WAIT_S = 20 * 60.0


class AgentGate:
    """At most ``capacity`` turns at once, granted by :class:`Priority`.

    ``side_limit`` says how many side turns may run now: the slots the open
    required work leaves idle (no limit when not given). A side turn takes a
    slot anyway once it has waited ``side_max_wait_s``.
    """

    def __init__(
        self,
        capacity: int,
        side_limit: Callable[[], int] | None = None,
        side_max_wait_s: float = SIDE_MAX_WAIT_S,
    ) -> None:
        self.capacity = max(1, capacity)
        self.reserve = min(1, self.capacity - 1)
        self.side_limit = side_limit
        self.side_max_wait_s = side_max_wait_s
        self._active: dict[Priority, int] = {p: 0 for p in Priority}
        self._waiting: dict[Priority, int] = {p: 0 for p in Priority}
        self._cond = threading.Condition()

    @property
    def task_slots(self) -> int:
        """How many turns other than the planner's may run at once."""
        return self.capacity - self.reserve

    def _admissible(self, priority: Priority, waited_s: float = 0.0) -> bool:
        active = sum(self._active.values())
        if priority == Priority.PLANNER:
            return active < self.capacity
        # The planner's turns in the reserved slots are not the tasks'.
        planner = self._active[Priority.PLANNER]
        if active - min(planner, self.reserve) >= self.task_slots:
            return False
        if any(self._waiting[p] for p in Priority if p < priority):
            return False
        if (
            priority == Priority.SIDE
            and self.side_limit is not None
            and waited_s < self.side_max_wait_s
        ):
            return self._active[Priority.SIDE] < max(0, self.side_limit())
        return True

    @contextlib.contextmanager
    def slot(self, priority: Priority | None = None) -> Iterator[None]:
        """Hold a slot for one turn, at ``priority`` or the thread's (:func:`turn_priority`)."""
        priority = current_priority() if priority is None else priority
        poll = _SIDE_POLL_S if priority == Priority.SIDE else None
        since = time.monotonic()
        with self._cond:
            self._waiting[priority] += 1
            try:
                while not self._admissible(priority, time.monotonic() - since):
                    self._cond.wait(timeout=poll)
            finally:
                self._waiting[priority] -= 1
            self._active[priority] += 1
            self._cond.notify_all()
        try:
            yield
        finally:
            with self._cond:
                self._active[priority] -= 1
                self._cond.notify_all()


__all__ = ["AgentGate", "Priority", "current_priority", "run_at", "turn_priority"]
