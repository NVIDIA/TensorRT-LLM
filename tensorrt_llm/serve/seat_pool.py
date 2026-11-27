# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The queue of a pooled generation router (``router.PooledLoadBalancingRouter``).

Every generation server has ``seats`` seats; a request holds one from its placement until it
finishes. A request that finds no free seat waits here, and each freed seat goes to the first
waiter in order that may use that server: the highest request ``priority`` first (the
orchestrator's predicted output length with ``output_length_prediction``), ties in the order
they joined this queue. With ``max_wait_s``, a request that has waited that long since its
arrival goes ahead of every request that has not, earliest arrival first (``(arrival, seq)``,
the order of the generation server's ``AgingPriorityWaitingQueue``).

A request that comes from a finished context phase carries ``orig_elapsed_s``, its time since it
reached the orchestrator: its wait counts from that arrival. With ``kv_transfer_timeout_s`` (the
context servers' ``kv_transfer_timeout_ms``) every waiter fails once it has been in this queue
that long, less ``KV_DEADLINE_MARGIN`` of it. A C++-transceiver context server counts that timeout
from its KV send, a little before the request joins this queue, so it has dropped the transfer by
then and the margin keeps a seat granted at the deadline from reaching a cancelled send; a
Python-transceiver one holds a send its generation peer has not asked for up to three timeouts
from the send, so there this bound is conservative (with ``generation_first`` the context server
waits the same timeout for the generation side).

Waiters sit in heaps per excluded server, so a placement costs O(E log W) for E distinct
excluded servers and W waiters, and expiry pops a deadline heap.
"""

import asyncio
import heapq
import itertools
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from tensorrt_llm.executor.request import DEFAULT_REQUEST_PRIORITY
from tensorrt_llm.logger import logger

STATS_LOG_PERIOD_S = 30.0
# A waiter fails this fraction of the KV transfer timeout before it (see the module docstring).
KV_DEADLINE_MARGIN = 0.05
# A heap is rebuilt from its live waiters once its removed entries outnumber them by this much.
_COMPACT_SLACK = 64


class NoGenerationSeatError(ValueError):
    """A generation request got no seat: past its KV transfer deadline, or the coordinator stopped.

    The server is overloaded rather than faulty; the disaggregated server answers 503.
    """


def priority_or_default(priority: Any) -> float:
    """A request priority as a number in [0, 1]; anything else is the default priority."""
    if isinstance(priority, (int, float)) and not isinstance(priority, bool):
        if math.isfinite(priority) and 0.0 <= priority <= 1.0:
            return float(priority)
    return DEFAULT_REQUEST_PRIORITY


@dataclass(eq=False)
class Waiter:
    """One request waiting for a seat; ``holders`` counts the callers waiting on its placement."""

    rid: int
    seq: int
    arrival: float
    priority: float
    exclude: Optional[str]
    future: Optional["asyncio.Future[str]"] = None
    holders: int = 0
    arriving: bool = True
    # When the context server's KV transfer times out; None: no bound.
    deadline: Optional[float] = None
    # When it joined this queue (``arrival`` may be earlier, at the orchestrator).
    entered: float = 0.0
    # False once it left the queue; its heap entries are then skipped.
    queued: bool = True


class _Bucket:
    """The waiters that exclude one server: by (-priority, seq) and by (arrival, seq)."""

    __slots__ = ("by_priority", "by_arrival", "live")

    def __init__(self) -> None:
        self.by_priority: List[Tuple[float, int, Waiter]] = []
        self.by_arrival: List[Tuple[float, int, Waiter]] = []
        self.live = 0

    def push(self, waiter: Waiter) -> None:
        heapq.heappush(self.by_priority, (-waiter.priority, waiter.seq, waiter))
        heapq.heappush(self.by_arrival, (waiter.arrival, waiter.seq, waiter))
        self.live += 1

    def discard(self) -> None:
        """One of its waiters left; drop the entries of departed waiters once they dominate.

        Either heap can hold them: a granted waiter leaves ``by_priority`` at its head, while
        ``by_arrival`` is popped only with a wait cap.
        """
        self.live -= 1
        if max(len(self.by_priority), len(self.by_arrival)) > 2 * self.live + _COMPACT_SLACK:
            self.by_priority = [e for e in self.by_priority if e[2].queued]
            heapq.heapify(self.by_priority)
            self.by_arrival = [e for e in self.by_arrival if e[2].queued]
            heapq.heapify(self.by_arrival)

    @staticmethod
    def _top(heap: List[Tuple[float, int, Waiter]]) -> Optional[Waiter]:
        while heap and not heap[0][2].queued:
            heapq.heappop(heap)
        return heap[0][2] if heap else None

    def first(self, now: float, max_wait_s: Optional[float]) -> Optional[Waiter]:
        """Its first waiter: the earliest arrival if that one is aged, else the highest priority."""
        if max_wait_s is not None:
            oldest = self._top(self.by_arrival)
            if oldest is not None and now - oldest.arrival >= max_wait_s:
                return oldest
        return self._top(self.by_priority)


def _pct(values: List[float], q: float) -> float:
    return values[min(len(values) - 1, int(q * len(values)))]


class SeatPool:
    """Waiters, their order and placement counters. Not thread safe: its router calls it under one lock."""

    def __init__(
        self,
        seats: int,
        max_wait_s: Optional[float] = None,
        clock: Callable[[], float] = time.monotonic,
        kv_transfer_timeout_s: Optional[float] = None,
    ) -> None:
        if isinstance(seats, bool) or not isinstance(seats, int) or seats < 1:
            raise ValueError(f"max_requests_per_server={seats!r} must be an integer >= 1")
        if max_wait_s is not None and not (math.isfinite(max_wait_s) and max_wait_s > 0.0):
            raise ValueError(f"max_queue_wait_s={max_wait_s!r} must be a positive number or None")
        self.seats = seats
        self.max_wait_s = max_wait_s
        self.kv_transfer_timeout_s = kv_transfer_timeout_s
        self._clock = clock
        # rid -> waiter, in the order they joined (a waiter is never re-inserted).
        self._waiters: Dict[int, Waiter] = {}
        self._buckets: Dict[Optional[str], _Bucket] = {}
        self._deadlines: List[Tuple[float, int, Waiter]] = []
        self._seq = itertools.count()
        self._pick_now: Optional[float] = None
        self.placed = 0
        self.placed_at_once = 0
        self.reordered = 0
        self.aged = 0
        self.dropped = 0
        self.expired = 0
        self._win_start = clock()
        self._win_placed = 0
        self._win_waits: List[float] = []

    @property
    def kv_transfer_timeout_s(self) -> Optional[float]:
        return self._kv_transfer_timeout_s

    @kv_transfer_timeout_s.setter
    def kv_transfer_timeout_s(self, value: Optional[float]) -> None:
        """Bounds the waiters that join from now on; those already waiting keep their deadline."""
        if value is not None and not (math.isfinite(value) and value > 0.0):
            raise ValueError(f"kv_transfer_timeout_s={value!r} must be a positive number or None")
        self._kv_transfer_timeout_s = value

    def describe(self, num_servers: int) -> str:
        cap = (
            f"a request that has waited {self.max_wait_s:g} s goes ahead, oldest first"
            if self.max_wait_s is not None
            else "no wait cap"
        )
        bound = (
            f"a request fails after {self.kv_transfer_timeout_s:g} s here (the context servers' "
            f"KV transfer timeout)"
            if self.kv_transfer_timeout_s is not None
            else "no KV transfer bound"
        )
        return (
            f"generation requests wait in one queue for a free seat ({self.seats} per server, "
            f"{num_servers} servers), highest request priority first, ties in arrival order; "
            f"{cap}; {bound}"
        )

    def __len__(self) -> int:
        return len(self._waiters)

    def waiter(self, rid: int) -> Optional[Waiter]:
        return self._waiters.get(rid)

    def waiters(self) -> List[Waiter]:
        """The waiting requests, in the order they joined."""
        return list(self._waiters.values())

    def add(
        self,
        rid: int,
        priority: Any,
        exclude: Optional[str],
        future: Optional["asyncio.Future[str]"],
        orig_elapsed_s: Any = None,
    ) -> Waiter:
        """A new waiter: it arrived ``orig_elapsed_s`` ago (a context phase has run), else now."""
        if rid in self._waiters:
            raise ValueError(f"request {rid} is already waiting")
        now = self._clock()
        arrival = now
        if isinstance(orig_elapsed_s, (int, float)) and not isinstance(orig_elapsed_s, bool):
            if math.isfinite(orig_elapsed_s):
                arrival = now - max(0.0, float(orig_elapsed_s))
        deadline = None
        if self.kv_transfer_timeout_s is not None:
            deadline = now + self.kv_transfer_timeout_s * (1.0 - KV_DEADLINE_MARGIN)
        waiter = Waiter(
            rid=rid,
            seq=next(self._seq),
            arrival=arrival,
            priority=priority_or_default(priority),
            exclude=exclude,
            future=future,
            deadline=deadline,
            entered=now,
        )
        self._waiters[rid] = waiter
        bucket = self._buckets.get(exclude)
        if bucket is None:
            bucket = self._buckets[exclude] = _Bucket()
        bucket.push(waiter)
        if deadline is not None:
            heapq.heappush(self._deadlines, (deadline, waiter.seq, waiter))
        return waiter

    def _discard(self, waiter: Waiter) -> None:
        """Take ``waiter`` (still queued) out of the queue."""
        waiter.queued = False
        del self._waiters[waiter.rid]
        bucket = self._buckets[waiter.exclude]
        bucket.discard()
        if not bucket.live:
            del self._buckets[waiter.exclude]
        if not self._waiters:
            self._deadlines.clear()

    def remove(self, rid: int) -> Optional[Waiter]:
        """Drop a waiter that finished or was cancelled before its placement."""
        waiter = self._waiters.get(rid)
        if waiter is not None:
            self._discard(waiter)
            self.dropped += 1
        return waiter

    def expire(self, waiter: Optional[Waiter] = None) -> List[Waiter]:
        """Remove and return the waiters past their deadline, and ``waiter`` if still waiting."""
        expired = []
        if self._deadlines:
            now = self._clock()
            while self._deadlines and self._deadlines[0][0] <= now:
                w = heapq.heappop(self._deadlines)[2]
                if w.queued:
                    self._discard(w)
                    expired.append(w)
        if waiter is not None and waiter.queued:
            self._discard(waiter)
            expired.append(waiter)
        self.expired += len(expired)
        return expired

    def remaining_s(self, waiter: Waiter) -> Optional[float]:
        """Seconds until ``waiter``'s deadline (>= 0), or None without one."""
        if waiter.deadline is None:
            return None
        return max(0.0, waiter.deadline - self._clock())

    def expiry_message(self, waiter: Waiter) -> str:
        return (
            f"generation request {waiter.rid} found no free seat within the context servers' KV "
            f"transfer timeout ({(waiter.deadline - waiter.entered) / (1.0 - KV_DEADLINE_MARGIN):g} s "
            f"less a {KV_DEADLINE_MARGIN:.0%} margin, after it joined the queue; "
            f"{self._clock() - waiter.arrival:.1f} s since it arrived); a context server with the "
            f"C++ transceiver has dropped the KV cache transfer by then"
        )

    def _is_aged(self, waiter: Waiter, now: float) -> bool:
        return self.max_wait_s is not None and now - waiter.arrival >= self.max_wait_s

    def sort_key(self, waiter: Waiter, now: float) -> Tuple:
        """Smaller goes first: aged waiters by (arrival, seq), then the rest by (-priority, seq)."""
        if self._is_aged(waiter, now):
            return (0, waiter.arrival, waiter.seq)
        return (1, -waiter.priority, waiter.seq)

    def pick(self, free_servers: Sequence[str]) -> Optional[Tuple[Waiter, List[str]]]:
        """The first waiter in order that may use one of ``free_servers``, with those servers."""
        if not self._waiters or not free_servers:
            return None
        now = self._pick_now = self._clock()
        best, best_key, best_servers = None, None, None
        for exclude, bucket in self._buckets.items():
            servers = (
                list(free_servers) if exclude is None else [s for s in free_servers if s != exclude]
            )
            if not servers:
                continue
            waiter = bucket.first(now, self.max_wait_s)
            if waiter is None:
                continue
            key = self.sort_key(waiter, now)
            if best_key is None or key < best_key:
                best, best_key, best_servers = waiter, key, servers
        if best is None:
            return None
        return best, best_servers

    def record_grant(self, waiter: Waiter) -> None:
        """``waiter`` (from :meth:`pick`) took a seat; it leaves the queue."""
        now = self._pick_now if self._pick_now is not None else self._clock()
        self._pick_now = None
        first_seq = next(iter(self._waiters.values())).seq
        if waiter.queued:
            self._discard(waiter)
        self.placed += 1
        self._win_placed += 1
        self.reordered += waiter.seq != first_seq
        self.aged += self._is_aged(waiter, now)
        self.placed_at_once += waiter.arriving
        self._win_waits.append(max(0.0, now - waiter.entered))

    def maybe_log_stats(self, seats_used: int, seats_total: int) -> Optional[str]:
        """The stats line every ``STATS_LOG_PERIOD_S`` with a placement (also returned)."""
        now = self._clock()
        if now - self._win_start < STATS_LOG_PERIOD_S or not self._win_placed:
            return None
        waits = sorted(self._win_waits)
        line = (
            f"[seat_pool] {self._win_placed} placed in {now - self._win_start:.0f} s, queue wait "
            f"p50 / p90 / max {_pct(waits, 0.5):.2f} / {_pct(waits, 0.9):.2f} / {waits[-1]:.2f} s; "
            f"{len(self._waiters)} waiting; seats in use {seats_used} / {seats_total}; total placed "
            f"{self.placed} ({self.placed_at_once} at once, {self.reordered} ahead of an earlier "
            f"arrival, {self.aged} past the cap), dropped {self.dropped}, "
            f"{self.expired} past the KV transfer timeout"
        )
        self._win_start = now
        self._win_placed = 0
        self._win_waits = []
        logger.info(line)
        return line
