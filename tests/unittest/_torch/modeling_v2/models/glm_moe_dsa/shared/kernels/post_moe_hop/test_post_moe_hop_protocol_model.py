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
"""Explicit-state model check of the post_moe_hop Lamport epoch/clear protocol (CPU only).

compute-sanitizer has no race detector for global memory, so the safety of
the post_moe_hop exchange protocol is checked here on an abstract model, exhaustively
over every interleaving of its individual memory events for small bounds.

Model of the cross-rank exchange (``kernels.py`` / ``cute_kernel.py``): ``W``
ranks run the same sequence of calls; call ``n`` has ``rows[n]`` token rows
and uses Lamport buffer ``n % 3`` (the epoch is the call index; the last
arriving program advances it, and the next call reads it after the kernel
boundary, so it is not modelled further). Every rank owns a local copy of the
three buffers, each ``[row][slot]``; a slot is either the sentinel or the
value ``(call, publisher)``. Within call ``n`` rank ``r`` performs, in any
order (the kernel's threads are concurrent):

* ``PUB(r, n, row, d)`` for every row and destination rank ``d``: one
  multicast store into ``d``'s copy of slot ``[row][r]``. Each store is
  issued when the call starts and may land at any later time, also after the
  call ended (no completion guarantee is assumed for remote stores);
* ``CLR(r, n, row, s)`` for every row of the previous call and every slot:
  a local store of the sentinel into buffer ``(n - 1) % 3`` of ``r``'s copy;
* ``POLL(r, n, row)``: enabled once no slot of the row in ``r``'s copy of
  buffer ``n % 3`` holds the sentinel; it reads every slot.

A call ends when its clears and polls are done; the next call of that rank
starts only after that (stream order, kernel boundary). Safety: every
completed poll reads exactly ``(n, s)`` in slot ``s``, every run reaches the
end without a stuck poll, and every remote store lands. The Triton kernel's
intra-GPU row-sum exchange is the same scheme with programs as "ranks", one
shared copy, one program clearing, and a kernel boundary between calls
(``shared_memory=True``), which is checked separately.

Mutated protocols (two buffers, clearing the current buffer, no clear,
clearing the current call's row count, polling only one slot) and a timed-out
poll must be rejected, which shows the checker can find the failures it is
meant to exclude.
"""

from collections import deque
from dataclasses import dataclass, replace
from typing import Optional, Tuple

import pytest

SENT = None


@dataclass(frozen=True)
class Protocol:
    buffers: int = 3
    clear: str = "previous"  # "previous", "current" or "none"
    clear_rows: str = "previous_call"  # or "current_call"
    poll_slots: str = "all"  # or "first"
    shared_memory: bool = False  # Triton row-sum exchange: one copy, one clearing program
    allow_timeout: bool = False  # a poll may give up once (error path)


def _events(proto: Protocol, world: int, rows: Tuple[int, ...], rank: int, n: int):
    """Local (clear, poll) events of call ``n`` of ``rank``; remote stores are separate."""
    clears = []
    if proto.clear != "none" and (not proto.shared_memory or rank == 0):
        if proto.clear_rows == "previous_call":
            clear_count = rows[n - 1] if n > 0 else 0
        else:
            clear_count = rows[n]
        clears = [("CLR", r, s) for r in range(clear_count) for s in range(world)]
    polls = [("POLL", r) for r in range(rows[n])]
    return frozenset(clears + polls)


class _Model:
    def __init__(self, proto: Protocol, world: int, rows: Tuple[int, ...]):
        self.proto = proto
        self.world = world
        self.rows = rows
        self.calls = len(rows)
        self.max_rows = max(rows)
        self.copies = 1 if proto.shared_memory else world

    def initial(self):
        mem = tuple(
            SENT for _ in range(self.copies * self.proto.buffers * self.max_rows * self.world)
        )
        # per rank: (call index, remaining local events, call started)
        ranks = tuple((0, None) for _ in range(self.world))
        return (ranks, frozenset(), mem, 0)  # (ranks, pending stores, memory, timeouts used)

    def _addr(self, copy: int, buf: int, row: int, slot: int) -> int:
        return ((copy * self.proto.buffers + buf) * self.max_rows + row) * self.world + slot

    def _cur(self, n: int) -> int:
        return n % self.proto.buffers

    def _dirty(self, n: int) -> int:
        if self.proto.clear == "current":
            return self._cur(n)
        return (n - 1) % self.proto.buffers

    def successors(self, state):
        """Yield ``(label, next_state, violation)`` for every enabled event."""
        ranks, pending, mem, timeouts = state
        for rank, (n, remaining) in enumerate(ranks):
            if n >= self.calls:
                continue
            if remaining is None:
                # Shared memory models one kernel's programs: call n starts
                # only after every program ended call n - 1 (kernel boundary).
                if self.proto.shared_memory and any(m < n for m, _ in ranks):
                    continue
                new_ranks = list(ranks)
                new_ranks[rank] = (n, _events(self.proto, self.world, self.rows, rank, n))
                stores = pending | frozenset(
                    (rank, n, row, dest)
                    for row in range(self.rows[n])
                    for dest in ((0,) if self.proto.shared_memory else range(self.world))
                )
                yield f"START r{rank} n{n}", (tuple(new_ranks), stores, mem, timeouts), None
                continue
            for event in remaining:
                kind = event[0]
                if kind == "CLR":
                    _, row, slot = event
                    copy = 0 if self.proto.shared_memory else rank
                    new_mem = list(mem)
                    new_mem[self._addr(copy, self._dirty(n), row, slot)] = SENT
                    nxt = self._advance(ranks, rank, n, remaining - {event})
                    yield (
                        f"CLR r{rank} n{n} row{row} s{slot}",
                        (nxt, pending, tuple(new_mem), timeouts),
                        None,
                    )
                else:
                    _, row = event
                    copy = 0 if self.proto.shared_memory else rank
                    slots = range(self.world) if self.proto.poll_slots == "all" else range(1)
                    values = [mem[self._addr(copy, self._cur(n), row, s)] for s in slots]
                    timed_out = False
                    if any(v is SENT for v in values):
                        if not self.proto.allow_timeout or timeouts:
                            continue
                        timed_out = True
                    violation = None
                    if not timed_out:
                        read = [
                            mem[self._addr(copy, self._cur(n), row, s)] for s in range(self.world)
                        ]
                        expected = [(n, s) for s in range(self.world)]
                        if read != expected:
                            violation = (
                                f"r{rank} call {n} row {row} read {read}, expected {expected}"
                            )
                    nxt = self._advance(ranks, rank, n, remaining - {event})
                    label = f"{'TIMEOUT' if timed_out else 'POLL'} r{rank} n{n} row{row}"
                    yield label, (nxt, pending, mem, timeouts + int(timed_out)), violation
        for store in pending:
            publisher, n, row, dest = store
            new_mem = list(mem)
            new_mem[self._addr(dest, self._cur(n), row, publisher)] = (n, publisher)
            yield (
                f"LAND r{publisher} n{n} row{row} -> r{dest}",
                (ranks, pending - {store}, tuple(new_mem), timeouts),
                None,
            )

    @staticmethod
    def _advance(ranks, rank, n, remaining):
        new_ranks = list(ranks)
        new_ranks[rank] = (n + 1, None) if not remaining else (n, remaining)
        return tuple(new_ranks)

    def done(self, state) -> bool:
        ranks, pending, _, _ = state
        return not pending and all(n >= self.calls for n, _ in ranks)


@dataclass
class CheckResult:
    states: int
    violation: Optional[str]
    trace: Tuple[str, ...]


def explore(proto: Protocol, world: int, rows: Tuple[int, ...], max_states: int = 3_000_000):
    """Breadth-first exploration; returns the first violation with its event trace."""
    model = _Model(proto, world, rows)
    start = model.initial()
    parent = {start: None}
    queue = deque([start])

    def trace_to(state, last_label):
        labels = [last_label]
        while parent[state] is not None:
            state, label = parent[state]
            labels.append(label)
        return tuple(reversed(labels))

    while queue:
        state = queue.popleft()
        moved = False
        for label, nxt, violation in model.successors(state):
            moved = True
            if violation is not None:
                return CheckResult(len(parent), violation, trace_to(state, label))
            if nxt not in parent:
                parent[nxt] = (state, label)
                if len(parent) > max_states:
                    raise RuntimeError(f"state bound {max_states} exceeded")
                queue.append(nxt)
        if not moved and not model.done(state):
            return CheckResult(
                len(parent), "stuck: no event enabled before the end", trace_to(state, "<stuck>")
            )
    return CheckResult(len(parent), None, ())


GOOD = Protocol()
ROW_PATTERNS = ((1, 1, 1, 1, 1, 1, 1), (2, 1, 2, 2, 1, 1, 2))


@pytest.mark.parametrize("rows", ROW_PATTERNS, ids=("rows1", "rows_varying"))
def test_cross_rank_protocol_is_safe_two_ranks(rows):
    result = explore(GOOD, world=2, rows=rows)
    print(
        f"[post_moe_hop-model] cross-rank W=2 rows={rows}: {result.states} states, "
        f"violation={result.violation}"
    )
    assert result.violation is None, result


def test_cross_rank_protocol_is_safe_three_ranks():
    # Four calls are the fewest that reuse a buffer (call 3 reuses call 0's),
    # after the clear of call 1 and with calls 1 and 2 in flight.
    rows = (1, 1, 1, 1)
    result = explore(GOOD, world=3, rows=rows, max_states=8_000_000)
    print(
        f"[post_moe_hop-model] cross-rank W=3 rows={rows}: {result.states} states, "
        f"violation={result.violation}"
    )
    assert result.violation is None, result


@pytest.mark.parametrize(
    "rows", ((1, 1, 1, 1, 1, 1), (2, 1, 2, 1, 2, 2)), ids=("rows1", "rows_varying")
)
def test_intra_gpu_row_sum_exchange_is_safe(rows):
    proto = replace(GOOD, shared_memory=True)
    result = explore(proto, world=3, rows=rows)
    print(
        f"[post_moe_hop-model] row-sum exchange 3 programs rows={rows}: {result.states} states, "
        f"violation={result.violation}"
    )
    assert result.violation is None, result


MUTANTS = {
    "two_buffers": (replace(GOOD, buffers=2), 2, (1, 1, 1, 1, 1)),
    "clear_current_buffer": (replace(GOOD, clear="current"), 2, (1, 1, 1, 1)),
    "no_clear": (replace(GOOD, clear="none"), 2, (1, 1, 1, 1, 1)),
    "clear_current_row_count": (replace(GOOD, clear_rows="current_call"), 2, (2, 1, 1, 2, 2)),
    "poll_first_slot_only": (replace(GOOD, poll_slots="first"), 2, (1, 1, 1)),
    "timeout_without_reset": (replace(GOOD, allow_timeout=True), 2, (1, 1, 1, 1, 1)),
}


@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_mutated_protocol_is_rejected(name):
    proto, world, rows = MUTANTS[name]
    result = explore(proto, world=world, rows=rows)
    print(
        f"[post_moe_hop-model] mutant {name}: {result.states} states, violation={result.violation}; "
        f"trace tail {result.trace[-6:]}"
    )
    assert result.violation is not None, f"mutant {name} was not rejected"
