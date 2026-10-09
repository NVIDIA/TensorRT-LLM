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
"""Staging sizing and slot bookkeeping, without memory. A lease takes all its slots at once, one
contiguous run per pool group, or none; leases waiting in line are served strictly in order."""

from __future__ import annotations

import bisect
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

import numpy as np

from ._types import StagingOptions, _integer

if TYPE_CHECKING:
    from ._layout import ManagerLayout


def fetch_rows(layout: ManagerLayout, fetch_tokens: int) -> dict[int, int]:
    """Rows per device pool group that any whole-block range of at most ``fetch_tokens`` tokens can
    need: ``ceil(fetch_tokens / tpb)`` per layer group, with a window at most its sink blocks plus
    ``ceil(window / tpb)``. ``ValueError`` if ``fetch_tokens < 1``, ``TypeError`` if not an int."""
    tpb = int(layout.tokens_per_block)
    target = -(-_integer("fetch_tokens", fetch_tokens) // tpb)
    rows: dict[int, int] = {}
    for lg, group in enumerate(layout.pool_group_of):
        blocks = target
        window = layout.windows[lg]
        # At a history on a block boundary, where every staging range ends, a window of W tokens
        # reads at most ceil((W - 1) / tpb) <= ceil(W / tpb) blocks besides its sinks.
        if window is not None:
            blocks = min(target, int(layout.sink_blocks[lg]) + -(-int(window) // tpb))
        rows[int(group)] = rows.get(int(group), 0) + blocks
    return rows


def slot_counts(layout: ManagerLayout, options: StagingOptions) -> dict[int, int]:
    """Slots per device pool group: ``max_fetches`` fetches, or ``max_bytes`` split by each group's
    share of one fetch and rounded down to whole slots. ``ValueError`` if ``max_bytes`` is below
    one fetch, so every group holds at least one fetch's rows."""
    rows = fetch_rows(layout, options.fetch_tokens)
    weight = {g: rows[g] * int(layout.page_bytes[g]) for g in rows}
    one = sum(weight.values())
    if one < 1:
        raise ValueError("the KV cache manager has no pages to stage")
    nbytes = options.max_fetches * one
    if options.max_bytes is not None:
        if options.max_bytes < one:
            raise ValueError(
                f"max_bytes of {options.max_bytes} is below one fetch of {options.fetch_tokens} "
                f"tokens, which needs {one} bytes"
            )
        nbytes = min(nbytes, options.max_bytes)
    # Each group's share of the bytes is its share of one fetch, so a cap keeps every group's
    # rows of one fetch; uncapped, a group gets exactly max_fetches times its rows.
    return {g: (nbytes * weight[g] // one) // int(layout.page_bytes[g]) for g in rows}


@dataclass(frozen=True)
class Runs:
    """The slots one lease holds: per pool group, ``count`` slots from ``start``."""

    runs: Mapping[int, tuple[int, int]]

    def slots(self, group: int) -> np.ndarray:
        """``int64`` slot indices of ``group``'s run, empty when the lease holds none there."""
        start, count = self.runs.get(group, (0, 0))
        return np.arange(start, start + count, dtype=np.int64)


class _RunAllocator:
    """First-fit allocator of contiguous slot runs with coalescing frees."""

    def __init__(self, num_slots: int) -> None:
        self.num_slots = num_slots
        self._free: list[tuple[int, int]] = [(0, num_slots)] if num_slots else []

    def find(self, count: int) -> int | None:
        for start, length in self._free:
            if length >= count:
                return start
        return None

    def take(self, start: int, count: int) -> None:
        for i, (s, length) in enumerate(self._free):
            if s <= start and start + count <= s + length:
                pieces = []
                if start > s:
                    pieces.append((s, start - s))
                if start + count < s + length:
                    pieces.append((start + count, s + length - start - count))
                self._free[i : i + 1] = pieces
                return
        raise ValueError(f"slots [{start}, {start + count}) are not free")

    def overlaps_free(self, start: int, count: int) -> bool:
        """Some slot of ``[start, start + count)`` is free already."""
        return any(s < start + count and start < s + length for s, length in self._free)

    def give(self, start: int, count: int) -> None:
        """Return a run ``Slots.give`` checked: inside the pool, and no slot of it free."""
        if count == 0:
            return
        i = bisect.bisect_left(self._free, (start, 0))
        self._free.insert(i, (start, count))
        merged: list[tuple[int, int]] = []
        for s, length in self._free:
            if merged and merged[-1][0] + merged[-1][1] == s:
                merged[-1] = (merged[-1][0], merged[-1][1] + length)
            else:
                merged.append((s, length))
        self._free = merged

    @property
    def free_slots(self) -> int:
        return sum(length for _, length in self._free)


class Slots:
    """First-come-first-served slot queue over every pool group. Not thread-safe: like the manager,
    one thread at a time uses it."""

    def __init__(self, num_slots: Mapping[int, int]) -> None:
        self._num_slots = {int(g): int(count) for g, count in num_slots.items()}
        self._alloc = {g: _RunAllocator(c) for g, c in self._num_slots.items()}
        self._waiting: OrderedDict[int, dict[int, int]] = OrderedDict()
        self._next_ticket = 0

    def num_slots(self, group: int) -> int:
        """Slots ``group`` holds in all."""
        return self._num_slots[group]

    def check(self, counts: Mapping[int, int]) -> None:
        """``ValueError`` if ``counts`` can never be granted: an unknown group, or more slots than a
        group holds. Queues nothing."""
        self._wanted(counts)

    def ask(self, counts: Mapping[int, int]) -> int:
        """Queue a request for ``counts`` slots per group and return its ticket; ``ValueError`` as
        ``check``."""
        wanted = self._wanted(counts)
        ticket = self._next_ticket
        self._next_ticket += 1
        self._waiting[ticket] = wanted
        return ticket

    def take(self, ticket: int) -> Runs | None:
        """Grant ``ticket`` if it is first in line and every group has a free run; else ``None``.
        ``KeyError`` for a ticket that is not waiting."""
        if ticket not in self._waiting:
            raise KeyError(f"staging ticket {ticket} is not waiting")
        if next(iter(self._waiting)) != ticket:
            return None
        wanted = self._waiting[ticket]
        starts = {}
        for g, c in wanted.items():
            start = self._alloc[g].find(c)
            if start is None:
                return None
            starts[g] = start
        for g, c in wanted.items():
            self._alloc[g].take(starts[g], c)
        del self._waiting[ticket]
        return Runs({g: (starts[g], c) for g, c in wanted.items()})

    def cancel(self, ticket: int) -> None:
        """Withdraw a waiting ticket; an unknown or granted ticket is ignored."""
        self._waiting.pop(ticket, None)

    def give(self, runs: Runs) -> None:
        """Return one lease's slots, all or none. ``ValueError`` for a run not held (freed twice or
        never granted)."""
        # Every run is checked before any is returned, so a bad one returns none.
        for g, (start, count) in runs.runs.items():
            alloc = self._alloc.get(g)
            if alloc is None:
                raise ValueError(f"no staging slots for pool group {g}")
            if count and (start < 0 or count < 0 or start + count > alloc.num_slots):
                raise ValueError(f"slots [{start}, {start + count}) are outside pool group {g}")
            if count and alloc.overlaps_free(start, count):
                raise ValueError(f"slots [{start}, {start + count}) of pool group {g} freed twice")
        for g, (start, count) in runs.runs.items():
            self._alloc[g].give(start, count)

    @property
    def num_waiting(self) -> int:
        """Tickets waiting in line."""
        return len(self._waiting)

    def free_slots(self, group: int) -> int:
        """Slots of ``group`` not held by any lease."""
        return self._alloc[group].free_slots

    def _wanted(self, counts: Mapping[int, int]) -> dict[int, int]:
        """``counts`` without zero entries; ``ValueError`` if it can never be granted."""
        wanted = {}
        for g, c in counts.items():
            g, c = int(g), int(c)
            if c < 0:
                raise ValueError(f"{c} staging slots asked of pool group {g}")
            if c == 0:
                continue
            if g not in self._num_slots:
                raise ValueError(f"no staging slots for pool group {g}")
            if c > self._num_slots[g]:
                raise ValueError(
                    f"{c} staging slots asked of pool group {g}, which has {self._num_slots[g]}"
                )
            wanted[g] = c
        return wanted
