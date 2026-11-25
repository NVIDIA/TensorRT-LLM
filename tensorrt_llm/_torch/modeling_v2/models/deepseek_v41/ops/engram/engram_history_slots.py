# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which row of a target's Engram look-back ring each request owns; a setup helper (host bookkeeping, no kernel).

The targets hash an n-gram's look-back tokens from a device ring `hist [rows + 1, ring]` (int32 compressed ids at
`position % ring`; the last row is the dummy that padded and unknown rows write) and the step's request -> row map
`req_slot [max_reqs]` (`engram_lookback_hash`). Row ownership is stock's (`EngramHashProvider._reserve_history_rows` /
`release_request_state`): a request takes a row the first step it is scheduled and keeps it, scheduled or not, until the
engine releases it (completion, cancellation, a warm-up dummy); a released row is reused. Stock grows its history rows
on demand (doubling); here the rows are sized up front to every request the rank can hold (the KV cache manager's
`max_batch_size`, which bounds the executor's active requests), at the first step, ahead of any CUDA-graph capture (the
graphs read `hist` by address), so any number of requests scheduled in one step each have a row. Should more requests
than rows be live anyway (a caller that never releases), a row of a request absent from the step is taken back (least
recently scheduled first), and past that the ring doubles, outside graph capture only.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterable, Optional, Tuple

import torch

# Rows the ring has before the engine says how many requests a rank holds.
MIN_ROWS = 512


class EngramHistorySlots:
    """`hist` / `req_slot` device buffers and the request -> row map; `dummy` is the padding row index."""

    def __init__(self, ring: int, pad_id: int, max_reqs: int, device, rows: int = MIN_ROWS):
        self.ring, self.pad_id, self.device = ring, int(pad_id), torch.device(device)
        self.rows = int(rows)
        self.hist = torch.full(
            (self.rows + 1, ring), self.pad_id, dtype=torch.int32, device=self.device
        )
        self.req_slot = torch.full((max_reqs,), self.rows, dtype=torch.int32, device=self.device)
        self._row_of: "OrderedDict[int, int]" = OrderedDict()  # least recently scheduled first
        self._free: list = []
        self._next = 0  # rows below this were handed out at least once

    @property
    def dummy(self) -> int:
        return self.rows

    def row_of(self, request_id: int) -> Optional[int]:
        return self._row_of.get(request_id)

    def __len__(self) -> int:
        return len(self._row_of)

    def reserve(self, requests: int) -> None:
        """At least `requests` rows (the engine's per-rank request capacity); grows only, outside graph capture."""
        if requests > self.rows:
            self._grow(requests)

    def _grow(self, need: int) -> None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"Engram history: {need} requests need a row but the ring has {self.rows}; it cannot grow during "
                "CUDA-graph capture"
            )
        rows = max(self.rows, 1)
        while rows < need:
            rows *= 2
        old, old_dummy = self.hist, self.rows
        self.hist = torch.full(
            (rows + 1, self.ring), self.pad_id, dtype=torch.int32, device=self.device
        )
        self.hist[:old_dummy].copy_(old[:old_dummy])
        self.req_slot.masked_fill_(self.req_slot == old_dummy, rows)
        self.rows = rows

    def acquire(self, request_id: int, live: Iterable[int]) -> Tuple[int, bool]:
        """-> (the request's row, whether it is new to the ring: the caller seeds its look-back). `live`: the ids
        scheduled in this step (their rows are never taken back)."""
        row = self._row_of.get(request_id)
        if row is not None:
            self._row_of.move_to_end(request_id)
            return row, False
        if self._free:
            row = self._free.pop()
        elif self._next < self.rows:
            row, self._next = self._next, self._next + 1
        else:
            live = live if isinstance(live, (set, frozenset)) else set(live)
            victim = next((k for k in self._row_of if k not in live), None)
            if victim is not None:
                row = self._row_of.pop(victim)
            else:
                self._grow(self.rows + 1)
                row, self._next = self._next, self._next + 1
        self._row_of[request_id] = row
        return row, True

    def release(self, request_id: int) -> None:
        """The engine is done with the request: its row is free for the next one (stock's `release_request_state`)."""
        row = self._row_of.pop(request_id, None)
        if row is not None:
            self._free.append(row)
