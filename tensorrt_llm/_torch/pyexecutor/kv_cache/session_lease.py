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
"""Session leases: keep a finished context request's KV cache active for the conversation's next turn.

A multi-turn conversation sends a prompt that, most of the time, starts with the previous turn's
prompt. On a context server, the normal path closes the finished cache (unlocking every page) and
the next turn walks the radix tree and locks the same pages again. A lease skips both: the closed
turn's cache stays ACTIVE, keyed by conversation, and the next turn whose prompt starts with the
leased tokens takes the cache over and only allocates and computes its new tail.

This module holds the bookkeeping only; ``KVCacheManagerV2`` owns the caches and calls in. Every
decision depends on request tokens, conversation keys, call order and pool utilization, which are
identical on every tensor-parallel rank, so ranks agree without a collective.
"""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Hashable, Optional, Sequence

import numpy as np


@dataclass(slots=True)
class SessionLease:
    """A leased cache: the request id it was closed under, the cache, and its committed prefix."""

    request_id: int
    kv_cache: Any
    tokens: np.ndarray
    # Per-cache indirection for the page-priority callback, refreshed when the cache is taken over.
    priority_cell: Optional[list] = None


class SessionLeases:
    """Leased caches by conversation key, oldest first, with the admissions that may lease.

    ``utilization`` returns the GPU pool utilization (0 to 1, the maximum over pool groups).
    ``close`` closes a cache the way the manager closes a finished request's cache.
    """

    def __init__(
        self,
        capacity: int,
        max_utilization: float,
        utilization: Callable[[], float],
        close: Callable[[int, Any], None],
    ) -> None:
        assert capacity > 0 and 0.0 < max_utilization <= 1.0
        self.capacity = capacity
        self.max_utilization = max_utilization
        self._utilization = utilization
        self._close = close
        self._leases: "OrderedDict[Hashable, SessionLease]" = OrderedDict()
        self._seq = 0
        # conversation key -> admission number of its newest admitted request
        self._newest: dict = {}
        # request id -> (conversation key, admission number), until the request is released
        self._admitted: dict = {}
        self.counters = dict.fromkeys(
            ("admitted", "taken_over", "no_lease", "mismatch", "leased", "stale", "shed"), 0
        )

    def __len__(self) -> int:
        return len(self._leases)

    @staticmethod
    def _matches(lease: SessionLease, tokens: Sequence[int]) -> bool:
        # At least one new token: the takeover's cursor must stay below the prompt length.
        depth = len(lease.tokens)
        return depth < len(tokens) and np.array_equal(
            np.asarray(tokens[:depth], dtype=np.int32), lease.tokens
        )

    def peek(self, key: Optional[Hashable], tokens: Sequence[int]) -> Optional[int]:
        """Depth a takeover by this prompt would reuse, or None. No side effects."""
        lease = self._leases.get(key) if key is not None else None
        return len(lease.tokens) if lease is not None and self._matches(lease, tokens) else None

    def admit(
        self, key: Hashable, request_id: int, tokens: Sequence[int]
    ) -> Optional[SessionLease]:
        """Register a first-chunk admission; return the lease it takes over, or None.

        A lease whose tokens are not a prefix of the prompt is closed, and the request takes the
        normal path, which still reuses the committed blocks from the radix tree.
        """
        self.shed()
        self._seq += 1
        self._newest[key] = self._seq
        self._admitted[request_id] = (key, self._seq)
        self.counters["admitted"] += 1
        lease = self._leases.pop(key, None)
        if lease is None:
            self.counters["no_lease"] += 1
            return None
        if not self._matches(lease, tokens):
            self.counters["mismatch"] += 1
            self._close(lease.request_id, lease.kv_cache)
            return None
        self.counters["taken_over"] += 1
        return lease

    def is_admitted(self, request_id: int) -> bool:
        return request_id in self._admitted

    def release(self, request_id: int) -> Optional[Hashable]:
        """Forget an admission; return its key if it may lease, that is, it is the newest turn."""
        admitted = self._admitted.pop(request_id, None)
        if admitted is None:
            return None
        key, seq = admitted
        if self._newest.get(key) != seq:
            # A later turn of this conversation was admitted first; it holds the newer history.
            self.counters["stale"] += 1
            return None
        del self._newest[key]
        return key

    def put(self, key: Hashable, lease: SessionLease) -> None:
        """Lease a cache under ``key``, replacing (and closing) an older lease of the same key."""
        old = self._leases.pop(key, None)
        if old is not None:
            self._close(old.request_id, old.kv_cache)
        self._leases[key] = lease
        self.counters["leased"] += 1
        self.shed()

    def close_oldest(self) -> bool:
        if not self._leases:
            return False
        _, lease = self._leases.popitem(last=False)
        self.counters["shed"] += 1
        self._close(lease.request_id, lease.kv_cache)
        return True

    def shed(self) -> None:
        """Close leases, oldest first, while over capacity or above the utilization bound."""
        while self._leases and (
            len(self._leases) > self.capacity or self._utilization() > self.max_utilization
        ):
            self.close_oldest()

    def close_all(self) -> None:
        while self._leases:
            _, lease = self._leases.popitem(last=False)
            self._close(lease.request_id, lease.kv_cache)
        self._newest.clear()
        self._admitted.clear()
