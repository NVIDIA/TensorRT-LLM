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
"""Save snapshots: copy a pass's saved pages out before anything can reuse their slots.

`TRTLLM_MC_SAVE_SNAPSHOT` (see `switches`). `wait_for_save` copies the pages
into slots of a pinned, store-registered host ring with `cudaMemcpyAsync` on
the execution stream, records the pass's event after the copies, and returns.
The save thread waits for that event and puts straight from the slots.

Why this closes R1 without blocking: every later writer of a page slot is
ordered behind the execution stream's earlier work. The next forward pass runs
on that stream. The KV cache manager moves pages on it (`KVCacheManagerV2._stream`
is the execution stream). A NIC load waits in `wait_allocation_fences` for the
GPU work that preceded the allocation. The commit that unlocks stale
sliding-window pages, and a suspension that migrates pages to a host tier,
both happen on the host after `wait_for_save` returns, so the copies are
queued before any of them. The executor pays the host time to issue the copies
(two `cudaMemcpyAsync` per MSO page) and the copy engine pays
bytes / D2H bandwidth on the stream. Waiting for the store write is gone.
"""

import threading
from typing import Callable, List, Optional, Sequence, Tuple

from .staging import HostStagingPool


class SnapshotRing:
    """Slots of a `HostStagingPool` handed out per page and returned once put."""

    def __init__(self, pool: HostStagingPool) -> None:
        self._pool = pool
        self._free: List[int] = list(range(pool.num_slots))
        self._cond = threading.Condition()
        #: Passes that had to wait for slots, and passes too large to fit.
        self.waits = 0
        self.oversized = 0

    @property
    def num_slots(self) -> int:
        return self._pool.num_slots

    @property
    def free_slots(self) -> int:
        with self._cond:
            return len(self._free)

    def try_reserve(self, count: int) -> Optional[List[int]]:
        """`count` slots, or None when that many are not free right now."""
        with self._cond:
            if count > len(self._free):
                return None
            taken, self._free = self._free[:count], self._free[count:]
            return taken

    def reserve(self, count: int, alive: Callable[[], bool], timeout: float = 1.0) -> Optional[List[int]]:
        """`count` slots, waiting for the save thread to return some.

        Returns None when `count` can never fit or `alive()` turns false while
        waiting, so the caller falls back to another way of protecting pages.
        """
        if count > self._pool.num_slots:
            self.oversized += 1
            return None
        with self._cond:
            waited = False
            while count > len(self._free):
                if not alive():
                    return None
                waited = True
                self._cond.wait(timeout)
            if waited:
                self.waits += 1
            taken, self._free = self._free[:count], self._free[count:]
            return taken

    def release(self, slots: Sequence[int]) -> None:
        with self._cond:
            self._free.extend(slots)
            self._cond.notify_all()

    def gather(
        self, slot: int, addresses: Sequence[int], sizes: Sequence[int], stream: int
    ) -> Tuple[int, int]:
        """Queue the copy of one page into `slot`; returns (slot address, bytes)."""
        return self._pool.gather(slot, addresses, sizes, stream)

    def gather_many(
        self,
        slots: Sequence[int],
        addresses: Sequence[Sequence[int]],
        sizes: Sequence[Sequence[int]],
        stream: int,
    ) -> List[Tuple[int, int]]:
        """`gather` for several pages at once (batched under `TRTLLM_MOONCAKE_STORE_BATCH_COPY`)."""
        return self._pool.gather_many(slots, addresses, sizes, stream)

    def close(self) -> None:
        self._pool.close()
