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
"""Step-ahead staging of KV cache tier copies for the KV cache manager v2 core.

The core moves pages between the GPU and lower tiers on demand: a cache that needs
GPU slots evicts least-recently-used pages to the host first, and a cache resumed
with a reused prefix onboards the prefix's host pages. Both copies run on the
core's own streams, and the requesting cache's stream waits on them. When that
stream is the execution stream, the next forward starts only once the copies end.

Both helpers below issue those copies one step early, after the current forward
has been launched, so they overlap its compute:

* ``OffloadHeadroom`` holds GPU slots in a private cache between steps. Growing
  the cache makes the core evict for it (offload); closing it hands the slots
  back to the free list. The next step's allocations take free slots before
  evicting, so their stream only waits on offloads issued a step earlier.
* ``prefetch_reuse_to_gpu`` onboards the reusable prefix of a request that has
  not been admitted yet, through a transient cache that matches the same blocks
  its admission will match.

Neither changes what any request reads: page contents and ordering stay with the
core, which tracks a ready event on every page and slot and makes each consumer
wait on it (a slot is handed out only after the copy reading it has finished; a
page is read only after the copy writing it has finished). Both act only on pages
no cache holds locked.
"""

from typing import Optional, Sequence

import torch

from tensorrt_llm.runtime.kv_cache_manager_v2 import GPU_LEVEL, ReuseScope

# Stats key of the headroom cache: far above the executor's request ids and
# below the reserved dummy ids at the top of the 64-bit range.
HEADROOM_KV_CACHE_ID = (1 << 62) + 0x4B56


def has_lower_tier(impl) -> bool:
    """Whether evicting from the GPU tier moves pages down rather than dropping them."""
    return len(impl.cache_tier_list) > 1


class OffloadHeadroom:
    """Keeps ``num_tokens`` of GPU KV slots evicted one step ahead of their use.

    ``reserve()`` (after a forward is launched) opens a private cache on a side
    stream and resizes it to ``num_tokens``; backing it evicts LRU pages, whose
    offload copies then run alongside the launched forward. Only the side stream
    waits on them. ``release()`` (before the next schedule) closes the cache: its
    slots return to the free list carrying an event recorded on the side stream
    after those waits, so a later owner waits for the offloads to finish before
    writing, and by then they normally have.

    The cache never commits a block, has no page-index buffer and no request row,
    and is excluded from the core's statistics, so it is invisible to scheduling
    and to the pool tuner. It must be released before anything that needs every
    cache suspended (pool rebalancing, shutdown).
    """

    def __init__(
        self,
        impl,
        num_tokens: int,
        device: Optional[torch.device] = None,
        kv_cache_id: int = HEADROOM_KV_CACHE_ID,
    ) -> None:
        self._impl = impl
        self.num_tokens = int(num_tokens)
        self._kv_cache_id = kv_cache_id
        self._stream = torch.cuda.Stream(device=device)
        self._kv_cache = None
        self.num_reserved = 0
        self.num_failed = 0
        impl.mark_stats_excluded(kv_cache_id)

    @property
    def stream(self) -> torch.cuda.Stream:
        return self._stream

    @property
    def is_reserved(self) -> bool:
        return self._kv_cache is not None

    def reserve(self) -> bool:
        """Evict for and hold the headroom. Returns whether it is held.

        Fails without side effects when the GPU tier cannot back it (every page
        locked, or utilization above the core's resume threshold).
        """
        if self._kv_cache is not None:
            return True
        if self.num_tokens <= 0:
            return False
        kv_cache = self._impl.create_kv_cache(None, None, id=self._kv_cache_id)
        kv_cache.stop_committing()
        kv_cache.enable_swa_scratch_reuse = False
        held = False
        try:
            held = bool(kv_cache.resume(self._stream.cuda_stream)) and bool(
                kv_cache.resize(self.num_tokens)
            )
        finally:
            if held:
                self._kv_cache = kv_cache
                self.num_reserved += 1
            else:
                kv_cache.close()
                self.num_failed += 1
        return held

    def release(self) -> None:
        """Return the held slots to the free list (no-op when nothing is held)."""
        kv_cache, self._kv_cache = self._kv_cache, None
        if kv_cache is not None:
            kv_cache.close()


def gpu_backed_reuse_tokens(impl, reuse_scope: ReuseScope, tokens: Sequence) -> int:
    """Reused tokens of ``tokens`` whose every required page is on the GPU, without claiming them.

    Pages a cache keeps below the GPU by design (sparse history) count as not
    backed, before and after any prefetch alike.
    """
    probe = impl.create_kv_cache(reuse_scope, tokens)
    try:
        levels = probe.cached_tokens_by_level
        return int(levels[GPU_LEVEL]) if len(levels) else 0
    finally:
        probe.close()


def prefetch_reuse_to_gpu(impl, reuse_scope: ReuseScope, tokens: Sequence) -> Optional[int]:
    """Onboard the reusable prefix matched by ``tokens`` to the GPU ahead of its admission.

    A transient cache claims the same radix-tree blocks a cache created with
    ``reuse_scope`` and ``tokens`` would, asks the core to bring the pages that
    cache would lock on the GPU there, and closes. The pages stay where the copy
    put them, as reusable blocks, until the request's own cache claims them or
    eviction moves them again.

    Returns how many reused tokens the GPU backed before the call (no copy is
    issued when that is all of them), or None when the GPU tier could not take
    the rest (nothing is moved then).
    """
    kv_cache = impl.create_kv_cache(reuse_scope, tokens)
    try:
        levels = kv_cache.cached_tokens_by_level
        backed = int(levels[GPU_LEVEL]) if len(levels) else 0
        if backed == sum(levels):
            return backed
        return backed if kv_cache.prefetch(GPU_LEVEL) else None
    finally:
        kv_cache.close()
