# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Hashable
from threading import RLock
from typing import TYPE_CHECKING, Generic, NamedTuple, TypeVar

import torch

from tensorrt_llm.logger import logger

from .resource_manager import BaseResourceManager

if TYPE_CHECKING:
    from .llm_request import LlmRequest

K = TypeVar("K", bound=Hashable)

RequestId = int


class _Entry(NamedTuple):
    value: torch.Tensor
    size_bytes: int


class MultimodalEncoderCacheStats(NamedTuple):
    max_bytes: int
    current_bytes: int
    held_bytes: int
    item_count: int
    held_count: int
    hits: int
    misses: int
    adoptions: int
    dedup_adoptions: int
    evictions: int


class MultimodalEncoderCacheManager(BaseResourceManager, Generic[K]):
    """Budgeted single storage for multimodal encoder outputs.

    Encoder outputs of item-scheduling models live *only* here: requests
    reference entries by key, and the byte budget therefore bounds the
    total GPU memory resident encoder outputs can occupy. Cross-request
    reuse is a side effect of the storage being content-addressed, not a
    separate cache.

    Registered as a ``BaseResourceManager`` so request teardown flows
    through the executor's standard ``free_resources`` funnel; admission is
    NOT capacity-gated here (the MM item scheduler enforces the byte budget
    via allocate-before-compute), so the scheduler-facing resource counts
    are zero.

    Semantics:

    - **Holding.** An entry is held while at least one request references
      it. Held entries are never evicted; a hold is the request's guarantee
      that the embedding survives until its prefill consumes it.
    - **Allocate-before-compute.** Callers (the MM item scheduler) must check
      `can_allocate()` before encoding an item. `adopt()` evicting only
      zero-reference entries relies on that contract and raises if it cannot
      make room.
    - **Adopt, not copy.** `adopt()` takes ownership of the caller's tensor
      without cloning when it exclusively owns its storage; views into a
      larger allocation are materialized so accounted bytes always equal
      physical residency. If the key already exists, the existing entry is
      held and returned instead — concurrent duplicate encodes of the same
      content collapse to one resident copy.
    - **Zero-ref residency.** Entries whose last hold was released stay
      resident in LRU order for reuse and are reclaimed only when a later
      allocation needs the space.

    Returned tensors alias manager-owned storage and must be treated as
    immutable by callers.

    Args:
        max_bytes: Maximum logical tensor bytes held by this manager.
        name: Short label used in debug log messages.
    """

    def __init__(
        self,
        max_bytes: int,
        *,
        name: str = "mm_encoder_cache_manager",
    ) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")

        self._max_bytes = max_bytes
        self._name = name
        self._current_bytes = 0
        self._held_bytes = 0
        # Recency order over all entries; eviction walks from the LRU end
        # skipping held entries.
        self._entries: OrderedDict[K, _Entry] = OrderedDict()
        self._key_holders: dict[K, set[RequestId]] = {}
        self._request_keys: dict[RequestId, set[K]] = {}
        self._lock = RLock()
        self._hits = 0
        self._misses = 0
        self._adoptions = 0
        self._dedup_adoptions = 0
        self._evictions = 0

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    @property
    def current_bytes(self) -> int:
        with self._lock:
            return self._current_bytes

    @property
    def held_bytes(self) -> int:
        with self._lock:
            return self._held_bytes

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def available_bytes(self, *, reserved_bytes: int = 0) -> int:
        """Bytes obtainable for new allocations: free plus zero-ref
        reclaimable space, minus the caller's outstanding reservation."""
        with self._lock:
            return max(0, self._max_bytes - self._held_bytes - reserved_bytes)

    def can_allocate(self, nbytes: int, *, reserved_bytes: int = 0) -> bool:
        """Whether `nbytes` can be admitted without evicting held entries.

        `reserved_bytes` lets the scheduler set space aside for the
        head-of-line request's remaining items before admitting items of
        later requests (deadlock avoidance)."""
        return nbytes <= self.available_bytes(reserved_bytes=reserved_bytes)

    def contains(self, key: K) -> bool:
        with self._lock:
            return key in self._entries

    def get_and_hold(self, key: K, request_id: RequestId) -> torch.Tensor | None:
        """Return the entry for `key` held on behalf of `request_id`,
        or `None` on miss. The hit is promoted to most-recently-used."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._misses += 1
                return None
            self._hits += 1
            self._entries.move_to_end(key)
            self._hold(key, entry, request_id)
            return entry.value

    def adopt(self, key: K, value: torch.Tensor, request_id: RequestId) -> torch.Tensor:
        """Store `value` (taking ownership) held for `request_id`.

        Tensors that exclusively own their storage are stored without a
        copy. Views into a larger backing allocation (e.g. `torch.split`
        outputs of a grouped encoder forward) are materialized first:
        entries from one batch would otherwise share storage, so evicting
        one while a sibling survives frees accounted bytes without freeing
        physical memory, breaking the budget's residency guarantee.

        If `key` is already resident the existing tensor is held and
        returned and `value` is dropped. Evicts zero-reference entries as
        needed; raises `RuntimeError` when space cannot be made — callers
        must have passed `can_allocate()` first (allocate-before-compute).
        """
        size_bytes = value.numel() * value.element_size()
        if value.untyped_storage().nbytes() != size_bytes:
            value = value.clone()

        with self._lock:
            existing = self._entries.get(key)
            if existing is not None:
                self._dedup_adoptions += 1
                self._entries.move_to_end(key)
                self._hold(key, existing, request_id)
                return existing.value

            if not self.can_allocate(size_bytes):
                raise RuntimeError(
                    f"{self._name}: cannot adopt {size_bytes} bytes; "
                    f"held={self._held_bytes}, max={self._max_bytes}. "
                    "The MM item scheduler must reserve space via "
                    "can_allocate() before encoding."
                )

            self._evict_zero_ref_until(self._max_bytes - size_bytes)
            entry = _Entry(value=value, size_bytes=size_bytes)
            self._entries[key] = entry
            self._current_bytes += size_bytes
            self._adoptions += 1
            self._hold(key, entry, request_id)
            return entry.value

    def release_holds(self, request_id: RequestId) -> None:
        """Release every hold of `request_id` (idempotent).

        This is the single teardown funnel: prefill completion, cancellation,
        and error paths all route here. Entries left with zero references
        stay resident for reuse until eviction needs their space.
        """
        with self._lock:
            keys = self._request_keys.pop(request_id, None)
            if not keys:
                return
            for key in keys:
                holders = self._key_holders.get(key)
                if holders is None:
                    continue
                holders.discard(request_id)
                if not holders:
                    del self._key_holders[key]
                    entry = self._entries.get(key)
                    if entry is not None:
                        self._held_bytes -= entry.size_bytes

    def free_resources(self, request: "LlmRequest") -> None:
        """`BaseResourceManager` teardown hook: release the request's holds.

        Runs for every request termination (completion, cancellation,
        errors) through `ResourceManager.free_resources`; the post-prefill
        strip additionally calls it early so entries become reclaimable as
        soon as their embedding has been consumed.
        """
        self.release_holds(request.request_id)

    def get_max_resource_count(self) -> int:
        """Encoder-output bytes are not a capacity-scheduler resource (the
        MM item scheduler budgets them at selection); return 0 so the
        executor does not gate admission on this manager."""
        return 0

    def get_needed_resource_to_completion(self, request: "LlmRequest") -> int:
        """See `get_max_resource_count`."""
        return 0

    def shutdown(self) -> None:
        self.clear()

    def clear(self) -> None:
        """Drop all entries and holds. Only safe when no request is active
        (e.g. warmup teardown or weight updates)."""
        with self._lock:
            self._entries.clear()
            self._key_holders.clear()
            self._request_keys.clear()
            self._current_bytes = 0
            self._held_bytes = 0

    def stats(self) -> MultimodalEncoderCacheStats:
        with self._lock:
            return MultimodalEncoderCacheStats(
                max_bytes=self._max_bytes,
                current_bytes=self._current_bytes,
                held_bytes=self._held_bytes,
                item_count=len(self._entries),
                held_count=len(self._key_holders),
                hits=self._hits,
                misses=self._misses,
                adoptions=self._adoptions,
                dedup_adoptions=self._dedup_adoptions,
                evictions=self._evictions,
            )

    def log_stats(self, reason: str) -> None:
        stats = self.stats()
        logger.debug(
            f"{self._name}: stats after {reason}: items={stats.item_count} "
            f"(held={stats.held_count}), bytes={stats.current_bytes}"
            f"/{stats.max_bytes} (held={stats.held_bytes}), "
            f"hits={stats.hits}, misses={stats.misses}, "
            f"adoptions={stats.adoptions} (dedup={stats.dedup_adoptions}), "
            f"evictions={stats.evictions}"
        )

    def _hold(self, key: K, entry: _Entry, request_id: RequestId) -> None:
        holders = self._key_holders.setdefault(key, set())
        if not holders:
            self._held_bytes += entry.size_bytes
        holders.add(request_id)
        self._request_keys.setdefault(request_id, set()).add(key)

    def _evict_zero_ref_until(self, target_bytes: int) -> None:
        if self._current_bytes <= target_bytes:
            return
        for key in list(self._entries):
            if self._current_bytes <= target_bytes:
                break
            if key in self._key_holders:
                continue
            entry = self._entries.pop(key)
            self._current_bytes -= entry.size_bytes
            self._evictions += 1
