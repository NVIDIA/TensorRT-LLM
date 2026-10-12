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
from dataclasses import dataclass
from enum import Enum, auto
from threading import RLock
from typing import Generic, NamedTuple, TypeVar

import torch

from tensorrt_llm.logger import logger

K = TypeVar("K", bound=Hashable)


class _CacheEntryState(Enum):
    """Lifecycle state of a cache key."""

    # Expected bytes and references exist, but no tensor has been committed.
    RESERVED = auto()
    # The cache owns a tensor that callers may read.
    READY = auto()


class CacheAcquireResult(Enum):
    """How `acquire` satisfied a successful request."""

    # A READY entry was found and referenced.
    READY_HIT = auto()
    # A missing key became a new RESERVED entry.
    NEW_RESERVATION = auto()
    # An existing RESERVED entry gained another reference.
    RESERVATION_HIT = auto()


@dataclass
class _Entry:
    state: _CacheEntryState
    size_bytes: int
    reference_count: int
    retain_after_release: bool
    value: torch.Tensor | None = None
    # CUDA event recorded on the producing stream right after the clone in `put`. Consumers on a
    # different stream wait on it before reading `value`. `None` for CPU tensors or when the cache
    # is not stream-aware.
    producer_event: torch.cuda.Event | None = None


class TensorLRUCacheStats(NamedTuple):
    max_bytes: int
    current_bytes: int
    reserved_bytes: int
    in_use_bytes: int
    item_count: int
    hits: int
    misses: int
    insertions: int
    replacements: int
    evictions: int
    rejected_insertions: int
    producer_misses: int
    inflight_deduplications: int
    hit_rate: float


@dataclass
class _CacheCounters:
    hits: int = 0
    misses: int = 0
    insertions: int = 0
    replacements: int = 0
    evictions: int = 0
    rejected_insertions: int = 0
    producer_misses: int = 0
    inflight_deduplications: int = 0

    @property
    def hit_rate(self) -> float:
        total_gets = self.hits + self.misses
        return self.hits / total_gets if total_gets else 0.0


class TensorLRUCache(Generic[K]):
    """Thread-safe LRU cache from hashable keys to tensor values.

    Size accounting uses logical tensor bytes: `tensor.numel() * tensor.element_size()`.
    Returned tensors alias the cache-owned tensor objects. Callers must treat them as immutable
    while they remain cache-owned.

    The cache owns a detached copy rather than the caller's tensor or view. This prevents later
    caller mutations from changing a cached value and prevents a small cached view from retaining
    the caller's larger backing allocation. The copy is made before acquiring the lock and before
    replacement or eviction, preserving existing cache entries if copying fails. Consequently,
    `max_bytes` bounds steady-state logical cache contents, not peak allocation: insertion
    temporarily needs both the source tensor and its copy and may exceed the cache limit until
    eviction completes.

    Managed entries add a RESERVED state and reference-counted READY state to the original cache:

    1. `acquire` adds a reference and creates a `RESERVED` entry on a miss.
    2. `ensure_capacity` evicts only unreferenced `READY` entries before selected
       producers materialize their outputs.
    3. `commit` stores a producer output in its `RESERVED` entry without
       choosing more eviction victims.
    4. `get` reads a `READY` tensor without changing its reference count.
    5. `release` drops the reference and applies the entry's retention policy.

    Reservations and referenced READY entries cannot be evicted. An entry is
    referenced after `acquire` increments its `reference_count` and until the
    matching `release` calls reduce it to zero. `in_use_bytes` is the logical
    size of those referenced READY tensors.

    `pop` is intentionally separate from `release`: it physically removes an
    unreferenced entry without applying reference or retention policy.

    In CUDA-stream-aware mode, each entry owns the event recorded after its clone. Replacement,
    eviction, and clear drop that event with the entry; events are not reused because an evicted
    tensor may still have outstanding consumers on another stream.

    Args:
        max_bytes: Maximum logical tensor bytes held by this cache.
        name: Short label used in debug log messages.
        cuda_stream_aware: When enabled, synchronize CUDA tensor producers and consumers across
            streams and extend allocation lifetime through every consuming stream. CPU tensors are
            unaffected.
    """

    def __init__(
        self,
        max_bytes: int,
        *,
        name: str = "tensor_lru_cache",
        cuda_stream_aware: bool = False,
    ) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")

        self._max_bytes = max_bytes
        self._name = name
        self._cuda_stream_aware = cuda_stream_aware
        self._current_bytes = 0
        self._reserved_bytes = 0
        # READY tensor bytes whose reference_count is positive, so they cannot be evicted.
        self._in_use_bytes = 0
        self._items: OrderedDict[K, _Entry] = OrderedDict()
        self._lock = RLock()
        self._counters = _CacheCounters()

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    @property
    def current_bytes(self) -> int:
        with self._lock:
            return self._current_bytes

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def acquire(
        self,
        key: K,
        expected_bytes: int,
        *,
        retain_after_release: bool = True,
    ) -> CacheAcquireResult | None:
        """Acquire a reference, reserving a missing entry when needed.

        A READY hit and an existing reservation both gain one reference. A
        missing key becomes RESERVED and accounts `expected_bytes` until the
        producer stores its tensor with `commit`.

        Args:
            key: Identity shared by producers and consumers of one tensor.
            expected_bytes: Exact tensor bytes that a missing key will produce.
            retain_after_release: Whether a READY entry remains reusable after
                its final reference is released. Stable content-key entries
                use `True` for cross-request reuse; request-local entries use
                `False` and are removed immediately.

        Returns:
            How the reference was acquired, or `None` when live references and
            reservations temporarily consume all capacity.

        Raises:
            ValueError: If the requested size is invalid or existing metadata
                conflicts with the request.
        """
        if expected_bytes <= 0:
            raise ValueError("expected_bytes must be positive")
        if expected_bytes > self._max_bytes:
            raise ValueError(
                f"expected_bytes ({expected_bytes}) exceeds cache capacity ({self._max_bytes})"
            )

        with self._lock:
            entry = self._items.get(key)
            if entry is not None:
                if entry.size_bytes != expected_bytes:
                    raise ValueError(
                        f"existing cache entry size ({entry.size_bytes}) does not match "
                        f"expected_bytes ({expected_bytes})"
                    )
                if entry.retain_after_release != retain_after_release:
                    raise ValueError("existing cache entry retention policy does not match")

                if entry.state is _CacheEntryState.RESERVED:
                    entry.reference_count += 1
                    self._counters.inflight_deduplications += 1
                    return CacheAcquireResult.RESERVATION_HIT
                if entry.state is not _CacheEntryState.READY:
                    raise RuntimeError(f"unexpected cache entry state: {entry.state}")

                if entry.reference_count == 0:
                    # A retained cache hit becomes non-evictable again. Make
                    # sure it fits beside current reservations and references
                    # before claiming it for this request.
                    if self._claimed_bytes + entry.size_bytes > self._max_bytes:
                        return None
                    self._in_use_bytes += entry.size_bytes
                entry.reference_count += 1
                self._items.move_to_end(key)
                self._counters.hits += 1
                return CacheAcquireResult.READY_HIT

            if self._claimed_bytes + expected_bytes > self._max_bytes:
                return None

            self._items[key] = _Entry(
                state=_CacheEntryState.RESERVED,
                size_bytes=expected_bytes,
                reference_count=1,
                retain_after_release=retain_after_release,
            )
            self._reserved_bytes += expected_bytes
            self._counters.misses += 1
            self._counters.producer_misses += 1
            return CacheAcquireResult.NEW_RESERVATION

    def get(self, key: K, *, record_stats: bool = True) -> torch.Tensor | None:
        """Return a cache-owned, immutable tensor and promote it to most-recently-used.

        The returned tensor aliases the cached value. Callers must not mutate it.
        Only READY entries are returned. `get` does not acquire a managed
        reference; `record_stats=False` suppresses its hit/miss accounting.
        """
        with self._lock:
            entry = self._items.get(key)
            if entry is None or entry.state is not _CacheEntryState.READY:
                if record_stats:
                    self._counters.misses += 1
                return None

            if record_stats:
                self._counters.hits += 1
            self._items.move_to_end(key)
            self._prepare_for_current_stream(entry)
            assert entry.value is not None
            return entry.value

    def put(
        self,
        key: K,
        value: torch.Tensor,
    ) -> bool:
        """Insert or replace a tensor.

        Returns `False` and leaves the cache contents unchanged when the value
        cannot fit beside current reservations and references.
        """
        size_bytes = self._tensor_size_bytes(value)

        if size_bytes > self._max_bytes:
            with self._lock:
                self._counters.rejected_insertions += 1
            logger.debug(
                f"{self._name}: rejected oversized tensor insertion, size_bytes={size_bytes}, "
                f"max_bytes={self._max_bytes}"
            )
            return False

        stored_value, producer_event = self._clone_for_storage(value)

        with self._lock:
            old_entry = self._items.get(key)
            if old_entry is not None:
                if old_entry.state is not _CacheEntryState.READY:
                    raise RuntimeError("cannot replace a reserved cache entry")
                if old_entry.reference_count:
                    raise RuntimeError("cannot replace a referenced cache entry")

            # `put` creates an unreferenced READY entry, but it must leave
            # enough room for every reservation and referenced entry to become
            # resident. Reject before changing the cache when those protected
            # bytes already claim the remaining capacity.
            if self._claimed_bytes + size_bytes > self._max_bytes:
                self._counters.rejected_insertions += 1
                return False

            if old_entry is not None:
                del self._items[key]
                self._current_bytes -= old_entry.size_bytes
                self._counters.replacements += 1
            else:
                self._counters.insertions += 1

            # The protected-byte check above guarantees that removable READY
            # entries can make this space without failure.
            self.ensure_capacity(size_bytes)

            self._items[key] = _Entry(
                state=_CacheEntryState.READY,
                size_bytes=size_bytes,
                reference_count=0,
                retain_after_release=True,
                value=stored_value,
                producer_event=producer_event,
            )
            self._current_bytes += size_bytes
            return True

    def commit(self, key: K, value: torch.Tensor) -> None:
        """Store a producer output in an acquired reservation.

        The cache owns the reservation size and verifies the producer output
        before changing the entry to READY. Capacity must already have been
        prepared with `ensure_capacity`.
        """
        size_bytes = self._tensor_size_bytes(value)
        with self._lock:
            entry = self._items.get(key)
            if entry is None or entry.state is not _CacheEntryState.RESERVED:
                raise RuntimeError("cache commit requires a reserved entry")
            if size_bytes != entry.size_bytes:
                raise ValueError(
                    f"tensor size ({size_bytes}) does not match reserved bytes ({entry.size_bytes})"
                )
            if self._current_bytes + size_bytes > self._max_bytes:
                raise RuntimeError("reserved entry does not have enough cache space")

            stored_value, producer_event = self._clone_for_storage(value)
            entry.state = _CacheEntryState.READY
            entry.value = stored_value
            entry.producer_event = producer_event
            self._reserved_bytes -= size_bytes
            self._in_use_bytes += size_bytes
            self._current_bytes += size_bytes
            self._items.move_to_end(key)
            self._counters.insertions += 1

    def ensure_capacity(self, incoming_bytes: int) -> None:
        """Ensure physical space for selected outputs before they are produced.

        `incoming_bytes` is the total size of outputs selected for the next
        producer commits. Only unreferenced READY entries are eligible LRU victims;
        reservations and referenced tensors are never removed. A failure
        leaves the cache unchanged.
        """
        if incoming_bytes < 0:
            raise ValueError("incoming_bytes must be non-negative")
        with self._lock:
            required_bytes = max(
                0,
                self._current_bytes + incoming_bytes - self._max_bytes,
            )
            if required_bytes == 0:
                return

            entries_to_evict: list[tuple[K, _Entry]] = []
            freed_bytes = 0
            for cache_key, entry in list(self._items.items()):
                if entry.state is not _CacheEntryState.READY or entry.reference_count != 0:
                    continue
                entries_to_evict.append((cache_key, entry))
                freed_bytes += entry.size_bytes
                if freed_bytes >= required_bytes:
                    break

            if freed_bytes < required_bytes:
                raise RuntimeError("cache does not have enough removable space")

            for cache_key, entry in entries_to_evict:
                del self._items[cache_key]
                self._current_bytes -= entry.size_bytes
            self._counters.evictions += len(entries_to_evict)

    def release(self, key: K) -> None:
        """Release one acquired reference and apply retention policy.

        A RESERVED entry disappears when its final producer or follower
        reference is released. A READY entry becomes evictable at zero
        references and remains cached when `retain_after_release` is true;
        otherwise it is removed immediately.
        """
        with self._lock:
            entry = self._items.get(key)
            if entry is None or entry.reference_count == 0:
                raise RuntimeError("cannot release an unreferenced cache entry")

            entry.reference_count -= 1
            if entry.reference_count:
                return None

            if entry.state is _CacheEntryState.RESERVED:
                self._reserved_bytes -= entry.size_bytes
                del self._items[key]
                return None

            if entry.state is not _CacheEntryState.READY:
                raise RuntimeError(f"unexpected cache entry state: {entry.state}")

            self._in_use_bytes -= entry.size_bytes
            if entry.retain_after_release:
                return None

            self._current_bytes -= entry.size_bytes
            del self._items[key]

    def pop(self, key: K) -> torch.Tensor | None:
        """Remove one unreferenced key without applying release policy.

        Returns the READY tensor, or `None` for a removed reservation or miss.
        """
        with self._lock:
            entry = self._items.get(key)
            if entry is None:
                return None
            if entry.reference_count:
                raise RuntimeError("cannot remove a referenced cache entry")

            del self._items[key]
            if entry.state is _CacheEntryState.RESERVED:
                self._reserved_bytes -= entry.size_bytes
                return None

            self._current_bytes -= entry.size_bytes
            self._prepare_for_current_stream(entry)
            assert entry.value is not None
            return entry.value

    def clear(self) -> None:
        """Remove all entries unless a managed reference or reservation is active."""
        with self._lock:
            if self._claimed_bytes:
                raise RuntimeError("cannot clear cache with live references or reservations")
            self._items.clear()
            self._current_bytes = 0

    def stats(self) -> TensorLRUCacheStats:
        with self._lock:
            return TensorLRUCacheStats(
                max_bytes=self._max_bytes,
                current_bytes=self._current_bytes,
                reserved_bytes=self._reserved_bytes,
                in_use_bytes=self._in_use_bytes,
                item_count=len(self._items),
                hits=self._counters.hits,
                misses=self._counters.misses,
                insertions=self._counters.insertions,
                replacements=self._counters.replacements,
                evictions=self._counters.evictions,
                rejected_insertions=self._counters.rejected_insertions,
                producer_misses=self._counters.producer_misses,
                inflight_deduplications=self._counters.inflight_deduplications,
                hit_rate=self._counters.hit_rate,
            )

    def log_stats(self, reason: str) -> None:
        stats = self.stats()
        logger.debug(
            f"{self._name}: stats after {reason}: items={stats.item_count}, "
            f"bytes={stats.current_bytes}/{stats.max_bytes}, hits={stats.hits}, "
            f"misses={stats.misses}, hit_rate={stats.hit_rate:.3f}, "
            f"insertions={stats.insertions}, replacements={stats.replacements}, "
            f"evictions={stats.evictions}, rejected_insertions={stats.rejected_insertions}, "
            f"producer_misses={stats.producer_misses}, "
            f"inflight_deduplications={stats.inflight_deduplications}"
        )

    @property
    def _claimed_bytes(self) -> int:
        return self._reserved_bytes + self._in_use_bytes

    @staticmethod
    def _tensor_size_bytes(tensor: torch.Tensor) -> int:
        return tensor.numel() * tensor.element_size()

    def _clone_for_storage(
        self, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.cuda.Event | None]:
        """Clone a value and record its producer event when stream-aware."""
        stored_value = value.detach().clone()
        producer_event = None
        if self._cuda_stream_aware and stored_value.is_cuda:
            producer_event = torch.cuda.Event()
            producer_event.record(torch.cuda.current_stream(stored_value.device))
        return stored_value, producer_event

    def _prepare_for_current_stream(self, entry: _Entry) -> None:
        """Order and anchor a cached tensor for consumption on the current stream.

        Called from `get` / `pop` before returning an entry it:
        * makes the current (consuming) stream wait on the entry's producer event, so a cross-stream
          read observes fully-written data
        * calls `record_stream` on the consuming stream so the caching allocator will not reuse
          the storage while consumer-stream work is still pending, even if a later replacement or
          eviction drops the cache's own reference.
        """
        if entry.value is None:
            raise RuntimeError("cannot access a cache entry before its value is stored")
        if not self._cuda_stream_aware or not entry.value.is_cuda:
            return

        consumer_stream = torch.cuda.current_stream(entry.value.device)
        # The producer event orders the data dependency; `record_stream` separately guards lifetime.
        if entry.producer_event is not None:
            consumer_stream.wait_event(entry.producer_event)
        entry.value.record_stream(consumer_stream)
