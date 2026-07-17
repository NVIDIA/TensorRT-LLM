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
from threading import RLock
from typing import Generic, NamedTuple, TypeVar

import torch

from tensorrt_llm.logger import logger

K = TypeVar("K", bound=Hashable)


class _Entry(NamedTuple):
    value: torch.Tensor
    size_bytes: int
    # CUDA event recorded on the producing stream right after the clone in `put`. Consumers on a
    # different stream wait on it before reading `value`. `None` for CPU tensors or when the cache
    # is not stream-aware.
    producer_event: torch.cuda.Event | None


class TensorLRUCacheStats(NamedTuple):
    max_bytes: int
    current_bytes: int
    item_count: int
    hits: int
    misses: int
    insertions: int
    replacements: int
    evictions: int
    rejected_insertions: int
    hit_rate: float


@dataclass
class _CacheCounters:
    hits: int = 0
    misses: int = 0
    insertions: int = 0
    replacements: int = 0
    evictions: int = 0
    rejected_insertions: int = 0

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

    def get(self, key: K) -> torch.Tensor | None:
        """Return a cache-owned, immutable tensor and promote it to most-recently-used.

        The returned tensor aliases the cached value. Callers must not mutate it.
        """
        with self._lock:
            entry = self._items.get(key)
            if entry is None:
                self._counters.misses += 1
                return None

            self._counters.hits += 1
            self._items.move_to_end(key)
            self._prepare_for_current_stream(entry)
            return entry.value

    def put(self, key: K, value: torch.Tensor) -> bool:
        """Insert or replace a tensor.

        Returns `False` and leaves the cache unchanged when `value` is larger than the full
        cache capacity.
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

        stored_value = value.detach().clone()
        producer_event = None
        if self._cuda_stream_aware and stored_value.is_cuda:
            producer_event = torch.cuda.Event()
            producer_event.record(torch.cuda.current_stream(stored_value.device))

        with self._lock:
            old_entry = self._items.pop(key, None)
            if old_entry is not None:
                self._current_bytes -= old_entry.size_bytes
                self._counters.replacements += 1
            else:
                self._counters.insertions += 1

            self._items[key] = _Entry(
                value=stored_value,
                size_bytes=size_bytes,
                producer_event=producer_event,
            )
            self._current_bytes += size_bytes

            evicted_count, evicted_bytes = self._evict_until_within_limit()
            if evicted_count:
                self._counters.evictions += evicted_count
                logger.debug(
                    f"{self._name}: evicted {evicted_count} LRU entries, "
                    f"freed_bytes={evicted_bytes}, current_bytes={self._current_bytes}, "
                    f"max_bytes={self._max_bytes}"
                )
            return True

    def pop(self, key: K) -> torch.Tensor | None:
        """Remove one key and return its tensor, or `None` on miss."""
        with self._lock:
            entry = self._items.pop(key, None)
            if entry is None:
                return None

            self._current_bytes -= entry.size_bytes
            self._prepare_for_current_stream(entry)
            return entry.value

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._current_bytes = 0

    def stats(self) -> TensorLRUCacheStats:
        with self._lock:
            return TensorLRUCacheStats(
                max_bytes=self._max_bytes,
                current_bytes=self._current_bytes,
                item_count=len(self._items),
                hits=self._counters.hits,
                misses=self._counters.misses,
                insertions=self._counters.insertions,
                replacements=self._counters.replacements,
                evictions=self._counters.evictions,
                rejected_insertions=self._counters.rejected_insertions,
                hit_rate=self._counters.hit_rate,
            )

    def log_stats(self, reason: str) -> None:
        stats = self.stats()
        logger.debug(
            f"{self._name}: stats after {reason}: items={stats.item_count}, "
            f"bytes={stats.current_bytes}/{stats.max_bytes}, hits={stats.hits}, "
            f"misses={stats.misses}, hit_rate={stats.hit_rate:.3f}, "
            f"insertions={stats.insertions}, replacements={stats.replacements}, "
            f"evictions={stats.evictions}, rejected_insertions={stats.rejected_insertions}"
        )

    @staticmethod
    def _tensor_size_bytes(tensor: torch.Tensor) -> int:
        return tensor.numel() * tensor.element_size()

    def _prepare_for_current_stream(self, entry: _Entry) -> None:
        """Order and anchor a cached tensor for consumption on the current stream.

        Called from `get` / `pop` before returning an entry it:
        * makes the current (consuming) stream wait on the entry's producer event, so a cross-stream
          read observes fully-written data
        * calls `record_stream` on the consuming stream so the caching allocator will not reuse
          the storage while consumer-stream work is still pending, even if a later replacement or
          eviction drops the cache's own reference.
        """
        if not self._cuda_stream_aware or not entry.value.is_cuda:
            return

        consumer_stream = torch.cuda.current_stream(entry.value.device)
        # The producer event orders the data dependency; `record_stream` separately guards lifetime.
        if entry.producer_event is not None:
            consumer_stream.wait_event(entry.producer_event)
        entry.value.record_stream(consumer_stream)

    def _evict_until_within_limit(self) -> tuple[int, int]:
        evicted_count = 0
        evicted_bytes = 0
        while self._current_bytes > self._max_bytes:
            _, entry = self._items.popitem(last=False)
            self._current_bytes -= entry.size_bytes
            evicted_count += 1
            evicted_bytes += entry.size_bytes
        return evicted_count, evicted_bytes
