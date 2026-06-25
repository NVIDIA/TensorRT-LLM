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
    Returned tensors are the cached tensor objects themselves, so callers should treat them as
    immutable while they remain cache-owned.

    Args:
        max_bytes: Maximum logical tensor bytes held by this cache.
        clone_on_insert: Store `value.detach().clone()` instead of the caller's tensor object.
        name: Short label used in debug log messages.
    """

    def __init__(
        self,
        max_bytes: int,
        *,
        clone_on_insert: bool = False,
        name: str = "tensor_lru_cache",
    ) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")

        self._max_bytes = max_bytes
        self._clone_on_insert = clone_on_insert
        self._name = name
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
        """Return a cached tensor and promote it to most-recently-used."""
        with self._lock:
            entry = self._items.get(key)
            if entry is None:
                self._counters.misses += 1
                return None

            self._counters.hits += 1
            self._items.move_to_end(key)
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

        stored_value = value.detach().clone() if self._clone_on_insert else value

        with self._lock:
            old_entry = self._items.pop(key, None)
            if old_entry is not None:
                self._current_bytes -= old_entry.size_bytes
                self._counters.replacements += 1
            else:
                self._counters.insertions += 1

            self._items[key] = _Entry(value=stored_value, size_bytes=size_bytes)
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

    def _evict_until_within_limit(self) -> tuple[int, int]:
        evicted_count = 0
        evicted_bytes = 0
        while self._current_bytes > self._max_bytes:
            _, entry = self._items.popitem(last=False)
            self._current_bytes -= entry.size_bytes
            evicted_count += 1
            evicted_bytes += entry.size_bytes
        return evicted_count, evicted_bytes
