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
"""KV cache event manager for KVCacheManagerV2.

This module mirrors the external v1 event contract while using v2 mutation
hooks as the event source:

- ``Created``, ``Stored``, ``Removed``, and ``Updated(cache_level)`` use
  dataclass names that the existing ``KVCacheEventSerializer`` recognizes.
- Removed events are coalesced per window and flushed before the next Stored
  event for the same window.
- Blocks are tracked per ``(window_size, block_key)`` so a single v2 radix
  block can produce independent full-attention and sliding-window event
  streams.
- ``flush_iteration_events()`` drains the per-iteration queue and, when
  attention DP is enabled, synchronously gathers events through an injected
  gather function.  It intentionally avoids a background Python collective on
  the model communicator.
"""

from collections import deque
from dataclasses import dataclass, field
from threading import Condition
from typing import Callable, NewType, Sequence

from ._common import PRIORITY_DEFAULT, CacheLevel, TokenIdExt

LifeCycleId = NewType("LifeCycleId", int)

_UINT32_MASK = (1 << 32) - 1
_UINT64_MASK = (1 << 64) - 1
_HASH64_CONST_1 = 0xBF58476D1CE4E5B9
_HASH64_CONST_2 = 0x94D049BB133111EB
_HASH_COMBINE_CONST = 0x9E3779B9
_PARENT_HASH_CONST = 0xBF58476D1CE4E5B9


@dataclass(slots=True, frozen=True)
class KVCacheEventDiffInt:
    old_value: int
    new_value: int


@dataclass(slots=True, frozen=True)
class KVCacheUniqueToken:
    token_id: int
    token_extra_id: int


@dataclass(slots=True, frozen=True)
class KVCacheCreatedData:
    num_blocks_per_cache_level: list[int]


@dataclass(slots=True, frozen=True)
class KVCacheStoredBlockData:
    block_hash: int
    tokens: list[KVCacheUniqueToken]
    lora_id: int | None
    cache_level: int
    priority: int
    mm_keys: list[tuple[bytes, int, str | None]] = field(default_factory=list)


@dataclass(slots=True, frozen=True)
class KVCacheStoredData:
    parent_hash: int | None
    blocks: list[KVCacheStoredBlockData]


@dataclass(slots=True, frozen=True)
class KVCacheRemovedData:
    block_hashes: list[int]


@dataclass(slots=True, frozen=True)
class KVCacheUpdatedData:
    block_hash: int
    cache_level: KVCacheEventDiffInt | None = None
    priority: KVCacheEventDiffInt | None = None


@dataclass(slots=True, frozen=True)
class KVCacheEvent:
    event_id: int
    data: KVCacheCreatedData | KVCacheStoredData | KVCacheRemovedData | KVCacheUpdatedData
    window_size: int
    attention_dp_rank: int | None = None


AttentionDpGatherFn = Callable[[list[KVCacheEvent]], Sequence[Sequence[KVCacheEvent]]]


class KVCacheEventManager:
    """Python KV cache event manager for KV cache manager v2."""

    def __init__(
        self,
        max_kv_event_entries: int,
        *,
        attention_dp_rank: int | None = None,
        attention_dp_gather_rank: int | None = None,
        attention_dp_size: int | None = None,
        attention_dp_gather_fn: AttentionDpGatherFn | None = None,
        default_window_size: int = 0,
    ) -> None:
        if max_kv_event_entries <= 0:
            raise ValueError("max_kv_event_entries must be positive")
        self._max_kv_event_entries = max_kv_event_entries
        self._attention_dp_rank = attention_dp_rank
        self._attention_dp_gather_rank = attention_dp_gather_rank
        if self._attention_dp_gather_rank is None:
            self._attention_dp_gather_rank = attention_dp_rank
        self._attention_dp_size = attention_dp_size
        self._attention_dp_gather_fn = attention_dp_gather_fn
        if (
            self._attention_dp_gather_rank == 0
            and self._attention_dp_size is not None
            and self._attention_dp_size > 1
        ):
            self._max_kv_event_entries *= self._attention_dp_size
        self._default_window_size = default_window_size
        self._event_id = 0
        self._event_queue: deque[KVCacheEvent] = deque()
        self._events: deque[KVCacheEvent] = deque()
        self._latest_removed_events: dict[int, KVCacheRemovedData | None] = {}
        self._life_cycle_window_sizes: dict[int, int | None] = {}
        self._block_hash_by_key: dict[bytes, int] = {}
        self._v1_hash_compatible_keys: set[bytes] = set()
        self._known_block_windows: dict[bytes, set[int]] = {}
        self._block_life_cycle_levels: dict[tuple[int, bytes], dict[int, int]] = {}
        self._block_logical_levels: dict[tuple[int, bytes], int] = {}
        self._condition = Condition()

    def set_default_window_size(self, window_size: int) -> None:
        self._default_window_size = window_size

    def set_life_cycle_window_sizes(self, life_cycle_window_sizes: dict[int, int | None]) -> None:
        self._life_cycle_window_sizes = dict(life_cycle_window_sizes)

    def enqueue_created_event(
        self,
        num_blocks_per_cache_level: list[int],
        window_size: int | None = None,
    ) -> None:
        with self._condition:
            self._enqueue_event_locked(
                KVCacheCreatedData(num_blocks_per_cache_level),
                self._resolve_window_size(window_size),
            )

    def on_blocks_stored(self, blocks: Sequence[object], window_size: int | None = None) -> None:
        with self._condition:
            stored_blocks_by_window: dict[int, list[KVCacheStoredBlockData]] = {}
            first_stored_block_by_window: dict[int, object] = {}
            for block in blocks:
                key = self._block_key(block)
                for resolved_window_size in self._block_window_sizes(block, window_size):
                    if self._is_known_in_window(key, resolved_window_size):
                        continue
                    first_stored_block_by_window.setdefault(resolved_window_size, block)
                    self._mark_known_in_window(key, resolved_window_size)
                    self._initialize_block_levels(block, resolved_window_size)
                    stored_blocks_by_window.setdefault(resolved_window_size, []).append(
                        self._stored_block_data(block, resolved_window_size)
                    )

            for resolved_window_size, stored_blocks in stored_blocks_by_window.items():
                self._flush_removed_events_locked(resolved_window_size)
                self._enqueue_event_locked(
                    KVCacheStoredData(
                        self._parent_hash(first_stored_block_by_window[resolved_window_size]),
                        stored_blocks,
                    ),
                    resolved_window_size,
                )

    def on_block_removed(
        self,
        block: object,
        window_size: int | None = None,
        life_cycle: LifeCycleId | None = None,
    ) -> None:
        with self._condition:
            key = self._block_key(block)
            known_windows = self._known_block_windows.get(key)
            if not known_windows:
                return

            block_hash = self._block_hash(block)
            if life_cycle is not None:
                windows = [self._window_size_for_life_cycle(life_cycle, window_size)]
            elif window_size is not None:
                windows = [self._resolve_window_size(window_size)]
            else:
                windows = list(known_windows)

            for resolved_window_size in windows:
                if not self._is_known_in_window(key, resolved_window_size):
                    continue
                latest_removed_event = self._latest_removed_events.get(resolved_window_size)
                if latest_removed_event is None:
                    self._latest_removed_events[resolved_window_size] = KVCacheRemovedData(
                        [block_hash]
                    )
                else:
                    latest_removed_event.block_hashes.append(block_hash)
                self._mark_unknown_in_window(key, resolved_window_size)
                self._block_life_cycle_levels.pop((resolved_window_size, key), None)
                self._block_logical_levels.pop((resolved_window_size, key), None)
                self._drop_block_cache_if_unknown(key)

    def on_block_updated(
        self,
        block: object | None,
        life_cycle: LifeCycleId,
        old_level: CacheLevel,
        new_level: CacheLevel,
        window_size: int | None = None,
    ) -> None:
        if block is None:
            return

        with self._condition:
            key = self._block_key(block)
            resolved_window_size = self._window_size_for_life_cycle(life_cycle, window_size)
            if not self._is_known_in_window(key, resolved_window_size):
                return

            state_key = (resolved_window_size, key)
            life_cycle_levels = self._block_life_cycle_levels.setdefault(state_key, {})
            old_logical_level = self._block_logical_levels.get(
                state_key, max(life_cycle_levels.values(), default=int(old_level))
            )
            life_cycle_levels[int(life_cycle)] = int(new_level)
            new_logical_level = max(life_cycle_levels.values(), default=int(new_level))
            self._block_logical_levels[state_key] = new_logical_level
            if old_logical_level == new_logical_level:
                return

            self._enqueue_event_locked(
                KVCacheUpdatedData(
                    self._block_hash(block),
                    cache_level=KVCacheEventDiffInt(old_logical_level, new_logical_level),
                ),
                resolved_window_size,
            )

    def flush_iteration_events(self) -> None:
        self.flush()
        self._exchange_attention_dp_events()

    def flush(self) -> None:
        with self._condition:
            for window_size in list(self._latest_removed_events):
                self._flush_removed_events_locked(window_size)
            if not self._event_queue:
                return

            events = list(self._event_queue)
            self._event_queue.clear()
            self._publish_events_locked(events)
            self._condition.notify_all()

    def get_latest_events(self, timeout_ms: float | None = 0) -> list[KVCacheEvent]:
        with self._condition:
            if not self._events:
                if timeout_ms is None:
                    self._condition.wait_for(lambda: bool(self._events))
                elif timeout_ms > 0:
                    self._condition.wait_for(lambda: bool(self._events), timeout_ms / 1000.0)

            events = list(self._events)
            self._events.clear()
            return events

    def _enqueue_event_locked(
        self,
        data: KVCacheCreatedData | KVCacheStoredData | KVCacheRemovedData | KVCacheUpdatedData,
        window_size: int,
    ) -> None:
        self._event_queue.append(
            KVCacheEvent(self._event_id, data, window_size, self._attention_dp_rank)
        )
        self._event_id += 1

    def _exchange_attention_dp_events(self) -> None:
        if self._attention_dp_rank is None or self._attention_dp_gather_fn is None:
            return
        if self._attention_dp_size is not None and self._attention_dp_size <= 1:
            return

        with self._condition:
            local_events = list(self._events)
            self._events.clear()

        gathered_events = self._attention_dp_gather_fn(local_events)

        with self._condition:
            if self._attention_dp_gather_rank == 0:
                self._publish_events_locked(
                    [event for rank_events in gathered_events for event in rank_events]
                )
            self._condition.notify_all()

    def _publish_events_locked(self, events: Sequence[KVCacheEvent]) -> None:
        elements_to_remove = len(self._events) + len(events) - self._max_kv_event_entries
        if elements_to_remove > 0:
            num_removed = min(len(self._events), elements_to_remove)
            for _ in range(num_removed):
                self._events.popleft()
            elements_to_remove -= num_removed
        if elements_to_remove > 0:
            events = events[elements_to_remove:]

        self._events.extend(events)

    def _flush_removed_events_locked(self, window_size: int) -> None:
        latest_removed_event = self._latest_removed_events.get(window_size)
        if latest_removed_event is None:
            return
        self._enqueue_event_locked(latest_removed_event, window_size)
        self._latest_removed_events[window_size] = None

    def _stored_block_data(self, block: object, window_size: int) -> KVCacheStoredBlockData:
        pages = self._block_pages(block, window_size)
        cache_level = self._block_logical_levels.get((window_size, self._block_key(block)), 0)
        priority = max(
            (int(getattr(page, "priority")) for _, page in pages),
            default=int(PRIORITY_DEFAULT),
        )
        return KVCacheStoredBlockData(
            block_hash=self._block_hash(block),
            tokens=[self._token_to_unique_token(token) for token in getattr(block, "tokens")],
            lora_id=self._lora_task_id(block),
            cache_level=cache_level,
            priority=priority,
        )

    def _initialize_block_levels(self, block: object, window_size: int) -> None:
        levels = {
            int(life_cycle): int(getattr(page, "cache_level"))
            for life_cycle, page in self._block_pages(block, window_size)
        }
        key = self._block_key(block)
        state_key = (window_size, key)
        self._block_life_cycle_levels[state_key] = levels
        self._block_logical_levels[state_key] = max(levels.values(), default=0)

    def _block_pages(
        self, block: object, window_size: int | None = None
    ) -> list[tuple[LifeCycleId, object]]:
        pages = []
        for life_cycle, page_ref in enumerate(getattr(block, "storage")):
            if page_ref is None:
                continue
            page = page_ref() if callable(page_ref) else page_ref
            if page is not None and self._life_cycle_matches_window(
                LifeCycleId(life_cycle), window_size
            ):
                pages.append((LifeCycleId(life_cycle), page))
        return pages

    def _block_window_sizes(self, block: object, window_size: int | None = None) -> list[int]:
        if window_size is not None:
            return [self._resolve_window_size(window_size)]
        if not self._life_cycle_window_sizes:
            return [self._resolve_window_size(None)]

        window_sizes = {
            self._window_size_for_life_cycle(life_cycle)
            for life_cycle, _ in self._block_pages(block)
            if int(life_cycle) in self._life_cycle_window_sizes
        }
        return sorted(window_sizes)

    def _life_cycle_matches_window(self, life_cycle: LifeCycleId, window_size: int | None) -> bool:
        if window_size is None or not self._life_cycle_window_sizes:
            return True
        if int(life_cycle) not in self._life_cycle_window_sizes:
            return False
        return self._window_size_for_life_cycle(life_cycle) == window_size

    def _window_size_for_life_cycle(
        self, life_cycle: LifeCycleId, fallback_window_size: int | None = None
    ) -> int:
        if int(life_cycle) not in self._life_cycle_window_sizes:
            return self._resolve_window_size(fallback_window_size)
        return self._resolve_window_size(self._life_cycle_window_sizes[int(life_cycle)])

    @staticmethod
    def _token_to_unique_token(token: TokenIdExt) -> KVCacheUniqueToken:
        if type(token) is int:
            return KVCacheUniqueToken(int(token), 0)
        token_hash = int.from_bytes(bytes(token)[:8].ljust(8, b"\0"), "little")
        return KVCacheUniqueToken(0, token_hash)

    def _parent_hash(self, block: object) -> int | None:
        parent = getattr(block, "prev")
        if not self._is_block(parent):
            return None
        return self._block_hash(parent)

    def _block_hash(self, block: object) -> int:
        key = self._block_key(block)
        cached = self._block_hash_by_key.get(key)
        if cached is not None:
            return cached

        parent = getattr(block, "prev")
        parent_is_block = self._is_block(parent)
        parent_is_v1_compatible = (
            not parent_is_block or self._block_key(parent) in self._v1_hash_compatible_keys
        )
        if self._has_text_tokens(block) and parent_is_v1_compatible:
            parent_hash = self._block_hash(parent) if parent_is_block else 0
            block_hash = self._hash_block_key(
                getattr(block, "tokens"), parent_hash, self._lora_task_id(block)
            )
            self._v1_hash_compatible_keys.add(key)
        else:
            block_hash = int.from_bytes(key[:8].ljust(8, b"\0"), "little")

        self._block_hash_by_key[key] = block_hash
        return block_hash

    @staticmethod
    def _has_text_tokens(block: object) -> bool:
        return all(type(token) is int for token in getattr(block, "tokens"))

    @staticmethod
    def _is_block(value: object) -> bool:
        return hasattr(value, "tokens") and hasattr(value, "key")

    @staticmethod
    def _block_key(block: object) -> bytes:
        return bytes(getattr(block, "key"))

    def _lora_task_id(self, block: object) -> int | None:
        current = block
        while self._is_block(getattr(current, "prev")):
            current = getattr(current, "prev")
        root = getattr(current, "prev")
        return getattr(root, "lora_task_id", None)

    def _resolve_window_size(self, window_size: int | None) -> int:
        return self._default_window_size if window_size is None else window_size

    def _is_known_in_window(self, key: bytes, window_size: int) -> bool:
        return window_size in self._known_block_windows.get(key, set())

    def _mark_known_in_window(self, key: bytes, window_size: int) -> None:
        self._known_block_windows.setdefault(key, set()).add(window_size)

    def _mark_unknown_in_window(self, key: bytes, window_size: int) -> None:
        windows = self._known_block_windows.get(key)
        if windows is None:
            return
        windows.discard(window_size)
        if not windows:
            self._known_block_windows.pop(key, None)

    def _drop_block_cache_if_unknown(self, key: bytes) -> None:
        if key in self._known_block_windows:
            return
        self._block_hash_by_key.pop(key, None)
        self._v1_hash_compatible_keys.discard(key)

    @staticmethod
    def _hash_block_key(
        tokens: Sequence[TokenIdExt],
        parent_hash: int,
        lora_task_id: int | None,
    ) -> int:
        seed = (len(tokens) ^ ((parent_hash * _PARENT_HASH_CONST) & _UINT64_MASK)) & _UINT64_MASK
        for token in tokens:
            assert type(token) is int, "v1-compatible hashing only supports text tokens"
            seed = KVCacheEventManager._hash32_mix(int(token) & _UINT32_MASK, seed)
        if lora_task_id is not None:
            seed = KVCacheEventManager._hash64_mix(lora_task_id, seed)
        return seed

    @staticmethod
    def _hash32_mix(value: int, seed: int) -> int:
        value &= _UINT32_MASK
        value = (((value >> 16) ^ value) * 0x45D9F3B) & _UINT32_MASK
        value = (((value >> 16) ^ value) * 0x45D9F3B) & _UINT32_MASK
        value = ((value >> 16) ^ value) & _UINT32_MASK
        combined = (
            value + _HASH_COMBINE_CONST + ((seed << 6) & _UINT64_MASK) + (seed >> 2)
        ) & _UINT64_MASK
        return (seed ^ combined) & _UINT64_MASK

    @staticmethod
    def _hash64_mix(value: int, seed: int) -> int:
        value &= _UINT64_MASK
        value = ((value ^ (value >> 30)) * _HASH64_CONST_1) & _UINT64_MASK
        value = ((value ^ (value >> 27)) * _HASH64_CONST_2) & _UINT64_MASK
        value = (value ^ (value >> 31)) & _UINT64_MASK
        combined = (
            value + _HASH_COMBINE_CONST + ((seed << 6) & _UINT64_MASK) + (seed >> 2)
        ) & _UINT64_MASK
        return (seed ^ combined) & _UINT64_MASK
