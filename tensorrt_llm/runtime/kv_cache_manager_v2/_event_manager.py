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

from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from threading import Condition
from typing import Any, Callable

from tensorrt_llm.logger import logger
from tensorrt_llm.runtime.kv_cache_hash import (
    KV_CACHE_HASH_ALGO_AUTO,
    KV_CACHE_HASH_ALGO_DEFAULT,
    KV_CACHE_HASH_ALGO_V1,
    KV_CACHE_HASH_ALGO_V2,
    KV_CACHE_HASH_ALGO_V2_SHA256_64,
    NonTextTokenHashError,
    hash_v1_block_key,
    truncate_sha256_hash_to_int64,
)

from ._common import GPU_LEVEL, PRIORITY_DEFAULT, CacheLevel, Priority, TokenIdExt

EventBlockHash = int | str
BlockHashLike = bytes | EventBlockHash
BlockHashesLike = BlockHashLike | Iterable[BlockHashLike]
LayerGroupId = int | None
EventTokenId = int | str
MmKey = tuple[bytes, int] | tuple[bytes, int, str | None]
AttentionDpGatherFn = Callable[[list["KVCacheEvent"]], list[list["KVCacheEvent"]]]


@dataclass(slots=True, frozen=True)
class UniqueToken:
    token_id: EventTokenId
    token_extra_id: int = 0


@dataclass(slots=True, frozen=True)
class KVCacheCreatedData:
    num_blocks_per_cache_level: list[int]


@dataclass(slots=True, frozen=True)
class KVCacheStoredBlockData:
    block_hash: EventBlockHash
    tokens: list[UniqueToken]
    cache_level: int
    priority: int
    mm_keys: list[MmKey] = field(default_factory=list)


@dataclass(slots=True, frozen=True)
class KVCacheStoredData:
    parent_hash: EventBlockHash | None
    blocks: list[KVCacheStoredBlockData]


@dataclass(slots=True, frozen=True)
class KVCacheRemovedData:
    block_hashes: list[EventBlockHash]


@dataclass(slots=True, frozen=True)
class KVCacheEventDiff:
    old_value: int
    new_value: int


@dataclass(slots=True, frozen=True)
class KVCacheUpdatedData:
    block_hash: EventBlockHash
    cache_level: KVCacheEventDiff | None
    priority: KVCacheEventDiff | None


@dataclass(slots=True, frozen=True)
class KVCacheEvent:
    event_id: int
    data: KVCacheCreatedData | KVCacheStoredData | KVCacheRemovedData | KVCacheUpdatedData
    window_size: int
    hash_algo: str | None = None
    attention_dp_rank: int | None = None
    layer_group_id: int | None = None


@dataclass(slots=True)
class _StoredBlockState:
    block_hash: EventBlockHash
    life_cycle_ids: set[int]


class KVCacheEventManager:
    """Python event queue matching the C++ KV cache event serializer contract."""

    def __init__(
        self,
        max_kv_event_entries: int,
        *,
        window_size: int = 0,
        attention_dp_rank: int | None = None,
        attention_dp_gather: AttentionDpGatherFn | None = None,
        hash_algo: str = KV_CACHE_HASH_ALGO_V2,
        window_size_by_layer_group: dict[int, int] | None = None,
    ) -> None:
        if hash_algo == KV_CACHE_HASH_ALGO_AUTO:
            hash_algo = KV_CACHE_HASH_ALGO_DEFAULT
        elif hash_algo not in (
            KV_CACHE_HASH_ALGO_V1,
            KV_CACHE_HASH_ALGO_V2,
            KV_CACHE_HASH_ALGO_V2_SHA256_64,
        ):
            raise ValueError(f"Unsupported V2 KV cache event hash algorithm: {hash_algo}")
        self._max_kv_event_entries = max_kv_event_entries
        self._window_size = window_size
        self._window_size_by_layer_group = dict(window_size_by_layer_group or {})
        self._attention_dp_rank = attention_dp_rank
        self._attention_dp_gather = attention_dp_gather
        self._hash_algo = hash_algo
        self._next_event_id = 0
        self._stored_blocks: dict[bytes, _StoredBlockState] = {}
        self._latest_stored_events: dict[LayerGroupId, KVCacheEvent] = {}
        self._latest_removed_block_hashes: dict[LayerGroupId, list[EventBlockHash]] = {}
        self._pending_events: list[KVCacheEvent] = []
        self._events: deque[KVCacheEvent] = deque()
        self._condition = Condition()
        self._v1_hash_by_block_key: dict[bytes, int] = {}
        self._v1_hash_compatible_keys: set[bytes] = set()
        self._v1_root_attrs_by_block_key: dict[bytes, tuple[int | None, int | None]] = {}
        self._warned_v1_hash_fallback = False

    def add_created_event(
        self,
        num_blocks_per_cache_level: Sequence[int],
        layer_group_ids: Sequence[int] | None = None,
    ) -> None:
        data = KVCacheCreatedData(list(num_blocks_per_cache_level))
        if layer_group_ids is None:
            self._add_event(data)
            return
        for layer_group_id in layer_group_ids:
            self._add_event(data, layer_group_id=int(layer_group_id))

    def set_layer_group_window_sizes(self, window_sizes: dict[int, int]) -> None:
        with self._condition:
            self._window_size_by_layer_group = dict(window_sizes)

    def add_stored_event(
        self,
        parent_hash: EventBlockHash | None,
        blocks: Sequence[KVCacheStoredBlockData],
        layer_group_id: int | None = None,
    ) -> None:
        if not blocks:
            return
        self._flush_removed_events(layer_group_id)
        self._add_stored_event(
            KVCacheStoredData(parent_hash, list(blocks)),
            layer_group_id=layer_group_id,
        )

    def add_stored_block_event_from_block(self, block: Any) -> None:
        life_cycle_ids = self._life_cycle_ids_from_radix_block(block)
        if not life_cycle_ids:
            return
        parent_hash = self._parent_hash_from_radix_block(block)
        self._stored_blocks[block.key] = _StoredBlockState(
            block_hash=self._hash_from_radix_block(block),
            life_cycle_ids=set(life_cycle_ids),
        )
        for life_cycle_id in sorted(life_cycle_ids):
            block_data = self._stored_block_from_radix_block(block, life_cycle_ids={life_cycle_id})
            if block_data is not None:
                self.add_stored_event(parent_hash, [block_data], life_cycle_id)

    def add_stored_life_cycle_event_from_block(self, block: Any, life_cycle_id: int) -> None:
        state = self._stored_blocks.get(block.key)
        life_cycle_id = int(life_cycle_id)
        if state is not None:
            if life_cycle_id in state.life_cycle_ids:
                return
            block_data = self._stored_block_from_radix_block(block, life_cycle_ids={life_cycle_id})
            if block_data is None:
                return
            state.life_cycle_ids.add(life_cycle_id)
            self.add_stored_event(
                self._parent_hash_from_radix_block(block),
                [block_data],
                layer_group_id=life_cycle_id,
            )
            return
        self.add_stored_block_event_from_block(block)

    def add_removed_event(self, block_hashes: BlockHashesLike) -> None:
        removed_block_hashes_by_layer_group: dict[int, list[EventBlockHash]] = {}
        removed_block_hashes_without_layer_group: list[EventBlockHash] = []
        for block_hash in self._iter_block_hashes(block_hashes):
            removed_state = self._pop_stored_block_state(block_hash)
            if removed_state is None:
                continue
            normalized_hash, life_cycle_ids = removed_state
            if life_cycle_ids:
                for life_cycle_id in sorted(life_cycle_ids):
                    removed_block_hashes_by_layer_group.setdefault(life_cycle_id, []).append(
                        normalized_hash
                    )
            else:
                removed_block_hashes_without_layer_group.append(normalized_hash)

        if removed_block_hashes_without_layer_group:
            self._enqueue_removed_event(removed_block_hashes_without_layer_group)
        for layer_group_id, removed_block_hashes in sorted(
            removed_block_hashes_by_layer_group.items()
        ):
            self._enqueue_removed_event(removed_block_hashes, layer_group_id=layer_group_id)

    def add_removed_life_cycle_event(self, block_hash: bytes, life_cycle_id: int) -> None:
        removed_state = self._pop_stored_life_cycle_block_state(block_hash, life_cycle_id)
        if removed_state is None:
            return
        normalized_hash, removed_life_cycle_id, _ = removed_state
        self._enqueue_removed_event(
            [normalized_hash],
            layer_group_id=removed_life_cycle_id,
        )

    def add_updated_event(
        self,
        block_hash: BlockHashLike,
        *,
        cache_level: KVCacheEventDiff | None = None,
        priority: KVCacheEventDiff | None = None,
        layer_group_id: int | None = None,
    ) -> None:
        if cache_level is None and priority is None:
            return
        normalized_block_hash = self._get_stored_block_hash(block_hash)
        if normalized_block_hash is None:
            return
        self._add_event(
            KVCacheUpdatedData(
                block_hash=normalized_block_hash,
                cache_level=cache_level,
                priority=priority,
            ),
            layer_group_id=layer_group_id,
        )

    def flush_iteration_events(self) -> None:
        if self._attention_dp_gather is not None:
            with self._condition:
                local_events = self._drain_pending_events_unlocked()
                local_events = self._trim_events(local_events, self._max_kv_event_entries)
            gathered_events = self._attention_dp_gather(local_events)
            if self._attention_dp_rank != 0:
                return
            events = [
                event
                for rank_events in gathered_events
                for event in self._trim_events(rank_events, self._max_kv_event_entries)
            ]
            with self._condition:
                self._publish_events_unlocked(
                    events,
                    max_kv_event_entries=(
                        self._max_kv_event_entries * max(1, len(gathered_events))
                    ),
                )
                self._condition.notify_all()
            return

        with self._condition:
            self._publish_events_unlocked(self._drain_pending_events_unlocked())
            self._condition.notify_all()

    def get_latest_events(self, timeout_ms: float | None = None) -> list[KVCacheEvent]:
        with self._condition:
            if not self._events and timeout_ms is None:
                while not self._events:
                    self._condition.wait()
            elif not self._events and timeout_ms > 0:
                deadline = time.monotonic() + timeout_ms / 1000
                while not self._events:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._condition.wait(timeout=remaining)

            events = list(self._events)
            self._events.clear()
            return events

    def _add_event(
        self,
        data: KVCacheCreatedData | KVCacheStoredData | KVCacheRemovedData | KVCacheUpdatedData,
        layer_group_id: LayerGroupId = None,
    ) -> None:
        if self._max_kv_event_entries <= 0:
            return
        with self._condition:
            self._add_event_unlocked(data, layer_group_id)

    def _add_stored_event(
        self,
        data: KVCacheStoredData,
        layer_group_id: LayerGroupId = None,
    ) -> None:
        if self._max_kv_event_entries <= 0:
            return
        with self._condition:
            has_pending_removed_events = bool(self._latest_removed_block_hashes)
            latest_event = self._latest_stored_events.get(layer_group_id)
            if (
                not has_pending_removed_events
                and latest_event is not None
                and isinstance(latest_event.data, KVCacheStoredData)
            ):
                latest_blocks = latest_event.data.blocks
                if latest_blocks and latest_blocks[-1].block_hash == data.parent_hash:
                    merged_data = replace(
                        latest_event.data,
                        blocks=[*latest_blocks, *data.blocks],
                    )
                    merged_event = replace(latest_event, data=merged_data)
                    self._replace_pending_event_unlocked(latest_event, merged_event)
                    self._latest_stored_events[layer_group_id] = merged_event
                    return

            event = self._add_event_unlocked(data, layer_group_id)
            self._latest_stored_events[layer_group_id] = event

    def _replace_pending_event_unlocked(
        self,
        old_event: KVCacheEvent,
        new_event: KVCacheEvent,
    ) -> None:
        for event_idx in range(len(self._pending_events) - 1, -1, -1):
            if self._pending_events[event_idx] is old_event:
                self._pending_events[event_idx] = new_event
                return
        raise RuntimeError("Stored event coalescing lost the pending event")

    def _enqueue_removed_event(
        self,
        block_hashes: Sequence[EventBlockHash],
        layer_group_id: LayerGroupId = None,
    ) -> None:
        if not block_hashes or self._max_kv_event_entries <= 0:
            return
        with self._condition:
            self._latest_removed_block_hashes.setdefault(layer_group_id, []).extend(block_hashes)
            self._latest_stored_events.pop(layer_group_id, None)

    def _flush_removed_events(self, layer_group_id: LayerGroupId) -> None:
        if self._max_kv_event_entries <= 0:
            return
        with self._condition:
            self._flush_removed_events_unlocked(layer_group_id)

    def _flush_removed_events_unlocked(self, layer_group_id: LayerGroupId) -> None:
        block_hashes = self._latest_removed_block_hashes.pop(layer_group_id, None)
        if not block_hashes:
            return
        self._add_event_unlocked(
            KVCacheRemovedData(block_hashes),
            layer_group_id=layer_group_id,
        )

    def _flush_all_removed_events_unlocked(self) -> None:
        layer_group_ids = list(self._latest_removed_block_hashes)
        for layer_group_id in layer_group_ids:
            self._flush_removed_events_unlocked(layer_group_id)

    def _add_event_unlocked(
        self,
        data: KVCacheCreatedData | KVCacheStoredData | KVCacheRemovedData | KVCacheUpdatedData,
        layer_group_id: LayerGroupId = None,
    ) -> KVCacheEvent:
        if not isinstance(data, KVCacheRemovedData):
            self._flush_all_removed_events_unlocked()
        event = KVCacheEvent(
            event_id=self._next_event_id,
            data=data,
            window_size=self._get_window_size(layer_group_id),
            hash_algo=self._hash_algo,
            attention_dp_rank=self._attention_dp_rank,
            layer_group_id=layer_group_id,
        )
        self._next_event_id += 1
        self._pending_events.append(event)
        if not isinstance(data, KVCacheStoredData):
            self._latest_stored_events.pop(layer_group_id, None)
        return event

    def _drain_pending_events_unlocked(self) -> list[KVCacheEvent]:
        self._flush_all_removed_events_unlocked()
        events = self._pending_events
        self._pending_events = []
        self._latest_stored_events.clear()
        return events

    def _publish_events_unlocked(
        self,
        events: Sequence[KVCacheEvent],
        *,
        max_kv_event_entries: int | None = None,
    ) -> None:
        if not events:
            return
        if max_kv_event_entries is None:
            max_kv_event_entries = self._max_kv_event_entries
        self._events.extend(events)
        while len(self._events) > max_kv_event_entries:
            self._events.popleft()

    @staticmethod
    def _trim_events(
        events: Sequence[KVCacheEvent], max_kv_event_entries: int
    ) -> list[KVCacheEvent]:
        if max_kv_event_entries <= 0:
            return []
        if len(events) <= max_kv_event_entries:
            return list(events)
        return list(events[-max_kv_event_entries:])

    def _get_window_size(self, layer_group_id: LayerGroupId) -> int:
        if layer_group_id is None:
            return self._window_size
        return self._window_size_by_layer_group.get(int(layer_group_id), self._window_size)

    @staticmethod
    def _iter_block_hashes(block_hashes: BlockHashesLike) -> Iterable[BlockHashLike]:
        if isinstance(block_hashes, (bytes, str, int)):
            return (block_hashes,)
        return block_hashes

    def _normalize_block_hash(self, block_hash: BlockHashLike) -> EventBlockHash:
        if isinstance(block_hash, bytes):
            if self._hash_algo == KV_CACHE_HASH_ALGO_V2_SHA256_64:
                return truncate_sha256_hash_to_int64(block_hash)
            return block_hash.hex()
        return block_hash

    def _get_stored_block_hash(self, block_hash: BlockHashLike) -> EventBlockHash | None:
        if isinstance(block_hash, bytes):
            state = self._stored_blocks.get(block_hash)
            return None if state is None else state.block_hash
        return block_hash

    def _pop_stored_block_state(
        self, block_hash: BlockHashLike
    ) -> tuple[EventBlockHash, set[int]] | None:
        if isinstance(block_hash, bytes):
            state = self._stored_blocks.pop(block_hash, None)
            if state is None:
                return None
            self._drop_hash_cache(block_hash)
            return state.block_hash, set(state.life_cycle_ids)
        return block_hash, set()

    def _pop_stored_life_cycle_block_state(
        self, block_hash: bytes, life_cycle_id: int
    ) -> tuple[EventBlockHash, int, bool] | None:
        state = self._stored_blocks.get(block_hash)
        if state is None or not state.life_cycle_ids:
            return None

        life_cycle_id = int(life_cycle_id)
        if life_cycle_id not in state.life_cycle_ids:
            return None

        state.life_cycle_ids.remove(life_cycle_id)
        is_last_life_cycle = not state.life_cycle_ids
        if is_last_life_cycle:
            self._stored_blocks.pop(block_hash, None)
            self._drop_hash_cache(block_hash)
        return state.block_hash, life_cycle_id, is_last_life_cycle

    def _drop_hash_cache(self, block_hash: bytes) -> None:
        self._v1_hash_by_block_key.pop(block_hash, None)
        self._v1_hash_compatible_keys.discard(block_hash)
        self._v1_root_attrs_by_block_key.pop(block_hash, None)

    @staticmethod
    def _normalize_token(token: TokenIdExt) -> UniqueToken:
        if isinstance(token, bytes):
            return UniqueToken(token.hex())
        return UniqueToken(int(token))

    def _stored_block_from_radix_block(
        self, block: Any, life_cycle_ids: set[int] | None = None
    ) -> KVCacheStoredBlockData | None:
        cache_level: CacheLevel = GPU_LEVEL
        priority: Priority = PRIORITY_DEFAULT
        found_page = False
        for life_cycle_id, page_ref in enumerate(block.storage):
            if life_cycle_ids is not None and life_cycle_id not in life_cycle_ids:
                continue
            if page_ref is None:
                continue
            page = page_ref()
            if page is None:
                continue
            cache_level = page.cache_level
            priority = page.priority
            found_page = True
            break

        if life_cycle_ids is not None and not found_page:
            return None

        return KVCacheStoredBlockData(
            block_hash=self._hash_from_radix_block(block),
            tokens=[self._normalize_token(token) for token in block.tokens],
            cache_level=int(cache_level),
            priority=int(priority),
            mm_keys=[],
        )

    @staticmethod
    def _life_cycle_ids_from_radix_block(block: Any) -> set[int]:
        return {
            life_cycle_id
            for life_cycle_id, page_ref in enumerate(block.storage)
            if page_ref is not None and page_ref() is not None
        }

    def _parent_hash_from_radix_block(self, block: Any) -> EventBlockHash | None:
        parent = block.prev
        if getattr(parent, "ordinal", -1) == -1:
            return None
        return self._hash_from_radix_block(parent)

    def _hash_from_radix_block(self, block: Any) -> EventBlockHash:
        if self._hash_algo == KV_CACHE_HASH_ALGO_V1:
            return self._v1_hash_from_radix_block(block)
        return self._normalize_block_hash(block.key)

    def _v1_hash_from_radix_block(self, block: Any) -> int:
        key = bytes(block.key)
        cached = self._v1_hash_by_block_key.get(key)
        if cached is not None:
            return cached

        chain: list[Any] = []
        current = block
        while self._is_radix_block(current):
            current_key = bytes(current.key)
            cached = self._v1_hash_by_block_key.get(current_key)
            if cached is not None:
                parent_hash = cached
                parent_is_v1_compatible = current_key in self._v1_hash_compatible_keys
                root_attrs = self._v1_root_attrs_by_block_key[current_key]
                break
            chain.append(current)
            current = current.prev

        if not self._is_radix_block(current):
            parent_hash = 0
            parent_is_v1_compatible = True
            root_attrs = self._root_attrs_from_root_block(current)

        lora_task_id, cache_salt_id = root_attrs
        for current in reversed(chain):
            current_key = bytes(current.key)
            if parent_is_v1_compatible:
                try:
                    parent_hash = self._hash_block_key(
                        current.tokens,
                        parent_hash,
                        lora_task_id,
                        cache_salt_id,
                    )
                    self._v1_hash_compatible_keys.add(current_key)
                except NonTextTokenHashError:
                    parent_hash = self._fallback_v1_hash(current_key)
                    parent_is_v1_compatible = False
            else:
                parent_hash = self._fallback_v1_hash(current_key)
            self._v1_hash_by_block_key[current_key] = parent_hash
            self._v1_root_attrs_by_block_key[current_key] = root_attrs

        return parent_hash

    def _fallback_v1_hash(self, block_key: bytes) -> int:
        if not self._warned_v1_hash_fallback:
            logger.warning(
                "V2 KV cache event hash algorithm %s only matches v1 for "
                "text-token radix blocks. Falling back to truncated V2 block "
                "hash for unsupported blocks.",
                KV_CACHE_HASH_ALGO_V1,
            )
            self._warned_v1_hash_fallback = True
        return truncate_sha256_hash_to_int64(block_key)

    @staticmethod
    def _is_radix_block(value: Any) -> bool:
        return hasattr(value, "tokens") and hasattr(value, "key")

    @staticmethod
    def _root_attrs_from_root_block(root: Any) -> tuple[int | None, int | None]:
        return getattr(root, "lora_task_id", None), getattr(root, "cache_salt_id", None)

    @staticmethod
    def _hash_block_key(
        tokens: Sequence[TokenIdExt],
        parent_hash: int,
        lora_task_id: int | None,
        cache_salt_id: int | None,
    ) -> int:
        return hash_v1_block_key(
            tokens,
            parent_hash=parent_hash,
            lora_task_id=lora_task_id,
            cache_salt_id=cache_salt_id,
        )
