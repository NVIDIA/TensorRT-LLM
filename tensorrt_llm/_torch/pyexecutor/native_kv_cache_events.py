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

"""Native KVCM2 event translation and background publication."""

from __future__ import annotations

import threading
import time
import traceback
from typing import Any

from tensorrt_llm.llmapi.llm_args import KVEventsConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    KVCacheCreatedData,
    KVCacheEvent,
    KVCacheRemovedData,
    KVCacheStoredData,
    KVCacheUpdatedData,
)

from .kv_cache_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVEventBatch,
    create_event_publisher,
)

NATIVE_EVENT_QUEUE_CAPACITY = 4_096
NATIVE_READER_MAX_EVENTS = 64
NATIVE_READER_POLL_MS = 1_000.0


def _signed_wire_hash(block_hash: int) -> int:
    """Preserve a native uint64 block hash in Dynamo's signed-int64 wire domain."""
    if isinstance(block_hash, bool) or not isinstance(block_hash, int):
        raise ValueError("Dynamo KV routing requires integer v1_block_key hashes")
    if block_hash < 0:
        if block_hash < -(2**63):
            raise ValueError(f"KV block hash {block_hash} does not fit in int64")
        return block_hash
    if block_hash >= 2**64:
        raise ValueError(f"KV block hash {block_hash} does not fit in uint64")
    return block_hash - 2**64 if block_hash >= 2**63 else block_hash


def _cache_level_medium(cache_level: int) -> str | None:
    if cache_level == 0:
        return "GPU"
    if cache_level == 1:
        return "CPU"
    if cache_level >= 2:
        return "DISK"
    return None


class _NativeEventTranslator:
    """Translate native KVCM2 events into the existing Dynamo msgpack contract."""

    def __init__(
        self,
        *,
        block_size: int,
        max_window_size: int,
        target_layer_group_id: int | None,
    ) -> None:
        self._block_size = block_size
        self._max_window_size = max_window_size
        self._target_layer_group_id = target_layer_group_id
        self._last_event_id: int | None = None
        self._accept_any_next_event_id = False
        self._suppressed_blocks: set[int] = set()

    def reset_after_loss(self) -> None:
        self._last_event_id = None
        self._accept_any_next_event_id = True
        self._suppressed_blocks.clear()

    def translate(
        self, events: list[KVCacheEvent]
    ) -> list[BlockStored | BlockRemoved | AllBlocksCleared]:
        output: list[BlockStored | BlockRemoved | AllBlocksCleared] = []
        for event in events:
            event_id = int(event.event_id)
            expected = 0 if self._last_event_id is None else self._last_event_id + 1
            if not self._accept_any_next_event_id and event_id != expected:
                self._suppressed_blocks.clear()
                output.append(AllBlocksCleared())
            self._accept_any_next_event_id = False
            self._last_event_id = event_id

            if event.hash_algo not in (None, "v1_block_key"):
                raise ValueError(
                    f"Dynamo KV routing requires v1_block_key hashes, got {event.hash_algo!r}"
                )
            if self._max_window_size and int(event.window_size) != self._max_window_size:
                continue
            group_idx = None if event.layer_group_id is None else int(event.layer_group_id)
            if self._target_layer_group_id is not None and group_idx != self._target_layer_group_id:
                continue
            # A nanobind variant conversion materializes a nested Python object.
            # Apply scalar filters first so rejected lifecycle events stay native.
            data = event.data
            if isinstance(data, KVCacheCreatedData):
                continue

            if isinstance(data, KVCacheStoredData):
                output.extend(self._stored(data, group_idx))
            elif isinstance(data, KVCacheRemovedData):
                output.extend(self._removed(data, group_idx))
            elif isinstance(data, KVCacheUpdatedData):
                output.extend(self._updated(data, group_idx))
            else:
                raise ValueError(f"Unsupported native KV event data: {type(data).__name__}")
        return output

    def _stored(self, data: KVCacheStoredData, group_idx: int | None) -> list[BlockStored]:
        output: list[BlockStored] = []
        parent_hash = None if data.parent_hash is None else _signed_wire_hash(data.parent_hash)
        if parent_hash in self._suppressed_blocks:
            self._suppressed_blocks.update(
                _signed_wire_hash(block.block_hash) for block in data.blocks
            )
            return output
        group_parent_hash = parent_hash
        group_cache_level: int | None = None
        group_hashes: list[int] = []
        group_token_ids: list[int] = []

        def flush_group() -> None:
            nonlocal group_parent_hash, group_cache_level, group_hashes, group_token_ids
            if not group_hashes or group_cache_level is None:
                return
            output.append(
                BlockStored(
                    block_hashes=group_hashes,
                    parent_block_hash=group_parent_hash,
                    token_ids=group_token_ids,
                    block_size=self._block_size,
                    lora_id=None,
                    medium=_cache_level_medium(group_cache_level),
                    lora_name=None,
                    group_idx=group_idx,
                )
            )
            group_parent_hash = parent_hash
            group_cache_level = None
            group_hashes = []
            group_token_ids = []

        for block in data.blocks:
            block_hash = _signed_wire_hash(block.block_hash)
            cache_level = int(block.cache_level)
            if cache_level != 0:
                # Dynamo routing currently targets directly reusable GPU KV.
                # Suppressing cold stores avoids retaining per-block medium
                # state solely to emit a matching later removal.
                self._suppressed_blocks.add(block_hash)
                break
            tokens = block.tokens
            if len(tokens) != self._block_size:
                self._suppressed_blocks.add(block_hash)
                break
            if block.cache_salt is not None or block.mm_keys:
                self._suppressed_blocks.add(block_hash)
                break
            token_ids: list[int] = []
            unsupported_tokens = False
            for token in tokens:
                if int(token.token_extra_id) != 0:
                    unsupported_tokens = True
                    break
                token_id = token.token_id
                if isinstance(token_id, bool) or not isinstance(token_id, int):
                    unsupported_tokens = True
                    break
                if token_id < 0 or token_id > 0xFFFFFFFF:
                    raise ValueError(f"KV token id {token_id} does not fit in uint32")
                token_ids.append(token_id)
            if unsupported_tokens:
                self._suppressed_blocks.add(block_hash)
                break

            if group_cache_level is not None and group_cache_level != cache_level:
                flush_group()
                group_parent_hash = parent_hash
            if group_cache_level is None:
                group_cache_level = cache_level
                group_parent_hash = parent_hash
            group_hashes.append(block_hash)
            group_token_ids.extend(token_ids)
            self._suppressed_blocks.discard(block_hash)
            parent_hash = block_hash
        flush_group()
        return output

    def _removed(self, data: KVCacheRemovedData, group_idx: int | None) -> list[BlockRemoved]:
        grouped: dict[str | None, list[int]] = {}
        for raw_hash in data.block_hashes:
            block_hash = _signed_wire_hash(raw_hash)
            if block_hash in self._suppressed_blocks:
                self._suppressed_blocks.remove(block_hash)
                continue
            medium = "GPU"
            grouped.setdefault(medium, []).append(block_hash)
        return [
            BlockRemoved(block_hashes=hashes, medium=medium, group_idx=group_idx)
            for medium, hashes in grouped.items()
        ]

    def _updated(self, data: KVCacheUpdatedData, group_idx: int | None) -> list[BlockRemoved]:
        if data.cache_level is None:
            return []
        block_hash = _signed_wire_hash(data.block_hash)
        if block_hash in self._suppressed_blocks:
            return []
        old_level = int(data.cache_level.old_value)
        # The update payload has no tokens/parent metadata with which to rebuild a
        # Stored event at the new tier. Conservatively remove the old location
        # instead of retaining a Python token graph for every live cache block.
        return [
            BlockRemoved(
                block_hashes=[block_hash],
                medium=_cache_level_medium(old_level),
                group_idx=group_idx,
            ),
        ]


class NativeKVCacheEventPublisher:
    """Drain the native KVCM2 queue and publish it without frontend RPC polling."""

    def __init__(
        self,
        config: KVEventsConfig,
        event_manager: Any,
        *,
        data_parallel_rank: int,
        block_size: int,
        max_window_size: int,
        window_sizes_by_layer_group: dict[int, int],
    ) -> None:
        target_ids = [
            int(group_idx)
            for group_idx, window_size in window_sizes_by_layer_group.items()
            if int(window_size) == max_window_size
        ]
        self._event_manager = event_manager
        self._publisher = create_event_publisher(config, data_parallel_rank)
        self._translator = _NativeEventTranslator(
            block_size=block_size,
            max_window_size=max_window_size,
            target_layer_group_id=min(target_ids) if target_ids else None,
        )
        self._rank = data_parallel_rank
        self._thread: threading.Thread | None = None
        self._shutdown_lock = threading.Lock()
        self._closed = False
        self.enqueued_batches = 0
        self.enqueued_events = 0
        self.dropped_batches = 0
        self.translation_errors = 0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._publisher.start()
        try:
            self._thread = threading.Thread(
                target=self._reader_thread,
                daemon=True,
                name=f"trtllm-native-kv-events-rank-{self._rank}",
            )
            self._thread.start()
        except Exception:
            self._thread = None
            self._publisher.shutdown()
            raise

    def shutdown(self) -> None:
        with self._shutdown_lock:
            if self._closed:
                return
            self._event_manager.close()
            self._closed = True
        if self._thread is not None:
            self._thread.join()
        self._publisher.shutdown()
        logger.info(
            f"Stopped native KV event reader rank={self._rank} "
            f"enqueued_batches={self.enqueued_batches} "
            f"enqueued_events={self.enqueued_events} "
            f"dropped_batches={self.dropped_batches} "
            f"translation_errors={self.translation_errors} "
            f"native_dropped_events={self._event_manager.dropped_event_count} "
            f"native_queue_high_watermark={self._event_manager.queue_high_watermark}"
        )

    def _reader_thread(self) -> None:
        observed_native_drops = 0
        while True:
            try:
                events = self._event_manager.get_latest_events(
                    timeout_ms=NATIVE_READER_POLL_MS,
                    max_events=NATIVE_READER_MAX_EVENTS,
                )
            except Exception:
                self.translation_errors += 1
                self._publish_clear()
                logger.error(
                    f"Native KV event reader failed on rank={self._rank}\n{traceback.format_exc()}"
                )
                if self._closed:
                    return
                time.sleep(0.1)
                continue
            native_drops = self._event_manager.dropped_event_count
            if native_drops != observed_native_drops:
                observed_native_drops = self._event_manager.discard_events()
                self._translator.reset_after_loss()
                self._publish_clear()
                # Native drops can remove either older queued events or newer
                # pending events. Discard this read on either case so no event
                # from before the recovery fence can rebuild stale router state.
                continue
            if not events:
                if self._event_manager.closed_and_empty:
                    return
                continue
            try:
                wire_events = self._translator.translate(events)
            except (TypeError, ValueError):
                self.translation_errors += 1
                self._publish_clear()
                logger.error(
                    f"Dropping malformed native KV event batch on rank={self._rank}\n"
                    f"{traceback.format_exc()}"
                )
                continue
            if not wire_events:
                continue
            batch = KVEventBatch(
                ts=time.time(),
                events=wire_events,
                data_parallel_rank=self._rank,
            )
            if self._publisher.publish(batch):
                self.enqueued_batches += 1
                self.enqueued_events += len(wire_events)
            else:
                self.dropped_batches += 1

    def _publish_clear(self) -> None:
        batch = KVEventBatch(
            ts=time.time(),
            events=[AllBlocksCleared()],
            data_parallel_rank=self._rank,
        )
        if self._publisher.publish(batch):
            self.enqueued_batches += 1
            self.enqueued_events += 1
        else:
            self.dropped_batches += 1
