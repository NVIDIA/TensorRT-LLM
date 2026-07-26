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
#
# The wire schema and ZeroMQ framing in this file are adapted from vLLM's
# vllm/distributed/kv_events.py.

from __future__ import annotations

import queue
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from itertools import count
from queue import Queue
from typing import Any, Optional

import msgspec
import zmq

from tensorrt_llm.llmapi.llm_args import KVEventsConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime.kv_cache_manager_v2._event_manager import (
    KVCacheCreatedData,
    KVCacheEvent,
    KVCacheEventDiff,
    KVCacheEventManager,
    KVCacheRemovedData,
    KVCacheStoredData,
    KVCacheUpdatedData,
)

ExternalBlockHash = bytes | int


class EventBatch(
        msgspec.Struct,
        array_like=True,  # type: ignore[call-arg]
        omit_defaults=True,  # type: ignore[call-arg]
        gc=False,  # type: ignore[call-arg]
):
    """vLLM-compatible event batch envelope."""

    ts: float
    events: list[Any]
    data_parallel_rank: int | None = None


class KVCacheWireEvent(
        msgspec.Struct,
        omit_defaults=True,  # type: ignore[call-arg]
        gc=False,  # type: ignore[call-arg]
        tag=True,
):
    """Base class for vLLM-compatible KV cache events."""


class BlockStored(KVCacheWireEvent):
    """A sequence of full KV cache blocks was stored."""

    block_hashes: list[ExternalBlockHash]
    parent_block_hash: ExternalBlockHash | None
    token_ids: list[int]
    block_size: int
    lora_id: int | None
    medium: str | None
    lora_name: str | None
    extra_keys: list[tuple[Any, ...] | None] | None = None
    group_idx: int | None = None
    kv_cache_spec_kind: str | None = None
    kv_cache_spec_sliding_window: int | None = None
    locality: str | None = None


class BlockRemoved(KVCacheWireEvent):
    """A sequence of KV cache blocks was removed."""

    block_hashes: list[ExternalBlockHash]
    medium: str | None
    group_idx: int | None = None
    locality: str | None = None


class AllBlocksCleared(KVCacheWireEvent):
    """All KV cache blocks were cleared."""


class KVEventBatch(EventBatch):
    """A batch containing only KV cache lifecycle events."""

    events: list[BlockStored | BlockRemoved | AllBlocksCleared]


class EventPublisher(ABC):
    """Publishes vLLM-compatible event batches for one cache rank."""

    def __init__(self, data_parallel_rank: int = 0) -> None:
        self._data_parallel_rank = data_parallel_rank

    @abstractmethod
    def publish(self, events: EventBatch) -> bool:
        """Enqueue an event batch without blocking the scheduler."""

    @abstractmethod
    def shutdown(self) -> None:
        """Flush pending batches and stop the publisher."""


class NullEventPublisher(EventPublisher):
    """Drains event batches locally without external I/O."""

    def publish(self, events: EventBatch) -> bool:
        return True

    def shutdown(self) -> None:
        return


class ZmqEventPublisher(EventPublisher):
    """Publishes event batches with vLLM's three-frame ZeroMQ protocol."""

    SHUTDOWN_TIMEOUT = 1.0
    END_SEQ = (-1).to_bytes(8, "big", signed=True)

    def __init__(
        self,
        data_parallel_rank: int,
        endpoint: str = "tcp://*:5557",
        replay_endpoint: str | None = None,
        buffer_steps: int = 10_000,
        hwm: int = 100_000,
        max_queue_size: int = 100_000,
        topic: str = "",
    ) -> None:
        super().__init__(data_parallel_rank)
        self._event_queue = Queue[EventBatch | None](maxsize=max_queue_size)
        self._buffer = deque[tuple[int, bytes]](maxlen=buffer_steps)
        self._ctx = zmq.Context.instance()
        self._pub: Optional[zmq.Socket] = None
        self._replay: Optional[zmq.Socket] = None
        self._rank = data_parallel_rank
        self._endpoint = self.offset_endpoint_port(endpoint, self._rank)
        self._replay_endpoint = self.offset_endpoint_port(
            replay_endpoint, self._rank)
        self._hwm = hwm
        self._seq_gen = count()
        self._topic_bytes = topic.encode("utf-8")
        self._running = True
        self._shutdown_lock = threading.Lock()
        self.enqueued_batches = 0
        self.published_batches = 0
        self.dropped_batches = 0
        self._socket_setup()
        self._thread = threading.Thread(
            target=self._publisher_thread,
            daemon=True,
            name=f"trtllm-kv-events-rank-{self._rank}",
        )
        self._thread.start()
        logger.info(f"Started native KV event publisher rank={self._rank} "
                    f"endpoint={self._endpoint} topic={topic!r}")

    def publish(self, events: EventBatch) -> bool:
        if not self._running:
            return False
        if events.data_parallel_rank is None:
            events.data_parallel_rank = self._data_parallel_rank
        try:
            self._event_queue.put_nowait(events)
            self.enqueued_batches += 1
            return True
        except queue.Full:
            self.dropped_batches += 1
            if self.dropped_batches == 1 or (self.dropped_batches &
                                             (self.dropped_batches - 1) == 0):
                logger.warning(
                    f"Dropping native KV event batch on rank={self._rank} because "
                    "the publisher queue is full; "
                    f"dropped_batches={self.dropped_batches}")
            return False

    def shutdown(self) -> None:
        with self._shutdown_lock:
            if not self._running:
                return
            self._running = False
            try:
                self._event_queue.put_nowait(None)
            except queue.Full:
                # The thread exits after draining the full queue.
                pass
        self._thread.join(timeout=self.SHUTDOWN_TIMEOUT)
        if self._thread.is_alive():
            logger.warning(
                f"Native KV event publisher rank={self._rank} did not stop "
                f"within {self.SHUTDOWN_TIMEOUT:.1f}s")
        logger.info(f"Stopped native KV event publisher rank={self._rank} "
                    f"enqueued_batches={self.enqueued_batches} "
                    f"published_batches={self.published_batches} "
                    f"dropped_batches={self.dropped_batches}")

    def _socket_setup(self) -> None:
        self._pub = self._ctx.socket(zmq.PUB)
        self._pub.set_hwm(self._hwm)
        if self._endpoint is None:
            raise ValueError("KV event publisher endpoint must not be empty")
        if ("*" in self._endpoint or "::" in self._endpoint
                or self._endpoint.startswith(("ipc://", "inproc://"))):
            self._pub.bind(self._endpoint)
        else:
            self._pub.connect(self._endpoint)

        if self._replay_endpoint is not None:
            self._replay = self._ctx.socket(zmq.ROUTER)
            self._replay.bind(self._replay_endpoint)

    def _publisher_thread(self) -> None:
        encoder = msgspec.msgpack.Encoder()
        assert self._pub is not None
        try:
            while self._running or not self._event_queue.empty():
                if self._replay is not None and self._replay.poll(0):
                    try:
                        self._service_replay()
                    except Exception:
                        logger.exception(
                            "Failed to service native KV event replay request")
                try:
                    event = self._event_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if event is None:
                    self._event_queue.task_done()
                    break
                seq = next(self._seq_gen)
                try:
                    payload = encoder.encode(event)
                    self._pub.send_multipart((
                        self._topic_bytes,
                        seq.to_bytes(8, "big"),
                        payload,
                    ))
                    self._buffer.append((seq, payload))
                    self.published_batches += 1
                except Exception:
                    self.dropped_batches += 1
                    logger.exception(f"Failed to publish native KV event batch "
                                     f"rank={self._rank} seq={seq}")
                    time.sleep(0.1)
                finally:
                    self._event_queue.task_done()
        finally:
            self._pub.close(linger=0)
            if self._replay is not None:
                self._replay.close(linger=0)

    def _service_replay(self) -> None:
        assert self._replay is not None
        frame = self._replay.recv_multipart()
        if len(frame) != 3:
            logger.warning(f"Invalid native KV event replay request: {frame}")
            return
        client_id, _, start_seq_bytes = frame
        start_seq = int.from_bytes(start_seq_bytes, "big")
        for seq, payload in self._buffer:
            if seq >= start_seq:
                self._replay.send_multipart((
                    client_id,
                    b"",
                    self._topic_bytes,
                    seq.to_bytes(8, "big"),
                    payload,
                ))
        self._replay.send_multipart((client_id, b"", b"", self.END_SEQ, b""))

    @staticmethod
    def offset_endpoint_port(endpoint: str | None,
                             data_parallel_rank: int) -> str | None:
        """Apply vLLM's base-port-plus-rank endpoint convention."""
        if not endpoint or data_parallel_rank == 0:
            return endpoint
        if "inproc" in endpoint:
            return f"{endpoint}_dp{data_parallel_rank}"
        if "tcp" in endpoint and ":" in endpoint:
            last_colon_idx = endpoint.rfind(":")
            base_addr = endpoint[:last_colon_idx]
            base_port = int(endpoint[last_colon_idx + 1:])
            new_port = base_port + data_parallel_rank
            if new_port > 65_535:
                raise ValueError(
                    f"KV event endpoint port exceeds 65535 for rank {data_parallel_rank}"
                )
            return f"{base_addr}:{new_port}"
        raise ValueError("Invalid endpoint: must contain 'inproc' or 'tcp'")


def create_event_publisher(config: KVEventsConfig,
                           data_parallel_rank: int) -> EventPublisher:
    """Create the configured publisher for one cache rank."""
    if config.publisher == "null":
        return NullEventPublisher(data_parallel_rank)
    if config.publisher == "zmq":
        return ZmqEventPublisher(
            data_parallel_rank=data_parallel_rank,
            endpoint=config.endpoint,
            replay_endpoint=config.replay_endpoint,
            buffer_steps=config.buffer_steps,
            hwm=config.hwm,
            max_queue_size=config.max_queue_size,
            topic=config.topic,
        )
    raise ValueError(f"Unsupported KV event publisher: {config.publisher!r}")


def _to_wire_hash(block_hash: int | str | None) -> ExternalBlockHash | None:
    if block_hash is None:
        return None
    if isinstance(block_hash, int):
        if block_hash >= 2**63:
            return block_hash - 2**64
        if block_hash < -(2**63):
            return ((block_hash + 2**63) % 2**64) - 2**63
        return block_hash
    try:
        return bytes.fromhex(block_hash)
    except ValueError as error:
        raise ValueError(
            f"Invalid hexadecimal KV block hash: {block_hash!r}") from error


def _vllm_wire_hash_from_radix_key(block_key: bytes) -> int:
    """Convert an existing SHA-256 radix key like vLLM's integer event hashes."""
    if len(block_key) < 8:
        raise ValueError("V2 radix block keys must contain at least 8 bytes")
    unsigned_hash = int.from_bytes(block_key[-8:], "big", signed=False)
    wire_hash = _to_wire_hash(unsigned_hash)
    assert isinstance(wire_hash, int)
    return wire_hash


class KVEventAdapter:
    """Converts local V2 events and publishes one wire batch per iteration."""

    def __init__(
        self,
        config: KVEventsConfig,
        *,
        data_parallel_rank: int,
        block_size: int,
        max_window_size: int,
    ) -> None:
        self._rank = data_parallel_rank
        self._block_size = block_size
        self._max_window_size = max_window_size
        self._publisher = create_event_publisher(config, data_parallel_rank)
        self._partial_block_hashes: set[int | str] = set()
        self._closed = False
        self.local_batches = 0
        self.local_events = 0
        self.enqueued_batches = 0
        self.enqueued_events = 0
        self.dropped_batches = 0

    def publish_local_events(
            self, events: list[KVCacheEvent]) -> list[list[KVCacheEvent]]:
        """Publish local events and return no gathered events to the manager."""
        if self._closed or not events:
            return []
        self.local_batches += 1
        self.local_events += len(events)
        try:
            wire_events = [
                wire_event for event in events
                if (wire_event := self._convert_event(event)) is not None
            ]
            if wire_events:
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
        except Exception:
            self.dropped_batches += 1
            logger.exception(
                f"Dropping native KV event iteration batch on rank={self._rank}"
            )
        return []

    def publish_wire_events(
        self,
        wire_events: list[BlockStored | BlockRemoved | AllBlocksCleared],
    ) -> None:
        """Enqueue an already-converted local iteration batch."""
        if self._closed or not wire_events:
            return
        self.local_batches += 1
        self.local_events += len(wire_events)
        try:
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
        except Exception:
            self.dropped_batches += 1
            logger.exception(
                f"Dropping native KV event iteration batch on rank={self._rank}"
            )

    def _convert_event(
        self, event: KVCacheEvent
    ) -> BlockStored | BlockRemoved | AllBlocksCleared | None:
        if event.window_size != self._max_window_size:
            return None
        data = event.data
        if isinstance(data, (KVCacheCreatedData, KVCacheUpdatedData)):
            return None
        if isinstance(data, KVCacheStoredData):
            block_hashes: list[ExternalBlockHash] = []
            token_ids: list[int] = []
            for block in data.blocks:
                num_tokens = len(block.tokens)
                if num_tokens > self._block_size:
                    raise ValueError(
                        f"KV block has {num_tokens} tokens, expected at most "
                        f"{self._block_size}")
                if num_tokens < self._block_size:
                    self._partial_block_hashes.add(block.block_hash)
                    break
                block_token_ids = [token.token_id for token in block.tokens]
                if any(not isinstance(token_id, int)
                       for token_id in block_token_ids):
                    raise ValueError(
                        "vLLM-compatible KV events require integer token IDs")
                wire_hash = _to_wire_hash(block.block_hash)
                assert wire_hash is not None
                block_hashes.append(wire_hash)
                token_ids.extend(block_token_ids)
            if not block_hashes:
                return None
            return BlockStored(
                block_hashes=block_hashes,
                parent_block_hash=_to_wire_hash(data.parent_hash),
                token_ids=token_ids,
                block_size=self._block_size,
                lora_id=None,
                medium="GPU",
                lora_name=None,
            )
        if isinstance(data, KVCacheRemovedData):
            block_hashes: list[ExternalBlockHash] = []
            for block_hash in data.block_hashes:
                if block_hash in self._partial_block_hashes:
                    self._partial_block_hashes.remove(block_hash)
                    continue
                wire_hash = _to_wire_hash(block_hash)
                assert wire_hash is not None
                block_hashes.append(wire_hash)
            if not block_hashes:
                return None
            return BlockRemoved(block_hashes=block_hashes, medium="GPU")
        return None

    def shutdown(self) -> None:
        """Close the publisher once and report direct-path counters."""
        if self._closed:
            return
        self._closed = True
        self._publisher.shutdown()
        logger.info(
            f"Native KV events rank={self._rank} "
            f"local_batches={self.local_batches} "
            f"local_events={self.local_events} "
            f"enqueued_batches={self.enqueued_batches} "
            f"enqueued_events={self.enqueued_events} "
            f"dropped_batches={self.dropped_batches} kv_event_allgathers=0")


class _NativeStoredBlockState:
    __slots__ = ("block_hash", )

    def __init__(self, block_hash: int) -> None:
        self.block_hash = block_hash


class NativeKVCacheEventManager(KVCacheEventManager):
    """Scheduler-local fast path that produces vLLM wire events directly."""

    def __init__(
        self,
        adapter: KVEventAdapter,
        *,
        block_size: int,
        max_window_size: int,
        max_entries: int = 50_000,
    ) -> None:
        self._adapter = adapter
        self._block_size = block_size
        self._max_window_size = max_window_size
        self._max_entries = max_entries
        self._target_life_cycle_id: int | None = None
        self._stored_blocks: dict[bytes, _NativeStoredBlockState] = {}
        self._pending_events: list[
            BlockStored | BlockRemoved | AllBlocksCleared] = []
        self._pending_entries = 0
        self._closed = False
        self.stored_blocks = 0
        self.removed_blocks = 0
        self.partial_blocks_suppressed = 0
        self.non_target_life_cycles_ignored = 0
        self.dropped_events = 0

    def set_layer_group_window_sizes(self,
                                     window_sizes: dict[int, int]) -> None:
        target_ids = [
            int(life_cycle_id)
            for life_cycle_id, window_size in window_sizes.items()
            if int(window_size) == self._max_window_size
        ]
        if not target_ids and window_sizes:
            largest_window = max(window_sizes.values())
            target_ids = [
                int(life_cycle_id)
                for life_cycle_id, window_size in window_sizes.items()
                if window_size == largest_window
            ]
        if not target_ids:
            raise ValueError(
                "Native KV events require an attention KV cache life cycle")
        self._target_life_cycle_id = min(target_ids)
        logger.info(
            "Native KV event fast path selected "
            f"lifecycle_id={self._target_life_cycle_id} "
            f"window_size={self._max_window_size}")

    def add_created_event(
        self,
        num_blocks_per_cache_level: Any,
        layer_group_ids: Any = None,
    ) -> None:
        return

    def add_stored_block_event_from_block(self, block: Any) -> None:
        if self._closed or self._target_life_cycle_id is None:
            return
        life_cycle_id = self._target_life_cycle_id
        if life_cycle_id >= len(block.storage):
            return
        page_ref = block.storage[life_cycle_id]
        if page_ref is None or page_ref() is None:
            return
        self._add_full_block(block)

    def add_stored_life_cycle_event_from_block(self, block: Any,
                                                life_cycle_id: int) -> None:
        if int(life_cycle_id) != self._target_life_cycle_id:
            self.non_target_life_cycles_ignored += 1
            return
        self.add_stored_block_event_from_block(block)

    def _add_full_block(self, block: Any) -> None:
        key = bytes(block.key)
        if key in self._stored_blocks:
            return
        if len(block.tokens) != self._block_size:
            self.partial_blocks_suppressed += 1
            return
        if not self._reserve_entries(1):
            return
        try:
            token_ids = self._token_ids(block.tokens)
            block_hash, parent_hash, state = self._block_hashes(block)
        except ValueError:
            self.dropped_events += 1
            self._pending_entries -= 1
            logger.exception(
                "Dropping native KV store event with unsupported token data")
            return
        self._stored_blocks[key] = state
        if self._pending_events and isinstance(self._pending_events[-1],
                                               BlockStored):
            previous = self._pending_events[-1]
            if previous.block_hashes and previous.block_hashes[
                    -1] == parent_hash:
                previous.block_hashes.append(block_hash)
                previous.token_ids.extend(token_ids)
                self.stored_blocks += 1
                return
        self._pending_events.append(
            BlockStored(
                block_hashes=[block_hash],
                parent_block_hash=parent_hash,
                token_ids=token_ids,
                block_size=self._block_size,
                lora_id=None,
                medium="GPU",
                lora_name=None,
            ))
        self.stored_blocks += 1

    @staticmethod
    def _token_ids(tokens: Any) -> list[int]:
        token_ids: list[int] = []
        for token in tokens:
            if type(token) is not int:
                raise ValueError(
                    "vLLM-compatible KV events require integer token IDs")
            token_ids.append(token)
        return token_ids

    def _block_hashes(
        self,
        block: Any,
    ) -> tuple[int, int | None, _NativeStoredBlockState]:
        parent = block.prev
        is_root_child = getattr(parent, "ordinal", -1) == -1
        block_hash = _vllm_wire_hash_from_radix_key(bytes(block.key))
        parent_hash = None if is_root_child else _vllm_wire_hash_from_radix_key(
            bytes(parent.key))
        return block_hash, parent_hash, _NativeStoredBlockState(block_hash)

    def add_removed_event(self, block_hashes: Any) -> None:
        if isinstance(block_hashes, (bytes, str, int)):
            block_hashes = (block_hashes, )
        removed_hashes: list[ExternalBlockHash] = []
        for block_key in block_hashes:
            if not isinstance(block_key, bytes):
                continue
            state = self._stored_blocks.pop(block_key, None)
            if state is not None:
                removed_hashes.append(state.block_hash)
        self._add_removed_hashes(removed_hashes)

    def add_removed_life_cycle_event(self, block_hash: bytes,
                                     life_cycle_id: int) -> None:
        if int(life_cycle_id) != self._target_life_cycle_id:
            self.non_target_life_cycles_ignored += 1
            return
        state = self._stored_blocks.pop(block_hash, None)
        if state is not None:
            self._add_removed_hashes([state.block_hash])

    def _add_removed_hashes(
            self, block_hashes: list[ExternalBlockHash]) -> None:
        if not block_hashes:
            return
        if not self._reserve_entries(len(block_hashes)):
            return
        if self._pending_events and isinstance(self._pending_events[-1],
                                               BlockRemoved):
            self._pending_events[-1].block_hashes.extend(block_hashes)
        else:
            self._pending_events.append(
                BlockRemoved(block_hashes=block_hashes, medium="GPU"))
        self.removed_blocks += len(block_hashes)

    def add_updated_event(
        self,
        block_hash: Any,
        *,
        cache_level: KVCacheEventDiff | None = None,
        priority: KVCacheEventDiff | None = None,
        layer_group_id: int | None = None,
    ) -> None:
        return

    def _reserve_entries(self, num_entries: int) -> bool:
        if self._pending_entries + num_entries <= self._max_entries:
            self._pending_entries += num_entries
            return True
        self.dropped_events += num_entries
        if self.dropped_events == num_entries or (
                self.dropped_events & (self.dropped_events - 1) == 0):
            logger.warning(
                "Dropping native KV events because the per-iteration safety "
                f"cap was exceeded; dropped_events={self.dropped_events}")
        return False

    def flush_iteration_events(self) -> None:
        if self._closed or not self._pending_events:
            return
        events = self._pending_events
        self._pending_events = []
        self._pending_entries = 0
        self._adapter.publish_wire_events(events)

    def get_latest_events(
            self, timeout_ms: float | None = None) -> list[KVCacheEvent]:
        raise RuntimeError(
            "KV cache event polling is unavailable while native publishing "
            "is enabled")

    def shutdown(self) -> None:
        if self._closed:
            return
        self.flush_iteration_events()
        self._closed = True
        logger.info(
            "Native KV event fast path "
            f"stored_blocks={self.stored_blocks} "
            f"removed_blocks={self.removed_blocks} "
            f"partial_blocks_suppressed={self.partial_blocks_suppressed} "
            f"non_target_life_cycles_ignored="
            f"{self.non_target_life_cycles_ignored} "
            f"dropped_events={self.dropped_events} "
            f"kv_event_allgathers=0")
