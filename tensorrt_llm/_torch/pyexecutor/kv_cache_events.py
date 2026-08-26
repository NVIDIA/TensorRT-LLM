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
# This module defines TensorRT-LLM's KV cache event wire format: msgpack event batches
# published over ZeroMQ in the three-frame (topic, seq, payload) framing that external
# KV-cache-aware routers expect. Each event encodes as a map tagged with a "type" key,
# the form documented for custom router backends, so routers consume these batches
# without translation. This differs from vLLM's vllm/distributed/kv_events.py, whose
# structs set array_like=True and encode as tagged positional arrays; keeping the map
# form leaves field order out of the wire contract. The batch envelope is positional
# in both.

from __future__ import annotations

import queue
import threading
import time
import traceback
from abc import ABC, abstractmethod
from collections import deque
from itertools import count
from queue import Queue
from typing import Any, Optional

import msgspec
import zmq

from tensorrt_llm.llmapi.llm_args import KVEventsConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime.kv_cache_hash import truncate_sha256_hash_to_int64
from tensorrt_llm.runtime.kv_cache_manager_v2._event_manager import KVCacheEvent, KVCacheEventDiff

# Subscribers decode block hashes as 64-bit ints, so a bytes value would fail the
# decode for the entire batch.
ExternalBlockHash = int


class EventBatch(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,  # type: ignore[call-arg]
):
    """KV cache event wire batch envelope."""

    ts: float
    events: list[Any]
    data_parallel_rank: int | None = None


class KVCacheWireEvent(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,  # type: ignore[call-arg]
    tag=True,
):
    """Base class for KV cache event wire messages."""


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
    """Publishes KV cache event wire batches for one cache rank."""

    def __init__(self, data_parallel_rank: int = 0) -> None:
        self._data_parallel_rank = data_parallel_rank

    def start(self) -> None:
        """Acquire external resources.

        Split from ``__init__`` so constructing a publisher has no side effects: the
        owner can build it early, finish its own validation, and only then commit to
        binding sockets and running threads.
        """

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
    """Publishes event batches over the three-frame ZeroMQ wire protocol.

    Delivery is best effort, but loss is observable: :meth:`publish` reserves a sequence
    number per accepted batch, so a dropped batch leaves a gap. Subscribers must treat a
    gap -- including a replay that starts above the requested ``start_seq`` because
    ``buffer_steps`` evicted older batches -- as lost KV-cache state and resynchronize.
    """

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
        self._event_queue = Queue[Optional[tuple[int, EventBatch]]](maxsize=max_queue_size)
        self._buffer = deque[tuple[int, bytes]](maxlen=buffer_steps)
        self._ctx = zmq.Context.instance()
        self._pub: Optional[zmq.Socket] = None
        self._replay: Optional[zmq.Socket] = None
        self._rank = data_parallel_rank
        self._endpoint = self.offset_endpoint_port(endpoint, self._rank)
        self._replay_endpoint = self.offset_endpoint_port(replay_endpoint, self._rank)
        self._hwm = hwm
        self._seq_gen = count()
        self._topic_bytes = topic.encode("utf-8")
        self._running = True
        self._shutdown_lock = threading.Lock()
        self.enqueued_batches = 0
        self.published_batches = 0
        self._queue_full_drops = 0
        self._send_error_drops = 0
        self._topic = topic
        # Nothing is bound and no thread runs until start(); see EventPublisher.start().
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None:
            return
        try:
            self._socket_setup()
        except Exception:
            # start() never returns on failure, so close whatever was opened rather
            # than leaking it on the shared context.
            if self._pub is not None:
                self._pub.close(linger=0)
                self._pub = None
            if self._replay is not None:
                self._replay.close(linger=0)
                self._replay = None
            raise
        self._thread = threading.Thread(
            target=self._publisher_thread,
            daemon=True,
            name=f"trtllm-kv-events-rank-{self._rank}",
        )
        self._thread.start()
        logger.info(
            f"Started streaming KV event publisher rank={self._rank} "
            f"endpoint={self._endpoint} topic={self._topic!r}"
        )

    @property
    def dropped_batches(self) -> int:
        # Two independent writers: the scheduler thread bumps _queue_full_drops
        # (queue full) and the publisher thread bumps _send_error_drops (send
        # failure). Each counter has a single writer, so the sum needs no lock.
        return self._queue_full_drops + self._send_error_drops

    def publish(self, events: EventBatch) -> bool:
        if not self._running:
            return False
        if events.data_parallel_rank is None:
            events.data_parallel_rank = self._data_parallel_rank
        # Reserve the sequence number here rather than in the publisher thread, so a
        # batch lost to a full queue or a failed send leaves a detectable gap instead of
        # a contiguous stream that hides the loss. publish() is the only allocator.
        seq = next(self._seq_gen)
        try:
            self._event_queue.put_nowait((seq, events))
            self.enqueued_batches += 1
            return True
        except queue.Full:
            self._queue_full_drops += 1
            drops = self._queue_full_drops
            if drops == 1 or (drops & (drops - 1) == 0):
                logger.warning(
                    f"Dropping streaming KV event batch on rank={self._rank} because "
                    f"the publisher queue is full; seq={seq} will be missing from the "
                    f"stream; dropped_batches={self.dropped_batches}"
                )
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
        if self._thread is not None:
            self._thread.join(timeout=self.SHUTDOWN_TIMEOUT)
            if self._thread.is_alive():
                logger.warning(
                    f"Streaming KV event publisher rank={self._rank} did not stop "
                    f"within {self.SHUTDOWN_TIMEOUT:.1f}s"
                )
        logger.info(
            f"Stopped streaming KV event publisher rank={self._rank} "
            f"enqueued_batches={self.enqueued_batches} "
            f"published_batches={self.published_batches} "
            f"dropped_batches={self.dropped_batches}"
        )

    def _socket_setup(self) -> None:
        self._pub = self._ctx.socket(zmq.PUB)
        self._pub.set_hwm(self._hwm)
        if not self._endpoint:
            raise ValueError("KV event publisher endpoint must not be empty")
        if not self._endpoint.startswith(("tcp://", "ipc://", "inproc://")):
            raise ValueError(f"Unsupported KV event endpoint scheme: {self._endpoint!r}")
        # The publisher owns its endpoint and subscribers connect to it, so the
        # PUB socket always binds -- including explicit-host TCP binds like
        # tcp://0.0.0.0:5557 that the previous '*'-only heuristic wrongly
        # treated as connect targets (silently dropping every event).
        self._pub.bind(self._endpoint)

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
                        logger.error(
                            "Failed to service streaming KV event replay request\n"
                            f"{traceback.format_exc()}"
                        )
                try:
                    item = self._event_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if item is None:
                    self._event_queue.task_done()
                    break
                seq, event = item
                try:
                    payload = encoder.encode(event)
                    self._pub.send_multipart(
                        (
                            self._topic_bytes,
                            seq.to_bytes(8, "big"),
                            payload,
                        )
                    )
                    self._buffer.append((seq, payload))
                    self.published_batches += 1
                except Exception:
                    self._send_error_drops += 1
                    logger.error(
                        f"Failed to publish streaming KV event batch rank={self._rank}; "
                        f"seq={seq} will be missing from the stream\n"
                        f"{traceback.format_exc()}"
                    )
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
            logger.warning(f"Invalid streaming KV event replay request: {frame}")
            return
        client_id, _, start_seq_bytes = frame
        start_seq = int.from_bytes(start_seq_bytes, "big")
        for seq, payload in self._buffer:
            if seq >= start_seq:
                self._replay.send_multipart(
                    (
                        client_id,
                        b"",
                        self._topic_bytes,
                        seq.to_bytes(8, "big"),
                        payload,
                    )
                )
        self._replay.send_multipart((client_id, b"", b"", self.END_SEQ, b""))

    @staticmethod
    def offset_endpoint_port(endpoint: str | None, data_parallel_rank: int) -> str | None:
        """Apply the base-port-plus-rank endpoint convention (each rank binds base_port + rank)."""
        if not endpoint or data_parallel_rank == 0:
            return endpoint
        # Match the scheme with startswith so detection agrees with
        # _socket_setup (substring tests misclassify hosts like "ipc-host").
        # ipc/inproc have no port; give each rank a distinct suffix instead.
        if endpoint.startswith(("inproc://", "ipc://")):
            return f"{endpoint}_dp{data_parallel_rank}"
        if endpoint.startswith("tcp://"):
            host_port = endpoint[len("tcp://") :]
            if ":" not in host_port:
                raise ValueError(f"TCP KV event endpoint must include a port: {endpoint!r}")
            last_colon_idx = endpoint.rfind(":")
            base_addr = endpoint[:last_colon_idx]
            port_text = endpoint[last_colon_idx + 1 :]
            # Validate the port value up front so a bad port names the endpoint
            # instead of surfacing as an opaque int()/ZeroMQ bind error on ranks > 0.
            if not (port_text.isdigit() and 1 <= int(port_text) <= 65_535):
                raise ValueError(
                    f"TCP KV event endpoint must have a port in [1, 65535]: {endpoint!r}"
                )
            base_port = int(port_text)
            new_port = base_port + data_parallel_rank
            if new_port > 65_535:
                raise ValueError(
                    f"KV event endpoint port exceeds 65535 for rank {data_parallel_rank}"
                )
            return f"{base_addr}:{new_port}"
        raise ValueError("Invalid endpoint: must start with 'inproc://', 'ipc://', or 'tcp://'")


def _tcp_base_port(endpoint: str | None) -> int | None:
    """Return the base port of a TCP endpoint, or None if it is not TCP."""
    if not endpoint or not endpoint.startswith("tcp://"):
        return None
    last_colon_idx = endpoint.rfind(":")
    port_text = endpoint[last_colon_idx + 1 :]
    if not port_text.isdigit():
        return None
    return int(port_text)


def validate_streaming_support(
    config: KVEventsConfig,
    *,
    pp_size: int,
    cp_size: int,
    ranks_per_host: int,
    data_parallel_size: int,
    backend: str,
) -> None:
    """Reject streaming-KV-event configurations the engine cannot honour.

    Split out of ``KVCacheManagerV2.__init__`` so the preconditions are testable
    without building a manager, which needs a GPU.
    """
    if pp_size > 1:
        raise ValueError("Streaming KV events do not support pipeline parallelism")
    if cp_size > 1:
        raise ValueError("Streaming KV events do not support context parallelism")
    if backend != "python":
        # StreamingKVCacheEventManager is a duck-typed Python event sink, which cannot
        # satisfy the nanobind constructor's nb::cast<std::shared_ptr<kv::EventManager>>
        # (and the C++ radix tree calls the sink natively, not through Python). Fail
        # with an actionable message instead of an opaque TypeError from the cast.
        raise ValueError(
            "Streaming KV events (kv_cache_config.kv_events_config) are only supported "
            f"by the Python KV cache manager V2 backend, but '{backend}' is active. Set "
            "TLLM_KV_CACHE_MANAGER_V2_BACKEND=python to enable streaming KV events, or "
            "use the buffered path via kv_cache_config.event_buffer_max_size."
        )
    validate_endpoint_ranges(config, ranks_per_host, data_parallel_size)


def validate_endpoint_ranges(
    config: KVEventsConfig, ranks_per_host: int, data_parallel_size: int
) -> None:
    """Reject configurations whose publish and replay port ranges overlap.

    Ranks bind ``base_port + rank`` using their **global** rank, so each rank's port is
    distinct cluster-wide and the sockets span ``[base, base + world - 1]``. Only ranks
    co-located on one host actually contend for a port, and a host holds a contiguous
    run of ranks, so the required spacing between the two base ports is the per-host
    rank count rather than the total. Catch it before any socket is created rather than
    as an opaque ``EADDRINUSE``.
    """
    pub_base = _tcp_base_port(config.endpoint)
    replay_base = _tcp_base_port(config.replay_endpoint)
    if pub_base is None or replay_base is None:
        return
    span = max(1, ranks_per_host)
    distance = abs(pub_base - replay_base)
    if distance < span:
        world = max(1, data_parallel_size)
        raise ValueError(
            f"KV event endpoint {config.endpoint!r} and replay_endpoint "
            f"{config.replay_endpoint!r} overlap: ranks bind base_port+rank by global "
            f"rank, so with {world} rank(s) the publish sockets span "
            f"[{pub_base}, {pub_base + world - 1}] and the replay sockets span "
            f"[{replay_base}, {replay_base + world - 1}]. Ranks co-located on a host "
            f"contend for ports, so the base ports must be at least {span} apart (the "
            f"per-host rank count) but are {distance} apart."
        )


def create_event_publisher(config: KVEventsConfig, data_parallel_rank: int) -> EventPublisher:
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


def _kv_event_wire_hash_from_radix_key(block_key: bytes) -> int:
    """Reuse an existing SHA-256 radix key as the KV cache event's signed int64 wire hash."""
    if len(block_key) < 8:
        raise ValueError("V2 radix block keys must contain at least 8 bytes")
    # Reuse the canonical SHA-256 -> int64 truncation (first 8 bytes) shared with
    # the rest of the KV-cache-event machinery instead of a second, divergent
    # truncation, then reinterpret the low 64 bits as the signed int64 wire hash.
    unsigned_hash = truncate_sha256_hash_to_int64(block_key)
    return unsigned_hash - 2**64 if unsigned_hash >= 2**63 else unsigned_hash


class _MultimodalBlockError(ValueError):
    """A block token is a multimodal cache-key digest (bytes), not a wire int.

    ``gen_multimodal_cache_key_tokens`` stores the per-item digest as ``bytes``,
    which has no integer wire representation. Such blocks are skipped
    quietly rather than routed through the malformed-data traceback path.
    """


class StreamingKVCacheEventManager:
    """Scheduler-local fast path that produces KV cache event wire messages directly.

    Implements the V2 KV-cache-manager event-sink hook interface by duck
    typing rather than inheriting ``KVCacheEventManager``: it fully replaces
    event production (reusing the radix block hashes) and shares none of the
    base manager's state, so subclassing would only risk partially initialised
    base attributes.
    """

    def __init__(
        self,
        config: KVEventsConfig,
        *,
        data_parallel_rank: int,
        block_size: int,
        max_window_size: int,
        max_entries: int = 50_000,
    ) -> None:
        self._rank = data_parallel_rank
        self._publisher = create_event_publisher(config, data_parallel_rank)
        self._block_size = block_size
        self._max_window_size = max_window_size
        self._max_entries = max_entries
        self._target_life_cycle_id: int | None = None
        self._stored_blocks: dict[bytes, int] = {}
        self._pending_events: list[BlockStored | BlockRemoved | AllBlocksCleared] = []
        self._pending_entries = 0
        self._closed = False
        self.stored_blocks = 0
        self.removed_blocks = 0
        self.partial_blocks_suppressed = 0
        self.multimodal_blocks_suppressed = 0
        self.non_target_life_cycles_ignored = 0
        self.dropped_events = 0
        self.enqueued_batches = 0
        self.enqueued_events = 0
        self.dropped_batches = 0

    def start(self) -> None:
        """Bind the publisher's sockets and start its background thread.

        Construction is side-effect free, so the owner calls this only once every
        other initialization check has passed. A failure before this point therefore
        leaves no socket bound and no thread running.
        """
        self._publisher.start()

    def set_layer_group_window_sizes(self, window_sizes: dict[int, int]) -> None:
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
            raise ValueError("Streaming KV events require an attention KV cache life cycle")
        self._target_life_cycle_id = min(target_ids)
        logger.info(
            "Streaming KV event fast path selected "
            f"lifecycle_id={self._target_life_cycle_id} "
            f"window_size={self._max_window_size}"
        )

    def add_created_event(
        self,
        num_blocks_per_cache_level: Any,
        layer_group_ids: Any = None,
    ) -> None:
        return

    def add_stored_event(self, *args: Any, **kwargs: Any) -> None:
        # Streaming publishing derives stored events from the per-block hooks
        # below; the aggregate stored-event hook is intentionally unused.
        return

    def add_stored_block_event_from_block(self, block: Any) -> None:
        if self._closed or self._target_life_cycle_id is None:
            return
        life_cycle_id = self._target_life_cycle_id
        if life_cycle_id >= len(block.storage):
            return
        page_ref = block.storage[life_cycle_id]
        page = None if page_ref is None else page_ref()
        if page is None:
            return
        # A non-null page does not imply it covers the whole radix block: V2 can attach
        # a page adopted from a shorter sibling. Publishing that as a BlockStored would
        # tell the router the engine holds a prefix it cannot fully reuse. The buffered
        # manager applies the same rule in _life_cycle_ids_from_radix_block().
        if page.num_tokens_in_block < len(block.tokens):
            self.partial_blocks_suppressed += 1
            return
        self._add_full_block(block)

    def add_stored_life_cycle_event_from_block(self, block: Any, life_cycle_id: int) -> None:
        if life_cycle_id is None or self._target_life_cycle_id is None:
            return
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
            block_hash, parent_hash = self._block_hashes(block)
        except _MultimodalBlockError:
            # Expected for multimodal cache-key blocks; skip without the
            # malformed-data traceback that would otherwise flood the log.
            self.multimodal_blocks_suppressed += 1
            self._pending_entries -= 1
            return
        except ValueError:
            self.dropped_events += 1
            self._pending_entries -= 1
            logger.error(
                "Dropping streaming KV store event with unsupported token data\n"
                f"{traceback.format_exc()}"
            )
            return
        self._stored_blocks[key] = block_hash
        if self._pending_events and isinstance(self._pending_events[-1], BlockStored):
            previous = self._pending_events[-1]
            if previous.block_hashes and previous.block_hashes[-1] == parent_hash:
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
            )
        )
        self.stored_blocks += 1

    @staticmethod
    def _token_ids(tokens: Any) -> list[int]:
        token_ids: list[int] = []
        for token in tokens:
            if type(token) is bytes:
                # Multimodal cache-key digest; not representable as a wire int.
                raise _MultimodalBlockError
            if type(token) is not int:
                raise ValueError("KV cache event wire format requires integer token IDs")
            token_ids.append(token)
        return token_ids

    def _block_hashes(
        self,
        block: Any,
    ) -> tuple[int, int | None]:
        parent = block.prev
        is_root_child = getattr(parent, "ordinal", -1) == -1
        block_hash = _kv_event_wire_hash_from_radix_key(bytes(block.key))
        parent_hash = (
            None if is_root_child else _kv_event_wire_hash_from_radix_key(bytes(parent.key))
        )
        return block_hash, parent_hash

    def add_removed_event(self, block_hashes: Any) -> None:
        if self._closed:
            return
        if isinstance(block_hashes, (bytes, str, int)):
            block_hashes = (block_hashes,)
        removed_hashes: list[ExternalBlockHash] = []
        for block_key in block_hashes:
            if not isinstance(block_key, bytes):
                continue
            stored_hash = self._stored_blocks.pop(block_key, None)
            if stored_hash is not None:
                removed_hashes.append(stored_hash)
        self._add_removed_hashes(removed_hashes)

    def add_removed_life_cycle_event(self, block_hash: bytes, life_cycle_id: int) -> None:
        if self._closed or life_cycle_id is None or self._target_life_cycle_id is None:
            return
        if int(life_cycle_id) != self._target_life_cycle_id:
            self.non_target_life_cycles_ignored += 1
            return
        stored_hash = self._stored_blocks.pop(block_hash, None)
        if stored_hash is not None:
            self._add_removed_hashes([stored_hash])

    def _add_removed_hashes(self, block_hashes: list[ExternalBlockHash]) -> None:
        if not block_hashes:
            return
        # Removals are never dropped by the per-iteration cap and, unlike stores,
        # do not consume the _pending_entries budget: each hash was already
        # reported as stored (so removals are bounded by the stored set), and
        # counting them against the store budget would starve legitimate
        # BlockStored events in a removal-heavy iteration.
        if self._pending_events and isinstance(self._pending_events[-1], BlockRemoved):
            self._pending_events[-1].block_hashes.extend(block_hashes)
        else:
            self._pending_events.append(BlockRemoved(block_hashes=block_hashes, medium="GPU"))
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
            self.dropped_events & (self.dropped_events - 1) == 0
        ):
            logger.warning(
                "Dropping streaming KV events because the per-iteration safety "
                f"cap was exceeded; dropped_events={self.dropped_events}"
            )
        return False

    def flush_iteration_events(self) -> None:
        if self._closed or not self._pending_events:
            return
        events = self._pending_events
        self._pending_events = []
        self._pending_entries = 0
        batch = KVEventBatch(
            ts=time.time(),
            events=events,
            data_parallel_rank=self._rank,
        )
        try:
            if self._publisher.publish(batch):
                self.enqueued_batches += 1
                self.enqueued_events += len(events)
            else:
                self.dropped_batches += 1
        except Exception:
            self.dropped_batches += 1
            logger.error(
                f"Dropping streaming KV event iteration batch on rank={self._rank}\n"
                f"{traceback.format_exc()}"
            )

    def get_latest_events(self, timeout_ms: float | None = None) -> list[KVCacheEvent]:
        # Streaming publishing pushes events out-of-band, so the pull API has
        # nothing to return. Return empty instead of raising so callers of the
        # buffered polling path degrade cleanly rather than erroring.
        return []

    def shutdown(self) -> None:
        if self._closed:
            return
        self.flush_iteration_events()
        self._closed = True
        self._publisher.shutdown()
        logger.info(
            "Streaming KV event fast path "
            f"rank={self._rank} "
            f"stored_blocks={self.stored_blocks} "
            f"removed_blocks={self.removed_blocks} "
            f"partial_blocks_suppressed={self.partial_blocks_suppressed} "
            f"non_target_life_cycles_ignored={self.non_target_life_cycles_ignored} "
            f"dropped_events={self.dropped_events} "
            f"enqueued_batches={self.enqueued_batches} "
            f"dropped_batches={self.dropped_batches}"
        )
