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
from queue import Queue
from typing import Any, Optional

import msgspec
import zmq

from tensorrt_llm.llmapi.llm_args import KVEventsConfig
from tensorrt_llm.logger import logger

# Subscribers decode block hashes as 64-bit ints, so a bytes value would fail the
# decode for the entire batch.
ExternalBlockHash = int
MAX_PUBLISH_QUEUE_BYTES = 64 * 1024 * 1024
MAX_REPLAY_BUFFER_BYTES = 64 * 1024 * 1024
HEARTBEAT_INTERVAL_SECONDS = 5.0


def _enable_ipv6_for_endpoint(socket_: zmq.Socket, endpoint: str) -> None:
    """Enable IPv6 on a socket whose TCP endpoint names an IPv6 host."""
    if endpoint.startswith("tcp://") and ":" in endpoint[len("tcp://") :].rpartition(":")[0]:
        socket_.setsockopt(zmq.IPV6, 1)


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

    Delivery is best effort. Internally detected queue or send loss schedules an
    ``AllBlocksCleared`` control batch. ZeroMQ may independently drop for a slow
    subscriber at its HWM; that subscriber observes the next sequence gap.
    """

    SHUTDOWN_TIMEOUT = 1.0
    END_SEQ = (-1).to_bytes(8, "big", signed=True)

    def __init__(
        self,
        data_parallel_rank: int,
        endpoint: str = "tcp://*:5557",
        replay_endpoint: str | None = None,
        buffer_steps: int = 10_000,
        hwm: int = 256,
        max_queue_size: int = 100_000,
        topic: str = "",
    ) -> None:
        super().__init__(data_parallel_rank)
        self._event_queue = Queue[Optional[tuple[int, bytes]]](maxsize=max_queue_size)
        self._buffer_steps = buffer_steps
        self._buffer_bytes = 0
        self._queue_lock = threading.Lock()
        self._queued_payload_bytes = 0
        self._queue_epoch = 0
        self._published_epoch = -1
        self._ctx = zmq.Context.instance()
        self._pub: Optional[zmq.Socket] = None
        self._replay: Optional[zmq.Socket] = None
        self._rank = data_parallel_rank
        self._endpoint = self.offset_endpoint_port(endpoint, self._rank)
        self._replay_endpoint = self.offset_endpoint_port(replay_endpoint, self._rank)
        self._buffer: deque[tuple[int, bytes]] | None = (
            deque() if self._replay_endpoint is not None else None
        )
        self._buffer_lock: threading.Lock | None = (
            threading.Lock() if self._buffer is not None else None
        )
        self._hwm = hwm
        # A wall-clock epoch keeps resume sequence numbers increasing across
        # publisher restarts while leaving ample uint64 headroom for batches.
        self._next_sequence_number = time.time_ns()
        self._topic_bytes = topic.encode("utf-8")
        self._running = True
        self._shutdown_lock = threading.Lock()
        self._resync_required = threading.Event()
        self._startup_complete = threading.Event()
        self._publisher_error: Exception | None = None
        self._last_send_monotonic = 0.0
        self._replay_ready = threading.Event()
        self._replay_error: Exception | None = None
        self.enqueued_batches = 0
        self.published_batches = 0
        self._queue_full_drops = 0
        self._send_error_drops = 0
        self._topic = topic
        # Nothing is bound and no thread runs until start(); see EventPublisher.start().
        self._thread: Optional[threading.Thread] = None
        self._replay_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None:
            return
        try:
            self._socket_setup()
        except Exception:
            raise
        self._thread = threading.Thread(
            target=self._publisher_thread,
            daemon=True,
            name=f"trtllm-kv-events-rank-{self._rank}",
        )
        try:
            if self._replay_endpoint is not None:
                self._replay_thread = threading.Thread(
                    target=self._replay_service_thread,
                    daemon=True,
                    name=f"trtllm-kv-events-replay-rank-{self._rank}",
                )
                self._replay_thread.start()
                if not self._replay_ready.wait(timeout=self.SHUTDOWN_TIMEOUT):
                    raise RuntimeError("KV event replay thread did not initialize")
                if self._replay_error is not None:
                    raise self._replay_error
            self._resync_required.set()
            self._thread.start()
            if not self._startup_complete.wait(timeout=self.SHUTDOWN_TIMEOUT):
                raise RuntimeError("KV event publisher thread did not initialize")
            if self._publisher_error is not None:
                raise self._publisher_error
        except Exception:
            self._running = False
            if self._thread is not None and self._thread.is_alive():
                self._thread.join()
            if self._replay_thread is not None and self._replay_thread.is_alive():
                self._replay_thread.join()
            self._thread = None
            raise
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
        try:
            payload = msgspec.msgpack.encode(events)
        except Exception:
            self._send_error_drops += 1
            self._signal_loss()
            logger.error(
                f"Failed to encode streaming KV event batch rank={self._rank}\n"
                f"{traceback.format_exc()}"
            )
            return False
        with self._queue_lock:
            if (
                self._event_queue.full()
                or self._queued_payload_bytes + len(payload) > MAX_PUBLISH_QUEUE_BYTES
            ):
                self._queue_full_drops += 1
                self._queue_epoch += 1
                self._resync_required.set()
                drops = self._queue_full_drops
                if drops == 1 or (drops & (drops - 1) == 0):
                    logger.warning(
                        f"Dropping streaming KV event batch on rank={self._rank} because "
                        "the bounded publisher queue is full; an AllBlocksCleared recovery "
                        f"batch will be published; dropped_batches={self.dropped_batches}"
                    )
                return False
            queued = (self._queue_epoch, payload)
            self._event_queue.put_nowait(queued)
            self._queued_payload_bytes += len(payload)
        self.enqueued_batches += 1
        return True

    def shutdown(self) -> None:
        with self._shutdown_lock:
            if not self._running:
                return
            with self._queue_lock:
                self._queue_epoch += 1
                self._resync_required.set()
            self._running = False
            try:
                self._event_queue.put_nowait(None)
            except queue.Full:
                # The publisher thread abandons queued data and fences it with a clear.
                pass
        if self._thread is not None:
            self._thread.join()
        if self._replay_thread is not None:
            self._replay_thread.join()
        if self._buffer is not None:
            assert self._buffer_lock is not None
            with self._buffer_lock:
                self._buffer.clear()
                self._buffer_bytes = 0
        logger.info(
            f"Stopped streaming KV event publisher rank={self._rank} "
            f"enqueued_batches={self.enqueued_batches} "
            f"published_batches={self.published_batches} "
            f"dropped_batches={self.dropped_batches}"
        )

    def _socket_setup(self) -> None:
        if not self._endpoint:
            raise ValueError("KV event publisher endpoint must not be empty")
        if not self._endpoint.startswith(("tcp://", "ipc://", "inproc://")):
            raise ValueError(f"Unsupported KV event endpoint scheme: {self._endpoint!r}")

    def _publisher_thread(self) -> None:
        # Keep the PUB socket on its owning thread for its entire lifetime.
        publisher = self._ctx.socket(zmq.PUB)
        self._pub = publisher
        bound = False
        try:
            _enable_ipv6_for_endpoint(publisher, self._endpoint)
            publisher.set_hwm(self._hwm)
            publisher.setsockopt(zmq.SNDTIMEO, int(self.SHUTDOWN_TIMEOUT * 1000))
            # The publisher owns its endpoint and subscribers connect to it, so
            # PUB always binds, including an explicit 0.0.0.0 host.
            publisher.bind(self._endpoint)
            bound = True
            while self._running:
                if self._publish_recovery_if_needed():
                    self._startup_complete.set()
                try:
                    item = self._event_queue.get(timeout=min(0.1, HEARTBEAT_INTERVAL_SECONDS))
                except queue.Empty:
                    if time.monotonic() - self._last_send_monotonic >= HEARTBEAT_INTERVAL_SECONDS:
                        self._send_payload(self._heartbeat_payload())
                    continue
                if item is None:
                    self._event_queue.task_done()
                    break
                epoch, payload = item
                try:
                    with self._queue_lock:
                        self._queued_payload_bytes -= len(payload)
                        current_epoch = self._queue_epoch
                    if epoch != current_epoch:
                        continue
                    self._publish_recovery_if_needed()
                    with self._queue_lock:
                        if epoch != self._queue_epoch or epoch != self._published_epoch:
                            continue
                    self._send_payload(payload)
                finally:
                    self._event_queue.task_done()
        except Exception as error:
            if not self._startup_complete.is_set():
                self._publisher_error = error
            else:
                logger.error(
                    f"Streaming KV event publisher thread failed\n{traceback.format_exc()}"
                )
            self._running = False
        finally:
            self._startup_complete.set()
            while True:
                try:
                    abandoned = self._event_queue.get_nowait()
                except queue.Empty:
                    break
                if abandoned is not None:
                    _, payload = abandoned
                    with self._queue_lock:
                        self._queued_payload_bytes -= len(payload)
                    self._queue_full_drops += 1
                self._event_queue.task_done()
            if bound:
                # A final clear fences any batch abandoned during bounded shutdown.
                self._send_payload(self._clear_payload())
            publisher.close(linger=0)
            self._pub = None

    def _publish_recovery_if_needed(self) -> bool:
        with self._queue_lock:
            target_epoch = self._queue_epoch
        if target_epoch == self._published_epoch:
            return False
        if self._send_payload(self._clear_payload()):
            with self._queue_lock:
                self._published_epoch = target_epoch
                if self._queue_epoch == target_epoch:
                    self._resync_required.clear()
            return True
        return False

    def _clear_payload(self) -> bytes:
        return msgspec.msgpack.encode(
            KVEventBatch(
                ts=time.time(),
                events=[AllBlocksCleared()],
                data_parallel_rank=self._rank,
            )
        )

    def _heartbeat_payload(self) -> bytes:
        return msgspec.msgpack.encode(
            KVEventBatch(
                ts=time.time(),
                events=[],
                data_parallel_rank=self._rank,
            )
        )

    def _signal_loss(self) -> None:
        with self._queue_lock:
            self._queue_epoch += 1
            self._resync_required.set()

    def _send_payload(self, payload: bytes) -> bool:
        assert self._pub is not None
        seq = self._next_sequence_number
        self._next_sequence_number += 1
        try:
            self._pub.send_multipart(
                (
                    self._topic_bytes,
                    seq.to_bytes(8, "big"),
                    payload,
                ),
                flags=zmq.NOBLOCK,
            )
            if self._buffer is not None:
                assert self._buffer_lock is not None
                with self._buffer_lock:
                    while self._buffer and (
                        len(self._buffer) >= self._buffer_steps
                        or self._buffer_bytes + len(payload) > MAX_REPLAY_BUFFER_BYTES
                    ):
                        _, evicted = self._buffer.popleft()
                        self._buffer_bytes -= len(evicted)
                    if len(payload) <= MAX_REPLAY_BUFFER_BYTES:
                        self._buffer.append((seq, payload))
                        self._buffer_bytes += len(payload)
            self.published_batches += 1
            self._last_send_monotonic = time.monotonic()
            return True
        except Exception:
            self._send_error_drops += 1
            self._signal_loss()
            logger.error(
                f"Failed to publish streaming KV event batch rank={self._rank}; "
                "an AllBlocksCleared recovery batch will be retried\n"
                f"{traceback.format_exc()}"
            )
            time.sleep(0.1)
            return False

    def _replay_service_thread(self) -> None:
        # Create, bind, use, and close the ROUTER socket on this thread. ZeroMQ
        # sockets are not thread-safe and must not migrate between threads.
        replay = self._ctx.socket(zmq.ROUTER)
        self._replay = replay
        try:
            assert self._replay_endpoint is not None
            _enable_ipv6_for_endpoint(replay, self._replay_endpoint)
            replay.setsockopt(zmq.SNDTIMEO, int(self.SHUTDOWN_TIMEOUT * 1000))
            replay.bind(self._replay_endpoint)
            self._replay_ready.set()
            while self._running:
                if not self._replay.poll(100):
                    continue
                try:
                    self._service_replay()
                except zmq.Again:
                    logger.warning(f"Aborting stalled KV event replay request on rank={self._rank}")
                except Exception:
                    logger.error(
                        "Failed to service streaming KV event replay request\n"
                        f"{traceback.format_exc()}"
                    )
        except Exception as error:
            if not self._replay_ready.is_set():
                self._replay_error = error
                self._replay_ready.set()
            else:
                logger.error(f"KV event replay service failed\n{traceback.format_exc()}")
        finally:
            self._replay_ready.set()
            replay.close(linger=0)
            self._replay = None

    def _service_replay(self) -> None:
        assert self._replay is not None
        frame = self._replay.recv_multipart()
        if len(frame) != 3:
            logger.warning(f"Invalid streaming KV event replay request: {frame}")
            return
        client_id, _, start_seq_bytes = frame
        start_seq = int.from_bytes(start_seq_bytes, "big")
        assert self._buffer is not None
        assert self._buffer_lock is not None
        with self._buffer_lock:
            buffered = tuple(self._buffer)
        for seq, payload in buffered:
            if not self._running:
                return
            if seq >= start_seq:
                self._send_replay_frames(
                    (
                        client_id,
                        b"",
                        self._topic_bytes,
                        seq.to_bytes(8, "big"),
                        payload,
                    ),
                )
        self._send_replay_frames((client_id, b"", b"", self.END_SEQ, b""))

    def _send_replay_frames(self, frames: tuple[bytes, ...]) -> None:
        assert self._replay is not None
        deadline = time.monotonic() + self.SHUTDOWN_TIMEOUT
        while self._running:
            try:
                self._replay.send_multipart(frames, flags=zmq.NOBLOCK)
                return
            except zmq.Again:
                remaining_ms = int(max(0.0, deadline - time.monotonic()) * 1000)
                if remaining_ms <= 0 or not self._replay.poll(min(remaining_ms, 100), zmq.POLLOUT):
                    if time.monotonic() >= deadline:
                        raise
        raise zmq.Again()

    @staticmethod
    def offset_endpoint_port(endpoint: str | None, data_parallel_rank: int) -> str | None:
        """Apply the base-port-plus-rank endpoint convention (each rank binds base_port + rank)."""
        if not endpoint:
            return endpoint
        # Match the scheme with startswith so detection agrees with
        # _socket_setup (substring tests misclassify hosts like "ipc-host").
        # ipc/inproc have no port; give each rank a distinct suffix instead.
        if endpoint.startswith(("inproc://", "ipc://")):
            return endpoint if data_parallel_rank == 0 else f"{endpoint}_dp{data_parallel_rank}"
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
            if data_parallel_rank == 0:
                return endpoint
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
    if backend not in ("cpp", "python"):
        raise ValueError(f"Unsupported KV cache manager V2 backend: {backend!r}")
    validate_endpoint_ranges(config, ranks_per_host, data_parallel_size)


def validate_endpoint_ranges(
    config: KVEventsConfig, ranks_per_host: int, data_parallel_size: int
) -> None:
    """Reject endpoint configurations the per-rank binding cannot honour.

    Ranks bind ``base_port + rank`` using their **global** rank, so each rank's port is
    distinct cluster-wide and the sockets span ``[base, base + world - 1]``. Two things
    can go wrong, and both are checked on every rank -- before any socket is created and
    before the initialization collectives -- so all ranks fail identically rather than
    one aborting while its peers wait in an all-reduce:

    * The span can run past port 65535. ``ZmqEventPublisher.__init__`` resolves
      ``base_port + rank`` itself, but only raises on the ranks that actually overflow.
    * The publish and replay spans can intersect. Only ranks co-located on one host
      contend for a port, and a host holds a contiguous run of ranks, so the required
      spacing between the two base ports is the per-host rank count, not the total.
    """
    world = max(1, data_parallel_size)
    pub_base = _tcp_base_port(config.endpoint)
    replay_base = _tcp_base_port(config.replay_endpoint)

    for name, endpoint, base_port in (
        ("endpoint", config.endpoint, pub_base),
        ("replay_endpoint", config.replay_endpoint, replay_base),
    ):
        if base_port is None:
            continue
        highest = base_port + world - 1
        if highest > 65_535:
            raise ValueError(
                f"KV event {name} {endpoint!r} does not fit {world} rank(s): ranks bind "
                f"base_port+rank, so the highest would be {highest}, above the maximum "
                f"port 65535. Use a base port at or below {65_535 - world + 1}."
            )

    if pub_base is None or replay_base is None:
        return
    span = max(1, ranks_per_host)
    distance = abs(pub_base - replay_base)
    if distance < span:
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
