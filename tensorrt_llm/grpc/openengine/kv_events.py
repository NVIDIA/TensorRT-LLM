# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KV cache event discovery and streaming for the OpenEngine Control service.

TensorRT-LLM publishes KV cache events from the engine itself: every attention-DP
rank binds its own ZeroMQ ``PUB`` socket and sends msgpack batches from a
background thread (``tensorrt_llm._torch.pyexecutor.kv_cache_events``). This
module exposes that publisher through the two OpenEngine RPCs:

``GetKvEventSources`` advertises the sockets so a client can subscribe to the
engine directly, which keeps events off the gRPC server's event loop.
``SubscribeKvEvents`` re-publishes the same batches as protobuf for clients that
cannot reach the ZeroMQ ports.

Both read one source of truth -- ``kv_cache_config.kv_events_config`` -- and
reuse the publisher's own ``base_port + rank`` convention rather than
reimplementing it, so the advertisement cannot drift from the binds.
"""

from __future__ import annotations

import asyncio
import socket
import traceback
from dataclasses import dataclass
from typing import Any, AsyncIterator, Iterable, Optional

import msgspec
import zmq
import zmq.asyncio
from openengine.v1 import kv_pb2

from tensorrt_llm._torch.pyexecutor.kv_cache_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVEventBatch,
    ZmqEventPublisher,
)
from tensorrt_llm.logger import logger

__all__ = [
    "KvEventsUnavailableError",
    "ResolvedKvEventSource",
    "resolve_advertise_host",
    "resolve_sources",
    "stream_batches",
    "to_proto_source",
]

# A bind wildcard is not a connectable address, and the protocol requires
# `endpoint_addr` to carry one.
_WILDCARD_HOSTS = frozenset({"*", "0.0.0.0", "::", ""})

# Batches held for one `SubscribeKvEvents` subscriber before the oldest are
# dropped. The engine's own publisher drops rather than blocks (a full queue
# leaves an observable sequence gap), so a slow gRPC consumer must not be the
# one thing that can apply backpressure all the way to the scheduler.
_MAX_QUEUED_BATCHES = 1024

_STORAGE_MEDIUM_BY_NAME = {
    "GPU": kv_pb2.STORAGE_MEDIUM_GPU,
    "CPU": kv_pb2.STORAGE_MEDIUM_CPU_PINNED,
    "CPU_PINNED": kv_pb2.STORAGE_MEDIUM_CPU_PINNED,
    "DISK": kv_pb2.STORAGE_MEDIUM_DISK,
    "EXTERNAL": kv_pb2.STORAGE_MEDIUM_EXTERNAL,
}


class KvEventsUnavailableError(RuntimeError):
    """Raised when KV events exist but cannot be advertised or subscribed to."""


@dataclass(frozen=True)
class ResolvedKvEventSource:
    """One attention-DP rank's publisher, addressed so a client can connect."""

    data_parallel_rank: int
    host: str
    port: int
    replay_endpoint: Optional[str]

    @property
    def endpoint(self) -> str:
        """Connectable ZeroMQ endpoint for this rank's PUB socket."""
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"tcp://{host}:{self.port}"


def resolve_advertise_host(bind_host: str) -> str:
    """Return a routable host for a server bound to `bind_host`.

    A wildcard bind means "every interface", which is not an address a client can
    connect to, so name this host explicitly instead.
    """
    cleaned = (bind_host or "").strip().strip("[]")
    if cleaned.lower() not in _WILDCARD_HOSTS and cleaned.lower() != "::":
        return cleaned
    hostname = socket.gethostname()
    try:
        return socket.gethostbyname(hostname)
    except OSError:
        # No resolvable A record; the name is still better than a wildcard.
        return hostname


def events_config(llm: Any) -> Optional[Any]:
    """Return the active streaming `KVEventsConfig`, or None when it is off.

    A `null` publisher is the documented way to build the event path without
    publishing, so it reads the same as disabled here.
    """
    cache_config = getattr(getattr(llm, "args", None), "kv_cache_config", None)
    config = getattr(cache_config, "kv_events_config", None)
    if config is None or not getattr(config, "enable_kv_cache_events", False):
        return None
    if getattr(config, "publisher", None) != "zmq":
        return None
    return config


def data_parallel_size(llm: Any) -> int:
    """Number of ranks that bind a publisher.

    Mirrors `Mapping.dp_size`, which is what actually decides how many sockets
    exist: without attention DP only rank 0 publishes.
    """
    args = getattr(llm, "args", None)
    if not getattr(args, "enable_attention_dp", False):
        return 1
    return max(1, int(getattr(args, "tensor_parallel_size", 1) or 1))


def _split_tcp_endpoint(endpoint: str) -> tuple[str, int]:
    """Split a `tcp://host:port` endpoint, rejecting anything else."""
    if not endpoint.startswith("tcp://"):
        raise KvEventsUnavailableError(
            f"KV cache events are published on {endpoint!r}; only tcp:// endpoints can be "
            "advertised because the protocol addresses a source by host and port"
        )
    remainder = endpoint[len("tcp://") :]
    host, _, port_text = remainder.rpartition(":")
    if not port_text.isdigit():
        raise KvEventsUnavailableError(f"KV cache event endpoint has no port: {endpoint!r}")
    return host.strip("[]"), int(port_text)


def resolve_sources(llm: Any, advertise_host: str) -> list[ResolvedKvEventSource]:
    """Address every rank's publisher, or return [] when events are disabled.

    Raises:
        KvEventsUnavailableError: Events are on but cannot be addressed.
    """
    config = events_config(llm)
    if config is None:
        return []

    dp_size = data_parallel_size(llm)
    gpus_per_node = getattr(getattr(llm, "args", None), "gpus_per_node", None) or dp_size
    if dp_size > gpus_per_node:
        # Ranks bind by *global* rank, so ranks beyond this node bind on a host
        # this process cannot name. Advertising only the local ones would hand a
        # router a partial view of the cache, which is worse than none.
        raise KvEventsUnavailableError(
            f"KV cache events span {dp_size} attention-DP ranks across more than one node "
            f"({gpus_per_node} GPUs per node); this server can only address the ranks on "
            "its own host"
        )

    sources = []
    for rank in range(dp_size):
        endpoint = ZmqEventPublisher.offset_endpoint_port(config.endpoint, rank)
        host, port = _split_tcp_endpoint(endpoint)
        if host.lower() in _WILDCARD_HOSTS:
            host = advertise_host
        replay = ZmqEventPublisher.offset_endpoint_port(config.replay_endpoint, rank)
        if replay:
            replay_host, replay_port = _split_tcp_endpoint(replay)
            if replay_host.lower() in _WILDCARD_HOSTS:
                replay_host = advertise_host
            replay = f"tcp://{replay_host}:{replay_port}"
        sources.append(
            ResolvedKvEventSource(
                data_parallel_rank=rank,
                host=host,
                port=port,
                replay_endpoint=replay or None,
            )
        )
    return sources


def to_proto_source(
    source: ResolvedKvEventSource,
    config: Any,
    schema_version: int,
) -> kv_pb2.KvEventSource:
    """Describe one resolved publisher to a client."""
    return kv_pb2.KvEventSource(
        transport="zmq",
        endpoint_addr=kv_pb2.KvEndpoint(
            host=source.host,
            port=source.port,
            protocol="tcp",
        ),
        topic=config.topic,
        replay_endpoint=source.replay_endpoint or "",
        data_parallel_rank=source.data_parallel_rank,
        encoding="msgpack",
        schema_version=schema_version,
        buffer_steps=config.buffer_steps,
        hwm=config.hwm,
        max_queue_size=config.max_queue_size,
    )


def _block_hash(value: int) -> kv_pb2.KvBlockHash:
    """Encode a publisher block hash, which is a signed int64."""
    return kv_pb2.KvBlockHash(value=int(value).to_bytes(8, "big", signed=True), encoding="int64")


def _storage_medium(medium: Optional[str]) -> int:
    if not medium:
        return kv_pb2.STORAGE_MEDIUM_UNSPECIFIED
    return _STORAGE_MEDIUM_BY_NAME.get(medium.upper(), kv_pb2.STORAGE_MEDIUM_UNSPECIFIED)


def _event_to_proto(event: Any) -> Optional[kv_pb2.KvEvent]:
    """Translate one wire event, or None when it has no protocol equivalent."""
    if isinstance(event, BlockStored):
        stored = kv_pb2.BlockStored(
            block_hashes=[_block_hash(value) for value in event.block_hashes],
            token_ids=event.token_ids,
            block_size=event.block_size,
            medium=_storage_medium(event.medium),
        )
        if event.parent_block_hash is not None:
            stored.parent_block_hash.CopyFrom(_block_hash(event.parent_block_hash))
        if event.lora_id is not None:
            stored.lora_id = event.lora_id
        if event.lora_name:
            stored.lora_name = event.lora_name
        if event.group_idx is not None:
            stored.group_idx = event.group_idx
        if event.kv_cache_spec_kind:
            stored.kv_cache_spec_kind = event.kv_cache_spec_kind
        if event.kv_cache_spec_sliding_window is not None:
            stored.kv_cache_spec_sliding_window = event.kv_cache_spec_sliding_window
        return kv_pb2.KvEvent(block_stored=stored)
    if isinstance(event, BlockRemoved):
        removed = kv_pb2.BlockRemoved(
            block_hashes=[_block_hash(value) for value in event.block_hashes],
            medium=_storage_medium(event.medium),
        )
        if event.group_idx is not None:
            removed.group_idx = event.group_idx
        return kv_pb2.KvEvent(block_removed=removed)
    if isinstance(event, AllBlocksCleared):
        return kv_pb2.KvEvent(all_blocks_cleared=kv_pb2.AllBlocksCleared())
    logger.warning(f"Skipping unrecognized streaming KV cache event: {type(event).__name__}")
    return None


def _batch_to_proto(
    batch: KVEventBatch,
    sequence_number: int,
    data_parallel_rank: int,
) -> kv_pb2.KvEventBatch:
    events = [message for message in (_event_to_proto(event) for event in batch.events) if message]
    return kv_pb2.KvEventBatch(
        sequence_number=sequence_number,
        timestamp_unix_nanos=int(batch.ts * 1_000_000_000),
        data_parallel_rank=data_parallel_rank,
        events=events,
    )


@dataclass
class _Envelope:
    """One decoded batch, or the terminal failure of a rank's subscription."""

    data_parallel_rank: int
    sequence_number: int = 0
    batch: Optional[KVEventBatch] = None
    error: Optional[str] = None


async def _replay(
    context: zmq.asyncio.Context,
    endpoint: str,
    start_sequence_number: int,
    decoder: msgspec.msgpack.Decoder,
    queue: asyncio.Queue,
    rank: int,
) -> int:
    """Drain retained batches from a rank's replay socket.

    Returns the highest sequence number replayed, so the live stream can skip
    what was already delivered. Only the last ``buffer_steps`` batches are
    retained, so a replay can legitimately start above the requested sequence --
    that gap is the subscriber's to notice.
    """
    socket_ = context.socket(zmq.DEALER)
    try:
        socket_.connect(endpoint)
        await socket_.send_multipart((b"", start_sequence_number.to_bytes(8, "big")))
        highest = -1
        while True:
            frames = await socket_.recv_multipart()
            if len(frames) != 4:
                raise KvEventsUnavailableError(
                    f"KV cache event replay returned {len(frames)} frames, expected four"
                )
            _, _, sequence, payload = frames
            if sequence == ZmqEventPublisher.END_SEQ and not payload:
                return highest
            sequence_number = int.from_bytes(sequence, "big")
            highest = max(highest, sequence_number)
            await queue.put(
                _Envelope(
                    data_parallel_rank=rank,
                    sequence_number=sequence_number,
                    batch=decoder.decode(payload),
                )
            )
    finally:
        socket_.close(linger=0)


async def _pump(
    context: zmq.asyncio.Context,
    source: ResolvedKvEventSource,
    topic: str,
    start_sequence_number: int,
    include_snapshot: bool,
    queue: asyncio.Queue,
) -> None:
    """Forward one rank's batches onto the shared queue until cancelled."""
    decoder = msgspec.msgpack.Decoder(KVEventBatch)
    socket_ = context.socket(zmq.SUB)
    try:
        socket_.connect(source.endpoint)
        socket_.setsockopt_string(zmq.SUBSCRIBE, topic)
        replayed_through = -1
        if (include_snapshot or start_sequence_number > 0) and source.replay_endpoint:
            # Subscribe first, then replay: PUB/SUB drops anything published
            # before the subscription lands, and the replay buffer is what
            # covers that window.
            replayed_through = await _replay(
                context,
                source.replay_endpoint,
                start_sequence_number,
                decoder,
                queue,
                source.data_parallel_rank,
            )
        while True:
            frames = await socket_.recv_multipart()
            if len(frames) != 3:
                logger.warning(
                    f"Discarding KV cache event message with {len(frames)} frames, expected three"
                )
                continue
            _, sequence, payload = frames
            sequence_number = int.from_bytes(sequence, "big")
            if sequence_number <= replayed_through:
                continue
            envelope = _Envelope(
                data_parallel_rank=source.data_parallel_rank,
                sequence_number=sequence_number,
                batch=decoder.decode(payload),
            )
            try:
                queue.put_nowait(envelope)
            except asyncio.QueueFull:
                # Same contract as the engine's publisher: drop and leave a gap
                # rather than stall the producer.
                logger.warning(
                    "Dropping KV cache event batch for a subscriber that is not keeping up; "
                    f"seq={sequence_number} will be missing from the stream"
                )
    except asyncio.CancelledError:
        raise
    except Exception as error:
        logger.error(
            f"KV cache event subscription for rank {source.data_parallel_rank} failed\n"
            f"{traceback.format_exc()}"
        )
        await queue.put(_Envelope(data_parallel_rank=source.data_parallel_rank, error=str(error)))
    finally:
        socket_.close(linger=0)


async def stream_batches(
    sources: Iterable[ResolvedKvEventSource],
    topic: str,
    start_sequence_number: int = 0,
    include_snapshot: bool = False,
) -> AsyncIterator[kv_pb2.KvEventBatch]:
    """Yield protobuf batches from every requested rank until the caller stops.

    Raises:
        KvEventsUnavailableError: A rank's subscription failed. The caller turns
            this into the stream's terminal error response.
    """
    sources = list(sources)
    context = zmq.asyncio.Context()
    queue: asyncio.Queue = asyncio.Queue(maxsize=_MAX_QUEUED_BATCHES)
    tasks = [
        asyncio.ensure_future(
            _pump(context, source, topic, start_sequence_number, include_snapshot, queue)
        )
        for source in sources
    ]
    try:
        while True:
            envelope = await queue.get()
            if envelope.error is not None:
                raise KvEventsUnavailableError(
                    f"KV cache event subscription for rank {envelope.data_parallel_rank} "
                    f"failed: {envelope.error}"
                )
            assert envelope.batch is not None
            yield _batch_to_proto(
                envelope.batch,
                envelope.sequence_number,
                envelope.data_parallel_rank,
            )
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        # Sockets are closed by each pump's `finally`; term() would otherwise
        # block on a socket the cancelled task has not finished releasing.
        context.destroy(linger=0)
