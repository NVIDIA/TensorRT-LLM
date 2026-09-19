# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for OpenEngine KV cache event streaming.

These run a real `ZmqEventPublisher` rather than a stand-in: the whole point of
`SubscribeKvEvents` is that it reproduces what the engine actually puts on the
wire, and a fake publisher would let the two encodings drift apart silently.
"""

import asyncio
import socket
from types import SimpleNamespace

import pytest

pytest.importorskip(
    "openengine",
    reason='OpenEngine dependency not installed (pip install "tensorrt_llm[openengine]")',
)

from conftest import FakeServicerContext  # noqa: E402
from openengine.v1 import kv_pb2  # noqa: E402

from tensorrt_llm._torch.pyexecutor import kv_cache_events as publisher_module  # noqa: E402
from tensorrt_llm._torch.pyexecutor.kv_cache_events import (  # noqa: E402
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVEventBatch,
    ZmqEventPublisher,
)
from tensorrt_llm.grpc.openengine import kv_events as kv_events_module  # noqa: E402
from tensorrt_llm.grpc.openengine.control import OpenEngineControlServicer  # noqa: E402
from tensorrt_llm.grpc.openengine.kv_events import (  # noqa: E402
    KvEventsUnavailableError,
    ResolvedKvEventSource,
    resolve_sources,
    stream_batches,
)
from tensorrt_llm.llmapi import KVEventsConfig  # noqa: E402

# Runs on the CPU stage: the publisher is a ZeroMQ socket, not an engine.
pytestmark = pytest.mark.cpu_only

_PUBLISH_TIMEOUT_SECONDS = 5.0
_SUBSCRIBE_ATTEMPTS = 50
_PROBE_INTERVAL_SECONDS = 0.05


def _unused_tcp_port(host: str = "127.0.0.1") -> int:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _publisher(
    *, with_replay: bool = False, topic: str = "kv", host: str = "127.0.0.1"
) -> ZmqEventPublisher:
    port = _unused_tcp_port(host)
    replay_port = _unused_tcp_port(host) if with_replay else None
    endpoint_host = f"[{host}]" if ":" in host else host
    publisher = ZmqEventPublisher(
        data_parallel_rank=0,
        endpoint=f"tcp://{endpoint_host}:{port}",
        replay_endpoint=f"tcp://{endpoint_host}:{replay_port}" if replay_port else None,
        topic=topic,
    )
    publisher.start()
    publisher._test_source = ResolvedKvEventSource(
        data_parallel_rank=0,
        host=host,
        port=port,
        replay_endpoint=f"tcp://{endpoint_host}:{replay_port}" if replay_port else None,
    )
    return publisher


async def _await_published(publisher: ZmqEventPublisher, count: int) -> None:
    """Wait until the publisher thread has drained `count` batches onto the socket."""
    deadline = asyncio.get_running_loop().time() + _PUBLISH_TIMEOUT_SECONDS
    while publisher.published_batches < count:
        if asyncio.get_running_loop().time() > deadline:
            pytest.fail(f"publisher sent {publisher.published_batches} batches, expected {count}")
        await asyncio.sleep(0.01)


async def _publish_and_wait(publisher: ZmqEventPublisher, batch: KVEventBatch) -> int:
    published_before = publisher.published_batches
    assert publisher.publish(batch)
    await _await_published(publisher, published_before + 1)
    return publisher._next_sequence_number - 1


def _stored() -> BlockStored:
    return BlockStored(
        block_hashes=[-7, 11],
        parent_block_hash=-3,
        token_ids=[5, 6, 7, 8],
        block_size=2,
        lora_id=None,
        medium="GPU",
        lora_name=None,
    )


@pytest.mark.asyncio
async def test_replayed_batches_are_translated_to_protobuf():
    """Replay, not the live socket, so the assertion does not race the subscription.

    A PUB socket drops anything published before a subscriber's subscription
    propagates; the replay buffer exists precisely to cover that window, so it is
    also the deterministic way to test translation.
    """
    publisher = _publisher(with_replay=True)
    try:
        sequence_number = await _publish_and_wait(
            publisher,
            KVEventBatch(
                ts=1.5,
                events=[_stored(), BlockRemoved(block_hashes=[11], medium="GPU")],
                data_parallel_rank=0,
            ),
        )

        batches = stream_batches(
            [publisher._test_source],
            "kv",
            start_sequence_number=sequence_number,
            include_snapshot=True,
        )
        try:
            batch = await asyncio.wait_for(batches.__anext__(), timeout=_PUBLISH_TIMEOUT_SECONDS)
        finally:
            await batches.aclose()
    finally:
        publisher.shutdown()

    assert batch.sequence_number == sequence_number
    assert batch.timestamp_unix_nanos == 1_500_000_000
    assert batch.data_parallel_rank == 0

    stored, removed = batch.events
    assert stored.block_stored.token_ids == [5, 6, 7, 8]
    assert stored.block_stored.block_size == 2
    assert stored.block_stored.medium == kv_pb2.STORAGE_MEDIUM_GPU
    # Publisher hashes are signed int64, so the encoding must round-trip a
    # negative value rather than silently reinterpreting it as unsigned.
    assert [
        int.from_bytes(hash_.value, "big", signed=True)
        for hash_ in stored.block_stored.block_hashes
    ] == [-7, 11]
    assert all(hash_.encoding == "int64" for hash_ in stored.block_stored.block_hashes)
    assert int.from_bytes(stored.block_stored.parent_block_hash.value, "big", signed=True) == -3
    assert [
        int.from_bytes(hash_.value, "big", signed=True)
        for hash_ in removed.block_removed.block_hashes
    ] == [11]


@pytest.mark.asyncio
async def test_a_cleared_cache_is_reported_as_its_own_event():
    publisher = _publisher(with_replay=True)
    try:
        sequence_number = await _publish_and_wait(
            publisher, KVEventBatch(ts=0.0, events=[AllBlocksCleared()])
        )

        batches = stream_batches(
            [publisher._test_source],
            "kv",
            start_sequence_number=sequence_number,
            include_snapshot=True,
        )
        try:
            batch = await asyncio.wait_for(batches.__anext__(), timeout=_PUBLISH_TIMEOUT_SECONDS)
        finally:
            await batches.aclose()
    finally:
        publisher.shutdown()

    (event,) = batch.events
    assert event.WhichOneof("event") == "all_blocks_cleared"


@pytest.mark.asyncio
async def test_replay_starts_at_the_requested_sequence_number():
    """Batches the subscriber already has must not be replayed to it again."""
    publisher = _publisher(with_replay=True)
    try:
        sequence_numbers = [
            await _publish_and_wait(publisher, KVEventBatch(ts=float(index), events=[_stored()]))
            for index in range(3)
        ]

        batches = stream_batches(
            [publisher._test_source], "kv", start_sequence_number=sequence_numbers[1]
        )
        try:
            batch = await asyncio.wait_for(batches.__anext__(), timeout=_PUBLISH_TIMEOUT_SECONDS)
        finally:
            await batches.aclose()
    finally:
        publisher.shutdown()

    assert batch.sequence_number == sequence_numbers[1]


@pytest.mark.asyncio
async def test_empty_replay_preserves_the_requested_live_sequence():
    """An empty replay must not reset the lower bound for subsequent live events."""
    publisher = _publisher(with_replay=True)
    assert publisher._buffer is not None
    assert publisher._buffer_lock is not None
    with publisher._buffer_lock:
        publisher._buffer.clear()
        publisher._buffer_bytes = 0

    start_sequence = publisher._next_sequence_number + 2
    batches = stream_batches([publisher._test_source], "kv", start_sequence_number=start_sequence)
    delivered: asyncio.Queue = asyncio.Queue()

    async def drain() -> None:
        async for batch in batches:
            await delivered.put(batch)

    consumer = asyncio.ensure_future(drain())
    try:
        for _ in range(_SUBSCRIBE_ATTEMPTS):
            await _publish_and_wait(publisher, KVEventBatch(ts=2.0, events=[_stored()]))
            if publisher._next_sequence_number <= start_sequence:
                continue
            try:
                received = await asyncio.wait_for(delivered.get(), timeout=_PROBE_INTERVAL_SECONDS)
                break
            except asyncio.TimeoutError:
                continue
        else:
            pytest.fail("subscription never delivered the requested sequence")
    finally:
        consumer.cancel()
        await asyncio.gather(consumer, return_exceptions=True)
        publisher.shutdown()

    assert received.sequence_number >= start_sequence


@pytest.mark.asyncio
async def test_ipv6_publisher_replay_and_bridge_round_trip():
    """Every ZMQ socket in the advertised IPv6 path must opt in to IPv6."""
    if not socket.has_ipv6:
        pytest.skip("Python was built without IPv6 support")
    try:
        publisher = _publisher(with_replay=True, host="::1")
    except OSError:
        pytest.skip("IPv6 loopback is unavailable")
    try:
        sequence_number = await _publish_and_wait(
            publisher, KVEventBatch(ts=3.0, events=[_stored()])
        )
        batches = stream_batches(
            [publisher._test_source],
            "kv",
            start_sequence_number=sequence_number,
            include_snapshot=True,
        )
        try:
            batch = await asyncio.wait_for(batches.__anext__(), timeout=_PUBLISH_TIMEOUT_SECONDS)
        finally:
            await batches.aclose()
    finally:
        publisher.shutdown()

    assert batch.sequence_number == sequence_number


@pytest.mark.asyncio
async def test_live_batches_reach_a_subscriber():
    """The live path, synchronised on an actual delivery rather than a sleep."""
    publisher = _publisher()
    batches = stream_batches([publisher._test_source], "kv")
    # Drained by a task rather than by awaiting `__anext__` under a timeout: a
    # cancelled `asend` leaves the generator unusable, so the retry loop would
    # be testing its own cleanup rather than delivery.
    delivered: asyncio.Queue = asyncio.Queue()

    async def drain() -> None:
        async for batch in batches:
            await delivered.put(batch)

    consumer = asyncio.ensure_future(drain())
    try:
        received = None
        for _ in range(_SUBSCRIBE_ATTEMPTS):
            publisher.publish(KVEventBatch(ts=2.0, events=[_stored()]))
            try:
                received = await asyncio.wait_for(delivered.get(), timeout=_PROBE_INTERVAL_SECONDS)
                break
            except asyncio.TimeoutError:
                continue
        assert received is not None, "subscription never propagated"
    finally:
        consumer.cancel()
        await asyncio.gather(consumer, return_exceptions=True)
        publisher.shutdown()

    (event,) = received.events
    assert event.block_stored.token_ids == [5, 6, 7, 8]


@pytest.mark.asyncio
async def test_live_stream_honors_start_sequence_without_replay():
    publisher = _publisher()
    start_sequence = publisher._next_sequence_number + 1
    batches = stream_batches([publisher._test_source], "kv", start_sequence_number=start_sequence)
    delivered: asyncio.Queue = asyncio.Queue()

    async def drain() -> None:
        async for batch in batches:
            await delivered.put(batch)

    consumer = asyncio.ensure_future(drain())
    try:
        for _ in range(_SUBSCRIBE_ATTEMPTS):
            await _publish_and_wait(publisher, KVEventBatch(ts=2.0, events=[_stored()]))
            if publisher._next_sequence_number <= start_sequence:
                continue
            try:
                received = await asyncio.wait_for(delivered.get(), timeout=_PROBE_INTERVAL_SECONDS)
                break
            except asyncio.TimeoutError:
                continue
        else:
            pytest.fail("subscription never delivered the requested sequence")
    finally:
        consumer.cancel()
        await asyncio.gather(consumer, return_exceptions=True)
        publisher.shutdown()

    assert received.sequence_number >= start_sequence


@pytest.mark.asyncio
async def test_closing_the_stream_releases_its_sockets():
    """A client that disconnects must not leak a subscription for the process's life."""
    publisher = _publisher(with_replay=True)
    try:
        sequence_number = await _publish_and_wait(publisher, KVEventBatch(ts=0.0, events=[]))

        batches = stream_batches(
            [publisher._test_source],
            "kv",
            start_sequence_number=sequence_number,
            include_snapshot=True,
        )
        await asyncio.wait_for(batches.__anext__(), timeout=_PUBLISH_TIMEOUT_SECONDS)
        await batches.aclose()
        # aclose runs the generator's finally, which cancels the per-rank pumps
        # and destroys the context; a second close must stay a no-op.
        await batches.aclose()
    finally:
        publisher.shutdown()


@pytest.mark.asyncio
async def test_heartbeats_keep_idle_sources_live_and_publisher_loss_terminates(monkeypatch):
    """An idle publisher is healthy, but its disappearance must not leave a hung RPC."""
    monkeypatch.setattr(publisher_module, "HEARTBEAT_INTERVAL_SECONDS", 0.02)
    monkeypatch.setattr(kv_events_module, "_LIVE_SOURCE_TIMEOUT_MS", 100)
    publisher = _publisher()
    batches = stream_batches([publisher._test_source], "kv")
    try:
        heartbeat = await asyncio.wait_for(batches.__anext__(), timeout=1.0)
        assert list(heartbeat.events) == []
        publisher.shutdown()
        with pytest.raises(KvEventsUnavailableError, match="silent"):
            while True:
                await batches.__anext__()
    finally:
        publisher.shutdown()
        await batches.aclose()


def _control_servicer(publisher: ZmqEventPublisher):
    """A Control servicer whose engine publishes on `publisher`'s socket."""
    source = publisher._test_source
    args = SimpleNamespace(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        context_parallel_size=1,
        enable_attention_dp=False,
        gpus_per_node=1,
        kv_cache_config=SimpleNamespace(
            tokens_per_block=32,
            kv_events_config=KVEventsConfig(
                enable_kv_cache_events=True,
                endpoint=f"tcp://127.0.0.1:{source.port}",
                replay_endpoint=source.replay_endpoint,
                topic="kv",
            ),
        ),
    )
    return OpenEngineControlServicer(
        SimpleNamespace(args=args), "test-model", None, bind_host="127.0.0.1"
    )


def test_source_discovery_rejects_ambiguous_multi_rank_ray_placement():
    """One frontend host cannot name publishers placed on arbitrary Ray nodes."""
    args = SimpleNamespace(
        enable_attention_dp=True,
        tensor_parallel_size=2,
        orchestrator_type="ray",
        gpus_per_node=8,
        kv_cache_config=SimpleNamespace(
            kv_events_config=KVEventsConfig(
                enable_kv_cache_events=True,
                endpoint="tcp://*:5557",
                publisher="zmq",
            )
        ),
    )
    with pytest.raises(KvEventsUnavailableError, match="Ray placement"):
        resolve_sources(SimpleNamespace(args=args), "127.0.0.1")


@pytest.mark.asyncio
async def test_subscribe_kv_events_streams_batches_and_stops_cleanly():
    """The servicer path, including the close a disconnecting client triggers."""
    publisher = _publisher(with_replay=True)
    try:
        sequence_number = await _publish_and_wait(
            publisher, KVEventBatch(ts=1.5, events=[_stored()])
        )

        responses = _control_servicer(publisher).SubscribeKvEvents(
            kv_pb2.SubscribeKvEventsRequest(
                start_sequence_number=sequence_number,
                include_snapshot=True,
            ),
            FakeServicerContext(),
        )
        response = await asyncio.wait_for(responses.__anext__(), timeout=_PUBLISH_TIMEOUT_SECONDS)
        assert response.WhichOneof("event") == "batch"
        (event,) = response.batch.events
        assert event.block_stored.token_ids == [5, 6, 7, 8]

        # What grpc.aio does when the client goes away mid-stream: the servicer
        # generator is closed at its yield, and its cleanup must not raise.
        await responses.aclose()
    finally:
        publisher.shutdown()


@pytest.mark.asyncio
async def test_subscribe_kv_events_reports_replay_timeout_as_terminal_error(monkeypatch):
    """An unavailable replay service must fail retryably instead of hanging the RPC."""
    monkeypatch.setattr(kv_events_module, "_REPLAY_TIMEOUT_MS", 20)
    publisher = _publisher()
    publisher._test_source = ResolvedKvEventSource(
        data_parallel_rank=0,
        host=publisher._test_source.host,
        port=publisher._test_source.port,
        replay_endpoint=f"tcp://127.0.0.1:{_unused_tcp_port()}",
    )
    responses = _control_servicer(publisher).SubscribeKvEvents(
        kv_pb2.SubscribeKvEventsRequest(include_snapshot=True), FakeServicerContext()
    )
    try:
        response = await asyncio.wait_for(responses.__anext__(), timeout=_PUBLISH_TIMEOUT_SECONDS)
        assert response.WhichOneof("event") == "error"
        assert response.error.retryable
        assert "replay timed out" in response.error.message
        with pytest.raises(StopAsyncIteration):
            await responses.__anext__()
    finally:
        await responses.aclose()
        publisher.shutdown()


@pytest.mark.asyncio
async def test_slow_subscriber_fails_instead_of_losing_a_batch(monkeypatch):
    """A full compatibility-bridge queue must terminate rather than diverge silently."""
    monkeypatch.setattr(kv_events_module, "_MAX_QUEUED_BATCHES", 1)
    publisher = _publisher()
    batches = stream_batches([publisher._test_source], "kv")
    try:
        first = asyncio.create_task(batches.__anext__())
        for _ in range(_SUBSCRIBE_ATTEMPTS):
            await _publish_and_wait(publisher, KVEventBatch(ts=4.0, events=[_stored()]))
            if first.done():
                break
            await asyncio.sleep(_PROBE_INTERVAL_SECONDS)
        await asyncio.wait_for(first, timeout=_PUBLISH_TIMEOUT_SECONDS)

        for _ in range(3):
            await _publish_and_wait(publisher, KVEventBatch(ts=5.0, events=[_stored()]))
        await asyncio.sleep(_PROBE_INTERVAL_SECONDS)

        with pytest.raises(KvEventsUnavailableError, match="not keeping up"):
            await asyncio.wait_for(batches.__anext__(), timeout=_PUBLISH_TIMEOUT_SECONDS)
    finally:
        await batches.aclose()
        publisher.shutdown()
