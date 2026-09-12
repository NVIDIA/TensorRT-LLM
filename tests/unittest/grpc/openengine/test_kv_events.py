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

from tensorrt_llm._torch.pyexecutor.kv_cache_events import (  # noqa: E402
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVEventBatch,
    ZmqEventPublisher,
)
from tensorrt_llm.grpc.openengine.control import OpenEngineControlServicer  # noqa: E402
from tensorrt_llm.grpc.openengine.kv_events import (  # noqa: E402
    ResolvedKvEventSource,
    stream_batches,
)
from tensorrt_llm.llmapi import KVEventsConfig  # noqa: E402

# Runs on the CPU stage: the publisher is a ZeroMQ socket, not an engine.
pytestmark = pytest.mark.cpu_only

_PUBLISH_TIMEOUT_SECONDS = 5.0
_SUBSCRIBE_ATTEMPTS = 50
_PROBE_INTERVAL_SECONDS = 0.05


def _unused_tcp_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _publisher(*, with_replay: bool = False, topic: str = "kv") -> ZmqEventPublisher:
    port = _unused_tcp_port()
    replay_port = _unused_tcp_port() if with_replay else None
    publisher = ZmqEventPublisher(
        data_parallel_rank=0,
        endpoint=f"tcp://127.0.0.1:{port}",
        replay_endpoint=f"tcp://127.0.0.1:{replay_port}" if replay_port else None,
        topic=topic,
    )
    publisher.start()
    publisher._test_source = ResolvedKvEventSource(
        data_parallel_rank=0,
        host="127.0.0.1",
        port=port,
        replay_endpoint=f"tcp://127.0.0.1:{replay_port}" if replay_port else None,
    )
    return publisher


async def _await_published(publisher: ZmqEventPublisher, count: int) -> None:
    """Wait until the publisher thread has drained `count` batches onto the socket."""
    deadline = asyncio.get_running_loop().time() + _PUBLISH_TIMEOUT_SECONDS
    while publisher.published_batches < count:
        if asyncio.get_running_loop().time() > deadline:
            pytest.fail(f"publisher sent {publisher.published_batches} batches, expected {count}")
        await asyncio.sleep(0.01)


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
        publisher.publish(
            KVEventBatch(
                ts=1.5,
                events=[_stored(), BlockRemoved(block_hashes=[11], medium="GPU")],
                data_parallel_rank=0,
            )
        )
        await _await_published(publisher, 1)

        batches = stream_batches(
            [publisher._test_source], "kv", start_sequence_number=0, include_snapshot=True
        )
        try:
            batch = await asyncio.wait_for(batches.__anext__(), timeout=_PUBLISH_TIMEOUT_SECONDS)
        finally:
            await batches.aclose()
    finally:
        publisher.shutdown()

    assert batch.sequence_number == 0
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
        publisher.publish(KVEventBatch(ts=0.0, events=[AllBlocksCleared()]))
        await _await_published(publisher, 1)

        batches = stream_batches([publisher._test_source], "kv", include_snapshot=True)
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
        for index in range(3):
            publisher.publish(KVEventBatch(ts=float(index), events=[_stored()]))
        await _await_published(publisher, 3)

        batches = stream_batches([publisher._test_source], "kv", start_sequence_number=2)
        try:
            batch = await asyncio.wait_for(batches.__anext__(), timeout=_PUBLISH_TIMEOUT_SECONDS)
        finally:
            await batches.aclose()
    finally:
        publisher.shutdown()

    assert batch.sequence_number == 2


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
async def test_closing_the_stream_releases_its_sockets():
    """A client that disconnects must not leak a subscription for the process's life."""
    publisher = _publisher(with_replay=True)
    try:
        publisher.publish(KVEventBatch(ts=0.0, events=[]))
        await _await_published(publisher, 1)

        batches = stream_batches([publisher._test_source], "kv", include_snapshot=True)
        await asyncio.wait_for(batches.__anext__(), timeout=_PUBLISH_TIMEOUT_SECONDS)
        await batches.aclose()
        # aclose runs the generator's finally, which cancels the per-rank pumps
        # and destroys the context; a second close must stay a no-op.
        await batches.aclose()
    finally:
        publisher.shutdown()


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


@pytest.mark.asyncio
async def test_subscribe_kv_events_streams_batches_and_stops_cleanly():
    """The servicer path, including the close a disconnecting client triggers."""
    publisher = _publisher(with_replay=True)
    try:
        publisher.publish(KVEventBatch(ts=1.5, events=[_stored()]))
        await _await_published(publisher, 1)

        responses = _control_servicer(publisher).SubscribeKvEvents(
            kv_pb2.SubscribeKvEventsRequest(include_snapshot=True), FakeServicerContext()
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
