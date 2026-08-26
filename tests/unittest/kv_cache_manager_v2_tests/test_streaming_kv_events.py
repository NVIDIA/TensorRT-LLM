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

import socket
from types import SimpleNamespace
from typing import Callable

import msgspec
import pytest
import zmq

from tensorrt_llm._torch.pyexecutor.kv_cache_events import (
    KVEventBatch,
    StreamingKVCacheEventManager,
    ZmqEventPublisher,
    validate_endpoint_ranges,
    validate_streaming_support,
)
from tensorrt_llm.llmapi.llm_args import KVEventsConfig

_ZMQ_SETUP_ATTEMPTS = 4
_RECEIVE_TIMEOUT_MS = 2_000
_SUBSCRIBE_ATTEMPTS = 50
_PROBE_TIMEOUT_MS = 100


class _NotReceived(Exception):
    """No batch arrived: the subscription had not propagated yet."""


def _unused_tcp_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _await_subscription(publisher: ZmqEventPublisher, subscriber: zmq.Socket) -> int:
    """Publish probe batches until the subscriber's subscription is live.

    A PUB socket silently drops everything published before a subscriber's
    subscription has propagated, and that window is not bounded by any delay the test
    can pick -- so synchronise on an actual received message instead of sleeping.
    Returns the number of probes published, which is the sequence number the next real
    batch will carry.
    """
    for probes in range(1, _SUBSCRIBE_ATTEMPTS + 1):
        publisher.publish(KVEventBatch(ts=0.0, events=[]))
        if subscriber.poll(_PROBE_TIMEOUT_MS):
            while subscriber.poll(0):
                subscriber.recv_multipart()
            return probes
    raise _NotReceived("subscription never propagated")


def _run_on_fresh_port(scenario: Callable[[int], None]) -> None:
    """Retry `scenario(port)` on a fresh port if its sockets could not come up.

    `_unused_tcp_port()` releases its port before the publisher binds it, so another
    process can take it in between. Assertion failures inside `scenario` are not
    retried.
    """
    for _ in range(_ZMQ_SETUP_ATTEMPTS):
        try:
            scenario(_unused_tcp_port())
            return
        except _NotReceived:
            pass
        except zmq.ZMQError as exc:
            if exc.errno != zmq.EADDRINUSE:
                raise
    pytest.fail(f"ZeroMQ setup failed after {_ZMQ_SETUP_ATTEMPTS} attempts")


def test_streaming_fast_path_publishes_only_full_max_window_blocks() -> None:
    """Protect radix hash reuse, filtering, wire format, and shutdown."""
    topic = "kv-events"
    context = zmq.Context.instance()

    # Wire hashes come from truncate_sha256_hash_to_int64 = the FIRST 8 bytes
    # of the radix key, so put the distinguishing bytes -- including the high
    # bit that exercises the signed-wraparound branch -- at the front.
    first_hash = b"\x80\x00\x00\x00\x00\x00\x00\x01" + b"\x11" * 24
    partial_hash = b"\x22" * 32
    second_hash = b"\x00\x00\x00\x00\x00\x00\x00\x02" + b"\x33" * 24
    first_wire_hash = int.from_bytes(first_hash[:8], "big") - 2**64
    second_wire_hash = int.from_bytes(second_hash[:8], "big")

    # A fresh manager per attempt restarts sequence numbers at 0 and clears the
    # stored-block dedup state, so a retry replays the scenario exactly.
    def scenario(port: int) -> None:
        bind_endpoint = f"tcp://*:{port}"
        subscriber = context.socket(zmq.SUB)
        subscriber.setsockopt_string(zmq.SUBSCRIBE, topic)
        subscriber.connect(f"tcp://127.0.0.1:{port}")
        manager = None
        try:
            manager = StreamingKVCacheEventManager(
                KVEventsConfig(
                    enable_kv_cache_events=True,
                    publisher="zmq",
                    endpoint=bind_endpoint,
                    topic=topic,
                    max_queue_size=8,
                ),
                data_parallel_rank=0,
                block_size=4,
                max_window_size=128,
            )
            manager.start()
            base_seq = _await_subscription(manager._publisher, subscriber)
            manager.set_layer_group_window_sizes({0: 128, 1: 64})

            root = SimpleNamespace(ordinal=-1)

            def block(key: bytes, tokens: list[int], prev: object) -> SimpleNamespace:
                max_window_page = SimpleNamespace(num_tokens_in_block=len(tokens))
                smaller_window_page = SimpleNamespace(num_tokens_in_block=len(tokens))
                return SimpleNamespace(
                    key=key,
                    tokens=tokens,
                    prev=prev,
                    ordinal=getattr(prev, "ordinal", -1) + 1,
                    storage=[lambda: max_window_page, lambda: smaller_window_page],
                )

            first = block(first_hash, [1, 2, 3, 4], root)
            partial = block(partial_hash, [5, 6], first)
            second = block(second_hash, [5, 6, 7, 8], first)

            manager.add_stored_block_event_from_block(first)
            manager.add_stored_block_event_from_block(partial)
            manager.add_stored_life_cycle_event_from_block(second, 1)
            manager.add_stored_life_cycle_event_from_block(second, 0)
            manager.flush_iteration_events()
            manager.add_removed_event([first_hash, partial_hash, second_hash])
            manager.flush_iteration_events()

            frames = []
            for _ in range(2):
                if not subscriber.poll(_RECEIVE_TIMEOUT_MS):
                    raise _NotReceived(port)
                frames.append(subscriber.recv_multipart())

            assert [frame[0] for frame in frames] == [topic.encode(), topic.encode()]
            # Sequence numbers stay dense across the probes and the real batches.
            assert [int.from_bytes(frame[1], "big") for frame in frames] == [
                base_seq,
                base_seq + 1,
            ]
            stored_batch = msgspec.msgpack.decode(frames[0][2])
            removed_batch = msgspec.msgpack.decode(frames[1][2])
            assert stored_batch[2] == 0
            assert stored_batch[1] == [
                {
                    "type": "BlockStored",
                    "block_hashes": [first_wire_hash, second_wire_hash],
                    "parent_block_hash": None,
                    "token_ids": [1, 2, 3, 4, 5, 6, 7, 8],
                    "block_size": 4,
                    "lora_id": None,
                    "medium": "GPU",
                    "lora_name": None,
                }
            ]
            assert removed_batch[1] == [
                {
                    "type": "BlockRemoved",
                    "block_hashes": [first_wire_hash, second_wire_hash],
                    "medium": "GPU",
                }
            ]
            assert manager.stored_blocks == 2
            assert manager.removed_blocks == 2
            assert manager.partial_blocks_suppressed == 1
            assert manager.non_target_life_cycles_ignored == 1
            assert manager.dropped_events == 0

            # Streaming publishing pushes events out-of-band, so the buffered pull
            # API must degrade to an empty result rather than raising.
            assert manager.get_latest_events() == []

            # shutdown() must be idempotent and must release the bound port.
            manager.shutdown()
            manager.shutdown()
            replacement = context.socket(zmq.PUB)
            replacement.bind(bind_endpoint)
            replacement.close(linger=0)
        finally:
            if manager is not None:
                manager.shutdown()
            subscriber.close(linger=0)

    _run_on_fresh_port(scenario)


def test_streaming_removals_are_never_dropped_by_the_entry_cap() -> None:
    """Removals must survive the per-iteration cap or the consumer desyncs."""
    manager = StreamingKVCacheEventManager(
        KVEventsConfig(enable_kv_cache_events=True, publisher="null"),
        data_parallel_rank=0,
        block_size=2,
        max_window_size=128,
        max_entries=2,
    )
    manager.start()
    try:
        manager.set_layer_group_window_sizes({0: 128})

        # Capture what actually reaches the publisher so the test proves the
        # removals are emitted on flush, not merely queued in _pending_events.
        published: list[object] = []
        manager._publisher.publish = lambda batch: published.append(batch) or True

        root = SimpleNamespace(ordinal=-1)

        def block(key: bytes, tokens: list[int], prev: object) -> SimpleNamespace:
            page = SimpleNamespace(num_tokens_in_block=len(tokens))
            return SimpleNamespace(
                key=key,
                tokens=tokens,
                prev=prev,
                ordinal=getattr(prev, "ordinal", -1) + 1,
                storage=[lambda: page],
            )

        first = block(b"\x01" * 32, [1, 2], root)
        second = block(b"\x02" * 32, [3, 4], first)
        manager.add_stored_block_event_from_block(first)
        manager.add_stored_block_event_from_block(second)

        # Both stores fill the entry cap (max_entries=2); the removals must still
        # be emitted rather than dropped, or the consumer treats the blocks as
        # resident forever.
        manager.add_removed_event([b"\x01" * 32, b"\x02" * 32])
        manager.flush_iteration_events()

        assert manager.removed_blocks == 2
        assert len(published) == 1
        # Round-trip through msgpack to prove the removals reach the wire as a
        # BlockRemoved batch carrying both hashes.
        decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(published[0]))
        removed = [event for event in decoded[1] if event.get("type") == "BlockRemoved"]
        assert sum(len(event["block_hashes"]) for event in removed) == 2
    finally:
        manager.shutdown()


def test_dropped_batches_leave_a_sequence_gap() -> None:
    """A batch lost to a full queue must be observable as a missing sequence number."""
    # Left unstarted on purpose: publish() only touches the queue, so the drop path is
    # exercised without binding a socket or draining the queue from a live thread.
    publisher = ZmqEventPublisher(
        data_parallel_rank=0,
        endpoint="inproc://kv-events-drop-test",
        max_queue_size=1,
    )
    try:
        assert publisher.publish(KVEventBatch(ts=0.0, events=[])) is True
        assert publisher.publish(KVEventBatch(ts=1.0, events=[])) is False
        assert publisher.dropped_batches == 1

        # The accepted batch kept seq 0 and the dropped batch consumed seq 1, so the
        # next batch is seq 2: subscribers see a hole rather than a contiguous stream
        # that hides the loss.
        seq, _ = publisher._event_queue.get_nowait()
        assert seq == 0
        assert publisher.publish(KVEventBatch(ts=2.0, events=[])) is True
        next_seq, _ = publisher._event_queue.get_nowait()
        assert next_seq == 2
    finally:
        publisher.shutdown()


def test_construction_binds_nothing_until_start() -> None:
    """A constructed-but-unstarted publisher must hold no socket and no thread."""
    port = _unused_tcp_port()
    endpoint = f"tcp://127.0.0.1:{port}"
    manager = StreamingKVCacheEventManager(
        KVEventsConfig(enable_kv_cache_events=True, publisher="zmq", endpoint=endpoint),
        data_parallel_rank=0,
        block_size=4,
        max_window_size=128,
    )
    try:
        publisher = manager._publisher
        assert publisher._pub is None
        assert publisher._thread is None

        manager.start()
        assert publisher._pub is not None
        assert publisher._thread is not None and publisher._thread.is_alive()
        # start() is idempotent.
        manager.start()
    finally:
        manager.shutdown()


def test_shutdown_without_start_is_safe() -> None:
    """Tearing down a manager that never started must not raise."""
    manager = StreamingKVCacheEventManager(
        KVEventsConfig(enable_kv_cache_events=True, publisher="zmq", endpoint="tcp://127.0.0.1:1"),
        data_parallel_rank=0,
        block_size=4,
        max_window_size=128,
    )
    # Never started, so nothing was bound -- shutdown must still be a clean no-op.
    manager.shutdown()
    manager.shutdown()


def test_validate_streaming_support_rejects_unsupported_setups() -> None:
    config = KVEventsConfig(enable_kv_cache_events=True, endpoint="tcp://*:5557")
    supported = dict(pp_size=1, cp_size=1, ranks_per_host=1, data_parallel_size=1, backend="python")

    # The supported baseline must not raise, or the negative cases prove nothing.
    validate_streaming_support(config, **supported)

    with pytest.raises(ValueError, match="pipeline parallelism"):
        validate_streaming_support(config, **{**supported, "pp_size": 2})
    with pytest.raises(ValueError, match="context parallelism"):
        validate_streaming_support(config, **{**supported, "cp_size": 2})
    # The default backend is "cpp", whose nanobind KVCacheManager cannot accept a
    # duck-typed Python event sink; the error must name the env var that fixes it.
    with pytest.raises(ValueError, match="TLLM_KV_CACHE_MANAGER_V2_BACKEND=python"):
        validate_streaming_support(config, **{**supported, "backend": "cpp"})


@pytest.mark.parametrize(
    "endpoint,replay_endpoint,ranks_per_host,overlaps",
    [
        # 2 ranks bind 5557-5558 and 5558-5559: rank 1's publish hits rank 0's replay.
        ("tcp://*:5557", "tcp://*:5558", 2, True),
        ("tcp://*:5557", "tcp://*:5558", 1, False),
        ("tcp://*:5557", "tcp://*:5657", 2, False),
        # Replay below the publish base overlaps just the same.
        ("tcp://*:5558", "tcp://*:5557", 2, True),
        (
            "tcp://*:5557",
            "tcp://*:5559",
            2,
            False,
        ),
        # No replay endpoint means no second range to collide with.
        ("tcp://*:5557", None, 8, False),
        # 16 attention-DP ranks over 2 nodes collide only within a node, so spacing
        # equal to the per-host rank count is legal even though it is under dp_size.
        ("tcp://*:5557", "tcp://*:5565", 8, False),
        # ipc/inproc endpoints have no ports, so the check does not apply.
        ("ipc:///tmp/kv-events", "ipc:///tmp/kv-replay", 8, False),
    ],
)
def test_validate_endpoint_ranges(endpoint, replay_endpoint, ranks_per_host, overlaps) -> None:
    kwargs = {"replay_endpoint": replay_endpoint} if replay_endpoint else {}
    config = KVEventsConfig(enable_kv_cache_events=True, endpoint=endpoint, **kwargs)
    if overlaps:
        with pytest.raises(ValueError, match="overlap"):
            validate_endpoint_ranges(config, ranks_per_host, ranks_per_host)
    else:
        validate_endpoint_ranges(config, ranks_per_host, ranks_per_host)


def test_partial_target_page_coverage_is_suppressed_until_fully_covered() -> None:
    """A page adopted from a shorter sibling must not be published as a full block."""
    manager = StreamingKVCacheEventManager(
        KVEventsConfig(enable_kv_cache_events=True, publisher="null"),
        data_parallel_rank=0,
        block_size=4,
        max_window_size=128,
    )
    manager.start()
    try:
        manager.set_layer_group_window_sizes({0: 128})
        published: list[object] = []
        manager._publisher.publish = lambda batch: published.append(batch) or True

        root = SimpleNamespace(ordinal=-1)
        # The block holds 4 tokens but its target page only covers 2 of them.
        page = SimpleNamespace(num_tokens_in_block=2)
        block = SimpleNamespace(
            key=b"\x01" * 32,
            tokens=[1, 2, 3, 4],
            prev=root,
            ordinal=0,
            storage=[lambda: page],
        )

        manager.add_stored_block_event_from_block(block)
        manager.flush_iteration_events()
        assert manager.stored_blocks == 0
        assert manager.partial_blocks_suppressed == 1
        assert published == []

        # Once the page covers the whole block, the same block is published.
        page.num_tokens_in_block = 4
        manager.add_stored_life_cycle_event_from_block(block, 0)
        manager.flush_iteration_events()
        assert manager.stored_blocks == 1
        assert len(published) == 1
        decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(published[0]))
        stored = [event for event in decoded[1] if event["type"] == "BlockStored"]
        assert sum(len(event["block_hashes"]) for event in stored) == 1
    finally:
        manager.shutdown()


def test_life_cycle_hooks_ignore_none_ids() -> None:
    """A None life-cycle id must not reach int() before the target is configured."""
    manager = StreamingKVCacheEventManager(
        KVEventsConfig(enable_kv_cache_events=True, publisher="null"),
        data_parallel_rank=0,
        block_size=4,
        max_window_size=128,
    )
    manager.start()
    try:
        # Before set_layer_group_window_sizes(), and with a None id, both hooks are
        # no-ops rather than raising TypeError.
        manager.add_stored_life_cycle_event_from_block(object(), None)
        manager.add_removed_life_cycle_event(b"\x01" * 32, None)
        manager.set_layer_group_window_sizes({0: 128})
        manager.add_stored_life_cycle_event_from_block(object(), None)
        manager.add_removed_life_cycle_event(b"\x01" * 32, None)
        assert manager.stored_blocks == 0
        assert manager.removed_blocks == 0
    finally:
        manager.shutdown()


def test_kv_events_config_publisher_default() -> None:
    """model_post_init resolves the publisher default (the common user path)."""
    assert KVEventsConfig(enable_kv_cache_events=True).publisher == "zmq"
    assert KVEventsConfig().publisher == "null"
    assert KVEventsConfig(enable_kv_cache_events=False).publisher == "null"
    # An explicitly set publisher is always respected.
    assert KVEventsConfig(enable_kv_cache_events=True, publisher="null").publisher == "null"
    assert KVEventsConfig(enable_kv_cache_events=False, publisher="zmq").publisher == "zmq"


@pytest.mark.parametrize(
    "endpoint,rank,expected",
    [
        ("tcp://*:5557", 0, "tcp://*:5557"),  # rank 0 is identity
        ("tcp://*:5557", 3, "tcp://*:5560"),  # tcp base_port + rank
        ("tcp://127.0.0.1:5557", 1, "tcp://127.0.0.1:5558"),
        ("ipc:///tmp/kv-events", 2, "ipc:///tmp/kv-events_dp2"),  # no port -> suffix
        ("inproc://kv-events", 2, "inproc://kv-events_dp2"),
        (None, 5, None),
    ],
)
def test_offset_endpoint_port(endpoint, rank, expected) -> None:
    assert ZmqEventPublisher.offset_endpoint_port(endpoint, rank) == expected


def test_offset_endpoint_port_rejects_bad_input() -> None:
    # base_port + rank must stay within the u16 range.
    with pytest.raises(ValueError):
        ZmqEventPublisher.offset_endpoint_port("tcp://*:65535", 1)
    # Unknown scheme is rejected for a non-zero rank.
    with pytest.raises(ValueError):
        ZmqEventPublisher.offset_endpoint_port("http://host:5557", 1)
    # A TCP endpoint without a port is rejected instead of raising an opaque
    # int() error on the scheme colon.
    with pytest.raises(ValueError):
        ZmqEventPublisher.offset_endpoint_port("tcp://host", 1)
    # Non-numeric or out-of-range ports are rejected with an endpoint-naming
    # error instead of an opaque int()/bind failure.
    for bad in ("tcp://host:abc", "tcp://host:0", "tcp://host:-5"):
        with pytest.raises(ValueError):
            ZmqEventPublisher.offset_endpoint_port(bad, 1)
