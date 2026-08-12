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
import time
from types import SimpleNamespace

import msgspec
import pytest
import zmq

from tensorrt_llm._torch.pyexecutor.kv_cache_events import (
    StreamingKVCacheEventManager,
    ZmqEventPublisher,
)
from tensorrt_llm.llmapi.llm_args import KVEventsConfig


def _unused_tcp_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_streaming_fast_path_publishes_only_full_max_window_blocks() -> None:
    """Protect radix hash reuse, filtering, wire format, and shutdown."""
    port = _unused_tcp_port()
    bind_endpoint = f"tcp://*:{port}"
    connect_endpoint = f"tcp://127.0.0.1:{port}"
    topic = "kv-events"
    context = zmq.Context.instance()
    subscriber = context.socket(zmq.SUB)
    subscriber.setsockopt_string(zmq.SUBSCRIBE, topic)
    subscriber.connect(connect_endpoint)

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
    try:
        manager.set_layer_group_window_sizes({0: 128, 1: 64})
        time.sleep(0.2)

        root = SimpleNamespace(ordinal=-1)

        def block(
            key: bytes,
            tokens: list[int],
            prev: object,
        ) -> SimpleNamespace:
            max_window_page = object()
            smaller_window_page = object()
            return SimpleNamespace(
                key=key,
                tokens=tokens,
                prev=prev,
                ordinal=getattr(prev, "ordinal", -1) + 1,
                storage=[
                    lambda: max_window_page,
                    lambda: smaller_window_page,
                ],
            )

        # Wire hashes come from truncate_sha256_hash_to_int64 = the FIRST 8 bytes
        # of the radix key, so put the distinguishing bytes -- including the high
        # bit that exercises the signed-wraparound branch -- at the front.
        first_hash = b"\x80\x00\x00\x00\x00\x00\x00\x01" + b"\x11" * 24
        partial_hash = b"\x22" * 32
        second_hash = b"\x00\x00\x00\x00\x00\x00\x00\x02" + b"\x33" * 24
        first_wire_hash = int.from_bytes(first_hash[:8], "big")
        second_wire_hash = int.from_bytes(second_hash[:8], "big")
        first_wire_hash = first_wire_hash - 2**64 if first_wire_hash >= 2**63 else first_wire_hash
        second_wire_hash = (
            second_wire_hash - 2**64 if second_wire_hash >= 2**63 else second_wire_hash
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
            assert subscriber.poll(2_000)
            frames.append(subscriber.recv_multipart())

        assert [frame[0] for frame in frames] == [topic.encode(), topic.encode()]
        assert [int.from_bytes(frame[1], "big") for frame in frames] == [0, 1]
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

        # shutdown() must be idempotent.
        manager.shutdown()
        manager.shutdown()
    finally:
        manager.shutdown()
        subscriber.close(linger=0)

    replacement = context.socket(zmq.PUB)
    replacement.bind(bind_endpoint)
    replacement.close(linger=0)


def test_streaming_removals_are_never_dropped_by_the_entry_cap() -> None:
    """Removals must survive the per-iteration cap or the consumer desyncs."""
    manager = StreamingKVCacheEventManager(
        KVEventsConfig(enable_kv_cache_events=True, publisher="null"),
        data_parallel_rank=0,
        block_size=2,
        max_window_size=128,
        max_entries=2,
    )
    try:
        manager.set_layer_group_window_sizes({0: 128})

        # Capture what actually reaches the publisher so the test proves the
        # removals are emitted on flush, not merely queued in _pending_events.
        published: list[object] = []
        manager._publisher.publish = lambda batch: published.append(batch) or True

        root = SimpleNamespace(ordinal=-1)

        def block(key: bytes, tokens: list[int], prev: object) -> SimpleNamespace:
            page = object()
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
