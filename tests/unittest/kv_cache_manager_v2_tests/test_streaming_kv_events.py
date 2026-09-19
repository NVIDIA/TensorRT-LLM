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
from typing import Callable

import msgspec
import pytest
import zmq

from tensorrt_llm._torch.pyexecutor.kv_cache_events import (
    AllBlocksCleared,
    BlockStored,
    KVEventBatch,
    ZmqEventPublisher,
    validate_endpoint_ranges,
    validate_streaming_support,
)
from tensorrt_llm._torch.pyexecutor.native_kv_cache_events import (
    NativeKVCacheEventPublisher,
    _NativeEventTranslator,
)
from tensorrt_llm.llmapi.llm_args import KVEventsConfig
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    KVCacheEvent,
    KVCacheEventDiff,
    KVCacheRemovedData,
    KVCacheStoredBlockData,
    KVCacheStoredData,
    KVCacheUpdatedData,
    UniqueToken,
)

_ZMQ_SETUP_ATTEMPTS = 4
_RECEIVE_TIMEOUT_MS = 2_000
_SUBSCRIBE_ATTEMPTS = 50
_PROBE_TIMEOUT_MS = 100


class _NotReceived(Exception):
    """No batch arrived because the subscription had not propagated yet."""


def _unused_tcp_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _await_subscription(publisher: ZmqEventPublisher, subscriber: zmq.Socket) -> None:
    for _ in range(_SUBSCRIBE_ATTEMPTS):
        publisher.publish(KVEventBatch(ts=0.0, events=[]))
        if subscriber.poll(_PROBE_TIMEOUT_MS):
            while subscriber.poll(0):
                subscriber.recv_multipart()
            return
    raise _NotReceived("subscription never propagated")


def _run_on_fresh_port(scenario: Callable[[int], None]) -> None:
    for _ in range(_ZMQ_SETUP_ATTEMPTS):
        try:
            scenario(_unused_tcp_port())
            return
        except _NotReceived:
            pass
        except zmq.ZMQError as error:
            if error.errno != zmq.EADDRINUSE:
                raise
    pytest.fail(f"ZeroMQ setup failed after {_ZMQ_SETUP_ATTEMPTS} attempts")


def test_zmq_publisher_round_trip_and_bounded_shutdown() -> None:
    topic = "kv-events"
    context = zmq.Context.instance()

    def scenario(port: int) -> None:
        endpoint = f"tcp://127.0.0.1:{port}"
        subscriber = context.socket(zmq.SUB)
        subscriber.setsockopt_string(zmq.SUBSCRIBE, topic)
        subscriber.connect(endpoint)
        publisher = ZmqEventPublisher(
            data_parallel_rank=0,
            endpoint=endpoint,
            topic=topic,
            max_queue_size=8,
        )
        try:
            publisher.start()
            _await_subscription(publisher, subscriber)
            assert publisher.publish(
                KVEventBatch(
                    ts=1.0,
                    events=[
                        BlockStored(
                            block_hashes=[7],
                            parent_block_hash=None,
                            token_ids=[1, 2],
                            block_size=2,
                            lora_id=None,
                            medium="GPU",
                            lora_name=None,
                        )
                    ],
                )
            )
            if not subscriber.poll(_RECEIVE_TIMEOUT_MS):
                raise _NotReceived(port)
            frames = subscriber.recv_multipart()
            assert frames[0] == topic.encode()
            decoded = msgspec.msgpack.decode(frames[2])
            assert decoded[2] == 0
            assert decoded[1][0]["type"] == "BlockStored"
            assert decoded[1][0]["block_hashes"] == [7]
        finally:
            publisher.shutdown()
            subscriber.close(linger=0)

        replacement = context.socket(zmq.PUB)
        replacement.bind(endpoint)
        replacement.close(linger=0)

    _run_on_fresh_port(scenario)


def test_queue_overflow_schedules_explicit_resynchronization() -> None:
    publisher = ZmqEventPublisher(
        data_parallel_rank=0,
        endpoint="inproc://kv-events-drop-test",
        max_queue_size=1,
    )
    try:
        assert publisher.publish(KVEventBatch(ts=0.0, events=[])) is True
        assert publisher.publish(KVEventBatch(ts=1.0, events=[])) is False
        assert publisher.dropped_batches == 1
        assert publisher._resync_required.is_set()
        queued_epoch, payload = publisher._event_queue.get_nowait()
        assert queued_epoch < publisher._queue_epoch
        assert msgspec.msgpack.decode(payload)[0] == 0.0
    finally:
        publisher.shutdown()


def test_validate_streaming_support_rejects_unsupported_setups() -> None:
    config = KVEventsConfig(enable_kv_cache_events=True, endpoint="tcp://*:5557")
    supported = dict(pp_size=1, cp_size=1, ranks_per_host=1, data_parallel_size=1, backend="python")

    validate_streaming_support(config, **supported)
    validate_streaming_support(config, **{**supported, "backend": "cpp"})

    with pytest.raises(ValueError, match="pipeline parallelism"):
        validate_streaming_support(config, **{**supported, "pp_size": 2})
    with pytest.raises(ValueError, match="context parallelism"):
        validate_streaming_support(config, **{**supported, "cp_size": 2})
    with pytest.raises(ValueError, match="Unsupported KV cache manager V2 backend"):
        validate_streaming_support(config, **{**supported, "backend": "unknown"})


def test_native_event_translation_is_conservative_and_detects_gaps() -> None:
    high_hash = 2**63 + 5
    signed_hash = high_hash - 2**64
    block = KVCacheStoredBlockData(
        high_hash,
        [UniqueToken(1), UniqueToken(2)],
        cache_level=0,
        priority=35,
    )
    translator = _NativeEventTranslator(
        block_size=2,
        max_window_size=128,
        target_layer_group_id=0,
    )

    stored = translator.translate(
        [KVCacheEvent(0, KVCacheStoredData(None, [block]), 128, "v1_block_key", 0, 0)]
    )
    migrated = translator.translate(
        [
            KVCacheEvent(
                1,
                KVCacheUpdatedData(high_hash, KVCacheEventDiff(0, 1), None),
                128,
                "v1_block_key",
                0,
                0,
            ),
            KVCacheEvent(
                3,
                KVCacheRemovedData([high_hash]),
                128,
                "v1_block_key",
                0,
                0,
            ),
        ]
    )

    stored_wire = msgspec.msgpack.decode(msgspec.msgpack.encode(stored))
    assert stored_wire[0]["block_hashes"] == [signed_hash]
    assert stored_wire[0]["medium"] == "GPU"
    assert [type(event).__name__ for event in migrated] == [
        "BlockRemoved",
        "AllBlocksCleared",
        "BlockRemoved",
    ]


def test_native_event_translation_suppresses_unroutable_descendants() -> None:
    translator = _NativeEventTranslator(
        block_size=2,
        max_window_size=128,
        target_layer_group_id=0,
    )
    unsupported = KVCacheStoredBlockData(
        17,
        [UniqueToken("multimodal-digest"), UniqueToken(2)],
        cache_level=0,
        priority=35,
    )
    descendant = KVCacheStoredBlockData(
        18,
        [UniqueToken(3), UniqueToken(4)],
        cache_level=0,
        priority=35,
    )

    assert (
        translator.translate(
            [KVCacheEvent(0, KVCacheStoredData(None, [unsupported]), 128, "v1_block_key", 0, 0)]
        )
        == []
    )
    assert (
        translator.translate(
            [KVCacheEvent(1, KVCacheStoredData(17, [descendant]), 128, "v1_block_key", 0, 0)]
        )
        == []
    )
    assert (
        translator.translate(
            [KVCacheEvent(2, KVCacheRemovedData([17, 18]), 128, "v1_block_key", 0, 0)]
        )
        == []
    )


def test_terminal_translation_error_publishes_clear_immediately() -> None:
    malformed = KVCacheEvent(
        0,
        KVCacheRemovedData([7]),
        128,
        "v2_sha256",
        0,
        0,
    )

    class FakeEventManager:
        dropped_event_count = 0
        queue_high_watermark = 0
        closed_and_empty = True

        def __init__(self) -> None:
            self._batches = [[malformed], []]

        def get_latest_events(self, timeout_ms=None, max_events=None):
            return self._batches.pop(0)

        def close(self) -> None:
            return

        def discard_events(self) -> int:
            return self.dropped_event_count

    class CapturingPublisher:
        def __init__(self) -> None:
            self.batches = []

        def start(self) -> None:
            return

        def publish(self, batch) -> bool:
            self.batches.append(batch)
            return True

        def shutdown(self) -> None:
            return

    native = NativeKVCacheEventPublisher(
        KVEventsConfig(enable_kv_cache_events=True, publisher="null"),
        FakeEventManager(),
        data_parallel_rank=0,
        block_size=2,
        max_window_size=128,
        window_sizes_by_layer_group={0: 128},
    )
    publisher = CapturingPublisher()
    native._publisher = publisher
    native._closed = True
    native._reader_thread()

    assert len(publisher.batches) == 1
    assert isinstance(publisher.batches[0].events[0], AllBlocksCleared)


def test_native_queue_loss_discards_co_returned_events_before_clear() -> None:
    stored = KVCacheEvent(
        0,
        KVCacheStoredData(
            None,
            [KVCacheStoredBlockData(7, [UniqueToken(1), UniqueToken(2)], 0, 35)],
        ),
        128,
        "v1_block_key",
        0,
        0,
    )

    class FakeEventManager:
        queue_high_watermark = 1
        closed_and_empty = True

        def __init__(self) -> None:
            self.dropped_event_count = 0
            self._reads = 0

        def get_latest_events(self, timeout_ms=None, max_events=None):
            self._reads += 1
            if self._reads == 1:
                self.dropped_event_count = 1
                return [stored]
            return []

        def close(self) -> None:
            return

        def discard_events(self) -> int:
            return self.dropped_event_count

    class CapturingPublisher:
        def __init__(self) -> None:
            self.batches = []

        def publish(self, batch) -> bool:
            self.batches.append(batch)
            return True

    manager = FakeEventManager()
    native = NativeKVCacheEventPublisher(
        KVEventsConfig(enable_kv_cache_events=True, publisher="null"),
        manager,
        data_parallel_rank=0,
        block_size=2,
        max_window_size=128,
        window_sizes_by_layer_group={0: 128},
    )
    publisher = CapturingPublisher()
    native._publisher = publisher
    native._closed = True

    native._reader_thread()

    assert len(publisher.batches) == 1
    assert isinstance(publisher.batches[0].events[0], AllBlocksCleared)


@pytest.mark.parametrize(
    "endpoint,replay_endpoint,ranks_per_host,overlaps",
    [
        ("tcp://*:5557", "tcp://*:5558", 2, True),
        ("tcp://*:5557", "tcp://*:5558", 1, False),
        ("tcp://*:5557", "tcp://*:5657", 2, False),
        ("tcp://*:5558", "tcp://*:5557", 2, True),
        ("tcp://*:5557", None, 8, False),
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


@pytest.mark.parametrize(
    "endpoint,replay_endpoint,dp_size,overflows",
    [
        ("tcp://*:65535", None, 2, True),
        ("tcp://*:65535", None, 1, False),
        ("tcp://*:65534", None, 2, False),
        ("tcp://*:65534", None, 3, True),
        ("tcp://*:5557", "tcp://*:65535", 2, True),
        ("ipc:///tmp/kv-events", None, 64, False),
    ],
)
def test_validate_endpoint_ranges_rejects_port_overflow(
    endpoint, replay_endpoint, dp_size, overflows
) -> None:
    kwargs = {"replay_endpoint": replay_endpoint} if replay_endpoint else {}
    config = KVEventsConfig(enable_kv_cache_events=True, endpoint=endpoint, **kwargs)
    if overflows:
        with pytest.raises(ValueError, match="above the maximum port 65535"):
            validate_endpoint_ranges(config, 1, dp_size)
    else:
        validate_endpoint_ranges(config, 1, dp_size)


def test_kv_events_config_publisher_default() -> None:
    assert KVEventsConfig(enable_kv_cache_events=True).publisher == "zmq"
    assert KVEventsConfig().publisher == "null"
    assert KVEventsConfig(enable_kv_cache_events=False).publisher == "null"
    assert KVEventsConfig(enable_kv_cache_events=True, publisher="null").publisher == "null"


@pytest.mark.parametrize(
    "endpoint,rank,expected",
    [
        ("tcp://*:5557", 0, "tcp://*:5557"),
        ("tcp://*:5557", 3, "tcp://*:5560"),
        ("tcp://127.0.0.1:5557", 1, "tcp://127.0.0.1:5558"),
        ("ipc:///tmp/kv-events", 2, "ipc:///tmp/kv-events_dp2"),
        ("inproc://kv-events", 2, "inproc://kv-events_dp2"),
        (None, 5, None),
    ],
)
def test_offset_endpoint_port(endpoint, rank, expected) -> None:
    assert ZmqEventPublisher.offset_endpoint_port(endpoint, rank) == expected


def test_offset_endpoint_port_rejects_bad_input() -> None:
    for endpoint in (
        "tcp://*:65535",
        "http://host:5557",
        "tcp://host",
        "tcp://host:abc",
        "tcp://host:0",
        "tcp://host:-5",
    ):
        with pytest.raises(ValueError):
            ZmqEventPublisher.offset_endpoint_port(endpoint, 1)
