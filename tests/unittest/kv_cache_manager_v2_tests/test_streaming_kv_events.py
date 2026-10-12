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

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache_events import (
    KVEventBatch,
    StreamingKVCacheEventManager,
    ZmqEventPublisher,
    validate_endpoint_ranges,
    validate_streaming_support,
)
from tensorrt_llm.llmapi.llm_args import KVEventsConfig


def _unused_tcp_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


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


def test_validate_streaming_support_rejects_unsupported_parallelism() -> None:
    config = KVEventsConfig(enable_kv_cache_events=True, endpoint="tcp://*:5557")
    supported = dict(pp_size=1, cp_size=1, ranks_per_host=1, data_parallel_size=1)

    validate_streaming_support(config, **supported)
    with pytest.raises(ValueError, match="pipeline parallelism"):
        validate_streaming_support(config, **{**supported, "pp_size": 2})
    with pytest.raises(ValueError, match="context parallelism"):
        validate_streaming_support(config, **{**supported, "cp_size": 2})


@pytest.mark.parametrize(
    "endpoint,replay_endpoint,ranks_per_host,overlaps",
    [
        # 2 ranks bind 5557-5558 and 5558-5559: rank 1's publish hits rank 0's replay.
        ("tcp://*:5557", "tcp://*:5558", 2, True),
        ("tcp://*:5557", "tcp://*:5558", 1, False),
        ("tcp://*:5557", "tcp://*:5657", 2, False),
        # Replay below the publish base overlaps just the same.
        ("tcp://*:5558", "tcp://*:5557", 2, True),
        ("tcp://*:5557", "tcp://*:5559", 2, False),
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


@pytest.mark.parametrize(
    "endpoint,replay_endpoint,dp_size,overflows",
    [
        ("tcp://*:65535", None, 2, True),
        ("tcp://*:65535", None, 1, False),
        ("tcp://*:65534", None, 2, False),
        ("tcp://*:65534", None, 3, True),
        # The replay endpoint is checked too, not just the publish endpoint.
        ("tcp://*:5557", "tcp://*:65535", 2, True),
        ("tcp://*:5557", "tcp://*:60000", 2, False),
        # ipc/inproc have no port, so the span does not apply.
        ("ipc:///tmp/kv-events", None, 64, False),
    ],
)
def test_validate_endpoint_ranges_rejects_port_overflow(
    endpoint, replay_endpoint, dp_size, overflows
) -> None:
    kwargs = {"replay_endpoint": replay_endpoint} if replay_endpoint else {}
    config = KVEventsConfig(enable_kv_cache_events=True, endpoint=endpoint, **kwargs)
    # ranks_per_host=1 isolates the span check from the overlap check.
    if overflows:
        with pytest.raises(ValueError, match="above the maximum port 65535"):
            validate_endpoint_ranges(config, 1, dp_size)
    else:
        validate_endpoint_ranges(config, 1, dp_size)


def test_validate_streaming_support_rejects_overflowing_port_span() -> None:
    """Every rank must reject the span, so none reaches the following collective."""
    config = KVEventsConfig(enable_kv_cache_events=True, endpoint="tcp://*:65535")
    supported = dict(pp_size=1, cp_size=1, ranks_per_host=1)
    # One rank fits; two do not, and rank 0 must refuse it just as rank 1 would.
    validate_streaming_support(config, **supported, data_parallel_size=1)
    with pytest.raises(ValueError, match="above the maximum port 65535"):
        validate_streaming_support(config, **supported, data_parallel_size=2)


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
