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

import msgspec
import zmq

from tensorrt_llm._torch.pyexecutor.kv_cache_events import KVEventAdapter
from tensorrt_llm.llmapi.llm_args import KVEventsConfig
from tensorrt_llm.runtime.kv_cache_manager_v2._event_manager import (
    KVCacheEvent,
    KVCacheEventManager,
    KVCacheRemovedData,
    KVCacheStoredBlockData,
    KVCacheStoredData,
    UniqueToken,
)


def _stored_block(block_hash: int, tokens: list[int]) -> KVCacheStoredBlockData:
    return KVCacheStoredBlockData(
        block_hash=block_hash,
        tokens=[UniqueToken(token) for token in tokens],
        cache_level=0,
        priority=0,
    )


def _unused_tcp_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_native_callback_drains_iteration_without_gathered_buffer():
    """The callback must leave the legacy pull buffer empty without a gather."""
    adapter = KVEventAdapter(
        KVEventsConfig(enable_kv_cache_events=True, publisher="null"),
        data_parallel_rank=0,
        block_size=4,
        max_window_size=128,
    )
    manager = KVCacheEventManager(
        50_000,
        window_size=128,
        attention_dp_rank=0,
        attention_dp_gather=adapter.publish_local_events,
    )
    manager.add_stored_event(
        None,
        [_stored_block(11, [1, 2, 3, 4])],
    )

    manager.flush_iteration_events()

    assert adapter.enqueued_batches == 1
    assert adapter.enqueued_events == 1
    assert manager.get_latest_events(timeout_ms=0) == []
    adapter.shutdown()
    adapter.shutdown()


def test_native_zmq_wire_filters_partial_blocks_and_reuses_port():
    """Decode the vLLM wire format while filtering partial block lifecycle."""
    port = _unused_tcp_port()
    bind_endpoint = f"tcp://*:{port}"
    connect_endpoint = f"tcp://127.0.0.1:{port}"
    topic = "kv-events"
    context = zmq.Context.instance()
    subscriber = context.socket(zmq.SUB)
    subscriber.setsockopt_string(zmq.SUBSCRIBE, topic)
    subscriber.connect(connect_endpoint)

    adapter = KVEventAdapter(
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
    time.sleep(0.2)

    full_hash = 2**63 + 5
    partial_hash = 29
    adapter.publish_local_events([
        KVCacheEvent(
            event_id=0,
            data=KVCacheStoredData(
                parent_hash=7,
                blocks=[
                    _stored_block(full_hash, [1, 2, 3, 4]),
                    _stored_block(partial_hash, [5, 6]),
                ],
            ),
            window_size=128,
            attention_dp_rank=0,
        )
    ])
    adapter.publish_local_events([
        KVCacheEvent(
            event_id=1,
            data=KVCacheRemovedData([full_hash, partial_hash]),
            window_size=128,
            attention_dp_rank=0,
        )
    ])

    frames = []
    for _ in range(2):
        assert subscriber.poll(2_000)
        frames.append(subscriber.recv_multipart())

    assert [frame[0] for frame in frames] == [topic.encode(), topic.encode()]
    assert [int.from_bytes(frame[1], "big") for frame in frames] == [0, 1]
    stored_batch = msgspec.msgpack.decode(frames[0][2])
    removed_batch = msgspec.msgpack.decode(frames[1][2])
    assert stored_batch[2] == 0
    assert stored_batch[1] == [{
        "type": "BlockStored",
        "block_hashes": [-(2**63) + 5],
        "parent_block_hash": 7,
        "token_ids": [1, 2, 3, 4],
        "block_size": 4,
        "lora_id": None,
        "medium": "GPU",
        "lora_name": None,
    }]
    assert removed_batch[1] == [{
        "type": "BlockRemoved",
        "block_hashes": [-(2**63) + 5],
        "medium": "GPU",
    }]

    adapter.shutdown()
    adapter.shutdown()
    subscriber.close(linger=0)

    replacement = context.socket(zmq.PUB)
    replacement.bind(bind_endpoint)
    replacement.close(linger=0)
