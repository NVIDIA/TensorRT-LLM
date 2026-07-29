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
import zmq

from tensorrt_llm._torch.pyexecutor.kv_cache_events import KVEventAdapter, NativeKVCacheEventManager
from tensorrt_llm.llmapi.llm_args import KVEventsConfig


def _unused_tcp_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_native_fast_path_publishes_only_full_max_window_blocks():
    """Protect radix hash reuse, filtering, wire format, and shutdown."""
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
    )
    manager = NativeKVCacheEventManager(
        adapter,
        block_size=4,
        max_window_size=128,
    )
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

    first_hash = b"\x11" * 24 + b"\x80\x00\x00\x00\x00\x00\x00\x01"
    partial_hash = b"\x22" * 32
    second_hash = b"\x33" * 24 + b"\x00\x00\x00\x00\x00\x00\x00\x02"
    first_wire_hash = int.from_bytes(first_hash[-8:], "big")
    second_wire_hash = int.from_bytes(second_hash[-8:], "big")
    first_wire_hash = first_wire_hash - 2**64 if first_wire_hash >= 2**63 else first_wire_hash
    second_wire_hash = second_wire_hash - 2**64 if second_wire_hash >= 2**63 else second_wire_hash
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

    # Native publishing pushes events out-of-band, so the legacy pull API must
    # degrade to an empty result rather than raising.
    assert manager.get_latest_events() == []

    manager.shutdown()
    manager.shutdown()
    adapter.shutdown()
    adapter.shutdown()
    subscriber.close(linger=0)

    replacement = context.socket(zmq.PUB)
    replacement.bind(bind_endpoint)
    replacement.close(linger=0)
