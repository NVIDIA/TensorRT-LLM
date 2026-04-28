# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import threading
from dataclasses import dataclass

import pytest

from tensorrt_llm._utils import KVCacheEventSerializer
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    CacheLevel,
    KVCacheCreatedData,
    KVCacheEvent,
    KVCacheEventManager,
    LifeCycleId,
)


@dataclass
class _Root:
    key: bytes
    lora_task_id: int | None = None
    cache_salt_id: int | None = None


@dataclass
class _Page:
    cache_level: int
    priority: int = 35


Token = int | bytes


class _Block:
    def __init__(
        self, key: bytes, tokens: list[Token], prev: _Root | "_Block", storage: list[_Page]
    ):
        self.key = key
        self.tokens = tokens
        self.prev = prev
        self.storage = storage


def _block(
    byte: int, tokens: list[Token], prev: _Root | _Block, cache_levels: list[int] | None = None
) -> _Block:
    return _Block(
        bytes([byte]) * 32, tokens, prev, [_Page(level) for level in (cache_levels or [0, 0])]
    )


def _event_manager(
    max_kv_event_entries: int = 1024,
    *,
    default_window_size: int = 16,
    **kwargs,
) -> KVCacheEventManager:
    return KVCacheEventManager(
        max_kv_event_entries,
        default_window_size=default_window_size,
        **kwargs,
    )


def test_v2_event_manager_requires_positive_buffer_size():
    with pytest.raises(ValueError, match="max_kv_event_entries"):
        KVCacheEventManager(0)


def test_v2_default_window_size_must_be_set_for_unspecified_window():
    event_manager = KVCacheEventManager(1024)

    with pytest.raises(ValueError, match="default_window_size"):
        event_manager.enqueue_created_event([1])


def test_v2_default_window_size_resolves_unspecified_window():
    event_manager = KVCacheEventManager(1024, default_window_size=4096)

    event_manager.enqueue_created_event([8, 2])
    event_manager.flush()

    events = event_manager.get_latest_events()
    assert len(events) == 1
    assert events[0].window_size == 4096


def test_v2_get_latest_events_blocks_until_event_arrives():
    event_manager = _event_manager()
    received = []

    def reader():
        received.extend(event_manager.get_latest_events(timeout_ms=2000))

    thread = threading.Thread(target=reader)
    thread.start()
    event_manager.enqueue_created_event([1])
    event_manager.flush()
    thread.join(timeout=3)

    assert not thread.is_alive()
    assert len(received) == 1


def test_v2_events_serialize_with_v1_type_names():
    event_manager = KVCacheEventManager(1024, default_window_size=16)
    root = _Root(b"root", lora_task_id=42)
    block0 = _block(1, [1, 2, 3, 4], root)
    block1 = _block(2, [5, 6], block0)

    event_manager.enqueue_created_event([8, 2])
    event_manager.on_blocks_stored([block0, block1])
    event_manager.flush()

    serialized = KVCacheEventSerializer.serialize(event_manager.get_latest_events())

    assert serialized[0]["event_id"] == 0
    assert serialized[0]["window_size"] == 16
    assert serialized[0]["data"]["type"] == "created"
    assert serialized[0]["data"]["num_blocks_per_cache_level"] == [8, 2]

    assert serialized[1]["event_id"] == 1
    assert serialized[1]["data"]["type"] == "stored"
    assert serialized[1]["data"]["parent_hash"] is None
    assert len(serialized[1]["data"]["blocks"]) == 2
    assert serialized[1]["data"]["blocks"][0]["type"] == "stored_block"
    assert serialized[1]["data"]["blocks"][0]["tokens"] == [
        {
            "type": "unique_token",
            "token_id": 1,
            "token_extra_id": 0,
        },
        {
            "type": "unique_token",
            "token_id": 2,
            "token_extra_id": 0,
        },
        {
            "type": "unique_token",
            "token_id": 3,
            "token_extra_id": 0,
        },
        {
            "type": "unique_token",
            "token_id": 4,
            "token_extra_id": 0,
        },
    ]


def test_v2_text_block_hash_matches_v1_algorithm():
    event_manager = _event_manager()
    root = _Root(b"root")
    block0 = _block(1, [1, 2, 3, 4], root)
    block1 = _block(2, [5, 6], block0)

    event_manager.on_blocks_stored([block0, block1])
    event_manager.flush()

    stored = event_manager.get_latest_events()[0].data

    assert stored.blocks[0].block_hash == 944812140882783
    assert stored.blocks[1].block_hash == 12460730416951841444


def test_v2_text_block_hash_mixes_cache_salt_on_first_block():
    event_manager = _event_manager()
    root = _Root(b"root", cache_salt_id=123)
    block0 = _block(1, [1, 2, 3, 4], root)
    block1 = _block(2, [5, 6], block0)

    event_manager.on_blocks_stored([block0, block1])
    event_manager.flush()

    stored = event_manager.get_latest_events()[0].data

    assert stored.blocks[0].block_hash == 6269282712567672529
    assert stored.blocks[1].block_hash == 9075042320698884212


def test_v2_bytes_token_uses_token_extra_id():
    event_manager = _event_manager()
    root = _Root(b"root")
    digest = b"\x01\x02\x03\x04\x05\x06\x07\x08\x09"
    block = _block(1, [digest], root, cache_levels=[0])

    event_manager.on_blocks_stored([block])
    event_manager.flush()

    stored = event_manager.get_latest_events()[0].data
    token = stored.blocks[0].tokens[0]
    assert token.token_id == 0
    assert token.token_extra_id == int.from_bytes(digest[:8], "little")


def test_v2_hash_block_key_rejects_non_text_tokens():
    with pytest.raises(ValueError, match="text tokens"):
        KVCacheEventManager._hash_block_key([b"digest"], 0, None, None)


def test_v2_removed_events_are_coalesced():
    event_manager = _event_manager()
    root = _Root(b"root")
    block0 = _block(1, [1, 2, 3, 4], root)
    block1 = _block(2, [5, 6, 7, 8], block0)

    event_manager.on_blocks_stored([block0, block1])
    event_manager.flush()
    event_manager.get_latest_events()

    event_manager.on_block_removed(block0)
    event_manager.on_block_removed(block1)
    event_manager.flush()

    events = event_manager.get_latest_events()
    assert len(events) == 1
    assert type(events[0].data).__name__ == "KVCacheRemovedData"
    assert events[0].data.block_hashes == [
        944812140882783,
        1938568495691091294,
    ]


def test_v2_stored_event_flushes_pending_removed_event_first():
    event_manager = _event_manager()
    root = _Root(b"root")
    old_block = _block(1, [1, 2, 3, 4], root)
    new_block = _block(3, [9, 10, 11, 12], root)

    event_manager.on_blocks_stored([old_block])
    event_manager.flush()
    event_manager.get_latest_events()

    event_manager.on_block_removed(old_block)
    event_manager.on_blocks_stored([new_block])
    event_manager.flush()

    events = event_manager.get_latest_events()
    assert [type(event.data).__name__ for event in events] == [
        "KVCacheRemovedData",
        "KVCacheStoredData",
    ]


def test_v2_stored_event_only_flushes_same_window_removed_event():
    event_manager = _event_manager()
    root = _Root(b"root")
    old_block = _block(1, [1, 2, 3, 4], root)
    new_block = _block(3, [9, 10, 11, 12], root)

    event_manager.on_blocks_stored([old_block], window_size=8)
    event_manager.flush()
    event_manager.get_latest_events()

    event_manager.on_block_removed(old_block, window_size=8)
    event_manager.on_blocks_stored([new_block], window_size=16)
    event_manager.flush()

    events = event_manager.get_latest_events()
    assert [type(event.data).__name__ for event in events] == [
        "KVCacheStoredData",
        "KVCacheRemovedData",
    ]
    assert [event.window_size for event in events] == [16, 8]


def test_v2_stored_events_are_partitioned_by_lifecycle_window():
    event_manager = KVCacheEventManager(1024, default_window_size=16)
    event_manager.set_life_cycle_window_sizes({0: None, 1: 8})
    root = _Root(b"root")
    block = _block(1, [1, 2, 3, 4], root, cache_levels=[0, 1])

    event_manager.on_blocks_stored([block])
    event_manager.flush()

    events = event_manager.get_latest_events()
    assert [event.window_size for event in events] == [8, 16]
    assert [type(event.data).__name__ for event in events] == [
        "KVCacheStoredData",
        "KVCacheStoredData",
    ]
    assert [event.data.blocks[0].cache_level for event in events] == [1, 0]


def test_v2_removed_events_can_target_lifecycle_window():
    event_manager = KVCacheEventManager(1024, default_window_size=16)
    event_manager.set_life_cycle_window_sizes({0: None, 1: 8})
    root = _Root(b"root")
    block = _block(1, [1, 2, 3, 4], root)

    event_manager.on_blocks_stored([block])
    event_manager.flush()
    event_manager.get_latest_events()

    event_manager.on_block_removed(block, life_cycle=LifeCycleId(0))
    event_manager.flush()
    events = event_manager.get_latest_events()
    assert len(events) == 1
    assert events[0].window_size == 16

    event_manager.on_block_removed(block, life_cycle=LifeCycleId(1))
    event_manager.flush()
    events = event_manager.get_latest_events()
    assert len(events) == 1
    assert events[0].window_size == 8


def test_v2_tree_scoped_removed_events_cover_all_known_windows():
    event_manager = _event_manager()
    event_manager.set_life_cycle_window_sizes({0: None, 1: 8})
    root = _Root(b"root")
    block = _block(1, [1, 2, 3, 4], root)

    event_manager.on_blocks_stored([block])
    event_manager.flush()
    event_manager.get_latest_events()

    event_manager.on_block_removed(block)
    event_manager.flush()

    events = event_manager.get_latest_events()
    assert len(events) == 2
    assert sorted(event.window_size for event in events) == [8, 16]
    assert all(type(event.data).__name__ == "KVCacheRemovedData" for event in events)


def test_v2_event_buffer_drops_oldest_events():
    event_manager = _event_manager(2)

    event_manager.enqueue_created_event([1])
    event_manager.enqueue_created_event([2])
    event_manager.enqueue_created_event([3])
    event_manager.flush()

    events = event_manager.get_latest_events()
    assert [event.event_id for event in events] == [1, 2]


def test_v2_event_ids_are_monotonic():
    event_manager = _event_manager()
    root = _Root(b"root")
    old_block = _block(1, [1, 2, 3, 4], root)
    new_block = _block(2, [5, 6, 7, 8], root)

    event_manager.enqueue_created_event([1])
    event_manager.on_blocks_stored([old_block])
    event_manager.on_block_removed(old_block)
    event_manager.on_blocks_stored([new_block])
    event_manager.flush()

    event_ids = [event.event_id for event in event_manager.get_latest_events()]
    assert event_ids == sorted(event_ids)
    assert len(set(event_ids)) == len(event_ids)


def test_v2_cache_level_updates_are_logical_block_events():
    event_manager = _event_manager()
    root = _Root(b"root")
    block = _block(1, [1, 2, 3, 4], root, cache_levels=[0, 0])

    event_manager.on_blocks_stored([block])
    event_manager.flush()
    event_manager.get_latest_events()

    event_manager.on_block_updated(block, LifeCycleId(0), CacheLevel(0), CacheLevel(1))
    event_manager.on_block_updated(block, LifeCycleId(1), CacheLevel(0), CacheLevel(1))
    event_manager.flush()

    events = event_manager.get_latest_events()
    assert len(events) == 1
    assert type(events[0].data).__name__ == "KVCacheUpdatedData"
    assert events[0].data.cache_level.old_value == 0
    assert events[0].data.cache_level.new_value == 1

    event_manager.on_block_updated(block, LifeCycleId(0), CacheLevel(1), CacheLevel(0))
    event_manager.flush()
    assert event_manager.get_latest_events() == []

    event_manager.on_block_updated(block, LifeCycleId(1), CacheLevel(1), CacheLevel(0))
    event_manager.flush()

    events = event_manager.get_latest_events()
    assert len(events) == 1
    assert events[0].data.cache_level.old_value == 1
    assert events[0].data.cache_level.new_value == 0


def test_v2_cache_level_updates_use_lifecycle_window():
    event_manager = KVCacheEventManager(1024, default_window_size=16)
    event_manager.set_life_cycle_window_sizes({0: None, 1: 8})
    root = _Root(b"root")
    block = _block(1, [1, 2, 3, 4], root, cache_levels=[0, 0])

    event_manager.on_blocks_stored([block])
    event_manager.flush()
    event_manager.get_latest_events()

    event_manager.on_block_updated(block, LifeCycleId(1), CacheLevel(0), CacheLevel(1))
    event_manager.flush()

    events = event_manager.get_latest_events()
    assert len(events) == 1
    assert events[0].window_size == 8
    assert type(events[0].data).__name__ == "KVCacheUpdatedData"
    assert events[0].data.cache_level.old_value == 0
    assert events[0].data.cache_level.new_value == 1


def test_v2_block_hash_cache_drops_after_last_known_window_removed():
    event_manager = KVCacheEventManager(1024, default_window_size=16)
    event_manager.set_life_cycle_window_sizes({0: None, 1: 8})
    root = _Root(b"root")
    block = _block(1, [1, 2, 3, 4], root)
    key = bytes(block.key)

    event_manager.on_blocks_stored([block])
    assert key in event_manager._block_hash_by_key

    event_manager.on_block_removed(block, life_cycle=LifeCycleId(0))
    assert key in event_manager._block_hash_by_key

    event_manager.on_block_removed(block, life_cycle=LifeCycleId(1))
    assert key not in event_manager._block_hash_by_key
    assert key not in event_manager._v1_hash_compatible_keys


def test_v2_attention_dp_flush_gathers_events_to_rank_zero():
    foreign_event = KVCacheEvent(
        event_id=100,
        data=KVCacheCreatedData([2]),
        window_size=16,
        attention_dp_rank=1,
    )

    def gather(local_events):
        return [local_events, [foreign_event]]

    event_manager = KVCacheEventManager(
        1024,
        attention_dp_rank=0,
        attention_dp_gather_rank=0,
        attention_dp_size=2,
        attention_dp_gather_fn=gather,
        default_window_size=16,
    )

    event_manager.enqueue_created_event([1])
    event_manager.flush_iteration_events()

    events = event_manager.get_latest_events()
    assert [event.attention_dp_rank for event in events] == [0, 1]
    assert [event.data.num_blocks_per_cache_level for event in events] == [
        [1],
        [2],
    ]


def test_v2_attention_dp_flush_clears_nonzero_rank_events():
    def gather(local_events):
        return [[], local_events]

    event_manager = KVCacheEventManager(
        1024,
        attention_dp_rank=1,
        attention_dp_gather_rank=1,
        attention_dp_size=2,
        attention_dp_gather_fn=gather,
        default_window_size=16,
    )

    event_manager.enqueue_created_event([1])
    event_manager.flush_iteration_events()

    assert event_manager.get_latest_events() == []


def test_v2_attention_dp_gather_failure_keeps_rank_zero_local_events():
    def gather(local_events):
        raise RuntimeError("simulated gather failure")

    event_manager = KVCacheEventManager(
        1024,
        attention_dp_rank=0,
        attention_dp_gather_rank=0,
        attention_dp_size=2,
        attention_dp_gather_fn=gather,
        default_window_size=16,
    )

    event_manager.enqueue_created_event([1])
    event_manager.flush_iteration_events()

    events = event_manager.get_latest_events()
    assert len(events) == 1
    assert events[0].attention_dp_rank == 0
    assert events[0].data.num_blocks_per_cache_level == [1]


def test_v2_attention_dp_gather_failure_drops_nonzero_rank_local_events():
    def gather(local_events):
        raise RuntimeError("simulated gather failure")

    event_manager = KVCacheEventManager(
        1024,
        attention_dp_rank=1,
        attention_dp_gather_rank=1,
        attention_dp_size=2,
        attention_dp_gather_fn=gather,
        default_window_size=16,
    )

    event_manager.enqueue_created_event([1])
    event_manager.flush_iteration_events()

    assert event_manager.get_latest_events() == []


def test_v2_attention_dp_rank_zero_buffer_scales_by_dp_size():
    event_manager = KVCacheEventManager(
        1,
        attention_dp_rank=0,
        attention_dp_gather_rank=0,
        attention_dp_size=2,
        attention_dp_gather_fn=lambda local: [local, []],
        default_window_size=16,
    )

    event_manager.enqueue_created_event([1])
    event_manager.enqueue_created_event([2])
    event_manager.flush_iteration_events()

    assert [
        event.data.num_blocks_per_cache_level for event in event_manager.get_latest_events()
    ] == [
        [1],
        [2],
    ]
