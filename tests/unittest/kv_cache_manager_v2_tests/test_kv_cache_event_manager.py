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

import gc
import os
import threading
import time
from importlib.util import find_spec
from typing import TYPE_CHECKING, cast

import pytest

from tensorrt_llm._utils import KVCacheEventSerializer
from tensorrt_llm.runtime.kv_cache_hash import (
    KV_CACHE_HASH_ALGO_V1,
    KV_CACHE_HASH_ALGO_V2_SHA256_64,
    truncate_sha256_hash_to_int64,
)
from tensorrt_llm.runtime.kv_cache_manager_v2._event_manager import (
    KVCacheEventDiff,
    KVCacheEventManager,
    KVCacheStoredBlockData,
    UniqueToken,
)

if not TYPE_CHECKING and find_spec("kv_cache_manager_v2") is not None:
    from kv_cache_manager_v2 import CacheLevel, CudaStream, KVCacheManager, TokenId
    from kv_cache_manager_v2._block_radix_tree import Block, RootBlock
    from kv_cache_manager_v2._utils import CachedCudaStream, init_cuda_once, temporary_sys_path
else:
    from tensorrt_llm.runtime.kv_cache_manager_v2 import (
        CacheLevel,
        CudaStream,
        KVCacheManager,
        TokenId,
    )
    from tensorrt_llm.runtime.kv_cache_manager_v2._block_radix_tree import Block, RootBlock
    from tensorrt_llm.runtime.kv_cache_manager_v2._utils import (
        CachedCudaStream,
        init_cuda_once,
        temporary_sys_path,
    )

try:
    import torch
except ImportError:
    torch = None


class _FakePage:
    def __init__(self, cache_level=CacheLevel(0), priority=0):
        self.cache_level = cache_level
        self.priority = priority


class _FakePageRef:
    def __init__(self, page):
        self._page = page

    def __call__(self):
        return self._page


class _FakeRootBlock:
    ordinal = -1


class _FakeBlock:
    def __init__(self, key, tokens, num_life_cycles=1, prev=None):
        self.key = key
        self.tokens = tokens
        self.prev = prev or _FakeRootBlock()
        self.ordinal = getattr(self.prev, "ordinal", -1) + 1
        self.storage = [_FakePageRef(_FakePage()) for _ in range(num_life_cycles)]


with temporary_sys_path(os.path.dirname(os.path.abspath(__file__))):
    from test_kv_cache_manager_v2 import create_config


def _create_test_manager(
    event_manager,
    *,
    tokens_per_block=4,
    gpu_quota=16 << 20,
    host_quota=0,
    window_size=None,
    kv_buf_size=8192,
):
    return KVCacheManager(
        create_config(
            tokens_per_block=tokens_per_block,
            gpu_quota=gpu_quota,
            host_quota=host_quota,
            disk_quota=0,
            num_layers=2,
            window_size=window_size,
            sink_tokens=0,
            kv_buf_size=kv_buf_size,
        ),
        event_manager=event_manager,
    )


def _token_ids(start, end):
    return [TokenId(token_id) for token_id in range(start, end)]


def _flush_serialized_events(event_manager):
    event_manager.flush_iteration_events()
    return KVCacheEventSerializer.serialize(event_manager.get_latest_events(0))


def _stored_events(events):
    return [event for event in events if event["data"]["type"] == "stored"]


def _stored_block_hashes(events):
    return [
        block["block_hash"] for event in _stored_events(events) for block in event["data"]["blocks"]
    ]


def _stored_block_hashes_by_layer_group(events):
    return {
        event["layer_group_id"]: [block["block_hash"] for block in event["data"]["blocks"]]
        for event in _stored_events(events)
    }


def _commit_and_close(manager, stream, tokens, *, input_tokens=None):
    kv_cache = manager.create_kv_cache(input_tokens=input_tokens)
    assert kv_cache.resume(stream)
    kv_cache.capacity = len(input_tokens or []) + len(tokens)
    kv_cache.commit(tokens)
    kv_cache.close()
    del kv_cache
    gc.collect()


def test_v2_kv_cache_event_manager_serialization():
    event_manager = KVCacheEventManager(max_kv_event_entries=4, window_size=128)
    event_manager.add_created_event([2, 3])
    event_manager.add_stored_event(
        parent_hash=None,
        blocks=[
            KVCacheStoredBlockData(
                block_hash="abcd",
                tokens=[UniqueToken(1), UniqueToken(2)],
                cache_level=0,
                priority=0,
            )
        ],
    )
    event_manager.add_removed_event("abcd")

    event_manager.flush_iteration_events()
    events = KVCacheEventSerializer.serialize(event_manager.get_latest_events())

    assert [event["event_id"] for event in events] == [0, 1, 2]
    assert [event["hash_algo"] for event in events] == ["v2_sha256"] * 3
    assert events[0]["window_size"] == 128
    assert events[0]["data"] == {
        "type": "created",
        "num_blocks_per_cache_level": [2, 3],
    }
    assert events[1]["data"]["type"] == "stored"
    assert events[1]["data"]["parent_hash"] is None
    assert events[1]["data"]["blocks"][0]["block_hash"] == "abcd"
    assert events[1]["data"]["blocks"][0]["tokens"][0] == {
        "type": "unique_token",
        "token_id": 1,
        "token_extra_id": 0,
    }
    assert events[1]["data"]["blocks"][0]["mm_keys"] == []
    assert events[2]["data"] == {
        "type": "removed",
        "block_hashes": ["abcd"],
    }
    assert event_manager.get_latest_events(0) == []


def test_v2_kv_cache_event_manager_default_get_latest_events_waits():
    event_manager = KVCacheEventManager(max_kv_event_entries=4, window_size=128)
    events = []

    def get_events():
        events.extend(event_manager.get_latest_events())

    thread = threading.Thread(target=get_events)
    thread.start()
    time.sleep(0.05)

    assert thread.is_alive()

    event_manager.add_created_event([1])
    event_manager.flush_iteration_events()
    thread.join(timeout=1)

    assert not thread.is_alive()
    assert len(events) == 1
    assert events[0].data.num_blocks_per_cache_level == [1]


def test_v2_kv_cache_event_manager_uses_layer_group_window_size():
    event_manager = KVCacheEventManager(
        max_kv_event_entries=4,
        window_size=4096,
        window_size_by_layer_group={
            0: 128,
            1: 4096,
        },
    )

    event_manager.add_created_event([1], layer_group_ids=[0, 1])
    event_manager.flush_iteration_events()
    events = event_manager.get_latest_events(0)

    assert [event.layer_group_id for event in events] == [0, 1]
    assert [event.window_size for event in events] == [128, 4096]


def test_truncate_sha256_hash_to_int64_uses_unsigned_value():
    assert truncate_sha256_hash_to_int64(b"\x80" + b"\x00" * 31) == 1 << 63
    assert truncate_sha256_hash_to_int64(b"\xff" * 32) == (1 << 64) - 1


def test_v2_kv_cache_event_diff_positional_order_matches_v1():
    diff = KVCacheEventDiff(3, 5)

    assert diff.old_value == 3
    assert diff.new_value == 5


def test_v2_kv_cache_event_manager_drops_oldest_events():
    event_manager = KVCacheEventManager(max_kv_event_entries=2, window_size=128)
    event_manager.add_created_event([1])
    event_manager.add_stored_event(
        parent_hash=None,
        blocks=[
            KVCacheStoredBlockData(
                block_hash="stored",
                tokens=[UniqueToken(1)],
                cache_level=0,
                priority=0,
            )
        ],
    )
    event_manager.add_removed_event("a")
    event_manager.add_removed_event("b")

    event_manager.flush_iteration_events()
    events = KVCacheEventSerializer.serialize(event_manager.get_latest_events())

    assert [event["event_id"] for event in events] == [1, 2]
    assert events[0]["data"]["blocks"][0]["block_hash"] == "stored"
    assert events[1]["data"]["block_hashes"] == ["a", "b"]


def test_v2_kv_cache_event_manager_accepts_removed_iterables():
    event_manager = KVCacheEventManager(max_kv_event_entries=2, window_size=128)
    event_manager.add_removed_event(["a", "b"])

    event_manager.flush_iteration_events()
    events = KVCacheEventSerializer.serialize(event_manager.get_latest_events())

    assert len(events) == 1
    assert events[0]["data"] == {
        "type": "removed",
        "block_hashes": ["a", "b"],
    }


def test_v2_kv_cache_event_manager_coalesces_contiguous_stored_events():
    event_manager = KVCacheEventManager(max_kv_event_entries=8, window_size=128)
    block0 = _FakeBlock(b"\xab\xcd", [1, 2], num_life_cycles=2)
    block1 = _FakeBlock(b"\xab\xce", [3, 4], num_life_cycles=2, prev=block0)

    event_manager.add_stored_block_event_from_block(block0)
    event_manager.add_stored_block_event_from_block(block1)

    events = _flush_serialized_events(event_manager)

    assert [event["data"]["type"] for event in events] == ["stored", "stored"]
    assert [event["layer_group_id"] for event in events] == [0, 1]
    assert [block["block_hash"] for block in events[0]["data"]["blocks"]] == [
        "abcd",
        "abce",
    ]
    assert [block["block_hash"] for block in events[1]["data"]["blocks"]] == [
        "abcd",
        "abce",
    ]
    assert events[0]["data"]["parent_hash"] is None
    assert events[1]["data"]["parent_hash"] is None


def test_v2_kv_cache_event_manager_serializes_layer_group_id():
    event_manager = KVCacheEventManager(max_kv_event_entries=4, window_size=128)
    event_manager.add_created_event([2, 3], [0, 1])

    events = _flush_serialized_events(event_manager)

    assert [event["layer_group_id"] for event in events] == [0, 1]
    assert [event["data"] for event in events] == [
        {
            "type": "created",
            "num_blocks_per_cache_level": [2, 3],
        },
        {
            "type": "created",
            "num_blocks_per_cache_level": [2, 3],
        },
    ]


def test_v2_kv_cache_event_manager_sha256_64_compatibility_mode():
    event_manager = KVCacheEventManager(
        max_kv_event_entries=8,
        window_size=128,
        hash_algo=KV_CACHE_HASH_ALGO_V2_SHA256_64,
    )
    block0 = _FakeBlock(bytes.fromhex("8000000000000001" + "00" * 24), [1, 2])
    block1 = _FakeBlock(bytes.fromhex("0102030405060708" + "00" * 24), [3, 4], prev=block0)

    event_manager.add_stored_block_event_from_block(block0)
    event_manager.add_stored_block_event_from_block(block1)
    event_manager.add_removed_event(block0.key)
    events = _flush_serialized_events(event_manager)

    expected_block0_hash = truncate_sha256_hash_to_int64(block0.key)
    expected_block1_hash = truncate_sha256_hash_to_int64(block1.key)
    assert [event["hash_algo"] for event in events] == [
        KV_CACHE_HASH_ALGO_V2_SHA256_64,
        KV_CACHE_HASH_ALGO_V2_SHA256_64,
    ]
    assert events[0]["data"]["type"] == "stored"
    assert events[0]["data"]["parent_hash"] is None
    assert [block["block_hash"] for block in events[0]["data"]["blocks"]] == [
        expected_block0_hash,
        expected_block1_hash,
    ]
    assert events[1]["data"] == {
        "type": "removed",
        "block_hashes": [expected_block0_hash],
    }
    assert all(isinstance(block_hash, int) for block_hash in events[1]["data"]["block_hashes"])


def test_v2_kv_cache_event_manager_v1_hash_algo_falls_back_to_sha256_64():
    event_manager = KVCacheEventManager(
        max_kv_event_entries=4,
        window_size=128,
        hash_algo=KV_CACHE_HASH_ALGO_V1,
    )
    block = _FakeBlock(bytes.fromhex("0102030405060708" + "00" * 24), [1, 2])

    event_manager.add_stored_block_event_from_block(block)
    events = _flush_serialized_events(event_manager)

    expected_block_hash = truncate_sha256_hash_to_int64(block.key)
    assert events[0]["hash_algo"] == KV_CACHE_HASH_ALGO_V2_SHA256_64
    assert events[0]["data"]["blocks"][0]["block_hash"] == expected_block_hash


def test_v2_kv_cache_event_manager_unknown_hash_algo_raises():
    with pytest.raises(ValueError, match="Unsupported V2 KV cache event hash algorithm"):
        KVCacheEventManager(
            max_kv_event_entries=4,
            window_size=128,
            hash_algo="unknown_hash_algo",
        )


def test_v2_kv_cache_event_manager_gathers_attention_dp_events_on_rank_zero():
    remote_manager = KVCacheEventManager(
        max_kv_event_entries=4,
        window_size=128,
        attention_dp_rank=1,
    )
    remote_manager.add_created_event([3])
    remote_manager.flush_iteration_events()
    remote_event_objects = remote_manager.get_latest_events()
    remote_events = KVCacheEventSerializer.serialize(remote_event_objects)
    assert remote_events[0]["attention_dp_rank"] == 1

    gathered_local_events = []

    def gather(local_events):
        gathered_local_events.extend(local_events)
        return [
            local_events,
            remote_event_objects,
        ]

    event_manager = KVCacheEventManager(
        max_kv_event_entries=4,
        window_size=128,
        attention_dp_rank=0,
        attention_dp_gather=gather,
    )
    event_manager.add_created_event([2])

    events = _flush_serialized_events(event_manager)

    assert len(gathered_local_events) == 1
    assert [event["attention_dp_rank"] for event in events] == [0, 1]
    assert [event["data"]["num_blocks_per_cache_level"] for event in events] == [
        [2],
        [3],
    ]


def test_v2_kv_cache_event_manager_attention_dp_nonzero_rank_sends_without_publishing():
    gathered_local_events = []

    def gather(local_events):
        gathered_local_events.extend(local_events)
        return [
            [],
            local_events,
        ]

    event_manager = KVCacheEventManager(
        max_kv_event_entries=1,
        window_size=128,
        attention_dp_rank=1,
        attention_dp_gather=gather,
    )
    event_manager.add_created_event([2])
    event_manager.add_created_event([3])

    events = _flush_serialized_events(event_manager)

    assert len(gathered_local_events) == 1
    assert gathered_local_events[0].attention_dp_rank == 1
    assert KVCacheEventSerializer.serialize(gathered_local_events)[0]["data"][
        "num_blocks_per_cache_level"
    ] == [3]
    assert events == []


def test_v2_kv_cache_event_manager_attention_dp_rank_zero_uses_dp_capacity():
    remote_manager = KVCacheEventManager(
        max_kv_event_entries=2,
        window_size=128,
        attention_dp_rank=1,
    )
    remote_manager.add_created_event([10])
    remote_manager.add_created_event([11])
    remote_manager.flush_iteration_events()
    remote_event_objects = remote_manager.get_latest_events()

    def gather(local_events):
        return [
            local_events,
            remote_event_objects,
        ]

    event_manager = KVCacheEventManager(
        max_kv_event_entries=1,
        window_size=128,
        attention_dp_rank=0,
        attention_dp_gather=gather,
    )
    event_manager.add_created_event([1])
    event_manager.add_created_event([2])

    events = _flush_serialized_events(event_manager)

    assert [event["attention_dp_rank"] for event in events] == [0, 1]
    assert [event["data"]["num_blocks_per_cache_level"] for event in events] == [
        [2],
        [11],
    ]


def test_v2_kv_cache_event_manager_serializes_updated_event():
    event_manager = KVCacheEventManager(max_kv_event_entries=2, window_size=128)
    event_manager.add_updated_event(
        "abcd",
        cache_level=KVCacheEventDiff(old_value=0, new_value=1),
        priority=KVCacheEventDiff(old_value=1, new_value=2),
    )

    event_manager.flush_iteration_events()
    events = KVCacheEventSerializer.serialize(event_manager.get_latest_events())

    assert len(events) == 1
    assert events[0]["hash_algo"] == "v2_sha256"
    assert events[0]["data"] == {
        "type": "updated",
        "block_hash": "abcd",
        "cache_level": {
            "type": "event_diff",
            "new_value": 1,
            "old_value": 0,
        },
        "priority": {
            "type": "event_diff",
            "new_value": 2,
            "old_value": 1,
        },
    }


def test_v2_kv_cache_event_manager_uses_stored_registry_for_removed_event():
    event_manager = KVCacheEventManager(max_kv_event_entries=8, window_size=128)
    block = _FakeBlock(b"\xab\xcd", [1, 2])

    event_manager.add_stored_block_event_from_block(block)
    block.storage = []
    event_manager.add_removed_event(block.key)
    event_manager.add_removed_event(block.key)

    events = _flush_serialized_events(event_manager)

    assert [event["data"]["type"] for event in events] == ["stored", "removed"]
    assert [event["layer_group_id"] for event in events] == [0, 0]
    assert events[0]["data"]["blocks"][0]["block_hash"] == "abcd"
    assert "layer_groups" not in events[0]["data"]["blocks"][0]
    assert events[1]["data"]["block_hashes"] == ["abcd"]
    assert "layer_groups" not in events[1]["data"]


def test_v2_kv_cache_event_manager_emits_partial_life_cycle_removed_events():
    event_manager = KVCacheEventManager(max_kv_event_entries=8, window_size=128)
    block = _FakeBlock(b"\xab\xcd", [1, 2], num_life_cycles=2)

    event_manager.add_stored_block_event_from_block(block)
    event_manager.add_removed_life_cycle_event(block.key, 0)

    events = _flush_serialized_events(event_manager)

    assert [event["data"]["type"] for event in events] == [
        "stored",
        "stored",
        "removed",
    ]
    assert [event["layer_group_id"] for event in events] == [0, 1, 0]
    assert events[0]["data"]["blocks"][0]["block_hash"] == "abcd"
    assert events[1]["data"]["blocks"][0]["block_hash"] == "abcd"
    assert "layer_groups" not in events[0]["data"]["blocks"][0]
    assert "layer_groups" not in events[1]["data"]["blocks"][0]
    assert events[2]["data"]["block_hashes"] == ["abcd"]
    assert "layer_groups" not in events[2]["data"]

    event_manager.add_removed_life_cycle_event(block.key, 1)
    event_manager.add_removed_life_cycle_event(block.key, 1)

    events = _flush_serialized_events(event_manager)

    assert [event["data"]["type"] for event in events] == ["removed"]
    assert events[0]["layer_group_id"] == 1
    assert events[0]["data"]["block_hashes"] == ["abcd"]
    assert "layer_groups" not in events[0]["data"]


def test_v2_kv_cache_event_manager_whole_block_removal_clears_life_cycle_state():
    event_manager = KVCacheEventManager(max_kv_event_entries=8, window_size=128)
    block = _FakeBlock(b"\xab\xce", [1, 2], num_life_cycles=2)

    event_manager.add_stored_block_event_from_block(block)
    event_manager.add_removed_life_cycle_event(block.key, 0)
    event_manager.add_removed_event(block.key)
    event_manager.add_removed_event(block.key)

    events = _flush_serialized_events(event_manager)

    assert [event["data"]["type"] for event in events] == [
        "stored",
        "stored",
        "removed",
        "removed",
    ]
    assert [event["layer_group_id"] for event in events] == [0, 1, 0, 1]
    assert events[2]["data"]["block_hashes"] == ["abce"]
    assert events[3]["data"]["block_hashes"] == ["abce"]


def test_v2_kv_cache_event_manager_readds_life_cycle_emits_stored_event():
    event_manager = KVCacheEventManager(max_kv_event_entries=8, window_size=128)
    block = _FakeBlock(b"\xab\xcf", [1, 2], num_life_cycles=2)

    event_manager.add_stored_block_event_from_block(block)
    event_types = [event["data"]["type"] for event in _flush_serialized_events(event_manager)]
    assert event_types == ["stored", "stored"]

    event_manager.add_removed_life_cycle_event(block.key, 0)
    event_manager.add_stored_life_cycle_event_from_block(block, 0)
    event_manager.add_removed_life_cycle_event(block.key, 1)
    events = _flush_serialized_events(event_manager)

    assert [event["data"]["type"] for event in events] == [
        "removed",
        "stored",
        "removed",
    ]
    assert [event["layer_group_id"] for event in events] == [0, 0, 1]
    assert events[0]["data"]["block_hashes"] == ["abcf"]
    assert events[1]["data"]["blocks"][0]["block_hash"] == "abcf"
    assert "layer_groups" not in events[1]["data"]["blocks"][0]
    assert events[2]["data"]["block_hashes"] == ["abcf"]

    event_manager.add_removed_life_cycle_event(block.key, 0)
    events = _flush_serialized_events(event_manager)

    assert [event["data"]["type"] for event in events] == ["removed"]
    assert events[0]["layer_group_id"] == 0
    assert events[0]["data"]["block_hashes"] == ["abcf"]
    assert "layer_groups" not in events[0]["data"]


def test_v2_kv_cache_event_manager_reemits_stored_after_all_life_cycles_were_removed():
    event_manager = KVCacheEventManager(max_kv_event_entries=8, window_size=128)
    block = _FakeBlock(b"\xab\xd0", [1, 2])

    event_manager.add_stored_block_event_from_block(block)
    event_manager.add_removed_life_cycle_event(block.key, 0)
    event_types = [event["data"]["type"] for event in _flush_serialized_events(event_manager)]
    assert event_types == ["stored", "removed"]

    event_manager.add_stored_life_cycle_event_from_block(block, 0)
    events = _flush_serialized_events(event_manager)

    assert [event["data"]["type"] for event in events] == ["stored"]
    assert events[0]["layer_group_id"] == 0
    assert events[0]["data"]["blocks"][0]["block_hash"] == "abd0"
    assert "layer_groups" not in events[0]["data"]["blocks"][0]


def test_v2_kv_cache_event_manager_flushes_removed_before_updated_event():
    event_manager = KVCacheEventManager(max_kv_event_entries=8, window_size=128)
    removed_block = _FakeBlock(b"\xab\xd1", [1, 2])
    updated_block = _FakeBlock(b"\xab\xd2", [3, 4])

    event_manager.add_stored_block_event_from_block(removed_block)
    event_manager.add_stored_block_event_from_block(updated_block)
    _flush_serialized_events(event_manager)

    event_manager.add_removed_event(removed_block.key)
    event_manager.add_updated_event(
        updated_block.key,
        cache_level=KVCacheEventDiff(old_value=0, new_value=1),
        layer_group_id=0,
    )
    events = _flush_serialized_events(event_manager)

    assert [event["data"]["type"] for event in events] == ["removed", "updated"]
    assert events[0]["data"]["block_hashes"] == ["abd1"]
    assert events[1]["data"]["block_hash"] == "abd2"


@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires CUDA")
def test_v2_stored_events_match_block_hash_chain():
    init_cuda_once()
    gc.collect()
    gc.disable()

    event_manager = KVCacheEventManager(max_kv_event_entries=16, window_size=128)
    manager = None
    try:
        tokens_per_block = 4
        manager = _create_test_manager(event_manager, tokens_per_block=tokens_per_block)
        stream_holder = CachedCudaStream()
        stream = cast(CudaStream, stream_holder.handle)
        tokens = _token_ids(0, 2 * tokens_per_block)

        _commit_and_close(manager, stream, tokens)
        events = _flush_serialized_events(event_manager)

        stored_events = _stored_events(events)
        assert len(stored_events) == 1

        root_key = RootBlock.make_key(None)
        block0_key = Block.make_key(root_key, tokens[:tokens_per_block])
        block1_key = Block.make_key(block0_key, tokens[tokens_per_block:])
        expected_hashes = [block0_key.hex(), block1_key.hex()]

        assert _stored_block_hashes(stored_events) == expected_hashes
        assert stored_events[0]["data"]["parent_hash"] is None
        assert [
            block["block_hash"] for block in stored_events[0]["data"]["blocks"]
        ] == expected_hashes
        assert [
            token["token_id"] for token in stored_events[0]["data"]["blocks"][0]["tokens"]
        ] == list(range(tokens_per_block))
        assert [
            token["token_id"] for token in stored_events[0]["data"]["blocks"][1]["tokens"]
        ] == list(range(tokens_per_block, 2 * tokens_per_block))
    finally:
        gc.enable()
        if manager is not None:
            manager.shutdown()


@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires CUDA")
def test_v2_reused_prefix_does_not_emit_duplicate_stored_events():
    init_cuda_once()
    gc.collect()
    gc.disable()

    event_manager = KVCacheEventManager(max_kv_event_entries=16, window_size=128)
    manager = None
    try:
        tokens_per_block = 4
        manager = _create_test_manager(event_manager, tokens_per_block=tokens_per_block)
        stream_holder = CachedCudaStream()
        stream = cast(CudaStream, stream_holder.handle)
        prefix_tokens = _token_ids(0, 2 * tokens_per_block)
        new_tokens = _token_ids(2 * tokens_per_block, 3 * tokens_per_block)

        _commit_and_close(manager, stream, prefix_tokens)
        first_events = _flush_serialized_events(event_manager)
        prefix_hashes = _stored_block_hashes(first_events)
        assert len(prefix_hashes) == 2

        _commit_and_close(
            manager,
            stream,
            new_tokens,
            input_tokens=prefix_tokens,
        )
        reuse_events = _flush_serialized_events(event_manager)
        reused_hashes = _stored_block_hashes(reuse_events)

        root_key = RootBlock.make_key(None)
        block0_key = Block.make_key(root_key, prefix_tokens[:tokens_per_block])
        block1_key = Block.make_key(block0_key, prefix_tokens[tokens_per_block:])
        block2_key = Block.make_key(block1_key, new_tokens)

        assert prefix_hashes == [block0_key.hex(), block1_key.hex()]
        assert reused_hashes == [block2_key.hex()]
        assert not (set(prefix_hashes) & set(reused_hashes))
    finally:
        gc.enable()
        if manager is not None:
            manager.shutdown()


@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires CUDA")
def test_v2_removed_events_match_stored_hashes():
    init_cuda_once()
    gc.collect()
    gc.disable()

    event_manager = KVCacheEventManager(max_kv_event_entries=16, window_size=128)
    manager = None
    try:
        tokens_per_block = 4
        manager = _create_test_manager(event_manager, tokens_per_block=tokens_per_block)
        stream_holder = CachedCudaStream()
        stream = cast(CudaStream, stream_holder.handle)
        tokens = _token_ids(0, 2 * tokens_per_block)

        _commit_and_close(manager, stream, tokens)
        stored_events = _flush_serialized_events(event_manager)
        stored_hashes = _stored_block_hashes(stored_events)
        assert len(stored_hashes) == 2

        manager.clear_reusable_blocks()
        removal_events = _flush_serialized_events(event_manager)
        removed_hashes = [
            block_hash
            for event in removal_events
            if event["data"]["type"] == "removed"
            for block_hash in event["data"]["block_hashes"]
        ]

        assert set(removed_hashes) == set(stored_hashes)
        assert len(removed_hashes) == len(stored_hashes)
    finally:
        gc.enable()
        if manager is not None:
            manager.shutdown()


@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires CUDA")
def test_v2_removed_event_emitted_when_last_level_page_is_dropped():
    init_cuda_once()
    gc.collect()
    gc.disable()

    event_manager = KVCacheEventManager(max_kv_event_entries=16, window_size=4096)
    manager = None
    try:
        tokens_per_block = 8
        manager = _create_test_manager(
            event_manager,
            tokens_per_block=tokens_per_block,
            gpu_quota=8 << 20,
            window_size=4096,
            kv_buf_size=1 << 20,
        )
        stream_holder = CachedCudaStream()
        stream = cast(CudaStream, stream_holder.handle)
        kv_cache = manager.create_kv_cache()
        assert kv_cache.resume(stream)
        kv_cache.capacity = tokens_per_block * 2
        kv_cache.commit(_token_ids(0, tokens_per_block * 2))

        stored_events = _flush_serialized_events(event_manager)
        stored_hashes_by_layer_group = _stored_block_hashes_by_layer_group(stored_events)
        assert set(stored_hashes_by_layer_group) == {0, 1}
        assert stored_hashes_by_layer_group[0] == stored_hashes_by_layer_group[1]

        kv_cache.close()
        del kv_cache
        gc.collect()

        assert manager.resize(CacheLevel(0), 4 << 20)
        removal_events = _flush_serialized_events(event_manager)
        removed_hashes_by_layer_group = {
            event["layer_group_id"]: event["data"]["block_hashes"]
            for event in removal_events
            if event["data"]["type"] == "removed"
        }

        assert set(removed_hashes_by_layer_group) == {0, 1}
        assert removed_hashes_by_layer_group[0] == removed_hashes_by_layer_group[1]
        for layer_group_id, removed_hashes in removed_hashes_by_layer_group.items():
            assert len(removed_hashes) == 1
            assert removed_hashes[0] in stored_hashes_by_layer_group[layer_group_id]
    finally:
        gc.enable()
        if manager is not None:
            manager.shutdown()


@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires CUDA")
def test_v2_kv_cache_event_manager_emits_updated_on_level_migration():
    init_cuda_once()
    gc.collect()
    gc.disable()

    event_manager = KVCacheEventManager(max_kv_event_entries=16, window_size=4096)
    manager = None
    try:
        tokens_per_block = 8
        manager = _create_test_manager(
            event_manager,
            tokens_per_block=tokens_per_block,
            gpu_quota=8 << 20,
            host_quota=8 << 20,
            window_size=4096,
            kv_buf_size=1 << 20,
        )
        stream_holder = CachedCudaStream()
        stream = cast(CudaStream, stream_holder.handle)
        kv_cache = manager.create_kv_cache()
        assert kv_cache.resume(stream)
        kv_cache.capacity = tokens_per_block * 2
        kv_cache.commit([TokenId(token_id) for token_id in range(tokens_per_block * 2)])

        stored_events = _flush_serialized_events(event_manager)
        stored_hashes_by_layer_group = _stored_block_hashes_by_layer_group(stored_events)
        assert set(stored_hashes_by_layer_group) == {0, 1}
        assert stored_hashes_by_layer_group[0] == stored_hashes_by_layer_group[1]

        kv_cache.close()
        del kv_cache
        gc.collect()

        assert manager.resize(CacheLevel(0), 4 << 20)

        event_manager.flush_iteration_events()
        events = KVCacheEventSerializer.serialize(event_manager.get_latest_events())
        updated_events = [event for event in events if event["data"]["type"] == "updated"]

        assert len(updated_events) == 2
        assert {event["layer_group_id"] for event in updated_events} == {0, 1}
        assert len({event["data"]["block_hash"] for event in updated_events}) == 1
        assert updated_events[0]["data"]["block_hash"] in stored_hashes_by_layer_group[0]
        for event in updated_events:
            assert event["data"]["cache_level"] == {
                "type": "event_diff",
                "old_value": 0,
                "new_value": 1,
            }
            assert event["data"]["priority"] is None
    finally:
        gc.enable()
        if manager is not None:
            manager.shutdown()
