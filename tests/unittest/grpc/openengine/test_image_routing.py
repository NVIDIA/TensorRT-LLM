# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Image routing hints preserve engine identities and survive the event wire."""

from types import SimpleNamespace

import msgspec
import pytest
from PIL import Image

from tensorrt_llm._torch.pyexecutor import kv_cache_events as events
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm.inputs.multimodal import apply_mm_hashes
from tensorrt_llm.llmapi.llm_args import KVEventsConfig

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("uuid", ["dynamo-image-v1:0123456789abcdef", "caller-owned"])
def test_streaming_image_hints_are_bounded_and_content_sensitive(monkeypatch, uuid):
    red = Image.new("RGB", (32, 32), "red")
    blue = Image.new("RGB", (32, 32), "blue")
    hashes = [
        apply_mm_hashes({"image": [image]}, {"image": [uuid]}).hashes["image"][0]
        for image in (red, blue)
    ]
    assert hashes[0] != hashes[1], "routing hints must not replace content-sensitive cache keys"

    manager = events.StreamingKVCacheEventManager(
        KVEventsConfig(enable_kv_cache_events=True, publisher="null"),
        data_parallel_rank=0,
        block_size=4,
        max_window_size=128,
        max_entries=1,
        mm_token_id_offset=1000,
    )
    source = manager._event_source
    native_sink = source._event_sink
    try:
        manager.remember_multimodal_uuids([(bytes.fromhex(hashes[0]), uuid)])

        class Stored:
            block_hashes = [123]
            parent_block_hash = None
            token_ids = [1, hashes[0], 1001, 7]
            lora_id = None
            mm_keys = [[(bytes.fromhex(hashes[0]), 0)]]

        monkeypatch.setattr(events.kv_cache_manager_v2_runtime, "StreamingBlockStoredData", Stored)
        source._event_sink = SimpleNamespace(drain_iteration_events=lambda: [Stored()])
        wire = msgspec.msgpack.decode(msgspec.msgpack.encode(source.drain_events()[0]))
        metadata = wire["trtllm_image_routing"]
        assert metadata["version"] == 1 and metadata["token_id_offset"] == 1000
        assert metadata["keys"][0][0]["uuid"] == uuid
        assert metadata["keys"][0][0]["hash"] == hashes[0]
        assert wire["token_ids"] == Stored.token_ids

        Stored.token_ids = [1, 2, 3, 4]
        Stored.mm_keys = [[]]
        text_wire = msgspec.msgpack.decode(msgspec.msgpack.encode(source.drain_events()[0]))
        assert "trtllm_image_routing" not in text_wire

        manager.remember_multimodal_uuids([(bytes.fromhex(hashes[1]), uuid)])
        assert len(source._uuids) == 1
        assert source._multimodal_key((bytes.fromhex(hashes[0]), 0)).uuid is None
        assert source._multimodal_key((bytes.fromhex(hashes[1]), 0)).uuid == uuid
        manager.remember_multimodal_uuids([(bytes.fromhex(hashes[0]), "x" * 257)])
        assert len(source._uuids) == 1
        manager.remember_multimodal_uuids([(bytes.fromhex(hashes[1]), "other")])
        assert source._multimodal_key((bytes.fromhex(hashes[1]), 0)).uuid is None
    finally:
        source._event_sink = native_sink
        manager.shutdown()


@pytest.mark.parametrize(
    "hashes,uuids,error",
    [
        ([tuple(range(8)), tuple(range(8, 16))], ["image-a"], "shorter"),
        ([tuple(range(8))], ["image-a", "image-b"], "longer"),
        ([tuple(range(8)), (1, 2, 3)], ["image-a", "image-b"], "Expected 8 int32"),
    ],
)
def test_invalid_request_uuid_pairs_leave_routing_hints_unchanged(hashes, uuids, error):
    event_manager = events.StreamingKVCacheEventManager(
        KVEventsConfig(enable_kv_cache_events=True, publisher="null"),
        data_parallel_rank=0,
        block_size=4,
        max_window_size=128,
        max_entries=1,
        mm_token_id_offset=1000,
    )
    try:
        event_manager.remember_multimodal_uuids([(b"x" * 32, "existing")])
        before = event_manager._event_source._uuids.copy()
        manager = KVCacheManagerV2.__new__(KVCacheManagerV2)
        manager.vocab_size = 1000
        manager.event_manager = event_manager
        request = SimpleNamespace(
            multimodal_hashes=hashes,
            multimodal_uuids=uuids,
            multimodal_positions=[0] * len(hashes),
            multimodal_lengths=[1] * len(hashes),
        )
        with pytest.raises(ValueError, match=error):
            manager._augment_tokens_for_block_reuse([1, 2], request)
        assert event_manager._event_source._uuids == before
    finally:
        event_manager.shutdown()


def test_request_uuid_pairs_preserve_partial_uuid_slots():
    hashes = [tuple(range(8)), tuple(range(8, 16))]
    event_manager = events.StreamingKVCacheEventManager(
        KVEventsConfig(enable_kv_cache_events=True, publisher="null"),
        data_parallel_rank=0,
        block_size=4,
        max_window_size=128,
        max_entries=1,
        mm_token_id_offset=1000,
    )
    try:
        manager = KVCacheManagerV2.__new__(KVCacheManagerV2)
        manager.vocab_size = 1000
        manager.event_manager = event_manager
        request = SimpleNamespace(
            multimodal_hashes=hashes,
            multimodal_uuids=["image-a", None],
            multimodal_positions=[0, 1],
            multimodal_lengths=[1, 1],
            multimodal_item_run_cu_offsets=None,
            multimodal_run_positions=None,
            multimodal_run_lengths=None,
        )
        manager._augment_tokens_for_block_reuse([1, 2], request)
        digest = b"".join(value.to_bytes(4, "big", signed=True) for value in hashes[0])
        assert dict(event_manager._event_source._uuids) == {digest: "image-a"}
    finally:
        event_manager.shutdown()


def test_conflicting_request_hints_preserve_inference_cache_keys():
    manager = events.StreamingKVCacheEventManager(
        KVEventsConfig(enable_kv_cache_events=True, publisher="null"),
        data_parallel_rank=0,
        block_size=4,
        max_window_size=128,
        max_entries=1,
        mm_token_id_offset=1000,
    )
    cache = KVCacheManagerV2.__new__(KVCacheManagerV2)
    cache.vocab_size = 1000
    cache.event_manager = manager
    hashes = [tuple(range(8))]
    digest = b"".join(value.to_bytes(4, "big", signed=True) for value in hashes[0])
    results = []
    try:
        for uuid in ["image-a", "image-b", "image-a"]:
            request = SimpleNamespace(
                multimodal_hashes=hashes,
                multimodal_uuids=[uuid],
                multimodal_positions=[0],
                multimodal_lengths=[1],
                multimodal_item_run_cu_offsets=None,
                multimodal_run_positions=None,
                multimodal_run_lengths=None,
            )
            results.append(cache._augment_tokens_for_block_reuse([1, 2], request))
            hint = manager._event_source._multimodal_key((digest, 0)).uuid
            assert hint == ("image-a" if len(results) == 1 else None)
        assert results[0] == results[1] == results[2]
    finally:
        manager.shutdown()


@pytest.mark.parametrize("hint_state", ["absent", "evicted", "ambiguous"])
def test_unmapped_integer_image_continuations_keep_routing_metadata(monkeypatch, hint_state):
    manager = events.StreamingKVCacheEventManager(
        KVEventsConfig(enable_kv_cache_events=True, publisher="null"),
        data_parallel_rank=0,
        block_size=4,
        max_window_size=128,
        max_entries=1,
        mm_token_id_offset=1000,
    )
    source = manager._event_source
    native_sink = source._event_sink
    digest = bytes(range(32))
    try:
        if hint_state != "absent":
            manager.remember_multimodal_uuids([(digest, "image-a")])
            if hint_state == "evicted":
                manager.remember_multimodal_uuids([(bytes(reversed(digest)), "image-b")])
            else:
                manager.remember_multimodal_uuids([(digest, "image-b")])

        class Stored:
            block_hashes = [123]
            parent_block_hash = 122
            token_ids = [1003, 1004, 1005, 7]
            lora_id = None
            mm_keys = [[(digest, 3)]]

        monkeypatch.setattr(events.kv_cache_manager_v2_runtime, "StreamingBlockStoredData", Stored)
        source._event_sink = SimpleNamespace(drain_iteration_events=lambda: [Stored()])
        wire = msgspec.msgpack.decode(msgspec.msgpack.encode(source.drain_events()[0]))
        metadata = wire["trtllm_image_routing"]
        assert metadata["version"] == 1 and metadata["token_id_offset"] == 1000
        key = metadata["keys"][0][0]
        assert key["hash"] == digest.hex() and key["start_offset"] == 3
        assert key.get("uuid") is None
        assert wire["token_ids"] == Stored.token_ids
    finally:
        source._event_sink = native_sink
        manager.shutdown()
