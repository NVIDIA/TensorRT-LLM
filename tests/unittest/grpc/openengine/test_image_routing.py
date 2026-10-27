# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Image routing hints preserve engine identities and survive the event wire."""

from types import SimpleNamespace

import msgspec
import pytest
from PIL import Image

from tensorrt_llm._torch.pyexecutor import kv_cache_events as events
from tensorrt_llm.inputs.multimodal import apply_mm_hashes
from tensorrt_llm.llmapi.llm_args import KVEventsConfig

pytestmark = pytest.mark.cpu_only


def test_streaming_image_hints_are_bounded_and_content_sensitive(monkeypatch):
    uuid = "dynamo-image-v1:0123456789abcdef"
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
        with pytest.raises(ValueError, match="Conflicting UUIDs"):
            manager.remember_multimodal_uuids([(bytes.fromhex(hashes[1]), "other")])
    finally:
        source._event_sink = native_sink
        manager.shutdown()
