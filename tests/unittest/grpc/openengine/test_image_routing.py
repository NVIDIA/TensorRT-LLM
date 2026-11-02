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


@pytest.mark.parametrize("uuid", ["dynamo-image-v1:0123456789abcdef", "caller-owned"])
def test_streaming_native_image_uuids_preserve_content_sensitive_keys(monkeypatch, uuid):
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

        class Stored:
            block_hashes = [123]
            parent_block_hash = None
            token_ids = [1, hashes[0], 1001, 7]
            lora_id = None
            mm_keys = [[(bytes.fromhex(hashes[0]), 0, uuid, True)]]

        monkeypatch.setattr(events.kv_cache_manager_v2_runtime, "StreamingBlockStoredData", Stored)
        source._event_sink = SimpleNamespace(drain_iteration_events=lambda: [Stored()])
        wire = msgspec.msgpack.decode(msgspec.msgpack.encode(source.drain_events()[0]))
        metadata = wire["trtllm_image_routing"]
        assert metadata["version"] == 1 and metadata["token_id_offset"] == 1000
        assert set(metadata) == {"version", "token_id_offset"}
        assert wire["mm_keys"][0][0]["uuid"] == uuid
        assert wire["mm_keys"][0][0]["hash"] == hashes[0]
        encoded = msgspec.msgpack.encode(source.drain_events()[0])
        assert encoded.count(hashes[0].encode()) == 2  # Token stream and mm_keys.
        assert encoded.count(uuid.encode()) == 1
        assert wire["token_ids"] == Stored.token_ids

        Stored.token_ids = [1, 2, 3, 4]
        Stored.mm_keys = [[]]
        text_wire = msgspec.msgpack.decode(msgspec.msgpack.encode(source.drain_events()[0]))
        assert "trtllm_image_routing" not in text_wire

    finally:
        source._event_sink = native_sink
        manager.shutdown()


@pytest.mark.parametrize("uuid", [None, "", "image-a"])
def test_native_integer_image_continuations_keep_routing_metadata(monkeypatch, uuid):
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

        class Stored:
            block_hashes = [123]
            parent_block_hash = 122
            token_ids = [1003, 1004, 1005, 7]
            lora_id = None
            mm_keys = [[(digest, 3, uuid, True)]]

        monkeypatch.setattr(events.kv_cache_manager_v2_runtime, "StreamingBlockStoredData", Stored)
        source._event_sink = SimpleNamespace(drain_iteration_events=lambda: [Stored()])
        wire = msgspec.msgpack.decode(msgspec.msgpack.encode(source.drain_events()[0]))
        metadata = wire["trtllm_image_routing"]
        assert metadata["version"] == 1 and metadata["token_id_offset"] == 1000
        assert set(metadata) == {"version", "token_id_offset"}
        key = wire["mm_keys"][0][0]
        assert key["hash"] == digest.hex() and key["start_offset"] == 3
        assert key.get("uuid") == uuid
        assert wire["token_ids"] == Stored.token_ids
    finally:
        source._event_sink = native_sink
        manager.shutdown()
