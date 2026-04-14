# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import time

import numpy as np
import pytest
import torch
from PIL import Image
from utils.util import skip_single_gpu

import tensorrt_llm
from tensorrt_llm import LLM
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._utils import KVCacheEventSerializer
from tensorrt_llm.bindings.internal.testing import \
    simulate_prefill_completion_only_use_for_testing
from tensorrt_llm.inputs.multimodal import (MultimodalInput,
                                            _find_mm_token_runs_from_mask,
                                            apply_mm_hashes)
from tensorrt_llm.inputs.multimodal_data import (AudioData, VideoData,
                                                 serialize_item)
from tensorrt_llm.llmapi import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.sampling_params import SamplingParams
from tensorrt_llm.scheduling_params import SchedulingParams

from .test_llm import get_model_path

default_model_name = "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"
llama_model_path = get_model_path(default_model_name)
global_kvcache_config = KvCacheConfig(free_gpu_memory_fraction=0.4,
                                      event_buffer_max_size=1024,
                                      enable_block_reuse=True,
                                      max_tokens=256)


def create_kv_cache_manager():
    num_layers = 2
    num_kv_heads = 2
    head_dim = 128
    tokens_per_block = 64
    max_seq_len = 1024
    max_batch_size = 1
    mapping = Mapping()
    return KVCacheManager(
        kv_cache_config=global_kvcache_config,
        kv_cache_type=tensorrt_llm.bindings.internal.batch_manager.CacheType.
        SELF,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        mapping=mapping,
    )


def create_llm(tensor_parallel_size=1):
    return LLM(model=llama_model_path,
               tensor_parallel_size=tensor_parallel_size,
               kv_cache_config=global_kvcache_config,
               enable_autotuner=False)


def create_llm_request(id, input_tokens, new_tokens=1):
    sampling_params = SamplingParams()
    req = LlmRequest(request_id=id,
                     max_new_tokens=new_tokens,
                     input_tokens=input_tokens,
                     sampling_config=tensorrt_llm.bindings.SamplingConfig(
                         sampling_params._get_sampling_config()),
                     is_streaming=False)
    return req


def flush_events(kv_cache_manager):
    kv_cache_manager.flush_iteration_events()
    time.sleep(0.001)


def test_kv_cache_event_data_serialization():
    kv_cache_manager = create_kv_cache_manager()
    flush_events(kv_cache_manager)
    events = kv_cache_manager.get_latest_events(10)
    serialized_event = KVCacheEventSerializer.serialize(events)
    assert len(serialized_event) == 1 and serialized_event[0][
        "event_id"] == 0 and serialized_event[0]["window_size"] == 256
    assert serialized_event[0]["data"]["type"] == "created"
    assert len(serialized_event[0]["data"]["num_blocks_per_cache_level"]) == 2

    req = create_llm_request(0, [1, 2, 3, 4, 5])
    kv_cache_manager.impl.add_sequence_batch(
        [(req.py_request_id, req.prompt_len, 1)], [req])
    simulate_prefill_completion_only_use_for_testing(req)
    kv_cache_manager.free_resources(req)

    flush_events(kv_cache_manager)
    events = kv_cache_manager.get_latest_events(10)
    serialized_event = KVCacheEventSerializer.serialize(events)

    assert serialized_event[0]["data"]["type"] == "stored"
    assert serialized_event[0]["data"]["parent_hash"] is None
    assert len(serialized_event[0]["data"]["blocks"]) == 1
    assert len(serialized_event[0]["data"]["blocks"][0]["tokens"]) == 4
    # Verify mm_keys field exists (empty for text-only requests)
    assert "mm_keys" in serialized_event[0]["data"]["blocks"][0]
    assert serialized_event[0]["data"]["blocks"][0]["mm_keys"] == []
    # Verify cache_salt field exists (None for unsalted requests)
    assert "cache_salt" in serialized_event[0]["data"]["blocks"][0]
    assert serialized_event[0]["data"]["blocks"][0]["cache_salt"] is None

    req2 = create_llm_request(1, [1, 2, 3, 4, 5])
    kv_cache_manager.impl.add_sequence_batch(
        [(req2.py_request_id, req2.prompt_len, 1)], [req2])
    simulate_prefill_completion_only_use_for_testing(req2)
    kv_cache_manager.free_resources(req2)

    flush_events(kv_cache_manager)
    events = kv_cache_manager.get_latest_events(10)
    serialized_event = KVCacheEventSerializer.serialize(events)


def test_mm_keys_serialization():
    """Test serialization of multimodal keys (mm_keys) in KV cache events."""
    # Test _mm_key_to_json with a mock mm_key tuple (bytes, int, uuid)
    # MmKey from C++ is converted to (bytes, int, optional<str>) tuple by pybind11
    mock_hash = b'\x01\x02\x03\x04\x05\x06\x07\x08' + b'\x00' * 24  # 32 bytes
    mock_offset = 42
    # New format: (hash, offset, uuid) - uuid is None for content-hashed items
    mock_mm_key = (mock_hash, mock_offset, None)

    result = KVCacheEventSerializer._mm_key_to_json(mock_mm_key)

    assert result["type"] == "mm_key"
    assert result["start_offset"] == 42
    # Hash should be converted to hex string when UUID is None
    assert result["hash"] == "0102030405060708" + "00" * 24
    assert len(result["hash"]) == 64  # 32 bytes = 64 hex chars

    # Test with different hash values
    mock_hash2 = bytes(range(32))  # 0x00 to 0x1f
    mock_mm_key2 = (mock_hash2, 100, None)
    result2 = KVCacheEventSerializer._mm_key_to_json(mock_mm_key2)

    assert result2["type"] == "mm_key"
    assert result2["start_offset"] == 100
    expected_hash = ''.join(f'{i:02x}' for i in range(32))
    assert result2["hash"] == expected_hash


def test_mm_keys_deserialization():
    """Test deserialization of mm_keys JSON back to 32-byte hash."""
    # Test case 1: Simple hash pattern (no UUID)
    mock_hash = b'\x01\x02\x03\x04\x05\x06\x07\x08' + b'\x00' * 24  # 32 bytes
    mock_offset = 42
    mock_mm_key = (mock_hash, mock_offset, None)  # New format with None UUID

    # Serialize to JSON
    json_result = KVCacheEventSerializer._mm_key_to_json(mock_mm_key)

    # Deserialize hex string back to bytes
    recovered_hash = bytes.fromhex(json_result["hash"])

    # Verify the recovered hash matches the original
    assert recovered_hash == mock_hash
    assert len(recovered_hash) == 32
    assert json_result["start_offset"] == mock_offset

    # Test case 2: Sequential bytes 0x00 to 0x1f
    mock_hash2 = bytes(range(32))
    mock_offset2 = 100
    mock_mm_key2 = (mock_hash2, mock_offset2, None)

    json_result2 = KVCacheEventSerializer._mm_key_to_json(mock_mm_key2)
    recovered_hash2 = bytes.fromhex(json_result2["hash"])

    assert recovered_hash2 == mock_hash2
    assert len(recovered_hash2) == 32
    assert json_result2["start_offset"] == mock_offset2

    # Test case 3: All 0xFF bytes
    mock_hash3 = b'\xff' * 32
    mock_offset3 = 255
    mock_mm_key3 = (mock_hash3, mock_offset3, None)

    json_result3 = KVCacheEventSerializer._mm_key_to_json(mock_mm_key3)
    recovered_hash3 = bytes.fromhex(json_result3["hash"])

    assert recovered_hash3 == mock_hash3
    assert len(recovered_hash3) == 32
    assert json_result3["hash"] == "ff" * 32

    # Test case 4: Random-like pattern
    mock_hash4 = bytes([0xde, 0xad, 0xbe, 0xef] + [0xca, 0xfe] * 14)
    mock_offset4 = 1024
    mock_mm_key4 = (mock_hash4, mock_offset4, None)

    json_result4 = KVCacheEventSerializer._mm_key_to_json(mock_mm_key4)
    recovered_hash4 = bytes.fromhex(json_result4["hash"])

    assert recovered_hash4 == mock_hash4
    assert len(recovered_hash4) == 32


def test_mm_key_with_uuid():
    """Test _mm_key_to_json returns UUID when provided in the tuple."""
    # Create a mock mm_key with new format (hash, offset, uuid)
    mock_hash = b'\x01\x02\x03\x04\x05\x06\x07\x08' + b'\x00' * 24  # 32 bytes
    mock_offset = 42
    expected_hash = "0102030405060708" + "00" * 24

    # Test 1: Without UUID (None), should return hex hash
    mock_mm_key_no_uuid = (mock_hash, mock_offset, None)
    result_no_uuid = KVCacheEventSerializer._mm_key_to_json(mock_mm_key_no_uuid)
    assert result_no_uuid["hash"] == expected_hash
    assert result_no_uuid["start_offset"] == 42

    # Test 2: With UUID in tuple, should return UUID directly
    test_uuid = "my-custom-image-uuid"
    mock_mm_key_with_uuid = (mock_hash, mock_offset, test_uuid)
    result_with_uuid = KVCacheEventSerializer._mm_key_to_json(
        mock_mm_key_with_uuid)
    assert result_with_uuid["hash"] == test_uuid
    assert result_with_uuid["start_offset"] == 42

    # Test 3: Backward compatibility - old format (2 elements) should return hex hash
    mock_mm_key_old_format = (mock_hash, mock_offset)
    result_old_format = KVCacheEventSerializer._mm_key_to_json(
        mock_mm_key_old_format)
    assert result_old_format["hash"] == expected_hash


def test_apply_mm_hashes_with_uuids():
    """Test apply_mm_hashes with user-provided UUIDs."""
    import torch

    from tensorrt_llm.inputs.multimodal import apply_mm_hashes

    # Create mock multimodal data - use fixed seed for reproducibility
    torch.manual_seed(42)
    mock_image1 = torch.randn(3, 224, 224)
    mock_image2 = torch.randn(3, 224, 224)
    mm_data = {"image": [mock_image1, mock_image2]}

    # Test without UUIDs - should use content-only hashing
    hashes_no_uuid, uuids_no_uuid = apply_mm_hashes(mm_data)
    assert len(hashes_no_uuid["image"]) == 2
    assert all(len(h) == 64 for h in hashes_no_uuid["image"])
    assert uuids_no_uuid is None

    # Test with partial UUIDs (first has UUID, second uses content-only hash)
    mm_uuids = {"image": ["sku-1234-a", None]}
    hashes_partial, uuids_partial = apply_mm_hashes(mm_data, mm_uuids)

    assert len(hashes_partial["image"]) == 2
    # First hash should be combined UUID+content (different from content-only)
    assert len(hashes_partial["image"][0]) == 64
    assert hashes_partial["image"][0] != hashes_no_uuid["image"][
        0]  # UUID changes hash
    # Second hash should be content-only (same as without UUID)
    assert hashes_partial["image"][1] == hashes_no_uuid["image"][1]
    # UUIDs list should have the UUID and None
    assert uuids_partial == ["sku-1234-a", None]

    # Test with all UUIDs
    mm_uuids_all = {"image": ["sku-1234-a", "sku-1234-b"]}
    hashes_all, uuids_all = apply_mm_hashes(mm_data, mm_uuids_all)

    assert len(hashes_all["image"]) == 2
    assert all(len(h) == 64 for h in hashes_all["image"])
    # Both hashes should differ from content-only hashes
    assert hashes_all["image"][0] != hashes_no_uuid["image"][0]
    assert hashes_all["image"][1] != hashes_no_uuid["image"][1]
    # Different UUIDs with different content should produce different hashes
    assert hashes_all["image"][0] != hashes_all["image"][1]
    assert uuids_all == ["sku-1234-a", "sku-1234-b"]


def test_apply_mm_hashes_uuid_content_combined():
    """Test that UUID + content hashing ensures cache correctness.

    This test verifies the key properties of combined UUID+content hashing:
    1. Same UUID + same content = same hash (cache hit expected)
    2. Same UUID + different content = different hash (no incorrect cache hit)
    3. Different UUID + same content = different hash (user isolation)
    """
    import torch

    from tensorrt_llm.inputs.multimodal import apply_mm_hashes

    # Create identical images
    torch.manual_seed(42)
    image_a = torch.randn(3, 224, 224)
    image_a_copy = image_a.clone()  # Identical content

    # Create a different image
    torch.manual_seed(123)
    image_b = torch.randn(3, 224, 224)

    # Property 1: Same UUID + same content = same hash
    mm_data_a = {"image": [image_a]}
    mm_data_a_copy = {"image": [image_a_copy]}
    mm_uuids = {"image": ["user-123-img"]}

    hashes_a, _ = apply_mm_hashes(mm_data_a, mm_uuids)
    hashes_a_copy, _ = apply_mm_hashes(mm_data_a_copy, mm_uuids)
    assert hashes_a["image"][0] == hashes_a_copy["image"][0], \
        "Same UUID + same content should produce identical hashes"

    # Property 2: Same UUID + different content = different hash
    mm_data_b = {"image": [image_b]}
    hashes_b, _ = apply_mm_hashes(mm_data_b, mm_uuids)
    assert hashes_a["image"][0] != hashes_b["image"][0], \
        "Same UUID + different content must produce different hashes"

    # Property 3: Different UUID + same content = different hash (user isolation)
    mm_uuids_user2 = {"image": ["user-456-img"]}
    hashes_user2, _ = apply_mm_hashes(mm_data_a, mm_uuids_user2)
    assert hashes_a["image"][0] != hashes_user2["image"][0], \
        "Different UUID + same content should produce different hashes"


def _make_video(frames, audio_samples=None, sample_rate=16000, metadata=None):
    """Module-level helper shared by the mm-hash tests."""
    audio = None
    if audio_samples is not None:
        audio = AudioData(samples=audio_samples, sample_rate=sample_rate)
    return VideoData(frames=frames,
                     metadata={} if metadata is None else metadata,
                     audio=audio)


def test_apply_mm_hashes_video_audio_affects_hash():
    """VideoData hashes include extracted audio when it affects model inputs."""
    frames = [
        Image.new("RGB", (2, 2), (10, 20, 30)),
        Image.new("RGB", (2, 2), (40, 50, 60)),
    ]
    audio_a = np.array([0.0, 0.25, -0.5, 1.0], dtype=np.float32)
    audio_a_copy = audio_a.copy()
    audio_b = np.array([0.0, 0.25, -0.5, -1.0], dtype=np.float32)

    hashes_a, _ = apply_mm_hashes({"video": [_make_video(frames, audio_a)]})
    hashes_a_copy, _ = apply_mm_hashes(
        {"video": [_make_video(frames, audio_a_copy)]})
    hashes_b, _ = apply_mm_hashes({"video": [_make_video(frames, audio_b)]})
    hashes_a_different_rate, _ = apply_mm_hashes(
        {"video": [_make_video(frames, audio_a, sample_rate=8000)]})
    hashes_no_audio, _ = apply_mm_hashes({"video": [_make_video(frames)]})
    hashes_frame_list, _ = apply_mm_hashes({"video": [frames]})

    assert hashes_a["video"][0] == hashes_a_copy["video"][0]
    assert hashes_a["video"][0] != hashes_b["video"][0]
    assert hashes_a["video"][0] != hashes_a_different_rate["video"][0]
    # nvbug 6226933: a VideoData (with metadata={} and no audio) and a bare
    # frame list are now hashed under distinct, self-describing schemes, so the
    # former intentional collision is removed -- they must differ.
    assert hashes_no_audio["video"][0] != hashes_frame_list["video"][0]

    mm_uuids = {"video": ["shared-video-id"]}
    hashes_uuid_a, _ = apply_mm_hashes(
        {"video": [_make_video(frames, audio_a)]}, mm_uuids)
    hashes_uuid_b, _ = apply_mm_hashes(
        {"video": [_make_video(frames, audio_b)]}, mm_uuids)
    assert hashes_uuid_a["video"][0] != hashes_uuid_b["video"][0]


def test_apply_mm_hashes_image_dims_distinguished():
    img_tall = Image.new("RGBA", (30, 100), (10, 20, 30, 40))
    img_wide = Image.new("RGBA", (100, 30), (10, 20, 30, 40))

    hashes_tall, _ = apply_mm_hashes({"image": [img_tall]})
    hashes_wide, _ = apply_mm_hashes({"image": [img_wide]})

    assert hashes_tall["image"][0] != hashes_wide["image"][0]


def test_apply_mm_hashes_ndarray_shape_distinguished():
    base = np.arange(12, dtype=np.uint8)
    arr_2x6 = base.reshape(2, 6, 1)
    arr_3x4 = base.reshape(3, 4, 1)
    arr_6x2 = base.reshape(6, 2, 1)

    h_2x6, _ = apply_mm_hashes({"image": [arr_2x6]})
    h_3x4, _ = apply_mm_hashes({"image": [arr_3x4]})
    h_6x2, _ = apply_mm_hashes({"image": [arr_6x2]})

    assert h_2x6["image"][0] != h_3x4["image"][0]
    assert h_2x6["image"][0] != h_6x2["image"][0]
    assert h_3x4["image"][0] != h_6x2["image"][0]


def test_apply_mm_hashes_tensor_shape_distinguished():
    data = torch.arange(3 * 224 * 224, dtype=torch.float32)
    t_chw = data.reshape(3, 224, 224)
    t_nchw = data.reshape(1, 3, 224, 224)

    h_chw, _ = apply_mm_hashes({"image": [t_chw]})
    h_nchw, _ = apply_mm_hashes({"image": [t_nchw]})

    assert h_chw["image"][0] != h_nchw["image"][0]


def test_apply_mm_hashes_video_metadata_distinguished():
    frames = [
        Image.new("RGB", (2, 2), (10, 20, 30)),
        Image.new("RGB", (2, 2), (40, 50, 60)),
    ]
    meta_a = {
        "fps": 30.0,
        "duration": 2.0,
        "frames_indices": [0, 1],
        "total_num_frames": 60,
    }
    meta_fps = {**meta_a, "fps": 24.0}
    meta_duration = {**meta_a, "duration": 3.0}
    meta_indices = {**meta_a, "frames_indices": [0, 2]}
    meta_total = {**meta_a, "total_num_frames": 90}

    h_a, _ = apply_mm_hashes({"video": [_make_video(frames, metadata=meta_a)]})
    h_a_copy, _ = apply_mm_hashes(
        {"video": [_make_video(frames, metadata=dict(meta_a))]})
    h_fps, _ = apply_mm_hashes(
        {"video": [_make_video(frames, metadata=meta_fps)]})
    h_duration, _ = apply_mm_hashes(
        {"video": [_make_video(frames, metadata=meta_duration)]})
    h_indices, _ = apply_mm_hashes(
        {"video": [_make_video(frames, metadata=meta_indices)]})
    h_total, _ = apply_mm_hashes(
        {"video": [_make_video(frames, metadata=meta_total)]})

    # Same frames + same metadata -> deterministic / equal.
    assert h_a["video"][0] == h_a_copy["video"][0]
    # Each differing metadata field changes the hash.
    assert h_a["video"][0] != h_fps["video"][0]
    assert h_a["video"][0] != h_duration["video"][0]
    assert h_a["video"][0] != h_indices["video"][0]
    assert h_a["video"][0] != h_total["video"][0]


def test_apply_mm_hashes_video_metadata_numpy_scalars():
    # numpy 2.x scalars (np.int64/np.float32/np.bool_) are not subclasses of
    # Python int/float/bool; numpy-typed metadata must hash like Python-typed.
    frames = [Image.new("RGB", (2, 2), (10, 20, 30))]
    meta_py = {
        "fps": 30.0,
        "duration": 2.0,
        "frames_indices": [0, 5],
        "total_num_frames": 100,
    }
    meta_np = {
        "fps": np.float32(30.0),
        "duration": np.float64(2.0),
        "frames_indices": [np.int64(0), np.int64(5)],
        "total_num_frames": np.int64(100),
    }

    h_py, _ = apply_mm_hashes(
        {"video": [_make_video(frames, metadata=meta_py)]})
    h_np, _ = apply_mm_hashes(
        {"video": [_make_video(frames, metadata=meta_np)]})

    assert h_np["video"][0] == h_py["video"][0]


def test_apply_mm_hashes_videodata_vs_framelist_distinguished():
    frames = [
        Image.new("RGB", (2, 2), (10, 20, 30)),
        Image.new("RGB", (2, 2), (40, 50, 60)),
    ]
    video = _make_video(frames, audio_samples=None)

    h_video, _ = apply_mm_hashes({"video": [video]})
    h_list, _ = apply_mm_hashes({"video": [list(frames)]})

    assert h_video["video"][0] != h_list["video"][0]


def test_apply_mm_hashes_identical_inputs_match():
    img = Image.new("RGBA", (8, 12), (1, 2, 3, 4))
    img_copy = Image.new("RGBA", (8, 12), (1, 2, 3, 4))
    h_img, _ = apply_mm_hashes({"image": [img]})
    h_img_copy, _ = apply_mm_hashes({"image": [img_copy]})
    assert h_img["image"][0] == h_img_copy["image"][0]

    tensor = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    h_t, _ = apply_mm_hashes({"image": [tensor]})
    h_t_copy, _ = apply_mm_hashes({"image": [tensor.clone()]})
    assert h_t["image"][0] == h_t_copy["image"][0]

    arr = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
    h_arr, _ = apply_mm_hashes({"image": [arr]})
    h_arr_copy, _ = apply_mm_hashes({"image": [arr.copy()]})
    assert h_arr["image"][0] == h_arr_copy["image"][0]

    frames = [
        Image.new("RGB", (2, 2), (10, 20, 30)),
        Image.new("RGB", (2, 2), (40, 50, 60)),
    ]
    audio = np.array([0.0, 0.25, -0.5], dtype=np.float32)
    meta = {"fps": 30.0, "frames_indices": [0, 1]}
    video = _make_video(frames, audio_samples=audio, metadata=meta)
    video_copy = _make_video(frames,
                             audio_samples=audio.copy(),
                             metadata=dict(meta))
    h_v, _ = apply_mm_hashes({"video": [video]})
    h_v_copy, _ = apply_mm_hashes({"video": [video_copy]})
    assert h_v["video"][0] == h_v_copy["video"][0]


def test_serialize_item_array_container_agnostic():
    # A torch.Tensor and an np.ndarray with identical dtype/shape/bytes are the
    # same multimodal content, so the container type is not part of the hash
    # identity. Shape and dtype still distinguish.
    tensor = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    array = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    assert serialize_item(tensor) == serialize_item(array)
    assert serialize_item(tensor.reshape(4, 3, 2)) != serialize_item(array)
    assert serialize_item(np.arange(24, dtype=np.int64)) != serialize_item(
        np.arange(24, dtype=np.float64))


def test_serialize_item_sequence_container_agnostic():
    # A tuple and a list with identical elements are the same ordered sequence.
    assert serialize_item((1, 2.0, b"x")) == serialize_item([1, 2.0, b"x"])


def test_apply_mm_hashes_audio_data_deterministic():
    """AudioData hashes are deterministic and include sample rate."""
    audio_a = AudioData(samples=np.array([0.0, 0.25, -0.5], dtype=np.float32),
                        sample_rate=16000)
    audio_a_copy = AudioData(samples=audio_a.samples.copy(), sample_rate=16000)
    audio_b = AudioData(samples=np.array([0.0, 0.25, -1.0], dtype=np.float32),
                        sample_rate=16000)
    audio_a_different_rate = AudioData(samples=audio_a.samples.copy(),
                                       sample_rate=8000)

    hashes_a, _ = apply_mm_hashes({"audio": [audio_a]})
    hashes_a_copy, _ = apply_mm_hashes({"audio": [audio_a_copy]})
    hashes_b, _ = apply_mm_hashes({"audio": [audio_b]})
    hashes_a_different_rate, _ = apply_mm_hashes(
        {"audio": [audio_a_different_rate]})

    assert hashes_a["audio"][0] == hashes_a_copy["audio"][0]
    assert hashes_a["audio"][0] != hashes_b["audio"][0]
    assert hashes_a["audio"][0] != hashes_a_different_rate["audio"][0]


def test_int32_hexdigest_roundtrip():
    """Test that hexdigest_to_int32 and int32_to_hexdigest are inverses."""
    from tensorrt_llm.inputs.multimodal import (hexdigest_to_int32,
                                                int32_to_hexdigest)

    # Test with various hash patterns
    test_hashes = [
        "0000000000000000000000000000000000000000000000000000000000000000",
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        "0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20",
        "deadbeefcafebabefeedfacebadc0ffedeadbeefcafebabefeedfacebadc0ffe",
    ]

    for original_hex in test_hashes:
        int32_values = hexdigest_to_int32(original_hex)
        recovered_hex = int32_to_hexdigest(int32_values)
        assert recovered_hex == original_hex, f"Roundtrip failed for {original_hex}"


def test_multimodal_input_dataclass_with_uuids():
    """Test Python MultimodalInput dataclass with UUIDs."""
    from tensorrt_llm.inputs.multimodal import MultimodalInput

    # Test with all UUIDs
    mm_input = MultimodalInput(multimodal_hashes=[[1, 2, 3, 4, 5, 6, 7, 8]],
                               multimodal_positions=[10],
                               multimodal_lengths=[50],
                               multimodal_uuids=["test-uuid-123"])

    assert mm_input.multimodal_uuids == ["test-uuid-123"]

    # Test with partial UUIDs (some None)
    mm_input_partial = MultimodalInput(
        multimodal_hashes=[[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]],
        multimodal_positions=[10, 100],
        multimodal_lengths=[50, 60],
        multimodal_uuids=["sku-001", None])

    assert mm_input_partial.multimodal_uuids == ["sku-001", None]

    # Test with None UUIDs (default)
    mm_input_no_uuids = MultimodalInput(
        multimodal_hashes=[[1, 2, 3, 4, 5, 6, 7, 8]],
        multimodal_positions=[10],
        multimodal_lengths=[50])

    assert mm_input_no_uuids.multimodal_uuids is None


def test_multimodal_input_dataclass_uuid_validation():
    """Test MultimodalInput validation for multimodal_uuids field."""
    from tensorrt_llm.inputs.multimodal import MultimodalInput

    # Test UUID list length mismatch
    with pytest.raises(ValueError, match="multimodal_uuids length"):
        MultimodalInput(multimodal_hashes=[[1, 2, 3, 4, 5, 6, 7, 8],
                                           [8, 7, 6, 5, 4, 3, 2, 1]],
                        multimodal_positions=[10, 100],
                        multimodal_lengths=[50, 60],
                        multimodal_uuids=["only-one-uuid"])

    # Test invalid UUID type
    with pytest.raises(TypeError, match="must be a string or None"):
        MultimodalInput(multimodal_hashes=[[1, 2, 3, 4, 5, 6, 7, 8]],
                        multimodal_positions=[10],
                        multimodal_lengths=[50],
                        multimodal_uuids=[123])  # Integer instead of string

    # Test invalid multimodal_uuids type (not a list)
    with pytest.raises(TypeError, match="multimodal_uuids must be a list"):
        MultimodalInput(multimodal_hashes=[[1, 2, 3, 4, 5, 6, 7, 8]],
                        multimodal_positions=[10],
                        multimodal_lengths=[50],
                        multimodal_uuids="not-a-list")


def test_multimodal_input_dataclass_rejects_mismatched_item_arrays():
    with pytest.raises(ValueError, match="must all have the same length"):
        MultimodalInput(multimodal_hashes=[[1, 2, 3, 4, 5, 6, 7, 8]],
                        multimodal_positions=[10, 100],
                        multimodal_lengths=[50, 60])


def test_multimodal_input_from_components_with_uuids():
    """Test MultimodalInput.from_components factory method with UUIDs."""
    from tensorrt_llm.inputs.multimodal import MultimodalInput

    mm_hashes = [[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]]
    mm_positions = [10, 100]
    mm_lengths = [50, 60]
    mm_uuids = ["uuid-a", "uuid-b"]

    mm_input = MultimodalInput.from_components(mm_hashes, mm_positions,
                                               mm_lengths, mm_uuids)

    assert mm_input.multimodal_hashes == mm_hashes
    assert mm_input.multimodal_positions == mm_positions
    assert mm_input.multimodal_lengths == mm_lengths
    assert mm_input.multimodal_uuids == mm_uuids

    # Test without UUIDs
    mm_input_no_uuids = MultimodalInput.from_components(mm_hashes, mm_positions,
                                                        mm_lengths)
    assert mm_input_no_uuids.multimodal_uuids is None


def test_multimodal_input_exact_run_buffers():
    """Test optional exact run buffers on Python MultimodalInput."""
    mm_mask = torch.tensor(
        [False, True, True, False, False, True, False, False, True, True])
    item_run_cu_offsets, run_positions, run_lengths = _find_mm_token_runs_from_mask(
        mm_mask, [3, 2])

    assert item_run_cu_offsets == [0, 2, 3]
    assert run_positions == [1, 5, 8]
    assert run_lengths == [2, 1, 2]

    mm_input = MultimodalInput.from_components(
        [[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]],
        [1, 8],
        [3, 2],
        ["item-a", None],
        item_run_cu_offsets,
        run_positions,
        run_lengths,
    )
    assert mm_input.multimodal_item_run_cu_offsets == item_run_cu_offsets
    assert mm_input.multimodal_run_positions == run_positions
    assert mm_input.multimodal_run_lengths == run_lengths

    with pytest.raises(ValueError, match="must be provided together"):
        MultimodalInput(
            multimodal_hashes=[[1, 2, 3, 4, 5, 6, 7, 8]],
            multimodal_positions=[1],
            multimodal_lengths=[2],
            multimodal_item_run_cu_offsets=[0, 1],
        )

    with pytest.raises(ValueError, match="sum to 3, expected 2"):
        MultimodalInput(
            multimodal_hashes=[[1, 2, 3, 4, 5, 6, 7, 8]],
            multimodal_positions=[1],
            multimodal_lengths=[2],
            multimodal_item_run_cu_offsets=[0, 1],
            multimodal_run_positions=[1],
            multimodal_run_lengths=[3],
        )


def test_multimodal_input_rejects_exact_run_int32_overflow():
    int32_max = 2_147_483_647
    with pytest.raises(ValueError, match="end position exceeds int32"):
        MultimodalInput(
            multimodal_hashes=[[1, 2, 3, 4, 5, 6, 7, 8]],
            multimodal_positions=[int32_max - 1],
            multimodal_lengths=[2],
            multimodal_item_run_cu_offsets=[0, 1],
            multimodal_run_positions=[int32_max - 1],
            multimodal_run_lengths=[2],
        )

    MultimodalInput(
        multimodal_hashes=[[1, 2, 3, 4, 5, 6, 7, 8]],
        multimodal_positions=[int32_max - 1],
        multimodal_lengths=[1],
        multimodal_item_run_cu_offsets=[0, 1],
        multimodal_run_positions=[int32_max - 1],
        multimodal_run_lengths=[1],
    )


def test_apply_mm_hashes_uuid_length_mismatch():
    """Test apply_mm_hashes raises error on UUID list length mismatch."""
    import torch

    from tensorrt_llm.inputs.multimodal import apply_mm_hashes

    mock_image1 = torch.randn(3, 224, 224)
    mock_image2 = torch.randn(3, 224, 224)
    mm_data = {"image": [mock_image1, mock_image2]}

    # Mismatched UUID list length
    mm_uuids_wrong_length = {"image": ["only-one-uuid"]}  # Should have 2

    with pytest.raises(ValueError,
                       match="UUID list length.*doesn't match.*data items"):
        apply_mm_hashes(mm_data, mm_uuids_wrong_length)


def test_apply_mm_hashes_multiple_modalities():
    """Test apply_mm_hashes with multiple modalities and UUIDs."""
    import torch

    from tensorrt_llm.inputs.multimodal import apply_mm_hashes

    # Create mock data for multiple modalities
    torch.manual_seed(42)
    mock_image = torch.randn(3, 224, 224)
    mock_video_frames = [torch.randn(3, 224, 224) for _ in range(4)]

    mm_data = {"image": [mock_image], "video": [mock_video_frames]}

    # First, get content-only hashes (without UUIDs)
    hashes_no_uuid, _ = apply_mm_hashes(mm_data)

    # UUIDs for each modality
    mm_uuids = {"image": ["img-uuid-001"], "video": ["vid-uuid-001"]}

    hashes, uuids_list = apply_mm_hashes(mm_data, mm_uuids)

    # Check hashes are 64-char hex strings (combined UUID+content hashes)
    assert len(hashes["image"][0]) == 64
    assert len(hashes["video"][0]) == 64

    # Verify UUIDs change the hashes (UUID+content != content-only)
    assert hashes["image"][0] != hashes_no_uuid["image"][0]
    assert hashes["video"][0] != hashes_no_uuid["video"][0]

    # Check flattened UUID list (order may vary based on dict iteration)
    assert set(uuids_list) == {"img-uuid-001", "vid-uuid-001"}


def test_mm_keys_in_stored_events():
    """Test that mm_keys field is present in stored block events."""
    llm = create_llm()
    sampling_params = SamplingParams(max_tokens=6, temperature=0.01)
    prompt = "Hello, my name is"

    _ = llm.generate(prompt, sampling_params=sampling_params)

    events = llm.get_kv_cache_events(5)

    # Find stored events and verify mm_keys and cache_salt fields
    for event in events:
        if event and event["data"]["type"] == "stored":
            blocks = event["data"]["blocks"]
            for block in blocks:
                # mm_keys should always be present (empty list for text-only)
                assert "mm_keys" in block
                assert isinstance(block["mm_keys"], list)
                # For text-only requests, mm_keys should be empty
                assert block["mm_keys"] == []
                # cache_salt should be present (None for unsalted requests)
                assert "cache_salt" in block
                assert block["cache_salt"] is None


def test_cache_salt_in_stored_events():
    """Test that cache_salt string is preserved in stored block events."""
    llm = create_llm()
    sampling_params = SamplingParams(max_tokens=6, temperature=0.01)
    prompt = "Hello, my name is"

    _ = llm.generate(prompt,
                     sampling_params=sampling_params,
                     cache_salt="tenant-A")

    events = llm.get_kv_cache_events(5)

    # Find stored events and verify cache_salt field
    found_stored = False
    for event in events:
        if event and event["data"]["type"] == "stored":
            found_stored = True
            blocks = event["data"]["blocks"]
            for block in blocks:
                assert "cache_salt" in block
                assert block["cache_salt"] == "tenant-A"

    assert found_stored, "No stored events found"


def test_expected_kv_cache_events():
    llm = create_llm()
    sampling_params = SamplingParams(max_tokens=6, temperature=0.01)
    prompt = "Hello, my name is"

    _ = llm.generate(prompt, sampling_params=sampling_params)

    events = llm.get_kv_cache_events(5)
    # created + stored events
    assert events and len(events) >= 2
    for event in events:
        if event:
            if event["event_id"] == 0:
                assert event["data"]["type"] == "created"
            elif event["event_id"] == 1:
                assert event["data"]["type"] == "stored"


def test_kv_cache_event_async_api():
    llm = create_llm()
    sampling_params = SamplingParams(max_tokens=6, temperature=0.01)
    prompt = "Hello, my name is"

    async def generate():
        async for output in llm.generate_async(prompt,
                                               streaming=True,
                                               sampling_params=sampling_params):
            pass

    events = []

    async def get_events():
        async for event in llm.get_kv_cache_events_async():
            events.append(event)

        assert events

    async def main():
        await generate()
        await asyncio.gather(generate(), get_events())
        await asyncio.gather(generate(), get_events())

    asyncio.run(main())


def check_events(llm,
                 requests,
                 sampling_params,
                 scheduling_params=None,
                 attention_dp_rank=None):

    _ = llm.generate(requests[0],
                     sampling_params=sampling_params,
                     scheduling_params=scheduling_params)
    time.sleep(1)
    events = llm.get_kv_cache_events(5)
    # Created or stored event
    total_stored_blocks = 0
    if attention_dp_rank is None:
        event = events.pop(0)  # created event
        assert event["event_id"] == 0
        assert event["data"]["type"] == "created"
        while events:
            event = events.pop(0)
            if event:
                assert event["data"]["type"] == "stored"
                assert event["event_id"] > 0
                total_stored_blocks += len(event["data"]["blocks"])
    else:
        while events:
            event = events.pop(0)
            if not event:
                continue
            assert "attention_dp_rank" in event
            if event["attention_dp_rank"] == attention_dp_rank:
                assert event["data"]["type"] in ["created", "stored"]
                if event["data"]["type"] == "created":
                    assert event["event_id"] == 0
                if event["data"]["type"] == "stored":
                    assert event["event_id"] > 0
                    total_stored_blocks += len(event["data"]["blocks"])

    assert total_stored_blocks == 5  # Should have 5 blocks in total

    _ = llm.generate(requests[1],
                     sampling_params=sampling_params,
                     scheduling_params=scheduling_params)
    time.sleep(1)
    events2 = llm.get_kv_cache_events(5)

    total_stored_blocks = 0
    has_removed_event = False
    while events2:
        event = events2.pop(0)
        if event and (attention_dp_rank is None
                      or event.get("attention_dp_rank") == attention_dp_rank):
            if event["data"]["type"] == "removed":
                has_removed_event = True
                assert event["data"]["block_hashes"]
            # stored events
            elif event["data"]["type"] == "stored":
                total_stored_blocks += len(event["data"]["blocks"])

    assert total_stored_blocks == 5  # Should have 5 blocks in total
    assert has_removed_event

    _ = llm.generate(requests[2],
                     sampling_params=sampling_params,
                     scheduling_params=scheduling_params)
    time.sleep(1)
    events3 = llm.get_kv_cache_events(5)

    total_stored_blocks = 0
    has_removed_event = False
    while events3:
        event = events3.pop(0)
        if event and (attention_dp_rank is None
                      or event.get("attention_dp_rank") == attention_dp_rank):

            if event["data"]["type"] == "removed":
                has_removed_event = True
                assert event["data"]["block_hashes"]
            elif event["data"]["type"] == "stored":
                total_stored_blocks += len(event["data"]["blocks"])

    assert total_stored_blocks == 5  # Should have 5 blocks in total
    assert has_removed_event

    # no more events after request is finished
    assert not llm.get_kv_cache_events(5)


def test_llm_kv_events_api():
    llm = create_llm()
    sampling_params = SamplingParams(max_tokens=6,
                                     temperature=0.01,
                                     ignore_eos=True)

    requests = []
    for i in range(3):
        input_tokens = list(range(127 + i))[i:]
        requests.append(input_tokens)

    check_events(llm, requests, sampling_params)


@skip_single_gpu
@pytest.mark.threadleak(enabled=False)
def test_llm_api_attention_dp_kv_events():

    llm = LLM(model=llama_model_path,
              tensor_parallel_size=2,
              enable_attention_dp=True,
              kv_cache_config=global_kvcache_config,
              enable_autotuner=False)

    sampling_params = SamplingParams(max_tokens=6,
                                     temperature=0.01,
                                     ignore_eos=True)

    for attention_dp_rank in range(2):
        requests = []
        for i in range(3):
            input_tokens = list(range(127 + i))[i:]
            requests.append(input_tokens)

        scheduling_params = SchedulingParams(
            attention_dp_rank=attention_dp_rank, attention_dp_relax=False)

        check_events(llm, requests, sampling_params, scheduling_params,
                     attention_dp_rank)
