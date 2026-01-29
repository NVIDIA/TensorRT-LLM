import asyncio
import time

import pytest
from utils.util import skip_single_gpu

import tensorrt_llm
from tensorrt_llm import LLM
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._utils import KVCacheEventSerializer
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
                                      onboard_blocks=True,
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
    kv_cache_manager.impl.add_sequence(req.py_request_id, req.prompt_len, 1,
                                       req)
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

    req2 = create_llm_request(1, [1, 2, 3, 4, 5])
    kv_cache_manager.impl.add_sequence(req2.py_request_id, req2.prompt_len, 1,
                                       req2)
    kv_cache_manager.free_resources(req2)

    flush_events(kv_cache_manager)
    events = kv_cache_manager.get_latest_events(10)
    serialized_event = KVCacheEventSerializer.serialize(events)


def test_mm_keys_serialization():
    """Test serialization of multimodal keys (mm_keys) in KV cache events."""
    # Test _mm_key_to_json with a mock mm_key tuple (bytes, int)
    # MmKey from C++ is converted to (bytes, int) tuple by pybind11
    mock_hash = b'\x01\x02\x03\x04\x05\x06\x07\x08' + b'\x00' * 24  # 32 bytes
    mock_offset = 42
    mock_mm_key = (mock_hash, mock_offset)

    result = KVCacheEventSerializer._mm_key_to_json(mock_mm_key)

    assert result["type"] == "mm_key"
    assert result["start_offset"] == 42
    # Hash should be converted to hex string
    assert result["hash"] == "0102030405060708" + "00" * 24
    assert len(result["hash"]) == 64  # 32 bytes = 64 hex chars

    # Test with different hash values
    mock_hash2 = bytes(range(32))  # 0x00 to 0x1f
    mock_mm_key2 = (mock_hash2, 100)
    result2 = KVCacheEventSerializer._mm_key_to_json(mock_mm_key2)

    assert result2["type"] == "mm_key"
    assert result2["start_offset"] == 100
    expected_hash = ''.join(f'{i:02x}' for i in range(32))
    assert result2["hash"] == expected_hash


def test_mm_keys_deserialization():
    """Test deserialization of mm_keys JSON back to 32-byte hash."""
    # Test case 1: Simple hash pattern
    mock_hash = b'\x01\x02\x03\x04\x05\x06\x07\x08' + b'\x00' * 24  # 32 bytes
    mock_offset = 42
    mock_mm_key = (mock_hash, mock_offset)

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
    mock_mm_key2 = (mock_hash2, mock_offset2)

    json_result2 = KVCacheEventSerializer._mm_key_to_json(mock_mm_key2)
    recovered_hash2 = bytes.fromhex(json_result2["hash"])

    assert recovered_hash2 == mock_hash2
    assert len(recovered_hash2) == 32
    assert json_result2["start_offset"] == mock_offset2

    # Test case 3: All 0xFF bytes
    mock_hash3 = b'\xff' * 32
    mock_offset3 = 255
    mock_mm_key3 = (mock_hash3, mock_offset3)

    json_result3 = KVCacheEventSerializer._mm_key_to_json(mock_mm_key3)
    recovered_hash3 = bytes.fromhex(json_result3["hash"])

    assert recovered_hash3 == mock_hash3
    assert len(recovered_hash3) == 32
    assert json_result3["hash"] == "ff" * 32

    # Test case 4: Random-like pattern
    mock_hash4 = bytes([0xde, 0xad, 0xbe, 0xef] + [0xca, 0xfe] * 14)
    mock_offset4 = 1024
    mock_mm_key4 = (mock_hash4, mock_offset4)

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


def test_uuid_to_hash():
    """Test UUID to hash conversion."""
    from tensorrt_llm.inputs.multimodal import uuid_to_hash

    # Short UUID (< 32 bytes) - should be zero-padded
    short_uuid = "sku-1234"
    short_hash = uuid_to_hash(short_uuid)
    assert len(short_hash) == 64  # 32 bytes = 64 hex chars
    # Verify it starts with the UUID bytes in hex
    expected_start = short_uuid.encode('utf-8').hex()
    assert short_hash.startswith(expected_start)
    # Rest should be zeros
    assert short_hash[len(expected_start):] == '00' * (32 - len(short_uuid))

    # Exactly 32 bytes - should not be hashed
    exact_uuid = "a" * 32
    exact_hash = uuid_to_hash(exact_uuid)
    assert len(exact_hash) == 64
    assert exact_hash == exact_uuid.encode('utf-8').hex()

    # Long UUID (> 32 bytes) - should be hashed
    long_uuid = "this-is-a-very-long-uuid-that-exceeds-32-bytes-limit"
    long_hash = uuid_to_hash(long_uuid)
    assert len(long_hash) == 64  # Should still be 64 hex chars (32 bytes)
    # Should not match the padded version (it was hashed)
    assert not long_hash.startswith(long_uuid[:32].encode('utf-8').hex())


def test_apply_mm_hashes_with_uuids():
    """Test apply_mm_hashes with user-provided UUIDs."""
    import torch

    from tensorrt_llm.inputs.multimodal import apply_mm_hashes, uuid_to_hash

    # Create mock multimodal data
    mock_image1 = torch.randn(3, 224, 224)
    mock_image2 = torch.randn(3, 224, 224)
    mm_data = {"image": [mock_image1, mock_image2]}

    # Test without UUIDs - should use content hashing
    hashes_no_uuid, uuids_no_uuid = apply_mm_hashes(mm_data)
    assert len(hashes_no_uuid["image"]) == 2
    assert all(len(h) == 64 for h in hashes_no_uuid["image"])
    assert uuids_no_uuid is None

    # Test with partial UUIDs (first has UUID, second uses content hash)
    mm_uuids = {"image": ["sku-1234-a", None]}
    hashes_partial, uuids_partial = apply_mm_hashes(mm_data, mm_uuids)

    assert len(hashes_partial["image"]) == 2
    # First hash should be from UUID
    assert hashes_partial["image"][0] == uuid_to_hash("sku-1234-a")
    # Second hash should be content-based (same as without UUID)
    assert hashes_partial["image"][1] == hashes_no_uuid["image"][1]
    # UUIDs list should have the UUID and None
    assert uuids_partial == ["sku-1234-a", None]

    # Test with all UUIDs
    mm_uuids_all = {"image": ["sku-1234-a", "sku-1234-b"]}
    hashes_all, uuids_all = apply_mm_hashes(mm_data, mm_uuids_all)

    assert len(hashes_all["image"]) == 2
    assert hashes_all["image"][0] == uuid_to_hash("sku-1234-a")
    assert hashes_all["image"][1] == uuid_to_hash("sku-1234-b")
    assert uuids_all == ["sku-1234-a", "sku-1234-b"]


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


def test_uuid_to_hash_edge_cases():
    """Test uuid_to_hash with edge cases."""
    from tensorrt_llm.inputs.multimodal import uuid_to_hash

    # Empty string
    empty_hash = uuid_to_hash("")
    assert len(empty_hash) == 64
    assert empty_hash == "00" * 32  # All zeros

    # Single character
    single_char_hash = uuid_to_hash("a")
    assert len(single_char_hash) == 64
    assert single_char_hash.startswith("61")  # 'a' = 0x61
    assert single_char_hash[2:] == "00" * 31

    # Special characters in UUID
    special_uuid = "sku-123/abc:def@test!"
    special_hash = uuid_to_hash(special_uuid)
    assert len(special_hash) == 64

    # Unicode characters (will be encoded as UTF-8)
    unicode_uuid = "测试-uuid-中文"
    unicode_hash = uuid_to_hash(unicode_uuid)
    assert len(unicode_hash) == 64

    # Exactly 32 bytes (boundary case)
    exactly_32 = "x" * 32
    exactly_32_hash = uuid_to_hash(exactly_32)
    assert len(exactly_32_hash) == 64
    # Should be the hex encoding of 32 'x' characters
    assert exactly_32_hash == "78" * 32  # 'x' = 0x78

    # 33 bytes (just over, should be hashed)
    over_32 = "y" * 33
    over_32_hash = uuid_to_hash(over_32)
    assert len(over_32_hash) == 64
    # Should NOT be simple hex encoding
    assert over_32_hash != "79" * 32 + "79"[:2]


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

    from tensorrt_llm.inputs.multimodal import apply_mm_hashes, uuid_to_hash

    # Create mock data for multiple modalities
    mock_image = torch.randn(3, 224, 224)
    mock_video_frames = [torch.randn(3, 224, 224) for _ in range(4)]

    mm_data = {"image": [mock_image], "video": [mock_video_frames]}

    # UUIDs for each modality
    mm_uuids = {"image": ["img-uuid-001"], "video": ["vid-uuid-001"]}

    hashes, uuids_list = apply_mm_hashes(mm_data, mm_uuids)

    # Check hashes are computed from UUIDs
    assert hashes["image"][0] == uuid_to_hash("img-uuid-001")
    assert hashes["video"][0] == uuid_to_hash("vid-uuid-001")

    # Check flattened UUID list (order may vary based on dict iteration)
    assert set(uuids_list) == {"img-uuid-001", "vid-uuid-001"}


def test_mm_keys_in_stored_events():
    """Test that mm_keys field is present in stored block events."""
    llm = create_llm()
    sampling_params = SamplingParams(max_tokens=6, temperature=0.01)
    prompt = "Hello, my name is"

    _ = llm.generate(prompt, sampling_params=sampling_params)

    events = llm.get_kv_cache_events(5)

    # Find stored events and verify mm_keys field
    for event in events:
        if event and event["data"]["type"] == "stored":
            blocks = event["data"]["blocks"]
            for block in blocks:
                # mm_keys should always be present (empty list for text-only)
                assert "mm_keys" in block
                assert isinstance(block["mm_keys"], list)
                # For text-only requests, mm_keys should be empty
                assert block["mm_keys"] == []


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
