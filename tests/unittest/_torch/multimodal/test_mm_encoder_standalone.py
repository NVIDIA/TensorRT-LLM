import copy
import json
import os
import time
from pathlib import Path
from typing import Generator

import pytest
import torch
from utils.llm_data import llm_models_root

from tensorrt_llm import MultimodalEncoder
from tensorrt_llm._torch.shared_tensor import SharedTensorContainer
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.inputs import default_multimodal_input_loader
from tensorrt_llm.llmapi import (CacheTransceiverConfig, DisaggregatedParams,
                                 KvCacheConfig, MoeConfig)
from tensorrt_llm.llmapi.llm import LLM, SamplingParams

test_data_root = Path(
    os.path.join(llm_models_root(), "multimodals", "test_data"))
example_images = [
    str(test_data_root / "seashore.png"),
    str(test_data_root / "inpaint.png"),
    str(test_data_root / "61.jpg"),
]

_LLAVA_DIR = llm_models_root() / "multimodals" / "llava-v1.6-mistral-7b-hf"
_QWEN_2_5_VL_DIR = llm_models_root() / "Qwen2.5-VL-3B-Instruct"
_QWEN_3_VL_DIR = llm_models_root() / "Qwen3" / "Qwen3-VL-2B-Instruct"
_QWEN_3_VL_30B_A3B_FP8_DIR = llm_models_root(
) / "Qwen3" / "Qwen3-VL-30B-A3B-Instruct-FP8"

_FAKE_QWEN3_VL_30B_A3B_FP8_SENTINEL = "qwen3_vl_30b_a3b_fp8_fake"
_FAKE_CHECKPOINT_MARKER = ".tllm_fake_checkpoint"


# Unlike the other models, we cannot fit a multimodal encoder + 2 copies of the LLM on a single
# H100 GPU in CI. We therefore resort to creating a slimmed down version of the model with less
# layers.
def _get_fake_qwen3_vl_30b_a3b_config() -> dict:
    config_path = _QWEN_3_VL_30B_A3B_FP8_DIR / "config.json"
    if not config_path.exists():
        pytest.skip(f"Qwen3-VL-30B-A3B config not found: {config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)
    config = copy.deepcopy(config)
    config["text_config"]["num_hidden_layers"] = 2
    return config


def _create_fake_qwen3_vl_30b_a3b_fp8_dir(
    tmp_path_factory: pytest.TempPathFactory,
    assets_dir: Path,
) -> Path:
    if not assets_dir.exists():
        pytest.skip(f"Base model dir not found: {assets_dir}")

    fake_dir = tmp_path_factory.mktemp("qwen3_vl_30b_a3b_fp8_fake")

    for item in assets_dir.iterdir():
        if item.name == "config.json":
            continue
        target = fake_dir / item.name
        if target.exists():
            continue
        os.symlink(item, target, target_is_directory=item.is_dir())

    config_path = fake_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(_get_fake_qwen3_vl_30b_a3b_config(), f, indent=2)

    (fake_dir /
     _FAKE_CHECKPOINT_MARKER).write_text("Synthetic checkpoint for CI tests.\n")
    return fake_dir


def _get_fake_checkpoint_kwargs(model_dir: Path) -> dict:
    if (model_dir / _FAKE_CHECKPOINT_MARKER).exists():
        return {"load_format": "dummy"}
    return {}


def _is_fake_checkpoint(model_dir: Path) -> bool:
    return (model_dir / _FAKE_CHECKPOINT_MARKER).exists()


def _get_moe_config_for_blackwell() -> MoeConfig:
    if get_sm_version() >= 100:
        return MoeConfig(backend="DEEPGEMM")
    return MoeConfig()


@pytest.mark.parametrize(
    "prompts,expected_num_duplicates",
    [
        # Full reuse: same media + same prompts
        # All blocks are reused, thus no duplicates
        (["Describe the natural environment in the image."] * 2, 0),
        # Partial reuse: same media + different prompts
        # Prefix blocks are reused, thus 2 duplicates
        ([
            "Describe the natural environment in the image.",
            "What objects can you see in the image?",
            "Describe the weather in the image.",
        ], 2),
    ])
def test_kv_event_mm_keys_with_reuse(prompts, expected_num_duplicates):
    """Test mm_keys in KV cache events with cache reuse scenarios.

    This test verifies:
    1. KV cache events contain mm_keys for multimodal blocks
    2. mm_keys have the expected structure (hash + start_offset)
    3. Cache reuse behavior based on media and prompts:
       - Same media + same prompts: full reuse (0 duplicate offsets)
       - Same media + different prompts: partial reuse (prefix blocks reused)
    """
    encoder_model_dir = _LLAVA_DIR

    max_tokens = 16
    free_gpu_memory_fraction = 0.2

    # Use same image for all prompts
    media = [example_images[0]] * len(prompts)

    sampling_params = SamplingParams(max_tokens=max_tokens)
    kv_cache_config = KvCacheConfig(
        enable_block_reuse=True,
        free_gpu_memory_fraction=free_gpu_memory_fraction,
        event_buffer_max_size=1024,  # Enable KV cache events
    )
    moe_config = _get_moe_config_for_blackwell()

    llm = LLM(model=encoder_model_dir,
              backend='pytorch',
              kv_cache_config=kv_cache_config,
              moe_config=moe_config,
              max_batch_size=1)

    inputs = _load_inputs(llm, prompts, media)

    with llm:
        # Generate for each input separately to test KV cache reuse
        for inp in inputs:
            _ = llm.generate([inp], sampling_params=sampling_params)

        time.sleep(0.5)  # Wait for events to be dispatched
        events = llm.get_kv_cache_events(10)

    # Extract mm_keys offsets from stored events
    mm_keys_offsets = []
    for event in events:
        if event and event.get("data", {}).get("type") == "stored":
            for block in event["data"].get("blocks", []):
                if block.get("mm_keys"):
                    for mm_key in block["mm_keys"]:
                        assert "hash" in mm_key, "mm_key should have 'hash' field"
                        assert "start_offset" in mm_key, "mm_key should have 'start_offset' field"
                        mm_keys_offsets.append(mm_key["start_offset"])

    num_duplicates = len(mm_keys_offsets) - len(set(mm_keys_offsets))
    assert num_duplicates == expected_num_duplicates, (
        f"Expected {expected_num_duplicates} duplicate mm_keys offsets, "
        f"got {num_duplicates}. Offsets: {mm_keys_offsets}")


@pytest.fixture(
    scope="module",
    params=[
        pytest.param(_LLAVA_DIR, id="llava_7b"),
        pytest.param(_QWEN_2_5_VL_DIR, id="qwen2.5_3b"),
        pytest.param(_QWEN_3_VL_DIR, id="qwen3_2b"),
        pytest.param(_FAKE_QWEN3_VL_30B_A3B_FP8_SENTINEL,
                     id="qwen3_30b_a3b_fp8"),
    ],
)
def model_dir(request, tmp_path_factory: pytest.TempPathFactory) -> Path:
    if request.param == _FAKE_QWEN3_VL_30B_A3B_FP8_SENTINEL:
        return _create_fake_qwen3_vl_30b_a3b_fp8_dir(tmp_path_factory,
                                                     _QWEN_3_VL_DIR)
    return request.param


@pytest.mark.parametrize(
    "use_uuids,expected_hash_type",
    [
        # Without UUIDs: mm_key hash should be a 64-char hex string
        (False, "hex"),
        # With UUIDs: mm_key hash should be the original UUID string
        (True, "uuid"),
    ])
def test_kv_event_mm_keys_with_uuid(use_uuids, expected_hash_type):
    """Test mm_keys in KV cache events return UUID when provided.

    This test verifies that when multi_modal_uuids is provided:
    1. The KV cache event mm_keys 'hash' field contains the original UUID string
    2. Without UUIDs, the hash field contains a 64-char hex string

    The UUID feature allows users to provide stable identifiers for multimodal
    items, which are returned in KV cache events for external cache management.
    """
    encoder_model_dir = _LLAVA_DIR

    max_tokens = 16
    free_gpu_memory_fraction = 0.2

    # Use different images to generate different prompts
    prompts = ["Describe the natural environment in the image."]
    media = [example_images[0]]

    # Define UUIDs if testing with them
    test_uuid = "my-test-image-uuid-12345"

    sampling_params = SamplingParams(max_tokens=max_tokens)
    kv_cache_config = KvCacheConfig(
        enable_block_reuse=True,
        free_gpu_memory_fraction=free_gpu_memory_fraction,
        event_buffer_max_size=1024,
    )

    llm = LLM(model=encoder_model_dir,
              backend='pytorch',
              kv_cache_config=kv_cache_config,
              max_batch_size=1)

    # Load inputs with or without UUIDs
    if use_uuids:
        # Create inputs with multi_modal_uuids
        inputs = _load_inputs_with_uuids(llm, prompts, media, [test_uuid])
    else:
        inputs = _load_inputs(llm, prompts, media)

    with llm:
        for inp in inputs:
            _ = llm.generate([inp], sampling_params=sampling_params)

        # Wait for KV cache events to be dispatched asynchronously
        time.sleep(0.5)
        events = llm.get_kv_cache_events(50)

    # Extract mm_keys from stored events
    mm_keys_found = []
    for event in events:
        if event and event.get("data", {}).get("type") == "stored":
            for block in event["data"].get("blocks", []):
                mm_keys_found.extend(block.get("mm_keys", []))

    # Verify mm_keys were found (multimodal model should have them)
    assert len(mm_keys_found) > 0, "Expected mm_keys in stored events"

    # Verify the hash field matches expected type
    for mm_key in mm_keys_found:
        hash_value = mm_key["hash"]
        if expected_hash_type == "uuid":
            # Should be the original UUID string
            assert hash_value == test_uuid, (
                f"Expected UUID '{test_uuid}', got '{hash_value}'")
        else:
            # Should be a 64-char hex string
            assert len(hash_value) == 64, (
                f"Expected 64-char hex hash, got {len(hash_value)} chars")
            # Verify it's valid hex (fromhex will raise ValueError if invalid)
            bytes.fromhex(hash_value)


def _load_inputs_with_uuids(llm: LLM, prompts, media, uuids):
    """Load inputs with multi_modal_uuids for testing.

    This function uses the same processing pipeline as _load_inputs but adds
    multi_modal_uuids to the processed inputs.
    """
    # Use the standard loader to get properly processed inputs with image tokens
    inputs = _load_inputs(llm, prompts, media)

    # Add multi_modal_uuids to the processed inputs
    for inp, uuid in zip(inputs, uuids):
        inp["multi_modal_uuids"] = {"image": [uuid]}

    return inputs


@pytest.mark.parametrize(
    "uuids,expected_patterns",
    [
        # First image has UUID, second uses content hash
        (["custom-uuid-first", None], ["custom-uuid-first", "hex"]),
        # Both have UUIDs
        (["uuid-img-a", "uuid-img-b"], ["uuid-img-a", "uuid-img-b"]),
        # Both use content hash (None)
        ([None, None], ["hex", "hex"]),
    ])
def test_kv_event_mm_keys_with_partial_uuids(uuids, expected_patterns):
    """Test mm_keys with partial UUIDs (some items with UUID, some without).

    This test verifies the mixed UUID scenario where:
    1. Some multimodal items have user-provided UUIDs
    2. Other items fall back to content-based hashing
    3. KV cache events correctly return UUID or hex hash based on input
    """
    encoder_model_dir = _LLAVA_DIR

    max_tokens = 16
    free_gpu_memory_fraction = 0.2

    # Two different images with potentially mixed UUIDs
    prompt = "Describe both images in detail."
    images = [example_images[0], example_images[1]]

    sampling_params = SamplingParams(max_tokens=max_tokens)
    kv_cache_config = KvCacheConfig(
        enable_block_reuse=True,
        free_gpu_memory_fraction=free_gpu_memory_fraction,
        event_buffer_max_size=1024,
    )

    llm = LLM(model=encoder_model_dir,
              backend='pytorch',
              kv_cache_config=kv_cache_config,
              max_batch_size=1)

    # Load input using the multimodal input loader directly for multiple images per prompt
    config_path = os.path.join(llm._hf_model_dir, 'config.json')
    with open(config_path, 'r') as f:
        model_config = json.load(f)
    model_type = model_config['model_type']

    inputs = default_multimodal_input_loader(tokenizer=llm.tokenizer,
                                             model_dir=llm._hf_model_dir,
                                             model_type=model_type,
                                             modality="multiple_image",
                                             prompts=[prompt],
                                             media=[images],
                                             image_data_format="pt")

    # Add multi_modal_uuids to the processed input
    inputs[0]["multi_modal_uuids"] = {"image": uuids}
    inp = inputs[0]

    with llm:
        _ = llm.generate([inp], sampling_params=sampling_params)

        # Wait for KV cache events to be dispatched asynchronously
        time.sleep(0.5)
        events = llm.get_kv_cache_events(50)

    # Collect all unique mm_key hashes from stored events
    mm_key_hashes = set()
    for event in events:
        if event and event.get("data", {}).get("type") == "stored":
            for block in event["data"].get("blocks", []):
                if block.get("mm_keys"):
                    for mm_key in block["mm_keys"]:
                        mm_key_hashes.add(mm_key["hash"])

    # Verify we got mm_keys
    assert len(mm_key_hashes) > 0, "Expected mm_keys in stored events"

    # Verify each expected pattern appears in the results
    for pattern in expected_patterns:
        if pattern == "hex":
            # Should find at least one 64-char hex string
            hex_found = any(
                len(h) == 64 and all(c in '0123456789abcdef' for c in h)
                for h in mm_key_hashes)
            assert hex_found, f"Expected hex hash pattern but got: {mm_key_hashes}"
        else:
            # Should find the exact UUID string
            assert pattern in mm_key_hashes, (
                f"Expected UUID '{pattern}' in mm_keys, got: {mm_key_hashes}")


def test_kv_event_mm_keys_with_uuid_multiple_prompts():
    """Test mm_keys with UUIDs across multiple prompts, each with its own image.

    This test verifies that when multiple prompts are processed, each with its own
    multimodal data and UUID:
    1. Each prompt's mm_keys correctly return the associated UUID
    2. Different UUIDs are preserved for different prompts
    3. KV cache events correctly associate UUIDs with their respective blocks
    """
    encoder_model_dir = _LLAVA_DIR

    max_tokens = 16
    free_gpu_memory_fraction = 0.2

    # Multiple prompts, each with its own image and UUID
    prompts = [
        "Describe the natural environment in the image.",
        "What objects can you see in the image?",
        "Describe the weather in the image.",
    ]
    media = [example_images[0], example_images[1], example_images[2]]
    uuids = ["uuid-image-seashore", "uuid-image-inpaint", "uuid-image-61"]

    sampling_params = SamplingParams(max_tokens=max_tokens)
    kv_cache_config = KvCacheConfig(
        enable_block_reuse=True,
        free_gpu_memory_fraction=free_gpu_memory_fraction,
        event_buffer_max_size=2048,
    )

    llm = LLM(model=encoder_model_dir,
              backend='pytorch',
              kv_cache_config=kv_cache_config,
              max_batch_size=1)

    # Load inputs with UUIDs for each prompt
    inputs = _load_inputs(llm, prompts, media)

    # Add multi_modal_uuids to each input
    for inp, uuid in zip(inputs, uuids):
        inp["multi_modal_uuids"] = {"image": [uuid]}

    with llm:
        # Generate for each input separately
        for inp in inputs:
            _ = llm.generate([inp], sampling_params=sampling_params)

        # Wait for KV cache events to be dispatched asynchronously
        time.sleep(0.5)
        events = llm.get_kv_cache_events(50)

    # Collect all unique mm_key hashes from stored events
    mm_key_hashes = set()
    for event in events:
        if event and event.get("data", {}).get("type") == "stored":
            for block in event["data"].get("blocks", []):
                if block.get("mm_keys"):
                    for mm_key in block["mm_keys"]:
                        mm_key_hashes.add(mm_key["hash"])

    # Verify we got mm_keys
    assert len(mm_key_hashes) > 0, "Expected mm_keys in stored events"

    # Verify each UUID appears in the results
    for uuid in uuids:
        assert uuid in mm_key_hashes, (
            f"Expected UUID '{uuid}' in mm_keys, got: {mm_key_hashes}")


def test_kv_event_mm_keys_with_very_long_uuid():
    """Test mm_keys with UUIDs that exceed 64 bytes.

    This test verifies that the system correctly handles UUIDs that are longer
    than the typical 64-character hex hash representation:
    1. Very long UUIDs (>64 bytes) are preserved and returned correctly
    2. No truncation or corruption occurs for long UUID strings
    3. The full UUID is returned in KV cache events
    """
    encoder_model_dir = _LLAVA_DIR

    max_tokens = 16
    free_gpu_memory_fraction = 0.2

    prompt = "Describe the natural environment in the image."

    # Create UUIDs of varying lengths, including very long ones
    # Normal UUID (36 chars): standard format
    # Medium UUID (80 chars): exceeds 64-byte hash representation
    # Very long UUID (200+ chars): stress test for string handling
    long_uuid_80 = "sku-product-image-" + "a" * 62  # 80 chars total
    very_long_uuid_200 = (
        "enterprise-asset-management-system/region/us-east-1/bucket/media-assets/"
        "category/electronics/subcategory/smartphones/brand/example-brand/"
        "product-line/flagship-series/sku/SKU-2024-FLAGSHIP-PRO-MAX-256GB-MIDNIGHT-BLACK"
    )  # ~200 chars

    # Use 2 images with different long UUIDs
    images = [example_images[0], example_images[1]]
    uuids = [long_uuid_80, very_long_uuid_200]

    sampling_params = SamplingParams(max_tokens=max_tokens)
    kv_cache_config = KvCacheConfig(
        enable_block_reuse=True,
        free_gpu_memory_fraction=free_gpu_memory_fraction,
        event_buffer_max_size=1024,
    )

    llm = LLM(model=encoder_model_dir,
              backend='pytorch',
              kv_cache_config=kv_cache_config,
              max_batch_size=1)

    # Load input using the multimodal input loader for multiple images
    config_path = os.path.join(llm._hf_model_dir, 'config.json')
    with open(config_path, 'r') as f:
        model_config = json.load(f)
    model_type = model_config['model_type']

    inputs = default_multimodal_input_loader(tokenizer=llm.tokenizer,
                                             model_dir=llm._hf_model_dir,
                                             model_type=model_type,
                                             modality="multiple_image",
                                             prompts=[prompt],
                                             media=[images],
                                             image_data_format="pt")

    # Add very long UUIDs
    inputs[0]["multi_modal_uuids"] = {"image": uuids}
    inp = inputs[0]

    with llm:
        _ = llm.generate([inp], sampling_params=sampling_params)

        # Wait for KV cache events to be dispatched asynchronously
        time.sleep(0.5)
        events = llm.get_kv_cache_events(50)

    # Collect all unique mm_key hashes from stored events
    mm_key_hashes = set()
    for event in events:
        if event and event.get("data", {}).get("type") == "stored":
            for block in event["data"].get("blocks", []):
                if block.get("mm_keys"):
                    for mm_key in block["mm_keys"]:
                        mm_key_hashes.add(mm_key["hash"])

    # Verify we got mm_keys
    assert len(mm_key_hashes) > 0, "Expected mm_keys in stored events"

    # Verify the 80-char UUID is present and not truncated
    assert long_uuid_80 in mm_key_hashes, (
        f"Expected 80-char UUID '{long_uuid_80}' in mm_keys, got: {mm_key_hashes}"
    )

    # Verify the 200-char UUID is present and not truncated
    assert very_long_uuid_200 in mm_key_hashes, (
        f"Expected 200-char UUID '{very_long_uuid_200}' in mm_keys, got: {mm_key_hashes}"
    )

    # Verify the UUIDs are exactly as provided (no truncation)
    for uuid in uuids:
        matching = [h for h in mm_key_hashes if h == uuid]
        assert len(matching) == 1, (
            f"UUID '{uuid}' (len={len(uuid)}) should appear exactly once, "
            f"found {len(matching)} times in {mm_key_hashes}")


@pytest.fixture(scope="module", params=[False, True])
def pd_disagg(request) -> bool:
    return request.param


@pytest.fixture(scope="module")
def llms(model_dir: Path,
         pd_disagg: bool) -> Generator[tuple[LLM, LLM | None], None, None]:
    """Get LLM for prefill and, if disagg, separate LLM for decode."""
    free_gpu_memory_fraction = 0.2
    disable_overlap_scheduler = pd_disagg
    # NOTE: if the number of tokens that need to pass from P -> D exceeds `max_tokens_in_buffer`,
    # one may see the following error:
    # >>> tensorrt_llm.executor.utils.RequestError: Error in kv cache transfer for generation
    #     requests.
    cache_transceiver_cfg = CacheTransceiverConfig(
        backend="DEFAULT", max_tokens_in_buffer=10240) if pd_disagg else None
    kv_cache_config = KvCacheConfig(
        enable_block_reuse=False,  # Disable for output 1:1 matching check
        free_gpu_memory_fraction=free_gpu_memory_fraction,
    )

    load_kwargs = _get_fake_checkpoint_kwargs(model_dir)
    moe_config = _get_moe_config_for_blackwell()
    llm = LLM(
        model=model_dir,
        backend='pytorch',
        kv_cache_config=kv_cache_config,
        moe_config=moe_config,
        trust_remote_code=True,
        cache_transceiver_config=cache_transceiver_cfg,
        disable_overlap_scheduler=disable_overlap_scheduler,
        max_batch_size=1,  # fix batch size to reduce non-determinism in tests
        **load_kwargs,
    )
    with llm:
        if pd_disagg:
            llm_decode = LLM(
                model=model_dir,
                backend='pytorch',
                kv_cache_config=kv_cache_config,
                moe_config=moe_config,
                trust_remote_code=True,
                cache_transceiver_config=cache_transceiver_cfg,
                **load_kwargs,
            )
            with llm_decode:
                yield (llm, llm_decode)
        else:
            yield (llm, None)


def _load_inputs(llm: LLM, prompts, media, mm_embeddings=None):
    # Load model configuration
    config_path = os.path.join(llm._hf_model_dir, 'config.json')
    assert os.path.exists(
        config_path), f"Model config not found at {config_path}"

    with open(config_path, 'r') as f:
        model_config = json.load(f)
    model_type = model_config['model_type']

    # Prepare multimodal inputs
    inputs = default_multimodal_input_loader(tokenizer=llm.tokenizer,
                                             model_dir=llm._hf_model_dir,
                                             model_type=model_type,
                                             modality="image",
                                             prompts=prompts,
                                             media=media,
                                             mm_embeddings=mm_embeddings,
                                             image_data_format="pt")
    # Validate inputs structure
    assert len(inputs) == len(
        prompts), f"Expected {len(prompts)} inputs, got {len(inputs)}"
    return inputs


def _assert_handles_are_different(x: dict | None, y: dict | None) -> None:
    # Helper function for checking that two SharedTensorContainer dict representations of the same
    # underlying data are different. Certain metadata should stay the same (basically those describing
    # the tensor's contents), while others should actually differ (those pertaining to the underlying
    # storage).
    matching_keys = [
        "dtype",
        "event_sync_required",
        "method_key",
        "requires_grad",
        # NOTE: this assumes the workers are on the same physical device, which is the case in
        # the tests in this file since `LLM` API does not expose a way to select the device ID.
        "storage_device",
        "storage_size_bytes",
        "tensor_offset",
        "tensor_size",
        "tensor_stride",
    ]

    different_keys = [
        "event_handle",
        "ref_counter_handle",
        "ref_counter_offset",
        "storage_handle",
        "storage_offset_bytes",
    ]

    assert set(matching_keys + different_keys) == x.keys() == y.keys()

    for key in matching_keys:
        assert x[key] == y[key]
    for key in different_keys:
        assert x[key] != y[key]


@pytest.mark.threadleak(enabled=False)
def test_single_request_chat_multiple_images(
    pd_disagg: bool,
    model_dir: Path,
    llms: tuple[LLM, LLM | None],
):
    """Test processing a single request with multiple images.

    This test verifies that encoder (pass mm_embeddings) + LLM API produces identical
    results to standard llm generation (pass raw image) by comparing outputs.
    """
    llm, llm_decode = llms

    # Test configuration
    max_tokens = 64
    max_batch_size = 1

    # Test data - OpenAI chat completion format
    prompts = ["Compare these 2 images."]
    media = [example_images[0], example_images[1]]

    # Sampling configuration
    sampling_params = SamplingParams(max_tokens=max_tokens)

    # Prepare multimodal inputs
    inputs = _load_inputs(llm, prompts, media)

    # Generate reference output with raw multimodal inputs
    outputs_ref = llm.generate(inputs, sampling_params=sampling_params)

    # Validate reference outputs
    assert outputs_ref is not None, "Reference generation returned None"
    assert len(outputs_ref) == len(
        prompts
    ), f"Expected {len(prompts)} reference outputs, got {len(outputs_ref)}"
    for i, output in enumerate(outputs_ref):
        assert len(
            output.outputs
        ) > 0, f"Reference generation has no output text for input {i}"

    # Prepare inputs for llm (pass mm_embeddings)
    # Process multimodal data using encoder (pass mm_embeddings)
    encoder = MultimodalEncoder(model=model_dir,
                                max_batch_size=max_batch_size,
                                **_get_fake_checkpoint_kwargs(model_dir))
    with encoder:
        encoder_outputs = encoder.generate(inputs)

        # Generate output using llm (pass mm_embeddings)
        ep_disaggregated_params = encoder_outputs[0].disaggregated_params

        assert ep_disaggregated_params is not None, "Encoder output disaggregated params is None"
        ep_disaggregated_params.request_type = "context_and_generation" if not pd_disagg else "context_only"

        outputs = llm.generate(inputs,
                               sampling_params=sampling_params,
                               disaggregated_params=ep_disaggregated_params)

        if pd_disagg:
            # Generation using llm_decode
            assert len(outputs) == 1
            pd_disaggregated_params = outputs[0].disaggregated_params

            ep_handle = ep_disaggregated_params.mrope_position_ids_handle
            pd_handle = pd_disaggregated_params.mrope_position_ids_handle
            assert type(ep_handle) is type(pd_handle)
            if ep_handle is not None:
                _assert_handles_are_different(ep_handle, pd_handle)
            pd_disaggregated_params.request_type = "generation_only"
            sampling_params = SamplingParams(max_tokens=max_tokens)
            # remove multimodal data from input as decoder worker doesn't need it
            inputs[0]['multi_modal_data'] = None
            # use prompt token ids from encoder output
            inputs[0]['prompt_token_ids'] = outputs[0].prompt_token_ids

            outputs = llm_decode.generate(
                inputs,
                sampling_params=sampling_params,
                disaggregated_params=pd_disaggregated_params)

    # Validate outputs
    assert len(outputs) == len(
        prompts), f"Expected {len(prompts)} outputs, got {len(outputs)}"
    for i, output in enumerate(outputs):
        assert len(
            output.outputs) > 0, f"generation has no output text for input {i}"

    # Compare outputs - they should match exactly
    assert len(outputs_ref) == len(
        outputs
    ), f"Number of outputs don't match: {len(outputs_ref)} vs {len(outputs)}"

    for i, (ref_output, test_output) in enumerate(zip(outputs_ref, outputs)):
        # Cannot compare prompts as decoder worker would void it
        #assert ref_output.prompt == test_output.prompt, \
        #    f"Prompts don't match for output {i}:\nReference: {ref_output.prompt!r}\nTest: {test_output.prompt!r}"

        # Compare number of generated outputs
        assert len(ref_output.outputs) == len(test_output.outputs), \
            f"Number of generated outputs don't match for output {i}: {len(ref_output.outputs)} vs {len(test_output.outputs)}"

        # Compare generated text and other attributes
        for j, (ref_gen, test_gen) in enumerate(
                zip(ref_output.outputs, test_output.outputs)):
            assert ref_gen.text == test_gen.text, \
                f"Generated text doesn't match for output {i}, generation {j}:\nReference: {ref_gen.text!r}\nTest: {test_gen.text!r}"

            # Compare token IDs if available
            if hasattr(ref_gen, 'token_ids') and hasattr(test_gen, 'token_ids'):
                assert ref_gen.token_ids == test_gen.token_ids, \
                    f"Token IDs don't match for output {i}, generation {j}"

            # Compare log probabilities if available
            if hasattr(ref_gen, 'logprobs') and hasattr(test_gen, 'logprobs'):
                assert ref_gen.logprobs == test_gen.logprobs, \
                    f"Log probabilities don't match for output {i}, generation {j}"


@pytest.mark.parametrize("model_dir", [_QWEN_3_VL_DIR], indirect=True)
@pytest.mark.parametrize("pd_disagg", [True], indirect=True)
@pytest.mark.threadleak(enabled=False)
def test_pd_disagg_with_image_input(
    model_dir: Path,
    pd_disagg: bool,
    llms: tuple[LLM, LLM | None],
):
    """Test P/D disagg with image input."""
    llm, llm_decode = llms
    assert llm_decode is not None, "Disaggregated decode worker required."

    prompts = ["Describe the image."]
    media = [example_images[-1]]
    sampling_params = SamplingParams(max_tokens=32, temperature=0)

    # Reference outputs: use desired `max_tokens`.
    inputs = _load_inputs(llm, prompts, media)
    outputs_ref = llm.generate(inputs, sampling_params=sampling_params)
    assert outputs_ref is not None and len(outputs_ref) == len(prompts)

    # Prefill: `max_tokens=0`.
    prefill_disagg_params = DisaggregatedParams(request_type="context_only")
    outputs = llm.generate(inputs,
                           sampling_params=SamplingParams(max_tokens=0,
                                                          temperature=0),
                           disaggregated_params=prefill_disagg_params)
    assert len(outputs) == 1
    pd_disaggregated_params = outputs[0].disaggregated_params
    pd_disaggregated_params.request_type = "generation_only"

    # Decode: use desired `max_tokens`.
    decode_inputs = [{
        "prompt": inputs[0]["prompt"],
        "multi_modal_data": None,
        "prompt_token_ids": outputs[0].prompt_token_ids,
    }]
    outputs_pd = llm_decode.generate(
        decode_inputs,
        sampling_params=sampling_params,
        disaggregated_params=pd_disaggregated_params)

    assert len(outputs_pd) == len(prompts)
    for i, (ref_output, test_output) in enumerate(zip(outputs_ref, outputs_pd)):
        assert len(ref_output.outputs) == len(test_output.outputs), \
            f"Number of generated outputs don't match for output {i}: {len(ref_output.outputs)} vs {len(test_output.outputs)}"
        for j, (ref_gen, test_gen) in enumerate(
                zip(ref_output.outputs, test_output.outputs)):
            assert ref_gen.text == test_gen.text, \
                f"Generated text doesn't match for output {i}, generation {j}:\nReference: {ref_gen.text!r}\nTest: {test_gen.text!r}"


# Explicit combinations instead of product([False, True], [False, True]) to avoid
# having to call `pytest.skip` within the test code itself. This saves on CI time, since `llms`
# take a long time to instantiate.
@pytest.mark.parametrize("use_mm_embeddings,pass_embeddings_through_loader", [
    (False, False),
    (True, False),
    (True, True),
])
@pytest.mark.threadleak(enabled=False)
def test_multi_request_batch_chat(
    model_dir: Path,
    llms: tuple[LLM, LLM | None],
    use_mm_embeddings: bool,
    pass_embeddings_through_loader: bool,
):
    """Test batching multiple multimodal requests and verify encoder path matches raw path.

    This mirrors test_single_image_chat but with a batch of size 3. It also tests passing
    embeddings alongside the prompt ("multi_modal_embeddings"), as well as the embedding
    handling within default_multimodal_input_loader.
    """
    if use_mm_embeddings and (model_dir in [_QWEN_2_5_VL_DIR, _QWEN_3_VL_DIR]
                              or _is_fake_checkpoint(model_dir)):
        pytest.skip("Qwen does not implement attach_multimodal_embeddings")

    # Qwen2.5/3 VL's vision encoder seems to output different embeddings based on this value.
    # The test only passes with this set to 1.
    encoder_max_batch_size = (1 if
                              model_dir in [_QWEN_2_5_VL_DIR, _QWEN_3_VL_DIR]
                              or _is_fake_checkpoint(model_dir) else 3)

    llm, llm_decode = llms
    if llm_decode is not None:
        pytest.skip("Disagg support not implemented in test case")

    # Guard against accidental reintroduction of invalid parameter combinations.
    if pass_embeddings_through_loader and not use_mm_embeddings:
        pytest.skip("Redundant test configuration")

    max_tokens = 64

    prompts = [
        "Describe the natural environment in the image.",
        "Describe the object and weather condition in the image.",
        "Describe the traffic condition on the road in the image.",
    ]
    media = [example_images[0], example_images[1], example_images[2]]

    sampling_params = SamplingParams(max_tokens=max_tokens)

    inputs = _load_inputs(llm, prompts, media)

    # Reference with raw inputs
    outputs_ref = llm.generate(inputs, sampling_params=sampling_params)
    assert outputs_ref is not None and len(outputs_ref) == len(prompts)
    for i, output in enumerate(outputs_ref):
        assert len(
            output.outputs
        ) > 0, f"Reference generation has no output text for input {i}"

    encoder = MultimodalEncoder(model=model_dir,
                                max_batch_size=encoder_max_batch_size,
                                **_get_fake_checkpoint_kwargs(model_dir))
    with encoder:
        # Encoder path
        encoder_outputs = encoder.generate(inputs)
        if use_mm_embeddings:
            for input, encoder_output in zip(inputs, encoder_outputs):
                disagg_params = encoder_output.disaggregated_params
                assert disagg_params is not None
                mm_embed_handles = disagg_params.multimodal_embedding_handles
                assert mm_embed_handles is not None
                # `mm_embed_handles` is list of handles (one per multimodal item).
                # Reconstruct and concatenate all embeddings for this request.
                mm_embeds = [
                    SharedTensorContainer.from_dict(handle).get_local_view()
                    for handle in mm_embed_handles
                ]
                mm_embed = torch.cat(
                    mm_embeds, dim=0) if len(mm_embeds) > 1 else mm_embeds[0]
                input["multi_modal_embeddings"] = {"image": mm_embed}

            if pass_embeddings_through_loader:
                # Test embedding support in default_multimodal_input_loader
                inputs_with_embeddings = _load_inputs(
                    llm,
                    prompts,
                    media=None,
                    mm_embeddings=[
                        input["multi_modal_embeddings"]["image"]
                        for input in inputs
                    ],
                )
                for input, input_with_embedding in zip(inputs,
                                                       inputs_with_embeddings):
                    assert isinstance(input, dict)
                    assert isinstance(input_with_embedding, dict)
                    assert list(
                        set(input.keys())
                        ^ set(input_with_embedding.keys())) == [
                            "multi_modal_data"
                        ]
                    assert set(input_with_embedding.keys()) == set(
                        ["prompt", "multi_modal_embeddings"])
                    assert input["prompt"] == input_with_embedding["prompt"]
                    assert list(
                        input["multi_modal_embeddings"].keys()) == ["image"]
                    assert list(input_with_embedding["multi_modal_embeddings"].
                                keys()) == ["image"]
                    mm_embed, = input_with_embedding["multi_modal_embeddings"][
                        "image"]
                    torch.testing.assert_close(
                        mm_embed, input["multi_modal_embeddings"]["image"])
                inputs = inputs_with_embeddings  # perform inference with embeddings returned by input loader

            extra_kwargs = {}
        else:
            for eo in encoder_outputs:
                eo.disaggregated_params.request_type = "context_and_generation"
            extra_kwargs = dict(disaggregated_params=[
                eo.disaggregated_params for eo in encoder_outputs
            ])
        outputs = llm.generate(inputs,
                               sampling_params=sampling_params,
                               **extra_kwargs)

        assert len(outputs) == len(prompts)
        for i, output in enumerate(outputs):
            assert len(output.outputs
                       ) > 0, f"generation has no output text for input {i}"

        # Compare
        for i, (ref_output, test_output) in enumerate(zip(outputs_ref,
                                                          outputs)):
            assert len(ref_output.outputs) == len(test_output.outputs), \
                f"Number of generated outputs don't match for output {i}: {len(ref_output.outputs)} vs {len(test_output.outputs)}"
            for j, (ref_gen, test_gen) in enumerate(
                    zip(ref_output.outputs, test_output.outputs)):
                assert ref_gen.text == test_gen.text, \
                    f"Generated text doesn't match for output {i}, generation {j}:\nReference: {ref_gen.text!r}\nTest: {test_gen.text!r}"
