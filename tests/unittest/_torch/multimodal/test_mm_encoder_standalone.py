import json
import os
import time
from itertools import product
from pathlib import Path
from typing import Generator

import pytest
import torch
from utils.llm_data import llm_models_root

from tensorrt_llm import MultimodalEncoder
from tensorrt_llm._torch.shared_tensor import SharedTensorContainer
from tensorrt_llm.inputs import default_multimodal_input_loader
from tensorrt_llm.llmapi import (CacheTransceiverConfig, DisaggregatedParams,
                                 KvCacheConfig)
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

    llm = LLM(model=encoder_model_dir,
              backend='pytorch',
              kv_cache_config=kv_cache_config,
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


@pytest.fixture(scope="module",
                params=[_LLAVA_DIR, _QWEN_2_5_VL_DIR, _QWEN_3_VL_DIR],
                ids=["llava_7b", "qwen2.5_3b", "qwen3_2b"])
def model_dir(request) -> Path:
    return request.param


@pytest.fixture(scope="module", params=[False, True])
def pd_disagg(request) -> bool:
    return request.param


@pytest.fixture(scope="module")
def llms(model_dir: Path,
         pd_disagg: bool) -> Generator[tuple[LLM, LLM | None], None, None]:
    """Get LLM for prefill and, if disagg, separate LLM for decode."""
    free_gpu_memory_fraction = 0.2
    disable_overlap_scheduler = pd_disagg
    cache_transceiver_cfg = CacheTransceiverConfig(
        backend="DEFAULT") if pd_disagg else None
    kv_cache_config = KvCacheConfig(
        enable_block_reuse=False,  # Disable for output 1:1 matching check
        free_gpu_memory_fraction=free_gpu_memory_fraction,
    )

    llm = LLM(
        model=model_dir,
        backend='pytorch',
        kv_cache_config=kv_cache_config,
        trust_remote_code=True,
        cache_transceiver_config=cache_transceiver_cfg,
        disable_overlap_scheduler=disable_overlap_scheduler,
        max_batch_size=1,  # fix batch size to reduce non-determinism in tests
    )
    with llm:
        if pd_disagg:
            llm_decode = LLM(
                model=model_dir,
                backend='pytorch',
                kv_cache_config=kv_cache_config,
                trust_remote_code=True,
                cache_transceiver_config=cache_transceiver_cfg,
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


# TODO: Add multi-image in single chat test
@pytest.mark.threadleak(enabled=False)
def test_single_image_chat(
    pd_disagg: bool,
    model_dir: Path,
    llms: tuple[LLM, LLM | None],
):
    """Test processing single image using encoder (pass mm_embeddings) + LLM API.

    This test verifies that encoder (pass mm_embeddings) + LLM API produces identical
    results to standard llm generation (pass raw image) by comparing outputs.
    """
    llm, llm_decode = llms

    # Test configuration
    max_tokens = 64
    max_batch_size = 1

    # Test data - OpenAI chat completion format
    prompts = ["Describe the natural environment in the image."]
    media = [example_images[0]]

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
    encoder = MultimodalEncoder(model=model_dir, max_batch_size=max_batch_size)
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


@pytest.mark.parametrize("use_mm_embeddings,pass_embeddings_through_loader",
                         product([False, True], [False, True]))
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
    if use_mm_embeddings and model_dir in [_QWEN_2_5_VL_DIR, _QWEN_3_VL_DIR]:
        pytest.skip("Qwen does not implement attach_multimodal_embeddings")

    # Qwen2.5/3 VL's vision encoder seems to output different embeddings based on this value.
    # The test only passes with this set to 1.
    encoder_max_batch_size = (1 if model_dir
                              in [_QWEN_2_5_VL_DIR, _QWEN_3_VL_DIR] else 3)

    llm, llm_decode = llms
    if llm_decode is not None:
        pytest.skip("Disagg support not implemented in test case")

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
                                max_batch_size=encoder_max_batch_size)
    with encoder:
        # Encoder path
        encoder_outputs = encoder.generate(inputs)
        if use_mm_embeddings:
            for input, encoder_output in zip(inputs, encoder_outputs):
                mm_embed_handle = encoder_output.mm_embedding_handle
                assert mm_embed_handle is not None
                mm_embed = SharedTensorContainer.from_dict(
                    mm_embed_handle).get_local_view()
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
