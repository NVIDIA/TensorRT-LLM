import json
import os

import pytest

from tensorrt_llm import MultimodalEncoder
from tensorrt_llm._torch.shared_tensor import SharedTensorContainer
from tensorrt_llm.inputs import default_multimodal_input_loader
from tensorrt_llm.llmapi import KvCacheConfig
from tensorrt_llm.llmapi.llm import LLM, SamplingParams

example_images = [
    "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/seashore.png",
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png",
    "https://huggingface.co/datasets/Sayali9141/traffic_signal_images/resolve/main/61.jpg",
]


@pytest.fixture(scope="function")
def multimodal_model_config():
    """Get multimodal model configuration similar to integration tests"""
    # You can extend this to support multiple models or get from environment
    model_configs = {
        'llava-v1.6-mistral-7b-hf': {
            'model_name': 'llava-v1.6-mistral-7b-hf',
            'hf_model_dir': 'llava-hf/llava-v1.6-mistral-7b-hf',
        }
    }

    return model_configs['llava-v1.6-mistral-7b-hf']


@pytest.mark.parametrize("model_key", [
    "llava-v1.6-mistral-7b-hf",
])
def test_single_image_chat(model_key, multimodal_model_config):
    """Test processing single image using encoder (pass mm_embeddings) + LLM API.

    This test verifies that encoder (pass mm_embeddings) + LLM API produces identical
    results to standard llm generation (pass raw image) by comparing outputs.
    """
    # Get model configuration
    if model_key != "llava-v1.6-mistral-7b-hf":
        #TODO: add more model tests progressively here
        pytest.skip(
            f"Skipping test for {model_key} - only testing llava-v1.6-mistral-7b-hf for now"
        )

    # Extract model information from config
    encoder_model_dir = multimodal_model_config['hf_model_dir']

    # Test configuration
    max_tokens = 64
    free_gpu_memory_fraction = 0.6
    max_batch_size = 1

    # Test data - OpenAI chat completion format
    prompts = ["Describe the natural environment in the image."]
    media = [example_images[0]]

    # Sampling configuration
    sampling_params = SamplingParams(max_tokens=max_tokens)
    kv_cache_config = KvCacheConfig(
        enable_block_reuse=False,
        free_gpu_memory_fraction=free_gpu_memory_fraction,
    )

    # Process multimodal data using encoder (pass mm_embeddings)
    encoder = MultimodalEncoder(model=encoder_model_dir,
                                max_batch_size=max_batch_size)
    llm = LLM(model=encoder_model_dir,
              backend='pytorch',
              kv_cache_config=kv_cache_config,
              trust_remote_code=True)

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
                                             image_data_format="pt")

    # Validate inputs structure
    assert len(inputs) == len(
        prompts), f"Expected {len(prompts)} inputs, got {len(inputs)}"
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
    encoder_outputs = encoder.generate(inputs)
    inputs = default_multimodal_input_loader(
        tokenizer=llm.tokenizer,
        model_dir=llm._hf_model_dir,
        model_type=model_type,
        modality="image",
        prompts=prompts,
        mm_embeddings=[
            SharedTensorContainer.from_dict(
                output.mm_embedding_handle).get_local_view()
            for output in encoder_outputs
        ],
        image_data_format="pt")

    # Generate output using llm (pass mm_embeddings)
    outputs = llm.generate(inputs, sampling_params=sampling_params)

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
        # Compare prompts
        assert ref_output.prompt == test_output.prompt, \
            f"Prompts don't match for output {i}:\nReference: {ref_output.prompt!r}\nTest: {test_output.prompt!r}"

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
