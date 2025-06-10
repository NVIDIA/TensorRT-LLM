import os
import pytest
import copy
import json

from tensorrt_llm.executor.multimodal.request import MultimodalRequest
from tensorrt_llm._torch.multimodal.mm_encoder import MultimodalEncoder
from tensorrt_llm.llmapi.llm import LLM, SamplingParams
from tensorrt_llm.llmapi import KvCacheConfig
from tensorrt_llm.inputs import default_multimodal_input_loader

example_images = [
    "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/seashore.png",
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png",
    "https://huggingface.co/datasets/Sayali9141/traffic_signal_images/resolve/main/61.jpg",
]


@pytest.mark.parametrize("encoder_model_dir", [
    "llava-hf/llava-v1.6-mistral-7b-hf",
])
def test_single_image_chat(encoder_model_dir):
    """Test processing single image using disaggregated encoder + LLM API.
    
    This test verifies that disaggregated multimodal generation produces identical
    results to standard multimodal generation by comparing outputs.
    """
    # Test configuration
    max_tokens = 64
    free_gpu_memory_fraction = 0.6
    max_batch_size = 1
    
    # Test data - OpenAI chat completion format
    prompts = ["Describe the natural environment in the image."]
    media = [example_images[0]]
    
    # Create OpenAI chat messages format
    messages_list = []
    for prompt, image_url in zip(prompts, media):
        messages = [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": prompt
            }, {
                "type": "image_url",
                "image_url": {
                    "url": image_url
                }
            }]
        }]
        messages_list.append(messages)
    
    # Sampling configuration
    sampling_params = SamplingParams(max_tokens=max_tokens)
    kv_cache_config = KvCacheConfig(
        enable_block_reuse=False,
        free_gpu_memory_fraction=free_gpu_memory_fraction,
    )
    
    # Step 1: Process multimodal data using disaggregated encoder
    encoder = None
    llm = None
    
    try:
        encoder = MultimodalEncoder(model=encoder_model_dir, max_batch_size=max_batch_size)
        
        # Process all messages through the encoder
        multimodal_requests = [MultimodalRequest.from_chat_messages(msgs) for msgs in messages_list]
        results = encoder.generate_from_mm_request(multimodal_requests)
        
        # Validate encoder output
        assert results is not None, "Encoder returned None results"
        assert len(results) == len(prompts), f"Expected {len(prompts)} results, got {len(results)}"

        mm_params_list = []
        for i, result in enumerate(results):
            mm_params = result.multimodal_params
            assert mm_params is not None, f"Multimodal parameters are None for request {i}"
            assert hasattr(mm_params, 'embeddings'), f"Multimodal parameters missing embeddings attribute for request {i}"
            assert mm_params.num_items > 0, f"Expected multimodal items > 0 for request {i}, got {mm_params.num_items}"
            mm_params_list.append(mm_params)
        
        # Step 2: Initialize LLM and prepare inputs
        llm = LLM(
            model=encoder_model_dir, 
            backend='pytorch', 
            kv_cache_config=kv_cache_config, 
            trust_remote_code=True
        )
        
        # Load model configuration
        config_path = os.path.join(llm._hf_model_dir, 'config.json')
        assert os.path.exists(config_path), f"Model config not found at {config_path}"
        
        with open(config_path, 'r') as f:
            model_config = json.load(f)
        model_type = model_config['model_type']
        
        # Prepare multimodal inputs
        inputs = default_multimodal_input_loader(
            tokenizer=llm.tokenizer,
            model_dir=llm._hf_model_dir,
            model_type=model_type,
            modality="image",
            prompts=prompts,  
            media=media,      
            image_data_format="pt"
        )
        
        # Validate inputs structure
        assert len(inputs) == len(prompts), f"Expected {len(prompts)} inputs, got {len(inputs)}"
        # Step 3: Generate reference output with raw multimodal inputs
        outputs_ref = llm.generate(inputs, sampling_params=sampling_params)
        
        # Validate reference outputs
        assert outputs_ref is not None, "Reference generation returned None"
        assert len(outputs_ref) == len(prompts), f"Expected {len(prompts)} reference outputs, got {len(outputs_ref)}"
        for i, output in enumerate(outputs_ref):
            assert len(output.outputs) > 0, f"Reference generation has no output text for input {i}"
        
        # Step 4: Prepare inputs for disaggregated multimodal generation
        inputs_disagg = copy.deepcopy(inputs)
        for i, input_data in enumerate(inputs_disagg):
            # disaggregated generation doesn't need raw image data, but keep the key
            input_data["multi_modal_data"]["image"] = []
        
        # Step 5: Generate output using disaggregated multimodal parameters
        # Note: For batch processing, we need to match mm_params with inputs
        outputs = llm.generate(inputs_disagg, sampling_params=sampling_params, disagg_mm_params=mm_params_list)
        
        # Validate disaggregated outputs
        assert outputs is not None, "Disaggregated generation returned None"
        assert len(outputs) == len(prompts), f"Expected {len(prompts)} disaggregated outputs, got {len(outputs)}"
        for i, output in enumerate(outputs):
            assert len(output.outputs) > 0, f"Disaggregated generation has no output text for input {i}"
        
        # Step 6: Compare outputs - they should match exactly
        assert len(outputs_ref) == len(outputs), f"Number of outputs don't match: {len(outputs_ref)} vs {len(outputs)}"
        
        for i, (ref_output, test_output) in enumerate(zip(outputs_ref, outputs)):
            # Compare prompts
            assert ref_output.prompt == test_output.prompt, \
                f"Prompts don't match for output {i}:\nReference: {ref_output.prompt!r}\nTest: {test_output.prompt!r}"
            
            # Compare number of generated outputs
            assert len(ref_output.outputs) == len(test_output.outputs), \
                f"Number of generated outputs don't match for output {i}: {len(ref_output.outputs)} vs {len(test_output.outputs)}"
            
            # Compare generated text and other attributes
            for j, (ref_gen, test_gen) in enumerate(zip(ref_output.outputs, test_output.outputs)):
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
    finally:
        # Cleanup resources
        if encoder is not None:
            del encoder
        if llm is not None:
            del llm
    
