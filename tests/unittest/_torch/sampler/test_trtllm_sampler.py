import pytest
from utils.llm_data import llm_models_root
from utils.util import similar

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import CudaGraphConfig
from tensorrt_llm.llmapi import KvCacheConfig as TRT_KvCacheConfig


@pytest.fixture(scope="module")
def model_path():
    return llm_models_root() / "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"


def create_llm(model_dir):
    """Create LLM with specific overlap scheduler setting"""
    trt_kv_cache_config = TRT_KvCacheConfig(enable_block_reuse=False)

    return LLM(
        model=str(model_dir),
        tensor_parallel_size=1,
        trust_remote_code=True,
        enable_chunked_prefill=True,
        cuda_graph_config=CudaGraphConfig(),
        kv_cache_config=trt_kv_cache_config,
        sampler_type="TRTLLMSampler",
        max_num_tokens=
        128  # Only one request longer than max_num_tokens is required to test chunked prefill
    )


@pytest.mark.high_cuda_memory
def test_trtllm_sampler(model_path):
    prompts = [
        "Magellan and Elcano lead the first",
        "The capital of France is",
        "The capital of Bolivia is",
    ]

    expected_outputs = [["circumnavigation of the world"], ["Paris"],
                        ["La Paz"]]

    # Test configuration
    max_new_tokens = 10
    temperature = 1.0
    top_p = None
    stop_words = ["."]

    sampling_config = SamplingParams(max_tokens=max_new_tokens,
                                     n=1,
                                     stop=stop_words,
                                     temperature=temperature,
                                     top_p=top_p)

    # Test with overlap scheduler disabled
    llm = create_llm(model_path)
    outputs = llm.generate(prompts,
                           sampling_params=sampling_config,
                           use_tqdm=True)
    texts = [[completion.text for completion in request_output.outputs]
             for request_output in outputs]
    llm.shutdown()

    # Remove any text after \n\n, consider texts is a list of list of strings
    texts = [[text.split('\n\n')[0] for text in request_output]
             for request_output in texts]

    # Verify outputs are consistent
    for text, expected in zip(texts, expected_outputs):
        assert similar(text, expected), f"text: {text}, expected: {expected}"
