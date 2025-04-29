import json
from pathlib import Path

import pytest
from utils.llm_data import llm_models_root
from utils.util import similar

from tensorrt_llm import SamplingParams
from tensorrt_llm._torch import LLM
from tensorrt_llm._torch.pyexecutor.config import PyTorchConfig
from tensorrt_llm.llmapi import KvCacheConfig as TRT_KvCacheConfig


# A test case of mmlu_llama from lm_eval
@pytest.fixture(scope="module")
def test_case():
    with open(Path(__file__).parent / "test_overlap_scheduler_input.json") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def model_path():
    return llm_models_root() / "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"


def create_llm(model_dir):
    """Create LLM with specific overlap scheduler setting"""
    pytorch_config = PyTorchConfig(use_cuda_graph=True,
                                   enable_trtllm_decoder=True)

    trt_kv_cache_config = TRT_KvCacheConfig(enable_block_reuse=False)

    return LLM(
        model=str(model_dir),
        tensor_parallel_size=1,
        trust_remote_code=True,
        enable_chunked_prefill=True,
        pytorch_backend_config=pytorch_config,
        kv_cache_config=trt_kv_cache_config,
        max_num_tokens=
        128  # Only one request longer than max_num_tokens is required to test chunked prefill
    )


def test_trtllm_decoder(model_path, test_case):
    prompts = [
        "Magellan and Elcano lead the first",
        "The capital of France is",
        "The capital of Bolivia is",
    ]

    expected_outputs = [["circumnavigation of the world."], ["Paris."],
                        ["La Paz."]]

    # Test configuration
    max_new_tokens = test_case["max_new_tokens"]
    temperature = test_case["temperature"]
    top_p = test_case["top_p"]
    stop_words = test_case["stop_words"]

    sampling_config = SamplingParams(max_tokens=max_new_tokens,
                                     beam_width=1,
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
        assert similar(text, expected)
