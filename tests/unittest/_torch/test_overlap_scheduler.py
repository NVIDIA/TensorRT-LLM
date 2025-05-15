import json
from pathlib import Path

import pytest
from utils.llm_data import llm_models_root

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


def create_llm(model_dir, disable_overlap_scheduler, enable_trtllm_decoder):
    """Create LLM with specific overlap scheduler setting"""
    pytorch_config = PyTorchConfig(
        use_cuda_graph=True,
        disable_overlap_scheduler=disable_overlap_scheduler,
        enable_trtllm_decoder=enable_trtllm_decoder)

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


@pytest.mark.parametrize("enable_trtllm_decoder", [False, True])
def test_overlap_scheduler_consistency(model_path, test_case,
                                       enable_trtllm_decoder):
    # Test configuration
    prompts = test_case["prompts"]
    max_new_tokens = test_case["max_new_tokens"]
    temperature = test_case["temperature"]
    top_p = test_case["top_p"]
    stop_words = test_case["stop_words"] if enable_trtllm_decoder else None

    sampling_config = SamplingParams(max_tokens=max_new_tokens,
                                     stop=stop_words,
                                     temperature=temperature,
                                     top_p=top_p,
                                     n=1,
                                     use_beam_search=True)

    # Test with overlap scheduler enabled
    llm = create_llm(model_path,
                     disable_overlap_scheduler=False,
                     enable_trtllm_decoder=enable_trtllm_decoder)
    outputs_with_overlap = llm.generate(prompts,
                                        sampling_params=sampling_config,
                                        use_tqdm=True)
    texts_with_overlap = [[
        completion.text for completion in request_output.outputs
    ] for request_output in outputs_with_overlap]
    llm.shutdown()

    # Test with overlap scheduler disabled
    llm = create_llm(model_path,
                     disable_overlap_scheduler=True,
                     enable_trtllm_decoder=enable_trtllm_decoder)
    outputs_without_overlap = llm.generate(prompts,
                                           sampling_params=sampling_config,
                                           use_tqdm=True)
    texts_without_overlap = [[
        completion.text for completion in request_output.outputs
    ] for request_output in outputs_without_overlap]
    llm.shutdown()

    # Verify outputs are consistent
    for with_overlap, without_overlap in zip(texts_with_overlap,
                                             texts_without_overlap):
        assert with_overlap == without_overlap


if __name__ == "__main__":
    test_overlap_scheduler_consistency()
