import os

import pytest
from utils.llm_data import llm_models_root
from utils.util import force_ampere

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig


@pytest.fixture(scope="module")
def input_prompts():
    return [
        "Born in north-east France, Soyer trained as a",
        "The future of AI is",
    ]


@pytest.fixture(scope="module")
def llm_torch(input_prompts):
    return LLM(
        model=os.path.join(llm_models_root(), "llama-models-v2",
                           "TinyLlama-1.1B-Chat-v1.0"),
        kv_cache_config=KvCacheConfig(max_tokens=10000),
        max_batch_size=len(
            input_prompts
        ),  # use small batch size to prevent large buffers from possibly hiding wrong data accesses.
        max_seq_len=32,
        disable_overlap_scheduler=False,
        sampler_type="TorchSampler",
        cuda_graph_config=CudaGraphConfig(batch_sizes=[1, 2, 4, 8],
                                          enable_padding=True),
        enable_mixed_sampler=True,
        max_top_logprobs=4,
    )


@force_ampere  # Save H100 resource
@pytest.mark.parametrize("top_logprobs", [0, 2, 4])
@pytest.mark.parametrize("top_k", [None, 2])
@pytest.mark.parametrize("top_p", [None, 0.5])
@pytest.mark.threadleak(enabled=False)
def test_generate_with_top_logprobs(top_logprobs: int, top_k: int, top_p: float,
                                    llm_torch, input_prompts):
    max_tokens = 8
    is_topk = top_k is not None and top_k > 1
    is_topp = top_p is not None and top_p > 0.0 and not is_topk
    is_valid_setup = (is_topk
                      and top_logprobs <= top_k) or is_topp or top_logprobs == 0
    if is_valid_setup:
        # passing testcases
        sampling_params = SamplingParams(max_tokens=max_tokens,
                                         logprobs=top_logprobs,
                                         top_k=top_k,
                                         top_p=top_p,
                                         temperature=1.0)
        for output in llm_torch.generate(input_prompts,
                                         sampling_params=sampling_params):
            if top_logprobs > 0:
                assert len(output.outputs[0].logprobs) == max_tokens
                assert len(
                    output.outputs[0].logprobs[0]) == top_logprobs or len(
                        output.outputs[0].logprobs[0]) == top_logprobs + 1
    else:
        # expected to fail testcases
        with pytest.raises(ValueError, match="logprobs.*"):
            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                logprobs=top_logprobs,
                top_k=top_k,
                top_p=top_p,
                temperature=1.0,
            )
