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
@pytest.mark.parametrize("top_logprobs", [None, 0, 2, 4])
@pytest.mark.parametrize("top_k", [None, 2])
@pytest.mark.parametrize("top_p", [None, 0.5])
@pytest.mark.threadleak(enabled=False)
def test_generate_with_top_logprobs(top_logprobs: int, top_k: int, top_p: float,
                                    llm_torch, input_prompts):
    max_tokens = 8
    is_topk = top_k is not None and top_k > 1
    is_topp = top_p is not None and top_p > 0.0 and not is_topk
    uses_top_logprobs = top_logprobs is not None
    is_valid_top_logprobs = top_logprobs > 0 if uses_top_logprobs else True
    is_valid_setup = not uses_top_logprobs or (is_valid_top_logprobs and (
        (is_topk and top_logprobs <= top_k) or is_topp))
    if is_valid_setup:
        # passing testcases
        sampling_params = SamplingParams(max_tokens=max_tokens,
                                         logprobs=True,
                                         top_k=top_k,
                                         top_p=top_p,
                                         temperature=1.0,
                                         top_logprobs=top_logprobs)
        for output in llm_torch.generate(input_prompts,
                                         sampling_params=sampling_params):
            if top_logprobs is not None and top_logprobs > 0:
                assert len(output.outputs[0].logprobs) == max_tokens
                assert len(output.outputs[0].top_logprobs) == max_tokens
                assert len(output.outputs[0].logprobs[0]) == 1
                assert len(output.outputs[0].top_logprobs[0]) == top_logprobs
                sampled_token = list(output.outputs[0].logprobs[0].keys())[0]
                sampled_token_logprob = output.outputs[0].logprobs[0][
                    sampled_token].logprob
                if sampled_token in output.outputs[0].top_logprobs[0]:
                    assert sampled_token_logprob == output.outputs[
                        0].top_logprobs[0][sampled_token].logprob
                else:
                    assert sampled_token_logprob <= list(
                        output.outputs[0].top_logprobs[0].values())[-1].logprob
    else:
        # expected to fail testcases
        with pytest.raises(ValueError, match="top_logprobs.*"):
            sampling_params = SamplingParams(max_tokens=max_tokens,
                                             logprobs=True,
                                             top_k=top_k,
                                             top_p=top_p,
                                             temperature=1.0,
                                             top_logprobs=top_logprobs)


@force_ampere  # Save H100 resource
@pytest.mark.threadleak(enabled=False)
def test_generate_with_top_logprobs_and_disabled_logprobs(
        llm_torch, input_prompts):
    max_tokens = 8
    top_logprobs = 2
    top_k = 2
    top_p = None

    # expected to fail testcases
    with pytest.raises(
            ValueError,
            match=".*You need to set logprobs to True to use top_logprobs.*"):
        sampling_params = SamplingParams(max_tokens=max_tokens,
                                         logprobs=False,
                                         top_k=top_k,
                                         top_p=top_p,
                                         temperature=1.0,
                                         top_logprobs=top_logprobs)
