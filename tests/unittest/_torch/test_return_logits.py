import os
import pickle

import pytest
import torch
from utils.llm_data import llm_models_root
from utils.util import force_ampere

from tensorrt_llm import SamplingParams
from tensorrt_llm._torch import LLM
from tensorrt_llm._torch.pyexecutor.llm_request import LlmResponse, PyResult
from tensorrt_llm.bindings.executor import Response, Result
from tensorrt_llm.executor.result import Logprob
from tensorrt_llm.llmapi.llm_utils import BuildConfig, KvCacheConfig

prompts = ["A B C"]
global_kvcache_config = KvCacheConfig(max_tokens=10000)


def test_LlmResponse_pickle():
    result = Result()
    result.decoding_iter = 1
    result.sequence_index = 1
    binding_response = Response(request_id=1, result=result, client_id=1)
    py_result = PyResult(prompt_len=1,
                         max_new_tokens=1,
                         use_device_memory=True,
                         streaming=False,
                         return_log_probs=True,
                         return_context_logits=True,
                         return_generation_logits=True)
    context_logits = torch.randn([1, 1, 128], device='cuda')
    generation_logits = torch.randn([1, 1, 128], device='cuda')
    logprobs = [[{1: Logprob(0.8, 1)}]]
    py_result.append_context_logits(context_logits)
    py_result.append_generation_logits(generation_logits)
    py_result.append_log_probs(logprobs)

    response = LlmResponse(binding_response, py_result)

    data = pickle.dumps(response)
    pickle_response: LlmResponse = pickle.loads(data)

    assert pickle_response._response.request_id == 1
    assert pickle_response._response.client_id == 1

    pickle_result = pickle_response.result

    assert pickle_result.decoding_iter == 1
    assert pickle_result.sequence_index == 1
    assert torch.all(torch.eq(pickle_result.context_logits, context_logits))
    assert torch.all(
        torch.eq(pickle_result.generation_logits, generation_logits))
    assert pickle_result.log_probs == logprobs


@force_ampere  # Save H100 resource
@pytest.mark.parametrize("return_log_probs", [False, True])
@pytest.mark.parametrize("gather_generation_logits", [False, True])
@pytest.mark.parametrize("gather_context_logits", [False, True])
@pytest.mark.parametrize("enable_trtllm_sampler", [False, True])
@pytest.mark.parametrize("disable_overlap_scheduler", [False, True])
def test_generate_with_return_logits(disable_overlap_scheduler: bool,
                                     enable_trtllm_sampler: bool,
                                     gather_context_logits: bool,
                                     gather_generation_logits: bool,
                                     return_log_probs: bool):
    if not (gather_context_logits or gather_generation_logits
            or return_log_probs):  # prune space
        pytest.skip("Nothing to test")

    if not enable_trtllm_sampler and gather_context_logits:
        pytest.skip("TorchSampler does not support gather_context_logits")

    llm = LLM(
        model=os.path.join(llm_models_root(), "llama-models-v2",
                           "TinyLlama-1.1B-Chat-v1.0"),
        kv_cache_config=global_kvcache_config,
        gather_context_logits=gather_context_logits,
        gather_generation_logits=gather_generation_logits,
        max_batch_size=
        128,  # reduce buffer sizes, specially for generation logits
        enable_trtllm_sampler=enable_trtllm_sampler,
        disable_overlap_scheduler=disable_overlap_scheduler,
    )

    sampling_params = SamplingParams(
        max_tokens=8,
        return_context_logits=gather_context_logits,
        return_generation_logits=gather_generation_logits,
        logprobs=return_log_probs,
    )

    with llm:
        for output in llm.generate(prompts, sampling_params=sampling_params):
            if gather_context_logits:
                assert output.context_logits is not None
                # NOTE: prompt_token_ids of "A B C" becomes [1, 319, 350, 315]
                expected_len = len(prompts[0].split()) + 1
                assert expected_len == output.context_logits.shape[0]
            else:
                assert output.context_logits is None

            if gather_generation_logits:
                gen_logits = output.outputs[0].generation_logits
                assert gen_logits is not None
                assert gen_logits.ndim == 2
                assert gen_logits.shape[0] == sampling_params.max_tokens
                assert torch.argmax(
                    gen_logits, dim=1).tolist() == output.outputs[0].token_ids
            else:
                assert output.outputs[0].generation_logits is None

            if return_log_probs:
                assert len(
                    output.outputs[0].logprobs) == sampling_params.max_tokens
            else:
                assert len(output.outputs[0].logprobs) == 0


@force_ampere  # Save H100 resource
@pytest.mark.parametrize("return_log_probs", [False, True])
@pytest.mark.parametrize("gather_generation_logits", [False, True])
@pytest.mark.parametrize("gather_context_logits", [False, True])
@pytest.mark.parametrize("enable_trtllm_sampler", [False, True])
@pytest.mark.parametrize("disable_overlap_scheduler", [False, True])
def test_generate_async_with_return_logits(disable_overlap_scheduler: bool,
                                           enable_trtllm_sampler: bool,
                                           gather_context_logits: bool,
                                           gather_generation_logits: bool,
                                           return_log_probs: bool):
    if not (gather_context_logits or gather_generation_logits
            or return_log_probs):  # prune space
        pytest.skip("Nothing to test")

    if not enable_trtllm_sampler and gather_context_logits:
        pytest.skip("TorchSampler does not support gather_context_logits")

    build_config = BuildConfig()
    build_config.gather_context_logits = gather_context_logits

    llm = LLM(
        model=os.path.join(llm_models_root(), "llama-models-v2",
                           "TinyLlama-1.1B-Chat-v1.0"),
        kv_cache_config=global_kvcache_config,
        build_config=build_config,
        gather_context_logits=gather_context_logits,
        gather_generation_logits=gather_generation_logits,
        max_batch_size=
        128,  # reduce buffer sizes, specially for generation logits
        enable_trtllm_sampler=enable_trtllm_sampler,
        disable_overlap_scheduler=disable_overlap_scheduler,
    )
    sampling_params = SamplingParams(
        max_tokens=8,
        return_context_logits=gather_context_logits,
        return_generation_logits=gather_generation_logits,
        logprobs=return_log_probs)

    with llm:
        for idx, output in enumerate(
                llm.generate_async(prompts[0],
                                   sampling_params=sampling_params,
                                   streaming=True)):
            if gather_context_logits:
                assert output.context_logits is not None
                # NOTE: prompt_token_ids of "A B C" becomes [1, 319, 350, 315]
                expected_len = len(prompts[0].split()) + 1
                assert expected_len == output.context_logits.shape[0]
            else:
                assert output.context_logits is None

            if gather_generation_logits:
                gen_logits = output.outputs[0].generation_logits
                assert gen_logits is not None
                assert gen_logits.ndim == 2
                assert gen_logits.shape[0] == 1
                assert torch.argmax(
                    gen_logits,
                    dim=1).tolist()[0] == output.outputs[0].token_ids[-1]
            else:
                assert output.outputs[0].generation_logits is None

            if return_log_probs:
                assert len(output.outputs[0].logprobs) == idx + 1
            else:
                assert len(output.outputs[0].logprobs) == 0
