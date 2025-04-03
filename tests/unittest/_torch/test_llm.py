import os

import pytest
from utils.llm_data import llm_models_root
from utils.util import force_ampere

from tensorrt_llm import SamplingParams
from tensorrt_llm._torch import LLM
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.llmapi.llm_utils import BuildConfig

prompts = ["A B C"]
global_kvcache_config = KvCacheConfig(max_tokens=256)


@force_ampere  # Save H100 resource
@pytest.mark.parametrize("gather_context_logits", [False])
@pytest.mark.parametrize("gather_generation_logits", [False, True])
@pytest.mark.parametrize("return_log_probs", [False, True])
def test_generate_with_OutputConfig(gather_context_logits: bool,
                                    gather_generation_logits: bool,
                                    return_log_probs: bool):
    if not (gather_context_logits or gather_generation_logits
            or return_log_probs):  # prune space
        return

    build_config = BuildConfig()
    build_config.max_batch_size = 128  # reduce buffer sizes, specially for generation logits
    build_config.gather_context_logits = gather_context_logits

    llm = LLM(
        model=os.path.join(llm_models_root(), "llama-models-v2",
                           "TinyLlama-1.1B-Chat-v1.0"),
        kv_cache_config=global_kvcache_config,
        build_config=build_config,
        gather_generation_logits=gather_generation_logits,
    )
    sampling_params = SamplingParams(
        max_tokens=8,
        return_context_logits=gather_context_logits,
        return_generation_logits=gather_generation_logits,
        return_log_probs=return_log_probs)

    for output in llm.generate(prompts, sampling_params=sampling_params):
        if gather_context_logits:
            assert output.context_logits is not None
            assert len(prompts[0].split()) + \
                   1 == output.context_logits.shape[0]
        else:
            assert output.context_logits is None

        if gather_generation_logits:
            assert output.outputs[0].generation_logits is not None
            assert output.outputs[0].generation_logits.ndim == 2
            assert output.outputs[0].generation_logits.shape[
                0] == sampling_params.max_tokens
        else:
            assert output.outputs[0].generation_logits is None

        if return_log_probs:
            assert len(output.outputs[0].logprobs) == sampling_params.max_tokens
        else:
            assert len(output.outputs[0].logprobs) == 0


@force_ampere  # Save H100 resource
@pytest.mark.parametrize("gather_context_logits", [False])
@pytest.mark.parametrize("gather_generation_logits", [False, True])
@pytest.mark.parametrize("return_log_probs", [False, True])
def test_generate_async_with_OutputConfig(gather_context_logits: bool,
                                          gather_generation_logits: bool,
                                          return_log_probs: bool):
    if not (gather_context_logits or gather_generation_logits
            or return_log_probs):  # prune space
        return

    build_config = BuildConfig()
    build_config.max_batch_size = 128  # reduce buffer sizes, specially for generation logits
    build_config.gather_context_logits = gather_context_logits

    llm = LLM(
        model=os.path.join(llm_models_root(), "llama-models-v2",
                           "TinyLlama-1.1B-Chat-v1.0"),
        kv_cache_config=global_kvcache_config,
        build_config=build_config,
        gather_generation_logits=gather_generation_logits,
    )
    sampling_params = SamplingParams(
        max_tokens=8,
        return_context_logits=gather_context_logits,
        return_generation_logits=gather_generation_logits,
        return_log_probs=return_log_probs)

    for idx, output in enumerate(
            llm.generate_async(prompts[0],
                               sampling_params=sampling_params,
                               streaming=True)):
        if gather_context_logits:
            assert output.context_logits is not None
            assert len(prompts[0].split()) + \
                   1 == output.context_logits.shape[0]
        else:
            assert output.context_logits is None

        if gather_generation_logits:
            assert output.outputs[0].generation_logits is not None
            assert output.outputs[0].generation_logits.ndim == 2
            assert output.outputs[0].generation_logits.shape[0] == 1
        else:
            assert output.outputs[0].generation_logits is None

        if return_log_probs:
            assert len(output.outputs[0].logprobs) == idx + 1
        else:
            assert len(output.outputs[0].logprobs) == 0
