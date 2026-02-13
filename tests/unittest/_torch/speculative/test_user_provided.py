import os
import sys
import unittest

import pytest
import torch

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch.speculative.ngram import NGramDrafter, NGramPoolManager
from tensorrt_llm.llmapi import (CudaGraphConfig, KvCacheConfig,
                                 NGramDecodingConfig,
                                 UserProvidedDecodingConfig)

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.llm_data import llm_models_root


# TODO: add disable_overlap_scheduler=False
@pytest.mark.parametrize(
    "disable_overlap_scheduler,use_cuda_graph,attn_backend",
    [[True, False, "TRTLLM"], [True, True, "TRTLLM"],
     [True, False, "FLASHINFER"]])
def test_llama_user_provided(disable_overlap_scheduler: bool,
                             use_cuda_graph: bool, attn_backend: str):
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if total_mem_gb < 20:
        pytest.skip("Not enough memory to load target model")

    max_batch_size = 2
    max_draft_len = 4
    kv_cache_config = KvCacheConfig(enable_block_reuse=False, max_tokens=8192)
    cuda_graph_config = CudaGraphConfig(
        batch_sizes=[1]) if use_cuda_graph else None

    llm_common_config = dict( \
        model=llm_models_root() / "llama-3.1-model" /"Meta-Llama-3.1-8B",
        backend='pytorch',
        attn_backend=attn_backend,
        disable_overlap_scheduler=disable_overlap_scheduler,
        cuda_graph_config=cuda_graph_config,
        max_batch_size=max_batch_size,
        kv_cache_config=kv_cache_config,
        max_num_tokens=2048,
    )

    ngram_config = NGramDecodingConfig(
        max_draft_len=max_draft_len,
        max_matching_ngram_size=2,
        is_keep_all=True,
        is_use_oldest=True,
        is_public_pool=True,
    )

    ngram_pool_manager = NGramPoolManager(
        spec_config=ngram_config,
        max_num_requests=max_batch_size,
    )

    drafter = NGramDrafter(
        spec_config=ngram_config,
        ngram_pool_manager=ngram_pool_manager,
    )

    spec_config = UserProvidedDecodingConfig(
        max_draft_len=max_draft_len,
        drafter=drafter,
    )

    prompts = [
        "The capital of France is",
        "The president of the United States is",
    ]
    sampling_params = SamplingParams(max_tokens=32)

    llm_spec = LLM(**llm_common_config, speculative_config=spec_config)
    results_spec = llm_spec.generate(prompts, sampling_params)
    generated_text_spec = [result.outputs[0].text for result in results_spec]
    llm_spec.shutdown()

    llm_ref = LLM(**llm_common_config)
    results_ref = llm_ref.generate(prompts, sampling_params)
    generated_text_ref = [result.outputs[0].text for result in results_ref]
    llm_ref.shutdown()

    for text_spec, text_ref in zip(generated_text_spec, generated_text_ref):
        # The spec decode algorithm currently guarantees identical results
        assert text_spec == text_ref


if __name__ == "__main__":
    unittest.main()
