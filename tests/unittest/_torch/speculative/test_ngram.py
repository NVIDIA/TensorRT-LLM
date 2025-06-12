import os
import sys
import unittest

import pytest
import torch

from tensorrt_llm import SamplingParams
from tensorrt_llm._torch import LLM
from tensorrt_llm.llmapi import KvCacheConfig, NGramDecodingConfig

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.llm_data import llm_models_root


# TODO: add attn_backend="FLASHINFER"
@pytest.mark.parametrize(
    "disable_overlap_scheduler,use_cuda_graph,attn_backend",
    [[True, False, "TRTLLM"], [False, False, "TRTLLM"], [True, True, "TRTLLM"],
     [False, True, "TRTLLM"]])
def test_llama_ngram(disable_overlap_scheduler: bool, use_cuda_graph: bool,
                     attn_backend: str):
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if total_mem_gb < 15:
        pytest.skip("Not enough memory to load target model")

    llm_common_config = dict( \
        model=llm_models_root() / "llama-models" / "llama-7b-hf",
        backend='pytorch',
        attn_backend=attn_backend,
        disable_overlap_scheduler=disable_overlap_scheduler,
        use_cuda_graph=use_cuda_graph,
        max_batch_size=4,
        kv_cache_config=KvCacheConfig(enable_block_reuse=True),
        max_num_tokens=2048,
    )

    spec_config = NGramDecodingConfig(
        prompt_lookup_num_tokens=4,
        max_matching_ngram_size=2,
        is_keep_all=True,
        is_use_oldest=True,
        is_public_pool=True,
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
