import os
import sys
import unittest

import pytest
import torch
from utils.llm_data import llm_models_root

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import (CudaGraphConfig, EagleDecodingConfig,
                                 KvCacheConfig)

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


@pytest.mark.parametrize("use_cuda_graph,attn_backend", [
    [True, "TRTLLM"],
    [False, "TRTLLM"],
])
@pytest.mark.high_cuda_memory
def test_kv_cache_reuse(use_cuda_graph: bool, attn_backend: str):
    # Eagle3 one model works with overlap scheduler and block reuse.
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if total_mem_gb < 35:
        pytest.skip("Not enough memory to load target + draft model")

    models_path = llm_models_root()
    eagle_model_dir = f"{models_path}/EAGLE3-LLaMA3.1-Instruct-8B"
    target_model_dir = f"{models_path}/llama-3.1-model/Llama-3.1-8B-Instruct"

    # bs > 1 gives non-deterministic when doing IFB. There are slight chances
    # that ref and spec does not match 100%
    max_batch_size = 1
    max_draft_len = 4
    kv_cache_config = KvCacheConfig(enable_block_reuse=True, max_tokens=8192)
    cuda_graph_config = CudaGraphConfig(
        batch_sizes=[1]) if use_cuda_graph else None

    llm_common_config = dict(
        model=target_model_dir,
        attn_backend=attn_backend,
        disable_overlap_scheduler=True,
        cuda_graph_config=cuda_graph_config,
        max_batch_size=max_batch_size,
        kv_cache_config=kv_cache_config,
        # This max_seq_len is larger than the one specified
        # in the llama 3 8B eagle's config. We want to make sure
        # that the draft model won't go above its max in warmup
        # in this test.
        max_seq_len=8192,
    )

    spec_config = EagleDecodingConfig(
        max_draft_len=max_draft_len,
        speculative_model_dir=eagle_model_dir,
        eagle3_one_model=False,
    )

    llm_spec = LLM(**llm_common_config, speculative_config=spec_config)

    # Output tests
    prompt = "The future of AI is"

    sampling_params = SamplingParams(max_tokens=10, temperature=0)

    # First run without KV cache
    results = llm_spec.generate(prompt, sampling_params)
    generated_text = results.outputs[0].text

    # Second run with KV cache
    results_kv_cache = llm_spec.generate(prompt, sampling_params)
    generated_text_kv_cache = results_kv_cache.outputs[0].text

    llm_spec.shutdown()

    assert generated_text == generated_text_kv_cache


if __name__ == "__main__":
    unittest.main()
