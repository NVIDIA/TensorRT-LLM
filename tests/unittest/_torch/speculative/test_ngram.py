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


# TODO: Add cuda graph enabled tests.
# Cuda graph cannot currently be enabled for ngram because cuda graph requires
# spec metadata and ngram does not have it.
@pytest.mark.parametrize("use_cuda_graph,attn_backend",
                         [[False, "TRTLLM"], [False, "FLASHINFER"]])
def test_llama_ngram(use_cuda_graph: bool, attn_backend: str):
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if total_mem_gb < 31:
        pytest.skip("Not enough memory to load target model")

    models_path = llm_models_root()

    pytorch_config = dict(
        disable_overlap_scheduler=True,
        use_cuda_graph=use_cuda_graph,
        # Only create a single CUDA graph to prevent OOM in CI
        attn_backend=attn_backend,
        cuda_graph_batch_sizes=[1],
    )

    kv_cache_config = KvCacheConfig(enable_block_reuse=False, max_tokens=2080)

    sampling_params = SamplingParams(
        max_tokens=32,
        temperature=0,
    )
    max_batch_size = 1

    target_model_dir = f"{models_path}/llama-models-v2/llama-v2-13b-hf"

    draft_len = 4
    spec_config = NGramDecodingConfig(
        prompt_lookup_num_tokens=draft_len,
        max_matching_ngram_size=draft_len,
        is_keep_all=True,
        is_use_oldest=True,
        is_public_pool=True,
    )
    llm_spec = LLM(model=target_model_dir,
                   max_batch_size=max_batch_size,
                   **pytorch_config,
                   kv_cache_config=kv_cache_config,
                   speculative_config=spec_config)

    prompts = [
        "The capital of France is", "The president of the United States is"
    ]
    results_spec = llm_spec.generate(prompts, sampling_params)
    generated_text_spec = [result.outputs[0].text for result in results_spec]
    llm_spec.shutdown()

    llm_ref = LLM(model=target_model_dir,
                  max_batch_size=max_batch_size,
                  **pytorch_config,
                  kv_cache_config=kv_cache_config)

    results_ref = llm_ref.generate(prompts, sampling_params)
    generated_text_ref = [result.outputs[0].text for result in results_ref]
    llm_ref.shutdown()

    for text_spec, text_ref in zip(generated_text_spec, generated_text_ref):
        # The spec decode algorithm currently guarantees identical results
        assert text_spec == text_ref


if __name__ == "__main__":
    unittest.main()
