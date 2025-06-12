import os
import sys
import unittest

import pytest
import torch

from tensorrt_llm import SamplingParams
from tensorrt_llm._torch import LLM
from tensorrt_llm.llmapi import DraftTargetDecodingConfig, KvCacheConfig

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.llm_data import llm_models_root


@pytest.mark.parametrize("use_cuda_graph,attn_backend",
                         [[False, "TRTLLM"], [True, "TRTLLM"]])
def test_llama_draft_target(use_cuda_graph: bool, attn_backend: str):
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if total_mem_gb < 60:
        pytest.skip("Not enough memory to load target model")

    models_path = llm_models_root()

    kv_cache_config = KvCacheConfig(enable_block_reuse=False, max_tokens=2080)

    sampling_params = SamplingParams(
        max_tokens=32,
        temperature=0,
    )
    max_batch_size = 1

    target_model_dir = f"{models_path}/llama-3.1-model/Llama-3.1-8B-Instruct"
    draft_model_dir = f"{models_path}/llama-3.1-model/Llama-3.1-8B-Instruct"

    draft_len = 4
    spec_config = DraftTargetDecodingConfig(
        max_draft_len=draft_len, pytorch_weights_path=draft_model_dir)
    llm_spec = LLM(model=target_model_dir,
                   max_batch_size=max_batch_size,
                   disable_overlap_scheduler=True,
                   use_cuda_graph=use_cuda_graph,
                   attn_backend=attn_backend,
                   cuda_graph_batch_sizes=[1],
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
                  disable_overlap_scheduler=True,
                  use_cuda_graph=use_cuda_graph,
                  attn_backend=attn_backend,
                  cuda_graph_batch_sizes=[1],
                  kv_cache_config=kv_cache_config)

    results_ref = llm_ref.generate(prompts, sampling_params)
    generated_text_ref = [result.outputs[0].text for result in results_ref]
    llm_ref.shutdown()

    for text_spec, text_ref in zip(generated_text_spec, generated_text_ref):
        # The spec decode algorithm currently guarantees identical results
        assert text_spec == text_ref


if __name__ == "__main__":
    unittest.main()
