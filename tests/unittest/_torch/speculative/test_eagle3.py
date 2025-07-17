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


@pytest.mark.parametrize(
    "use_cuda_graph,attn_backend,disable_overlap_scheduler,enable_block_reuse,use_one_model",
    [
        [True, "TRTLLM", True, False, False],
        [False, "TRTLLM", True, False, False],
        [True, "TRTLLM", True, True, False],
        [False, "TRTLLM", True, True, False],
        [True, "FLASHINFER", True, False, False],
        [False, "FLASHINFER", True, False, False],
        [False, "TRTLLM", False, True, True],
        [True, "TRTLLM", False, True, True],
    ])
@pytest.mark.high_cuda_memory
def test_llama_eagle3(use_cuda_graph: bool, attn_backend: str,
                      disable_overlap_scheduler: bool, enable_block_reuse: bool,
                      use_one_model: bool):
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
    kv_cache_config = KvCacheConfig(enable_block_reuse=enable_block_reuse,
                                    free_gpu_memory_fraction=0.5)
    cuda_graph_config = CudaGraphConfig(
        batch_sizes=[1]) if use_cuda_graph else None

    llm_common_config = dict(
        model=target_model_dir,
        attn_backend=attn_backend,
        disable_overlap_scheduler=disable_overlap_scheduler,
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
        # Llama 3 does not support one model eagle.
        eagle3_one_model=use_one_model,
    )

    llm_spec = LLM(**llm_common_config, speculative_config=spec_config)

    # Acceptance rate tests
    tok_ids = llm_spec.tokenizer.encode("The future of AI is")
    num_tokens = 0
    num_drafted = 0
    num_accepted = 0
    sampling_params = SamplingParams(max_tokens=128, temperature=0)
    for output in llm_spec.generate_async(tok_ids,
                                          sampling_params,
                                          streaming=True):
        new_tokens = output.outputs[0].token_ids
        num_drafted += max_draft_len
        num_accepted += len(new_tokens) - num_tokens - 1
        num_tokens = len(new_tokens)

    accept_rate = num_accepted / num_drafted
    assert accept_rate > 0.15

    # Output tests
    prompts = [
        "The capital of France is",
        "The president of the United States is",
    ]
    sampling_params = SamplingParams(max_tokens=10, temperature=0)

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
