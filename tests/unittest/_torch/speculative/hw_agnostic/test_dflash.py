import os
import sys
import unittest

import pytest
import torch
from utils.llm_data import llm_models_root

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import CudaGraphConfig, DFlashDecodingConfig, KvCacheConfig

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


@pytest.mark.parametrize("disable_overlap_scheduler", [True, False])
def test_dflash(disable_overlap_scheduler: bool):
    """Test DFlash speculative decoding with CUDA graph support.

    This test verifies that DFlash speculative decoding works
    correctly with CUDA graphs and padding enabled.
    """
    attn_backend = "TRTLLM"
    enable_block_reuse = False
    enable_chunked_prefill = False

    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if total_mem_gb < 35:
        pytest.skip("Not enough memory to load target + draft model")

    models_path = llm_models_root()
    dflash_model_dir = f"{models_path}/Qwen3-8B-DFlash-b16"
    target_model_dir = f"{models_path}/Qwen3/Qwen3-8B"

    # Test with 3 requests and max_batch_size=4 to trigger padding
    max_batch_size = 4
    max_draft_len = 4
    kv_cache_config = KvCacheConfig(enable_block_reuse=enable_block_reuse, max_tokens=2048)
    use_cuda_graph = True
    cuda_graph_config = (
        CudaGraphConfig(batch_sizes=[1, 2, 4], enable_padding=True) if use_cuda_graph else None
    )

    llm_common_config = dict(
        model=target_model_dir,
        attn_backend=attn_backend,
        disable_overlap_scheduler=disable_overlap_scheduler,
        cuda_graph_config=cuda_graph_config,
        max_batch_size=max_batch_size,
        kv_cache_config=kv_cache_config,
        max_seq_len=2048,
        enable_chunked_prefill=enable_chunked_prefill,
    )

    spec_config = DFlashDecodingConfig(
        max_draft_len=max_draft_len,
        speculative_model=dflash_model_dir,
    )

    # Create the LLM instance
    llm_spec = LLM(**llm_common_config, speculative_config=spec_config)

    prompts = [
        "The capital of France is",
        "The president of the United States is",
        "The future of AI is",
    ]

    sampling_params = SamplingParams(max_tokens=1024, temperature=0)
    llm_spec.generate(prompts, sampling_params)
    llm_spec.shutdown()


if __name__ == "__main__":
    unittest.main()
