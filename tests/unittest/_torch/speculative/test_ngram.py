import os
import sys
import unittest

import pytest
import torch

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import (CudaGraphConfig, KvCacheConfig,
                                 NGramDecodingConfig)

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.llm_data import llm_models_root


# TODO: add disable_overlap_scheduler=False
@pytest.mark.parametrize(
    "disable_overlap_scheduler,use_cuda_graph,attn_backend",
    [[True, False, "TRTLLM"], [True, True, "TRTLLM"],
     [True, False, "FLASHINFER"]])
@pytest.mark.high_cuda_memory
def test_llama_ngram(disable_overlap_scheduler: bool, use_cuda_graph: bool,
                     attn_backend: str):
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

    # NGram now uses suffix automaton for pattern matching
    # max_matching_ngram_size=2 means try 2-gram, then 1-gram matching
    spec_config = NGramDecodingConfig(
        max_draft_len=max_draft_len,
        max_matching_ngram_size=2,
    )

    prompts = [
        "The capital of France is",
        "The president of the United States is",
    ]
    sampling_params = SamplingParams(max_tokens=32, ignore_eos=True)

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


@pytest.mark.parametrize("max_matching_ngram_size", [2, 4, -1])
def test_ngram_config_validation(max_matching_ngram_size: int):
    """Test NGramDecodingConfig validation."""
    # Valid configuration
    config = NGramDecodingConfig(
        max_draft_len=4,
        max_matching_ngram_size=max_matching_ngram_size,
    )
    assert config.max_matching_ngram_size == max_matching_ngram_size


def test_ngram_config_invalid_zero():
    """Test that max_matching_ngram_size=0 raises error."""
    with pytest.raises(ValueError, match="max_matching_ngram_size must be"):
        NGramDecodingConfig(
            max_draft_len=4,
            max_matching_ngram_size=0,
        )


def test_ngram_config_deprecated_options():
    """Test that deprecated options raise errors."""
    # Test is_public_pool
    with pytest.raises(ValueError, match="is_public_pool=True is not supported"):
        NGramDecodingConfig(
            max_draft_len=4,
            max_matching_ngram_size=2,
            is_public_pool=True,
        )

    # Test is_keep_all
    with pytest.raises(ValueError, match="is_keep_all=True is not supported"):
        NGramDecodingConfig(
            max_draft_len=4,
            max_matching_ngram_size=2,
            is_keep_all=True,
        )

    # Test is_use_oldest
    with pytest.raises(ValueError, match="is_use_oldest=True is not supported"):
        NGramDecodingConfig(
            max_draft_len=4,
            max_matching_ngram_size=2,
            is_use_oldest=True,
        )


@pytest.mark.high_cuda_memory
def test_llama_ngram_longest_match():
    """Test NGram with longest match mode (max_matching_ngram_size=-1)."""
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if total_mem_gb < 20:
        pytest.skip("Not enough memory to load target model")

    max_batch_size = 2
    max_draft_len = 4
    kv_cache_config = KvCacheConfig(enable_block_reuse=False, max_tokens=8192)

    llm_common_config = dict( \
        model=llm_models_root() / "llama-3.1-model" /"Meta-Llama-3.1-8B",
        backend='pytorch',
        disable_overlap_scheduler=True,
        max_batch_size=max_batch_size,
        kv_cache_config=kv_cache_config,
        max_num_tokens=2048,
    )

    # Test longest match mode
    spec_config = NGramDecodingConfig(
        max_draft_len=max_draft_len,
        max_matching_ngram_size=-1,  # Longest match mode
    )

    prompts = [
        "The capital of France is",
        "The president of the United States is",
    ]
    sampling_params = SamplingParams(max_tokens=32, ignore_eos=True)

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
