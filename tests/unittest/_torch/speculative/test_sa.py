import os
import sys
import unittest

import pytest
import torch

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig, SADecodingConfig

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.llm_data import llm_models_root


# Test parameter combinations:
# - disable_overlap_scheduler: Controls scheduler mode (False=overlap enabled)
# - use_cuda_graph: Whether to use CUDA graph capture
# - attn_backend: Attention implementation (TRTLLM only - FLASHINFER not supported)
# - max_matching_ngram_size: SA matching mode (2=fixed size, -1=longest match)
#
# NOTE: FLASHINFER backend is NOT supported for one-engine speculative decoding modes
# (SA, MTP, Eagle3-one-model). The FLASHINFER backend's decode path expects exactly
# 1 token per sequence, but one-engine speculative decoding requires processing multiple
# tokens per generation sequence (last_accepted + draft_tokens). This is a fundamental
# architectural limitation of the FLASHINFER integration that would require significant
# changes to support multi-token generation sequences.
@pytest.mark.parametrize(
    "disable_overlap_scheduler,use_cuda_graph,attn_backend,max_matching_ngram_size",
    [
        [False, False, "TRTLLM", 2],
        [False, True, "TRTLLM", 2],
        [True, False, "TRTLLM", 2],
        [True, True, "TRTLLM", 2],
        [False, False, "TRTLLM", -1],
    ])
@pytest.mark.high_cuda_memory
def test_llama_sa(disable_overlap_scheduler: bool, use_cuda_graph: bool,
                  attn_backend: str, max_matching_ngram_size: int):
    """Test SA (Suffix Automaton) speculative decoding acceptance rate.

    Verifies:
    1. SA drafting produces draft tokens that get accepted
    2. Multi-token acceptance occurs (acceptanceLength > 1)

    Output correctness is validated by integration accuracy tests in
    tests/integration/defs/accuracy/test_llm_api_pytorch.py.
    """
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if total_mem_gb < 20:
        pytest.skip("Not enough memory to load target model")

    print(
        f"\nTest config: disable_overlap_scheduler={disable_overlap_scheduler}, "
        f"use_cuda_graph={use_cuda_graph}, attn_backend={attn_backend}, "
        f"max_matching_ngram_size={max_matching_ngram_size}")

    max_batch_size = 1
    max_draft_len = 4
    kv_cache_config = KvCacheConfig(enable_block_reuse=False, max_tokens=8192)
    cuda_graph_config = CudaGraphConfig(
        batch_sizes=[1]) if use_cuda_graph else None

    llm_common_config = dict(
        model=llm_models_root() / "llama-3.1-model" / "Meta-Llama-3.1-8B",
        backend='pytorch',
        attn_backend=attn_backend,
        disable_overlap_scheduler=disable_overlap_scheduler,
        cuda_graph_config=cuda_graph_config,
        max_batch_size=max_batch_size,
        kv_cache_config=kv_cache_config,
        max_num_tokens=2048,
        enable_iter_perf_stats=True,
    )

    spec_config = SADecodingConfig(
        max_draft_len=max_draft_len,
        max_matching_ngram_size=max_matching_ngram_size,
    )

    # Use prompts that encourage repetitive patterns for better SA/ngram matching
    prompts = [
        "Count from 1 to 50: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, "
        "16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, "
        "34, 35,",
    ]
    sampling_params = SamplingParams(max_tokens=64,
                                     ignore_eos=True,
                                     temperature=0)

    llm_spec = LLM(**llm_common_config, speculative_config=spec_config)
    llm_spec.generate(prompts, sampling_params)

    stats = llm_spec.get_stats(timeout=5)
    iterations_with_spec = []
    for stat in stats:
        if 'specDecodingStats' in stat:
            spec_stats = stat['specDecodingStats']
            if spec_stats.get('numDraftTokens', 0) > 0:
                iterations_with_spec.append(spec_stats)

    llm_spec.shutdown()

    # Verify 1: Spec decoding stats show drafting occurred
    assert len(iterations_with_spec) > 0, (
        f"SA should have iterations with specDecodingStats. "
        f"Got {len(stats)} total stats but 0 with draft tokens.")

    total_draft = sum(s['numDraftTokens'] for s in iterations_with_spec)
    total_accepted = sum(s['numAcceptedTokens'] for s in iterations_with_spec)
    avg_acceptance_len = (sum(s['acceptanceLength']
                              for s in iterations_with_spec) /
                          len(iterations_with_spec))

    print(f"Spec decoding stats:")
    print(f"  Iterations with drafting: {len(iterations_with_spec)}")
    print(f"  Total draft tokens: {total_draft}")
    print(f"  Total accepted tokens: {total_accepted}")
    print(f"  Average acceptance length: {avg_acceptance_len:.2f}")
    print(f"  Acceptance rate: {total_accepted / total_draft * 100:.1f}%")

    assert total_draft > 0, "SA should produce draft tokens"
    assert total_accepted > 0, (
        f"SA should accept some draft tokens. "
        f"Got {total_accepted} accepted out of {total_draft} drafted")

    # Verify 2: Multi-token acceptance (acceptanceLength > 1)
    has_multi_token_acceptance = any(s['acceptanceLength'] > 1.0
                                     for s in iterations_with_spec)
    print(f"  Has multi-token acceptance: {has_multi_token_acceptance}")

    assert has_multi_token_acceptance, (
        "Expected at least one iteration with acceptanceLength > 1 "
        "for repetitive pattern")

    torch.cuda.synchronize()


@pytest.mark.parametrize("max_matching_ngram_size", [2, 4, -1])
def test_sa_config_validation(max_matching_ngram_size: int):
    """Test SADecodingConfig validation."""
    # Valid configuration
    config = SADecodingConfig(
        max_draft_len=4,
        max_matching_ngram_size=max_matching_ngram_size,
    )
    assert config.max_matching_ngram_size == max_matching_ngram_size


def test_sa_config_invalid_zero():
    """Test that max_matching_ngram_size=0 raises error for SA."""
    with pytest.raises(ValueError, match="max_matching_ngram_size must be"):
        SADecodingConfig(
            max_draft_len=4,
            max_matching_ngram_size=0,
        )


def test_sa_config_global_pool():
    """Test SADecodingConfig with enable_global_pool."""
    config = SADecodingConfig(
        max_draft_len=4,
        enable_global_pool=True,
    )
    assert config.enable_global_pool is True

    config_off = SADecodingConfig(
        max_draft_len=4,
        enable_global_pool=False,
    )
    assert config_off.enable_global_pool is False

    # Default should be False
    config_default = SADecodingConfig(max_draft_len=4)
    assert config_default.enable_global_pool is False


@pytest.mark.parametrize("disable_overlap_scheduler,use_cuda_graph", [
    [False, False],
    [False, True],
])
@pytest.mark.high_cuda_memory
def test_llama_sa_global_pool(disable_overlap_scheduler: bool,
                              use_cuda_graph: bool):
    """Test SA speculative decoding with global pool enabled.

    Verifies that SA drafting with global pool produces draft tokens that
    get accepted. Output correctness is validated by integration accuracy
    tests in tests/integration/defs/accuracy/test_llm_api_pytorch.py.
    """
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if total_mem_gb < 20:
        pytest.skip("Not enough memory to load target model")

    print(
        f"\nTest config: disable_overlap_scheduler={disable_overlap_scheduler}, "
        f"use_cuda_graph={use_cuda_graph}, enable_global_pool=True")

    max_batch_size = 2
    max_draft_len = 4
    kv_cache_config = KvCacheConfig(enable_block_reuse=False, max_tokens=8192)
    cuda_graph_config = CudaGraphConfig(
        batch_sizes=[1, 2]) if use_cuda_graph else None

    llm_common_config = dict(
        model=llm_models_root() / "llama-3.1-model" / "Meta-Llama-3.1-8B",
        backend='pytorch',
        attn_backend='TRTLLM',
        disable_overlap_scheduler=disable_overlap_scheduler,
        cuda_graph_config=cuda_graph_config,
        max_batch_size=max_batch_size,
        kv_cache_config=kv_cache_config,
        max_num_tokens=2048,
        enable_iter_perf_stats=True,
    )

    spec_config = SADecodingConfig(
        max_draft_len=max_draft_len,
        max_matching_ngram_size=-1,
        enable_global_pool=True,
    )

    # Use two prompts with similar patterns so global pool can help
    prompts = [
        "Count from 1 to 50: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, "
        "14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,",
        "Count from 1 to 50: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, "
        "14, 15, 16, 17, 18, 19, 20, 21, 22, 23,",
    ]
    sampling_params = SamplingParams(max_tokens=64,
                                     ignore_eos=True,
                                     temperature=0)

    llm_spec = LLM(**llm_common_config, speculative_config=spec_config)
    llm_spec.generate(prompts, sampling_params)

    stats = llm_spec.get_stats(timeout=5)
    iterations_with_spec = []
    for stat in stats:
        if 'specDecodingStats' in stat:
            spec_stats = stat['specDecodingStats']
            if spec_stats.get('numDraftTokens', 0) > 0:
                iterations_with_spec.append(spec_stats)

    llm_spec.shutdown()

    # Verify 1: Spec decoding stats show drafting occurred
    assert len(iterations_with_spec) > 0, (
        "SA global pool should have iterations with specDecodingStats.")

    total_draft = sum(s['numDraftTokens'] for s in iterations_with_spec)
    total_accepted = sum(s['numAcceptedTokens'] for s in iterations_with_spec)

    print(f"Global pool spec decoding stats:")
    print(f"  Iterations with drafting: {len(iterations_with_spec)}")
    print(f"  Total draft tokens: {total_draft}")
    print(f"  Total accepted tokens: {total_accepted}")

    assert total_draft > 0, "SA global pool should produce draft tokens"
    assert total_accepted > 0, "SA global pool should accept some draft tokens"

    torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()
