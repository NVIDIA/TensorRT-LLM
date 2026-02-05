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


# Test parameter combinations (5 out of 16 possible):
# - disable_overlap_scheduler: Controls scheduler mode (False=overlap enabled)
# - use_cuda_graph: Whether to use CUDA graph capture
# - attn_backend: Attention implementation (TRTLLM or FLASHINFER)
# - max_matching_ngram_size: NGram matching mode (2=fixed size, -1=longest match)
#
# Combinations are pruned because parameters are largely orthogonal:
# - NGram matching logic is independent of attention backend/scheduler/cuda graph
# - Each parameter variation is tested at least once
# - Removed combinations like [FLASHINFER, -1] since ngram mode is tested with TRTLLM
#   and FlashInfer is tested with fixed ngram size
@pytest.mark.parametrize(
    "disable_overlap_scheduler,use_cuda_graph,attn_backend,max_matching_ngram_size",
    [
        [False, False, "TRTLLM", 2],
        [False, False, "TRTLLM", -1],
        [False, False, "FLASHINFER", 2],
        [False, True, "TRTLLM", 2],
        [True, False, "TRTLLM", 2],
    ])
@pytest.mark.high_cuda_memory
def test_llama_ngram(disable_overlap_scheduler: bool, use_cuda_graph: bool,
                     attn_backend: str, max_matching_ngram_size: int):
    """Test NGram speculative decoding correctness and acceptance rate.

    Verifies:
    1. Speculative decoding produces identical results to baseline
    2. NGram drafting produces draft tokens that get accepted
    3. Multi-token acceptance occurs (acceptanceLength > 1)
    """
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if total_mem_gb < 20:
        pytest.skip("Not enough memory to load target model")

    print(f"\nTest config: disable_overlap_scheduler={disable_overlap_scheduler}, "
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

    spec_config = NGramDecodingConfig(
        max_draft_len=max_draft_len,
        max_matching_ngram_size=max_matching_ngram_size,
    )

    # Use prompts that encourage repetitive patterns for better ngram matching
    prompts = [
        "Count from 1 to 50: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, "
        "16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, "
        "34, 35,",
    ]
    sampling_params = SamplingParams(max_tokens=64, ignore_eos=True, temperature=0)

    # Run with speculative decoding
    llm_spec = LLM(**llm_common_config, speculative_config=spec_config)
    results_spec = llm_spec.generate(prompts, sampling_params)
    generated_text_spec = [result.outputs[0].text for result in results_spec]

    # Get spec decoding stats before shutdown
    stats = llm_spec.get_stats(timeout=5)
    iterations_with_spec = []
    for stat in stats:
        if 'specDecodingStats' in stat:
            spec_stats = stat['specDecodingStats']
            if spec_stats.get('numDraftTokens', 0) > 0:
                iterations_with_spec.append(spec_stats)

    llm_spec.shutdown()

    # Run reference without speculative decoding
    llm_ref = LLM(**llm_common_config)
    results_ref = llm_ref.generate(prompts, sampling_params)
    generated_text_ref = [result.outputs[0].text for result in results_ref]
    llm_ref.shutdown()

    # Verify 1: Identical results (correctness)
    for i, (text_spec, text_ref) in enumerate(zip(generated_text_spec,
                                                   generated_text_ref)):
        assert text_spec == text_ref, (
            f"Prompt {i}: Spec decode result differs from baseline.\n"
            f"Spec: {text_spec}\nRef: {text_ref}"
        )
    print(f"Correctness verified: spec decode matches baseline")

    # Verify 2: Spec decoding stats show drafting occurred
    assert len(iterations_with_spec) > 0, (
        f"NGram should have iterations with specDecodingStats. "
        f"Got {len(stats)} total stats but 0 with draft tokens."
    )

    total_draft = sum(s['numDraftTokens'] for s in iterations_with_spec)
    total_accepted = sum(s['numAcceptedTokens'] for s in iterations_with_spec)
    avg_acceptance_len = (sum(s['acceptanceLength'] for s in iterations_with_spec)
                          / len(iterations_with_spec))

    print(f"Spec decoding stats:")
    print(f"  Iterations with drafting: {len(iterations_with_spec)}")
    print(f"  Total draft tokens: {total_draft}")
    print(f"  Total accepted tokens: {total_accepted}")
    print(f"  Average acceptance length: {avg_acceptance_len:.2f}")
    print(f"  Acceptance rate: {total_accepted / total_draft * 100:.1f}%")

    assert total_draft > 0, "NGram should produce draft tokens"
    assert total_accepted > 0, (
        f"NGram should accept some draft tokens. "
        f"Got {total_accepted} accepted out of {total_draft} drafted"
    )

    # Verify 3: Multi-token acceptance (acceptanceLength > 1)
    has_multi_token_acceptance = any(
        s['acceptanceLength'] > 1.0 for s in iterations_with_spec
    )
    print(f"  Has multi-token acceptance: {has_multi_token_acceptance}")

    assert has_multi_token_acceptance, (
        "Expected at least one iteration with acceptanceLength > 1 "
        "for repetitive pattern"
    )


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
    with pytest.raises(ValueError,
                       match="is_public_pool=True is not supported"):
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


if __name__ == "__main__":
    unittest.main()
