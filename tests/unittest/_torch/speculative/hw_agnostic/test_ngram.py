import unittest

import pytest
import torch
from utils.llm_data import llm_models_root

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig, NGramDecodingConfig


# TODO: add disable_overlap_scheduler=False
@pytest.mark.parametrize(
    "disable_overlap_scheduler,use_cuda_graph,attn_backend",
    [[True, False, "TRTLLM"], [True, True, "TRTLLM"], [True, False, "FLASHINFER"]],
)
@pytest.mark.high_cuda_memory
def test_llama_ngram(disable_overlap_scheduler: bool, use_cuda_graph: bool, attn_backend: str):
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if total_mem_gb < 20:
        pytest.skip("Not enough memory to load target model")

    max_batch_size = 2
    max_draft_len = 4
    kv_cache_config = KvCacheConfig(enable_block_reuse=False, max_tokens=8192)
    cuda_graph_config = CudaGraphConfig(batch_sizes=[1]) if use_cuda_graph else None

    llm_common_config = dict(
        model=llm_models_root() / "llama-3.1-model" / "Meta-Llama-3.1-8B",
        backend="pytorch",
        attn_backend=attn_backend,
        disable_overlap_scheduler=disable_overlap_scheduler,
        cuda_graph_config=cuda_graph_config,
        max_batch_size=max_batch_size,
        kv_cache_config=kv_cache_config,
        max_num_tokens=2048,
    )

    spec_config = NGramDecodingConfig(
        max_draft_len=max_draft_len,
        max_matching_ngram_size=2,
        is_keep_all=True,
        is_use_oldest=True,
        is_public_pool=True,
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


# -- CPU-only pool-logic tests --------------------------------------------------


class _MockConfig:
    def __init__(self, is_keep_all=True, is_use_oldest=True):
        self.tokens_per_gen_step = 6
        self.max_matching_ngram_size = 2
        self.is_keep_all = is_keep_all
        self.is_use_oldest = is_use_oldest
        self.is_public_pool = True


def _make_manager(**kwargs):
    from tensorrt_llm._torch.speculative.ngram import NGramPoolManager

    return NGramPoolManager(_MockConfig(**kwargs), max_num_requests=8)


@pytest.mark.cpu_only
def test_pruning_removes_all_shorter_candidates():
    """A new match must prune every shorter candidate and keep the rest."""
    mgr = _make_manager(is_keep_all=True, is_use_oldest=True)
    # Seed a longer and an equal-length candidate, then three shorter ones.
    mgr.get_draft_tokens([100, 101, 20, 21, 22, 23, 24], 0, 100)
    mgr.get_draft_tokens([100, 101, 8, 9, 10, 11], 1, 100)
    for rid in range(2, 5):
        mgr.get_draft_tokens([100, 101, rid], rid, 100)
    mgr.get_draft_tokens([100, 101, 4, 5, 6, 7], 5, 100)

    pool = mgr.pool if mgr.is_public_pool else mgr.pool[5]
    assert list(pool[(100, 101)]) == [
        (20, 21, 22, 23, 24),
        (8, 9, 10, 11),
        (4, 5, 6, 7),
    ]


@pytest.mark.cpu_only
def test_single_match_mode_keeps_one_entry():
    """is_keep_all=False must retain at most one match per pattern."""
    mgr = _make_manager(is_keep_all=False, is_use_oldest=True)
    mgr.get_draft_tokens([100, 101, 1, 2, 3], 0, 100)
    mgr.get_draft_tokens([100, 101, 1, 2], 1, 100)
    mgr.get_draft_tokens([100, 101, 9], 2, 100)

    pool = mgr.pool if mgr.is_public_pool else mgr.pool[2]
    candidates = list(pool.get((100, 101), []))
    assert len(candidates) <= 1, f"is_keep_all=False but {len(candidates)} candidates in pool"


@pytest.mark.cpu_only
def test_draft_returns_longest_after_pruning():
    """After pruning, the drafter should return the longest remaining match."""
    mgr = _make_manager(is_keep_all=True, is_use_oldest=True)
    mgr.get_draft_tokens([100, 101, 1], 0, 100)
    mgr.get_draft_tokens([100, 101, 4, 5, 6, 7], 1, 100)
    draft = mgr.get_draft_tokens([50, 100, 101], 2, 100)
    assert draft == [4, 5, 6, 7], (
        f"draft returned {draft}, expected [4, 5, 6, 7] from the longer match"
    )


if __name__ == "__main__":
    unittest.main()
