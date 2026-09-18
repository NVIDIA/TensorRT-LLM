import types
import unittest

import pytest
import torch
from utils.llm_data import llm_models_root

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.speculative.interface import SpeculativeDecodingMode
from tensorrt_llm._torch.speculative.ngram import NGramPoolManager
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig, NGramDecodingConfig


# Test parameter combinations:
# - disable_overlap_scheduler: NGram drafts inside the target forward, so it
#   supports the overlap scheduler (False = overlap enabled).
# - use_cuda_graph: the drafting kernel is captured together with the target.
# - is_public_pool: search all in-flight histories vs. only the request's own.
@pytest.mark.parametrize(
    "disable_overlap_scheduler,use_cuda_graph,attn_backend,is_public_pool",
    [
        [False, False, "TRTLLM", True],
        [False, True, "TRTLLM", True],
        [True, False, "TRTLLM", False],
        [True, True, "TRTLLM", False],
        [True, False, "FLASHINFER", True],
    ],
)
@pytest.mark.high_cuda_memory
def test_llama_ngram(
    disable_overlap_scheduler: bool, use_cuda_graph: bool, attn_backend: str, is_public_pool: bool
):
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if total_mem_gb < 20:
        pytest.skip("Not enough memory to load target model")

    max_batch_size = 2
    max_draft_len = 4
    kv_cache_config = KvCacheConfig(enable_block_reuse=False, max_tokens=8192)
    cuda_graph_config = CudaGraphConfig(batch_sizes=[1, 2]) if use_cuda_graph else None

    llm_common_config = dict(
        model=llm_models_root() / "llama-3.1-model" / "Meta-Llama-3.1-8B",
        backend="pytorch",
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
        max_matching_ngram_size=2,
        is_use_oldest=True,
        is_public_pool=is_public_pool,
    )

    # Repetitive prompts so the suffix lookup finds continuations to draft.
    prompts = [
        "Count from 1 to 50: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, "
        "16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, "
        "34, 35,",
        "The capital of France is",
    ]
    sampling_params = SamplingParams(max_tokens=64, ignore_eos=True, temperature=0)

    llm_spec = LLM(**llm_common_config, speculative_config=spec_config)
    results_spec = llm_spec.generate(prompts, sampling_params)
    generated_text_spec = [result.outputs[0].text for result in results_spec]

    stats = llm_spec.get_stats(timeout=5)
    iterations_with_spec = [
        stat["specDecodingStats"]
        for stat in stats
        if stat.get("specDecodingStats", {}).get("numDraftTokens", 0) > 0
    ]
    llm_spec.shutdown()

    llm_ref = LLM(**llm_common_config)
    results_ref = llm_ref.generate(prompts, sampling_params)
    generated_text_ref = [result.outputs[0].text for result in results_ref]
    llm_ref.shutdown()

    # Greedy verification guarantees identical results.
    for text_spec, text_ref in zip(generated_text_spec, generated_text_ref):
        assert text_spec == text_ref

    # The counting prompt must have produced accepted drafts.
    assert len(iterations_with_spec) > 0, "NGram should report iterations with draft tokens"
    total_accepted = sum(s["numAcceptedTokens"] for s in iterations_with_spec)
    assert total_accepted > 0, "NGram should accept some draft tokens on a repetitive prompt"
    assert any(s["acceptanceLength"] > 1.0 for s in iterations_with_spec)

    torch.cuda.synchronize()


@pytest.mark.cpu_only
def test_ngram_is_a_one_engine_mode():
    """NGram drafts inside the target forward: no host drafter, overlap and CUDA graph friendly."""
    mode = NGramDecodingConfig(max_draft_len=2).spec_dec_mode
    assert mode == SpeculativeDecodingMode.NGRAM
    assert mode.use_one_engine()
    assert mode.is_retrieval_drafter()
    assert mode.support_overlap_scheduler()
    assert mode.support_capturable_guided_decoder()
    assert not mode.has_spec_drafter()
    assert mode.needs_kv_cache_rewind()


class _FakeRequest:
    """Minimal request stand-in for NGramPoolManager bookkeeping tests."""

    def __init__(
        self,
        request_id: int,
        tokens: list,
        *,
        generation_only: bool = False,
        is_dummy: bool = False,
        is_first_context_chunk: bool = True,
    ):
        self.request_id = request_id
        self._tokens = list(tokens)
        self.is_generation_only_request = generation_only
        self.is_dummy = is_dummy
        self.is_first_context_chunk = is_first_context_chunk

    def get_tokens(self, beam: int) -> list:
        assert beam == 0
        return list(self._tokens)


def _make_manager(max_num_requests: int = 4, max_seq_len: int = 32) -> NGramPoolManager:
    config = types.SimpleNamespace(
        max_draft_len=3, max_matching_ngram_size=2, is_use_oldest=True, is_public_pool=True
    )
    return NGramPoolManager(config, max_num_requests, max_seq_len)


def _history(manager: NGramPoolManager, slot: int) -> list:
    torch.cuda.synchronize()
    length = manager.history_lens[slot].item()
    return manager.history_tokens[slot, :length].tolist()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NGram token histories live on CUDA")
class TestNGramPoolManagerBookkeeping:
    def test_first_context_chunk_seeds_the_prompt(self):
        manager = _make_manager()
        try:
            req = _FakeRequest(7, [1, 2, 3, 1, 2])
            batch = ScheduledRequests()
            batch.context_requests_last_chunk = [req]
            manager.prepare_resources(batch)

            slot = manager.slot_manager.get_slot(7)
            assert slot is not None
            assert _history(manager, slot) == [1, 2, 3, 1, 2]

            # Subsequent iterations (the request now in generation) must not reseed.
            gen_batch = ScheduledRequests()
            gen_batch.generation_requests = [req]
            manager.prepare_resources(gen_batch)
            assert manager.slot_manager.get_slot(7) == slot
        finally:
            manager.shutdown()

    def test_chunking_context_rows_are_masked_and_dummies_map_to_the_dummy_slot(self):
        manager = _make_manager()
        try:
            chunking = _FakeRequest(1, [5, 6, 7, 8], is_first_context_chunk=True)
            last = _FakeRequest(2, [9, 9, 9])
            batch = ScheduledRequests()
            batch.context_requests_chunking = [chunking]
            batch.context_requests_last_chunk = [last]
            manager.prepare_resources(batch)
            manager.add_dummy_requests([99])

            slot_ids = torch.zeros(4, dtype=torch.int32, device="cuda")
            row_mask = torch.zeros(4, dtype=torch.int32, device="cuda")
            manager.prepare([1, 2, 99], slot_ids, row_mask)
            torch.cuda.synchronize()

            assert slot_ids[:3].tolist() == [
                manager.slot_manager.get_slot(1),
                manager.slot_manager.get_slot(2),
                manager.dummy_slot,
            ]
            assert row_mask[:3].tolist() == [0, 1, 0]
            # The first chunk still seeds the full prompt even though the row is masked.
            assert _history(manager, manager.slot_manager.get_slot(1)) == [5, 6, 7, 8]

            # Once the request reaches its last chunk it is extended again.
            batch = ScheduledRequests()
            batch.context_requests_last_chunk = [chunking]
            manager.prepare_resources(batch)
            manager.prepare([1], slot_ids, row_mask)
            torch.cuda.synchronize()
            assert row_mask[:1].tolist() == [1]
        finally:
            manager.shutdown()

    def test_disagg_gen_init_is_seeded_from_the_generation_schedule(self):
        manager = _make_manager()
        try:
            prompt = [1, 2, 3, 4]
            req = _FakeRequest(11, prompt, generation_only=True)

            # _prepare_disagg_gen_resources presents the request as a context request before the
            # context server's first token has arrived: nothing may be seeded yet.
            init_batch = ScheduledRequests()
            init_batch.context_requests_last_chunk = [req]
            manager.prepare_resources(init_batch)
            assert manager.slot_manager.get_slot(11) is None

            req._tokens.append(6)
            gen_batch = ScheduledRequests()
            gen_batch.generation_requests = [req]
            manager.prepare_resources(gen_batch)
            slot = manager.slot_manager.get_slot(11)
            assert _history(manager, slot) == prompt + [6]

            dummy_gen = _FakeRequest(12, [1], generation_only=True, is_dummy=True)
            gen_batch.generation_requests = [dummy_gen]
            manager.prepare_resources(gen_batch)
            assert manager.slot_manager.get_slot(12) is None
        finally:
            manager.shutdown()

    def test_free_resources_clears_the_history_and_recycles_the_slot(self):
        manager = _make_manager(max_num_requests=1)
        try:
            req = _FakeRequest(3, [4, 4, 4])
            batch = ScheduledRequests()
            batch.context_requests_last_chunk = [req]
            manager.prepare_resources(batch)
            slot = manager.slot_manager.get_slot(3)

            manager.free_resources(req)
            assert manager.slot_manager.get_slot(3) is None
            assert _history(manager, slot) == []

            other = _FakeRequest(4, [8, 9])
            batch.context_requests_last_chunk = [other]
            manager.prepare_resources(batch)
            assert manager.slot_manager.get_slot(4) == slot
            assert _history(manager, slot) == [8, 9]

            manager.add_dummy_requests([99])
            manager.free_resources(_FakeRequest(99, []))
            assert 99 not in manager._dummy_request_ids
        finally:
            manager.shutdown()

    def test_extend_and_draft_round_trip(self):
        manager = _make_manager(max_num_requests=2)
        try:
            req = _FakeRequest(5, [1, 2, 3, 1, 2, 3, 1])
            batch = ScheduledRequests()
            batch.context_requests_last_chunk = [req]
            manager.prepare_resources(batch)

            slot_ids = torch.zeros(2, dtype=torch.int32, device="cuda")
            row_mask = torch.zeros(2, dtype=torch.int32, device="cuda")
            manager.prepare([5], slot_ids, row_mask)

            # The context step accepts one token (2): the suffix [1, 2] first occurs at the start
            # and is followed by 3, 1, 2.
            accepted = torch.tensor([[2, 0, 0, 0]], dtype=torch.int32, device="cuda")
            num_accepted = torch.ones(1, dtype=torch.int32, device="cuda")
            drafts = manager.extend_and_draft(accepted, num_accepted, slot_ids, row_mask, 1, 3)
            torch.cuda.synchronize()
            assert drafts.tolist() == [[3, 1, 2]]
            assert manager.match_lens[:1].tolist() == [2]
            assert _history(manager, manager.slot_manager.get_slot(5)) == [1, 2, 3, 1, 2, 3, 1, 2]

            # draft_len == 0 only extends the history.
            drafts = manager.extend_and_draft(accepted, num_accepted, slot_ids, row_mask, 1, 0)
            torch.cuda.synchronize()
            assert drafts.shape == (1, 0)
            assert _history(manager, manager.slot_manager.get_slot(5)) == [
                1,
                2,
                3,
                1,
                2,
                3,
                1,
                2,
                2,
            ]
        finally:
            manager.shutdown()


if __name__ == "__main__":
    unittest.main()
