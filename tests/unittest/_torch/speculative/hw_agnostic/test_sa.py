import os
import sys
import unittest

import pytest
import torch

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.speculative.suffix_automaton import SAConfig, SuffixAutomatonManager
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig, SADecodingConfig

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.llm_data import llm_models_root


def get_perf_metrics(result):
    """Extract performance metrics from result using built-in request_perf_metrics."""
    metrics = {}
    if result.outputs and result.outputs[0].request_perf_metrics:
        perf = result.outputs[0].request_perf_metrics
        timing = perf.timing_metrics
        # Convert timedelta to seconds
        metrics["arrival_time"] = timing.arrival_time.total_seconds()
        metrics["first_token_time"] = timing.first_token_time.total_seconds()
        metrics["last_token_time"] = timing.last_token_time.total_seconds()
        # Calculate TTFT and E2E latency
        metrics["ttft"] = metrics["first_token_time"] - metrics["arrival_time"]
        metrics["e2e"] = metrics["last_token_time"] - metrics["arrival_time"]
    return metrics


# Test parameter combinations:
# - disable_overlap_scheduler: Controls scheduler mode (False=overlap enabled)
# - use_cuda_graph: Whether to use CUDA graph capture
# - attn_backend: Attention implementation (TRTLLM only - FLASHINFER not supported)
# - max_matching_ngram_size: SA matching mode (2=fixed size, -1=longest match)
#
# NOTE: FLASHINFER target decode supports multiple queries per request, but
# non-shared one-engine modes still require a separate draft KV cache. The
# draft KV metadata/manager swap is currently implemented only for TRTLLM
# attention. Shared-target-KV modes use a separate FlashInfer metadata view.
@pytest.mark.parametrize(
    "disable_overlap_scheduler,use_cuda_graph,attn_backend,max_matching_ngram_size",
    [
        [False, False, "TRTLLM", 2],
        [False, True, "TRTLLM", 2],
        [True, False, "TRTLLM", 2],
        [True, True, "TRTLLM", 2],
        [False, False, "TRTLLM", -1],
    ],
)
@pytest.mark.high_cuda_memory
def test_llama_sa(
    disable_overlap_scheduler: bool,
    use_cuda_graph: bool,
    attn_backend: str,
    max_matching_ngram_size: int,
):
    """Test SA (Suffix Automaton) speculative decoding correctness and acceptance rate.

    Verifies:
    1. Speculative decoding produces identical results to baseline
    2. SA drafting produces draft tokens that get accepted
    3. Multi-token acceptance occurs (acceptanceLength > 1)
    """
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if total_mem_gb < 20:
        pytest.skip("Not enough memory to load target model")

    print(
        f"\nTest config: disable_overlap_scheduler={disable_overlap_scheduler}, "
        f"use_cuda_graph={use_cuda_graph}, attn_backend={attn_backend}, "
        f"max_matching_ngram_size={max_matching_ngram_size}"
    )

    max_batch_size = 1
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
    # Enable perf metrics collection via return_perf_metrics=True
    sampling_params = SamplingParams(
        max_tokens=64, ignore_eos=True, temperature=0, return_perf_metrics=True
    )

    # Run with speculative decoding
    llm_spec = LLM(**llm_common_config, speculative_config=spec_config)
    results_spec = llm_spec.generate(prompts, sampling_params)
    generated_text_spec = [result.outputs[0].text for result in results_spec]

    # Get spec decoding stats before shutdown
    stats = llm_spec.get_stats(timeout=5)
    iterations_with_spec = []
    for stat in stats:
        if "specDecodingStats" in stat:
            spec_stats = stat["specDecodingStats"]
            if spec_stats.get("numDraftTokens", 0) > 0:
                iterations_with_spec.append(spec_stats)

    # Get perf metrics using built-in request_perf_metrics
    spec_metrics = get_perf_metrics(results_spec[0]) if results_spec else {}

    llm_spec.shutdown()

    # Run reference without speculative decoding
    llm_ref = LLM(**llm_common_config)
    results_ref = llm_ref.generate(prompts, sampling_params)
    generated_text_ref = [result.outputs[0].text for result in results_ref]

    # Get perf metrics for reference
    ref_metrics = get_perf_metrics(results_ref[0]) if results_ref else {}

    llm_ref.shutdown()

    # Verify 1: Identical results (correctness)
    for i, (text_spec, text_ref) in enumerate(zip(generated_text_spec, generated_text_ref)):
        assert text_spec == text_ref, (
            f"Prompt {i}: Spec decode result differs from baseline.\n"
            f"Spec: {text_spec}\nRef: {text_ref}"
        )
    print("Correctness verified: spec decode matches baseline")

    # Verify 2: Spec decoding stats show drafting occurred
    assert len(iterations_with_spec) > 0, (
        f"SA should have iterations with specDecodingStats. "
        f"Got {len(stats)} total stats but 0 with draft tokens."
    )

    total_draft = sum(s["numDraftTokens"] for s in iterations_with_spec)
    total_accepted = sum(s["numAcceptedTokens"] for s in iterations_with_spec)
    avg_acceptance_len = sum(s["acceptanceLength"] for s in iterations_with_spec) / len(
        iterations_with_spec
    )

    print("Spec decoding stats:")
    print(f"  Iterations with drafting: {len(iterations_with_spec)}")
    print(f"  Total draft tokens: {total_draft}")
    print(f"  Total accepted tokens: {total_accepted}")
    print(f"  Average acceptance length: {avg_acceptance_len:.2f}")
    print(f"  Acceptance rate: {total_accepted / total_draft * 100:.1f}%")

    assert total_draft > 0, "SA should produce draft tokens"
    assert total_accepted > 0, (
        f"SA should accept some draft tokens. "
        f"Got {total_accepted} accepted out of {total_draft} drafted"
    )

    # Verify 3: Multi-token acceptance (acceptanceLength > 1)
    has_multi_token_acceptance = any(s["acceptanceLength"] > 1.0 for s in iterations_with_spec)
    print(f"  Has multi-token acceptance: {has_multi_token_acceptance}")

    assert has_multi_token_acceptance, (
        "Expected at least one iteration with acceptanceLength > 1 for repetitive pattern"
    )

    # Print performance comparison using built-in metrics
    print("\n" + "=" * 70)
    print("PERFORMANCE COMPARISON (using request_perf_metrics)")
    print("=" * 70)
    print(
        f"Config: overlap_scheduler={'enabled' if not disable_overlap_scheduler else 'disabled'}, "
        f"cuda_graph={'enabled' if use_cuda_graph else 'disabled'}"
    )
    print("-" * 70)
    print(f"{'Metric':<30} {'Spec Decoding':<20} {'Reference':<20}")
    print("-" * 70)

    # Print TTFT (Time to First Token)
    ttft_spec = spec_metrics.get("ttft", None)
    ttft_ref = ref_metrics.get("ttft", None)
    ttft_spec_str = f"{ttft_spec * 1000:.2f} ms" if ttft_spec else "N/A"
    ttft_ref_str = f"{ttft_ref * 1000:.2f} ms" if ttft_ref else "N/A"
    print(f"{'TTFT':<30} {ttft_spec_str:<20} {ttft_ref_str:<20}")

    # Print E2E latency
    e2e_spec = spec_metrics.get("e2e", None)
    e2e_ref = ref_metrics.get("e2e", None)
    e2e_spec_str = f"{e2e_spec * 1000:.2f} ms" if e2e_spec else "N/A"
    e2e_ref_str = f"{e2e_ref * 1000:.2f} ms" if e2e_ref else "N/A"
    print(f"{'E2E Latency':<30} {e2e_spec_str:<20} {e2e_ref_str:<20}")

    # Calculate and print speedup
    if e2e_spec and e2e_ref and e2e_spec > 0:
        speedup = e2e_ref / e2e_spec
        print("-" * 70)
        print(f"{'Speedup (E2E)':<30} {speedup:.2f}x")
    print("=" * 70 + "\n")

    # Synchronize CUDA to catch any async memory errors before test completes.
    # This ensures errors are attributed to this test rather than propagating
    # to subsequent tests.
    torch.cuda.synchronize()


@pytest.mark.cpu_only
@pytest.mark.parametrize("max_matching_ngram_size", [2, 4, -1])
def test_sa_config_validation(max_matching_ngram_size: int):
    """Test SADecodingConfig validation."""
    # Valid configuration
    config = SADecodingConfig(
        max_draft_len=4,
        max_matching_ngram_size=max_matching_ngram_size,
    )
    assert config.max_matching_ngram_size == max_matching_ngram_size


@pytest.mark.cpu_only
def test_sa_config_invalid_zero():
    """Test that max_matching_ngram_size=0 raises error for SA."""
    with pytest.raises(ValueError, match="max_matching_ngram_size must be"):
        SADecodingConfig(
            max_draft_len=4,
            max_matching_ngram_size=0,
        )


class _FakeSARequest:
    """Minimal request stand-in for SuffixAutomatonManager.prepare_resources.

    Provides only the attributes/methods that prepare_resources and
    free_resources touch. Mirrors an LlmRequest on a disaggregated
    generation server, where the request skips the context phase
    (DISAGG_GENERATION_INIT) and is scheduled directly as a generation
    request with LLMREQUEST_TYPE_GENERATION_ONLY type.
    """

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
        self._generation_only = generation_only
        self.is_dummy = is_dummy
        self.is_first_context_chunk = is_first_context_chunk

    def get_tokens(self, beam: int) -> list:
        assert beam == 0
        return list(self._tokens)

    def is_generation_only_request(self) -> bool:
        return self._generation_only


class TestSADisaggGenInit:
    """CPU-only tests for the disagg-generation SA init path.

    On a disagg generation server, requests never appear as context
    requests, so prepare_resources must build the automaton from the
    generation request's token history (prompt + first generated token).
    These tests exercise only host-side automaton construction and slot
    bookkeeping; no GPU workspace is allocated.
    """

    @staticmethod
    def _make_manager(max_slots: int = 8) -> SuffixAutomatonManager:
        config = SAConfig(max_seq_len=1024, max_slots=max_slots)
        return SuffixAutomatonManager(config, max_num_requests=max_slots)

    def test_disagg_gen_request_initializes_automaton(self):
        """A generation-only request must be initialized from its tokens."""
        manager = self._make_manager()
        try:
            # Prompt tokens + first generated token (appended by
            # _prepare_disagg_gen_transmission_complete on the gen server).
            tokens = [1, 2, 3, 4, 5, 1, 2, 3] + [6]
            req = _FakeSARequest(7, tokens, generation_only=True)

            batch = ScheduledRequests()
            batch.generation_requests = [req]

            free_slots_before = len(manager._free_slots)
            manager.prepare_resources(batch)

            assert req.request_id in manager._initialized_requests
            assert req.request_id in manager._request_to_slot
            assert req.request_id in manager._host_states_native
            assert req.request_id in manager._pending_copies
            assert len(manager._free_slots) == free_slots_before - 1

            # A second prepare_resources with the same request (every
            # subsequent decode iteration) must be a no-op: same slot, no
            # extra slot consumed, host state not rebuilt.
            slot = manager._request_to_slot[req.request_id]
            state = manager._host_states_native[req.request_id]
            manager.prepare_resources(batch)
            assert manager._request_to_slot[req.request_id] == slot
            assert manager._host_states_native[req.request_id] is state
            assert len(manager._free_slots) == free_slots_before - 1
        finally:
            manager.shutdown()

    def test_regular_and_dummy_generation_requests_are_skipped(self):
        """Only disagg (generation-only) non-dummy requests are initialized."""
        manager = self._make_manager()
        try:
            regular_gen = _FakeSARequest(1, [1, 2, 3], generation_only=False)
            dummy_gen = _FakeSARequest(2, [1, 2, 3], generation_only=True, is_dummy=True)

            batch = ScheduledRequests()
            batch.generation_requests = [regular_gen, dummy_gen]

            free_slots_before = len(manager._free_slots)
            manager.prepare_resources(batch)

            assert regular_gen.request_id not in manager._initialized_requests
            assert dummy_gen.request_id not in manager._initialized_requests
            assert len(manager._free_slots) == free_slots_before
        finally:
            manager.shutdown()

    def test_context_initialized_request_not_reinitialized_in_generation(self):
        """Aggregated flow: context-phase init must not be redone in generation."""
        manager = self._make_manager()
        try:
            req = _FakeSARequest(3, [10, 20, 30], generation_only=False)

            ctx_batch = ScheduledRequests()
            ctx_batch.context_requests_last_chunk = [req]
            manager.prepare_resources(ctx_batch)
            slot = manager._request_to_slot[req.request_id]
            state = manager._host_states_native[req.request_id]

            # The same request later shows up as a generation request.
            gen_batch = ScheduledRequests()
            gen_batch.generation_requests = [req]
            manager.prepare_resources(gen_batch)

            assert manager._request_to_slot[req.request_id] == slot
            assert manager._host_states_native[req.request_id] is state
        finally:
            manager.shutdown()

    def test_free_resources_after_disagg_gen_init(self):
        """free_resources must fully release slots/bookkeeping for the gen path."""
        manager = self._make_manager()
        try:
            req = _FakeSARequest(4, [1, 2, 3, 4], generation_only=True)

            batch = ScheduledRequests()
            batch.generation_requests = [req]

            free_slots_before = len(manager._free_slots)
            manager.prepare_resources(batch)
            manager.free_resources(req)

            assert req.request_id not in manager._initialized_requests
            assert req.request_id not in manager._request_to_slot
            assert req.request_id not in manager._host_states_native
            assert req.request_id not in manager._pending_copies
            assert len(manager._free_slots) == free_slots_before

            # The request can be initialized again from scratch.
            manager.prepare_resources(batch)
            assert req.request_id in manager._initialized_requests
        finally:
            manager.shutdown()

    def test_disagg_gen_init_defers_to_generation_schedule(self):
        """Ctx/gen spec split: init must NOT happen at _prepare_disagg_gen_init.

        The executor's _prepare_disagg_gen_init routes DISAGG_GENERATION_INIT
        requests through prepare_resources as context_requests_last_chunk
        BEFORE the ctx server's first generated token has been appended
        (that happens later, in _prepare_disagg_gen_transmission_complete).
        Initializing there would freeze the automaton at prompt-only —
        permanently missing the ctx first token relative to an aggregated
        run — and pin an SA slot for the whole KV-transfer duration. The
        context loop must skip generation-only requests and let the
        generation loop initialize them with the full token history.
        """
        manager = self._make_manager()
        try:
            prompt = [1, 2, 3, 4, 5]
            req = _FakeSARequest(11, prompt, generation_only=True)

            # Phase 1: _prepare_disagg_gen_init — request arrives as a
            # context_requests_last_chunk entry with prompt-only tokens.
            init_batch = ScheduledRequests()
            init_batch.context_requests_last_chunk = [req]

            free_slots_before = len(manager._free_slots)
            manager.prepare_resources(init_batch)

            assert req.request_id not in manager._initialized_requests
            assert req.request_id not in manager._request_to_slot
            # No SA slot pinned during the KV transfer.
            assert len(manager._free_slots) == free_slots_before

            # Phase 2: transmission complete — the executor appends the ctx
            # first token and the request is scheduled as a generation
            # request. Init must now use the FULL history.
            req._tokens.append(6)

            seen_tokens = {}
            orig_add_request = manager.add_request

            def spy_add_request(request_id, context_tokens):
                seen_tokens[request_id] = list(context_tokens)
                return orig_add_request(request_id, context_tokens)

            manager.add_request = spy_add_request
            gen_batch = ScheduledRequests()
            gen_batch.generation_requests = [req]
            manager.prepare_resources(gen_batch)
            manager.add_request = orig_add_request

            assert req.request_id in manager._initialized_requests
            assert seen_tokens[req.request_id] == prompt + [6]
        finally:
            manager.shutdown()


class TestKdaReplaySeedOnDisaggTransfer(unittest.TestCase):
    """seed_kda_replay_caches_for_disagg_gen must mirror
    _sync_kda_replay_conv_window for transferred requests: committed conv
    window seeded from the (transferred) conv pool, draft tail columns and
    pending-draft scratch cleared, other slots untouched."""

    L, SLOTS, D, W, M, NHEADS = 2, 4, 6, 4, 2, 3  # committed = W - 1 = 3

    def _make_manager(self, use_kda_replay=True):
        from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import (
            PythonMambaCacheManager,
        )

        L, SLOTS, D, W, M, NH = (self.L, self.SLOTS, self.D, self.W, self.M, self.NHEADS)
        committed = W - 1

        class _FakeSpecState:
            pass

        cache = _FakeSpecState()
        torch.manual_seed(0)
        cache.conv = torch.randn(L, SLOTS, 3 * D, W)
        cache.kda_conv_q = torch.full((L, SLOTS, D, committed + M), 7.0)
        cache.kda_conv_k = torch.full((L, SLOTS, D, committed + M), 7.0)
        cache.kda_conv_v = torch.full((L, SLOTS, D, committed + M), 7.0)
        cache.kda_qkg_cache = torch.full((L, SLOTS, M, 3, D), 7.0)
        cache.kda_v_cache = torch.full((L, SLOTS, M, D), 7.0)
        cache.kda_beta_cache = torch.full((L, SLOTS, M, NH), 7.0)
        cache.prev_num_accepted_tokens = torch.full((SLOTS,), 5, dtype=torch.int32)

        mgr = PythonMambaCacheManager.__new__(PythonMambaCacheManager)
        mgr._use_kda_replay_update = use_kda_replay
        mgr.SpeculativeState = _FakeSpecState
        mgr.mamba_cache = cache
        mgr.mamba_cache_index = {101: 1, 202: 3}
        return mgr, cache

    def test_seeds_committed_window_and_clears_scratch(self):
        mgr, cache = self._make_manager()
        committed = self.W - 1
        conv_before = cache.conv.clone()
        mgr.seed_kda_replay_caches_for_disagg_gen([101, 202])
        d = self.D
        for slot in (1, 3):
            for kda, lo, hi in (
                (cache.kda_conv_q, 0, d),
                (cache.kda_conv_k, d, 2 * d),
                (cache.kda_conv_v, 2 * d, 3 * d),
            ):
                torch.testing.assert_close(
                    kda[:, slot, :, :committed], conv_before[:, slot, lo:hi, 1:].to(kda.dtype)
                )
                assert (kda[:, slot, :, committed:] == 0).all()
            assert (cache.kda_qkg_cache[:, slot] == 0).all()
            assert (cache.kda_v_cache[:, slot] == 0).all()
            assert (cache.kda_beta_cache[:, slot] == 0).all()
            assert cache.prev_num_accepted_tokens[slot] == 0
        # Conv pool itself must not be modified.
        torch.testing.assert_close(cache.conv, conv_before)
        # Untouched slots keep their contents.
        for slot in (0, 2):
            assert (cache.kda_conv_q[:, slot] == 7.0).all()
            assert (cache.kda_qkg_cache[:, slot] == 7.0).all()
            assert cache.prev_num_accepted_tokens[slot] == 5

    def test_noop_without_kda_replay_or_unknown_ids(self):
        mgr, cache = self._make_manager(use_kda_replay=False)
        mgr.seed_kda_replay_caches_for_disagg_gen([101])
        assert (cache.kda_conv_q == 7.0).all()

        mgr2, cache2 = self._make_manager()
        mgr2.seed_kda_replay_caches_for_disagg_gen([999])  # not in index
        assert (cache2.kda_conv_q == 7.0).all()


if __name__ == "__main__":
    unittest.main()
