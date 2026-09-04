import os
import sys
import unittest

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.speculative.suffix_automaton import SAConfig, SuffixAutomatonManager
from tensorrt_llm.llmapi import SADecodingConfig

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


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

    L, SLOTS, D, W, M, NHEADS = 2, 4, 6, 4, 2, 3

    def _make_manager(self, use_kda_replay=True):
        from tensorrt_llm._torch.pyexecutor.mamba_cache_manager import PythonMambaCacheManager

        L, SLOTS, D, W, M, NH = (self.L, self.SLOTS, self.D, self.W, self.M, self.NHEADS)
        committed = W - 1

        class _FakeSpecState:
            pass

        cache = _FakeSpecState()
        torch.manual_seed(0)
        cache.conv = torch.randn(L, SLOTS, 3 * D, committed)
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
                    kda[:, slot, :, :committed], conv_before[:, slot, lo:hi].to(kda.dtype)
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
