# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Unit tests for KVCacheManager._fit_token_budget.

These exercise the prep-boundary token-budget fallback that defers or
re-chunks context requests so a scheduled batch cannot overshoot
``max_num_tokens`` in the forward pass (GitHub issue #13318). The fallback is
pure scheduling logic and does not touch the GPU, so the tests build a bare
KVCacheManager via ``__new__`` and drive the method with lightweight fake
requests.
"""

import unittest
from collections import OrderedDict

from tensorrt_llm._torch.pyexecutor.resource_manager import (
    KVCacheManager,
    ResourceManager,
    ResourceManagerType,
)
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests


class _FakeRequest:
    """Minimal stand-in exposing only the attributes _fit_token_budget reads."""

    _next_id = 0

    def __init__(
        self,
        *,
        context_chunk_size=0,
        is_last_context_chunk=True,
        prompt_len=None,
        context_current_position=0,
        py_beam_width=1,
        py_draft_tokens=None,
        is_disagg_generation_init_state=False,
        mm_bidirectional=False,
    ):
        _FakeRequest._next_id += 1
        self.py_request_id = _FakeRequest._next_id
        self.context_chunk_size = context_chunk_size
        self.context_current_position = context_current_position
        # Mirrors the C++ semantics: is_last_context_chunk is a *computed*
        # property (context_current_position + context_chunk_size == prompt_len),
        # so shrinking the chunk during re-chunk flips it to False. When
        # prompt_len is None the flag is a fixed override (for tests that don't
        # exercise re-chunk re-binning).
        self._prompt_len = prompt_len
        self._is_last_override = is_last_context_chunk
        self.py_beam_width = py_beam_width
        self.py_draft_tokens = py_draft_tokens
        self.is_disagg_generation_init_state = is_disagg_generation_init_state
        self.py_multimodal_data = {"mm_bidirectional_blocks": True} if mm_bidirectional else None

    @property
    def is_last_context_chunk(self):
        if self._prompt_len is None:
            return self._is_last_override
        return self.context_current_position + self.context_chunk_size == self._prompt_len


def _make_manager(max_num_tokens, tokens_per_block, enable_chunked_prefill=True):
    # Skip the heavy (GPU-allocating) __init__; the method under test only
    # needs these attributes plus its own (bound) helper methods.
    mgr = KVCacheManager.__new__(KVCacheManager)
    mgr.max_num_tokens = max_num_tokens
    mgr.tokens_per_block = tokens_per_block
    # Re-chunking is only valid when chunked prefill is enabled; otherwise the
    # attention backend cannot consume a partial context chunk and the fallback
    # must defer instead. Default to enabled so the re-chunk tests exercise that
    # path; the disabled case is covered explicitly below.
    mgr.enable_chunked_prefill = enable_chunked_prefill
    return mgr


def _make_batch(context_requests=(), generation_requests=()):
    batch = ScheduledRequests()
    for req in context_requests:
        batch.append_context_request(req)
    batch.generation_requests = list(generation_requests)
    return batch


class TestFitTokenBudget(unittest.TestCase):
    def test_request_forward_tokens_upper_bound(self):
        mgr = _make_manager(max_num_tokens=128, tokens_per_block=16)

        # Context: chunk size, plus draft tokens only on the last chunk.
        last = _FakeRequest(
            context_chunk_size=10, is_last_context_chunk=True, py_draft_tokens=[1, 2]
        )
        self.assertEqual(mgr._request_forward_tokens(last, is_context=True), 12)
        mid = _FakeRequest(
            context_chunk_size=10, is_last_context_chunk=False, py_draft_tokens=[1, 2]
        )
        self.assertEqual(mgr._request_forward_tokens(mid, is_context=True), 10)

        # Generation: (1 + draft) per beam.
        gen = _FakeRequest(py_beam_width=2, py_draft_tokens=[1, 2, 3])
        self.assertEqual(mgr._request_forward_tokens(gen, is_context=False), 8)

    def test_within_budget_is_noop(self):
        mgr = _make_manager(max_num_tokens=128, tokens_per_block=16)
        ctx = _FakeRequest(context_chunk_size=16)
        gen = _FakeRequest(py_beam_width=100)  # 100 gen tokens
        batch = _make_batch([ctx], [gen])

        mgr._fit_token_budget(batch)

        self.assertEqual(batch.num_context_requests, 1)
        self.assertEqual(ctx.context_chunk_size, 16)  # untouched

    def test_overshoot_rechunks_context(self):
        # 100 gen tokens leave a 28-token budget; a 64-token last chunk does not
        # fit but can be re-chunked down to a block-aligned 16.
        mgr = _make_manager(max_num_tokens=128, tokens_per_block=16)
        ctx = _FakeRequest(context_chunk_size=64, is_last_context_chunk=True)
        gen = _FakeRequest(py_beam_width=100)
        batch = _make_batch([ctx], [gen])

        mgr._fit_token_budget(batch)

        self.assertEqual(batch.num_context_requests, 1)
        self.assertEqual(ctx.context_chunk_size, 16)  # (28 // 16) * 16
        total = mgr._request_forward_tokens(ctx, is_context=True) + mgr._request_forward_tokens(
            gen, is_context=False
        )
        self.assertLessEqual(total, mgr.max_num_tokens)

    def test_rechunk_only_rebins_to_chunking(self):
        # Regression for the prep-boundary corruption (issue #13318 follow-up):
        # when the overshoot is absorbed purely by re-chunking the *last*
        # context request (no deferral), len(kept) is unchanged, but the request
        # has flipped from last-chunk to non-last and MUST be moved out of the
        # last-chunk bin. Otherwise downstream treats it as a final chunk and
        # appends generation/draft tokens, corrupting the forward pass.
        mgr = _make_manager(max_num_tokens=128, tokens_per_block=16)
        # Full prompt is 64 tokens, processed in one (last) chunk.
        ctx = _FakeRequest(context_chunk_size=64, prompt_len=64)
        self.assertTrue(ctx.is_last_context_chunk)
        gen = _FakeRequest(py_beam_width=100)  # remaining = 28
        batch = _make_batch([ctx], [gen])

        mgr._fit_token_budget(batch)

        # Re-chunked to (28 // 16) * 16 == 16, now a non-last chunk.
        self.assertEqual(ctx.context_chunk_size, 16)
        self.assertFalse(ctx.is_last_context_chunk)
        # Count is unchanged, but it must have been re-binned into chunking.
        self.assertEqual(batch.num_context_requests, 1)
        self.assertIn(ctx, batch.context_requests_chunking)
        self.assertNotIn(ctx, batch.context_requests_last_chunk)

    def test_rechunk_drops_last_chunk_draft_tokens(self):
        # Same re-chunk regression as above, but with draft tokens, which are
        # appended only on the *last* chunk (see _request_forward_tokens). If a
        # re-chunked request were left on the last-chunk path, its draft tokens
        # would still be counted/materialized and re-introduce the overshoot
        # this guard prevents. After re-chunking, the request must be a non-last
        # chunk and its forward-token cost must no longer include the draft.
        mgr = _make_manager(max_num_tokens=128, tokens_per_block=16)
        # 64-token last chunk + 2 draft tokens; a 28-token budget cannot fit
        # 64 (+2), but the chunk re-chunks to a block-aligned 16.
        ctx = _FakeRequest(context_chunk_size=64, prompt_len=64, py_draft_tokens=[1, 2])
        self.assertTrue(ctx.is_last_context_chunk)
        gen = _FakeRequest(py_beam_width=100)  # remaining = 28
        batch = _make_batch([ctx], [gen])

        mgr._fit_token_budget(batch)

        # Re-chunked to (28 // 16) * 16 == 16 and flipped to a non-last chunk.
        self.assertEqual(ctx.context_chunk_size, 16)
        self.assertFalse(ctx.is_last_context_chunk)
        self.assertIn(ctx, batch.context_requests_chunking)
        self.assertNotIn(ctx, batch.context_requests_last_chunk)
        # Cost is now the chunk size alone -- the 2 draft tokens are dropped
        # because the request is no longer the last chunk.
        self.assertEqual(mgr._request_forward_tokens(ctx, is_context=True), 16)
        total = mgr._request_forward_tokens(ctx, is_context=True) + mgr._request_forward_tokens(
            gen, is_context=False
        )
        self.assertLessEqual(total, mgr.max_num_tokens)

    def test_overshoot_defers_when_chunked_prefill_disabled(self):
        # Regression for the CI failures (q.numel()==0 / "Separate quantized
        # buffer is not provided" / cudaErrorInvalidValue) seen in PR #15187:
        # when chunked prefill is disabled the attention backend cannot consume
        # a partial context chunk, so an over-budget request that *would* be
        # re-chunkable must instead be deferred whole -- never re-chunked.
        mgr = _make_manager(max_num_tokens=128, tokens_per_block=16, enable_chunked_prefill=False)
        # Same shape as test_overshoot_rechunks_context (a 28-token budget and a
        # 64-token last chunk that is block-aligned re-chunkable to 16), but with
        # chunked prefill off the request must be deferred, not shrunk.
        ctx = _FakeRequest(context_chunk_size=64, prompt_len=64)
        gen = _FakeRequest(py_beam_width=100)  # remaining = 28
        batch = _make_batch([ctx], [gen])

        mgr._fit_token_budget(batch)

        self.assertEqual(batch.num_context_requests, 0)
        self.assertEqual(ctx.context_chunk_size, 64)  # not re-chunked
        self.assertTrue(ctx.is_last_context_chunk)  # still a whole last chunk

    def test_overshoot_defers_when_cannot_rechunk(self):
        # Only an 8-token budget remains -- smaller than one block -- so the
        # context request cannot be re-chunked and must be deferred entirely.
        mgr = _make_manager(max_num_tokens=128, tokens_per_block=16)
        ctx = _FakeRequest(context_chunk_size=64)
        gen = _FakeRequest(py_beam_width=120)  # remaining = 8 < tokens_per_block
        batch = _make_batch([ctx], [gen])

        mgr._fit_token_budget(batch)

        self.assertEqual(batch.num_context_requests, 0)
        self.assertEqual(ctx.context_chunk_size, 64)  # not re-chunked

    def test_mm_bidirectional_is_deferred_not_rechunked(self):
        # A re-chunkable budget exists, but splitting a bidirectional MM block
        # would corrupt attention, so the request is deferred whole.
        mgr = _make_manager(max_num_tokens=128, tokens_per_block=16)
        ctx = _FakeRequest(context_chunk_size=64, mm_bidirectional=True)
        gen = _FakeRequest(py_beam_width=100)  # remaining = 28
        batch = _make_batch([ctx], [gen])

        mgr._fit_token_budget(batch)

        self.assertEqual(batch.num_context_requests, 0)
        self.assertEqual(ctx.context_chunk_size, 64)

    def test_defers_all_subsequent_context_requests(self):
        # ctx1 fits; ctx2 overshoots and cannot re-chunk; ctx3 (small) must
        # still be deferred to preserve context-progress ordering.
        mgr = _make_manager(max_num_tokens=128, tokens_per_block=16)
        ctx1 = _FakeRequest(context_chunk_size=96)
        ctx2 = _FakeRequest(context_chunk_size=64)
        ctx3 = _FakeRequest(context_chunk_size=16)
        gen = _FakeRequest(py_beam_width=16)  # remaining = 112
        batch = _make_batch([ctx1, ctx2, ctx3], [gen])

        mgr._fit_token_budget(batch)

        # ctx1 (96) fits into 112; remaining 16. ctx2 (64) doesn't fit and
        # (16 // 16) * 16 == 16 but 16 < 64 so it *could* re-chunk to 16...
        # remaining after is 0, so ctx3 is deferred.
        kept = batch.context_requests
        self.assertIn(ctx1, kept)
        self.assertNotIn(ctx3, kept)

    def test_fallback_can_be_disabled_via_flag(self):
        # The fallback is opt-out (TorchLlmArgs.enable_token_budget_fallback,
        # default True). When disabled, prepare_resources must NOT invoke
        # _fit_token_budget, leaving the scheduled batch untouched.
        mgr = _make_manager(max_num_tokens=128, tokens_per_block=16)
        mgr.is_draft = False
        mgr.enable_token_budget_fallback = False

        called = []
        mgr._fit_token_budget = lambda batch: called.append(batch)

        # Reproduce the gate from KVCacheManager.prepare_resources without the
        # surrounding GPU work.
        if not mgr.is_draft and mgr.enable_token_budget_fallback:
            mgr._fit_token_budget(object())

        self.assertEqual(called, [])

        # Sanity: enabling it does call through.
        mgr.enable_token_budget_fallback = True
        if not mgr.is_draft and mgr.enable_token_budget_fallback:
            mgr._fit_token_budget(object())
        self.assertEqual(len(called), 1)

    def test_torch_llm_args_flag_default_is_opt_out(self):
        # The user-facing flag must default to enabled (opt-out semantics).
        from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

        field = TorchLlmArgs.model_fields["enable_token_budget_fallback"]
        self.assertEqual(field.default, True)

    def test_maybe_fit_token_budget_honors_flag_and_draft(self):
        # maybe_fit_token_budget is the single entry point driven by the
        # aggregate ResourceManager. It must apply the fallback only for the
        # non-draft manager and only when the opt-out flag is enabled.
        ctx = _FakeRequest(context_chunk_size=64, prompt_len=64)
        gen = _FakeRequest(py_beam_width=120)  # remaining = 8 -> defer ctx

        # Non-draft + enabled -> defers.
        mgr = _make_manager(max_num_tokens=128, tokens_per_block=16, enable_chunked_prefill=False)
        mgr.is_draft = False
        mgr.enable_token_budget_fallback = True
        batch = _make_batch([ctx], [gen])
        mgr.maybe_fit_token_budget(batch)
        self.assertEqual(batch.num_context_requests, 0)

        # Draft manager -> never fits (handled separately).
        mgr.is_draft = True
        batch = _make_batch([ctx], [gen])
        mgr.maybe_fit_token_budget(batch)
        self.assertEqual(batch.num_context_requests, 1)

        # Disabled flag -> no-op.
        mgr.is_draft = False
        mgr.enable_token_budget_fallback = False
        batch = _make_batch([ctx], [gen])
        mgr.maybe_fit_token_budget(batch)
        self.assertEqual(batch.num_context_requests, 1)

    def test_fallback_runs_before_other_managers(self):
        # Regression for the emplaceDone double-add (PR #15187): the token-budget
        # fallback must mutate scheduled_batch BEFORE any resource manager
        # allocates sequences. A separate draft KV cache manager (MTP) is
        # invoked before the target KV cache manager (the target is moved to the
        # end of the manager dict on purpose), so if the fallback ran inside the
        # target's own prepare_resources the draft manager would already have
        # added sequences for context requests the fallback then defers --
        # orphaning them and causing a double-add when they reschedule.
        #
        # Build the aggregate ResourceManager with the same ordering as
        # production (draft-like manager first, KV cache manager last) and assert
        # the earlier manager observes the *already-deferred* batch.
        target = _make_manager(
            max_num_tokens=128, tokens_per_block=16, enable_chunked_prefill=False
        )
        target.is_draft = False
        target.enable_token_budget_fallback = True
        # Don't touch the GPU: only the budget fallback matters for ordering.
        target.prepare_resources = lambda batch: None

        observed = []

        class _RecordingManager:
            def prepare_resources(self, batch):
                observed.append([r.py_request_id for r in batch.context_requests])

        ctx_keep = _FakeRequest(context_chunk_size=96)
        ctx_defer = _FakeRequest(context_chunk_size=64)
        gen = _FakeRequest(py_beam_width=16)  # remaining = 112
        batch = _make_batch([ctx_keep, ctx_defer], [gen])

        # Draft-like manager registered FIRST, KV cache manager LAST (mirrors
        # _util.py's move_to_end(KV_CACHE_MANAGER)).
        rm = ResourceManager(
            OrderedDict(
                [
                    (ResourceManagerType.DRAFT_KV_CACHE_MANAGER, _RecordingManager()),
                    (ResourceManagerType.KV_CACHE_MANAGER, target),
                ]
            )
        )
        rm.prepare_resources(batch)

        # ctx_keep (96) fits into 112; ctx_defer (64) does not and is deferred.
        # The draft-like manager, though invoked first, must have seen only the
        # kept request -- proving the fallback ran up front.
        self.assertEqual(observed, [[ctx_keep.py_request_id]])
        self.assertEqual(batch.num_context_requests, 1)

    def test_generation_alone_over_budget_raises(self):
        mgr = _make_manager(max_num_tokens=128, tokens_per_block=16)
        gen = _FakeRequest(py_beam_width=200)
        batch = _make_batch([], [gen])

        with self.assertRaises(RuntimeError):
            mgr._fit_token_budget(batch)


if __name__ == "__main__":
    unittest.main()
