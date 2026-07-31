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
        # PyExecutor's inflight-set bookkeeping reads ``request_id`` (the C++
        # binding's name) rather than ``py_request_id``; keep them in sync so the
        # same fake drives both the fallback and TestInflightIdsSurviveTrim.
        self.request_id = self.py_request_id
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

    def test_maybe_fit_token_budget_skips_draft_manager(self):
        # maybe_fit_token_budget is the single entry point driven by the
        # aggregate ResourceManager. It must apply the fallback for the target
        # manager only -- the draft-model engine builds inputs with a different
        # token shape and its budget is handled separately.
        ctx = _FakeRequest(context_chunk_size=64, prompt_len=64)
        gen = _FakeRequest(py_beam_width=120)  # remaining = 8 -> defer ctx

        # Non-draft -> defers.
        mgr = _make_manager(max_num_tokens=128, tokens_per_block=16, enable_chunked_prefill=False)
        mgr.is_draft = False
        batch = _make_batch([ctx], [gen])
        mgr.maybe_fit_token_budget(batch)
        self.assertEqual(batch.num_context_requests, 0)

        # Draft manager -> never fits (handled separately).
        mgr.is_draft = True
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
        # The executor loop drives the fallback via
        # ResourceManager.maybe_fit_token_budget before it calls
        # prepare_resources; build the aggregate ResourceManager with the same
        # ordering as production (draft-like manager first, KV cache manager
        # last) and assert the earlier manager observes the *already-deferred*
        # batch.
        target = _make_manager(
            max_num_tokens=128, tokens_per_block=16, enable_chunked_prefill=False
        )
        target.is_draft = False
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
        rm.maybe_fit_token_budget(batch)
        rm.prepare_resources(batch)

        # ctx_keep (96) fits into 112; ctx_defer (64) does not and is deferred.
        # The draft-like manager, though invoked first, must have seen only the
        # kept request -- proving the fallback ran up front.
        self.assertEqual(observed, [[ctx_keep.py_request_id]])
        self.assertEqual(batch.num_context_requests, 1)

    def test_prepare_resources_does_not_trim(self):
        # The fallback must NOT be reachable from prepare_resources: the
        # executor loop drives it earlier so that _can_queue's attention-DP
        # tp_allgather(batch_size) and the PP inflight-set registration both see
        # the trimmed batch. A second trim here would be redundant at best, and
        # leaving it as the *only* trim point is what let a rank shed its way to
        # an empty batch after the ranks had already voted to run.
        target = _make_manager(
            max_num_tokens=128, tokens_per_block=16, enable_chunked_prefill=False
        )
        target.is_draft = False
        target.prepare_resources = lambda batch: None

        ctx_over_budget = _FakeRequest(context_chunk_size=64)
        gen = _FakeRequest(py_beam_width=100)  # remaining = 28, so ctx cannot fit
        batch = _make_batch([ctx_over_budget], [gen])

        rm = ResourceManager(OrderedDict([(ResourceManagerType.KV_CACHE_MANAGER, target)]))
        rm.prepare_resources(batch)

        self.assertEqual(batch.num_context_requests, 1)

        # ...and the batch is trimmed only once maybe_fit_token_budget is called.
        rm.maybe_fit_token_budget(batch)
        self.assertEqual(batch.num_context_requests, 0)

    def test_generation_alone_over_budget_raises(self):
        # A context request must be present for the fallback to engage at all
        # (see test_gen_only_batch_is_left_alone); generation requests that
        # exceed the budget by themselves leave nothing to shed.
        mgr = _make_manager(max_num_tokens=128, tokens_per_block=16)
        gen = _FakeRequest(py_beam_width=200)
        ctx = _FakeRequest(context_chunk_size=16)
        batch = _make_batch([ctx], [gen])

        with self.assertRaises(RuntimeError):
            mgr._fit_token_budget(batch)

    def test_gen_only_batch_is_left_alone(self):
        # Context requests are the only thing the fallback can shed, so a
        # gen-only batch returns before the generation-token accounting: it
        # keeps that scan off the executor loop's hottest path, and it avoids
        # raising on a condition nothing here can fix. An over-budget gen-only
        # batch stays the concern of the _prepare_tp_inputs assert, which fails
        # one batch rather than killing the (possibly only rank-local) event
        # loop.
        mgr = _make_manager(max_num_tokens=128, tokens_per_block=16)
        gen = _FakeRequest(py_beam_width=200)  # 200 > max_num_tokens
        batch = _make_batch([], [gen])

        mgr._fit_token_budget(batch)  # must not raise

        self.assertEqual(batch.num_context_requests, 0)
        self.assertEqual(batch.generation_requests, [gen])


class TestInflightIdsSurviveTrim(unittest.TestCase):
    """The fallback must not strand ids in PyExecutor's inflight set.

    ``_executor_loop_pp`` calls ``_add_inflight_ids`` before
    ``ResourceManager.prepare_resources`` and ``_remove_inflight_ids`` after, so
    the fallback runs between them and mutates the batch: a deferred context
    request is dropped from it, and a re-chunked one moves out of
    ``context_requests_last_chunk``. Removal must therefore erase the ids that
    were actually inserted, not re-derive them from the trimmed batch -- an id
    left behind makes the scheduler skip that request forever (scheduler.py's
    ``if req.request_id in inflight_request_ids: continue``).
    """

    @staticmethod
    def _bare_executor():
        # Same trick as _make_manager: PyExecutor.__init__ builds an engine, so
        # instantiate bare and supply only the inflight set the methods touch.
        from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
        from tensorrt_llm.bindings.internal.batch_manager import ReqIdsSet

        executor = PyExecutor.__new__(PyExecutor)
        executor.inflight_req_ids = ReqIdsSet()
        return executor

    def test_trimmed_context_requests_leave_no_inflight_ids(self):
        executor = self._bare_executor()
        mgr = _make_manager(max_num_tokens=128, tokens_per_block=16)

        gen = _FakeRequest(py_beam_width=100)  # leaves a 28-token budget
        # Both start as last-chunk requests, so both are registered inflight.
        # ctx_rechunked shrinks to a block-aligned 16 and flips to a non-last
        # chunk; ctx_deferred no longer fits and is dropped from the batch.
        ctx_rechunked = _FakeRequest(context_chunk_size=64, prompt_len=64)
        ctx_deferred = _FakeRequest(context_chunk_size=64, prompt_len=64)
        batch = _make_batch([ctx_rechunked, ctx_deferred], [gen])

        executor._add_inflight_ids(batch)
        self.assertEqual(
            sorted(batch.added_inflight_req_ids),
            sorted([ctx_rechunked.request_id, ctx_deferred.request_id, gen.request_id]),
        )

        mgr._fit_token_budget(batch)

        # Preconditions for the regression: the batch really did change shape.
        self.assertEqual(ctx_rechunked.context_chunk_size, 16)
        self.assertIn(ctx_rechunked, batch.context_requests_chunking)
        self.assertEqual(batch.context_requests_last_chunk, [])
        self.assertNotIn(ctx_deferred, batch.context_requests)

        executor._remove_inflight_ids(batch)

        for req in (ctx_rechunked, ctx_deferred, gen):
            self.assertNotIn(
                req.request_id,
                executor.inflight_req_ids,
                f"request {req.request_id} left in the inflight set; the scheduler "
                "would never schedule it again",
            )
        self.assertEqual(batch.added_inflight_req_ids, [])

    def test_untrimmed_batch_round_trips(self):
        # The batch the fallback leaves alone must behave exactly as before.
        executor = self._bare_executor()
        mgr = _make_manager(max_num_tokens=128, tokens_per_block=16)

        gen = _FakeRequest(py_beam_width=1)
        ctx = _FakeRequest(context_chunk_size=16)
        batch = _make_batch([ctx], [gen])

        executor._add_inflight_ids(batch)
        for req in (ctx, gen):
            self.assertIn(req.request_id, executor.inflight_req_ids)

        mgr._fit_token_budget(batch)
        self.assertEqual(batch.num_context_requests, 1)

        executor._remove_inflight_ids(batch)
        for req in (ctx, gen):
            self.assertNotIn(req.request_id, executor.inflight_req_ids)


if __name__ == "__main__":
    unittest.main()
