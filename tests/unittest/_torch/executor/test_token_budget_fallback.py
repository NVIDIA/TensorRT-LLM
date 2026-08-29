# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Unit tests for KVCacheManager.fit_token_budget.

These exercise the post-allocation token-budget trim that shrinks over-budget
context chunks so a scheduled batch cannot overshoot ``max_num_tokens`` in the
forward pass (GitHub issue #13318). The trim is pure scheduling logic and does
not touch the GPU, so the tests build a bare KVCacheManager via ``__new__`` and
drive the method with lightweight fake requests.

The trim runs at the end of ``ResourceManager.prepare_resources``, which is the
first point where ``context_current_position`` and ``context_chunk_size`` mean
"forward-pass tokens" -- before ``setPrepopulatedPromptLen`` the chunk still
spans the reusable KV prefix. ``TestReuseDiscountedChunk`` is the regression
test for reading it too early.
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
    """Minimal stand-in exposing only the attributes fit_token_budget reads."""

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
        # same fake drives both the trim and TestInflightIdsSurviveTrim.
        self.request_id = self.py_request_id
        self.context_chunk_size = context_chunk_size
        self.context_current_position = context_current_position
        # Mirrors the C++ semantics: is_last_context_chunk is a *computed*
        # property (context_current_position + context_chunk_size == prompt_len),
        # so shrinking the chunk flips it to False. When prompt_len is None the
        # flag is a fixed override (for tests that don't exercise re-binning).
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

    @property
    def context_remaining_length(self):
        # C++: mPromptLen - getContextCurrentPosition(). With no prompt_len the
        # chunk is by definition all that is left.
        if self._prompt_len is None:
            return self.context_chunk_size
        return self._prompt_len - self.context_current_position


def _make_manager(max_num_tokens, tokens_per_block, enable_chunked_prefill=True):
    # Skip the heavy (GPU-allocating) __init__; the method under test only
    # needs these attributes plus its own (bound) helper methods.
    mgr = KVCacheManager.__new__(KVCacheManager)
    mgr.max_num_tokens = max_num_tokens
    mgr.tokens_per_block = tokens_per_block
    # Shrinking produces a partial context chunk, which only chunked prefill's
    # attention path can consume. Default to enabled; the disabled case is
    # covered explicitly below.
    mgr.enable_chunked_prefill = enable_chunked_prefill
    mgr.is_draft = False
    # Read by publish_connector_scheduler_output; most tests run without a
    # connector attached.
    mgr.kv_connector_manager = None
    return mgr


def _make_batch(context_requests=(), generation_requests=()):
    batch = ScheduledRequests()
    for req in context_requests:
        batch.append_context_request(req)
    batch.generation_requests = list(generation_requests)
    return batch


def _forward_tokens(mgr, batch):
    """What _prepare_tp_inputs will materialize for this batch."""
    return sum(
        mgr._request_forward_tokens(r, is_context=False) for r in batch.generation_requests
    ) + sum(
        mgr._request_forward_tokens(r, is_context=True)
        for r in batch.context_requests
        if not r.is_disagg_generation_init_state
    )


class TestReuseDiscountedChunk(unittest.TestCase):
    """Regression for the defect this trim shipped with.

    Read before ``prepare_resources``, ``context_chunk_size`` spans the reusable
    KV prefix: a 19212-token prompt with a 19200-token cache hit still reports
    ``context_chunk_size == 19212`` while the forward pass will compute 12
    tokens. Costing it at 19212 makes every reuse hit look 1600x too expensive,
    so it is endlessly re-chunked (chunked prefill on) or deferred (off).

    Read after ``prepare_resources`` -- where the trim now runs --
    ``setPrepopulatedPromptLen`` has advanced ``context_current_position`` past
    the prefix and the same request costs 12.
    """

    def test_reuse_hit_costs_only_the_uncached_tail(self):
        mgr = _make_manager(max_num_tokens=8192, tokens_per_block=32)
        # Post-setPrepopulatedPromptLen state for a 19212-token prompt with a
        # 19200-token cache hit.
        req = _FakeRequest(context_chunk_size=12, context_current_position=19200, prompt_len=19212)
        self.assertEqual(mgr._request_forward_tokens(req, is_context=True), 12)

    def test_reuse_hit_is_not_trimmed(self):
        mgr = _make_manager(max_num_tokens=8192, tokens_per_block=32)
        reqs = [
            _FakeRequest(context_chunk_size=12, context_current_position=19200, prompt_len=19212)
            for _ in range(4)
        ]
        batch = _make_batch(reqs, [_FakeRequest(py_beam_width=1) for _ in range(3)])

        mgr.fit_token_budget(batch)

        self.assertEqual(batch.num_context_requests, 4, "no request may be dropped")
        for req in reqs:
            self.assertEqual(req.context_chunk_size, 12, "chunk must be untouched")

    def test_chunk_beyond_prompt_end_is_clamped(self):
        # _prepare_tp_inputs slices all_prompt_tokens[pos:pos + chunk], and
        # Python clamps that to the end of the list. A chunk that overhangs the
        # prompt must be costed at what is actually left, not at its nominal
        # size.
        mgr = _make_manager(max_num_tokens=8192, tokens_per_block=32)
        req = _FakeRequest(
            context_chunk_size=8160, context_current_position=19200, prompt_len=19212
        )
        self.assertEqual(mgr._request_forward_tokens(req, is_context=True), 12)


class TestFitTokenBudget(unittest.TestCase):
    def test_request_forward_tokens(self):
        mgr = _make_manager(max_num_tokens=128, tokens_per_block=16)

        # Context: materialized chunk, plus draft tokens only on the last chunk.
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

        mgr.fit_token_budget(batch)

        self.assertEqual(batch.num_context_requests, 1)
        self.assertEqual(ctx.context_chunk_size, 16)  # untouched

    def test_overshoot_shrinks_context_to_fit(self):
        # 100 gen tokens leave a 28-token budget; a 64-token last chunk does not
        # fit and must be shrunk to the largest block-aligned chunk that does.
        mgr = _make_manager(max_num_tokens=128, tokens_per_block=16)
        ctx = _FakeRequest(context_chunk_size=64, prompt_len=64)
        gen = _FakeRequest(py_beam_width=100)
        batch = _make_batch([ctx], [gen])

        mgr.fit_token_budget(batch)

        self.assertEqual(ctx.context_chunk_size, 16)
        self.assertLessEqual(_forward_tokens(mgr, batch), 128)

    def test_nothing_is_ever_dropped(self):
        # The defining property of the post-allocation trim: KV is allocated and
        # sequences are added, so a request may be shrunk but never removed.
        mgr = _make_manager(max_num_tokens=128, tokens_per_block=16)
        ctxs = [_FakeRequest(context_chunk_size=64, prompt_len=64) for _ in range(3)]
        gen = _FakeRequest(py_beam_width=100)
        batch = _make_batch(ctxs, [gen])

        mgr.fit_token_budget(batch)

        self.assertEqual(batch.num_context_requests, 3)
        for ctx in ctxs:
            self.assertIn(ctx, batch.context_requests)

    def test_shrink_keeps_chunk_end_block_aligned(self):
        # setPrepopulatedPromptLen asserts (pos + chunk) % tokens_per_block == 0
        # for every non-last chunk, to keep the KV cache unfragmented.
        mgr = _make_manager(max_num_tokens=100, tokens_per_block=16)
        ctx = _FakeRequest(context_chunk_size=96, context_current_position=32, prompt_len=128)
        gen = _FakeRequest(py_beam_width=53)
        batch = _make_batch([ctx], [gen])

        mgr.fit_token_budget(batch)

        self.assertLess(ctx.context_chunk_size, 96)
        self.assertEqual(
            (ctx.context_current_position + ctx.context_chunk_size) % 16,
            0,
            "chunk end must land on a block boundary",
        )

    def test_shrink_never_produces_a_zero_token_chunk(self):
        # A zero-token chunk leaves the request scheduled but computing nothing,
        # which never terminates. One block of progress is the floor even when
        # that overshoots the budget.
        mgr = _make_manager(max_num_tokens=128, tokens_per_block=16)
        ctx = _FakeRequest(context_chunk_size=64, prompt_len=64)
        gen = _FakeRequest(py_beam_width=127)  # leaves a 1-token budget
        batch = _make_batch([ctx], [gen])

        mgr.fit_token_budget(batch)

        self.assertEqual(ctx.context_chunk_size, 16)

    def test_shrink_rebins_to_chunking(self):
        # Shrinking flips is_last_context_chunk to False, so the request must
        # move out of context_requests_last_chunk -- otherwise downstream treats
        # it as a final chunk and appends generation / draft tokens to it.
        mgr = _make_manager(max_num_tokens=128, tokens_per_block=16)
        ctx = _FakeRequest(context_chunk_size=64, prompt_len=64)
        gen = _FakeRequest(py_beam_width=100)
        batch = _make_batch([ctx], [gen])
        self.assertEqual(batch.context_requests_last_chunk, [ctx])

        mgr.fit_token_budget(batch)

        self.assertEqual(batch.context_requests_last_chunk, [])
        self.assertEqual(batch.context_requests_chunking, [ctx])

    def test_shrink_drops_last_chunk_draft_tokens(self):
        # Draft tokens ride only on the last chunk, so a shrunk request stops
        # contributing them and the budget must account for that.
        mgr = _make_manager(max_num_tokens=128, tokens_per_block=16)
        ctx = _FakeRequest(context_chunk_size=64, prompt_len=64, py_draft_tokens=[1, 2, 3, 4])
        gen = _FakeRequest(py_beam_width=100)
        batch = _make_batch([ctx], [gen])
        self.assertEqual(mgr._request_forward_tokens(ctx, is_context=True), 68)

        mgr.fit_token_budget(batch)

        self.assertFalse(ctx.is_last_context_chunk)
        self.assertEqual(mgr._request_forward_tokens(ctx, is_context=True), 16)

    def test_sheds_the_last_chunk_first(self):
        # context_requests is chunking + last_chunk, so the trim walks last-chunk
        # requests first. That is the request whose cost the scheduler can have
        # under-charged (only a last chunk carries a reuse discount), so it is
        # the right one to repair; mid-prefill chunks are touched only if
        # trimming it is not enough.
        mgr = _make_manager(max_num_tokens=128, tokens_per_block=16)
        chunking = _FakeRequest(
            context_chunk_size=96, prompt_len=256
        )  # pos 0 + 96 != 256 -> chunking
        last = _FakeRequest(context_chunk_size=64, prompt_len=64)
        batch = _make_batch([chunking, last], [])
        self.assertEqual(batch.context_requests_chunking, [chunking])
        self.assertEqual(batch.context_requests_last_chunk, [last])

        mgr.fit_token_budget(batch)  # 160 tokens vs a 128 budget

        self.assertEqual(chunking.context_chunk_size, 96, "mid-prefill untouched")
        self.assertEqual(last.context_chunk_size, 32, "last chunk absorbed the excess")
        self.assertLessEqual(_forward_tokens(mgr, batch), 128)

    def test_shrinks_multiple_requests_when_one_is_not_enough(self):
        mgr = _make_manager(max_num_tokens=64, tokens_per_block=16)
        ctxs = [_FakeRequest(context_chunk_size=64, prompt_len=64) for _ in range(3)]
        batch = _make_batch(ctxs, [])

        mgr.fit_token_budget(batch)

        self.assertLessEqual(_forward_tokens(mgr, batch), 64)
        self.assertEqual(batch.num_context_requests, 3)

    def test_no_shrink_when_chunked_prefill_disabled(self):
        # A partial context chunk is only valid under chunked prefill; forcing
        # one produces an invalid forward pass. Nothing safe is left to do.
        mgr = _make_manager(max_num_tokens=128, tokens_per_block=16, enable_chunked_prefill=False)
        ctx = _FakeRequest(context_chunk_size=64, prompt_len=64)
        gen = _FakeRequest(py_beam_width=100)
        batch = _make_batch([ctx], [gen])

        mgr.fit_token_budget(batch)

        self.assertEqual(ctx.context_chunk_size, 64)
        self.assertEqual(batch.num_context_requests, 1)

    def test_mm_bidirectional_is_not_shrunk(self):
        # Splitting a bidirectional multimodal block silently breaks attention.
        mgr = _make_manager(max_num_tokens=128, tokens_per_block=16)
        ctx = _FakeRequest(context_chunk_size=64, prompt_len=64, mm_bidirectional=True)
        gen = _FakeRequest(py_beam_width=100)
        batch = _make_batch([ctx], [gen])

        mgr.fit_token_budget(batch)

        self.assertEqual(ctx.context_chunk_size, 64)

    def test_disagg_gen_init_requests_are_left_alone(self):
        # They only allocate/transfer KV cache and contribute no compute tokens.
        mgr = _make_manager(max_num_tokens=128, tokens_per_block=16)
        disagg = _FakeRequest(context_chunk_size=4096, is_disagg_generation_init_state=True)
        ctx = _FakeRequest(context_chunk_size=16, prompt_len=16)
        batch = _make_batch([disagg, ctx], [])

        mgr.fit_token_budget(batch)

        self.assertEqual(disagg.context_chunk_size, 4096)
        self.assertEqual(ctx.context_chunk_size, 16)

    def test_gen_only_batch_is_left_alone(self):
        # Generation cannot be shed, so a batch with no context requests returns
        # immediately -- keeping the executor loop's hottest path free of an
        # O(num_generation_requests) scan.
        mgr = _make_manager(max_num_tokens=8, tokens_per_block=16)
        batch = _make_batch([], [_FakeRequest(py_beam_width=64)])

        mgr.fit_token_budget(batch)  # must not raise

        self.assertEqual(len(batch.generation_requests), 1)

    def test_generation_alone_over_budget_does_not_raise(self):
        # There is nothing to shed, but raising here would be rank-local and
        # would deadlock the surviving ranks under attention DP. Warn and let
        # the forward pass report it.
        mgr = _make_manager(max_num_tokens=64, tokens_per_block=16)
        ctx = _FakeRequest(context_chunk_size=16, prompt_len=16)
        gen = _FakeRequest(py_beam_width=100)
        batch = _make_batch([ctx], [gen])

        mgr.fit_token_budget(batch)  # must not raise

        self.assertEqual(batch.num_context_requests, 1)

    def test_maybe_fit_token_budget_skips_draft_manager(self):
        # The draft-model engine builds inputs with a different token shape and
        # its budget is handled separately.
        gen = _FakeRequest(py_beam_width=100)

        mgr = _make_manager(max_num_tokens=128, tokens_per_block=16)
        mgr.is_draft = False
        ctx = _FakeRequest(context_chunk_size=64, prompt_len=64)
        mgr.maybe_fit_token_budget(_make_batch([ctx], [gen]))
        self.assertEqual(ctx.context_chunk_size, 16)

        mgr.is_draft = True
        ctx = _FakeRequest(context_chunk_size=64, prompt_len=64)
        mgr.maybe_fit_token_budget(_make_batch([ctx], [gen]))
        self.assertEqual(ctx.context_chunk_size, 64)

    def test_trim_runs_after_every_manager(self):
        # The trim must observe the batch as prepare_resources leaves it: only
        # after addSequence has run does context_chunk_size mean forward-pass
        # tokens (setPrepopulatedPromptLen advances context_current_position
        # past the reusable prefix). Registering the KV cache manager last
        # mirrors _util.py's move_to_end(KV_CACHE_MANAGER); the trim must still
        # run after that.
        observed = []

        target = _make_manager(max_num_tokens=128, tokens_per_block=16)
        target.prepare_resources = lambda batch: observed.append("kv_cache_manager")

        class _RecordingManager:
            def prepare_resources(self, batch):
                observed.append("draft_manager")

        ctx = _FakeRequest(context_chunk_size=64, prompt_len=64)
        gen = _FakeRequest(py_beam_width=100)
        batch = _make_batch([ctx], [gen])

        rm = ResourceManager(
            OrderedDict(
                [
                    (ResourceManagerType.DRAFT_KV_CACHE_MANAGER, _RecordingManager()),
                    (ResourceManagerType.KV_CACHE_MANAGER, target),
                ]
            )
        )
        rm.prepare_resources(batch)

        self.assertEqual(observed, ["draft_manager", "kv_cache_manager"])
        self.assertEqual(ctx.context_chunk_size, 16, "trim ran after both managers")

    def test_prepare_resources_trims(self):
        # prepare_resources is the only entry point; the executor loops must not
        # need their own call.
        target = _make_manager(max_num_tokens=128, tokens_per_block=16)
        target.prepare_resources = lambda batch: None

        ctx = _FakeRequest(context_chunk_size=64, prompt_len=64)
        gen = _FakeRequest(py_beam_width=100)
        batch = _make_batch([ctx], [gen])

        rm = ResourceManager(OrderedDict([(ResourceManagerType.KV_CACHE_MANAGER, target)]))
        rm.prepare_resources(batch)

        self.assertEqual(ctx.context_chunk_size, 16)


class TestInflightIdsSurviveTrim(unittest.TestCase):
    """The trim must not strand ids in PyExecutor's inflight set.

    ``_executor_loop_pp`` calls ``_add_inflight_ids`` before
    ``ResourceManager.prepare_resources`` and ``_remove_inflight_ids`` after, so
    the trim runs between them and moves shrunk requests out of
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

    def test_shrunk_context_requests_leave_no_inflight_ids(self):
        executor = self._bare_executor()
        mgr = _make_manager(max_num_tokens=128, tokens_per_block=16)

        gen = _FakeRequest(py_beam_width=100)  # leaves a 28-token budget
        # Starts as a last-chunk request, so it is registered inflight; the trim
        # then shrinks it to a block-aligned 16 and it stops being a last chunk.
        ctx = _FakeRequest(context_chunk_size=64, prompt_len=64)
        batch = _make_batch([ctx], [gen])

        executor._add_inflight_ids(batch)
        self.assertEqual(
            sorted(batch.added_inflight_req_ids),
            sorted([ctx.request_id, gen.request_id]),
        )

        mgr.fit_token_budget(batch)

        # Precondition for the regression: the batch really did change shape.
        self.assertEqual(ctx.context_chunk_size, 16)
        self.assertIn(ctx, batch.context_requests_chunking)
        self.assertEqual(batch.context_requests_last_chunk, [])

        executor._remove_inflight_ids(batch)

        for req in (ctx, gen):
            self.assertNotIn(
                req.request_id,
                executor.inflight_req_ids,
                f"request {req.request_id} left in the inflight set; the scheduler "
                "would never schedule it again",
            )
        self.assertEqual(batch.added_inflight_req_ids, [])

    def test_untrimmed_batch_round_trips(self):
        # The batch the trim leaves alone must behave exactly as before.
        executor = self._bare_executor()
        mgr = _make_manager(max_num_tokens=128, tokens_per_block=16)

        gen = _FakeRequest(py_beam_width=1)
        ctx = _FakeRequest(context_chunk_size=16)
        batch = _make_batch([ctx], [gen])

        executor._add_inflight_ids(batch)
        for req in (ctx, gen):
            self.assertIn(req.request_id, executor.inflight_req_ids)

        mgr.fit_token_budget(batch)
        self.assertEqual(batch.num_context_requests, 1)

        executor._remove_inflight_ids(batch)
        for req in (ctx, gen):
            self.assertNotIn(req.request_id, executor.inflight_req_ids)


class TestConnectorSeesTheTrimmedBatch(unittest.TestCase):
    """The KV connector must see the batch that will actually run.

    RequestData.num_scheduled_tokens is documented as "the number of scheduled
    tokens for the upcoming forward pass" and is built from context_chunk_size,
    so reporting the batch before the trim over-states it for every chunk the
    trim shrinks -- and a connector that decides what to save or offload from
    that count would publish KV for tokens the forward pass never computed.
    """

    class _RecordingKvManager:
        """Stands in for KVCacheManager: records when it is asked to publish."""

        def __init__(self, log, batch, shrink_to):
            self._log = log
            self._batch = batch
            self._shrink_to = shrink_to

        def prepare_resources(self, scheduled_batch):
            self._log.append(("prepare", self._chunks()))

        def maybe_fit_token_budget(self, scheduled_batch):
            for req in scheduled_batch.context_requests:
                req.context_chunk_size = self._shrink_to
            self._log.append(("trim", self._chunks()))

        def publish_connector_scheduler_output(self, scheduled_batch):
            self._log.append(("publish", self._chunks()))

        def _chunks(self):
            return [r.context_chunk_size for r in self._batch.context_requests]

    def _resource_manager(self, log, batch, shrink_to):
        return ResourceManager(
            OrderedDict(
                [
                    (
                        ResourceManagerType.KV_CACHE_MANAGER,
                        self._RecordingKvManager(log, batch, shrink_to),
                    )
                ]
            )
        )

    def test_the_connector_is_told_after_the_trim(self):
        log = []
        req = _FakeRequest(context_chunk_size=4096, prompt_len=4096)
        batch = _make_batch([req])

        self._resource_manager(log, batch, shrink_to=64).prepare_resources(batch)

        self.assertEqual([step for step, _ in log], ["prepare", "trim", "publish"])
        # The count the connector sees is the trimmed one, not the scheduled one.
        self.assertEqual(dict(log)["publish"], [64])

    def test_managers_without_a_connector_hook_are_skipped(self):
        rm = ResourceManager(OrderedDict([(ResourceManagerType.KV_CACHE_MANAGER, object())]))
        rm.prepare_resources(_make_batch())  # must not raise

    def test_publishing_is_a_no_op_without_a_connector(self):
        mgr = _make_manager(max_num_tokens=128, tokens_per_block=16)
        mgr.kv_connector_manager = None
        mgr.publish_connector_scheduler_output(_make_batch())  # must not raise

    def test_publishing_forwards_the_batch_to_the_connector(self):
        class _FakeConnector:
            def __init__(self):
                self.calls = []

            def build_scheduler_output(self, scheduled_batch, kv_cache_manager):
                self.calls.append((scheduled_batch, kv_cache_manager))

        mgr = _make_manager(max_num_tokens=128, tokens_per_block=16)
        mgr.kv_connector_manager = _FakeConnector()
        batch = _make_batch([_FakeRequest(context_chunk_size=16, prompt_len=16)])

        mgr.publish_connector_scheduler_output(batch)

        self.assertEqual(mgr.kv_connector_manager.calls, [(batch, mgr)])


if __name__ == "__main__":
    unittest.main()
