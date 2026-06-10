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

from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests


class _FakeRequest:
    """Minimal stand-in exposing only the attributes _fit_token_budget reads."""

    def __init__(
        self,
        *,
        context_chunk_size=0,
        is_last_context_chunk=True,
        py_beam_width=1,
        py_draft_tokens=None,
        is_disagg_generation_init_state=False,
        mm_bidirectional=False,
    ):
        self.context_chunk_size = context_chunk_size
        self.is_last_context_chunk = is_last_context_chunk
        self.py_beam_width = py_beam_width
        self.py_draft_tokens = py_draft_tokens
        self.is_disagg_generation_init_state = is_disagg_generation_init_state
        self.py_multimodal_data = {"mm_bidirectional_blocks": True} if mm_bidirectional else None


def _make_manager(max_num_tokens, tokens_per_block):
    # Skip the heavy (GPU-allocating) __init__; the method under test only
    # needs these two attributes plus its own (bound) helper methods.
    mgr = KVCacheManager.__new__(KVCacheManager)
    mgr.max_num_tokens = max_num_tokens
    mgr.tokens_per_block = tokens_per_block
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

    def test_generation_alone_over_budget_raises(self):
        mgr = _make_manager(max_num_tokens=128, tokens_per_block=16)
        gen = _FakeRequest(py_beam_width=200)
        batch = _make_batch([], [gen])

        with self.assertRaises(RuntimeError):
            mgr._fit_token_budget(batch)


if __name__ == "__main__":
    unittest.main()
