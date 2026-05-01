# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tier 1 mock unit tests for KVCacheV2Scheduler.

All KVCacheManagerV2, LlmRequest, and PeftCacheManager objects are mocked.
No GPU required.
"""

from unittest.mock import Mock, patch

import pytest

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm.llmapi.llm_args import CapacitySchedulerPolicy

# ---------------------------------------------------------------------------
# State value constants
# ---------------------------------------------------------------------------
ENCODER_INIT = LlmRequestState.ENCODER_INIT.value  # 1
DISAGG_GEN_INIT = LlmRequestState.DISAGG_GENERATION_INIT.value  # 8
CONTEXT_INIT = LlmRequestState.CONTEXT_INIT.value  # 10
GEN_IN_PROGRESS = LlmRequestState.GENERATION_IN_PROGRESS.value  # 13
GEN_TO_COMPLETE = LlmRequestState.GENERATION_TO_COMPLETE.value  # 14


# ---------------------------------------------------------------------------
# Mock factories
# ---------------------------------------------------------------------------
def make_gen_request(
    request_id, beam_width=1, num_draft_tokens=0, lora_task_id=None, is_first_context_chunk=False
):
    req = Mock()
    req.request_id = request_id
    req.py_request_id = request_id
    req.state_value = GEN_IN_PROGRESS
    req.get_beam_width_by_iter.return_value = beam_width
    req.num_draft_tokens = num_draft_tokens
    req.has_draft_tokens = num_draft_tokens > 0
    req.py_draft_tokens = [0] * num_draft_tokens if num_draft_tokens > 0 else []
    req.lora_task_id = lora_task_id
    req.is_context_init_state = False
    req.is_generation_in_progress_state = True
    req.is_first_context_chunk = is_first_context_chunk
    return req


def make_ctx_request(
    request_id,
    context_remaining_length,
    prompt_len=None,
    num_draft_tokens=0,
    is_first_context_chunk=True,
    is_last_context_chunk=True,
    lora_task_id=None,
    encoder_output_len=None,
):
    req = Mock()
    req.request_id = request_id
    req.py_request_id = request_id
    req.state_value = CONTEXT_INIT
    req.context_remaining_length = context_remaining_length
    req.prompt_len = prompt_len or context_remaining_length
    req.context_current_position = 0
    req.num_draft_tokens = num_draft_tokens
    req.has_draft_tokens = num_draft_tokens > 0
    req.py_draft_tokens = [0] * num_draft_tokens if num_draft_tokens > 0 else []
    req.is_first_context_chunk = is_first_context_chunk
    req.is_last_context_chunk = is_last_context_chunk
    req.context_chunk_size = 0
    req.lora_task_id = lora_task_id
    req.is_context_init_state = True
    req.is_generation_in_progress_state = False
    req.encoder_output_len = encoder_output_len
    return req


def make_encoder_request(request_id, encoder_output_len, lora_task_id=None):
    req = Mock()
    req.request_id = request_id
    req.py_request_id = request_id
    req.state_value = ENCODER_INIT
    req.encoder_output_len = encoder_output_len
    req.lora_task_id = lora_task_id
    req.is_context_init_state = False
    req.is_generation_in_progress_state = False
    req.is_first_context_chunk = True
    return req


def make_disagg_request(request_id, context_remaining_length=1, num_draft_tokens=0):
    req = Mock()
    req.request_id = request_id
    req.py_request_id = request_id
    req.state_value = DISAGG_GEN_INIT
    req.is_context_init_state = False
    req.is_generation_in_progress_state = False
    req.is_first_context_chunk = True
    req.context_remaining_length = context_remaining_length
    req.num_draft_tokens = num_draft_tokens
    req.has_draft_tokens = num_draft_tokens > 0
    req.py_draft_tokens = [0] * num_draft_tokens if num_draft_tokens > 0 else []
    return req


def make_filtered_request(request_id, state_value=0):
    req = Mock()
    req.request_id = request_id
    req.py_request_id = request_id
    req.state_value = state_value
    req.is_context_init_state = False
    req.is_generation_in_progress_state = False
    req.is_first_context_chunk = True
    return req


class _KVCacheMap(dict):
    """Dict that auto-creates active mock KV cache entries on first access.

    This mirrors real KVCacheManagerV2 behaviour where every scheduled request
    has a kv_cache entry. Tests that need a suspended entry can explicitly set
    ``kv_cache_map[req_id].is_active = False``.
    """

    def __missing__(self, key):
        entry = Mock()
        entry.is_active = True
        self[key] = entry
        return entry


def make_kv_cache_manager(
    tokens_per_block=64,
    prepare_context_fn=None,
    resize_context_fn=None,
    try_allocate_generation_fn=None,
):
    mgr = Mock()
    mgr.tokens_per_block = tokens_per_block
    mgr.kv_cache_map = _KVCacheMap()
    mgr.prepare_context.side_effect = prepare_context_fn or (lambda req: True)
    mgr.resize_context.side_effect = resize_context_fn or (lambda req, n: True)
    mgr.try_allocate_generation.side_effect = try_allocate_generation_fn or (lambda req: True)
    mgr.suspend_request.return_value = None
    mgr.is_request_active.side_effect = lambda req_id: mgr.kv_cache_map[req_id].is_active
    return mgr


def make_peft_cache_manager(max_device_pages, pages_per_task=1):
    mgr = Mock()
    mgr.max_device_pages = max_device_pages
    mgr.determine_num_pages.return_value = pages_per_task
    return mgr


def make_scheduler(
    kv_cache_manager,
    max_batch_size=100,
    max_num_tokens=1024,
    ctx_chunk_config=None,
    peft_cache_manager=None,
    scheduler_capacity=None,
    no_schedule_until_state=None,
    no_schedule_after_state=None,
):
    """Create KVCacheV2Scheduler, patching isinstance check for mock mgr."""
    from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import KVCacheV2Scheduler

    with patch(
        "tensorrt_llm._torch.pyexecutor.resource_manager.KVCacheManagerV2",
        new=type(kv_cache_manager),
    ):
        kwargs = {}
        if no_schedule_until_state is not None:
            kwargs["no_schedule_until_state"] = no_schedule_until_state
        if no_schedule_after_state is not None:
            kwargs["no_schedule_after_state"] = no_schedule_after_state
        return KVCacheV2Scheduler(
            max_batch_size=max_batch_size,
            max_num_tokens=max_num_tokens,
            kv_cache_manager=kv_cache_manager,
            scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
            ctx_chunk_config=ctx_chunk_config,
            peft_cache_manager=peft_cache_manager,
            scheduler_capacity=scheduler_capacity,
            **kwargs,
        )


def make_encoder_scheduler(kv_cache_manager, **kwargs):
    """Scheduler with state range widened to include ENCODER_INIT (matches
    C++ trtEncoderModel pattern)."""
    return make_scheduler(
        kv_cache_manager,
        no_schedule_until_state=LlmRequestState.ENCODER_INIT,
        **kwargs,
    )


def ids(reqs):
    return [r.request_id for r in reqs]


# ===========================================================================
# Construction & Configuration
# ===========================================================================


class TestConstruction:
    def test_valid_construction(self):
        mgr = make_kv_cache_manager()
        sched = make_scheduler(mgr)
        assert sched.max_num_tokens == 1024
        assert sched.max_num_requests == 100

    def test_reject_non_v2_manager(self):
        from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import KVCacheV2Scheduler

        with pytest.raises(AssertionError, match="KVCacheManagerV2"):
            KVCacheV2Scheduler(
                max_batch_size=10,
                max_num_tokens=100,
                kv_cache_manager=Mock(),  # not KVCacheManagerV2
                scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
            )

    def test_scheduler_capacity_overrides(self):
        mgr = make_kv_cache_manager()
        sched = make_scheduler(mgr, max_batch_size=100, scheduler_capacity=5)
        assert sched.max_num_requests == 5

    def test_chunk_unit_no_alignment(self):
        mgr = make_kv_cache_manager(tokens_per_block=64)
        sched = make_scheduler(mgr, ctx_chunk_config=(None, 100))
        # chunk_unit_size is used as-is, no alignment to tokens_per_block
        assert sched.chunk_unit_size == 100

    def test_chunk_unit_already_aligned(self):
        mgr = make_kv_cache_manager(tokens_per_block=64)
        sched = make_scheduler(mgr, ctx_chunk_config=(None, 128))
        assert sched.chunk_unit_size == 128

    def test_max_num_tokens_none(self):
        mgr = make_kv_cache_manager()
        sched = make_scheduler(mgr, max_num_tokens=None)
        assert sched.max_num_tokens is None


# ===========================================================================
# Token Budget Limits
# ===========================================================================


class TestTokenBudget:
    def test_gen_budget_exhausted(self):
        mgr = make_kv_cache_manager()
        sched = make_scheduler(mgr, max_num_tokens=3)
        reqs = [make_gen_request(i) for i in range(5)]
        out = sched.schedule_request(reqs, set())
        assert len(out.generation_requests) == 3
        assert ids(out.generation_requests) == [0, 1, 2]

    def test_ctx_budget_exhausted(self):
        mgr = make_kv_cache_manager()
        sched = make_scheduler(mgr, max_num_tokens=100)
        reqs = [make_ctx_request(0, context_remaining_length=200)]
        out = sched.schedule_request(reqs, set())
        assert len(out.context_requests) == 0

    def test_encoder_budget_exhausted(self):
        mgr = make_kv_cache_manager()
        sched = make_encoder_scheduler(mgr, max_num_tokens=100)
        reqs = [make_encoder_request(0, encoder_output_len=200)]
        out = sched.schedule_request(reqs, set())
        assert len(out.context_requests) == 0

    def test_gen_with_draft_tokens(self):
        mgr = make_kv_cache_manager()
        sched = make_scheduler(mgr, max_num_tokens=5)
        reqs = [
            make_gen_request(0, num_draft_tokens=3),  # 1+3=4
            make_gen_request(1, num_draft_tokens=3),  # 1+3=4, total 8>5
        ]
        out = sched.schedule_request(reqs, set())
        assert len(out.generation_requests) == 1
        assert out.generation_requests[0].request_id == 0

    def test_ctx_with_draft_no_chunk(self):
        """Context(80) + draft(5) = 85, fits in budget=100."""
        mgr = make_kv_cache_manager()
        sched = make_scheduler(mgr, max_num_tokens=100)
        reqs = [make_ctx_request(0, context_remaining_length=80, num_draft_tokens=5)]
        out = sched.schedule_request(reqs, set())
        assert len(out.context_requests) == 1

    def test_mixed_gen_ctx_budget(self):
        mgr = make_kv_cache_manager()
        sched = make_scheduler(mgr, max_num_tokens=110)
        reqs = [
            make_gen_request(0),  # 1
            make_ctx_request(1, context_remaining_length=100),  # 100
        ]
        out = sched.schedule_request(reqs, set())
        assert len(out.generation_requests) == 1
        assert len(out.context_requests) == 1

    def test_unlimited_budget(self):
        mgr = make_kv_cache_manager()
        sched = make_scheduler(mgr, max_num_tokens=None)
        reqs = [make_gen_request(i) for i in range(10)]
        out = sched.schedule_request(reqs, set())
        assert len(out.generation_requests) == 10

    def test_budget_exactly_full(self):
        mgr = make_kv_cache_manager()
        sched = make_scheduler(mgr, max_num_tokens=3)
        reqs = [make_gen_request(i) for i in range(3)]
        out = sched.schedule_request(reqs, set())
        assert len(out.generation_requests) == 3

    def test_batch_size_limit(self):
        mgr = make_kv_cache_manager()
        sched = make_scheduler(mgr, max_batch_size=2)
        reqs = [make_gen_request(i) for i in range(3)]
        out = sched.schedule_request(reqs, set())
        assert len(out.generation_requests) == 2

    def test_combined_batch_and_budget(self):
        mgr = make_kv_cache_manager()
        sched = make_scheduler(mgr, max_batch_size=3, max_num_tokens=2)
        reqs = [make_gen_request(i) for i in range(5)]
        out = sched.schedule_request(reqs, set())
        assert len(out.generation_requests) == 2  # budget limit first


# ===========================================================================
# Draft Token Budget Interaction (Non-Chunked)
# ===========================================================================


class TestDraftTokenBudget:
    def test_ctx_fits_draft_exceeds_budget(self):
        """ctx(80) fits in budget=100, but 80+30=110 > 100.
        V2 scheduler checks ctx+draft as a single unit, so this breaks.
        """
        mgr = make_kv_cache_manager(tokens_per_block=64)
        sched = make_scheduler(mgr, max_num_tokens=100)
        req = make_ctx_request(0, context_remaining_length=80, num_draft_tokens=30)
        out = sched.schedule_request([req], set())
        assert len(out.context_requests) == 0

    def test_ctx_fits_exactly_draft_pushes_over(self):
        """ctx(100) fits in budget=100, but 100+5=105 > 100.
        V2 scheduler checks ctx+draft as a single unit, so this breaks.
        """
        mgr = make_kv_cache_manager(tokens_per_block=64)
        sched = make_scheduler(mgr, max_num_tokens=100)
        req = make_ctx_request(0, context_remaining_length=100, num_draft_tokens=5)
        out = sched.schedule_request([req], set())
        assert len(out.context_requests) == 0

    def test_ctx_plus_draft_fits_exactly(self):
        """ctx(80) + draft(20) = 100 == budget. Should schedule with all drafts."""
        mgr = make_kv_cache_manager(tokens_per_block=64)
        sched = make_scheduler(mgr, max_num_tokens=100)
        req = make_ctx_request(0, context_remaining_length=80, num_draft_tokens=20)
        out = sched.schedule_request([req], set())
        assert len(out.context_requests) == 1

    def test_ctx_alone_exceeds_budget(self):
        """ctx(200) > budget=100. Should break."""
        mgr = make_kv_cache_manager()
        sched = make_scheduler(mgr, max_num_tokens=100)
        req = make_ctx_request(0, context_remaining_length=200, num_draft_tokens=5)
        out = sched.schedule_request([req], set())
        assert len(out.context_requests) == 0

    def test_gen_with_draft_exceeds_budget(self):
        """gen(beam=1, draft=5) → 1+5=6 > budget=3. Break."""
        mgr = make_kv_cache_manager()
        sched = make_scheduler(mgr, max_num_tokens=3)
        req = make_gen_request(0, num_draft_tokens=5)
        out = sched.schedule_request([req], set())
        assert len(out.generation_requests) == 0


# ===========================================================================
# KV Cache Allocation Failures
# ===========================================================================


class TestKVCacheFailuresGen:
    """Generation KV failures."""

    def test_gen_alloc_succeeds(self):
        mgr = make_kv_cache_manager()
        sched = make_scheduler(mgr, max_num_tokens=100)
        reqs = [make_gen_request(i) for i in range(3)]
        out = sched.schedule_request(reqs, set())
        assert len(out.generation_requests) == 3

    def test_gen_alloc_fails_no_victim(self):
        call_count = [0]

        def alloc_fn(req):
            call_count[0] += 1
            return call_count[0] <= 1  # first succeeds, second fails

        mgr = make_kv_cache_manager(try_allocate_generation_fn=alloc_fn)
        sched = make_scheduler(mgr, max_num_tokens=100)
        # gen0 ok, gen1 fails, gen2 is first-chunk ctx (not evictable)
        reqs = [
            make_gen_request(0),
            make_gen_request(1),
            make_ctx_request(2, context_remaining_length=10),  # first chunk, not evictable
        ]
        out = sched.schedule_request(reqs, set())
        assert len(out.generation_requests) == 1
        assert out.generation_requests[0].request_id == 0

    def test_gen_alloc_fails_evict_succeeds(self):
        """gen fails, started req at tail, retry succeeds after eviction."""
        call_count = [0]

        def alloc_fn(req):
            call_count[0] += 1
            # First call (gen0): success
            # Second call (gen1): fail
            # Third call (gen1 retry after evict): success
            return call_count[0] != 2

        mgr = make_kv_cache_manager(try_allocate_generation_fn=alloc_fn)
        sched = make_scheduler(mgr, max_num_tokens=100)
        victim = make_gen_request(99)  # started, at tail → evictable
        reqs = [make_gen_request(0), make_gen_request(1), victim]
        out = sched.schedule_request(reqs, set())
        assert ids(out.generation_requests) == [0, 1]
        assert ids(out.paused_requests) == [99]
        mgr.suspend_request.assert_called_once_with(victim)

    def test_gen_alloc_fails_evict_insufficient(self):
        """gen fails, evict victim, retry still fails → self-evict."""
        call_count = [0]

        def alloc_fn(req):
            call_count[0] += 1
            return call_count[0] <= 1  # only first succeeds

        mgr = make_kv_cache_manager(try_allocate_generation_fn=alloc_fn)
        sched = make_scheduler(mgr, max_num_tokens=100)
        victim = make_gen_request(99)
        reqs = [make_gen_request(0), make_gen_request(1), victim]
        out = sched.schedule_request(reqs, set())
        assert ids(out.generation_requests) == [0]
        # gen99 evicted as victim for gen1; gen1 self-evicts after
        assert set(ids(out.paused_requests)) == {1, 99}

    def test_multiple_evictions_needed(self):
        """gen fails, 2 victims needed to free enough space."""
        call_count = [0]

        def alloc_fn(req):
            call_count[0] += 1
            # gen0: ok(1), gen1: fail(2), retry: fail(3), retry2: ok(4)
            return call_count[0] not in (2, 3)

        mgr = make_kv_cache_manager(try_allocate_generation_fn=alloc_fn)
        sched = make_scheduler(mgr, max_num_tokens=100)
        v1 = make_gen_request(98)
        v2 = make_gen_request(99)
        reqs = [make_gen_request(0), make_gen_request(1), v1, v2]
        out = sched.schedule_request(reqs, set())
        assert ids(out.generation_requests) == [0, 1]
        assert len(out.paused_requests) == 2

    def test_gen_fail_does_not_break_immediately(self):
        """[gen1_ok, gen2_fail(no victim)] → gen1 scheduled, gen2 causes break."""
        call_count = [0]

        def alloc_fn(req):
            call_count[0] += 1
            return call_count[0] <= 1

        mgr = make_kv_cache_manager(try_allocate_generation_fn=alloc_fn)
        sched = make_scheduler(mgr, max_num_tokens=100)
        reqs = [make_gen_request(0), make_gen_request(1)]
        out = sched.schedule_request(reqs, set())
        assert ids(out.generation_requests) == [0]

    def test_suspended_request_not_evictable(self):
        """A started request with suspended (inactive) KV cache is not a valid victim."""
        call_count = [0]

        def alloc_fn(req):
            call_count[0] += 1
            return call_count[0] <= 1  # only first succeeds

        mgr = make_kv_cache_manager(try_allocate_generation_fn=alloc_fn)
        sched = make_scheduler(mgr, max_num_tokens=100)
        victim = make_gen_request(99)
        # Mark victim's KV cache as suspended (inactive)
        mgr.kv_cache_map[victim.py_request_id].is_active = False
        reqs = [make_gen_request(0), make_gen_request(1), victim]
        out = sched.schedule_request(reqs, set())
        # gen0 succeeds, gen1 fails, victim is not evictable (suspended)
        # → gen1 self-evicts, victim is not in paused list from eviction
        assert ids(out.generation_requests) == [0]
        assert 99 not in ids(out.paused_requests)


class TestKVCacheFailuresCtx:
    """Context KV failures (non-chunked)."""

    def test_ctx_prepare_fails(self):
        mgr = make_kv_cache_manager(prepare_context_fn=lambda req: False)
        sched = make_scheduler(mgr, max_num_tokens=1000)
        reqs = [make_ctx_request(0, context_remaining_length=100)]
        out = sched.schedule_request(reqs, set())
        assert len(out.context_requests) == 0

    def test_ctx_resize_fails_continue(self):
        """resize fails → ctx skipped (continue), not break."""
        mgr = make_kv_cache_manager(resize_context_fn=lambda req, n: False)
        sched = make_scheduler(mgr, max_num_tokens=1000)
        reqs = [
            make_ctx_request(0, context_remaining_length=1000),
            make_gen_request(1),
        ]
        out = sched.schedule_request(reqs, set())
        assert len(out.context_requests) == 0
        # Gen after failed ctx should still be scheduled
        assert len(out.generation_requests) == 1

    def test_ctx_resize_fail_then_gen_succeeds(self):
        mgr = make_kv_cache_manager(resize_context_fn=lambda req, n: False)
        sched = make_scheduler(mgr, max_num_tokens=1000)
        reqs = [make_ctx_request(0, 500), make_gen_request(1)]
        out = sched.schedule_request(reqs, set())
        assert len(out.context_requests) == 0
        assert ids(out.generation_requests) == [1]

    def test_ctx_resize_fail_then_smaller_ctx(self):
        call_count = [0]

        def resize_fn(req, n):
            call_count[0] += 1
            return call_count[0] > 1  # first fails, second succeeds

        mgr = make_kv_cache_manager(resize_context_fn=resize_fn)
        sched = make_scheduler(mgr, max_num_tokens=1000)
        reqs = [
            make_ctx_request(0, context_remaining_length=1000),
            make_ctx_request(1, context_remaining_length=100),
        ]
        out = sched.schedule_request(reqs, set())
        assert ids(out.context_requests) == [1]

    def test_zero_waste_failed_ctx_budget(self):
        """Failed ctx doesn't consume budget — next ctx gets full budget."""
        call_count = [0]

        def resize_fn(req, n):
            call_count[0] += 1
            return call_count[0] > 1

        mgr = make_kv_cache_manager(resize_context_fn=resize_fn)
        sched = make_scheduler(mgr, max_num_tokens=500)
        reqs = [
            make_ctx_request(0, context_remaining_length=400),
            make_ctx_request(1, context_remaining_length=500),  # needs full 500
        ]
        out = sched.schedule_request(reqs, set())
        assert ids(out.context_requests) == [1]


class TestKVCacheFailuresCtxChunked:
    """Context KV failures (chunked)."""

    def test_chunked_prepare_fails(self):
        mgr = make_kv_cache_manager(
            tokens_per_block=64,
            prepare_context_fn=lambda req: False,
        )
        sched = make_scheduler(mgr, max_num_tokens=1000, ctx_chunk_config=(None, 64))
        reqs = [make_ctx_request(0, context_remaining_length=500)]
        out = sched.schedule_request(reqs, set())
        assert len(out.context_requests) == 0

    def test_chunked_resize_fails(self):
        mgr = make_kv_cache_manager(
            tokens_per_block=64,
            resize_context_fn=lambda req, n: False,
        )
        sched = make_scheduler(mgr, max_num_tokens=1000, ctx_chunk_config=(None, 64))
        reqs = [make_ctx_request(0, context_remaining_length=500)]
        out = sched.schedule_request(reqs, set())
        assert len(out.context_requests) == 0

    def test_chunked_fail_then_gen(self):
        mgr = make_kv_cache_manager(
            tokens_per_block=64,
            resize_context_fn=lambda req, n: False,
        )
        sched = make_scheduler(mgr, max_num_tokens=1000, ctx_chunk_config=(None, 64))
        reqs = [make_ctx_request(0, 500), make_gen_request(1)]
        out = sched.schedule_request(reqs, set())
        assert len(out.context_requests) == 0
        assert ids(out.generation_requests) == [1]


class TestKVCacheFailuresEncoder:
    """Encoder KV failures."""

    def test_encoder_prepare_fails(self):
        mgr = make_kv_cache_manager(prepare_context_fn=lambda req: False)
        sched = make_encoder_scheduler(mgr, max_num_tokens=1000)
        reqs = [make_encoder_request(0, encoder_output_len=100)]
        out = sched.schedule_request(reqs, set())
        assert len(out.context_requests) == 0

    def test_encoder_resize_fails(self):
        mgr = make_kv_cache_manager(resize_context_fn=lambda req, n: False)
        sched = make_encoder_scheduler(mgr, max_num_tokens=1000)
        reqs = [make_encoder_request(0, encoder_output_len=100)]
        out = sched.schedule_request(reqs, set())
        assert len(out.context_requests) == 0


# ===========================================================================
# Eviction (MAX_UTILIZATION)
# ===========================================================================


class TestEviction:
    def test_evict_gen_from_tail(self):
        call_count = [0]

        def alloc_fn(req):
            call_count[0] += 1
            return call_count[0] != 1  # first fails, retry succeeds

        mgr = make_kv_cache_manager(try_allocate_generation_fn=alloc_fn)
        sched = make_scheduler(mgr, max_num_tokens=100)
        victim = make_gen_request(99)  # gen in progress → evictable
        reqs = [make_gen_request(0), victim]
        out = sched.schedule_request(reqs, set())
        assert ids(out.generation_requests) == [0]
        assert ids(out.paused_requests) == [99]

    def test_evict_nonfirst_ctx_chunk(self):
        call_count = [0]

        def alloc_fn(req):
            call_count[0] += 1
            return call_count[0] != 1

        mgr = make_kv_cache_manager(try_allocate_generation_fn=alloc_fn)
        sched = make_scheduler(mgr, max_num_tokens=100)
        # Non-first ctx chunk is evictable (is_context_init_state=True, is_first=False)
        victim = make_ctx_request(99, context_remaining_length=100, is_first_context_chunk=False)
        reqs = [make_gen_request(0), victim]
        out = sched.schedule_request(reqs, set())
        assert ids(out.generation_requests) == [0]
        assert ids(out.paused_requests) == [99]

    def test_first_chunk_ctx_not_evictable(self):
        mgr = make_kv_cache_manager(
            try_allocate_generation_fn=lambda req: False,
        )
        sched = make_scheduler(mgr, max_num_tokens=100)
        victim = make_ctx_request(99, context_remaining_length=100, is_first_context_chunk=True)
        reqs = [make_gen_request(0), victim]
        out = sched.schedule_request(reqs, set())
        assert len(out.generation_requests) == 0
        # gen0 self-evicts; break stops loop before ctx99
        assert ids(out.paused_requests) == [0]
        assert len(out.context_requests) == 0

    def test_evicted_in_paused_requests(self):
        call_count = [0]

        def alloc_fn(req):
            call_count[0] += 1
            return call_count[0] != 1

        mgr = make_kv_cache_manager(try_allocate_generation_fn=alloc_fn)
        sched = make_scheduler(mgr, max_num_tokens=100)
        victim = make_gen_request(99)
        reqs = [make_gen_request(0), victim]
        out = sched.schedule_request(reqs, set())
        assert victim in out.paused_requests

    def test_suspend_called_on_victim(self):
        call_count = [0]

        def alloc_fn(req):
            call_count[0] += 1
            return call_count[0] != 1

        mgr = make_kv_cache_manager(try_allocate_generation_fn=alloc_fn)
        sched = make_scheduler(mgr, max_num_tokens=100)
        victim = make_gen_request(99)
        reqs = [make_gen_request(0), victim]
        sched.schedule_request(reqs, set())
        mgr.suspend_request.assert_called_once_with(victim)

    def test_unknown_state_not_evictable(self):
        """UNKNOWN state at tail is not evictable; gen0 self-evicts."""
        mgr = make_kv_cache_manager(
            try_allocate_generation_fn=lambda req: False,
        )
        sched = make_scheduler(mgr, max_num_tokens=100)
        tail = make_filtered_request(99, state_value=0)
        reqs = [make_gen_request(0), tail]
        out = sched.schedule_request(reqs, set())
        assert len(out.generation_requests) == 0
        # gen0 self-evicts; UNKNOWN-state tail is neither evicted nor scheduled
        assert ids(out.paused_requests) == [0]

    def test_multiple_evictions_order(self):
        """victim2 evicted first (tail), retry fail, victim1 evicted, retry success."""
        call_count = [0]

        def alloc_fn(req):
            call_count[0] += 1
            # gen0: fail(1), retry after v2: fail(2), retry after v1: ok(3)
            return call_count[0] == 3

        mgr = make_kv_cache_manager(try_allocate_generation_fn=alloc_fn)
        sched = make_scheduler(mgr, max_num_tokens=100)
        v1 = make_gen_request(98)
        v2 = make_gen_request(99)
        reqs = [make_gen_request(0), v1, v2]
        out = sched.schedule_request(reqs, set())
        assert ids(out.generation_requests) == [0]
        assert len(out.paused_requests) == 2
        # v2 evicted first (higher index), then v1
        assert v2 in out.paused_requests
        assert v1 in out.paused_requests

    def test_req_it_end_shrinks_after_eviction(self):
        """After eviction, requests beyond victim index not visited."""
        call_count = [0]

        def alloc_fn(req):
            call_count[0] += 1
            return call_count[0] != 1

        mgr = make_kv_cache_manager(try_allocate_generation_fn=alloc_fn)
        sched = make_scheduler(mgr, max_num_tokens=100)
        victim = make_gen_request(5)  # at index 1, evictable
        # index 2: first-chunk ctx is NOT evictable, so eviction search
        # skips it and finds victim at index 1 instead.
        gen_after = make_ctx_request(6, context_remaining_length=100, is_first_context_chunk=True)
        reqs = [make_gen_request(0), victim, gen_after]
        out = sched.schedule_request(reqs, set())
        assert ids(out.generation_requests) == [0]
        assert ids(out.paused_requests) == [5]

    def test_self_eviction_on_alloc_fail(self):
        """Gen request self-evicts when alloc fails and no victims exist."""
        mgr = make_kv_cache_manager(
            try_allocate_generation_fn=lambda req: False,
        )
        sched = make_scheduler(mgr, max_num_tokens=100)
        # Only one gen req — alloc fails, no victims → self-evicts
        reqs = [make_gen_request(0)]
        out = sched.schedule_request(reqs, set())
        assert len(out.generation_requests) == 0
        assert ids(out.paused_requests) == [0]

    def test_self_eviction_no_started_in_range(self):
        """Gen self-evicts when all behind it are first-chunk ctx (not evictable)."""
        mgr = make_kv_cache_manager(
            try_allocate_generation_fn=lambda req: False,
        )
        sched = make_scheduler(mgr, max_num_tokens=200)
        reqs = [
            make_gen_request(0),
            make_ctx_request(1, 50, is_first_context_chunk=True),
            make_ctx_request(2, 50, is_first_context_chunk=True),
        ]
        out = sched.schedule_request(reqs, set())
        assert len(out.generation_requests) == 0
        # gen0 self-evicts; break stops loop before ctx1, ctx2
        assert ids(out.paused_requests) == [0]
        assert len(out.context_requests) == 0


# ===========================================================================
# PEFT / LoRA
# ===========================================================================


class TestPEFT:
    def test_no_peft_manager(self):
        mgr = make_kv_cache_manager()
        sched = make_scheduler(mgr, peft_cache_manager=None)
        reqs = [make_gen_request(0, lora_task_id=1)]
        out = sched.schedule_request(reqs, set())
        assert len(out.generation_requests) == 1

    def test_peft_sufficient(self):
        mgr = make_kv_cache_manager()
        peft = make_peft_cache_manager(max_device_pages=10, pages_per_task=1)
        sched = make_scheduler(mgr, peft_cache_manager=peft)
        reqs = [
            make_ctx_request(0, 100, lora_task_id=1),
            make_ctx_request(1, 100, lora_task_id=2),
        ]
        out = sched.schedule_request(reqs, set())
        assert len(out.context_requests) == 2

    def test_peft_exceeded(self):
        mgr = make_kv_cache_manager()
        peft = make_peft_cache_manager(max_device_pages=1, pages_per_task=1)
        sched = make_scheduler(mgr, peft_cache_manager=peft)
        reqs = [
            make_ctx_request(0, 100, lora_task_id=1),
            make_ctx_request(1, 100, lora_task_id=2),
        ]
        out = sched.schedule_request(reqs, set())
        assert len(out.context_requests) == 1
        assert out.context_requests[0].request_id == 0

    def test_same_task_id_deduplication(self):
        mgr = make_kv_cache_manager()
        peft = make_peft_cache_manager(max_device_pages=1, pages_per_task=1)
        sched = make_scheduler(mgr, peft_cache_manager=peft)
        reqs = [
            make_ctx_request(0, 100, lora_task_id=1),
            make_ctx_request(1, 100, lora_task_id=1),
            make_ctx_request(2, 100, lora_task_id=1),
        ]
        out = sched.schedule_request(reqs, set())
        assert len(out.context_requests) == 3  # same task_id, counted once

    def test_peft_fail_before_kv_alloc(self):
        mgr = make_kv_cache_manager()
        peft = make_peft_cache_manager(max_device_pages=0, pages_per_task=1)
        sched = make_scheduler(mgr, peft_cache_manager=peft)
        reqs = [make_ctx_request(0, 100, lora_task_id=1)]
        out = sched.schedule_request(reqs, set())
        assert len(out.context_requests) == 0
        mgr.prepare_context.assert_not_called()

    def test_peft_commit_only_after_kv_success(self):
        """If resize fails, PEFT pages NOT committed."""
        call_count = [0]

        def resize_fn(req, n):
            call_count[0] += 1
            return call_count[0] > 1  # first fails

        mgr = make_kv_cache_manager(resize_context_fn=resize_fn)
        peft = make_peft_cache_manager(max_device_pages=1, pages_per_task=1)
        sched = make_scheduler(mgr, peft_cache_manager=peft)
        reqs = [
            make_ctx_request(0, 100, lora_task_id=1),  # resize fails
            make_ctx_request(1, 100, lora_task_id=2),  # should see full PEFT budget
        ]
        out = sched.schedule_request(reqs, set())
        # req0 skipped (resize fail), req1 should get PEFT budget
        assert ids(out.context_requests) == [1]

    def test_no_lora_task_id(self):
        mgr = make_kv_cache_manager()
        peft = make_peft_cache_manager(max_device_pages=1, pages_per_task=1)
        sched = make_scheduler(mgr, peft_cache_manager=peft)
        reqs = [make_ctx_request(0, 100, lora_task_id=None)]
        out = sched.schedule_request(reqs, set())
        assert len(out.context_requests) == 1

    def test_gen_peft_check(self):
        """Gen with lora, PEFT exhausted → break."""
        mgr = make_kv_cache_manager()
        peft = make_peft_cache_manager(max_device_pages=0, pages_per_task=1)
        sched = make_scheduler(mgr, peft_cache_manager=peft)
        reqs = [make_gen_request(0, lora_task_id=1)]
        out = sched.schedule_request(reqs, set())
        assert len(out.generation_requests) == 0

    def test_encoder_peft_check(self):
        """Encoder with lora, PEFT exhausted → break."""
        mgr = make_kv_cache_manager()
        peft = make_peft_cache_manager(max_device_pages=0, pages_per_task=1)
        sched = make_encoder_scheduler(mgr, peft_cache_manager=peft)
        reqs = [make_encoder_request(0, encoder_output_len=100, lora_task_id=1)]
        out = sched.schedule_request(reqs, set())
        assert len(out.context_requests) == 0

    def test_mixed_peft_gen_claims_reduce_ctx(self):
        """[gen(task1), ctx(task2)], pages for 2 total → both ok."""
        mgr = make_kv_cache_manager()
        peft = make_peft_cache_manager(max_device_pages=2, pages_per_task=1)
        sched = make_scheduler(mgr, peft_cache_manager=peft)
        reqs = [
            make_gen_request(0, lora_task_id=1),
            make_ctx_request(1, 100, lora_task_id=2),
        ]
        out = sched.schedule_request(reqs, set())
        assert len(out.generation_requests) == 1
        assert len(out.context_requests) == 1

    def test_pages_per_task_varies(self):
        """determine_num_pages returns different values per req."""
        mgr = make_kv_cache_manager()
        peft = make_peft_cache_manager(max_device_pages=5, pages_per_task=1)

        def determine_pages(req):
            return 3  # 3+3=6 > 5

        peft.determine_num_pages.side_effect = determine_pages
        sched = make_scheduler(mgr, peft_cache_manager=peft)
        reqs = [
            make_ctx_request(0, 100, lora_task_id=1),
            make_ctx_request(1, 100, lora_task_id=2),
        ]
        out = sched.schedule_request(reqs, set())
        # First claims 3, second needs 3 but 3+3=6>5
        assert len(out.context_requests) == 1

    def test_peft_resets_between_calls(self):
        mgr = make_kv_cache_manager()
        peft = make_peft_cache_manager(max_device_pages=1, pages_per_task=1)
        sched = make_scheduler(mgr, peft_cache_manager=peft)
        # First call claims the one page
        reqs1 = [make_ctx_request(0, 100, lora_task_id=1)]
        out1 = sched.schedule_request(reqs1, set())
        assert len(out1.context_requests) == 1
        # Second call: budget should reset
        reqs2 = [make_ctx_request(1, 100, lora_task_id=2)]
        out2 = sched.schedule_request(reqs2, set())
        assert len(out2.context_requests) == 1


# ===========================================================================
# Encoder Requests
# ===========================================================================


class TestEncoder:
    def test_encoder_scheduled(self):
        mgr = make_kv_cache_manager()
        sched = make_encoder_scheduler(mgr, max_num_tokens=200)
        reqs = [make_encoder_request(0, encoder_output_len=100)]
        out = sched.schedule_request(reqs, set())
        assert ids(out.context_requests) == [0]

    def test_encoder_budget_overflow(self):
        mgr = make_kv_cache_manager()
        sched = make_encoder_scheduler(mgr, max_num_tokens=100)
        reqs = [make_encoder_request(0, encoder_output_len=200)]
        out = sched.schedule_request(reqs, set())
        assert len(out.context_requests) == 0

    def test_encoder_exceeds_budget(self):
        """encoder_output_len > max_num_tokens → break (not scheduled)."""
        mgr = make_kv_cache_manager()
        sched = make_encoder_scheduler(mgr, max_num_tokens=1000)
        reqs = [make_encoder_request(0, encoder_output_len=2000)]
        out = sched.schedule_request(reqs, set())
        assert len(out.context_requests) == 0

    def test_encoder_plus_gen(self):
        mgr = make_kv_cache_manager()
        sched = make_encoder_scheduler(mgr, max_num_tokens=100)
        reqs = [
            make_encoder_request(0, encoder_output_len=50),
            make_gen_request(1),
        ]
        out = sched.schedule_request(reqs, set())
        assert ids(out.context_requests) == [0]
        assert ids(out.generation_requests) == [1]

    def test_encoder_prepare_fails(self):
        mgr = make_kv_cache_manager(prepare_context_fn=lambda req: False)
        sched = make_encoder_scheduler(mgr, max_num_tokens=1000)
        reqs = [make_encoder_request(0, encoder_output_len=100)]
        out = sched.schedule_request(reqs, set())
        assert len(out.context_requests) == 0

    def test_encoder_resize_fails(self):
        mgr = make_kv_cache_manager(resize_context_fn=lambda req, n: False)
        sched = make_encoder_scheduler(mgr, max_num_tokens=1000)
        reqs = [make_encoder_request(0, encoder_output_len=100)]
        out = sched.schedule_request(reqs, set())
        assert len(out.context_requests) == 0

    def test_multiple_encoders(self):
        mgr = make_kv_cache_manager()
        sched = make_encoder_scheduler(mgr, max_num_tokens=100)
        reqs = [
            make_encoder_request(0, encoder_output_len=50),
            make_encoder_request(1, encoder_output_len=50),
        ]
        out = sched.schedule_request(reqs, set())
        assert ids(out.context_requests) == [0, 1]

    def test_encoder_counts_toward_batch(self):
        """Gen wins phase 1; encoder scheduled in phase 2 still occupies a batch slot."""
        mgr = make_kv_cache_manager()
        sched = make_encoder_scheduler(mgr, max_batch_size=2, max_num_tokens=1000)
        reqs = [
            make_encoder_request(0, encoder_output_len=50),
            make_gen_request(1),
            make_encoder_request(2, encoder_output_len=50),
        ]
        out = sched.schedule_request(reqs, set())
        # gen(1) scheduled first (phase 1), encoder(0) fills remaining slot (phase 2)
        assert ids(out.generation_requests) == [1]
        assert ids(out.context_requests) == [0]
        # encoder(2) excluded — encoder(0) counted toward batch


# ===========================================================================
# Disaggregated Serving
# ===========================================================================


class TestDisagg:
    def test_disagg_bypasses_state_gate(self):
        mgr = make_kv_cache_manager()
        sched = make_scheduler(mgr, max_num_tokens=100)
        reqs = [make_disagg_request(0)]
        out = sched.schedule_request(reqs, set())
        assert ids(out.fitting_disagg_gen_init_requests) == [0]

    def test_disagg_does_not_count_toward_batch(self):
        mgr = make_kv_cache_manager()
        sched = make_scheduler(mgr, max_batch_size=1, max_num_tokens=100)
        reqs = [make_disagg_request(0), make_gen_request(1)]
        out = sched.schedule_request(reqs, set())
        assert ids(out.fitting_disagg_gen_init_requests) == [0]
        assert ids(out.generation_requests) == [1]

    def test_disagg_does_not_consume_budget(self):
        mgr = make_kv_cache_manager()
        sched = make_scheduler(mgr, max_num_tokens=1)
        reqs = [make_disagg_request(0), make_gen_request(1)]
        out = sched.schedule_request(reqs, set())
        assert ids(out.fitting_disagg_gen_init_requests) == [0]
        assert ids(out.generation_requests) == [1]

    def test_multiple_disagg(self):
        mgr = make_kv_cache_manager()
        sched = make_scheduler(mgr, max_num_tokens=100)
        reqs = [make_disagg_request(i) for i in range(3)]
        out = sched.schedule_request(reqs, set())
        assert ids(out.fitting_disagg_gen_init_requests) == [0, 1, 2]

    def test_disagg_mixed_with_gen_ctx(self):
        mgr = make_kv_cache_manager()
        sched = make_scheduler(mgr, max_num_tokens=200)
        reqs = [
            make_disagg_request(0),
            make_gen_request(1),
            make_ctx_request(2, context_remaining_length=100),
            make_disagg_request(3),
        ]
        out = sched.schedule_request(reqs, set())
        assert ids(out.fitting_disagg_gen_init_requests) == [0, 3]
        assert ids(out.generation_requests) == [1]
        assert ids(out.context_requests) == [2]

    def test_disagg_skips_peft_check(self):
        """Disagg req should NOT trigger PEFT check."""
        mgr = make_kv_cache_manager()
        peft = make_peft_cache_manager(max_device_pages=0, pages_per_task=1)
        sched = make_scheduler(mgr, peft_cache_manager=peft)
        req = make_disagg_request(0)
        req.lora_task_id = 1
        out = sched.schedule_request([req], set())
        assert ids(out.fitting_disagg_gen_init_requests) == [0]
        # PEFT check is never called for disagg
        peft.determine_num_pages.assert_not_called()


# ===========================================================================
# Chunked Context
# ===========================================================================


class TestChunkedContext:
    def test_budget_enough_for_all(self):
        mgr = make_kv_cache_manager(tokens_per_block=64)
        sched = make_scheduler(mgr, max_num_tokens=200, ctx_chunk_config=(None, 64))
        req = make_ctx_request(0, context_remaining_length=100)
        out = sched.schedule_request([req], set())
        assert ids(out.context_requests) == [0]
        assert req.context_chunk_size == 100  # last chunk, no rounding

    def test_budget_limits_chunk(self):
        mgr = make_kv_cache_manager(tokens_per_block=64)
        sched = make_scheduler(mgr, max_num_tokens=300, ctx_chunk_config=(None, 64))
        req = make_ctx_request(0, context_remaining_length=1000)
        out = sched.schedule_request([req], set())
        assert ids(out.context_requests) == [0]
        # 300 // 64 * 64 = 256
        assert req.context_chunk_size == 256

    def test_min_budget_check(self):
        mgr = make_kv_cache_manager(tokens_per_block=64)
        sched = make_scheduler(mgr, max_num_tokens=50, ctx_chunk_config=(None, 64))
        req = make_ctx_request(0, context_remaining_length=1000)
        out = sched.schedule_request([req], set())
        assert len(out.context_requests) == 0  # 50 < 64 min budget

    def test_last_chunk_no_rounding(self):
        mgr = make_kv_cache_manager(tokens_per_block=64)
        sched = make_scheduler(mgr, max_num_tokens=200, ctx_chunk_config=(None, 128))
        req = make_ctx_request(0, context_remaining_length=100)
        out = sched.schedule_request([req], set())
        assert ids(out.context_requests) == [0]
        assert req.context_chunk_size == 100  # 100 < 200, is last, no rounding

    def test_has_chunking_flag_set(self):
        mgr = make_kv_cache_manager(tokens_per_block=64)
        sched = make_scheduler(mgr, max_num_tokens=100, ctx_chunk_config=(None, 64))
        req = make_ctx_request(0, context_remaining_length=1000)
        # chunk_size will be 64 < 1000 → has_chunking=True
        out = sched.schedule_request([req], set())
        # has_chunking manifests in sorting behavior — non-last chunks before last
        # We verify indirectly: the request should be scheduled
        assert len(out.context_requests) == 1

    def test_multiple_ctx_share_budget(self):
        """Two ctx requests share the budget."""
        mgr = make_kv_cache_manager(tokens_per_block=64)
        sched = make_scheduler(mgr, max_num_tokens=500, ctx_chunk_config=(None, 64))
        req1 = make_ctx_request(0, context_remaining_length=200)
        req2 = make_ctx_request(1, context_remaining_length=200)
        out = sched.schedule_request([req1, req2], set())
        assert ids(out.context_requests) == [0, 1]
        # req1 gets 200 (last chunk), req2 gets up to 300 remaining (also last chunk → 200)
        assert req1.context_chunk_size == 200
        assert req2.context_chunk_size == 200

    def test_chunk_fail_flows_budget(self):
        """Chunked ctx fails → budget flows to next request."""
        call_count = [0]

        def resize_fn(req, n):
            call_count[0] += 1
            return call_count[0] > 1  # first fails

        mgr = make_kv_cache_manager(tokens_per_block=64, resize_context_fn=resize_fn)
        sched = make_scheduler(mgr, max_num_tokens=500, ctx_chunk_config=(None, 64))
        req1 = make_ctx_request(0, context_remaining_length=200)
        req2 = make_ctx_request(1, context_remaining_length=200)
        out = sched.schedule_request([req1, req2], set())
        assert ids(out.context_requests) == [1]

    def test_chunk_rounding_non_last(self):
        """Non-last chunk rounds down to chunk_unit_size boundary."""
        mgr = make_kv_cache_manager(tokens_per_block=64)
        sched = make_scheduler(mgr, max_num_tokens=200, ctx_chunk_config=(None, 128))
        req = make_ctx_request(0, context_remaining_length=1000)
        out = sched.schedule_request([req], set())
        assert ids(out.context_requests) == [0]
        # 200 // 128 * 128 = 128
        assert req.context_chunk_size == 128

    def test_has_chunking_flag_not_set(self):
        """Last chunk (all fits) → has_chunking=False → plain sort."""
        mgr = make_kv_cache_manager(tokens_per_block=64)
        sched = make_scheduler(mgr, max_num_tokens=500, ctx_chunk_config=(None, 64))
        req = make_ctx_request(0, context_remaining_length=100)
        out = sched.schedule_request([req], set())
        assert ids(out.context_requests) == [0]
        assert req.context_chunk_size == 100  # == remaining → last chunk, no chunking flag

    def test_max_context_length_caps_chunk(self):
        """max_context_length (= max_num_tokens) caps the chunk size."""
        mgr = make_kv_cache_manager(tokens_per_block=64)
        # max_num_tokens=500 → max_context_length=500
        sched = make_scheduler(mgr, max_num_tokens=500, ctx_chunk_config=(None, 64))
        req = make_ctx_request(0, context_remaining_length=2000)
        out = sched.schedule_request([req], set())
        assert ids(out.context_requests) == [0]
        # chunk capped at min(500, 2000) = 500, non-last → round: 500//64*64 = 448
        assert req.context_chunk_size == 448

    def test_chunk_rounds_to_zero(self):
        mgr = make_kv_cache_manager(tokens_per_block=64)
        sched = make_scheduler(mgr, max_num_tokens=100, ctx_chunk_config=(None, 128))
        req = make_ctx_request(0, context_remaining_length=1000)
        out = sched.schedule_request([req], set())
        # 100 // 128 * 128 = 0, ctx skipped
        assert len(out.context_requests) == 0

    def test_subsequent_chunk_non_first(self):
        """Non-first chunk (continuing prefill) should be scheduled."""
        mgr = make_kv_cache_manager(tokens_per_block=64)
        sched = make_scheduler(mgr, max_num_tokens=500, ctx_chunk_config=(None, 64))
        req = make_ctx_request(0, context_remaining_length=200, is_first_context_chunk=False)
        out = sched.schedule_request([req], set())
        assert ids(out.context_requests) == [0]


# ===========================================================================
# Draft Token Fitting (Chunked Last Chunk)
# ===========================================================================


class TestDraftTokenFitting:
    def test_draft_fits_in_page_remainder(self):
        """chunk=100, tokens_per_block=64. 100%64=36, space=64-36=28. draft=5 fits."""
        mgr = make_kv_cache_manager(tokens_per_block=64)
        sched = make_scheduler(mgr, max_num_tokens=200, ctx_chunk_config=(None, 64))
        req = make_ctx_request(0, context_remaining_length=100, num_draft_tokens=5)
        out = sched.schedule_request([req], set())
        assert ids(out.context_requests) == [0]
        # No discard needed (28 >= 5)
        req.discard_draft_tokens.assert_not_called()

    def test_draft_exceeds_page_remainder(self):
        """chunk=128, tokens_per_block=64. V2 scheduler adds full draft_len
        to chunk_tokens without page-remainder trimming."""
        mgr = make_kv_cache_manager(tokens_per_block=64)
        sched = make_scheduler(mgr, max_num_tokens=200, ctx_chunk_config=(None, 64))
        req = make_ctx_request(0, context_remaining_length=128, num_draft_tokens=5)
        out = sched.schedule_request([req], set())
        assert ids(out.context_requests) == [0]
        # V2 does not trim drafts to page remainder — full draft_len included
        req.discard_draft_tokens.assert_not_called()

    def test_draft_partially_fits(self):
        """chunk=100, tokens_per_block=64. V2 scheduler adds full draft_len
        to chunk_tokens without page-remainder trimming."""
        mgr = make_kv_cache_manager(tokens_per_block=64)
        sched = make_scheduler(mgr, max_num_tokens=200, ctx_chunk_config=(None, 64))
        req = make_ctx_request(0, context_remaining_length=100, num_draft_tokens=30)
        out = sched.schedule_request([req], set())
        assert ids(out.context_requests) == [0]
        # V2 does not trim drafts — full draft_len included
        req.discard_draft_tokens.assert_not_called()

    def test_budget_limits_draft_space(self):
        """chunk=100, draft=20, budget=110. V2 scheduler adds full draft_len
        to chunk_tokens (100+20=120) which passes resize but budget tracks 120."""
        mgr = make_kv_cache_manager(tokens_per_block=64)
        sched = make_scheduler(mgr, max_num_tokens=110, ctx_chunk_config=(None, 64))
        req = make_ctx_request(0, context_remaining_length=100, num_draft_tokens=20)
        out = sched.schedule_request([req], set())
        assert ids(out.context_requests) == [0]
        # V2 does not trim drafts — full draft_len included
        req.discard_draft_tokens.assert_not_called()

    def test_max_context_length_limits_draft(self):
        """max_context_length=110, chunk=100, draft=20. V2 scheduler adds
        full draft_len to chunk_tokens without trimming."""
        mgr = make_kv_cache_manager(tokens_per_block=64)
        sched = make_scheduler(mgr, max_num_tokens=110, ctx_chunk_config=(None, 64))
        req = make_ctx_request(0, context_remaining_length=100, num_draft_tokens=20)
        out = sched.schedule_request([req], set())
        assert ids(out.context_requests) == [0]
        # V2 does not trim drafts — full draft_len included
        req.discard_draft_tokens.assert_not_called()

    def test_draft_tokens_add_to_batch_budget(self):
        """Last chunk with draft: batch_num_tokens += chunk_size + num_draft_tokens."""
        mgr = make_kv_cache_manager(tokens_per_block=64)
        sched = make_scheduler(mgr, max_num_tokens=200, ctx_chunk_config=(None, 64))
        ctx = make_ctx_request(0, context_remaining_length=100, num_draft_tokens=5)
        gen = make_gen_request(1)  # 1 token
        out = sched.schedule_request([ctx, gen], set())
        assert ids(out.context_requests) == [0]
        assert ids(out.generation_requests) == [1]
        # Budget: 100 (ctx) + 5 (draft, fits in 28 page remainder) + 1 (gen) = 106

    def test_non_last_chunk_no_draft_fitting(self):
        """Non-last chunk should NOT call _fit_draft_tokens_single."""
        mgr = make_kv_cache_manager(tokens_per_block=64)
        sched = make_scheduler(mgr, max_num_tokens=200, ctx_chunk_config=(None, 64))
        req = make_ctx_request(
            0, context_remaining_length=1000, num_draft_tokens=5, is_last_context_chunk=False
        )
        out = sched.schedule_request([req], set())
        assert ids(out.context_requests) == [0]
        req.discard_draft_tokens.assert_not_called()


# ===========================================================================
# Beam Width
# ===========================================================================


class TestBeamWidth:
    def test_all_same_beam(self):
        mgr = make_kv_cache_manager()
        sched = make_scheduler(mgr, max_num_tokens=100)
        reqs = [make_gen_request(i, beam_width=1) for i in range(3)]
        out = sched.schedule_request(reqs, set())
        assert len(out.generation_requests) == 3

    def test_beam_mismatch_skip(self):
        mgr = make_kv_cache_manager()
        sched = make_scheduler(mgr, max_num_tokens=100)
        reqs = [
            make_gen_request(0, beam_width=1),
            make_gen_request(1, beam_width=2),  # mismatch → skip
            make_gen_request(2, beam_width=1),
        ]
        out = sched.schedule_request(reqs, set())
        assert ids(out.generation_requests) == [0, 2]

    def test_first_gen_sets_beam(self):
        mgr = make_kv_cache_manager()
        sched = make_scheduler(mgr, max_num_tokens=100)
        reqs = [
            make_gen_request(0, beam_width=2),
            make_gen_request(1, beam_width=2),
        ]
        out = sched.schedule_request(reqs, set())
        assert len(out.generation_requests) == 2

    def test_beam_mismatch_after_budget(self):
        """Budget hit before beam mismatch is relevant."""
        mgr = make_kv_cache_manager()
        sched = make_scheduler(mgr, max_num_tokens=1)
        reqs = [
            make_gen_request(0, beam_width=1),
            make_gen_request(1, beam_width=2),
        ]
        out = sched.schedule_request(reqs, set())
        assert ids(out.generation_requests) == [0]


# ===========================================================================
# State Filtering & Inflight
# ===========================================================================


class TestStateFiltering:
    def test_inflight_skipped(self):
        mgr = make_kv_cache_manager()
        sched = make_scheduler(mgr, max_num_tokens=100)
        reqs = [make_gen_request(0), make_gen_request(1)]
        out = sched.schedule_request(reqs, {0})
        assert ids(out.generation_requests) == [1]

    def test_unknown_state_filtered(self):
        mgr = make_kv_cache_manager()
        sched = make_scheduler(mgr, max_num_tokens=100)
        reqs = [make_filtered_request(0, state_value=0)]
        out = sched.schedule_request(reqs, set())
        assert len(out.context_requests) == 0
        assert len(out.generation_requests) == 0

    def test_gen_to_complete_filtered(self):
        mgr = make_kv_cache_manager()
        sched = make_scheduler(mgr, max_num_tokens=100)
        reqs = [make_filtered_request(0, state_value=GEN_TO_COMPLETE)]
        out = sched.schedule_request(reqs, set())
        assert len(out.context_requests) == 0
        assert len(out.generation_requests) == 0

    def test_gen_complete_filtered(self):
        """state_value=20 (GENERATION_COMPLETE) → outside [10,14), filtered."""
        mgr = make_kv_cache_manager()
        sched = make_scheduler(mgr, max_num_tokens=100)
        reqs = [make_filtered_request(0, state_value=20)]
        out = sched.schedule_request(reqs, set())
        assert out.num_fitting_requests == 0

    def test_context_init_passes(self):
        """CONTEXT_INIT (10) is in [10,14) range → not filtered."""
        mgr = make_kv_cache_manager()
        sched = make_scheduler(mgr, max_num_tokens=1000)
        reqs = [make_ctx_request(0, context_remaining_length=100)]
        out = sched.schedule_request(reqs, set())
        assert len(out.context_requests) == 1

    def test_gen_in_progress_passes(self):
        """GEN_IN_PROGRESS (13) is in [10,14) range → not filtered."""
        mgr = make_kv_cache_manager()
        sched = make_scheduler(mgr, max_num_tokens=100)
        reqs = [make_gen_request(0)]
        out = sched.schedule_request(reqs, set())
        assert len(out.generation_requests) == 1

    def test_all_requests_filtered(self):
        mgr = make_kv_cache_manager()
        sched = make_scheduler(mgr, max_num_tokens=100)
        reqs = [
            make_filtered_request(0, state_value=0),
            make_filtered_request(1, state_value=GEN_TO_COMPLETE),
        ]
        out = sched.schedule_request(reqs, set())
        assert out.num_fitting_requests == 0

    def test_inflight_does_not_count_toward_batch(self):
        mgr = make_kv_cache_manager()
        sched = make_scheduler(mgr, max_batch_size=1, max_num_tokens=100)
        reqs = [make_gen_request(0), make_gen_request(1)]
        out = sched.schedule_request(reqs, {0})
        assert ids(out.generation_requests) == [1]


# ===========================================================================
# Sorting (LoRA Task ID)
# ===========================================================================


class TestSorting:
    def test_no_lora_stable_order(self):
        mgr = make_kv_cache_manager()
        sched = make_scheduler(mgr, max_num_tokens=100)
        reqs = [make_gen_request(i) for i in range(3)]
        out = sched.schedule_request(reqs, set())
        assert ids(out.generation_requests) == [0, 1, 2]

    def test_lora_sorted_ascending(self):
        mgr = make_kv_cache_manager()
        sched = make_scheduler(mgr, max_num_tokens=100)
        reqs = [
            make_gen_request(0, lora_task_id=3),
            make_gen_request(1, lora_task_id=1),
            make_gen_request(2, lora_task_id=2),
        ]
        out = sched.schedule_request(reqs, set())
        assert ids(out.generation_requests) == [1, 2, 0]

    def test_none_before_lora(self):
        mgr = make_kv_cache_manager()
        sched = make_scheduler(mgr, max_num_tokens=100)
        reqs = [
            make_gen_request(0, lora_task_id=1),
            make_gen_request(1, lora_task_id=None),
        ]
        out = sched.schedule_request(reqs, set())
        assert ids(out.generation_requests) == [1, 0]

    def test_ctx_with_chunks_nonlast_before_last(self):
        """With chunking: non-last chunks sorted before last chunks."""
        mgr = make_kv_cache_manager(tokens_per_block=64)
        sched = make_scheduler(mgr, max_num_tokens=2000, ctx_chunk_config=(None, 64))
        # req0: last chunk (remaining=100, fits in budget)
        req0 = make_ctx_request(0, context_remaining_length=100, lora_task_id=2)
        # req1: non-last chunk (remaining=1000, will be chunked)
        req1 = make_ctx_request(
            1, context_remaining_length=1000, lora_task_id=1, is_last_context_chunk=False
        )
        out = sched.schedule_request([req0, req1], set())
        assert len(out.context_requests) == 2
        # Non-last before last in sorted output
        assert out.context_requests[0].request_id == 1
        assert out.context_requests[1].request_id == 0

    def test_ctx_without_chunks_plain_sort(self):
        mgr = make_kv_cache_manager()
        sched = make_scheduler(mgr, max_num_tokens=1000)
        reqs = [
            make_ctx_request(0, 100, lora_task_id=3),
            make_ctx_request(1, 100, lora_task_id=1),
        ]
        out = sched.schedule_request(reqs, set())
        assert ids(out.context_requests) == [1, 0]


# ===========================================================================
# SchedulerOutput Structure
# ===========================================================================


class TestSchedulerOutput:
    def test_output_fields_correct(self):
        mgr = make_kv_cache_manager()
        sched = make_scheduler(mgr, max_num_tokens=1000)
        reqs = [
            make_gen_request(0),
            make_ctx_request(1, 100),
            make_disagg_request(2),
        ]
        out = sched.schedule_request(reqs, set())
        assert len(out.context_requests) == 1
        assert len(out.generation_requests) == 1
        assert len(out.fitting_disagg_gen_init_requests) == 1
        assert out.num_fitting_requests == 2

    def test_num_fitting_requests(self):
        mgr = make_kv_cache_manager()
        sched = make_scheduler(mgr, max_num_tokens=1000)
        reqs = [
            make_ctx_request(0, 100),
            make_ctx_request(1, 100),
            make_gen_request(2),
            make_gen_request(3),
            make_gen_request(4),
        ]
        out = sched.schedule_request(reqs, set())
        assert out.num_fitting_requests == 5

    def test_empty_output(self):
        mgr = make_kv_cache_manager()
        sched = make_scheduler(mgr, max_num_tokens=100)
        out = sched.schedule_request([], set())
        assert out.num_fitting_requests == 0
        assert len(out.context_requests) == 0
        assert len(out.generation_requests) == 0


# ===========================================================================
# Mixed Ordering & Interaction
# ===========================================================================


class TestMixedOrdering:
    def test_mixed_gen_ctx_order(self):
        mgr = make_kv_cache_manager()
        sched = make_scheduler(mgr, max_num_tokens=300)
        reqs = [
            make_gen_request(0),  # 1
            make_ctx_request(1, 100),  # 100
            make_gen_request(2),  # 1
            make_ctx_request(3, 100),  # 100
        ]
        out = sched.schedule_request(reqs, set())
        assert len(out.generation_requests) == 2
        assert len(out.context_requests) == 2

    def test_ctx_fail_gen_after_succeeds(self):
        mgr = make_kv_cache_manager(resize_context_fn=lambda req, n: False)
        sched = make_scheduler(mgr, max_num_tokens=1000)
        reqs = [make_ctx_request(0, 500), make_gen_request(1)]
        out = sched.schedule_request(reqs, set())
        assert len(out.context_requests) == 0
        assert ids(out.generation_requests) == [1]

    def test_gen_fail_self_evicts_then_breaks(self):
        """Gen fails → self-evicts → break. Ctx after is NOT tried."""
        mgr = make_kv_cache_manager(
            try_allocate_generation_fn=lambda req: False,
        )
        sched = make_scheduler(mgr, max_num_tokens=1000)
        reqs = [make_gen_request(0), make_ctx_request(1, 100)]
        out = sched.schedule_request(reqs, set())
        assert len(out.generation_requests) == 0
        assert len(out.context_requests) == 0
        assert ids(out.paused_requests) == [0]

    def test_budget_builds_across_types(self):
        mgr = make_kv_cache_manager()
        sched = make_scheduler(mgr, max_num_tokens=102)
        reqs = [
            make_gen_request(0),  # 1
            make_ctx_request(1, 100),  # 100
            make_gen_request(2),  # 1
        ]
        out = sched.schedule_request(reqs, set())
        assert len(out.generation_requests) == 2
        assert len(out.context_requests) == 1

    def test_peft_across_types(self):
        """Gen and ctx both check PEFT, both scheduled."""
        mgr = make_kv_cache_manager()
        peft = make_peft_cache_manager(max_device_pages=2, pages_per_task=1)
        sched = make_scheduler(mgr, max_num_tokens=1000, peft_cache_manager=peft)
        reqs = [
            make_gen_request(0, lora_task_id=1),
            make_ctx_request(1, 100, lora_task_id=2),
        ]
        out = sched.schedule_request(reqs, set())
        assert len(out.generation_requests) == 1
        assert len(out.context_requests) == 1

    def test_encoder_then_gen(self):
        mgr = make_kv_cache_manager()
        sched = make_encoder_scheduler(mgr, max_num_tokens=100)
        reqs = [
            make_encoder_request(0, encoder_output_len=50),
            make_gen_request(1),
        ]
        out = sched.schedule_request(reqs, set())
        assert ids(out.context_requests) == [0]
        assert ids(out.generation_requests) == [1]

    def test_ctx_fail_budget_preserved(self):
        """Failed ctx doesn't consume budget — gen gets full budget."""
        mgr = make_kv_cache_manager(resize_context_fn=lambda req, n: False)
        sched = make_scheduler(mgr, max_num_tokens=100)
        reqs = [make_ctx_request(0, 50), make_gen_request(1)]
        out = sched.schedule_request(reqs, set())
        assert ids(out.generation_requests) == [1]

    def test_multiple_gen_after_gen_fail(self):
        """Gen failure: req0 fails, evicts tail victims, then self-evicts.

        When req0 can't allocate, eviction evicts req2 and req1 (both
        started gen requests found backwards from tail). Still fails,
        so req0 self-evicts. All 3 paused.
        """

        def selective_gen_alloc(req):
            # req0 always fails, req1 and req2 succeed
            return req.request_id != 0

        mgr = make_kv_cache_manager(
            try_allocate_generation_fn=selective_gen_alloc,
        )
        sched = make_scheduler(mgr, max_num_tokens=1000)
        reqs = [make_gen_request(0), make_gen_request(1), make_gen_request(2)]
        out = sched.schedule_request(reqs, set())
        # req0 fails alloc → evicts req2 (tail), then req1 → still fails
        # → req0 self-evicts. All 3 are paused.
        assert len(out.generation_requests) == 0
        assert set(ids(out.paused_requests)) == {0, 1, 2}


# ===========================================================================
# can_schedule
# ===========================================================================


class TestCanSchedule:
    def test_always_true(self):
        mgr = make_kv_cache_manager()
        sched = make_scheduler(mgr)
        assert sched.can_schedule([make_gen_request(0)]) is True

    def test_empty_list(self):
        mgr = make_kv_cache_manager()
        sched = make_scheduler(mgr)
        assert sched.can_schedule([]) is True


# ===========================================================================
# Non-chunked Context: Prepare Before Budget
# ===========================================================================


class TestPrepareBeforeBudget:
    def test_block_reuse_reduces_budget(self):
        """prepare_context reduces context_remaining_length via block reuse."""

        def prepare_with_reuse(req):
            req.context_remaining_length = 500  # reduced from 1000
            return True

        mgr = make_kv_cache_manager(prepare_context_fn=prepare_with_reuse)
        sched = make_scheduler(mgr, max_num_tokens=600)
        req = make_ctx_request(0, context_remaining_length=1000)
        out = sched.schedule_request([req], set())
        assert ids(out.context_requests) == [0]

    def test_without_reuse_budget_would_fail(self):
        """Without block reuse, 1000 > 800 would fail. With reuse → 500 ≤ 800."""

        def prepare_with_reuse(req):
            req.context_remaining_length = 500
            return True

        mgr = make_kv_cache_manager(prepare_context_fn=prepare_with_reuse)
        sched = make_scheduler(mgr, max_num_tokens=800)
        req = make_ctx_request(0, context_remaining_length=1000)
        out = sched.schedule_request([req], set())
        assert ids(out.context_requests) == [0]


# ===========================================================================
# Edge Cases
# ===========================================================================


class TestEdgeCases:
    def test_empty_active_requests(self):
        mgr = make_kv_cache_manager()
        sched = make_scheduler(mgr, max_num_tokens=100)
        out = sched.schedule_request([], set())
        assert out.num_fitting_requests == 0

    def test_single_request_each_type(self):
        """Single gen, ctx, encoder each scheduled correctly."""
        mgr = make_kv_cache_manager()
        sched = make_scheduler(mgr, max_num_tokens=1000)
        # Single gen
        out = sched.schedule_request([make_gen_request(0)], set())
        assert len(out.generation_requests) == 1
        # Single ctx
        out = sched.schedule_request([make_ctx_request(1, 100)], set())
        assert len(out.context_requests) == 1
        # Single encoder (needs widened state range)
        enc_sched = make_encoder_scheduler(mgr, max_num_tokens=1000)
        out = enc_sched.schedule_request([make_encoder_request(2, 50)], set())
        assert len(out.context_requests) == 1

    def test_all_inflight(self):
        mgr = make_kv_cache_manager()
        sched = make_scheduler(mgr, max_num_tokens=100)
        reqs = [make_gen_request(0), make_gen_request(1)]
        out = sched.schedule_request(reqs, {0, 1})
        assert out.num_fitting_requests == 0

    def test_max_num_tokens_zero(self):
        mgr = make_kv_cache_manager()
        sched = make_scheduler(mgr, max_num_tokens=0)
        reqs = [make_gen_request(0)]
        out = sched.schedule_request(reqs, set())
        assert len(out.generation_requests) == 0

    def test_max_batch_size_zero(self):
        """max_batch_size=0 → nothing scheduled."""
        mgr = make_kv_cache_manager()
        sched = make_scheduler(mgr, max_num_tokens=1000, max_batch_size=0)
        reqs = [make_gen_request(0), make_ctx_request(1, 50)]
        out = sched.schedule_request(reqs, set())
        assert out.num_fitting_requests == 0

    def test_very_large_batch(self):
        """gen requests with sufficient budget → all scheduled."""
        mgr = make_kv_cache_manager()
        sched = make_scheduler(mgr, max_batch_size=2000, max_num_tokens=2000)
        reqs = [make_gen_request(i) for i in range(1000)]
        out = sched.schedule_request(reqs, set())
        assert len(out.generation_requests) == 1000

    def test_interleaved_states(self):
        mgr = make_kv_cache_manager()
        sched = make_scheduler(mgr, max_num_tokens=1000)
        reqs = [
            make_filtered_request(0, state_value=0),
            make_disagg_request(1),
            make_gen_request(2),
            make_filtered_request(3, state_value=GEN_TO_COMPLETE),
            make_ctx_request(4, context_remaining_length=100),
        ]
        out = sched.schedule_request(reqs, set())
        assert ids(out.fitting_disagg_gen_init_requests) == [1]
        assert ids(out.generation_requests) == [2]
        assert ids(out.context_requests) == [4]


# ===========================================================================
# Block Reuse Boundary Alignment (Chunked + Partial Reuse)
# ===========================================================================


class TestPartialReuse:
    """Verify chunked context behaviour when context_current_position is not
    block-aligned (partial reuse scenario).  Block boundary alignment is NOT
    performed by the scheduler — the KV cache manager handles it internally."""

    def test_partial_reuse_no_block_alignment(self):
        """Partial reuse sets context_current_position=50 (not block-aligned).
        tokens_per_block=64, chunk_unit_size=64, max_num_tokens=128.
        remaining after reuse = 950.
        chunk_size = min(128, 950) = 128 → unit round: 128//64*64 = 128.
        No block boundary floor — chunk_size stays 128.
        """

        def prepare_with_partial_reuse(req):
            req.context_current_position = 50
            req.context_remaining_length = 950  # 1000 - 50
            return True

        mgr = make_kv_cache_manager(
            tokens_per_block=64, prepare_context_fn=prepare_with_partial_reuse
        )
        sched = make_scheduler(mgr, max_num_tokens=128, ctx_chunk_config=(None, 64))
        req = make_ctx_request(0, context_remaining_length=1000, prompt_len=1000)
        out = sched.schedule_request([req], set())
        assert ids(out.context_requests) == [0]
        assert req.context_chunk_size == 128

    def test_block_aligned_position_no_change(self):
        """When context_current_position is already block-aligned (full block
        reuse), chunk_size should not be further adjusted."""

        def prepare_with_full_reuse(req):
            req.context_current_position = 128  # 2 * 64, block-aligned
            req.context_remaining_length = 872  # 1000 - 128
            return True

        mgr = make_kv_cache_manager(tokens_per_block=64, prepare_context_fn=prepare_with_full_reuse)
        sched = make_scheduler(mgr, max_num_tokens=300, ctx_chunk_config=(None, 64))
        req = make_ctx_request(0, context_remaining_length=1000, prompt_len=1000)
        out = sched.schedule_request([req], set())
        assert ids(out.context_requests) == [0]
        # 300 // 64 * 64 = 256 (unit round only)
        assert req.context_chunk_size == 256

    def test_partial_reuse_last_chunk_no_alignment(self):
        """Last chunk should NOT be rounded — all remaining tokens must be
        processed regardless of block boundary."""

        def prepare_with_partial_reuse(req):
            req.context_current_position = 50
            req.context_remaining_length = 80  # 130 - 50
            return True

        mgr = make_kv_cache_manager(
            tokens_per_block=64, prepare_context_fn=prepare_with_partial_reuse
        )
        sched = make_scheduler(mgr, max_num_tokens=500, ctx_chunk_config=(None, 64))
        req = make_ctx_request(0, context_remaining_length=130, prompt_len=130)
        out = sched.schedule_request([req], set())
        assert ids(out.context_requests) == [0]
        # 80 <= 500 budget, 80 == remaining → last chunk, no rounding
        assert req.context_chunk_size == 80

    def test_partial_reuse_resize_uses_chunk_size(self):
        """Verify resize_context is called with the chunk_size directly
        (no block boundary alignment by the scheduler)."""
        resize_calls = []

        def prepare_with_partial_reuse(req):
            req.context_current_position = 50
            req.context_remaining_length = 950
            return True

        def track_resize(req, n):
            resize_calls.append(n)
            return True

        mgr = make_kv_cache_manager(
            tokens_per_block=64,
            prepare_context_fn=prepare_with_partial_reuse,
            resize_context_fn=track_resize,
        )
        sched = make_scheduler(mgr, max_num_tokens=128, ctx_chunk_config=(None, 64))
        req = make_ctx_request(0, context_remaining_length=1000, prompt_len=1000)
        sched.schedule_request([req], set())
        # resize receives chunk_size directly (128), no block alignment
        assert resize_calls == [128]

    def test_partial_reuse_budget_accounting(self):
        """Token budget should account for the chunk_size as-is.
        pos=50, budget=80, remaining=950 → chunk=64 (unit round of 80).
        No block boundary floor — chunk stays 64.
        64 + 1(gen) = 65 <= 80: gen fits.
        """

        def prepare_with_partial_reuse(req):
            if req.request_id == 0:
                req.context_current_position = 50
                req.context_remaining_length = 950
            return True

        mgr = make_kv_cache_manager(
            tokens_per_block=64, prepare_context_fn=prepare_with_partial_reuse
        )
        sched = make_scheduler(mgr, max_num_tokens=80, ctx_chunk_config=(None, 64))
        ctx = make_ctx_request(0, context_remaining_length=1000, prompt_len=1000)
        gen = make_gen_request(1)  # 1 token
        out = sched.schedule_request([ctx, gen], set())
        assert ids(out.context_requests) == [0]
        assert ctx.context_chunk_size == 64
        # 64 + 1 = 65 <= 80: gen fits
        assert ids(out.generation_requests) == [1]
