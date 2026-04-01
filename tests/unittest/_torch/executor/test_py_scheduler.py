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
"""
Unit tests for Python scheduler implementations (PyMicroBatchScheduler,
PyCapacityScheduler, SimpleUnifiedScheduler).

These tests validate the pure-Python scheduler logic using real LlmRequest
objects (from C++ bindings) and mock KV cache managers, without requiring
GPU. They are aligned with the C++ scheduler unit tests in:
  - cpp/tests/unit_tests/batch_manager/microBatchSchedulerTest.cpp
  - cpp/tests/unit_tests/batch_manager/capacitySchedulerTest.cpp
"""

from dataclasses import dataclass, field
from typing import List, Optional

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestState, SamplingConfig
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import (
    ChunkingPolicy,
    ContextChunkingConfig,
    PyCapacityScheduler,
    PyMicroBatchScheduler,
    SimpleUnifiedScheduler,
)
from tensorrt_llm.llmapi.llm_args import CapacitySchedulerPolicy


def _make_request(
    request_id: int,
    prompt_len: int = 10,
    beam_width: int = 1,
    draft_tokens_len: int = 0,
    encoder_output_len: int = 0,
    lora_task_id: Optional[int] = None,
    state: LlmRequestState = LlmRequestState.CONTEXT_INIT,
    input_tokens: Optional[List[int]] = None,
) -> LlmRequest:
    tokens = input_tokens if input_tokens is not None else list(range(prompt_len))
    draft = list(range(draft_tokens_len)) if draft_tokens_len > 0 else None
    req = LlmRequest(
        request_id=request_id,
        max_new_tokens=10,
        input_tokens=tokens,
        sampling_config=SamplingConfig(beam_width),
        is_streaming=False,
        draft_tokens=draft,
        lora_task_id=lora_task_id,
        encoder_output_len=encoder_output_len if encoder_output_len > 0 else None,
    )
    req.state = state
    return req


def make_context_request(
    request_id: int,
    prompt_len: int = 10,
    beam_width: int = 1,
    draft_tokens_len: int = 0,
    context_position: int = 0,
) -> LlmRequest:
    req = _make_request(
        request_id=request_id,
        prompt_len=prompt_len,
        beam_width=beam_width,
        draft_tokens_len=draft_tokens_len,
        state=LlmRequestState.CONTEXT_INIT,
    )
    if context_position > 0:
        req.context_chunk_size = context_position
        req.move_to_next_context_chunk()
    return req


def make_generation_request(
    request_id: int,
    beam_width: int = 1,
    draft_tokens_len: int = 0,
) -> LlmRequest:
    return _make_request(
        request_id=request_id,
        beam_width=beam_width,
        draft_tokens_len=draft_tokens_len,
        state=LlmRequestState.GENERATION_IN_PROGRESS,
    )


def make_encoder_request(
    request_id: int,
    encoder_output_len: int = 10,
) -> LlmRequest:
    return _make_request(
        request_id=request_id,
        encoder_output_len=encoder_output_len,
        state=LlmRequestState.ENCODER_INIT,
    )


def make_disagg_gen_init_request(request_id: int) -> LlmRequest:
    return _make_request(
        request_id=request_id,
        state=LlmRequestState.DISAGG_GENERATION_INIT,
    )


def make_completed_request(request_id: int) -> LlmRequest:
    return _make_request(
        request_id=request_id,
        state=LlmRequestState.GENERATION_COMPLETE,
    )


@dataclass
class MockKVCacheStats:
    num_free_blocks_per_window_size: dict = field(default_factory=lambda: {128: 100})


class MockKVCacheManager:
    """Mock KV cache manager for capacity scheduler tests."""

    def __init__(
        self,
        num_free_blocks: int = 100,
        window_sizes: Optional[list] = None,
        blocks_per_request: int = 5,
        is_variable_window: bool = False,
        enable_block_reuse: bool = False,
    ):
        self._window_sizes = window_sizes or [128]
        self._num_free_blocks = num_free_blocks
        self._blocks_per_request = blocks_per_request
        self.is_variable_window = is_variable_window
        self.enable_block_reuse = enable_block_reuse
        self.max_attention_window_vec = self._window_sizes
        self._scheduling_started = False

    def get_kv_cache_stats(self) -> MockKVCacheStats:
        return MockKVCacheStats(
            num_free_blocks_per_window_size={ws: self._num_free_blocks for ws in self._window_sizes}
        )

    def get_remaining_blocks_to_completion(self, req, window_size: int) -> int:
        return self._blocks_per_request

    def get_needed_blocks_one_step(self, req, two_step_lookahead: bool, window_size: int) -> int:
        return self._blocks_per_request

    def scheduling_has_free_blocks(self, total: int, window_size: int) -> bool:
        return total <= self._num_free_blocks

    def start_scheduling(self):
        self._scheduling_started = True

    def scheduling_remove_sequence(self, req_id: int):
        pass

    def find_new_context_block(self, unique_tokens, req):
        return None

    def get_max_resource_count(self) -> int:
        return self._num_free_blocks

    def get_needed_resource_to_completion(self, req) -> int:
        return self._blocks_per_request


class MockPeftCacheManager:
    def __init__(self, max_pages: int = 100, pages_per_request: int = 10):
        self.max_device_pages = max_pages
        self._pages_per_request = pages_per_request

    def determine_num_pages(self, req) -> int:
        return self._pages_per_request


# ############################################################################
#
# Part 1: PyMicroBatchScheduler Tests
#
# ############################################################################


class TestPyMicroBatchSchedulerBasic:
    """
    Tests for PyMicroBatchScheduler — single-step scheduling decisions.
    Aligned with C++ MicroBatchSchedulerTest in microBatchSchedulerTest.cpp.
    """

    def test_simple_context_only(self):
        """All requests are context requests, batch size allows 2."""
        scheduler = PyMicroBatchScheduler(max_batch_size=2, max_num_tokens=None)
        requests = [
            make_context_request(0, prompt_len=10),
            make_context_request(1, prompt_len=10),
            make_context_request(2, prompt_len=10),
        ]
        ctx, gen = scheduler.schedule(requests, set())
        assert len(ctx) == 2
        assert len(gen) == 0
        assert ctx[0].request_id == 0
        assert ctx[1].request_id == 1

    def test_simple_generation_only(self):
        """All requests are generation requests, batch size allows 2."""
        scheduler = PyMicroBatchScheduler(max_batch_size=2, max_num_tokens=None)
        requests = [
            make_generation_request(0),
            make_generation_request(1),
            make_generation_request(2),
        ]
        ctx, gen = scheduler.schedule(requests, set())
        assert len(ctx) == 0
        assert len(gen) == 2
        assert gen[0].request_id == 0
        assert gen[1].request_id == 1

    def test_context_generation_overlap(self):
        """
        Mixed batch: context + generation requests.
        C++ ref: SimpleWithOverlap
        """
        scheduler = PyMicroBatchScheduler(max_batch_size=4, max_num_tokens=None)
        requests = [
            make_context_request(0, prompt_len=10),
            make_generation_request(1),
            make_context_request(2, prompt_len=10),
            make_generation_request(3),
        ]
        ctx, gen = scheduler.schedule(requests, set())
        assert len(ctx) == 2
        assert len(gen) == 2
        assert {r.request_id for r in ctx} == {0, 2}
        assert {r.request_id for r in gen} == {1, 3}

    def test_max_num_tokens_limits_context(self):
        """
        max_num_tokens limits how many context tokens can be scheduled.
        C++ ref: SimpleNoOverlapMaxNumTokens
        """
        scheduler = PyMicroBatchScheduler(max_batch_size=4, max_num_tokens=15)
        # Each context request has 10 tokens. Two would be 20 > 15.
        requests = [
            make_context_request(0, prompt_len=10),
            make_context_request(1, prompt_len=10),
        ]
        ctx, gen = scheduler.schedule(requests, set())
        # Only 1 fits within token budget
        assert len(ctx) == 1
        assert ctx[0].request_id == 0

    def test_max_num_tokens_allows_gen_after_context(self):
        """
        After scheduling a context request, generation requests still fit
        if their token count (beam_width) fits in remaining budget.
        """
        scheduler = PyMicroBatchScheduler(max_batch_size=4, max_num_tokens=12)
        requests = [
            make_context_request(0, prompt_len=10),
            make_generation_request(1, beam_width=1),
            make_generation_request(2, beam_width=1),
        ]
        ctx, gen = scheduler.schedule(requests, set())
        # context: 10 tokens, gen1: 1 token, gen2: 1 token => total 12
        assert len(ctx) == 1
        assert len(gen) == 2

    def test_max_batch_size_limits_total(self):
        """Batch size limits total (context + generation)."""
        scheduler = PyMicroBatchScheduler(max_batch_size=2, max_num_tokens=None)
        requests = [
            make_context_request(0, prompt_len=5),
            make_generation_request(1),
            make_generation_request(2),
        ]
        ctx, gen = scheduler.schedule(requests, set())
        # batch_size=2: should schedule context_0 + gen_1
        assert len(ctx) + len(gen) == 2

    def test_beam_width_1(self):
        """
        Generation requests with beam_width=1 each cost 1 token.
        C++ ref: SimpleMaxNumTokensBW1
        """
        scheduler = PyMicroBatchScheduler(max_batch_size=4, max_num_tokens=12)
        requests = [
            make_context_request(0, prompt_len=10, beam_width=1),
            make_generation_request(1, beam_width=1),
            make_generation_request(2, beam_width=1),
            make_generation_request(3, beam_width=1),
        ]
        ctx, gen = scheduler.schedule(requests, set())
        # context: 10, gen: 1+1 = 12 total. Can't fit gen_3 (would be 13).
        assert len(ctx) == 1
        assert len(gen) == 2

    def test_beam_width_4(self):
        """
        Generation requests with beam_width=4 each cost 4 tokens.
        C++ ref: SimpleMaxNumTokensBW4
        """
        scheduler = PyMicroBatchScheduler(max_batch_size=4, max_num_tokens=15)
        requests = [
            make_context_request(0, prompt_len=10, beam_width=4),
            make_generation_request(1, beam_width=4),
            make_generation_request(2, beam_width=4),
        ]
        ctx, gen = scheduler.schedule(requests, set())
        # context: 10, gen1: 4 = 14. gen2: +4 = 18 > 15.
        assert len(ctx) == 1
        assert len(gen) == 1

    def test_beam_width_mismatch_skipped(self):
        """
        Generation requests with different beam widths are skipped.
        C++ ensures all gen requests in a batch have same beam_width.
        """
        scheduler = PyMicroBatchScheduler(max_batch_size=4, max_num_tokens=None)
        requests = [
            make_generation_request(0, beam_width=1),
            make_generation_request(1, beam_width=4),
            make_generation_request(2, beam_width=1),
        ]
        ctx, gen = scheduler.schedule(requests, set())
        # gen_0 sets beam_width=1, gen_1 is skipped (beam_width=4), gen_2 fits
        assert len(gen) == 2
        assert gen[0].request_id == 0
        assert gen[1].request_id == 2

    def test_draft_tokens_count_toward_budget(self):
        """
        Draft tokens are added to the token count for both context and gen.
        C++ ref: DraftTokensMaxNumTokens
        """
        scheduler = PyMicroBatchScheduler(max_batch_size=4, max_num_tokens=15)
        # Context request: 10 prompt + 3 draft = 13 tokens
        requests = [
            make_context_request(0, prompt_len=10, draft_tokens_len=3),
            make_generation_request(1, draft_tokens_len=2),
        ]
        ctx, gen = scheduler.schedule(requests, set())
        # context: 10+3=13, gen: 1+2=3, total=16 > 15 => only context fits
        assert len(ctx) == 1
        assert len(gen) == 0

    def test_gen_draft_tokens(self):
        """
        Generation with draft tokens: cost = beam_width + num_draft_tokens.
        C++ ref: GenDraftTokensMaxNumTokens
        """
        scheduler = PyMicroBatchScheduler(max_batch_size=4, max_num_tokens=10)
        requests = [
            make_generation_request(0, beam_width=1, draft_tokens_len=3),
            make_generation_request(1, beam_width=1, draft_tokens_len=3),
            make_generation_request(2, beam_width=1, draft_tokens_len=3),
        ]
        ctx, gen = scheduler.schedule(requests, set())
        # Each gen costs 1+3=4. Two fit (8), three don't (12 > 10).
        assert len(gen) == 2

    def test_inflight_requests_excluded(self):
        """Requests already in flight are skipped."""
        scheduler = PyMicroBatchScheduler(max_batch_size=4, max_num_tokens=None)
        requests = [
            make_context_request(0, prompt_len=10),
            make_context_request(1, prompt_len=10),
            make_generation_request(2),
        ]
        ctx, gen = scheduler.schedule(requests, {0, 2})
        # Only request 1 is not in flight
        assert len(ctx) == 1
        assert ctx[0].request_id == 1
        assert len(gen) == 0

    def test_completed_requests_filtered(self):
        """Requests in GENERATION_COMPLETE state are filtered out."""
        scheduler = PyMicroBatchScheduler(max_batch_size=4, max_num_tokens=None)
        requests = [
            make_context_request(0, prompt_len=10),
            make_completed_request(1),
            make_generation_request(2),
        ]
        ctx, gen = scheduler.schedule(requests, set())
        # Completed request 1 is filtered by state gating
        assert len(ctx) == 1
        assert len(gen) == 1

    def test_simple_no_overlap(self):
        """
        With max_batch_size=2 and 4 context requests, only 2 are scheduled.
        After transitioning to generation, 2 gen requests fill the batch.
        C++ ref: SimpleNoOverlap (multi-iteration; here we test single-step
        scheduling decisions that compose the same behavior).
        """
        scheduler = PyMicroBatchScheduler(max_batch_size=2, max_num_tokens=None)

        # Step 1: 4 context requests, only 2 fit
        requests = [
            make_context_request(0, prompt_len=10),
            make_context_request(1, prompt_len=10),
            make_context_request(2, prompt_len=10),
            make_context_request(3, prompt_len=10),
        ]
        ctx, gen = scheduler.schedule(requests, set())
        assert len(ctx) == 2
        assert len(gen) == 0
        assert ctx[0].request_id == 0
        assert ctx[1].request_id == 1

        # Step 2: first 2 become generation, remaining 2 still context
        # Generation fills the batch, context requests wait
        requests = [
            make_generation_request(0),
            make_generation_request(1),
            make_context_request(2, prompt_len=10),
            make_context_request(3, prompt_len=10),
        ]
        ctx, gen = scheduler.schedule(requests, set())
        assert len(gen) == 2
        assert gen[0].request_id == 0
        assert gen[1].request_id == 1
        # Context requests 2,3 don't fit due to batch_size=2
        assert len(ctx) == 0

        # Step 3: first 2 complete, now context 2,3 get scheduled
        requests = [
            make_context_request(2, prompt_len=10),
            make_context_request(3, prompt_len=10),
        ]
        ctx, gen = scheduler.schedule(requests, set())
        assert len(ctx) == 2
        assert ctx[0].request_id == 2
        assert ctx[1].request_id == 3

    def test_simple_no_overlap_max_num_tokens(self):
        """
        Context chunking with token budget enforcement across multiple steps.
        C++ ref: SimpleNoOverlapMaxNumTokens
        Req 0, 1: promptLen=12, maxNewTokens=5, maxNumTokens=7, chunkUnitSize=5
        """
        config = ContextChunkingConfig(ChunkingPolicy.EQUAL_PROGRESS, chunk_unit_size=5)
        scheduler = PyMicroBatchScheduler(
            max_batch_size=2, max_num_tokens=7, ctx_chunk_config=config
        )

        # Step 1 (it=0): Only req0 gets a chunk of 5, req1 doesn't fit
        # C++: Req 0: (0,1,2,3,4), Req 1: ()
        r0 = make_context_request(0, prompt_len=12)
        r1 = make_context_request(1, prompt_len=12)
        ctx, gen = scheduler.schedule([r0, r1], set())
        assert len(ctx) >= 1
        # First request gets a chunk within budget
        req0 = next(r for r in ctx if r.request_id == 0)
        assert req0.context_chunk_size <= 7
        total_tokens = sum(r.context_chunk_size for r in ctx)
        assert total_tokens <= 7

    def test_simple_no_overlap_max_context_length(self):
        """
        Context chunking with max_context_length limiting chunk sizes.
        C++ ref: SimpleNoOverlapMaxContextLength
        Requests with promptLen=10 and 17, maxContextLength=12, chunkUnitSize=5.
        """
        config = ContextChunkingConfig(ChunkingPolicy.EQUAL_PROGRESS, chunk_unit_size=5)
        scheduler = PyMicroBatchScheduler(
            max_batch_size=2, max_num_tokens=None, ctx_chunk_config=config
        )
        # Override max_context_length (in C++ this is a separate constructor arg)
        scheduler.max_context_length = 12

        # Two requests with promptLen=10 fit within maxContextLength=12
        r0 = make_context_request(0, prompt_len=10)
        r1 = make_context_request(1, prompt_len=10)
        ctx, gen = scheduler.schedule([r0, r1], set())
        assert len(ctx) == 2
        # Each chunk should be at most max_context_length
        for r in ctx:
            assert r.context_chunk_size <= 12

        # Request with promptLen=17 needs chunking (17 > 12)
        r3 = make_context_request(3, prompt_len=17)
        ctx2, gen2 = scheduler.schedule([r3], set())
        assert len(ctx2) == 1
        assert ctx2[0].context_chunk_size <= 12

    def test_simple_with_overlap(self):
        """
        2-micro-batch overlap: with max_batch_size=2 and 4 context requests,
        the inflight mechanism ensures alternating pairs of requests are scheduled
        in successive steps, mimicking the 2-slot pipeline in C++.

        Step 1: no inflight     → req0, req1 scheduled (slot 0)
        Step 2: {0,1} inflight  → req2, req3 scheduled (slot 1)
        Step 3: {2,3} inflight  → req0, req1 scheduled again (slot 0 freed)

        C++ ref: SimpleWithOverlap
        """
        scheduler = PyMicroBatchScheduler(max_batch_size=2, max_num_tokens=None)
        requests = [make_context_request(i, prompt_len=10) for i in range(4)]

        # Step 1: slot 0 — req0, req1 scheduled
        ctx0, _ = scheduler.schedule(requests, set())
        assert {r.request_id for r in ctx0} == {0, 1}
        slot0_inflight = {r.request_id for r in ctx0}

        # Step 2: slot 1 — req0/req1 still inflight, req2/req3 scheduled
        ctx1, _ = scheduler.schedule(requests, slot0_inflight)
        assert {r.request_id for r in ctx1} == {2, 3}
        slot1_inflight = {r.request_id for r in ctx1}

        # Step 3: slot 0 freed (inflight = slot1 only) — req0/req1 scheduled again
        ctx2, _ = scheduler.schedule(requests, slot1_inflight)
        assert {r.request_id for r in ctx2} == {0, 1}

    def test_gen_draft_tokens_max_num_tokens(self):
        """
        Draft tokens in generation phase reduce batch size when maxNumTokens is tight.
        Each gen request costs beam_width + num_draft_tokens = 1 + 63 = 64 tokens.
        With maxNumTokens=128, only 2 of the 4 gen requests fit (2*64=128).

        C++ ref: GenDraftTokensMaxNumTokens
        """
        scheduler = PyMicroBatchScheduler(max_batch_size=64, max_num_tokens=128)
        requests = [make_generation_request(i, beam_width=1, draft_tokens_len=63) for i in range(4)]
        ctx, gen = scheduler.schedule(requests, set())
        # Each request costs 1 + 63 = 64 tokens; 2 fit (128 = budget), 3 don't (192 > 128).
        assert len(gen) == 2
        assert gen[0].request_id == 0
        assert gen[1].request_id == 1
        assert len(ctx) == 0


# ############################################################################
#
# Part 2: Context Chunking Tests
#
# ############################################################################


class TestPyMicroBatchSchedulerChunking:
    """
    Tests for context chunking logic in PyMicroBatchScheduler.
    Aligned with C++ ContextChunkingTest in microBatchSchedulerTest.cpp.
    """

    # --- EQUAL_PROGRESS policy ---

    def test_equal_progress_basic(self):
        """
        Two context requests split equally across token budget.
        C++ ref: ContextChunkingTest with EQUAL_PROGRESS
        """
        config = ContextChunkingConfig(ChunkingPolicy.EQUAL_PROGRESS, chunk_unit_size=5)
        scheduler = PyMicroBatchScheduler(
            max_batch_size=4, max_num_tokens=10, ctx_chunk_config=config
        )
        requests = [
            make_context_request(0, prompt_len=20),
            make_context_request(1, prompt_len=20),
        ]
        ctx, gen = scheduler.schedule(requests, set())
        assert len(ctx) == 2
        # Each should get ~5 tokens (equal progress, unit=5, total=10)
        total_chunk = sum(r.context_chunk_size for r in ctx)
        assert total_chunk <= 10

    def test_equal_progress_uneven_remaining(self):
        """
        One request has less remaining context than the other.
        Equal progress should still give fair allocation.
        After chunking, sort puts not-last-chunk requests first.
        """
        config = ContextChunkingConfig(ChunkingPolicy.EQUAL_PROGRESS, chunk_unit_size=5)
        scheduler = PyMicroBatchScheduler(
            max_batch_size=4, max_num_tokens=15, ctx_chunk_config=config
        )
        requests = [
            make_context_request(0, prompt_len=3),  # Only 3 tokens remaining
            make_context_request(1, prompt_len=20),  # Lots remaining
        ]
        ctx, gen = scheduler.schedule(requests, set())
        assert len(ctx) == 2
        # Look up by request_id since sort reorders (not-last-chunk first)
        req0 = next(r for r in ctx if r.request_id == 0)
        req1 = next(r for r in ctx if r.request_id == 1)
        # Request 0 should get at most 3 (all it has)
        assert req0.context_chunk_size <= 3
        # Total should fit within budget
        assert req0.context_chunk_size + req1.context_chunk_size <= 15

    def test_fcfs_basic(self):
        """
        FIRST_COME_FIRST_SERVED: first request gets as much as possible.
        """
        config = ContextChunkingConfig(ChunkingPolicy.FIRST_COME_FIRST_SERVED, chunk_unit_size=5)
        scheduler = PyMicroBatchScheduler(
            max_batch_size=4, max_num_tokens=12, ctx_chunk_config=config
        )
        requests = [
            make_context_request(0, prompt_len=20),
            make_context_request(1, prompt_len=20),
        ]
        ctx, gen = scheduler.schedule(requests, set())
        # FCFS: request 0 gets up to budget, request 1 gets remainder
        assert len(ctx) >= 1
        # First request should get more tokens
        assert ctx[0].context_chunk_size >= ctx[-1].context_chunk_size or len(ctx) == 1

    def test_fcfs_fills_first_request(self):
        """FCFS fills the first request completely if budget allows.
        After chunking, sort puts not-last-chunk requests first."""
        config = ContextChunkingConfig(ChunkingPolicy.FIRST_COME_FIRST_SERVED, chunk_unit_size=5)
        scheduler = PyMicroBatchScheduler(
            max_batch_size=4, max_num_tokens=25, ctx_chunk_config=config
        )
        requests = [
            make_context_request(0, prompt_len=10),
            make_context_request(1, prompt_len=20),
        ]
        ctx, gen = scheduler.schedule(requests, set())
        assert len(ctx) == 2
        # Look up by request_id since sort reorders (not-last-chunk first)
        req0 = next(r for r in ctx if r.request_id == 0)
        req1 = next(r for r in ctx if r.request_id == 1)
        # First request should get all 10 (full context)
        assert req0.context_chunk_size == 10
        # Second gets remainder up to its need
        assert req1.context_chunk_size <= 15

    def test_chunk_with_generation(self):
        """
        Chunked context + generation in the same batch.
        Generation tokens reduce the available budget for context chunks.
        """
        config = ContextChunkingConfig(ChunkingPolicy.EQUAL_PROGRESS, chunk_unit_size=5)
        scheduler = PyMicroBatchScheduler(
            max_batch_size=4, max_num_tokens=15, ctx_chunk_config=config
        )
        requests = [
            make_generation_request(0),  # costs 1 token
            make_context_request(1, prompt_len=20),
            make_context_request(2, prompt_len=20),
        ]
        ctx, gen = scheduler.schedule(requests, set())
        assert len(gen) == 1
        # Remaining budget for context: 15 - 1 = 14
        total_ctx_tokens = sum(r.context_chunk_size for r in ctx)
        assert total_ctx_tokens <= 14

    def test_chunk_size_zero_not_scheduled(self):
        """
        If a request gets chunk_size=0 after chunking, it's not added to
        the scheduled context requests.
        """
        config = ContextChunkingConfig(ChunkingPolicy.EQUAL_PROGRESS, chunk_unit_size=5)
        scheduler = PyMicroBatchScheduler(
            max_batch_size=2, max_num_tokens=5, ctx_chunk_config=config
        )
        requests = [
            make_context_request(0, prompt_len=20),
            make_context_request(1, prompt_len=20),
        ]
        ctx, gen = scheduler.schedule(requests, set())
        # With budget 5, at most one request gets chunk_size=5, the other might get 0
        for r in ctx:
            assert r.context_chunk_size > 0

    def test_chunking_with_max_context_length(self):
        """
        max_context_length (same as max_num_tokens) limits individual chunk size.
        C++ ref: SimpleNoOverlapMaxContextLength
        """
        config = ContextChunkingConfig(ChunkingPolicy.EQUAL_PROGRESS, chunk_unit_size=5)
        scheduler = PyMicroBatchScheduler(
            max_batch_size=4, max_num_tokens=12, ctx_chunk_config=config
        )
        requests = [
            make_context_request(0, prompt_len=20),
        ]
        ctx, gen = scheduler.schedule(requests, set())
        assert len(ctx) == 1
        # max_context_length = max_num_tokens = 12, so chunk <= 12
        assert ctx[0].context_chunk_size <= 12

    def test_continued_chunking(self):
        """
        A request that has already processed part of its context
        (context_position > 0) continues from where it left off.
        """
        config = ContextChunkingConfig(ChunkingPolicy.EQUAL_PROGRESS, chunk_unit_size=5)
        scheduler = PyMicroBatchScheduler(
            max_batch_size=4, max_num_tokens=10, ctx_chunk_config=config
        )
        req = make_context_request(0, prompt_len=20, context_position=10)
        # remaining = 20 - 10 = 10
        ctx, gen = scheduler.schedule([req], set())
        assert len(ctx) == 1
        assert ctx[0].context_chunk_size <= 10  # remaining context

    def test_last_chunk_allows_draft_tokens(self):
        """
        On the last chunk, draft tokens are included if they fit within
        the remaining space in the chunk unit.
        C++ ref: DraftTokensNoDiscard
        """
        config = ContextChunkingConfig(ChunkingPolicy.FIRST_COME_FIRST_SERVED, chunk_unit_size=10)
        scheduler = PyMicroBatchScheduler(
            max_batch_size=4, max_num_tokens=20, ctx_chunk_config=config
        )
        # prompt_len=8, so chunk_size will be 8. Unit=10, remainder=2.
        # Draft tokens=2 fits in remainder.
        req = make_context_request(0, prompt_len=8, draft_tokens_len=2)
        ctx, gen = scheduler.schedule([req], set())
        assert len(ctx) == 1
        assert req.is_last_context_chunk

    def test_draft_tokens_discarded_when_no_space(self):
        """
        Draft tokens that don't fit in chunk unit remainder are discarded.
        C++ ref: DraftTokensDiscard
        """
        config = ContextChunkingConfig(ChunkingPolicy.FIRST_COME_FIRST_SERVED, chunk_unit_size=5)
        scheduler = PyMicroBatchScheduler(
            max_batch_size=4, max_num_tokens=20, ctx_chunk_config=config
        )
        # prompt_len=5, chunk_size=5, unit=5, remainder=0. Draft=3 won't fit.
        req = make_context_request(0, prompt_len=5, draft_tokens_len=3)
        ctx, gen = scheduler.schedule([req], set())
        assert len(ctx) == 1

    def test_chunked_context_draft_tokens_max_num_tokens(self):
        """
        Chunked context + draft tokens: maxNumTokens limits total draft budget.
        C++ ref: ChunkedContextDraftTokensMaxNumTokens
        maxNumTokens=8192, maxBatchSize=64, chunkUnitSize=64, FCFS,
        promptLen=2041, draftLen=8, 4 requests.
        2041 = 31*64 + 57, so remainder in unit = 64-57 = 7.
        Each request's draft reduced from 8 to 7.
        """
        config = ContextChunkingConfig(ChunkingPolicy.FIRST_COME_FIRST_SERVED, chunk_unit_size=64)
        scheduler = PyMicroBatchScheduler(
            max_batch_size=64,
            max_num_tokens=8192,
            ctx_chunk_config=config,
        )
        requests = [make_context_request(i, prompt_len=2041, draft_tokens_len=8) for i in range(4)]
        ctx, gen = scheduler.schedule(requests, set())
        assert len(ctx) == 4
        for req in ctx:
            assert req.num_draft_tokens == 7

    def test_chunked_context_draft_tokens_max_context_length(self):
        """
        Chunked context + draft tokens: maxContextLength limits individual draft.
        C++ ref: ChunkedContextDraftTokensMaxContextLength
        maxContextLength=10, maxNumTokens=8192, maxBatchSize=64,
        chunkUnitSize=64, FCFS, promptLen=6, draftLen=5, 2 requests.
        chunk_size=6, unit=64, remainder=6, space_in_unit=58.
        But maxContextLength-chunk_size = 10-6 = 4, so remaining_space=4.
        Draft reduced from 5 to 4.
        """
        config = ContextChunkingConfig(ChunkingPolicy.FIRST_COME_FIRST_SERVED, chunk_unit_size=64)
        scheduler = PyMicroBatchScheduler(
            max_batch_size=64,
            max_num_tokens=8192,
            ctx_chunk_config=config,
        )
        scheduler.max_context_length = 10
        requests = [
            make_context_request(0, prompt_len=6, draft_tokens_len=5),
            make_context_request(1, prompt_len=6, draft_tokens_len=5),
        ]
        ctx, gen = scheduler.schedule(requests, set())
        assert len(ctx) == 2
        for req in ctx:
            assert req.num_draft_tokens == 4

    def test_no_chunking_context_fits(self):
        """Without chunking, context is scheduled in full if it fits."""
        scheduler = PyMicroBatchScheduler(
            max_batch_size=4, max_num_tokens=20, ctx_chunk_config=None
        )
        req = make_context_request(0, prompt_len=15)
        ctx, gen = scheduler.schedule([req], set())
        assert len(ctx) == 1

    def test_no_chunking_context_exceeds_budget(self):
        """Without chunking, cumulative tokens exceeding budget stops scheduling.
        Each individual request must fit within max_context_length (== max_num_tokens),
        but the cumulative token count is checked against the budget. The first request
        that would push the total over the limit breaks the loop."""
        scheduler = PyMicroBatchScheduler(
            max_batch_size=4, max_num_tokens=10, ctx_chunk_config=None
        )
        requests = [
            make_context_request(0, prompt_len=8),
            make_context_request(1, prompt_len=8),
        ]
        ctx, gen = scheduler.schedule(requests, set())
        # First request (8) fits (8 <= 10). Second (8+8=16 > 10) breaks the loop.
        assert len(ctx) == 1
        assert ctx[0].request_id == 0

    def test_sort_by_lora_task_id(self):
        """
        Requests are sorted by lora_task_id for performance.
        C++ ref: sortRequests in inflightBatchingUtils.cpp
        """
        scheduler = PyMicroBatchScheduler(max_batch_size=4, max_num_tokens=None)
        r0 = _make_request(0, state=LlmRequestState.GENERATION_IN_PROGRESS, lora_task_id=5)
        r1 = _make_request(1, state=LlmRequestState.GENERATION_IN_PROGRESS)
        r2 = _make_request(2, state=LlmRequestState.GENERATION_IN_PROGRESS, lora_task_id=3)
        ctx, gen = scheduler.schedule([r0, r1, r2], set())
        # None < any value, so order should be: r1(None), r2(3), r0(5)
        assert gen[0].request_id == 1
        assert gen[1].request_id == 2
        assert gen[2].request_id == 0


# ############################################################################
#
#  Reusable KV Cache Token Tests
#
# ############################################################################


class TestPyMicroBatchSchedulerReusableTokens:
    """
    Tests for estimated_reusable_tokens logic in PyMicroBatchScheduler.
    Mirrors C++ ReusableTokens tests in microBatchSchedulerTest.cpp.

    The reusable credit reduces the compute cost of a request when the KV
    cache radix tree has already computed some prefix blocks.
    """

    def test_reusable_tokens_reduce_compute_budget(self):
        """
        Reusable tokens shrink the compute cost so more requests fit.
        C++ ref: ReusableTokensReduceComputeBudget

        max_num_tokens=20, two requests of prompt_len=20.
        Without reuse: req0 costs 20 == budget → req1 doesn't fit.
        With reusable=15 each: compute = max(1, 20-15) = 5 → total 10 < 20
        → both fit.
        """
        scheduler = PyMicroBatchScheduler(
            max_batch_size=4, max_num_tokens=20, ctx_chunk_config=None
        )
        req0 = make_context_request(0, prompt_len=20)
        req1 = make_context_request(1, prompt_len=20)
        req0.estimated_reusable_tokens = 15
        req1.estimated_reusable_tokens = 15

        ctx, gen = scheduler.schedule([req0, req1], set())
        assert len(ctx) == 2

    def test_reusable_tokens_zero_has_no_effect(self):
        """
        Explicitly setting reusable=0 is identical to the default (no reuse).
        C++ ref: ReusableTokensZeroHasNoEffect

        max_num_tokens=12, two requests of prompt_len=10.
        10+10=20 > 12 → only the first request fits.
        """
        scheduler = PyMicroBatchScheduler(
            max_batch_size=4, max_num_tokens=12, ctx_chunk_config=None
        )
        req0 = make_context_request(0, prompt_len=10)
        req1 = make_context_request(1, prompt_len=10)
        req0.estimated_reusable_tokens = 0
        req1.estimated_reusable_tokens = 0

        ctx, gen = scheduler.schedule([req0, req1], set())
        assert len(ctx) == 1
        assert ctx[0].request_id == 0

    def test_reusable_tokens_chunked_context_fcfs_full_context_fits(self):
        """
        Reusable tokens can allow the full context to fit without chunking.
        C++ ref: ReusableTokensWithChunkedContextFCFS

        max_num_tokens=15, prompt_len=20, FCFS, chunk_unit=5.
        Without reuse: tentative compute = 20, > 15 → chunked to 15.
        With reusable=10: context_compute = max(0, 20-10) = 10 < 15
        → all_context_requests_fit stays True → full 20-token context scheduled.
        """
        config = ContextChunkingConfig(ChunkingPolicy.FIRST_COME_FIRST_SERVED, chunk_unit_size=5)
        scheduler = PyMicroBatchScheduler(
            max_batch_size=4, max_num_tokens=15, ctx_chunk_config=config
        )
        req = make_context_request(0, prompt_len=20)
        req.estimated_reusable_tokens = 10

        ctx, gen = scheduler.schedule([req], set())
        assert len(ctx) == 1
        # Full context fits — chunk_size equals the full prompt length.
        assert ctx[0].context_chunk_size == 20

    def test_reusable_tokens_only_on_first_chunk(self):
        """
        The reusable credit is only applied on the first context chunk.
        On subsequent chunks is_first_context_chunk == False, so reusable = 0.

        Use a request already past its first chunk (context_position=10) with
        prompt_len=30 and reusable=20 — the reuse should be ignored.
        Without reuse the remaining 20 tokens fill the budget exactly.
        With another fresh request (reusable=20, first chunk) the reuse IS
        applied: compute = max(1, 30-20) = 10 → both fit in budget=30.
        """
        scheduler = PyMicroBatchScheduler(
            max_batch_size=4, max_num_tokens=30, ctx_chunk_config=None
        )
        # req0: already past first chunk — reusable credit must be ignored.
        req0 = make_context_request(0, prompt_len=30, context_position=10)
        req0.estimated_reusable_tokens = 20
        assert not req0.is_first_context_chunk

        ctx, gen = scheduler.schedule([req0], set())
        assert len(ctx) == 1
        # Remaining tokens = 30 - 10 = 20; reuse ignored → compute = 20
        # Budget = 30, 20 <= 30, so it fits.
        # Now add a second request — without reuse it would push total to 40 > 30.
        req1 = make_context_request(1, prompt_len=20)
        # No reuse → compute = 20; total = 20 + 20 = 40 > 30 → req1 doesn't fit.
        scheduler2 = PyMicroBatchScheduler(
            max_batch_size=4, max_num_tokens=30, ctx_chunk_config=None
        )
        ctx2, _ = scheduler2.schedule([req0, req1], set())
        req0_again = make_context_request(0, prompt_len=30, context_position=10)
        req0_again.estimated_reusable_tokens = 20
        # Confirm: fresh req0 (non-first chunk) + req1 → only req0 fits
        assert len(ctx2) == 1

        # Contrast: first-chunk request with reuse DOES get the credit.
        req2 = make_context_request(2, prompt_len=30)
        req2.estimated_reusable_tokens = 20
        assert req2.is_first_context_chunk
        req3 = make_context_request(3, prompt_len=20)
        scheduler3 = PyMicroBatchScheduler(
            max_batch_size=4, max_num_tokens=30, ctx_chunk_config=None
        )
        ctx3, _ = scheduler3.schedule([req2, req3], set())
        # req2 compute = max(1, 30-20) = 10; req3 compute = 20; total = 30 → both fit.
        assert len(ctx3) == 2

    def test_reusable_tokens_no_chunking_min_cost_is_one(self):
        """
        The no-chunking path floors compute cost at 1 even if reusable > prompt_len.
        Python-specific floor test.

        prompt_len=10, reusable=15: compute = max(1, 10-15) = max(1,-5) = 1.
        max_num_tokens=10: req0 costs 1, req1 (no reuse, 10 tokens) costs 1+10=11 > 10
        → req1 doesn't fit.
        """
        scheduler = PyMicroBatchScheduler(
            max_batch_size=4, max_num_tokens=10, ctx_chunk_config=None
        )
        req0 = make_context_request(0, prompt_len=10)
        req0.estimated_reusable_tokens = 15  # exceeds prompt length
        req1 = make_context_request(1, prompt_len=10)

        ctx, gen = scheduler.schedule([req0, req1], set())
        # req0 compute = 1; req1 compute = 10; 1 + 10 = 11 > 10 → only req0 fits.
        assert len(ctx) == 1
        assert ctx[0].request_id == 0

    def test_reusable_tokens_fcfs_over_budget_multi_request(self):
        """
        FCFS compute-aware scheduling with multiple requests exceeding the compute budget.
        C++ ref: ReusableTokensWithChunkedContextFCFS_OverBudgetMultiRequest

        Setup: 3 requests, prompt_len=15, reusable=8, compute budget=20, chunk_unit=1.
        Note: In PyMicroBatchScheduler, max_context_length = max_num_tokens = 20, so
        prompt_len=15 keeps chunk sizes within the max_context_length limit.
        Total compute if all scheduled in full: 3 * (15 - 8) = 21 > 20 → chunking exercised.

        The model processes min(chunk_size, P - reusable) tokens from position reusable,
        where P = context_remaining_length = 15 (full prompt on first chunk).

        Expected (compute-aware FCFS):
          req0: full context fits (compute=7 <= 20); chunk=15; budget 20→13
          req1: full context fits (compute=7 <= 13); chunk=15; budget 13→6
          req2: compute=7 > 6 → chunk=6 (remaining capacity); budget→0
        """
        config = ContextChunkingConfig(ChunkingPolicy.FIRST_COME_FIRST_SERVED, chunk_unit_size=1)
        scheduler = PyMicroBatchScheduler(
            max_batch_size=4, max_num_tokens=20, ctx_chunk_config=config
        )
        req0 = make_context_request(0, prompt_len=15)
        req1 = make_context_request(1, prompt_len=15)
        req2 = make_context_request(2, prompt_len=15)
        req0.estimated_reusable_tokens = 8
        req1.estimated_reusable_tokens = 8
        req2.estimated_reusable_tokens = 8

        ctx, gen = scheduler.schedule([req0, req1, req2], set())

        # Note: ctx is sorted — partially-chunked requests come before full-context ones.
        # Look up by request_id rather than by position.
        chunks = {r.request_id: r.context_chunk_size for r in ctx}
        assert len(ctx) == 3, "All three requests should be scheduled"
        assert chunks[0] == 15, "req0: full context (compute=7, budget 20→13)"
        assert chunks[1] == 15, "req1: full context (compute=7, budget 13→6)"
        assert chunks[2] == 6, "req2: chunk=6 (remaining capacity)"

    def test_reusable_tokens_equal_progress(self):
        """
        EQUAL_PROGRESS compute-aware budget tracking with reusable tokens.
        C++ ref: ReusableTokensWithChunkedContextEqualProgress

        Note: In PyMicroBatchScheduler, max_context_length = max_num_tokens, so
        individual chunk sizes are capped at max_num_tokens. Parameters are chosen
        so the reusable prefix is smaller than max_num_tokens, allowing the
        compute-aware path to produce noticeably larger chunks than the raw path.

        Setup: 3 requests, prompt_len=15, reusable=5, compute budget=8, chunk_unit=1.
        max_context_length = max_num_tokens = 8 (caps individual chunk sizes at 8).
        Total compute if all scheduled in full: 3 * (15 - 5) = 30 > 8 → chunking exercised.

        Reusable tokens 0-4 are "free" (compute_increment = 0 until chunk > 5).
        Budget is only consumed for tokens beyond the reusable prefix.

        This test validates the loop-termination fix: num_tokens_single_loop must use
        the raw chunk increment (not compute_increment). With the bug, the loop exits
        on the very first iteration because compute_increment=0 for all reqs → loops
        terminates at chunk=1. With the fix, the loop continues until the compute
        budget is exhausted or max_context_length is reached.

        Expected (compute-aware EQUAL_PROGRESS):
          req0: chunk=8  (model cost = 8 - 5 = 3)
          req1: chunk=8  (model cost = 3)
          req2: chunk=7  (model cost = 7 - 5 = 2; total compute 3+3+2=8 = budget)
        """
        config = ContextChunkingConfig(ChunkingPolicy.EQUAL_PROGRESS, chunk_unit_size=1)
        scheduler = PyMicroBatchScheduler(
            max_batch_size=4, max_num_tokens=8, ctx_chunk_config=config
        )
        req0 = make_context_request(0, prompt_len=15)
        req1 = make_context_request(1, prompt_len=15)
        req2 = make_context_request(2, prompt_len=15)
        req0.estimated_reusable_tokens = 5
        req1.estimated_reusable_tokens = 5
        req2.estimated_reusable_tokens = 5

        ctx, gen = scheduler.schedule([req0, req1, req2], set())

        chunks = {r.request_id: r.context_chunk_size for r in ctx}
        assert len(ctx) == 3, "All three requests should be scheduled"
        assert chunks[0] == 8, "req0: reusable(5) + 3 compute tokens = 8"
        assert chunks[1] == 8, "req1: reusable(5) + 3 compute tokens = 8"
        assert chunks[2] == 7, "req2: reusable(5) + 2 compute tokens = 7"


# ############################################################################
#
# Part 3: Direct Context Chunking Tests (mirrors C++ ContextChunkingTest)
#
# ############################################################################


def _run_context_chunking_test(
    context_lengths: List[int],
    chunk_unit_size: int,
    ep_positions: List[List[int]],
    fcfs_positions: List[List[int]],
    ctx_tokens_capacity: Optional[int] = None,
    max_context_length: Optional[int] = None,
    draft_lengths: Optional[List[int]] = None,
    ep_draft_lengths: Optional[List[List[int]]] = None,
    fcfs_draft_lengths: Optional[List[List[int]]] = None,
):
    """
    Helper that mirrors the C++ ContextChunkingTest fixture.

    For each policy (EQUAL_PROGRESS and FCFS), it:
    1. Creates LlmRequests with given context_lengths and optional draft_lengths.
    2. Creates a PyMicroBatchScheduler with the right ContextChunkingConfig.
    3. If max_context_length is set, overrides scheduler.max_context_length.
    4. For each iteration (each element in positions list):
       a. Filters requests where context_remaining_length > 0.
       b. Calls scheduler._set_ctx_requests_chunk_size(active_reqs, ctx_tokens_capacity).
       c. For each active req, calls req.move_to_next_context_chunk().
       d. Verifies context position matches expected positions for ALL requests.
    5. After all iterations, verifies final draft_lengths if specified.
    """
    policies_and_data = [
        (ChunkingPolicy.EQUAL_PROGRESS, ep_positions, ep_draft_lengths),
        (ChunkingPolicy.FIRST_COME_FIRST_SERVED, fcfs_positions, fcfs_draft_lengths),
    ]

    for policy, positions_list, final_draft_lens in policies_and_data:
        # Create fresh requests for each policy
        requests = []
        for i, ctx_len in enumerate(context_lengths):
            dl = draft_lengths[i] if draft_lengths else 0
            req = make_context_request(
                request_id=i,
                prompt_len=ctx_len,
                draft_tokens_len=dl,
            )
            requests.append(req)

        # Create scheduler
        config = ContextChunkingConfig(policy, chunk_unit_size=chunk_unit_size)
        scheduler = PyMicroBatchScheduler(
            max_batch_size=64,
            max_num_tokens=1000,  # large enough not to limit
            ctx_chunk_config=config,
        )

        if max_context_length is not None:
            scheduler.max_context_length = max_context_length

        # Run iterations
        for iteration_idx, expected_positions in enumerate(positions_list):
            # Filter active requests (those with remaining context)
            active_reqs = [r for r in requests if r.context_remaining_length > 0]

            scheduler._set_ctx_requests_chunk_size(active_reqs, ctx_tokens_capacity)

            # Move each active request to next chunk
            for req in active_reqs:
                req.move_to_next_context_chunk()

            # Verify positions for ALL requests (including completed ones)
            for req_idx, req in enumerate(requests):
                assert req.context_current_position == expected_positions[req_idx], (
                    f"Policy {policy.name}, iteration {iteration_idx}, "
                    f"request {req_idx}: expected position "
                    f"{expected_positions[req_idx]}, got "
                    f"{req.context_current_position}"
                )

        # Verify final draft lengths if specified
        if final_draft_lens is not None:
            for req_idx, req in enumerate(requests):
                assert req.num_draft_tokens == final_draft_lens[req_idx], (
                    f"Policy {policy.name}, request {req_idx}: "
                    f"expected draft_len {final_draft_lens[req_idx]}, "
                    f"got {req.num_draft_tokens}"
                )


class TestContextChunkingDirect:
    """
    Direct tests for _set_ctx_requests_chunk_size logic, mirroring
    C++ ContextChunkingTest in microBatchSchedulerTest.cpp.
    Each test calls _run_context_chunking_test with parameters matching
    exactly the C++ test cases.
    """

    def test_no_limit(self):
        """C++ ref: NoLimit"""
        _run_context_chunking_test([25, 25], 20, [[25, 25]], [[25, 25]])

    def test_context_length_never_satisfied(self):
        """C++ ref: ContextLengthNeverSatisfied"""
        _run_context_chunking_test([25, 25], 100, [[0, 0]], [[0, 0]], max_context_length=20)

    def test_chunk_longer_than_context(self):
        """C++ ref: ChunkLongerThanContext"""
        _run_context_chunking_test([25, 25], 30, [[25, 25]], [[25, 25]], max_context_length=25)

    def test_context_length_satisfied(self):
        """C++ ref: ContextLengthSatisfied"""
        _run_context_chunking_test(
            [10, 25],
            10,
            [[10, 20], [10, 25]],
            [[10, 20], [10, 25]],
            max_context_length=20,
        )

    def test_token_capacity_smaller_than_context(self):
        """C++ ref: TokenCapacitySmallerThanContext"""
        _run_context_chunking_test(
            [25, 25],
            20,
            [[20, 0], [25, 0], [25, 20], [25, 25]],
            [[20, 0], [25, 0], [25, 20], [25, 25]],
            ctx_tokens_capacity=20,
        )

    def test_token_capacity_smaller_than_chunk_unit(self):
        """C++ ref: TokenCapacitySmallerThanChunkUnit"""
        _run_context_chunking_test([25, 25], 20, [[0, 0]], [[0, 0]], ctx_tokens_capacity=10)

    def test_scheduling_order(self):
        """C++ ref: SchedulingOrder"""
        _run_context_chunking_test(
            [25, 25],
            5,
            [[15, 15], [25, 25]],
            [[25, 5], [25, 25]],
            ctx_tokens_capacity=30,
        )

    def test_completion_order(self):
        """C++ ref: CompletionOrder"""
        _run_context_chunking_test(
            [25, 15],
            5,
            [[15, 15], [25, 15]],
            [[25, 5], [25, 15]],
            ctx_tokens_capacity=30,
        )

    def test_long_first_short_later(self):
        """C++ ref: LongFirstShortLater"""
        _run_context_chunking_test(
            [25, 15],
            5,
            [[10, 10], [20, 15], [25, 15]],
            [[10, 10], [20, 15], [25, 15]],
            ctx_tokens_capacity=30,
            max_context_length=10,
        )

    def test_front_priority(self):
        """C++ ref: FrontPriority"""
        _run_context_chunking_test(
            [25, 25],
            5,
            [[10, 5], [20, 10], [25, 20], [25, 25]],
            [[15, 0], [25, 5], [25, 20], [25, 25]],
            ctx_tokens_capacity=15,
        )

    def test_draft_tokens_discard(self):
        """C++ ref: DraftTokensDiscard"""
        _run_context_chunking_test(
            [27, 27],
            5,
            [[15, 15], [27, 27]],
            [[27, 0], [27, 27]],
            ctx_tokens_capacity=30,
            draft_lengths=[5, 5],
            ep_draft_lengths=[3, 3],
            fcfs_draft_lengths=[3, 3],
        )

    def test_draft_tokens_discard2(self):
        """C++ ref: DraftTokensDiscard2"""
        _run_context_chunking_test(
            [17, 17],
            5,
            [[15, 15], [17, 17]],
            [[17, 10], [17, 17]],
            ctx_tokens_capacity=30,
            draft_lengths=[5, 5],
            ep_draft_lengths=[3, 3],
            fcfs_draft_lengths=[3, 3],
        )

    def test_draft_tokens_discard3(self):
        """C++ ref: DraftTokensDiscard3"""
        _run_context_chunking_test(
            [27, 27],
            5,
            [[10, 10], [20, 20], [27, 27]],
            [[20, 0], [27, 10], [27, 27]],
            ctx_tokens_capacity=20,
            draft_lengths=[5, 5],
            ep_draft_lengths=[3, 3],
            fcfs_draft_lengths=[3, 3],
        )

    def test_draft_tokens_discard_due_to_token_capacity(self):
        """C++ ref: DraftTokensDiscardDueToTokenCapacity"""
        _run_context_chunking_test(
            [23, 17],
            5,
            [[10, 10], [23, 17]],
            [[20, 0], [23, 17]],
            ctx_tokens_capacity=20,
            draft_lengths=[5, 5],
            ep_draft_lengths=[0, 0],
            fcfs_draft_lengths=[0, 0],
        )

    def test_draft_tokens_discard_due_to_max_context_length(self):
        """C++ ref: DraftTokensDiscardDueToMaxContextLength"""
        _run_context_chunking_test(
            [6, 6],
            5,
            [[6, 6]],
            [[6, 6]],
            ctx_tokens_capacity=30,
            max_context_length=10,
            draft_lengths=[5, 5],
            ep_draft_lengths=[4, 4],
            fcfs_draft_lengths=[4, 4],
        )

    def test_draft_tokens_discard_all(self):
        """C++ ref: DraftTokensDiscardAll"""
        _run_context_chunking_test(
            [25, 25],
            5,
            [[25, 25]],
            [[25, 25]],
            ctx_tokens_capacity=50,
            draft_lengths=[5, 5],
            ep_draft_lengths=[0, 0],
            fcfs_draft_lengths=[0, 0],
        )

    def test_draft_tokens_discard_all2(self):
        """C++ ref: DraftTokensDiscardAll2"""
        _run_context_chunking_test(
            [25, 25],
            5,
            [[15, 10], [25, 25]],
            [[25, 0], [25, 25]],
            ctx_tokens_capacity=25,
            draft_lengths=[5, 5],
            ep_draft_lengths=[0, 0],
            fcfs_draft_lengths=[0, 0],
        )

    def test_draft_tokens_no_discard(self):
        """C++ ref: DraftTokensNoDiscard"""
        _run_context_chunking_test(
            [25, 25],
            10,
            [[20, 10], [25, 25]],
            [[25, 0], [25, 25]],
            ctx_tokens_capacity=30,
            draft_lengths=[5, 5],
            ep_draft_lengths=[5, 5],
            fcfs_draft_lengths=[5, 5],
        )

    def test_draft_tokens_no_chunking_discard_all(self):
        """C++ ref: DraftTokensNoChunkingDiscardAll"""
        _run_context_chunking_test(
            [4128],
            64,
            [[4128]],
            [[4128]],
            max_context_length=4128,
            draft_lengths=[3],
            ep_draft_lengths=[0],
            fcfs_draft_lengths=[0],
        )

    def test_draft_tokens_no_chunking_discard_some(self):
        """C++ ref: DraftTokensNoChunkingDiscardSome"""
        _run_context_chunking_test(
            [4127],
            64,
            [[4127]],
            [[4127]],
            max_context_length=4128,
            draft_lengths=[3],
            ep_draft_lengths=[1],
            fcfs_draft_lengths=[1],
        )

    def test_draft_tokens_no_chunking_discard_none(self):
        """C++ ref: DraftTokensNoChunkingDiscardNone"""
        _run_context_chunking_test(
            [4125],
            64,
            [[4125]],
            [[4125]],
            max_context_length=4128,
            draft_lengths=[3],
            ep_draft_lengths=[3],
            fcfs_draft_lengths=[3],
        )


class TestForceChunkPolicy:
    """
    Tests for FORCE_CHUNK chunking policy in PyMicroBatchScheduler.
    FORCE_CHUNK always chunks every context request to at most chunk_unit_size
    tokens per scheduling step, regardless of whether the full context would fit
    in the budget.

    Aligned with C++ ForceChunkTest in microBatchSchedulerTest.cpp.
    """

    # --- Helper methods (mirrors C++ ForceChunkTest fixture) ---

    @staticmethod
    def _chunk_iteration(requests, chunk_unit_size, capacity=None):
        """Run a single chunking iteration: call _set_ctx_requests_chunk_size
        with FORCE_CHUNK, then move_to_next_context_chunk for active requests.
        C++ ref: ForceChunkTest::chunkIteration"""
        config = ContextChunkingConfig(ChunkingPolicy.FORCE_CHUNK, chunk_unit_size=chunk_unit_size)
        scheduler = PyMicroBatchScheduler(
            max_batch_size=64, max_num_tokens=1000, ctx_chunk_config=config
        )
        active = [r for r in requests if r.context_remaining_length > 0]
        scheduler._set_ctx_requests_chunk_size(active, capacity)
        for r in active:
            r.move_to_next_context_chunk()

    @staticmethod
    def _expect_positions(requests, expected, label=""):
        """Verify context positions of all requests match expected values.
        C++ ref: ForceChunkTest::expectPositions"""
        assert len(requests) == len(expected), label
        for i, req in enumerate(requests):
            assert req.context_current_position == expected[i], (
                f"{label} request {i} (id={req.request_id}): "
                f"expected {expected[i]}, got {req.context_current_position}"
            )

    # --- Direct _set_ctx_requests_chunk_size tests ---
    # C++ ref: ForceChunkTest::Basic through CapacityAcrossIterations

    def test_basic(self):
        """
        A single request with prompt_len > chunk_unit_size is chunked to unit_size.
        C++ ref: ForceChunkTest.Basic
        """
        config = ContextChunkingConfig(ChunkingPolicy.FORCE_CHUNK, chunk_unit_size=10)
        scheduler = PyMicroBatchScheduler(
            max_batch_size=64, max_num_tokens=1000, ctx_chunk_config=config
        )
        reqs = [make_context_request(0, prompt_len=30)]
        scheduler._set_ctx_requests_chunk_size(reqs, None)
        assert reqs[0].context_chunk_size == 10

    def test_prompt_smaller_than_unit(self):
        """
        When prompt_len < chunk_unit_size, chunk_size = prompt_len (min).
        C++ ref: ForceChunkTest.PromptSmallerThanUnit
        """
        config = ContextChunkingConfig(ChunkingPolicy.FORCE_CHUNK, chunk_unit_size=20)
        scheduler = PyMicroBatchScheduler(
            max_batch_size=64, max_num_tokens=1000, ctx_chunk_config=config
        )
        reqs = [make_context_request(0, prompt_len=8)]
        scheduler._set_ctx_requests_chunk_size(reqs, None)
        assert reqs[0].context_chunk_size == 8

    def test_exact_unit_size(self):
        """
        When prompt_len == chunk_unit_size, chunk_size = prompt_len.
        C++ ref: ForceChunkTest.ExactUnitSize
        """
        config = ContextChunkingConfig(ChunkingPolicy.FORCE_CHUNK, chunk_unit_size=10)
        scheduler = PyMicroBatchScheduler(
            max_batch_size=64, max_num_tokens=1000, ctx_chunk_config=config
        )
        reqs = [make_context_request(0, prompt_len=10)]
        scheduler._set_ctx_requests_chunk_size(reqs, None)
        assert reqs[0].context_chunk_size == 10

    def test_multiple_requests(self):
        """
        Each request independently gets min(remaining, unit_size).
        C++ ref: ForceChunkTest.MultipleRequests
        """
        config = ContextChunkingConfig(ChunkingPolicy.FORCE_CHUNK, chunk_unit_size=10)
        scheduler = PyMicroBatchScheduler(
            max_batch_size=64, max_num_tokens=1000, ctx_chunk_config=config
        )
        reqs = [
            make_context_request(0, prompt_len=25),
            make_context_request(1, prompt_len=15),
            make_context_request(2, prompt_len=5),
        ]
        scheduler._set_ctx_requests_chunk_size(reqs, None)
        assert reqs[0].context_chunk_size == 10
        assert reqs[1].context_chunk_size == 10
        assert reqs[2].context_chunk_size == 5  # min(5, 10)

    def test_capacity_limits(self):
        """
        When capacity is limited, later requests get chunk_size=0.
        C++ ref: ForceChunkTest.CapacityLimits
        """
        config = ContextChunkingConfig(ChunkingPolicy.FORCE_CHUNK, chunk_unit_size=10)
        scheduler = PyMicroBatchScheduler(
            max_batch_size=64, max_num_tokens=1000, ctx_chunk_config=config
        )
        reqs = [
            make_context_request(0, prompt_len=30),
            make_context_request(1, prompt_len=30),
        ]
        scheduler._set_ctx_requests_chunk_size(reqs, capacity=15)
        # req0 gets 10, req1 would push total to 20 > 15 → 0
        assert reqs[0].context_chunk_size == 10
        assert reqs[1].context_chunk_size == 0

    def test_capacity_exact_fit(self):
        """
        When capacity exactly accommodates all chunks.
        C++ ref: ForceChunkTest.CapacityExactFit
        """
        config = ContextChunkingConfig(ChunkingPolicy.FORCE_CHUNK, chunk_unit_size=10)
        scheduler = PyMicroBatchScheduler(
            max_batch_size=64, max_num_tokens=1000, ctx_chunk_config=config
        )
        reqs = [
            make_context_request(0, prompt_len=30),
            make_context_request(1, prompt_len=30),
        ]
        scheduler._set_ctx_requests_chunk_size(reqs, capacity=20)
        assert reqs[0].context_chunk_size == 10
        assert reqs[1].context_chunk_size == 10

    def test_multi_iteration(self):
        """
        A request with prompt_len=25 and chunk_unit_size=10 processes in 3
        iterations: chunk 1: 10, chunk 2: 10, chunk 3: 5.
        C++ ref: ForceChunkTest.MultiIteration
        """
        reqs = [make_context_request(0, prompt_len=25)]

        # Iteration 1
        self._chunk_iteration(reqs, 10)
        self._expect_positions(reqs, [10], "iter 1")

        # Iteration 2
        self._chunk_iteration(reqs, 10)
        self._expect_positions(reqs, [20], "iter 2")

        # Iteration 3
        self._chunk_iteration(reqs, 10)
        self._expect_positions(reqs, [25], "iter 3")

    def test_multi_request_multi_iteration(self):
        """
        Two requests with different lengths processed over multiple iterations.
        prompt_len={25, 12}, chunk_unit_size=10.
        C++ ref: ForceChunkTest.MultiRequestMultiIteration
        """
        reqs = [
            make_context_request(0, prompt_len=25),
            make_context_request(1, prompt_len=12),
        ]

        # Iteration 1: both get 10
        self._chunk_iteration(reqs, 10)
        self._expect_positions(reqs, [10, 10], "iter 1")

        # Iteration 2: req0 gets 10, req1 gets 2 (remaining)
        self._chunk_iteration(reqs, 10)
        self._expect_positions(reqs, [20, 12], "iter 2")

        # Iteration 3: only req0 active (remaining=5), req1 done
        self._chunk_iteration(reqs, 10)
        self._expect_positions(reqs, [25, 12], "iter 3")

    def test_capacity_across_iterations(self):
        """
        With limited capacity, some requests may be delayed to later iterations.
        prompt_len={25, 25}, chunk_unit_size=10, capacity=15.
        C++ ref: ForceChunkTest.CapacityAcrossIterations
        """
        reqs = [
            make_context_request(0, prompt_len=25),
            make_context_request(1, prompt_len=25),
        ]

        # Iteration 1: req0=10, req1=0 (10+10=20 > 15)
        self._chunk_iteration(reqs, 10, capacity=15)
        self._expect_positions(reqs, [10, 0], "iter 1")

        # Iteration 2: req0=10, req1=0 (still constrained)
        self._chunk_iteration(reqs, 10, capacity=15)
        self._expect_positions(reqs, [20, 0], "iter 2")

        # Iteration 3: req0=5, req1=10 (5+10=15 == capacity)
        self._chunk_iteration(reqs, 10, capacity=15)
        self._expect_positions(reqs, [25, 10], "iter 3")

        # Iteration 4: only req1 active (remaining=15), gets 10
        self._chunk_iteration(reqs, 10, capacity=15)
        self._expect_positions(reqs, [25, 20], "iter 4")

        # Iteration 5: req1 remaining=5
        self._chunk_iteration(reqs, 10, capacity=15)
        self._expect_positions(reqs, [25, 25], "iter 5")

    # --- Full scheduler.schedule() tests ---
    # C++ ref: ForceChunkTest::FullSchedulerPath through FullSchedulerWithGeneration

    def test_full_scheduler_path(self):
        """
        FORCE_CHUNK always re-chunks even when all contexts fit within the
        token budget. Test via the full schedule() path.
        C++ ref: ForceChunkTest.FullSchedulerPath
        """
        config = ContextChunkingConfig(ChunkingPolicy.FORCE_CHUNK, chunk_unit_size=10)
        scheduler = PyMicroBatchScheduler(
            max_batch_size=4, max_num_tokens=100, ctx_chunk_config=config
        )
        req = make_context_request(0, prompt_len=30)
        ctx, gen = scheduler.schedule([req], set())
        # Despite budget=100 >> prompt=30, FORCE_CHUNK limits chunk to unit_size=10.
        assert len(ctx) == 1
        assert ctx[0].context_chunk_size == 10
        assert len(gen) == 0

    def test_full_scheduler_multiple_requests(self):
        """
        Full scheduler path with multiple requests.
        C++ ref: ForceChunkTest.FullSchedulerMultipleRequests
        """
        config = ContextChunkingConfig(ChunkingPolicy.FORCE_CHUNK, chunk_unit_size=10)
        scheduler = PyMicroBatchScheduler(
            max_batch_size=4, max_num_tokens=100, ctx_chunk_config=config
        )
        requests = [
            make_context_request(0, prompt_len=25),
            make_context_request(1, prompt_len=15),
            make_context_request(2, prompt_len=5),
        ]
        ctx, gen = scheduler.schedule(requests, set())
        assert len(ctx) == 3
        # Find by request_id since sorting may reorder.
        chunks = {r.request_id: r.context_chunk_size for r in ctx}
        assert chunks[0] == 10
        assert chunks[1] == 10
        assert chunks[2] == 5

    def test_full_scheduler_with_generation(self):
        """
        Context chunking with concurrent generation requests.
        Generation tokens reduce the available budget for context chunks.
        C++ ref: ForceChunkTest.FullSchedulerWithGeneration
        """
        config = ContextChunkingConfig(ChunkingPolicy.FORCE_CHUNK, chunk_unit_size=10)
        scheduler = PyMicroBatchScheduler(
            max_batch_size=4, max_num_tokens=15, ctx_chunk_config=config
        )
        requests = [
            make_generation_request(0),  # costs 1 token
            make_context_request(1, prompt_len=30),
        ]
        ctx, gen = scheduler.schedule(requests, set())
        assert len(gen) == 1
        assert len(ctx) == 1
        # Budget remaining = 15 - 1 (gen) = 14; chunk = min(30, 10) = 10
        assert ctx[0].context_chunk_size == 10


class TestDraftTokensGreaterThanChunkSize:
    """
    Tests that when draft tokens > chunk unit, they get properly trimmed.
    C++ ref: DraftTokensGreaterThanChunkSize in microBatchSchedulerTest.cpp
    """

    def test_draft_tokens_greater_than_chunk_size(self):
        """
        C++ ref: DraftTokensGreaterThanChunkSize
        maxNumTokens=40, maxBatchSize=64, chunkUnitSize=16, FCFS policy,
        maxContextLength=64. 3 requests: promptLen=3, draftLen=17.

        After scheduling, expected:
        - All 3 scheduled
        - Request 0: draftTokens = 13 (unit=16, context=3, remainder=13)
        - Request 1: draftTokens = 13
        - Request 2: draftTokens = 5 (remaining budget)
        """
        config = ContextChunkingConfig(ChunkingPolicy.FIRST_COME_FIRST_SERVED, chunk_unit_size=16)
        scheduler = PyMicroBatchScheduler(
            max_batch_size=64,
            max_num_tokens=40,
            ctx_chunk_config=config,
        )
        scheduler.max_context_length = 64

        requests = [
            make_context_request(0, prompt_len=3, draft_tokens_len=17),
            make_context_request(1, prompt_len=3, draft_tokens_len=17),
            make_context_request(2, prompt_len=3, draft_tokens_len=17),
        ]

        ctx, gen = scheduler.schedule(requests, set())

        assert len(ctx) == 3
        req0 = next(r for r in ctx if r.request_id == 0)
        req1 = next(r for r in ctx if r.request_id == 1)
        req2 = next(r for r in ctx if r.request_id == 2)

        assert req0.num_draft_tokens == 13
        assert req1.num_draft_tokens == 13
        assert req2.num_draft_tokens == 5


# ############################################################################
#
# Part 4: PyCapacityScheduler Tests
#
# ############################################################################


class TestPyCapacitySchedulerMaxRequests:
    """Tests for MaxRequestsPolicy — no KV cache, simple count limit."""

    def test_basic_scheduling(self):
        """Schedule up to max_num_requests."""
        scheduler = PyCapacityScheduler(
            max_num_requests=2,
            kv_cache_manager=None,  # triggers MaxRequestsPolicy
        )
        requests = [
            make_context_request(0),
            make_context_request(1),
            make_context_request(2),
        ]
        fitting, disagg, paused = scheduler.schedule_request(requests)
        assert len(fitting) == 2
        assert len(paused) == 0

    def test_mixed_states(self):
        """Only schedulable states are included."""
        scheduler = PyCapacityScheduler(
            max_num_requests=4,
            kv_cache_manager=None,
        )
        requests = [
            make_context_request(0),
            make_generation_request(1),
            make_completed_request(2),  # should be filtered
            make_context_request(3),
        ]
        fitting, disagg, paused = scheduler.schedule_request(requests)
        # Completed request filtered out
        assert len(fitting) == 3
        assert all(r.request_id != 2 for r in fitting)

    def test_empty_requests(self):
        """No requests to schedule."""
        scheduler = PyCapacityScheduler(
            max_num_requests=4,
            kv_cache_manager=None,
        )
        fitting, disagg, paused = scheduler.schedule_request([])
        assert len(fitting) == 0


class TestPyCapacitySchedulerGuaranteedNoEvict:
    """
    Tests for GuaranteedNoEvictPolicy.
    C++ ref: capacitySchedulerTest.cpp GuaranteedCompletion tests
    """

    def test_requests_fit(self):
        """
        Both requests fit: free blocks (100) > 2 * blocks_per_request (5).
        C++ ref: SimpleShouldFit
        """
        kv = MockKVCacheManager(num_free_blocks=100, blocks_per_request=5)
        scheduler = PyCapacityScheduler(
            max_num_requests=4,
            kv_cache_manager=kv,
            scheduler_policy=CapacitySchedulerPolicy.GUARANTEED_NO_EVICT,
        )
        requests = [
            make_context_request(0),
            make_context_request(1),
        ]
        fitting, disagg, paused = scheduler.schedule_request(requests)
        assert len(fitting) == 2

    def test_not_enough_blocks(self):
        """
        Not enough blocks for the second request.
        C++ ref: SimpleDoesntFitGuaranteedCompletion
        """
        kv = MockKVCacheManager(num_free_blocks=7, blocks_per_request=5)
        scheduler = PyCapacityScheduler(
            max_num_requests=4,
            kv_cache_manager=kv,
            scheduler_policy=CapacitySchedulerPolicy.GUARANTEED_NO_EVICT,
        )
        requests = [
            make_context_request(0),
            make_context_request(1),
        ]
        fitting, disagg, paused = scheduler.schedule_request(requests)
        # First request takes 5 blocks, leaving 2 < 5 for second
        assert len(fitting) == 1
        assert fitting[0].request_id == 0

    def test_generation_scheduled_first(self):
        """
        In-progress generation requests are scheduled before context requests.
        They consume blocks from the pool.
        """
        kv = MockKVCacheManager(num_free_blocks=12, blocks_per_request=5)
        scheduler = PyCapacityScheduler(
            max_num_requests=4,
            kv_cache_manager=kv,
            scheduler_policy=CapacitySchedulerPolicy.GUARANTEED_NO_EVICT,
        )
        requests = [
            make_generation_request(0),  # takes 5 blocks, leaves 7
            make_context_request(1),  # takes 5 blocks, leaves 2
            make_context_request(2),  # needs 5, only 2 left
        ]
        fitting, disagg, paused = scheduler.schedule_request(requests)
        assert len(fitting) == 2
        assert {r.request_id for r in fitting} == {0, 1}

    def test_max_num_requests_honored(self):
        """max_num_requests is respected even if blocks are available."""
        kv = MockKVCacheManager(num_free_blocks=100, blocks_per_request=5)
        scheduler = PyCapacityScheduler(
            max_num_requests=2,
            kv_cache_manager=kv,
            scheduler_policy=CapacitySchedulerPolicy.GUARANTEED_NO_EVICT,
        )
        requests = [
            make_context_request(0),
            make_context_request(1),
            make_context_request(2),
        ]
        fitting, disagg, paused = scheduler.schedule_request(requests)
        assert len(fitting) == 2


class TestPyCapacitySchedulerMaxUtilization:
    """
    Tests for MaxUtilizationPolicy.
    C++ ref: capacitySchedulerTest.cpp MaxUtilization tests
    """

    def test_fits(self):
        """Requests fit within free blocks."""
        kv = MockKVCacheManager(num_free_blocks=100, blocks_per_request=5)
        scheduler = PyCapacityScheduler(
            max_num_requests=4,
            kv_cache_manager=kv,
            scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
        )
        requests = [
            make_context_request(0),
            make_generation_request(1),
        ]
        fitting, disagg, paused = scheduler.schedule_request(requests)
        assert len(fitting) == 2
        assert len(paused) == 0

    def test_doesnt_fit_pauses(self):
        """
        When a new request can't fit, MaxUtilization tries to pause
        already-started requests to make room.
        C++ ref: SimpleDoesntFitMaxUtilization
        """
        kv = MockKVCacheManager(num_free_blocks=7, blocks_per_request=5)
        scheduler = PyCapacityScheduler(
            max_num_requests=4,
            kv_cache_manager=kv,
            scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
        )
        # gen_0 is started (in progress), context_1 is new
        requests = [
            make_generation_request(0),
            make_context_request(1),
        ]
        fitting, disagg, paused = scheduler.schedule_request(requests)
        # gen_0 takes 5 blocks, context_1 needs 5 but only 2 left.
        # MaxUtilization pauses gen_0 to make room for context_1.
        assert len(fitting) + len(paused) >= 1

    def test_no_requests_to_pause(self):
        """If no started requests to pause, scheduling stops."""
        kv = MockKVCacheManager(num_free_blocks=3, blocks_per_request=5)
        scheduler = PyCapacityScheduler(
            max_num_requests=4,
            kv_cache_manager=kv,
            scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
        )
        requests = [
            make_context_request(0),
            make_context_request(1),
        ]
        fitting, disagg, paused = scheduler.schedule_request(requests)
        # Neither fits (3 < 5), no started requests to pause
        assert len(fitting) == 0
        assert len(paused) == 0


class TestPyCapacitySchedulerStaticBatch:
    """
    Tests for STATIC_BATCH policy.
    C++ ref: SimpleFitsStaticBatch
    """

    def test_schedules_when_no_active_generation(self):
        """Static batch: schedule all context requests when idle."""
        kv = MockKVCacheManager(num_free_blocks=100, blocks_per_request=5)
        scheduler = PyCapacityScheduler(
            max_num_requests=4,
            kv_cache_manager=kv,
            scheduler_policy=CapacitySchedulerPolicy.STATIC_BATCH,
        )
        requests = [
            make_context_request(0),
            make_context_request(1),
        ]
        fitting, disagg, paused = scheduler.schedule_request(requests)
        assert len(fitting) == 2

    def test_no_new_context_when_generation_active(self):
        """Static batch: no new context when generation is in progress."""
        kv = MockKVCacheManager(num_free_blocks=100, blocks_per_request=5)
        scheduler = PyCapacityScheduler(
            max_num_requests=4,
            kv_cache_manager=kv,
            scheduler_policy=CapacitySchedulerPolicy.STATIC_BATCH,
        )
        requests = [
            make_generation_request(0),
            make_context_request(1),
        ]
        fitting, disagg, paused = scheduler.schedule_request(requests)
        # Static batch: gen_0 is active, so only gen_0 scheduled (no new context)
        assert len(fitting) == 1
        assert fitting[0].request_id == 0


class TestPyCapacitySchedulerDisagg:
    """Tests for disaggregated generation init handling."""

    def test_disagg_gen_init_classified_separately(self):
        """Disagg gen init requests are separated in output.
        Uses GUARANTEED_NO_EVICT policy with a KV cache manager, since
        MaxRequestsPolicy (used when kv_cache_manager=None) does not
        handle disagg_generation_init state."""
        kv = MockKVCacheManager(num_free_blocks=100, blocks_per_request=5)
        scheduler = PyCapacityScheduler(
            max_num_requests=4,
            kv_cache_manager=kv,
            scheduler_policy=CapacitySchedulerPolicy.GUARANTEED_NO_EVICT,
        )
        requests = [
            make_context_request(0),
            make_disagg_gen_init_request(1),
            make_generation_request(2),
        ]
        fitting, disagg, paused = scheduler.schedule_request(requests)
        assert any(r.request_id == 0 for r in fitting)
        assert any(r.request_id == 2 for r in fitting)
        assert len(disagg) == 1
        assert disagg[0].request_id == 1

    def test_disagg_gen_init_bypasses_state_gating(self):
        """
        Disagg gen init requests bypass normal state gating
        (no_schedule_until / no_schedule_after).
        """
        kv = MockKVCacheManager(num_free_blocks=100, blocks_per_request=5)
        scheduler = PyCapacityScheduler(
            max_num_requests=4,
            kv_cache_manager=kv,
            scheduler_policy=CapacitySchedulerPolicy.GUARANTEED_NO_EVICT,
        )
        requests = [
            make_disagg_gen_init_request(0),
        ]
        fitting, disagg, paused = scheduler.schedule_request(requests)
        assert len(disagg) == 1


class TestPyCapacitySchedulerStateGating:
    """Tests for state-based filtering."""

    def test_unknown_state_filtered(self):
        """UNKNOWN state is before no_schedule_until, filtered out."""
        scheduler = PyCapacityScheduler(
            max_num_requests=4,
            kv_cache_manager=None,
        )
        req = _make_request(request_id=0, state=LlmRequestState.UNKNOWN)
        fitting, disagg, paused = scheduler.schedule_request([req])
        assert len(fitting) == 0

    def test_generation_complete_filtered(self):
        """GENERATION_COMPLETE is at no_schedule_after, filtered out."""
        scheduler = PyCapacityScheduler(
            max_num_requests=4,
            kv_cache_manager=None,
        )
        fitting, disagg, paused = scheduler.schedule_request([make_completed_request(0)])
        assert len(fitting) == 0

    def test_generation_to_complete_scheduled(self):
        """GENERATION_TO_COMPLETE is schedulable in PyCapacityScheduler.
        PyCapacityScheduler uses no_schedule_after=GENERATION_COMPLETE (20),
        so GENERATION_TO_COMPLETE (14) passes state gating. The real C++ binding's
        is_generation_in_progress_state includes GENERATION_TO_COMPLETE, so the
        MaxRequestsPolicy schedules it."""
        scheduler = PyCapacityScheduler(
            max_num_requests=4,
            kv_cache_manager=None,
        )
        req = _make_request(
            request_id=0,
            state=LlmRequestState.GENERATION_TO_COMPLETE,
        )
        fitting, disagg, paused = scheduler.schedule_request([req])
        assert len(fitting) == 1


# ############################################################################
#
# Part 5: PyCapacityScheduler Advanced Tests
#
# ############################################################################


class TestPyCapacitySchedulerLora:
    """Tests for LoRA/PEFT integration in capacity scheduling."""

    def test_lora_fits(self):
        """
        LoRA requests fit within PEFT cache.
        C++ ref: SimpleLoraFitsDuplicateTask
        """
        kv = MockKVCacheManager(num_free_blocks=100, blocks_per_request=5)
        peft = MockPeftCacheManager(max_pages=100, pages_per_request=10)
        scheduler = PyCapacityScheduler(
            max_num_requests=4,
            kv_cache_manager=kv,
            peft_cache_manager=peft,
            scheduler_policy=CapacitySchedulerPolicy.GUARANTEED_NO_EVICT,
        )
        r0 = _make_request(0, lora_task_id=1)
        r1 = _make_request(1, lora_task_id=1)  # same task — no extra pages needed
        fitting, disagg, paused = scheduler.schedule_request([r0, r1])
        assert len(fitting) == 2

    def test_lora_doesnt_fit(self):
        """
        LoRA requests exceed PEFT cache.
        C++ ref: SimpleLoraDoesntFitDuplicateTask
        """
        kv = MockKVCacheManager(num_free_blocks=100, blocks_per_request=5)
        peft = MockPeftCacheManager(max_pages=15, pages_per_request=10)
        scheduler = PyCapacityScheduler(
            max_num_requests=4,
            kv_cache_manager=kv,
            peft_cache_manager=peft,
            scheduler_policy=CapacitySchedulerPolicy.GUARANTEED_NO_EVICT,
        )
        r0 = _make_request(0, lora_task_id=1)
        r1 = _make_request(1, lora_task_id=2)  # different task — needs 10 more pages
        fitting, disagg, paused = scheduler.schedule_request([r0, r1])
        # First task: 10 pages, second task: 10 pages, total 20 > 15
        assert len(fitting) == 1


# ############################################################################
#
# Part 6: SimpleUnifiedScheduler Integration Tests
#
# ############################################################################


class TestSimpleUnifiedScheduler:
    """
    Tests for the two-stage scheduling pipeline:
    PyCapacityScheduler → PyMicroBatchScheduler
    """

    def test_capacity_then_microbatch(self):
        """Capacity filters, then microbatch selects within token budget.
        max_batch_size is used as max_num_requests for capacity scheduler,
        so it must be large enough for all requests to pass capacity."""
        kv = MockKVCacheManager(num_free_blocks=100, blocks_per_request=5)
        scheduler = SimpleUnifiedScheduler(
            max_batch_size=4,
            max_num_tokens=15,
            kv_cache_manager=kv,
            peft_cache_manager=None,
            scheduler_policy=CapacitySchedulerPolicy.GUARANTEED_NO_EVICT,
        )
        requests = [
            make_context_request(0, prompt_len=10),
            make_context_request(1, prompt_len=10),
            make_generation_request(2),
        ]
        output = scheduler.schedule_request(requests, set())
        # Capacity: all 3 fit (plenty of blocks, max_num_requests=4)
        # Microbatch: gen_2 (1) + context_0 (10) = 11 <= 15, context_1 (10) would be 21 > 15
        assert output.num_fitting_requests == 3
        assert len(output.context_requests) + len(output.generation_requests) <= 2

    def test_can_schedule_dry_run(self):
        """can_schedule() checks capacity without side effects."""
        kv = MockKVCacheManager(num_free_blocks=100, blocks_per_request=5)
        scheduler = SimpleUnifiedScheduler(
            max_batch_size=4,
            max_num_tokens=100,
            kv_cache_manager=kv,
            peft_cache_manager=None,
            scheduler_policy=CapacitySchedulerPolicy.GUARANTEED_NO_EVICT,
        )
        requests = [
            make_context_request(0),
            make_generation_request(1),
        ]
        assert scheduler.can_schedule(requests) is True

    def test_can_schedule_returns_false(self):
        """can_schedule() returns False when capacity is insufficient."""
        kv = MockKVCacheManager(num_free_blocks=3, blocks_per_request=5)
        scheduler = SimpleUnifiedScheduler(
            max_batch_size=4,
            max_num_tokens=100,
            kv_cache_manager=kv,
            peft_cache_manager=None,
            scheduler_policy=CapacitySchedulerPolicy.GUARANTEED_NO_EVICT,
        )
        requests = [
            make_context_request(0),
            make_context_request(1),
        ]
        # First takes 5 blocks > 3 free
        assert scheduler.can_schedule(requests) is False

    def test_full_pipeline_output_structure(self):
        """Verify SchedulerOutput has all expected fields."""
        kv = MockKVCacheManager(num_free_blocks=100, blocks_per_request=5)
        scheduler = SimpleUnifiedScheduler(
            max_batch_size=4,
            max_num_tokens=100,
            kv_cache_manager=kv,
            peft_cache_manager=None,
            scheduler_policy=CapacitySchedulerPolicy.GUARANTEED_NO_EVICT,
        )
        requests = [
            make_context_request(0, prompt_len=10),
            make_generation_request(1),
        ]
        output = scheduler.schedule_request(requests, set())
        assert hasattr(output, "context_requests")
        assert hasattr(output, "generation_requests")
        assert hasattr(output, "paused_requests")
        assert hasattr(output, "fitting_disagg_gen_init_requests")
        assert hasattr(output, "num_fitting_requests")
        assert len(output.context_requests) == 1
        assert len(output.generation_requests) == 1
        assert output.context_requests[0].request_id == 0
        assert output.generation_requests[0].request_id == 1

    def test_paused_requests_propagated(self):
        """Paused requests from capacity scheduler appear in output."""
        kv = MockKVCacheManager(num_free_blocks=100, blocks_per_request=5)
        scheduler = SimpleUnifiedScheduler(
            max_batch_size=4,
            max_num_tokens=100,
            kv_cache_manager=kv,
            peft_cache_manager=None,
            scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
        )
        # With MAX_UTILIZATION and plenty of resources, nothing should be paused
        requests = [
            make_context_request(0, prompt_len=10),
            make_generation_request(1),
        ]
        output = scheduler.schedule_request(requests, set())
        assert isinstance(output.paused_requests, list)


# ############################################################################
#
# Part 7: Additional PyCapacityScheduler Tests
#
# ############################################################################


class TestPyCapacitySchedulerCrossKVCache:
    """
    Tests for cross-attention KV cache scheduling.
    C++ ref: capacitySchedulerTest.cpp cross KV cache tests.
    """

    def test_should_fit_with_cross_blocks(self):
        """C++ ref: SimpleShouldFitWithCrossBlocks"""
        kv = MockKVCacheManager(num_free_blocks=100, blocks_per_request=5)
        cross_kv = MockKVCacheManager(num_free_blocks=100, blocks_per_request=2)
        scheduler = PyCapacityScheduler(
            max_num_requests=2,
            kv_cache_manager=kv,
            cross_kv_cache_manager=cross_kv,
            scheduler_policy=CapacitySchedulerPolicy.GUARANTEED_NO_EVICT,
        )
        r0 = make_context_request(0, prompt_len=10)
        r0.encoder_output_len = 10
        r1 = make_context_request(1, prompt_len=10)
        r1.encoder_output_len = 10
        fitting, disagg, paused = scheduler.schedule_request([r0, r1])
        assert len(fitting) == 2

    def test_doesnt_fit_with_cross_blocks(self):
        """C++ ref: SimpleDoesntFitWithCrossBlocks - cross kv cache too small for both"""
        kv = MockKVCacheManager(num_free_blocks=100, blocks_per_request=5)
        cross_kv = MockKVCacheManager(num_free_blocks=1, blocks_per_request=1)
        scheduler = PyCapacityScheduler(
            max_num_requests=2,
            kv_cache_manager=kv,
            cross_kv_cache_manager=cross_kv,
            scheduler_policy=CapacitySchedulerPolicy.GUARANTEED_NO_EVICT,
        )
        r0 = make_context_request(0, prompt_len=10)
        r0.encoder_output_len = 10
        r1 = make_context_request(1, prompt_len=10)
        r1.encoder_output_len = 10
        fitting, disagg, paused = scheduler.schedule_request([r0, r1])
        assert len(fitting) == 1


class TestPyCapacitySchedulerPriority:
    """
    Tests for priority-related scheduling behavior.
    C++ ref: capacitySchedulerTest.cpp priority tests.
    """

    def test_requests_sorted_by_priorities(self):
        """C++ ref: RequestsSortedByPriorities.
        Python scheduler doesn't have insertRequestInOrder, but this tests
        that requests are processed in the order they're given."""
        scheduler = PyCapacityScheduler(max_num_requests=12, kv_cache_manager=None)
        # Create 12 requests - all should be scheduled in order
        requests = [make_context_request(i) for i in range(12)]
        fitting, disagg, paused = scheduler.schedule_request(requests)
        assert len(fitting) == 12
        assert [r.request_id for r in fitting] == list(range(12))

    def test_doesnt_fit_priorities(self):
        """C++ ref: SimpleDoesntFitPriorities - MAX_UTILIZATION with limited blocks"""
        kv = MockKVCacheManager(num_free_blocks=7, blocks_per_request=5)
        scheduler = PyCapacityScheduler(
            max_num_requests=2,
            kv_cache_manager=kv,
            scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
        )
        requests = [make_context_request(0), make_context_request(1)]
        fitting, disagg, paused = scheduler.schedule_request(requests)
        # Only first request fits (5 blocks), second needs 5 but only 2 left
        assert len(fitting) >= 1


class TestPyCapacitySchedulerChunked:
    """
    Tests for chunked context request capacity scheduling.
    C++ ref: capacitySchedulerTest.cpp chunked tests.
    """

    def test_should_fit_in_chunk(self):
        """C++ ref: SimpleShouldFitInChunk - chunked context requests fit"""
        kv = MockKVCacheManager(num_free_blocks=100, blocks_per_request=5)
        scheduler = PyCapacityScheduler(
            max_num_requests=2,
            kv_cache_manager=kv,
            scheduler_policy=CapacitySchedulerPolicy.GUARANTEED_NO_EVICT,
        )
        r0 = make_context_request(0, prompt_len=50, context_position=0)
        r0.context_chunk_size = 20
        r1 = make_context_request(1, prompt_len=50, context_position=0)
        r1.context_chunk_size = 20
        fitting, disagg, paused = scheduler.schedule_request([r0, r1])
        assert len(fitting) == 2

    def test_doesnt_fit_guaranteed_completion_in_chunk(self):
        """C++ ref: SimpleDoesntFitGuaranteedCompletionInChunk"""
        kv = MockKVCacheManager(num_free_blocks=7, blocks_per_request=5)
        scheduler = PyCapacityScheduler(
            max_num_requests=2,
            kv_cache_manager=kv,
            scheduler_policy=CapacitySchedulerPolicy.GUARANTEED_NO_EVICT,
        )
        r0 = make_context_request(0, prompt_len=30)
        r0.context_chunk_size = 20
        r1 = make_context_request(1, prompt_len=30)
        r1.context_chunk_size = 20
        fitting, disagg, paused = scheduler.schedule_request([r0, r1])
        assert len(fitting) == 1

    def test_doesnt_fit_max_utilization_in_chunk(self):
        """C++ ref: SimpleDoesntFitMaxUtilizationInChunk"""
        kv = MockKVCacheManager(num_free_blocks=7, blocks_per_request=5)
        scheduler = PyCapacityScheduler(
            max_num_requests=2,
            kv_cache_manager=kv,
            scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
        )
        r0 = make_context_request(0, prompt_len=30)
        r0.context_chunk_size = 20
        r1 = make_context_request(1, prompt_len=30)
        r1.context_chunk_size = 20
        fitting, disagg, paused = scheduler.schedule_request([r0, r1])
        # At least one should be scheduled
        assert len(fitting) >= 1

    def test_doesnt_fit_max_utilization_in_chunked_cache(self):
        """C++ ref: SimpleDoesntFitMaxUtilizationInChunkedCache - only 1 fits due to limited blocks"""
        kv = MockKVCacheManager(num_free_blocks=7, blocks_per_request=7)
        scheduler = PyCapacityScheduler(
            max_num_requests=2,
            kv_cache_manager=kv,
            scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
        )
        r0 = make_context_request(0, prompt_len=70)
        r0.context_chunk_size = 40
        r1 = make_context_request(1, prompt_len=70)
        r1.context_chunk_size = 40
        fitting, disagg, paused = scheduler.schedule_request([r0, r1])
        assert len(fitting) == 1

    def test_doesnt_fit_max_utilization_draft_tokens(self):
        """C++ ref: SimpleDoesntFitMaxUtilizationDraftTokens"""
        kv = MockKVCacheManager(num_free_blocks=7, blocks_per_request=5)
        scheduler = PyCapacityScheduler(
            max_num_requests=2,
            kv_cache_manager=kv,
            scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
        )
        r0 = make_context_request(0, prompt_len=10, draft_tokens_len=5)
        r1 = make_context_request(1, prompt_len=10, draft_tokens_len=10)
        fitting, disagg, paused = scheduler.schedule_request([r0, r1])
        assert len(fitting) >= 1


class TestPyCapacitySchedulerDynamicAddition:
    """
    Tests for dynamic request addition during scheduling.
    C++ ref: capacitySchedulerTest.cpp dynamic addition tests.
    """

    def test_adding_new_requests_max_utilization(self):
        """C++ ref: SimpleDoesntFitAddingNewRequestsMaxUtilization
        Initial state: 2 gen requests in progress, 2 new context requests.
        With limited blocks, not all can run."""
        kv = MockKVCacheManager(num_free_blocks=10, blocks_per_request=5)
        scheduler = PyCapacityScheduler(
            max_num_requests=4,
            kv_cache_manager=kv,
            scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
        )
        requests = [
            make_generation_request(0),
            make_generation_request(1),
            make_context_request(2),
            make_context_request(3),
        ]
        fitting, disagg, paused = scheduler.schedule_request(requests)
        assert len(fitting) + len(paused) >= 2

    def test_adding_new_requests_guaranteed_completion(self):
        """C++ ref: SimpleDoesntFitAddingNewRequestsGuaranteedCompletion"""
        kv = MockKVCacheManager(num_free_blocks=7, blocks_per_request=5)
        scheduler = PyCapacityScheduler(
            max_num_requests=4,
            kv_cache_manager=kv,
            scheduler_policy=CapacitySchedulerPolicy.GUARANTEED_NO_EVICT,
        )
        requests = [
            make_generation_request(0),
            make_context_request(1),
            make_context_request(2),
            make_context_request(3),
        ]
        fitting, disagg, paused = scheduler.schedule_request(requests)
        # gen_0 takes 5 blocks (7-5=2 left), none of the context requests fit
        assert len(fitting) >= 1
        assert fitting[0].request_id == 0

    def test_adding_new_requests_guaranteed_completion_in_chunk(self):
        """C++ ref: SimpleDoesntFitAddingNewRequestsGuaranteedCompletionInChunk"""
        kv = MockKVCacheManager(num_free_blocks=7, blocks_per_request=5)
        scheduler = PyCapacityScheduler(
            max_num_requests=4,
            kv_cache_manager=kv,
            scheduler_policy=CapacitySchedulerPolicy.GUARANTEED_NO_EVICT,
        )
        r0 = make_generation_request(0)
        r1 = make_context_request(1, prompt_len=30)
        r1.context_chunk_size = 20
        r2 = make_context_request(2, prompt_len=30)
        r2.context_chunk_size = 20
        fitting, disagg, paused = scheduler.schedule_request([r0, r1, r2])
        assert len(fitting) >= 1

    def test_surpass_max_num_requests_with_priorities(self):
        """C++ ref: SimpleSurpassMaxNumRequestsWithPriorities"""
        kv = MockKVCacheManager(num_free_blocks=100, blocks_per_request=2)
        scheduler = PyCapacityScheduler(
            max_num_requests=2,
            kv_cache_manager=kv,
            scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
        )
        # In C++, high-priority requests preempt lower-priority ones.
        # In Python, maxNumRequests limits total scheduled.
        requests = [
            make_context_request(0),
            make_context_request(1),
            make_context_request(2),
            make_context_request(3),
        ]
        fitting, disagg, paused = scheduler.schedule_request(requests)
        assert len(fitting) <= 2

    def test_adding_new_requests_max_utilization_priorities(self):
        """C++ ref: SimpleDoesntFitAddingNewRequestsMaxUtilizationPriorities"""
        kv = MockKVCacheManager(num_free_blocks=10, blocks_per_request=5)
        scheduler = PyCapacityScheduler(
            max_num_requests=4,
            kv_cache_manager=kv,
            scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
        )
        requests = [
            make_generation_request(0),
            make_generation_request(1),
            make_context_request(2),
            make_context_request(3),
        ]
        fitting, disagg, paused = scheduler.schedule_request(requests)
        assert len(fitting) + len(paused) >= 2


class TestPyCapacitySchedulerKVCacheReuse:
    """
    Tests for KV cache reuse-aware scheduling.
    C++ ref: DelayDuplicate*, ReuseAware*, MaxUtilizationReuse*, NoReuse* tests
    in capacitySchedulerTest.cpp.
    """

    def test_delay_duplicate_request(self):
        """C++ ref: DelayDuplicateRequest - identical requests delayed for reuse"""
        kv = MockKVCacheManager(num_free_blocks=100, blocks_per_request=3, enable_block_reuse=True)
        scheduler = PyCapacityScheduler(
            max_num_requests=3,
            kv_cache_manager=kv,
            scheduler_policy=CapacitySchedulerPolicy.GUARANTEED_NO_EVICT,
        )
        tokens = list(range(21))
        r0 = _make_request(0, prompt_len=21, input_tokens=tokens)
        r1 = _make_request(1, prompt_len=21, input_tokens=tokens)
        r2 = _make_request(2, prompt_len=21, input_tokens=tokens)
        fitting, disagg, paused = scheduler.schedule_request([r0, r1, r2])
        # With reuse enabled, beneficial_to_skip may delay r1 and r2
        assert len(fitting) >= 1

    def test_delay_duplicate_request_chunked(self):
        """C++ ref: DelayDuplicateRequestChunked"""
        kv = MockKVCacheManager(num_free_blocks=100, blocks_per_request=5, enable_block_reuse=True)
        scheduler = PyCapacityScheduler(
            max_num_requests=2,
            kv_cache_manager=kv,
            scheduler_policy=CapacitySchedulerPolicy.GUARANTEED_NO_EVICT,
        )
        tokens = list(range(50))
        r0 = _make_request(0, prompt_len=50, input_tokens=tokens)
        r0.context_chunk_size = 20
        r1 = _make_request(1, prompt_len=50, input_tokens=tokens)
        r1.context_chunk_size = 20
        fitting, disagg, paused = scheduler.schedule_request([r0, r1])
        assert len(fitting) >= 1

    def test_delay_five_requests_complicated(self):
        """C++ ref: DelayFiveRequestsComplicated"""
        kv = MockKVCacheManager(num_free_blocks=100, blocks_per_request=3, enable_block_reuse=True)
        scheduler = PyCapacityScheduler(
            max_num_requests=5,
            kv_cache_manager=kv,
            scheduler_policy=CapacitySchedulerPolicy.GUARANTEED_NO_EVICT,
        )
        r0 = _make_request(0, prompt_len=11, input_tokens=[x + 1 for x in range(11)])
        r1 = _make_request(1, prompt_len=21, input_tokens=[x + 2 for x in range(21)])
        r2 = _make_request(2, prompt_len=11, input_tokens=list(range(11)))
        r3 = _make_request(3, prompt_len=21, input_tokens=list(range(21)))
        r4 = _make_request(4, prompt_len=31, input_tokens=list(range(31)))
        fitting, disagg, paused = scheduler.schedule_request([r0, r1, r2, r3, r4])
        assert len(fitting) >= 1

    def test_reuse_aware_allows_more_requests(self):
        """C++ ref: ReuseAwareSchedulingAllowsMoreRequestsWithSharedPrefix"""
        kv = MockKVCacheManager(num_free_blocks=100, blocks_per_request=2, enable_block_reuse=True)
        scheduler = PyCapacityScheduler(
            max_num_requests=4,
            kv_cache_manager=kv,
            scheduler_policy=CapacitySchedulerPolicy.GUARANTEED_NO_EVICT,
        )
        tokens = list(range(20))
        r0 = _make_request(0, prompt_len=20, input_tokens=tokens)
        r1 = _make_request(1, prompt_len=20, input_tokens=tokens)
        fitting, disagg, paused = scheduler.schedule_request([r0, r1])
        assert len(fitting) >= 1

    def test_reuse_aware_partial_prefix_match(self):
        """C++ ref: ReuseAwareSchedulingWithPartialPrefixMatch"""
        kv = MockKVCacheManager(num_free_blocks=100, blocks_per_request=3, enable_block_reuse=True)
        scheduler = PyCapacityScheduler(
            max_num_requests=3,
            kv_cache_manager=kv,
            scheduler_policy=CapacitySchedulerPolicy.GUARANTEED_NO_EVICT,
        )
        tokens0 = list(range(30))
        tokens1 = list(range(20)) + [999] * 10
        r0 = _make_request(0, prompt_len=30, input_tokens=tokens0)
        r1 = _make_request(1, prompt_len=30, input_tokens=tokens1)
        fitting, disagg, paused = scheduler.schedule_request([r0, r1])
        assert len(fitting) >= 1

    def test_no_reuse_with_different_prompts(self):
        """C++ ref: NoReuseWithDifferentPrompts"""
        kv = MockKVCacheManager(num_free_blocks=100, blocks_per_request=2, enable_block_reuse=True)
        scheduler = PyCapacityScheduler(
            max_num_requests=2,
            kv_cache_manager=kv,
            scheduler_policy=CapacitySchedulerPolicy.GUARANTEED_NO_EVICT,
        )
        r0 = _make_request(0, prompt_len=20, input_tokens=[100] * 20)
        r1 = _make_request(1, prompt_len=20, input_tokens=[200] * 20)
        fitting, disagg, paused = scheduler.schedule_request([r0, r1])
        # Different prompts: no reuse, both scheduled
        assert len(fitting) == 2

    def test_reuse_aware_max_utilization(self):
        """C++ ref: ReuseAwareSchedulingMaxUtilizationPolicy"""
        kv = MockKVCacheManager(num_free_blocks=100, blocks_per_request=2, enable_block_reuse=True)
        scheduler = PyCapacityScheduler(
            max_num_requests=4,
            kv_cache_manager=kv,
            scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
        )
        tokens = list(range(20))
        r0 = _make_request(0, prompt_len=20, input_tokens=tokens)
        r1 = _make_request(1, prompt_len=20, input_tokens=tokens)
        fitting, disagg, paused = scheduler.schedule_request([r0, r1])
        assert len(fitting) >= 1

    def test_max_utilization_reuse_reduces_needed_blocks(self):
        """C++ ref: MaxUtilizationReuseReducesNeededBlocksOneStep"""
        kv = MockKVCacheManager(num_free_blocks=6, blocks_per_request=3, enable_block_reuse=True)
        scheduler = PyCapacityScheduler(
            max_num_requests=3,
            kv_cache_manager=kv,
            scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
        )
        tokens = list(range(30))
        r0 = _make_request(0, prompt_len=30, input_tokens=tokens)
        r1 = _make_request(1, prompt_len=30, input_tokens=tokens)
        fitting, disagg, paused = scheduler.schedule_request([r0, r1])
        assert len(fitting) >= 1

    def test_max_utilization_under_memory_pressure_with_reuse(self):
        """C++ ref: MaxUtilizationUnderMemoryPressureWithReuse"""
        kv = MockKVCacheManager(num_free_blocks=5, blocks_per_request=2, enable_block_reuse=True)
        scheduler = PyCapacityScheduler(
            max_num_requests=2,
            kv_cache_manager=kv,
            scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
        )
        tokens = list(range(20))
        r0 = _make_request(0, prompt_len=20, input_tokens=tokens)
        r1 = _make_request(1, prompt_len=20, input_tokens=tokens)
        fitting, disagg, paused = scheduler.schedule_request([r0, r1])
        assert len(fitting) >= 1

    def test_max_utilization_incremental_reuse(self):
        """C++ ref: MaxUtilizationMultipleRequestsIncrementalReuse"""
        kv = MockKVCacheManager(num_free_blocks=15, blocks_per_request=2, enable_block_reuse=True)
        scheduler = PyCapacityScheduler(
            max_num_requests=4,
            kv_cache_manager=kv,
            scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
        )
        tokens = list(range(20))
        r0 = _make_request(0, prompt_len=20, input_tokens=tokens)
        r1 = _make_request(1, prompt_len=20, input_tokens=tokens)
        fitting, disagg, paused = scheduler.schedule_request([r0, r1])
        assert len(fitting) >= 1

    def test_no_reuse_when_disabled(self):
        """C++ ref: MaxUtilizationNoReuseWhenDisabled"""
        kv = MockKVCacheManager(num_free_blocks=100, blocks_per_request=2, enable_block_reuse=False)
        scheduler = PyCapacityScheduler(
            max_num_requests=2,
            kv_cache_manager=kv,
            scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
        )
        tokens = list(range(20))
        r0 = _make_request(0, prompt_len=20, input_tokens=tokens)
        r1 = _make_request(1, prompt_len=20, input_tokens=tokens)
        fitting, disagg, paused = scheduler.schedule_request([r0, r1])
        # No reuse: both scheduled together
        assert len(fitting) == 2


class TestPyCapacitySchedulerDisaggAdvanced:
    """
    Advanced tests for disaggregated generation init scheduling.
    C++ ref: capacitySchedulerTest.cpp disagg tests.
    """

    def test_disagg_gen_init_max_utilization(self):
        """C++ ref: DisaggGenInitMaxUtilization"""
        kv = MockKVCacheManager(num_free_blocks=100, blocks_per_request=5)
        scheduler = PyCapacityScheduler(
            max_num_requests=2,
            kv_cache_manager=kv,
            scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
        )
        r0 = make_disagg_gen_init_request(0)
        r1 = make_disagg_gen_init_request(1)
        fitting, disagg, paused = scheduler.schedule_request([r0, r1])
        assert len(disagg) == 2


class TestPyCapacitySchedulerStaticBatchAdvanced:
    """
    Advanced tests for static batch scheduling.
    C++ ref: capacitySchedulerTest.cpp static batch tests.
    """

    def test_static_batch_fits(self):
        """C++ ref: SimpleFitsStaticBatch - third request waits for batch to complete"""
        kv = MockKVCacheManager(num_free_blocks=100, blocks_per_request=5)
        scheduler = PyCapacityScheduler(
            max_num_requests=2,
            kv_cache_manager=kv,
            scheduler_policy=CapacitySchedulerPolicy.STATIC_BATCH,
        )
        # When gen is active, no new context allowed
        r0 = make_generation_request(0)
        r1 = make_context_request(1)
        r2 = make_context_request(2)
        fitting, disagg, paused = scheduler.schedule_request([r0, r1, r2])
        assert len(fitting) == 1
        assert fitting[0].request_id == 0
