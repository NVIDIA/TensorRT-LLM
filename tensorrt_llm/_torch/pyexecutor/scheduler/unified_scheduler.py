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

import dataclasses
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Set

from strenum import StrEnum

from tensorrt_llm.llmapi.llm_args import CapacitySchedulerPolicy
from tensorrt_llm.logger import logger

from ..llm_request import LlmRequest, LlmRequestState
from .scheduler import (
    RequestList,
    RequestScheduler,
    SchedulerOutput,
    ScheduleStepConfig,
    compute_fcfs_context_chunk_size,
    sort_requests_by_lora,
)


class ChunkingPolicy(Enum):
    EQUAL_PROGRESS = 1
    FIRST_COME_FIRST_SERVED = 2


@dataclasses.dataclass
class ContextChunkingConfig:
    chunking_policy: ChunkingPolicy
    chunk_unit_size: int


class TokenBudgetTracker:
    """Fused capacity + microbatch token budget tracker.

    Integrates token-budget checks, request type classification, context
    chunking, and sorting into the capacity policy loop — replacing the
    old two-pass (capacity → microbatch) pipeline with a single pass.

    Provides type-specialized methods for hot paths:
    - try_add_generation(): fast path for generation requests (no type checks)
    - try_add_context(): fast path for context requests (no beam/gen checks)
    - try_add(): generic path for mixed/unknown request types
    """

    __slots__ = [
        "max_batch_size",
        "max_num_tokens",
        "max_context_length",
        "_has_token_limit",
        "ctx_chunk_config",
        "_inflight_ids",
        "_batch_num_tokens",
        "_scheduled_req_size",
        "_scheduled_beam_width",
        "_context_requests",
        "_generation_requests",
        "_contexts_to_be_chunked",
        "_num_chunked_tokens",
        "_all_context_requests_fit",
        "_num_fitting",
        "batch_full",
    ]

    # Cache state values once at class level to avoid repeated .value access
    _CONTEXT_INIT_VALUE = LlmRequestState.CONTEXT_INIT.value
    _ENCODER_INIT_VALUE = LlmRequestState.ENCODER_INIT.value
    _GEN_IN_PROGRESS_VALUE = LlmRequestState.GENERATION_IN_PROGRESS.value
    _NO_SCHEDULE_UNTIL_VALUE = LlmRequestState.CONTEXT_INIT.value
    _NO_SCHEDULE_AFTER_VALUE = LlmRequestState.GENERATION_TO_COMPLETE.value

    def __init__(
        self,
        max_batch_size: int,
        max_num_tokens: Optional[int],
        ctx_chunk_config: Optional[ContextChunkingConfig],
        inflight_request_ids: object = None,
    ):
        self.max_batch_size = max_batch_size
        self.max_num_tokens = max_num_tokens
        self.max_context_length = max_num_tokens
        self._has_token_limit = max_num_tokens is not None
        self.ctx_chunk_config = ctx_chunk_config
        # Accepts set[int] or C++ ReqIdsSet — only `in` operator is used.
        self._inflight_ids = inflight_request_ids
        self._batch_num_tokens = 0
        self._scheduled_req_size = 0
        self._scheduled_beam_width = 0
        self._context_requests: RequestList = []
        self._generation_requests: RequestList = []
        self._contexts_to_be_chunked: RequestList = []
        self._num_chunked_tokens = 0
        self._all_context_requests_fit = True
        self._num_fitting = 0
        self.batch_full = False

    def try_add(self, req: LlmRequest) -> bool:
        """Try to add a request to the batch. Returns False if token budget exceeded."""
        # Skip if in flight
        if self._inflight_ids is not None and req.request_id in self._inflight_ids:
            return True  # don't reject, just skip token accounting

        # Disagg gen init requests bypass token accounting — they are
        # classified separately by the capacity policy.
        if req.is_disagg_generation_init_state:
            return True

        req_state_value = req.state_value

        # Skip if not in schedulable state range
        if not (
            req_state_value >= self._NO_SCHEDULE_UNTIL_VALUE
            and req_state_value < self._NO_SCHEDULE_AFTER_VALUE
        ):
            return True  # don't reject from capacity, just skip

        req_num_tokens = 0

        # --- Encoder ---
        if req_state_value == self._ENCODER_INIT_VALUE:
            req_num_tokens = req.encoder_output_len
            if self.max_context_length is not None:
                assert req_num_tokens <= self.max_context_length, (
                    f"The number of encoder tokens ({req_num_tokens}) exceeds "
                    f"the limit value ({self.max_context_length})"
                )
            if self._has_token_limit and (
                self._batch_num_tokens + req_num_tokens > self.max_num_tokens
            ):
                return False
            self._context_requests.append(req)
            self._batch_num_tokens += req_num_tokens

        # --- Context ---
        elif req_state_value == self._CONTEXT_INIT_VALUE:
            if not self.ctx_chunk_config:
                base_tokens = req.get_num_tokens(0)
                draft_tokens = req.num_draft_tokens if req.has_draft_tokens else 0
                req_num_tokens = base_tokens + draft_tokens
                if self.max_context_length is not None:
                    assert req_num_tokens <= self.max_context_length, (
                        f"Context tokens ({req_num_tokens}) exceeds "
                        f"limit ({self.max_context_length})"
                    )
                if self._has_token_limit and (
                    self._batch_num_tokens + req_num_tokens > self.max_num_tokens
                ):
                    return False
                self._context_requests.append(req)
                self._batch_num_tokens += req_num_tokens
            else:
                # Chunking: tentative schedule (finalized later)
                req.context_chunk_size = req.context_remaining_length
                draft_tokens = (
                    req.num_draft_tokens
                    if (req.is_last_context_chunk and req.has_draft_tokens)
                    else 0
                )
                req_num_tokens = req.context_chunk_size + draft_tokens
                if self.max_context_length is not None:
                    if self.max_context_length < req_num_tokens:
                        req_num_tokens = self.max_context_length
                        self._all_context_requests_fit = False
                self._contexts_to_be_chunked.append(req)
                self._num_chunked_tokens += req_num_tokens

        # --- Generation ---
        else:
            beam_width = req.get_beam_width_by_iter(for_next_iteration=False)
            req_num_tokens = beam_width + req.num_draft_tokens
            if self._has_token_limit and (
                self._batch_num_tokens + req_num_tokens > self.max_num_tokens
            ):
                return False
            # Beam width consistency
            if self._scheduled_beam_width == 0:
                self._scheduled_beam_width = beam_width
            elif self._scheduled_beam_width != beam_width:
                return True  # skip this request, don't reject from capacity
            self._generation_requests.append(req)
            self._batch_num_tokens += req_num_tokens

        self._scheduled_req_size += 1
        self._num_fitting += 1
        if self._scheduled_req_size >= self.max_batch_size:
            self.batch_full = True
        return True

    def try_add_generation(self, req: LlmRequest) -> int:
        """Fast path for generation-in-progress requests.

        Returns:
            1: accepted, continue scheduling
            0: rejected (token budget exceeded)
           -1: accepted but batch is full (stop scheduling)
        """
        if self._inflight_ids is not None and req.request_id in self._inflight_ids:
            return 1  # skip token accounting for inflight requests
        beam_width = req.get_beam_width_by_iter(for_next_iteration=False)
        req_num_tokens = beam_width + req.num_draft_tokens
        if self._has_token_limit and (
            self._batch_num_tokens + req_num_tokens > self.max_num_tokens
        ):
            return 0
        # Beam width consistency
        if self._scheduled_beam_width == 0:
            self._scheduled_beam_width = beam_width
        elif self._scheduled_beam_width != beam_width:
            return 1  # skip, don't reject from capacity
        self._generation_requests.append(req)
        self._batch_num_tokens += req_num_tokens
        self._scheduled_req_size += 1
        self._num_fitting += 1
        if self._scheduled_req_size >= self.max_batch_size:
            self.batch_full = True
            return -1
        return 1

    def try_add_context(self, req: LlmRequest) -> bool:
        """Fast path for context-init requests.

        Skips state-range, encoder, generation, and beam checks.
        Called from the second pass of capacity policies where requests
        are known to be context_init or disagg_generation_init.
        """
        if self._inflight_ids is not None and req.request_id in self._inflight_ids:
            return True  # skip token accounting for inflight requests
        if req.is_disagg_generation_init_state:
            return True  # classified separately by the capacity policy

        if not self.ctx_chunk_config:
            base_tokens = req.get_num_tokens(0)
            draft_tokens = req.num_draft_tokens if req.has_draft_tokens else 0
            req_num_tokens = base_tokens + draft_tokens
            if self.max_context_length is not None:
                assert req_num_tokens <= self.max_context_length, (
                    f"Context tokens ({req_num_tokens}) exceeds limit ({self.max_context_length})"
                )
            if self._has_token_limit and (
                self._batch_num_tokens + req_num_tokens > self.max_num_tokens
            ):
                return False
            self._context_requests.append(req)
            self._batch_num_tokens += req_num_tokens
        else:
            # Chunking: tentative schedule (finalized later)
            req.context_chunk_size = req.context_remaining_length
            draft_tokens = (
                req.num_draft_tokens if (req.is_last_context_chunk and req.has_draft_tokens) else 0
            )
            req_num_tokens = req.context_chunk_size + draft_tokens
            if self.max_context_length is not None:
                if self.max_context_length < req_num_tokens:
                    req_num_tokens = self.max_context_length
                    self._all_context_requests_fit = False
            self._contexts_to_be_chunked.append(req)
            self._num_chunked_tokens += req_num_tokens

        self._scheduled_req_size += 1
        self._num_fitting += 1
        if self._scheduled_req_size >= self.max_batch_size:
            self.batch_full = True
        return True

    def finalize(self) -> tuple[RequestList, RequestList, int]:
        """Apply chunking and sorting. Returns (context, generation, num_fitting).

        Note: num_fitting reflects requests admitted by try_add/try_add_*
        before chunking. Requests with context_chunk_size == 0 are dropped
        from context_requests below but num_fitting is not decremented.
        See class docstring item 3 for the full semantics.
        """
        # Verify chunking fits
        if self._has_token_limit and self._num_chunked_tokens > (
            self.max_num_tokens - self._batch_num_tokens
        ):
            self._all_context_requests_fit = False

        # Apply chunking strategy
        if not self._all_context_requests_fit and self._contexts_to_be_chunked:
            remaining_capacity = (
                (self.max_num_tokens - self._batch_num_tokens) if self._has_token_limit else None
            )
            self._set_ctx_requests_chunk_size(self._contexts_to_be_chunked, remaining_capacity)

        # Finalize chunked requests
        for req in self._contexts_to_be_chunked:
            if req.context_chunk_size > 0:
                self._context_requests.append(req)
                self._batch_num_tokens += req.context_chunk_size

        # Sort requests for consistency with C++
        sort_requests_by_lora(
            self._context_requests, self._generation_requests, not self._all_context_requests_fit
        )

        return (self._context_requests, self._generation_requests, self._num_fitting)

    # ------------------------------------------------------------------
    # Context chunking and sorting helpers
    # ------------------------------------------------------------------

    def _set_ctx_requests_chunk_size(self, requests: RequestList, capacity: Optional[int]):
        for req in requests:
            req.context_chunk_size = 0

        policy = self.ctx_chunk_config.chunking_policy
        unit_size = self.ctx_chunk_config.chunk_unit_size

        if policy == ChunkingPolicy.EQUAL_PROGRESS:
            self._chunk_equal_progress(requests, capacity, unit_size)
        elif policy == ChunkingPolicy.FIRST_COME_FIRST_SERVED:
            self._chunk_fcfs(requests, capacity, unit_size)
        else:
            raise ValueError(f"Invalid chunking policy: {policy}")

        self._fit_draft_tokens(requests, capacity, unit_size)

    def _chunk_equal_progress(self, requests: RequestList, capacity: Optional[int], unit_size: int):
        num_ctx_tokens = 0
        num_tokens_single_loop = 1

        while (capacity is None or num_ctx_tokens < capacity) and num_tokens_single_loop > 0:
            num_tokens_single_loop = 0
            for req in requests:
                past_size = req.context_chunk_size
                suggested_size = min(past_size + unit_size, req.context_remaining_length)
                req.context_chunk_size = suggested_size
                actual_size = req.context_chunk_size
                actual_increment = actual_size - past_size
                if capacity is not None and (num_ctx_tokens + actual_increment > capacity):
                    req.context_chunk_size = past_size
                    continue
                if self.max_context_length is not None and actual_size > self.max_context_length:
                    req.context_chunk_size = past_size
                    continue
                num_ctx_tokens += actual_increment
                num_tokens_single_loop += actual_increment

    def _chunk_fcfs(self, requests: RequestList, capacity: Optional[int], unit_size: int):
        current_capacity = capacity if capacity is not None else float("inf")

        for req in requests:
            req.context_chunk_size = compute_fcfs_context_chunk_size(
                req.context_remaining_length,
                current_capacity if capacity is not None else None,
                self.max_context_length,
                unit_size,
            )
            if capacity is not None:
                current_capacity -= req.context_chunk_size

    def _fit_draft_tokens(self, requests: RequestList, capacity: Optional[int], unit_size: int):
        num_ctx_tokens = sum(req.context_chunk_size for req in requests)

        for req in requests:
            if req.is_last_context_chunk and req.has_draft_tokens:
                remainder = req.context_chunk_size % unit_size
                remaining_space = 0 if remainder == 0 else unit_size - remainder
                if self.max_context_length is not None:
                    remaining_context_len = self.max_context_length - req.context_chunk_size
                    remaining_space = min(remaining_space, remaining_context_len)
                if capacity is not None:
                    remaining_space = min(remaining_space, capacity - num_ctx_tokens)
                    num_ctx_tokens += remaining_space
                draft_discard = req.num_draft_tokens - remaining_space
                if draft_discard > 0:
                    logger.debug(f"Discarding {draft_discard} draft tokens")
                    if hasattr(req, "discard_draft_tokens"):
                        req.discard_draft_tokens(draft_discard)


class SchedulerPolicyBase(ABC):
    """
    Abstract base class for capacity scheduler policies.
    Each policy implements its own scheduling logic.
    """

    @abstractmethod
    def schedule(
        self,
        scheduler: "PyCapacityScheduler",
        active_requests: RequestList,
        token_tracker: Optional[TokenBudgetTracker] = None,
    ) -> tuple[RequestList, RequestList, RequestList]:
        """
        Schedule requests according to the policy.

        Args:
            scheduler: The capacity scheduler instance (for accessing shared state)
            active_requests: List of active requests to schedule
            token_tracker: If provided, fuses token-budget checks into the
                capacity loop for single-pass scheduling.

        Returns:
            Tuple of (fitting_requests, fitting_disagg_gen_init_requests,
            paused_requests)
        """
        raise NotImplementedError


class MaxRequestsPolicy(SchedulerPolicyBase):
    """
    MaxRequestsScheduler: Simple request count limiting without KV cache checks.
    C++ reference: capacityScheduler.cpp:154-176
    """

    def schedule(
        self,
        scheduler: "PyCapacityScheduler",
        active_requests: RequestList,
        token_tracker: Optional[TokenBudgetTracker] = None,
    ) -> tuple[RequestList, RequestList, RequestList]:
        scheduled_requests: RequestList = []

        for req in active_requests:
            if not scheduler._can_be_scheduled(req):
                continue

            if len(scheduled_requests) >= scheduler.max_num_requests:
                break

            if (
                req.is_encoder_init_state
                or req.is_context_init_state
                or req.is_generation_in_progress_state
            ):
                if token_tracker is not None:
                    if not token_tracker.try_add(req):
                        break
                    if token_tracker.batch_full:
                        scheduled_requests.append(req)
                        break
                scheduled_requests.append(req)

        return scheduled_requests, [], []


class GuaranteedNoEvictPolicy(SchedulerPolicyBase):
    """
    GuaranteedNoEvictScheduler: Reserve blocks for requests to complete without eviction.
    C++ reference: capacityScheduler.cpp:194-331
    """

    def __init__(self, static_batch: bool = False):
        self.static_batch = static_batch

    def schedule(
        self,
        scheduler: "PyCapacityScheduler",
        active_requests: RequestList,
        token_tracker: Optional[TokenBudgetTracker] = None,
    ) -> tuple[RequestList, RequestList, RequestList]:
        scheduled_requests: RequestList = []
        fitting_disagg: RequestList = []
        has_peft = scheduler.peft_cache_manager is not None

        skipping_is_relevant = scheduler._is_skipping_relevant()

        newly_contributed_context_blocks: Set = set()
        newly_contributed_cross_context_blocks: Set = set()
        if not self.static_batch and skipping_is_relevant:
            newly_contributed_context_blocks, newly_contributed_cross_context_blocks = (
                scheduler._prefill_contributed_blocks(active_requests)
            )

        reserved_blocks = NoEvictScheduledBlocksManager(scheduler.kv_cache_manager)
        reserved_cross_blocks: Optional[NoEvictScheduledBlocksManager] = None
        if scheduler.cross_kv_cache_manager is not None:
            reserved_cross_blocks = NoEvictScheduledBlocksManager(scheduler.cross_kv_cache_manager)

        # PEFT state - only used when has_peft
        claimed_peft_pages = 0
        available_peft_pages = scheduler._get_max_peft_pages() if has_peft else 0
        uniq_task_ids: Optional[set[int]] = set() if has_peft else None

        pending_requests: RequestList = []
        pending_dis_gen_init_requests: RequestList = []

        # Cache hot-path locals to avoid repeated attribute/method lookups
        _has_tracker = token_tracker is not None
        _until = scheduler._no_schedule_until_state_value
        _after = scheduler._no_schedule_after_state_value
        _max_num = scheduler.max_num_requests
        _has_cross = reserved_cross_blocks is not None
        _gen_in_progress = scheduler._gen_in_progress_state_value
        _sched_append = scheduled_requests.append
        _pending_append = pending_requests.append
        _disagg_pending_append = pending_dis_gen_init_requests.append
        num_scheduled = 0

        # First pass: process in-progress generation and classify requests.
        # Block decrements are deferred to batch_decrement_list after the loop
        # for a single Python→C++ call (available_blocks is not read in first pass).
        # At this point scheduled_requests contains only generation requests
        # (context → pending_requests, disagg → pending_dis_gen_init_requests),
        # so we pass it directly to batch_decrement_list instead of a separate list.
        for req in active_requests:
            # Inlined _can_be_scheduled_with_disagg_exception
            is_disagg = req.is_disagg_generation_init_state
            if not is_disagg:
                sv = req.state_value
                if not (_until <= sv < _after):
                    continue

            if num_scheduled >= _max_num:
                break

            # sv is defined (set above on the non-disagg branch);
            # replaces req.is_generation_in_progress_state to reuse cached state_value.
            if not is_disagg and sv == _gen_in_progress:
                # rc: 1=continue, 0=token budget exceeded, -1=batch full
                rc = 1
                if _has_tracker:
                    rc = token_tracker.try_add_generation(req)
                    if rc == 0:
                        break  # token budget exceeded

                _sched_append(req)
                num_scheduled += 1

                if has_peft:
                    lora_task_id, is_new_task, peft_pages = scheduler._get_peft_task_info(
                        req, uniq_task_ids
                    )
                    if is_new_task:
                        claimed_peft_pages += peft_pages
                        uniq_task_ids.add(lora_task_id)

                if rc < 0:
                    break  # batch full (after all bookkeeping)

            elif is_disagg:
                _disagg_pending_append(req)
            else:
                _pending_append(req)

        # Batch-decrement blocks using C++ batch API (single boundary crossing)
        reserved_blocks.batch_decrement_list(scheduled_requests)
        if _has_cross:
            reserved_cross_blocks.batch_decrement_list(scheduled_requests)
        # Sync single-window scalar back to dict before second pass reads it
        reserved_blocks.sync_to_dict()
        if _has_cross:
            reserved_cross_blocks.sync_to_dict()

        # Second pass: process pending requests
        # Skip entirely if first pass already filled the batch
        if (_has_tracker and token_tracker.batch_full) or num_scheduled >= _max_num:
            return scheduled_requests, fitting_disagg, []
        if not self.static_batch or num_scheduled == 0:
            if has_peft:
                available_peft_pages -= claimed_peft_pages

            # Disagg requests: all are disagg_generation_init — skip
            # beneficial_to_skip (never applies) and route directly to
            # fitting_disagg.
            _fitting_disagg_append = fitting_disagg.append
            for req in pending_dis_gen_init_requests:
                if num_scheduled >= _max_num:
                    break

                if not reserved_blocks.preview_reserve(req):
                    break
                if _has_cross:
                    if not reserved_cross_blocks.preview_reserve(req):
                        break

                if has_peft:
                    lora_task_id, is_new_task, needed_peft_pages = scheduler._get_peft_task_info(
                        req, uniq_task_ids
                    )
                    if needed_peft_pages > available_peft_pages:
                        continue
                    available_peft_pages -= needed_peft_pages
                    if is_new_task:
                        uniq_task_ids.add(lora_task_id)

                if _has_tracker:
                    if not token_tracker.try_add_context(req):
                        break
                _fitting_disagg_append(req)
                num_scheduled += 1
                reserved_blocks.commit_preview()
                if _has_cross:
                    reserved_cross_blocks.commit_preview()
                if _has_tracker and token_tracker.batch_full:
                    break

            # Context/encoder requests: none are disagg — skip disagg
            # checks and route directly to scheduled_requests.
            _skip_check = not self.static_batch and skipping_is_relevant
            for req in pending_requests:
                if _skip_check and scheduler._beneficial_to_skip(
                    req,
                    newly_contributed_context_blocks,
                    newly_contributed_cross_context_blocks,
                ):
                    continue

                if num_scheduled >= _max_num:
                    break

                if req.is_context_init_state:
                    if not reserved_blocks.preview_reserve(req):
                        break
                    if _has_cross:
                        if not reserved_cross_blocks.preview_reserve(req):
                            break

                    if has_peft:
                        lora_task_id, is_new_task, needed_peft_pages = (
                            scheduler._get_peft_task_info(req, uniq_task_ids)
                        )
                        if needed_peft_pages > available_peft_pages:
                            continue
                        available_peft_pages -= needed_peft_pages
                        if is_new_task:
                            uniq_task_ids.add(lora_task_id)

                    if _has_tracker:
                        if not token_tracker.try_add_context(req):
                            break
                    _sched_append(req)
                    num_scheduled += 1
                    reserved_blocks.commit_preview()
                    if _has_cross:
                        reserved_cross_blocks.commit_preview()
                    if _has_tracker and token_tracker.batch_full:
                        break

        return scheduled_requests, fitting_disagg, []


class MaxUtilizationPolicy(SchedulerPolicyBase):
    """
    MaxUtilizationScheduler: Maximize utilization, may pause started requests.
    C++ reference: capacityScheduler.cpp:341-425
    """

    def schedule(
        self,
        scheduler: "PyCapacityScheduler",
        active_requests: RequestList,
        token_tracker: Optional[TokenBudgetTracker] = None,
    ) -> tuple[RequestList, RequestList, RequestList]:
        scheduler.kv_cache_manager.start_scheduling()

        skipping_is_relevant = scheduler._is_skipping_relevant()

        scheduled_blocks_manager = MaxUtilizationScheduledBlocksManager(
            scheduler.kv_cache_manager, scheduler.two_step_lookahead
        )

        num_scheduled_peft_pages = 0
        seen_task_ids: set[int] = set()
        _max_peft_pages = scheduler._get_max_peft_pages()

        newly_contributed_context_blocks, _ = scheduler._prefill_contributed_blocks(active_requests)

        # Cache hot-path locals
        _until = scheduler._no_schedule_until_state_value
        _after = scheduler._no_schedule_after_state_value
        _max_num = scheduler.max_num_requests
        _has_tracker = token_tracker is not None
        # MaxUtilization doesn't pre-compute cross-context blocks (line 678
        # discards the second return). Use a reusable set cleared per
        # iteration to avoid per-call allocation while matching the C++
        # semantics of not accumulating cross-context blocks across requests.
        _cross_ctx_blocks: Set = set()

        def is_started_request(req: LlmRequest) -> bool:
            sv = req.state_value
            if not (_until <= sv < _after):
                return False
            return (
                req.is_context_init_state and not req.is_first_context_chunk
            ) or req.is_generation_in_progress_state

        scheduled_requests: RequestList = []
        fitting_disagg: RequestList = []
        paused_requests: RequestList = []
        num_scheduled = 0

        requests_list = list(active_requests)
        req_it_end = len(requests_list)
        req_it = 0

        while req_it < req_it_end:
            req = requests_list[req_it]

            # Inlined _can_be_scheduled_with_disagg_exception
            if not req.is_disagg_generation_init_state:
                sv = req.state_value
                if not (_until <= sv < _after):
                    req_it += 1
                    continue

            _cross_ctx_blocks.clear()
            if skipping_is_relevant and scheduler._beneficial_to_skip(
                req, newly_contributed_context_blocks, _cross_ctx_blocks
            ):
                req_it += 1
                continue

            result, num_scheduled_peft_pages = self._try_scheduling_request(
                scheduler,
                req,
                scheduled_requests,
                fitting_disagg,
                scheduled_blocks_manager,
                num_scheduled_peft_pages,
                _max_peft_pages,
                num_scheduled,
                _max_num,
                seen_task_ids,
                token_tracker,
            )

            if result is True:
                num_scheduled += 1
                if _has_tracker and token_tracker.batch_full:
                    break
                req_it += 1
            elif result is None:
                # Token budget exhausted — pausing won't help, stop.
                break
            else:
                # Capacity failure — try pausing an older request.
                last_started_idx = None
                for i in range(req_it_end - 1, req_it - 1, -1):
                    if is_started_request(requests_list[i]):
                        last_started_idx = i
                        break

                if last_started_idx is not None:
                    paused_req = requests_list[last_started_idx]
                    scheduler.kv_cache_manager.scheduling_remove_sequence(paused_req.py_request_id)
                    paused_requests.append(paused_req)
                    # Don't decrement num_scheduled: the paused request is at
                    # index >= req_it (unprocessed), so it was never counted.
                    req_it_end = last_started_idx
                else:
                    break

        return scheduled_requests, fitting_disagg, paused_requests

    def _try_scheduling_request(
        self,
        scheduler: "PyCapacityScheduler",
        req: LlmRequest,
        scheduled_requests: RequestList,
        fitting_disagg: RequestList,
        scheduled_blocks_manager: "MaxUtilizationScheduledBlocksManager",
        num_scheduled_peft_pages: int,
        max_peft_pages: int,
        num_scheduled: int,
        max_num_requests: int,
        seen_task_ids: set[int],
        token_tracker: Optional[TokenBudgetTracker] = None,
    ) -> tuple[Optional[bool], int]:
        """Try to schedule a request.

        Returns a tuple of (result, num_scheduled_peft_pages):
            result is True: request scheduled successfully.
            result is False: capacity failure (KV blocks, PEFT, max_num_requests)
                   — caller may pause an older request and retry.
            result is None: token budget exhausted — caller should stop scheduling
                  (pausing won't help because it doesn't free token budget).
            num_scheduled_peft_pages: updated running total of scheduled PEFT
                  pages (mirrors C++ pass-by-reference semantics in
                  capacityScheduler.cpp:429).
        """
        if num_scheduled >= max_num_requests:
            return False, num_scheduled_peft_pages

        if scheduled_blocks_manager.prepare_blocks_if_schedulable(req) is None:
            return False, num_scheduled_peft_pages

        # PEFT check only when needed — compute required pages but do NOT
        # commit (add to seen_task_ids / accumulate pages) until all checks
        # pass. Matches C++ which commits atomically on success only.
        _peft_lora_task_id = 0
        _peft_is_new_task = False
        _peft_required_pages = 0
        if scheduler.peft_cache_manager is not None:
            _peft_lora_task_id, _peft_is_new_task, _peft_required_pages = (
                scheduler._get_peft_task_info(req, seen_task_ids)
            )
            if _peft_required_pages + num_scheduled_peft_pages > max_peft_pages:
                return False, num_scheduled_peft_pages

        # Token budget check — return None (not False) so the caller
        # does NOT enter the pause/backtrack path. Pausing frees KV
        # blocks but not token budget, so retrying would fail again.
        if token_tracker is not None:
            if not token_tracker.try_add(req):
                return None, num_scheduled_peft_pages

        # All checks passed — commit all state atomically.
        scheduled_blocks_manager.update_scheduled_blocks()
        if _peft_is_new_task:
            seen_task_ids.add(_peft_lora_task_id)
            num_scheduled_peft_pages += _peft_required_pages
        if req.is_disagg_generation_init_state:
            fitting_disagg.append(req)
        else:
            scheduled_requests.append(req)
        return True, num_scheduled_peft_pages


class NoEvictScheduledBlocksManager:
    """
    Python equivalent of C++ kv_cache_manager::NoEvictScheduledBlocksManager.
    Tracks available blocks per window size for GUARANTEED_NO_EVICT scheduling.

    Includes single-window fast path: when only one window size exists
    (the common case), avoids dict iteration overhead.

    Reference: cpp/tensorrt_llm/batch_manager/scheduledBlocksManager.h:29-62
    """

    def __init__(self, kv_cache_manager):
        self.kv_cache_manager = kv_cache_manager
        stats = kv_cache_manager.get_kv_cache_stats()
        self.available_blocks: dict[int, int] = dict(stats.num_free_blocks_per_window_size)
        self._preview_valid = False
        # Single-window fast path: avoid dict iteration when only one window
        if len(self.available_blocks) == 1:
            ws, avail = next(iter(self.available_blocks.items()))
            self._single_ws = ws
            self._single_avail = avail
        else:
            self._single_ws = None
            self._single_avail = 0

    def batch_decrement_list(self, requests: RequestList) -> None:
        """Batch-decrement blocks for a list of requests using C++ batch API.

        Uses get_remaining_blocks_to_completion_batch for a single Python→C++
        call instead of N individual calls.
        """
        if not requests:
            return
        if self._single_ws is not None:
            needed_list = self.kv_cache_manager.get_remaining_blocks_to_completion_batch(
                requests, self._single_ws
            )
            self._single_avail -= sum(needed_list)
        else:
            for ws in self.available_blocks:
                needed_list = self.kv_cache_manager.get_remaining_blocks_to_completion_batch(
                    requests, ws
                )
                self.available_blocks[ws] -= sum(needed_list)

    def sync_to_dict(self) -> None:
        """Write single-window scalar back to dict. Call before dict is read."""
        if self._single_ws is not None:
            self.available_blocks[self._single_ws] = self._single_avail

    def preview_reserve(self, req: LlmRequest) -> bool:
        """Check if request fits (no mutation). Caches needed blocks for commit_preview.

        Call commit_preview() after all intermediate checks (PEFT, token) pass
        to apply the cached decrement. This avoids a second C++ call.
        """
        self._preview_valid = False
        if self._single_ws is not None:
            needed = self.kv_cache_manager.get_remaining_blocks_to_completion(req, self._single_ws)
            if needed > self._single_avail:
                return False
            self._preview_needed_single = needed
            self._preview_valid = True
            return True
        needed_per_ws = {}
        for ws, avail in self.available_blocks.items():
            needed = self.kv_cache_manager.get_remaining_blocks_to_completion(req, ws)
            if needed > avail:
                return False
            needed_per_ws[ws] = needed
        self._preview_needed_multi = needed_per_ws
        self._preview_valid = True
        return True

    def commit_preview(self) -> None:
        """Apply the cached decrement from the last preview_reserve call."""
        assert self._preview_valid, "commit_preview called without a successful preview_reserve"
        self._preview_valid = False
        if self._single_ws is not None:
            self._single_avail -= self._preview_needed_single
            return
        for ws, needed in self._preview_needed_multi.items():
            self.available_blocks[ws] -= needed


class MaxUtilizationScheduledBlocksManager:
    """
    Python equivalent of C++ kv_cache_manager::MaxUtilizationScheduledBlocksManager.
    Tracks scheduled blocks per window size for MAX_UTILIZATION scheduling.

    Includes single-window fast path for the common case.

    Reference: cpp/tensorrt_llm/batch_manager/scheduledBlocksManager.h:64-117
    """

    def __init__(self, kv_cache_manager, two_steps_look_ahead: bool):
        self.kv_cache_manager = kv_cache_manager
        self.two_steps_look_ahead = two_steps_look_ahead
        window_sizes = set(kv_cache_manager.max_attention_window_vec)
        self.num_scheduled_blocks: dict[int, int] = {ws: 0 for ws in window_sizes}
        self._pending_total: int = 0
        self._pending_blocks: dict[int, int] = {}
        # Single-window fast path
        if len(self.num_scheduled_blocks) == 1:
            self._single_ws: Optional[int] = next(iter(self.num_scheduled_blocks))
            self._single_scheduled: int = 0
        else:
            self._single_ws = None
            self._single_scheduled = 0

    def prepare_blocks_if_schedulable(self, req: LlmRequest) -> Optional[bool]:
        """Check if request can be scheduled. Returns True or None (can't fit).

        For single-window: returns True and caches the scheduled_total
        internally. Call update_scheduled_blocks() to commit.
        For multi-window: returns a dict of new block counts (legacy).
        """
        if self._single_ws is not None:
            required = self.kv_cache_manager.get_needed_blocks_one_step(
                req, self.two_steps_look_ahead, self._single_ws
            )
            scheduled_total = self._single_scheduled + required
            if not self.kv_cache_manager.scheduling_has_free_blocks(
                scheduled_total, self._single_ws
            ):
                return None
            self._pending_total = scheduled_total
            return True
        # Multi-window path
        blocks_if_scheduled = {}
        for window_size, num_scheduled in self.num_scheduled_blocks.items():
            required = self.kv_cache_manager.get_needed_blocks_one_step(
                req, self.two_steps_look_ahead, window_size
            )
            scheduled_total = num_scheduled + required
            if not self.kv_cache_manager.scheduling_has_free_blocks(scheduled_total, window_size):
                return None
            blocks_if_scheduled[window_size] = scheduled_total
        self._pending_blocks = blocks_if_scheduled
        return True

    def update_scheduled_blocks(self) -> None:
        """Commit the block counts from the last prepare_blocks_if_schedulable call."""
        if self._single_ws is not None:
            self._single_scheduled = self._pending_total
            self.num_scheduled_blocks[self._single_ws] = self._single_scheduled
            return
        for window_size, total in self._pending_blocks.items():
            self.num_scheduled_blocks[window_size] = total


class PyCapacityScheduler:
    """KV cache capacity scheduler with optional fused token budget tracking.

    Python implementation based on the C++ CapacityScheduler. Core KV-block
    logic follows cpp/tensorrt_llm/batch_manager/capacityScheduler.cpp.

    Extension: accepts an optional TokenBudgetTracker via schedule_request()
    to fuse token-budget checks into the capacity loop (single-pass
    scheduling). When no tracker is provided, behaves identically to the
    C++ implementation.

    Policies:
    - MaxRequestsScheduler: No KV cache manager, simple request count limit
    - GuaranteedNoEvictScheduler: Reserve blocks for completion, no eviction
    - StaticBatchScheduler: Only schedule when no requests are active
    - MaxUtilizationScheduler: Maximize utilization, may pause requests
    """

    def __init__(
        self,
        max_num_requests: int,
        kv_cache_manager=None,
        peft_cache_manager=None,
        scheduler_policy: CapacitySchedulerPolicy = CapacitySchedulerPolicy.GUARANTEED_NO_EVICT,
        cross_kv_cache_manager=None,
        two_step_lookahead: bool = False,
        no_schedule_until_state: LlmRequestState = LlmRequestState.CONTEXT_INIT,
        no_schedule_after_state: LlmRequestState = LlmRequestState.GENERATION_COMPLETE,
    ):
        self.max_num_requests = max_num_requests
        self.kv_cache_manager = kv_cache_manager
        self.peft_cache_manager = peft_cache_manager
        self.cross_kv_cache_manager = cross_kv_cache_manager
        self.scheduler_policy = scheduler_policy
        self.two_step_lookahead = two_step_lookahead
        self.no_schedule_until_state = no_schedule_until_state
        self.no_schedule_after_state = no_schedule_after_state
        # Cache state values to avoid repeated .value access (optimization)
        self._no_schedule_until_state_value = no_schedule_until_state.value
        self._no_schedule_after_state_value = no_schedule_after_state.value
        self._gen_in_progress_state_value = LlmRequestState.GENERATION_IN_PROGRESS.value

        # Initialize the appropriate policy
        self._policy = self._create_policy()

    def _create_policy(self) -> SchedulerPolicyBase:
        """Create the appropriate policy based on configuration."""
        if self.kv_cache_manager is None:
            return MaxRequestsPolicy()
        elif self.scheduler_policy == CapacitySchedulerPolicy.MAX_UTILIZATION:
            return MaxUtilizationPolicy()
        elif self.scheduler_policy == CapacitySchedulerPolicy.GUARANTEED_NO_EVICT:
            return GuaranteedNoEvictPolicy(static_batch=False)
        elif self.scheduler_policy == CapacitySchedulerPolicy.STATIC_BATCH:
            return GuaranteedNoEvictPolicy(static_batch=True)
        else:
            raise ValueError(f"Unsupported scheduler policy: {self.scheduler_policy}")

    def _can_be_scheduled(self, req: LlmRequest) -> bool:
        """
        Check if request is within the schedulable state range.
        Returns True if request has reached no_schedule_until_state
        but has not yet reached no_schedule_after_state.
        Optimized: use state_value property to avoid enum object creation
        """
        # Use state_value property (returns int directly, avoids enum object creation)
        state_value = req.state_value
        # Inline comparison: must have reached until_state but not after_state
        return (
            state_value >= self._no_schedule_until_state_value
            and state_value < self._no_schedule_after_state_value
        )

    def _is_skipping_relevant(self) -> bool:
        """
        Check if block reuse skip optimization is relevant.
        Disabled for VSWA (Variable Sliding Window Attention).
        C++ reference: capacityScheduler.cpp:207-208, 348
        """
        if self.kv_cache_manager is None:
            return False
        if self.kv_cache_manager.is_variable_window:
            return False
        if (
            self.cross_kv_cache_manager is not None
            and self.cross_kv_cache_manager.is_variable_window
        ):
            return False
        return True

    def _prefill_contributed_blocks(self, active_requests: RequestList) -> tuple[set, set]:
        """
        Collect blocks contributed by chunked context requests already executing.
        These blocks can be reused by later requests.

        C++ reference: capacityScheduler.cpp:34-68 (prefillWithChunkedContextsAlreadyExecuting)
        """
        newly_contributed_context_blocks: Set = set()
        newly_contributed_cross_context_blocks: Set = set()

        if self.kv_cache_manager is None:
            return newly_contributed_context_blocks, newly_contributed_cross_context_blocks

        enable_block_reuse = self.kv_cache_manager.enable_block_reuse
        cross_enable_reuse = (
            self.cross_kv_cache_manager is not None
            and self.cross_kv_cache_manager.enable_block_reuse
        )

        for req in active_requests:
            # Check: isContextInitState() && !isFirstContextChunk()
            if req.is_context_init_state and not req.is_first_context_chunk:
                # Chunked context request already executing
                if enable_block_reuse:
                    unique_tokens = req.get_unique_tokens(0)
                    block_key = self.kv_cache_manager.find_new_context_block(unique_tokens, req)
                    if block_key is not None:
                        newly_contributed_context_blocks.add(block_key)

                if cross_enable_reuse:
                    encoder_unique_tokens = req.get_encoder_unique_tokens()
                    if encoder_unique_tokens is not None:
                        block_key = self.cross_kv_cache_manager.find_new_context_block(
                            encoder_unique_tokens, req
                        )
                        if block_key is not None:
                            newly_contributed_cross_context_blocks.add(block_key)

        return newly_contributed_context_blocks, newly_contributed_cross_context_blocks

    def _one_manager_beneficial_to_skip(
        self, kv_cache_manager, unique_tokens, req: LlmRequest, newly_contributed_blocks: set
    ) -> bool:
        """
        Check if skipping is beneficial for one KV cache manager.
        C++ reference: capacityScheduler.cpp:70-92 (oneManagerBeneficialToSkip)
        """
        new_context_block = kv_cache_manager.find_new_context_block(unique_tokens, req)
        if new_context_block is not None:
            if new_context_block in newly_contributed_blocks:
                return True
            newly_contributed_blocks.add(new_context_block)
        return False

    def _beneficial_to_skip(
        self,
        req: LlmRequest,
        newly_contributed_context_blocks: set,
        newly_contributed_cross_context_blocks: set,
    ) -> bool:
        """
        Check if it's beneficial to skip this request.
        A request should be skipped if it can reuse blocks contributed by
        already scheduled context requests.

        C++ reference: capacityScheduler.cpp:97-123 (beneficialToSkip)
        """
        if not (req.is_context_init_state and req.is_first_context_chunk):
            return False

        if self.kv_cache_manager is not None and self.kv_cache_manager.enable_block_reuse:
            unique_tokens = req.get_unique_tokens(0)
            if self._one_manager_beneficial_to_skip(
                self.kv_cache_manager, unique_tokens, req, newly_contributed_context_blocks
            ):
                return True

        if (
            self.cross_kv_cache_manager is not None
            and self.cross_kv_cache_manager.enable_block_reuse
        ):
            encoder_unique_tokens = req.get_encoder_unique_tokens()
            if encoder_unique_tokens is not None:
                if self._one_manager_beneficial_to_skip(
                    self.cross_kv_cache_manager,
                    encoder_unique_tokens,
                    req,
                    newly_contributed_cross_context_blocks,
                ):
                    return True

        return False

    def _get_max_peft_pages(self) -> int:
        """Get maximum PEFT cache pages."""
        if self.peft_cache_manager is None:
            return 2**31 - 1  # INT_MAX equivalent
        return self.peft_cache_manager.max_device_pages

    def _get_peft_pages_for_request(self, req: LlmRequest) -> int:
        """Get PEFT pages needed for a request."""
        if self.peft_cache_manager is None:
            return 0
        return self.peft_cache_manager.determine_num_pages(req)

    def _get_peft_task_info(
        self, req: LlmRequest, seen_task_ids: set[int]
    ) -> tuple[Optional[int], bool, int]:
        """
        Get PEFT task information for a request.
        Returns (lora_task_id, is_new_task, required_pages).
        """
        lora_task_id = getattr(req, "lora_task_id", None)
        is_new_task = lora_task_id is not None and lora_task_id not in seen_task_ids
        required_pages = self._get_peft_pages_for_request(req) if is_new_task else 0
        return lora_task_id, is_new_task, required_pages

    def schedule_request(
        self,
        active_requests: RequestList,
        token_tracker: Optional["TokenBudgetTracker"] = None,
    ) -> tuple[RequestList, RequestList, RequestList]:
        """
        Schedule requests based on the configured policy.

        Args:
            active_requests: List of active requests to consider
            token_tracker: If provided, fuses token-budget checks into
                the capacity policy loop (single-pass scheduling).

        Returns:
            Tuple of (fitting_requests, fitting_disagg_gen_init_requests, paused_requests)

        C++ reference: capacityScheduler.cpp:488-539 (CapacityScheduler::operator())
        """
        fitting_requests, fitting_disagg_gen_init_requests, paused = self._policy.schedule(
            self, active_requests, token_tracker
        )

        logger.debug(
            f"[Summary] Capacity scheduler allows {len(fitting_requests)} requests, "
            f"pauses {len(paused)} requests"
        )

        return fitting_requests, fitting_disagg_gen_init_requests, paused


class UnifiedScheduler(RequestScheduler):
    """Python-only scheduler — drop-in replacement for SimpleScheduler.

    Replaces the two-pass pipeline in SimpleScheduler (C++ bindings:
    BindCapacityScheduler → BindMicroBatchScheduler) with a single-pass
    fused approach. Gated by SchedulerConfig(use_python_scheduler=True).

    Implements the same schedule_request() interface as SimpleScheduler,
    so py_executor.py uses the same code path for both schedulers.

    Key difference: capacity and token-budget checks run in one loop via
    TokenBudgetTracker, instead of capacity first then microbatch second.
    This eliminates one full iteration over fitting_requests and reduces
    scheduler-side KV bookkeeping on requests that would be dropped by the
    token budget.
    """

    def __init__(
        self,
        max_batch_size: int,
        max_num_tokens: int,
        kv_cache_manager,
        peft_cache_manager,
        scheduler_policy: CapacitySchedulerPolicy,
        ctx_chunk_config: Optional[tuple[StrEnum, int]] = None,
        cross_kv_cache_manager=None,
        two_step_lookahead: bool = False,
        scheduler_capacity: Optional[int] = None,
        schedule_step_config: Optional[ScheduleStepConfig] = None,
        dist=None,
    ):
        super(UnifiedScheduler, self).__init__(schedule_step_config=schedule_step_config, dist=dist)
        # Use scheduler_capacity if provided, otherwise fall back to max_batch_size
        # scheduler_capacity may differ from max_batch_size (e.g., adjusted for attention_dp + disagg)
        capacity = scheduler_capacity if scheduler_capacity is not None else max_batch_size

        self.capacity_scheduler = PyCapacityScheduler(
            max_num_requests=capacity,
            kv_cache_manager=kv_cache_manager,
            peft_cache_manager=peft_cache_manager,
            scheduler_policy=scheduler_policy,
            cross_kv_cache_manager=cross_kv_cache_manager,
            two_step_lookahead=two_step_lookahead,
        )

        self._max_batch_size = max_batch_size
        self._max_num_tokens = max_num_tokens
        self._ctx_chunk_config = None
        if ctx_chunk_config:
            input_policy = ctx_chunk_config[0]
            if "EQUAL_PROGRESS" in str(input_policy):
                policy_enum = ChunkingPolicy.EQUAL_PROGRESS
            else:
                policy_enum = ChunkingPolicy.FIRST_COME_FIRST_SERVED
            self._ctx_chunk_config = ContextChunkingConfig(policy_enum, ctx_chunk_config[1])

    def schedule_request(
        self, active_requests: RequestList, inflight_request_ids: object = None
    ) -> SchedulerOutput:
        """Single-pass fused capacity + token budget scheduling.

        A TokenBudgetTracker is passed into the capacity policy loop so that
        each request admission check includes both KV-block and token-budget
        gates simultaneously. The tracker classifies requests into
        context/generation and handles chunking/sorting.
        """
        tracker = TokenBudgetTracker(
            max_batch_size=self._max_batch_size,
            max_num_tokens=self._max_num_tokens,
            ctx_chunk_config=self._ctx_chunk_config,
            inflight_request_ids=inflight_request_ids,
        )
        _, fitting_disagg_gen_init, paused_requests = self.capacity_scheduler.schedule_request(
            active_requests, tracker
        )
        # num_fitting reflects requests passing both capacity AND token budget.
        context_requests, generation_requests, num_fitting = tracker.finalize()
        return SchedulerOutput(
            context_requests=context_requests,
            generation_requests=generation_requests,
            paused_requests=paused_requests,
            fitting_disagg_gen_init_requests=fitting_disagg_gen_init,
            num_fitting_requests=num_fitting,
        )

    def can_schedule(self, requests: RequestList) -> bool:
        fitting, _, _ = self.capacity_scheduler.schedule_request(requests)
        return len(fitting) == len(requests)
