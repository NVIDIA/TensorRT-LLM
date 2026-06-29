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

import enum
from typing import Optional

from tensorrt_llm.llmapi.llm_args import CapacitySchedulerPolicy
from tensorrt_llm.logger import logger

from ..llm_request import LlmRequest, LlmRequestState, get_draft_token_length
from .scheduler import (
    RequestList,
    RequestScheduler,
    SchedulerOutput,
    _get_lora_task_id,
    drop_decoder_context_requests_waiting_for_encoder_output,
)


class ScheduleAction(enum.Enum):
    """Result of a per-request scheduling attempt."""

    SCHEDULED = "scheduled"  # success — payload contains tokens etc.
    SKIP = "skip"  # skip this request, continue the loop
    STOP = "stop"  # stop the scheduling loop


class BudgetTracker:
    """Tracks token, request, and PEFT budgets for one scheduling iteration.

    Centralizes the budget logic that was previously spread across the main
    scheduling loop.  Designed for future extensibility to multiple memory
    pools — callers interact through ``can_fit_tokens`` /
    ``remaining_tokens`` / ``commit`` / ``peft_pages_needed`` without knowing the underlying pool
    topology.
    """

    def __init__(
        self,
        max_num_tokens: Optional[int],
        max_num_requests: int,
        peft_cache_manager=None,
    ):
        self.max_num_tokens = max_num_tokens
        self.max_num_requests = max_num_requests
        self.num_tokens = 0
        self.num_requests = 0

        # PEFT accounting
        self._peft_cache_manager = peft_cache_manager
        self._max_peft_pages = (
            peft_cache_manager.max_device_pages if peft_cache_manager is not None else 0
        )
        self._claimed_peft_pages = 0
        self._seen_peft_task_ids: set[int] = set()

    # ---- Token / request budget ----

    @property
    def requests_full(self) -> bool:
        return self.num_requests >= self.max_num_requests

    def can_fit_tokens(self, num_tokens: int) -> bool:
        """Check if *num_tokens* fits within the remaining token budget."""
        if self.max_num_tokens is not None and (self.num_tokens + num_tokens > self.max_num_tokens):
            return False
        return True

    @property
    def remaining_tokens(self) -> Optional[int]:
        """Remaining token budget, or ``None`` if unlimited."""
        if self.max_num_tokens is None:
            return None
        return self.max_num_tokens - self.num_tokens

    def commit(self, req: LlmRequest, num_tokens: int, peft_pages: int) -> None:
        """Record a successfully scheduled request's token and PEFT consumption."""
        self.num_tokens += num_tokens
        self.num_requests += 1
        if peft_pages > 0:
            self.commit_peft(req, peft_pages)

    def commit_peft(self, req: LlmRequest, peft_pages: int) -> None:
        """Record only PEFT consumption without affecting token/request budgets.

        Used by disagg_gen_init requests which need PEFT accounting but don't
        participate in the forward pass.
        """
        lora_task_id = getattr(req, "lora_task_id", None)
        self._claimed_peft_pages += peft_pages
        self._seen_peft_task_ids.add(lora_task_id)

    # ---- PEFT budget ----

    def peft_pages_needed(self, req: LlmRequest) -> Optional[int]:
        """Return PEFT pages needed for *req*, or ``None`` if budget exceeded."""
        if self._peft_cache_manager is None:
            return 0
        lora_task_id = getattr(req, "lora_task_id", None)
        if lora_task_id is None or lora_task_id in self._seen_peft_task_ids:
            return 0
        required = self._peft_cache_manager.determine_num_pages(req)
        if self._claimed_peft_pages + required > self._max_peft_pages:
            return None
        return required

    def pre_claim_peft(self, req: LlmRequest) -> None:
        """Reserve PEFT pages for a non-scheduled request whose adapter is
        still loaded on device (e.g. GENERATION_TO_COMPLETE in the overlap
        executor — not yet terminated, so ensure_batch cannot evict it)."""
        if self._peft_cache_manager is None:
            return
        lora_task_id = getattr(req, "lora_task_id", None)
        if lora_task_id is None or lora_task_id in self._seen_peft_task_ids:
            return
        pages = self._peft_cache_manager.determine_num_pages(req)
        self._claimed_peft_pages += pages
        self._seen_peft_task_ids.add(lora_task_id)


class KVCacheV2Scheduler(RequestScheduler):
    """Interleaved scheduler for KV Cache Manager V2.

    Merges capacity checking (KV cache allocation via resize()) and token budget
    assignment into one unified loop.
    """

    def __init__(
        self,
        max_batch_size: int,
        max_num_tokens: Optional[int],
        kv_cache_manager,  # KVCacheManagerV2
        scheduler_policy: CapacitySchedulerPolicy,
        ctx_chunk_config: Optional[tuple] = None,
        peft_cache_manager=None,
        scheduler_capacity: Optional[int] = None,
        no_schedule_until_state: LlmRequestState = LlmRequestState.CONTEXT_INIT,
        no_schedule_after_state: LlmRequestState = LlmRequestState.GENERATION_TO_COMPLETE,
        draft_kv_cache_manager=None,  # KVCacheManagerV2 for MTP draft layers
        cross_kv_cache_manager=None,  # KVCacheManagerV2 for enc-dec cross-attn
    ):
        self.max_num_tokens = max_num_tokens
        self.max_num_requests = (
            scheduler_capacity if scheduler_capacity is not None else max_batch_size
        )
        from ..kv_cache_manager_v2 import KVCacheManagerV2

        assert isinstance(kv_cache_manager, KVCacheManagerV2), (
            f"KVCacheV2Scheduler requires KVCacheManagerV2, got {type(kv_cache_manager).__name__}"
        )
        self.kv_cache_manager = kv_cache_manager
        self.draft_kv_cache_manager = draft_kv_cache_manager
        self.cross_kv_cache_manager = cross_kv_cache_manager
        if scheduler_policy != CapacitySchedulerPolicy.MAX_UTILIZATION:
            logger.warning(
                "KVCacheV2Scheduler only supports MAX_UTILIZATION for now, "
                f"got {scheduler_policy.name}, setting to MAX_UTILIZATION"
            )
            scheduler_policy = CapacitySchedulerPolicy.MAX_UTILIZATION
        self.policy = scheduler_policy
        self.peft_cache_manager = peft_cache_manager

        # Chunking config.
        self.chunking_enabled = False
        self.chunking_policy = None
        self.chunk_unit_size = 0
        self.max_context_length = max_num_tokens
        self.tokens_per_block = kv_cache_manager.tokens_per_block
        draft_mgr_name = (
            type(draft_kv_cache_manager).__name__ if draft_kv_cache_manager is not None else "None"
        )
        cross_mgr_name = (
            type(cross_kv_cache_manager).__name__ if cross_kv_cache_manager is not None else "None"
        )
        logger.info(
            f"KVCacheV2Scheduler: tokens_per_block={self.tokens_per_block}, "
            f"max_num_tokens={max_num_tokens}, max_batch_size={max_batch_size}, "
            f"draft_mgr={draft_mgr_name}, cross_mgr={cross_mgr_name}"
        )
        if ctx_chunk_config is not None:
            self.chunking_enabled = True
            self.chunking_policy = ctx_chunk_config[0]
            self.chunk_unit_size = ctx_chunk_config[1]

        # State value caches for fast comparison.
        # Default range [CONTEXT_INIT, GENERATION_TO_COMPLETE) matches C++
        # MicroBatchScheduler. For encoder-decoder models, caller should pass
        # no_schedule_until_state=ENCODER_INIT to widen the range (same as
        # C++ trtEncoderModel which passes kENCODER_INIT).
        self._no_schedule_until_state_value = no_schedule_until_state.value
        self._no_schedule_after_state_value = no_schedule_after_state.value
        self._context_init_state_value = LlmRequestState.CONTEXT_INIT.value
        self._encoder_init_state_value = LlmRequestState.ENCODER_INIT.value
        self._disagg_gen_init_state_value = LlmRequestState.DISAGG_GENERATION_INIT.value
        self._gen_to_complete_state_value = LlmRequestState.GENERATION_TO_COMPLETE.value

    def schedule_request(
        self, active_requests: RequestList, inflight_request_ids: set[int]
    ) -> SchedulerOutput:
        active_requests = drop_decoder_context_requests_waiting_for_encoder_output(active_requests)
        # Main scheduling loop
        (
            scheduled_encoder,
            scheduled_ctx,
            scheduled_gen,
            evicted,
            disagg_candidates,
            has_chunking,
        ) = self._schedule_loop(active_requests, inflight_request_ids)

        # Sort by LoRA task ID
        scheduled_encoder.sort(key=_get_lora_task_id)
        self._sort_requests(scheduled_ctx, scheduled_gen, has_chunking)

        return SchedulerOutput(
            encoder_requests=scheduled_encoder,
            context_requests=scheduled_ctx,
            generation_requests=scheduled_gen,
            paused_requests=evicted,
            fitting_disagg_gen_init_requests=disagg_candidates,
            num_fitting_requests=(len(scheduled_encoder) + len(scheduled_ctx) + len(scheduled_gen)),
        )

    # ---- Main scheduling loop ----

    def _schedule_loop(self, active_requests, inflight_request_ids):
        scheduled_ctx: RequestList = []
        scheduled_encoder: RequestList = []
        scheduled_gen: RequestList = []
        evicted: RequestList = []
        disagg_candidates: RequestList = []
        scheduled_beam_width = 0
        has_chunking = False

        budget = BudgetTracker(
            self.max_num_tokens,
            self.max_num_requests,
            self.peft_cache_manager,
        )

        # TODO: block reuse skip optimization (_beneficial_to_skip).
        # V1 skips first-chunk ctx requests whose next block overlaps with
        # an executing chunked ctx's current chunk. Waiting one iteration
        # lets the new request reuse the committed block instead of
        # recomputing it. Saves computation, not just memory.
        # Requires a read-only radix tree probe API.

        # Use indexed iteration (while + req_it_end) so that MAX_UTIL
        # eviction can shrink the range from the tail.
        requests_list = list(active_requests)
        req_it_end = len(requests_list)
        req_it = 0

        # Context requests are always deferred to a second phase so that
        # generation requests are fully accounted for in the budget before
        # any context request competes for resources.  This mirrors the
        # two-phase approach of the old KVCacheV2DummyScheduler and
        # prevents PEFT adapter eviction failures when gen requests hold
        # adapters that can't be evicted mid-iteration.
        pending_ctx: RequestList = []

        # Pre-claim PEFT pages for GENERATION_TO_COMPLETE requests.
        # In the overlap executor these requests are no longer scheduled
        # (state outside schedulable range) but their adapters haven't
        # been released yet (mark_request_done runs after prepare_resources
        # in the next iteration).  Without this, the budget would appear
        # empty and context requests with a different adapter could be
        # admitted, causing ensure_batch to crash when it can't evict the
        # still-active adapter.
        for req in requests_list:
            if req.state_value == self._gen_to_complete_state_value:
                budget.pre_claim_peft(req)

        # --- Phase 1: generation / disagg only ---
        while req_it < req_it_end:
            req = requests_list[req_it]

            # --- Filter ---
            if req.request_id in inflight_request_ids:
                req_it += 1
                continue

            req_state_value = req.state_value

            # Disagg gen init bypasses both state gating and budget.requests_full
            # (same as C++ / V1 scheduler), but the V2 scheduler owns inline KV
            # allocation so we must allocate here. V1 defers allocation to
            # prepare_resources; V2 prepare_resources is a no-op for the primary
            # manager, so allocation must happen in the scheduling loop.
            #
            # Disagg does NOT count toward budget.num_requests because it
            # doesn't participate in the forward pass. Capacity is gated by
            # IndexMapper slot availability: prepare_context returns False when
            # no free slots remain, so the request is skipped and retried next
            # iteration. PEFT budget is still checked and committed.
            if req_state_value == self._disagg_gen_init_state_value:
                peft_pages = budget.peft_pages_needed(req)
                if peft_pages is None:
                    break

                action, tokens = self._try_schedule_disagg_gen_init(req, budget)
                if action is ScheduleAction.STOP:
                    break
                if action is ScheduleAction.SKIP:
                    req_it += 1
                    continue
                disagg_candidates.append(req)
                # Disagg requests only commit PEFT (not num_requests/num_tokens)
                # because they don't participate in the forward pass. Counting
                # them toward num_requests would steal batch slots from gen/ctx
                # and delay KV transfer initiation. IndexMapper slot availability
                # (via prepare_context) is the real capacity guard.
                if peft_pages > 0:
                    budget.commit_peft(req, peft_pages)
                req_it += 1
                continue

            # Budget check for non-disagg requests (disagg bypasses this
            # because it doesn't participate in the forward pass).
            if budget.requests_full:
                break

            if not (
                req_state_value >= self._no_schedule_until_state_value
                and req_state_value < self._no_schedule_after_state_value
            ):
                req_it += 1
                continue

            # --- Dispatch by request type ---
            if (
                req_state_value == self._encoder_init_state_value
                or req_state_value == self._context_init_state_value
            ):
                # Defer to phase 2 so gen PEFT budget is settled first.
                pending_ctx.append(req)
                req_it += 1
                continue

            else:
                peft_pages = budget.peft_pages_needed(req)
                if peft_pages is None:
                    break
                action, tokens, scheduled_beam_width, req_it_end = self._try_schedule_generation(
                    req,
                    budget,
                    requests_list,
                    req_it,
                    req_it_end,
                    evicted,
                    scheduled_beam_width,
                )
                if action is ScheduleAction.STOP:
                    break
                if action is ScheduleAction.SKIP:
                    req_it += 1
                    continue
                scheduled_gen.append(req)
                budget.commit(req, tokens, peft_pages)

            req_it += 1

        # --- Phase 2: schedule deferred context / encoder requests ---
        # Generation PEFT pages are now fully committed in the budget.
        for req in pending_ctx:
            if budget.requests_full:
                break
            peft_pages = budget.peft_pages_needed(req)
            if peft_pages is None:
                continue
            if req.state_value == self._encoder_init_state_value:
                action, tokens = self._try_schedule_encoder(req, budget)
                if action is ScheduleAction.STOP:
                    break
                scheduled_encoder.append(req)
                budget.commit(req, tokens, peft_pages)
            else:
                action, tokens, chunking_flag = self._try_schedule_context(req, budget)
                if action is ScheduleAction.STOP:
                    break
                if action is ScheduleAction.SKIP:
                    continue
                has_chunking = has_chunking or chunking_flag
                scheduled_ctx.append(req)
                budget.commit(req, tokens, peft_pages)

        # Deadlock detection: if generation requests exist but none were
        # scheduled and none were evicted, no forward pass will run and no
        # KV cache pages will ever be freed — the scheduler will spin
        # forever.  This typically happens when the KV cache pool is
        # exhausted and no host cache tier is available for suspend/resume.
        if not scheduled_gen and not scheduled_ctx:
            num_gen_candidates = sum(
                1
                for r in active_requests
                if r.is_generation_in_progress_state
                and not r.is_generation_to_complete_state
                and r.request_id not in inflight_request_ids
            )
            if num_gen_candidates > 0 and not evicted:
                raise RuntimeError(
                    f"V2 scheduler deadlock: {num_gen_candidates} generation "
                    f"request(s) active but none could be scheduled or "
                    f"evicted. KV cache pool is likely exhausted with no "
                    f"host cache tier for suspend/resume offload. "
                    f"Configure kv_cache_config.host_cache_size or increase "
                    f"kv_cache_config.max_tokens."
                )

        return (
            scheduled_encoder,
            scheduled_ctx,
            scheduled_gen,
            evicted,
            disagg_candidates,
            has_chunking,
        )

    # ---- Per-type scheduling methods ----

    def _try_schedule_encoder(
        self, req: LlmRequest, budget: BudgetTracker
    ) -> tuple[ScheduleAction, int]:
        """Try to schedule an encoder request.

        Encoder admission does not need KV blocks for decoder work. Decoder
        context is the first step that reserves self- and cross-KV blocks and
        writes K/V projections into the cross cache.

        Returns ``(action, tokens)`` where *tokens* is meaningful only
        when *action* is ``SCHEDULED``.
        """
        # Encoder-decoder runtime requires a cross pool for the later decoder
        # context step. If the runtime did not plumb one through, surface this
        # loudly rather than silently routing to the self pool, which would
        # corrupt the dual-pool contract.
        if self.cross_kv_cache_manager is None:
            raise RuntimeError(
                f"Encoder-init request {req.py_request_id} requires a cross_kv_cache_manager."
            )

        req_tokens = req.encoder_output_len
        if not budget.can_fit_tokens(req_tokens):
            return ScheduleAction.STOP, 0
        assert self.max_context_length is None or req_tokens <= self.max_context_length, (
            f"The number of encoder tokens ({req_tokens}) exceeds the limit value ({self.max_context_length})"
        )
        return ScheduleAction.SCHEDULED, req_tokens

    def _try_schedule_disagg_gen_init(
        self, req: LlmRequest, budget: BudgetTracker
    ) -> tuple[ScheduleAction, int]:
        """Try to schedule a disagg generation init request.

        Disagg gen init requests bypass normal state gating but still need
        KV cache allocation inline (V2 prepare_resources is a no-op for
        the primary manager).  Aligned with C++ CapacityScheduler which
        treats disagg_gen_init identically to context_init for block/PEFT/
        maxNumRequests accounting.

        Returns ``(action, tokens)``.  *tokens* is 0 because disagg requests
        don't participate in the forward pass token budget.
        """
        if not self.kv_cache_manager.prepare_context(req):
            logger.debug("prepare_context failed for disagg gen init request %s", req.py_request_id)
            return ScheduleAction.SKIP, 0

        if not self.kv_cache_manager.resize_context(
            req, req.context_remaining_length + get_draft_token_length(req)
        ):
            return ScheduleAction.SKIP, 0

        return ScheduleAction.SCHEDULED, 0

    def _try_schedule_context(
        self, req: LlmRequest, budget: BudgetTracker
    ) -> tuple[ScheduleAction, int, bool]:
        """Try to schedule a context request (chunked or non-chunked).

        Returns ``(action, tokens, chunking_flag)``.  *tokens* and
        *chunking_flag* are meaningful only when *action* is ``SCHEDULED``.
        """
        if self.chunking_enabled:
            return self._try_schedule_context_chunked(req, budget)
        return self._try_schedule_context_full(req, budget)

    def _try_schedule_context_full(
        self, req: LlmRequest, budget: BudgetTracker
    ) -> tuple[ScheduleAction, int, bool]:
        """Try to schedule a non-chunked context request.

        Returns ``(action, tokens, chunking_flag)``.
        """
        # Prepare first so block reuse updates context_remaining_length
        # before budget check.
        if not self.kv_cache_manager.prepare_context(req):
            logger.debug(f"prepare_context failed for context request {req.py_request_id}")
            return ScheduleAction.STOP, 0, False

        context_tokens = req.context_remaining_length
        draft_len = get_draft_token_length(req)
        req_tokens = context_tokens + draft_len

        if not budget.can_fit_tokens(req_tokens):
            return ScheduleAction.STOP, 0, False

        assert self.max_context_length is None or context_tokens <= self.max_context_length, (
            f"Context tokens ({context_tokens}) exceeds limit ({self.max_context_length})"
        )

        # V2 resizes KV cache directly in the scheduler (no separate
        # prepareResources for main cache), so include draft tokens.
        if not self.kv_cache_manager.resize_context(req, req_tokens):
            return ScheduleAction.SKIP, 0, False

        cross_action = self._try_schedule_cross_context(req)
        if cross_action is not ScheduleAction.SCHEDULED:
            self._suspend_request(req)
            return cross_action, 0, False

        return ScheduleAction.SCHEDULED, req_tokens, False

    def _try_schedule_context_chunked(
        self, req: LlmRequest, budget: BudgetTracker
    ) -> tuple[ScheduleAction, int, bool]:
        """FCFS interleaved chunking for a single context request.

        Returns ``(action, chunk_tokens, chunking_flag)``.

        Chunking uses implicit skip on failure (not break): a ctx that can't
        be chunked small enough is skipped, allowing subsequent gen requests
        (needing only ~1 token) to still be scheduled.
        """
        remaining_budget = budget.remaining_tokens

        if remaining_budget is not None and remaining_budget <= 0:
            return ScheduleAction.SKIP, 0, False

        # Prepare context (create _KVCache, block reuse, resume — no resize)
        if not self.kv_cache_manager.prepare_context(req):
            logger.debug(f"prepare_context failed for chunked context request {req.py_request_id}")
            return ScheduleAction.SKIP, 0, False

        # Calculate chunk size from remaining budget
        #    (context_remaining_length is now correct after block reuse)
        context_remaining = req.context_remaining_length
        force_chunk = self._is_force_chunking_policy()
        if force_chunk:
            chunk_size = self._force_chunk_size(req, context_remaining)
        else:
            # Min budget check — FCFS chunking needs at least one chunk unit.
            if remaining_budget is not None and remaining_budget < self.chunk_unit_size:
                return ScheduleAction.SKIP, 0, False
            chunk_size = (
                min(remaining_budget, context_remaining)
                if remaining_budget is not None
                else context_remaining
            )

        if self.max_context_length is not None:
            chunk_size = min(chunk_size, self.max_context_length)

        if remaining_budget is not None and chunk_size > remaining_budget:
            chunk_size = remaining_budget

        # Round down to chunk_unit_size boundary when we are not stopping at
        # an exact forced point or the prompt end.
        if chunk_size < context_remaining and not (
            force_chunk and self._is_forced_chunk_boundary(req, chunk_size)
        ):
            chunk_size = (chunk_size // self.chunk_unit_size) * self.chunk_unit_size

        if chunk_size <= 0:
            # TODO: consider suspending first-chunk KVCache to release
            # GPU pages. Currently we skip without suspend to avoid
            # pathological suspend/resume cycles. suspend_request is
            # only called from eviction (_try_evict_for_gen).
            return ScheduleAction.SKIP, 0, False

        chunk_size = self._align_chunk_to_mm_block(
            req, chunk_size, remaining_budget, context_remaining
        )
        # Alignment returns 0 when neither snap-up nor snap-down can preserve
        # bidirectional MM attention this iteration — defer the request rather
        # than schedule a chunk that would corrupt attention. Next iteration
        # gets a fresh budget; under steady-state load this resolves quickly.
        if chunk_size <= 0:
            return ScheduleAction.SKIP, 0, False

        req.context_chunk_size = chunk_size

        # Draft tokens only matter for last chunk (budget + resize)
        chunk_tokens = chunk_size
        if req.is_last_context_chunk:
            draft_len = get_draft_token_length(req)
            if draft_len > 0:
                chunk_tokens += draft_len

        # V2 resizes KV cache directly in the scheduler, so include
        # draft tokens for last chunk.
        if not self.kv_cache_manager.resize_context(req, chunk_tokens):
            return ScheduleAction.SKIP, 0, False

        cross_action = self._try_schedule_cross_context(req)
        if cross_action is not ScheduleAction.SCHEDULED:
            self._suspend_request(req)
            return cross_action, 0, False

        chunking_flag = req.context_chunk_size < req.context_remaining_length

        return ScheduleAction.SCHEDULED, chunk_tokens, chunking_flag

    def _is_force_chunking_policy(self) -> bool:
        policy = self.chunking_policy
        if policy is None:
            return False
        return (
            getattr(policy, "name", None) == "FORCE_CHUNK"
            or getattr(policy, "value", None) == "FORCE_CHUNK"
            or str(policy).endswith("FORCE_CHUNK")
        )

    def _force_chunk_size(self, req: LlmRequest, context_remaining: int) -> int:
        points = getattr(req, "expect_chunking_points", None)
        if not isinstance(points, (list, tuple)) or not points:
            raise RuntimeError(
                "FORCE_CHUNK requires request.expect_chunking_points to be "
                f"set before scheduling request {req.py_request_id}"
            )

        current = req.context_current_position
        next_point = min((point for point in points if point > current), default=None)
        if next_point is None:
            return context_remaining

        next_position = min(next_point, req.prompt_len)
        return max(0, next_position - current)

    def _is_forced_chunk_boundary(self, req: LlmRequest, chunk_size: int) -> bool:
        next_position = req.context_current_position + chunk_size
        if next_position >= req.prompt_len:
            return True
        points = getattr(req, "expect_chunking_points", None)
        return bool(isinstance(points, (list, tuple)) and next_position in points)

    def _align_chunk_to_mm_block(
        self,
        req: LlmRequest,
        chunk_size: int,
        remaining_budget: Optional[int],
        context_remaining: int,
    ) -> int:
        """Adjust *chunk_size* so a boundary never lands inside a multimodal
        soft-token run that requires intact bidirectional attention.

        Returns the adjusted chunk size, or ``0`` to signal that the caller
        should SKIP this request this iteration (defer to next iteration with
        a fresh budget).

        Two-pronged strategy:
          1. snap-UP: extend the chunk to swallow the rest of the block
             (preferred — block lands wholly in this iteration).
          2. snap-DOWN fallback: if budget / max_context_length / context_remaining
             can't absorb snap-up, shrink the chunk to end *before* the block
             starts; the block lands wholly in the next iteration.

        Last-resort defer (return 0): if the block starts at the chunk's
        left boundary (snap-down would zero the chunk anyway), defer the
        request. We never let a bidirectional MM block straddle a chunk
        boundary — it would silently break attention correctness.

        Raises ``ValueError`` if the block itself exceeds max_context_length —
        no chunk size can fit it, deferring would livelock; surface the
        config error loudly rather than spin silently.

        Gated on ``mm_bidirectional_blocks`` written by the input processor —
        causal-MM models (Llava, Qwen-VL) skip this entirely.
        """
        # `getattr(..., None) or {}` is not enough: on a Mock req the attribute
        # exists as a Mock (truthy) and `.get` returns another Mock (also
        # truthy). Require an actual dict — that's also what the input
        # processor writes in inputs/registry.py.
        mm_data = getattr(req, "py_multimodal_data", None)
        if not isinstance(mm_data, dict):
            return chunk_size
        if not mm_data.get("mm_bidirectional_blocks", False):
            return chunk_size
        cumsum = mm_data.get("multimodal_embed_mask_cumsum")
        if cumsum is None:
            return chunk_size

        unit_size = self.chunk_unit_size
        lo = req.context_current_position
        end_abs = lo + chunk_size
        prompt_len = lo + context_remaining
        if end_abs >= prompt_len or end_abs <= 0:
            return chunk_size

        # Cumsum is INCLUSIVE [P] (length prompt_len) with
        # cumsum[i] = sum(mask[0..i]). So mask[i] = cumsum[i]-cumsum[i-1] for
        # i>=1, and mask[0] = cumsum[0]. Aligned with inputs/registry.py
        # (``embed_mask.cumsum(0)``) and MultimodalRuntimeData.
        #
        # Three-cumsum boundary-in-block detection: positions end_abs-1 and
        # end_abs are both MM iff mask[end_abs-1]==1 AND mask[end_abs]==1.
        # Gemma4's HF processor wraps every soft-token run with non-MM
        # specials (boi/eoi/boa/eoa, embed_mask=0), so two consecutive
        # embed_mask=1 positions always belong to the same block.
        cs_prev = int(cumsum[end_abs - 1].item())
        cs_cur = int(cumsum[end_abs].item())
        if (cs_cur - cs_prev) != 1:
            return chunk_size  # mask[end_abs] != 1
        if end_abs == 1:
            if cs_prev != 1:
                return chunk_size  # mask[0] != 1
        else:
            if (cs_prev - int(cumsum[end_abs - 2].item())) != 1:
                return chunk_size  # mask[end_abs - 1] != 1

        # --- compute block extent (forward + backward walks from end_abs) ---
        # Forward walk: advance while position block_end_abs is MM
        # (mask[block_end_abs] == cumsum[block_end_abs]-cumsum[block_end_abs-1]).
        # Final block_end_abs is the first non-MM position (exclusive end),
        # or prompt_len if the block extends through EOS.
        # Backward walk uses ``> 0`` (not ``> lo``) so block_size below
        # reflects the TRUE block size — important for the impossibility
        # check, which must not be undercounted if a prior iteration somehow
        # advanced into a block (would otherwise mask a real config error).
        block_end_abs = end_abs
        while block_end_abs < prompt_len:
            if (int(cumsum[block_end_abs].item()) - int(cumsum[block_end_abs - 1].item())) != 1:
                break
            block_end_abs += 1

        # Backward walk: decrement while position (block_start_abs - 1) is MM.
        # Final block_start_abs is the first MM position (inclusive start),
        # or 0 if the block extends to BOS.
        block_start_abs = end_abs
        while block_start_abs > 1:
            if (
                int(cumsum[block_start_abs - 1].item()) - int(cumsum[block_start_abs - 2].item())
            ) != 1:
                break
            block_start_abs -= 1
        # Position 0: mask[0] == cumsum[0]; no cumsum[-1] sentinel.
        if block_start_abs == 1 and int(cumsum[0].item()) == 1:
            block_start_abs = 0

        # --- impossibility check ---
        # If the block itself exceeds max_context_length, no chunk size can
        # ever fit it. Deferring would livelock the request. Raise a clear
        # config error rather than letting the scheduler thrash silently.
        block_size = block_end_abs - block_start_abs
        if self.max_context_length is not None and block_size > self.max_context_length:
            raise ValueError(
                f"req {req.py_request_id}: bidirectional multimodal block of "
                f"{block_size} tokens exceeds max_context_length="
                f"{self.max_context_length}. The block must fit in a single "
                f"chunk to preserve bidirectional attention; deferring would "
                f"livelock. Increase max_num_tokens to at least {block_size}."
            )

        # --- snap-up target ---
        # Round up to unit_size so setPrepopulatedPromptLen's downstream
        # floor() doesn't clip the extension back into the block.
        if unit_size > 0 and block_end_abs < prompt_len:
            up_block_end = ((block_end_abs + unit_size - 1) // unit_size) * unit_size
        else:
            up_block_end = block_end_abs
        up_block_end = min(up_block_end, prompt_len)
        up_chunk_size = up_block_end - lo

        up_fits_budget = remaining_budget is None or up_chunk_size <= remaining_budget
        up_fits_max_ctx = (
            self.max_context_length is None or up_chunk_size <= self.max_context_length
        )
        up_fits_remaining = up_chunk_size <= context_remaining

        if up_fits_budget and up_fits_max_ctx and up_fits_remaining and up_chunk_size > chunk_size:
            return up_chunk_size

        # --- snap-down fallback ---
        # Round down to unit_size so the next chunk starts on a kv-block boundary.
        if unit_size > 0:
            down_block_start = (block_start_abs // unit_size) * unit_size
        else:
            down_block_start = block_start_abs

        if down_block_start <= lo:
            logger.warning(
                "req %s: MM block at chunk left edge "
                "(lo=%s, block=[%s, %s)); snap-up does not fit "
                "(up_size=%s, budget=%s, max_ctx=%s, ctx_rem=%s) and "
                "snap-down would zero the chunk. Deferring request to next "
                "iteration to preserve bidirectional MM attention.",
                req.py_request_id,
                lo,
                block_start_abs,
                block_end_abs,
                up_chunk_size,
                remaining_budget,
                self.max_context_length,
                context_remaining,
            )
            return 0

        return down_block_start - lo

    @staticmethod
    def _get_optional_encoder_output_len(req: LlmRequest) -> Optional[int]:
        get_encoder_output_len = getattr(type(req), "try_get_encoder_output_len", None)
        encoder_output_len = (
            get_encoder_output_len(req)
            if get_encoder_output_len is not None
            else getattr(req, "encoder_output_len", None)
        )
        if encoder_output_len is None:
            return None
        return int(encoder_output_len)

    @classmethod
    def _needs_cross_context_allocation(cls, req: LlmRequest) -> bool:
        """Return whether decoder context must reserve cross-KV for *req*."""
        if cls._get_optional_encoder_output_len(req) is None:
            return False
        skip_projection = getattr(req, "py_skip_cross_kv_projection", False)
        return not (isinstance(skip_projection, bool) and skip_projection)

    def _try_schedule_cross_context(self, req: LlmRequest) -> ScheduleAction:
        """Reserve cross-KV blocks for the first decoder context step."""
        if not self._needs_cross_context_allocation(req):
            return ScheduleAction.SCHEDULED

        if self.cross_kv_cache_manager is None:
            logger.warning(
                "Decoder context request %s requires cross-KV cache but "
                "no cross_kv_cache_manager is configured. Skipping.",
                req.py_request_id,
            )
            return ScheduleAction.STOP

        req_tokens = self._get_optional_encoder_output_len(req)
        if req_tokens is None:
            return ScheduleAction.SCHEDULED
        from ..kv_cache_manager_v2 import KVCacheManagerV2

        if isinstance(self.cross_kv_cache_manager, KVCacheManagerV2):
            if not self._try_schedule_cross_context_v2(
                self.cross_kv_cache_manager, req, req_tokens
            ):
                return ScheduleAction.SKIP
            return ScheduleAction.SCHEDULED

        if not self.cross_kv_cache_manager.prepare_context(req):
            logger.debug(
                "cross prepare_context failed for decoder context request %s",
                req.py_request_id,
            )
            return ScheduleAction.SKIP
        if not self.cross_kv_cache_manager.resize_context(req, req_tokens):
            return ScheduleAction.SKIP
        return ScheduleAction.SCHEDULED

    @staticmethod
    def _try_schedule_cross_context_v2(
        cross_kv_cache_manager, req: LlmRequest, req_tokens: int
    ) -> bool:
        """Reserve V2 cross-KV without mutating decoder context position."""
        kv_cache = cross_kv_cache_manager.kv_cache_map.get(req.py_request_id)
        if kv_cache is None:
            if not req.is_first_context_chunk:
                logger.debug(
                    "cross KV cache missing for non-first context chunk, request %s",
                    req.py_request_id,
                )
                return False
            input_tokens = (
                req.get_encoder_unique_tokens()
                if cross_kv_cache_manager.enable_block_reuse
                else None
            )
            kv_cache = cross_kv_cache_manager._create_kv_cache(
                req.py_request_id, req.lora_task_id, input_tokens
            )
            kv_cache.cuda_stream = cross_kv_cache_manager._stream.cuda_stream

        if not cross_kv_cache_manager.enable_block_reuse:
            kv_cache.stop_committing()

        if not cross_kv_cache_manager._resume_and_restore(req.py_request_id, kv_cache):
            return False

        target_capacity = req_tokens + cross_kv_cache_manager.num_extra_kv_tokens
        if not kv_cache.resize(max(kv_cache.capacity, target_capacity)):
            if req.is_first_context_chunk:
                kv_cache.suspend()
            return False

        return True

    def _try_schedule_generation(
        self,
        req: LlmRequest,
        budget: BudgetTracker,
        requests_list: list,
        req_it: int,
        req_it_end: int,
        evicted: RequestList,
        scheduled_beam_width: int,
    ) -> tuple[ScheduleAction, int, int, int]:
        """Try to schedule a generation request.

        Returns ``(action, tokens, scheduled_beam_width, req_it_end)``.
        *tokens* is meaningful only when *action* is ``SCHEDULED``.
        """
        beam_width = req.get_beam_width_by_iter(for_next_iteration=False)
        req_tokens = beam_width + get_draft_token_length(req)

        if not budget.can_fit_tokens(req_tokens):
            return ScheduleAction.STOP, 0, scheduled_beam_width, req_it_end

        if scheduled_beam_width == 0:
            scheduled_beam_width = beam_width
        elif scheduled_beam_width != beam_width:
            return ScheduleAction.SKIP, 0, scheduled_beam_width, req_it_end

        success = self.kv_cache_manager.try_allocate_generation(req)

        if not success:
            req_it_end, success = self._try_evict_for_gen(
                req, requests_list, req_it, req_it_end, evicted
            )

        if success:
            return ScheduleAction.SCHEDULED, req_tokens, scheduled_beam_width, req_it_end

        # Self-eviction: suspend this gen request to free its
        # GPU pages so other requests can resume().
        # Skip if already suspended — suspending again is a no-op
        # that frees no pages.
        if self.kv_cache_manager.is_request_active(req.py_request_id):
            logger.debug(
                f"[V2Scheduler] Self-evicting request {req.py_request_id} "
                f"(state={req.state.name}) to free GPU pages"
            )
            self._suspend_request(req)
            evicted.append(req)

        return ScheduleAction.STOP, 0, scheduled_beam_width, req_it_end

    # ---- Eviction ----

    @staticmethod
    def _is_started_request(req: LlmRequest) -> bool:
        """A request that has begun execution and can be paused.
        Matches V1: (context_init && !first_chunk) || generation_in_progress.
        """
        return (
            req.is_context_init_state and not req.is_first_context_chunk
        ) or req.is_generation_in_progress_state

    def _suspend_request(self, req: LlmRequest) -> None:
        """Suspend a request's KV cache in both main and draft managers.

        TODO: Also release PEFT resources (mark_request_done) for the
        suspended request so the C++ PeftCacheManager can evict its
        adapter pages.  Currently only KV cache is freed; the adapter
        remains "active" on device, which could cause ensure_batch to
        fail if it needs to load a different adapter into a full cache.
        """
        self._clear_request_runtime_state(req)
        self.kv_cache_manager.suspend_request(req)
        if self.draft_kv_cache_manager is not None:
            self.draft_kv_cache_manager.suspend_request(req)

    def _clear_request_runtime_state(self, req: LlmRequest) -> None:
        req.py_batch_idx = None

    def _is_evictable(self, req: LlmRequest) -> bool:
        """A started request whose KV cache is still active on GPU.

        Already-suspended requests are not useful eviction victims
        because suspending them again is a no-op that frees no pages.
        """
        if not self._is_started_request(req):
            return False
        return self.kv_cache_manager.is_request_active(req.py_request_id)

    def _try_evict_for_gen(self, req, requests_list, req_it, req_it_end, evicted):
        """Evict started requests from active_requests tail to make room.

        Search backwards from req_it_end
        for evictable requests (started AND KV cache active on GPU),
        suspend them to free pages, then retry allocation.

        Victims are always at indices >= req_it (not yet processed by the
        main loop), so they are never in scheduled_ctx/scheduled_gen and
        no token budget reclaim is needed.

        Returns (new_req_it_end, success) tuple. new_req_it_end is always
        updated to reflect evicted victims (even on failure) so the caller
        can skip already-evicted requests.
        """
        while req_it_end > req_it:
            victim_idx = None
            for i in range(req_it_end - 1, req_it, -1):
                if self._is_evictable(requests_list[i]):
                    victim_idx = i
                    break

            if victim_idx is None:
                break

            victim = requests_list[victim_idx]
            logger.debug(
                f"[V2Scheduler] Evicting request {victim.py_request_id} "
                f"(state={victim.state.name}) to free pages for request {req.py_request_id}"
            )
            self._suspend_request(victim)
            evicted.append(victim)
            req_it_end = victim_idx

            if self.kv_cache_manager.try_allocate_generation(req):
                return req_it_end, True

        return req_it_end, False

    # ---- Sorting ----

    @staticmethod
    def _lora_key(req: LlmRequest):
        return _get_lora_task_id(req)

    def _sort_requests(self, context_requests, generation_requests, has_chunks):
        """Sort by LoRA task ID. Non-last chunks before last chunks."""
        if has_chunks:
            not_last = [r for r in context_requests if not r.is_last_context_chunk]
            last = [r for r in context_requests if r.is_last_context_chunk]
            not_last.sort(key=self._lora_key)
            last.sort(key=self._lora_key)
            context_requests.clear()
            context_requests.extend(not_last)
            context_requests.extend(last)
        else:
            context_requests.sort(key=self._lora_key)
        generation_requests.sort(key=self._lora_key)

    # ---- can_schedule (PP dry-run) ----

    def can_schedule(self, requests: RequestList) -> bool:
        """Conservative heuristic for PP retry loop. Does NOT allocate.

        TODO: V2's try-and-see model lacks a free-blocks query API.
        Implementing this requires exposing storage statistics from the
        V2 runtime (e.g., free GPU slots per pool group). For now,
        always returns True — PP is not yet supported (asserted in
        KVCacheManagerV2 constructor via kv_connector_manager=None).
        """
        return True
