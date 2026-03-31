# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from .scheduler import RequestList, RequestScheduler, SchedulerOutput


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
    ):
        self.max_num_tokens = max_num_tokens
        self.max_num_requests = (
            scheduler_capacity if scheduler_capacity is not None else max_batch_size
        )
        from ..resource_manager import KVCacheManagerV2

        assert isinstance(kv_cache_manager, KVCacheManagerV2), (
            f"KVCacheV2Scheduler requires KVCacheManagerV2, got {type(kv_cache_manager).__name__}"
        )
        self.kv_cache_manager = kv_cache_manager
        self.draft_kv_cache_manager = draft_kv_cache_manager
        if scheduler_policy != CapacitySchedulerPolicy.MAX_UTILIZATION:
            logger.warning(
                "KVCacheV2Scheduler only supports MAX_UTILIZATION for now, got %s, setting to MAX_UTILIZATION",
                scheduler_policy,
            )
        self.policy = scheduler_policy
        self.peft_cache_manager = peft_cache_manager

        # Chunking config — only FCFS supported
        self.chunking_enabled = False
        self.chunk_unit_size = 0
        self.max_context_length = max_num_tokens
        self.tokens_per_block = kv_cache_manager.tokens_per_block
        logger.info(
            "KVCacheV2Scheduler: tokens_per_block=%d, max_num_tokens=%s, max_batch_size=%s, draft_mgr=%s",
            self.tokens_per_block,
            max_num_tokens,
            max_batch_size,
            type(draft_kv_cache_manager).__name__ if draft_kv_cache_manager is not None else "None",
        )
        if ctx_chunk_config is not None:
            self.chunking_enabled = True
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
        # Main scheduling loop
        (scheduled_ctx, scheduled_gen, evicted, disagg_candidates, has_chunking) = (
            self._schedule_loop(active_requests, inflight_request_ids)
        )

        # Sort by LoRA task ID
        self._sort_requests(scheduled_ctx, scheduled_gen, has_chunking)

        return SchedulerOutput(
            context_requests=scheduled_ctx,
            generation_requests=scheduled_gen,
            paused_requests=evicted,
            fitting_disagg_gen_init_requests=disagg_candidates,
            num_fitting_requests=len(scheduled_ctx) + len(scheduled_gen),
        )

    # ---- Main scheduling loop ----

    def _schedule_loop(self, active_requests, inflight_request_ids):
        scheduled_ctx: RequestList = []
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
            if budget.requests_full:
                break

            req = requests_list[req_it]

            # --- Filter ---
            if req.request_id in inflight_request_ids:
                req_it += 1
                continue

            req_state_value = req.state_value

            # Disagg gen init bypasses normal state gating (same as C++ / V1 scheduler)
            if req_state_value == self._disagg_gen_init_state_value:
                disagg_candidates.append(req)
                req_it += 1
                continue

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
                scheduled_ctx.append(req)
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

        return scheduled_ctx, scheduled_gen, evicted, disagg_candidates, has_chunking

    # ---- Per-type scheduling methods ----

    def _try_schedule_encoder(
        self, req: LlmRequest, budget: BudgetTracker
    ) -> tuple[ScheduleAction, int]:
        """Try to schedule an encoder request.

        Returns ``(action, tokens)`` where *tokens* is meaningful only
        when *action* is ``SCHEDULED``.
        """
        req_tokens = req.encoder_output_len
        if not budget.can_fit_tokens(req_tokens):
            return ScheduleAction.STOP, 0
        assert self.max_context_length is None or req_tokens <= self.max_context_length, (
            f"The number of encoder tokens ({req_tokens}) exceeds the limit value ({self.max_context_length})"
        )
        if not self.kv_cache_manager.prepare_context(req):
            logger.debug("prepare_context failed for encoder request %s", req.py_request_id)
            return ScheduleAction.STOP, 0
        if not self.kv_cache_manager.resize_context(req, req_tokens):
            return ScheduleAction.STOP, 0
        return ScheduleAction.SCHEDULED, req_tokens

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
            logger.debug("prepare_context failed for context request %s", req.py_request_id)
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

        # Min budget check — need at least one chunk unit
        if remaining_budget is not None and remaining_budget < self.chunk_unit_size:
            return ScheduleAction.SKIP, 0, False

        # Prepare context (create _KVCache, block reuse, resume — no resize)
        if not self.kv_cache_manager.prepare_context(req):
            logger.debug("prepare_context failed for chunked context request %s", req.py_request_id)
            return ScheduleAction.SKIP, 0, False

        # Calculate chunk size from remaining budget
        #    (context_remaining_length is now correct after block reuse)
        context_remaining = req.context_remaining_length
        chunk_size = (
            min(remaining_budget, context_remaining)
            if remaining_budget is not None
            else context_remaining
        )

        if self.max_context_length is not None:
            chunk_size = min(chunk_size, self.max_context_length)

        # Round down to chunk_unit_size boundary (unless last chunk).
        if chunk_size < context_remaining:
            chunk_size = (chunk_size // self.chunk_unit_size) * self.chunk_unit_size

        if chunk_size <= 0:
            # TODO: consider suspending first-chunk KVCache to release
            # GPU pages. Currently we skip without suspend to avoid
            # pathological suspend/resume cycles. suspend_request is
            # only called from eviction (_try_evict_for_gen).
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
        chunking_flag = req.context_chunk_size < req.context_remaining_length

        return ScheduleAction.SCHEDULED, chunk_tokens, chunking_flag

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
                "[V2Scheduler] Self-evicting request %s (state=%s) to free GPU pages",
                req.py_request_id,
                req.state,
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
        self.kv_cache_manager.suspend_request(req)
        if self.draft_kv_cache_manager is not None:
            self.draft_kv_cache_manager.suspend_request(req)

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
                "[V2Scheduler] Evicting request %s (state=%s) to free pages for request %s",
                victim.py_request_id,
                victim.state,
                req.py_request_id,
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
        lora_id = getattr(req, "lora_task_id", None)
        if lora_id is None:
            return (0, 0)
        return (1, lora_id)

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
