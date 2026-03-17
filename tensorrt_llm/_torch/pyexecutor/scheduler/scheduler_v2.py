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

from typing import Optional

from tensorrt_llm.llmapi.llm_args import CapacitySchedulerPolicy
from tensorrt_llm.logger import logger

from ..llm_request import LlmRequest, LlmRequestState
from .scheduler import RequestList, RequestScheduler, SchedulerOutput


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

        # PEFT config (constant after construction)
        self._max_peft_pages = (
            peft_cache_manager.max_device_pages if peft_cache_manager is not None else 0
        )

        # PEFT accounting — reset at the start of each scheduling iteration
        self._claimed_peft_pages = 0
        self._seen_peft_task_ids: set[int] = set()

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

    def _schedule_loop(self, active_requests, inflight_request_ids):
        scheduled_ctx: RequestList = []
        scheduled_gen: RequestList = []
        evicted: RequestList = []
        disagg_candidates: RequestList = []
        batch_num_tokens = 0
        max_num_tokens = self.max_num_tokens
        num_scheduled = 0
        scheduled_beam_width = 0
        has_chunking = False
        # Reset PEFT accounting for this iteration
        self._claimed_peft_pages = 0
        self._seen_peft_task_ids.clear()
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

        while req_it < req_it_end:
            if num_scheduled >= self.max_num_requests:
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

            scheduled = False

            peft_pages = self._check_peft(req)
            if peft_pages < 0:
                break

            # --- Encoder ---
            if req_state_value == self._encoder_init_state_value:
                req_tokens = req.encoder_output_len
                if max_num_tokens is not None and (batch_num_tokens + req_tokens > max_num_tokens):
                    break
                assert self.max_context_length is None or req_tokens <= self.max_context_length, (
                    f"The number of encoder tokens ({req_tokens}) exceeds the limit value ({self.max_context_length})"
                )
                if not self.kv_cache_manager.prepare_context(req):
                    break
                if self.kv_cache_manager.resize_context(req, req_tokens):
                    self._commit_peft(req, peft_pages)
                    scheduled_ctx.append(req)
                    batch_num_tokens += req_tokens
                    num_scheduled += 1
                    scheduled = True
                else:
                    break

            # --- Context ---
            elif req_state_value == self._context_init_state_value:
                if self.chunking_enabled:
                    # Chunking uses implicit continue on failure (not break):
                    # a ctx that can't be chunked small enough is skipped,
                    # allowing subsequent gen requests (needing only ~1 token)
                    # to still be scheduled.
                    prev_len = len(scheduled_ctx)
                    self._schedule_context_chunked(
                        req, scheduled_ctx, batch_num_tokens, max_num_tokens, peft_pages
                    )
                    if len(scheduled_ctx) > prev_len:
                        chunk_tokens = req.context_chunk_size
                        if req.is_last_context_chunk and req.has_draft_tokens:
                            chunk_tokens += req.num_draft_tokens
                        has_chunking = has_chunking or (
                            req.context_chunk_size < req.context_remaining_length
                        )
                        batch_num_tokens += chunk_tokens
                        num_scheduled += 1
                        scheduled = True
                else:
                    # Non-chunking: prepare first so block reuse updates
                    # context_remaining_length before budget check.
                    if not self.kv_cache_manager.prepare_context(req):
                        break

                    context_tokens = req.context_remaining_length

                    if max_num_tokens is not None and (
                        batch_num_tokens + context_tokens > max_num_tokens
                    ):
                        break

                    assert (
                        self.max_context_length is None or context_tokens <= self.max_context_length
                    ), (
                        f"Context tokens ({context_tokens}) exceeds limit ({self.max_context_length})"
                    )
                    if self.kv_cache_manager.resize_context(req, context_tokens):
                        self._commit_peft(req, peft_pages)
                        req_tokens = context_tokens
                        if req.has_draft_tokens:
                            req.context_chunk_size = context_tokens
                            budget_remaining = (
                                (max_num_tokens - batch_num_tokens - context_tokens)
                                if max_num_tokens is not None
                                else None
                            )
                            self._fit_draft_tokens_single(req, budget_remaining)
                            req_tokens += req.num_draft_tokens
                        scheduled_ctx.append(req)
                        batch_num_tokens += req_tokens
                        num_scheduled += 1
                        scheduled = True

            # --- Generation ---
            else:
                beam_width = req.get_beam_width_by_iter(for_next_iteration=False)
                req_tokens = beam_width + req.num_draft_tokens

                if max_num_tokens is not None and (batch_num_tokens + req_tokens > max_num_tokens):
                    break

                if scheduled_beam_width == 0:
                    scheduled_beam_width = beam_width
                elif scheduled_beam_width != beam_width:
                    req_it += 1
                    continue

                success = self.kv_cache_manager.try_allocate_generation(req)

                if not success:
                    req_it_end, success = self._try_evict_for_gen(
                        req, requests_list, req_it, req_it_end, evicted
                    )

                if success:
                    self._commit_peft(req, peft_pages)
                    scheduled_gen.append(req)
                    batch_num_tokens += req_tokens
                    num_scheduled += 1
                    scheduled = True
                else:
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

            if scheduled:
                req_it += 1
            elif req_state_value == self._context_init_state_value:
                # Context failure: skip and try next request — a
                # smaller ctx behind this one may still fit.
                req_it += 1
            else:
                # Gen/encoder failure (including self-eviction): stop.
                break

        return scheduled_ctx, scheduled_gen, evicted, disagg_candidates, has_chunking

    def _schedule_context_chunked(
        self, req, scheduled_ctx, batch_num_tokens, max_num_tokens, peft_pages
    ):
        """FCFS interleaved chunking for a single context request."""
        remaining_budget = (
            (max_num_tokens - batch_num_tokens) if max_num_tokens is not None else None
        )

        # 1. Min budget check — need at least one chunk unit
        if remaining_budget is not None and remaining_budget < self.chunk_unit_size:
            return

        # 2. Prepare context (create _KVCache, block reuse, resume — no resize)
        if not self.kv_cache_manager.prepare_context(req):
            return

        # 3. Calculate chunk size from remaining budget
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
            return

        # 5. Resize to chunk size
        if not self.kv_cache_manager.resize_context(req, chunk_size):
            return

        req.context_chunk_size = chunk_size

        # Draft tokens for last chunk
        if req.is_last_context_chunk and req.has_draft_tokens:
            budget_after_chunk = (
                (remaining_budget - chunk_size) if remaining_budget is not None else None
            )
            self._fit_draft_tokens_single(req, budget_after_chunk)

        self._commit_peft(req, peft_pages)
        scheduled_ctx.append(req)

    # ---- PEFT helpers ----

    def _check_peft(self, req: LlmRequest) -> int:
        """Check PEFT pages. Returns required pages (>= 0) if fits, -1 if not."""
        if self.peft_cache_manager is None:
            return 0
        lora_task_id = getattr(req, "lora_task_id", None)
        if lora_task_id is None or lora_task_id in self._seen_peft_task_ids:
            return 0
        required = self.peft_cache_manager.determine_num_pages(req)
        if self._claimed_peft_pages + required > self._max_peft_pages:
            return -1
        return required

    def _commit_peft(self, req: LlmRequest, peft_pages: int) -> None:
        """Commit PEFT page accounting after successful scheduling."""
        if peft_pages > 0:
            lora_task_id = getattr(req, "lora_task_id", None)
            self._claimed_peft_pages += peft_pages
            self._seen_peft_task_ids.add(lora_task_id)

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
        """Suspend a request's KV cache in both main and draft managers."""
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

    # ---- Draft tokens ----

    def _fit_draft_tokens_single(self, req, budget_after_chunk):
        """Fit draft tokens into the last allocated page's remaining space."""
        chunk_size = req.context_chunk_size
        # Use tokens_per_block (not _chunk_align_unit): draft tokens should
        # fit within already-allocated pages to avoid extra page allocation.
        remainder = chunk_size % self.tokens_per_block
        remaining_space = 0 if remainder == 0 else self.tokens_per_block - remainder

        if self.max_context_length is not None:
            remaining_context_len = self.max_context_length - chunk_size
            remaining_space = min(remaining_space, remaining_context_len)

        if budget_after_chunk is not None:
            remaining_space = min(remaining_space, budget_after_chunk)

        draft_discard = req.num_draft_tokens - min(req.num_draft_tokens, remaining_space)
        if draft_discard > 0 and hasattr(req, "discard_draft_tokens"):
            req.discard_draft_tokens(draft_discard)

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
