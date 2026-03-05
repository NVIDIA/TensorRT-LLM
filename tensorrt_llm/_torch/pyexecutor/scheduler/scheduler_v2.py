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

from math import gcd
from typing import Optional

from tensorrt_llm.llmapi.llm_args import CapacitySchedulerPolicy

from ..llm_request import LlmRequest, LlmRequestState

from .scheduler import (
    RequestList,
    RequestScheduler,
    SchedulerOutput,
)


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
    ):
        self.max_num_tokens = max_num_tokens
        self.max_num_requests = scheduler_capacity if scheduler_capacity is not None else max_batch_size
        from ..resource_manager import KVCacheManagerV2
        assert isinstance(kv_cache_manager, KVCacheManagerV2), (
            f"KVCacheV2Scheduler requires KVCacheManagerV2, got {type(kv_cache_manager).__name__}"
        )
        self.kv_cache_manager = kv_cache_manager
        assert scheduler_policy == CapacitySchedulerPolicy.MAX_UTILIZATION, (
            f"KVCacheV2Scheduler only supports MAX_UTILIZATION for now, got {scheduler_policy}"
        )
        self.policy = scheduler_policy
        self.peft_cache_manager = peft_cache_manager

        # Chunking config — only FCFS supported
        self.chunking_enabled = False
        self.chunk_unit_size = 0
        self.max_context_length = max_num_tokens
        self.tokens_per_block = kv_cache_manager.tokens_per_block
        if ctx_chunk_config is not None:
            self.chunking_enabled = True
            self.chunk_unit_size = ctx_chunk_config[1]

        # Non-last chunks must end at a position aligned to BOTH
        # chunk_unit_size (scheduler convention) and tokens_per_block
        # (KV cache block boundary — prevents cache fragmentation).
        # Using lcm guarantees alignment to both simultaneously.
        if self.chunking_enabled:
            u, b = self.chunk_unit_size, self.tokens_per_block
            self._chunk_align_unit = u * b // gcd(u, b)
        else:
            self._chunk_align_unit = 0

        # State value caches for fast comparison.
        # Use GENERATION_TO_COMPLETE as upper bound (exclusive) — same as V1's
        # MicroBatchScheduler. GENERATION_TO_COMPLETE requests are handled by the
        # overlap scheduler / decoder; we must not assign them token budget.
        self._no_schedule_until_state_value = LlmRequestState.CONTEXT_INIT.value
        self._no_schedule_after_state_value = LlmRequestState.GENERATION_TO_COMPLETE.value
        self._context_init_state_value = LlmRequestState.CONTEXT_INIT.value
        self._encoder_init_state_value = LlmRequestState.ENCODER_INIT.value
        self._disagg_gen_init_state_value = LlmRequestState.DISAGG_GENERATION_INIT.value

        # PEFT config (constant after construction)
        self._max_peft_pages = (peft_cache_manager.max_device_pages
                                if peft_cache_manager is not None else 0)

        # PEFT accounting — reset at the start of each scheduling iteration
        self._claimed_peft_pages = 0
        self._seen_peft_task_ids: set[int] = set()

    def schedule_request(
        self, active_requests: RequestList, inflight_request_ids: set[int]
    ) -> SchedulerOutput:
        # Main scheduling loop
        (scheduled_ctx, scheduled_gen, evicted, disagg_candidates,
         has_chunking) = self._schedule_loop(active_requests,
                                             inflight_request_ids)

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

            if not (req_state_value >= self._no_schedule_until_state_value
                    and req_state_value < self._no_schedule_after_state_value):
                req_it += 1
                continue

            scheduled = False

            if not self._check_peft(req):
                break

            # --- Encoder ---
            if req_state_value == self._encoder_init_state_value:
                req_tokens = req.encoder_output_len
                assert self.max_context_length is None or req_tokens <= self.max_context_length, (
                    f"The number of encoder tokens ({req_tokens}) exceeds the limit value ({self.max_context_length})"
                )
                if max_num_tokens is not None and (
                        batch_num_tokens + req_tokens > max_num_tokens):
                    break
                if not self.kv_cache_manager.prepare_context(req):
                    break
                if self.kv_cache_manager.resize_context(
                        req, req_tokens):
                    self._commit_peft(req)
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
                        req, scheduled_ctx, batch_num_tokens, max_num_tokens)
                    if len(scheduled_ctx) > prev_len:
                        chunk_tokens = req.context_chunk_size
                        if req.is_last_context_chunk and req.has_draft_tokens:
                            chunk_tokens += req.num_draft_tokens
                        has_chunking = has_chunking or (
                            req.context_chunk_size
                            < req.context_remaining_length)
                        batch_num_tokens += chunk_tokens
                        num_scheduled += 1
                        scheduled = True
                else:
                    # Non-chunking: prepare first so block reuse updates
                    # context_remaining_length before budget check.
                    if not self.kv_cache_manager.prepare_context(req):
                        break

                    context_tokens = req.context_remaining_length
                    draft_tokens = req.num_draft_tokens if req.has_draft_tokens else 0
                    req_tokens = context_tokens + draft_tokens

                    assert self.max_context_length is None or req_tokens <= self.max_context_length, (
                        f"Context tokens ({req_tokens}) exceeds limit ({self.max_context_length})"
                    )

                    if max_num_tokens is not None and (
                            batch_num_tokens + req_tokens > max_num_tokens):
                        break
                    if self.kv_cache_manager.resize_context(
                            req, context_tokens):
                        self._commit_peft(req)
                        scheduled_ctx.append(req)
                        batch_num_tokens += req_tokens
                        num_scheduled += 1
                        scheduled = True

            # --- Generation ---
            else:
                beam_width = req.get_beam_width_by_iter(
                    for_next_iteration=False)
                req_tokens = beam_width + req.num_draft_tokens

                if max_num_tokens is not None and (
                        batch_num_tokens + req_tokens > max_num_tokens):
                    break

                if scheduled_beam_width == 0:
                    scheduled_beam_width = beam_width
                elif scheduled_beam_width != beam_width:
                    req_it += 1
                    continue

                success = self.kv_cache_manager.try_allocate_generation(req)

                if not success:
                    evict_result = self._try_evict_for_gen(
                        req, requests_list, req_it, req_it_end,
                        scheduled_gen, evicted, batch_num_tokens)
                    if evict_result is not None:
                        success = True
                        req_it_end, batch_num_tokens = evict_result

                if success:
                    self._commit_peft(req)
                    scheduled_gen.append(req)
                    batch_num_tokens += req_tokens
                    num_scheduled += 1
                    scheduled = True

            if scheduled:
                req_it += 1
            elif req_state_value != self._context_init_state_value:
                # For MAX_UTIL gen/encoder failures not handled above, stop.
                break
            else:
                req_it += 1

        return scheduled_ctx, scheduled_gen, evicted, disagg_candidates, has_chunking

    def _schedule_context_chunked(self, req, scheduled_ctx, batch_num_tokens,
                                  max_num_tokens):
        """FCFS interleaved chunking for a single context request."""
        remaining_budget = (max_num_tokens -
                            batch_num_tokens) if max_num_tokens is not None else None
        align = self._chunk_align_unit

        # 1. Min budget check — need at least one alignment unit
        if remaining_budget is not None and remaining_budget < align:
            return

        # 2. Prepare context (create _KVCache, block reuse, resume — no resize)
        if not self.kv_cache_manager.prepare_context(req):
            return

        # 3. Calculate chunk size from remaining budget
        #    (context_remaining_length is now correct after block reuse)
        context_remaining = req.context_remaining_length
        chunk_size = min(
            remaining_budget,
            context_remaining) if remaining_budget is not None else context_remaining

        if self.max_context_length is not None:
            chunk_size = min(chunk_size, self.max_context_length)

        # Round down to alignment boundary (unless last chunk).
        # align = lcm(chunk_unit_size, tokens_per_block) ensures that
        # the chunk end position is a multiple of both, preventing
        # KV cache block fragmentation.
        if chunk_size < context_remaining:
            chunk_size = (chunk_size // align) * align

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
            budget_after_chunk = (remaining_budget -
                                  chunk_size) if remaining_budget is not None else None
            self._fit_draft_tokens_single(req, budget_after_chunk)

        self._commit_peft(req)
        scheduled_ctx.append(req)

    # ---- PEFT helpers ----

    def _check_peft(self, req: LlmRequest) -> bool:
        """Check if PEFT pages are available. Does NOT mutate state."""
        if self.peft_cache_manager is None:
            return True
        lora_task_id = getattr(req, "lora_task_id", None)
        if lora_task_id is None or lora_task_id in self._seen_peft_task_ids:
            return True
        required_pages = self.peft_cache_manager.determine_num_pages(req)
        return self._claimed_peft_pages + required_pages <= self._max_peft_pages

    def _commit_peft(self, req: LlmRequest) -> None:
        """Commit PEFT page accounting after successful scheduling."""
        if self.peft_cache_manager is None:
            return
        lora_task_id = getattr(req, "lora_task_id", None)
        if lora_task_id is not None and lora_task_id not in self._seen_peft_task_ids:
            self._claimed_peft_pages += self.peft_cache_manager.determine_num_pages(
                req)
            self._seen_peft_task_ids.add(lora_task_id)

    # ---- Eviction ----

    @staticmethod
    def _is_started_request(req: LlmRequest) -> bool:
        """A request that has begun execution and can be paused.
        Matches V1: (context_init && !first_chunk) || generation_in_progress.
        """
        return (
            (req.is_context_init_state and not req.is_first_context_chunk)
            or req.is_generation_in_progress_state
        )

    def _try_evict_for_gen(self, req, requests_list, req_it, req_it_end,
                           scheduled_gen, evicted, batch_num_tokens):
        """Evict started requests from active_requests tail to make room.

        Matches V1 MaxUtilizationScheduler: search backwards from req_it_end
        for "started" requests (gen in progress or non-first ctx chunk),
        suspend them to free pages, then retry allocation.

        Returns (new_req_it_end, new_batch_num_tokens) on success, None on failure.
        """
        while req_it_end > req_it:
            # Find last started request from tail
            victim_idx = None
            for i in range(req_it_end - 1, req_it - 1, -1):
                if self._is_started_request(requests_list[i]):
                    victim_idx = i
                    break

            if victim_idx is None:
                return None

            victim = requests_list[victim_idx]
            self.kv_cache_manager.suspend_request(victim)
            evicted.append(victim)

            # If the victim was already scheduled this iteration,
            # remove it and reclaim its token budget.
            if victim in scheduled_gen:
                scheduled_gen.remove(victim)
                v_beam = victim.get_beam_width_by_iter(
                    for_next_iteration=False)
                batch_num_tokens -= (v_beam + victim.num_draft_tokens)

            req_it_end = victim_idx

            if self.kv_cache_manager.try_allocate_generation(req):
                return req_it_end, batch_num_tokens

        return None

    # ---- Draft tokens ----

    def _fit_draft_tokens_single(self, req, budget_after_chunk):
        """Fit draft tokens into remaining space for a single request."""
        chunk_size = req.context_chunk_size
        align = self._chunk_align_unit
        remainder = chunk_size % align
        remaining_space = 0 if remainder == 0 else align - remainder

        if self.max_context_length is not None:
            remaining_context_len = self.max_context_length - chunk_size
            remaining_space = min(remaining_space, remaining_context_len)

        if budget_after_chunk is not None:
            remaining_space = min(remaining_space, budget_after_chunk)

        draft_tokens = min(req.num_draft_tokens, remaining_space)
        req.num_draft_tokens = draft_tokens

    # ---- Sorting ----

    @staticmethod
    def _lora_key(req: LlmRequest):
        lora_id = getattr(req, "lora_task_id", None)
        if lora_id is None:
            return (0, 0)
        return (1, lora_id)

    def _sort_requests(self, context_requests, generation_requests,
                       has_chunks):
        """Sort by LoRA task ID. Non-last chunks before last chunks."""
        if has_chunks:
            not_last = [
                r for r in context_requests if not r.is_last_context_chunk
            ]
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
        # TODO: implement this
        # """Conservative heuristic for PP retry loop. Does NOT allocate."""
        # stats = self.kv_cache_manager.get_kv_cache_stats()
        # available = stats.max_blocks - stats.used_blocks

        # for req in requests:
        #     if req.state == LlmRequestState.GENERATION_IN_PROGRESS:
        #         available -= 1
        #     elif req.state == LlmRequestState.CONTEXT_INIT:
        #         available -= (
        #             req.prompt_len + self.kv_cache_manager.tokens_per_block - 1
        #         ) // self.kv_cache_manager.tokens_per_block

        #     if available < 0:
        #         return False

        return True
