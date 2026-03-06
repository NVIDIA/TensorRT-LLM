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
"""SimpleScheduler: two-pass scheduling (capacity -> microbatch) using C++ bindings.

Also includes KVCacheV2DummyScheduler for the v2 KV cache path.
"""

from typing import Optional

from strenum import StrEnum

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestState
from tensorrt_llm.bindings import internal as tb_internal
from tensorrt_llm.llmapi.llm_args import CapacitySchedulerPolicy

from .scheduler import (
    CapacityScheduler,
    MicroBatchScheduler,
    RequestList,
    RequestScheduler,
    SchedulerOutput,
)


class BindCapacityScheduler(CapacityScheduler):
    def __init__(
        self,
        max_num_requests: int,
        kv_cache_manager,
        peft_cache_manager: tb_internal.batch_manager.PeftCacheManager | None,
        scheduler_policy: CapacitySchedulerPolicy = CapacitySchedulerPolicy.GUARANTEED_NO_EVICT,
        two_step_lookahead: bool = False,
    ):
        super(BindCapacityScheduler, self).__init__()
        self.kv_cache_manager = kv_cache_manager
        self.peft_cache_manager = peft_cache_manager

        self.impl = tb_internal.algorithms.CapacityScheduler(
            max_num_requests=max_num_requests,
            capacity_scheduler_policy=scheduler_policy._to_pybind(),
            has_kv_cache_manager=kv_cache_manager is not None,
            two_step_lookahead=two_step_lookahead,
            no_schedule_until_state=LlmRequestState.CONTEXT_INIT,
            no_schedule_after_state=LlmRequestState.GENERATION_COMPLETE,
        )

    def schedule_request(
        self, active_requests: RequestList
    ) -> tuple[list[LlmRequest], list[LlmRequest], list[LlmRequest]]:
        return self.impl(active_requests, self.kv_cache_manager, self.peft_cache_manager)


class KVCacheV2DummyScheduler(CapacityScheduler):
    # only schedule requests has no_schedule_until_state <= state < no_schedule_after_state
    no_schedule_until_state = LlmRequestState.CONTEXT_INIT
    no_schedule_after_state = LlmRequestState.GENERATION_COMPLETE

    def __init__(self, max_num_requests: int, kv_cache_manager, peft_cache_manager=None):
        super(KVCacheV2DummyScheduler, self).__init__()
        self.max_num_requests = max_num_requests
        self.kv_cache_manager = kv_cache_manager
        self.peft_cache_manager = peft_cache_manager

    def _get_max_peft_pages(self) -> int:
        if self.peft_cache_manager is None:
            return 2**31 - 1
        return self.peft_cache_manager.max_device_pages

    def _get_peft_task_info(
        self, req: LlmRequest, seen_task_ids: set[int]
    ) -> tuple[Optional[int], bool, int]:
        lora_task_id = getattr(req, "lora_task_id", None)
        is_new_task = lora_task_id is not None and lora_task_id not in seen_task_ids
        if is_new_task and self.peft_cache_manager is not None:
            required_pages = self.peft_cache_manager.determine_num_pages(req)
        else:
            required_pages = 0
        return lora_task_id, is_new_task, required_pages

    def schedule_request(
        self, active_requests: RequestList
    ) -> tuple[list[LlmRequest], list[LlmRequest], list[LlmRequest]]:
        scheduled_requests = []
        scheduled_disagg_gen_init_requests = []
        pending_requests = []
        reserved_blocks = 0
        max_blocks = self.kv_cache_manager.get_max_resource_count()

        has_peft = self.peft_cache_manager is not None
        claimed_peft_pages = 0
        available_peft_pages = self._get_max_peft_pages() if has_peft else 0
        uniq_task_ids: set[int] = set() if has_peft else None

        for request in active_requests:
            req_state = request.state
            # if request cannot be scheduled yet or request should no longer be scheduled, skip
            if not req_state == LlmRequestState.DISAGG_GENERATION_INIT and (
                req_state.value < self.no_schedule_until_state.value
                or req_state.value >= self.no_schedule_after_state.value
            ):
                continue

            if len(scheduled_requests) >= self.max_num_requests or reserved_blocks >= max_blocks:
                break
            elif (
                req_state == LlmRequestState.GENERATION_IN_PROGRESS
                or req_state == LlmRequestState.GENERATION_TO_COMPLETE
            ):
                scheduled_requests.append(request)
                reserved_blocks += self.kv_cache_manager.get_needed_resource_to_completion(request)

                if has_peft:
                    lora_task_id, is_new_task, peft_pages = self._get_peft_task_info(
                        request, uniq_task_ids
                    )
                    if is_new_task:
                        claimed_peft_pages += peft_pages
                        uniq_task_ids.add(lora_task_id)

            elif req_state == LlmRequestState.DISAGG_GENERATION_INIT:
                scheduled_disagg_gen_init_requests.append(request)
                reserved_blocks += self.kv_cache_manager.get_needed_resource_to_completion(request)
            else:
                pending_requests.append(request)

        if has_peft:
            available_peft_pages -= claimed_peft_pages

        available_blocks = max_blocks - reserved_blocks
        for request in pending_requests:
            req_state = request.state
            if len(scheduled_requests) >= self.max_num_requests:
                break
            elif req_state == LlmRequestState.CONTEXT_INIT:
                needed_blocks = self.kv_cache_manager.get_needed_resource_to_completion(request)
                if needed_blocks <= available_blocks:
                    if has_peft:
                        lora_task_id, is_new_task, needed_peft_pages = self._get_peft_task_info(
                            request, uniq_task_ids
                        )
                        if needed_peft_pages > available_peft_pages:
                            continue
                        available_peft_pages -= needed_peft_pages
                        if is_new_task:
                            uniq_task_ids.add(lora_task_id)

                    scheduled_requests.append(request)
                    available_blocks -= needed_blocks
                elif needed_blocks > available_blocks:
                    # If one requests fails to be scheduled, break
                    break

        return scheduled_requests, scheduled_disagg_gen_init_requests, []


class BindMicroBatchScheduler(MicroBatchScheduler):
    def __init__(
        self,
        max_batch_size: int,
        max_num_tokens: int = None,
        ctx_chunk_config: Optional[tuple[StrEnum, int]] = None,
    ) -> None:
        super(BindMicroBatchScheduler, self).__init__()
        self.max_batch_size = max_batch_size
        self.max_num_tokens = max_num_tokens

        ctx_chunk_config_cpp = None
        if ctx_chunk_config is not None:
            policy = ctx_chunk_config[0]
            ctx_chunk_config_cpp = tb_internal.batch_manager.ContextChunkingConfig(
                policy._to_pybind(),
                ctx_chunk_config[1],  # type: ignore[attr-defined]
            )

        self.impl = tb_internal.algorithms.MicroBatchScheduler(ctx_chunk_config_cpp, max_num_tokens)

    def schedule(
        self, active_requests: RequestList, inflight_request_ids: set[int]
    ) -> tuple[list[LlmRequest], list[LlmRequest]]:
        return self.impl(
            active_requests, inflight_request_ids, self.max_batch_size, self.max_num_tokens
        )


class SimpleScheduler(RequestScheduler):
    def __init__(
        self, capacity_scheduler: CapacityScheduler, micro_batch_scheduler: MicroBatchScheduler
    ):
        super(SimpleScheduler, self).__init__()
        self.capacity_scheduler = capacity_scheduler
        self.micro_batch_scheduler = micro_batch_scheduler

    def schedule_request(
        self, active_requests: RequestList, inflight_request_ids: set[int]
    ) -> SchedulerOutput:
        fitting_requests, fitting_disagg_gen_init_requests, paused_requests = (
            self.capacity_scheduler.schedule_request(active_requests)
        )

        context_requests, generation_requests = self.micro_batch_scheduler.schedule(
            fitting_requests, inflight_request_ids
        )
        # Convert from binding type RequestVector to list[LlmRequest],
        # so Python fields on LlmRequest won't be stripped away
        return SchedulerOutput(
            list(context_requests),
            list(generation_requests),
            list(paused_requests),
            list(fitting_disagg_gen_init_requests),
            len(fitting_requests),
        )

    def can_schedule(self, requests: RequestList) -> bool:
        fitting_requests, _, _ = self.capacity_scheduler.schedule_request(requests)
        return len(fitting_requests) == len(requests)
