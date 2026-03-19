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
"""SimpleScheduler: two-pass scheduling (capacity -> microbatch) using C++ bindings."""

from typing import Optional

from strenum import StrEnum

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestState
from tensorrt_llm.bindings import internal as tb_internal
from tensorrt_llm.llmapi.llm_args import CapacitySchedulerPolicy

from .scheduler import RequestList, RequestScheduler, SchedulerOutput, ScheduleStepConfig


class BindCapacityScheduler:
    def __init__(
        self,
        max_num_requests: int,
        kv_cache_manager,
        peft_cache_manager: tb_internal.batch_manager.PeftCacheManager | None,
        scheduler_policy: CapacitySchedulerPolicy = CapacitySchedulerPolicy.GUARANTEED_NO_EVICT,
        two_step_lookahead: bool = False,
    ):
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


class BindMicroBatchScheduler:
    def __init__(
        self,
        max_batch_size: int,
        max_num_tokens: int = None,
        ctx_chunk_config: Optional[tuple[StrEnum, int]] = None,
    ) -> None:
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
        self,
        capacity_scheduler,
        micro_batch_scheduler,
        schedule_step_config: Optional[ScheduleStepConfig] = None,
        dist=None,
    ):
        super(SimpleScheduler, self).__init__(schedule_step_config=schedule_step_config, dist=dist)
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
