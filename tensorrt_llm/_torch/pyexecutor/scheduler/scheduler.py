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
"""Scheduler interfaces, shared data structures, and helper utilities.

This module defines the abstract base classes and data types used by all
scheduler implementations, along with small backend-agnostic helpers.

Implementations:
    simple_scheduler.py     — SimpleScheduler (C++ binding wrappers)
    unified_scheduler.py    — UnifiedScheduler (pure-Python)
    scheduler_v2.py         — KVCacheV2Scheduler
"""

from abc import ABC, abstractmethod
from collections import namedtuple
from dataclasses import dataclass, field
from typing import Optional

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._utils import nvtx_range

RequestList = list[LlmRequest]


def sort_requests_by_lora(
    context_requests: list[LlmRequest],
    generation_requests: list[LlmRequest],
    chunks_present: bool,
) -> None:
    def sort_key(req: LlmRequest) -> tuple[int, int]:
        lora_id = getattr(req, "lora_task_id", None)
        if lora_id is None:
            return (0, 0)
        return (1, lora_id)

    if chunks_present:
        not_last_chunk = [req for req in context_requests if not req.is_last_context_chunk]
        last_chunk = [req for req in context_requests if req.is_last_context_chunk]
        not_last_chunk.sort(key=sort_key)
        last_chunk.sort(key=sort_key)
        context_requests.clear()
        context_requests.extend(not_last_chunk)
        context_requests.extend(last_chunk)
    else:
        context_requests.sort(key=sort_key)

    generation_requests.sort(key=sort_key)


def compute_fcfs_context_chunk_size(
    context_remaining_length: int,
    capacity: Optional[int],
    max_context_length: Optional[int],
    unit_size: int,
) -> int:
    actual_size = context_remaining_length
    if capacity is not None and capacity < actual_size:
        actual_size = capacity
    if max_context_length is not None:
        actual_size = min(max_context_length, actual_size)
    if actual_size <= 0:
        return 0
    if actual_size < context_remaining_length:
        actual_size = (int(actual_size) // unit_size) * unit_size
    return int(actual_size)


SchedulerOutput = namedtuple(
    "SchedulerOutput",
    [
        "context_requests",
        "generation_requests",
        "paused_requests",
        "fitting_disagg_gen_init_requests",
        "num_fitting_requests",
    ],
)


class ScheduledRequests:
    """Scheduled requests separated into disjoint sets.

    The reason for the separation is that requests are handled differently in different phases.
    For example,
    - context requests and generation requests execute different attention kernels.
    - only context requests that are at the last chunk and generation requests sample new tokens.
    """

    context_requests_chunking: RequestList
    """Requests that are in the middle of the context phase."""
    context_requests_last_chunk: RequestList
    """Requests that are in the last chunk of the context phase."""
    generation_requests: RequestList
    """Requests that are in the generation phase."""
    paused_requests: RequestList
    """Requests that are paused."""

    def __init__(self):
        self.context_requests_chunking: RequestList = []
        self.context_requests_last_chunk: RequestList = []
        self.generation_requests: RequestList = []
        self.paused_requests: RequestList = []

    @property
    def is_generation_only(self) -> bool:
        return self.num_context_requests == 0 and all(
            len(req.draft_tokens) == 0 for req in self.generation_requests
        )

    @property
    def can_run_cuda_graph(self) -> bool:
        return self.num_context_requests == 0

    @property
    def batch_size(self) -> int:
        return self.num_context_requests + len(self.generation_requests)

    @property
    def num_context_requests(self) -> int:
        return len(self.context_requests_chunking) + len(self.context_requests_last_chunk)

    @property
    def num_generation_requests(self) -> int:
        return len(self.generation_requests)

    @property
    def context_requests(self) -> RequestList:
        return self.context_requests_chunking + self.context_requests_last_chunk

    def all_requests(self) -> RequestList:
        return self.context_requests + self.generation_requests

    def append_context_request(self, request: LlmRequest) -> None:
        if request.is_last_context_chunk:
            self.context_requests_last_chunk.append(request)
        else:
            self.context_requests_chunking.append(request)

    def append_generation_request(self, request: LlmRequest) -> None:
        self.generation_requests.append(request)

    def reset_context_requests(self, context_requests: RequestList | None = None) -> None:
        context_requests = (
            context_requests if context_requests is not None else self.context_requests
        )
        self.context_requests_chunking = []
        self.context_requests_last_chunk = []
        for req in context_requests:
            self.append_context_request(req)


@dataclass
class ScheduleStepConfig:
    """Configuration for executor-facing scheduler step post-processing."""

    enable_attention_dp: bool = False
    attention_dp_enable_balance: bool = False
    attention_dp_time_out_iters: int = 0
    attention_dp_batching_wait_iters: int = 0
    batch_wait_timeout_iters: int = 0
    batch_wait_max_tokens_ratio: float = 0.0
    max_batch_size: int = 0
    max_num_tokens: int = 0


@dataclass
class ScheduleStepResult:
    """Finalized scheduling result consumed by the executor."""

    scheduled_requests: ScheduledRequests = field(default_factory=ScheduledRequests)
    fitting_disagg_gen_init_requests: RequestList = field(default_factory=list)
    num_fitting_requests: int = 0


class RequestScheduler(ABC):
    def __init__(self, schedule_step_config: Optional[ScheduleStepConfig] = None, dist=None):
        self._schedule_step_config = schedule_step_config or ScheduleStepConfig()
        self._dist = dist
        self._adp_ctx_waiting_iters_count = 0
        self._adp_ctx_batching_wait_iters_count = 0
        self._batch_wait_iters_count = 0

    @abstractmethod
    def schedule_request(
        self, active_requests: RequestList, inflight_request_ids: set[int]
    ) -> SchedulerOutput:
        """
        :param active_requests: list of active requests, up to maximum number of sequences
        :param inflight_request_ids: set of request ids that are inflight (of all micro batches)
        :return: SchedulerOutput
        """
        raise NotImplementedError

    @abstractmethod
    def can_schedule(self, requests: RequestList) -> bool:
        """
        Check if current rank can schedule the requests.
        :param requests: list of requests to be scheduled
        :return: True if current rank can schedule the requests, False otherwise
        """
        raise NotImplementedError

    @nvtx_range("_schedule")
    def schedule_step(
        self, active_requests: RequestList, inflight_request_ids: set[int]
    ) -> ScheduleStepResult:
        scheduler_output = self.schedule_request(active_requests, inflight_request_ids)

        scheduled_context_requests = scheduler_output.context_requests
        cfg = self._schedule_step_config

        if cfg.enable_attention_dp and cfg.attention_dp_enable_balance:
            scheduled_context_requests = self._balance_adp_context_requests(
                scheduler_output.context_requests,
                scheduler_output.generation_requests,
            )

        enable_batch_waiting = (
            cfg.batch_wait_timeout_iters > 0 or cfg.batch_wait_max_tokens_ratio > 0
        )
        should_check_waiting = (
            not cfg.enable_attention_dp
            and enable_batch_waiting
            and len(scheduler_output.context_requests) > 0
            and len(scheduler_output.generation_requests) > 0
        )
        if should_check_waiting:
            scheduled_context_requests = self._apply_batch_waiting(
                scheduled_context_requests, scheduler_output.generation_requests
            )

        scheduled_requests = ScheduledRequests()
        scheduled_requests.reset_context_requests(scheduled_context_requests)
        scheduled_requests.generation_requests = scheduler_output.generation_requests
        scheduled_requests.paused_requests = scheduler_output.paused_requests

        return ScheduleStepResult(
            scheduled_requests=scheduled_requests,
            fitting_disagg_gen_init_requests=scheduler_output.fitting_disagg_gen_init_requests,
            num_fitting_requests=scheduler_output.num_fitting_requests,
        )

    def _balance_adp_context_requests(
        self, context_requests: RequestList, generation_requests: RequestList
    ) -> RequestList:
        if self._dist is None:
            raise RuntimeError(
                "RequestScheduler.schedule_step requires dist for attention-dp balancing"
            )

        cfg = self._schedule_step_config
        balanced_context_requests = context_requests
        num_scheduled_context_requests = len(context_requests)
        num_scheduled_generation_requests = len(generation_requests)
        responses_list = self._dist.tp_allgather(
            [
                num_scheduled_context_requests,
                num_scheduled_generation_requests,
            ]
        )
        all_ranks_num_scheduled_context_requests = [response[0] for response in responses_list]
        all_ranks_num_scheduled_generation_requests = [response[1] for response in responses_list]
        all_ranks_have_free_ctx_slots = all(
            num_gen < cfg.max_batch_size for num_gen in all_ranks_num_scheduled_generation_requests
        )
        all_ranks_have_ctx_requests = all(
            num_ctx > 0 for num_ctx in all_ranks_num_scheduled_context_requests
        )
        all_ranks_have_gen_requests = all(
            num_gen > 0 for num_gen in all_ranks_num_scheduled_generation_requests
        )

        if all_ranks_have_free_ctx_slots and all_ranks_have_ctx_requests:
            self._adp_ctx_waiting_iters_count = 0
            if all_ranks_have_gen_requests:
                if self._adp_ctx_batching_wait_iters_count < cfg.attention_dp_batching_wait_iters:
                    self._adp_ctx_batching_wait_iters_count += 1
                    balanced_context_requests = []
                else:
                    self._adp_ctx_batching_wait_iters_count = 0
        else:
            self._adp_ctx_waiting_iters_count += 1
            balanced_context_requests = []
            timeout_reached = self._adp_ctx_waiting_iters_count >= cfg.attention_dp_time_out_iters
            if timeout_reached or not all_ranks_have_gen_requests:
                self._adp_ctx_waiting_iters_count = 0
                balanced_context_requests = context_requests

        return balanced_context_requests

    def _apply_batch_waiting(
        self, context_requests: RequestList, generation_requests: RequestList
    ) -> RequestList:
        cfg = self._schedule_step_config
        num_scheduled_ctx_tokens = self._get_num_scheduled_context_tokens(context_requests)
        num_scheduled_gen_tokens = sum(1 + req.num_draft_tokens for req in generation_requests)
        num_scheduled_tokens = num_scheduled_ctx_tokens + num_scheduled_gen_tokens

        should_waiting = (
            self._batch_wait_iters_count < cfg.batch_wait_timeout_iters
            and num_scheduled_tokens < cfg.batch_wait_max_tokens_ratio * cfg.max_num_tokens
        )
        if should_waiting:
            self._batch_wait_iters_count += 1
            return []

        self._batch_wait_iters_count = 0
        return context_requests

    def _get_num_scheduled_context_tokens(self, context_requests: RequestList) -> int:
        return sum(len(req.get_tokens(0)) for req in context_requests)


@dataclass
class SerializableSchedulerOutput:
    """
    Serializable version of SchedulerOutput, used for sending schedule result to other ranks.

    Analogous to ScheduledRequests the lists are disjoint sets of request IDs.
    Need this class because LlmRequest is not serializable by pickle.
    """

    context_requests_chunking: list[int]  # request ids of context requests chunking
    context_requests_last_chunk: list[int]  # request ids of context requests last chunk
    generation_requests: list[int]  # request ids of generation requests
    paused_requests: list[int]  # request ids of paused requests
    fitting_disagg_gen_init_requests: list[
        int
    ]  # request ids of fitting disaggregated generation initialization requests
    num_fitting_requests: int  # number of fitting requests

    @classmethod
    def from_scheduler_result(
        cls,
        scheduled_requests: ScheduledRequests,
        fitting_disagg_gen_init_requests: RequestList,
        num_fitting_requests: int,
    ) -> "SerializableSchedulerOutput":
        return cls(
            context_requests_chunking=[
                req.request_id for req in scheduled_requests.context_requests_chunking
            ],
            context_requests_last_chunk=[
                req.request_id for req in scheduled_requests.context_requests_last_chunk
            ],
            generation_requests=[req.request_id for req in scheduled_requests.generation_requests],
            paused_requests=[req.request_id for req in scheduled_requests.paused_requests],
            fitting_disagg_gen_init_requests=[
                req.request_id for req in fitting_disagg_gen_init_requests
            ],
            num_fitting_requests=num_fitting_requests,
        )

    def to_scheduler_result(
        self, active_requests: RequestList
    ) -> tuple[ScheduledRequests, RequestList, int]:
        id_to_request = {req.request_id: req for req in active_requests}
        scheduled_requests = ScheduledRequests()
        scheduled_requests.context_requests_chunking = [
            id_to_request[req_id] for req_id in self.context_requests_chunking
        ]
        scheduled_requests.context_requests_last_chunk = [
            id_to_request[req_id] for req_id in self.context_requests_last_chunk
        ]
        scheduled_requests.generation_requests = [
            id_to_request[req_id] for req_id in self.generation_requests
        ]
        scheduled_requests.paused_requests = [
            id_to_request[req_id] for req_id in self.paused_requests
        ]
        fitting_disagg_gen_init_requests = [
            id_to_request[req_id] for req_id in self.fitting_disagg_gen_init_requests
        ]
        return scheduled_requests, fitting_disagg_gen_init_requests, self.num_fitting_requests
