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
"""Scheduler interfaces and shared data structures.

This module defines the abstract base classes and data types used by all
scheduler implementations. No concrete scheduling logic lives here.

Implementations:
    simple_scheduler.py     — SimpleScheduler (C++ binding wrappers)
    unified_scheduler.py    — SimpleUnifiedScheduler (pure-Python)
"""

from abc import ABC, abstractmethod
from collections import namedtuple
from dataclasses import dataclass

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest

RequestList = list[LlmRequest]

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


class RequestScheduler(ABC):
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


class CapacityScheduler(ABC):
    @abstractmethod
    def schedule_request(
        self, active_requests: RequestList
    ) -> tuple[list[LlmRequest], list[LlmRequest], list[LlmRequest]]:
        """
        :param active_requests: list of active requests, up to maximum number of sequences
        :return: (scheduledRequests, pausedRequests)
        """
        raise NotImplementedError


class MicroBatchScheduler(ABC):
    @abstractmethod
    def schedule(
        self, active_requests: RequestList, inflight_request_ids: set[int]
    ) -> tuple[list[LlmRequest], list[LlmRequest]]:
        """
        :param active_requests: list of active requests, up to maximum number of sequences
        :param inflight_request_ids: set of request ids that are inflight (of all micro batches)
        :return: (contextRequests, generationRequests)
        """
        raise NotImplementedError
