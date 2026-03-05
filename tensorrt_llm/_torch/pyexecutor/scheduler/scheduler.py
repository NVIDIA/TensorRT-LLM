from abc import ABC, abstractmethod
from collections import namedtuple
from dataclasses import dataclass
from typing import Optional

from strenum import StrEnum

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestState
from tensorrt_llm.bindings import internal as tb_internal
from tensorrt_llm.llmapi.llm_args import CapacitySchedulerPolicy

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
        # to be aligned with RequestScheduler::scheduleRequests in cpp/tensorrt_llm/batch_manager/requestScheduler.h
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
        # to be aligned with CapacityScheduler::scheduleRequests in cpp/tensorrt_llm/batch_manager/capacityScheduler.h
        raise NotImplementedError


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
        # to be aligned with MicroBatchScheduler::scheduleRequests
        # in cpp/tensorrt_llm/batch_manager/microBatchScheduler.h
        raise NotImplementedError


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
