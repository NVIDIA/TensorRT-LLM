from abc import ABC, abstractmethod
from typing import Optional

from tensorrt_llm.bindings import executor as tb_executor
from tensorrt_llm.bindings import internal as tb_internal

from .llm_request import LlmRequest, LlmRequestState

RequestList = list[LlmRequest]


class ScheduledRequests:
    # to be aligned with ScheduledRequests in cpp/tensorrt_llm/batch_manager/common.h
    context_requests: RequestList
    generation_requests: RequestList
    paused_requests: RequestList

    @property
    def is_generation_only(self) -> bool:
        return (not self.context_requests and all(
            len(req.draft_tokens) == 0 for req in self.generation_requests))

    @property
    def batch_size(self) -> int:
        return len(self.context_requests) + len(self.generation_requests)


class RequestScheduler(ABC):

    @abstractmethod
    def schedule_request(
        self, active_requests: RequestList, inflight_request_ids: set[int]
    ) -> tuple[list[LlmRequest], list[LlmRequest], list[LlmRequest]]:
        """
        :param active_requests: list of active requests, up to maximum number of sequences
        :param inflight_request_ids: set of request ids that are inflight (of all micro batches)
        :return: (contextRequests, generationRequests, pausedRequests)
        """
        # to be aligned with RequestScheduler::scheduleRequests in cpp/tensorrt_llm/batch_manager/requestScheduler.h
        raise NotImplementedError


class CapacityScheduler(ABC):

    @abstractmethod
    def schedule_request(
        self, active_requests: RequestList
    ) -> tuple[list[LlmRequest], list[LlmRequest]]:
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
        scheduler_policy: tb_executor.CapacitySchedulerPolicy = tb_executor.
        CapacitySchedulerPolicy.GUARANTEED_NO_EVICT):
        super(BindCapacityScheduler, self).__init__()
        self.kv_cache_manager = kv_cache_manager
        self.impl = tb_internal.algorithms.CapacityScheduler(
            max_num_requests, scheduler_policy, True, False,
            LlmRequestState.CONTEXT_INIT, LlmRequestState.GENERATION_COMPLETE)

    def schedule_request(
        self, active_requests: RequestList
    ) -> tuple[list[LlmRequest], list[LlmRequest]]:
        return self.impl(active_requests, self.kv_cache_manager)


class GuaranteedNoEvictScheduler(CapacityScheduler):
    # only schedule requests has no_schedule_until_state <= state < no_schedule_after_state
    no_schedule_until_state = LlmRequestState.CONTEXT_INIT
    no_schedule_after_state = LlmRequestState.GENERATION_COMPLETE

    def __init__(self, max_num_requests: int, kv_cache_manager):
        super(GuaranteedNoEvictScheduler, self).__init__()
        self.max_num_requests = max_num_requests
        self.kv_cache_manager = kv_cache_manager

    def schedule_request(
        self, active_requests: RequestList
    ) -> tuple[list[LlmRequest], list[LlmRequest]]:
        scheduled_requests = []
        pending_requests = []
        reserved_blocks = 0
        max_blocks = self.kv_cache_manager.get_max_resource_count()
        for request in active_requests:
            req_state = request.state
            # if request cannot be scheduled yet or request should no longer be scheduled, skip
            if req_state.value < self.no_schedule_until_state.value or req_state.value >= self.no_schedule_after_state.value:
                continue

            if len(scheduled_requests
                   ) >= self.max_num_requests or reserved_blocks >= max_blocks:
                break
            elif req_state == LlmRequestState.GENERATION_IN_PROGRESS or req_state == LlmRequestState.GENERATION_TO_COMPLETE:
                scheduled_requests.append(request)
                reserved_blocks += self.kv_cache_manager.get_needed_resource_to_completion(
                    request)
            else:
                pending_requests.append(request)

        avaiable_blocks = max_blocks - reserved_blocks
        for request in pending_requests:
            req_state = request.state
            if len(scheduled_requests) >= self.max_num_requests:
                break
            elif req_state == LlmRequestState.CONTEXT_INIT:
                needed_blocks = self.kv_cache_manager.get_needed_resource_to_completion(
                    request)
                if needed_blocks <= avaiable_blocks:
                    scheduled_requests.append(request)
                    avaiable_blocks -= needed_blocks
                elif needed_blocks > avaiable_blocks:
                    # If one requests fails to be scheduled, break
                    break

        assert len(scheduled_requests) > 0, (
            "no pending request can get enough resource to complete, "
            "please increase KV cache pool size.")
        return scheduled_requests, []


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
        # to be aligned with MicroBatchScheduler::scheduleRequests in cpp/tensorrt_llm/batch_manager/microBatchScheduler.h
        raise NotImplementedError


class BindMicroBatchScheduler(MicroBatchScheduler):

    def __init__(
        self,
        max_batch_size: int,
        max_num_tokens: int = None,
        ctx_chunk_config: Optional[
            tb_internal.batch_manager.ContextChunkingConfig] = None,
    ) -> None:
        super(BindMicroBatchScheduler, self).__init__()
        self.max_batch_size = max_batch_size
        self.max_num_tokens = max_num_tokens
        self.impl = tb_internal.algorithms.MicroBatchScheduler(
            ctx_chunk_config, max_num_tokens)

    def schedule(
        self, active_requests: RequestList, inflight_request_ids: set[int]
    ) -> tuple[list[LlmRequest], list[LlmRequest]]:
        return self.impl(active_requests, inflight_request_ids,
                         self.max_batch_size, self.max_num_tokens)


class SimpleScheduler(RequestScheduler):

    def __init__(self, capacity_scheduler: CapacityScheduler,
                 micro_batch_scheduler: MicroBatchScheduler):
        super(SimpleScheduler, self).__init__()
        self.capacity_scheduler = capacity_scheduler
        self.micro_batch_scheduler = micro_batch_scheduler

    def schedule_request(
        self, active_requests: RequestList, inflight_request_ids: set[int]
    ) -> tuple[list[LlmRequest], list[LlmRequest], list[LlmRequest]]:
        fitting_requests, paused_requests = self.capacity_scheduler.schedule_request(
            active_requests)
        #print(f'capacitor scheduler scheduled {len(fitting_requests)} fitting_request')
        context_requests, generation_requests = self.micro_batch_scheduler.schedule(
            fitting_requests, inflight_request_ids)
        return context_requests, generation_requests, paused_requests
