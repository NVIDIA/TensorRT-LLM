import dataclasses
from abc import ABC, abstractmethod
from collections import namedtuple
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Set, Tuple

from strenum import StrEnum

from tensorrt_llm.bindings import internal as tb_internal
from tensorrt_llm.llmapi.llm_args import CapacitySchedulerPolicy

from .llm_request import LlmRequest, LlmRequestState
from .resource_manager import KVCacheManager

RequestList = list[LlmRequest]

SchedulerOutput = namedtuple("SchedulerOutput", [
    "context_requests", "generation_requests", "paused_requests",
    "fitting_disagg_gen_init_requests", "num_fitting_requests"
])


class ScheduledRequests:
    # to be aligned with ScheduledRequests in cpp/tensorrt_llm/batch_manager/common.h
    def __init__(self):
        self.context_requests: RequestList = []
        self.generation_requests: RequestList = []
        self.paused_requests: RequestList = []

    @property
    def is_generation_only(self) -> bool:
        return (not self.context_requests and all(
            len(req.draft_tokens) == 0 for req in self.generation_requests))

    @property
    def can_run_cuda_graph(self) -> bool:
        return (not self.context_requests)

    @property
    def batch_size(self) -> int:
        return len(self.context_requests) + len(self.generation_requests)

    def all_requests(self) -> list[LlmRequest]:
        return self.context_requests + self.generation_requests


class RequestScheduler(ABC):

    @abstractmethod
    def schedule_request(self, active_requests: RequestList,
                         inflight_request_ids: set[int]) -> SchedulerOutput:
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
    Serializable version of SchedulerOutput, used for sending schedule result to other ranks. Need this class because LlmRequest is not serializable by pickle.
    """
    context_requests: list[int]  # request ids of context requests
    generation_requests: list[int]  # request ids of generation requests
    paused_requests: list[int]  # request ids of paused requests
    fitting_disagg_gen_init_requests: list[
        int]  # request ids of fitting disaggregated generation initialization requests
    num_fitting_requests: int  # number of fitting requests

    @classmethod
    def from_scheduler_result(
            cls, scheduled_requests: ScheduledRequests,
            fitting_disagg_gen_init_requests: RequestList,
            num_fitting_requests: int) -> "SerializableSchedulerOutput":
        return cls(context_requests=[
            req.request_id for req in scheduled_requests.context_requests
        ],
                   generation_requests=[
                       req.request_id
                       for req in scheduled_requests.generation_requests
                   ],
                   paused_requests=[
                       req.request_id
                       for req in scheduled_requests.paused_requests
                   ],
                   fitting_disagg_gen_init_requests=[
                       req.request_id
                       for req in fitting_disagg_gen_init_requests
                   ],
                   num_fitting_requests=num_fitting_requests)

    def to_scheduler_result(
        self, active_requests: RequestList
    ) -> Tuple[ScheduledRequests, RequestList, int]:
        id_to_request = {req.request_id: req for req in active_requests}
        scheduled_requests = ScheduledRequests()
        scheduled_requests.context_requests = [
            id_to_request[req_id] for req_id in self.context_requests
        ]
        scheduled_requests.generation_requests = [
            id_to_request[req_id] for req_id in self.generation_requests
        ]
        scheduled_requests.paused_requests = [
            id_to_request[req_id] for req_id in self.paused_requests
        ]
        fitting_disagg_gen_init_requests = [
            id_to_request[req_id]
            for req_id in self.fitting_disagg_gen_init_requests
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
        scheduler_policy: CapacitySchedulerPolicy = CapacitySchedulerPolicy.
        GUARANTEED_NO_EVICT,
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
            no_schedule_after_state=LlmRequestState.GENERATION_COMPLETE)

    def schedule_request(
        self, active_requests: RequestList
    ) -> tuple[list[LlmRequest], list[LlmRequest], list[LlmRequest]]:
        return self.impl(active_requests, self.kv_cache_manager,
                         self.peft_cache_manager)


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
        ctx_chunk_config: Optional[Tuple[StrEnum, int]] = None,
    ) -> None:
        super(BindMicroBatchScheduler, self).__init__()
        self.max_batch_size = max_batch_size
        self.max_num_tokens = max_num_tokens

        ctx_chunk_config_cpp = None
        if ctx_chunk_config is not None:
            ctx_chunk_config_cpp = tb_internal.batch_manager.ContextChunkingConfig(
                ctx_chunk_config[0]._to_pybind(), ctx_chunk_config[1])

        self.impl = tb_internal.algorithms.MicroBatchScheduler(
            ctx_chunk_config_cpp, max_num_tokens)

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

    def schedule_request(self, active_requests: RequestList,
                         inflight_request_ids: set[int]) -> SchedulerOutput:
        fitting_requests, fitting_disagg_gen_init_requests, paused_requests = self.capacity_scheduler.schedule_request(
            active_requests)

        context_requests, generation_requests = self.micro_batch_scheduler.schedule(
            fitting_requests, inflight_request_ids)
        # Convert from binding type RequestVector to list[LlmRequest],
        # so Python fields on LlmRequest won't be stripped away
        return SchedulerOutput(list(context_requests),
                               list(generation_requests), list(paused_requests),
                               list(fitting_disagg_gen_init_requests),
                               len(fitting_requests))

    def can_schedule(self, requests: RequestList) -> bool:
        fitting_requests, _, _ = self.capacity_scheduler.schedule_request(
            requests)
        return len(fitting_requests) == len(requests)


class ChunkingPolicy(Enum):
    EQUAL_PROGRESS = 1
    FIRST_COME_FIRST_SERVED = 2


@dataclasses.dataclass
class ContextChunkingConfig:
    chunking_policy: ChunkingPolicy
    chunk_unit_size: int


class PyMicroBatchScheduler(MicroBatchScheduler):

    def __init__(
        self,
        max_batch_size: int,
        max_num_tokens: Optional[int] = None,
        ctx_chunk_config: Optional[ContextChunkingConfig] = None,
    ):
        super().__init__()
        self.max_batch_size = max_batch_size
        self.max_num_tokens = max_num_tokens
        self.ctx_chunk_config = ctx_chunk_config

    def schedule(
            self, active_requests: RequestList,
            inflight_request_ids: Set[int]) -> Tuple[RequestList, RequestList]:

        context_requests: RequestList = []
        generation_requests: RequestList = []

        current_batch_tokens = 0
        scheduled_req_count = 0
        scheduled_beam_width = 0

        contexts_to_be_chunked: RequestList = []
        num_chunked_tokens = 0
        all_context_fits = True

        # 1. First Pass: Filter & Categorize (Generation First)
        for req in active_requests:
            # Skip invalid states (Simplified check, assuming caller filters mostly)
            if req.request_id in inflight_request_ids:
                continue

            # --- Generation Handling ---
            if req.state == LlmRequestState.GENERATION_IN_PROGRESS:
                beam_width = req.sampling_config.beam_width
                req_num_tokens = beam_width + req.num_draft_tokens

                # Check Global Token Budget
                if self.max_num_tokens is not None and (current_batch_tokens +
                                                        req_num_tokens
                                                        > self.max_num_tokens):
                    break

                # Check Beam Width Consistency (Batch constraint)
                if scheduled_beam_width == 0:
                    scheduled_beam_width = beam_width
                elif scheduled_beam_width != beam_width:
                    continue

                generation_requests.append(req)
                current_batch_tokens += req_num_tokens

            # --- Context Handling ---
            elif req.state == LlmRequestState.CONTEXT_INIT:
                if not self.ctx_chunk_config:
                    # No Chunking: Greedy allocation
                    req_num_tokens = req.get_context_remaining_length()
                    draft_tokens = req.num_draft_tokens if req.has_draft_tokens else 0
                    total_tokens = req_num_tokens + draft_tokens

                    if self.max_num_tokens is not None and (
                            current_batch_tokens + total_tokens
                            > self.max_num_tokens):
                        break

                    context_requests.append(req)
                    current_batch_tokens += total_tokens
                else:
                    # Chunking Enabled: Defer calculation
                    remaining = req.get_context_remaining_length()
                    # Just an estimate for budget check
                    req.context_chunk_size = remaining

                    draft_tokens = req.num_draft_tokens if (
                        req.is_last_context_chunk
                        and req.has_draft_tokens) else 0
                    req_num_tokens = remaining + draft_tokens

                    contexts_to_be_chunked.append(req)
                    num_chunked_tokens += req_num_tokens

            # Batch Size Check
            scheduled_req_count += 1
            if scheduled_req_count >= self.max_batch_size:
                break

        # 2. Check if chunking logic is needed
        if self.max_num_tokens is not None and num_chunked_tokens > (
                self.max_num_tokens - current_batch_tokens):
            all_context_fits = False

        # 3. Apply Chunking Strategy
        if not all_context_fits and contexts_to_be_chunked:
            if not self.ctx_chunk_config:
                # Should effectively be handled above, but as a fallback
                pass
            else:
                remaining_capacity = (
                    self.max_num_tokens - current_batch_tokens
                ) if self.max_num_tokens is not None else None
                self._set_ctx_requests_chunk_size(contexts_to_be_chunked,
                                                  remaining_capacity)

        # 4. Finalize Context Requests
        for req in contexts_to_be_chunked:
            if req.context_chunk_size > 0:
                context_requests.append(req)
                current_batch_tokens += req.context_chunk_size

        return context_requests, generation_requests

    def _set_ctx_requests_chunk_size(self, requests: RequestList,
                                     capacity: Optional[int]):
        # Reset
        for req in requests:
            req.context_chunk_size = 0

        policy = self.ctx_chunk_config.chunking_policy
        unit_size = self.ctx_chunk_config.chunk_unit_size

        if policy == ChunkingPolicy.EQUAL_PROGRESS:
            self._chunk_equal_progress(requests, capacity, unit_size)
        elif policy == ChunkingPolicy.FIRST_COME_FIRST_SERVED:
            self._chunk_fcfs(requests, capacity, unit_size)

        # Optimization: Fit draft tokens if space allows
        self._fit_draft_tokens(requests, capacity, unit_size)

    def _chunk_equal_progress(self, requests, capacity, unit_size):
        num_ctx_tokens = 0
        made_progress = True

        while (capacity is None or num_ctx_tokens < capacity) and made_progress:
            made_progress = False
            for req in requests:
                past_size = req.context_chunk_size
                remaining = req.get_context_remaining_length()

                if past_size >= remaining:
                    continue

                suggested_size = past_size + unit_size
                actual_size = min(suggested_size, remaining)
                increment = actual_size - past_size

                if increment > 0:
                    if capacity is not None and (num_ctx_tokens + increment
                                                 > capacity):
                        # Cannot fit this increment, stop growing this request
                        req.context_chunk_size = past_size
                        continue

                    req.context_chunk_size = actual_size
                    num_ctx_tokens += increment
                    made_progress = True

    def _chunk_fcfs(self, requests, capacity, unit_size):
        current_capacity = capacity if capacity is not None else float('inf')

        for req in requests:
            remaining = req.get_context_remaining_length()
            actual_size = remaining

            if current_capacity < actual_size:
                actual_size = current_capacity

            # Align if truncated
            if actual_size < remaining:
                actual_size = (int(actual_size) // unit_size) * unit_size

            req.context_chunk_size = int(actual_size)
            current_capacity -= req.context_chunk_size

            if current_capacity <= 0:
                break

    def _fit_draft_tokens(self, requests, capacity, unit_size):
        # Python port of fitDraftTokens
        # Logic: If it is the last chunk, try to fit draft tokens without using a new KV block
        current_tokens = sum(r.context_chunk_size for r in requests)

        for req in requests:
            if req.is_last_context_chunk and req.has_draft_tokens:
                chunk_size = req.context_chunk_size
                remainder = chunk_size % unit_size
                # Space left in the last block
                space_in_block = 0 if remainder == 0 else (unit_size -
                                                           remainder)

                # Check constraints
                allowed_space = space_in_block
                if capacity is not None:
                    allowed_space = min(allowed_space,
                                        capacity - current_tokens)

                # If we can't fit all draft tokens in the existing block/capacity, discard them
                draft_needed = req.num_draft_tokens
                if draft_needed > allowed_space:
                    # In python we might need a method to discard/update draft tokens on req
                    # req.discard_draft_tokens(draft_needed - allowed_space)
                    pass
                else:
                    current_tokens += draft_needed


class PyCapacityScheduler:
    """
    Python implementation of the C++ CapacityScheduler.
    """

    def __init__(
        self,
        max_num_requests: int,
        kv_cache_manager: KVCacheManager,
        scheduler_policy: CapacitySchedulerPolicy = CapacitySchedulerPolicy.
        MAX_UTILIZATION,
        no_schedule_until_state=LlmRequestState.CONTEXT_INIT,
        no_schedule_after_state=LlmRequestState.GENERATION_COMPLETE,
    ):
        self.max_num_requests = max_num_requests
        self.kv_cache_manager = kv_cache_manager
        # Use the underlying C++ object via bindings
        self.kv_cache_manager_cpp = kv_cache_manager.impl
        self.policy = scheduler_policy
        self.no_schedule_until_state = no_schedule_until_state
        self.no_schedule_after_state = no_schedule_after_state

        # [FIX]: Get this from config/wrapper to ensure C++ API compatibility
        self.default_window_size = self.kv_cache_manager.max_seq_len

    def schedule_request(
        self, active_requests: RequestList
    ) -> Tuple[RequestList, RequestList, RequestList]:

        if self.policy == CapacitySchedulerPolicy.MAX_UTILIZATION:
            return self._schedule_max_utilization(active_requests)
        elif self.policy == CapacitySchedulerPolicy.GUARANTEED_NO_EVICT:
            return self._schedule_guaranteed_no_evict(active_requests)
        else:
            raise NotImplementedError(
                f"Policy {self.policy} not implemented in PyCapacityScheduler")

    def _schedule_max_utilization(self, active_requests: RequestList):
        scheduled_requests: RequestList = []
        paused_requests: RequestList = []

        # Track free blocks manually in Python to simulate transactional state.
        # get_num_free_blocks() returns the current physical free blocks.
        # We subtract from this as we tentatively schedule requests.
        current_free_blocks = self.kv_cache_manager_cpp.get_num_free_blocks()

        cached_active_list = list(active_requests)
        idx = 0

        while idx < len(cached_active_list):
            req = cached_active_list[idx]

            # 1. State Filter
            # Allow Disagg Gen Init to pass through (matching C++ logic)
            is_disagg_init = (
                req.state == LlmRequestState.DISAGG_GENERATION_INIT)

            if not is_disagg_init and (
                    req.state.value < self.no_schedule_until_state.value
                    or req.state.value >= self.no_schedule_after_state.value):
                idx += 1
                continue

            # 2. Max Requests Check
            if len(scheduled_requests) >= self.max_num_requests:
                break

            # 3. Try Allocation (Python Manual Check)
            # Use get_needed_blocks_one_step to calculate incremental need
            needed_blocks = 0
            if is_disagg_init:
                # Disagg Init needs special calculation usually same as Context
                needed_blocks = self.kv_cache_manager_cpp.get_needed_blocks_one_step(
                    req, False, self.default_window_size)
            elif req.state == LlmRequestState.GENERATION_IN_PROGRESS:
                # Generation usually needs 0 or 1 block depending on boundary
                needed_blocks = self.kv_cache_manager_cpp.get_needed_blocks_one_step(
                    req, False, self.default_window_size)
            elif req.state == LlmRequestState.CONTEXT_INIT:
                needed_blocks = self.kv_cache_manager_cpp.get_needed_blocks_one_step(
                    req, False, self.default_window_size)

            if current_free_blocks >= needed_blocks:
                # Commit locally (transactional update)
                current_free_blocks -= needed_blocks
                scheduled_requests.append(req)
                idx += 1
                continue

            # 4. Backtracking / Eviction Logic
            # If current request doesn't fit, try to evict a previously scheduled
            # GENERATION request to make room.
            victim_idx = -1
            for i in range(len(scheduled_requests) - 1, -1, -1):
                r = scheduled_requests[i]
                if r.state == LlmRequestState.GENERATION_IN_PROGRESS:
                    victim_idx = i
                    break

            if victim_idx != -1:
                # Found a victim. Evict it.
                victim_req = scheduled_requests.pop(victim_idx)
                paused_requests.append(victim_req)

                # Reclaim victim's blocks manually
                # We simply give back what we subtracted earlier for this victim.
                victim_needed = self.kv_cache_manager_cpp.get_needed_blocks_one_step(
                    victim_req, False, self.default_window_size)
                current_free_blocks += victim_needed

                # Retry current req (do NOT increment idx)
                continue
            else:
                # No victim found, and current request doesn't fit. Stop.
                break

        # 5. Output Classification
        fitting_requests = []
        fitting_disagg_gen_init = []

        scheduled_ids = set(r.request_id for r in scheduled_requests)
        evicted_ids = set(r.request_id for r in paused_requests)

        for req in active_requests:
            if (req.state == LlmRequestState.GENERATION_IN_PROGRESS
                    and req.request_id not in scheduled_ids
                    and req.request_id not in evicted_ids):
                # Request was running but dropped due to capacity/limit
                paused_requests.append(req)

        for r in scheduled_requests:
            if r.state == LlmRequestState.DISAGG_GENERATION_INIT:
                fitting_disagg_gen_init.append(r)
            else:
                fitting_requests.append(r)

        return fitting_requests, fitting_disagg_gen_init, paused_requests

    def _schedule_guaranteed_no_evict(self, active_requests: RequestList):
        scheduled_requests: RequestList = []
        pending_requests: RequestList = []

        max_blocks = self.kv_cache_manager_cpp.max_num_blocks

        # [FIX] Must subtract used blocks to get available for *new* allocations
        used_blocks = self.kv_cache_manager_cpp.get_used_num_blocks()

        # We track 'reserved' as blocks we PLAN to add on top of 'used'
        available_blocks = max_blocks - used_blocks

        # --- Pass 1: Running Requests ---
        for request in active_requests:
            req_state = request.state

            if (req_state.value < self.no_schedule_until_state.value
                    or req_state.value >= self.no_schedule_after_state.value):
                continue

            if len(scheduled_requests) >= self.max_num_requests:
                pending_requests.append(request)
                continue

            if (req_state == LlmRequestState.GENERATION_IN_PROGRESS
                    or req_state == LlmRequestState.GENERATION_TO_COMPLETE):

                # [FIX] Added window_size argument
                needed = self.kv_cache_manager_cpp.get_remaining_blocks_to_completion(
                    request, self.default_window_size)

                if needed > available_blocks:
                    # System Overload (should ideally not happen in NoEvict)
                    pending_requests.append(request)
                    continue

                scheduled_requests.append(request)
                available_blocks -= needed
            else:
                # Defer Context/Init requests to Pass 2
                pending_requests.append(request)

        # --- Pass 2: New / Context Requests ---
        for request in pending_requests:
            if len(scheduled_requests) >= self.max_num_requests:
                break

            if (request.state == LlmRequestState.CONTEXT_INIT
                    or request.state == LlmRequestState.DISAGG_GENERATION_INIT):

                # [FIX] Added window_size argument
                needed_blocks = self.kv_cache_manager_cpp.get_remaining_blocks_to_completion(
                    request, self.default_window_size)

                if needed_blocks <= available_blocks:
                    scheduled_requests.append(request)
                    available_blocks -= needed_blocks
                else:
                    # Cannot fit new request. Stop.
                    break

        # --- Output Classification ---
        fitting_requests = []
        fitting_disagg_gen_init = []
        paused_requests = []

        scheduled_ids = set(r.request_id for r in scheduled_requests)

        # Identify running requests that were implicitly paused
        for req in active_requests:
            if (req.request_id not in scheduled_ids
                    and req.state == LlmRequestState.GENERATION_IN_PROGRESS):
                paused_requests.append(req)

        for req in scheduled_requests:
            if req.state == LlmRequestState.DISAGG_GENERATION_INIT:
                fitting_disagg_gen_init.append(req)
            else:
                fitting_requests.append(req)

        return fitting_requests, fitting_disagg_gen_init, paused_requests


class SimpleUnifiedScheduler(RequestScheduler):

    def __init__(
        self,
        max_batch_size: int,
        max_num_tokens: int,
        kv_cache_manager,
        scheduler_policy: CapacitySchedulerPolicy,
        ctx_chunk_config: Optional[Tuple[StrEnum, int]] = None,
    ):
        # 1. Initialize Python Capacity Scheduler
        self.capacity_scheduler = PyCapacityScheduler(
            max_num_requests=max_batch_size,
            kv_cache_manager=kv_cache_manager,
            scheduler_policy=scheduler_policy)

        # 2. Initialize Python MicroBatch Scheduler
        py_chunk_config = None
        if ctx_chunk_config:
            # Convert StrEnum to our Python Enum
            policy_enum = ChunkingPolicy.EQUAL_PROGRESS if ctx_chunk_config[
                0] == tb_internal.batch_manager.ChunkingPolicy.EQUAL_PROGRESS else ChunkingPolicy.FIRST_COME_FIRST_SERVED
            py_chunk_config = ContextChunkingConfig(policy_enum,
                                                    ctx_chunk_config[1])

        self.micro_batch_scheduler = PyMicroBatchScheduler(
            max_batch_size=max_batch_size,
            max_num_tokens=max_num_tokens,
            ctx_chunk_config=py_chunk_config)

    def schedule_request(self, active_requests: RequestList,
                         inflight_request_ids: set[int]) -> SchedulerOutput:
        # Step 1: Capacity Check (Who fits in memory?)
        fitting_requests, fitting_disagg_gen_init, paused_requests = self.capacity_scheduler.schedule_request(
            active_requests)

        # Step 2: MicroBatch Check (Who fits in token budget? + Chunking)
        context_requests, generation_requests = self.micro_batch_scheduler.schedule(
            fitting_requests, inflight_request_ids)

        return SchedulerOutput(
            context_requests=context_requests,
            generation_requests=generation_requests,
            paused_requests=paused_requests,
            fitting_disagg_gen_init_requests=fitting_disagg_gen_init,
            num_fitting_requests=len(fitting_requests))

    def can_schedule(self, requests: RequestList) -> bool:
        # Dry run capacity check
        fitting, _, _ = self.capacity_scheduler.schedule_request(requests)
        return len(fitting) == len(requests)
