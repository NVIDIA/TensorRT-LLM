import dataclasses
from abc import ABC, abstractmethod
from collections import namedtuple
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Set, Tuple

from strenum import StrEnum

from tensorrt_llm.bindings import internal as tb_internal
from tensorrt_llm.llmapi.llm_args import CapacitySchedulerPolicy

# Assuming these imports exist in your environment
from .llm_request import LlmRequest, LlmRequestState

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


class MicroBatchScheduler:
    """Base class to match structure."""


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
        self.max_context_length = max_num_tokens

    def schedule(
            self, active_requests: RequestList,
            inflight_request_ids: Set[int]) -> Tuple[RequestList, RequestList]:

        context_requests: RequestList = []
        generation_requests: RequestList = []

        # Current total tokens in the scheduled batch (Generation + Context)
        batch_num_tokens = 0
        scheduled_req_size = 0
        scheduled_beam_width = 0

        contexts_to_be_chunked: RequestList = []
        # Total tokens required by chunked requests (calculated tentatively)
        num_chunked_tokens = 0
        all_context_requests_fit = True

        # 1. Main Scheduling Loop
        for req in active_requests:
            # Skip requests already in flight (should be filtered by caller, but C++ checks)
            if req.request_id in inflight_request_ids:
                continue

            req_num_tokens = 0

            # --- A. Encoder Request Handling (Previously Missing) ---
            if req.state == LlmRequestState.ENCODER_INIT:
                # C++: reqNumTokens = llmReq->getEncoderOutputLen();
                req_num_tokens = req.encoder_output_len

                if self.max_context_length is not None and req_num_tokens > self.max_context_length:
                    # C++ does TLLM_CHECK here. We skip or log.
                    continue

                # Check Batch Token Budget
                if self.max_num_tokens is not None and (batch_num_tokens +
                                                        req_num_tokens
                                                        > self.max_num_tokens):
                    break

                context_requests.append(req)
                batch_num_tokens += req_num_tokens

            # --- B. Context Request Handling ---
            elif req.state == LlmRequestState.CONTEXT_INIT:
                if not self.ctx_chunk_config:
                    # No Chunking: Schedule full context
                    # C++: getNumTokens(beam) + (hasDraft ? getNumDraftTokens : 0)
                    base_tokens = req.context_remaining_length  # effectively getNumTokens(0)
                    draft_tokens = req.num_draft_tokens if req.has_draft_tokens else 0
                    req_num_tokens = base_tokens + draft_tokens

                    if self.max_context_length is not None and req_num_tokens > self.max_context_length:
                        continue

                    if self.max_num_tokens is not None and (
                            batch_num_tokens + req_num_tokens
                            > self.max_num_tokens):
                        break

                    context_requests.append(req)
                    batch_num_tokens += req_num_tokens
                else:
                    # Chunking Enabled: Tentative schedule
                    # C++: setContextChunkSize(remaining); reqNumTokens = size + draft
                    req.context_chunk_size = req.context_remaining_length

                    draft_tokens = req.num_draft_tokens if (
                        req.is_last_context_chunk
                        and req.has_draft_tokens) else 0
                    req_num_tokens = req.context_chunk_size + draft_tokens

                    # C++: Check maxContextLength constraints
                    if self.max_context_length is not None:
                        if self.max_context_length < req_num_tokens:
                            req_num_tokens = self.max_context_length
                            all_context_requests_fit = False

                    contexts_to_be_chunked.append(req)
                    num_chunked_tokens += req_num_tokens

            # --- C. Generation Request Handling ---
            else:
                beam_width = req.sampling_config.beam_width
                req_num_tokens = beam_width + req.num_draft_tokens

                if self.max_num_tokens is not None and (batch_num_tokens +
                                                        req_num_tokens
                                                        > self.max_num_tokens):
                    break

                # Beam Width Consistency Check (C++ Logic)
                if scheduled_beam_width == 0:
                    scheduled_beam_width = beam_width
                elif scheduled_beam_width != beam_width:
                    # Skip requests with different beam width in this batch
                    continue

                generation_requests.append(req)
                batch_num_tokens += req_num_tokens

            # --- Batch Size Limit Check ---
            scheduled_req_size += 1
            if scheduled_req_size >= self.max_batch_size:
                break

        # 2. Verify Chunking Fits
        if self.max_num_tokens is not None and num_chunked_tokens > (
                self.max_num_tokens - batch_num_tokens):
            all_context_requests_fit = False

        # 3. Apply Chunking Strategy if needed
        if not all_context_requests_fit and contexts_to_be_chunked:
            if not self.ctx_chunk_config:
                pass  # Error in C++: "If chunking not enabled..."
            else:
                remaining_capacity = (
                    self.max_num_tokens - batch_num_tokens
                ) if self.max_num_tokens is not None else None

                self._set_ctx_requests_chunk_size(contexts_to_be_chunked,
                                                  remaining_capacity)

        # 4. Finalize Chunked Requests
        for req in contexts_to_be_chunked:
            if req.context_chunk_size > 0:
                context_requests.append(req)
                # C++: batchNumTokens += chunk size
                batch_num_tokens += req.context_chunk_size

        # Note: C++ calls utils::sortRequests here. Python lists preserve order,
        # usually acceptable unless specific downstream kernel requirements exist.

        return context_requests, generation_requests

    def _set_ctx_requests_chunk_size(self, requests: RequestList,
                                     capacity: Optional[int]):
        # C++: Resets all chunk sizes to 0 at start
        for req in requests:
            req.context_chunk_size = 0

        policy = self.ctx_chunk_config.chunking_policy
        unit_size = self.ctx_chunk_config.chunk_unit_size

        if policy == ChunkingPolicy.EQUAL_PROGRESS:
            self._chunk_equal_progress(requests, capacity, unit_size)
        elif policy == ChunkingPolicy.FIRST_COME_FIRST_SERVED:
            self._chunk_fcfs(requests, capacity, unit_size)

        self._fit_draft_tokens(requests, capacity, unit_size)

    def _chunk_equal_progress(self, requests: RequestList,
                              capacity: Optional[int], unit_size: int):
        num_ctx_tokens = 0
        num_tokens_single_loop = 1

        # C++ Loop: while ((!capacity || numCtxTokens < capacity) && numTokensSingleLoop)
        while (capacity is None
               or num_ctx_tokens < capacity) and num_tokens_single_loop > 0:
            num_tokens_single_loop = 0
            for req in requests:
                past_size = req.context_chunk_size

                # C++ logic: suggested = past + unit
                suggested_size = past_size + unit_size

                # Ensure we don't exceed what the request actually needs
                remaining_total = req.context_remaining_length
                suggested_size = min(suggested_size, remaining_total)

                req.context_chunk_size = suggested_size

                actual_size = req.context_chunk_size
                actual_increment = actual_size - past_size

                # Check Constraints
                # 1. Capacity
                if capacity is not None and (num_ctx_tokens + actual_increment
                                             > capacity):
                    req.context_chunk_size = past_size  # Revert
                    continue

                # 2. Max Context Length
                if self.max_context_length is not None and actual_size > self.max_context_length:
                    req.context_chunk_size = past_size  # Revert
                    continue

                num_ctx_tokens += actual_increment
                num_tokens_single_loop += actual_increment

    def _chunk_fcfs(self, requests: RequestList, capacity: Optional[int],
                    unit_size: int):
        current_capacity = capacity if capacity is not None else float('inf')

        for req in requests:
            suggested_size = req.context_remaining_length
            actual_size = suggested_size

            if current_capacity < actual_size:
                actual_size = current_capacity

            if self.max_context_length is not None:
                actual_size = min(self.max_context_length, actual_size)

            # Round down to unit size if we had to truncate
            if actual_size < suggested_size:
                actual_size = (int(actual_size) // unit_size) * unit_size

            req.context_chunk_size = int(actual_size)

            # C++: ctxTokensCapacity = ctxTokensCapacity - actualChunkSize
            if capacity is not None:
                current_capacity -= req.context_chunk_size

    def _fit_draft_tokens(self, requests: RequestList, capacity: Optional[int],
                          unit_size: int):
        # Calculate tokens already taken by the batch so far
        num_ctx_tokens = sum(req.context_chunk_size for req in requests)

        for req in requests:
            if req.is_last_context_chunk and req.has_draft_tokens:
                remainder = req.context_chunk_size % unit_size
                remaining_space = 0 if remainder == 0 else unit_size - remainder

                if self.max_context_length is not None:
                    remaining_context_len = self.max_context_length - req.context_chunk_size
                    remaining_space = min(remaining_space,
                                          remaining_context_len)

                if capacity is not None:
                    remaining_space = min(remaining_space,
                                          capacity - num_ctx_tokens)
                    num_ctx_tokens += remaining_space

                draft_discard = req.num_draft_tokens - remaining_space
                if draft_discard > 0:
                    if hasattr(req, "discard_draft_tokens"):
                        req.discard_draft_tokens(draft_discard)


class PyCapacityScheduler:
    """
    Python implementation of the C++ CapacityScheduler.
    Aligned with C++ logic to support Multiple Window Sizes (VSWA).
    """

    def __init__(
        self,
        max_num_requests: int,
        kv_cache_manager,
        scheduler_policy: CapacitySchedulerPolicy = CapacitySchedulerPolicy.
        MAX_UTILIZATION,
        no_schedule_until_state=LlmRequestState.CONTEXT_INIT,
        no_schedule_after_state=LlmRequestState.GENERATION_COMPLETE,
    ):
        self.max_num_requests = max_num_requests
        self.kv_cache_manager = kv_cache_manager
        self.policy = scheduler_policy
        self.no_schedule_until_state = no_schedule_until_state
        self.no_schedule_after_state = no_schedule_after_state

        if self.kv_cache_manager is not None:
            self.kv_cache_manager_cpp = kv_cache_manager.impl
            self.default_window_size = self.kv_cache_manager.max_seq_len

    def schedule_request(
        self, active_requests: RequestList
    ) -> Tuple[RequestList, RequestList, RequestList]:

        if self.kv_cache_manager is None:
            return self._schedule_max_requests(active_requests)

        if self.policy == CapacitySchedulerPolicy.MAX_UTILIZATION:
            return self._schedule_max_utilization(active_requests)
        elif self.policy == CapacitySchedulerPolicy.GUARANTEED_NO_EVICT:
            return self._schedule_guaranteed_no_evict(active_requests)
        else:
            raise NotImplementedError(
                f"Policy {self.policy} not implemented in PyCapacityScheduler")

    def _get_initial_available_blocks_map(self) -> Dict[int, int]:
        """
        Mimics C++: mKvCacheManager.getBlockManager().getNumFreeBlocksPerWindowSize()
        Returns a dict {window_size: free_blocks}.
        """
        stats = self.kv_cache_manager_cpp.get_kv_cache_stats()

        # Nanobind binds std::map to python dict
        # Property name from binding: .def_rw("num_free_blocks_per_window_size", ...)
        free_map = stats.num_free_blocks_per_window_size

        if not free_map:
            # Fallback for simple cases or if map is empty (though unlikely in C++)
            # Calculate scalar free blocks
            free_scalar = stats.max_num_blocks - stats.used_num_blocks
            return {self.default_window_size: free_scalar}

        # Ensure we return a copy so we can modify it during scheduling
        return dict(free_map)

    def _req_check_and_update_map(self, req: LlmRequest,
                                  available_map: Dict[int, int],
                                  is_guaranteed_no_evict: bool) -> bool:
        """
        Checks if a request fits in ALL window sizes tracked in available_map.
        If it fits, decrements the map and returns True.
        If it doesn't fit, leaves map untouched and returns False.
        """
        # 1. Calculate needed blocks for all window sizes
        needed_per_window = {}
        for window_size in available_map.keys():
            if is_guaranteed_no_evict:
                # C++: getRemainingBlocksToCompletion(req, windowSize)
                needed = self.kv_cache_manager_cpp.get_remaining_blocks_to_completion(
                    req, window_size)
            else:
                # C++: getNeededBlocksOneStep(req, twoStepsLookAhead, windowSize)
                needed = self.kv_cache_manager_cpp.get_needed_blocks_one_step(
                    req, False, window_size)
            needed_per_window[window_size] = needed

        # 2. Check if fits (All or Nothing)
        for window_size, available in available_map.items():
            if needed_per_window[window_size] > available:
                return False

        # 3. Commit update
        for window_size in available_map.keys():
            available_map[window_size] -= needed_per_window[window_size]

        return True

    def _req_force_update_map(self, req: LlmRequest, available_map: Dict[int,
                                                                         int],
                              is_guaranteed_no_evict: bool):
        """
        Unconditionally decrements the available blocks (used for Running requests in NoEvict).
        Allowed to go negative.
        """
        for window_size in available_map.keys():
            if is_guaranteed_no_evict:
                needed = self.kv_cache_manager_cpp.get_remaining_blocks_to_completion(
                    req, window_size)
            else:
                needed = self.kv_cache_manager_cpp.get_needed_blocks_one_step(
                    req, False, window_size)

            available_map[window_size] -= needed

    def _req_revert_map(self, req: LlmRequest, available_map: Dict[int, int],
                        is_guaranteed_no_evict: bool):
        """
        Reverts a decrement (used for Backtracking in MaxUtilization).
        """
        for window_size in available_map.keys():
            if is_guaranteed_no_evict:
                needed = self.kv_cache_manager_cpp.get_remaining_blocks_to_completion(
                    req, window_size)
            else:
                needed = self.kv_cache_manager_cpp.get_needed_blocks_one_step(
                    req, False, window_size)

            available_map[window_size] += needed

    def _schedule_max_requests(self, active_requests: RequestList):
        scheduled_requests: RequestList = []
        paused_requests: RequestList = []

        for req in active_requests:
            is_disagg_init = (
                req.state == LlmRequestState.DISAGG_GENERATION_INIT)

            if not is_disagg_init and (
                    req.state.value < self.no_schedule_until_state.value
                    or req.state.value >= self.no_schedule_after_state.value):
                continue

            if len(scheduled_requests) >= self.max_num_requests:
                break

            if (req.state == LlmRequestState.ENCODER_INIT
                    or req.state == LlmRequestState.CONTEXT_INIT
                    or req.state == LlmRequestState.GENERATION_IN_PROGRESS
                    or is_disagg_init):
                scheduled_requests.append(req)

        return self._classify_output(active_requests, scheduled_requests,
                                     paused_requests)

    def _schedule_max_utilization(self, active_requests: RequestList):
        scheduled_requests: RequestList = []
        paused_requests: RequestList = []

        if hasattr(self.kv_cache_manager, "start_scheduling"):
            self.kv_cache_manager.start_scheduling()

        # [FIX] Use Map tracking for multiple window sizes
        current_free_blocks_map = self._get_initial_available_blocks_map()

        cached_active_list = list(active_requests)
        idx = 0

        while idx < len(cached_active_list):
            req = cached_active_list[idx]

            is_disagg_init = (
                req.state == LlmRequestState.DISAGG_GENERATION_INIT)

            if not is_disagg_init and (
                    req.state.value < self.no_schedule_until_state.value
                    or req.state.value >= self.no_schedule_after_state.value):
                idx += 1
                continue

            if len(scheduled_requests) >= self.max_num_requests:
                break

            # 3. Try Allocation
            # C++ Logic: Checks if it fits in *all* window sizes
            if self._req_check_and_update_map(req,
                                              current_free_blocks_map,
                                              is_guaranteed_no_evict=False):
                scheduled_requests.append(req)
                idx += 1
                continue

            # 4. Backtracking (Evict Generation requests only)
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

                # Revert victim's usage in the map
                self._req_revert_map(victim_req,
                                     current_free_blocks_map,
                                     is_guaranteed_no_evict=False)

                # Retry current req (do NOT increment idx)
                continue
            else:
                # No victim found, and current request doesn't fit. Stop.
                break

        return self._classify_output(active_requests, scheduled_requests,
                                     paused_requests)

    def _schedule_guaranteed_no_evict(self, active_requests: RequestList):
        scheduled_requests: RequestList = []
        pending_disagg_requests: RequestList = []
        pending_context_requests: RequestList = []

        # [FIX] Use Map tracking for multiple window sizes
        # Note: C++ NoEvictScheduledBlocksManager initializes with getNumFreeBlocksPerWindowSize()
        available_blocks_map = self._get_initial_available_blocks_map()

        # --- Pass 1: Running Requests ---
        for request in active_requests:
            req_state = request.state
            is_disagg_init = (
                req_state == LlmRequestState.DISAGG_GENERATION_INIT)

            if not is_disagg_init and (
                    req_state.value < self.no_schedule_until_state.value
                    or req_state.value >= self.no_schedule_after_state.value):
                continue

            if len(scheduled_requests) >= self.max_num_requests:
                if is_disagg_init:
                    pending_disagg_requests.append(request)
                else:
                    pending_context_requests.append(request)
                continue

            # Unconditionally schedule Running Requests
            if (req_state == LlmRequestState.GENERATION_IN_PROGRESS
                    or req_state == LlmRequestState.GENERATION_TO_COMPLETE):

                scheduled_requests.append(request)

                # [FIX] Update Map unconditionally (can go negative)
                self._req_force_update_map(request,
                                           available_blocks_map,
                                           is_guaranteed_no_evict=True)
            else:
                if is_disagg_init:
                    pending_disagg_requests.append(request)
                else:
                    pending_context_requests.append(request)

        # --- Pass 2: New / Context Requests (Disagg First) ---
        all_pending = pending_disagg_requests + pending_context_requests

        for request in all_pending:
            if len(scheduled_requests) >= self.max_num_requests:
                break

            # [FIX] Check using Map logic
            # C++ enoughAvailableBlocks checks: needed <= available for ALL window sizes
            if self._req_check_and_update_map(request,
                                              available_blocks_map,
                                              is_guaranteed_no_evict=True):
                scheduled_requests.append(request)
            else:
                # Head-of-line blocking
                break

        return self._classify_output(active_requests, scheduled_requests, [])

    def _classify_output(self, active_requests, scheduled_requests,
                         explicit_paused_requests):
        fitting_requests = []
        fitting_disagg_gen_init = []
        paused_requests = list(explicit_paused_requests)

        scheduled_ids = set(r.request_id for r in scheduled_requests)
        paused_ids = set(r.request_id for r in paused_requests)

        for req in active_requests:
            if (req.request_id not in scheduled_ids
                    and req.request_id not in paused_ids
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
            # Fix: Use string comparison to identify the policy.
            # This works regardless of whether the input is a Python Enum, C++ Binding Enum, or String.
            input_policy = ctx_chunk_config[0]

            if "EQUAL_PROGRESS" in str(input_policy):
                policy_enum = ChunkingPolicy.EQUAL_PROGRESS
            else:
                # Default to FCFS for FIRST_COME_FIRST_SERVED or others
                policy_enum = ChunkingPolicy.FIRST_COME_FIRST_SERVED

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
