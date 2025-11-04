import dataclasses
from abc import ABC, abstractmethod
from collections import namedtuple
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from strenum import StrEnum

from tensorrt_llm.bindings import internal as tb_internal
from tensorrt_llm.llmapi.llm_args import CapacitySchedulerPolicy
from tensorrt_llm.logger import logger

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
    ) -> tuple[ScheduledRequests, RequestList, int]:
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


class KVCacheV2DummyScheduler(CapacityScheduler):
    # only schedule requests has no_schedule_until_state <= state < no_schedule_after_state
    no_schedule_until_state = LlmRequestState.CONTEXT_INIT
    no_schedule_after_state = LlmRequestState.GENERATION_COMPLETE

    def __init__(self, max_num_requests: int, kv_cache_manager):
        super(KVCacheV2DummyScheduler, self).__init__()
        self.max_num_requests = max_num_requests
        self.kv_cache_manager = kv_cache_manager

    def schedule_request(
        self, active_requests: RequestList
    ) -> tuple[list[LlmRequest], list[LlmRequest], list[LlmRequest]]:
        scheduled_requests = []
        scheduled_disagg_gen_init_requests = []
        pending_requests = []
        reserved_blocks = 0
        max_blocks = self.kv_cache_manager.get_max_resource_count()
        for request in active_requests:
            req_state = request.state
            # if request cannot be scheduled yet or request should no longer be scheduled, skip
            if not req_state == LlmRequestState.DISAGG_GENERATION_INIT and (
                    req_state.value < self.no_schedule_until_state.value
                    or req_state.value >= self.no_schedule_after_state.value):
                continue

            if len(scheduled_requests
                   ) >= self.max_num_requests or reserved_blocks >= max_blocks:
                break
            elif req_state == LlmRequestState.GENERATION_IN_PROGRESS or req_state == LlmRequestState.GENERATION_TO_COMPLETE:
                scheduled_requests.append(request)
                reserved_blocks += self.kv_cache_manager.get_needed_resource_to_completion(
                    request)
            elif req_state == LlmRequestState.DISAGG_GENERATION_INIT:
                scheduled_disagg_gen_init_requests.append(request)
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

        assert len(scheduled_requests) + len(
            scheduled_disagg_gen_init_requests) > 0, (
                "no pending request can get enough resource to complete, "
                "please increase KV cache pool size.")
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
        # to be aligned with MicroBatchScheduler::scheduleRequests in cpp/tensorrt_llm/batch_manager/microBatchScheduler.h
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
        no_schedule_until_state: LlmRequestState = LlmRequestState.CONTEXT_INIT,
        no_schedule_after_state: LlmRequestState = LlmRequestState.
        GENERATION_TO_COMPLETE,
    ):
        super().__init__()
        self.max_batch_size = max_batch_size
        self.max_num_tokens = max_num_tokens
        self.ctx_chunk_config = ctx_chunk_config
        self.max_context_length = max_num_tokens
        # Match C++ MicroBatchScheduler defaults (see algorithms.cpp line 68-70)
        self.no_schedule_until_state = no_schedule_until_state
        self.no_schedule_after_state = no_schedule_after_state
        # Cache state values to avoid repeated .value access (optimization)
        self._no_schedule_until_state_value = no_schedule_until_state.value
        self._no_schedule_after_state_value = no_schedule_after_state.value
        self._context_init_state_value = LlmRequestState.CONTEXT_INIT.value
        self._encoder_init_state_value = LlmRequestState.ENCODER_INIT.value

    def _can_be_scheduled(self, req: LlmRequest) -> bool:
        """
        Check if request is within the schedulable state range.
        C++ reference: microBatchScheduler.cpp line 192-195
        Optimized: use state_value property to avoid enum object creation
        """
        # Use state_value property (returns int directly, avoids enum object creation)
        state_value = req.state_value
        # Inline comparison: must have reached until_state but not after_state
        return (state_value >= self._no_schedule_until_state_value
                and state_value < self._no_schedule_after_state_value)

    def schedule(
            self, active_requests: RequestList,
            inflight_request_ids: set[int]) -> tuple[RequestList, RequestList]:

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

        # Cache instance attributes as locals for faster access in loop
        max_batch_size = self.max_batch_size
        max_num_tokens = self.max_num_tokens
        max_context_length = self.max_context_length
        ctx_chunk_config = self.ctx_chunk_config

        # 1. Main Scheduling Loop
        for req in active_requests:
            req_state_value = req.state_value
            # Skip requests already in flight (should be filtered by caller, but C++ checks)
            if req.request_id in inflight_request_ids:
                continue

            # Skip if request cannot be scheduled yet or should no longer be scheduled, manually inline the condition to reuse req.state_value
            if not (req_state_value >= self._no_schedule_until_state_value
                    and req_state_value < self._no_schedule_after_state_value):
                continue

            req_num_tokens = 0

            # --- A. Encoder Request Handling ---
            if req_state_value == self._encoder_init_state_value:
                req_num_tokens = req.encoder_output_len

                assert max_context_length is None or req_num_tokens <= max_context_length, \
                    f"The number of encoder tokens ({req_num_tokens}) exceeds the limit value ({max_context_length})"

                if max_num_tokens is not None and (
                        batch_num_tokens + req_num_tokens > max_num_tokens):
                    break

                logger.debug(f"encoder request scheduled: ID {req.request_id}")
                context_requests.append(req)
                batch_num_tokens += req_num_tokens

            # --- B. Context Request Handling ---
            elif req_state_value == self._context_init_state_value:
                if not ctx_chunk_config:
                    # No Chunking: Schedule full context
                    # C++ uses getNumTokens(beam=0) which is tokens.size() - numPreDecodedTokens
                    base_tokens = req.get_num_tokens(0)
                    draft_tokens = req.num_draft_tokens if req.has_draft_tokens else 0
                    req_num_tokens = base_tokens + draft_tokens

                    assert max_context_length is None or req_num_tokens <= max_context_length, \
                        f"The number of context tokens ({req_num_tokens}) exceeds the limit value ({max_context_length})"

                    if max_num_tokens is not None and (
                            batch_num_tokens + req_num_tokens > max_num_tokens):
                        break

                    logger.debug(
                        f"context request scheduled: ID {req.request_id}")
                    context_requests.append(req)
                    batch_num_tokens += req_num_tokens
                else:
                    # Chunking Enabled: Tentative schedule
                    req.context_chunk_size = req.context_remaining_length

                    draft_tokens = req.num_draft_tokens if (
                        req.is_last_context_chunk
                        and req.has_draft_tokens) else 0
                    req_num_tokens = req.context_chunk_size + draft_tokens

                    if max_context_length is not None:
                        if max_context_length < req_num_tokens:
                            req_num_tokens = max_context_length
                            all_context_requests_fit = False

                    logger.debug(
                        f"contexts-to-be-chunked request scheduled: ID {req.request_id}"
                    )
                    contexts_to_be_chunked.append(req)
                    num_chunked_tokens += req_num_tokens

            # --- C. Generation Request Handling ---
            else:
                # C++ uses getBeamWidthByIter() which returns dynamic beam width
                # during beam search (1->2->3->...->beamWidth)
                beam_width = req.get_beam_width_by_iter(
                    for_next_iteration=False)
                req_num_tokens = beam_width + req.num_draft_tokens

                if max_num_tokens is not None and (
                        batch_num_tokens + req_num_tokens > max_num_tokens):
                    break

                # Beam Width Consistency Check
                if scheduled_beam_width == 0:
                    scheduled_beam_width = beam_width
                elif scheduled_beam_width != beam_width:
                    logger.debug(
                        f"generation request skipped: ID {req.request_id} since its "
                        f"beam width ({beam_width}) is different from scheduled ones "
                        f"({scheduled_beam_width})")
                    continue
                generation_requests.append(req)
                batch_num_tokens += req_num_tokens

            # --- Batch Size Limit Check ---
            scheduled_req_size += 1
            if scheduled_req_size >= max_batch_size:
                break

        # 2. Verify Chunking Fits
        if max_num_tokens is not None and num_chunked_tokens > (
                max_num_tokens - batch_num_tokens):
            all_context_requests_fit = False

        # 3. Apply Chunking Strategy if needed
        if not all_context_requests_fit and contexts_to_be_chunked:
            assert ctx_chunk_config is not None, \
                "If chunking is not enabled, context scheduling should be completed."
            remaining_capacity = (
                max_num_tokens -
                batch_num_tokens) if max_num_tokens is not None else None

            self._set_ctx_requests_chunk_size(contexts_to_be_chunked,
                                              remaining_capacity)

        # 4. Finalize Chunked Requests
        for req in contexts_to_be_chunked:
            if req.context_chunk_size > 0:
                context_requests.append(req)
                batch_num_tokens += req.context_chunk_size
                logger.debug(f"context request scheduled: ID {req.request_id}, "
                             f"chunk size {req.context_chunk_size}")

        # Sort requests for consistency with C++
        # C++ reference: utils::sortRequests in inflightBatchingUtils.cpp
        self._sort_requests(context_requests, generation_requests,
                            not all_context_requests_fit)

        # Summary logs
        logger.debug(f"batchSize (num ctx/enc requests + num gen requests): "
                     f"{len(context_requests) + len(generation_requests)}")
        logger.debug(f"batchNumTokens / maxNumTokens: {batch_num_tokens} / "
                     f"{max_num_tokens or 0}")

        return context_requests, generation_requests

    def _sort_requests(self, context_requests: RequestList,
                       generation_requests: RequestList,
                       chunks_present: bool) -> None:
        """
        Sort requests for consistency with C++.
        C++ reference: utils::sortRequests in inflightBatchingUtils.cpp

        1. If chunks are present, move context requests that reached the last
           context chunk to the end of the vector.
        2. Sort all requests by lora task id for performance.
        """

        def get_lora_task_id(req: LlmRequest):
            # C++ uses std::optional comparison where nullopt < any_value
            # So requests without LoRA (nullopt) should come first
            lora_id = getattr(req, 'lora_task_id', None)
            if lora_id is None:
                return (0, 0)  # (has_value=False, value=0) - comes first
            return (1, lora_id)  # (has_value=True, value) - sorted by value

        if chunks_present:
            # Partition: non-last-chunk first, last-chunk at end
            not_last_chunk = [
                r for r in context_requests if not r.is_last_context_chunk
            ]
            last_chunk = [
                r for r in context_requests if r.is_last_context_chunk
            ]
            # Sort each group by lora_task_id
            not_last_chunk.sort(key=get_lora_task_id)
            last_chunk.sort(key=get_lora_task_id)
            # Rebuild the list in-place
            context_requests.clear()
            context_requests.extend(not_last_chunk)
            context_requests.extend(last_chunk)
        else:
            context_requests.sort(key=get_lora_task_id)

        generation_requests.sort(key=get_lora_task_id)

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
        else:
            raise ValueError(f"Invalid chunking policy: {policy}")

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
                    logger.debug(f"Discarding {draft_discard} draft tokens")
                    if hasattr(req, "discard_draft_tokens"):
                        req.discard_draft_tokens(draft_discard)


class SchedulerPolicyBase(ABC):
    """
    Abstract base class for capacity scheduler policies.
    Each policy implements its own scheduling logic.
    """

    @abstractmethod
    def schedule(
            self, scheduler: 'PyCapacityScheduler',
            active_requests: RequestList) -> tuple[RequestList, RequestList]:
        """
        Schedule requests according to the policy.

        Args:
            scheduler: The capacity scheduler instance (for accessing shared state)
            active_requests: List of active requests to schedule

        Returns:
            Tuple of (scheduled_requests, paused_requests)
        """
        raise NotImplementedError


class MaxRequestsPolicy(SchedulerPolicyBase):
    """
    MaxRequestsScheduler: Simple request count limiting without KV cache checks.
    C++ reference: capacityScheduler.cpp:154-176
    """

    def schedule(
            self, scheduler: 'PyCapacityScheduler',
            active_requests: RequestList) -> tuple[RequestList, RequestList]:
        scheduled_requests: RequestList = []

        for req in active_requests:
            if not scheduler._can_be_scheduled(req):
                continue

            if len(scheduled_requests) >= scheduler.max_num_requests:
                break

            if (req.is_encoder_init_state or req.is_context_init_state
                    or req.is_generation_in_progress_state):
                scheduled_requests.append(req)

        return scheduled_requests, []


class GuaranteedNoEvictPolicy(SchedulerPolicyBase):
    """
    GuaranteedNoEvictScheduler: Reserve blocks for requests to complete without eviction.
    C++ reference: capacityScheduler.cpp:194-331
    """

    def __init__(self, static_batch: bool = False):
        self.static_batch = static_batch

    def schedule(
            self, scheduler: 'PyCapacityScheduler',
            active_requests: RequestList) -> tuple[RequestList, RequestList]:
        scheduled_requests: RequestList = []
        has_peft = scheduler.peft_cache_manager is not None

        skipping_is_relevant = scheduler._is_skipping_relevant()

        newly_contributed_context_blocks: Set = set()
        newly_contributed_cross_context_blocks: Set = set()
        if not self.static_batch and skipping_is_relevant:
            newly_contributed_context_blocks, newly_contributed_cross_context_blocks = \
                scheduler._prefill_contributed_blocks(active_requests)

        reserved_blocks = NoEvictScheduledBlocksManager(
            scheduler.kv_cache_manager)
        reserved_cross_blocks: Optional[NoEvictScheduledBlocksManager] = None
        if scheduler.cross_kv_cache_manager is not None:
            reserved_cross_blocks = NoEvictScheduledBlocksManager(
                scheduler.cross_kv_cache_manager)

        # PEFT state - only used when has_peft
        claimed_peft_pages = 0
        available_peft_pages = scheduler._get_max_peft_pages(
        ) if has_peft else 0
        uniq_task_ids: set[int] = set() if has_peft else None

        pending_requests: RequestList = []
        pending_dis_gen_init_requests: RequestList = []

        # First pass: process in-progress generation and classify requests
        for req in active_requests:
            if not scheduler._can_be_scheduled_with_disagg_exception(req):
                continue

            if len(scheduled_requests) >= scheduler.max_num_requests:
                break

            if req.is_generation_in_progress_state:
                scheduled_requests.append(req)
                reserved_blocks.decrement_reserved_blocks(req)
                if reserved_cross_blocks is not None:
                    reserved_cross_blocks.decrement_reserved_blocks(req)

                if has_peft:
                    lora_task_id, is_new_task, peft_pages = scheduler._get_peft_task_info(
                        req, uniq_task_ids)
                    if is_new_task:
                        claimed_peft_pages += peft_pages
                        uniq_task_ids.add(lora_task_id)

            elif req.is_disagg_generation_init_state:
                pending_dis_gen_init_requests.append(req)
            else:
                pending_requests.append(req)

        # Second pass: process pending requests
        if not self.static_batch or len(scheduled_requests) == 0:
            if has_peft:
                available_peft_pages -= claimed_peft_pages

            for requests in [pending_dis_gen_init_requests, pending_requests]:
                for req in requests:
                    if (not self.static_batch and skipping_is_relevant
                            and not req.is_disagg_generation_init_state
                            and scheduler._beneficial_to_skip(
                                req, newly_contributed_context_blocks,
                                newly_contributed_cross_context_blocks)):
                        continue

                    if len(scheduled_requests) >= scheduler.max_num_requests:
                        break

                    if req.is_context_init_state or req.is_disagg_generation_init_state:
                        enough_blocks = reserved_blocks.enough_available_blocks(
                            req)
                        enough_cross_blocks = True
                        if reserved_cross_blocks is not None:
                            enough_cross_blocks = reserved_cross_blocks.enough_available_blocks(
                                req)

                        if not enough_blocks or not enough_cross_blocks:
                            break

                        # PEFT check only when needed
                        if has_peft:
                            lora_task_id, is_new_task, needed_peft_pages = scheduler._get_peft_task_info(
                                req, uniq_task_ids)
                            if needed_peft_pages > available_peft_pages:
                                continue
                            available_peft_pages -= needed_peft_pages
                            if is_new_task:
                                uniq_task_ids.add(lora_task_id)

                        scheduled_requests.append(req)
                        reserved_blocks.decrement_reserved_blocks(req)
                        if reserved_cross_blocks is not None:
                            reserved_cross_blocks.decrement_reserved_blocks(req)

        return scheduled_requests, []


class MaxUtilizationPolicy(SchedulerPolicyBase):
    """
    MaxUtilizationScheduler: Maximize utilization, may pause started requests.
    C++ reference: capacityScheduler.cpp:341-425
    """

    def schedule(
            self, scheduler: 'PyCapacityScheduler',
            active_requests: RequestList) -> tuple[RequestList, RequestList]:
        scheduler.kv_cache_manager.start_scheduling()

        skipping_is_relevant = scheduler._is_skipping_relevant()

        scheduled_blocks_manager = MaxUtilizationScheduledBlocksManager(
            scheduler.kv_cache_manager, scheduler.two_step_lookahead)

        num_scheduled_peft_pages = 0
        seen_task_ids: set[int] = set()

        newly_contributed_context_blocks, _ = scheduler._prefill_contributed_blocks(
            active_requests)

        def is_started_request(req: LlmRequest) -> bool:
            if not scheduler._can_be_scheduled(req):
                return False
            return ((req.is_context_init_state
                     and not req.is_first_context_chunk)
                    or req.is_generation_in_progress_state)

        scheduled_requests: RequestList = []
        paused_requests: RequestList = []

        requests_list = list(active_requests)
        req_it_end = len(requests_list)
        req_it = 0

        while req_it < req_it_end:
            req = requests_list[req_it]
            logger.debug(
                f"MaxUtilizationScheduler: scheduling request ID {req.request_id}"
            )

            if not scheduler._can_be_scheduled_with_disagg_exception(req):
                logger.debug(
                    f"MaxUtilizationScheduler: request ID {req.request_id} "
                    "cannot / should not be scheduled")
                req_it += 1
                continue

            if (skipping_is_relevant and scheduler._beneficial_to_skip(
                    req, newly_contributed_context_blocks, set())):
                req_it += 1
                continue

            was_scheduled = self._try_scheduling_request(
                scheduler, req, scheduled_requests, scheduled_blocks_manager,
                num_scheduled_peft_pages, seen_task_ids)

            if was_scheduled:
                logger.debug(
                    f"MaxUtilizationScheduler: request ID {req.request_id} -> start"
                )
                req_it += 1
            else:
                last_started_idx = None
                for i in range(req_it_end - 1, req_it - 1, -1):
                    if is_started_request(requests_list[i]):
                        last_started_idx = i
                        break

                if last_started_idx is not None:
                    paused_req = requests_list[last_started_idx]
                    scheduler.kv_cache_manager.scheduling_remove_sequence(
                        paused_req.py_request_id)
                    paused_requests.append(paused_req)
                    logger.debug(
                        f"MaxUtilizationScheduler: request ID {paused_req.request_id} -> pause"
                    )
                    req_it_end = last_started_idx
                else:
                    break

        return scheduled_requests, paused_requests

    def _try_scheduling_request(
            self, scheduler: 'PyCapacityScheduler', req: LlmRequest,
            scheduled_requests: RequestList,
            scheduled_blocks_manager: 'MaxUtilizationScheduledBlocksManager',
            num_scheduled_peft_pages: int, seen_task_ids: set[int]) -> bool:
        if len(scheduled_requests) >= scheduler.max_num_requests:
            return False

        blocks_if_scheduled = scheduled_blocks_manager.prepare_blocks_if_schedulable(
            req)
        if blocks_if_scheduled is None:
            return False

        # PEFT check only when needed
        if scheduler.peft_cache_manager is not None:
            lora_task_id, is_new_task, num_required_peft_pages = scheduler._get_peft_task_info(
                req, seen_task_ids)
            logger.debug(
                f"MaxUtilizationScheduler: request ID {req.request_id} "
                f"required peft pages: {num_required_peft_pages}")
            max_peft_pages = scheduler._get_max_peft_pages()
            if num_required_peft_pages + num_scheduled_peft_pages > max_peft_pages:
                return False
            logger.debug(
                f"MaxUtilizationScheduler: scheduled peft pages: {num_required_peft_pages}"
            )
            if is_new_task:
                seen_task_ids.add(lora_task_id)

        scheduled_blocks_manager.update_scheduled_blocks(blocks_if_scheduled)
        scheduled_requests.append(req)
        return True


class NoEvictScheduledBlocksManager:
    """
    Python equivalent of C++ kv_cache_manager::NoEvictScheduledBlocksManager.
    Tracks available blocks per window size for GUARANTEED_NO_EVICT scheduling.

    Reference: cpp/tensorrt_llm/batch_manager/scheduledBlocksManager.h:29-62
    """

    def __init__(self, kv_cache_manager):
        """
        Initialize with free blocks from KVCacheManager.
        C++ equivalent: mAvailableBlocks = mKvCacheManager.getBlockManager().getNumFreeBlocksPerWindowSize()
        """
        self.kv_cache_manager = kv_cache_manager
        stats = kv_cache_manager.get_kv_cache_stats()
        self.available_blocks: dict[int, int] = dict(
            stats.num_free_blocks_per_window_size)

    def decrement_reserved_blocks(self, req: LlmRequest) -> None:
        """
        Decrement available blocks by the blocks needed to complete this request.
        C++ reference: scheduledBlocksManager.h:40-46
        """
        for window_size in self.available_blocks:
            needed = self.kv_cache_manager.get_remaining_blocks_to_completion(
                req, window_size)
            self.available_blocks[window_size] -= needed

    def enough_available_blocks(self, req: LlmRequest) -> bool:
        """
        Check if there are enough available blocks for this request across all window sizes.
        C++ reference: scheduledBlocksManager.h:48-57
        """
        return all(
            self.kv_cache_manager.get_remaining_blocks_to_completion(req, ws) <=
            avail for ws, avail in self.available_blocks.items())


class MaxUtilizationScheduledBlocksManager:
    """
    Python equivalent of C++ kv_cache_manager::MaxUtilizationScheduledBlocksManager.
    Tracks scheduled blocks per window size for MAX_UTILIZATION scheduling.

    Reference: cpp/tensorrt_llm/batch_manager/scheduledBlocksManager.h:64-117
    """

    def __init__(self, kv_cache_manager, two_steps_look_ahead: bool):
        """
        Initialize scheduled blocks count per window size.
        C++ equivalent: iterate windowSizes and set mNumScheduledBlocks[windowSize] = 0
        """
        self.kv_cache_manager = kv_cache_manager
        self.two_steps_look_ahead = two_steps_look_ahead
        window_sizes = set(kv_cache_manager.max_attention_window_vec)
        self.num_scheduled_blocks: dict[int, int] = {
            ws: 0
            for ws in window_sizes
        }

    def prepare_blocks_if_schedulable(
            self, req: LlmRequest) -> Optional[dict[int, int]]:
        """
        Check if request can be scheduled and return new block counts if so.
        Returns None if request cannot fit.
        C++ reference: scheduledBlocksManager.h:80-100
        """
        blocks_if_scheduled = {}
        for window_size, num_scheduled in self.num_scheduled_blocks.items():
            required = self.kv_cache_manager.get_needed_blocks_one_step(
                req, self.two_steps_look_ahead, window_size)
            logger.debug(
                f"MaxUtilizationScheduler: request ID {req.request_id} "
                f"required blocks {required} for {window_size} window size")
            scheduled_total = num_scheduled + required
            has_free = self.kv_cache_manager.scheduling_has_free_blocks(
                scheduled_total, window_size)
            if not has_free:
                return None
            blocks_if_scheduled[window_size] = scheduled_total
        return blocks_if_scheduled

    def update_scheduled_blocks(self, blocks: dict[int, int]) -> None:
        """
        Update the scheduled blocks after successfully scheduling a request.
        C++ reference: scheduledBlocksManager.h:102-110
        """
        assert len(blocks) == len(self.num_scheduled_blocks), \
            f"Block count mismatch: {len(blocks)} vs {len(self.num_scheduled_blocks)}"
        for window_size, blocks_if_scheduled in blocks.items():
            logger.debug(
                f"MaxUtilizationScheduler: scheduled blocks {blocks_if_scheduled} "
                f"for window size {window_size}")
            self.num_scheduled_blocks[window_size] = blocks_if_scheduled


class PyCapacityScheduler:
    """
    Python implementation of the C++ CapacityScheduler.
    Aligned 1:1 with C++ logic in cpp/tensorrt_llm/batch_manager/capacityScheduler.cpp.
    Supports Multiple Window Sizes (VSWA), block reuse optimization, and all policies.

    Policies:
    - MaxRequestsScheduler: No KV cache manager, simple request count limit
    - GuaranteedNoEvictScheduler: Reserve blocks for completion, no eviction
    - StaticBatchScheduler: Only schedule when no requests are active
    - MaxUtilizationScheduler: Maximize utilization, may pause requests

    Reference: cpp/include/tensorrt_llm/batch_manager/capacityScheduler.h
    """

    def __init__(
        self,
        max_num_requests: int,
        kv_cache_manager=None,
        peft_cache_manager=None,
        scheduler_policy: CapacitySchedulerPolicy = CapacitySchedulerPolicy.
        GUARANTEED_NO_EVICT,
        cross_kv_cache_manager=None,
        two_step_lookahead: bool = False,
        no_schedule_until_state: LlmRequestState = LlmRequestState.CONTEXT_INIT,
        no_schedule_after_state: LlmRequestState = LlmRequestState.
        GENERATION_COMPLETE,
    ):
        """
        Initialize the capacity scheduler.

        Args:
            max_num_requests: Maximum number of requests to schedule
            kv_cache_manager: KV cache manager (None for MaxRequestsScheduler)
            peft_cache_manager: PEFT/LoRA cache manager (optional)
            scheduler_policy: Scheduling policy
            cross_kv_cache_manager: Cross-attention KV cache manager for encoder-decoder
            two_step_lookahead: Enable two-step lookahead for MAX_UTILIZATION
            no_schedule_until_state: Don't schedule until this state is reached
            no_schedule_after_state: Don't schedule after this state is reached
        """
        self.max_num_requests = max_num_requests
        self.kv_cache_manager = kv_cache_manager
        self.peft_cache_manager = peft_cache_manager
        self.cross_kv_cache_manager = cross_kv_cache_manager
        self.scheduler_policy = scheduler_policy
        self.two_step_lookahead = two_step_lookahead
        self.no_schedule_until_state = no_schedule_until_state
        self.no_schedule_after_state = no_schedule_after_state
        # Cache state values to avoid repeated .value access (optimization)
        self._no_schedule_until_state_value = no_schedule_until_state.value
        self._no_schedule_after_state_value = no_schedule_after_state.value

        # Initialize the appropriate policy
        self._policy = self._create_policy()

    def _create_policy(self) -> SchedulerPolicyBase:
        """Create the appropriate policy based on configuration."""
        if self.kv_cache_manager is None:
            return MaxRequestsPolicy()
        elif self.scheduler_policy == CapacitySchedulerPolicy.MAX_UTILIZATION:
            return MaxUtilizationPolicy()
        elif self.scheduler_policy == CapacitySchedulerPolicy.GUARANTEED_NO_EVICT:
            return GuaranteedNoEvictPolicy(static_batch=False)
        elif self.scheduler_policy == CapacitySchedulerPolicy.STATIC_BATCH:
            return GuaranteedNoEvictPolicy(static_batch=True)
        else:
            raise ValueError(
                f"Unsupported scheduler policy: {self.scheduler_policy}")

    def _can_be_scheduled(self, req: LlmRequest) -> bool:
        """
        Check if request is within the schedulable state range.
        Returns True if request has reached no_schedule_until_state
        but has not yet reached no_schedule_after_state.
        Optimized: use state_value property to avoid enum object creation
        """
        # Use state_value property (returns int directly, avoids enum object creation)
        state_value = req.state_value
        # Inline comparison: must have reached until_state but not after_state
        return (state_value >= self._no_schedule_until_state_value
                and state_value < self._no_schedule_after_state_value)

    def _is_skipping_relevant(self) -> bool:
        """
        Check if block reuse skip optimization is relevant.
        Disabled for VSWA (Variable Sliding Window Attention).
        C++ reference: capacityScheduler.cpp:207-208, 348
        """
        if self.kv_cache_manager is None:
            return False
        if self.kv_cache_manager.is_variable_window:
            return False
        if (self.cross_kv_cache_manager is not None
                and self.cross_kv_cache_manager.is_variable_window):
            return False
        return True

    def _prefill_contributed_blocks(
            self, active_requests: RequestList) -> tuple[set, set]:
        """
        Collect blocks contributed by chunked context requests already executing.
        These blocks can be reused by later requests.

        C++ reference: capacityScheduler.cpp:34-68 (prefillWithChunkedContextsAlreadyExecuting)
        """
        newly_contributed_context_blocks: Set = set()
        newly_contributed_cross_context_blocks: Set = set()

        if self.kv_cache_manager is None:
            return newly_contributed_context_blocks, newly_contributed_cross_context_blocks

        enable_block_reuse = self.kv_cache_manager.enable_block_reuse
        cross_enable_reuse = (self.cross_kv_cache_manager is not None and
                              self.cross_kv_cache_manager.enable_block_reuse)

        for req in active_requests:
            # Check: isContextInitState() && !isFirstContextChunk()
            if req.is_context_init_state and not req.is_first_context_chunk:
                # Chunked context request already executing
                if enable_block_reuse:
                    unique_tokens = req.get_unique_tokens(0)
                    block_key = self.kv_cache_manager.find_new_context_block(
                        unique_tokens, req)
                    if block_key is not None:
                        newly_contributed_context_blocks.add(block_key)

                if cross_enable_reuse:
                    encoder_unique_tokens = req.get_encoder_unique_tokens()
                    if encoder_unique_tokens is not None:
                        block_key = self.cross_kv_cache_manager.find_new_context_block(
                            encoder_unique_tokens, req)
                        if block_key is not None:
                            newly_contributed_cross_context_blocks.add(
                                block_key)

        return newly_contributed_context_blocks, newly_contributed_cross_context_blocks

    def _one_manager_beneficial_to_skip(self, kv_cache_manager, unique_tokens,
                                        req: LlmRequest,
                                        newly_contributed_blocks: set) -> bool:
        """
        Check if skipping is beneficial for one KV cache manager.
        C++ reference: capacityScheduler.cpp:70-92 (oneManagerBeneficialToSkip)
        """
        new_context_block = kv_cache_manager.find_new_context_block(
            unique_tokens, req)
        if new_context_block is not None:
            if new_context_block in newly_contributed_blocks:
                return True
            newly_contributed_blocks.add(new_context_block)
        return False

    def _beneficial_to_skip(
            self, req: LlmRequest, newly_contributed_context_blocks: set,
            newly_contributed_cross_context_blocks: set) -> bool:
        """
        Check if it's beneficial to skip this request.
        A request should be skipped if it can reuse blocks contributed by
        already scheduled context requests.

        C++ reference: capacityScheduler.cpp:97-123 (beneficialToSkip)
        """
        if not (req.is_context_init_state and req.is_first_context_chunk):
            return False

        if (self.kv_cache_manager is not None
                and self.kv_cache_manager.enable_block_reuse):
            unique_tokens = req.get_unique_tokens(0)
            if self._one_manager_beneficial_to_skip(
                    self.kv_cache_manager, unique_tokens, req,
                    newly_contributed_context_blocks):
                return True

        if (self.cross_kv_cache_manager is not None
                and self.cross_kv_cache_manager.enable_block_reuse):
            encoder_unique_tokens = req.get_encoder_unique_tokens()
            if encoder_unique_tokens is not None:
                if self._one_manager_beneficial_to_skip(
                        self.cross_kv_cache_manager, encoder_unique_tokens, req,
                        newly_contributed_cross_context_blocks):
                    return True

        return False

    def _get_max_peft_pages(self) -> int:
        """Get maximum PEFT cache pages."""
        if self.peft_cache_manager is None:
            return 2**31 - 1  # INT_MAX equivalent
        return self.peft_cache_manager.max_device_pages

    def _get_peft_pages_for_request(self, req: LlmRequest) -> int:
        """Get PEFT pages needed for a request."""
        if self.peft_cache_manager is None:
            return 0
        return self.peft_cache_manager.determine_num_pages(req)

    def _get_peft_task_info(
            self, req: LlmRequest,
            seen_task_ids: set[int]) -> tuple[Optional[int], bool, int]:
        """
        Get PEFT task information for a request.
        Returns (lora_task_id, is_new_task, required_pages).
        """
        lora_task_id = getattr(req, 'lora_task_id', None)
        is_new_task = lora_task_id is not None and lora_task_id not in seen_task_ids
        required_pages = self._get_peft_pages_for_request(
            req) if is_new_task else 0
        return lora_task_id, is_new_task, required_pages

    def _can_be_scheduled_with_disagg_exception(self, req: LlmRequest) -> bool:
        """
        Check if request can be scheduled, with exception for disagg generation init state.
        Disagg generation init requests bypass the normal state gating.
        """
        if req.is_disagg_generation_init_state:
            return True
        return self._can_be_scheduled(req)

    def schedule_request(
        self, active_requests: RequestList
    ) -> tuple[RequestList, RequestList, RequestList]:
        """
        Schedule requests based on the configured policy.

        Args:
            active_requests: List of active requests to consider

        Returns:
            Tuple of (fitting_requests, fitting_disagg_gen_init_requests, paused_requests)

        C++ reference: capacityScheduler.cpp:488-539 (CapacityScheduler::operator())
        """
        scheduled, paused = self._policy.schedule(self, active_requests)

        fitting_requests, fitting_disagg_gen_init_requests = self._classify_output(
            scheduled)

        logger.debug(
            f"[Summary] Capacity scheduler allows {len(fitting_requests)} requests, "
            f"pauses {len(paused)} requests")

        return fitting_requests, fitting_disagg_gen_init_requests, paused

    def _classify_output(
            self,
            scheduled_requests: RequestList) -> tuple[RequestList, RequestList]:
        """
        Separate scheduled requests into normal requests and disagg gen init requests.
        C++ reference: capacityScheduler.cpp:522-534
        """
        fitting_requests: RequestList = []
        fitting_disagg_gen_init_requests: RequestList = []
        for req in scheduled_requests:
            if req.is_disagg_generation_init_state:
                fitting_disagg_gen_init_requests.append(req)
            else:
                fitting_requests.append(req)
        return fitting_requests, fitting_disagg_gen_init_requests


class SimpleUnifiedScheduler(RequestScheduler):

    def __init__(
        self,
        max_batch_size: int,
        max_num_tokens: int,
        kv_cache_manager,
        peft_cache_manager,
        scheduler_policy: CapacitySchedulerPolicy,
        ctx_chunk_config: Optional[tuple[StrEnum, int]] = None,
        cross_kv_cache_manager=None,
        two_step_lookahead: bool = False,
        scheduler_capacity: Optional[int] = None,
    ):
        # Use scheduler_capacity if provided, otherwise fall back to max_batch_size
        # scheduler_capacity may differ from max_batch_size (e.g., adjusted for attention_dp + disagg)
        capacity = scheduler_capacity if scheduler_capacity is not None else max_batch_size

        # 1. Initialize Python Capacity Scheduler
        # Now fully aligned with C++ CapacityScheduler
        self.capacity_scheduler = PyCapacityScheduler(
            max_num_requests=capacity,
            kv_cache_manager=kv_cache_manager,
            peft_cache_manager=peft_cache_manager,
            scheduler_policy=scheduler_policy,
            cross_kv_cache_manager=cross_kv_cache_manager,
            two_step_lookahead=two_step_lookahead)

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
        fitting_requests, fitting_disagg_gen_init, paused_requests = \
            self.capacity_scheduler.schedule_request(active_requests)

        # Step 2: MicroBatch Check (Who fits in token budget? + Chunking)
        context_requests, generation_requests = \
            self.micro_batch_scheduler.schedule(fitting_requests, inflight_request_ids)

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
