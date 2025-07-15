from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Optional

import torch

from tensorrt_llm.bindings import executor as tb_executor
from tensorrt_llm.bindings import internal as tb_internal

from .llm_request import LlmRequest, LlmRequestState

RequestList = list[LlmRequest]

SchedulerOutput = namedtuple("SchedulerOutput", [
    "context_requests", "generation_requests", "paused_requests",
    "fitting_disagg_gen_init_requests", "num_fitting_requests"
])

# New named tuple for parallel stream scheduling
ParallelSchedulerOutput = namedtuple("ParallelSchedulerOutput", [
    "stream_0_context_requests", "stream_0_generation_requests",
    "stream_1_context_requests", "stream_1_generation_requests",
    "paused_requests", "fitting_disagg_gen_init_requests",
    "num_fitting_requests"
])


class ParallelExecutionConfig:
    """
    Configuration for parallel stream execution.
    """

    def __init__(
            self,
            enable_parallel_execution: bool = True,
            load_balancing_strategy:
        str = "smart",  # "round_robin", "smart", "balanced"
            min_requests_for_parallel: int = 2,
            stream_priority: int = 0,
            enable_stream_synchronization: bool = True,
            context_generation_fusion: bool = False):
        """
        Initialize parallel execution configuration.

        Args:
            enable_parallel_execution: Whether to enable parallel execution
            load_balancing_strategy: Strategy for splitting requests between streams
            min_requests_for_parallel: Minimum number of requests to enable parallel execution
            stream_priority: Priority for CUDA streams
            enable_stream_synchronization: Whether to synchronize streams after execution
            context_generation_fusion: Whether to fuse context and generation on same stream
        """
        self.enable_parallel_execution = enable_parallel_execution
        self.load_balancing_strategy = load_balancing_strategy
        self.min_requests_for_parallel = min_requests_for_parallel
        self.stream_priority = stream_priority
        self.enable_stream_synchronization = enable_stream_synchronization
        self.context_generation_fusion = context_generation_fusion


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


class ParallelStreamScheduler(RequestScheduler):
    """
    A scheduler that can run two batches in parallel on different CUDA streams.
    This scheduler splits requests between two streams to maximize GPU utilization.
    """

    def __init__(self,
                 capacity_scheduler: 'CapacityScheduler',
                 micro_batch_scheduler: 'MicroBatchScheduler',
                 config: Optional[ParallelExecutionConfig] = None,
                 stream_0: Optional[torch.cuda.Stream] = None,
                 stream_1: Optional[torch.cuda.Stream] = None):
        """
        Initialize the parallel stream scheduler.

        Args:
            capacity_scheduler: Scheduler to determine which requests can run
            micro_batch_scheduler: Scheduler to organize requests into batches
            config: Configuration for parallel execution
            stream_0: First CUDA stream for parallel execution
            stream_1: Second CUDA stream for parallel execution
        """
        super(ParallelStreamScheduler, self).__init__()
        self.capacity_scheduler = capacity_scheduler
        self.micro_batch_scheduler = micro_batch_scheduler
        self.config = config or ParallelExecutionConfig()

        # Create streams with specified priority
        if stream_0 is None:
            self.stream_0 = torch.cuda.Stream(
                priority=self.config.stream_priority)
        else:
            self.stream_0 = stream_0

        if stream_1 is None:
            self.stream_1 = torch.cuda.Stream(
                priority=self.config.stream_priority)
        else:
            self.stream_1 = stream_1

        # Events for synchronization between streams
        self.stream_0_event = torch.cuda.Event()
        self.stream_1_event = torch.cuda.Event()

    def schedule_request(self, active_requests: RequestList,
                         inflight_request_ids: set[int]) -> SchedulerOutput:
        """
        Schedule requests for parallel execution on two streams.

        Args:
            active_requests: List of active requests
            inflight_request_ids: Set of inflight request IDs

        Returns:
            SchedulerOutput with requests distributed across streams
        """
        # First, use capacity scheduler to determine which requests can run
        fitting_requests, fitting_disagg_gen_init_requests, paused_requests = self.capacity_scheduler.schedule_request(
            active_requests)

        if (not self.config.enable_parallel_execution or len(fitting_requests)
                < self.config.min_requests_for_parallel):
            # Fall back to single stream execution
            context_requests, generation_requests = self.micro_batch_scheduler.schedule(
                fitting_requests, inflight_request_ids)
            return SchedulerOutput(list(context_requests),
                                   list(generation_requests),
                                   list(paused_requests),
                                   list(fitting_disagg_gen_init_requests),
                                   len(fitting_requests))

        # Split requests between two streams based on configuration
        stream_0_requests, stream_1_requests = self._split_requests_for_streams(
            fitting_requests)

        # Schedule each stream's requests
        stream_0_context, stream_0_generation = self.micro_batch_scheduler.schedule(
            stream_0_requests, inflight_request_ids)
        stream_1_context, stream_1_generation = self.micro_batch_scheduler.schedule(
            stream_1_requests, inflight_request_ids)

        # Combine results for backward compatibility
        all_context_requests = list(stream_0_context) + list(stream_1_context)
        all_generation_requests = list(stream_0_generation) + list(
            stream_1_generation)

        return SchedulerOutput(all_context_requests, all_generation_requests,
                               list(paused_requests),
                               list(fitting_disagg_gen_init_requests),
                               len(fitting_requests))

    def schedule_request_parallel(
            self, active_requests: RequestList,
            inflight_request_ids: set[int]) -> ParallelSchedulerOutput:
        """
        Schedule requests for parallel execution and return stream-specific results.

        Args:
            active_requests: List of active requests
            inflight_request_ids: Set of inflight request IDs

        Returns:
            ParallelSchedulerOutput with requests separated by stream
        """
        # First, use capacity scheduler to determine which requests can run
        fitting_requests, fitting_disagg_gen_init_requests, paused_requests = self.capacity_scheduler.schedule_request(
            active_requests)

        if (not self.config.enable_parallel_execution or len(fitting_requests)
                < self.config.min_requests_for_parallel):
            # Fall back to single stream execution - put everything on stream 0
            context_requests, generation_requests = self.micro_batch_scheduler.schedule(
                fitting_requests, inflight_request_ids)
            return ParallelSchedulerOutput(
                list(context_requests), list(generation_requests), [], [],
                list(paused_requests), list(fitting_disagg_gen_init_requests),
                len(fitting_requests))

        # Split requests between two streams
        stream_0_requests, stream_1_requests = self._split_requests_for_streams(
            fitting_requests)

        # Schedule each stream's requests
        stream_0_context, stream_0_generation = self.micro_batch_scheduler.schedule(
            stream_0_requests, inflight_request_ids)
        stream_1_context, stream_1_generation = self.micro_batch_scheduler.schedule(
            stream_1_requests, inflight_request_ids)

        return ParallelSchedulerOutput(list(stream_0_context),
                                       list(stream_0_generation),
                                       list(stream_1_context),
                                       list(stream_1_generation),
                                       list(paused_requests),
                                       list(fitting_disagg_gen_init_requests),
                                       len(fitting_requests))

    def _split_requests_for_streams(
            self, requests: RequestList) -> tuple[RequestList, RequestList]:
        """
        Split requests between two streams based on workload characteristics.

        Args:
            requests: List of requests to split

        Returns:
            Tuple of (stream_0_requests, stream_1_requests)
        """
        if len(requests) <= 1:
            return requests, []

        if self.config.load_balancing_strategy == "round_robin":
            return self._split_requests_round_robin(requests)
        elif self.config.load_balancing_strategy == "balanced":
            return self._split_requests_balanced(requests)
        else:  # "smart" is default
            return self._split_requests_smart(requests)

    def _split_requests_round_robin(
            self, requests: RequestList) -> tuple[RequestList, RequestList]:
        """Simple round-robin splitting."""
        stream_0_requests = []
        stream_1_requests = []

        for i, request in enumerate(requests):
            if i % 2 == 0:
                stream_0_requests.append(request)
            else:
                stream_1_requests.append(request)

        return stream_0_requests, stream_1_requests

    def _split_requests_smart(
            self, requests: RequestList) -> tuple[RequestList, RequestList]:
        """
        Smart load balancing strategy that considers request characteristics.

        Args:
            requests: List of requests to split

        Returns:
            Tuple of (stream_0_requests, stream_1_requests)
        """
        if len(requests) <= 1:
            return requests, []

        # Enhanced load balancing strategy
        stream_0_requests = []
        stream_1_requests = []

        # Separate context and generation requests for better load balancing
        context_requests = []
        generation_requests = []

        for request in requests:
            if request.state in [
                    LlmRequestState.CONTEXT_INIT,
                    LlmRequestState.DISAGG_CONTEXT_INIT_AND_TRANS
            ]:
                context_requests.append(request)
            else:
                generation_requests.append(request)

        # Distribute context requests (typically more compute-intensive)
        for i, request in enumerate(context_requests):
            if i % 2 == 0:
                stream_0_requests.append(request)
            else:
                stream_1_requests.append(request)

        # Distribute generation requests
        for i, request in enumerate(generation_requests):
            if i % 2 == 0:
                stream_0_requests.append(request)
            else:
                stream_1_requests.append(request)

        # If one stream is empty, try to balance by moving requests
        if not stream_0_requests and stream_1_requests:
            # Move half of stream_1 requests to stream_0
            mid = len(stream_1_requests) // 2
            stream_0_requests = stream_1_requests[:mid]
            stream_1_requests = stream_1_requests[mid:]
        elif not stream_1_requests and stream_0_requests:
            # Move half of stream_0 requests to stream_1
            mid = len(stream_0_requests) // 2
            stream_1_requests = stream_0_requests[:mid]
            stream_0_requests = stream_0_requests[mid:]

        return stream_0_requests, stream_1_requests

    def _split_requests_balanced(
            self, requests: RequestList) -> tuple[RequestList, RequestList]:
        """
        Alternative load balancing strategy that considers request characteristics.

        Args:
            requests: List of requests to split

        Returns:
            Tuple of (stream_0_requests, stream_1_requests)
        """
        if len(requests) <= 1:
            return requests, []

        # Calculate workload for each request
        request_workloads = []
        for request in requests:
            workload = self._calculate_request_workload(request)
            request_workloads.append((request, workload))

        # Sort by workload (heaviest first)
        request_workloads.sort(key=lambda x: x[1], reverse=True)

        # Use greedy assignment to balance streams
        stream_0_requests = []
        stream_1_requests = []
        stream_0_workload = 0
        stream_1_workload = 0

        for request, workload in request_workloads:
            if stream_0_workload <= stream_1_workload:
                stream_0_requests.append(request)
                stream_0_workload += workload
            else:
                stream_1_requests.append(request)
                stream_1_workload += workload

        return stream_0_requests, stream_1_requests

    def _calculate_request_workload(self, request: LlmRequest) -> float:
        """
        Calculate the estimated workload for a request.

        Args:
            request: The request to evaluate

        Returns:
            Estimated workload score
        """
        # Base workload from input length
        workload = len(request.input_token_ids) if hasattr(
            request, 'input_token_ids') else 1.0

        # Adjust based on request state
        if request.state == LlmRequestState.CONTEXT_INIT:
            # Context processing is typically more compute-intensive
            workload *= 2.0
        elif request.state == LlmRequestState.GENERATION_IN_PROGRESS:
            # Generation is typically lighter
            workload *= 1.0
        elif request.state == LlmRequestState.GENERATION_TO_COMPLETE:
            # Final generation step
            workload *= 0.5

        # Adjust based on beam width
        if hasattr(request, 'sampling_config') and hasattr(
                request.sampling_config, 'beam_width'):
            workload *= request.sampling_config.beam_width

        # Adjust based on draft tokens (speculative decoding)
        if hasattr(request, 'draft_tokens') and request.draft_tokens:
            workload *= (1.0 + len(request.draft_tokens) * 0.1)

        return workload

    def execute_parallel_batches(self, stream_0_context: RequestList,
                                 stream_0_generation: RequestList,
                                 stream_1_context: RequestList,
                                 stream_1_generation: RequestList,
                                 context_executor, generation_executor) -> None:
        """
        Execute two batches in parallel on different streams.

        Args:
            stream_0_context: Context requests for stream 0
            stream_0_generation: Generation requests for stream 0
            stream_1_context: Context requests for stream 1
            stream_1_generation: Generation requests for stream 1
            context_executor: Function to execute context phase
            generation_executor: Function to execute generation phase
        """
        if not self.config.enable_parallel_execution:
            # Fall back to sequential execution
            if stream_0_context or stream_0_generation:
                context_executor(stream_0_context, stream_0_generation)
            if stream_1_context or stream_1_generation:
                context_executor(stream_1_context, stream_1_generation)
            return

        # Execute stream 0
        with torch.cuda.stream(self.stream_0):
            if self.config.context_generation_fusion:
                # Execute context and generation together on stream 0
                if stream_0_context or stream_0_generation:
                    context_executor(stream_0_context, stream_0_generation)
            else:
                # Execute context and generation separately
                if stream_0_context:
                    context_executor(stream_0_context, [])
                if stream_0_generation:
                    generation_executor([], stream_0_generation)
            self.stream_0_event.record()

        # Execute stream 1
        with torch.cuda.stream(self.stream_1):
            if self.config.context_generation_fusion:
                # Execute context and generation together on stream 1
                if stream_1_context or stream_1_generation:
                    context_executor(stream_1_context, stream_1_generation)
            else:
                # Execute context and generation separately
                if stream_1_context:
                    context_executor(stream_1_context, [])
                if stream_1_generation:
                    generation_executor([], stream_1_generation)
            self.stream_1_event.record()

        # Synchronize streams if enabled
        if self.config.enable_stream_synchronization:
            self.stream_0_event.wait()
            self.stream_1_event.wait()

    def execute_parallel_batches_async(
            self, stream_0_context: RequestList,
            stream_0_generation: RequestList, stream_1_context: RequestList,
            stream_1_generation: RequestList, context_executor,
            generation_executor) -> tuple[torch.cuda.Event, torch.cuda.Event]:
        """
        Execute two batches in parallel on different streams asynchronously.

        Args:
            stream_0_context: Context requests for stream 0
            stream_0_generation: Generation requests for stream 0
            stream_1_context: Context requests for stream 1
            stream_1_generation: Generation requests for stream 1
            context_executor: Function to execute context phase
            generation_executor: Function to execute generation phase

        Returns:
            Tuple of (stream_0_event, stream_1_event) for synchronization
        """
        if not self.config.enable_parallel_execution:
            # Fall back to sequential execution
            if stream_0_context or stream_0_generation:
                context_executor(stream_0_context, stream_0_generation)
            if stream_1_context or stream_1_generation:
                context_executor(stream_1_context, stream_1_generation)
            return self.stream_0_event, self.stream_1_event

        # Execute stream 0
        with torch.cuda.stream(self.stream_0):
            if self.config.context_generation_fusion:
                # Execute context and generation together on stream 0
                if stream_0_context or stream_0_generation:
                    context_executor(stream_0_context, stream_0_generation)
            else:
                # Execute context and generation separately
                if stream_0_context:
                    context_executor(stream_0_context, [])
                if stream_0_generation:
                    generation_executor([], stream_0_generation)
            self.stream_0_event.record()

        # Execute stream 1
        with torch.cuda.stream(self.stream_1):
            if self.config.context_generation_fusion:
                # Execute context and generation together on stream 1
                if stream_1_context or stream_1_generation:
                    context_executor(stream_1_context, stream_1_generation)
            else:
                # Execute context and generation separately
                if stream_1_context:
                    context_executor(stream_1_context, [])
                if stream_1_generation:
                    generation_executor([], stream_1_generation)
            self.stream_1_event.record()

        return self.stream_0_event, self.stream_1_event

    def get_streams(self) -> tuple[torch.cuda.Stream, torch.cuda.Stream]:
        """Get the two CUDA streams used for parallel execution."""
        return self.stream_0, self.stream_1

    def get_events(self) -> tuple[torch.cuda.Event, torch.cuda.Event]:
        """Get the CUDA events used for synchronization."""
        return self.stream_0_event, self.stream_1_event


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
        scheduler_policy: tb_executor.CapacitySchedulerPolicy = tb_executor.
        CapacitySchedulerPolicy.GUARANTEED_NO_EVICT,
        two_step_lookahead: bool = False,
    ):
        super(BindCapacityScheduler, self).__init__()
        self.kv_cache_manager = kv_cache_manager

        self.impl = tb_internal.algorithms.CapacityScheduler(
            max_num_requests=max_num_requests,
            capacity_scheduler_policy=scheduler_policy,
            has_kv_cache_manager=kv_cache_manager is not None,
            two_step_lookahead=two_step_lookahead,
            no_schedule_until_state=LlmRequestState.CONTEXT_INIT,
            no_schedule_after_state=LlmRequestState.GENERATION_COMPLETE)

    def schedule_request(
        self, active_requests: RequestList
    ) -> tuple[list[LlmRequest], list[LlmRequest], list[LlmRequest]]:
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
        for request in active_requests:
            if len(request.py_draft_tokens) > 0:
                request.draft_tokens = request.py_draft_tokens
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
