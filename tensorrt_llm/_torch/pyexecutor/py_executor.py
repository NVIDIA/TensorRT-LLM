import dataclasses
import datetime
import functools
import os
import pickle  # nosec B403
import threading
import time
import traceback
from collections import deque
from contextlib import contextmanager
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch

from tensorrt_llm._torch.expert_statistic import ExpertStatistic
from tensorrt_llm.llmapi import DisaggScheduleStyle
from tensorrt_llm.serve.responses_utils import get_steady_clock_now_in_seconds

try:
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cudart

from tensorrt_llm._torch.pyexecutor.resource_manager import (
    ResourceManagerType, request_context)
from tensorrt_llm._utils import (customized_gc_thresholds, is_trace_enabled,
                                 mpi_disabled, nvtx_range, trace_func)
from tensorrt_llm.bindings.executor import (DisServingRequestStats,
                                            FinishReason, InflightBatchingStats,
                                            IterationStats, KvCacheStats,
                                            RequestStage, RequestStats,
                                            SpecDecodingStats,
                                            StaticBatchingStats)
from tensorrt_llm.bindings.internal.batch_manager import (LlmRequestType,
                                                          ReqIdsSet)
from tensorrt_llm.llmapi.llm_args import PeftCacheConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import CpType
from tensorrt_llm.runtime.generation import CUASSERT
from tensorrt_llm.tools.layer_wise_benchmarks import get_calibrator

from ..distributed import Distributed
from ..models.modeling_utils import DecoderModelForCausalLM
from ..modules.decoder_layer import DecoderLayer
from ..speculative.drafter import Drafter
from ..speculative.mtp import SampleStateTensorsMTP
from ..speculative.speculation_gate import SpeculationGate
from .executor_request_queue import ExecutorRequestQueue, RequestQueueItem
from .guided_decoder import GuidedDecoder
from .handle_additional_outputs import HandleAdditionalOutputs
from .handle_logits import HandleLogits
from .hang_detector import HangDetector
from .kv_cache_connector import KvCacheConnectorManager
from .kv_cache_transceiver import KvCacheTransceiver
from .llm_request import (ExecutorRequest, LlmRequest, LlmRequestState,
                          LlmResponse, get_draft_token_length)
from .model_engine import ModelEngine
from .request_utils import (RequestBroadcaster, attach_py_objects_to_requests,
                            get_from_waiting_queue, merge_requests,
                            schedule_attention_dp_requests)
from .resource_manager import ResourceManager
from .sampler import (AsyncWorkerMixin, Sampler, SamplerEvent, SampleState,
                      SampleStateTensors, TRTLLMSampler)
from .scheduler import (RequestScheduler, ScheduledRequests,
                        SerializableSchedulerOutput)

# Environment variable to specify iteration ranges for profiling start/stop.
# Format: "start1-stop1,start2-stop2,..." or single iterations "iter1,iter2,..."
PROFILE_START_STOP_ENV_VAR_NAME = "TLLM_PROFILE_START_STOP"

# Environment variable to enable PyTorch profiler tracing.
# Set to a path to save detailed tracing of PyTorch operations.
PROFILE_TRACE_ENV_VAR_NAME = "TLLM_TORCH_PROFILE_TRACE"

# Unique tag base to avoid collisions with token/logits comms
TERMINATION_COMM_TAG_BASE = 20000
PP_COMM_TAG_SCHEDULE_RESULT = 21000
PP_COMM_TAG_SAMPLE_STATE_BASE = 21001


@functools.cache
def _load_iteration_indexes(env_var: str):
    spans = os.environ.get(env_var, None)
    starts, stops = [], []

    if spans:
        spans = spans.split(',')

        for span in spans:
            try:
                if '-' in span:
                    start, stop = span.strip().split('-')
                    starts.append(int(start))
                    stops.append(int(stop))
                else:
                    it = int(span.strip())
                    starts.append(it)
                    stops.append(it)
            except ValueError as e:
                raise ValueError(
                    f"Cannot parse span in environment variable `{env_var}`: {e}"
                ) from None

    return frozenset(starts), frozenset(stops)


@dataclasses.dataclass
class BatchState:
    sample_state: SampleState

    iter_start_time: float = 0
    iter_stats: IterationStats = None
    all_requests: list[LlmRequest] = None


@dataclasses.dataclass
class BatchStatePP(BatchState):
    microbatch_id: int = -1
    scheduled_ctx_reqs: list[LlmRequest] = None
    finished_ctx_reqs: list[LlmRequest] = None


class AsyncTransferManager:
    """
    Handle asynchronous transfer or KV cache after a request has completed.
    When running with both the KV cache transceiver and the KV cache connector, we must ensure that BOTH transfers (if any) are completed before we can release the KV cache blocks.
    The AsyncTransferManager has a few key responsibilities:
    1. Track requests in transfer.
    2. Pin blocks for reuse while blocks are in transfer.
    3. Unpin blocks after all transfers are complete.

    TODO(jthomson04): This only handles async send/saving, not loading. Loading kv cache is handled through a separate codepath. Eventually, we'll want to merge these two paths.
    """

    class RequestTransferMetadata:

        def __init__(self, block_id: Optional[int]):
            self.block_id = block_id
            self.counter = 0

        def start_transfer(self):
            self.counter += 1

        def end_transfer(self) -> bool:
            """
            Returns:
                bool: True if there are no more transfers for this request
            """
            self.counter -= 1
            return self.counter == 0

    def __init__(self,
                 resource_manager: "ResourceManager",
                 should_store_blocks: bool = True):
        self.resource_manager = resource_manager
        self.kv_cache_manager = resource_manager.resource_managers.get(
            ResourceManagerType.KV_CACHE_MANAGER)

        self.should_store_blocks = should_store_blocks

        # Mapping of request id to the LlmRequest
        self._requests_in_transfer: Dict[int, LlmRequest] = dict()

        # Mapping of request id to the the request metadata
        self._request_transfer_metadata: Dict[
            int, self.RequestTransferMetadata] = dict()

    def requests_in_transfer(self) -> Dict[int, LlmRequest]:
        return self._requests_in_transfer

    def start_transfer(self, request: LlmRequest):
        """
        Called when a Cache transceiver or connector transfer is started.
        1. Increment the counter for the request.
        2. Releases all resources except for the KV cache, if not already released.
        3. Store KV cache blocks for reuse.
        """

        req_id = request.py_request_id

        if req_id not in self._requests_in_transfer:
            for resource_mgr_type in (
                    ResourceManagerType.SEQ_SLOT_MANAGER,
                    ResourceManagerType.SPEC_RESOURCE_MANAGER):
                if resource_mgr_type in self.resource_manager.resource_managers and self.resource_manager.resource_managers[
                        resource_mgr_type] is not None:
                    self.resource_manager.resource_managers[
                        resource_mgr_type].free_resources(request)

            request.state = LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS

            if self.should_store_blocks:
                block_id = self.kv_cache_manager.store_blocks_for_reuse(
                    request, True)
            else:
                block_id = None

            self._requests_in_transfer[req_id] = request
            self._request_transfer_metadata[
                req_id] = self.RequestTransferMetadata(block_id)

        self._request_transfer_metadata[req_id].start_transfer()

    def end_transfer(self, request: LlmRequest) -> bool:
        """
        Called after a send of KV cache is complete.
        1. Decrements counter for request.
        2. If there are no more inflight transfers for this request, unpin the blocks and mark the request as complete.

        Returns:
            bool: True if the request should be terminated after call to end_transfer
        """
        try:
            transfer_metadata = self._request_transfer_metadata[
                request.py_request_id]
        except KeyError:
            logger.warning(
                f"Request {request.py_request_id} not found in transfer manager"
            )
            return

        if transfer_metadata.end_transfer():
            self._requests_in_transfer.pop(request.py_request_id)
            self._request_transfer_metadata.pop(request.py_request_id)

            if self.should_store_blocks:
                self.kv_cache_manager.unpin_blocks_by_id(
                    transfer_metadata.block_id)

            # We don't want to overwrite any error state.
            if request.state != LlmRequestState.DISAGG_TRANS_ERROR:
                request.state = LlmRequestState.DISAGG_CONTEXT_COMPLETE

            return True

        return False

    def has_any_inflight_requests(self) -> bool:
        return len(self._requests_in_transfer) > 0


class PyExecutor:

    def __init__(self,
                 resource_manager,
                 scheduler: RequestScheduler,
                 model_engine: ModelEngine,
                 sampler: Sampler,
                 dist: Distributed,
                 max_num_sequences: int,
                 drafter: Optional[Drafter] = None,
                 disable_overlap_scheduler: bool = False,
                 max_input_len: int = 0x7fffffff,
                 max_batch_size: int = 8,
                 max_beam_width: int = 1,
                 max_draft_len: int = 0,
                 max_total_draft_tokens: int = 0,
                 kv_cache_transceiver: Optional[KvCacheTransceiver] = None,
                 guided_decoder: Optional[GuidedDecoder] = None,
                 garbage_collection_gen0_threshold: Optional[int] = None,
                 start_worker: bool = True,
                 kv_connector_manager: Optional[KvCacheConnectorManager] = None,
                 max_seq_len: Optional[int] = None,
                 peft_cache_config: Optional[PeftCacheConfig] = None,
                 virtual_memory_pools: Optional[dict] = None,
                 hang_detection_timeout: Optional[int] = None,
                 execution_stream: Optional[torch.cuda.Stream] = None):
        super(PyExecutor, self).__init__()
        self.device_id = torch.cuda.current_device()
        self.global_rank = dist.rank
        # Store the execution stream for model forward operations.
        # This stream is used for proper synchronization with KVCacheTransferManager.
        # execution_stream can be provided by create_py_executor
        # Create a new stream if none provided
        self.execution_stream = execution_stream if execution_stream is not None else torch.cuda.Stream(
        )
        logger.info(
            f"[PyExecutor] execution_stream initialized: {self.execution_stream}. "
        )

        self.peft_cache_config = peft_cache_config

        self.iter_counter = 0
        # profile config
        self.profile_start_iters, self.profile_stop_iters = _load_iteration_indexes(
            PROFILE_START_STOP_ENV_VAR_NAME)

        # related modules
        self.resource_manager = resource_manager
        self.scheduler = scheduler
        self.model_engine = model_engine
        self.enable_attention_dp = model_engine.enable_attention_dp
        self.sampler = sampler
        self.drafter = drafter
        self.draft_model_engine = getattr(self.drafter, "draft_model_engine",
                                          None)
        self.guided_decoder = guided_decoder
        self.dist = dist
        self.disable_overlap_scheduler = disable_overlap_scheduler
        self.virtual_memory_pools = virtual_memory_pools

        # enqueue and _fetch_new_requests used data
        self.active = True
        self.max_beam_width = max_beam_width
        self.max_draft_len = max_draft_len
        self.max_total_draft_tokens = max_total_draft_tokens
        self.llm_args = self.model_engine.llm_args
        self.max_stats_len = max(self.llm_args.max_stats_len, 1)
        self.max_num_tokens = self.llm_args.max_num_tokens
        self.print_log = self.llm_args.print_iter_log
        self.enable_iter_perf_stats = self.llm_args.enable_iter_perf_stats
        self.enable_iter_req_stats = self.llm_args.enable_iter_req_stats
        self.stream_interval = self.llm_args.stream_interval
        self.attention_dp_enable_balance = (
            self.llm_args.attention_dp_config is not None
            and self.llm_args.attention_dp_config.enable_balance)
        if self.attention_dp_enable_balance:
            self.attention_dp_time_out_iters = self.llm_args.attention_dp_config.timeout_iters
            self.attention_dp_batching_wait_iters = self.llm_args.attention_dp_config.batching_wait_iters
        self.batch_wait_timeout_ms = self.llm_args.batch_wait_timeout_ms
        self.batch_wait_timeout_iters = self.llm_args.batch_wait_timeout_iters
        self.batch_wait_max_tokens_ratio = self.llm_args.batch_wait_max_tokens_ratio
        self.enable_batch_waiting = self.batch_wait_timeout_iters > 0 or self.batch_wait_max_tokens_ratio > 0

        self.num_fetch_requests_cur_rank = 0
        self.num_fetch_requests = 0
        self.shutdown_event = threading.Event()

        # Rolling acceptance tracking for spec decode (disable speculation if rolling acceptance is below threshold)
        spec_config = getattr(self.model_engine, 'spec_config', None)
        self.acceptance_window = getattr(
            spec_config, 'acceptance_window',
            None) if spec_config is not None else None
        self.acceptance_length_threshold = getattr(
            spec_config, 'acceptance_length_threshold',
            None) if spec_config is not None else None
        self.speculation_permanently_disabled = False
        self.speculation_gate = None
        if self.acceptance_window and self.acceptance_length_threshold is not None:
            self.speculation_gate = SpeculationGate(
                self.acceptance_window, self.acceptance_length_threshold)

        # response used data
        self.response_lock = threading.Lock()
        self.response_cv = threading.Condition(self.response_lock)
        self.responses = {}
        self.result_wait_queues = {}

        # kv cache events
        self.kv_cache_manager = self.resource_manager.resource_managers.get(
            ResourceManagerType.KV_CACHE_MANAGER)
        self.enable_kv_cache_events = self.kv_cache_manager is not None and self.kv_cache_manager.event_buffer_max_size > 0
        self.enable_kv_cache_reuse = self.kv_cache_manager is not None and self.kv_cache_manager.enable_block_reuse

        self.max_input_len = max_input_len
        # _executor_loop private data
        self.max_num_active_requests = model_engine.get_max_num_sequences()
        self.active_requests: List[LlmRequest] = []
        self.expected_num_active_requests = 0
        self.async_transfer_manager = AsyncTransferManager(
            self.resource_manager,
            should_store_blocks=self.enable_kv_cache_reuse
            and not self.kv_cache_manager.is_vswa)
        self.previous_batch: Optional[BatchState] = None
        self.has_previous_draft_tokens = False
        self.num_scheduled_requests: int = 0
        self.benchmark_req_queues_size = int(
            os.environ.get("TLLM_BENCHMARK_REQ_QUEUES_SIZE", 0))

        # list of requests in each PP micro batch
        self.num_micro_batches = self.dist.pp_size
        self.micro_batches: List[BatchStatePP
                                 | None] = [None] * self.num_micro_batches
        self.send_handles = [None] * self.num_micro_batches
        # schedule handle for PP to propagate the first PP rank's schedule result
        self.send_schedule_handler = None
        self.pp_scheduler_max_retry_count = int(
            os.environ.get("TLLM_PP_SCHEDULER_MAX_RETRY_COUNT", 10))
        self.pp_multi_stream_sample = os.environ.get(
            "TRTLLM_PP_MULTI_STREAM_SAMPLE", "1") == "1"
        self.sample_stream = torch.cuda.Stream()
        self.start_sample_event = torch.cuda.Event()
        self.finish_sample_event = torch.cuda.Event()
        if (self.dist.pp_size > 1 and self.pp_multi_stream_sample
                and isinstance(self.sampler, TRTLLMSampler)):
            # TRTLLM sampler uses default stream for store and algorithms.
            # To enable multi-stream sampling, we need to re-initialize
            # the sampler store and algorithms on the sample stream.
            with torch.cuda.stream(self.sample_stream):
                self.sampler._initialize_store()
                self.sampler._instantiate_algorithms()

        # Set of request IDs that are currently in flight across all micro batches.
        # The scheduler will avoid scheduling requests that are already in flight.
        self.inflight_req_ids = ReqIdsSet()

        # During warmup, we don't enable the profiler
        # Run warmup on the execution_stream for proper synchronization with
        # KVCacheTransferManager's onboard/offload operations.
        self.is_warmup = True

        self.execution_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.execution_stream):
            self.model_engine.warmup(self.resource_manager)
            if self.draft_model_engine is not None:
                self.draft_model_engine.warmup(self.resource_manager)

        # Ensure the default stream waits for execution_stream to complete
        # before subsequent operations.
        torch.cuda.current_stream().wait_stream(self.execution_stream)
        self.is_warmup = False

        self.is_shutdown = False
        self.max_batch_size = max_batch_size
        self.adp_ctx_waiting_iters_count = 0
        self.adp_ctx_batching_wait_iters_count = 0
        self.batch_wait_iters_count = 0

        def on_detected():
            self._handle_errors(
                f"Hang detected on rank {self.global_rank} in PyExecutor.")
            self.shutdown_event.set()
            self.is_shutdown = True

        self.hang_detector = HangDetector(timeout=hang_detection_timeout,
                                          on_detected=on_detected)

        # request fetcher initialization
        self._set_global_steady_clock_offset()
        self.executor_request_queue = ExecutorRequestQueue(
            dist=self.dist,
            max_batch_size=max_batch_size,
            enable_iter_perf_stats=self.enable_iter_perf_stats,
            batch_wait_timeout_ms=self.batch_wait_timeout_ms,
        )
        # When overlap scheduler is enabled then when starting to handle a new prompt,
        # sample_async is called twice before the first call to update_requests:
        # - 1st time as a context request that handles on the 1st generated token
        # - 2nd time as a generation request that handles on the 2nd generated token.
        # and only after these two calls the sampler's update_request method is called.
        # So in a sampler that works by the expected flow of handling the logits in
        # sample_async, every update_request doesn't handle the newest token, but one
        # before it. Since all these calls work on the same request object, then its
        # logits storage contains the logits of both the token update_requests should work
        # on, and also its next token. Thus, excluding the last generation logits from any
        # getter is required.
        self.should_exclude_last_generation_logits = (
            not self.disable_overlap_scheduler and self.dist.pp_size == 1)

        # Request processing state (managed by executor)
        self.canceled_req_ids: List[int] = []
        self.control_requests: List[RequestQueueItem] = []
        self.request_accumulated: List[RequestQueueItem] = []
        self.new_active_requests_queue_latency_ms = 0.0
        self._disable_mpi = mpi_disabled()
        self.request_broadcaster = RequestBroadcaster(self.dist,
                                                      self.hang_detector)

        # Waiting queue for requests that have been fetched but not yet scheduled
        self.waiting_queue: deque[RequestQueueItem] = deque()

        self.control_request_barrier = threading.Event()
        self.control_action_done = threading.Event()

        self.stats_lock = threading.Lock()
        self.stats = []
        self.gather_all_responses = False

        self.kv_cache_transceiver = kv_cache_transceiver

        # Initialize disagg PP termination handler if needed
        self._disagg_pp_termination_handler = None
        if self.dist.pp_size > 1 and self.enable_kv_cache_reuse and self.kv_cache_transceiver:
            self._disagg_pp_termination_handler = DisaggPPTerminationHandler(
                self.dist, self._do_terminate_request)

        if self.dist.pp_size > 1:
            self.event_loop = self._executor_loop_pp
        else:
            self.event_loop = self._executor_loop if self.disable_overlap_scheduler else self._executor_loop_overlap
        if is_trace_enabled("TLLM_TRACE_EXECUTOR_LOOP"):
            self.event_loop = trace_func(self.event_loop)

        if self.drafter is not None:
            if self.event_loop.__name__ == self._executor_loop_pp.__name__:
                raise NotImplementedError(
                    "Drafting is not supported for selected executor loop. "
                    "Please disable disagg/pipeline parallelism scheduler.")
        self.garbage_collection_gen0_threshold = garbage_collection_gen0_threshold
        self.max_seq_len = max_seq_len

        self.worker_started = False
        self.worker_lock = threading.Lock()

        self.kv_connector_manager = kv_connector_manager

        self._maybe_init_kv_connector_manager()

        if start_worker:
            self.start_worker()

    def _maybe_init_kv_connector_manager(self):
        if self.kv_connector_manager is not None:
            if self.kv_cache_transceiver is not None:
                logger.warning(
                    "Both KV Cache Connector and KV Cache Transceiver are enabled. Are you sure you want to do this?"
                )

            if self.dist.pp_size > 1:
                raise NotImplementedError(
                    "KV Cache Connector is not supported with pipeline parallelism."
                )

            if self.kv_cache_manager is None:
                raise ValueError(
                    "KV Cache Connector requires a KV Cache Manager.")

            kv_tensor = self.kv_cache_manager.get_unique_primary_pool()
            self.kv_connector_manager.worker.register_kv_caches(kv_tensor)

            # For each of our layers, we need to register the pre/post hooks.
            # These are used for methods like `wait_for_layer_load` and `save_kv_layer`.
            for _name, module in self.model_engine.model.named_modules():
                if isinstance(module, DecoderLayer):
                    module.register_forward_pre_hook(
                        self.kv_connector_manager.layer_pre_hook)
                    module.register_forward_hook(
                        self.kv_connector_manager.layer_post_hook)

    def _end_transfer_and_maybe_terminate(self, request: LlmRequest):
        if self.async_transfer_manager.end_transfer(request):
            self._terminate_request(request)

    def _event_loop_wrapper(self):
        try:
            with customized_gc_thresholds(
                    self.garbage_collection_gen0_threshold):
                self.event_loop()
        except Exception as e:
            logger.error(f"Error in event loop: {e}")
            logger.error(traceback.format_exc())
            raise e
        finally:
            self._executor_loop_cleanup()

    @property
    def is_warmup(self) -> bool:
        return getattr(self, "_is_warmup", False)

    @is_warmup.setter
    def is_warmup(self, value: bool):
        self._is_warmup = value
        # Set warmup flag in model engine to trigger torch compile and avoid moe load balancer statistics update
        self.model_engine.is_warmup = value
        if self.draft_model_engine is not None:
            self.draft_model_engine.is_warmup = value

    def start_worker(self):
        with self.worker_lock:
            if self.worker_started == False:
                self.worker_thread = threading.Thread(
                    target=self._event_loop_wrapper, daemon=True)
                self.worker_thread.start()
                self.worker_started = True
            # Start the sampler's async worker, if it is enabled
            if (isinstance(self.sampler, AsyncWorkerMixin)
                    and self.sampler.async_worker_enabled()):
                self.sampler.async_worker_start()

    def _set_global_steady_clock_offset(self):
        assert self.global_rank >= 0, "rank should be >= 0"

        # Sync all ranks
        self.dist.barrier()
        # Immediately take the local steady clock timestamp
        local_timestamp = get_steady_clock_now_in_seconds()
        all_rank_timestamps = self.dist.allgather(local_timestamp)
        if self.global_rank == 0:
            logger.info(
                f"global_steady_clock_offset at each rank: {[local_timestamp - ts for ts in all_rank_timestamps]}"
            )
        # Compute the steady clock offset between rank 0 and current rank
        global_steady_clock_offset = all_rank_timestamps[0] - local_timestamp
        LlmRequest.global_steady_clock_offset = global_steady_clock_offset
        logger.info(
            f"Setting global_steady_clock_offset: {global_steady_clock_offset} seconds for rank {self.global_rank}"
        )

    def __enter__(self):
        return self

    def __exit__(self):
        self.shutdown()

    def enqueue_requests(
        self,
        requests: List[ExecutorRequest],
        result_wait_queue: "Optional[ray.actor.ActorHandle]" = None
    ) -> List[int]:
        """
        Enqueue new requests
        """
        req_ids = self.executor_request_queue.enqueue_requests(requests)
        if result_wait_queue is not None:
            with self.response_cv:
                for req_id in req_ids:
                    self.result_wait_queues[req_id] = result_wait_queue
        return req_ids

    def await_responses(
        self,
        id: Optional[Union[List[int], int]] = None,
        timeout: Optional[datetime.timedelta] = None,
    ) -> Union[List[List[LlmResponse]], List[LlmResponse]]:
        """
        Await for ready responses
        Args:
            id (Optional[Union[List[int], int]]): Request id
            timeout (Optional[datetime.timedelta]): The maximum time to wait for new responses
        Returns:
            Union[List[LlmResponse], List[List[LlmResponse]]]: Responses
        """
        timeout = timeout.total_seconds() if timeout is not None else None
        if id is None:
            return self._await_any_response(timeout=timeout)
        if isinstance(id, int):
            return self._await_single_response(id=id, timeout=timeout)
        responses = []
        for req_id in id:
            responses.append(
                self._await_single_response(id=req_id, timeout=timeout))

        return responses

    def cancel_request(self, id: int):
        """
        Cancel the request with provided request id
        Args:
            id (int): The request id for which to cancel the response
        """
        self.executor_request_queue.enqueue_cancel_request(id)

    def shutdown(self):
        """
        Signals the server to shutdown.
        """
        self.executor_request_queue.enqueue_shutdown_request()
        self.shutdown_event.wait()
        if self.hang_detector.detected():
            # Early return here to avoid waiting for hanging threads.
            # Since `on_detected` has sent the error message as response,
            # this worker will be asked to shutdown immediately.
            # Since the whole process will shutdown after this `shutdown` call,
            # All threads and memory pools will be freed properly.
            logger.error("Hang detected, shutting down immediately.")
            return
        self.worker_thread.join()
        self.worker_started = False
        for manager in self.resource_manager.resource_managers.values():
            if manager:
                manager.shutdown()
        del self.model_engine
        if self.draft_model_engine is not None:
            del self.draft_model_engine
        if self.virtual_memory_pools is not None:
            keys = list(self.virtual_memory_pools.keys())
            for key in keys:
                del self.virtual_memory_pools[key]
        # Stop the sampler's async worker, if it was used
        if (isinstance(self.sampler, AsyncWorkerMixin)
                and self.sampler.async_worker_enabled()):
            self.sampler.async_worker_stop()

    def can_enqueue_requests(self) -> bool:
        """
        Indicates if the current process is allowed to enqueue requests
        """
        return self.executor_request_queue.can_enqueue_request()

    def get_latest_iteration_stats(self):
        """
        Returns the per-iterations statistics computed since last call to this method.
        Contains at most iter_stats_max_iterations iterations.
        """
        if self.enable_iter_perf_stats == False:
            return []

        latest_stats = (IterationStats(), None)
        with self.stats_lock:
            latest_stats = self.stats
            self.stats = []
        return latest_stats

    def get_latest_kv_cache_events(self):
        kv_cache_manager = self.resource_manager.resource_managers.get(
            ResourceManagerType.KV_CACHE_MANAGER)
        if not kv_cache_manager or not self.enable_kv_cache_events:
            return []

        events = kv_cache_manager.get_latest_events(0)
        return events

    def wait_shutdown(self):
        self.shutdown_event.wait()

    def enqueue_request(
            self,
            request: ExecutorRequest,
            query: Optional[List] = None,
            result_wait_queue: "Optional[ray.actor.ActorHandle]" = None) -> int:
        """
        Enqueue a new request, query is only used in `StarAttention`.
        """
        req_id = self.executor_request_queue.enqueue_request(request, query)
        if result_wait_queue is not None:
            with self.response_cv:
                self.result_wait_queues[req_id] = result_wait_queue
        return req_id

    def set_gather_responses(self, gather_all_responses):
        self.gather_all_responses = gather_all_responses

    @property
    def should_stop_processing(self):
        return self.is_shutdown and len(self.active_requests) == 0 and \
            len(self.waiting_queue) == 0

    @contextmanager
    def _profiler(self):
        it = -1
        enabled = False
        start_time = None

        # These events are used to record the time of the previous batch.
        # We need two set of the start-end events to record the time through
        # a ping-pong way so that it works with overlap scheduler.
        start_event_1 = None
        end_event_1 = torch.cuda.Event(enable_timing=True)
        start_event_2 = None
        end_event_2 = torch.cuda.Event(enable_timing=True)
        prev_device_step_time = None

        torch_trace_path = os.environ.get(PROFILE_TRACE_ENV_VAR_NAME, None)
        profile_start_stop = os.environ.get(PROFILE_START_STOP_ENV_VAR_NAME,
                                            None)
        enable_torch_trace = bool(torch_trace_path and profile_start_stop)
        if torch_trace_path and profile_start_stop is None:
            logger.warning(
                f"{PROFILE_START_STOP_ENV_VAR_NAME} environment variable "
                "needs to be set to enable the torch trace. Example to profile "
                f"iteration 10-20: export {PROFILE_START_STOP_ENV_VAR_NAME}=10-20"
            )

        if enable_torch_trace:
            activities = [
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
                torch.profiler.ProfilerActivity.XPU,
            ]
            torch_profiler = torch.profiler.profile(activities=activities,
                                                    record_shapes=True,
                                                    with_modules=True)

        calibrator = get_calibrator()

        def profile_step():
            nonlocal it, enabled, start_time, start_event_1, end_event_1, start_event_2, end_event_2, prev_device_step_time
            calibrator.post_step(it)
            if it in self.profile_stop_iters and not self.is_warmup:
                assert enabled, "Inconsistent CUDA profiling state"
                if enable_torch_trace:
                    torch_profiler.stop()
                    torch_profiler.export_chrome_trace(torch_trace_path)
                    logger.info(f"Profiling stopped at iteration {it}, "
                                f"trace saved to {torch_trace_path}")
                torch.cuda.cudart().cudaProfilerStop()
                calibrator.stop()
                enabled = False

            if start_time is not None and self.print_log and self.dist.rank == 0:
                end_time = time.time()
                if it % 2 == 0:
                    end_event_1.record()
                    if start_event_2 is not None:
                        end_event_2.synchronize()
                        prev_device_step_time = start_event_2.elapsed_time(
                            end_event_2)
                else:
                    end_event_2.record()
                    if start_event_1 is not None:
                        end_event_1.synchronize()
                        prev_device_step_time = start_event_1.elapsed_time(
                            end_event_1)

                if prev_device_step_time is None:
                    prev_device_step_time = "N/A"  # Handle first iteration
                else:
                    prev_device_step_time = f"{prev_device_step_time}ms"
                host_step_time = (end_time - start_time) * 1000  # milliseconds
                formatted_timestamp = datetime.datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S")
                logger.info(
                    f"iter = {self.iter_counter}, "
                    f"global_rank = {self.global_rank}, "
                    f"rank = {self.dist.rank}, "
                    f"currank_total_requests = {self.num_fetch_requests_cur_rank}/"
                    f"{self.num_fetch_requests}, "
                    f"host_step_time = {host_step_time}ms, "
                    f"prev_device_step_time = {prev_device_step_time}, "
                    f"timestamp = {formatted_timestamp}, "
                    f"num_scheduled_requests: {self.num_scheduled_requests}, "
                    f"states = {self.model_engine.iter_states}")

            it += 1

            if it in self.profile_start_iters and not self.is_warmup:
                assert not enabled, "Inconsistent CUDA profiling state"
                calibrator.start()
                torch.cuda.cudart().cudaProfilerStart()
                if enable_torch_trace:
                    torch_profiler.start()
                logger.info(f"Profiling started at iteration {it}.")
                enabled = True
            calibrator.pre_step(it)
            start_time = time.time()
            if it % 2 == 0:
                if start_event_1 is None:
                    start_event_1 = torch.cuda.Event(enable_timing=True)
                start_event_1.record()
            else:
                if start_event_2 is None:
                    start_event_2 = torch.cuda.Event(enable_timing=True)
                start_event_2.record()

        try:
            yield profile_step
        finally:
            if enabled:
                # Stop on early exit / exception
                if enable_torch_trace:
                    torch_profiler.stop()
                    torch_profiler.export_chrome_trace(torch_trace_path)
                    logger.info(f"Profiling stopped at iteration {it}, "
                                f"trace saved to {torch_trace_path}")
                torch.cuda.cudart().cudaProfilerStop()
                calibrator.stop()

    def _get_init_iter_stats(self, num_new_active_requests,
                             new_active_requests_queue_latency_ms):
        stats = IterationStats()
        stats.timestamp = datetime.datetime.now().strftime(
            "%m-%d-%Y %H:%M:%S.%f")

        stats.num_new_active_requests = num_new_active_requests
        stats.num_active_requests = len(self.active_requests)
        stats.new_active_requests_queue_latency_ms = new_active_requests_queue_latency_ms
        stats.inflight_batching_stats = InflightBatchingStats()
        # staticBatchingStats is not used in pytorch path
        stats.static_batching_stats = StaticBatchingStats()

        # Create specdec_stats if speculative decoding is enabled
        # Either via spec_resource_manager (two-model mode) or spec_config (one-model mode)
        spec_resource_manager = self.resource_manager.resource_managers.get(
            ResourceManagerType.SPEC_RESOURCE_MANAGER)
        has_spec_config = self.model_engine.spec_config is not None

        if spec_resource_manager is not None or has_spec_config:
            stats.specdec_stats = SpecDecodingStats()
            # Reset draft latency at the start of each iteration to prevent stale values
            # from previous iterations when speculation is disabled
            if self.drafter is not None and hasattr(self.drafter,
                                                    'last_draft_latency_ms'):
                self.drafter.last_draft_latency_ms = 0.0

        return stats

    def _populate_req_stats(
            self, finished_requests: List[LlmRequest],
            active_requests: List[LlmRequest],
            scheduled_requests: ScheduledRequests
    ) -> Optional[List[RequestStats]]:

        def get_req_stats(req: LlmRequest) -> RequestStats:
            req_stat = RequestStats()
            req_stat.id = req.request_id
            req_stat.context_prefill_position = req.context_current_position
            req_stat.num_generated_tokens = req.max_beam_num_tokens - req.orig_prompt_len
            req_stat.avg_num_decoded_tokens_per_iter = req.avg_decoded_tokens_per_iter
            req_stat.alloc_total_blocks_per_request = req.alloc_total_blocks
            req_stat.alloc_new_blocks_per_request = req.alloc_new_blocks
            req_stat.reused_blocks_per_request = req.reused_blocks
            req_stat.missed_blocks_per_request = req.missed_blocks
            req_stat.kv_cache_hit_rate_per_request = req.kv_cache_hit_rate
            req_stat.scheduled = req in scheduled_requests.context_requests or req in scheduled_requests.generation_requests
            if req.llm_request_type == LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY or req.llm_request_type == LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY:
                req_stat.dis_serving_stats = DisServingRequestStats()
                req_stat.dis_serving_stats.kv_cache_transfer_ms = req.kv_cache_transfer_time_ms
                req_stat.dis_serving_stats.kv_cache_size = req.kv_cache_size
            return req_stat

        def get_queued_req_stats(request_id: int) -> RequestStats:
            req_stat = RequestStats()
            req_stat.id = request_id
            req_stat.context_prefill_position = 0
            req_stat.num_generated_tokens = 0
            req_stat.avg_num_decoded_tokens_per_iter = 0
            req_stat.alloc_total_blocks_per_request = 0
            req_stat.alloc_new_blocks_per_request = 0
            req_stat.reused_blocks_per_request = 0
            req_stat.missed_blocks_per_request = 0
            req_stat.kv_cache_hit_rate_per_request = 0
            return req_stat

        req_stats = []
        for req in active_requests:
            req_stat = get_req_stats(req)
            req_stat.stage = req.stage
            req_stats.append(req_stat)

        for req in list(self.executor_request_queue.get_request_queue().queue):
            if isinstance(req, RequestQueueItem):
                req_stat = get_queued_req_stats(req.id)
                req_stat.stage = RequestStage.QUEUED
                req_stats.append(req_stat)

        for req in finished_requests:
            req_stat = get_req_stats(req)
            req_stat.stage = RequestStage.GENERATION_COMPLETE
            req_stats.append(req_stat)

        return req_stats

    def _update_iter_stats(self, stats, iter_latency_ms, num_completed_requests,
                           scheduled_batch, micro_batch_id) -> IterationStats:
        stats.iter_latency_ms = iter_latency_ms

        stats.num_queued_requests = self.executor_request_queue.get_request_queue_size(
        )
        stats.num_completed_requests = num_completed_requests
        stats.max_num_active_requests = self.max_num_active_requests

        end, total_gpu_memory = torch.cuda.mem_get_info()
        stats.gpu_mem_usage = total_gpu_memory - end
        stats.cpu_mem_usage = 0
        stats.pinned_mem_usage = 0

        stats.iter = self.iter_counter

        kv_cache_manager = self.resource_manager.resource_managers.get(
            ResourceManagerType.KV_CACHE_MANAGER)
        if kv_cache_manager is not None:
            kv_stats = kv_cache_manager.get_kv_cache_stats()
            kv_stats_to_save = KvCacheStats()
            kv_stats_to_save.max_num_blocks = kv_stats.max_num_blocks
            kv_stats_to_save.free_num_blocks = kv_stats.free_num_blocks
            kv_stats_to_save.used_num_blocks = kv_stats.used_num_blocks
            kv_stats_to_save.tokens_per_block = kv_stats.tokens_per_block
            kv_stats_to_save.alloc_total_blocks = kv_stats.alloc_total_blocks
            kv_stats_to_save.alloc_new_blocks = kv_stats.alloc_new_blocks
            kv_stats_to_save.reused_blocks = kv_stats.reused_blocks
            kv_stats_to_save.missed_blocks = kv_stats.missed_blocks
            kv_stats_to_save.cache_hit_rate = kv_stats.cache_hit_rate
            stats.kv_cache_stats = kv_stats_to_save

        stats.inflight_batching_stats.num_scheduled_requests = len(
            scheduled_batch.context_requests) + len(
                scheduled_batch.generation_requests)
        stats.inflight_batching_stats.num_context_requests = len(
            scheduled_batch.context_requests)
        stats.inflight_batching_stats.num_gen_requests = len(
            scheduled_batch.generation_requests)
        stats.inflight_batching_stats.num_paused_requests = len(
            scheduled_batch.paused_requests)
        stats.inflight_batching_stats.avg_num_decoded_tokens_per_iter = 0
        stats.inflight_batching_stats.micro_batch_id = micro_batch_id

        if stats.specdec_stats is not None:
            total_draft_tokens = 0
            total_accepted_tokens = 0
            num_requests_with_draft = 0

            # Aggregate stats from all generation requests
            for req in scheduled_batch.generation_requests:
                draft_len = getattr(req, 'num_draft_tokens', 0)
                py_draft_tokens = getattr(req, 'py_draft_tokens', None)
                py_num_accepted = getattr(req, 'py_num_accepted_draft_tokens',
                                          None)

                # Use py_draft_tokens length if num_draft_tokens is 0
                if draft_len == 0 and py_draft_tokens is not None:
                    # Count non-zero draft tokens
                    draft_len = sum(1 for t in py_draft_tokens if t != 0)

                if draft_len > 0:
                    total_draft_tokens += draft_len
                    accepted_tokens = py_num_accepted if py_num_accepted is not None else 0
                    total_accepted_tokens += accepted_tokens
                    num_requests_with_draft += 1

            stats.specdec_stats.num_draft_tokens = total_draft_tokens
            stats.specdec_stats.num_accepted_tokens = total_accepted_tokens
            stats.specdec_stats.num_requests_with_draft_tokens = num_requests_with_draft

            # Calculate acceptance length: average tokens produced per step for requests with draft tokens
            if num_requests_with_draft > 0:
                # acceptance_length = (total_accepted_tokens + num_requests_with_draft) / num_requests_with_draft
                # Each request produces 1 target token + accepted draft tokens per iteration
                stats.specdec_stats.acceptance_length = float(
                    total_accepted_tokens +
                    num_requests_with_draft) / float(num_requests_with_draft)
            else:
                stats.specdec_stats.acceptance_length = 0.0

            # Get draft latency from drafter if available (only for two-model mode)
            # Only use draft latency if there were actually draft tokens in this iteration
            draft_latency_ms = 0.0
            if total_draft_tokens > 0 and self.drafter is not None and hasattr(
                    self.drafter, 'last_draft_latency_ms'):
                draft_latency_ms = getattr(self.drafter,
                                           'last_draft_latency_ms', 0.0)

            stats.specdec_stats.iter_latency_ms = draft_latency_ms

            # Calculate draft overhead
            stats.specdec_stats.draft_overhead = 0.0 if iter_latency_ms <= 0.0 else float(
                draft_latency_ms) / float(iter_latency_ms)
        return stats

    def _append_iter_stats(self,
                           stats: IterationStats,
                           req_stats: Optional[List[RequestStats]] = None):

        with self.stats_lock:
            if len(self.stats) > self.max_stats_len:
                self.stats.pop(0)
            self.stats.append((stats, req_stats))

    def _process_iter_stats(
        self,
        finished_requests: list[LlmRequest],
        active_requests: List[LlmRequest],
        batch_state: BatchState,
        micro_batch_id: int = 0,
    ):
        iter_end_time = time.time()
        iter_latency_ms = (iter_end_time - batch_state.iter_start_time) * 1e3
        if batch_state.iter_stats is None:
            return

        req_stats = self._populate_req_stats(
            finished_requests, active_requests,
            batch_state.sample_state.scheduled_requests) if (
                self.enable_iter_req_stats
                and self.enable_iter_perf_stats) else None

        self._append_iter_stats(
            self._update_iter_stats(batch_state.iter_stats, iter_latency_ms,
                                    len(finished_requests),
                                    batch_state.sample_state.scheduled_requests,
                                    micro_batch_id), req_stats)

    def _executor_loop_cleanup(self):

        for h in self.send_handles:
            if h is not None:
                h.wait()

        with self.response_cv:
            self.is_shutdown = True
            self.response_cv.notify_all()
        self.shutdown_event.set()

    def _pp_schedule_and_propagate(self):
        """The first PP rank schedules the requests and propagates the result to all other PP ranks."""

        # The first PP rank schedules the requests, other ranks receive the schedule result from the previous PP rank.
        if self.dist.is_first_pp_rank:
            scheduled_batch, fitting_disagg_gen_init_requests, num_fitting_reqs = self._schedule(
            )
            serializable_schedule = SerializableSchedulerOutput.from_scheduler_result(
                scheduled_batch, fitting_disagg_gen_init_requests,
                num_fitting_reqs)
        else:
            with nvtx_range("recv_schedule_from_prev_pp"):
                serializable_schedule = self.dist.recv_object(
                    self.dist.prev_pp_rank, PP_COMM_TAG_SCHEDULE_RESULT)
            scheduled_batch, fitting_disagg_gen_init_requests, num_fitting_reqs = serializable_schedule.to_scheduler_result(
                self.active_requests)

        # Propagate the schedule result to the next PP rank except the last PP rank.
        if not self.dist.is_last_pp_rank:
            if self.send_schedule_handler is not None:
                with nvtx_range("wait_send_schedule_handler"):
                    self.send_schedule_handler.wait()
            with nvtx_range("send_schedule_to_next_pp"):
                self.send_schedule_handler = self.dist.isend_object(
                    serializable_schedule, self.dist.next_pp_rank,
                    PP_COMM_TAG_SCHEDULE_RESULT)
        return scheduled_batch, fitting_disagg_gen_init_requests, num_fitting_reqs

    def _pp_retry_until_can_schedule(self, scheduled_batch):
        """
        If current rank cannot run the scheduled batch, it will retry following steps until it has enough KV cache resources or reach maximum retry count:
        1. Wait for cache transceiver to finish at least one cache transmission.
        2. Terminate requests that have finished context cache transmission.
        3. Check if current rank has enough KV cache resources to run the scheduled batch.
        """
        scheduled_batch_requests = scheduled_batch.all_requests()
        if self.scheduler.can_schedule(scheduled_batch_requests):
            return

        logger.warning(
            "Cannot run first PP's schedule result due to limited KV cache resources. This may cause bubbles in the PP pipeline. Please consider increasing the KV cache size by setting `free_gpu_memory_fraction` to a larger value."
        )
        if self.kv_cache_transceiver is None:
            raise RuntimeError(
                "KV cache transceiver is not enabled, but current rank cannot run first PP's schedule result due to limited KV cache resources. This is not expected."
            )
        if not self.async_transfer_manager.has_any_inflight_requests():
            raise RuntimeError(
                "No context cache transmission is in progress, but current rank cannot run first PP's schedule result due to limited KV cache resources. This is not expected."
            )
        if self.enable_kv_cache_reuse and self._disagg_pp_termination_handler is not None:
            raise RuntimeError(
                "Cannot terminate requests in cache transmission and release their KV cache resources when block reuse is enabled. Please consider increasing the KV cache size."
            )

        for retry_count in range(self.pp_scheduler_max_retry_count):
            if self.scheduler.can_schedule(scheduled_batch_requests):
                break
            logger.debug(
                f"Retrying to run first PP's schedule result ({retry_count + 1}/{self.pp_scheduler_max_retry_count})"
            )

            # Let cache transceiver finish at least one cache transmission and release requests' KV cache resources
            self._check_disagg_ctx_cache_transfer_status(1)
            self._check_kv_transfer_timeout()
        else:
            raise RuntimeError(
                f"Reach maximum PP retry count ({self.pp_scheduler_max_retry_count}) but still cannot run first PP's schedule result. Please consider increasing the KV cache size by setting `free_gpu_memory_fraction` to a larger value. Or you can set `TLLM_PP_SCHEDULER_MAX_RETRY_COUNT` to a larger value to allow more retries."
            )

    def _executor_loop_pp(self):
        logger.debug(f"Starting executor loop for pp_rank {self.dist.pp_rank}")
        torch.cuda.set_device(self.device_id)
        # ensure the context is created, otherwise, some MPI calls will fail.
        CUASSERT(cudart.cudaSetDevice(self.device_id))
        microbatch_id = 0
        with self._profiler() as profile_step, self.hang_detector:
            iter_start_time = time.time()
            iter_stats = None
            while True:
                self.hang_detector.checkpoint()
                profile_step()
                if self.enable_iter_perf_stats:
                    iter_start_time = time.time()

                # Fetch new requests from request queue
                new_requests = self._fetch_and_activate_new_requests()
                if self.should_stop_processing:
                    break

                self._handle_control_request()

                if self.kv_cache_transceiver:
                    self._check_disagg_gen_transfer_status()

                if self.enable_iter_perf_stats:
                    iter_stats = self._get_init_iter_stats(
                        len(new_requests),
                        self._get_new_active_requests_queue_latency())

                self._pad_attention_dp_dummy_request()

                # Stage 0: first PP rank schedules requests and propagates the result to all other PP ranks.
                scheduled_batch, fitting_disagg_gen_init_requests, num_fitting_reqs = self._pp_schedule_and_propagate(
                )
                if not self.dist.is_first_pp_rank:
                    # Retry until current rank can run first PP's schedule result.
                    self._pp_retry_until_can_schedule(scheduled_batch)
                    # Run scheduler locally because scheduler may change llm requests' state.
                    self.scheduler.schedule_request(self.active_requests,
                                                    self.inflight_req_ids)

                # For requests that are fitting disagg gen init, also prepare resources for KV cache manager
                if self.kv_cache_transceiver:
                    self._prepare_disagg_gen_init(
                        fitting_disagg_gen_init_requests)

                    if num_fitting_reqs == 0 and not fitting_disagg_gen_init_requests:
                        logger.warning(
                            "num_fitting_reqs=0 and fitting_disagg_gen_init_requests is empty, may not have enough kvCache"
                        )
                        self._check_disagg_ctx_cache_transfer_status(1)

                self.num_scheduled_requests = scheduled_batch.batch_size

                logger.debug(
                    f'iteration {self.iter_counter}, microbatch {microbatch_id}, '
                    f'has {len(self.active_requests)} active_requests, '
                    f'scheduled {len(scheduled_batch.context_requests)} context requests and '
                    f'{len(scheduled_batch.generation_requests)} generation requests'
                )

                can_queue, _ = self._can_queue(scheduled_batch)
                if not can_queue:
                    logger.debug(
                        f"microbatch {microbatch_id} cannot be queued, skipping"
                    )
                    self.micro_batches[microbatch_id] = None
                else:
                    logger.debug(f"microbatch {microbatch_id} can be queued")
                    finished_ctx_reqs = self._add_inflight_ids(scheduled_batch)

                    if self.kv_cache_transceiver:
                        # For generation requests which have completed KV cache transfer
                        self._prepare_disagg_gen_transmission_complete(
                            scheduled_batch)

                    self.resource_manager.prepare_resources(scheduled_batch)

                    # The generation requests that are do not have batch_idx,
                    # needs to be in front of the batch due to the assumptions
                    # made in model_engine.py::_forward_step. This is only important
                    # for disaggregated serving. For non-disaggregated serving,
                    # the generation requests always have batch_idx.
                    scheduled_batch.generation_requests = sorted(  # stable sort
                        scheduled_batch.generation_requests,
                        key=lambda req: int(req.py_batch_idx is not None),
                    )

                    if self.kv_cache_transceiver:
                        # Return the first token to the client
                        self._handle_first_token_response(scheduled_batch)

                    # Stage 1: Async forward (all ranks) and decoding pass (last rank only)
                    if not self.dist.is_last_pp_rank:
                        with torch.cuda.nvtx.range(
                                f"_forward_step_inter_pp pp_rank {self.dist.pp_rank}"
                        ):
                            sample_state = self._forward_step_inter_pp(
                                scheduled_batch)
                    else:
                        with torch.cuda.nvtx.range(
                                f"_forward_step_last_pp pp_rank {self.dist.pp_rank}"
                        ):
                            # init_disagg_gen_requests must be before engine forward, where the prev_seq_slot is updated.
                            if self.guided_decoder is not None and self.kv_cache_transceiver:
                                self.guided_decoder.add_batch(scheduled_batch)
                                self.guided_decoder.init_disagg_gen_requests()

                            batch_outputs = self._forward_step(scheduled_batch)

                            guided_decoder_failed_requests = None
                            if self.guided_decoder is not None:
                                self.guided_decoder.add_batch(scheduled_batch)
                                guided_decoder_failed_requests = self.guided_decoder.execute(
                                    batch_outputs['logits'])

                            if self.pp_multi_stream_sample:
                                # Wait for the previous sample to finish.
                                self.finish_sample_event.wait()
                                # Copy the batch outputs as sampler inputs
                                # to avoid next forward step overwriting them.
                                batch_outputs_copy = {
                                    name: tensor.clone()
                                    for name, tensor in batch_outputs.items()
                                }
                                self.start_sample_event.record()
                                with torch.cuda.stream(self.sample_stream):
                                    self.start_sample_event.wait()
                                    sample_state = self._sample_async(
                                        scheduled_batch, batch_outputs_copy)
                                    self.finish_sample_event.record()
                            else:
                                sample_state = self._sample_async(
                                    scheduled_batch, batch_outputs)
                            assert sample_state is not None, "Sampling failed"

                            # Handle guided decoder errors after _sample_async to avoid state conflicts.
                            # If called before, failed requests would be marked as GENERATION_COMPLETE,
                            # causing _sample_async to fail when accessing context_chunk_size property.
                            self._handle_guided_decoder_errors(
                                scheduled_batch, guided_decoder_failed_requests)

                            self._update_request_states(scheduled_batch)

                    if self.enable_iter_perf_stats:
                        iter_stats.inflight_batching_stats.num_ctx_tokens = self.model_engine.iter_states[
                            'num_ctx_tokens']
                    batch_state = BatchStatePP(
                        sample_state=sample_state,
                        iter_start_time=iter_start_time,
                        iter_stats=iter_stats,
                        microbatch_id=microbatch_id,
                        scheduled_ctx_reqs=scheduled_batch.context_requests,
                        finished_ctx_reqs=finished_ctx_reqs,
                    )

                    self.micro_batches[microbatch_id] = batch_state

                # sync sampler for previous microbatch to start new sample state comm chain.
                prev_microbatch_id = (microbatch_id -
                                      1) % self.num_micro_batches
                previous_batch = self.micro_batches[prev_microbatch_id]
                if previous_batch is not None:
                    with nvtx_range("sync_previous_sampler_event"):
                        previous_batch.sample_state.sampler_event.synchronize()

                # Stage 2: Communicate sample state for previous batch between ranks
                # send/recv chain: (pp_size - 1) -> 0 -> 1 -> ... -> (pp_size - 2)
                # intermediate ranks: send/recv sample state for next microbatch to allow overlap
                offset = -1 if self.dist.is_last_pp_rank else 1
                prev_microbatch_id = (microbatch_id +
                                      offset) % self.num_micro_batches
                previous_batch = self.micro_batches[prev_microbatch_id]
                tag = PP_COMM_TAG_SAMPLE_STATE_BASE + prev_microbatch_id
                if previous_batch is not None:
                    sample_state = previous_batch.sample_state
                    if not self.dist.is_last_pp_rank:
                        # Receive tokens from previous pp rank (w.r.t model forward direction)
                        with nvtx_range("recv_sample_state"):
                            sample_state.host = self.dist.recv_object(
                                src=self.dist.prev_pp_rank,
                                tag=tag,
                            )

                    # Send tokens to next pp rank (w.r.t model forward direction)
                    # Second last rank does not need to since last rank has original decoded tokens
                    if not self.dist.is_second_last_pp_rank:
                        self.wait_on_pp_send_handles(prev_microbatch_id)
                        with nvtx_range("send_sample_state"):
                            self.send_handles[
                                prev_microbatch_id] = self.dist.isend_object(
                                    sample_state.host,
                                    dest=self.dist.next_pp_rank,
                                    tag=tag)

                # Stage 3: Finalize previous batch that finished sample state communication
                # In last pp rank, stage 2 and 3 process different previous batches
                prev_microbatch_id = (microbatch_id +
                                      1) % self.num_micro_batches
                previous_batch = self.micro_batches[prev_microbatch_id]
                finished_requests = []
                if previous_batch is not None:
                    with torch.cuda.nvtx.range("_handle_previous_batch_pp"):
                        sample_state = previous_batch.sample_state
                        sample_state.scheduled_requests.context_requests = previous_batch.finished_ctx_reqs
                        self._update_requests(previous_batch.sample_state)

                        if self.kv_cache_transceiver:
                            self._send_kv_async(
                                previous_batch.scheduled_ctx_reqs)
                        self._handle_canceled_requests()

                        self._handle_logits_communication(
                            previous_batch, prev_microbatch_id)

                        finished_requests = self._handle_responses()
                        previous_scheduled_batch = previous_batch.sample_state.scheduled_requests
                        attn_metadata = getattr(self.model_engine,
                                                'attn_metadata', None)
                        kv_cache_dtype_byte_size = getattr(
                            self.model_engine, 'kv_cache_dtype_byte_size', None)
                        self.resource_manager.update_resources(
                            previous_scheduled_batch, attn_metadata,
                            kv_cache_dtype_byte_size)

                        self._remove_inflight_ids(previous_batch)

                    self.wait_on_pp_send_handles(prev_microbatch_id)
                    self.micro_batches[prev_microbatch_id] = None

                if self.kv_cache_transceiver and self.async_transfer_manager.has_any_inflight_requests(
                ):
                    self._check_kv_transfer_timeout()

                if self._disagg_pp_termination_handler is not None:
                    self._disagg_pp_termination_handler.terminate_pending_requests(
                    )

                # march forward in microbatch slots
                microbatch_id = (microbatch_id + 1) % self.num_micro_batches

                if self.enable_iter_perf_stats and previous_batch is not None:
                    sample_state = previous_batch.sample_state
                    sample_state.scheduled_requests.context_requests = previous_batch.scheduled_ctx_reqs
                    self._process_iter_stats(finished_requests,
                                             self.active_requests,
                                             previous_batch, microbatch_id)

                self.iter_counter += 1

    @nvtx_range("wait_on_pp_send_handles")
    def wait_on_pp_send_handles(self, microbatch_id):
        if self.send_handles[microbatch_id] is not None:
            self.send_handles[microbatch_id].wait()
            self.send_handles[microbatch_id] = None

    def _can_queue(self, scheduled_batch):

        # can_queue_this_rank is for case that the batch is not empty on this rank, but empty on other ranks
        # For bs == 1, we cannot pad dummy request to make the batch non-empty since it will cause the batch size to be 2.
        # 1 for dummy request, 1 for the to complete but haven't updated request.
        if self.enable_attention_dp:
            tp_batch_sizes = self.dist.tp_allgather(scheduled_batch.batch_size)
            can_queue = 0 not in tp_batch_sizes
            can_queue_this_rank = scheduled_batch.batch_size > 0
        else:
            can_queue = can_queue_this_rank = scheduled_batch.batch_size > 0

        return can_queue, can_queue_this_rank

    def _prepare_and_schedule_batch(self):
        new_requests = self._fetch_and_activate_new_requests()
        if self.should_stop_processing:
            return None, None

        if self.kv_cache_transceiver:
            self._check_disagg_ctx_schedulable_status(new_requests)
            self._check_disagg_gen_transfer_status()
            self._check_kv_transfer_timeout()

        iter_stats = None
        if self.enable_iter_perf_stats:
            iter_stats = self._get_init_iter_stats(
                len(new_requests),
                self._get_new_active_requests_queue_latency())

        self._pad_attention_dp_dummy_request()

        if self.drafter is not None:
            # Honor permanent disable flag based on rolling acceptance first
            if self.drafter.draft_len_schedule is not None:
                batch_size_input = len(self.active_requests)

                self.max_total_draft_tokens = self.drafter.get_draft_len_for_batch_size(
                    batch_size_input)

                self.drafter.update_max_total_draft_tokens(
                    self.max_total_draft_tokens)

            # Check if draft_len=0 → immediately disable
            # self.max_total_draft_tokens==0 is only possible when draft_len_schedule is provided
            # for example, draft_len_schedule = {1:4, 4:2, 8:0}, batch_size >= 8 will set self.max_draft_len = 0
            if self.drafter.draft_len_schedule is not None and self.max_total_draft_tokens == 0:
                self.use_spec_decode = False
            elif getattr(self, 'speculation_permanently_disabled', False):
                self.use_spec_decode = False
            else:
                self.use_spec_decode = self.drafter.should_use_spec_decode(
                    self.active_requests, self.max_batch_size,
                    self.model_engine.llm_args.max_num_tokens,
                    self.max_total_draft_tokens)
            logger.debug(f"Use spec decode: {self.use_spec_decode}")
            self.model_engine.enable_spec_decode = self.use_spec_decode

            # Set up draft_tokens in active_requests, because they could be used in the scheduling stage.
            for request in self.active_requests:
                if request.state not in (
                        LlmRequestState.GENERATION_IN_PROGRESS,
                        LlmRequestState.DISAGG_GENERATION_INIT):
                    continue
                request.draft_tokens = [
                    0
                ] * self.max_total_draft_tokens if self.max_total_draft_tokens > 0 else []

            # If speculation is off, this function sets py_draft_tokens to []
            # for all active requests. If it's on, we initialize py_draft_tokens
            # with dummy draft tokens to make the scheduler aware of the fact
            # that speculation is about to happen.
            self._prepare_draft_requests()

        scheduled_batch, fitting_disagg_gen_init_requests, num_fitting_reqs = self._schedule(
        )

        if self.drafter is not None and not self.use_spec_decode:
            for request in scheduled_batch.all_requests():
                request.py_disable_speculative_decoding = True

        if self.kv_cache_transceiver:
            # For requests that are fitting disagg gen init, also prepare resources for KV cache manager
            self._prepare_disagg_gen_init(fitting_disagg_gen_init_requests)

            if num_fitting_reqs == 0 and not fitting_disagg_gen_init_requests:
                logger.warning(
                    "num_fitting_reqs=0 and fitting_disagg_gen_init_requests is empty, may not have enough kvCache"
                )
                self._check_disagg_ctx_cache_transfer_status(1)

        self.num_scheduled_requests = scheduled_batch.batch_size
        logger.debug(
            f'has {len(self.active_requests)} active_requests, '
            f'scheduled {len(scheduled_batch.context_requests)} context requests and '
            f'{len(scheduled_batch.generation_requests)} generation requests')
        return scheduled_batch, iter_stats

    def _kv_connector_start_batch(self, scheduled_batch):
        if self.kv_connector_manager:
            self.kv_connector_manager.take_scheduled_requests_pending_load(
                scheduled_batch)
            self.kv_connector_manager.handle_metadata()
            self.kv_connector_manager.worker.start_load_kv(
                torch.cuda.current_stream())

    def _kv_connector_terminate_requests(self):
        if self.kv_connector_manager:
            reqs_to_terminate = self.kv_connector_manager.get_finished()
            for req in reqs_to_terminate:
                self._end_transfer_and_maybe_terminate(req)

    def _kv_connector_wait_for_save(self):
        if self.kv_connector_manager is not None:
            self.kv_connector_manager.worker.wait_for_save(
                torch.cuda.current_stream())

    def _executor_loop(self):
        torch.cuda.set_device(self.device_id)
        # ensure the context is created, otherwise, some MPI calls will fail.
        CUASSERT(cudart.cudaSetDevice(self.device_id))
        with self._profiler() as profile_step, self.hang_detector:
            sample_state = None
            iter_start_time = time.time()
            iter_stats = None
            while True:
                self.hang_detector.checkpoint()
                profile_step()
                if self.enable_iter_perf_stats:
                    iter_start_time = time.time()

                scheduled_batch, iter_stats = self._prepare_and_schedule_batch()
                self._handle_control_request()

                if scheduled_batch is None:
                    break

                self._terminate_requests(scheduled_batch.paused_requests)
                self._pause_requests(scheduled_batch.paused_requests)

                finished_requests = []

                can_queue, _ = self._can_queue(scheduled_batch)
                if can_queue:
                    if self.kv_cache_transceiver:
                        # For generation requests which have completed KV cache transfer
                        self._prepare_disagg_gen_transmission_complete(
                            scheduled_batch)

                        # Return the first token to the client
                        self._handle_first_token_response(scheduled_batch)
                    self.resource_manager.prepare_resources(scheduled_batch)

                    self._kv_connector_start_batch(scheduled_batch)

                # if using a kv connector, we need to call can_queue again since scheduled_batch might have changed
                if self.kv_connector_manager:
                    can_queue, _ = self._can_queue(scheduled_batch)

                if can_queue:
                    # init_disagg_gen_requests must be before drafter loop, otherwise draft requests do not have initialized matchers.
                    # init_disagg_gen_requests must be before engine forward, where the prev_seq_slot is updated.
                    if self.guided_decoder is not None:
                        self.guided_decoder.add_batch(scheduled_batch)
                        if self.kv_cache_transceiver:
                            self.guided_decoder.init_disagg_gen_requests()

                    if self.drafter is not None and self.use_spec_decode:
                        if self.guided_decoder is not None:
                            self.guided_decoder.rollback_rejected_tokens()
                        with request_context(
                                is_draft=self.draft_model_engine is not None,
                                scheduled_requests=scheduled_batch):
                            self.drafter.prepare_draft_tokens(
                                scheduled_batch, self.resource_manager)
                            # Pad draft tokens to the max draft length. This is for CUDA graph compatibility.
                            self.drafter.pad_draft_tokens_for_cuda_graph(
                                scheduled_batch)
                        # add_batch must be called again to restore to target requests with updated draft tokens.
                        if self.guided_decoder is not None:
                            self.guided_decoder.add_batch(scheduled_batch)
                            if hasattr(self.drafter, "guided_decoder"):
                                self.guided_decoder.rollback_draft_tokens()

                    batch_outputs = self._forward_step(scheduled_batch)

                    guided_decoder_failed_requests = None
                    if self.guided_decoder is not None:
                        guided_decoder_failed_requests = self.guided_decoder.execute(
                            batch_outputs['logits'])

                    sample_state = self._sample_async(scheduled_batch,
                                                      batch_outputs)

                    # Handle guided decoder errors after _sample_async to avoid state conflicts.
                    # If called before, failed requests would be marked as GENERATION_COMPLETE,
                    # causing _sample_async to fail when accessing context_chunk_size property.
                    self._handle_guided_decoder_errors(
                        scheduled_batch, guided_decoder_failed_requests)

                    if self.drafter is not None:
                        self.drafter.run_drafter_post(scheduled_batch,
                                                      self.resource_manager,
                                                      self.is_warmup)

                    self._update_request_states(scheduled_batch)
                    self._update_requests(sample_state, self.resource_manager)

                    self._send_kv_async(scheduled_batch.context_requests +
                                        scheduled_batch.generation_requests)

                    self._handle_canceled_requests()
                    finished_requests = self._handle_responses()
                    attn_metadata = getattr(self.model_engine, 'attn_metadata',
                                            None)
                    kv_cache_dtype_byte_size = getattr(
                        self.model_engine, 'kv_cache_dtype_byte_size', None)
                    self.resource_manager.update_resources(
                        scheduled_batch, attn_metadata,
                        kv_cache_dtype_byte_size)
                    if self.enable_kv_cache_events:
                        self._add_kv_cache_events()

                if self.kv_cache_transceiver and self.async_transfer_manager.has_any_inflight_requests(
                ):
                    self._check_kv_transfer_timeout()

                self._kv_connector_terminate_requests()

                if self.enable_iter_perf_stats and sample_state is not None:
                    iter_stats.inflight_batching_stats.num_ctx_tokens = self.model_engine.iter_states[
                        'num_ctx_tokens']
                    self._process_iter_stats(
                        finished_requests, self.active_requests,
                        BatchState(sample_state=sample_state,
                                   iter_stats=iter_stats,
                                   iter_start_time=iter_start_time))

                self.iter_counter += 1

    def _prepare_draft_requests(self):
        try:
            # Set draft tokens here to make the KV cache manager
            # and scheduler aware of them.
            for req in self.active_requests:
                if req.state not in (LlmRequestState.GENERATION_IN_PROGRESS,
                                     LlmRequestState.DISAGG_GENERATION_INIT):
                    continue

                req.py_last_draft_tokens = req.py_draft_tokens

                if self.max_total_draft_tokens > 0 and self.use_spec_decode and not req.py_disable_speculative_decoding:
                    req.py_draft_tokens = [0] * self.max_total_draft_tokens
                    req.py_draft_pages_allocated = self.max_total_draft_tokens
                else:
                    req.py_draft_tokens = []
                    req.py_draft_pages_allocated = 0

        except Exception as e:
            traceback.print_exc()
            error_msg = str(e)
            logger.error(f"Encountered an error in decode: {error_msg}")
            self._handle_errors(error_msg)

    def _handle_control_request(self):
        if len(self.active_requests) == 0 and \
            len(self.waiting_queue) == 0 and \
            len(self.control_requests) > 0:
            assert len(self.control_requests) == 1, (
                f"Expected exactly one control request to be processed at a time, "
                f"but found {len(self.control_requests)} control requests. "
                f"This may indicate a race condition or improper control request handling."
            )
            self.control_requests.pop(0)
            self.control_request_barrier.set()
            self.control_action_done.wait()
            self.control_action_done.clear()

    @contextmanager
    def control_action(self):
        """
        Context manager for synchronized control actions.

        Usage:
            with control_action():
                # Eventloop thread has finished all previous requests and paused
                do some actions here
            # Eventloop thread resumes automatically after exiting
        """

        if self.dist.rank == 0:
            self.executor_request_queue.enqueue_control_request()

        # Wait for worker to finish all previous requests
        self.control_request_barrier.wait()

        try:
            # Yield control to the with block
            # Worker is now paused, safe to execute actions
            yield self
        finally:
            # Cleanup: signal worker to resume
            self.control_action_done.set()
            self.control_request_barrier.clear()

    def _executor_loop_overlap(self):
        torch.cuda.set_device(self.device_id)
        # ensure the context is created, otherwise, some MPI calls will fail.
        CUASSERT(cudart.cudaSetDevice(self.device_id))
        with self._profiler() as profile_step, self.hang_detector:
            iter_start_time = time.time()
            iter_stats = None
            target_inputs = None
            previous_tensors_device = None
            can_forward = False if self.benchmark_req_queues_size > 0 and self.kv_cache_transceiver else True
            while True:
                self.hang_detector.checkpoint()
                profile_step()
                if self.enable_iter_perf_stats:
                    iter_start_time = time.time()

                scheduled_batch, iter_stats = self._prepare_and_schedule_batch()
                self._handle_control_request()

                if scheduled_batch is None:
                    break
                # In gen-only benchmarking mode, wait until the number of scheduled generation
                # requests reaches the required threshold before starting forward pass,
                # to ensure consistent batch sizes for accurate performance measurement.
                if not self.is_warmup and not can_forward:
                    if self.enable_attention_dp:
                        local_can_forward = self.num_fetch_requests + \
                            len(scheduled_batch.generation_requests) >= self.benchmark_req_queues_size
                        all_can_forward = self.dist.tp_allgather(
                            local_can_forward)
                        if all(all_can_forward):
                            can_forward = True
                            time.sleep(10)
                        else:
                            if self.dist.rank == 0:
                                logger.info(
                                    f"sleep 10 seconds, num_fetched_requests: {self.num_fetch_requests}, scheduled_gen_batch: {len(scheduled_batch.generation_requests)}"
                                )
                            time.sleep(10)
                            continue
                    else:
                        if len(scheduled_batch.generation_requests
                               ) < self.benchmark_req_queues_size:
                            if self.dist.rank == 0:
                                logger.info(
                                    f"sleep 10 seconds, scheduled_gen_batch: {len(scheduled_batch.generation_requests)}"
                                )
                            time.sleep(10)
                            continue
                        else:
                            can_forward = True

                self._terminate_requests(scheduled_batch.paused_requests)

                can_queue, can_queue_this_rank = self._can_queue(
                    scheduled_batch)
                if can_queue:
                    if self.kv_cache_transceiver:
                        # For generation requests which have completed KV cache transfer
                        self._prepare_disagg_gen_transmission_complete(
                            scheduled_batch)

                    has_draft_batch = self.drafter is not None and self.previous_batch is not None and self.use_spec_decode and self.drafter.should_forward_draft_model(
                        scheduled_batch)
                    # Reset the draft tokens to avoid preparing resources for the draft model.
                    if self.drafter is not None and self.use_spec_decode and not has_draft_batch:
                        self.use_spec_decode = False
                        # We are not running the draft model. Remove the draft tokens and turn off spec
                        # decode so that the requests get handled correctly.
                        # One corner case: when we have at least one context request, we have to keep spec
                        # dec on. This ensures that we capture hidden states for requests that haven't done
                        # prefill yet.
                        self.use_spec_decode = False
                        self.model_engine.enable_spec_decode = len(
                            scheduled_batch.context_requests) > 0
                        if not self.model_engine.enable_spec_decode:
                            for request in scheduled_batch.all_requests():
                                request.py_draft_tokens = []

                    self.resource_manager.prepare_resources(scheduled_batch)

                    self._kv_connector_start_batch(scheduled_batch)

                # if using a kv connector, we need to call can_queue again since scheduled_batch might have changed
                if self.kv_connector_manager:
                    can_queue, can_queue_this_rank = self._can_queue(
                        scheduled_batch)

                # If the batch is not empty on this rank, but empty on other ranks,
                # we need to delay the update of the previous batch's sample state,
                # and let the later iteration to update it.
                should_process_previous_batch = can_queue or not can_queue_this_rank
                if can_queue:

                    # The generation requests that are do not have batch_idx,
                    # needs to be in front of the batch due to the assumptions
                    # made in model_engine.py::_forward_step. This is only important
                    # for disaggregated serving. For non-disaggregated serving,
                    # the generation requests always have batch_idx.
                    scheduled_batch.generation_requests = sorted(  # stable sort
                        scheduled_batch.generation_requests,
                        key=lambda req: int(req.py_batch_idx is not None),
                    )

                    if self.kv_cache_transceiver:
                        # Return the first token to the client
                        self._handle_first_token_response(scheduled_batch)

                    # init_disagg_gen_requests must be before engine forward, where the prev_seq_slot is updated.
                    if self.guided_decoder is not None and self.kv_cache_transceiver:
                        self.guided_decoder.add_batch(scheduled_batch)
                        self.guided_decoder.init_disagg_gen_requests()

                    previous_tensors = self.previous_batch and self.previous_batch.sample_state
                    # If there are previous draft tokens, we need to update the target requests to accept some draft tokens.
                    # When there's any accepted tokens, we can't directly use the previous batch's outputs in this iteration for the target model,
                    # so we'll set the target model's input to None and skip updating the target requests after target model forward.
                    use_previous_draft_tokens = self.has_previous_draft_tokens
                    num_accepted_tokens_device = None

                    target_inputs = None
                    num_accepted_tokens_device = None

                    if has_draft_batch:
                        target_inputs, num_accepted_tokens_device = self._handle_speculative_decoding(
                            scheduled_batch, previous_tensors,
                            previous_tensors_device)

                    # Use the draft_model's outputs if we've launched the draft model.
                    # Otherwise, use the previous batch's outputs.
                    if (target_inputs is not None
                            and target_inputs.next_draft_tokens
                            is not None) or use_previous_draft_tokens:
                        previous_tensors_device = target_inputs
                    else:
                        previous_tensors_device = self.previous_batch and self.previous_batch.sample_state and self.previous_batch.sample_state.device

                    batch_outputs = self._forward_step(
                        scheduled_batch, previous_tensors_device,
                        num_accepted_tokens_device)

                if self.previous_batch is not None and should_process_previous_batch:
                    self._update_requests(self.previous_batch.sample_state)
                    self._send_kv_async(self.previous_batch.all_requests)

                if self.drafter is not None and self.use_spec_decode and should_process_previous_batch:
                    # Cleanup previous draft resources used in the draft model
                    self.drafter.cleanup_previous_draft_resources()

                self._pause_requests(scheduled_batch.paused_requests)

                if can_queue:
                    guided_decoder_failed_requests = None
                    if self.guided_decoder is not None:
                        # add_batch must be called again to have updated new tokens.
                        self.guided_decoder.add_batch(scheduled_batch)
                        guided_decoder_failed_requests = self.guided_decoder.execute(
                            batch_outputs['logits'])

                    sample_state = self._sample_async(scheduled_batch,
                                                      batch_outputs)
                    assert sample_state is not None, "Sampling failed"

                    # Handle guided decoder errors after _sample_async to avoid state conflicts.
                    # If called before, failed requests would be marked as GENERATION_COMPLETE,
                    # causing _sample_async to fail when accessing context_chunk_size property.
                    self._handle_guided_decoder_errors(
                        scheduled_batch, guided_decoder_failed_requests)

                    self._update_request_states(scheduled_batch)

                if self.previous_batch is not None and should_process_previous_batch:
                    self._process_previous_batch()
                else:
                    self._enqueue_responses([])

                if can_queue:
                    if self.enable_iter_perf_stats:
                        iter_stats.inflight_batching_stats.num_ctx_tokens = self.model_engine.iter_states[
                            'num_ctx_tokens']

                    self.previous_batch = BatchState(
                        sample_state=sample_state,
                        iter_start_time=iter_start_time,
                        iter_stats=iter_stats,
                        all_requests=scheduled_batch.all_requests())
                elif not can_queue_this_rank:
                    # If the batch is empty on this rank, we need to clear the previous batch.
                    self.previous_batch = None

                if self.kv_cache_transceiver and self.async_transfer_manager.has_any_inflight_requests(
                ):
                    self._check_kv_transfer_timeout()

                self._kv_connector_terminate_requests()

                self.iter_counter += 1

    @nvtx_range("_accept_draft_tokens")
    def _accept_draft_tokens(
        self, scheduled_batch: ScheduledRequests,
        target_outputs: SampleStateTensors,
        target_inputs: Optional[SampleStateTensors]
    ) -> Tuple[SampleStateTensorsMTP, Optional[torch.Tensor]]:
        """
        Prepare target device inputs after computing draft token acceptance.

        This function:
        1. If draft tokens exist: compares sampled tokens with draft tokens to compute acceptance
        2. If no draft tokens: directly uses the first sampled token
        3. Creates new_tokens by extracting accepted tokens per request

        Args:
            scheduled_batch: The scheduled requests
            target_outputs: Contains new_tokens [max_draft_len + 1, batch_size, beam_width]
                                or [1, batch_size, beam_width] if no draft tokens
            target_inputs: Contains next_draft_tokens [batch_size, max_draft_len]
        Returns:
            Tuple of:
            - SampleStateTensorsMTP with new_tokens set to accepted tokens,
              new_tokens_lens and next_draft_tokens set to None
            - num_accepted_tokens: [batch_size] tensor with acceptance counts per request,
              or None if no draft tokens
        """
        has_draft_tokens = target_inputs is not None and isinstance(
            target_inputs, SampleStateTensorsMTP
        ) and target_inputs.next_draft_tokens is not None
        target_tokens = target_outputs.new_tokens  # [max_draft_len + 1, batch_size, beam_width] or [1, batch_size, beam_width]
        new_tokens = torch.zeros_like(target_tokens)

        # Squeeze the beam dimension (beam_width=1 for greedy or single beam)
        target_tokens = target_tokens.squeeze(
            -1)  # [max_draft_len + 1, batch_size] or [1, batch_size]

        batch_size = target_tokens.shape[1]
        device = target_tokens.device
        # Compute number of accepted tokens per request
        num_accepted_tokens = torch.zeros(batch_size,
                                          dtype=torch.int32,
                                          device=device)

        if has_draft_tokens:
            # Draft tokens exist, compute acceptance
            draft_tokens = target_inputs.next_draft_tokens  # [batch_size, max_draft_len]
            max_draft_len = draft_tokens.shape[1]

            # Compute number of accepted tokens per request
            # Generation requests: compare with draft tokens to find acceptance
            num_contexts = len(scheduled_batch.context_requests)
            if batch_size > num_contexts:
                # Use .T to transpose: [max_draft_len + 1, num_gens] -> [num_gens, max_draft_len + 1]
                gen_target_tokens = target_tokens[:,
                                                  num_contexts:].T  # [num_gens, max_draft_len + 1]

                # Compare draft tokens with target tokens to find acceptance
                # Use cumprod to find the first rejection point
                draft_tokens_gen = draft_tokens[
                    num_contexts:, :].int()  # [num_gens, max_draft_len]
                num_accepted_tokens[num_contexts:] += torch.cumprod(
                    (draft_tokens_gen == gen_target_tokens[:, :max_draft_len]
                     ).int(),
                    dim=-1).sum(dim=1)

            # Vectorized extraction using advanced indexing (no GPU-CPU sync)
            # Use num_accepted_tokens as indices to gather the right tokens
            batch_indices = torch.arange(batch_size, device=device)
            new_tokens[0, :, 0] = target_tokens[num_accepted_tokens,
                                                batch_indices]
        else:
            # No draft tokens to accept, just use the first (and only) sampled token
            batch_indices = torch.arange(batch_size, device=device)
            new_tokens[0, :, 0] = target_tokens[0, batch_indices]

        # Create the updated SampleStateTensorsMTP
        # new_tokens_lens and next_draft_tokens are left as None
        result_tensors = SampleStateTensorsMTP(
            new_tokens=new_tokens,
            log_probs=target_outputs.log_probs,
            new_tokens_lens=None,
            next_draft_tokens=None)

        # Copy logits if available
        if hasattr(target_outputs, 'logits'):
            result_tensors.logits = target_outputs.logits

        return result_tensors, num_accepted_tokens

    def _process_previous_batch(self):
        self._handle_canceled_requests()
        finished_requests = self._handle_responses()
        scheduled_requests = self.previous_batch.sample_state.scheduled_requests
        attn_metadata = getattr(self.model_engine, 'attn_metadata', None)
        kv_cache_dtype_byte_size = getattr(self.model_engine,
                                           'kv_cache_dtype_byte_size', None)
        self.resource_manager.update_resources(scheduled_requests,
                                               attn_metadata,
                                               kv_cache_dtype_byte_size)
        if self.enable_kv_cache_events:
            self._add_kv_cache_events()

        if self.enable_iter_perf_stats:
            self._process_iter_stats(finished_requests, self.active_requests,
                                     self.previous_batch)

    def _forward_step_inter_pp(self, scheduled_batch) -> SampleState:
        self._forward_step(scheduled_batch)
        sampler_event = torch.cuda.Event()
        sampler_event.record()
        self._update_request_states(scheduled_batch)
        return self.sampler.SampleState(
            scheduled_requests=scheduled_batch,
            sampler_event=SamplerEvent(cuda_event=sampler_event),
        )

    def _validate_request(self, request: LlmRequest):
        # Validate beam width
        sampling_config = request.sampling_config
        if sampling_config is not None:
            if sampling_config.beam_width != self.max_beam_width:
                raise ValueError(
                    f"Request beam width {sampling_config.beam_width} "
                    f"is not equal to max_beam_width {self.max_beam_width}. This is not supported!"
                )

        # Check token ID ranges
        if isinstance(self.model_engine.model, DecoderModelForCausalLM):
            # Only skip token‐range checks for Llama4 when the request has multimodal data
            from ..models.modeling_llama import Llama4ForConditionalGeneration
            if isinstance(self.model_engine.model,
                          Llama4ForConditionalGeneration):
                has_mm = bool(request.py_multimodal_data)
                if has_mm:
                    logger.debug(
                        f"Skipping token-range validation for {type(self.model_engine.model).__name__} "
                        "(multimodal request)")
                    return

            # FIXME: This check is necessary because of how Qwen2ForProcessRewardModel
            #        subclasses DecoderModelForCausalLM. Perhaps the functionality
            #        of DecoderModelForCausalLM reused by Qwen2ForProcessRewardModel
            #        should be factored out into a separate class instead.
            if not hasattr(self.model_engine.model, "lm_head"):
                return

            if not request.check_token_id_range(
                    self.model_engine.model.lm_head.num_embeddings):
                raise ValueError("Token ID out of range")

    def _fetch_and_enqueue_requests(self,
                                    waiting_queue: deque[RequestQueueItem],
                                    total_num_active_requests: int) -> None:
        """Fetch requests from request_queue and enqueue to waiting_queue."""
        # Block new requests while control requests are pending
        if len(self.control_requests) != 0:
            return

        # Calculate timeout
        idle = (total_num_active_requests == 0) and len(waiting_queue) == 0
        if idle:
            # In Ray path (TLLM_DISABLE_MPI=1), use a periodic heartbeat timeout so rank 0
            # reaches the broadcast path regularly to prevent trtllm-serve timeout when idle.
            timeout = datetime.timedelta(
                seconds=1200) if self._disable_mpi else None
        else:
            timeout = datetime.timedelta(0)

        # Fetch requests from rank 0
        new_requests = []
        if self.dist.rank == 0:
            # Process accumulated requests that were queued during control request handling.
            if len(self.request_accumulated) != 0:
                new_requests.extend(self.request_accumulated)
                self.request_accumulated.clear()
                # Reset timeout to 0 to avoid hanging when no new requests are available
                timeout = datetime.timedelta(0)
            with self.hang_detector.pause():
                new_requests.extend(
                    self.executor_request_queue.get_from_request_queue(timeout))

        # Broadcast requests and handle Python objects
        new_requests, py_request_objects = self.request_broadcaster.broadcast(
            new_requests)

        # Validate and filter requests
        new_requests = self._handle_special_queue_items(new_requests)

        # Attach Python objects to requests
        if py_request_objects and (self.dist.tp_size > 1 or self.dist.has_pp
                                   or self.dist.cp_size
                                   > 1) and self.dist.rank > 0:
            attach_py_objects_to_requests(new_requests, py_request_objects)

        waiting_queue.extend(new_requests)

    def _pop_from_waiting_queue(
        self,
        waiting_queue: deque[RequestQueueItem],
        total_num_active_requests: int,
        all_ranks_num_active_requests: Optional[List[int]] = None
    ) -> List[RequestQueueItem]:
        """Pop requests from waiting_queue based on available capacity."""
        if self.enable_attention_dp:
            total_max = self.dist.tp_size * self.max_num_active_requests
        else:
            total_max = self.max_num_active_requests

        max_new_requests = total_max - total_num_active_requests

        return get_from_waiting_queue(
            waiting_queue,
            max_new_requests,
            enable_attention_dp=self.enable_attention_dp,
            max_num_active_requests=self.max_num_active_requests,
            all_ranks_num_active_requests=all_ranks_num_active_requests)

    @nvtx_range("_fetch_new_requests")
    def _fetch_new_requests(
            self, waiting_queue: deque[RequestQueueItem],
            activate_requests: List[LlmRequest]) -> List[LlmRequest]:
        """Fetch new requests and return LlmRequests ready for execution."""
        # 1. Gather info and calculate total_num_active_requests
        if self.enable_attention_dp:
            all_ranks_num_active_requests = []
            all_ranks_num_active_tokens = []
            if self.dist.has_cp_helix:
                num_active_tokens = sum(
                    [req.total_input_len_cp for req in activate_requests])
            else:
                num_active_tokens = sum(
                    [req.py_orig_prompt_len for req in activate_requests])
            responses_list = self.dist.tp_allgather(
                [len(activate_requests), num_active_tokens])
            for num_active_requests, num_active_tokens in responses_list:
                all_ranks_num_active_requests.append(num_active_requests)
                all_ranks_num_active_tokens.append(num_active_tokens)
            total_num_active_requests = sum(all_ranks_num_active_requests)
        else:
            total_num_active_requests = len(activate_requests)
            all_ranks_num_active_requests = None

        # 2. Fetch and enqueue to waiting queue
        self._fetch_and_enqueue_requests(waiting_queue,
                                         total_num_active_requests)

        # 3. Pop requests from waiting queue
        new_requests = self._pop_from_waiting_queue(
            waiting_queue, total_num_active_requests,
            all_ranks_num_active_requests)

        # 4. Update performance metrics (before DP scheduling to clear all start_times)
        if self.enable_iter_perf_stats and self.dist.rank == 0:
            self._update_new_active_requests_queue_latency(new_requests)

        # 5. Schedule requests across ranks (DP only)
        if self.enable_attention_dp:
            all_ranks_new_requests, self.expected_num_active_requests = \
                schedule_attention_dp_requests(
                    new_requests, all_ranks_num_active_requests,
                    all_ranks_num_active_tokens, self.dist.tp_size,
                    self.max_num_active_requests)
            new_requests_cur_rank = all_ranks_new_requests[self.dist.tp_rank]

            # Update counters for DP
            self.num_fetch_requests += len(new_requests)
            self.num_fetch_requests_cur_rank += len(new_requests_cur_rank)

            new_requests = new_requests_cur_rank

        # 6. Merge requests
        return merge_requests(new_requests,
                              cp_config=self.dist.cp_config,
                              cp_rank=self.dist.cp_rank,
                              cp_size=self.dist.cp_size,
                              exclude_last_generation_logits=self.
                              _should_exclude_last_generation_logits())

    def _handle_special_queue_items(
            self,
            new_requests: List[RequestQueueItem]) -> List[RequestQueueItem]:
        """Handle special signals."""
        accepted_new_requests = []
        for idx, req_item in enumerate(new_requests):
            if req_item.is_shutdown_request:
                self.is_shutdown = True
                break
            elif req_item.is_canceled_request:
                self.canceled_req_ids.append(req_item.id)
            elif req_item.is_control_request:
                self.control_requests.append(req_item)
                if self.dist.rank == 0:
                    self.request_accumulated.extend(new_requests[idx + 1:])
                break
            else:
                accepted_new_requests.append(req_item)

        return accepted_new_requests

    def _update_new_active_requests_queue_latency(
            self, new_requests: List[RequestQueueItem]):
        """Update queue latency metrics for new requests."""
        now = time.time()
        latency = self.executor_request_queue.calculate_queue_latency(
            new_requests, now)
        self.new_active_requests_queue_latency_ms += latency

    def _get_new_active_requests_queue_latency(self) -> float:
        return self.new_active_requests_queue_latency_ms

    def _should_exclude_last_generation_logits(self) -> bool:
        return self.should_exclude_last_generation_logits

    def _fetch_and_activate_new_requests(self) -> List[LlmRequest]:

        def _respond_if_invalid(request: LlmRequest) -> bool:
            """Immediately fail invalid request.

            Return True if invalid request was encountered and
            handled.
            """
            try:
                self._validate_request(request)
                return False
            except Exception as e:
                self._handle_errors(str(e), requests=[request])
                return True

        new_requests_cur_rank = self._fetch_new_requests(
            self.waiting_queue, self.active_requests)

        validated_requests = [
            request for request in new_requests_cur_rank
            if not _respond_if_invalid(request)
        ]

        self.active_requests.extend(validated_requests)
        return validated_requests

    def _add_kv_cache_events(self):
        kv_cache_manager = self.resource_manager.resource_managers.get(
            ResourceManagerType.KV_CACHE_MANAGER)
        if not kv_cache_manager:
            return
        # Flush iteration events at each iteration to ensure that events have enough time
        # to be transferred to main thread when user needs them.
        kv_cache_manager.flush_iteration_events()

    def _balance_adp_requests(self, context_requests: list[LlmRequest],
                              generation_requests: list[LlmRequest]):
        balanced_context_requests = context_requests
        num_scheduled_context_requests = len(context_requests)
        num_scheduled_generation_requests = len(generation_requests)
        num_scheduled_tokens = sum(
            [len(req.get_tokens(0))
             for req in context_requests]) + num_scheduled_generation_requests
        # Note: We use tp_allgather instead of tp_cp_allgather because we want to
        # balance the requests across DP ranks; not CP ranks within those DP ranks.
        responses_list = self.dist.tp_allgather([
            num_scheduled_context_requests, num_scheduled_generation_requests,
            num_scheduled_tokens
        ])
        all_ranks_num_scheduled_context_requests = [
            response[0] for response in responses_list
        ]
        all_ranks_num_scheduled_generation_requests = [
            response[1] for response in responses_list
        ]
        all_ranks_have_free_ctx_slots = all([
            num_gen < self.max_batch_size
            for num_gen in all_ranks_num_scheduled_generation_requests
        ])
        all_ranks_have_ctx_requests = all([
            num_ctx > 0 for num_ctx in all_ranks_num_scheduled_context_requests
        ])
        all_ranks_have_gen_requests = all([
            num_gen > 0
            for num_gen in all_ranks_num_scheduled_generation_requests
        ])

        if self.attention_dp_enable_balance:
            # wait for all ranks have context requests
            if all_ranks_have_free_ctx_slots and all_ranks_have_ctx_requests:
                self.adp_ctx_waiting_iters_count = 0
                # balance number of context requests across ranks
                if all_ranks_have_gen_requests:
                    if self.adp_ctx_batching_wait_iters_count < self.attention_dp_batching_wait_iters:
                        self.adp_ctx_batching_wait_iters_count += 1
                        balanced_context_requests = []
                    else:
                        self.adp_ctx_batching_wait_iters_count = 0
            else:
                self.adp_ctx_waiting_iters_count += 1
                balanced_context_requests = []
                timeout_reached = self.adp_ctx_waiting_iters_count >= self.attention_dp_time_out_iters
                if timeout_reached or not all_ranks_have_gen_requests:
                    self.adp_ctx_waiting_iters_count = 0
                    balanced_context_requests = context_requests
        return balanced_context_requests

    def _waiting_requests(self, context_requests: list[LlmRequest],
                          generation_requests: list[LlmRequest]):
        """
        Return an empty list if scheduled requests fulfill the waiting conditions, otherwise return the original context requests.
        Waiting conditions:
        - The number of scheduled tokens (both context and generation) is smaller than `self.batch_wait_max_tokens_ratio * self.max_num_tokens`
        - The number of waiting iterations is smaller than `self.batch_wait_timeout_iters`.
        """

        num_scheduled_ctx_tokens = sum(
            len(ctx_req.get_tokens(0)) for ctx_req in context_requests)
        num_scheduled_gen_tokens = sum(1 + gen_req.num_draft_tokens
                                       for gen_req in generation_requests)
        num_scheduled_tokens = num_scheduled_ctx_tokens + num_scheduled_gen_tokens

        should_waiting = self.batch_wait_iters_count < self.batch_wait_timeout_iters and num_scheduled_tokens < self.batch_wait_max_tokens_ratio * self.max_num_tokens
        if should_waiting:
            self.batch_wait_iters_count += 1
            return []

        self.batch_wait_iters_count = 0
        return context_requests

    @nvtx_range("_schedule")
    def _schedule(self):
        scheduler_output = self.scheduler.schedule_request(
            self.active_requests, self.inflight_req_ids)

        scheduled_context_requests = scheduler_output.context_requests
        if self.enable_attention_dp and self.attention_dp_enable_balance:
            scheduled_context_requests = self._balance_adp_requests(
                scheduler_output.context_requests,
                scheduler_output.generation_requests)

        # If no generation requests, no need to wait, to avoid dead waiting
        should_check_waiting = not self.enable_attention_dp and self.enable_batch_waiting and len(
            scheduler_output.context_requests) > 0 and len(
                scheduler_output.generation_requests) > 0
        if should_check_waiting:
            scheduled_context_requests = self._waiting_requests(
                scheduler_output.context_requests,
                scheduler_output.generation_requests)

        scheduled_requests = ScheduledRequests()
        scheduled_requests.context_requests = scheduled_context_requests
        scheduled_requests.generation_requests = scheduler_output.generation_requests
        scheduled_requests.paused_requests = scheduler_output.paused_requests

        return scheduled_requests, scheduler_output.fitting_disagg_gen_init_requests, scheduler_output.num_fitting_requests

    @nvtx_range("_check_disagg_gen_transfer_status")
    def _check_disagg_gen_transfer_status(self):

        need_check = any([
            req.is_disagg_generation_transmission_in_progress
            for req in self.active_requests
        ])
        need_check_one = all([
            req.is_disagg_generation_transmission_in_progress
            for req in self.active_requests
        ])

        if need_check:
            at_least_num = 1 if need_check_one else 0
            self._check_disagg_gen_cache_transfer_status(at_least_num)

        return

    @nvtx_range("_check_kv_transfer_timeout")
    def _check_kv_transfer_timeout(self):
        if not self.kv_cache_transceiver:
            return
        timeout_ms = self.kv_cache_transceiver.kv_transfer_timeout_ms
        if timeout_ms is None:
            return

        def flag_if_kv_transfer_timed_out(req: LlmRequest, type: str) -> None:
            current_time = time.time()
            if req.py_kv_transfer_start_time is None:
                return
            elapsed_time = (current_time - req.py_kv_transfer_start_time) * 1000
            if elapsed_time > timeout_ms and not req.py_kv_transfer_timed_out:
                logger.warning(
                    f"Terminating {type} request {req.py_request_id} due to KV cache transfer timeout"
                )
                req.py_kv_transfer_timed_out = True

        for req in self.async_transfer_manager.requests_in_transfer().values():
            flag_if_kv_transfer_timed_out(req, "context")

        for req in self.active_requests:
            if req.is_disagg_generation_transmission_in_progress:
                flag_if_kv_transfer_timed_out(req, "generation")

        return

    @nvtx_range("_check_disagg_ctx_schedulable_status")
    def _check_disagg_ctx_schedulable_status(self,
                                             new_requests: List[LlmRequest]):
        """
        In context-first mode, context requests are scheduable immediately,
        otherwise, we need to check if context requests are ready to be scheduled by querying kv cache transceiver
        """
        if not self.kv_cache_transceiver:
            return
        ctx_only_requests = [
            req for req in new_requests
            if req.is_context_only_request and req.py_disaggregated_params.
            schedule_style == DisaggScheduleStyle.GENERATION_FIRST
        ]
        if ctx_only_requests:
            self.kv_cache_transceiver.prepare_context_requests(
                ctx_only_requests)

    @nvtx_range("_pad_attention_dp_dummy_request")
    def _pad_attention_dp_dummy_request(self):
        """
        Pad with a generation dummy request, if required, to ensure every attention_dp rank has at least one active request.
        """
        if not self.enable_attention_dp:
            return

        assert self.expected_num_active_requests >= len(self.active_requests)
        if self.kv_cache_transceiver is None:
            num_active_request = len(self.active_requests)
        else:
            num_active_request = len([
                req for req in self.active_requests
                if not (req.is_disagg_generation_init_state
                        or req.is_disagg_generation_transmission_in_progress)
            ])

        if self.expected_num_active_requests - num_active_request > 0 and num_active_request == 0:
            llm_request = self.kv_cache_manager.add_dummy_requests(
                request_ids=[0],
                is_gen=True,
                prepare_resource=True,
                max_num_draft_tokens=self.max_total_draft_tokens,
            )[0]
            llm_request.is_attention_dp_dummy = True
            spec_resource_manager = self.resource_manager.get_resource_manager(
                ResourceManagerType.SPEC_RESOURCE_MANAGER)
            if spec_resource_manager is not None:
                spec_resource_manager.add_dummy_requests([0])
            self.active_requests.append(llm_request)

    @nvtx_range("_prepare_disagg_gen_init")
    def _prepare_disagg_gen_init(self, fitting_disagg_gen_init_requests):
        if fitting_disagg_gen_init_requests:
            disagg_gen_init_to_prepare = ScheduledRequests()
            disagg_gen_init_to_prepare.context_requests = fitting_disagg_gen_init_requests
            disagg_gen_init_to_prepare.generation_requests = []
            disagg_gen_init_to_prepare.paused_requests = []

            for resource_mgr_type in (
                    ResourceManagerType.KV_CACHE_MANAGER,
                    ResourceManagerType.SPEC_RESOURCE_MANAGER,
                    ResourceManagerType.DRAFT_KV_CACHE_MANAGER):
                if (resource_mgr_type in self.resource_manager.resource_managers
                        and self.resource_manager.
                        resource_managers[resource_mgr_type] is not None):
                    self.resource_manager.resource_managers[
                        resource_mgr_type].prepare_resources(
                            disagg_gen_init_to_prepare)

            # Trigger KV cache exchange for new disagg_gen_init_requests
            self._recv_disagg_gen_cache(fitting_disagg_gen_init_requests)

    @nvtx_range("_prepare_disagg_gen_transmission_complete")
    def _prepare_disagg_gen_transmission_complete(self, scheduled_batch):
        cache_trans_complete_requests = []
        for req in scheduled_batch.generation_requests:
            if req.is_disagg_generation_transmission_complete:
                cache_trans_complete_requests.append(req)
        if len(cache_trans_complete_requests) > 0:
            requests = ScheduledRequests()
            requests.context_requests = cache_trans_complete_requests
            self.resource_manager.resource_managers[
                ResourceManagerType.SEQ_SLOT_MANAGER].prepare_resources(
                    requests)
            self._setup_sampler_step(requests)

        for req in scheduled_batch.generation_requests:
            if req.is_disagg_generation_transmission_complete:
                req.state = LlmRequestState.GENERATION_IN_PROGRESS
                req.context_current_position = req.prompt_len
                req.decoding_iter = 1
                req.py_decoding_iter = 1
                req.py_kv_transfer_start_time = None
                first_gen_tokens = req.context_phase_params.first_gen_tokens
                ctx_draft_tokens = req.context_phase_params.draft_tokens
                req.py_draft_tokens = [] if ctx_draft_tokens is None else ctx_draft_tokens
                beam_width = req.sampling_config.beam_width
                for beam in range(0, beam_width):
                    req.add_new_token(first_gen_tokens[beam], beam)

    @nvtx_range("_recv_disagg_gen_cache")
    def _recv_disagg_gen_cache(self, new_gen_reqs):

        # For gen-only benchmarking, mark new gen request as transmission complete right away
        if os.getenv("TRTLLM_DISAGG_BENCHMARK_GEN_ONLY") == "1":
            for req in new_gen_reqs:
                req.state = LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE
            return

        if os.getenv("TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP") == "1":
            for req in new_gen_reqs:
                self.kv_cache_transceiver.request_and_receive_sync(req)
        else:
            for req in new_gen_reqs:
                self.kv_cache_transceiver.request_and_receive_async(req)

        if self.kv_cache_transceiver.kv_transfer_timeout_ms is not None:
            for req in new_gen_reqs:
                if req.state == LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS:
                    req.py_kv_transfer_start_time = time.time()

        block_transfer = all([
            req.is_disagg_generation_transmission_in_progress
            for req in self.active_requests
        ])
        self._check_disagg_gen_cache_transfer_status(1 if block_transfer else 0)

        return

    @nvtx_range("_send_kv_async")
    def _send_kv_async(self, scheduled_requests: List[LlmRequest]):

        def kv_connector_request_finished(req: LlmRequest):
            try:
                cache_block_ids = self.kv_cache_manager.get_cache_indices(req)
            except Exception as e:
                logger.warning(
                    f"Unable to get cache blocks for request {req.py_request_id}. Skipping asynchronous saving: {e}"
                )
            else:
                if self.kv_connector_manager.request_finished(
                        req, cache_block_ids):
                    self.async_transfer_manager.start_transfer(req)

        if self.kv_cache_transceiver:
            for req in scheduled_requests:
                if req.is_context_only_request and (
                        req.is_context_finished or req.is_finished_due_to_length
                ) and not req.is_finished_due_to_cancellation:
                    self.kv_cache_transceiver.respond_and_send_async(req)

                    self.async_transfer_manager.start_transfer(req)

                    if self.kv_cache_transceiver.kv_transfer_timeout_ms is not None:
                        req.py_kv_transfer_start_time = time.time()

        if self.kv_connector_manager:
            if not self.disable_overlap_scheduler:
                requests = self.previous_batch.sample_state.scheduled_requests.all_requests(
                ) if self.previous_batch is not None else []
            else:
                requests = scheduled_requests
            for req in requests:
                if req.is_finished:
                    kv_connector_request_finished(req)

        if self.kv_cache_transceiver:
            self._check_disagg_ctx_cache_transfer_status(0)

    def _get_disagg_reqs_in_error_state(self):
        return [
            req for req in self.active_requests
            if req.state == LlmRequestState.DISAGG_TRANS_ERROR
        ]

    def _check_cache_transfer_errors(self, error_msg_prefix: str):
        """Common helper to check for and handle cache transfer errors."""
        error_requests = self._get_disagg_reqs_in_error_state()
        if error_requests:
            self._handle_errors(
                f"Error in kv cache transfer for {error_msg_prefix}",
                requests=error_requests)

    @nvtx_range("_check_disagg_ctx_cache_transfer_status")
    def _check_disagg_ctx_cache_transfer_status(self, atLeastNum: int = 0):
        finished_requests, error_requests = self.kv_cache_transceiver.check_context_transfer_status(
            atLeastNum)

        completed_req_ids = set(finished_requests + error_requests)

        requests_in_transfer = self.async_transfer_manager.requests_in_transfer(
        )

        for request_id in completed_req_ids:

            if request_id not in requests_in_transfer:
                logger.warning(
                    f"Request {request_id} not found in transfer manager")
                continue

            request = requests_in_transfer[request_id]

            self._end_transfer_and_maybe_terminate(request)

        # The set of requests in transfer may have changed since we terminated some requests.
        requests_in_transfer = self.async_transfer_manager.requests_in_transfer(
        )

        for request_id in list(requests_in_transfer.keys()):
            request = requests_in_transfer[request_id]
            if request.py_kv_transfer_timed_out and request_id not in completed_req_ids:
                is_cancelled = self.kv_cache_transceiver.cancel_request(request)
                # If cancel is successful, mark as complete so it can be cleaned up
                # Otherwise, try at next iteration
                if is_cancelled:
                    request.py_kv_transfer_start_time = None
                    request.state = LlmRequestState.DISAGG_CONTEXT_COMPLETE

                    self._end_transfer_and_maybe_terminate(request)

        self._check_cache_transfer_errors("context requests")

    @nvtx_range("_check_disagg_gen_cache_transfer_status")
    def _check_disagg_gen_cache_transfer_status(self, atLeastNum: int = 0):
        self.kv_cache_transceiver.check_gen_transfer_status(atLeastNum)
        self._check_cache_transfer_errors("generation requests")

    def _forward_step(
            self,
            scheduled_requests: ScheduledRequests,
            new_tensors_device: Optional[SampleStateTensors] = None,
            num_accepted_tokens_device: Optional[torch.Tensor] = None):
        ExpertStatistic.set_iter(self.iter_counter)

        @nvtx_range(
            f"[Executor] _forward_step {self.iter_counter}: {len(scheduled_requests.context_requests)} ctx reqs, {len(scheduled_requests.generation_requests)} gen reqs"
        )
        def forward(scheduled_requests, resource_manager, new_tensors_device,
                    gather_context_logits, cache_indirection_buffer,
                    num_accepted_tokens_device):
            return self.model_engine.forward(
                scheduled_requests,
                resource_manager,
                new_tensors_device,
                gather_context_logits=gather_context_logits,
                cache_indirection_buffer=cache_indirection_buffer,
                num_accepted_tokens_device=num_accepted_tokens_device)

        try:
            gather_context_logits = any(
                a.py_return_context_logits
                for a in scheduled_requests.context_requests)
            cache_indirection_buffer = self.sampler.get_cache_indirection()

            # Run model forward on the execution stream for proper synchronization
            # with KVCacheTransferManager's onboard/offload operations.
            self.execution_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.execution_stream):
                outputs = forward(scheduled_requests, self.resource_manager,
                                  new_tensors_device, gather_context_logits,
                                  cache_indirection_buffer,
                                  num_accepted_tokens_device)

            # Ensure the default stream waits for execution_stream to complete
            # before downstream operations use the outputs.
            torch.cuda.current_stream().wait_stream(self.execution_stream)

            self._kv_connector_wait_for_save()

            return outputs
        except Exception as e:
            traceback.print_exc()
            error_msg = str(e)
            logger.error(
                f"Encountered an error in forward function: {error_msg}")
            self._handle_errors(error_msg)
            return None

    def _update_request_states_tp(self, scheduled_requests: ScheduledRequests):
        # handle potential attention dp dummy request
        if self.active_requests and self.active_requests[
                -1].is_attention_dp_dummy:
            request = self.active_requests[-1]
            request.state = LlmRequestState.GENERATION_COMPLETE
            self.inflight_req_ids.erase(request.py_request_id)
            self._terminate_request(request)
            self.active_requests.remove(request)

        for request in scheduled_requests.context_requests:
            if request.state != LlmRequestState.GENERATION_COMPLETE:  # skip failed requests
                request.py_last_context_chunk = (
                    request.context_current_position,
                    request.context_current_position +
                    request.context_chunk_size)
                request.move_to_next_context_chunk()
            if request.context_remaining_length == 0:
                if not self.disable_overlap_scheduler and request.will_complete_next_iteration(
                ):
                    request.set_exclude_last_generation_logits(False)
                    request.state = LlmRequestState.GENERATION_TO_COMPLETE
                else:
                    request.state = LlmRequestState.GENERATION_IN_PROGRESS

        for request in scheduled_requests.generation_requests:
            if request.state != LlmRequestState.GENERATION_COMPLETE:
                if not self.disable_overlap_scheduler and request.will_complete_next_iteration(
                ):
                    request.set_exclude_last_generation_logits(False)
                    request.state = LlmRequestState.GENERATION_TO_COMPLETE

    def _update_request_states_star_attention(
            self, scheduled_requests: ScheduledRequests):
        for request in scheduled_requests.context_requests:
            if request.ctx_iters >= len(request.ctx_blocks) - 2:
                request.state = LlmRequestState.GENERATION_IN_PROGRESS
            request.ctx_iters += 1

        for request in scheduled_requests.generation_requests:
            request.gen_iters += 1

    @nvtx_range("_update_request_states")
    def _update_request_states(self, scheduled_requests: ScheduledRequests):
        cp_config = self.dist.cp_config
        if 'cp_type' in cp_config:
            cp_type = cp_config['cp_type']
            if cp_type == CpType.STAR:
                self._update_request_states_star_attention(scheduled_requests)
            elif cp_type == CpType.HELIX:
                # Take the usual route with _update_request_states_tp().
                pass
            else:
                raise NotImplementedError(
                    f'Unsupported cp type {cp_type.name}.')
        self._update_request_states_tp(scheduled_requests)

    @nvtx_range("_sample_async")
    def _sample_async(self, scheduled_batch,
                      batch_outputs) -> SampleState | None:
        try:
            if batch_outputs is not None:
                num_context_logits_prefix_sum = [0]
                prefix_sum = 0
                num_context_tokens = 0
                for request in scheduled_batch.context_requests:
                    context_chunk_size = request.context_chunk_size
                    prefix_sum += context_chunk_size if request.py_return_context_logits else 1
                    num_context_logits_prefix_sum.append(prefix_sum)
                    num_context_tokens += context_chunk_size

                beam_width = self.sampler.beam_width(
                    scheduled_batch.all_requests())

                HandleLogits()(scheduled_batch.context_requests,
                               scheduled_batch.generation_requests,
                               batch_outputs["logits"], beam_width,
                               num_context_logits_prefix_sum,
                               self.sampler.is_generation_model())

                HandleAdditionalOutputs()(scheduled_batch.context_requests,
                                          scheduled_batch.generation_requests,
                                          batch_outputs, beam_width,
                                          num_context_tokens)

                return self.sampler.sample_async(scheduled_batch, batch_outputs,
                                                 num_context_logits_prefix_sum)
        except Exception as e:
            traceback.print_exc()
            error_msg = str(e)
            logger.error(f"Encountered an error in sampling: {error_msg}")
            self._handle_errors(error_msg)

    @nvtx_range("_setup_sampler_step")
    def _setup_sampler_step(self, requests: ScheduledRequests):
        try:
            return self.sampler.setup_sampler_step(requests)
        except Exception as e:
            traceback.print_exc()
            error_msg = str(e)
            logger.error(f"Encountered an error in sampling: {error_msg}")
            self._handle_errors(error_msg)

    @nvtx_range("_update_requests")
    def _update_requests(self,
                         sample_state: SampleState,
                         resource_manager: Optional[ResourceManager] = None):
        try:
            self.sampler.update_requests(sample_state, resource_manager)
        except Exception as e:
            traceback.print_exc()
            error_msg = str(e)
            logger.error(f"Encountered an error in sampling: {error_msg}")
            self._handle_errors(error_msg)

    def _handle_errors(self,
                       error_msg: Optional[str] = None,
                       *,
                       requests: Optional[List[LlmRequest]] = None):
        error_responses: Dict[int, LlmResponse] = {}
        error_msg = error_msg or "error"
        failed_requests = requests if requests is not None else self.active_requests
        for request in failed_requests:
            req_id = request.py_request_id
            request.state = LlmRequestState.GENERATION_COMPLETE
            error_responses[req_id] = LlmResponse(
                request_id=req_id,
                error_msg=error_msg,
                client_id=request.py_client_id)
        if requests is None:
            self.active_requests.clear()
        else:
            self.active_requests = [
                request for request in self.active_requests
                if request not in requests
            ]
        self._enqueue_responses(list(error_responses.items()))
        for request in failed_requests:
            self._terminate_request(request)

    def _terminate_request(self, request: LlmRequest):
        if self._disagg_pp_termination_handler is not None:
            self._disagg_pp_termination_handler.terminate(request)
        else:
            self._do_terminate_request(request)

    def _do_terminate_request(self, request: LlmRequest):
        self.resource_manager.free_resources(request)

        if self.gather_all_responses or self.dist.rank == 0:
            self.result_wait_queues.pop(request.py_request_id, None)

    def _is_request_in_transmission(self, request) -> bool:
        """Check if a request is currently in transmission state."""
        return (request.state
                == LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS
                or request.state
                == LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS)

    def _try_cancel_request(self, request) -> bool:
        """Check if a request can be canceled and attempt cancellation if needed.

        Returns:
            bool: True if the request can be canceled (either successfully cancelled or doesn't need cancellation).
        """
        if self.kv_cache_transceiver is None:
            return True

        if not self._is_request_in_transmission(request):
            return True

        return self.kv_cache_transceiver.cancel_request(request)

    @nvtx_range("_handle_canceled_requests")
    def _handle_canceled_requests(self):
        if len(self.canceled_req_ids) == 0:
            return

        # Create set from list of canceled request ids to speed up canceled test
        canceled_req_ids_set = set(self.canceled_req_ids)

        # Remove canceled requests from the waiting queue
        self.waiting_queue = deque(req for req in self.waiting_queue
                                   if req.id not in canceled_req_ids_set)

        still_pending_canceled_ids = []
        for request in self.active_requests:
            req_id = request.py_request_id if not request.is_child else request.parent_request_id
            if req_id not in canceled_req_ids_set:
                continue

            is_cancelled = self._try_cancel_request(request)
            if is_cancelled:
                # Mark requests as finished, then, we reuse all existing code
                # to clean up the KV cache resources.
                request.finish_by_reason(FinishReason.CANCELLED)
                request.decoding_iter = request.py_decoding_iter
            else:
                still_pending_canceled_ids.append(req_id)

        # Clear list of requests marked for cancellation and add back those that failed to cancel.
        self.canceled_req_ids.clear()
        self.canceled_req_ids.extend(still_pending_canceled_ids)

    @nvtx_range("_enqueue_responses")
    def _enqueue_responses(self, responses: Iterable[Tuple[int, LlmResponse]]):
        if 0 not in self.dist.mapping.tp_group and not self.gather_all_responses:
            return

        if self.enable_attention_dp and self.dist.world_size != 1:
            if not self.gather_all_responses:
                responses_list = self.dist.tp_gather(responses)
            else:
                responses_list = self.dist.allgather(responses)
            if self.dist.rank == 0 or self.gather_all_responses:
                gather_responses = []
                if responses_list is not None:
                    for resp in responses_list:
                        if resp is not None:
                            gather_responses.extend(resp)
                    responses = gather_responses
        logger.debug(
            f'after gather, rank = {self.dist.rank}, responses = {responses}')

        if self.dist.rank == 0 or self.gather_all_responses:
            with self.response_cv:
                for req_id, resp in responses:
                    if req_id in self.responses.keys():
                        self.responses[req_id].append(resp)
                    else:
                        self.responses.update({req_id: [resp]})
                    # (TODO: joyang) There are other types of responses, we need to sort out.
                    if type(
                            resp
                    ) == LlmResponse and req_id in self.result_wait_queues and self.result_wait_queues[
                            req_id] is not None:
                        self.result_wait_queues[req_id].put_response.remote(
                            resp.client_id, resp)
                self.response_cv.notify_all()

    @nvtx_range("_handle_first_token_response")
    def _handle_first_token_response(self, scheduled_batch):
        new_responses = []
        for req in scheduled_batch.generation_requests:
            if req.py_decoding_iter == 1:
                logger.debug(
                    f'Send first token response for request {req.py_request_id}'
                )
                response = req.create_response(False, self.dist.rank)
                new_responses.append((req.py_request_id, response))

        self._enqueue_responses(new_responses)

    @nvtx_range("_handle_responses")
    def _handle_responses(self):
        new_responses = []
        requests_to_terminate = []
        new_active_requests = []
        logger.debug(
            f'------before _handle_responses, rank = {self.dist.rank}, output = {self.active_requests}'
        )
        for request in self.active_requests:
            req_id = request.py_request_id
            # no responses for dummy request, and finish it
            if request.is_attention_dp_dummy:
                requests_to_terminate.append(request)
                continue

            # Check if generation request needs cleanup due to KV cache transfer timeout
            if request.py_kv_transfer_timed_out:
                is_cancelled = self.kv_cache_transceiver.cancel_request(request)
                if is_cancelled:
                    self._handle_errors(
                        error_msg=f"Request {request.py_request_id} timed out",
                        requests=[request])
                continue

            if request.is_generation_only_request():
                # If request is in transmission, so we don't need to emit a response
                # Also, for the first iteration with overlap, we should skip since first
                # token has already been emitted previously
                if request.is_disagg_generation_transmission_in_progress or (
                        not self.disable_overlap_scheduler
                        and request.py_decoding_iter <= 1):
                    new_active_requests.append(request)
                    continue

            request.draft_tokens = request.py_draft_tokens if get_draft_token_length(
                request) > 0 else []
            request.decoding_iter = request.py_decoding_iter

            # Skip active requests that are not scheduled
            if request.return_perf_metrics and request.py_decoding_iter >= 1:
                request.update_perf_metrics(self.iter_counter)

            request_done = False
            if request.py_decoding_iter == 1 or request.is_finished or \
                    request.py_decoding_iter % self.stream_interval == 0:
                response = request.create_response(False, self.dist.rank)
                if response:
                    request_done = request.is_finished
                    response.result.cached_tokens = request.cached_tokens
                    new_responses.append((req_id, response))

            if request_done:
                if (self.drafter is not None and getattr(
                        self.model_engine, 'enable_spec_decode', False)
                        and not self.speculation_permanently_disabled
                        and not request.is_dummy and not self.is_warmup):
                    if self.speculation_gate is not None:
                        # Response handling runs on multiple PP ranks. Only the last PP rank performs
                        # sampling; restrict rolling stat updates to it to avoid overcounting.
                        if (not getattr(self.dist, 'has_pp',
                                        False)) or self.dist.is_last_pp_rank:
                            avg_decoded = getattr(
                                request, 'avg_decoded_tokens_per_iter', None)
                            if avg_decoded is not None:
                                disabled_now, _ = self.speculation_gate.record_avg_decoded(
                                    avg_decoded,
                                    request_id=getattr(request, 'py_request_id',
                                                       None))
                                if disabled_now:
                                    # disable speculation permanently
                                    # starting from next iteration, _prepare_and_schedule_batch will set self.use_spec_decode to False
                                    self.speculation_permanently_disabled = True
                            else:
                                logger.debug(
                                    f"Request {request.py_request_id} has no avg_decoded_tokens_per_iter"
                                )
                if self.enable_kv_cache_reuse and not self.kv_cache_manager.is_vswa:
                    requests_to_terminate.append(request)
                else:
                    if not request.is_disagg_context_transmission_state:
                        requests_to_terminate.append(request)
            else:
                new_active_requests.append(request)

        self.active_requests.clear()
        self.active_requests.extend(new_active_requests)
        # Request should be terminated after enqueueing response to ensure we can enqueue response successfully.
        self._enqueue_responses(new_responses)
        for request in requests_to_terminate:
            self._terminate_request(request)
        return requests_to_terminate

    def _handle_logits_communication(self, previous_batch, prev_microbatch_id):
        """Handle logits communication between pipeline parallel ranks.

        If logits were requested, the last PP rank sends to the first PP rank (who sends responses)
        the logits of the requests that have finished.

        Args:
            previous_batch: The previous batch state
            prev_microbatch_id: The microbatch ID for the previous batch
        """
        # NOTE: If the rank processing the logits ever becomes the same as
        # the rank sending the responses, this code can be removed.
        finished_reqs = [
            r for r in
            previous_batch.sample_state.scheduled_requests.all_requests()
            if r.state == LlmRequestState.GENERATION_COMPLETE and (
                r.py_return_context_logits or r.py_return_generation_logits
                or r.py_additional_outputs is not None)
        ]
        if self.dist.is_first_pp_rank and len(finished_reqs):
            finished_reqs_py_results = [r.py_result for r in finished_reqs]
            finished_reqs_py_results = self.dist.recv_object(
                src=self.dist.prev_pp_rank,
                tag=prev_microbatch_id,
            )
            for req, py_result in zip(finished_reqs, finished_reqs_py_results):
                req.py_result = py_result

        elif self.dist.is_last_pp_rank and len(finished_reqs):
            self.wait_on_pp_send_handles(prev_microbatch_id)
            self.send_handles[prev_microbatch_id] = self.dist.isend_object(
                [r.py_result for r in finished_reqs],
                dest=self.dist.next_pp_rank,
                tag=prev_microbatch_id)

    def _await_any_response(self,
                            timeout: Optional[float] = None
                            ) -> List[LlmResponse]:

        def any_responses_ready():
            return len(self.responses) > 0 or self.is_shutdown

        responses = []
        with self.response_cv:
            self.response_cv.wait_for(any_responses_ready, timeout=timeout)
            for req_id, response in self.responses.items():
                responses += response
            self.responses = {}

        return responses

    def _await_single_response(
            self,
            id: int,
            timeout: Optional[float] = None) -> List[LlmResponse]:
        with self.response_cv:

            def key_has_response():
                return id in self.responses.keys()

            self.response_cv.wait_for(key_has_response, timeout=timeout)
            response = self.responses[id]
            self.responses.pop(id)
            return response

    def _terminate_requests(self, requests_to_pause):
        # todo: support work with self.inflight_req_ids.
        #       Currently, self.inflight_req_ids is not.
        for req in requests_to_pause:
            self._terminate_request(req)

    def _pause_requests(self, requests_to_pause):
        for req in requests_to_pause:
            req.pause(self.max_input_len)

    def _add_inflight_ids(self, scheduled_requests):
        """Add request IDs of current requests to self.inflight_req_ids.

        Non‑final context chunks are not added to the inflight set, so the scheduler can keep scheduling further
        context chunks while earlier ones are in the PP pipeline. Only context requests that finish context phase
        are inserted into the inflight set and collected into finished_ctx_reqs.
        All generation requests are still inserted into the inflight set.
        """
        finished_ctx_reqs = []
        for req in scheduled_requests.context_requests:
            if req.is_last_context_chunk:
                logger.debug(
                    f"Context request with ID {req.request_id} added to DECODER model inflight set"
                )
                self.inflight_req_ids.insert(req.request_id)
                finished_ctx_reqs.append(req)
        for req in scheduled_requests.generation_requests:
            logger.debug(
                f"Generation request with ID {req.request_id} added to DECODER model inflight set"
            )
            self.inflight_req_ids.insert(req.request_id)
        return finished_ctx_reqs

    def _remove_inflight_ids(self, batch_state: BatchStatePP):
        """Remove request IDs of current requests from self.inflight_req_ids.

        Context IDs are erased from the inflight set using batch_state.finished_ctx_reqs.
        Generation IDs are erased using batch_state.sample_state.scheduled_requests.generation_requests.
        """
        for req in batch_state.finished_ctx_reqs:
            logger.debug(
                f"Context request with ID {req.request_id} removed from DECODER model inflight set"
            )
            self.inflight_req_ids.erase(req.request_id)
        for req in batch_state.sample_state.scheduled_requests.generation_requests:
            logger.debug(
                f"Generation request with ID {req.request_id} removed from DECODER model inflight set"
            )
            self.inflight_req_ids.erase(req.request_id)

    def _handle_speculative_decoding(
        self, scheduled_batch, previous_tensors, target_inputs
    ) -> Tuple[Optional[SampleStateTensorsMTP], Optional[torch.Tensor]]:
        with request_context(is_draft=self.draft_model_engine is not None,
                             scheduled_requests=scheduled_batch):
            target_outputs = self.previous_batch.sample_state and self.previous_batch.sample_state.device
            assert target_outputs is not None, "target_outputs should not be None"
            new_target_inputs, num_accepted_tokens_device = self._accept_draft_tokens(
                scheduled_batch=scheduled_batch,
                target_inputs=target_inputs,
                target_outputs=target_outputs)

            self.drafter.generate_draft_tokens_with_overlap(
                scheduled_batch, self.resource_manager,
                previous_tensors.device if previous_tensors else None,
                new_target_inputs, num_accepted_tokens_device)

            # Pad draft tokens to the max draft length for CUDA graph compatibility
            self.has_previous_draft_tokens = new_target_inputs is not None and new_target_inputs.next_draft_tokens is not None

        return new_target_inputs, num_accepted_tokens_device

    def reset_prefix_cache(self):
        self.kv_cache_manager.reset_reuse_state()

    def _handle_guided_decoder_errors(
            self, scheduled_batch: ScheduledRequests,
            failed_requests: Optional[List[Tuple[int, str]]]):
        """Handle errors that occurred during guided decoding.

        Args:
            scheduled_batch: The current batch of scheduled requests
            failed_requests: List of (request_id, error_message) tuples for failed requests,
                           or None if no failures occurred
        """
        if not failed_requests:
            return

        failed_req_id_to_err = {req_id: err for req_id, err in failed_requests}

        for request in scheduled_batch.all_requests():
            if request.py_request_id not in failed_req_id_to_err:
                continue
            error_msg = failed_req_id_to_err[request.py_request_id]
            self._handle_errors(error_msg, requests=[request])


class DisaggPPTerminationHandler:
    """Handles termination synchronization across pipeline parallel ranks under disaggregated serving.

    We require synchronization when terminating requests in disaggregated PP when
    KV cache reuse is enabled. All PP ranks need to reach consensus before freeing
    resources to avoid a NCCL hang.
    """

    def __init__(self, dist, terminator_func: Callable[[LlmRequest], None]):
        self._dist = dist
        self._terminator_func = terminator_func
        self._pending_termination = {}
        self._terminating_iteration = 0
        self._send_handle = None
        self._comm_tag = TERMINATION_COMM_TAG_BASE

    def terminate(self, request: LlmRequest):
        self._pending_termination[request.py_request_id] = request

    @nvtx_range("_disagg_pp_termination_handler_sync")
    def terminate_pending_requests(self):
        """
        Ring-style communicating to decide which requests to be terminated and avoid bubbles.
        This ensures that one request is terminated from rank_0 to rank_(pp_size-1) in order.
        """
        terminate_req_ids = []
        term_state = None
        if self._send_handle:
            self._send_handle.wait()

        if not (self._dist.is_first_pp_rank
                and self._terminating_iteration == 0):
            term_state = self._dist.recv_object(src=self._dist.prev_pp_rank,
                                                tag=self._comm_tag)

        ready_req_map = term_state["ready"] if term_state else {
        }  # {req_id: num_ranks} ranks vote in the ready dict
        terminate_req_ids = term_state["term"] if term_state else [
        ]  # request ids to be terminated in the current iteration

        reqs_to_terminate = {
            req_id: self._pending_termination.pop(req_id, None)
            for req_id in terminate_req_ids
            if req_id in self._pending_termination
        }

        if self._dist.is_first_pp_rank:
            # rank0 proposes the requests to be terminated
            ready_req_map = {req_id: 1 for req_id in self._pending_termination}
        else:
            # if a rank agrees to terminate a request, increase the vote count for the request id
            for req_id in ready_req_map.keys():
                if req_id in self._pending_termination:
                    ready_req_map[req_id] += 1

        if self._dist.is_last_pp_rank:
            new_terminate_req_ids = [
                req_id for req_id, num_ranks in ready_req_map.items()
                if num_ranks == self._dist.pp_size
            ]
            # by determining the terminate ids in the last rank, we can save the overhead of sending the ready dict back to rank0
            new_term_state = {"ready": {}, "term": new_terminate_req_ids}
        else:
            # other pp ranks pass the updated ready dict and terminate request ids to the next rank, and the
            # terminate_req_ids will not change in a given iteration, so we can terminate the requests synchronously
            new_term_state = {"ready": ready_req_map, "term": terminate_req_ids}

        self._send_handle = self._dist.isend_object(
            new_term_state, dest=self._dist.next_pp_rank, tag=self._comm_tag)

        if reqs_to_terminate:
            logger.debug(
                f'rank {self._dist.pp_rank} terminates {list(reqs_to_terminate.keys())} in iter {self._terminating_iteration}'
            )
        for req_id, req in reqs_to_terminate.items():
            if req:
                self._terminator_func(req)
        self._terminating_iteration += 1
