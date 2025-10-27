import dataclasses
import datetime
import functools
import gc
import os
import pickle  # nosec B403
import threading
import time
import traceback
import weakref
from contextlib import contextmanager
from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch

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

from ..distributed import Distributed
from ..models.modeling_utils import DecoderModelForCausalLM
from ..modules.decoder_layer import DecoderLayer
from ..speculative.drafter import Drafter
from ..speculative.speculation_gate import SpeculationGate
from .executor_request_queue import ExecutorRequestQueue, RequestQueueItem
from .guided_decoder import GuidedDecoder
from .handle_additional_outputs import HandleAdditionalOutputs
from .handle_logits import HandleLogits
from .kv_cache_connector import KvCacheConnectorManager
from .kv_cache_transceiver import KvCacheTransceiver
from .llm_request import (ExecutorRequest, LlmRequest, LlmRequestState,
                          LlmResponse, get_draft_token_length)
from .model_engine import ModelEngine
from .resource_manager import ResourceManager
from .sampler import Sampler, SampleState, SampleStateTensors
from .scheduler import RequestScheduler, ScheduledRequests

# Environment variable to specify iteration ranges for profiling start/stop.
# Format: "start1-stop1,start2-stop2,..." or single iterations "iter1,iter2,..."
PROFILE_START_STOP_ENV_VAR_NAME = "TLLM_PROFILE_START_STOP"

# Environment variable to enable garbage collection profiling.
# Set to "1" to enable recording of garbage collection events during profiling.
PROFILE_RECORD_GC_ENV_VAR_NAME = "TLLM_PROFILE_RECORD_GC"

# Environment variable to enable PyTorch profiler tracing.
# Set to a path to save detailed tracing of PyTorch operations.
PROFILE_TRACE_ENV_VAR_NAME = "TLLM_TORCH_PROFILE_TRACE"

# Unique tag base to avoid collisions with token/logits comms
TERMINATION_COMM_TAG_BASE = 20000


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


class _GCNvtxHandle:
    pass


def _gc_nvtx_watcher():
    enabled = os.environ.get(PROFILE_RECORD_GC_ENV_VAR_NAME, None)
    if not enabled:
        return None

    range_id: Optional[int] = None

    def gc_callback(phase, _):
        nonlocal range_id
        if phase == "start":
            assert range_id is None, "Unexpected state in GC callback: another GC while last GC not finished?"
            range_id = torch.cuda.nvtx.range_start("Python GC")
        elif phase == "stop":
            assert range_id is not None, "Unexpected state in GC callback: no active GC but got GC finished?"
            torch.cuda.nvtx.range_end(range_id)
            range_id = None

    gc.callbacks.append(gc_callback)

    def gc_cleanup(callback):
        try:
            gc.callbacks.remove(callback)
        except ValueError:
            pass

    handle = _GCNvtxHandle()
    weakref.finalize(handle, gc_cleanup, gc_callback)
    return handle


@dataclasses.dataclass
class BatchState:
    sample_state: SampleState

    iter_start_time: float = 0
    iter_stats: IterationStats = None
    ctx_transmission_reqs: list[LlmRequest] = None


@dataclasses.dataclass
class BatchStatePP(BatchState):
    microbatch_id: int = -1
    scheduled_ctx_reqs: list[LlmRequest] = None


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
                 max_input_len: int = 2048,
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
                 peft_cache_config: Optional[PeftCacheConfig] = None):
        super(PyExecutor, self).__init__()
        self.device_id = torch.cuda.current_device()
        self.global_rank = dist.rank

        self.peft_cache_config = peft_cache_config

        # profile config
        self.profile_start_iters, self.profile_stop_iters = _load_iteration_indexes(
            PROFILE_START_STOP_ENV_VAR_NAME)
        self.gc_nvtx_watcher_handle = _gc_nvtx_watcher()

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

        # enqueue and _fetch_new_requests used data
        self.active = True
        self.max_beam_width = max_beam_width
        self.max_draft_len = max_draft_len
        self.max_total_draft_tokens = max_total_draft_tokens
        self.max_num_tokens = model_engine.pytorch_backend_config.max_num_tokens
        self.print_log = model_engine.pytorch_backend_config.print_iter_log
        self.enable_iter_perf_stats = model_engine.pytorch_backend_config.enable_iter_perf_stats
        self.enable_iter_req_stats = model_engine.pytorch_backend_config.enable_iter_req_stats
        self.stream_interval = model_engine.pytorch_backend_config.stream_interval
        self.attention_dp_enable_balance = model_engine.pytorch_backend_config.attention_dp_enable_balance
        self.attention_dp_time_out_iters = model_engine.pytorch_backend_config.attention_dp_time_out_iters
        self.attention_dp_batching_wait_iters = model_engine.pytorch_backend_config.attention_dp_batching_wait_iters
        self.batch_wait_timeout_ms = model_engine.pytorch_backend_config.batch_wait_timeout_ms
        self.batch_wait_timeout_iters = model_engine.pytorch_backend_config.batch_wait_timeout_iters
        self.batch_wait_max_tokens_ratio = model_engine.pytorch_backend_config.batch_wait_max_tokens_ratio
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
        self.block_reuse_enabled = True if self.kv_cache_manager is not None and self.kv_cache_manager.enable_block_reuse else False
        self.enable_kv_cache_events = self.kv_cache_manager is not None and self.kv_cache_manager.event_buffer_max_size > 0
        self.enable_kv_cache_reuse = self.kv_cache_manager is not None and self.kv_cache_manager.enable_block_reuse

        self.max_input_len = max_input_len
        # _executor_loop private data
        self.max_num_active_requests = model_engine.get_max_num_sequences()
        self.active_requests: List[LlmRequest] = []
        self.expected_num_active_requests = 0
        self.ctx_in_transmission_requests = dict()
        self.ctx_in_transmission_counter = (1 if kv_cache_transceiver else
                                            0) + (1 if kv_connector_manager else
                                                  0)
        self.previous_batch: Optional[BatchState] = None
        self.has_previous_draft_tokens = False
        self.num_scheduled_requests: int = 0
        self.benchmark_req_queues_size = int(
            os.environ.get("TLLM_BENCHMARK_REQ_QUEUES_SIZE", 0))
        self._disable_mpi = mpi_disabled()

        # list of requests in each PP micro batch
        self.num_micro_batches = self.dist.pp_size
        self.micro_batches: List[BatchStatePP
                                 | None] = [None] * self.num_micro_batches
        self.send_handles = [None] * self.num_micro_batches

        self.inflight_req_ids = ReqIdsSet()

        # During warmup, we don't enable the profiler
        self.is_warmup = True
        self.model_engine.warmup(self.resource_manager)
        if self.draft_model_engine is not None:
            self.draft_model_engine.warmup(self.resource_manager)
        self.is_warmup = False

        self.is_shutdown = False
        self.max_batch_size = max_batch_size
        self.adp_ctx_waiting_iters_count = 0
        self.adp_ctx_batching_wait_iters_count = 0
        self.batch_wait_iters_count = 0

        # request fetcher initialization
        self._set_global_steady_clock_offset()
        self.executor_request_queue = ExecutorRequestQueue(
            dist=self.dist,
            enable_attention_dp=self.enable_attention_dp,
            max_batch_size=max_batch_size,
            max_beam_width=self.max_beam_width,
            max_num_active_requests=self.max_num_active_requests,
            enable_iter_perf_stats=self.enable_iter_perf_stats,
            batch_wait_timeout_ms=self.batch_wait_timeout_ms,
            is_disaggregated=kv_cache_transceiver is not None,
        )
        self.executor_request_queue.set_exclude_last_generation_logits(
            self.disable_overlap_scheduler, self.dist.pp_size)

        self.stats_lock = threading.Lock()
        self.stats = []
        self.gather_all_responses = False

        self.kv_cache_transceiver = kv_cache_transceiver

        # Initialize disagg PP termination handler if needed
        self._disagg_pp_termination_handler = None
        if self.dist.pp_size > 1 and self.enable_kv_cache_reuse and self.kv_cache_transceiver:
            self._disagg_pp_termination_handler = DisaggPPTerminationHandler(
                self.num_micro_batches, self.dist)

        if self.dist.pp_size > 1:
            self.event_loop = self._executor_loop_pp
        else:
            self.event_loop = self._executor_loop if disable_overlap_scheduler else self._executor_loop_overlap
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
        self.worker_thread.join()
        self.worker_started = False
        for manager in self.resource_manager.resource_managers.values():
            if manager:
                manager.shutdown()
        del self.model_engine
        if self.draft_model_engine is not None:
            del self.draft_model_engine

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
            self.executor_request_queue.get_waiting_queue_size() == 0

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

        def profile_step():
            nonlocal it, enabled, start_time, start_event_1, end_event_1, start_event_2, end_event_2, prev_device_step_time
            if it in self.profile_stop_iters and not self.is_warmup:
                assert enabled, "Inconsistent CUDA profiling state"
                if enable_torch_trace:
                    torch_profiler.stop()
                    torch_profiler.export_chrome_trace(torch_trace_path)
                    logger.info(f"Profiling stopped at iteration {it}, "
                                f"trace saved to {torch_trace_path}")
                torch.cuda.cudart().cudaProfilerStop()
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
                    f"iter = {self.model_engine.iter_counter}, "
                    f"global_rank = {self.global_rank}, "
                    f"rank = {self.dist.rank}, "
                    f"currank_total_requests = {self.executor_request_queue.num_fetch_requests_cur_rank}/"
                    f"{self.executor_request_queue.num_fetch_requests}, "
                    f"host_step_time = {host_step_time}ms, "
                    f"prev_device_step_time = {prev_device_step_time}, "
                    f"timestamp = {formatted_timestamp}, "
                    f"num_scheduled_requests: {self.num_scheduled_requests}, "
                    f"states = {self.model_engine.iter_states}")

            it += 1

            if it in self.profile_start_iters and not self.is_warmup:
                assert not enabled, "Inconsistent CUDA profiling state"
                torch.cuda.cudart().cudaProfilerStart()
                if enable_torch_trace:
                    torch_profiler.start()
                logger.info(f"Profiling started at iteration {it}.")
                enabled = True
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
        spec_resource_manager = self.resource_manager.resource_managers.get(
            ResourceManagerType.SPEC_RESOURCE_MANAGER)
        if spec_resource_manager is not None:
            stats.specdec_stats = SpecDecodingStats()
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
                           scheduled_batch) -> IterationStats:
        stats.iter_latency_ms = iter_latency_ms

        stats.num_queued_requests = self.executor_request_queue.get_request_queue_size(
        )
        stats.num_completed_requests = num_completed_requests
        stats.max_num_active_requests = self.max_num_active_requests

        end, total_gpu_memory = torch.cuda.mem_get_info()
        stats.gpu_mem_usage = total_gpu_memory - end
        stats.cpu_mem_usage = 0
        stats.pinned_mem_usage = 0

        stats.iter = self.model_engine.iter_counter

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
        stats.inflight_batching_stats.micro_batch_id = 0
        if stats.specdec_stats is not None:
            stats.specdec_stats.draft_overhead = 0.0 if iter_latency_ms <= 0.0 else float(
                stats.specdec_stats.iter_latency_ms) / float(iter_latency_ms)
        return stats

    def _append_iter_stats(self,
                           stats: IterationStats,
                           req_stats: Optional[List[RequestStats]] = None):

        with self.stats_lock:
            self.stats.append((stats, req_stats))

    def _process_iter_stats(self, finished_requests: list[LlmRequest],
                            active_requests: List[LlmRequest],
                            batch_state: BatchState):
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
            self._update_iter_stats(
                batch_state.iter_stats, iter_latency_ms, len(finished_requests),
                batch_state.sample_state.scheduled_requests), req_stats)

    def _executor_loop_cleanup(self):

        for h in self.send_handles:
            if h is not None:
                h.wait()

        if self._disagg_pp_termination_handler is not None:
            self._disagg_pp_termination_handler.cleanup()

        with self.response_cv:
            self.is_shutdown = True
            self.response_cv.notify_all()
        self.shutdown_event.set()

    def _executor_loop_pp(self):
        logger.debug(f"Starting executor loop for pp_rank {self.dist.pp_rank}")
        torch.cuda.set_device(self.device_id)
        # ensure the context is created, otherwise, some MPI calls will fail.
        CUASSERT(cudart.cudaSetDevice(self.device_id))
        microbatch_id = 0
        with self._profiler() as profile_step:
            iter_start_time = time.time()
            iter_stats = None
            while True:
                profile_step()
                if self.enable_iter_perf_stats:
                    iter_start_time = time.time()
                new_requests = self._fetch_and_activate_new_requests()
                if self.should_stop_processing:
                    break

                if self.kv_cache_transceiver:
                    self._check_disagg_gen_transfer_status()

                if self.enable_iter_perf_stats:
                    iter_stats = self._get_init_iter_stats(
                        len(new_requests),
                        self.executor_request_queue.
                        get_new_active_requests_queue_latency())

                self._pad_attention_dp_dummy_request()

                scheduled_batch, fitting_disagg_gen_init_requests, num_fitting_reqs = self._schedule(
                )

                if self.kv_cache_transceiver:
                    # For requests that are fitting disagg gen init, also prepare resources for KV cache manager
                    self._prepare_disagg_gen_init(
                        fitting_disagg_gen_init_requests)

                    if num_fitting_reqs == 0 and not fitting_disagg_gen_init_requests:
                        logger.warning(
                            "num_fitting_reqs=0 and fitting_disagg_gen_init_requests is empty, may not have enough kvCache"
                        )
                        self._check_disagg_ctx_cache_transfer_status(1)

                self.num_scheduled_requests = scheduled_batch.batch_size

                logger.debug(
                    f'has {len(self.active_requests)} active_request, '
                    f'scheduled {len(scheduled_batch.context_requests)} context requests and '
                    f'{len(scheduled_batch.generation_requests)} generation requests'
                )

                if self.enable_attention_dp:
                    tp_batch_sizes = self.dist.tp_allgather(
                        scheduled_batch.batch_size)
                    can_queue = 0 not in tp_batch_sizes
                else:
                    can_queue = scheduled_batch.batch_size > 0

                if not can_queue:
                    self.micro_batches[microbatch_id] = None
                else:
                    self._add_inflight_ids(scheduled_batch)

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
                        sample_state = self._forward_step_inter_pp(
                            scheduled_batch)
                    else:
                        with torch.cuda.nvtx.range("_forward_step_last_pp"):
                            # init_disagg_gen_requests must be before engine forward, where the prev_seq_slot is updated.
                            if self.guided_decoder is not None and self.kv_cache_transceiver:
                                self.guided_decoder.add_batch(scheduled_batch)
                                self.guided_decoder.init_disagg_gen_requests()

                            batch_outputs = self._forward_step(scheduled_batch)

                            if self.guided_decoder is not None:
                                self.guided_decoder.add_batch(scheduled_batch)
                                self.guided_decoder.execute(
                                    batch_outputs['logits'])

                            sample_state = self._sample_async(
                                scheduled_batch, batch_outputs)
                            assert sample_state is not None, "Sampling failed"
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
                    )

                    self.micro_batches[microbatch_id] = batch_state

                # Stage 2: Communicate new tokens for previous batch between ranks
                # send/recv chain: (pp_size - 1) -> 0 -> 1 -> ... -> (pp_size - 2)
                # last rank: sync sampler for previous microbatch to start new tokens comm chain.
                # other ranks: send/recv tokens for next microbatch to allow overlap
                offset = -1 if self.dist.is_last_pp_rank else 1
                prev_microbatch_id = (microbatch_id +
                                      offset) % self.num_micro_batches
                previous_batch = self.micro_batches[prev_microbatch_id]
                if previous_batch is not None:
                    sample_state = previous_batch.sample_state
                    if not self.dist.is_last_pp_rank:
                        recv_object_funct = self.dist.recv_object_from_isend if self._disable_mpi \
                            else self.dist.recv_object
                        torch.cuda.nvtx.range_push(
                            "_handle_new_tokens_inter_pp")
                        # Receive tokens from previous pp rank (w.r.t model forward direction)
                        sample_state.host = recv_object_funct(
                            src=self.dist.prev_pp_rank,
                            tag=prev_microbatch_id,
                        )
                    else:
                        torch.cuda.nvtx.range_push("_handle_new_tokens_last_pp")
                        sample_state.sampler_event.synchronize()

                    # Send tokens to next pp rank (w.r.t model forward direction)
                    # Second last rank does not need to since last rank has original decoded tokens
                    if not self.dist.is_second_last_pp_rank:
                        self.wait_on_pp_send_handles(prev_microbatch_id)
                        self.send_handles[
                            prev_microbatch_id] = self.dist.isend_object(
                                sample_state.host,
                                dest=self.dist.next_pp_rank,
                                tag=prev_microbatch_id)
                    torch.cuda.nvtx.range_pop()

                # Stage 3: Finalize previous batch that finished tokens communication
                # In last pp rank, stage 2 and 3 process different previous batches
                prev_microbatch_id = (microbatch_id +
                                      1) % self.num_micro_batches
                previous_batch = self.micro_batches[prev_microbatch_id]
                finished_requests = []
                if previous_batch is not None:
                    with torch.cuda.nvtx.range("_handle_previous_batch_pp"):
                        self._update_requests(previous_batch.sample_state)

                        if self.block_reuse_enabled and not self.kv_cache_manager.is_vswa and self.kv_cache_transceiver:
                            for req in previous_batch.scheduled_ctx_reqs:
                                if req.is_context_only_request and (
                                        req.is_context_finished
                                        or req.is_finished_due_to_length):
                                    block_id = self.kv_cache_manager.store_blocks_for_reuse(
                                        req, True)
                                    self.ctx_in_transmission_requests[
                                        req.py_request_id] = (
                                            (req, block_id,
                                             self.ctx_in_transmission_counter))

                        if self.kv_cache_transceiver:
                            self._send_disagg_ctx_cache(
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
                        self._remove_inflight_ids(previous_scheduled_batch)

                    self.wait_on_pp_send_handles(prev_microbatch_id)
                    self.micro_batches[prev_microbatch_id] = None

                if self.kv_cache_transceiver and self.ctx_in_transmission_requests:
                    self._check_kv_transfer_timeout()
                    self._terminate_disagg_ctx_finished_requests()

                if self._disagg_pp_termination_handler is not None:
                    requests_to_terminate = self._disagg_pp_termination_handler.sync(
                        prev_microbatch_id)
                    for req in requests_to_terminate:
                        self._do_terminate_request(req)

                # march forward in microbatch slots
                microbatch_id = (microbatch_id + 1) % self.num_micro_batches

                if self.enable_iter_perf_stats and previous_batch is not None:
                    self._process_iter_stats(finished_requests,
                                             self.active_requests,
                                             previous_batch)

    def wait_on_pp_send_handles(self, microbatch_id):
        if self.send_handles[microbatch_id] is not None:
            self.send_handles[microbatch_id].wait()
            self.send_handles[microbatch_id] = None

    def _prepare_and_schedule_batch(self):
        new_requests = self._fetch_and_activate_new_requests()
        if self.should_stop_processing:
            return None, None

        if self.kv_cache_transceiver:
            self._check_disagg_gen_transfer_status()
            self._check_kv_transfer_timeout()

        iter_stats = None
        if self.enable_iter_perf_stats:
            iter_stats = self._get_init_iter_stats(
                len(new_requests),
                self.executor_request_queue.
                get_new_active_requests_queue_latency())

        self._pad_attention_dp_dummy_request()

        if self.drafter is not None:
            # Honor permanent disable flag based on rolling acceptance first
            if getattr(self, 'speculation_permanently_disabled', False):
                self.use_spec_decode = False
            else:
                self.use_spec_decode = self.drafter.should_use_spec_decode(
                    self.active_requests, self.max_batch_size,
                    self.model_engine.max_num_tokens,
                    self.model_engine.spec_config.max_total_draft_tokens)
            logger.debug(f"Use spec decode: {self.use_spec_decode}")
            self.model_engine.enable_spec_decode = self.use_spec_decode

            # Set up draft_tokens in active_requests, because they could be used in the scheduling stage.
            for request in self.active_requests:
                if request.state not in (
                        LlmRequestState.GENERATION_IN_PROGRESS,
                        LlmRequestState.DISAGG_GENERATION_INIT):
                    continue
                max_total_draft_tokens = self.model_engine.spec_config.max_total_draft_tokens
                request.draft_tokens = [
                    0
                ] * max_total_draft_tokens if max_total_draft_tokens > 0 else []

            # When overlap scheduler is enabled, and we already prepared the draft tokens in the previous batch,
            # we don't need to initialize py_draft_tokens at this stage because we haven't append the accepted tokens to the request yet.
            if not self.has_previous_draft_tokens:
                # If speculation is off, this function sets py_draft_tokens to []
                # for all active requests. If it's on, we initialize py_draft_tokens
                # with dummy draft tokens to make the scheduler aware of the fact
                # that speculation is about to happen.
                self._prepare_draft_requests()

        scheduled_batch, fitting_disagg_gen_init_requests, num_fitting_reqs = self._schedule(
        )

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
            f'has {len(self.active_requests)} active_request, '
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
                if req.py_request_id in self.ctx_in_transmission_requests:
                    request, block_id, counter = self.ctx_in_transmission_requests.pop(
                        req.py_request_id)
                    if counter == 1:
                        self.kv_cache_manager.unpin_blocks_by_id(block_id)
                    else:
                        self.ctx_in_transmission_requests[req.py_request_id] = (
                            request, block_id, counter - 1)
                    break

    def _kv_connector_wait_for_save(self):
        if self.kv_connector_manager is not None:
            self.kv_connector_manager.worker.wait_for_save(
                torch.cuda.current_stream())

    def _executor_loop(self):
        torch.cuda.set_device(self.device_id)
        # ensure the context is created, otherwise, some MPI calls will fail.
        CUASSERT(cudart.cudaSetDevice(self.device_id))
        with self._profiler() as profile_step:
            sample_state = None
            iter_start_time = time.time()
            iter_stats = None
            while True:
                profile_step()
                if self.enable_iter_perf_stats:
                    iter_start_time = time.time()

                scheduled_batch, iter_stats = self._prepare_and_schedule_batch()
                if scheduled_batch is None:
                    break

                self._pause_requests(scheduled_batch.paused_requests)

                finished_requests = []

                if scheduled_batch.batch_size > 0 or (
                        self.enable_attention_dp and self.dist.tp_size > 1):
                    if self.kv_cache_transceiver:
                        # For generation requests which have completed KV cache transfer
                        self._prepare_disagg_gen_transmission_complete(
                            scheduled_batch)

                        # Return the first token to the client
                        self._handle_first_token_response(scheduled_batch)
                    self.resource_manager.prepare_resources(scheduled_batch)

                    self._kv_connector_start_batch(scheduled_batch)

                if scheduled_batch.batch_size > 0 or (
                        self.enable_attention_dp and self.dist.tp_size > 1):
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
                    if self.guided_decoder is not None:
                        self.guided_decoder.execute(batch_outputs['logits'])

                    sample_state = self._sample_async(scheduled_batch,
                                                      batch_outputs)
                    if self.drafter is not None:
                        self.drafter.run_drafter_post(scheduled_batch,
                                                      self.resource_manager,
                                                      self.is_warmup)

                    self._update_request_states(scheduled_batch)
                    self._update_requests(sample_state, self.resource_manager)
                    if self.block_reuse_enabled and not self.kv_cache_manager.is_vswa and self.kv_cache_transceiver:
                        for req in scheduled_batch.context_requests:
                            if req.is_context_only_request and (
                                    req.is_context_finished
                                    or req.is_finished_due_to_length):
                                block_id = self.kv_cache_manager.store_blocks_for_reuse(
                                    req, True)
                                self.ctx_in_transmission_requests[
                                    req.py_request_id] = (
                                        (req, block_id,
                                         self.ctx_in_transmission_counter))

                    if self.kv_cache_transceiver:
                        ctx_transmission_reqs = self._send_disagg_ctx_cache(
                            scheduled_batch.context_requests)
                        # For context only req in transmission, we reset the state since sampler might have changed it
                        for req in ctx_transmission_reqs:
                            req.state = LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS

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

                if self.kv_cache_transceiver and self.ctx_in_transmission_requests:
                    self._check_kv_transfer_timeout()
                    self._terminate_disagg_ctx_finished_requests()

                self._kv_connector_terminate_requests()

                if self.enable_iter_perf_stats and sample_state is not None:
                    iter_stats.inflight_batching_stats.num_ctx_tokens = self.model_engine.iter_states[
                        'num_ctx_tokens']
                    self._process_iter_stats(
                        finished_requests, self.active_requests,
                        BatchState(sample_state=sample_state,
                                   iter_stats=iter_stats,
                                   iter_start_time=iter_start_time))

    def _prepare_draft_requests(self):
        try:
            # Set draft tokens here to make the KV cache manager
            # and scheduler aware of them.
            for req in self.active_requests:
                if req.state not in (LlmRequestState.GENERATION_IN_PROGRESS,
                                     LlmRequestState.DISAGG_GENERATION_INIT):
                    continue

                req.py_last_draft_tokens = req.py_draft_tokens
                max_total_draft_tokens = self.model_engine.spec_config.max_total_draft_tokens

                if max_total_draft_tokens > 0 and self.use_spec_decode:
                    req.py_draft_tokens = [0] * max_total_draft_tokens
                    req.py_draft_pages_allocated = max_total_draft_tokens
                else:
                    req.py_draft_tokens = []
                    req.py_draft_pages_allocated = 0

        except Exception as e:
            traceback.print_exc()
            error_msg = str(e)
            logger.error(f"Encountered an error in decode: {error_msg}")
            self._handle_errors(error_msg)

    def _executor_loop_overlap(self):
        torch.cuda.set_device(self.device_id)
        # ensure the context is created, otherwise, some MPI calls will fail.
        CUASSERT(cudart.cudaSetDevice(self.device_id))
        with self._profiler() as profile_step:
            iter_start_time = time.time()
            iter_stats = None
            can_forward = False if self.benchmark_req_queues_size > 0 and self.kv_cache_transceiver else True
            while True:
                profile_step()
                if self.enable_iter_perf_stats:
                    iter_start_time = time.time()

                scheduled_batch, iter_stats = self._prepare_and_schedule_batch()
                if scheduled_batch is None:
                    break
                # In gen-only benchmarking mode, wait until the number of scheduled generation
                # requests reaches the required threshold before starting forward pass,
                # to ensure consistent batch sizes for accurate performance measurement.
                if not self.is_warmup and not can_forward:
                    if self.enable_attention_dp:
                        local_can_forward = self.executor_request_queue.num_fetch_requests + \
                            len(scheduled_batch.generation_requests) >= self.benchmark_req_queues_size
                        all_can_forward = self.dist.tp_allgather(
                            local_can_forward)
                        if all(all_can_forward):
                            can_forward = True
                            time.sleep(10)
                        else:
                            if self.dist.rank == 0:
                                logger.info(
                                    f"sleep 10 seconds, num_fetched_requests: {self.executor_request_queue.num_fetch_requests}, scheduled_gen_batch: {len(scheduled_batch.generation_requests)}"
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

                self._pause_requests(scheduled_batch.paused_requests)

                if scheduled_batch.batch_size > 0:
                    if self.kv_cache_transceiver:
                        # For generation requests which have completed KV cache transfer
                        self._prepare_disagg_gen_transmission_complete(
                            scheduled_batch)
                    self.resource_manager.prepare_resources(scheduled_batch)

                    self._kv_connector_start_batch(scheduled_batch)

                if scheduled_batch.batch_size > 0:

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
                    target_inputs = None
                    draft_outputs = None
                    # If there are previous draft tokens, we need to update the target requests to accept some draft tokens.
                    # When there's any accepted tokens, we can't directly use the previous batch's outputs in this iteration for the target model,
                    # so we'll set the target model's input to None and skip updating the target requests after target model forward.
                    use_previous_draft_tokens = self.has_previous_draft_tokens
                    if self.drafter is not None and (self.use_spec_decode or
                                                     use_previous_draft_tokens):
                        target_inputs, draft_outputs, draft_batch = self._handle_speculative_decoding(
                            scheduled_batch, previous_tensors)

                    # Use the draft_model's outputs if we've launched the draft model.
                    # Otherwise, use the previous batch's outputs.
                    if target_inputs is not None or use_previous_draft_tokens:
                        previous_tensors_device = target_inputs
                    else:
                        previous_tensors_device = self.previous_batch and self.previous_batch.sample_state and self.previous_batch.sample_state.device

                    batch_outputs = self._forward_step(scheduled_batch,
                                                       previous_tensors_device)

                    if target_inputs is not None:
                        self._process_draft_results(scheduled_batch,
                                                    draft_outputs, draft_batch)
                    elif self.previous_batch is not None and not use_previous_draft_tokens:
                        self._update_requests(self.previous_batch.sample_state)

                        if self.block_reuse_enabled and not self.kv_cache_manager.is_vswa and self.kv_cache_transceiver:
                            for req in self.previous_batch.sample_state.scheduled_requests.context_requests:
                                if req.is_context_only_request and (
                                        req.is_context_finished
                                        or req.is_finished_due_to_length):
                                    block_id = self.kv_cache_manager.store_blocks_for_reuse(
                                        req, True)
                                    self.ctx_in_transmission_requests[
                                        req.py_request_id] = (
                                            (req, block_id,
                                             self.ctx_in_transmission_counter))

                    if self.guided_decoder is not None:
                        # add_batch must be called again to have updated new tokens.
                        self.guided_decoder.add_batch(scheduled_batch)
                        self.guided_decoder.execute(batch_outputs['logits'])

                    sample_state = self._sample_async(scheduled_batch,
                                                      batch_outputs)
                    assert sample_state is not None, "Sampling failed"

                    self._update_request_states(scheduled_batch)

                    ctx_transmission_reqs = self._send_disagg_ctx_cache(
                        scheduled_batch.context_requests
                    ) if self.kv_cache_transceiver else []

                    if self.previous_batch is not None:
                        self._process_previous_batch()

                    if self.enable_iter_perf_stats:
                        iter_stats.inflight_batching_stats.num_ctx_tokens = self.model_engine.iter_states[
                            'num_ctx_tokens']

                    self.previous_batch = BatchState(
                        sample_state=sample_state,
                        iter_start_time=iter_start_time,
                        iter_stats=iter_stats,
                        ctx_transmission_reqs=ctx_transmission_reqs)

                if self.kv_cache_transceiver and self.ctx_in_transmission_requests:
                    self._check_kv_transfer_timeout()
                    self._terminate_disagg_ctx_finished_requests()

                self._kv_connector_terminate_requests()

    def _process_previous_batch(self):
        if self.kv_cache_transceiver and self.previous_batch.ctx_transmission_reqs:
            for req in self.previous_batch.ctx_transmission_reqs:
                req.state = LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS

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

    @nvtx_range("_forward_step_inter_pp")
    def _forward_step_inter_pp(self, scheduled_batch) -> SampleState:
        self._forward_step(scheduled_batch)
        sampler_event = torch.cuda.Event()
        sampler_event.record()
        self._update_request_states(scheduled_batch)
        sampler_event.synchronize()
        return self.sampler.SampleState(
            scheduled_requests=scheduled_batch,
            sampler_event=sampler_event,
        )

    def _validate_request(self, request: LlmRequest):
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

    @nvtx_range("_fetch_and_activate_new_requests")
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

        new_requests_cur_rank = self.executor_request_queue.fetch_new_requests(
            self.active_requests)
        self.is_shutdown = self.executor_request_queue.is_shutdown
        self.expected_num_active_requests = self.executor_request_queue.get_expected_num_active_requests(
        )

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
        if not self.enable_batch_waiting:
            return context_requests

        waited_context_requests = []
        stop_waiting = False
        num_scheduled_ctx_tokens = sum(
            len(ctx_req.get_tokens(0)) for ctx_req in context_requests)
        num_scheduled_gen_tokens = sum(1 + gen_req.num_draft_tokens
                                       for gen_req in generation_requests)
        num_scheduled_tokens = num_scheduled_ctx_tokens + num_scheduled_gen_tokens

        stop_waiting = self.batch_wait_iters_count >= self.batch_wait_timeout_iters or num_scheduled_tokens >= self.batch_wait_max_tokens_ratio * self.max_num_tokens
        if stop_waiting:
            waited_context_requests = context_requests
            self.batch_wait_iters_count = 0
        else:
            self.batch_wait_iters_count += 1
        return waited_context_requests

    @nvtx_range("_schedule")
    def _schedule(self):
        scheduler_output = self.scheduler.schedule_request(
            self.active_requests, self.inflight_req_ids)
        scheduled_context_requests = scheduler_output.context_requests
        if self.enable_attention_dp and self.attention_dp_enable_balance:
            scheduled_context_requests = self._balance_adp_requests(
                scheduler_output.context_requests,
                scheduler_output.generation_requests)

        # if no generation requests, no need to wait, to avoid dead waiting
        if not self.enable_attention_dp and self.enable_batch_waiting and len(
                scheduler_output.context_requests) > 0 and len(
                    scheduler_output.generation_requests) > 0:
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

        for req, _ in self.ctx_in_transmission_requests:
            flag_if_kv_transfer_timed_out(req, "context")

        for req in self.active_requests:
            if req.is_disagg_generation_transmission_in_progress:
                flag_if_kv_transfer_timed_out(req, "generation")

        return

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
            num_active_request = sum([
                0 if req.is_disagg_generation_init_state
                or req.is_disagg_generation_transmission_in_progress else 1
                for req in self.active_requests
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

    @nvtx_range("_send_disagg_ctx_cache")
    def _send_disagg_ctx_cache(self, scheduled_ctx_requests):
        if (scheduled_ctx_requests is None or len(scheduled_ctx_requests) == 0):
            return []
        for req in scheduled_ctx_requests:
            if req.is_context_only_request and (req.is_context_finished or
                                                req.is_finished_due_to_length):
                self.kv_cache_transceiver.respond_and_send_async(req)
                for resource_mgr_type in (
                        ResourceManagerType.SEQ_SLOT_MANAGER,
                        ResourceManagerType.SPEC_RESOURCE_MANAGER):
                    if resource_mgr_type in self.resource_manager.resource_managers and self.resource_manager.resource_managers[
                            resource_mgr_type] is not None:
                        self.resource_manager.resource_managers[
                            resource_mgr_type].free_resources(req)

        self._check_disagg_ctx_cache_transfer_status(0)

        # Keep track of ctx requests that are in transmission
        ctx_transmission_reqs = [
            req for req in scheduled_ctx_requests
            if req.state == LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS
        ]

        if self.kv_cache_transceiver.kv_transfer_timeout_ms is not None:
            for req in ctx_transmission_reqs:
                req.py_kv_transfer_start_time = time.time()

        return ctx_transmission_reqs

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
        self.kv_cache_transceiver.check_context_transfer_status(atLeastNum)
        self._check_cache_transfer_errors("context requests")

    @nvtx_range("_check_disagg_gen_cache_transfer_status")
    def _check_disagg_gen_cache_transfer_status(self, atLeastNum: int = 0):
        self.kv_cache_transceiver.check_gen_transfer_status(atLeastNum)
        self._check_cache_transfer_errors("generation requests")

    def _forward_step(self,
                      scheduled_requests,
                      new_tensors_device: Optional[SampleStateTensors] = None):

        @nvtx_range(
            f"[Executor] _forward_step {self.model_engine.iter_counter + 1}: {len(scheduled_requests.context_requests)} ctx reqs, {len(scheduled_requests.generation_requests)} gen reqs"
        )
        def forward(scheduled_requests, resource_manager, new_tensors_device,
                    gather_context_logits, cache_indirection_buffer):
            return self.model_engine.forward(
                scheduled_requests,
                resource_manager,
                new_tensors_device,
                gather_context_logits=gather_context_logits,
                cache_indirection_buffer=cache_indirection_buffer)

        try:
            gather_context_logits = any(
                a.py_return_context_logits
                for a in scheduled_requests.context_requests)
            cache_indirection_buffer = self.sampler.get_cache_indirection()
            outputs = forward(scheduled_requests, self.resource_manager,
                              new_tensors_device, gather_context_logits,
                              cache_indirection_buffer)

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
                request.state = LlmRequestState.GENERATION_IN_PROGRESS

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
        if self.kv_connector_manager is not None:
            # Only call request_finished on the connector if the request has already been added to the kv cache manager.
            try:
                cache_block_ids = self.kv_cache_manager.get_cache_indices(
                    request)
            except IndexError:
                # If the request has not yet been added to the kv cache manager,
                # we still need to free resources corresponding to other resource managers.
                self.resource_manager.free_resources(request)
            else:
                if self.kv_connector_manager.request_finished(
                        request,
                        cache_block_ids) and not self.kv_cache_transceiver:
                    block_id = self.kv_cache_manager.store_blocks_for_reuse(
                        request, True)
                    self.ctx_in_transmission_requests[request.py_request_id] = (
                        (request, block_id, self.ctx_in_transmission_counter))

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
        if self.executor_request_queue.get_canceled_req_ids_size() == 0:
            return

        # Remove cancel request in the waiting queue
        self.executor_request_queue.update_waiting_queue()

        for request in self.active_requests:
            req_id = request.py_request_id if not request.is_child else request.parent_request_id
            if req_id not in self.executor_request_queue.get_canceled_req_ids():
                continue

            is_cancelled = self._try_cancel_request(request)
            if is_cancelled:
                # Mark requests as finished, then, we reuse all existing code
                # to clean up the KV cache resources.
                request.finish_by_reason(FinishReason.CANCELLED)
                request.decoding_iter = request.py_decoding_iter
                self.executor_request_queue.canceled_req_ids.remove(req_id)

        if self.enable_attention_dp:
            # TODO: revisit the cancel logic of attention dp
            # When enable attention dp, each rank does not have full copy of requests
            # so we need to remove the cancel requests not in the local rank
            self.executor_request_queue.clear_canceled_req_ids()

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
                request.update_perf_metrics(self.model_engine.iter_counter)

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
                if self.block_reuse_enabled and not self.kv_cache_manager.is_vswa:
                    requests_to_terminate.append(request)
                else:
                    if request.is_disagg_context_transmission_state:
                        self.ctx_in_transmission_requests[
                            request.py_request_id] = (
                                (request, None,
                                 self.ctx_in_transmission_counter))
                    else:
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

    @nvtx_range("_terminate_disagg_ctx_finished_requests")
    def _terminate_disagg_ctx_finished_requests(self):
        for request_id in list(self.ctx_in_transmission_requests.keys()):
            request, block_id, counter = self.ctx_in_transmission_requests[
                request_id]

            if request.py_kv_transfer_timed_out:
                is_cancelled = self.kv_cache_transceiver.cancel_request(request)
                # If cancel is successful, mark as complete so it can be cleaned up
                # Otherwise, try at next iteration
                if is_cancelled:
                    request.py_kv_transfer_start_time = None
                    request.state = LlmRequestState.DISAGG_CONTEXT_COMPLETE

            if request.is_disagg_context_complete_state:
                del self.ctx_in_transmission_requests[request_id]
                if not self.block_reuse_enabled or self.kv_cache_manager.is_vswa:
                    self._terminate_request(request)
                elif counter == 1:
                    self.kv_cache_manager.unpin_blocks_by_id(block_id)
                else:
                    self.ctx_in_transmission_requests[request_id] = ((request,
                                                                      block_id,
                                                                      counter -
                                                                      1))

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
                r.py_return_context_logits or r.py_return_generation_logits)
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

    def _pause_requests(self, requests_to_pause):
        # todo: support work with self.inflight_req_ids.
        #       Currently, self.inflight_req_ids is not.
        max_input_len = self.max_input_len
        for req in requests_to_pause:
            req.pause(max_input_len)
            self._terminate_request(req)

    def _add_inflight_ids(self, scheduled_requests):
        """Add reqids of current requests to self.inflight_req_ids."""
        for req in scheduled_requests.all_requests():
            self.inflight_req_ids.insert(req.request_id)

    def _remove_inflight_ids(self, scheduled_requests):
        """Remove reqids of current requests from self.inflight_req_ids."""
        for req in scheduled_requests.all_requests():
            self.inflight_req_ids.erase(req.request_id)

    def _handle_speculative_decoding(self, scheduled_batch, previous_tensors):
        with request_context(is_draft=self.draft_model_engine is not None,
                             scheduled_requests=scheduled_batch):
            # Do an early checking to see if we need to forward the draft model.
            # If needed, the overlap should happen between the target requests and the draft requests.
            # Otherwise, we can still do overlap between the previous target requests and the current target requests.
            has_draft_batch = (
                self.previous_batch is not None and self.use_spec_decode
                and self.drafter.should_forward_draft_model(scheduled_batch))

            if has_draft_batch or self.has_previous_draft_tokens:
                self._update_requests(self.previous_batch.sample_state)
                if self.has_previous_draft_tokens:
                    self._prepare_draft_requests()

            if has_draft_batch:
                target_inputs, draft_outputs, draft_batch = self.drafter.generate_draft_tokens_with_overlap(
                    scheduled_batch, self.resource_manager,
                    previous_tensors.device if previous_tensors else None)

                self.has_previous_draft_tokens = target_inputs is not None and target_inputs.next_draft_tokens is not None
            else:
                self.has_previous_draft_tokens = False
                target_inputs, draft_outputs, draft_batch = None, None, None

        return target_inputs, draft_outputs, draft_batch

    def _process_draft_results(self, scheduled_batch, draft_outputs,
                               draft_batch):
        """
        Append the draft tokens to the target requests, and clean up the draft resources.
        """
        with request_context(is_draft=self.draft_model_engine is not None,
                             scheduled_requests=scheduled_batch):
            req_id_to_old_request = {
                req.py_request_id: req
                for req in scheduled_batch.all_requests()
            }

            if self.drafter.use_static_draft_loop:
                self.drafter.process_static_draft_outputs(
                    draft_outputs, draft_batch, req_id_to_old_request)
            elif draft_outputs is not None:
                self.drafter.process_dynamic_draft_outputs(
                    draft_outputs, req_id_to_old_request)

            # Pad draft tokens to the max draft length. This is for CUDA graph compatibility.
            self.drafter.pad_draft_tokens_for_cuda_graph(scheduled_batch)
            # add_batch must be called again to restore to target requests with updated draft tokens.
            if self.guided_decoder is not None:
                self.guided_decoder.add_batch(scheduled_batch)
                if hasattr(self.drafter, "guided_decoder"):
                    self.guided_decoder.rollback_draft_tokens()


class DisaggPPTerminationHandler:
    """Handles termination synchronization across pipeline parallel ranks under disaggregated serving.

    We require synchronization when terminating requests in disaggregated PP when
    KV cache reuse is enabled. All PP ranks need to reach consensus before freeing
    resources to avoid a NCCL hang.
    """

    def __init__(self, num_micro_batches: int, dist):
        self.dist = dist
        # Request termination synchronization across PP ranks
        # {request_id: {'ready_to_terminate': set{ranks}, 'terminated': {ranks}}}
        self.pending_termination = {}
        self.termination_handles = [None] * num_micro_batches
        # Local map from request_id -> local LlmRequest awaiting consensus termination
        self.local_termination = {}

    def terminate(self, request: LlmRequest) -> bool:
        req_key = request.py_request_id
        self.local_termination[req_key] = request
        state = self.pending_termination.get(req_key, None)
        if state is None:
            state = {'ready_to_terminate': set(), 'terminated': set()}
            self.pending_termination[req_key] = state
        if self.dist.rank not in state['ready_to_terminate']:
            state['ready_to_terminate'].add(self.dist.rank)
        return False

    def sync(self, microbatch_id: int) -> List[LlmRequest]:
        """Ring-communicate pending termination state and apply local terminations upon consensus.

        Each rank sends its current pending_termination snapshot to the next PP rank
        and receives the previous rank's snapshot. After merging, apply any terminations
        that have reached consensus (i.e., all PP ranks are ready).
        """
        snapshot = {
            req_id: {
                'ready_to_terminate': state.get('ready_to_terminate', set()),
                'terminated': state.get('terminated', set()),
            }
            for req_id, state in self.pending_termination.items()
        }

        if self.termination_handles[microbatch_id] is not None:
            self.termination_handles[microbatch_id].wait()

        term_tag = TERMINATION_COMM_TAG_BASE + microbatch_id
        self.termination_handles[microbatch_id] = self.dist.isend_object(
            snapshot,
            dest=self.dist.next_pp_rank,
            tag=term_tag,
        )
        remote_state = self.dist.recv_object(
            src=self.dist.prev_pp_rank,
            tag=term_tag,
        )
        logger.debug(
            f"received remote state for microbatch {microbatch_id}, prev pp rank: {self.dist.prev_pp_rank} state {remote_state}"
        )

        if remote_state:
            for req_id, state in remote_state.items():
                local = self.pending_termination.get(req_id)
                if local is None:
                    self.pending_termination[req_id] = {
                        'ready_to_terminate': state.get('ready_to_terminate',
                                                        set()),
                        'terminated': state.get('terminated', set()),
                    }
                else:
                    for key in ('ready_to_terminate', 'terminated'):
                        for r in state.get(key, []):
                            if r not in local[key]:
                                local[key].add(r)

        requests_to_terminate = []
        to_delete = []
        for req_id, state in self.pending_termination.items():
            ready = state.get('ready_to_terminate', set())
            done = state.get('terminated', set())
            # If all PP ranks are ready to terminate the request, we can free the resources
            if len(ready) >= self.dist.pp_size and self.dist.rank not in done:
                local_req = self.local_termination.get(req_id)
                if local_req is not None:
                    requests_to_terminate.append(local_req)
                done.add(self.dist.rank)
            if len(done) >= self.dist.pp_size:
                to_delete.append(req_id)
                if req_id in self.local_termination:
                    self.local_termination.pop(req_id, None)
        for req_id in to_delete:
            self.pending_termination.pop(req_id, None)

        return requests_to_terminate

    def cleanup(self):
        for h in self.termination_handles:
            if h is not None:
                h.wait()
