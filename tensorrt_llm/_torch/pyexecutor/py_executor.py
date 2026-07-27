# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import datetime
import functools
import os
import threading
import time
import traceback
from contextlib import contextmanager
from enum import IntEnum
from queue import Queue
from typing import (TYPE_CHECKING, Callable, Dict, Iterable, List, Optional,
                    Tuple, Union)

import torch
from strenum import StrEnum

from tensorrt_llm.llmapi import DisaggScheduleStyle
from tensorrt_llm.serve.responses_utils import get_steady_clock_now_in_seconds

try:
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cudart

from tensorrt_llm._utils import (CUASSERT, customized_gc_thresholds,
                                 is_trace_enabled, mpi_comm, mpi_disabled,
                                 nvtx_range, set_thread_local_mpi_comm,
                                 trace_func)
from tensorrt_llm.bindings.executor import (DisServingRequestStats,
                                            FinishReason, InflightBatchingStats,
                                            IterationStats, KvCacheStats,
                                            RequestStage, RequestStats,
                                            RequestType, SpecDecodingStats,
                                            StaticBatchingStats)
from tensorrt_llm.bindings.internal.batch_manager import (LlmRequestType,
                                                          ReqIdsSet)
from tensorrt_llm.executor.request import TruncateKVCacheRequest
from tensorrt_llm.inputs.multimodal import strip_mm_data_for_generation
from tensorrt_llm.llmapi.llm_args import PeftCacheConfig, WaitingQueuePolicy
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import CpType
from tensorrt_llm.runtime.kv_cache_manager_v2._exceptions import OutOfPagesError
from tensorrt_llm.tools.layer_wise_benchmarks import get_calibrator
from tensorrt_llm.tools.profiler.host_profile_tools.host_profiler import (
    get_global_profiler, host_profiler_context)

from ..distributed import Distributed
from ..distributed.communicator import ReduceOp
from ..expert_statistic import ExpertStatistic
from ..models.modeling_llama import Llama4ForConditionalGeneration
from ..models.modeling_multimodal_mixin import \
    maybe_prefetch_mm_encoder_for_next_iter
from ..models.modeling_utils import DecoderModelForCausalLM
from ..modules.decoder_layer import DecoderLayer
from ..speculative.drafter import Drafter
from ..speculative.spec_sampler_base import SampleStateTensorsSpec
from ..speculative.speculation_gate import SpeculationGate
from .adp_iter_stats import ADPIterStatsBuffer
from .connectors.kv_cache_connector import KvCacheConnectorManager
from .dwdp import DwdpManager
from .error_classification import ErrorBudget
from .executor_request_queue import ExecutorRequestQueue, RequestQueueItem
from .guided_decoder import GuidedDecoder
from .handle_additional_outputs import HandleAdditionalOutputs
from .handle_logits import HandleLogits
from .hang_detector import HangDetector, propagate_hard_kill
from .kv_cache_manager_v2 import KVCacheManagerV2
from .kv_cache_stats import append_kv_cache_iteration_stats
from .kv_cache_transceiver import (KvCacheTransceiver,
                                   is_disagg_inflight_cancel_enabled)
from .llm_request import (ATTENTION_DP_DUMMY_REQUEST_ID,
                          MAX_SPEC_DECODE_POSITIONS, ExecutorRequest,
                          LlmRequest, LlmRequestState, LlmResponse,
                          get_draft_token_length)
from .mamba_cache_manager import (BaseMambaCacheManager,
                                  MixedMambaHybridCacheManager)
from .model_engine import ModelEngine
from .perf_metrics_manager import PerfMetricsManager
from .request_utils import (RequestBroadcaster, attach_py_objects_to_requests,
                            derive_attention_dp_per_rank_request_cap,
                            get_from_waiting_queue, merge_requests)
from .resource_manager import (NoFreeSlotsError, ResourceManager,
                               ResourceManagerType, request_context)
from .sampler import (AsyncWorkerMixin, Sampler, SamplerEvent, SampleState,
                      SampleStateTensors, TRTLLMSampler)
from .scheduler import (RequestScheduler, ScheduledRequests,
                        SerializableSchedulerOutput, WaitingQueue,
                        create_waiting_queue)
from .scheduler.adp_router import ADPRouter

if TYPE_CHECKING:
    from ray.actor import ActorHandle

_UNBOUNDED_STATS_MAX_LEN = -1


def _stats_buffer_is_unbounded(max_stats_len: int) -> bool:
    return max_stats_len == _UNBOUNDED_STATS_MAX_LEN


# Environment variable to specify iteration ranges for profiling start/stop.
# Format: "start1-stop1,start2-stop2,..." or single iterations "iter1,iter2,..."
PROFILE_START_STOP_ENV_VAR_NAME = "TLLM_PROFILE_START_STOP"

# Environment variable to enable PyTorch profiler tracing.
# Set to a path to save detailed tracing of PyTorch operations.
PROFILE_TRACE_ENV_VAR_NAME = "TLLM_TORCH_PROFILE_TRACE"

# Environment variable to control which ranks print step logging.
# Format: comma-separated rank IDs, e.g. "0,1,3", or "all" for all ranks.
# Default: "0" (only rank 0 prints, matching existing behavior).
PROFILE_LOG_RANKS_ENV_VAR_NAME = "TLLM_PROFILE_LOG_RANKS"


class PPCommTag(IntEnum):
    """
    Unique tags for pipeline parallelism communication.
    """
    TERMINATION = 20000
    SCHEDULE_RESULT = 20001
    EXECUTED_BATCH_NUM = 20002
    SAMPLE_STATE = 20003


class _SleepWakeupTag(IntEnum):
    """MPI message tags for the dedicated sleep/wakeup communicator.

    Because ``_sleep_wakeup_comm`` is a duplicated communicator isolated from
    all other MPI traffic, small tag values are safe and do not conflict with
    ``PPCommTag``.
    """

    ACTION = 0  # rank-0 -> non-rank-0: sleep/wakeup/shutdown msg
    ACK = 1  # non-rank-0 -> rank-0: ACK after action completes


class _SleepWakeupAction(StrEnum):
    """Action values carried in sleep/wakeup control messages."""

    SLEEP = "sleep"
    WAKEUP = "wakeup"
    PREPARE = "prepare"
    COMMIT = "commit"
    ABORT = "abort"
    SHUTDOWN = "shutdown"


_SLEEP_WAKEUP_LISTENER_JOIN_TIMEOUT_S = 30.0
_SLEEP_WAKEUP_ACK_TIMEOUT_S = 30.0
_SLEEP_WAKEUP_ACK_POLL_INTERVAL_S = 0.01


def _sleep_wakeup_ack_ready(comm, source: int, tag: _SleepWakeupTag) -> bool:
    """Return whether an ACK is ready without blocking on recv."""
    if hasattr(comm, "iprobe"):
        return comm.iprobe(source=source, tag=tag)
    if hasattr(comm, "Iprobe"):
        return comm.Iprobe(source=source, tag=tag)
    raise RuntimeError(
        "_sleep_wakeup_comm does not support nonblocking ACK probe")


def _recv_sleep_wakeup_ack_until(comm,
                                 source: int,
                                 deadline: float,
                                 expected_op_id: Optional[str] = None,
                                 expected_phase: Optional[str] = None) -> dict:
    """Receive a sleep/wakeup ACK before an absolute monotonic deadline."""
    while True:
        if _sleep_wakeup_ack_ready(comm, source, _SleepWakeupTag.ACK):
            ack = comm.recv(source=source, tag=_SleepWakeupTag.ACK)
            if (expected_op_id is not None
                    and ack.get("op_id") != expected_op_id):
                logger.warning(
                    "Ignoring stale sleep/wakeup ACK from rank %d for op_id=%s "
                    "(expected op_id=%s).",
                    source,
                    ack.get("op_id"),
                    expected_op_id,
                )
                continue
            ack_phase = ack.get("phase")
            if (expected_phase is not None and ack_phase is not None
                    and ack_phase != expected_phase):
                logger.warning(
                    "Ignoring stale sleep/wakeup ACK from rank %d for phase=%s "
                    "(expected phase=%s).",
                    source,
                    ack_phase,
                    expected_phase,
                )
                continue
            return ack
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"timed out waiting for sleep/wakeup ACK from rank {source} "
                f"by deadline {deadline:.6f}")
        time.sleep(_SLEEP_WAKEUP_ACK_POLL_INTERVAL_S)


def _recv_sleep_wakeup_ack(comm, source: int, timeout_s: float) -> dict:
    """Receive a sleep/wakeup ACK with a bounded nonblocking poll loop."""
    return _recv_sleep_wakeup_ack_until(comm, source,
                                        time.monotonic() + timeout_s)


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


def _strip_py_multimodal_data_post_prefill(request: LlmRequest) -> None:
    """Drop pinned encoder cache and raw pre-encoder tensors after prefill completes.

    Wraps `strip_mm_data_for_generation` and mutates the shared `request.py_multimodal_data`
    in-place so the `LlmRequest`'s multimodal tensors actually get freed (unlike
    `MultimodalParams.strip_for_generation`, which rebinds a per-forward-call wrapper's attribute
    and leaves the request's dict untouched).
    """
    mm_data = getattr(request, "py_multimodal_data", None)
    if not mm_data:
        return
    strip_mm_data_for_generation(mm_data)


@dataclasses.dataclass
class DisaggTransferAdmissionResult:
    admitted_requests: List[LlmRequest]
    active_transfer_blocks: int = 0
    admitted_transfer_blocks: int = 0
    deferred_request_count: int = 0
    limited_by_budget: bool = False

    def is_blocked_by_active_transfers(self) -> bool:
        return (self.limited_by_budget and not self.admitted_requests
                and self.active_transfer_blocks > 0)


class DisaggTransferAdmissionController:
    """FCFS admission gate for disaggregated generation KV transfers."""

    def __init__(self, max_tokens_in_buffer: Optional[int],
                 tokens_per_block: Optional[int]) -> None:
        self.max_transfer_blocks = self._to_block_budget(
            max_tokens_in_buffer, tokens_per_block)
        self.tokens_per_block = tokens_per_block or 0

    def enabled(self) -> bool:
        return self.max_transfer_blocks is not None

    @staticmethod
    def _to_block_budget(max_tokens_in_buffer: Optional[int],
                         tokens_per_block: Optional[int]) -> Optional[int]:
        if (max_tokens_in_buffer is None or max_tokens_in_buffer == 0
                or tokens_per_block is None or tokens_per_block <= 0):
            return None
        return (max_tokens_in_buffer + tokens_per_block - 1) // tokens_per_block

    @staticmethod
    def _to_nonnegative_int(value) -> Optional[int]:
        try:
            return max(int(value), 0)
        except (TypeError, ValueError):
            return None

    def _get_request_transfer_token_count(self, request: LlmRequest) -> int:
        for attr_name in ("total_input_len_cp", "py_prompt_len", "prompt_len"):
            token_count = self._to_nonnegative_int(
                getattr(request, attr_name, None))
            if token_count is not None:
                return token_count
        return 0

    def _estimate_request_blocks(self, request: LlmRequest) -> int:
        if self.tokens_per_block <= 0:
            return 0
        prompt_len = self._get_request_transfer_token_count(request)
        return (prompt_len + self.tokens_per_block - 1) // self.tokens_per_block

    def _estimate_requests_blocks(self, requests: Iterable[LlmRequest]) -> int:
        return sum(
            self._estimate_request_blocks(request) for request in requests)

    def _estimate_active_transfer_blocks(
            self, active_requests: Iterable[LlmRequest]) -> int:
        return sum(
            self._estimate_request_blocks(request)
            for request in active_requests
            if request.is_disagg_generation_transmission_in_progress)

    def select(self, active_requests: Iterable[LlmRequest],
               candidates: List[LlmRequest]) -> DisaggTransferAdmissionResult:
        if not self.enabled():
            return DisaggTransferAdmissionResult(
                admitted_requests=list(candidates),
                active_transfer_blocks=self._estimate_active_transfer_blocks(
                    active_requests),
                admitted_transfer_blocks=self._estimate_requests_blocks(
                    candidates),
            )

        result = DisaggTransferAdmissionResult(admitted_requests=[])
        result.active_transfer_blocks = self._estimate_active_transfer_blocks(
            active_requests)

        used_blocks = result.active_transfer_blocks
        max_transfer_blocks = self.max_transfer_blocks
        assert max_transfer_blocks is not None
        for request in candidates:
            request_blocks = self._estimate_request_blocks(request)
            fits_budget = used_blocks + request_blocks <= max_transfer_blocks
            admit_oversized_head = (not result.admitted_requests
                                    and result.active_transfer_blocks == 0
                                    and request_blocks > max_transfer_blocks)
            if not fits_budget and not admit_oversized_head:
                result.limited_by_budget = True
                break

            result.admitted_requests.append(request)
            used_blocks += request_blocks
            result.admitted_transfer_blocks += request_blocks

        result.deferred_request_count = len(candidates) - len(
            result.admitted_requests)
        return result


@dataclasses.dataclass
class ScheduledBatchStats:
    # None means the counter was not captured and _update_iter_stats should
    # fall back to the existing scheduled_batch/request accessors.
    num_ctx_requests: Optional[int] = None
    num_ctx_tokens: Optional[int] = None
    num_ctx_kv_tokens: Optional[int] = None
    num_gen_requests: Optional[int] = None
    num_gen_kv_tokens: Optional[int] = None
    num_paused_requests: Optional[int] = None


@dataclasses.dataclass
class BatchState:
    scheduled_requests: ScheduledRequests
    sample_state: SampleState

    iter_start_time: float = 0
    iter_stats: IterationStats = None
    scheduled_batch_stats: Optional[ScheduledBatchStats] = None
    gpu_forward_start_event: Optional[torch.cuda.Event] = None
    gpu_forward_end_event: Optional[torch.cuda.Event] = None
    gpu_forward_events_from_perf_pool: bool = False


@dataclasses.dataclass
class BatchStatePP(BatchState):
    microbatch_id: int = -1


class AsyncTransferManager:
    """
    Handle asynchronous transfer of KV cache after a request has completed.
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

        # Mapping of request id to the request metadata
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
            return False

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
    # Minimum number of async micro batches for async PP execution.
    # This is a trade-off between memory usage and performance.
    # If the number of micro batches is too small, the executor will spend too much time in synchronization.
    # If the number of micro batches is too large, the executor will spend too much host memory (No additional GPU memory is required).
    # 1024 in-flight micro batches can avoid synchronization in most cases and keep host memory usage low.
    MIN_ASYNC_MICRO_BATCH_NUM = 1024

    def __init__(
            self,
            resource_manager,
            scheduler: RequestScheduler,
            model_engine: ModelEngine,
            sampler: Sampler,
            dist: Distributed,
            max_num_sequences: int,
            drafter: Optional[Drafter] = None,
            disable_overlap_scheduler: bool = False,
            enable_early_first_token_response: bool = False,
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
            resource_governor_queue=None,
            max_seq_len: Optional[int] = None,
            peft_cache_config: Optional[PeftCacheConfig] = None,
            virtual_memory_pools: Optional[dict] = None,
            hang_detection_timeout: Optional[int] = None,
            execution_stream: Optional[torch.cuda.Stream] = None,
            waiting_queue_policy: WaitingQueuePolicy = WaitingQueuePolicy.FCFS,
            adp_router: Optional[ADPRouter] = None,
            dwdp_manager: Optional[DwdpManager] = None,
            enable_kv_pool_rebalance: bool = False):
        super(PyExecutor, self).__init__()
        self.device_id = torch.cuda.current_device()
        self.global_rank = dist.rank
        # Store the execution stream for decoder/model forward operations.
        # This stream is used for proper synchronization with
        # KVCacheTransferManager. execution_stream can be provided by
        # create_py_executor. Create a new stream if none provided.
        self.execution_stream = execution_stream if execution_stream is not None else torch.cuda.Stream(
        )
        # Encoder-decoder requests use a dedicated encoder stream so the
        # encoder forward does not serialize the decoder forward when the
        # two operate on disjoint request sets. Per-request CUDA events
        # carry the encoder->decoder dependency to the eventual consumer.
        self.encoder_stream = torch.cuda.Stream()
        logger.info(
            f"[PyExecutor] execution_stream initialized: {self.execution_stream}; "
            f"encoder_stream initialized: {self.encoder_stream}.")

        self.peft_cache_config = peft_cache_config

        self.iter_counter = 0
        # profile config
        self.profile_start_iters, self.profile_stop_iters = _load_iteration_indexes(
            PROFILE_START_STOP_ENV_VAR_NAME)

        # related modules
        self.resource_manager = resource_manager
        self.scheduler = scheduler
        self.model_engine = model_engine
        self._enable_dsv4_adp_dummy_fixes = getattr(
            model_engine, "_enable_dsv4_adp_dummy_fixes", False)
        self.enable_attention_dp = model_engine.enable_attention_dp
        self.dist = dist
        self.sampler = sampler
        self.drafter = drafter
        self.draft_model_engine = getattr(self.drafter, "draft_model_engine",
                                          None)
        self.guided_decoder = guided_decoder
        self.disable_overlap_scheduler = disable_overlap_scheduler
        self.enable_kv_pool_rebalance = enable_kv_pool_rebalance
        self.enable_early_first_token_response = enable_early_first_token_response
        self.virtual_memory_pools = virtual_memory_pools

        # enqueue and _fetch_new_requests used data
        self.active = True
        self.max_beam_width = max_beam_width
        self.max_draft_len = max_draft_len
        self.max_total_draft_tokens = max_total_draft_tokens
        self.llm_args = self.model_engine.llm_args
        self.max_stats_len = self.llm_args.max_stats_len
        self.max_num_tokens = self.llm_args.max_num_tokens
        self.print_log = self.llm_args.print_iter_log
        self.enable_iter_perf_stats = self.llm_args.enable_iter_perf_stats
        self.enable_iter_req_stats = self.llm_args.enable_iter_req_stats
        self.stream_interval = self.llm_args.stream_interval
        self.perf_manager = PerfMetricsManager(
            enabled=getattr(self.llm_args, 'return_perf_metrics', False))
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

        # Rolling true acceptance-rate tracking for permanent speculation
        # disable.
        self.speculation_permanently_disabled = False
        self.speculation_gate = None
        spec_config = getattr(self.model_engine, 'spec_config', None)
        if spec_config is not None:
            window = getattr(spec_config, 'acceptance_rate_window_size', None)
            threshold = getattr(spec_config, 'acceptance_rate_threshold', None)
            if window and threshold is not None:
                self.speculation_gate = SpeculationGate(window, threshold)

        # response used data
        self.response_lock = threading.Lock()
        self.response_cv = threading.Condition(self.response_lock)
        self.responses = {}
        self.result_wait_queues = {}
        # If the event-loop thread (PyExecutor._event_loop_wrapper) raises
        # an exception (e.g. KV cache OOM), it is stashed here and read by
        # two local consumers so they can surface it instead of hanging:
        #   - _await_single_response re-raises it as a RuntimeError, and
        #   - BaseWorker.AwaitResponseHelper (the bridge between this
        #     engine and per-request GenerationResult queues) reads it to
        #     broadcast an ErrorResponse to every pending request, waking
        #     callers parked in queue.get() / aqueue.get().
        self._event_loop_error: Optional[BaseException] = None

        # kv cache events
        self.kv_cache_manager = self.resource_manager.resource_managers.get(
            ResourceManagerType.KV_CACHE_MANAGER)
        # V2 owns KV allocation, suspend, resume, and context finalization.
        # The executor skips the V1 terminate/pause paths and finalizes V2
        # context resources before transfer or response handling can terminate
        # a request.
        self._is_kv_manager_v2 = isinstance(self.kv_cache_manager,
                                            KVCacheManagerV2)
        self._prefetched_request_ids: set[int] = set()
        self.enable_kv_cache_events = self.kv_cache_manager is not None and self.kv_cache_manager.event_buffer_max_size > 0
        self.enable_kv_cache_reuse = self.kv_cache_manager is not None and self.kv_cache_manager.enable_block_reuse
        # AsyncTransferManager pin/unpin path is V1-only; V2 holds blocks via _KVCache refcount.
        self.enable_partial_reuse_for_disagg = (
            self.enable_kv_cache_reuse
            and self.kv_cache_manager.enable_partial_reuse
            and not isinstance(self.kv_cache_manager, KVCacheManagerV2))
        # Store+pin ctx blocks into the reuse trie at transfer start so the next
        # request reuses them immediately (PP>1 otherwise commits too late and
        # only partially matches). No collective is required for the
        # store-and-pin operation. store_blocks_for_reuse is V1-only;
        # KVCacheManagerV2 has no equivalent yet.
        self.enable_disagg_partial_reuse_store = (
            self.enable_partial_reuse_for_disagg
            and not self.kv_cache_manager.is_vswa)
        # Early-terminating the ctx request in _handle_responses (at prefill
        # done) is a PP=1-only latency win. Under PP>1, ctx termination is
        # routed through the DisaggPPTerminationHandler cross-rank ring
        # consensus, which is only validated against the transfer-complete
        # trigger; driving it from the early path regressed to a hang in CI.
        # So PP>1 keeps eager-store but terminates via the transfer-complete path.
        self.force_terminate_ctx_for_partial_reuse = (
            self.enable_disagg_partial_reuse_store and self.dist.pp_size == 1)

        self.max_input_len = max_input_len
        # _executor_loop private data
        self.max_num_active_requests = model_engine.get_max_num_sequences()
        # nvbug-6133201: under attention DP, tighten the per-rank request
        # cap so per-rank gen-phase step-token load cannot exceed
        # max_num_tokens. No-op for correctly-sized configs.
        if self.enable_attention_dp:
            derived_cap = derive_attention_dp_per_rank_request_cap(
                base_cap=self.max_num_active_requests,
                max_num_tokens=self.max_num_tokens,
                max_total_draft_tokens=self.max_total_draft_tokens,
            )
            if derived_cap < self.max_num_active_requests:
                step_tokens_per_req = 1 + max(self.max_total_draft_tokens, 0)
                required_max_num_tokens = (self.max_num_active_requests *
                                           step_tokens_per_req)
                logger.warning(
                    f"[PyExecutor] enable_attention_dp: max_num_tokens="
                    f"{self.max_num_tokens} cannot fit max_batch_size="
                    f"{self.max_num_active_requests} at {step_tokens_per_req} "
                    f"step-tokens each; capping per-rank max_num_active_requests "
                    f"to {derived_cap}. Raise max_num_tokens to "
                    f"{required_max_num_tokens} to run at the declared "
                    f"max_batch_size.")
            self.max_num_active_requests = derived_cap
        self.active_requests: List[LlmRequest] = []
        self.expected_num_active_requests = 0
        # Buffer for responses generated inside _end_transfer_and_maybe_terminate.
        # With ADP, _enqueue_responses does a tp_gather collective.  When called
        # from _send_kv_async the owning DP rank has a response but the other
        # rank does not, causing a collective mismatch deadlock.  Buffering the
        # responses and flushing them at a synchronised point in the executor
        # loop avoids the mismatch.
        self._pending_transfer_responses: List[Tuple[int, LlmResponse]] = []
        # Same buffer-then-synced-flush pattern as _pending_transfer_responses
        # above: _handle_responses and _append_iter_stats are reached from
        # per-rank-divergent gates, so their tp_allgather collectives are
        # lifted to _handle_kv_transfer_timeouts_synced / _flush_iter_stats_synced.
        self._pending_timed_out_requests: List[LlmRequest] = []
        self._pending_iter_stats_dict: Optional[Dict] = None
        # ADP dummy role for _pad_attention_dp_dummy_request. Default is gen;
        # updated from observed request types.
        self._adp_dummy_is_gen: bool = True
        # Dummy allocated by the current scheduling iteration. It is committed
        # to the normal forward/termination lifecycle only after every ADP rank
        # can queue; otherwise only this exact allocation is rolled back.
        self._pending_adp_dummy_request: Optional[LlmRequest] = None
        self.async_transfer_manager = AsyncTransferManager(
            self.resource_manager,
            should_store_blocks=self.enable_disagg_partial_reuse_store)

        # Router is built after async_transfer_manager so KVCacheAwareADPRouter
        # can receive the transfer-manager reference at construction time.
        self.adp_router: ADPRouter = ADPRouter.create(
            dist=self.dist,
            kv_cache_manager=self.kv_cache_manager,
            attention_dp_config=self.llm_args.attention_dp_config,
            async_transfer_manager=self.async_transfer_manager,
        )

        self.previous_batch: Optional[BatchState] = None
        self.has_previous_draft_tokens = False
        self.num_scheduled_requests: int = 0
        self.benchmark_req_queues_size = int(
            os.environ.get("TLLM_BENCHMARK_REQ_QUEUES_SIZE", 0))

        # list of requests in each PP micro batch
        self.num_micro_batches = max(self.dist.pp_size,
                                     self.MIN_ASYNC_MICRO_BATCH_NUM)
        self.micro_batches: List[BatchStatePP
                                 | None] = [None] * self.num_micro_batches
        self.send_handles = [None] * self.num_micro_batches
        # schedule handle for PP to propagate the first PP rank's schedule result
        self.send_schedule_handles = [None] * self.num_micro_batches
        self.send_expected_batch_num_handles = [None] * self.num_micro_batches
        self.unhandled_batch_counter = 0
        self.pp_scheduler_max_retry_count = int(
            os.environ.get("TLLM_PP_SCHEDULER_MAX_RETRY_COUNT", 10))
        self.pp_multi_stream_sample = os.environ.get(
            "TRTLLM_PP_MULTI_STREAM_SAMPLE", "1") == "1"
        self.sample_stream = torch.cuda.Stream()
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

        # Encoder-decoder models execute the encoder and decoder in separate
        # iterations in both executor loops. PP usage is very rare for these
        # models, so encoder PP send/recv support is not implemented in the
        # PyTorch path for now. Reject pp_size > 1.
        # TODO: Add support for pp + encoder models
        is_encoder_decoder = bool(
            getattr(getattr(self.model_engine.model, "model_config", None),
                    "is_encoder_decoder", False))
        if is_encoder_decoder:
            if self.dist.pp_size > 1:
                raise NotImplementedError(
                    "pp_size > 1 is not supported for encoder-decoder models "
                    "in the PyTorch flow; encoder send/recv hooks are out of "
                    "scope. Set pp_size=1 to run T5/BART/mBART.")
            if getattr(self.model_engine, "_torch_compile_piecewise_cuda_graph",
                       False):
                raise NotImplementedError(
                    "Piecewise CUDA graph is not supported for "
                    "encoder-decoder models. Disable "
                    "torch_compile_config.enable_piecewise_cuda_graph.")
            if (getattr(self.model_engine, "cuda_graph_config", None)
                    is not None and getattr(self.model_engine, "spec_config",
                                            None) is not None):
                raise NotImplementedError(
                    "Speculative decoding with CUDA graph is not supported "
                    "for encoder-decoder models.")

        # Synchronize all ranks before warmup. This prevents a PP
        # communication deadlock when ranks on the last PP stage are delayed
        # by heavy initialisation (e.g. guided-decoder / llguidance tokenizer
        # creation) while earlier PP stages already start warmup forward
        # passes that require matching pp_recv on the later stages.
        self.dist.barrier()

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

        # Snapshot some cumulative KV cache counters so that stats reported to
        # users exclude blocks reused and missed during warmup dummy requests.
        if hasattr(self.kv_cache_manager, 'snapshot_warmup_baseline'):
            self.kv_cache_manager.snapshot_warmup_baseline()

        self.is_shutdown = False
        self._fatal_error: Optional[BaseException] = None
        self._error_budget = ErrorBudget()
        self._disagg_timed_out_ctx_cancelled_ids: set[int] = set()
        self._disagg_timed_out_gen_cancelled_ids: set[int] = set()
        self._disagg_inflight_cancel_unsupported_logged = False
        self.max_batch_size = max_batch_size
        self.adp_ctx_waiting_iters_count = 0
        self.adp_ctx_batching_wait_iters_count = 0
        self.batch_wait_iters_count = 0

        def on_detected():
            # The graceful shutdown path can itself deadlock on collectives
            # while peer ranks stay blocked in NCCL holding their GPUs. Hard-kill
            # this rank and propagate the kill outward instead.
            # Guard the diagnostic log in try/finally so propagate_hard_kill()
            # always runs even if logging itself raises.
            try:
                logger.error(
                    f"Hang detected on rank {self.global_rank} in PyExecutor; "
                    "hard-killing and propagating to peer ranks.")
            finally:
                propagate_hard_kill()

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
        # _sample_async is called twice before the first call to update_requests:
        # - 1st time as a context request that operates on the 1st generated token
        # - 2nd time as a generation request that operates on the 2nd generated token.
        # and only after these two calls the sampler's update_request method is called.
        # So in a sampler that works by the expected flow of handling the logits in
        # _sample_async, every update_request doesn't handle the newest token, but one
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
        self.waiting_queue: WaitingQueue = create_waiting_queue(
            waiting_queue_policy)

        self.control_request_barrier = threading.Event()
        self.control_action_done = threading.Event()
        self._active_control_id: Optional[str] = None
        self._sleep_wakeup_pending_aborts: Dict[str, str] = {}
        self._sleep_wakeup_pending_abort_lock = threading.Lock()

        # Resource governor queue (IpcQueue in multi-process mode,
        # IntraProcessQueue in single-process mode) for receiving cache-
        # management requests (e.g. truncation) from ResourceGovernor.
        # The decode loop only enters the collective path when the flag is
        # enabled, so both the queue and the flag must be set before
        # start_worker() to keep the MPI collective order identical on all
        # ranks from iteration 0.
        self._resource_governor_queue = resource_governor_queue
        self._resource_governor_enabled = resource_governor_queue is not None

        self.stats_lock = threading.Lock()
        self.stats = []
        self._latest_kv_iter_stats = None
        self._last_kv_iter_stats_fetch_iter = None
        self._kv_iter_stats_interval = getattr(
            getattr(self.llm_args, 'kv_cache_config', None),
            'iteration_stats_interval', 1)
        self._adp_iter_stats = ADPIterStatsBuffer()
        # Per-loop CPU wall and GPU forward time captured by the profile_step
        # closure (see _profiler). Populated whenever enable_iter_perf_stats or
        # print_iter_log is on so the /metrics serializer can read them
        # without depending on the log line. host_step_time describes the loop
        # body that just finished; prev_device_step_time is the GPU forward
        # time read via the ping-pong CUDA event pair (lags host by one loop
        # under steady state — see ping-pong comment in _profiler).
        self._latest_host_step_time_ms: Optional[float] = None
        self._latest_prev_device_step_time_ms: Optional[float] = None
        self._emit_initial_stats()
        self.gather_all_responses = False

        self.kv_cache_transceiver = kv_cache_transceiver
        cache_transceiver_config = getattr(self.llm_args,
                                           "cache_transceiver_config", None)
        max_tokens_in_buffer = getattr(cache_transceiver_config,
                                       "max_tokens_in_buffer", None)
        tokens_per_block = getattr(self.kv_cache_manager, "tokens_per_block",
                                   None)
        self._disagg_transfer_admission_controller = DisaggTransferAdmissionController(
            max_tokens_in_buffer, tokens_per_block)
        self.is_benchmark_disagg = (self.benchmark_req_queues_size > 0
                                    and self.kv_cache_transceiver is not None)
        # True while the benchmark disagg fill phase is in progress (waiting
        # for all benchmark requests to complete KV transfer before the first
        # forward pass).  Cleared by _check_benchmark_disagg_gate when the
        # can_forward gate opens.  Used by _should_skip_dummy_for_benchmark_disagg
        # to prevent permanent dummy insertion during fill; once False, the
        # normal dummy add-forward-terminate lifecycle handles taper-down.
        # Only relevant in benchmark disagg mode; False otherwise.
        self._benchmark_fill_phase_active = self.is_benchmark_disagg
        # Set when a blocking generation transfer returns during benchmark
        # fill. The gate consumes this signal to retry immediately instead of
        # adding an unnecessary polling delay after synchronous progress.
        self._sync_disagg_transfer_made_progress = False
        self._benchmark_sync_progress_global = False
        # Slow-start admission cap for benchmark disagg fill (see
        # _pop_from_waiting_queue).  0 = uninitialised; first throttled iter
        # seeds it to tp_size and each subsequent iter doubles it.
        self._fill_admit_cap: int = 0

        # Initialize disagg PP termination handler if needed
        self._disagg_pp_termination_handler = None
        if self.dist.pp_size > 1 and self.enable_kv_cache_reuse and self.kv_cache_transceiver:
            self._disagg_pp_termination_handler = DisaggPPTerminationHandler(
                self.dist, self._do_terminate_request)

        if self.dist.pp_size > 1:
            self.event_loop = self._executor_loop_pp
            # `TLLM_PP_ASYNC_BROADCAST_SAMPLE_STATE` controls whether to broadcast the sample state asynchronously.
            # If true, the executor loop can broadcast and handle sample states asynchronously to achieve best perf.
            # If false, the executor loop can only broadcast and handle each sample state in a pre-defined iteration.
            # It is only for debugging purposes.
            # Some tests can disable it to get a deterministic behavior.
            self.pp_async_broadcast_sample_state = os.environ.get(
                "TLLM_PP_ASYNC_BROADCAST_SAMPLE_STATE", "1") == "1"
        else:
            self.event_loop = self._executor_loop if self.disable_overlap_scheduler else self._executor_loop_overlap
        if is_trace_enabled("TLLM_TRACE_EXECUTOR_LOOP"):
            self.event_loop = trace_func(self.event_loop)

        if dwdp_manager is not None and not self.disable_overlap_scheduler:
            raise ValueError(
                "DWDP requires disable_overlap_scheduler=True. "
                "Overlap scheduler is not yet supported with DWDP.")

        if self.drafter is not None:
            if self.event_loop.__name__ == self._executor_loop_pp.__name__:
                raise NotImplementedError(
                    "Drafting is not supported for selected executor loop. "
                    "Please disable disagg/pipeline parallelism scheduler.")
        self.garbage_collection_gen0_threshold = garbage_collection_gen0_threshold
        self.max_seq_len = max_seq_len

        self.worker_started = False
        self.worker_lock = threading.Lock()
        self._broadcast_mpi_comm = None
        # Secondary MPI communicator and listener thread for multi-rank
        # sleep/wakeup control messages.  Both are None until start_worker()
        # calls Dup() (a collective) on the main thread.
        self._sleep_wakeup_comm = None
        self._sleep_wakeup_listener_thread: Optional[threading.Thread] = None
        # Serialises concurrent sleep/wakeup calls so that the control_action
        # + _sleep_wakeup_comm send/recv sequence is never interleaved.  Without
        # this, two concurrent RPC calls could both pass control_request_barrier
        # and race on the shared communicator, consuming the wrong ACKs.
        self._sleep_wakeup_lock = threading.Lock()

        self.kv_connector_manager = kv_connector_manager

        self._maybe_init_kv_connector_manager()

        self.dwdp_manager = dwdp_manager

        if start_worker:
            self.start_worker()

    def _maybe_init_kv_connector_manager(self):
        if self.kv_connector_manager is not None:
            if self.kv_cache_transceiver is not None:
                logger.warning(
                    "Both KV Cache Connector and KV Cache Transceiver are enabled. Are you sure you want to do this?"
                )

            if self.dist.pp_size > 1 or self.dist.cp_size > 1:
                raise NotImplementedError(
                    "KV Cache Connector is not supported with pipeline or "
                    "context parallelism: the connector worker is registered "
                    "with the local rank's primary pool only, with no "
                    "coordination across ranks.")

            if self.max_beam_width > 1:
                raise NotImplementedError(
                    "KV Cache Connector is not supported with beam search "
                    "(max_beam_width > 1). The connector's per-request block "
                    "list has no notion of beams, so non-leading beams would "
                    "save and restore the wrong KV state.")

            if self.enable_attention_dp:
                raise NotImplementedError(
                    "KV Cache Connector is not supported with attention data "
                    "parallelism (enable_attention_dp). Dummy requests "
                    "inserted for cross-DP balancing flow through the "
                    "connector scheduler / worker hooks and are not "
                    "distinguished from real requests.")

            kv_cache_config = getattr(self.llm_args, 'kv_cache_config', None)
            if kv_cache_config is not None and kv_cache_config.host_cache_size:
                raise NotImplementedError(
                    "KV Cache Connector is not supported with KV cache host "
                    "offloading (KvCacheConfig.host_cache_size). The connector "
                    "worker is only registered with the primary (GPU) pool and "
                    "cannot read or write blocks that have been evicted to the "
                    "secondary (host) pool, and the connector's load/save "
                    "streams are not synchronized with the internal "
                    "onboard/offload streams.")

            if self.kv_cache_manager is None:
                raise ValueError(
                    "KV Cache Connector requires a KV Cache Manager.")

            if getattr(self.kv_cache_manager, 'is_vswa', False):
                raise NotImplementedError(
                    "KV Cache Connector is not supported with variable "
                    "sliding-window attention (per-layer max_attention_window "
                    "with more than one distinct window size). The connector "
                    "is registered with a single primary pool, but VSWA "
                    "models allocate one pool per window size.")

            if isinstance(self.kv_cache_manager, BaseMambaCacheManager):
                raise NotImplementedError(
                    "KV Cache Connector is not supported with Mamba / hybrid "
                    "linear-attention models. Mamba layers carry SSM state "
                    "rather than per-token KV blocks, so the connector's "
                    "per-layer load/save hooks have nothing meaningful to "
                    "transfer for those layers.")

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

            self.kv_connector_manager.wait_for_initialization()

    def _end_transfer_and_maybe_terminate(self, request: LlmRequest):
        transfer_failed = request.state == LlmRequestState.DISAGG_TRANS_ERROR
        if self.kv_cache_transceiver and request in self.active_requests:
            if transfer_failed:
                # End only the transfer that just became terminal. Keep the
                # request active so the synchronized error path can emit an
                # error response after every async transfer releases ownership.
                self.async_transfer_manager.end_transfer(request)
                return
            # Fast-transfer: KV transfer completed in the same iteration
            # before _handle_responses could run. Create the response now
            # while state is still TRANS_IN_PROGRESS (required by C++
            # createResult). Then proceed with end_transfer + termination.
            response = request.create_response(False, self.dist.rank)
            if response:
                response.result.cached_tokens = request.cached_tokens
                self._maybe_attach_ctx_usage(request, response)
                # Buffer the response instead of enqueueing immediately.
                # With ADP, _enqueue_responses does a tp_gather collective.
                # Calling it here would deadlock because only the owning DP
                # rank reaches this point; the other DP rank never enters
                # the matching collective.  The buffer is flushed later at
                # _flush_pending_transfer_responses where all ranks
                # participate.
                self._pending_transfer_responses.append(
                    (request.py_request_id, response))
            if self.async_transfer_manager.end_transfer(request):
                self.active_requests.remove(request)
                self._terminate_request(request)
            return
        if self.async_transfer_manager.end_transfer(request):
            if transfer_failed:
                return
            # Skip if the PP=1 early path already terminated this request;
            # under PP>1 that path is off, so terminate here on transfer-complete.
            if not self.force_terminate_ctx_for_partial_reuse:
                self._terminate_request(request)

    def _flush_pending_transfer_responses(self):
        """Enqueue buffered transfer-completion responses.

        Must be called at a point where ALL DP ranks execute in lockstep so
        that the tp_gather inside _enqueue_responses does not deadlock.
        """
        responses = self._pending_transfer_responses
        self._pending_transfer_responses = []
        if responses or self.enable_attention_dp:
            # Even when this rank has no responses we must participate in the
            # collective when ADP is enabled so that the other rank's gather
            # can complete.
            self._enqueue_responses(responses)

    def _handle_kv_transfer_timeouts_synced(self):
        """ADP-safe drain of the KV-transfer-timeout consensus collective.

        Lifted out of _handle_responses, which is reached from per-rank-
        divergent gates (_process_previous_batch, _handle_executed_batch).
        Non-ADP runs handle timeouts inline; the buffer is empty here.
        """
        if not (self.enable_attention_dp and self.dist.world_size != 1):
            return
        timed_out = self._pending_timed_out_requests
        self._pending_timed_out_requests = []
        any_timed_out = any(self.dist.tp_allgather(bool(timed_out)))
        if any_timed_out:
            self._handle_errors(error_msg="Request timed out (KV transfer)",
                                requests=timed_out,
                                charge_budget=False)

    def _flush_iter_stats_synced(self):
        """ADP-safe drain of the TLLM_METRICS_ALL_RANKS dict-gather collective.

        Lifted out of _append_iter_stats (same divergent-gate problem as
        _handle_kv_transfer_timeouts_synced).  Under ADP every rank sends
        either the local dict or ``None`` every iter so the gather stays in
        lockstep with peer collectives.  When the gate doesn't hold,
        _append_iter_stats takes its legacy path and never populates the
        buffer.
        """
        tp_size = self.dist.tp_size
        gather_all_ranks = os.environ.get("TLLM_METRICS_ALL_RANKS", "0") == "1"
        if not (gather_all_ranks and self.enable_iter_perf_stats and tp_size > 1
                and self.enable_attention_dp):
            return
        local_dict = self._pending_iter_stats_dict
        self._pending_iter_stats_dict = None
        gathered = self.dist.tp_allgather(local_dict)
        if self.dist.tp_rank != 0:
            return
        rank_dicts = [d for d in gathered if d is not None]
        if not rank_dicts:
            return
        with self.stats_lock:
            if not _stats_buffer_is_unbounded(self.max_stats_len):
                cap = self.max_stats_len * tp_size
                overflow = max(0, len(self.stats) + len(rank_dicts) - cap)
                if overflow:
                    del self.stats[:overflow]
            for d in rank_dicts:
                self.stats.append(("per_rank_dict", d))

    # Performance metrics methods are in PerfMetricsManager (self.perf_manager)

    def _event_loop_wrapper(self):
        try:
            # Skip line profiler during warmup/memory estimation phase to avoid
            # saving incomplete results that would be overwritten anyway
            enable_profiler = bool(os.environ.get(
                "TLLM_LINE_PROFILER_PATH")) and not self.is_warmup
            with host_profiler_context(enable=enable_profiler), \
                 customized_gc_thresholds(self.garbage_collection_gen0_threshold):
                self.event_loop()
        except Exception as e:
            logger.error(f"Error in event loop: {e}")
            logger.error(traceback.format_exc())
            # Stash the original error so local consumers
            # (_await_single_response and BaseWorker.AwaitResponseHelper)
            # can surface it instead of letting callers hang. We do NOT
            # call _handle_errors / _enqueue_responses here: they trigger
            # tp_gather / allgather collectives that would deadlock when
            # only this rank crashed. The is_shutdown notification in
            # _executor_loop_cleanup is enough to wake local waiters.
            self._event_loop_error = e
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
            if not self.worker_started:
                if self.dist.pp_size > 1:
                    self.executed_batch_queue: Queue[BatchStatePP] = Queue(
                        maxsize=self.num_micro_batches)
                    self.executed_batch_response_queue: Queue[
                        BatchStatePP] = Queue(maxsize=-1)
                    # Duplicate the communicator on the main thread before the
                    # PP event loop starts. MPI_Comm_dup is collective across
                    # ranks, so doing it here avoids racing with the worker
                    # thread's point-to-point traffic on the original
                    # communicator during startup.
                    logger.info(
                        "Create new MPI comm for broadcast sample state thread to avoid deadlock."
                    )
                    self._broadcast_mpi_comm = mpi_comm().Dup()
                    broadcast_sample_state_loop = self._broadcast_sample_state_loop
                    if is_trace_enabled("TLLM_TRACE_EXECUTOR_LOOP"):
                        broadcast_sample_state_loop = trace_func(
                            broadcast_sample_state_loop)
                    self.broadcast_sample_state_handler = threading.Thread(
                        target=broadcast_sample_state_loop,
                        daemon=True,
                        name="broadcast_sample_state_handler",
                    )
                    self.broadcast_sample_state_handler.start()
                # Duplicate the communicator on the main thread for
                # sleep/wakeup control messages.  MPI_Comm_dup is collective
                # across all ranks and must run before any worker thread
                # starts to avoid racing with the worker's MPI traffic.
                # Only needed when multi-rank sleep/wakeup is actually
                # possible, i.e. MPI executor path (not Ray), sleep_config
                # enabled, and more than one rank participating.
                if (self.dist.world_size > 1
                        and not self._disable_mpi and getattr(
                            self.llm_args, "sleep_config", None) is not None):
                    logger.info(
                        "Create new MPI comm for sleep/wakeup control listener."
                    )
                    self._sleep_wakeup_comm = mpi_comm().Dup()
                    if self.dist.rank != 0:
                        self._sleep_wakeup_listener_thread = threading.Thread(
                            target=self._sleep_wakeup_listener_loop,
                            daemon=True,
                            name="sleep_wakeup_listener_thread",
                        )
                        self._sleep_wakeup_listener_thread.start()
                self.worker_thread = threading.Thread(
                    target=self._event_loop_wrapper, daemon=True)
                self.worker_thread.start()
                self.worker_started = True
            # Start the sampler's async worker, if it is enabled
            if (isinstance(self.sampler, AsyncWorkerMixin)
                    and self.sampler.async_worker_enabled()):
                logger.info("Starting the async worker for sampler D2H copies")
                self.sampler.async_worker_start()

    def _set_global_steady_clock_offset(self):
        assert self.global_rank >= 0, "rank should be >= 0"

        # First calibration wins (mirrors the C++ guard in CacheTransceiver).
        # PyExecutor is constructed twice per process (memory-profiling dry run,
        # then the real executor), so this method runs twice. Recalibration is
        # idempotent as long as the measurement below reads the raw
        # steady_clock, but skipping it avoids a redundant barrier+allgather
        # and protects the offset if the measurement path ever becomes
        # offset-aware (which would make a second pass observe ~zero skew and
        # wipe the correct value).
        if LlmRequest.global_steady_clock_offset is not None:
            logger.info(
                f"global_steady_clock_offset already set "
                f"({LlmRequest.global_steady_clock_offset}); skipping recalibration "
                f"for rank {self.global_rank}")
            return

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

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    def _shutdown_sleep_wakeup_listeners(self) -> None:
        """Signal sleep/wakeup listener threads to exit and drain bounded ACKs."""
        if self._sleep_wakeup_comm is None or self.dist.world_size <= 1:
            return

        if self.dist.rank == 0:
            logger.info(
                "Sending shutdown to %d sleep/wakeup listener thread(s).",
                self.dist.world_size - 1,
            )
            shutdown_ack_deadline = (time.monotonic() +
                                     _SLEEP_WAKEUP_ACK_TIMEOUT_S)
            shutdown_errors = []
            shutdown_ranks = []
            for dest in range(1, self.dist.world_size):
                try:
                    self._sleep_wakeup_comm.send(
                        {
                            "action": _SleepWakeupAction.SHUTDOWN,
                            "tags": []
                        },
                        dest=dest,
                        tag=_SleepWakeupTag.ACTION,
                    )
                    shutdown_ranks.append(dest)
                except Exception as exc:
                    shutdown_errors.append(
                        f"rank {dest} shutdown send failed: {exc}")
                    logger.warning(
                        "Failed to send sleep/wakeup listener shutdown to "
                        "rank %d: %s", dest, exc)
            for src in shutdown_ranks:
                try:
                    ack = _recv_sleep_wakeup_ack_until(self._sleep_wakeup_comm,
                                                       src,
                                                       shutdown_ack_deadline)
                except Exception as exc:
                    shutdown_errors.append(
                        f"rank {src} shutdown ACK recv failed: {exc}")
                    logger.warning(
                        "Failed to receive sleep/wakeup listener shutdown ACK "
                        "from rank %d: %s", src, exc)
                    continue
                if ack.get("status") != "ok":
                    shutdown_errors.append(
                        ack.get("error")
                        or f"rank {src} returned unknown shutdown ACK")
            if shutdown_errors:
                logger.warning(
                    "Sleep/wakeup listener shutdown completed with errors: %s",
                    "; ".join(shutdown_errors))
        elif self._sleep_wakeup_listener_thread is not None:
            self._sleep_wakeup_listener_thread.join(
                timeout=_SLEEP_WAKEUP_LISTENER_JOIN_TIMEOUT_S)
            if self._sleep_wakeup_listener_thread.is_alive():
                logger.warning(
                    "Sleep/wakeup listener thread did not exit within %.1f "
                    "seconds on rank %d.",
                    _SLEEP_WAKEUP_LISTENER_JOIN_TIMEOUT_S,
                    self.dist.rank,
                )

    def _record_sleep_wakeup_abort(self, control_id: str,
                                   error_msg: str) -> None:
        """Remember an early ABORT until the matching control request fires."""
        if not hasattr(self, "_sleep_wakeup_pending_abort_lock"):
            self._sleep_wakeup_pending_abort_lock = threading.Lock()
        if not hasattr(self, "_sleep_wakeup_pending_aborts"):
            self._sleep_wakeup_pending_aborts = {}
        with self._sleep_wakeup_pending_abort_lock:
            self._sleep_wakeup_pending_aborts[control_id] = error_msg

    def _pop_sleep_wakeup_abort(self,
                                control_id: Optional[str]) -> Optional[str]:
        """Pop a pending ABORT for a control request, if one arrived early."""
        if control_id is None:
            return None
        if not hasattr(self, "_sleep_wakeup_pending_abort_lock"):
            self._sleep_wakeup_pending_abort_lock = threading.Lock()
        if not hasattr(self, "_sleep_wakeup_pending_aborts"):
            self._sleep_wakeup_pending_aborts = {}
        with self._sleep_wakeup_pending_abort_lock:
            return self._sleep_wakeup_pending_aborts.pop(control_id, None)

    def enqueue_requests(
            self,
            requests: List[ExecutorRequest],
            result_wait_queue: "Optional[ActorHandle]" = None) -> List[int]:
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
        Await ready responses
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
        if self.dist.pp_size > 1:
            self.executed_batch_queue.put(None)
            self.broadcast_sample_state_handler.join()
        # Signal non-rank-0 sleep/wakeup listener threads to exit.  This runs
        # after the worker thread has joined, which guarantees that the non-rank-0
        # executor loops have already processed the shutdown broadcast and are
        # no longer driving NCCL, so the send cannot deadlock.
        self._shutdown_sleep_wakeup_listeners()
        self.worker_started = False
        # Release CUDA graphs before resource managers free their GPU memory.
        # Resource managers (e.g. SuffixAutomatonManager) allocate GPU workspace
        # that is referenced by raw pointers inside captured CUDA graphs.  If
        # the workspace is freed first (and returned to the driver via
        # empty_cache), the subsequent CUDA graph teardown can trigger a
        # device-wide cudaErrorIllegalAddress when the driver touches metadata
        # for the now-freed memory regions.
        for engine in (self.model_engine, self.draft_model_engine):
            if engine is not None and hasattr(engine, '_release_cuda_graphs'):
                engine._release_cuda_graphs()
        # Ensure graph destruction has fully completed on device before
        # resource managers start freeing GPU-backed workspaces.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        for manager in self.resource_manager.resource_managers.values():
            if manager:
                manager.shutdown()
        # Note: do NOT call engine.cleanup() here. PyExecutor.shutdown() is
        # also invoked mid-init by configure_kv_cache_capacity() in
        # tensorrt_llm/_torch/pyexecutor/_util.py — the warmup pass calls
        # shutdown() and then immediately reads model_engine.model.model_config
        # to compute kv_cache_max_memory. cleanup() would set
        # model_engine.model = None, breaking that read with
        # `'NoneType' object has no attribute 'model_config'`.
        # The engine's __del__ still calls cleanup() at terminal teardown
        # (when the executor's reference is dropped), which is sufficient for
        # the GMS daemon registry eviction the cleanup hook was added for.
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
        if self.dwdp_manager is not None:
            self.dwdp_manager.__exit__(None, None, None)
            self.dwdp_manager = None

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

    def get_kv_cache_capacity(self) -> dict:
        kv_cache_manager = self.resource_manager.resource_managers.get(
            ResourceManagerType.KV_CACHE_MANAGER)
        if kv_cache_manager is None:
            return {}

        kv_stats = kv_cache_manager.get_kv_cache_stats()
        max_num_blocks = getattr(kv_stats, "max_num_blocks", 0)
        tokens_per_block = getattr(kv_stats, "tokens_per_block", 0)

        if not max_num_blocks:
            max_num_blocks = getattr(kv_cache_manager, "blocks_in_primary_pool",
                                     0)
        if not max_num_blocks:
            max_num_blocks = kv_cache_manager.get_max_resource_count()
        if not tokens_per_block:
            tokens_per_block = getattr(kv_cache_manager, "tokens_per_block", 0)

        if not max_num_blocks or not tokens_per_block:
            return {}

        max_num_blocks = int(max_num_blocks)
        tokens_per_block = int(tokens_per_block)
        return {
            "maxNumBlocks": max_num_blocks,
            "tokensPerBlock": tokens_per_block,
            "maxNumTokens": max_num_blocks * tokens_per_block,
        }

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
            result_wait_queue: "Optional[ActorHandle]" = None) -> int:
        """
        Enqueue a new request, query is only used in `StarAttention`.
        """
        req_id = self.executor_request_queue.enqueue_request(request, query)
        if result_wait_queue is not None:
            with self.response_cv:
                self.result_wait_queues[req_id] = result_wait_queue
        return req_id

    def set_resource_governor_queue(self, queue):
        """Swap the queue used by ResourceGovernor.

        ``queue`` is an IpcQueue (multi-process / proxy path) or an
        IntraProcessQueue (single-process / BaseWorker path). The resource
        governor enablement flag must already have been established during
        construction before the worker thread starts.
        """
        self._resource_governor_queue = queue

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
        if torch_trace_path is not None:
            # Append the rank so each rank writes to its own file. Without
            # this, TP/PP/DP > 1 runs have every rank calling
            # torch_profiler.export_chrome_trace() on the same path
            # concurrently, producing interleaved output that fails to
            # parse in Chrome tracing / Perfetto.
            trace_base, trace_ext = os.path.splitext(torch_trace_path)
            torch_trace_path = f"{trace_base}-rank-{self.global_rank}{trace_ext}"
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

        log_ranks_str = os.environ.get(PROFILE_LOG_RANKS_ENV_VAR_NAME, "0")
        if log_ranks_str.strip().lower() == "all":
            log_all_ranks = True
            log_ranks = set()
        else:
            log_all_ranks = False
            log_ranks = {int(r) for r in log_ranks_str.split(",")}

        calibrator = get_calibrator()

        def profile_step():
            nonlocal it, enabled, start_time, start_event_1, end_event_1, start_event_2, end_event_2, prev_device_step_time
            calibrator.post_step(it)
            if (self.iter_counter in self.profile_stop_iters
                    and not self.is_warmup):
                assert enabled, "Inconsistent CUDA profiling state"
                if enable_torch_trace:
                    torch_profiler.stop()
                    torch_profiler.export_chrome_trace(torch_trace_path)
                    logger.info(
                        f"Profiling stopped at iteration {self.iter_counter}, "
                        f"trace saved to {torch_trace_path}")
                # Persist calibration data and quiesce the device before
                # cudaProfilerStop(): profiling tools may stop collecting or
                # even end the process (e.g. nsys with
                # --capture-range-end=stop-shutdown) as soon as the capture
                # range closes, so the calibration file dump must happen
                # before the stop, and the capture should end on an idle
                # device with no in-flight async copies.
                calibrator.stop()
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStop()
                enabled = False

            # Capture per-loop timing whenever stats or the iter log are
            # enabled. The reading of the OTHER parity's event pair (the
            # ping-pong) is what keeps synchronize() from blocking the GPU
            # — the events being read have already passed by the time we
            # read them. Stashing on self lets the /metrics serializer pick
            # up the values without going through the log line.
            should_capture_timing = start_time is not None and (
                self.print_log or self.enable_iter_perf_stats)
            if should_capture_timing:
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

                host_step_time = (end_time - start_time) * 1000  # milliseconds
                self._latest_host_step_time_ms = host_step_time
                self._latest_prev_device_step_time_ms = prev_device_step_time

                if self.print_log and (log_all_ranks
                                       or self.dist.rank in log_ranks):
                    if prev_device_step_time is None:
                        prev_device_step_time_str = "N/A"  # Handle first iteration
                    else:
                        prev_device_step_time_str = f"{prev_device_step_time}ms"
                    kv_util_str = "N/A"
                    if self.kv_cache_manager is not None:
                        kv_stats = self.kv_cache_manager.get_kv_cache_stats()
                        if kv_stats.max_num_blocks > 0:
                            kv_util_str = f"{1.0 - kv_stats.free_num_blocks / kv_stats.max_num_blocks:.3f}"
                    formatted_timestamp = datetime.datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S")
                    logger.info(
                        f"iter = {self.iter_counter}, "
                        f"global_rank = {self.global_rank}, "
                        f"rank = {self.dist.rank}, "
                        f"num_scheduled_requests = {self.num_scheduled_requests}, "
                        f"kv_cache_util = {kv_util_str}, "
                        f"currank_total_requests = {self.num_fetch_requests_cur_rank}/"
                        f"{self.num_fetch_requests}, "
                        f"host_step_time = {host_step_time}ms, "
                        f"prev_device_step_time = {prev_device_step_time_str}, "
                        f"timestamp = {formatted_timestamp}, "
                        f"states = {self.model_engine.iter_states}")

            it += 1

            if (self.iter_counter in self.profile_start_iters
                    and not self.is_warmup):
                assert not enabled, "Inconsistent CUDA profiling state"
                calibrator.start()
                torch.cuda.cudart().cudaProfilerStart()
                if enable_torch_trace:
                    torch_profiler.start()
                logger.info(
                    f"Profiling started at iteration {self.iter_counter}.")
                enabled = True

            # Notify host line profiler of iteration for iteration-aware profiling
            host_profiler = get_global_profiler()
            if host_profiler is not None:
                host_profiler.notify_iteration(self.iter_counter)

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
                # See the ordering rationale at the in-loop cudaProfilerStop().
                calibrator.stop()
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStop()

    def _get_init_iter_stats(self, num_new_active_requests,
                             new_active_requests_queue_latency_ms):
        stats = IterationStats()
        # Stamp iter at construction time so the JSON `iter` field identifies
        # the loop that CONSTRUCTED this batch. Under the overlap and PP
        # schedulers _process_iter_stats consumes this record in a later
        # loop, so stamping at consumption time (via self.iter_counter)
        # would mislabel the record by 1+ iterations.
        stats.iter = self.iter_counter
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

    @staticmethod
    def _is_stats_dummy_request(req) -> bool:
        return bool(getattr(req, "is_dummy", False))

    def _collect_scheduled_batch_stats(
            self, scheduled_batch: ScheduledRequests) -> ScheduledBatchStats:
        """Collect scheduled-batch counters before forward mutates requests."""
        filter_dummies = getattr(self, "enable_attention_dp", False)

        num_context_requests = 0
        num_ctx_tokens = 0
        num_ctx_kv_tokens = 0
        for req in scheduled_batch.context_requests:
            if filter_dummies and self._is_stats_dummy_request(req):
                continue
            num_context_requests += 1
            try:
                start = req.context_current_position
                chunk = req.context_chunk_size
            except RuntimeError:
                last_chunk = getattr(req, "py_last_context_chunk", None)
                if last_chunk is None or last_chunk[0] is None:
                    continue
                start, end = last_chunk
                chunk = end - start
            num_ctx_tokens += chunk
            num_ctx_kv_tokens += start

        num_gen_requests = 0
        num_gen_kv_tokens = 0
        for req in scheduled_batch.generation_requests:
            if filter_dummies and self._is_stats_dummy_request(req):
                continue
            num_gen_requests += 1
            try:
                num_gen_kv_tokens += req.get_num_tokens(0)
            except RuntimeError:
                pass

        num_paused_requests = 0
        for req in scheduled_batch.paused_requests:
            if filter_dummies and self._is_stats_dummy_request(req):
                continue
            num_paused_requests += 1

        return ScheduledBatchStats(
            num_ctx_requests=num_context_requests,
            num_ctx_tokens=num_ctx_tokens,
            num_ctx_kv_tokens=num_ctx_kv_tokens,
            num_gen_requests=num_gen_requests,
            num_gen_kv_tokens=num_gen_kv_tokens,
            num_paused_requests=num_paused_requests,
        )

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
            req_stat.scheduled = (req in scheduled_requests.encoder_requests
                                  or req in scheduled_requests.context_requests
                                  or req
                                  in scheduled_requests.generation_requests)
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

    def _update_iter_stats(
        self,
        stats,
        iter_latency_ms,
        num_completed_requests,
        scheduled_batch,
        micro_batch_id,
        scheduled_batch_stats: Optional[ScheduledBatchStats] = None
    ) -> IterationStats:
        stats.iter_latency_ms = iter_latency_ms
        scheduled_batch_stats = (scheduled_batch_stats or ScheduledBatchStats())

        stats.num_queued_requests = self.executor_request_queue.get_request_queue_size(
        )
        stats.num_completed_requests = num_completed_requests
        stats.max_num_active_requests = self.max_num_active_requests

        end, total_gpu_memory = torch.cuda.mem_get_info()
        stats.gpu_mem_usage = total_gpu_memory - end
        stats.cpu_mem_usage = 0
        stats.pinned_mem_usage = 0

        # NOTE: stats.iter was stamped at construction time in
        # _get_init_iter_stats. Do NOT re-stamp here from self.iter_counter
        # — under the overlap and PP schedulers this method runs in a later
        # loop than the one that built the batch, so the live counter would
        # mislabel the record.

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

            # Collect per-iteration stats (with deltas) at configured interval.
            # Between calls, C++ deltas accumulate so the reported values cover multiple iterations.
            # Guard: only fetch once per iter_counter to avoid draining deltas in PP multi-batch.
            if (self.iter_counter % self._kv_iter_stats_interval == 0 and
                    self._last_kv_iter_stats_fetch_iter != self.iter_counter):
                self._latest_kv_iter_stats = kv_cache_manager.get_iteration_stats(
                )
                self._last_kv_iter_stats_fetch_iter = self.iter_counter
            else:
                self._latest_kv_iter_stats = None

        # Attention-DP may add dummy requests to keep ranks aligned during
        # distributed scheduling. CUDA graph padding can add dummies too.
        # Those placeholders are not user work, so count the request lists
        # with a filter instead of using the cached batch counters, which
        # include every scheduled item.
        if getattr(self, "enable_attention_dp", False):
            # The sum(...) scans are small and keep planner-facing metrics
            # from treating dummy requests as real load.
            num_context_requests = sum(
                1 for req in scheduled_batch.context_requests
                if not self._is_stats_dummy_request(req))
            num_gen_requests = sum(
                1 for req in scheduled_batch.generation_requests
                if not self._is_stats_dummy_request(req))
            num_paused_requests = sum(1
                                      for req in scheduled_batch.paused_requests
                                      if not self._is_stats_dummy_request(req))
        else:
            num_context_requests = scheduled_batch.num_context_requests
            num_gen_requests = scheduled_batch.num_generation_requests
            num_paused_requests = len(scheduled_batch.paused_requests)
        num_context_requests = int(
            scheduled_batch_stats.num_ctx_requests if scheduled_batch_stats.
            num_ctx_requests is not None else num_context_requests)
        num_gen_requests = int(
            scheduled_batch_stats.num_gen_requests if scheduled_batch_stats.
            num_gen_requests is not None else num_gen_requests)
        num_paused_requests = int(
            scheduled_batch_stats.num_paused_requests if scheduled_batch_stats.
            num_paused_requests is not None else num_paused_requests)

        stats.inflight_batching_stats.num_context_requests = num_context_requests
        stats.inflight_batching_stats.num_gen_requests = num_gen_requests
        stats.inflight_batching_stats.num_scheduled_requests = (
            stats.inflight_batching_stats.num_context_requests +
            stats.inflight_batching_stats.num_gen_requests)
        stats.inflight_batching_stats.num_paused_requests = num_paused_requests
        stats.inflight_batching_stats.avg_num_decoded_tokens_per_iter = 0
        stats.inflight_batching_stats.micro_batch_id = micro_batch_id

        if stats.specdec_stats is not None:
            total_draft_tokens = 0
            total_accepted_tokens = 0
            num_requests_with_draft = 0

            # Aggregate stats from all generation requests
            for req in scheduled_batch.generation_requests:
                # exclude attention dp dummy / CUDA Graph padding requests from AL calculation
                if getattr(req, 'is_dummy', False):
                    continue
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

        # Extra per-iteration request-aggregate counters attached to
        # inflight_batching_stats. These complement the existing
        # num_context_requests / num_gen_requests / num_ctx_tokens /
        # num_paused_requests members with token-weighted counts and
        # queue/paused KV accounting.
        stats.inflight_batching_stats.num_ctx_tokens = int(
            scheduled_batch_stats.num_ctx_tokens if scheduled_batch_stats.
            num_ctx_tokens is not None else stats.inflight_batching_stats.
            num_ctx_tokens)

        # Tokens read from prior state (prefix-cache hits and
        # previously-chunked tokens) summed across scheduled context
        # requests; complements num_ctx_tokens (tokens computed this
        # iteration). Read from py_last_context_chunk, a Python-side
        # cache set by _update_request_states before state mutation — it
        # stays valid after the request transitions to
        # GENERATION_IN_PROGRESS, unlike the C++ getContextChunkSize() /
        # getContextCurrentPosition() accessors that would raise
        # RuntimeError on a mutated request.
        num_ctx_kv_tokens = 0
        if scheduled_batch_stats.num_ctx_kv_tokens is not None:
            num_ctx_kv_tokens = int(scheduled_batch_stats.num_ctx_kv_tokens)
        else:
            for req in scheduled_batch.context_requests:
                if self._is_stats_dummy_request(req):
                    continue
                last_chunk = getattr(req, "py_last_context_chunk", None)
                if last_chunk is not None and last_chunk[0] is not None:
                    start, _end = last_chunk
                    num_ctx_kv_tokens += start
                else:
                    try:
                        num_ctx_kv_tokens += \
                            req.context_current_position
                    except RuntimeError:
                        pass

        # Total KV context length (prompt + tokens generated so far)
        # summed across scheduled generation requests.
        num_gen_kv_tokens = 0
        if scheduled_batch_stats.num_gen_kv_tokens is not None:
            num_gen_kv_tokens = int(scheduled_batch_stats.num_gen_kv_tokens)
        else:
            for req in scheduled_batch.generation_requests:
                if self._is_stats_dummy_request(req):
                    continue
                try:
                    num_gen_kv_tokens += req.get_num_tokens(0)
                except RuntimeError:
                    pass

        # Normal requests waiting in the executor_request_queue that have
        # never been scheduled. Excludes non-normal control items
        # (shutdown/cancel) and items with a missing payload. Each queued
        # item is a RequestQueueItem wrapping an ExecutorRequest
        # (tle::Request). Requests are routed by request_type:
        #   - CONTEXT_AND_GENERATION (default) and CONTEXT_ONLY
        #     (disagg-prefill side) -> queued-context counters.
        #   - GENERATION_ONLY (disagg-decode side, awaiting KV transfer
        #     before they can start decoding) -> queued-gen counters.
        # On a non-disagg engine all items land in the context counters;
        # on a disagg-decode engine all items land in the gen counters.
        num_queued_context_requests = 0
        num_queued_ctx_tokens = 0
        num_queued_gen_requests = 0
        num_queued_gen_kv_tokens = 0
        for item in list(self.executor_request_queue.get_request_queue().queue):
            if not item.is_normal_request:
                continue
            if item.request is None:
                continue
            try:
                token_count = len(item.request.input_token_ids)
            except (AttributeError, TypeError) as e:
                # Unusual request shape with no usable token payload;
                # exclude from all queued counters so downstream consumers
                # see consistent per-request averages. Not expected on the
                # current API (ExecutorRequest construction requires a
                # non-empty input_token_ids), logged so future API drift
                # surfaces instead of being silently dropped.
                logger.warning(f"Excluding queued item {item.id} from queued "
                               f"counters: input_token_ids not readable "
                               f"({type(e).__name__})")
                continue
            if item.request.request_type == RequestType.REQUEST_TYPE_GENERATION_ONLY:
                num_queued_gen_requests += 1
                num_queued_gen_kv_tokens += token_count
            else:
                num_queued_context_requests += 1
                num_queued_ctx_tokens += token_count

        # Total KV context length summed across paused (preempted-decode)
        # requests — were decoding but got evicted back to the waiting
        # pool for this iteration.
        num_paused_kv_tokens = 0
        for req in scheduled_batch.paused_requests:
            if self._is_stats_dummy_request(req):
                continue
            try:
                num_paused_kv_tokens += req.get_num_tokens(0)
            except RuntimeError:
                pass

        stats.inflight_batching_stats.num_ctx_kv_tokens = num_ctx_kv_tokens
        stats.inflight_batching_stats.num_gen_kv_tokens = num_gen_kv_tokens
        stats.inflight_batching_stats.num_queued_context_requests = num_queued_context_requests
        stats.inflight_batching_stats.num_queued_ctx_tokens = num_queued_ctx_tokens
        stats.inflight_batching_stats.num_queued_gen_requests = num_queued_gen_requests
        stats.inflight_batching_stats.num_queued_gen_kv_tokens = num_queued_gen_kv_tokens
        stats.inflight_batching_stats.num_paused_kv_tokens = num_paused_kv_tokens

        return stats

    def _update_batch_acceptance_rate(
            self,
            scheduled_batch: ScheduledRequests,
            sample_state: SampleState,
            iteration_id: Optional[int] = None) -> Tuple[bool, Optional[float]]:
        if (self.speculation_gate is None
                or self.speculation_permanently_disabled or self.is_warmup):
            return False, None

        if (getattr(self.dist, 'has_pp', False)
                and not self.dist.is_last_pp_rank):
            return False, None
        new_tokens_lens = getattr(sample_state.host, 'new_tokens_lens', None)
        if new_tokens_lens is None:
            return False, None
        new_tokens_lens_list = (new_tokens_lens.tolist() if hasattr(
            new_tokens_lens, 'tolist') else list(new_tokens_lens))
        total_draft_tokens = 0
        total_accepted_tokens = 0
        for request in scheduled_batch.generation_requests:
            draft_len = request.num_draft_tokens
            if draft_len <= 0 or request.is_dummy:
                continue
            total_draft_tokens += draft_len
            total_accepted_tokens += request.py_num_accepted_draft_tokens

        if total_draft_tokens <= 0:
            return False, None

        acceptance_rate = total_accepted_tokens / total_draft_tokens
        disabled_now, avg = self.speculation_gate.record_acceptance_rate(
            acceptance_rate, sample_id=iteration_id)
        if disabled_now:
            self.speculation_permanently_disabled = True
        return disabled_now, avg

    def _append_iter_stats(self,
                           stats: IterationStats,
                           req_stats: Optional[List[RequestStats]] = None,
                           kv_iter_stats: Optional[Dict[int, object]] = None,
                           attention_dp_rank: Optional[int] = None,
                           host_step_time_ms: Optional[float] = None,
                           prev_device_step_time_ms: Optional[float] = None,
                           gpu_forward_time_ms: Optional[float] = None):
        """Append one iteration's finalized stats to the export buffer.

        The normal Attention-DP path fans out rank-local rows before calling
        this method; those calls pass ``attention_dp_rank`` and must not enter
        the collective all-rank gather below.

        Args:
            stats: Iteration-level stats.
            req_stats: Optional per-request stats.
            kv_iter_stats: Optional KV iteration stats captured with ``stats``.
            attention_dp_rank: Optional ADP rank for fanned-out rank-local rows.
            host_step_time_ms: Per-loop CPU wall captured by profile_step
                for the loop that built this batch. Surfaces as
                ``hostStepTimeMS`` in the /metrics JSON. Always a clean
                single-loop measurement (matches the log line's
                ``host_step_time``).
            prev_device_step_time_ms: GPU forward time captured by the
                ping-pong CUDA event pair in profile_step. Surfaces as
                ``prevDeviceStepTimeMS`` in the /metrics JSON. Note the
                value lags by one loop relative to ``host_step_time_ms``
                (its sibling on the same record describes a slightly
                older batch); see _profiler ping-pong comment.
            gpu_forward_time_ms: Batch-matched GPU forward time captured by
                the events surrounding this batch's ``_forward_step``.
                Surfaces as ``gpuForwardTimeMS`` in the /metrics JSON.
        """
        # Non-ADP appends immediately, so the latest KV stats belong to this
        # IterationStats. ADP appends later and passes the saved iter-matched
        # KV stats explicitly.
        if not self.enable_attention_dp:
            kv_iter_stats = self._latest_kv_iter_stats
            # Same reasoning for the per-loop timings: non-ADP captures them
            # at append time. ADP passes the values that were saved with the
            # IterationStats at queue time.
            if host_step_time_ms is None:
                host_step_time_ms = self._latest_host_step_time_ms
            if prev_device_step_time_ms is None:
                prev_device_step_time_ms = self._latest_prev_device_step_time_ms

        # Per-record indicator so consumers can interpret iterLatencyMS
        # without inspecting server config. iterLatencyMS spans ~2 loops
        # under "overlap" and a clean single loop under "non_overlap"; the
        # always-clean alternative is hostStepTimeMS.
        # PP (pp_size > 1) is structurally overlap-style regardless of
        # disable_overlap_scheduler: _executor_loop_pp queues a BatchStatePP
        # in iter N and _process_previous_batch_pp consumes it in a later
        # loop, so iterLatencyMS spans ~2 loops there too.
        is_overlap_like = (not self.disable_overlap_scheduler
                           or self.dist.pp_size > 1)
        scheduler_mode = "overlap" if is_overlap_like else "non_overlap"

        tp_size = getattr(self.dist, "tp_size", 1)
        gather_all_ranks = os.environ.get("TLLM_METRICS_ALL_RANKS", "0") == "1"
        if (gather_all_ranks and self.enable_iter_perf_stats and tp_size > 1
                and self.enable_attention_dp and attention_dp_rank is None):
            import json as _json
            local_dict = _json.loads(stats.to_json_str())
            if req_stats:
                local_dict["requestStats"] = [
                    _json.loads(r.to_json_str()) for r in req_stats
                ]
            append_kv_cache_iteration_stats(local_dict, kv_iter_stats)
            if host_step_time_ms is not None:
                local_dict["hostStepTimeMS"] = host_step_time_ms
            if prev_device_step_time_ms is not None:
                local_dict["prevDeviceStepTimeMS"] = prev_device_step_time_ms
            if gpu_forward_time_ms is not None:
                local_dict["gpuForwardTimeMS"] = gpu_forward_time_ms
            local_dict["schedulerMode"] = scheduler_mode
            local_dict["rank"] = self.dist.tp_rank
            # Buffer for _flush_iter_stats_synced; in-place tp_allgather would
            # desync (reached from per-rank-divergent gates).
            self._pending_iter_stats_dict = local_dict
            return

        # Legacy path: rank-0-only (single-rank or iter stats disabled).
        # Tuple layout (length is sniffed by _stats_serializer with
        # len() > N guards so older 4-tuples remain readable):
        #   [0] stats: IterationStats
        #   [1] req_stats: Optional[List[RequestStats]]
        #   [2] kv_iter_stats: Optional[Dict[int, KvCacheIterationStats]]
        #   [3] attention_dp_rank: Optional[int]
        #   [4] host_step_time_ms: Optional[float]
        #   [5] prev_device_step_time_ms: Optional[float]
        #   [6] scheduler_mode: "overlap" | "non_overlap"
        #   [7] gpu_forward_time_ms: Optional[float]
        with self.stats_lock:
            if (not _stats_buffer_is_unbounded(self.max_stats_len)
                    and len(self.stats) > self.max_stats_len):
                self.stats.pop(0)
            self.stats.append(
                (stats, req_stats, kv_iter_stats, attention_dp_rank,
                 host_step_time_ms, prev_device_step_time_ms, scheduler_mode,
                 gpu_forward_time_ms))

    def _process_iter_stats(
        self,
        finished_requests: list[LlmRequest],
        active_requests: List[LlmRequest],
        batch_state: BatchState,
        micro_batch_id: int = 0,
    ):
        """All ranks: build local stats; ADP queues them for later fanout."""
        iter_end_time = time.time()
        # iterLatencyMS semantics differ by scheduler:
        # - Overlap / PP: batch_state.iter_start_time was captured at the top
        #   of the loop that BUILT this batch (one or more loops ago). The
        #   end timestamp is taken here, during the consuming loop. So
        #   iter_latency_ms spans roughly two loops and tracks the batch's
        #   full lifecycle, not the loop body that produced it. Under steady
        #   state, sum(iterLatencyMS) ≈ 2 × wall_time.
        # - Non-overlap: iter_start_time was captured at the top of THIS
        #   loop and the end is taken at the bottom of the same loop, so
        #   iterLatencyMS is the clean per-loop CPU wall.
        # The always-clean alternative is host_step_time_ms (set on every
        # IterationStats record); the per-record schedulerMode field tells
        # consumers which interpretation applies.
        iter_latency_ms = (iter_end_time - batch_state.iter_start_time) * 1e3
        if batch_state.iter_stats is None:
            if batch_state.gpu_forward_events_from_perf_pool:
                self.perf_manager.release_forward_timing_events(
                    batch_state.gpu_forward_start_event,
                    batch_state.gpu_forward_end_event)
            return

        # Snapshot per-loop profiler timings plus the batch-matched GPU
        # forward time. The FPM GPU value is read from CUDA events without
        # synchronizing here; the normal sampler/update path has already
        # established the required completion point for processed batches.
        host_step_time_ms = self._latest_host_step_time_ms
        prev_device_step_time_ms = self._latest_prev_device_step_time_ms
        gpu_forward_time_ms = self.perf_manager.try_compute_gpu_elapsed_time_ms(
            batch_state.gpu_forward_start_event,
            batch_state.gpu_forward_end_event)
        if batch_state.gpu_forward_events_from_perf_pool:
            self.perf_manager.release_forward_timing_events(
                batch_state.gpu_forward_start_event,
                batch_state.gpu_forward_end_event)

        req_stats = self._populate_req_stats(
            finished_requests, active_requests,
            batch_state.scheduled_requests) if (
                self.enable_iter_req_stats
                and self.enable_iter_perf_stats) else None

        stats = self._update_iter_stats(batch_state.iter_stats, iter_latency_ms,
                                        len(finished_requests),
                                        batch_state.scheduled_requests,
                                        micro_batch_id,
                                        batch_state.scheduled_batch_stats)
        if self.enable_attention_dp:
            self._adp_iter_stats.queue(
                stats,
                req_stats,
                kv_iter_stats=self._latest_kv_iter_stats,
                is_rank0=self.dist.rank == 0,
                host_step_time_ms=host_step_time_ms,
                prev_device_step_time_ms=prev_device_step_time_ms,
                gpu_forward_time_ms=gpu_forward_time_ms)
        else:
            self._append_iter_stats(
                stats,
                req_stats,
                host_step_time_ms=host_step_time_ms,
                prev_device_step_time_ms=prev_device_step_time_ms,
                gpu_forward_time_ms=gpu_forward_time_ms)

    def _executor_loop_cleanup(self):
        # Wake any waiters in await_responses BEFORE potentially-blocking
        # work below. If wait_on_pp_send_handles hangs (e.g. after a
        # crash leaves PP send handles in a bad state), the await loop
        # must not hang with it.
        with self.response_cv:
            self.is_shutdown = True
            self.response_cv.notify_all()
        self.shutdown_event.set()

        for i in range(self.num_micro_batches):
            try:
                self.wait_on_pp_send_handles(self.send_handles, i)
                self.wait_on_pp_send_handles(self.send_schedule_handles, i)
                self.wait_on_pp_send_handles(
                    self.send_expected_batch_num_handles, i)
            except Exception:
                # PP send handles may be in a broken state after an
                # event-loop crash. Log and continue; the waiters have
                # already been notified above.
                logger.error(
                    f"Error waiting on PP send handles during cleanup: "
                    f"{traceback.format_exc()}")

    def _pp_schedule_and_propagate(self, microbatch_id: int):
        """The first PP rank schedules the requests and propagates the result to all other PP ranks."""

        # For TP/CP cases, the first rank schedules the requests.
        # For DP cases, the first PP rank schedules the requests.
        scheduled_batch = None
        serializable_schedule = None
        wait_for_disagg_gen_transfer_progress = False
        is_dp_broadcast = self.dist.tp_size > 1 and self.enable_attention_dp
        if self.dist.rank == 0 or (self.dist.is_first_pp_rank
                                   and is_dp_broadcast):
            scheduled_batch, fitting_disagg_gen_init_requests, num_fitting_reqs = self._schedule(
            )
            if self.kv_cache_transceiver:
                fitting_disagg_gen_init_requests, wait_for_disagg_gen_transfer_progress = (
                    self._apply_disagg_transfer_admission(
                        fitting_disagg_gen_init_requests))
            serializable_schedule = SerializableSchedulerOutput.from_scheduler_result(
                scheduled_batch, fitting_disagg_gen_init_requests,
                num_fitting_reqs, wait_for_disagg_gen_transfer_progress)

        # Broadcast within first tp+cp group before send/recv chain to other tp+cp groups
        if self.dist.is_first_pp_rank:
            if self.dist.tp_size > 1 and not self.enable_attention_dp:
                with nvtx_range("tp_broadcast_schedule"):
                    serializable_schedule = self.dist.tp_broadcast(
                        serializable_schedule, root=0)
            if self.dist.cp_size > 1:
                with nvtx_range("cp_broadcast_schedule"):
                    serializable_schedule = self.dist.cp_broadcast(
                        serializable_schedule, root=0)

        # Other ranks receive the schedule result from the previous PP rank.
        if not self.dist.is_first_pp_rank:
            with nvtx_range("recv_schedule_from_prev_pp"):
                serializable_schedule = self.dist.recv_object(
                    self.dist.prev_pp_rank, PPCommTag.SCHEDULE_RESULT)

        # Propagate the schedule result to the next PP rank except the last PP rank.
        if not self.dist.is_last_pp_rank:
            self.wait_on_pp_send_handles(self.send_schedule_handles,
                                         microbatch_id)
            with nvtx_range("send_schedule_to_next_pp"):
                self.send_schedule_handles[
                    microbatch_id] = self.dist.isend_object(
                        serializable_schedule, self.dist.next_pp_rank,
                        PPCommTag.SCHEDULE_RESULT)

        if scheduled_batch is None:
            scheduled_batch, fitting_disagg_gen_init_requests, num_fitting_reqs = serializable_schedule.to_scheduler_result(
                self.active_requests)
            wait_for_disagg_gen_transfer_progress = (
                serializable_schedule.wait_for_disagg_gen_transfer_progress)
        return (scheduled_batch, fitting_disagg_gen_init_requests,
                num_fitting_reqs, wait_for_disagg_gen_transfer_progress)

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

                self._handle_disagg_cache_errors_synced()

                # Fetch new requests from request queue
                new_requests = self._fetch_and_activate_new_requests()
                if self.should_stop_processing:
                    break

                self._handle_control_request()

                if self.kv_cache_transceiver:
                    self._check_disagg_ctx_schedulable_status(new_requests)
                    self._check_disagg_gen_transfer_status()

                if self.enable_iter_perf_stats:
                    iter_stats = self._get_init_iter_stats(
                        len(new_requests),
                        self._get_new_active_requests_queue_latency())

                self._pad_attention_dp_dummy_request()

                # Stage 0: first PP rank schedules requests and propagates the result to all other PP ranks.
                (scheduled_batch, fitting_disagg_gen_init_requests,
                 num_fitting_reqs, wait_for_disagg_gen_transfer_progress
                 ) = self._pp_schedule_and_propagate(microbatch_id)
                if self.dist.rank != 0:
                    # Retry until current rank can run first PP's schedule result.
                    self._pp_retry_until_can_schedule(scheduled_batch)
                    # Run scheduler locally because scheduler may change llm requests' state.
                    if hasattr(self.kv_cache_manager,
                               "prepare_expect_snapshot_points"):
                        self.kv_cache_manager.prepare_expect_snapshot_points(
                            self.active_requests)
                    local_scheduler_output = self.scheduler.schedule_request(
                        self.active_requests, self.inflight_req_ids)
                    if self.kv_cache_transceiver:
                        local_disagg_candidates = getattr(
                            local_scheduler_output,
                            "fitting_disagg_gen_init_requests", [])
                        self._revert_deferred_disagg_gen_init_alloc(
                            local_disagg_candidates,
                            fitting_disagg_gen_init_requests)

                # For requests that are fitting disagg gen init, also prepare resources for KV cache manager
                if self.kv_cache_transceiver:
                    self._prepare_disagg_gen_init(
                        fitting_disagg_gen_init_requests)

                    all_gen_first = self.active_requests and all(
                        req.py_disaggregated_params
                        and req.py_disaggregated_params.schedule_style ==
                        DisaggScheduleStyle.GENERATION_FIRST
                        for req in self.active_requests)
                    self._check_disagg_transfer_progress_when_idle(
                        num_fitting_reqs, fitting_disagg_gen_init_requests,
                        wait_for_disagg_gen_transfer_progress, all_gen_first)

                self.num_scheduled_requests = scheduled_batch.batch_size

                logger.debug(
                    f'iteration {self.iter_counter}, microbatch {microbatch_id}, '
                    f'has {len(self.active_requests)} active_requests, '
                    f'scheduled {scheduled_batch.num_encoder_requests} encoder requests, '
                    f'{scheduled_batch.num_context_requests} context requests and '
                    f'{scheduled_batch.num_generation_requests} generation requests'
                )

                if scheduled_batch.encoder_requests:
                    self._run_encoder_step(scheduled_batch.encoder_requests)

                can_queue, _ = self._can_queue(scheduled_batch)
                if not can_queue:
                    self._revert_gen_alloc(scheduled_batch)
                if not can_queue:
                    logger.debug(
                        f"microbatch {microbatch_id} cannot be queued, skipping"
                    )
                    self.micro_batches[microbatch_id] = None
                else:
                    logger.debug(f"microbatch {microbatch_id} can be queued")

                    self._add_inflight_ids(scheduled_batch)

                    if self.kv_cache_transceiver:
                        # For generation requests which have completed KV cache transfer
                        self._prepare_disagg_gen_transmission_complete(
                            scheduled_batch)

                    self._handle_dynamic_draft_len(scheduled_batch)

                    self.resource_manager.prepare_resources(scheduled_batch)

                    # The generation requests that do not have batch_idx
                    # need to be in front of the batch due to the assumptions
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

                    scheduled_batch_stats = (
                        self._collect_scheduled_batch_stats(scheduled_batch)
                        if self.enable_iter_perf_stats else None)
                    gpu_forward_start = None
                    gpu_forward_end = None
                    gpu_forward_events_from_perf_pool = False
                    if self.enable_iter_perf_stats:
                        gpu_forward_start, gpu_forward_end = self.perf_manager.borrow_forward_timing_events(
                        )
                        gpu_forward_events_from_perf_pool = True

                    # Stage 1.1: Async forward (all ranks) and decoding pass (last rank only)
                    if not self.dist.is_last_pp_rank:
                        with torch.cuda.nvtx.range(
                                f"_forward_step_inter_pp pp_rank {self.dist.pp_rank}"
                        ):
                            sample_state = self._forward_step_inter_pp(
                                scheduled_batch, gpu_forward_start,
                                gpu_forward_end)
                    else:
                        with torch.cuda.nvtx.range(
                                f"_forward_step_last_pp pp_rank {self.dist.pp_rank}"
                        ):
                            # init_disagg_gen_requests must be before engine forward, where the prev_seq_slot is updated.
                            if self.guided_decoder is not None and self.kv_cache_transceiver:
                                self.guided_decoder.add_batch(scheduled_batch)
                                self.guided_decoder.init_disagg_gen_requests()

                            with self.perf_manager.record_perf_events(
                                    gpu_forward_start, gpu_forward_end):
                                batch_outputs = self._forward_step(
                                    scheduled_batch)

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
                                self.sample_stream.wait_stream(
                                    torch.cuda.current_stream())
                                with torch.cuda.stream(self.sample_stream):
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
                            if not self.disable_overlap_scheduler:
                                self._update_generation_requests_that_will_complete_next_iteration(
                                    scheduled_batch.generation_requests)

                    batch_state = BatchStatePP(
                        scheduled_requests=scheduled_batch,
                        sample_state=sample_state,
                        iter_start_time=iter_start_time,
                        iter_stats=iter_stats,
                        scheduled_batch_stats=scheduled_batch_stats,
                        gpu_forward_start_event=gpu_forward_start,
                        gpu_forward_end_event=gpu_forward_end,
                        gpu_forward_events_from_perf_pool=
                        gpu_forward_events_from_perf_pool,
                        microbatch_id=microbatch_id,
                    )

                    self.micro_batches[microbatch_id] = batch_state

                # Stage 1.2: Sync sampler for previous microbatch to start new sample state comm chain.
                # For last PP rank, we must synchronize the previous batch
                # since we need to broadcast its sample state soon afterwards in the same iteration.
                # For other PP ranks, we can delay the synchronization if the current batch cannot be queued.
                previous_batch = self.previous_batch
                if can_queue:
                    self.previous_batch = batch_state
                if (self.dist.is_last_pp_rank
                        or can_queue) and previous_batch is not None:
                    with nvtx_range("sync_previous_sampler_event"):
                        previous_batch.sample_state.sampler_event.synchronize()

                # Stage 2: Enqueue sample state for executed batch to ring broadcast it in background thread asynchronously.
                # send/recv chain: (pp_size - 1) -> 0 -> 1 -> ... -> (pp_size - 2)
                # intermediate ranks: send/recv sample state for next microbatch to allow overlap
                offset = -1 if self.dist.is_last_pp_rank else (
                    1 - self.dist.pp_size)
                executed_microbatch_id = (microbatch_id +
                                          offset) % self.num_micro_batches
                executed_batch = self.micro_batches[executed_microbatch_id]
                if executed_batch is not None:
                    self.executed_batch_queue.put(executed_batch)
                    self.unhandled_batch_counter += 1
                self.micro_batches[executed_microbatch_id] = None

                def fetch_executed_batches() -> list[BatchStatePP]:
                    executed_batches = []
                    if self.pp_async_broadcast_sample_state:
                        # Wait for at least one batch to finish if no new request is available.
                        must_get = not can_queue
                    else:
                        must_get = True
                    while not self.executed_batch_response_queue.empty() or (
                            must_get and self.unhandled_batch_counter > 0):
                        with nvtx_range("get_executed_batch"):
                            executed_batches.append(
                                self.executed_batch_response_queue.get())
                        must_get = False
                    return executed_batches

                def ring_broadcast_executed_batch_num(
                        executed_batch_num: int) -> int:
                    if self.dist.is_first_pp_rank and self.dist.tp_size * self.dist.cp_size > 1:
                        with nvtx_range("tp_cp_broadcast_executed_batch_num"):
                            executed_batch_num = self.dist.tp_cp_broadcast(
                                executed_batch_num,
                                root=0,
                            )
                    if not self.dist.is_first_pp_rank:
                        with nvtx_range("recv_expected_batch_num"):
                            executed_batch_num = self.dist.recv_object(
                                src=self.dist.prev_pp_rank,
                                tag=PPCommTag.EXECUTED_BATCH_NUM,
                            )
                    if not self.dist.is_last_pp_rank:
                        self.wait_on_pp_send_handles(
                            self.send_expected_batch_num_handles, microbatch_id)
                        with nvtx_range("send_expected_batch_num"):
                            self.send_expected_batch_num_handles[
                                microbatch_id] = self.dist.isend_object(
                                    executed_batch_num,
                                    dest=self.dist.next_pp_rank,
                                    tag=PPCommTag.EXECUTED_BATCH_NUM,
                                )
                    return executed_batch_num

                def handle_executed_batches(executed_batch_num: int):
                    if self.dist.rank != 0:
                        dequeue_counter = 0
                        while dequeue_counter < executed_batch_num:
                            with nvtx_range("get_executed_batch"):
                                executed_batch = self.executed_batch_response_queue.get(
                                )
                            self._handle_executed_batch(executed_batch)
                            dequeue_counter += 1
                    else:
                        for executed_batch in executed_batches:
                            self._handle_executed_batch(executed_batch)
                    self.unhandled_batch_counter -= executed_batch_num

                executed_batch_num = 0

                # Stage 3.1: The first rank determines the number of executed batches.
                if self.dist.rank == 0:
                    executed_batches = fetch_executed_batches()
                    executed_batch_num = len(executed_batches)

                # Stage 3.2: Broadcast the number of executed batches to other ranks.
                executed_batch_num = ring_broadcast_executed_batch_num(
                    executed_batch_num)

                # Stage 3.3: Handle executed batches.
                handle_executed_batches(executed_batch_num)

                # Stage 4: March forward in microbatch slots
                microbatch_id = (microbatch_id + 1) % self.num_micro_batches
                self.iter_counter += 1

            # Stage 5: Handle remaining executed batches in the queue.
            while self.unhandled_batch_counter > 0:
                with nvtx_range("get_executed_batch"):
                    executed_batch = self.executed_batch_response_queue.get()
                self._handle_executed_batch(executed_batch)
                self.unhandled_batch_counter -= 1

    def _broadcast_sample_state_loop(self):
        logger.debug(
            f"Starting broadcast sample state loop for pp_rank {self.dist.pp_rank}"
        )
        torch.cuda.set_device(self.device_id)
        # ensure the context is created, otherwise, some MPI calls will fail.
        CUASSERT(cudart.cudaSetDevice(self.device_id))
        # pkl5.Intracomm serializes send/recv through internal locks. Sharing
        # one communicator between the executor loop and this background
        # thread can serialize unrelated traffic or deadlock. Use the
        # communicator duplicated on the main thread so this thread does not
        # perform pkl5 operations on the worker's startup communicator.
        broadcast_mpi_comm = self._broadcast_mpi_comm
        assert broadcast_mpi_comm is not None
        set_thread_local_mpi_comm(broadcast_mpi_comm)
        try:
            while True:
                executed_batch = self.executed_batch_queue.get()
                if executed_batch is None:
                    break
                self._ring_broadcast_sample_state(executed_batch)
                # Do not wait for PP send handles here. The next
                # _ring_broadcast_sample_state call drains the previous isend
                # for the same microbatch_id before reusing that slot.
                #
                # Waiting here can hang during shutdown. Peer ranks may have
                # left this loop and moved to the next executor setup, so no
                # rank is polling the broadcast communicator. In that state,
                # pkl5's final MPI_Waitall can spin in ucp_worker_progress
                # because UCX rendezvous sends need receive-side progress to
                # complete.
                #
                # UCX workers are shared across communicators in the process.
                # Later MPI activity on the main communicator can still
                # advance these pending broadcast-comm sends, and they
                # complete before MPI_Finalize. (nvbug/6095421)
        finally:
            # Keep the duplicated communicator alive until process teardown.
            # With PP >= 3, the ring has asymmetric send/recv roles: the
            # second-last PP rank never issues an isend, so its broadcast
            # thread can exit with no pending sends. Freeing it tears down the
            # receive-side handle for peer ranks' in-flight isends; their
            # MPI_Test then stops making progress on the remaining pkl5
            # subsidiary requests, leaving wait_on_pp_send_handles spinning.
            # (nvbug/6095421)
            #
            # This retention is bounded: PyExecutor is created at most a few
            # times per process (KV-cache estimation executor and real
            # executor), and MPI reclaims the communicators on MPI_Finalize.
            set_thread_local_mpi_comm(None)

    def _sleep_wakeup_listener_loop(self) -> None:
        """Listener loop for sleep/wakeup control messages on non-rank-0 workers.

        Runs in a dedicated daemon thread on every non-rank-0 rank.  Blocks on
        ``_sleep_wakeup_comm.recv`` waiting for a control message from rank-0.
        ``PREPARE`` quiesces the rank and ACKs without changing VMM state;
        ``COMMIT`` executes the requested VMM operation
        (``release_with_tag`` or ``materialize_with_tag``), releases the local
        control barrier, and ACKs.  ``ABORT`` either releases the matching
        active control barrier or records the ``op_id`` so the executor loop can
        skip that control request when it arrives later.  ``SHUTDOWN`` ACKs
        rank-0 and breaks the loop cleanly.

        Safety notes:
        - ``_sleep_wakeup_comm`` is a dedicated duplicated MPI communicator, isolated
          from all regular executor MPI traffic.
        - The ``CONTROL_REQUEST_ID`` sentinel is broadcast to all ranks via the
          normal request-broadcaster path, so each non-rank-0 executor loop also
          enters ``_handle_control_request()`` and blocks on ``control_action_done``.
          This listener thread owns the ``control_action_done`` / ``control_request_barrier``
          lifecycle on non-rank-0 ranks:
            1. Wait on ``control_request_barrier`` to confirm the executor loop has
               drained all in-flight work (mirroring rank-0's ``control_action()``
               barrier handshake).
            2. ACK ``PREPARE`` while keeping the executor loop parked.
            3. On ``COMMIT``, perform the VMM operation with
               ``torch.cuda.synchronize()`` guards.
            4. Clear ``control_request_barrier`` (reset for the next cycle) then set
               ``control_action_done`` to unblock the executor loop.
            5. Send the final ACK to rank-0 only *after* step 4, so rank-0 cannot exit
               ``control_action()`` and resume broadcasting new requests before this
               rank's executor loop is guaranteed to have cleared the control barrier.
        - ``gc.collect()`` and ``torch.cuda.empty_cache()`` are safe to call
          from a non-main thread.
        """
        import gc
        import sys

        from tensorrt_llm._torch.virtual_memory import (materialize_with_tag,
                                                        release_with_tag)
        from tensorrt_llm.llmapi.llm_args import ExecutorMemoryType

        torch.cuda.set_device(self.device_id)
        CUASSERT(cudart.cudaSetDevice(self.device_id))
        set_thread_local_mpi_comm(self._sleep_wakeup_comm)
        try:
            while True:
                msg = self._sleep_wakeup_comm.recv(source=0,
                                                   tag=_SleepWakeupTag.ACTION)
                if msg.get("action") == _SleepWakeupAction.SHUTDOWN:
                    logger.debug(
                        "Sleep/wakeup listener (rank %d): received shutdown, "
                        "exiting.", self.dist.rank)
                    self._sleep_wakeup_comm.send(
                        {
                            "status": "ok",
                            "error": None,
                            "phase": _SleepWakeupAction.SHUTDOWN,
                        },
                        dest=0,
                        tag=_SleepWakeupTag.ACK,
                    )
                    break
                # Use .get() so a missing "action" key surfaces as an unknown
                # action inside the try rather than a bare KeyError here.
                action = msg.get("action", "<unknown>")
                op_id = msg.get("op_id")
                error_msg = None
                release_control_request = True
                try:
                    # Decode tags inside the try so KeyError / ValueError from
                    # a malformed message still results in an error ACK being
                    # sent via the finally below, rather than deadlocking rank-0.
                    tags = [ExecutorMemoryType(t) for t in msg["tags"]]
                    target_action = msg.get("target_action", action)
                    # Wait for the executor loop to drain all active requests and
                    # enter _handle_control_request().  It signals readiness by
                    # setting control_request_barrier (the same event rank-0's
                    # control_action() waits on locally).  This guarantees no
                    # in-flight CUDA kernels from the executor are still running
                    # on this rank when the VMM mapping is changed below.
                    if action == _SleepWakeupAction.ABORT:
                        # ABORT is sent after rank-0 fails to broadcast an
                        # ACTION to every peer.  If the matching control
                        # request is already parked, release it below.  If the
                        # ABORT arrived early, record it so _handle_control_request()
                        # can skip the matching sentinel later.  This keeps
                        # stale ABORT messages from clearing the wrong barrier
                        # or parking the listener thread forever.
                        error_msg = (
                            "rank 0 aborted sleep/wakeup before local "
                            f"execution: {msg.get('reason', 'unknown error')}")
                        active_control_id = getattr(self, "_active_control_id",
                                                    None)
                        if op_id is None:
                            release_control_request = False
                        elif (not self.control_request_barrier.is_set()
                              or active_control_id != op_id):
                            self._record_sleep_wakeup_abort(op_id, error_msg)
                            release_control_request = False
                        logger.warning("Sleep/wakeup listener: %s", error_msg)
                    elif action in (_SleepWakeupAction.PREPARE,
                                    _SleepWakeupAction.COMMIT,
                                    _SleepWakeupAction.SLEEP,
                                    _SleepWakeupAction.WAKEUP):
                        self.control_request_barrier.wait()
                        active_control_id = getattr(self, "_active_control_id",
                                                    None)
                        if op_id is not None and active_control_id != op_id:
                            error_msg = (
                                f"stale control message for op_id={op_id}; "
                                f"active control_id={active_control_id}")
                            release_control_request = False
                            logger.warning("Sleep/wakeup listener: %s",
                                           error_msg)
                        else:
                            torch.cuda.synchronize()
                            if action == _SleepWakeupAction.PREPARE:
                                # Prepared means this rank is quiesced and
                                # ready to commit, but VMM state is unchanged.
                                release_control_request = False
                            elif target_action == _SleepWakeupAction.SLEEP:
                                release_with_tag(*tags)
                                torch.cuda.synchronize()
                                gc.collect()
                                torch.cuda.empty_cache()
                            elif target_action == _SleepWakeupAction.WAKEUP:
                                materialize_with_tag(*tags)
                                torch.cuda.synchronize()
                            else:
                                error_msg = (
                                    f"unknown target action '{target_action}'")
                                logger.warning(
                                    "Sleep/wakeup listener: %s, ignoring.",
                                    error_msg)
                    else:
                        error_msg = f"unknown action '{action}'"
                        logger.warning("Sleep/wakeup listener: %s, ignoring.",
                                       error_msg)
                except (KeyError, TypeError, ValueError, RuntimeError,
                        torch.OutOfMemoryError) as exc:
                    error_msg = (f"rank {self.dist.rank} '{action}' failed: "
                                 f"{exc}\n{traceback.format_exc()}")
                    logger.error("Sleep/wakeup listener: error executing '%s':",
                                 action,
                                 exc_info=True)
                finally:
                    # Always ACK so rank-0 does not deadlock; carry error
                    # details so rank-0 can raise after all ranks respond.
                    # If an exception bypassed the narrow except above (e.g.,
                    # MemoryError, SystemError), detect it via sys.exc_info()
                    # so the ACK correctly reflects failure rather than falsely
                    # reporting status=ok while the thread is unwinding.
                    if error_msg is None and sys.exc_info()[0] is not None:
                        exc = sys.exc_info()[1]
                        error_msg = (
                            f"rank {self.dist.rank} '{action}' failed with "
                            f"uncaught {type(exc).__name__}: {exc!r}")
                        logger.error(
                            "Sleep/wakeup listener: uncaught exception on "
                            "rank %d during '%s':",
                            self.dist.rank,
                            action,
                            exc_info=True,
                        )
                    # Unblock the executor loop that is waiting in
                    # _handle_control_request().  Clear control_request_barrier
                    # first so that it is clean for the next sleep/wakeup cycle
                    # before the executor loop resumes and could potentially
                    # set it again.  Only then signal control_action_done.
                    if release_control_request:
                        self.control_request_barrier.clear()
                        self.control_action_done.set()
                    # Send the ACK to rank-0 only after the executor loop on
                    # this rank has been unblocked.  This prevents rank-0 from
                    # exiting control_action() and broadcasting new requests
                    # before our executor loop has cleared its control barrier
                    # and is ready to participate in the next collective.
                    self._sleep_wakeup_comm.send(
                        {
                            "status": "ok" if error_msg is None else "error",
                            "error": error_msg,
                            "op_id": op_id,
                            "phase": action,
                        },
                        dest=0,
                        tag=_SleepWakeupTag.ACK,
                    )
        finally:
            set_thread_local_mpi_comm(None)

    def _ring_broadcast_sample_state(
        self,
        executed_batch: Optional[BatchStatePP],
    ) -> None:
        if executed_batch is None:
            return

        tag = PPCommTag.SAMPLE_STATE
        microbatch_id = executed_batch.microbatch_id
        sample_state = executed_batch.sample_state
        requests = sample_state.requests

        if not self.dist.is_last_pp_rank:
            # Receive tokens from previous pp rank (w.r.t model forward direction)
            with nvtx_range("recv_sample_state"):
                sample_state.host, py_result_diffs = self.dist.recv_object(
                    src=self.dist.prev_pp_rank,
                    tag=tag,
                )

            for request, py_result_diff in zip(requests, py_result_diffs):
                request.py_result.apply_diff(py_result_diff)

        self.executed_batch_response_queue.put(executed_batch)

        # Send tokens to next pp rank (w.r.t model forward direction)
        # Second last rank does not need to since last rank has original decoded tokens
        if not self.dist.is_second_last_pp_rank:
            py_result_diffs = []
            for request in requests:
                diff = request.py_result.get_diff()
                py_result_diffs.append(diff)
                request.py_result.reset_diff()
            self.wait_on_pp_send_handles(self.send_handles, microbatch_id)
            with nvtx_range("send_sample_state"):
                self.send_handles[microbatch_id] = self.dist.isend_object(
                    (sample_state.host, py_result_diffs),
                    dest=self.dist.next_pp_rank,
                    tag=tag,
                )

    def _handle_executed_batch(self, executed_batch: Optional[BatchStatePP]):
        finished_requests = []
        if executed_batch is not None:
            with torch.cuda.nvtx.range("_handle_executed_batch_pp"):
                self._update_requests(executed_batch.sample_state)
                if self.speculation_gate is not None:
                    self._update_batch_acceptance_rate(
                        executed_batch.scheduled_requests,
                        executed_batch.sample_state,
                        iteration_id=self.iter_counter)

                scheduled_requests = executed_batch.scheduled_requests
                if self._is_kv_manager_v2:
                    # Finalize V2 context KV before disagg transfer/response
                    # handling can terminate the request.
                    self.kv_cache_manager.update_context_resources(
                        scheduled_requests)
                if self.kv_cache_transceiver:
                    finished_ctx_reqs = scheduled_requests.context_requests_last_chunk
                    self._send_kv_async(finished_ctx_reqs)
                self._flush_pending_transfer_responses()
                self._handle_canceled_requests()

                finished_requests = self._handle_responses()
                # Complete ctx send sessions AFTER responses are created so
                # _handle_responses sees the request before it is terminated.
                if self.kv_cache_transceiver:
                    self._check_disagg_ctx_cache_transfer_status(0)
                sample_state_scheduled_requests = executed_batch.scheduled_requests
                attn_metadata = getattr(self.model_engine, 'attn_metadata',
                                        None)
                kv_cache_dtype_byte_size = getattr(self.model_engine,
                                                   'kv_cache_dtype_byte_size',
                                                   None)
                self.resource_manager.update_resources(
                    sample_state_scheduled_requests, attn_metadata,
                    kv_cache_dtype_byte_size)

                self._remove_inflight_ids(scheduled_requests)

        if self.kv_cache_transceiver and self.async_transfer_manager.has_any_inflight_requests(
        ):
            self._check_kv_transfer_timeout()

        if self._disagg_pp_termination_handler is not None:
            self._disagg_pp_termination_handler.terminate_pending_requests()

        if executed_batch is not None:
            self._commit_kv_cache_stats(executed_batch.scheduled_requests)

        if self.enable_iter_perf_stats and executed_batch is not None:
            self._process_iter_stats(
                finished_requests,
                self.active_requests,
                executed_batch,
                executed_batch.microbatch_id % self.dist.pp_size,
            )

        # Drain the synced-collective buffers outside the per-rank-divergent
        # ``executed_batch is not None`` branch.  handle_executed_batches
        # invokes this helper the same number of times on every rank
        # (broadcast count), so the flushes are rank-symmetric.
        self._handle_kv_transfer_timeouts_synced()
        self._flush_iter_stats_synced()

    @nvtx_range("wait_on_pp_send_handles")
    def wait_on_pp_send_handles(self, send_handles, microbatch_id):
        if send_handles[microbatch_id] is not None:
            send_handles[microbatch_id].wait()
            send_handles[microbatch_id] = None

    def _handle_dynamic_draft_len(self,
                                  scheduled_batch: ScheduledRequests) -> None:
        """Handle dynamic draft length for the current batch.

        Must be called BEFORE prepare_resources so that KV cache allocation
        uses the correct draft length.

        Two things happen here:
        1. Determine the runtime draft length from the draft_len_schedule
           based on the current batch size, and store it on model_engine so
           that the rest of the forward path can read it.
        2. Pad / truncate each request's py_draft_tokens to exactly match
           the determined draft length, ensuring uniform draft token counts across the
           batch (required by CUDA graph replay and the attention kernel).

        When dynamic draft length is not enabled, runtime_draft_len is simply
        set to max_draft_len (the static maximum).
        """
        if not hasattr(self.model_engine, 'max_draft_len'):
            return

        if self.speculation_permanently_disabled:
            for request in scheduled_batch.generation_requests:
                request.py_draft_tokens = []
            self.model_engine.runtime_draft_len = 0
            return

        if (self.model_engine.spec_config is not None
                and self.model_engine.spec_config.draft_len_schedule is not None
                and self.model_engine.spec_config.spec_dec_mode.
                support_dynamic_draft_len()):
            from tensorrt_llm._torch.speculative.utils import \
                get_draft_len_for_batch_size

            spec_dec_mode = self.model_engine.spec_config.spec_dec_mode

            # 1. Resolve runtime draft length from schedule
            runtime_draft_len = get_draft_len_for_batch_size(
                self.model_engine.spec_config.draft_len_schedule,
                scheduled_batch.batch_size, self.model_engine.max_draft_len)
            # 2. Pad or truncate draft tokens to the resolved length
            DRAFT_BUFFER_PAD = 0  # Buffer sentinel, not PARD mask_token_id.
            rejection_on = getattr(self.model_engine.spec_config,
                                   "use_rejection_sampling", False)
            for request in scheduled_batch.generation_requests:
                current_num_draft_tokens = len(request.py_draft_tokens)
                # One-model rejection: a gen request entering with 0 real draft
                # tokens produced no draft-prob scatter for its slot last iter,
                # so next iter's rejection kernel would read a stale draft_probs
                # row. Mark it (pre-pad signal) so _prepare_tp_inputs writes a
                # one-hot placeholder row after spec_metadata.prepare().
                request.py_needs_onehot_draft_probs = (
                    rejection_on and current_num_draft_tokens == 0)
                if spec_dec_mode.is_pard():
                    # special case: PARD carries 2K-1 draft tokens per request
                    runtime_draft_token_buffer_width = (
                        self.model_engine.spec_config.
                        get_runtime_tokens_per_gen_step(runtime_draft_len) - 1)
                    current_runtime_draft_len = (
                        current_num_draft_tokens +
                        1) // 2 if current_num_draft_tokens > 0 else 0
                    real_draft_tokens = request.py_draft_tokens[:min(
                        current_runtime_draft_len, runtime_draft_len)]
                    real_draft_tokens.extend(
                        [DRAFT_BUFFER_PAD] *
                        (runtime_draft_len - len(real_draft_tokens)))
                    request.py_draft_tokens = real_draft_tokens + [
                        DRAFT_BUFFER_PAD
                    ] * (runtime_draft_token_buffer_width -
                         len(real_draft_tokens))
                else:
                    if current_num_draft_tokens < runtime_draft_len:
                        padding_needed = (runtime_draft_len -
                                          current_num_draft_tokens)
                        request.py_draft_tokens.extend([DRAFT_BUFFER_PAD] *
                                                       padding_needed)
                    elif current_num_draft_tokens > runtime_draft_len:
                        request.py_draft_tokens = request.py_draft_tokens[:
                                                                          runtime_draft_len]

            self.model_engine.runtime_draft_len = runtime_draft_len
        else:
            # Linear-tree modes (incl. PARD) use logical K; tree decoding
            # (e.g. EAGLE3 dynamic tree) uses total tree tokens. Same
            # selection as _prepare_tp_inputs and _get_graphs_to_capture.
            spec_config = self.model_engine.spec_config
            self.model_engine.runtime_draft_len = (
                self.model_engine.max_draft_len
                if spec_config is not None and spec_config.is_linear_tree else
                self.model_engine.max_total_draft_tokens)

    @nvtx_range("_can_queue")
    def _can_queue(self, scheduled_batch):

        # can_queue_this_rank is for case that the batch is not empty on this rank, but empty on other ranks
        # For bs == 1, we cannot pad dummy request to make the batch non-empty since it will cause the batch size to be 2.
        # 1 for dummy request, 1 for the yet-to-complete but not-yet-updated request.
        if self.enable_attention_dp:
            tp_batch_sizes = self.dist.tp_allgather(scheduled_batch.batch_size)
            can_queue = 0 not in tp_batch_sizes
            can_queue_this_rank = scheduled_batch.batch_size > 0
        else:
            can_queue = can_queue_this_rank = scheduled_batch.batch_size > 0

        return can_queue, can_queue_this_rank

    def _revert_gen_alloc(self, scheduled_batch):
        """Revert KV cache capacity growth when the batch is skipped.

        With attention DP, can_queue=False means another rank has an empty
        batch so no forward pass will run.  The V2 scheduler already grew
        each generation request's KV cache capacity during scheduling;
        revert that growth so it does not accumulate across skipped
        iterations and overflow the host page-index buffer.

        Only applies to KV cache manager V2 + scheduler V2, because the V2
        scheduler allocates KV cache capacity during scheduling, before the
        can_queue check.  V1 allocates in prepare_resources() after the
        can_queue check, so no revert is needed.
        """
        if self._is_kv_manager_v2:
            for req in scheduled_batch.generation_requests:
                self.kv_cache_manager.revert_allocate_generation(req)

    def _finalize_adp_dummy_allocation(self, can_queue: bool) -> None:
        """Commit or roll back this iteration's tentative ADP dummy.

        Dummy allocation is rank-local, while ``can_queue`` is a TP-wide
        decision. If one rank cannot allocate its dummy, peers that succeeded
        must release theirs before retrying or the fixed dummy request ID leaks
        cache resources on every skipped iteration.
        """
        if not self._enable_dsv4_adp_dummy_fixes:
            return

        dummy_request = self._pending_adp_dummy_request
        self._pending_adp_dummy_request = None
        if dummy_request is None or can_queue:
            return

        dummy_request.state = LlmRequestState.GENERATION_COMPLETE
        if getattr(dummy_request, "py_seq_slot", None) is not None:
            # Full resource preparation already ran before a connector changed
            # the final queue decision. Every manager can use normal teardown.
            self._terminate_request(dummy_request)
            self.active_requests.remove(dummy_request)
            return

        # Before prepare_resources(), only the KV and speculative managers have
        # allocated anything for the dummy. Do not run the full teardown path:
        # other resource managers may require a prior prepare call.
        spec_resource_manager = self.resource_manager.get_resource_manager(
            ResourceManagerType.SPEC_RESOURCE_MANAGER)
        if spec_resource_manager is not None:
            spec_resource_manager.free_resources(dummy_request)
        self.kv_cache_manager.free_resources(dummy_request)
        self.active_requests.remove(dummy_request)

    def _revert_ctx_alloc(self, dropped_context_requests):
        """Revert V2 context KV growth for requests deferred after scheduling."""
        for req in dropped_context_requests:
            self.kv_cache_manager.revert_allocate_context(req)

    @nvtx_range("_prefetch_for_context_requests")
    def _prefetch_for_context_requests(self) -> None:
        """Pre-stage disk blocks to host for upcoming context requests with block reuse."""
        if not isinstance(getattr(self, "kv_cache_manager", None),
                          KVCacheManagerV2):
            return
        if not self.kv_cache_manager.enable_block_reuse:
            return
        if self.kv_cache_manager.disk_prefetch_num_reqs <= 0:
            return
        max_prefetch = self.kv_cache_manager.disk_prefetch_num_reqs
        candidates = []
        for req in self.active_requests:
            if len(candidates) >= max_prefetch:
                break
            if (req.is_first_context_chunk and req.py_request_id
                    not in self.kv_cache_manager.kv_cache_map
                    and req.py_request_id not in self._prefetched_request_ids):
                candidates.append(req)
                self._prefetched_request_ids.add(req.py_request_id)

        if candidates:
            self.kv_cache_manager.prefetch_for_context_tokens(candidates)

    def _commit_kv_cache_stats(self,
                               scheduled_batch: ScheduledRequests) -> None:
        if self._is_kv_manager_v2:
            self.kv_cache_manager.commit_scheduled_kv_cache_stats(
                scheduled_batch)

    def _get_disagg_transfer_admission_controller(
            self) -> DisaggTransferAdmissionController:
        controller = getattr(self, "_disagg_transfer_admission_controller",
                             None)
        if controller is not None:
            return controller

        cache_transceiver_config = getattr(getattr(self, "llm_args", None),
                                           "cache_transceiver_config", None)
        kv_cache_manager = getattr(self, "kv_cache_manager", None)
        return DisaggTransferAdmissionController(
            getattr(cache_transceiver_config, "max_tokens_in_buffer", None),
            getattr(kv_cache_manager, "tokens_per_block", None),
        )

    @staticmethod
    def _is_disagg_gen_only_no_context_benchmark() -> bool:
        """Return whether ``gen_only_no_context`` skips KV transfer."""
        return os.getenv("TRTLLM_DISAGG_BENCHMARK_GEN_ONLY") == "1"

    def _uses_async_disagg_gen_transfer(self) -> bool:
        """Return whether generation KV transfers can remain in flight."""
        return (not self._is_disagg_gen_only_no_context_benchmark() and
                os.getenv("TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP") != "1")

    def _uses_kv_manager_v2(self) -> bool:
        explicit_flag = getattr(self, "_is_kv_manager_v2", None)
        if explicit_flag is not None:
            return bool(explicit_flag)
        return isinstance(getattr(self, "kv_cache_manager", None),
                          KVCacheManagerV2)

    def _apply_disagg_transfer_admission(
        self, fitting_disagg_gen_init_requests: List[LlmRequest]
    ) -> Tuple[List[LlmRequest], bool]:
        # gen_only_no_context has no CTX worker and does not transfer data.
        # Real synchronous gen_only transfers still honor the budget to bound
        # the number of blocking transfers started in one executor iteration.
        if self._is_disagg_gen_only_no_context_benchmark():
            return fitting_disagg_gen_init_requests, False

        controller = self._get_disagg_transfer_admission_controller()
        if not (getattr(self, "kv_cache_transceiver", None)
                and controller.enabled() and fitting_disagg_gen_init_requests):
            return fitting_disagg_gen_init_requests, False

        admission_result = controller.select(self.active_requests,
                                             fitting_disagg_gen_init_requests)
        if admission_result.deferred_request_count > 0:
            logger.debug("Disagg transfer admission deferred "
                         f"{admission_result.deferred_request_count} requests; "
                         f"active transfer blocks="
                         f"{admission_result.active_transfer_blocks}, "
                         f"admitted transfer blocks="
                         f"{admission_result.admitted_transfer_blocks}, "
                         f"budget={controller.max_transfer_blocks}")

        self._revert_deferred_disagg_gen_init_alloc(
            fitting_disagg_gen_init_requests,
            admission_result.admitted_requests)

        return (admission_result.admitted_requests,
                admission_result.is_blocked_by_active_transfers())

    def _revert_deferred_disagg_gen_init_alloc(
            self, candidates: List[LlmRequest],
            admitted_requests: List[LlmRequest]) -> None:
        if not (self._uses_kv_manager_v2() and candidates):
            return

        admitted_request_ids = {
            request.py_request_id
            for request in admitted_requests
        }
        deferred_requests = [
            request for request in candidates
            if request.py_request_id not in admitted_request_ids
        ]
        if deferred_requests:
            self._revert_ctx_alloc(deferred_requests)

    @staticmethod
    def _dist_size(dist, name: str) -> int:
        try:
            return int(getattr(dist, name))
        except (AttributeError, TypeError, ValueError):
            return 1

    def _allgather_model_parallel_status(
            self, local_status: Tuple[int, bool]) -> List[Tuple[int, bool]]:
        """Gather a status over the TP+CP scheduling group.

        Args:
            local_status: Caller-defined ``(state, flag)`` pair from this rank.
                The fill gate uses ``(ready, synchronous_progress)`` and the
                fail-fast path uses ``(all_fetched, terminal_no_fit)``.

        Returns:
            One status pair per TP+CP rank in the current pipeline-parallel
            slice. A singleton group returns ``[local_status]``.
        """
        # CP may coexist with TP; tp_cp_allgather covers both CP-only and
        # TP+CP configurations.
        if self._dist_size(self.dist, "cp_size") > 1:
            return self.dist.tp_cp_allgather(local_status)
        if self._dist_size(self.dist, "tp_size") > 1:
            return self.dist.tp_allgather(local_status)
        return [local_status]

    def _sync_disagg_gen_status_entry(self, local_need_check: bool) -> int:
        if self._dist_size(self.dist, "world_size") > 1:
            return self.dist.allreduce(int(local_need_check), op=ReduceOp.MAX)
        return int(local_need_check)

    def _sync_disagg_ctx_status_entry(self, local_need_check: bool) -> int:
        if self._dist_size(self.dist, "cp_size") > 1:
            return int(any(self.dist.tp_cp_allgather(int(local_need_check))))
        if self._dist_size(self.dist, "tp_size") > 1:
            return self.dist.tp_allreduce(int(local_need_check),
                                          op=ReduceOp.MAX)
        return int(local_need_check)

    def _check_disagg_transfer_progress_when_idle(
            self, num_fitting_reqs: int,
            fitting_disagg_gen_init_requests: List[LlmRequest],
            wait_for_disagg_gen_transfer_progress: bool,
            all_gen_first: bool) -> None:
        local_need_check = (num_fitting_reqs == 0
                            and not fitting_disagg_gen_init_requests)

        # A synchronous GEN receive is rank-local and blocking. One rank can
        # still be receiving while another is idle, so entering either the
        # generation or context progress collective here is unsafe.
        if not self._uses_async_disagg_gen_transfer():
            return

        local_need_gen_check = (local_need_check
                                and wait_for_disagg_gen_transfer_progress)

        any_need_gen_check = self._sync_disagg_gen_status_entry(
            local_need_gen_check)
        if any_need_gen_check > 0:
            if local_need_gen_check:
                logger.debug(
                    "Waiting for generation KV cache transfer progress to "
                    "free disagg admission budget")
            self._check_disagg_gen_cache_transfer_status(1)
            return

        any_need_check = self._sync_disagg_ctx_status_entry(local_need_check)
        if any_need_check > 0:
            if local_need_check and not all_gen_first:
                logger.warning(
                    "num_fitting_reqs=0 and fitting_disagg_gen_init_requests is empty, may not have enough kvCache"
                )
                # Local conditions warrant a blocking wait for at least one
                # in-flight transfer to complete so KV blocks can be freed.
                self._check_disagg_ctx_cache_transfer_status(1)
            else:
                # Either (a) a peer rank needed the call but we didn't, or
                # (b) all active requests are gen-first so we don't
                # actively block. In both cases the non-blocking variant
                # still runs the internal allgather (keeping all ranks in
                # sync) and reaps any already-completed transfers without
                # blocking on un-finished ones.
                self._check_disagg_ctx_cache_transfer_status(0)

    def _sync_gen_only_benchmark_has_insufficient_kv(
            self, scheduler_fitting_disagg_gen_init_requests: List[LlmRequest],
            wait_for_disagg_gen_transfer_progress: bool) -> bool:
        """Return whether benchmark fill has terminal KV exhaustion.

        Model-parallel ranks can make different local scheduling decisions.
        Every rank must therefore vote before entering the collective error-
        handling path. One terminal rank prevents the global benchmark fill
        gate from opening. The vote is fill-only to avoid adding a collective
        to every decode iteration after the gate opens.

        Args:
            scheduler_fitting_disagg_gen_init_requests: Generation INIT
                requests that fit KV capacity before transfer admission. A
                nonempty list means KV capacity exists even if transfer
                admission temporarily defers every request.
            wait_for_disagg_gen_transfer_progress: Whether active generation
                transfers are consuming the admission budget and transfer
                progress can unblock a deferred request.

        Returns:
            True when every TP+CP rank has fetched its full benchmark queue and
            at least one rank has an INIT request that cannot fit KV capacity
            and has no transfer progress that can unblock it; otherwise False.
        """
        if (self.benchmark_req_queues_size <= 0 or self.is_warmup
                or not self._benchmark_fill_phase_active):
            return False

        local_has_stuck = any(req.is_disagg_generation_init_state
                              for req in self.active_requests)
        local_all_fetched = (self.num_fetch_requests
                             >= self.benchmark_req_queues_size)
        local_terminal_no_fit = (local_has_stuck and
                                 not scheduler_fitting_disagg_gen_init_requests
                                 and not wait_for_disagg_gen_transfer_progress)
        local_status = (local_all_fetched, local_terminal_no_fit)

        all_rank_status = self._allgather_model_parallel_status(local_status)
        all_ranks_fetched = all(status[0] for status in all_rank_status)
        any_rank_terminal_no_fit = any(status[1] for status in all_rank_status)
        return all_ranks_fetched and any_rank_terminal_no_fit

    def _prepare_and_schedule_batch(self):
        self._sync_disagg_transfer_made_progress = False
        new_requests = self._fetch_and_activate_new_requests()
        if self.should_stop_processing:
            return None, None

        self._handle_control_request()

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

        self._prefetch_for_context_requests()

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
        elif (getattr(self, "model_engine", None) is not None
              and self.model_engine.is_spec_decode
              and self.max_total_draft_tokens > 0):
            # One-model speculative decoding (MTP / Eagle3 one-model) has no separate
            # drafter, so the block above is skipped -- but the model still verifies
            # max_total_draft_tokens draft tokens per generation step. The C++ micro-batch
            # scheduler budgets each gen request as beam_width + getNumDraftTokens(); without
            # populating the draft-token count here it under-reserves and, under chunked
            # prefill + overlap scheduler, the forward's uniform (1 + runtime_draft_len)
            # build overshoots max_num_tokens (total_num_tokens > max_num_tokens). Mirror the
            # two-model normalization so scheduling reserves the correct token budget.
            # model_engine is guarded first so partially-constructed executors in unit tests
            # (which may not set model_engine) do not raise AttributeError.
            for request in self.active_requests:
                if request.state not in (
                        LlmRequestState.GENERATION_IN_PROGRESS,
                        LlmRequestState.DISAGG_GENERATION_INIT):
                    continue
                request.draft_tokens = [0] * self.max_total_draft_tokens

        scheduled_batch, scheduler_fitting_disagg_gen_init_requests, num_fitting_reqs = self._schedule(
        )

        if self.drafter is not None and not self.use_spec_decode:
            for request in scheduled_batch.all_requests():
                request.py_disable_speculative_decoding = True

        if self.kv_cache_transceiver:
            wait_for_disagg_gen_transfer_progress = False
            admitted_disagg_gen_init_requests, wait_for_disagg_gen_transfer_progress = (
                self._apply_disagg_transfer_admission(
                    scheduler_fitting_disagg_gen_init_requests))
            # Prepare KV cache manager resources only for requests admitted
            # into the transfer window this iteration.
            self._prepare_disagg_gen_init(admitted_disagg_gen_init_requests)

            all_gen_first = self.active_requests and all(
                req.py_disaggregated_params and req.py_disaggregated_params.
                schedule_style == DisaggScheduleStyle.GENERATION_FIRST
                for req in self.active_requests)
            self._check_disagg_transfer_progress_when_idle(
                num_fitting_reqs, admitted_disagg_gen_init_requests,
                wait_for_disagg_gen_transfer_progress, all_gen_first)

            # In gen-only benchmark mode, all requests must fit in KV cache
            # simultaneously. If some requests are stuck in INIT state and the
            # scheduler could not allocate KV for any of them, the benchmark
            # will hang forever because in-progress generation requests won't
            # release their KV cache.
            # Check the scheduler result from before transfer admission. An
            # empty admitted list can mean that active transfers are
            # temporarily consuming the transfer budget.
            has_insufficient_kv = self._sync_gen_only_benchmark_has_insufficient_kv(
                scheduler_fitting_disagg_gen_init_requests,
                wait_for_disagg_gen_transfer_progress)
            if has_insufficient_kv:
                error_msg = (
                    f"Insufficient KV cache for gen-only benchmark mode: "
                    f"one or more requests are waiting for KV cache allocation "
                    f"on a model-parallel rank whose scheduler could not fit "
                    f"any of them. Increase free_gpu_memory_fraction or reduce "
                    f"TLLM_BENCHMARK_REQ_QUEUES_SIZE (currently "
                    f"{self.benchmark_req_queues_size}).")
                logger.error(error_msg)
                # Fail all active and waiting requests on every rank so every
                # client receives an error instead of hanging.
                self._handle_errors(error_msg, requests=self.active_requests)
                return None, None

        self.num_scheduled_requests = scheduled_batch.batch_size
        logger.debug(
            f'has {len(self.active_requests)} active_requests, '
            f'scheduled {scheduled_batch.num_encoder_requests} encoder requests, '
            f'{scheduled_batch.num_context_requests} context requests and '
            f'{scheduled_batch.num_generation_requests} generation requests')
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

    def _is_benchmark_disagg_fill_complete(
            self,
            scheduled_batch: ScheduledRequests,
            local_sync_progress: bool = False) -> bool:
        """State-based fill-complete predicate for benchmark disagg mode.

        The gate opens when all three conditions hold globally:

        (A) The executor has fetched at least ``benchmark_req_queues_size``
            requests cumulatively.
        (B) Every request in ``active_requests`` on this rank completed the
            KV-transfer phase (not in INIT, TRANS_IN_PROGRESS, or ERROR).
        (C) The KV cache transceiver has no pending receive sessions.

        The conditions and synchronous-progress signal are gathered across the
        TP+CP scheduling group so every model-parallel rank makes the same gate
        and sleep decision.

        This method must only be called when ``is_benchmark_disagg`` is True.

        Args:
            scheduled_batch: Passed for API compatibility with callers
                but no longer used by this predicate.
            local_sync_progress: Whether this rank completed a synchronous KV
                transfer in the current iteration.

        Returns:
            True when the fill phase is complete and the first forward
            pass can proceed.
        """
        if not self.is_benchmark_disagg:
            raise RuntimeError(
                "_is_benchmark_disagg_fill_complete() should not be called "
                "outside benchmark disagg mode.")

        # (A) All benchmark requests have been fetched from the queue. Keep
        # going to the shared allgather even when this rank is not done so
        # model-parallel ranks cannot diverge in collective order.
        local_all_fetched = (self.num_fetch_requests
                             >= self.benchmark_req_queues_size)
        if not local_all_fetched:
            if self.dist.rank == 0:
                logger.debug(
                    f"Benchmark disagg fill: fetching "
                    f"{self.num_fetch_requests}/{self.benchmark_req_queues_size}"
                )

        # (B) Every active request on this rank is past KV-transfer states.
        local_all_past_transfer = not any(
            req.is_disagg_generation_init_state
            or req.is_disagg_generation_transmission_in_progress
            or req.state == LlmRequestState.DISAGG_TRANS_ERROR
            for req in self.active_requests)

        # Also require the transceiver's async receive bookkeeping to be
        # drained before opening the fill gate.
        local_no_inflight = (
            self.kv_cache_transceiver is None
            or self.kv_cache_transceiver.check_gen_transfer_complete())

        local_ok = int(local_all_fetched and local_all_past_transfer
                       and local_no_inflight)
        local_status = (local_ok, bool(local_sync_progress))

        all_rank_status = self._allgather_model_parallel_status(local_status)
        all_ranks_ok = [status[0] for status in all_rank_status]
        global_ok = min(all_ranks_ok) == 1
        self._benchmark_sync_progress_global = any(
            status[1] for status in all_rank_status)

        if self.dist.rank == 0:
            if global_ok:
                logger.info(
                    f"Benchmark disagg fill complete: "
                    f"{len(self.active_requests)} active requests ready, "
                    f"gate opening.")
            else:
                blocked_ranks = [
                    rank for rank, ok in enumerate(all_ranks_ok) if not ok
                ]
                num_init = sum(1 for req in self.active_requests
                               if req.is_disagg_generation_init_state)
                num_in_progress = sum(
                    1 for req in self.active_requests
                    if req.is_disagg_generation_transmission_in_progress)
                num_errors = sum(
                    1 for req in self.active_requests
                    if req.state == LlmRequestState.DISAGG_TRANS_ERROR)
                logger.debug(
                    f"Benchmark disagg fill: blocked on ranks {blocked_ranks} "
                    f"(rank {self.dist.rank} local: {num_init} INIT, "
                    f"{num_in_progress} in-progress, {num_errors} errors, "
                    f"inflight={not local_no_inflight})")
        return global_ok

    def _check_benchmark_disagg_gate(self, scheduled_batch: ScheduledRequests,
                                     can_forward: bool) -> tuple[bool, bool]:
        """Gate the forward pass until all benchmark disagg requests are ready.

        In benchmark disagg mode the GEN executor must defer the forward
        pass until every request has completed KV transfer.  This helper
        consolidates the check used by both ``_executor_loop`` and
        ``_executor_loop_overlap``.

        A short sleep (0.1s) yields the CPU between retries that made no
        transfer progress while keeping the polling interval short enough to
        avoid KV transfer backpressure on the CTX server. Synchronous receives
        already block until they make progress, so those retries do not sleep.

        Args:
            scheduled_batch: The current scheduled batch.
            can_forward: Current gate state.

        Returns:
            ``(can_forward, should_retry)`` — when *should_retry* is True
            the caller should ``continue`` to the next loop iteration.
        """
        if not self.is_warmup and not can_forward:
            sync_transfer_made_progress = getattr(
                self, "_sync_disagg_transfer_made_progress", False)
            self._sync_disagg_transfer_made_progress = False
            can_forward = self._is_benchmark_disagg_fill_complete(
                scheduled_batch, sync_transfer_made_progress)
            sync_transfer_made_progress = self._benchmark_sync_progress_global
            if can_forward:
                self._benchmark_fill_phase_active = False
                self._fill_admit_cap = 0
            elif not sync_transfer_made_progress:
                time.sleep(0.1)
            if not can_forward:
                return can_forward, True
        return can_forward, False

    @nvtx_range("_handle_disagg_cache_errors_synced")
    def _handle_disagg_cache_errors_synced(self):
        """Rank-safe disagg cache error and poison handler.

        Called from the top of every executor iteration. Buffer poison is
        reduced over the full executor world because one poisoned PP/DP rank
        requires the whole distributed executor to stop. ADP TP ranks then
        vote on failed request IDs and fail matching local replicas together;
        otherwise the downstream ``tp_gather`` in ``_enqueue_responses``
        deadlocks or leaves peer replicas running.
        """
        if not self.kv_cache_transceiver:
            return

        if self._is_disagg_inflight_cancel_active():
            local_poisoned = self.kv_cache_transceiver.has_poisoned_transfer_buffer(
            )
            if self.dist.world_size != 1:
                any_poisoned = bool(
                    self.dist.allreduce(int(local_poisoned), op=ReduceOp.MAX))
            else:
                any_poisoned = local_poisoned
            if any_poisoned:
                error_msg = (
                    "Disagg KV cache transfer buffer is poisoned; process "
                    "restart is required")
                self._fatal_error = RuntimeError(f"Fatal error: {error_msg}")
                self.is_shutdown = True
                self._handle_errors(error_msg,
                                    requests=None,
                                    charge_budget=False)
                return

        if not (self.enable_attention_dp and self.dist.world_size != 1):
            return

        local_error_requests = [
            request for request in self.active_requests
            if request.state == LlmRequestState.DISAGG_TRANS_ERROR
        ]
        local_vote = {
            "error_ids": [
                self._request_vote_id(request)
                for request in local_error_requests
            ],
            "blocked_ids": [
                self._request_vote_id(request)
                for request in local_error_requests
                if self._is_disagg_error_cleanup_blocked(request)
            ],
        }
        all_votes = self.dist.tp_allgather(local_vote)
        voted_error_ids = {
            request_id
            for rank_vote in all_votes
            for request_id in rank_vote["error_ids"]
        }
        blocked_error_ids = {
            request_id
            for rank_vote in all_votes
            for request_id in rank_vote["blocked_ids"]
        }
        ready_error_ids = voted_error_ids - blocked_error_ids
        if not ready_error_ids:
            return
        local_voted_error_requests = [
            request for request in self.active_requests
            if self._request_vote_id(request) in ready_error_ids
        ]
        logger.warning(
            f"Disagg KV cache transfer error: rank={self.dist.rank} "
            f"local_err_count={len(local_error_requests)}, "
            f"voted_err_count={len(voted_error_ids)}, "
            f"blocked_err_count={len(voted_error_ids & blocked_error_ids)}")
        self._handle_errors(
            "Disagg KV cache transfer error",
            requests=local_voted_error_requests,
            charge_budget=False,
        )

    def _emit_initial_stats(self) -> None:
        """Emit a startup stats snapshot so that cache_config_info is
        immediately available to external metric scrapers (e.g. the
        Kubernetes Inference Gateway EPP) before any inference request."""
        if not self.enable_iter_perf_stats:
            return
        stats = self._get_init_iter_stats(0, 0)
        kv_cache_manager = self.resource_manager.resource_managers.get(
            ResourceManagerType.KV_CACHE_MANAGER)
        if kv_cache_manager is not None:
            kv_stats = kv_cache_manager.get_kv_cache_stats()
            kv_stats_to_save = KvCacheStats()
            kv_stats_to_save.max_num_blocks = kv_stats.max_num_blocks
            kv_stats_to_save.tokens_per_block = kv_stats.tokens_per_block
            kv_stats_to_save.free_num_blocks = kv_stats.free_num_blocks
            kv_stats_to_save.used_num_blocks = kv_stats.used_num_blocks
            stats.kv_cache_stats = kv_stats_to_save
        self._append_iter_stats(stats)

    def _executor_loop(self):
        torch.cuda.set_device(self.device_id)
        # ensure the context is created, otherwise, some MPI calls will fail.
        CUASSERT(cudart.cudaSetDevice(self.device_id))
        with self._profiler() as profile_step, self.hang_detector:
            sample_state = None
            iter_start_time = time.time()
            iter_stats = None
            can_forward = not self.is_benchmark_disagg
            while True:
                self.hang_detector.checkpoint()
                profile_step()
                if self.enable_iter_perf_stats:
                    iter_start_time = time.time()

                if self._resource_governor_enabled:
                    self._sync_and_process_resource_governor_queue()

                if self._is_kv_manager_v2 and self._can_pause_for_rebalance():
                    self._maybe_rebalance_kv_pools()

                self._handle_disagg_cache_errors_synced()

                scheduled_batch, iter_stats = self._prepare_and_schedule_batch()

                if scheduled_batch is None:
                    break

                can_forward, should_retry = self._check_benchmark_disagg_gate(
                    scheduled_batch, can_forward)
                if should_retry:
                    if self._is_kv_manager_v2:
                        for req in scheduled_batch.generation_requests:
                            self.kv_cache_manager.revert_allocate_generation(
                                req)
                    self._finalize_adp_dummy_allocation(False)
                    continue

                if not self._is_kv_manager_v2:
                    self._terminate_requests(scheduled_batch.paused_requests)
                    self._pause_requests(scheduled_batch.paused_requests)

                finished_requests = []
                sample_state = None
                scheduled_batch_stats = None
                gpu_forward_start = None
                gpu_forward_end = None
                gpu_forward_events_from_perf_pool = False

                # Run the encoder iteration first.  After scatter the
                # encoder requests transition to ``CONTEXT_INIT`` and are
                # picked up by the next scheduler iteration as decoder
                # context. The encoder pass is independent of the decoder
                # ``can_queue`` gate, so an iteration with only encoder-init
                # requests still makes forward progress.
                if scheduled_batch.encoder_requests:
                    self._run_encoder_step(scheduled_batch.encoder_requests)

                can_queue, _ = self._can_queue(scheduled_batch)

                if can_queue:
                    if self.kv_cache_transceiver:
                        # For generation requests which have completed KV cache transfer
                        self._prepare_disagg_gen_transmission_complete(
                            scheduled_batch)

                        # Return the first token to the client
                        self._handle_first_token_response(scheduled_batch)

                    self._handle_dynamic_draft_len(scheduled_batch)

                    self.resource_manager.prepare_resources(scheduled_batch)

                if self.kv_connector_manager:
                    self.kv_connector_manager.handle_metadata()

                if can_queue:
                    self._kv_connector_start_batch(scheduled_batch)

                # if using a kv connector, we need to call can_queue again since scheduled_batch might have changed
                if self.kv_connector_manager:
                    can_queue, _ = self._can_queue(scheduled_batch)

                if not can_queue:
                    self._revert_gen_alloc(scheduled_batch)
                self._finalize_adp_dummy_allocation(can_queue)

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
                            self.execution_stream.wait_stream(
                                torch.cuda.current_stream())
                            with torch.cuda.stream(self.execution_stream):
                                self.drafter.prepare_draft_tokens(
                                    scheduled_batch, self.resource_manager)
                                # Pad draft tokens to the max draft length and extend KV cache
                                # capacity to match. This is for CUDA graph compatibility.
                                self.drafter.pad_draft_tokens_for_cuda_graph(
                                    scheduled_batch, self.resource_manager)
                            torch.cuda.current_stream().wait_stream(
                                self.execution_stream)
                        # add_batch must be called again to restore to target requests with updated draft tokens.
                        if self.guided_decoder is not None:
                            self.guided_decoder.add_batch(scheduled_batch)
                            if hasattr(self.drafter, "guided_decoder"):
                                self.guided_decoder.rollback_draft_tokens()

                    scheduled_batch_stats = (
                        self._collect_scheduled_batch_stats(scheduled_batch)
                        if self.enable_iter_perf_stats else None)
                    self._commit_kv_cache_stats(scheduled_batch)

                    # GPU and CPU timing for perf metrics
                    gpu_forward_start, gpu_forward_end, gpu_sample_end = self.perf_manager.create_timing_events(
                    )
                    if self.enable_iter_perf_stats and gpu_forward_start is None:
                        gpu_forward_start, gpu_forward_end = self.perf_manager.borrow_forward_timing_events(
                        )
                        gpu_forward_events_from_perf_pool = True

                    with self.perf_manager.record_perf_events(
                            gpu_forward_start, gpu_forward_end) as fwd_timing:
                        if self.dwdp_manager is not None:
                            self.dwdp_manager.prefetch_first_layers()
                        batch_outputs = self._forward_step(scheduled_batch)

                    self._maybe_prefetch_next_iter_mm_encoders(scheduled_batch)

                    guided_decoder_failed_requests = None
                    if self.guided_decoder is not None:
                        guided_decoder_failed_requests = self.guided_decoder.execute(
                            batch_outputs['logits'])

                    with self.perf_manager.record_perf_events(
                            None, gpu_sample_end) as sample_timing:
                        sample_state = self._sample_async(
                            scheduled_batch, batch_outputs)

                    if self.perf_manager.enabled:
                        self.perf_manager.save_timing_to_requests(
                            scheduled_batch.all_requests(), gpu_forward_start,
                            gpu_forward_end, gpu_sample_end,
                            fwd_timing.start_time, fwd_timing.end_time,
                            sample_timing.start_time, sample_timing.end_time)

                    # Handle guided decoder errors after _sample_async to avoid state conflicts.
                    # If called before, failed requests would be marked as GENERATION_COMPLETE,
                    # causing _sample_async to fail when accessing context_chunk_size property.
                    self._handle_guided_decoder_errors(
                        scheduled_batch, guided_decoder_failed_requests)

                    # Handle SaveHiddenStates mode - save hidden states after forward
                    if not self.is_warmup:
                        spec_resource_mgr = self.resource_manager.resource_managers.get(
                            ResourceManagerType.SPEC_RESOURCE_MANAGER)
                        if spec_resource_mgr is not None and hasattr(
                                spec_resource_mgr, 'process_and_save'):
                            spec_metadata = getattr(self.model_engine,
                                                    'spec_metadata', None)
                            spec_resource_mgr.process_and_save(
                                scheduled_batch, spec_metadata)

                    self._update_request_states(scheduled_batch)
                    self._update_requests(sample_state, self.resource_manager)
                    if self.speculation_gate is not None:
                        self._update_batch_acceptance_rate(
                            scheduled_batch,
                            sample_state,
                            iteration_id=self.iter_counter)

                    if self._is_kv_manager_v2:
                        # Finalize V2 context KV before disagg transfer/response
                        # handling can terminate the request.
                        self.kv_cache_manager.update_context_resources(
                            scheduled_batch)
                    self._send_kv_async(scheduled_batch.all_requests())
                    self._flush_pending_transfer_responses()

                    self._handle_canceled_requests()
                    finished_requests = self._handle_responses()
                    # Complete ctx send sessions AFTER responses are created so
                    # _handle_responses sees the request before it is terminated.
                    if self.kv_cache_transceiver:
                        self._check_disagg_ctx_cache_transfer_status(0)
                    # Compute GPU times after _handle_responses creates metric entries
                    # (safe in non-overlap mode: no next iteration to overwrite events)
                    self.perf_manager.compute_batch_gpu_times(
                        scheduled_batch.all_requests())
                    attn_metadata = getattr(self.model_engine, 'attn_metadata',
                                            None)
                    kv_cache_dtype_byte_size = getattr(
                        self.model_engine, 'kv_cache_dtype_byte_size', None)
                    self.resource_manager.update_resources(
                        scheduled_batch, attn_metadata,
                        kv_cache_dtype_byte_size)
                    if self.enable_kv_cache_events:
                        self._add_kv_cache_events()

                # Drain timeout buffer outside ``if can_queue`` so the synced
                # collective fires every iter regardless of future restructuring.
                self._handle_kv_transfer_timeouts_synced()

                if self.kv_cache_transceiver and self.async_transfer_manager.has_any_inflight_requests(
                ):
                    self._check_kv_transfer_timeout()

                self._kv_connector_terminate_requests()

                if self.enable_iter_perf_stats and sample_state is not None:
                    self._process_iter_stats(
                        finished_requests, self.active_requests,
                        BatchState(scheduled_requests=scheduled_batch,
                                   sample_state=sample_state,
                                   iter_stats=iter_stats,
                                   iter_start_time=iter_start_time,
                                   scheduled_batch_stats=scheduled_batch_stats,
                                   gpu_forward_start_event=gpu_forward_start,
                                   gpu_forward_end_event=gpu_forward_end,
                                   gpu_forward_events_from_perf_pool=
                                   gpu_forward_events_from_perf_pool))
                elif gpu_forward_events_from_perf_pool:
                    self.perf_manager.release_forward_timing_events(
                        gpu_forward_start, gpu_forward_end)
                # Same lockstep guarantee for iter-stats; no-op when
                # TLLM_METRICS_ALL_RANKS=0.
                self._flush_iter_stats_synced()

                self.iter_counter += 1

    def _prepare_draft_requests(self):
        try:
            # Set draft tokens here to make the KV cache manager
            # and scheduler aware of them.
            for req in self.active_requests:
                if req.state not in (LlmRequestState.GENERATION_IN_PROGRESS,
                                     LlmRequestState.DISAGG_GENERATION_INIT):
                    continue

                # Skip DISAGG_GENERATION_INIT: snapshotting the dummy py_draft_tokens leaks stale state into the first real draft.
                if req.state == LlmRequestState.GENERATION_IN_PROGRESS:
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
        """Fire the next pending control action at the next step boundary.

        drain=True  (default): wait until ``active_requests`` and
            ``waiting_queue`` are empty — exclusive engine access.
        drain=False: as soon as a fetched batch contains a control request,
            fire the action first, then continue forwarding requests.
        """
        # Return when control_requests is an empty list, or absent (some
        # unit tests build a mock PyExecutor that never sets the attribute).
        if not getattr(self, "control_requests", None):
            return

        assert len(self.control_requests) == 1, (
            f"Expected exactly one control request to be processed at a time, "
            f"but found {len(self.control_requests)} control requests. "
            f"This may indicate a race condition or improper control request handling."
        )

        pending = self.control_requests[0]

        if pending.control_requires_drain and (len(self.active_requests) != 0
                                               or len(self.waiting_queue) != 0):
            # drain=True: keep the sentinel parked until the engine drains.
            return

        logger.debug(f"[control_action] firing control request "
                     f"drain={pending.control_requires_drain} "
                     f"active_requests={len(self.active_requests)} "
                     f"waiting_queue={len(self.waiting_queue)}")
        # Quiesce the device before the action mutates GPU state. Under the
        # overlap scheduler a previous batch's forward/sample kernels may still
        # be in flight, so an in-place update_weights reload (or sleep/wakeup
        # freeing memory) could race with kernels still reading those tensors.
        torch.cuda.synchronize()
        self.control_requests.pop(0)
        control_id = getattr(pending, "control_id", None)
        self._active_control_id = control_id
        pending_abort = self._pop_sleep_wakeup_abort(control_id)
        if pending_abort is not None:
            logger.warning(
                "[control_action] skipping aborted control request %s: %s",
                control_id,
                pending_abort,
            )
            self.control_request_barrier.set()
            self.control_request_barrier.clear()
            self._active_control_id = None
            return
        self.control_request_barrier.set()
        self.control_action_done.wait()
        self.control_action_done.clear()
        self._active_control_id = None
        logger.debug("[control_action] control request finished")

    def _sync_and_process_resource_governor_queue(self):
        """Synchronize and process resource governor requests across all ranks.

        Only called when ``_resource_governor_enabled`` is `True`.
        Uses a two-phase broadcast: first broadcast the count (a single int),
        then broadcast the actual requests only when count > 0.  This avoids
        serializing and deserializing an empty Python list on every iteration.
        """
        if self.dist.rank == 0:
            if self._resource_governor_queue is not None:
                resource_governor_requests = self._resource_governor_queue.drain(
                )
            else:
                resource_governor_requests = []
            count = len(resource_governor_requests)
        else:
            resource_governor_requests = None
            count = 0

        count = self.dist.broadcast(count, root=0)
        if count == 0:
            return

        resource_governor_requests = self.dist.broadcast(
            resource_governor_requests, root=0)

        for request in resource_governor_requests:
            if isinstance(request, TruncateKVCacheRequest):
                self.kv_cache_manager.truncate_blocks(
                    request.messages, len(request.messages_to_retain))
            else:
                raise ValueError(f"Invalid request type: {type(request)}.")

    def _can_pause_for_rebalance(self) -> bool:
        """Gate KV pool rebalance to the cases the v1 hook supports.

        MVP scope: single-GPU aggregated, no in-flight disagg transfer,
        no beam search, no drafter, not during warmup or shutdown.
        Honors the ``enable_kv_pool_rebalance`` opt-in flag (default off).
        """
        if not self.enable_kv_pool_rebalance:
            return False
        if self.dist.pp_size > 1:
            return False
        if self.kv_cache_transceiver is not None:
            return False
        if self.is_warmup:
            return False
        if self.is_shutdown:
            return False
        if self.kv_cache_manager.max_beam_width > 1:
            return False
        if self.drafter is not None:
            return False
        return True

    def _consume_previous_batch_for_rebalance(self) -> None:
        """Drain ``previous_batch`` so its _KVCache instances are quiescent.

        No-op when ``previous_batch is None`` -- i.e., always a no-op in
        the non-overlap loop, since that loop never sets previous_batch.
        In the overlap loop this fires when the rebalance hook catches a
        pending in-flight iteration; we consume it inline so suspend can
        safely run.

        Mirrors the inline sequence in ``_executor_loop_overlap`` that
        handles ``previous_batch``.  Unlike the inline code we are not
        guarded by ``should_process_previous_batch``: the rebalance gate
        already excludes the multi-rank-divergence cases that flag exists
        to handle.
        """
        if self.previous_batch is None:
            return
        self._update_requests(self.previous_batch.sample_state)
        self._send_kv_async(
            self.previous_batch.scheduled_requests.all_requests())
        self._flush_pending_transfer_responses()
        self._process_previous_batch()
        self.perf_manager.compute_batch_gpu_times(
            self.previous_batch.scheduled_requests.all_requests())
        self.previous_batch = None

    def _maybe_rebalance_kv_pools(self) -> None:
        """Rebalance KV pool ratios when the V2 auto-tuner asks for it.

        Fast path: ``need_adjustment`` checks the sample counter and the
        120s cooldown before doing any real work.  On the slow path we
        drain pending GPU work, consume any in-flight ``previous_batch``
        (overlap loop only), suspend every active request, call
        ``adjust()``, and resume.  Resume failures stay suspended; the
        scheduler reactivates them through prepare_context /
        try_allocate_generation on the next iteration, the same path it
        uses today after eviction.
        """
        mgr = self.kv_cache_manager
        if not mgr.impl.need_adjustment:
            return

        torch.cuda.current_stream().synchronize()
        self._consume_previous_batch_for_rebalance()

        paused: List[LlmRequest] = []
        for req in self.active_requests:
            if mgr.is_request_active(req.py_request_id):
                mgr.suspend_request(req)
                paused.append(req)

        try:
            mgr.impl.adjust()
        except OutOfPagesError as e:
            logger.warning(f"KV pool adjust() failed: {e!r}")

        for req in paused:
            mgr.resume_request(req)

    @contextmanager
    def control_action(self,
                       *,
                       drain: bool = True,
                       control_id: Optional[str] = None):
        """Run an action at a scheduler step boundary.

        drain=True  (default): block until ``active_requests`` and
                               ``waiting_queue`` are empty before yielding.
        drain=False: yield at the next step boundary without draining.
                     In-flight requests keep their KV caches across the
                     action; same-batch requests fetched after the sentinel
                     are parked until the ``with`` block exits.
        """

        if self.dist.rank == 0:
            self.executor_request_queue.enqueue_control_request(
                drain=drain, control_id=control_id)

        self.control_request_barrier.wait()

        try:
            yield self
        finally:
            self.control_action_done.set()
            self.control_request_barrier.clear()

    def _wait_for_model_engine_input_copy(self):
        wait_for_input_copy = getattr(self.model_engine, "wait_for_input_copy",
                                      None)
        if wait_for_input_copy is not None:
            wait_for_input_copy()

    def _executor_loop_overlap(self):
        torch.cuda.set_device(self.device_id)
        # ensure the context is created, otherwise, some MPI calls will fail.
        CUASSERT(cudart.cudaSetDevice(self.device_id))
        with self._profiler() as profile_step, self.hang_detector:
            iter_start_time = time.time()
            iter_stats = None
            target_inputs = None
            previous_tensors_device = None
            can_forward = not self.is_benchmark_disagg
            while True:
                self.hang_detector.checkpoint()
                profile_step()
                if self.enable_iter_perf_stats:
                    iter_start_time = time.time()

                if self._resource_governor_enabled:
                    self._sync_and_process_resource_governor_queue()

                if self._is_kv_manager_v2 and self._can_pause_for_rebalance():
                    self._maybe_rebalance_kv_pools()

                self._handle_disagg_cache_errors_synced()

                # Need to wait for the copy of previous iteration before
                # modifying any host memory copied to GPU. Scheduler V2
                # modifies the host page table, so wait before scheduling.
                # This wait is also needed for legacy scheduler, but it can
                # be pushed later, e.g. before model_engine._prepare_inputs().
                self._wait_for_model_engine_input_copy()
                scheduled_batch, iter_stats = self._prepare_and_schedule_batch()

                if scheduled_batch is None:
                    break

                can_forward, should_retry = self._check_benchmark_disagg_gate(
                    scheduled_batch, can_forward)
                if should_retry:
                    if self._is_kv_manager_v2:
                        for req in scheduled_batch.generation_requests:
                            self.kv_cache_manager.revert_allocate_generation(
                                req)
                    self._finalize_adp_dummy_allocation(False)
                    continue

                if not self._is_kv_manager_v2:
                    self._terminate_requests(scheduled_batch.paused_requests)

                if scheduled_batch.encoder_requests:
                    self._run_encoder_step(scheduled_batch.encoder_requests)

                gpu_forward_events_from_perf_pool = False
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
                        self.model_engine.enable_spec_decode = scheduled_batch.num_context_requests > 0
                        if not self.model_engine.enable_spec_decode:
                            for request in scheduled_batch.all_requests():
                                request.py_draft_tokens = []

                    self._handle_dynamic_draft_len(scheduled_batch)

                    self.resource_manager.prepare_resources(scheduled_batch)

                if self.kv_connector_manager:
                    self.kv_connector_manager.handle_metadata()

                if can_queue:
                    self._kv_connector_start_batch(scheduled_batch)

                # if using a kv connector, we need to call can_queue again since scheduled_batch might have changed
                if self.kv_connector_manager:
                    can_queue, can_queue_this_rank = self._can_queue(
                        scheduled_batch)

                if not can_queue:
                    self._revert_gen_alloc(scheduled_batch)
                self._finalize_adp_dummy_allocation(can_queue)

                # If the batch is not empty on this rank, but empty on other ranks,
                # we need to delay the update of the previous batch's sample state,
                # and let the later iteration to update it.
                should_process_previous_batch = can_queue or not can_queue_this_rank
                if can_queue:

                    # The generation requests that do not have batch_idx
                    # need to be in front of the batch due to the assumptions
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
                        self.execution_stream.wait_stream(
                            torch.cuda.current_stream())
                        with torch.cuda.stream(self.execution_stream):
                            target_inputs, num_accepted_tokens_device = self._handle_speculative_decoding(
                                scheduled_batch, previous_tensors,
                                previous_tensors_device)
                        torch.cuda.current_stream().wait_stream(
                            self.execution_stream)

                    # Use the draft_model's outputs if we've launched the draft model.
                    # Otherwise, use the previous batch's outputs.
                    if (target_inputs is not None
                            and target_inputs.next_draft_tokens
                            is not None) or use_previous_draft_tokens:
                        previous_tensors_device = target_inputs
                    else:
                        previous_tensors_device = self.previous_batch and self.previous_batch.sample_state and self.previous_batch.sample_state.device

                    scheduled_batch_stats = (
                        self._collect_scheduled_batch_stats(scheduled_batch)
                        if self.enable_iter_perf_stats else None)

                    # GPU timing for perf metrics
                    gpu_forward_start, gpu_forward_end, gpu_sample_end = self.perf_manager.create_timing_events(
                    )
                    if self.enable_iter_perf_stats and gpu_forward_start is None:
                        gpu_forward_start, gpu_forward_end = self.perf_manager.borrow_forward_timing_events(
                        )
                        gpu_forward_events_from_perf_pool = True

                    with self.perf_manager.record_perf_events(
                            gpu_forward_start, gpu_forward_end) as fwd_timing:
                        batch_outputs = self._forward_step(
                            scheduled_batch, previous_tensors_device,
                            num_accepted_tokens_device)

                    self._maybe_prefetch_next_iter_mm_encoders(scheduled_batch)

                if self.previous_batch is not None and should_process_previous_batch:
                    self._update_requests(self.previous_batch.sample_state)
                    # Turning off speculative decoding when Acceptance Rate is low.
                    # In overlap scheduler path, it will do an extra iter with spec decode on.
                    if self.speculation_gate is not None:
                        self._update_batch_acceptance_rate(
                            self.previous_batch.scheduled_requests,
                            self.previous_batch.sample_state,
                            iteration_id=self.iter_counter)

                    self._send_kv_async(
                        self.previous_batch.scheduled_requests.all_requests())

                if self.enable_early_first_token_response:
                    if self.previous_batch is not None and should_process_previous_batch:
                        # Early first-token emission. Must run after
                        # `_update_requests` (so `py_decoding_iter` is current)
                        # and `_send_kv_async` (so disagg ctx state has advanced).
                        self._emit_first_token_responses(
                            self.previous_batch.scheduled_requests)
                    else:
                        # Pair the attention-DP gather invoked by
                        # `_emit_first_token_responses` on the active branch.
                        self._enqueue_responses([])

                # Flush outside the conditional so that all DP ranks
                # participate in the tp_gather collective even when
                # should_process_previous_batch differs between ranks.
                self._flush_pending_transfer_responses()

                if self.drafter is not None and self.use_spec_decode and should_process_previous_batch:
                    # Cleanup previous draft resources used in the draft model
                    self.drafter.cleanup_previous_draft_resources()

                if not self._is_kv_manager_v2:
                    self._pause_requests(scheduled_batch.paused_requests)

                if can_queue:
                    guided_decoder_failed_requests = None
                    with self.perf_manager.record_perf_events(
                            None, gpu_sample_end) as sample_timing:
                        if self.guided_decoder is not None:
                            # add_batch must be called again to have updated new tokens.
                            self.guided_decoder.add_batch(scheduled_batch)
                            guided_decoder_failed_requests = self.guided_decoder.execute(
                                batch_outputs['logits'])

                        sample_state = self._sample_async(
                            scheduled_batch, batch_outputs)

                    assert sample_state is not None, "Sampling failed"

                    # Handle guided decoder errors after _sample_async to avoid state conflicts.
                    # If called before, failed requests would be marked as GENERATION_COMPLETE,
                    # causing _sample_async to fail when accessing context_chunk_size property.
                    self._handle_guided_decoder_errors(
                        scheduled_batch, guided_decoder_failed_requests)
                    # _update_request_states() can terminate attention-DP
                    # dummy requests, which frees V2 KV pages and overwrites
                    # host page-index entries with BAD_PAGE_INDEX. Wait until
                    # the current input preparation has consumed those buffers.
                    self._wait_for_model_engine_input_copy()
                    self._update_request_states(scheduled_batch)

                    # Update context requests' KV cache so that sliding-window
                    # blocks freed by this chunk are visible to the next
                    # iteration's scheduler.
                    # Only applies to KV cache manager V2 + scheduler V2.
                    if (self._is_kv_manager_v2
                            and scheduled_batch.context_requests):
                        self.kv_cache_manager.update_context_resources(
                            scheduled_batch)

                if self.previous_batch is not None and should_process_previous_batch:
                    self._commit_kv_cache_stats(
                        self.previous_batch.scheduled_requests)
                    # _process_previous_batch may terminate requests or resize
                    # generation KV caches, both of which can mutate V2 page
                    # indices used by the current batch's input preparation.
                    self._wait_for_model_engine_input_copy()
                    self._process_previous_batch()
                    self.perf_manager.compute_batch_gpu_times(
                        self.previous_batch.scheduled_requests.all_requests())
                else:
                    self._enqueue_responses([])

                # Drain buffers from the (per-rank-divergent)
                # _process_previous_batch above; rank-symmetric companion to
                # the _enqueue_responses([]) call in the else branch.
                self._handle_kv_transfer_timeouts_synced()
                self._flush_iter_stats_synced()

                # Call set_exclude_last_generation_logits after _process_previous_batch.
                # If set before, the response of a request may be incorrect, as it will
                # use the wrong indices for generation logits when streaming is enabled.
                if can_queue:
                    self._update_generation_requests_that_will_complete_next_iteration(
                        scheduled_batch.generation_requests)

                if can_queue:
                    if self.perf_manager.enabled:
                        self.perf_manager.save_timing_to_requests(
                            scheduled_batch.all_requests(), gpu_forward_start,
                            gpu_forward_end, gpu_sample_end,
                            fwd_timing.start_time, fwd_timing.end_time,
                            sample_timing.start_time, sample_timing.end_time)

                    self.previous_batch = BatchState(
                        scheduled_requests=scheduled_batch,
                        sample_state=sample_state,
                        iter_start_time=iter_start_time,
                        iter_stats=iter_stats,
                        scheduled_batch_stats=scheduled_batch_stats,
                        gpu_forward_start_event=gpu_forward_start,
                        gpu_forward_end_event=gpu_forward_end,
                        gpu_forward_events_from_perf_pool=
                        gpu_forward_events_from_perf_pool)
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
    ) -> Tuple[SampleStateTensorsSpec, Optional[torch.Tensor]]:
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
            - SampleStateTensorsSpec with new_tokens set to accepted tokens,
              new_tokens_lens and next_draft_tokens set to None
            - num_accepted_tokens: [batch_size] tensor with acceptance counts per request,
              or None if no draft tokens
        """
        has_draft_tokens = target_inputs is not None and isinstance(
            target_inputs, SampleStateTensorsSpec
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
            num_contexts = scheduled_batch.num_context_requests
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

        # Create the updated SampleStateTensorsSpec
        # new_tokens_lens and next_draft_tokens are left as None
        result_tensors = SampleStateTensorsSpec(
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
        # Skip iter-1 emission when `_emit_first_token_responses` already
        # handled it.
        finished_requests = self._handle_responses(
            emit_first_iter=not self.enable_early_first_token_response)
        scheduled_requests = self.previous_batch.scheduled_requests
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

    def _forward_step_inter_pp(self,
                               scheduled_batch,
                               gpu_forward_start=None,
                               gpu_forward_end=None) -> SampleState:
        with self.perf_manager.record_perf_events(gpu_forward_start,
                                                  gpu_forward_end):
            self._forward_step(scheduled_batch)
        sampler_event = torch.cuda.Event()
        sampler_event.record()
        self._update_request_states(scheduled_batch)
        sampling_requests = scheduled_batch.context_requests_last_chunk + scheduled_batch.generation_requests
        return self.sampler.SampleState(
            requests=sampling_requests,
            sampler_event=SamplerEvent(cuda_event=sampler_event),
            runtime_draft_len=self.model_engine.runtime_draft_len,
        )

    def _validate_token_id_range(self, request: LlmRequest) -> None:
        if isinstance(self.model_engine.model, DecoderModelForCausalLM):
            # Only skip token‐range checks for Llama4 when the request has multimodal data
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
        self._validate_token_id_range(request)

        # Perform sampler-specific validation
        self.sampler.validate_request(request)

    def _fetch_and_enqueue_requests(self, waiting_queue: WaitingQueue,
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

        # Broadcast requests and handle Python objects. RequestBroadcaster probes
        # the request count first and can skip the heavy payload broadcast on
        # empty iterations.
        new_requests, py_request_objects = self.request_broadcaster.broadcast(
            new_requests)

        # Validate and filter requests
        new_requests = self._handle_special_queue_items(new_requests)

        # Attach Python objects to requests
        if py_request_objects and (self.dist.tp_size > 1 or self.dist.has_pp
                                   or self.dist.cp_size
                                   > 1) and self.dist.rank > 0:
            attach_py_objects_to_requests(new_requests, py_request_objects)

        waiting_queue.add_requests(new_requests)

    def _pop_from_waiting_queue(
        self,
        waiting_queue: WaitingQueue,
        total_num_active_requests: int,
        all_ranks_num_active_requests: Optional[List[int]] = None
    ) -> List[RequestQueueItem]:
        """Pop requests from waiting_queue based on available capacity."""
        if self.enable_attention_dp:
            total_max = self.dist.tp_size * self.max_num_active_requests
        else:
            total_max = self.max_num_active_requests

        max_new_requests = total_max - total_num_active_requests

        # Benchmark disagg fill-phase admission throttle (slow-start ramp).
        if (self.is_benchmark_disagg and self._benchmark_fill_phase_active
                and not self.is_warmup):
            if self._fill_admit_cap == 0:
                self._fill_admit_cap = self.dist.tp_size
            else:
                self._fill_admit_cap = min(self._fill_admit_cap * 2, total_max)
            max_new_requests = min(max_new_requests, self._fill_admit_cap)

        return get_from_waiting_queue(
            waiting_queue,
            max_new_requests,
            enable_attention_dp=self.enable_attention_dp,
            max_num_active_requests=self.max_num_active_requests,
            all_ranks_num_active_requests=all_ranks_num_active_requests)

    @nvtx_range("_fetch_new_requests")
    def _fetch_new_requests(
            self, waiting_queue: WaitingQueue,
            active_requests: List[LlmRequest]) -> List[LlmRequest]:
        """Fetch new requests and return LlmRequests ready for execution."""
        # 1. Gather rank states and calculate total_num_active_requests
        if self.enable_attention_dp:
            # NOTE: gather_all_rank_states is called here (before step 3)
            # because _pop_from_waiting_queue needs all_ranks_num_active_requests
            # from the allgather result. Moving it to step 5 would require an
            # extra allgather. When introducing new router implementations
            # (e.g. KV-cache-aware) that need new_requests to gather additional
            # info, the allgather position may need to be revisited.

            # The rank-state allgather is always required for ADP routing.
            # When iteration stats are enabled, piggyback this rank's oldest
            # pending payload and use the gathered states below to fan out or
            # clear stats once every rank is aligned.
            iter_stats_payload = (self._adp_iter_stats.next_payload()
                                  if self.enable_iter_perf_stats else None)
            all_rank_states = self.adp_router.gather_all_rank_states(
                active_requests, iter_stats_payload=iter_stats_payload)
            if self.enable_iter_perf_stats:
                for record in self._adp_iter_stats.finalize(
                        all_rank_states, is_rank0=self.dist.rank == 0):
                    self._append_iter_stats(
                        record.stats,
                        record.req_stats,
                        kv_iter_stats=record.kv_iter_stats,
                        attention_dp_rank=record.attention_dp_rank,
                        host_step_time_ms=record.host_step_time_ms,
                        prev_device_step_time_ms=record.
                        prev_device_step_time_ms,
                        gpu_forward_time_ms=record.gpu_forward_time_ms)
            all_ranks_num_active_requests = [
                s.num_active_requests for s in all_rank_states
            ]
            total_num_active_requests = sum(all_ranks_num_active_requests)
        else:
            total_num_active_requests = len(active_requests)
            all_ranks_num_active_requests = None
            all_rank_states = None

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

        # 5. Update total fetch counter (used by benchmark disagg gating)
        self.num_fetch_requests += len(new_requests)

        # 6. Schedule requests across ranks (DP only)
        if self.enable_attention_dp:
            # Symmetric skip — after _pop_from_waiting_queue all ranks see identical new_requests.
            if self.adp_router.needs_prefix_matches and new_requests:
                self.adp_router.gather_prefix_matches(new_requests)

            all_ranks_new_requests, self.expected_num_active_requests = \
                self.adp_router.route_requests(
                    all_rank_states, new_requests,
                    self.max_num_active_requests)
            new_requests_cur_rank = all_ranks_new_requests[self.dist.tp_rank]

            all_new_flat = [
                req for reqs in all_ranks_new_requests.values() for req in reqs
            ]
            self._update_adp_dummy_role(all_new_flat)

            # Update per-rank counter for DP
            self.num_fetch_requests_cur_rank += len(new_requests_cur_rank)

            new_requests = new_requests_cur_rank

        # 7. Merge requests
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
                self._handle_errors(str(e),
                                    requests=[request],
                                    charge_budget=False)
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

    @staticmethod
    def _compute_scheduled_tokens(context_requests, generation_requests):
        """Compute the total number of scheduled tokens for batch waiting decisions.

        For context requests, we estimate the actual compute tokens for this
        iteration (excluding tokens served from KV cache).

        For generation requests, each contributes 1 + num_draft_tokens.

        Note on reusable token handling:
        estimated_reusable_tokens is an absolute count from position 0.
        Depending on the scheduler, context_current_position may or may not
        have been advanced past the reusable prefix by the time this method
        is called:
        - V1 scheduler: prepare_context runs after scheduling, so
          context_current_position is still 0.
        - V2 scheduler: prepare_context runs during scheduling, so
          context_current_position is already advanced to the reused offset.
        To handle both correctly, the reusable credit applied to the current
        chunk is max(0, reusable - context_current_position), i.e. only the
        portion of the reusable range that falls within this chunk's span.
        """
        num_scheduled_ctx_tokens = 0
        for ctx_req in context_requests:
            reusable = (ctx_req.estimated_reusable_tokens
                        if ctx_req.is_first_context_chunk else 0)
            # Credit only the reusable tokens that overlap with the current
            # chunk: if context_current_position has already been advanced past
            # the reusable prefix (V2), the credit is 0; if not (V1), the full
            # reusable count is subtracted.
            reusable_in_chunk = max(0,
                                    reusable - ctx_req.context_current_position)
            remaining = ctx_req.context_remaining_length
            if reusable_in_chunk <= 0:
                compute = ctx_req.context_chunk_size
            elif reusable_in_chunk + ctx_req.context_chunk_size < remaining:
                compute = ctx_req.context_chunk_size
            else:
                compute = max(1, remaining - reusable_in_chunk)
            num_scheduled_ctx_tokens += compute
        num_scheduled_gen_tokens = sum(1 + gen_req.num_draft_tokens
                                       for gen_req in generation_requests)
        return num_scheduled_ctx_tokens + num_scheduled_gen_tokens

    def _waiting_requests(self, context_requests: list[LlmRequest],
                          generation_requests: list[LlmRequest]):
        """
        Return an empty list if scheduled requests fulfill the waiting conditions, otherwise return the original context requests.
        Waiting conditions:
        - The number of scheduled tokens (both context and generation) is smaller than `self.batch_wait_max_tokens_ratio * self.max_num_tokens`
        - The number of waiting iterations is smaller than `self.batch_wait_timeout_iters`.
        """

        num_scheduled_tokens = self._compute_scheduled_tokens(
            context_requests, generation_requests)

        should_waiting = self.batch_wait_iters_count < self.batch_wait_timeout_iters and num_scheduled_tokens < self.batch_wait_max_tokens_ratio * self.max_num_tokens
        if should_waiting:
            self.batch_wait_iters_count += 1
            return []

        self.batch_wait_iters_count = 0
        return context_requests

    @nvtx_range("_schedule")
    def _schedule(self):
        if hasattr(self.kv_cache_manager, "prepare_expect_snapshot_points"):
            self.kv_cache_manager.prepare_expect_snapshot_points(
                self.active_requests)

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
            # With KV cache manager V2, scheduling has already grown context request KV cache capacity. Requests dropped
            # for batch waiting still occupy KV cache and may reduce the batch size available for generation requests.
            scheduled_context_requests = self._waiting_requests(
                scheduler_output.context_requests,
                scheduler_output.generation_requests)

        num_fitting = scheduler_output.num_fitting_requests
        #TODO(TRTLLM-12359): remove the WAR when PythonMambaCacheManager is deprecated.
        if isinstance(
                self.kv_cache_manager,
                MixedMambaHybridCacheManager) and self.kv_cache_transceiver:
            if len(scheduled_context_requests) > 0:
                scheduled_context_requests = self.kv_cache_manager.filter_ctx_requests_by_capacity(
                    scheduled_context_requests)
                num_fitting = len(scheduled_context_requests)

        scheduled_requests = ScheduledRequests()
        scheduled_requests.encoder_requests = scheduler_output.encoder_requests
        scheduled_requests.reset_context_requests(scheduled_context_requests)
        scheduled_requests.generation_requests = scheduler_output.generation_requests
        scheduled_requests.paused_requests = scheduler_output.paused_requests

        return scheduled_requests, scheduler_output.fitting_disagg_gen_init_requests, num_fitting

    # ---------------------------------------------------------------
    # Encoder-decoder support: encoder iteration in the executor loop.
    #
    # At a scheduling pass, the scheduler may admit encoder-init requests
    # alongside decoder-context and generation requests.  It returns them in
    # disjoint buckets:
    #
    #   * encoder requests (``LlmRequestState.ENCODER_INIT``), which run
    #     through ``ModelEngine.forward_encoder`` on this iteration.
    #     After scatter, they transition to ``CONTEXT_INIT`` and are
    #     re-admitted by the *next* iteration's scheduler pass for the
    #     decoder context step.
    #
    #   * decoder-context requests (``CONTEXT_INIT`` and disagg-gen-init),
    #     which flow through the normal decoder IFB step.
    #
    # The invariant is that encoder and decoder context never share one
    # micro-batch; this preserves the cross-KV lifecycle and the
    # dual-pool budget.
    # ---------------------------------------------------------------
    @nvtx_range("_run_encoder_step")
    def _run_encoder_step(self, encoder_requests: List[LlmRequest]) -> None:
        """Drive one encoder iteration for ``encoder_requests``.

        Runs the encoder stack on the dedicated encoder stream, then
        scatters the packed hidden states back onto the per-request
        ``py_encoder_output`` field and transitions request state to
        ``CONTEXT_INIT`` so the next scheduler pass picks them up as
        decoder-context requests. A separate CUDA event is recorded for
        each request on the encoder stream; the scheduler queries that
        event before admitting the request to a decoder context step.
        """
        if not encoder_requests:
            return

        try:
            self.encoder_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.encoder_stream):
                encoder_hidden_states, encoder_seq_lens = (
                    self.model_engine.forward_encoder(
                        encoder_requests,
                        resource_manager=self.resource_manager,
                    ))
        except Exception as e:
            traceback.print_exc()
            error_msg = str(e)
            logger.error(
                f"Encountered an error in encoder forward: {error_msg}")
            self._handle_errors(error_msg, requests=encoder_requests)
            return

        self._scatter_encoder_output(encoder_requests, encoder_hidden_states,
                                     encoder_seq_lens)
        for req in encoder_requests:
            req.py_encoder_output_ready_event = torch.cuda.Event()
            req.py_encoder_output_ready_event.record(self.encoder_stream)
            # TODO(TRTLLM-12339): Honor return_encoder_output once the public
            # LLM API shape for returned encoder hidden states is finalized.

    @nvtx_range("_scatter_encoder_output")
    def _scatter_encoder_output(
        self,
        encoder_requests: List[LlmRequest],
        encoder_hidden_states: torch.Tensor,
        encoder_seq_lens: List[int],
    ) -> None:
        """Slice packed encoder hidden states into per-request tensors.

        Stores the slice for each request in ``req.py_encoder_output``
        as a temporary GPU buffer (consumed by the first decoder
        context step), and transitions the request from
        ``ENCODER_INIT`` to ``CONTEXT_INIT`` so the next scheduler
        iteration admits it on the decoder side.

        ``py_skip_cross_kv_projection`` is initialized to ``False`` so
        the *first* decoder context step projects K/V from
        ``encoder_output`` and writes the cross-KV pool; the decoder
        step flips it to ``True`` for later steps and chunks.
        """
        if encoder_hidden_states is None:
            raise RuntimeError(
                "Encoder forward returned None hidden states; cannot "
                "scatter encoder output to requests.")

        assert len(encoder_seq_lens) == len(encoder_requests), (
            "Encoder packed sequence lengths must match the number of "
            "encoder requests")
        assert encoder_hidden_states.shape[0] == sum(encoder_seq_lens), (
            "Encoder packed hidden states first dim must equal "
            "sum(encoder_seq_lens)")

        offset = 0
        for req, seq_len in zip(encoder_requests, encoder_seq_lens):
            req.py_encoder_output = encoder_hidden_states[offset:offset +
                                                          seq_len]
            req.py_skip_cross_kv_projection = False
            req.state = LlmRequestState.CONTEXT_INIT
            offset += seq_len

    @nvtx_range("_attach_encoder_output_to_execution_stream")
    def _attach_encoder_output_to_execution_stream(
            self, scheduled_requests: ScheduledRequests) -> None:
        """Hand encoder-produced tensors over to the execution stream.

        Per-request encoder output tensors are produced on the dedicated
        ``encoder_stream`` and consumed by the decoder forward on
        ``execution_stream``. Cross-stream correctness is guaranteed by
        the scheduler:
        ``drop_decoder_context_requests_waiting_for_encoder_output`` excludes
        any ``CONTEXT_INIT`` request whose ``py_encoder_output_ready_event``
        has not completed, so by the time a request reaches this point the
        encoder kernels for that request are already done. No
        ``wait_event`` is therefore needed on the execution stream.

        Two pieces of bookkeeping remain that this helper performs:

        * ``record_stream`` is called on the encoder-output tensor so the
          PyTorch caching allocator knows the storage is still in use on
          the execution stream and must not be reused until the decoder
          forward releases it.
        * The spent ``py_encoder_output_ready_event`` is cleared so it
          cannot be queried again on a later iteration.
        """
        for req in scheduled_requests.context_requests:
            ready_event = getattr(req, "py_encoder_output_ready_event", None)
            if ready_event is None:
                continue

            if req.py_encoder_output is not None:
                req.py_encoder_output.record_stream(self.execution_stream)
            req.py_encoder_output_ready_event = None

    def _mark_cross_kv_projection_consumed(
            self, scheduled_requests: ScheduledRequests) -> None:
        """Release temporary encoder outputs after decoder context consumes them."""
        for req in scheduled_requests.context_requests:
            if getattr(req, "py_encoder_output", None) is None:
                continue
            req.py_encoder_output = None
            req.py_skip_cross_kv_projection = True

    @nvtx_range("_check_disagg_gen_transfer_status")
    def _check_disagg_gen_transfer_status(self):
        if not self._uses_async_disagg_gen_transfer():
            return

        # Gen-transfer status performs cross-rank consensus internally.
        # Enter it symmetrically; ranks with no ready local future contribute
        # an empty ready set.
        self._check_disagg_gen_cache_transfer_status(0)

        if self._is_disagg_inflight_cancel_active():
            self._cancel_timed_out_gen_transfers()
            self._check_gen_cache_transfer_errors_consensus()

        return

    def _is_disagg_inflight_cancel_active(self) -> bool:
        if not is_disagg_inflight_cancel_enabled():
            return False

        transceiver = getattr(self, "kv_cache_transceiver", None)
        if transceiver is None:
            return False

        supports = getattr(transceiver,
                           "supports_inflight_request_cancellation", None)
        if callable(supports) and supports() is True:
            return True

        if not getattr(self, "_disagg_inflight_cancel_unsupported_logged",
                       False):
            logger.warning(
                "TRTLLM_DISAGG_ENABLE_INFLIGHT_CANCEL=1 was requested, but "
                f"{type(transceiver).__name__} does not advertise in-flight "
                "request cancellation support. Cancellation and transfer-buffer "
                "quarantine are currently scoped to the C++ NIXL transceiver "
                "with the UCX plugin; using the existing timeout and "
                "cancellation behavior for this transceiver.")
            self._disagg_inflight_cancel_unsupported_logged = True
        return False

    def _request_kv_transfer_cancellation(self, request: LlmRequest) -> bool:
        """Best-effort cancellation that leaves ownership intact on errors."""
        try:
            return self.kv_cache_transceiver.cancel_request(request)
        except Exception as error:
            logger.error(f"KV transfer cancellation failed for request "
                         f"{request.py_request_id}; will retry: {error}")
            return False

    @nvtx_range("_cancel_timed_out_gen_transfers")
    def _cancel_timed_out_gen_transfers(self) -> None:
        """Request cancellation for timed-out generation transfers.

        This path is deliberately rank-synchronized. Under attention-DP, each
        TP rank can own a different request subset, but later error responses
        still pass through a TP collective. The timeout/cancel decision must
        therefore be based on the TP-wide request-id union rather than a
        rank-local timeout observation.
        """
        timeout_ms = self.kv_cache_transceiver.kv_transfer_timeout_ms
        if timeout_ms is None:
            return

        requests_in_transfer = {
            req.py_request_id: req
            for req in self.active_requests
            if req.is_disagg_generation_transmission_in_progress
        }
        current_time = time.monotonic()
        for request in requests_in_transfer.values():
            if request.py_kv_transfer_start_time is None:
                continue
            elapsed_time = ((current_time - request.py_kv_transfer_start_time) *
                            1000)
            if (elapsed_time > timeout_ms
                    and not request.py_kv_transfer_timed_out):
                logger.warning(
                    f"Requesting cancellation for generation request "
                    f"{request.py_request_id} due to KV cache transfer timeout")
                request.py_kv_transfer_timed_out = True

        user_canceled_ids = set(self.canceled_req_ids)
        local_timed_out_ids = sorted(
            request_id for request_id, request in requests_in_transfer.items()
            if request.py_kv_transfer_timed_out
            and request_id not in user_canceled_ids
            and request_id not in self._disagg_timed_out_gen_cancelled_ids)

        if self.dist.tp_size > 1:
            any_timed_out = self.dist.tp_allreduce(int(
                bool(local_timed_out_ids)),
                                                   op=ReduceOp.MAX)
        else:
            any_timed_out = int(bool(local_timed_out_ids))
        if not any_timed_out:
            return

        if self.dist.tp_size > 1:
            gathered_timed_out_ids = self.dist.tp_allgather(local_timed_out_ids)
            timed_out_ids = sorted(set().union(*gathered_timed_out_ids))
        else:
            timed_out_ids = local_timed_out_ids

        for request_id in timed_out_ids:
            request = requests_in_transfer.get(request_id)
            if request is None:
                continue

            # A peer rank may have crossed the timeout first.  Mirror the
            # TP-wide decision locally so a failed cancel attempt keeps
            # retrying even if this rank's wall clock had not yet expired.
            request.py_kv_transfer_timed_out = True

            if request_id in self._disagg_timed_out_gen_cancelled_ids:
                continue

            is_cancelled = self._request_kv_transfer_cancellation(request)
            if is_cancelled:
                self._disagg_timed_out_gen_cancelled_ids.add(request_id)
                logger.warning(
                    f"Cancelled timed-out generation KV transfer for request "
                    f"{request.py_request_id}; waiting for C++ transfer "
                    "status to report final cleanup")

    @nvtx_range("_check_gen_cache_transfer_errors_consensus")
    def _check_gen_cache_transfer_errors_consensus(self) -> None:
        """Flush generation transfer errors through a TP-uniform path."""
        error_requests = [
            req for req in self._get_disagg_reqs_in_error_state()
            if req.is_generation_only_request()
        ]
        local_needs_flush = bool(error_requests)

        if self.dist.tp_size > 1:
            any_needs_flush = self.dist.tp_allreduce(int(local_needs_flush),
                                                     op=ReduceOp.MAX)
        else:
            any_needs_flush = int(local_needs_flush)
        if not any_needs_flush:
            return

        error_msg = "Error in kv cache transfer for generation requests"
        self._handle_errors(error_msg,
                            requests=error_requests,
                            charge_budget=False)

    @nvtx_range("_check_kv_transfer_timeout")
    def _check_kv_transfer_timeout(self):
        if not self.kv_cache_transceiver:
            return
        timeout_ms = self.kv_cache_transceiver.kv_transfer_timeout_ms
        if timeout_ms is None:
            return

        def flag_if_kv_transfer_timed_out(req: LlmRequest, type: str) -> None:
            current_time = time.monotonic()
            if req.py_kv_transfer_start_time is None:
                return
            elapsed_time = (current_time - req.py_kv_transfer_start_time) * 1000
            if elapsed_time > timeout_ms and not req.py_kv_transfer_timed_out:
                verb = ("Requesting cancellation for"
                        if self._is_disagg_inflight_cancel_active() else
                        "Observed timeout on")
                logger.warning(
                    f"{verb} {type} request {req.py_request_id} due to KV "
                    f"cache transfer timeout: elapsed {elapsed_time:.0f}ms > "
                    f"kv_transfer_timeout_ms={timeout_ms}ms")
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
        In context-first mode, context requests are schedulable immediately,
        otherwise, we need to check if context requests are ready to be scheduled by querying kv cache transceiver
        """
        if not self.kv_cache_transceiver:
            return
        gen_first_ctx_requests = [
            req for req in new_requests
            if req.is_context_only_request and req.py_disaggregated_params.
            schedule_style == DisaggScheduleStyle.GENERATION_FIRST
        ]
        # Always call prepare_context_requests when there are new requests
        # or previously-waiting requests, so the tp_allgather consensus
        # can promote requests whose peer info has arrived on all ranks.
        self.kv_cache_transceiver.prepare_context_requests(
            gen_first_ctx_requests)

    def _count_schedulable_active_requests(self) -> int:
        """Count active requests that are ready for scheduling.

        The non-PP DeepSeek-V4 disaggregated ADP path mirrors the decoder
        scheduler's state window [CONTEXT_INIT, GENERATION_TO_COMPLETE). This
        covers generation-first context requests below the lower bound and
        terminal requests at the upper bound. Other configurations retain the
        established ADP behavior; PP eligibility remains follow-up scope.

        Returns:
            The number of active requests eligible for scheduling.
        """
        if (not self._enable_dsv4_adp_dummy_fixes
                or self.kv_cache_transceiver is None):
            if self.kv_cache_transceiver is None:
                return len(self.active_requests)

            return sum(
                1 for req in self.active_requests
                if not (req.is_disagg_generation_init_state
                        or req.is_disagg_generation_transmission_in_progress))

        schedule_from_value = LlmRequestState.CONTEXT_INIT.value
        to_complete_value = LlmRequestState.GENERATION_TO_COMPLETE.value

        return sum(
            1 for req in self.active_requests
            if schedule_from_value <= req.state_value < to_complete_value)

    def _has_adp_dummy_kv_capacity(self,
                                   token_nums: Optional[List[int]]) -> bool:
        """Check the full dummy allocation before entering rank-local code.

        V1's dummy allocator historically checked only that one block was
        free. A context-side ADP dummy can be ``max_num_tokens`` long, while a
        generation dummy also reserves draft tokens, so that check can still
        let the C++ batch allocation fail after a peer rank has succeeded.
        """
        token_num = (token_nums[0] if token_nums is not None else 1 +
                     self.max_total_draft_tokens)
        mapping = getattr(self.kv_cache_manager, "mapping", None)
        has_cp_helix = getattr(mapping, "has_cp_helix", None)
        if callable(has_cp_helix) and has_cp_helix():
            token_num = max(token_num, 2)
        draft_reserve = (self.max_total_draft_tokens
                         if self._adp_dummy_is_gen else 0)
        available_tokens = self.kv_cache_manager.get_num_available_tokens(
            token_num_upper_bound=token_num, max_num_draft_tokens=draft_reserve)
        return available_tokens >= token_num

    def _should_skip_dummy_for_benchmark_disagg(
            self, num_schedulable_requests: int) -> bool:
        """Decide whether to skip ADP dummy insertion during benchmark disagg fill.

        During the fill phase (``_benchmark_fill_phase_active`` is True),
        the ``can_forward`` gate prevents forward-pass collectives, so
        temporarily-empty ranks don't need dummies.  Dummies added during
        the fill phase would never be cleaned up (termination only runs
        after a forward pass), permanently wasting KV cache slots.

        Once the fill phase completes and the gate opens, the flag is
        cleared and this method stops skipping — the normal dummy
        add-forward-terminate lifecycle handles taper-down correctly
        (e.g., when ranks empty out at different rates due to varied
        speculative decoding acceptance rates).

        Args:
            num_schedulable_requests: Number of active requests that have
                completed KV transfer and are ready for the forward pass.

        Returns:
            True if dummy insertion should be skipped for this iteration.
        """
        if not self._benchmark_fill_phase_active or self.is_warmup:
            return False

        logger.info(f"Skipped adding dummy requests: "
                    f"num_fetch_requests={self.num_fetch_requests}, "
                    f"num_schedulable_requests={num_schedulable_requests}")
        return True

    def _update_adp_dummy_role(self, candidates: List[LlmRequest]) -> None:
        if not self.enable_attention_dp or self.kv_cache_transceiver is None:
            return
        has_ctx = False
        has_gen = False
        for req in candidates:
            rt = getattr(req, "llm_request_type", None)
            if rt == LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY:
                has_ctx = True
            elif rt == LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY:
                has_gen = True
        # Prefer the CTX role when both types are present this iteration: a CTX
        # dummy is padded to max_num_tokens so idle ranks keep MoE all-to-all
        # token counts comparable with ranks doing real context work.
        if has_ctx:
            self._adp_dummy_is_gen = False
        elif has_gen:
            self._adp_dummy_is_gen = True

    @nvtx_range("_pad_attention_dp_dummy_request")
    def _pad_attention_dp_dummy_request(self):
        """
        Pad with a generation dummy request, if required, to ensure every attention_dp rank has at least one active request.
        """
        if not self.enable_attention_dp:
            return

        assert self.expected_num_active_requests >= len(self.active_requests)
        num_active_request = self._count_schedulable_active_requests()

        if self._should_skip_dummy_for_benchmark_disagg(num_active_request):
            return

        needs_dummy = (self.expected_num_active_requests > 0
                       and num_active_request == 0)
        if not needs_dummy:
            return

        dummy_request_ids = [ATTENTION_DP_DUMMY_REQUEST_ID]
        token_nums = None
        if (not self._adp_dummy_is_gen and self.kv_cache_transceiver is not None
                and self.max_num_tokens is not None):
            token_nums = [self.max_num_tokens]

        if (not self._enable_dsv4_adp_dummy_fixes
                or self.kv_cache_transceiver is None):
            llm_request = self.kv_cache_manager.add_dummy_requests(
                request_ids=dummy_request_ids,
                token_nums=token_nums,
                is_gen=self._adp_dummy_is_gen,
                prepare_resource=True,
                max_num_draft_tokens=self.max_total_draft_tokens,
            )[0]
            llm_request.is_attention_dp_dummy = True
            spec_resource_manager = self.resource_manager.get_resource_manager(
                ResourceManagerType.SPEC_RESOURCE_MANAGER)
            if spec_resource_manager is not None:
                spec_resource_manager.add_dummy_requests(dummy_request_ids)
            self.active_requests.append(llm_request)
            return

        assert self._pending_adp_dummy_request is None
        has_live_adp_dummy = any(
            request.py_request_id == ATTENTION_DP_DUMMY_REQUEST_ID
            for request in self.active_requests)
        if has_live_adp_dummy or not self._has_adp_dummy_kv_capacity(
                token_nums):
            return

        try:
            dummy_requests = self.kv_cache_manager.add_dummy_requests(
                request_ids=dummy_request_ids,
                token_nums=token_nums,
                is_gen=self._adp_dummy_is_gen,
                prepare_resource=True,
                max_num_draft_tokens=self.max_total_draft_tokens,
            )
        except OutOfPagesError:
            dummy_requests = None
        if not dummy_requests:
            logger.warning(
                "Cannot allocate DeepSeek-V4 ADP pad dummy; rank schedules "
                "an empty batch and the fleet will retry.")
            return

        dummy_request = dummy_requests[0]
        spec_resource_manager = self.resource_manager.get_resource_manager(
            ResourceManagerType.SPEC_RESOURCE_MANAGER)
        if spec_resource_manager is not None:
            try:
                spec_resource_manager.add_dummy_requests(dummy_request_ids)
            except NoFreeSlotsError:
                self.kv_cache_manager.free_resources(dummy_request)
                return

        assert dummy_request is not None
        dummy_request.is_attention_dp_dummy = True
        self.active_requests.append(dummy_request)
        self._pending_adp_dummy_request = dummy_request

    @nvtx_range("_prepare_disagg_gen_init")
    def _prepare_disagg_gen_init(self, fitting_disagg_gen_init_requests):
        if fitting_disagg_gen_init_requests:
            disagg_gen_init_to_prepare = ScheduledRequests()
            disagg_gen_init_to_prepare.context_requests_last_chunk = fitting_disagg_gen_init_requests

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
            requests.context_requests_last_chunk = cache_trans_complete_requests
            self.resource_manager.resource_managers[
                ResourceManagerType.SEQ_SLOT_MANAGER].prepare_resources(
                    requests)
            self._setup_sampler_step(requests)

        for req in scheduled_batch.generation_requests:
            if req.is_disagg_generation_transmission_complete:
                req.state = LlmRequestState.GENERATION_IN_PROGRESS
                req.context_current_position = req.prompt_len
                if self.kv_cache_transceiver is not None:
                    self.kv_cache_transceiver.commit_blocks_for_reuse(req)
                req.decoding_iter = 1
                req.py_decoding_iter = 1
                req.py_kv_transfer_start_time = None
                req.py_kv_transfer_timed_out = False
                first_gen_tokens = req.context_phase_params.first_gen_tokens
                ctx_draft_tokens = req.context_phase_params.draft_tokens
                if not ctx_draft_tokens and self.model_engine.enable_spec_decode:
                    # CTX has no speculative decoding — fill dummy draft tokens
                    # so model_engine builds the correct input shape (1 + draft_len
                    # tokens per gen request). Dummies will be rejected on verify,
                    # and the draft model will produce real tokens for next step.
                    ctx_draft_tokens = [
                        0
                    ] * self.model_engine.max_total_draft_tokens
                req.py_draft_tokens = [] if ctx_draft_tokens is None else ctx_draft_tokens
                beam_width = req.py_beam_width
                if not self._update_sampler_state_for_disagg_gen_request(
                        req, beam_width, first_gen_tokens):
                    continue
                for beam in range(0, beam_width):
                    req.add_new_token(first_gen_tokens[beam], beam)

                self._maybe_prepend_logprobs_and_logits(req, beam_width)

    def _update_sampler_state_for_disagg_gen_request(self, req, beam_width,
                                                     first_gen_tokens) -> bool:
        """Update beam sampler state with context-side first-token data."""
        if beam_width <= 1:
            return True

        def fail_request(message: str) -> bool:
            logger.error(message)
            req.state = LlmRequestState.DISAGG_TRANS_ERROR
            return False

        seq_slot = req.py_seq_slot
        if seq_slot is None:
            return fail_request(
                "Cannot update sampler state for disagg beam request "
                f"{req.py_request_id}: seq slot is not assigned.")

        sampler_store = getattr(self.sampler, 'store', None)
        beam_search_store = getattr(sampler_store, 'beam_search_store', None)
        if beam_search_store is None:
            return fail_request(
                "Cannot update sampler state for disagg beam request "
                f"{req.py_request_id}: sampler has no beam search store.")
        if len(first_gen_tokens) < beam_width:
            return fail_request(
                "Invalid first_gen_tokens length for disagg beam "
                f"request {req.py_request_id}: "
                f"{len(first_gen_tokens)} < {beam_width}")

        disagg_params = getattr(req, 'py_disaggregated_params', None)
        if disagg_params is None:
            return fail_request(
                "No disaggregated params available for disagg beam "
                f"request {req.py_request_id}.")

        first_gen_log_probs = getattr(disagg_params, 'first_gen_log_probs',
                                      None)
        if first_gen_log_probs is None:
            return fail_request(
                "No first_gen_log_probs available for disagg beam "
                f"request {req.py_request_id}.")

        if len(first_gen_log_probs) != beam_width:
            return fail_request(
                "Invalid first_gen_log_probs length for disagg beam "
                f"request {req.py_request_id}: "
                f"{len(first_gen_log_probs)} != {beam_width}")

        first_gen_scores = []
        for beam_idx, (token_id, token_log_probs) in enumerate(
                zip(first_gen_tokens[:beam_width], first_gen_log_probs)):
            token_log_prob = token_log_probs.get(token_id)
            if token_log_prob is None:
                return fail_request(
                    "Missing first_gen_log_probs entry for disagg beam "
                    f"request {req.py_request_id}: beam={beam_idx}, "
                    f"token_id={token_id}")
            first_gen_scores.append(token_log_prob.logprob)

        original_tokens = beam_search_store.original_tokens
        first_gen_token_values = torch.tensor(first_gen_tokens[:beam_width],
                                              device=original_tokens.device,
                                              dtype=original_tokens.dtype)
        original_tokens[seq_slot, :beam_width,
                        req.prompt_len].copy_(first_gen_token_values)
        cache_indirection = beam_search_store.cache_indirection
        beam_idx_arange = torch.arange(beam_width,
                                       device=original_tokens.device,
                                       dtype=cache_indirection.dtype)
        cache_indirection[seq_slot, :beam_width,
                          req.prompt_len].copy_(beam_idx_arange)

        cum_log_probs = beam_search_store.cum_log_probs
        values = torch.tensor(first_gen_scores,
                              device=cum_log_probs.device,
                              dtype=cum_log_probs.dtype)
        cum_log_probs[seq_slot, :beam_width].copy_(values)
        return True

    @staticmethod
    def _maybe_attach_ctx_usage(request: LlmRequest, response):
        """Surface gen-first ctx usage (delivered via the KV-transfer aux
        buffer in RxSession.unpack_aux) onto the response so the postprocessor
        adopts the context-side prompt/cached token accounting."""
        disagg_params = request.py_disaggregated_params
        if disagg_params is not None and disagg_params.ctx_usage is not None:
            response.result.ctx_usage = disagg_params.ctx_usage

    def _maybe_prepend_logprobs_and_logits(self, req, beam_width):
        """Prepend logprobs and generation logits for first_gen_tokens
        if transferred from prefill."""
        disagg_params = getattr(req, 'py_disaggregated_params', None)
        if disagg_params is None:
            return

        if getattr(disagg_params, 'first_gen_log_probs', None) is not None:
            if beam_width != 1:
                pass
            else:
                req.py_result.append_log_probs(
                    [disagg_params.first_gen_log_probs])

        if getattr(disagg_params, 'first_gen_logits', None) is not None:
            if beam_width != 1:
                logger.warning(
                    "Skipping first_gen_logits prepend for "
                    "request %s: beam_width=%s is not supported.",
                    req.py_request_id, beam_width)
            else:
                device = torch.device('cuda', self.device_id)
                for logits_tensor in disagg_params.first_gen_logits:
                    req.py_result.append_generation_logits(
                        logits_tensor.to(device))

    def _has_prepended_logits(self, req) -> bool:
        """Check whether the request has first-gen logits prepended from
        prefill that need a snapshot before response creation."""
        if not self.should_exclude_last_generation_logits:
            return False
        disagg_params = getattr(req, 'py_disaggregated_params', None)
        if disagg_params is None:
            return False
        return getattr(disagg_params, 'first_gen_logits', None) is not None

    @nvtx_range("_recv_disagg_gen_cache")
    def _recv_disagg_gen_cache(self, new_gen_reqs):

        # gen_only_no_context has no CTX worker, so mark each request as
        # transmission-complete immediately.
        if self._is_disagg_gen_only_no_context_benchmark():
            for req in new_gen_reqs:
                req.state = LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE
            return

        if not self._uses_async_disagg_gen_transfer():
            # Resources have already been prepared for every request in this
            # batch. Drain all synchronous receives even after one fails so no
            # prepared request is left in DISAGG_GENERATION_INIT.
            for req in new_gen_reqs:
                self.kv_cache_transceiver.request_and_receive_sync(req)
                if req.state == LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE:
                    self._sync_disagg_transfer_made_progress = True
            self._check_cache_transfer_errors("generation requests")
            return

        for req in new_gen_reqs:
            self.kv_cache_transceiver.request_and_receive_async(req)

        if self.kv_cache_transceiver.kv_transfer_timeout_ms is not None:
            for req in new_gen_reqs:
                if req.state == LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS:
                    req.py_kv_transfer_start_time = time.monotonic()

        self._check_disagg_gen_cache_transfer_status(0)

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
                    # Forward is done for this request — release the
                    # IndexMapper slot so new requests can reuse it.
                    # KV blocks stay allocated for the upcoming transfer.
                    if hasattr(self.kv_cache_manager, 'release_index_slot'):
                        self.kv_cache_manager.release_index_slot(
                            req.py_request_id)
                    # Order is important here: we need to start the transfer before responding
                    # to make sure the blocks are stored for reuse before they are sent.
                    self.async_transfer_manager.start_transfer(req)
                    self.kv_cache_transceiver.respond_and_send_async(req)

                    if self.kv_cache_transceiver.kv_transfer_timeout_ms is not None:
                        req.py_kv_transfer_start_time = time.monotonic()

        if self.kv_connector_manager:
            if not self.disable_overlap_scheduler:
                requests = self.previous_batch.scheduled_requests.all_requests(
                ) if self.previous_batch is not None else []
            else:
                requests = scheduled_requests
            for req in requests:
                if req.is_finished:
                    kv_connector_request_finished(req)

        if self.kv_cache_transceiver:
            self._check_disagg_ctx_cache_transfer_status(0)

    @staticmethod
    def _request_vote_id(request: LlmRequest) -> int:
        return (request.parent_request_id
                if request.is_child else request.py_request_id)

    def _is_disagg_error_cleanup_blocked(self, request: LlmRequest) -> bool:
        request_id = self._request_vote_id(request)
        if request_id in getattr(self, "canceled_req_ids", ()):
            return True

        async_transfer_manager = getattr(self, "async_transfer_manager", None)
        if (getattr(request, "is_context_only_request", False) is True
                and async_transfer_manager is not None and request.py_request_id
                in async_transfer_manager.requests_in_transfer()):
            return True
        return False

    def _get_disagg_reqs_in_error_state(self):
        return [
            req for req in self.active_requests
            if req.state == LlmRequestState.DISAGG_TRANS_ERROR
            and not self._is_disagg_error_cleanup_blocked(req)
        ]

    def _check_cache_transfer_errors(self, error_msg_prefix: str):
        """Check and handle cache transfer errors.

        Under ADP this is a no-op: errors are handled by
        ``_handle_disagg_cache_errors_synced`` at the loop top.
        """
        if self.enable_attention_dp and self.dist.world_size != 1:
            return
        error_requests = self._get_disagg_reqs_in_error_state()
        if error_requests:
            error_msg = f"Error in kv cache transfer for {error_msg_prefix}"
            self._handle_errors(error_msg,
                                requests=error_requests,
                                charge_budget=False)

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
            if (not request.py_kv_transfer_timed_out
                    or request_id in completed_req_ids
                    or request_id in self._disagg_timed_out_ctx_cancelled_ids):
                continue

            is_cancelled = self._request_kv_transfer_cancellation(request)
            if not is_cancelled:
                continue

            if self._is_disagg_inflight_cancel_active():
                self._disagg_timed_out_ctx_cancelled_ids.add(request_id)
                logger.warning(f"Cancelled timed-out context KV transfer for "
                               f"request {request.py_request_id}; waiting for "
                               "C++ transfer status to report final cleanup")
            else:
                # Preserve the legacy timeout behavior when in-flight
                # cancellation is disabled: a queued transfer that can be
                # cancelled is immediately released from the async manager.
                request.py_kv_transfer_start_time = None
                request.state = LlmRequestState.DISAGG_CONTEXT_COMPLETE
                self._end_transfer_and_maybe_terminate(request)

        self._check_cache_transfer_errors("context requests")

    @nvtx_range("_check_disagg_gen_cache_transfer_status")
    def _check_disagg_gen_cache_transfer_status(self, atLeastNum: int = 0):
        result = self.kv_cache_transceiver.check_gen_transfer_status(atLeastNum)
        if isinstance(result, tuple):
            _, _, cancelled_reqs = result
            user_canceled_set = set(self.canceled_req_ids)
            for req in cancelled_reqs:
                req_id = req.py_request_id if not req.is_child else req.parent_request_id
                if req_id not in user_canceled_set:
                    req.state = LlmRequestState.DISAGG_TRANS_ERROR
        if not self._is_disagg_inflight_cancel_active():
            self._check_cache_transfer_errors("generation requests")

    def _maybe_prefetch_next_iter_mm_encoders(
            self, scheduled_batch: ScheduledRequests) -> None:
        """Best-effort hook for cross-iter MM encoder prefetch.

        Called immediately after `_forward_step`, so the side-stream encoder
        work can overlap current-iteration sampling in the non-overlap loop and
        previous-batch `_update_requests` in the overlap loop. No-op unless
        `multimodal_config.encoder_side_stream_max_ahead` is positive and the
        model is a `MultimodalModelMixin` subclass.

        Walks `active_requests` for context-init candidates that are NOT
        in the just-scheduled batch (and, in overlap mode, not in the
        previous batch either) and dispatches one of them, subject to the
        outstanding-ahead cap in `maybe_prefetch_mm_encoder_for_next_iter`.
        That helper runs the encoder on a side CUDA stream and stashes
        results back into `request.py_multimodal_data`. The next iteration's
        `_prepare_inputs` then picks up the cached embedding and the mixin
        consume site waits on the recorded CUDA event.

        Shared between `_executor_loop` (non-overlap) and
        `_executor_loop_overlap`. `self.previous_batch` is always None in
        non-overlap mode, so the second union term is a no-op there.
        """
        model = getattr(self.model_engine, "model", None)
        if model is None:
            return
        in_flight = {r.py_request_id for r in scheduled_batch.all_requests()}
        if self.previous_batch is not None:
            in_flight |= {
                r.py_request_id
                for r in self.previous_batch.scheduled_requests.all_requests()
            }
        pending = [
            r for r in self.active_requests
            if r.state == LlmRequestState.CONTEXT_INIT
        ]
        if not pending:
            return
        max_prefetch_ahead = (
            self.llm_args.multimodal_config.encoder_side_stream_max_ahead)
        try:
            maybe_prefetch_mm_encoder_for_next_iter(
                model=model,
                pending_requests=pending,
                in_flight_request_ids=in_flight,
                max_prefetch=1,
                max_prefetch_ahead=max_prefetch_ahead,
            )
        except Exception:
            # Speculative prefetch is best-effort and must never crash the
            # executor loop. On failure, `py_mm_encoder_event` is not stamped,
            # so the next iteration's `_prepare_inputs` falls back to the
            # standard in-iter encode path (which re-runs `to_device` and the
            # encoder unconditionally when no cached embedding is present).
            logger.warning(
                f"Cross-iter MM encoder prefetch failed; falling back to "
                f"in-iter encode.\n{traceback.format_exc()}")

    def _forward_step(
            self,
            scheduled_requests: ScheduledRequests,
            new_tensors_device: Optional[SampleStateTensors] = None,
            num_accepted_tokens_device: Optional[torch.Tensor] = None):
        ExpertStatistic.set_iter(self.iter_counter)

        num_ctx_tokens = sum(req.context_chunk_size
                             for req in scheduled_requests.context_requests)

        @nvtx_range(
            f"[Executor] _forward_step {self.iter_counter}: {scheduled_requests.num_context_requests} ctx reqs, {num_ctx_tokens} ctx tokens, {scheduled_requests.num_generation_requests} gen reqs"
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
            self._attach_encoder_output_to_execution_stream(scheduled_requests)
            with torch.cuda.stream(self.execution_stream):
                outputs = forward(scheduled_requests, self.resource_manager,
                                  new_tensors_device, gather_context_logits,
                                  cache_indirection_buffer,
                                  num_accepted_tokens_device)
                self._mark_cross_kv_projection_consumed(scheduled_requests)

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

    def _update_generation_requests_that_will_complete_next_iteration(
            self, generation_requests: list[LlmRequest]):
        """ Update the generation requests that will complete next iteration.

        If overlap scheduling is enabled, we need update the state of generation requests that will complete next iteration
        and adjust the exclude_last_generation_logits flag accordingly.
        """
        for request in generation_requests:
            if request.state != LlmRequestState.GENERATION_COMPLETE and request.will_complete_next_iteration(
            ):
                request.set_exclude_last_generation_logits(False)
                request.state = LlmRequestState.GENERATION_TO_COMPLETE

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
                # Prefill is done for this request; drop pinned encoder outputs
                # (multimodal_embedding) and raw pre-encoder tensors that multimodal models stashed
                # on `py_multimodal_data`. Without this, encoder inputs and outputs for multi-modal
                # requests stay pinned on GPU through the full decode lifetime and can lead to OOMs
                # at high concurrency.
                _strip_py_multimodal_data_post_prefill(request)
                if not self.disable_overlap_scheduler and request.will_complete_next_iteration(
                ):
                    request.set_exclude_last_generation_logits(False)
                    request.state = LlmRequestState.GENERATION_TO_COMPLETE
                else:
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
                       requests: Optional[List[LlmRequest]] = None,
                       charge_budget: bool = True) -> None:
        """Fail requests and optionally initiate shutdown on fatal errors.

        When ``charge_budget`` is True (the default), classifies the error
        via the error budget.  If deemed fatal (immediate-fatal pattern or
        budget exhausted), **all** active requests are failed and a shutdown
        is enqueued.  Otherwise only the requests in *requests* are failed.

        When ``charge_budget`` is False, the error is treated as a
        per-request failure unless the executor was already marked fatal by
        the caller. The error budget is not consumed. Use this for
        request-scoped errors (validation, KV-transfer timeout,
        guided-decoder) that should not affect server health, and for the
        cleanup phase of an already-classified fatal error.

        .. note::
            The ``charge_budget=False`` path reuses the full
            ``_handle_errors`` machinery (queue drain, response
            enqueue, terminate) even though it only needs to fail a
            single request.  A future improvement would be to extract
            a lightweight ``_fail_request(request, error_msg)`` helper
            for request-scoped failures, keeping ``_handle_errors``
            focused on system-level errors that may crash the engine.

        Args:
            error_msg: Human-readable error description.  Defaults to
                ``"error"`` when ``None``.
            requests: Subset of active requests to fail.  When ``None``
                (or when the error is fatal), all ``active_requests`` are
                failed.
            charge_budget: Whether to consume the error budget.  Set to
                False for request-scoped errors that should not affect
                server health.
        """
        error_responses: Dict[int, LlmResponse] = {}
        error_msg = error_msg or "error"

        budget_fatal = (self._error_budget.consume(error_msg)
                        if charge_budget else False)
        is_fatal = self._fatal_error is not None or budget_fatal
        if budget_fatal and self._error_budget.budget < 1e-9:
            logger.error(f"Error budget exhausted "
                         f"(budget={self._error_budget.budget:.3f}), "
                         "treating as fatal")

        if is_fatal:
            if self._fatal_error is None:
                self._fatal_error = RuntimeError(f"Fatal error: {error_msg}")
            self.is_shutdown = True
            logger.error(
                f"Fatal error detected, initiating shutdown: {error_msg}")
            requests = None

            # Drain waiting_queue so that queued-but-not-yet-activated
            # requests don't get picked up on the next iteration.
            # These are RequestQueueItems (not yet LlmRequests), so we
            # fail them via error responses.  Buffer all responses and
            # call _enqueue_responses once after the loop so every rank
            # enters the same number of collectives (attention-DP /
            # gather-all modes use collective gathers internally).
            waiting_responses: List[Tuple[int, LlmResponse]] = []
            while self.waiting_queue:
                item = self.waiting_queue.pop_request()
                if (self.gather_all_responses
                        or self.dist.rank == 0) and item.request is not None:
                    waiting_responses.append(
                        (item.id,
                         LlmResponse(request_id=item.id,
                                     error_msg=error_msg,
                                     client_id=getattr(item.request,
                                                       'client_id', None))))
            # Also drain executor_request_queue so items already queued
            # but not yet fetched by the main loop are not scheduled
            # after the CUDA context is corrupted.  Safe to use empty()
            # here because is_shutdown is True and the queue's active
            # flag is about to be set False, so no new items arrive.
            raw_queue = self.executor_request_queue.get_request_queue()
            while not raw_queue.empty():
                item = raw_queue.get_nowait()
                if item.is_shutdown_request:
                    continue
                if ((self.gather_all_responses or self.dist.rank == 0)
                        and item.request is not None):
                    waiting_responses.append(
                        (item.id,
                         LlmResponse(request_id=item.id,
                                     error_msg=error_msg,
                                     client_id=getattr(item.request,
                                                       'client_id', None))))

            adp_collective_required = (self.enable_attention_dp
                                       and self.dist.world_size != 1)
            if waiting_responses or adp_collective_required:
                self._enqueue_responses(waiting_responses)
                if waiting_responses:
                    logger.info(
                        f"Drained {len(waiting_responses)} queued requests "
                        "on fatal error")

        failed_requests = (list(self.active_requests)
                           if requests is None else requests)
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

        if self._fatal_error is not None:
            self.executor_request_queue.enqueue_shutdown_request()

    def _terminate_request(self, request: LlmRequest):
        # Dummy requests don't participate in disagg KV cache transfers,
        # so they must bypass the PP termination handler to avoid stale
        # sequences in the KV cache manager (the handler delays removal,
        # but the dummy ID is reused every iteration).
        if (self._disagg_pp_termination_handler is not None
                and not request.is_dummy_request):
            self._disagg_pp_termination_handler.terminate(request)
        else:
            self._do_terminate_request(request)

    def _do_terminate_request(self, request: LlmRequest):
        self.resource_manager.free_resources(request)
        self._prefetched_request_ids.discard(request.py_request_id)
        self._disagg_timed_out_ctx_cancelled_ids.discard(request.py_request_id)
        self._disagg_timed_out_gen_cancelled_ids.discard(request.py_request_id)

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

        async_transfer_manager = getattr(self, "async_transfer_manager", None)
        if (getattr(request, "is_context_only_request", False) is True
                and async_transfer_manager is not None and request.py_request_id
                in async_transfer_manager.requests_in_transfer()):
            if self._is_disagg_inflight_cancel_active():
                self._request_kv_transfer_cancellation(request)
            return False

        if not self._is_request_in_transmission(request):
            return True

        if self._is_disagg_inflight_cancel_active():
            self._request_kv_transfer_cancellation(request)
            return False

        return self._request_kv_transfer_cancellation(request)

    @nvtx_range("_handle_canceled_requests")
    def _handle_canceled_requests(self):
        if len(self.canceled_req_ids) == 0:
            return

        # Create set from list of canceled request ids to speed up canceled test
        canceled_req_ids_set = set(self.canceled_req_ids)

        # Remove canceled requests from the waiting queue
        self.waiting_queue.remove_by_ids(canceled_req_ids_set)

        still_pending_canceled_ids = []
        for request in self.active_requests:
            req_id = request.py_request_id if not request.is_child else request.parent_request_id
            if req_id not in canceled_req_ids_set:
                continue

            is_cancelled = self._try_cancel_request(request)
            if is_cancelled:
                # Mark requests as finished, then, we reuse all existing code
                # to clean up the KV cache resources.
                request.py_kv_transfer_timed_out = False
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
                # Snapshot prepended first_gen_logits for the response:
                #
                # 1. generation_logits is not stored on LlmResult; every
                #    access goes through __getattr__ -> PyResult property
                #    -> LogitsStorage, re-reading shared mutable state.
                # 2. All streaming responses reference the same PyResult,
                #    so later tokens mutate what earlier responses would
                #    read.
                # 3. With the overlap scheduler, exclude_last_generation_logits
                #    is True, which would hide the prepended logits since
                #    they are the only entry at this point.
                #
                # WAR: read the logits now (bypassing exclusion), then set
                # the tensor directly on LlmResult as an instance attribute.
                # This shadows __getattr__ so the consumer always gets the
                # correct, frozen logits.
                has_prepended_logits = self._has_prepended_logits(req)
                logits_snapshot = (req.py_result.get_latest_logits_unexcluded()
                                   if has_prepended_logits else None)
                response = req.create_response(False, self.dist.rank)
                if logits_snapshot is not None and response is not None:
                    response.result.generation_logits = logits_snapshot
                new_responses.append((req.py_request_id, response))

        self._enqueue_responses(new_responses)

    @nvtx_range("_emit_first_token_responses")
    def _emit_first_token_responses(self, prev_scheduled_requests):
        """Emit first-token responses ahead of `_sample_async` to reduce
        TTFT. Termination, cleanup, and perf stats remain in
        `_handle_responses`. Only invoked when
        `enable_early_first_token_response` is set.
        """
        new_responses = []
        for request in prev_scheduled_requests.all_requests():
            if request.py_decoding_iter != 1:
                continue
            if request.is_attention_dp_dummy or request.is_cuda_graph_dummy:
                continue
            # Terminal response is issued by `_handle_responses`; an
            # early-emitted response is never final.
            if request.is_finished:
                continue

            logger.debug(
                f'Send first token response for request {request.py_request_id}'
            )

            request.draft_tokens = request.py_draft_tokens if get_draft_token_length(
                request) > 0 else []
            request.decoding_iter = request.py_decoding_iter

            # Snapshot first-token logits before `exclude_last_generation_logits`
            # would hide them; only the first logits chunk is appended at this
            # point. Same approach as in `_handle_first_token_response`.
            logits_snapshot = None
            if (self.should_exclude_last_generation_logits
                    and request.py_return_generation_logits):
                logits_snapshot = request.py_result.get_latest_logits_unexcluded(
                )

            response = request.create_response(False, self.dist.rank)
            if response is None:
                continue
            response.result.cached_tokens = request.cached_tokens
            self._maybe_attach_ctx_usage(request, response)
            if logits_snapshot is not None:
                response.result.generation_logits = logits_snapshot
            new_responses.append((request.py_request_id, response))

        self._enqueue_responses(new_responses)

    @nvtx_range("_handle_responses")
    def _handle_responses(self, emit_first_iter: bool = True):
        new_responses = []
        requests_to_terminate = []
        # Requests terminated by _check_disagg_ctx_cache_transfer_status (DISAGG_CONTEXT_COMPLETE);
        # included in the return value for stats but not re-terminated here.
        requests_finished_by_transfer = []
        new_active_requests = []
        timed_out_requests = []
        logger.debug(
            f'------before _handle_responses, rank = {self.dist.rank}, output = {self.active_requests}'
        )

        batch_token_time = self.perf_manager.get_timestamp()

        for request in self.active_requests:
            req_id = request.py_request_id
            # no responses for dummy request, and finish it
            if request.is_attention_dp_dummy:
                requests_to_terminate.append(request)
                continue

            # Check if a generation request needs cleanup due to KV cache transfer timeout.
            if request.py_kv_transfer_timed_out:
                if self._is_disagg_inflight_cancel_active():
                    if (request.is_disagg_generation_transmission_in_progress
                            or request.state
                            == LlmRequestState.DISAGG_TRANS_ERROR):
                        new_active_requests.append(request)
                    else:
                        # The transfer completed after the deadline but before
                        # cancellation won the race. Fail the request without
                        # touching the now-quiesced transfer buffer.
                        timed_out_requests.append(request)
                    continue

                is_cancelled = self._request_kv_transfer_cancellation(request)
                if is_cancelled:
                    # _handle_errors enters response collectives under ADP.
                    # Defer it until the rank-uniform vote below.
                    timed_out_requests.append(request)
                else:
                    # Legacy transceivers cannot cancel every in-flight
                    # transfer. Keep polling instead of dropping ownership of
                    # the request and its KV resources.
                    new_active_requests.append(request)
                continue

            if request.is_generation_only_request() and not request.is_finished:
                # If request is in transmission, so we don't need to emit a response
                # Also, for the first iteration with overlap, we should skip since first
                # token has already been emitted previously
                if request.is_disagg_generation_transmission_in_progress or (
                        not self.disable_overlap_scheduler
                        and request.py_decoding_iter <= 1):
                    self.perf_manager.append_step_metrics(
                        request,
                        self.iter_counter,
                        batch_token_time=batch_token_time)
                    new_active_requests.append(request)
                    continue

            request.draft_tokens = request.py_draft_tokens or []
            request.decoding_iter = request.py_decoding_iter

            py_num_accepted = getattr(request, 'py_num_accepted_draft_tokens',
                                      0)
            draft_len = get_draft_token_length(request)
            if draft_len > 0:
                for pos in range(min(draft_len, MAX_SPEC_DECODE_POSITIONS)):
                    request.py_per_pos_drafted[pos] += 1
                for pos in range(min(py_num_accepted,
                                     MAX_SPEC_DECODE_POSITIONS)):
                    request.py_per_pos_accepted[pos] += 1

            self.perf_manager.append_step_metrics(
                request, self.iter_counter, batch_token_time=batch_token_time)

            # Ensure C++ perf metrics (lastTokenTime, etc.) are always updated
            # independently of whether append_step_metrics early-returned.
            # This is critical for E2E latency computation in tracing/Prometheus.
            if request.return_perf_metrics and request.py_decoding_iter >= 1:
                request.update_perf_metrics(self.iter_counter)

            request_done = False
            should_emit = (request.py_decoding_iter == 1 or request.is_finished
                           or request.py_decoding_iter % self.stream_interval
                           == 0)
            # The early-emit prototype issues the (non-terminal) iter-1
            # response from `_emit_first_token_responses`; suppress it here.
            if (not emit_first_iter and request.py_decoding_iter == 1
                    and not request.is_finished):
                should_emit = False
            if should_emit:
                if request.return_perf_metrics:
                    # Response creation may finalize and copy scalar ctx GPU totals.
                    self.perf_manager.compute_batch_gpu_times([request])
                response = request.create_response(False, self.dist.rank)
                if response:
                    request_done = request.is_finished
                    response.result.cached_tokens = request.cached_tokens
                    self._maybe_attach_ctx_usage(request, response)
                    response.result.per_pos_drafted = request.py_per_pos_drafted
                    response.result.per_pos_accepted = request.py_per_pos_accepted
                    new_responses.append((req_id, response))

            if request_done:
                # PP=1-only early termination; _end_transfer_and_maybe_terminate
                # gates on the same flag so the request terminates exactly once.
                force_terminate_for_partial_reuse = (
                    self.force_terminate_ctx_for_partial_reuse)
                if request.is_disagg_context_complete_state:
                    # Already terminated by _check_disagg_ctx_cache_transfer_status;
                    # track for stats only to avoid double-free (nvbug/5961736).
                    requests_finished_by_transfer.append(request)
                elif force_terminate_for_partial_reuse:
                    requests_to_terminate.append(request)
                elif not request.is_disagg_context_transmission_state:
                    requests_to_terminate.append(request)
            else:
                new_active_requests.append(request)

        self.active_requests.clear()
        self.active_requests.extend(new_active_requests)
        # Request should be terminated after enqueueing response to ensure we can enqueue response successfully.
        self._enqueue_responses(new_responses)
        for request in requests_to_terminate:
            self._terminate_request(request)
        if (self.kv_cache_transceiver is not None and self.enable_attention_dp
                and self.dist.world_size != 1):
            # Buffer for _handle_kv_transfer_timeouts_synced; in-place
            # tp_allgather would desync (reached from per-rank-divergent gates).
            self._pending_timed_out_requests.extend(timed_out_requests)
        else:
            for req in timed_out_requests:
                self._handle_errors(
                    error_msg=f"Request {req.py_request_id} timed out",
                    requests=[req],
                    charge_budget=False)
        return requests_to_terminate + requests_finished_by_transfer

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
                # Wake on shutdown too so that an event-loop crash
                # cannot trap callers here forever (nvbug 6038228).
                return id in self.responses or self.is_shutdown

            self.response_cv.wait_for(key_has_response, timeout=timeout)
            if id in self.responses:
                return self.responses.pop(id)
            if self.is_shutdown:
                # The event-loop thread terminated before producing a
                # response for this request. Re-raise the original
                # exception (if any) so callers see a meaningful error
                # instead of hanging here or hitting a KeyError below.
                error = self._event_loop_error
                if error is not None:
                    raise RuntimeError(
                        f"Event loop terminated with error: {error}") from error
                raise RuntimeError(
                    f"Event loop shut down before a response was received "
                    f"for request {id}.")
            # Timed out without shutdown. Return [] to match the
            # documented timeout contract used by _await_any_response and
            # the other executor-API timeouts; the pre-fix code raised a
            # bare KeyError on this path, which all known callers had to
            # defend against anyway.
            return []

    def _terminate_requests(self, requests_to_terminate):
        # todo: support work with self.inflight_req_ids.
        #       Currently, self.inflight_req_ids is not updated.
        for req in requests_to_terminate:
            self._terminate_request(req)

    def _pause_requests(self, requests_to_pause):
        for req in requests_to_pause:
            req.pause(self.max_input_len)

    def _add_inflight_ids(self, scheduled_requests: ScheduledRequests):
        """Add request IDs of current sampling requests to self.inflight_req_ids.

        Non-final context chunks should not be added to the inflight set, so the scheduler can keep scheduling
        further context chunks while earlier ones are in the PP pipeline.
        Only requests that sample new tokens should be added to the inflight set since their next iteration depends
        on these new tokens, so they should be skipped in the scheduler until the new tokens are generated.
        This includes context requests that finish context phase and generation requests.
        """
        for req in scheduled_requests.context_requests_last_chunk:
            logger.debug(
                f"Context request with ID {req.request_id} added to DECODER model inflight set"
            )
            self.inflight_req_ids.insert(req.request_id)
        for req in scheduled_requests.generation_requests:
            logger.debug(
                f"Generation request with ID {req.request_id} added to DECODER model inflight set"
            )
            self.inflight_req_ids.insert(req.request_id)

    def _remove_inflight_ids(self, scheduled_requests: ScheduledRequests):
        """Remove request IDs of current sampling requests from self.inflight_req_ids."""
        for req in scheduled_requests.context_requests_last_chunk:
            logger.debug(
                f"Context request with ID {req.request_id} removed from DECODER model inflight set"
            )
            self.inflight_req_ids.erase(req.request_id)
        for req in scheduled_requests.generation_requests:
            logger.debug(
                f"Generation request with ID {req.request_id} removed from DECODER model inflight set"
            )
            self.inflight_req_ids.erase(req.request_id)

    def _handle_speculative_decoding(
        self, scheduled_batch, previous_tensors, target_inputs
    ) -> Tuple[Optional[SampleStateTensorsSpec], Optional[torch.Tensor]]:
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
            self._handle_errors(error_msg,
                                requests=[request],
                                charge_budget=False)


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
        self._comm_tag = PPCommTag.TERMINATION

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
