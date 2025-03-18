import dataclasses
import datetime
import functools
import gc
import os
import queue
import threading
import time
import traceback
import weakref
from contextlib import contextmanager
from itertools import chain
from typing import Dict, List, Optional, Union

import torch

import tensorrt_llm.bindings.executor as trtllm

from ..._utils import global_mpi_rank, nvtx_range
from ...logger import logger
from .decoder import *
from .distributed import *
from .kv_cache_transceiver import KvCacheTransceiver
from .llm_request import *
from .model_engine import *
from .resource_manager import *
from .scheduler import *


def _is_executor_request(req_queue_item) -> bool:
    return isinstance(req_queue_item, tuple)


def _is_cancel_request(req_queue_item) -> bool:
    return isinstance(req_queue_item, int)


def _get_from_request_queue(request_queue, timeout: datetime.timedelta,
                            max_req_count: int):
    items = []
    timeout = timeout.total_seconds() if timeout is not None else None
    req_count = 0
    try:
        if request_queue.empty() and (timeout is None or timeout > 0):
            # if queue is empty and want to wait, wait
            items.append(request_queue.get(timeout=timeout))
        else:
            # if not empty or don't want to wait, just return all items in queue
            while req_count < max_req_count:
                queue_item = request_queue.get_nowait()
                items.append(queue_item)
                if _is_executor_request(queue_item):
                    # if it is request, (Not finish signal or cancel signal)
                    req_count += 1
    except queue.Empty:
        pass
    return items


PROFILE_START_STOP_ENV_VAR_NAME = "TLLM_PROFILE_START_STOP"


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


PROFILE_RECORD_GC_ENV_VAR_NAME = "TLLM_PROFILE_RECORD_GC"


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
class RuntimeRequest:
    request_id: int
    request: ExecutorRequest
    response: ExecutorResponse


class PyExecutor:

    def __init__(self,
                 resource_manager,
                 scheduler,
                 model_engine: ModelEngine,
                 decoder: Decoder,
                 dist: Distributed,
                 enable_overlap_scheduler: bool = False,
                 max_input_len: int = 2048,
                 kv_cache_transceiver: KvCacheTransceiver = None):
        super(PyExecutor, self).__init__()
        self.device_id = torch.cuda.current_device()
        self.global_rank = global_mpi_rank()
        self.request_queue = queue.Queue()

        # profile config
        self.profile_start_iters, self.profile_stop_iters = _load_iteration_indexes(
            PROFILE_START_STOP_ENV_VAR_NAME)
        self.gc_nvtx_watcher_handle = _gc_nvtx_watcher()

        # related modules
        self.resource_manager = resource_manager
        self.scheduler = scheduler
        self.model_engine = model_engine
        self.enable_attention_dp = model_engine.enable_attention_dp
        self.decoder = decoder
        self.dist = dist

        # enqueue and _fetch_new_requests used data
        self.enqueue_lock = threading.Lock()
        self.active = True
        self.next_req_id = 1
        self.print_log = model_engine.pytorch_backend_config.print_iter_log
        self.enable_iter_perf_stats = model_engine.pytorch_backend_config.enable_iter_perf_stats
        self.num_fetch_requests_cur_rank = 0
        self.num_fetch_requests = 0
        self.shutdown_event = threading.Event()

        # response used data
        self.response_lock = threading.Lock()
        self.response_cv = threading.Condition(self.response_lock)
        self.responses = {}

        # kv cache events
        kv_cache_manager = self.resource_manager.resource_managers.get(
            "kv_cache_manager")
        self.enable_kv_cache_events = kv_cache_manager is not None and kv_cache_manager.event_buffer_max_size > 0

        # todo: we need pass this by builder config from LLM and LLMargs
        self.max_input_len = max_input_len
        # _executor_loop private data
        self.max_num_active_requests = model_engine.get_max_num_sequences()
        self.active_requests = []
        self.all_ranks_num_active_requests = [
            0
        ] * self.dist.world_size if self.enable_attention_dp else []
        self.ctx_in_transmission_requests = []
        self.previous_batch = None
        # list of requests in each micro batch
        self.micro_batches = [None] * self.dist.pp_size
        self.send_handles = [None] * self.dist.pp_size

        self.inflight_req_ids = tensorrt_llm.bindings.internal.batch_manager.ReqIdsSet(
        )
        self.canceled_req_ids = tensorrt_llm.bindings.internal.batch_manager.ReqIdsSet(
        )

        self.model_engine.warmup(self.resource_manager)

        self.is_shutdown = False

        self.kv_cache_transceiver = kv_cache_transceiver
        if self.dist.pp_size > 1:
            event_loop = self._executor_loop_pp
        elif kv_cache_transceiver is not None:
            event_loop = self._executor_disagg_loop_overlap if enable_overlap_scheduler else self._executor_disagg_loop
        else:
            event_loop = self._executor_loop_overlap if enable_overlap_scheduler else self._executor_loop
        self.worker_thread = threading.Thread(target=event_loop, daemon=True)
        self.worker_thread.start()
        self.stats_lock = threading.Lock()
        self.stats = []
        self.start_times = {}
        self.new_active_requests_queue_latency_ms = 0

    def __enter__(self):
        return self

    def __exit__(self):
        self.shutdown()

    def wait_shutdown(self):
        self.shutdown_event.wait()

    @contextmanager
    def _profiler(self):
        it = -1
        enabled = False
        start_time = None

        def profile_step():
            nonlocal it, enabled, start_time
            if it in self.profile_stop_iters:
                assert enabled, "Inconsistent CUDA profiling state"
                torch.cuda.cudart().cudaProfilerStop()
                enabled = False

            if start_time is not None and self.print_log and self.dist.rank == 0:
                end_time = time.time()

                formatted_timestamp = datetime.datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S")
                print(
                    f'iter = {self.model_engine.iter_counter}, global_rank = {self.global_rank}, rank = {self.dist.rank}, currank_total_requests = {self.num_fetch_requests_cur_rank}/{self.num_fetch_requests}, elapsed_time = {end_time - start_time}s, timestamp = {formatted_timestamp}, states = {self.model_engine.iter_states}'
                )

            it += 1

            if it in self.profile_start_iters:
                assert not enabled, "Inconsistent CUDA profiling state"
                torch.cuda.cudart().cudaProfilerStart()
                enabled = True
            start_time = time.time()

        try:
            yield profile_step
        finally:
            if enabled:
                # Stop on early exit / exception
                torch.cuda.cudart().cudaProfilerStop()

    def _get_init_iter_stats(self, num_new_active_requests,
                             new_active_requests_queue_latency_ms):
        stats = trtllm.IterationStats()
        stats.timestamp = ""

        stats.num_new_active_requests = num_new_active_requests
        stats.num_active_requests = len(self.active_requests)
        stats.new_active_requests_queue_latency_ms = new_active_requests_queue_latency_ms
        return stats

    def _update_iter_stats(self, stats, iter_latency_ms, num_completed_requests,
                           scheduled_batch):
        stats.iter_latency_ms = iter_latency_ms

        stats.num_queued_requests = self.request_queue.qsize()
        stats.num_completed_requests = num_completed_requests
        stats.max_num_active_requests = self.max_num_active_requests

        end, total_gpu_memory = torch.cuda.mem_get_info()
        stats.gpu_mem_usage = total_gpu_memory - end
        stats.cpu_mem_usage = 0
        stats.pinned_mem_usage = 0

        stats.iter = self.model_engine.iter_counter

        kv_cache_manager = self.resource_manager.resource_managers.get(
            "kv_cache_manager")
        if kv_cache_manager is not None:
            kv_stats = kv_cache_manager.get_kv_cache_stats()
            kv_stats_to_save = trtllm.KvCacheStats()
            kv_stats_to_save.max_num_blocks = kv_stats.max_num_blocks
            kv_stats_to_save.free_num_blocks = kv_stats.free_num_blocks
            kv_stats_to_save.used_num_blocks = kv_stats.used_num_blocks
            kv_stats_to_save.tokens_per_block = kv_stats.tokens_per_block
            kv_stats_to_save.alloc_total_blocks = kv_stats.alloc_total_blocks
            kv_stats_to_save.alloc_new_blocks = kv_stats.alloc_new_blocks
            kv_stats_to_save.reused_blocks = kv_stats.reused_blocks
            stats.kv_cache_stats = kv_stats_to_save

        model_stats = trtllm.InflightBatchingStats()
        model_stats.num_scheduled_requests = len(
            scheduled_batch.context_requests) + len(
                scheduled_batch.generation_requests)
        model_stats.num_context_requests = len(scheduled_batch.context_requests)
        model_stats.num_gen_requests = len(scheduled_batch.generation_requests)
        model_stats.num_paused_requests = len(scheduled_batch.paused_requests)
        model_stats.avg_num_decoded_tokens_per_iter = 0
        model_stats.num_ctx_tokens = 0
        model_stats.micro_batch_id = 0
        stats.inflight_batching_stats = model_stats
        return stats

    def _append_iter_stats(self, stats):
        try:
            self.stats_lock.acquire()
            self.stats.append(stats)
        finally:
            self.stats_lock.release()

    def _process_iter_stats(self, finished_requests, scheduled_batch,
                            iter_start_time, iter_stats):
        iter_end_time = time.time()
        iter_latency_ms = iter_end_time - iter_start_time
        self._append_iter_stats(
            self._update_iter_stats(iter_stats, iter_latency_ms,
                                    len(finished_requests), scheduled_batch))

        return

    def _executor_loop_pp(self):
        torch.cuda.set_device(self.device_id)
        got_finish_signal = False
        attn_dp_idle_iter = False
        microbatch_id = 0
        with self._profiler() as profile_step:
            iter_start_time = time.time()
            iter_end_time = iter_start_time
            iter_stats = None
            while not got_finish_signal or len(self.active_requests) > 0:
                profile_step()
                if self.enable_attention_dp:
                    new_requests = self._fetch_adp_new_requests()
                else:
                    new_requests = self._fetch_new_requests()
                got_finish_signal = self._merge_requests(
                    new_requests) or got_finish_signal
                if got_finish_signal and len(self.active_requests) == 0:
                    break

                finished_requests = []
                if self.enable_iter_perf_stats:
                    iter_stats = self._get_init_iter_stats(
                        len(new_requests),
                        self.new_active_requests_queue_latency_ms)
                attn_dp_idle_iter = ((not got_finish_signal)
                                     and len(self.active_requests) == 0
                                     and self.enable_attention_dp)
                if attn_dp_idle_iter:
                    self._merge_one_dummy_request()
                scheduled_batch, _, _ = self._schedule()

                if scheduled_batch.batch_size == 0:
                    assert len(self.inflight_req_ids) > 0, (
                        "fail to schedule any pending request, probably run out of resource"
                    )
                    self.micro_batches[microbatch_id] = None
                else:
                    #TODO: add pause_requests together with inflight_req_ids for pp
                    self._add_inflight_ids(
                        scheduled_batch)  # lock inflight requests
                    # TODO: handle draft_tokens (speculative decoding) and add pause_requests handling
                    self.resource_manager.prepare_resources(scheduled_batch)

                    # Stage 1: Forward + (decoding) pass ([should be] async)
                    if self.dist.is_last_pp_rank:
                        scheduled_batch, new_tensors_host, finished_requests = self._forward_step_last_pp(
                            scheduled_batch, microbatch_id)
                        self.resource_manager.update_resources(scheduled_batch)
                    else:
                        new_tensors_host = self._forward_step_inter_pp(
                            scheduled_batch)
                    self.micro_batches[microbatch_id] = (scheduled_batch,
                                                         new_tensors_host)

                # marching forward in the microbatch slots
                prev_microbatch_id = (microbatch_id + 1) % self.dist.pp_size
                previous_batch = self.micro_batches[prev_microbatch_id]
                # Stage 2: Handle previous batch that only processed forward_step
                if previous_batch is not None:
                    previous_scheduled_batch, previous_new_tensors_host = previous_batch
                    if not self.dist.is_last_pp_rank:
                        finished_requests = self._handle_previous_batch_inter_pp(
                            previous_scheduled_batch, previous_new_tensors_host,
                            prev_microbatch_id)
                        self.resource_manager.update_resources(
                            previous_scheduled_batch)
                    self._remove_inflight_ids(
                        previous_scheduled_batch)  # unlock inflight requests
                microbatch_id = prev_microbatch_id

                if self.enable_iter_perf_stats:
                    iter_end_time = time.time()
                    iter_latency_ms = iter_end_time - iter_start_time
                    self._append_iter_stats(
                        self._update_iter_stats(iter_stats, iter_latency_ms,
                                                len(finished_requests),
                                                scheduled_batch))
                    iter_start_time = iter_end_time
        # Cleanup
        with self.response_cv:
            self.is_shutdown = True
            self.response_cv.notify_all()
        self.shutdown_event.set()

    def _executor_loop(self):
        torch.cuda.set_device(self.device_id)
        got_finish_signal = False
        attn_dp_idle_iter = False
        with self._profiler() as profile_step:
            iter_start_time = time.time()
            iter_stats = None
            while not got_finish_signal or len(self.active_requests) > 0:
                profile_step()
                if self.enable_iter_perf_stats:
                    iter_start_time = time.time()
                if self.enable_attention_dp:
                    new_requests = self._fetch_adp_new_requests()
                else:
                    new_requests = self._fetch_new_requests()
                got_finish_signal = self._merge_requests(
                    new_requests) or got_finish_signal
                if got_finish_signal and len(self.active_requests) == 0:
                    break

                if self.enable_iter_perf_stats:
                    iter_stats = self._get_init_iter_stats(
                        len(new_requests),
                        self.new_active_requests_queue_latency_ms)

                attn_dp_idle_iter = ((not got_finish_signal)
                                     and len(self.active_requests) == 0
                                     and self.enable_attention_dp)
                if attn_dp_idle_iter:
                    self._merge_one_dummy_request()
                scheduled_batch, _, _ = self._schedule()

                assert scheduled_batch.batch_size > 0, (
                    "fail to schedule any pending request, "
                    "probably run out of resource.")

                self.pause_requests(scheduled_batch.paused_requests)
                self.resource_manager.prepare_resources(scheduled_batch)
                batch_outputs = self._forward_step(scheduled_batch)
                self._decode(scheduled_batch, batch_outputs)
                self._handle_cancelled_requests()

                finished_requests = self._handle_responses()
                self.resource_manager.update_resources(scheduled_batch)

                self._gather_dp_requests_num()

                if self.enable_kv_cache_events:
                    self._add_kv_cache_events()

                if self.enable_iter_perf_stats:
                    self._process_iter_stats(finished_requests, scheduled_batch,
                                             iter_start_time, iter_stats)

        with self.response_cv:
            self.is_shutdown = True
            self.response_cv.notify_all()
        self.shutdown_event.set()

    def _executor_loop_overlap(self):
        torch.cuda.set_device(self.device_id)
        got_finish_signal = False
        with self._profiler() as profile_step:
            iter_start_time = time.time()
            iter_stats = None
            while not got_finish_signal or len(self.active_requests) > 0:
                profile_step()
                if self.enable_iter_perf_stats:
                    iter_start_time = time.time()
                if self.enable_attention_dp:
                    new_requests = self._fetch_adp_new_requests()
                else:
                    new_requests = self._fetch_new_requests()
                got_finish_signal = self._merge_requests(
                    new_requests) or got_finish_signal
                if got_finish_signal and len(self.active_requests) == 0:
                    break

                if self.enable_iter_perf_stats:
                    iter_stats = self._get_init_iter_stats(
                        len(new_requests),
                        self.new_active_requests_queue_latency_ms)

                attn_dp_idle_iter = ((not got_finish_signal)
                                     and len(self.active_requests) == 0
                                     and self.enable_attention_dp)

                if attn_dp_idle_iter:
                    self._merge_one_dummy_request()

                scheduled_batch, _, _ = self._schedule()

                assert scheduled_batch.batch_size > 0, (
                    "fail to schedule any pending request, "
                    "probably run out of resource.")
                logger.debug(
                    f'has {len(self.active_requests)} active_request, '
                    f'scheduled {len(scheduled_batch.context_requests)} context requests and '
                    f'{len(scheduled_batch.generation_requests)} generation requests'
                )

                previous_new_tensors_device = None
                if self.previous_batch is not None:
                    _, previous_new_tensors_device, _, _, _, _ = self.previous_batch

                self.resource_manager.prepare_resources(scheduled_batch)
                batch_outputs = self._forward_step(scheduled_batch,
                                                   previous_new_tensors_device)

                new_tensors_device, new_tensors_host, decoder_event = self._decode_async(
                    scheduled_batch, batch_outputs)

                if attn_dp_idle_iter:
                    self._finish_one_dummy_request(scheduled_batch)
                has_previous_batch = self.previous_batch is not None
                if has_previous_batch:
                    previous_scheduled_batch, _, previous_new_tensors_host, previous_decoder_event, previous_iter_start_time, previous_iter_stats = self.previous_batch
                    self._update_requests(previous_scheduled_batch,
                                          previous_new_tensors_host,
                                          previous_decoder_event)
                    self._handle_cancelled_requests()
                    finished_requests = self._handle_responses()
                    self.resource_manager.update_resources(
                        previous_scheduled_batch)
                    if self.enable_kv_cache_events:
                        self._add_kv_cache_events()

                    if self.enable_iter_perf_stats:
                        self._process_iter_stats(finished_requests,
                                                 previous_scheduled_batch,
                                                 previous_iter_start_time,
                                                 previous_iter_stats)

                # Separate chunked requests so we can handle them in _update_requests w/o relying on the request state.
                # This is necessary because _forward_step updates the state before _update_requests is executed.
                scheduled_batch.chunked_requests = [
                    r for r in scheduled_batch.context_requests
                    if r.get_context_remaining_length() != 0
                ]
                scheduled_batch.context_requests = [
                    r for r in scheduled_batch.context_requests
                    if r.get_context_remaining_length() == 0
                ]

                self.previous_batch = (scheduled_batch, new_tensors_device,
                                       new_tensors_host, decoder_event,
                                       iter_start_time, iter_stats)
                self._gather_dp_requests_num()

        with self.response_cv:
            self.is_shutdown = True
            self.response_cv.notify_all()
        self.shutdown_event.set()

    def _executor_disagg_loop(self):
        torch.cuda.set_device(self.device_id)
        got_finish_signal = False
        with self._profiler() as profile_step:
            iter_start_time = time.time()
            iter_stats = None
            while not got_finish_signal or len(self.active_requests) > 0:
                profile_step()
                if self.enable_iter_perf_stats:
                    iter_start_time = time.time()
                if self.enable_attention_dp:
                    new_requests = self._fetch_adp_new_requests()
                else:
                    new_requests = self._fetch_new_requests()
                got_finish_signal = self._merge_requests(
                    new_requests) or got_finish_signal
                if got_finish_signal and len(self.active_requests) == 0:
                    break

                if self.enable_iter_perf_stats:
                    iter_stats = self._get_init_iter_stats(
                        len(new_requests),
                        self.new_active_requests_queue_latency_ms)

                self._check_disagg_gen_transfer_status()

                attn_dp_idle_iter = (
                    not got_finish_signal
                ) and self.enable_attention_dp and self._check_need_one_dummy_request(
                )
                # TODO: if the requests in  all dp rank are all in disagg_generation_init, we don't need to merge one dummy request
                if attn_dp_idle_iter:
                    self._merge_one_dummy_request()

                scheduled_batch, fitting_disagg_gen_init_requests, num_fitting_reqs = self._schedule(
                )

                # For requests that are fitting disagg gen init, also prepare resources for KV cache manager
                self._prepare_disagg_gen_init(fitting_disagg_gen_init_requests)
                if num_fitting_reqs == 0 and (
                    (fitting_disagg_gen_init_requests == None)
                        or len(fitting_disagg_gen_init_requests) == 0):
                    # TODO: should be true , to free kvCache
                    logger.warning(
                        "num_fitting_reqs =0 and fitting_disagg_gen_init_requests is empty , may not have enough kvCache"
                    )
                    self.kv_cache_transceiver.check_context_transfer_status(
                        True)

                self.pause_requests(scheduled_batch.paused_requests)

                finished_requests = []

                if scheduled_batch.batch_size > 0:
                    self.resource_manager.prepare_resources(scheduled_batch)

                    # For generation requests which have completed KV cache transfer
                    self._prepare_disagg_gen_transmission_complete(
                        scheduled_batch)

                    batch_outputs = self._forward_step(scheduled_batch)

                    self._send_disagg_ctx_cache(
                        scheduled_batch.context_requests)

                    # Keep track of ctx requests that are in transmission
                    ctx_transmission_reqs = [
                        req for req in scheduled_batch.context_requests
                        if req.state ==
                        LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS
                    ]

                    self._decode(scheduled_batch, batch_outputs)

                    # For context only req in transmission, we reset the state since decoder might have changed it
                    for req in ctx_transmission_reqs:
                        req.state = LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS

                    self._handle_cancelled_requests()
                    finished_requests = self._handle_responses()
                    self.resource_manager.update_resources(scheduled_batch)
                    if self.enable_kv_cache_events:
                        self._add_kv_cache_events()

                if self.ctx_in_transmission_requests:
                    self._terminate_ctx_finished_requests()

                self._gather_dp_requests_num()

                if self.enable_iter_perf_stats:
                    self._process_iter_stats(finished_requests, scheduled_batch,
                                             iter_start_time, iter_stats)

        with self.response_cv:
            self.is_shutdown = True
            self.response_cv.notify_all()
        self.shutdown_event.set()

    def _process_previous_batch(self):
        previous_scheduled_batch, _, previous_new_tokens_host, previous_decoder_event, previous_iter_start_time, previous_iter_stats, previous_ctx_transmission_reqs = self.previous_batch

        self._update_requests(previous_scheduled_batch,
                              previous_new_tokens_host, previous_decoder_event)
        if previous_ctx_transmission_reqs:
            for req in previous_ctx_transmission_reqs:
                req.state = LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS
        self._handle_cancelled_requests()
        finished_requests = self._handle_responses()
        self.resource_manager.update_resources(previous_scheduled_batch)

        if self.enable_iter_perf_stats:
            self._process_iter_stats(finished_requests,
                                     previous_scheduled_batch,
                                     previous_iter_start_time,
                                     previous_iter_stats)

        self.previous_batch = None

        return

    def _executor_disagg_loop_overlap(self):
        torch.cuda.set_device(self.device_id)
        got_finish_signal = False
        with self._profiler() as profile_step:
            iter_start_time = time.time()
            iter_stats = None
            while not got_finish_signal or len(self.active_requests) > 0:
                profile_step()

                if self.enable_iter_perf_stats:
                    iter_start_time = time.time()

                if self.enable_attention_dp:
                    new_requests = self._fetch_adp_new_requests()
                else:
                    new_requests = self._fetch_new_requests()
                got_finish_signal = self._merge_requests(
                    new_requests) or got_finish_signal
                if got_finish_signal and len(self.active_requests) == 0:
                    break

                self._check_disagg_gen_transfer_status()

                attn_dp_idle_iter = (
                    not got_finish_signal
                ) and self.enable_attention_dp and self._check_need_one_dummy_request(
                )

                # TODO: if the requests in  all dp rank are all in disagg_generation_init, we don't need to merge one dummy request
                if attn_dp_idle_iter:
                    self._merge_one_dummy_request()

                if self.enable_iter_perf_stats:
                    iter_stats = self._get_init_iter_stats(
                        len(new_requests),
                        self.new_active_requests_queue_latency_ms)

                scheduled_batch, fitting_disagg_gen_init_requests, num_fitting_reqs = self._schedule(
                )

                # For requests that are fitting disagg gen init, also prepare resources for KV cache manager
                self._prepare_disagg_gen_init(fitting_disagg_gen_init_requests)

                if num_fitting_reqs == 0 and (
                    (fitting_disagg_gen_init_requests == None)
                        or len(fitting_disagg_gen_init_requests) == 0):
                    # TODO: should be true , to free kvCache
                    logger.warning(
                        "num_fitting_reqs =0 and fitting_disagg_gen_init_requests is empty , may not have enough kvCache"
                    )
                    self.kv_cache_transceiver.check_context_transfer_status(
                        True)

                self.pause_requests(scheduled_batch.paused_requests)

                if scheduled_batch.batch_size > 0:
                    self.resource_manager.prepare_resources(scheduled_batch)

                    # For generation requests which have completed KV cache transfer
                    self._prepare_disagg_gen_transmission_complete(
                        scheduled_batch)

                    previous_new_tokens_device = None
                    if self.previous_batch is not None:
                        _, previous_new_tokens_device, _, _, _, _, _ = self.previous_batch

                    batch_outputs = self._forward_step(
                        scheduled_batch, previous_new_tokens_device)

                    self._send_disagg_ctx_cache(
                        scheduled_batch.context_requests)

                    # Keep track of ctx requests that are in transmission
                    ctx_transmission_reqs = [
                        req for req in scheduled_batch.context_requests
                        if req.state ==
                        LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS
                    ]

                    new_tokens_device, new_tokens_host, decoder_event = self._decode_async(
                        scheduled_batch, batch_outputs)

                    if attn_dp_idle_iter:
                        self._finish_one_dummy_request(scheduled_batch)

                    has_previous_batch = self.previous_batch is not None
                    if has_previous_batch:
                        self._process_previous_batch()

                    self.previous_batch = (scheduled_batch, new_tokens_device,
                                           new_tokens_host, decoder_event,
                                           iter_start_time, iter_stats,
                                           ctx_transmission_reqs)
                    self._gather_dp_requests_num()
                if self.ctx_in_transmission_requests:
                    self._terminate_ctx_finished_requests()

        with self.response_cv:
            self.is_shutdown = True
            self.response_cv.notify_all()
        self.shutdown_event.set()

    def _handle_previous_batch_inter_pp(self, previous_scheduled_batch,
                                        previous_new_tensors_host,
                                        prev_microbatch_id):
        # Receive tokens from previous pp rank (or next pp rank w.r.t model forward direction)
        self.dist.recv_tensor(
            previous_new_tensors_host["new_tokens_host"],
            src=self.dist.next_pp_rank,
            tag=prev_microbatch_id  # not necessary and may discard
        )

        # Send tokens to next rank if not first (or previous rank w.r.t model forward direction)
        if self.send_handles[prev_microbatch_id] is not None:
            self.send_handles[prev_microbatch_id].Wait()
        if not self.dist.is_first_pp_rank:
            self.send_handles[prev_microbatch_id] = self.dist.isend_tensor(
                tensor=previous_new_tensors_host["new_tokens_host"],
                dest=self.dist.prev_pp_rank,
                tag=prev_microbatch_id)
        # TODO: how to handle draft_tokens (speculative decoding)?
        self._update_requests(previous_scheduled_batch,
                              previous_new_tensors_host, None)
        self._handle_cancelled_requests()
        finished_requests = self._handle_responses()
        return finished_requests

    @nvtx_range("_forward_step_last_pp")
    def _forward_step_last_pp(self, scheduled_batch, microbatch_id):
        batch_outputs = self._forward_step(scheduled_batch)
        _, new_tensors_host, decoder_event = self._decode_async(
            scheduled_batch, batch_outputs)
        if self.send_handles[microbatch_id] is not None:
            self.send_handles[microbatch_id].Wait()
        decoder_event.synchronize()

        self.send_handles[microbatch_id] = self.dist.isend_tensor(
            new_tensors_host["new_tokens_host"],
            dest=self.dist.prev_pp_rank,
            tag=microbatch_id)

        self._update_requests(scheduled_batch, new_tensors_host, None)
        self._handle_cancelled_requests()
        finished_requests = self._handle_responses()
        return scheduled_batch, new_tensors_host, finished_requests

    @nvtx_range("_forward_step_inter_pp")
    def _forward_step_inter_pp(self, scheduled_batch):
        batch_outputs = self._forward_step(scheduled_batch)
        tokens_shape = batch_outputs["hidden_states"].shape[:-1]
        new_tokens_host = torch.empty(tokens_shape,
                                      dtype=torch.int64,
                                      device='cpu',
                                      pin_memory=True)
        return {"new_tokens_host": new_tokens_host}

    @nvtx_range("_fetch_new_requests")
    def _fetch_new_requests(self):
        timeout = None if len(
            self.active_requests) == 0 else datetime.timedelta(0)
        new_requests = []
        if self.dist.rank == 0:
            new_requests = _get_from_request_queue(
                self.request_queue, timeout,
                self.max_num_active_requests - len(self.active_requests))

        new_requests = self.dist.broadcast(new_requests, root=0)

        if self.enable_iter_perf_stats and self.dist.rank == 0:
            now = time.time()
            for req in new_requests:
                if isinstance(req, tuple):
                    req_id = req[0]
                    if req_id in self.start_times:
                        self.new_active_requests_queue_latency_ms += now - self.start_times.pop(
                            req_id)

        return new_requests

    @nvtx_range("_fetch_adp_new_requests")
    def _fetch_adp_new_requests(self):
        total_num_active_requests = sum(self.all_ranks_num_active_requests)
        total_max_num_active_requests = self.dist.world_size * self.max_num_active_requests
        timeout = None if total_num_active_requests == 0 else datetime.timedelta(
            0)
        new_requests = []
        if self.dist.rank == 0:
            new_requests = _get_from_request_queue(
                self.request_queue, timeout,
                total_max_num_active_requests - total_num_active_requests)

        new_requests = self.dist.broadcast(new_requests, root=0)
        num_new_requests_all_ranks = len(new_requests)
        new_requests_cur_rank = []
        if new_requests != [] and new_requests[0] != None:
            now = time.time()
            for idx, request in enumerate(new_requests):
                if (idx + self.num_fetch_requests
                    ) % self.dist.world_size == self.dist.rank:
                    new_requests_cur_rank.append(request)

                    if self.enable_iter_perf_stats and self.dist.rank == 0:
                        self.new_active_requests_queue_latency_ms += now - self.start_times[
                            request[0]]
                        self.start_times.pop(request[0])

        self.num_fetch_requests = self.num_fetch_requests + num_new_requests_all_ranks
        self.num_fetch_requests_cur_rank = self.num_fetch_requests_cur_rank + len(
            new_requests_cur_rank)

        if len(new_requests) == 1 and new_requests[0] == None:
            new_requests_cur_rank = new_requests
        return new_requests_cur_rank

    @nvtx_range("_gather_dp_requests_num")
    def _gather_dp_requests_num(self):
        if self.enable_attention_dp:
            gather_active_requests = []
            resonses_list = self.dist.allgather(len(self.active_requests))
            for num_active_requests in resonses_list:
                gather_active_requests.append(num_active_requests)
            self.all_ranks_num_active_requests = gather_active_requests

    def _add_kv_cache_events(self):
        kv_cache_manager = self.resource_manager.resource_managers.get(
            "kv_cache_manager")
        if not kv_cache_manager:
            return
        # Flush iteration events at each iteration to ensure that events have enough time
        # to be transferred to main thread when user needs them.
        kv_cache_manager.flush_iteration_events()

    def _merge_tp_requests(self, new_requests: List[ExecutorRequest]):
        got_finish_signal = False
        for request in new_requests:
            # return finish signal and drop all request on shutdown
            if request is None:
                return True
        for req_item in new_requests:
            if _is_executor_request(req_item):
                req_id, exe_req = req_item
                req = executor_request_to_llm_request(req_id, exe_req)
                req.is_dummy = False
                self.active_requests.append(req)
            elif _is_cancel_request(req_item):
                self.canceled_req_ids.insert(req_item)

        return got_finish_signal

    def _merge_one_dummy_request(self):
        sampling_params = SamplingParams()
        kwargs = {
            "request_id":
            0,
            "max_new_tokens":
            1,
            "input_tokens": [1],
            "sampling_config":
            tensorrt_llm.bindings.SamplingConfig(
                sampling_params._get_sampling_config()),
            "is_streaming":
            False,
        }
        llm_request = LlmRequest(**kwargs)
        llm_request.is_dummy = True
        self.active_requests.append(llm_request)

    def _finish_one_dummy_request(self, scheduled_requests: ScheduledRequests):
        for req in scheduled_requests.context_requests:
            if req.is_dummy:
                req.state = LlmRequestState.GENERATION_COMPLETE

        for req in self.active_requests:
            if req.is_dummy:
                self._terminate_request(req)
                self.active_requests.remove(req)
                break

    def _remove_dummy_request(self, scheduled_requests):
        for request in scheduled_requests.context_requests:
            if request.is_dummy:
                scheduled_requests.context_requests.remove(request)
        for request in self.active_requests:
            if request.is_dummy:
                self.active_requests.remove(request)

    def _partition_context(self, ctx_ids_list):
        ctx_ids = torch.tensor(ctx_ids_list).unsqueeze(0)
        ctx_len = ctx_ids.shape[-1]
        block_size = self.dist.cp_config['block_size']
        if block_size is None:
            block_size = ctx_len // self.dist.cp_size
        anchor_block_size = self.dist.cp_config['cp_anchor_size']
        if anchor_block_size is None:
            anchor_block_size = block_size

        assert anchor_block_size <= block_size, f'cp_anchor_size {anchor_block_size} should be smaller than block_size {block_size}'
        padding = 0
        if ctx_len % block_size != 0:
            padding = block_size - (ctx_len % block_size)
            assert padding <= ctx_len, f'block size is too large for context, please set it smaller'
            ctx_ids = torch.cat(
                (ctx_ids, torch.zeros_like(ctx_ids)[:, :padding]), dim=-1)
        position_ids = torch.arange(0, ctx_ids.shape[-1]).unsqueeze(0)

        ctx_ids_blocks = torch.tensor_split(
            torch.stack(ctx_ids.split(block_size, dim=-1)), self.dist.cp_size)
        position_ids_blocks = torch.tensor_split(
            torch.stack(position_ids.split(block_size, dim=-1)),
            self.dist.cp_size)
        if self.dist.cp_rank != 0:
            ctx_blocks, position_blocks = [
                ctx_ids_blocks[0][0].tolist()[0][:anchor_block_size]
            ], [position_ids_blocks[0][0].tolist()[0][:anchor_block_size]]
        else:
            ctx_blocks, position_blocks = [], []

        for idx in range(len(ctx_ids_blocks[self.dist.cp_rank])):
            ctx_block = ctx_ids_blocks[self.dist.cp_rank][idx]
            position_block = position_ids_blocks[self.dist.cp_rank][idx]
            ctx_blocks.append(ctx_block.tolist()[0])
            position_blocks.append(position_block.tolist()[0])
            #(f'rank = {self.dist.cp_rank}, block_id = {idx}, block_size = {ctx_block.shape}, device = {ctx_block.get_device()}')
        return ctx_blocks, position_blocks, padding

    def _merge_star_attention_requests(self,
                                       new_requests: List[ExecutorRequest]):
        got_finish_signal = False
        for request in new_requests:
            # return finish signal and drop all request on shutdown
            if request is None:
                return True
        for req_item in new_requests:
            if _is_executor_request(req_item):
                req_id, exe_req, query_token_ids = req_item
                ctx_len0 = len(exe_req.input_token_ids)
                ctx_blocks, position_blocks, last_block_padding_num = [
                    exe_req.input_token_ids
                ], [[i for i in range(ctx_len0)]], 0
                ctx_blocks, position_blocks, last_block_padding_num = self._partition_context(
                    exe_req.input_token_ids)
                if self.dist.cp_rank == self.dist.cp_size - 1 and last_block_padding_num > 0:
                    ctx_blocks[-1] = ctx_blocks[-1][:-last_block_padding_num]
                    position_blocks[-1] = position_blocks[
                        -1][:-last_block_padding_num]
                #if has query
                if query_token_ids:
                    ctx_blocks.append(query_token_ids)
                    position_blocks.append([
                        i for i in range(ctx_len0, ctx_len0 +
                                         len(query_token_ids))
                    ])

                # insert the dummy block to align the number of ctx iterations of each rank
                block_size = self.dist.cp_config['block_size']
                total_blocks = (ctx_len0 + block_size - 1) // block_size
                num_blocks_per_rank = (
                    total_blocks + self.dist.cp_size -
                    1) // self.dist.cp_size + 1  # 1 for query block
                if len(ctx_blocks) == num_blocks_per_rank:
                    ctx_blocks.insert(1, [])
                    position_blocks.insert(1, [])
                elif len(ctx_blocks) == num_blocks_per_rank + 1:
                    # anchor + ctx_blocks + qry_block
                    pass
                else:
                    print(
                        f'rank = {self.dist.cp_rank}, len(ctx_blocks)  = {len(ctx_blocks) }, num_blocks_per_rank = {num_blocks_per_rank}'
                    )
                    assert False, f'invalid context partition'

                # fake data for scheduler
                ctx_blocks_list = [0] * (block_size +
                                         self.dist.cp_config['cp_anchor_size'])

                req = executor_request_to_llm_request(req_id, exe_req,
                                                      ctx_blocks_list)
                req.gen_iters = 0
                req.ctx_iters = 0
                req.ctx_blocks = ctx_blocks
                req.ctx_position_blocks = position_blocks
                req.query_id = query_token_ids
                req.is_dummy = False
                self.active_requests.append(req)
            elif _is_cancel_request(req_item):
                self.canceled_req_ids.insert(req_item)

        return got_finish_signal

    @nvtx_range("_merge_requests")
    def _merge_requests(self, new_requests: List[ExecutorRequest]):
        cp_config = self.dist.cp_config
        if 'cp_type' in cp_config:
            cp_type = cp_config['cp_type']
            if cp_type == 'star_attention':
                ret = self._merge_star_attention_requests(new_requests)
            elif cp_type == 'ring_attention':
                raise NotImplementedError("ring attention not implemented yet")
            else:
                raise NotImplementedError(f'unsupport cp type {cp_type}')
        else:
            ret = self._merge_tp_requests(new_requests)
        return ret

    @nvtx_range("_schedule")
    def _schedule(self):
        scheduler_output = self.scheduler.schedule_request(
            self.active_requests, self.inflight_req_ids)
        scheduled_requests = ScheduledRequests()

        scheduled_requests.context_requests = scheduler_output.context_requests
        scheduled_requests.generation_requests = scheduler_output.generation_requests
        scheduled_requests.paused_requests = scheduler_output.paused_requests
        return scheduled_requests, scheduler_output.fitting_disagg_gen_init_requests, scheduler_output.num_fitting_requests

    @nvtx_range("_check_disagg_gen_transfer_status")
    def _check_disagg_gen_transfer_status(self):

        if self.kv_cache_transceiver is not None:

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
                self.kv_cache_transceiver.check_gen_transfer_status(
                    at_least_num)

        return

    @nvtx_range("_check_need_one_dummy_request")
    def _check_need_one_dummy_request(self):
        if len(self.active_requests) == 0:
            return True
        if self.kv_cache_transceiver is not None:
            return all([
                req.is_disagg_generation_init_state
                or req.is_disagg_generation_transmission_in_progress
                for req in self.active_requests
            ])
        return False

    @nvtx_range("_prepare_disagg_gen_init")
    def _prepare_disagg_gen_init(self, fitting_disagg_gen_init_requests):
        if fitting_disagg_gen_init_requests:
            disagg_gen_init_to_prepare = ScheduledRequests()
            disagg_gen_init_to_prepare.context_requests = fitting_disagg_gen_init_requests
            disagg_gen_init_to_prepare.generation_requests = []
            disagg_gen_init_to_prepare.paused_requests = []

            self.resource_manager.resource_managers[
                'kv_cache_manager'].prepare_resources(
                    disagg_gen_init_to_prepare)

            # Trigger KV cache exchange for new disagg_gen_init_requests
            self._recv_disagg_gen_cache(fitting_disagg_gen_init_requests)

    @nvtx_range("_prepare_disagg_gen_transmission_complete")
    def _prepare_disagg_gen_transmission_complete(self, scheduled_batch):

        for req in scheduled_batch.generation_requests:
            if req.is_disagg_generation_transmission_complete:
                req.state = LlmRequestState.GENERATION_IN_PROGRESS
                req.context_current_position = req.prompt_len
                req.decoding_iter = 1
                first_gen_tokens = req.context_phase_params.first_gen_tokens
                req.py_draft_tokens = req.context_phase_params.draft_tokens
                beam_width = req.sampling_config.beam_width
                for beam in range(0, beam_width):
                    req.add_new_token(first_gen_tokens[beam], beam)

    @nvtx_range("_recv_disagg_gen_cache")
    def _recv_disagg_gen_cache(self, new_gen_reqs):

        for req in new_gen_reqs:
            self.kv_cache_transceiver.request_and_receive_async(req)

        block_transfer = all([
            req.is_disagg_generation_transmission_in_progress
            for req in self.active_requests
        ])
        self.kv_cache_transceiver.check_gen_transfer_status(
            1 if block_transfer else 0)

        return

    @nvtx_range("_send_disagg_ctx_cache")
    def _send_disagg_ctx_cache(self, scheduled_ctx_requests):
        if (scheduled_ctx_requests is None or len(scheduled_ctx_requests) == 0):
            return
        for req in scheduled_ctx_requests:
            if req.is_context_only_request and req.is_context_finished:
                self.kv_cache_transceiver.respond_and_send_async(req)

        self.kv_cache_transceiver.check_context_transfer_status(False)

        return

    def _forward_step(self,
                      scheduled_requests,
                      new_tensors_device: Optional[Dict[str,
                                                        torch.Tensor]] = None):

        @nvtx_range(
            f"[Executor] _forward_step: {len(scheduled_requests.context_requests)} ctx reqs, {len(scheduled_requests.generation_requests)} gen reqs"
        )
        def forward(scheduled_requests, resource_manager, new_tensors_device):
            return self.model_engine.forward(scheduled_requests,
                                             resource_manager,
                                             new_tensors_device)

        try:
            outputs = forward(scheduled_requests, self.resource_manager,
                              new_tensors_device)
            self._setup_decoder(scheduled_requests, outputs)
            self._update_request_states(scheduled_requests)
            return outputs
        except Exception as e:
            traceback.print_exc()
            error_msg = str(e)
            logger.error(
                f"Encountered an error in forward function: {error_msg}")
            self._handle_errors(error_msg)
            return None

    def _update_request_states_tp(self, scheduled_requests: ScheduledRequests):
        for request in scheduled_requests.context_requests:
            request.move_to_next_context_chunk()
            if request.get_context_remaining_length() == 0:
                request.state = LlmRequestState.GENERATION_IN_PROGRESS
            if request.is_dummy:
                request.state = LlmRequestState.GENERATION_COMPLETE

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
            if cp_type == 'star_attention':
                self._update_request_states_star_attention(scheduled_requests)
            else:
                assert False, f'Unsupport cp_type {cp_type}'
        else:
            self._update_request_states_tp(scheduled_requests)

    @nvtx_range("_decode")
    def _decode(self, scheduled_batch, batch_outputs):
        try:
            if batch_outputs is not None:
                self.decoder.decode(scheduled_batch, batch_outputs)
        except Exception as e:
            traceback.print_exc()
            error_msg = str(e)
            logger.error(f"Encountered an error in decode: {error_msg}")
            self._handle_errors(error_msg)

    @nvtx_range("_setup_decoder")
    def _setup_decoder(self, scheduled_batch, batch_outputs):
        try:
            self.decoder.setup_decoder(scheduled_batch, batch_outputs)
        except Exception as e:
            traceback.print_exc()
            error_msg = str(e)
            logger.error(f"Encountered an error in setup_decoder: {error_msg}")
            self._handle_errors(error_msg)

    @nvtx_range("_decode_async")
    def _decode_async(self, scheduled_batch, batch_outputs):
        try:
            if batch_outputs is not None:
                return self.decoder.decode_async(scheduled_batch, batch_outputs)
        except Exception as e:
            traceback.print_exc()
            error_msg = str(e)
            logger.error(f"Encountered an error in decode: {error_msg}")
            self._handle_errors(error_msg)

    @nvtx_range("_update_requests")
    def _update_requests(self, scheduled_requests: ScheduledRequests,
                         new_tensors_host: Dict[str, torch.tensor],
                         event: torch.cuda.Event):
        try:
            self.decoder.update_requests(scheduled_requests, new_tensors_host,
                                         event)
        except Exception as e:
            traceback.print_exc()
            error_msg = str(e)
            logger.error(f"Encountered an error in decode: {error_msg}")
            self._handle_errors(error_msg)

    def _handle_errors(self, error_msg: Optional[str] = None):
        error_responses = {}
        error_msg = error_msg or "error"
        for request in self.active_requests:
            req_id = request.py_request_id
            request.state = LlmRequestState.GENERATION_COMPLETE
            self._terminate_request(request)
            error_responses[req_id] = ExecutorResponse(req_id, error_msg)
        self._enqueue_responses(error_responses)

    def _terminate_request(self, request: LlmRequest):
        self.resource_manager.free_resources(request)

    @nvtx_range("_handle_cancelled_requests")
    def _handle_cancelled_requests(self):
        if not self.canceled_req_ids:
            return

        #TODO: properly handle canceled ids in pp case
        if self.dist.has_tp and self.canceled_req_ids:
            self.canceled_req_ids = self.dist.broadcast(self.canceled_req_ids,
                                                        root=0)

        cancelled_responses = {}
        left_requests = []
        for request in self.active_requests:
            req_id = request.py_request_id
            if req_id in self.canceled_req_ids:
                self._terminate_request(request)
                request.finish_by_reason(trtllm.FinishReason.CANCELLED)
                cancelled_responses[req_id] = request.create_response(
                    False, self.dist.rank)
            else:
                left_requests.append(request)
        self.active_requests = left_requests

        # enqueue the cancelled requests' responses as they are not
        # active_requests and be discarded in the decoder loop.
        self._enqueue_responses(cancelled_responses)

    @nvtx_range("_enqueue_responses")
    def _enqueue_responses(self, responses: Dict[int, ExecutorResponse]):

        logger.debug(
            f'before ag, rank = {self.dist.rank}, responses = {responses}')
        if self.enable_attention_dp:
            resonses_list = self.dist.allgather(responses)
            gather_responses = {}
            for resp in resonses_list:
                gather_responses.update(resp)
            responses = gather_responses
        logger.debug(
            f'after ag, rank = {self.dist.rank}, responses = {responses}')
        if self.dist.rank == 0:
            with self.response_cv:
                for req_id, resp in responses.items():
                    if req_id in self.responses.keys():
                        self.responses[req_id].append(resp)
                    else:
                        self.responses.update({req_id: [resp]})
                self.response_cv.notify_all()

    @nvtx_range("_handle_responses")
    def _handle_responses(self):
        new_responses = {}
        requests_to_terminate = []
        new_active_requests = []
        logger.debug(
            f'------before _handle_responses, rank = {self.dist.rank}, output = {self.active_requests}'
        )
        for request in self.active_requests:
            req_id = request.py_request_id
            #no responses for dummy request, and finish it
            if request.is_dummy == True:
                requests_to_terminate.append(request)
                continue

            request.draft_tokens = request.py_draft_tokens
            response = request.create_response(False, self.dist.rank)
            request_done = False

            if response:
                request_done = response.result.is_final
                new_responses.update({req_id: response})
            if request_done:
                if request.is_disagg_context_transmission_state:
                    self.ctx_in_transmission_requests.append(request)
                else:
                    requests_to_terminate.append(request)
            else:
                new_active_requests.append(request)
        self.active_requests = new_active_requests
        self._enqueue_responses(new_responses)
        for request in requests_to_terminate:
            self._terminate_request(request)

        return requests_to_terminate

    @nvtx_range("_terminate_ctx_finished_requests")
    def _terminate_ctx_finished_requests(self):
        for request in self.ctx_in_transmission_requests:
            if request.is_disagg_context_complete_state:
                self._terminate_request(request)
                self.ctx_in_transmission_requests.remove(request)

    def shutdown(self):
        try:
            self.enqueue_lock.acquire()
            self.request_queue.put(None)
            self.active = False
        finally:
            self.enqueue_lock.release()
        self.shutdown_event.wait()
        self.worker_thread.join()
        for manager in self.resource_manager.resource_managers.values():
            if manager:
                manager.shutdown()

    def enqueue_request(self,
                        request: ExecutorRequest,
                        query: Optional[List] = None):
        try:
            self.enqueue_lock.acquire()
            assert self.active, "PyExecutor has already been shutdown."
            req_id = self.next_req_id
            if self.enable_iter_perf_stats:
                self.start_times[req_id] = time.time()

            if query is not None:
                self.request_queue.put((req_id, request, query))
            else:
                self.request_queue.put((req_id, request))
            self.next_req_id += 1
        finally:
            self.enqueue_lock.release()
        return req_id

    def enqueue_requests(self, requests: List[ExecutorRequest]):
        req_ids = []
        try:
            self.enqueue_lock.acquire()
            assert self.active, "PyExecutor has already been shutdown."
            start_time = time.time()
            for request in requests:
                self.start_times[self.next_req_id] = start_time
                self.request_queue.put((self.next_req_id, request))
                req_ids.append(self.next_req_id)
                self.next_req_id += 1
        finally:
            self.enqueue_lock.release()
        return req_ids

    def _await_any_response(self,
                            timeout: Union[float, None] = None
                            ) -> List[ExecutorResponse]:

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
            timeout: Union[float, None] = None) -> List[ExecutorResponse]:
        with self.response_cv:

            def key_has_response():
                return id in self.responses.keys()

            self.response_cv.wait_for(key_has_response, timeout=timeout)
            response = self.responses[id]
            self.responses.pop(id)
            return response

    def await_responses(
        self,
        id: Union[List[int], int, None] = None,
        timeout: Union[datetime.timedelta, None] = None,
    ) -> Union[List[List[ExecutorResponse]], List[ExecutorResponse]]:
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
        self.canceled_req_ids.insert(id)

    def get_num_responses_ready(self, id: Union[int, None]) -> int:
        with self.response_cv:
            if isinstance(id, int):
                if id in self.responses.keys():
                    return len(self.responses[id])
                else:
                    return 0
            else:
                num_ready_responses = 0
                for req_id, response in self.responses.items():
                    num_ready_responses += len(response)
                return num_ready_responses

    def can_enqueue_requests(self) -> bool:
        self.enqueue_lock.acquire()
        can_enqueue = self.active
        self.enqueue_lock.release()
        return can_enqueue and self.dist.rank == 0

    def get_latest_iteration_stats(self):
        if self.enable_iter_perf_stats == False:
            return []

        latest_stats = tuple()
        try:
            self.stats_lock.acquire()
            latest_stats = tuple(self.stats)
            self.stats = []
        finally:
            self.stats_lock.release()

        return latest_stats

    def get_latest_request_stats(self):
        #raise NotImplementedError("get_latest_request_stats not implemented")
        # todo: implement it
        return []

    def get_latest_kv_cache_events(self):
        kv_cache_manager = self.resource_manager.resource_managers.get(
            "kv_cache_manager")
        if not kv_cache_manager or not self.enable_kv_cache_events:
            return []

        events = kv_cache_manager.get_latest_events(0)
        return events

    def pause_requests(self, requests_to_pause):
        # todo: support work with self.inflight_req_ids.
        #       Currently, self.inflight_req_ids is not.
        max_input_len = self.max_input_len
        for req in requests_to_pause:
            req.pause(max_input_len)
            self._terminate_request(req)

    def _add_inflight_ids(self, scheduled_requests):
        """Add reqids of current requests to self.inflight_req_ids."""
        for req in chain(scheduled_requests.context_requests,
                         scheduled_requests.generation_requests):
            self.inflight_req_ids.insert(req.request_id)

    def _remove_inflight_ids(self, scheduled_requests):
        """Remove reqids of current requests from self.inflight_req_ids."""
        for req in chain(scheduled_requests.context_requests,
                         scheduled_requests.generation_requests):
            self.inflight_req_ids.erase(req.request_id)
