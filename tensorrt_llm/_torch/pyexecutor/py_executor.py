import datetime
import functools
import gc
import heapq
import os
import queue
import threading
import time
import traceback
import weakref
from collections import namedtuple
from contextlib import contextmanager
from itertools import chain
from typing import Dict, List, Optional, Tuple, Union

import dill  # nosec B403
import numpy as np
import torch

from tensorrt_llm._utils import global_mpi_rank, nvtx_range
from tensorrt_llm.bindings.executor import (FinishReason, InflightBatchingStats,
                                            IterationStats, KvCacheStats,
                                            RequestType)
from tensorrt_llm.bindings.internal.batch_manager import ReqIdsSet
from tensorrt_llm.logger import logger

from .decoder import Decoder
from .distributed import Distributed
from .kv_cache_transceiver import KvCacheTransceiver
from .llm_request import (ExecutorRequest, ExecutorResponse, LlmRequest,
                          LlmRequestState, executor_request_to_llm_request)
from .model_engine import ModelEngine
from .scheduler import ScheduledRequests


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


class PyExecutor:

    def __init__(self,
                 resource_manager,
                 scheduler,
                 model_engine: ModelEngine,
                 decoder: Decoder,
                 dist: Distributed,
                 enable_overlap_scheduler: bool = False,
                 max_input_len: int = 2048,
                 max_batch_size: int = 8,
                 max_draft_tokens: int = 0,
                 kv_cache_transceiver: KvCacheTransceiver = None,
                 draft_model_engine: Optional[ModelEngine] = None):
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

        # Draft model for certain spec decode algorithms, e.g. EAGLE3
        self.draft_model_engine = draft_model_engine

        # enqueue and _fetch_new_requests used data
        self.enqueue_lock = threading.Lock()
        self.active = True
        self.next_req_id = max_batch_size  # The first max_batch_size request IDs are reserved for dummy requests
        self.max_draft_tokens = max_draft_tokens
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
        self.kv_cache_manager = self.resource_manager.resource_managers.get(
            "kv_cache_manager")
        self.enable_kv_cache_events = self.kv_cache_manager is not None and self.kv_cache_manager.event_buffer_max_size > 0

        if self.draft_model_engine is not None and self.kv_cache_manager is not None:
            if self.kv_cache_manager.enable_block_reuse:
                raise NotImplementedError(
                    "Draft model engine + KV cache reuse is not supported yet. "
                    "This will be fixed in the near future!")

        self.max_input_len = max_input_len
        # _executor_loop private data
        self.max_num_active_requests = model_engine.get_max_num_sequences()
        self.active_requests = []
        self.all_ranks_num_active_requests = [
            0
        ] * self.dist.tp_size if self.enable_attention_dp else []
        self.expected_num_active_requests = 0
        self.has_context_request = False
        self.ctx_in_transmission_requests = []
        self.previous_batch = None

        # list of requests in each PP micro batch
        self.num_micro_batches = self.dist.pp_size + enable_overlap_scheduler
        self.micro_batches = [None] * self.num_micro_batches
        self.send_handles = [None] * self.num_micro_batches
        # one handle each for metadata and serialized new_reqs buffer
        self.send_new_reqs_handle = [None] * 2

        self.inflight_req_ids = ReqIdsSet()
        self.canceled_req_ids = ReqIdsSet()

        self.model_engine.warmup(self.resource_manager)
        if self.draft_model_engine is not None:
            self.draft_model_engine.warmup(self.resource_manager)

        self.is_shutdown = False

        self.stats_lock = threading.Lock()
        self.stats = []
        self.start_times = {}
        self.new_active_requests_queue_latency_ms = 0

        self.kv_cache_transceiver = kv_cache_transceiver
        if self.dist.pp_size > 1:
            event_loop = self._executor_loop_pp_overlap if enable_overlap_scheduler else self._executor_loop_pp
        else:
            event_loop = self._executor_loop_overlap if enable_overlap_scheduler else self._executor_loop

        if self.draft_model_engine is not None and event_loop.__name__ != self._executor_loop.__name__:
            raise NotImplementedError(
                "Drafting is not supported for selected executor loop. "
                "Please disable disagg/pipeline parallelism/overlap scheduler.")

        self.worker_thread = threading.Thread(target=event_loop, daemon=True)
        self.worker_thread.start()

    def __enter__(self):
        return self

    def __exit__(self):
        self.shutdown()

    def enqueue_requests(self, requests: List[ExecutorRequest]):
        """
        Enqueue new requests
        """
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

    def await_responses(
        self,
        id: Optional[Union[List[int], int]] = None,
        timeout: Optional[datetime.timedelta] = None,
    ) -> Union[List[List[ExecutorResponse]], List[ExecutorResponse]]:
        """
        Await for ready responses
        Args:
            id (Optional[Union[List[int], int]]): Request id
            timeout (Optional[datetime.timedelta]): The maximum time to wait for new responses
        Returns:
            Union[List[tensorrt_llm.bindings.executor.Response], List[List[tensorrt_llm.bindings.executor.Response]]]: Responses
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
        if id is None:
            return
        self.canceled_req_ids.insert(id)

    def shutdown(self):
        """
        Signals the server to shutdown.
        """
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
        del self.model_engine
        if self.draft_model_engine is not None:
            del self.draft_model_engine

    def can_enqueue_requests(self) -> bool:
        """
        Indicates if the current process is allowed to enqueue requests
        """
        self.enqueue_lock.acquire()
        can_enqueue = self.active
        self.enqueue_lock.release()
        return can_enqueue and self.dist.rank == 0

    def get_latest_iteration_stats(self):
        """
        Returns the per-iterations statistics computed since last call to this method.
        Contains at most iter_stats_max_iterations iterations.
        """
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

    def get_latest_kv_cache_events(self):
        kv_cache_manager = self.resource_manager.resource_managers.get(
            "kv_cache_manager")
        if not kv_cache_manager or not self.enable_kv_cache_events:
            return []

        events = kv_cache_manager.get_latest_events(0)
        return events

    def wait_shutdown(self):
        self.shutdown_event.wait()

    def enqueue_request(self,
                        request: ExecutorRequest,
                        query: Optional[List] = None):
        """
        Enqueue a new request, only used in `StarAttention`.
        """
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
        stats = IterationStats()
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

        model_stats = InflightBatchingStats()
        model_stats.num_scheduled_requests = len(
            scheduled_batch.context_requests) + len(
                scheduled_batch.generation_requests)
        model_stats.num_context_requests = self.model_engine.iter_states[
            'num_ctx_requests']
        model_stats.num_gen_requests = len(scheduled_batch.generation_requests)
        model_stats.num_paused_requests = len(scheduled_batch.paused_requests)
        model_stats.avg_num_decoded_tokens_per_iter = 0
        model_stats.num_ctx_tokens = self.model_engine.iter_states[
            'num_ctx_tokens']
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

    def _executor_loop_cleanup(self):
        with self.response_cv:
            self.is_shutdown = True
            self.response_cv.notify_all()
        self.shutdown_event.set()

    def _executor_loop_pp(self):
        torch.cuda.set_device(self.device_id)
        got_finish_signal = False
        num_dummy_request = 0
        microbatch_id = 0
        with self._profiler() as profile_step:
            iter_start_time = time.time()
            iter_stats = None
            while not got_finish_signal or len(self.active_requests) > 0:
                profile_step()
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
                if not got_finish_signal:
                    num_dummy_request = self._get_num_dummy_request()
                if num_dummy_request > 0:
                    self._merge_dummy_request(num_dummy_request)
                scheduled_batch, _, _ = self._schedule()

                if self.enable_attention_dp:
                    tp_batch_sizes = self.dist.tp_allgather(
                        scheduled_batch.batch_size)
                    can_queue = 0 not in tp_batch_sizes
                else:
                    can_queue = scheduled_batch.batch_size > 0
                    if not can_queue:
                        assert len(self.inflight_req_ids) > 0, (
                            "fail to schedule any pending request, probably run out of resource"
                        )

                if not can_queue:
                    self.micro_batches[microbatch_id] = None
                else:
                    # TODO: add pause_requests together with inflight_req_ids and handle draft_tokens
                    self._add_inflight_ids(scheduled_batch)
                    self.resource_manager.prepare_resources(scheduled_batch)

                    # Stage 1: Forward + (decoding) pass ([should be] async)
                    if self.dist.is_last_pp_rank:
                        new_tensors_host = self._forward_step_last_pp(
                            scheduled_batch, microbatch_id)
                    else:
                        new_tensors_host = self._forward_step_inter_pp(
                            scheduled_batch)

                    if num_dummy_request > 0:
                        self._finish_dummy_request(scheduled_batch)
                    self.micro_batches[microbatch_id] = (scheduled_batch,
                                                         new_tensors_host)

                # marching forward in the microbatch slots
                prev_microbatch_id = (microbatch_id +
                                      1) % self.num_micro_batches
                previous_batch = self.micro_batches[prev_microbatch_id]

                # Stage 2: Handle previous batch that only processed forward_step
                if previous_batch is not None:
                    previous_scheduled_batch, previous_new_tensors_host = previous_batch
                    if not self.dist.is_last_pp_rank:
                        self._handle_previous_batch_inter_pp(
                            previous_scheduled_batch, previous_new_tensors_host,
                            prev_microbatch_id)

                    self._update_requests(previous_scheduled_batch,
                                          previous_new_tensors_host, None)
                    self._handle_cancelled_requests()
                    finished_requests = self._handle_responses()
                    self.resource_manager.update_resources(
                        previous_scheduled_batch)
                    self._remove_inflight_ids(previous_scheduled_batch)

                microbatch_id = prev_microbatch_id
                self._gather_dp_requests_num()

                if self.enable_iter_perf_stats:
                    iter_end_time = time.time()
                    iter_latency_ms = iter_end_time - iter_start_time
                    self._append_iter_stats(
                        self._update_iter_stats(iter_stats, iter_latency_ms,
                                                len(finished_requests),
                                                scheduled_batch))
                    iter_start_time = iter_end_time
        self._executor_loop_cleanup()

    def _executor_loop_pp_overlap(self):
        torch.cuda.set_device(self.device_id)
        got_finish_signal = False
        num_dummy_request = 0
        microbatch_id = 0
        with self._profiler() as profile_step:
            iter_start_time = time.time()
            iter_end_time = iter_start_time
            iter_stats = None
            while not got_finish_signal or len(self.active_requests) > 0:
                profile_step()
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
                if not got_finish_signal:
                    num_dummy_request = self._get_num_dummy_request()
                if num_dummy_request > 0:
                    self._merge_dummy_request(num_dummy_request)

                scheduled_batch, _, _ = self._schedule()

                if self.enable_attention_dp:
                    tp_batch_sizes = self.dist.tp_allgather(
                        scheduled_batch.batch_size)
                    can_queue = 0 not in tp_batch_sizes
                else:
                    can_queue = scheduled_batch.batch_size > 0
                    if not can_queue:
                        assert len(self.inflight_req_ids) > 0, (
                            "fail to schedule any pending request, probably run out of resource"
                        )

                if not can_queue:
                    self.micro_batches[microbatch_id] = None
                else:
                    self._add_inflight_ids(scheduled_batch)
                    self.resource_manager.prepare_resources(scheduled_batch)

                    # Stage 1: Async forward (all ranks) and decoding pass (last rank only)
                    if not self.dist.is_last_pp_rank:
                        new_tensors_host = self._forward_step_inter_pp(
                            scheduled_batch)
                        decoder_event = None
                    else:
                        torch.cuda.nvtx.range_push("_forward_step_last_pp")
                        batch_outputs = self._forward_step(scheduled_batch)
                        new_tensors_device, new_tensors_host, decoder_event = self._decode_async(
                            scheduled_batch, batch_outputs)
                        torch.cuda.nvtx.range_pop()

                    if num_dummy_request > 0:
                        self._finish_dummy_request(scheduled_batch)
                    self.micro_batches[microbatch_id] = (scheduled_batch,
                                                         new_tensors_host,
                                                         decoder_event)

                # Stage 2: Communicate new tokens for previous batch between ranks
                # send/recv chain: (pp_size - 1) -> 0 -> 1 -> ... -> (pp_size - 2)
                # last rank: sync decoder for previous microbatch to start new tokens comm chain.
                # other ranks: send/recv tokens for next microbatch to allow overlap
                offset = -1 if self.dist.is_last_pp_rank else 1
                prev_microbatch_id = (microbatch_id +
                                      offset) % self.num_micro_batches
                previous_batch = self.micro_batches[prev_microbatch_id]
                if previous_batch is not None:
                    if not self.dist.is_last_pp_rank:
                        torch.cuda.nvtx.range_push(
                            "_handle_new_tokens_inter_pp")
                        _, previous_new_tensors_host, _ = previous_batch
                        # Receive tokens from previous pp rank (w.r.t model forward direction)
                        self.dist.recv_tensor(
                            previous_new_tensors_host["new_tokens_host"],
                            src=self.dist.prev_pp_rank,
                            tag=prev_microbatch_id)
                    else:
                        torch.cuda.nvtx.range_push("_handle_new_tokens_last_pp")
                        _, previous_new_tensors_host, previous_decoder_event = previous_batch
                        previous_decoder_event.synchronize()

                    # Send tokens to next pp rank (w.r.t model forward direction)
                    # Second last rank does not need to since last rank has original decoded tokens
                    if not self.dist.is_second_last_pp_rank:
                        if self.send_handles[prev_microbatch_id] is not None:
                            self.send_handles[prev_microbatch_id].Wait()
                        self.send_handles[
                            prev_microbatch_id] = self.dist.isend_tensor(
                                previous_new_tensors_host["new_tokens_host"],
                                dest=self.dist.next_pp_rank,
                                tag=prev_microbatch_id)
                    torch.cuda.nvtx.range_pop()

                # Stage 3: Finalize previous batch that finished tokens communication
                # In last pp rank, stage 2 and 3 process different previous batches
                prev_microbatch_id = (microbatch_id +
                                      1) % self.num_micro_batches
                previous_batch = self.micro_batches[prev_microbatch_id]
                if previous_batch is not None:
                    torch.cuda.nvtx.range_push("_handle_previous_batch_pp")
                    previous_scheduled_batch, previous_new_tensors_host, previous_decoder_event = previous_batch
                    self._update_requests(previous_scheduled_batch,
                                          previous_new_tensors_host,
                                          previous_decoder_event)
                    self._handle_cancelled_requests()
                    finished_requests = self._handle_responses()
                    self.resource_manager.update_resources(
                        previous_scheduled_batch)
                    self._remove_inflight_ids(previous_scheduled_batch)
                    torch.cuda.nvtx.range_pop()
                    self.micro_batches[prev_microbatch_id] = None

                # march forward in microbatch slots
                microbatch_id = (microbatch_id + 1) % self.num_micro_batches

                self._gather_dp_requests_num()

                if self.enable_iter_perf_stats:
                    iter_end_time = time.time()
                    iter_latency_ms = iter_end_time - iter_start_time
                    self._append_iter_stats(
                        self._update_iter_stats(iter_stats, iter_latency_ms,
                                                len(finished_requests),
                                                scheduled_batch))
                    iter_start_time = iter_end_time
        self._executor_loop_cleanup()

    def _executor_loop(self):
        torch.cuda.set_device(self.device_id)
        got_finish_signal = False
        num_dummy_request = 0
        with self._profiler() as profile_step:
            iter_start_time = time.time()
            iter_stats = None
            while not got_finish_signal or len(self.active_requests) > 0:
                profile_step()
                if self.enable_iter_perf_stats:
                    iter_start_time = time.time()
                new_requests = self._fetch_new_requests()
                got_finish_signal = self._merge_requests(
                    new_requests) or got_finish_signal
                if got_finish_signal and len(self.active_requests) == 0:
                    break

                if self.enable_iter_perf_stats:
                    iter_stats = self._get_init_iter_stats(
                        len(new_requests),
                        self.new_active_requests_queue_latency_ms)

                if self.kv_cache_transceiver:
                    self._check_disagg_gen_transfer_status()

                if not got_finish_signal:
                    num_dummy_request = self._get_num_dummy_request()
                if num_dummy_request > 0:
                    self._merge_dummy_request(num_dummy_request)

                if self.draft_model_engine is not None:
                    self._prepare_draft_requests()

                scheduled_batch, fitting_disagg_gen_init_requests, num_fitting_reqs = self._schedule(
                )

                if self.kv_cache_transceiver:
                    self._prepare_disagg_gen_init(
                        fitting_disagg_gen_init_requests)
                    if num_fitting_reqs == 0 and not fitting_disagg_gen_init_requests:
                        logger.warning(
                            "num_fitting_reqs=0 and fitting_disagg_gen_init_requests is empty, may not have enough kvCache"
                        )
                        self.kv_cache_transceiver.check_context_transfer_status(
                            True)
                else:
                    assert scheduled_batch.batch_size > 0, (
                        "fail to schedule any pending request, "
                        "probably run out of resource.")

                self._pause_requests(scheduled_batch.paused_requests)

                finished_requests = []

                if scheduled_batch.batch_size > 0:
                    self.resource_manager.prepare_resources(scheduled_batch)
                    if self.draft_model_engine is not None:
                        self._prepare_draft_tokens(scheduled_batch)

                    if self.kv_cache_transceiver:
                        # For generation requests which have completed KV cache transfer
                        self._prepare_disagg_gen_transmission_complete(
                            scheduled_batch)

                    batch_outputs = self._forward_step(scheduled_batch)

                    ctx_transmission_reqs = self._send_disagg_ctx_cache(
                        scheduled_batch.context_requests
                    ) if self.kv_cache_transceiver else []

                    self._decode(scheduled_batch, batch_outputs)

                    if self.kv_cache_transceiver:
                        # For context only req in transmission, we reset the state since decoder might have changed it
                        for req in ctx_transmission_reqs:
                            req.state = LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS

                    self._handle_cancelled_requests()
                    finished_requests = self._handle_responses()
                    self.resource_manager.update_resources(scheduled_batch)
                    if self.enable_kv_cache_events:
                        self._add_kv_cache_events()

                if self.kv_cache_transceiver and self.ctx_in_transmission_requests:
                    self._terminate_ctx_finished_requests()

                self._gather_dp_requests_num()

                if self.enable_iter_perf_stats:
                    self._process_iter_stats(finished_requests, scheduled_batch,
                                             iter_start_time, iter_stats)

        self._executor_loop_cleanup()

    def _prepare_draft_requests(self):
        try:
            # Set draft tokens here to make the KV cache manager
            # and scheduler aware of them.
            for req in self.active_requests:
                if req.state != LlmRequestState.GENERATION_IN_PROGRESS:
                    continue
                req.py_last_draft_tokens = req.py_draft_tokens
                max_draft_len = self.model_engine.spec_config.max_draft_tokens
                max_seq_len = self.model_engine.max_seq_len

                # Subtract 1 to account for the token we will add on this forward
                # pass.
                draft_len = min(max_seq_len - 1 - req.get_num_tokens(0),
                                max_draft_len)

                if draft_len > 0:
                    req.py_draft_tokens = [0] * draft_len
                    req.py_draft_pages_allocated = draft_len
                else:
                    req.py_draft_tokens = None
                    req.py_draft_pages_allocated = 0

        except Exception as e:
            traceback.print_exc()
            error_msg = str(e)
            logger.error(f"Encountered an error in decode: {error_msg}")
            self._handle_errors(error_msg)

    def _executor_loop_overlap(self):
        torch.cuda.set_device(self.device_id)
        got_finish_signal = False
        num_dummy_request = 0
        with self._profiler() as profile_step:
            iter_start_time = time.time()
            iter_stats = None
            while not got_finish_signal or len(self.active_requests) > 0:
                profile_step()
                if self.enable_iter_perf_stats:
                    iter_start_time = time.time()
                new_requests = self._fetch_new_requests()
                got_finish_signal = self._merge_requests(
                    new_requests) or got_finish_signal
                if got_finish_signal and len(self.active_requests) == 0:
                    break

                if self.kv_cache_transceiver:
                    self._check_disagg_gen_transfer_status()

                if self.enable_iter_perf_stats:
                    iter_stats = self._get_init_iter_stats(
                        len(new_requests),
                        self.new_active_requests_queue_latency_ms)

                if not got_finish_signal:
                    num_dummy_request = self._get_num_dummy_request()
                if num_dummy_request > 0:
                    self._merge_dummy_request(num_dummy_request)
                scheduled_batch, fitting_disagg_gen_init_requests, num_fitting_reqs = self._schedule(
                )

                if self.kv_cache_transceiver:

                    # For requests that are fitting disagg gen init, also prepare resources for KV cache manager
                    self._prepare_disagg_gen_init(
                        fitting_disagg_gen_init_requests)

                    if num_fitting_reqs == 0 and not fitting_disagg_gen_init_requests:
                        logger.warning(
                            "num_fitting_reqs =0 and fitting_disagg_gen_init_requests is empty , may not have enough kvCache"
                        )
                        self.kv_cache_transceiver.check_context_transfer_status(
                            True)
                else:
                    assert scheduled_batch.batch_size > 0, (
                        "fail to schedule any pending request, "
                        "probably run out of resource.")

                logger.debug(
                    f'has {len(self.active_requests)} active_request, '
                    f'scheduled {len(scheduled_batch.context_requests)} context requests and '
                    f'{len(scheduled_batch.generation_requests)} generation requests'
                )

                self._pause_requests(scheduled_batch.paused_requests)

                if scheduled_batch.batch_size > 0:
                    self.resource_manager.prepare_resources(scheduled_batch)

                    if self.kv_cache_transceiver:
                        # For generation requests which have completed KV cache transfer
                        self._prepare_disagg_gen_transmission_complete(
                            scheduled_batch)

                    previous_new_tensors_device = None
                    if self.previous_batch is not None:
                        _, previous_new_tensors_device, _, _, _, _, _ = self.previous_batch

                    batch_outputs = self._forward_step(
                        scheduled_batch, previous_new_tensors_device)

                    ctx_transmission_reqs = self._send_disagg_ctx_cache(
                        scheduled_batch.context_requests
                    ) if self.kv_cache_transceiver else []

                    new_tensors_device, new_tensors_host, decoder_event = self._decode_async(
                        scheduled_batch, batch_outputs)

                    if num_dummy_request > 0:
                        self._finish_dummy_request(scheduled_batch)
                    has_previous_batch = self.previous_batch is not None
                    if has_previous_batch:
                        self._process_previous_batch()
                        self.previous_batch = None

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
                                           iter_start_time, iter_stats,
                                           ctx_transmission_reqs)
                    self._gather_dp_requests_num()

                if self.kv_cache_transceiver and self.ctx_in_transmission_requests:
                    self._terminate_ctx_finished_requests()

        self._executor_loop_cleanup()

    def _process_previous_batch(self):
        previous_scheduled_batch, _, previous_new_tensors_host, previous_decoder_event, previous_iter_start_time, previous_iter_stats, previous_ctx_transmission_reqs = self.previous_batch
        self._update_requests(previous_scheduled_batch,
                              previous_new_tensors_host, previous_decoder_event)

        if self.kv_cache_transceiver and previous_ctx_transmission_reqs:
            for req in previous_ctx_transmission_reqs:
                req.state = LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS

        self._handle_cancelled_requests()
        finished_requests = self._handle_responses()
        self.resource_manager.update_resources(previous_scheduled_batch)
        if self.enable_kv_cache_events:
            self._add_kv_cache_events()

        if self.enable_iter_perf_stats:
            self._process_iter_stats(finished_requests,
                                     previous_scheduled_batch,
                                     previous_iter_start_time,
                                     previous_iter_stats)

    @nvtx_range("_forward_step_inter_pp")
    def _forward_step_inter_pp(self, scheduled_batch):
        batch_outputs = self._forward_step(scheduled_batch)
        tokens_shape = batch_outputs["hidden_states"].shape[:-1]
        new_tokens_host = torch.empty(tokens_shape,
                                      dtype=torch.int64,
                                      device='cpu',
                                      pin_memory=True)
        return {"new_tokens_host": new_tokens_host}

    @nvtx_range("_handle_previous_batch_inter_pp")
    def _handle_previous_batch_inter_pp(self, previous_scheduled_batch,
                                        previous_new_tensors_host,
                                        prev_microbatch_id):
        # Receive tokens from prev pp rank w.r.t model forward direction
        self.dist.recv_tensor(
            previous_new_tensors_host["new_tokens_host"],
            src=self.dist.prev_pp_rank,
            tag=prev_microbatch_id  # not necessary and may discard
        )

        # Send tokens to next pp rank w.r.t model forward direction
        # Second last rank not need since last rank has original decoded tokens
        if not self.dist.is_second_last_pp_rank:
            if self.send_handles[prev_microbatch_id] is not None:
                self.send_handles[prev_microbatch_id].Wait()
            self.send_handles[prev_microbatch_id] = self.dist.isend_tensor(
                tensor=previous_new_tensors_host["new_tokens_host"],
                dest=self.dist.next_pp_rank,
                tag=prev_microbatch_id)

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
            dest=self.dist.next_pp_rank,
            tag=microbatch_id)

        return new_tensors_host

    def _update_new_active_requests_queue_latency(self, new_requests):
        if self.enable_iter_perf_stats and self.dist.rank == 0:
            now = time.time()
            for req in new_requests:
                if isinstance(req, tuple):
                    req_id = req[0]
                    if req_id in self.start_times:
                        self.new_active_requests_queue_latency_ms += now - self.start_times.pop(
                            req_id)

    @nvtx_range("_broadcast_new_requests")
    def _broadcast_new_requests(self, new_requests):
        if not self.dist.has_pp:
            return self.dist.broadcast(new_requests, root=0)

        # broadcast within first tp group before send/recv chain to other tp groups
        if self.dist.tp_size > 1 and self.dist.is_first_pp_rank:
            new_requests = self.dist.tp_broadcast(new_requests, root=0)

        # tag = [0, num_micro_batches - 1] used for new_tokens send/recv
        tag = self.num_micro_batches

        # 1. send metadata: len(num_requests) and serialized buffer size
        if self.dist.is_first_pp_rank and len(new_requests) > 0:
            buf = np.array(bytearray(dill.dumps(new_requests)))
            buf_size = len(buf)
        else:
            buf, buf_size = None, 0
        metadata_arr = np.array([len(new_requests), buf_size])

        if not self.dist.is_first_pp_rank:
            self.dist.recv(metadata_arr, self.dist.prev_pp_rank, tag)

        if not self.dist.is_last_pp_rank:
            if self.send_new_reqs_handle[0] is not None:
                self.send_new_reqs_handle[0].Wait()
            self.send_new_reqs_handle[0] = self.dist.isend(
                metadata_arr, self.dist.next_pp_rank, tag)

        # 2. send serialized buffer when new requests is not empty
        num_new_requests = metadata_arr[0]
        if num_new_requests > 0:
            buf_size = metadata_arr[1]
            if not self.dist.is_first_pp_rank:
                buf = np.array(bytearray(buf_size))
                self.dist.recv(buf, self.dist.prev_pp_rank, tag)

            if not self.dist.is_last_pp_rank:
                if self.send_new_reqs_handle[1] is not None:
                    self.send_new_reqs_handle[1].Wait()
                self.send_new_reqs_handle[1] = self.dist.isend(
                    buf, self.dist.next_pp_rank, tag)

            if not self.dist.is_first_pp_rank:
                new_requests = dill.loads(buf.tobytes())  # nosec B301
                assert len(new_requests) == num_new_requests

        return new_requests

    @nvtx_range("_fetch_new_requests")
    def _fetch_new_requests(self):
        if self.enable_attention_dp:
            total_num_active_requests = sum(self.all_ranks_num_active_requests)
            total_max_num_active_requests = self.dist.tp_size * self.max_num_active_requests
        else:
            total_num_active_requests = len(self.active_requests)
            total_max_num_active_requests = self.max_num_active_requests

        timeout = None if total_num_active_requests == 0 else datetime.timedelta(
            0)
        new_requests = []
        if self.dist.rank == 0:
            new_requests = _get_from_request_queue(
                self.request_queue, timeout,
                total_max_num_active_requests - total_num_active_requests)

        new_requests = self._broadcast_new_requests(new_requests)

        if not self.enable_attention_dp:
            self._update_new_active_requests_queue_latency(new_requests)
            return new_requests

        num_new_requests_all_ranks = len(new_requests)
        self.expected_num_active_requests = max(
            (total_num_active_requests + num_new_requests_all_ranks +
             self.dist.tp_size - 1) // self.dist.tp_size,
            max(self.all_ranks_num_active_requests),
        )
        self.has_context_request = False
        new_requests_cur_rank = []
        if new_requests != [] and new_requests[
                0] != None and self.expected_num_active_requests > self.all_ranks_num_active_requests[
                    self.dist.tp_rank]:
            # Balance context tokens across ranks
            HeapVal = namedtuple(
                'HeapVal',
                [
                    'num_tokens',  # number of context tokens that have been added
                    'num_requests',  # number of requests to be added
                    'rank',  # rank
                    'request_list',  # new requests that have been added
                ],
            )
            all_ranks_new_requests_heap = [
                HeapVal(0, self.expected_num_active_requests - val, tp_rank, [])
                for tp_rank, val in enumerate(
                    self.all_ranks_num_active_requests)
            ]
            new_requests_cur_rank = all_ranks_new_requests_heap[
                self.dist.tp_rank].request_list
            all_ranks_new_requests_heap = [
                val for val in all_ranks_new_requests_heap
                if val.num_requests > 0
            ]
            heapq.heapify(all_ranks_new_requests_heap)
            new_requests = sorted(new_requests,
                                  key=lambda x: len(x[1].input_token_ids),
                                  reverse=True)
            for request in new_requests:
                val = heapq.heappop(all_ranks_new_requests_heap)
                val = val._replace(
                    num_tokens=val.num_tokens + len(request[1].input_token_ids),
                    num_requests=val.num_requests - 1,
                )
                val.request_list.append(request)
                if val.num_requests > 0:
                    heapq.heappush(all_ranks_new_requests_heap, val)
                elif val.rank == self.dist.tp_rank:
                    break

            # In disaggregated serving, we might get either context request or
            # generation request. In IFB, we only get context request from request queue
            if self.kv_cache_transceiver:
                for req in new_requests_cur_rank:
                    if req[1].request_type == RequestType.REQUEST_TYPE_CONTEXT_ONLY:
                        self.has_context_request = True
                        break
            else:
                self.has_context_request = len(new_requests_cur_rank) > 0
            self._update_new_active_requests_queue_latency(
                new_requests_cur_rank)

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
            responses_list = self.dist.tp_allgather(len(self.active_requests))
            for num_active_requests in responses_list:
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
        for request in new_requests:
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

        return False

    def _merge_dummy_request(self, num_dummy_request: int):
        llm_request_list = self.kv_cache_manager.add_dummy_requests(
            request_ids=list(range(num_dummy_request)),
            is_gen=not self.has_context_request,
            prepare_resource=not self.has_context_request,
            max_num_draft_tokens=0
            if self.has_context_request else self.max_draft_tokens,
        )
        for llm_request in llm_request_list:
            llm_request.is_dummy = True
        self.active_requests += llm_request_list

    def _finish_dummy_request(self, scheduled_requests: ScheduledRequests):
        for req in scheduled_requests.context_requests:
            if req.is_dummy:
                req.state = LlmRequestState.GENERATION_COMPLETE
        for req in scheduled_requests.generation_requests:
            if req.is_dummy:
                req.state = LlmRequestState.GENERATION_COMPLETE
        for req in self.active_requests[:]:
            if req.is_dummy:
                self.inflight_req_ids.erase(req.request_id)
                self._terminate_request(req)
                self.active_requests.remove(req)

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
        return ctx_blocks, position_blocks, padding

    def _merge_star_attention_requests(self,
                                       new_requests: List[ExecutorRequest]):
        for request in new_requests:
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

        return False

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
            self.kv_cache_transceiver.check_gen_transfer_status(at_least_num)

        return

    @nvtx_range("_get_num_dummy_request")
    def _get_num_dummy_request(self):
        if self.enable_attention_dp:
            assert self.expected_num_active_requests >= len(
                self.active_requests)
            if self.kv_cache_transceiver is None:
                num_active_request = len(self.active_requests)
            else:
                num_active_request = sum([
                    0 if req.is_disagg_generation_init_state
                    or req.is_disagg_generation_transmission_in_progress else 1
                    for req in self.active_requests
                ])
            num_dummy_request = self.expected_num_active_requests - num_active_request
        else:
            num_dummy_request = 0
        return num_dummy_request

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
            return []
        for req in scheduled_ctx_requests:
            if req.is_context_only_request and req.is_context_finished:
                self.kv_cache_transceiver.respond_and_send_async(req)

        self.kv_cache_transceiver.check_context_transfer_status(False)

        # Keep track of ctx requests that are in transmission
        ctx_transmission_reqs = [
            req for req in scheduled_ctx_requests
            if req.state == LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS
        ]

        return ctx_transmission_reqs

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

    @nvtx_range("_prepare_draft_batch")
    def _prepare_draft_batch(
        self, scheduled_requests: ScheduledRequests
    ) -> Tuple[ScheduledRequests, Dict[int, LlmRequest]]:
        """
        Prepares a batch for the draft model engine. Draft tokens are only produced
        for generation requests.

        The requests are prepared as follows:
        1. The first time the draft engine sees a request, it's a context request.
        2. Otherwise, if draft tokens were accepted on the last target model decoding
        step, it's a chunked context request (we process all the accepted tokens together).
        3. Otherwise, it's a generation request.
        """
        try:
            draft_batch = ScheduledRequests()
            req_id_to_num_rejected_tokens = {}

            for request in scheduled_requests.generation_requests:
                if request.py_draft_pages_allocated == 0:
                    # No space for draft tokens.
                    continue

                num_draft_tokens = len(
                    request.py_last_draft_tokens
                ) if request.py_last_draft_tokens is not None else 0
                request.py_draft_tokens = []

                num_accepted_tokens = getattr(request,
                                              "py_num_accepted_draft_tokens", 0)
                num_rejected_tokens = num_draft_tokens - num_accepted_tokens
                assert num_rejected_tokens >= 0
                req_id_to_num_rejected_tokens[
                    request.py_request_id] = num_rejected_tokens

                spec_config = self.model_engine.spec_config
                beam_idx = 0
                input_tokens = spec_config.get_draft_model_prompt(
                    request.get_tokens()[beam_idx])

                if request.max_beam_num_tokens - 1 == request.py_prompt_len:
                    # This is the first time the draft model is seeing this request.
                    # Prepare a context request. We discard the first token and take
                    # the newly decoded one - this is the convention for EAGLE 2 and 3.
                    assert num_draft_tokens == 0
                    new_request = LlmRequest(
                        request_id=request.py_request_id,
                        max_new_tokens=request.py_max_new_tokens,
                        input_tokens=input_tokens,
                        sampling_config=request.sampling_config,
                        is_streaming=False)

                    draft_batch.context_requests.append(new_request)
                elif getattr(request, "py_num_accepted_draft_tokens", 0) == 0:
                    new_request = LlmRequest(
                        request_id=request.py_request_id,
                        max_new_tokens=request.py_max_new_tokens,
                        input_tokens=input_tokens[:-1],
                        sampling_config=request.sampling_config,
                        is_streaming=False)
                    # Explicitly add the last token so get_last_tokens() returns
                    # the right value
                    new_request.add_new_token(input_tokens[-1], beam_idx)
                    new_request.state = LlmRequestState.GENERATION_IN_PROGRESS
                    draft_batch.generation_requests.append(new_request)
                else:
                    new_request = LlmRequest(
                        request_id=request.py_request_id,
                        max_new_tokens=request.py_max_new_tokens,
                        input_tokens=input_tokens,
                        sampling_config=request.sampling_config,
                        is_streaming=False)
                    new_request.context_chunk_size = num_accepted_tokens + 1
                    new_request.context_current_position = len(
                        input_tokens) - num_accepted_tokens - 1

                    draft_batch.context_requests.append(new_request)

                new_request.py_stop_words_list = request.py_stop_words_list
                new_request.is_dummy = False

            return draft_batch, req_id_to_num_rejected_tokens

        except Exception as e:
            traceback.print_exc()
            error_msg = str(e)
            logger.error(f"Encountered an error in decode: {error_msg}")
            self._handle_errors(error_msg)

    @nvtx_range("_prepare_draft_tokens")
    def _prepare_draft_tokens(self, scheduled_requests: ScheduledRequests):
        try:
            draft_batch, num_rejected_tokens = self._prepare_draft_batch(
                scheduled_requests)

            if draft_batch.batch_size == 0:
                return

            req_id_to_old_request = {
                req.py_request_id: req
                for req in chain(scheduled_requests.context_requests,
                                 scheduled_requests.generation_requests)
            }

            spec_metadata = self.model_engine.last_spec_metadata

            hidden_states = spec_metadata.get_hidden_states(
                draft_batch, num_rejected_tokens)

            if spec_metadata.spec_dec_mode.is_eagle3():
                # Hack for eagle3. We might need to run a matmul to reduce
                # the dimensionality of the hidden states on the first pass
                # through the draft model. Shape dependent control flow will
                # not work with CUDA graphs. So we just do it here.
                hidden_states = self.draft_model_engine.model.apply_eagle3_fc(
                    hidden_states)

            extra_model_inputs = {'hidden_states': hidden_states}

            outputs = self.draft_model_engine.forward(
                draft_batch,
                self.resource_manager,
                extra_model_inputs=extra_model_inputs)

            if spec_metadata.spec_dec_mode.is_eagle3() and hasattr(
                    self.draft_model_engine.model.model, 'd2t'):
                outputs['d2t'] = self.draft_model_engine.model.model.d2t.data

            self._update_request_states(draft_batch)

            self._decode(draft_batch, outputs)

            def _process_decoded_tokens():
                new_requests = []
                for req in chain(draft_batch.context_requests,
                                 draft_batch.generation_requests):
                    target_model_req = req_id_to_old_request[req.py_request_id]
                    target_model_req.py_draft_tokens.append(
                        req.get_last_tokens(0))
                    if req.state != LlmRequestState.GENERATION_COMPLETE and len(
                            target_model_req.py_draft_tokens
                    ) < target_model_req.py_draft_pages_allocated:
                        new_requests.append(req)

                return new_requests

            new_requests = _process_decoded_tokens()
            if not new_requests:
                return

            draft_batch.generation_requests = new_requests
            draft_batch.context_requests = []

            for _ in range(spec_metadata.max_draft_tokens - 1):
                draft_spec_metadata = self.draft_model_engine.last_spec_metadata
                hidden_states = draft_spec_metadata.get_hidden_states(
                    draft_batch)
                extra_model_inputs = {'hidden_states': hidden_states}

                outputs = self.draft_model_engine.forward(
                    draft_batch,
                    self.resource_manager,
                    extra_model_inputs=extra_model_inputs)

                if spec_metadata.spec_dec_mode.is_eagle3() and hasattr(
                        self.draft_model_engine.model.model, 'd2t'):
                    outputs[
                        'd2t'] = self.draft_model_engine.model.model.d2t.data
                self._update_request_states(draft_batch)
                self._decode(draft_batch, outputs)

                new_requests = _process_decoded_tokens()
                if not new_requests:
                    return
                draft_batch.generation_requests = new_requests

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
        #TODO: properly handle canceled ids in pp case
        if self.dist.has_tp:
            self.canceled_req_ids = self.dist.broadcast(self.canceled_req_ids,
                                                        root=0)

        if len(self.canceled_req_ids) == 0:
            return

        cancelled_responses = {}
        left_requests = []
        # Tracks canceled requests for proper handling in overlap mode during `decoder.update_requests`.
        self.canceled_requests = []
        for request in self.active_requests:
            req_id = request.py_request_id
            if req_id in self.canceled_req_ids:
                self._terminate_request(request)
                request.finish_by_reason(FinishReason.CANCELLED)
                request.decoding_iter = request.py_decoding_iter
                cancelled_responses[req_id] = request.create_response(
                    False, self.dist.rank)
                self.canceled_requests.append(request)
                self.canceled_req_ids.erase(req_id)
            else:
                left_requests.append(request)
        self.active_requests = left_requests

        # enqueue the cancelled requests' responses as they are not
        # active_requests and be discarded in the decoder loop.
        self._enqueue_responses(cancelled_responses)

    @nvtx_range("_enqueue_responses")
    def _enqueue_responses(self, responses: Dict[int, ExecutorResponse]):
        if 0 not in self.dist.mapping.tp_group:
            return

        logger.debug(
            f'before gather, rank = {self.dist.rank}, responses = {responses}')
        if self.enable_attention_dp:
            responses_list = self.dist.tp_gather(responses)
            if self.dist.rank == 0:
                gather_responses = {}
                for resp in responses_list:
                    gather_responses.update(resp)
                responses = gather_responses
        logger.debug(
            f'after gather, rank = {self.dist.rank}, responses = {responses}')
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
            request.decoding_iter = request.py_decoding_iter
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
        for request in self.ctx_in_transmission_requests[:]:
            if request.is_disagg_context_complete_state:
                self._terminate_request(request)
                self.ctx_in_transmission_requests.remove(request)

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

    def _pause_requests(self, requests_to_pause):
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
