import dataclasses
import datetime
import functools
import gc
import os
import queue
import threading
import traceback
import weakref
from contextlib import contextmanager
from typing import Dict, List, Optional, Union

import torch

from ..._utils import nvtx_range
from ...logger import logger
from .decoder import *
from .distributed import *
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
                 max_input_len: int = 2048):
        super(PyExecutor, self).__init__()
        self.device_id = torch.cuda.current_device()
        self.request_queue = queue.Queue()

        # profile config
        self.profile_start_iters, self.profile_stop_iters = _load_iteration_indexes(
            PROFILE_START_STOP_ENV_VAR_NAME)
        self.gc_nvtx_watcher_handle = _gc_nvtx_watcher()

        # related modules
        self.resource_manager = resource_manager
        self.scheduler = scheduler
        self.model_engine = model_engine
        self.decoder = decoder
        self.dist = dist

        # enqueue and _fetch_new_requests used data
        self.enqueue_lock = threading.Lock()
        self.active = True
        self.next_req_id = 1

        self.shutdown_event = threading.Event()

        # response used data
        self.response_lock = threading.Lock()
        self.response_cv = threading.Condition(self.response_lock)
        self.responses = {}

        # todo: we need pass this by builder config from LLM and LLMargs
        self.max_input_len = max_input_len
        # _executor_loop private data
        self.max_num_active_requests = model_engine.get_max_num_sequences()
        self.active_requests = []
        self.previous_batch = None
        self.inflight_req_ids = tensorrt_llm.bindings.internal.batch_manager.ReqIdsSet(
        )
        self.canceled_req_ids = tensorrt_llm.bindings.internal.batch_manager.ReqIdsSet(
        )

        self.model_engine.warmup(self.resource_manager)

        self.is_shutdown = False
        event_loop = self._executor_loop_overlap if enable_overlap_scheduler else self._executor_loop
        self.worker_thread = threading.Thread(target=event_loop, daemon=True)
        self.worker_thread.start()

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

        def profile_step():
            nonlocal it, enabled

            if it in self.profile_stop_iters:
                assert enabled, "Inconsistent CUDA profiling state"
                torch.cuda.cudart().cudaProfilerStop()
                enabled = False

            it += 1

            if it in self.profile_start_iters:
                assert not enabled, "Inconsistent CUDA profiling state"
                torch.cuda.cudart().cudaProfilerStart()
                enabled = True

        try:
            yield profile_step
        finally:
            if enabled:
                # Stop on early exit / exception
                torch.cuda.cudart().cudaProfilerStop()

    def _executor_loop(self):
        torch.cuda.set_device(self.device_id)
        got_finish_signal = False
        with self._profiler() as profile_step:
            while not got_finish_signal or len(self.active_requests) > 0:
                profile_step()
                new_requests = self._fetch_new_requests()
                got_finish_signal = self._merge_requests(
                    new_requests) or got_finish_signal
                if got_finish_signal and len(self.active_requests) == 0:
                    break
                scheduled_batch = self._schedule()
                assert scheduled_batch.batch_size > 0, (
                    "fail to schedule any pending request, "
                    "probably run out of resource.")
                # print(
                #     f'has {len(self.active_requests)} active_request, scheduled {len(scheduled_batch.context_requests)} context requests and {len(scheduled_batch.generation_requests)} generation requests'
                # )
                self.pause_requests(scheduled_batch.paused_requests)
                self.resource_manager.prepare_resources(scheduled_batch)
                batch_outputs = self._forward_step(scheduled_batch)
                self._decode(scheduled_batch, batch_outputs)
                self._handle_cancelled_requests()
                self._handle_responses()
                self.resource_manager.update_resources(scheduled_batch)

        with self.response_cv:
            self.is_shutdown = True
            self.response_cv.notify_all()
        self.shutdown_event.set()

    def _executor_loop_overlap(self):
        torch.cuda.set_device(self.device_id)
        got_finish_signal = False
        with self._profiler() as profile_step:
            while not got_finish_signal or len(self.active_requests) > 0:
                profile_step()
                new_requests = self._fetch_new_requests()
                got_finish_signal = self._merge_requests(
                    new_requests) or got_finish_signal
                if got_finish_signal and len(self.active_requests) == 0:
                    break
                scheduled_batch = self._schedule()
                assert scheduled_batch.batch_size > 0, (
                    "fail to schedule any pending request, "
                    "probably run out of resource.")
                # print(
                #     f'has {len(self.active_requests)} active_request, scheduled {len(scheduled_batch.context_requests)} context requests and {len(scheduled_batch.generation_requests)} generation requests'
                # )

                previous_new_tokens_device = None
                if self.previous_batch is not None:
                    _, previous_new_tokens_device, _, _ = self.previous_batch

                self.resource_manager.prepare_resources(scheduled_batch)
                batch_outputs = self._forward_step(scheduled_batch,
                                                   previous_new_tokens_device)

                new_tokens_device, new_tokens_host, decoder_event = self._decode_async(
                    batch_outputs)

                if self.previous_batch is not None:
                    previous_scheduled_batch, _, previous_new_tokens_host, previous_decoder_event = self.previous_batch
                    self._update_requests(previous_scheduled_batch,
                                          previous_new_tokens_host,
                                          previous_decoder_event)
                    self._handle_cancelled_requests()
                    self._handle_responses()
                    self.resource_manager.update_resources(
                        previous_scheduled_batch)

                self.previous_batch = (scheduled_batch, new_tokens_device,
                                       new_tokens_host, decoder_event)

        with self.response_cv:
            self.is_shutdown = True
            self.response_cv.notify_all()
        self.shutdown_event.set()

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
        return new_requests

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
                self.active_requests.append(req)
            elif _is_cancel_request(req_item):
                self.canceled_req_ids.add(req_item)
        return got_finish_signal

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
                self.active_requests.append(req)
            elif _is_cancel_request(req_item):
                self.canceled_req_ids.add(req_item)
        return got_finish_signal

    @nvtx_range("_merge_requests")
    def _merge_requests(self, new_requests: List[ExecutorRequest]):
        cp_config = self.dist.cp_config
        if 'cp_type' in cp_config:
            cp_type = cp_config['cp_type']
            if cp_type == 'star_attention':
                return self._merge_star_attention_requests(new_requests)
            elif cp_type == 'ring_attention':
                assert False, 'unsupport ring attention now'
            else:
                assert False, f'unsupport sp type {cp_type}'
        else:
            return self._merge_tp_requests(new_requests)

    @nvtx_range("_schedule")
    def _schedule(self):
        scheduled_requests = ScheduledRequests()
        scheduled_requests.context_requests, scheduled_requests.generation_requests, scheduled_requests.paused_requests = \
            self.scheduler.schedule_request(self.active_requests, self.inflight_req_ids)
        return scheduled_requests

    def _forward_step(self,
                      scheduled_requests,
                      new_tokens_device: Optional[torch.Tensor] = None):

        @nvtx_range(
            f"_forward_step: {len(scheduled_requests.context_requests)} ctx reqs, {len(scheduled_requests.generation_requests)} gen reqs"
        )
        def forward(scheduled_requests, resource_manager, new_tokens_device):
            return self.model_engine.forward(scheduled_requests,
                                             resource_manager,
                                             new_tokens_device)

        try:
            outputs = forward(scheduled_requests, self.resource_manager,
                              new_tokens_device)
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

    @nvtx_range("_decode_async")
    def _decode_async(self, batch_outputs):
        try:
            if batch_outputs is not None:
                return self.decoder.decode_async(batch_outputs)
        except Exception as e:
            traceback.print_exc()
            error_msg = str(e)
            logger.error(f"Encountered an error in decode: {error_msg}")
            self._handle_errors(error_msg)

    @nvtx_range("_update_requests")
    def _update_requests(self, scheduled_requests: ScheduledRequests,
                         new_tokens_host: torch.tensor,
                         event: torch.cuda.Event):
        try:
            self.decoder.update_requests(scheduled_requests, new_tokens_host,
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

        if self.dist.has_tp() and len(self.canceled_req_ids) > 0:
            self.canceled_req_ids = self.dist.broadcast(self.canceled_req_ids,
                                                        root=0)

        left_requests = []
        for request in self.active_requests:
            req_id = request.py_request_id
            if req_id in self.canceled_req_ids:
                self._terminate_request(request)
            else:
                left_requests.append(request)
        self.active_requests = left_requests

    def _enqueue_responses(self, responses: Dict[int, ExecutorResponse]):
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
        finish_requests = []
        new_active_requests = []
        for request in self.active_requests:
            req_id = request.py_request_id
            response = request.create_response(False, self.dist.rank)
            request_done = False
            if response:
                request_done = response.result.is_final
                new_responses.update({req_id: response})
            if request_done:
                finish_requests.append(request)
            else:
                new_active_requests.append(request)
        self.active_requests = new_active_requests
        self._enqueue_responses(new_responses)
        for request in finish_requests:
            self._terminate_request(request)

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
            manager.shutdown()

    def enqueue_request(self,
                        request: ExecutorRequest,
                        query: Optional[List] = None):
        try:
            self.enqueue_lock.acquire()
            assert self.active, "PyExecutor has already been shutdown."
            req_id = self.next_req_id
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
            for request in requests:
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
        raise NotImplementedError("Cancel request not implemented")

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
        # raise NotImplementedError("get_latest_iteration_stats not implemented")
        # todo: implement it
        return []

    def get_latest_request_stats(self):
        #raise NotImplementedError("get_latest_iteration_stats not implemented")
        # todo: implement it
        return []

    def pause_requests(self, requests_to_pause):
        # todo: support work with self.inflight_req_ids.
        #       Currently, self.inflight_req_ids is not.
        max_input_len = self.max_input_len
        for req in requests_to_pause:
            req.pause(max_input_len)
            self._terminate_request(req)
