# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import datetime
import os
import threading
import traceback
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING

import torch

try:
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cudart

from tensorrt_llm._utils import CUASSERT, customized_gc_thresholds, release_gc
from tensorrt_llm.bindings.executor import FinishReason
from tensorrt_llm.executor.utils import check_prompt_capacity
from tensorrt_llm.llmapi.utils import _reapply_current_thread_affinity_to_all_threads
from tensorrt_llm.logger import logger
from tensorrt_llm.tools.profiler.host_profile_tools.host_profiler import host_profiler_context

from ..moe.fused_moe.moe_comm_timeout_guard import set_moe_comm_warmup
from .engine.model_setup import build_encoder_runner
from .engine.runners.encoder import EncoderRunner, resolve_scheduled_encoder_args
from .engine.runners.interface import ScheduledInputs
from .error_classification import ErrorBudget
from .executor_request_queue import ExecutorRequestQueue, RequestQueueItem
from .hang_detector import (
    HangDetector,
    distributed_warmup_guard,
    hard_kill_on_rank_crash,
    propagate_hard_kill,
    start_rank_crash_kill_watchdog,
)
from .llm_request import LlmRequest, LlmRequestState, LlmResponse
from .perf_metrics_manager import PerfMetricsManager
from .profiling import ProfileController
from .request_delivery import ResponseChannel
from .request_utils import RequestBroadcaster, drain_queued_items, merge_requests_to_llm_requests
from .scheduler import ScheduledRequests
from .scheduler.waiting_queue import create_waiting_queue

if TYPE_CHECKING:
    from ray.actor import ActorHandle

    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
    from tensorrt_llm.mapping import Mapping

    from ..distributed import Distributed
    from ..models.checkpoints.base_checkpoint_loader import BaseCheckpointLoader
    from .llm_request import ExecutorRequest
    from .model_loader import ModelLoader

__all__ = ["ScheduledEncoderExecutor", "fit_batch", "prompt_cost"]

_MULTIMODAL_INPUT_ERROR = (
    "Scheduled requests to a non-generation model cannot carry multimodal inputs."
)


def prompt_cost(item: RequestQueueItem) -> tuple[int, int]:
    """Rows and tokens of one queued request; each child request is another full row."""
    rows = 1 + len(item.child_req_ids or ())
    return rows, rows * item.request.num_input_tokens


def fit_batch(
    items: Sequence[RequestQueueItem],
    *,
    max_batch_size: int,
    max_num_tokens: int,
    cost: Callable[[RequestQueueItem], tuple[int, int]] = prompt_cost,
) -> int:
    """Count the leading items one forward can take; the first that does not fit starts the next batch."""
    rows = tokens = 0
    for count, item in enumerate(items):
        item_rows, item_tokens = cost(item)
        rows += item_rows
        tokens += item_tokens
        if rows > max_batch_size or tokens > max_num_tokens:
            return count
    return len(items)


class ScheduledEncoderExecutor:
    """Serve scheduled requests to a non-generation model: queue, admission, batching, one forward.

    Each request runs whole in one forward and completes with it, so there is no
    sampler, KV cache, generation scheduler or overlap loop. Rank 0 receives requests
    and publishes responses; every rank runs the same batches. Settings that only tune
    generation are ignored: the overlap scheduler, chunked prefill, guided decoding,
    ``scheduler_config``, the waiting-queue policy and the iteration-based batch waits.
    Iteration statistics and KV cache events are always empty, and ``print_iter_log``
    prints nothing.
    """

    is_generation_model = False
    kv_cache_transceiver = None

    def __init__(
        self,
        runner: EncoderRunner,
        *,
        model_loader: ModelLoader,
        llm_args: TorchLlmArgs,
        mapping: Mapping,
        dist: Distributed,
        max_batch_size: int,
        max_num_tokens: int,
        max_seq_len: int,
    ) -> None:
        """Assemble the executor around a built runner and warm the runner up on every rank.

        Args:
            runner: The model's runner; every forward goes through its scheduled entry point.
            model_loader: The loader that loaded the model, cleaned up at shutdown.
            llm_args: The resolved arguments; the executor reads its timeouts, garbage
                collection threshold and metrics switches.
            mapping: The parallel mapping, for the warmup's rank-crash handling.
            dist: The communicator that broadcasts requests from rank 0.
            max_batch_size: The most sequences one forward takes.
            max_num_tokens: The most prompt tokens one forward takes.
            max_seq_len: The longest prompt a request may have.
        """
        self._runner = runner
        self._model_loader = model_loader
        self.dist = dist
        self.max_batch_size = max_batch_size
        self.max_num_tokens = max_num_tokens
        self.max_seq_len = max_seq_len
        self.metrics: dict[str, float | dict[str, float]] = {}

        self.is_shutdown = False
        self.shutdown_event = threading.Event()
        self._event_loop_error: BaseException | None = None
        self._event_loop_error_delivered = threading.Event()
        self._loop_completed = False
        self._worker_thread: threading.Thread | None = None
        self._gc_gen0_threshold = llm_args.garbage_collection_gen0_threshold

        self._hang_detector = HangDetector(on_detected=self._on_hang_detected)
        self._request_queue = ExecutorRequestQueue(
            dist,
            max_batch_size,
            enable_iter_perf_stats=False,
            batch_wait_timeout_ms=llm_args.batch_wait_timeout_ms,
        )
        self._broadcaster = RequestBroadcaster(dist, self._hang_detector)
        self._waiting_queue = create_waiting_queue()
        self._error_budget = ErrorBudget()
        self._perf_manager = PerfMetricsManager(enabled=llm_args.return_perf_metrics)
        self._response_channel = ResponseChannel()
        self._iter_counter = 0
        self._profiler = ProfileController(
            rank=dist.rank,
            current_iteration=lambda: self._iter_counter,
            is_warmup=lambda: False,
            broadcast_start=self._request_queue.enqueue_profile_start_request,
            broadcast_stop=self._request_queue.enqueue_profile_stop_request,
        )

        if llm_args.enable_iter_perf_stats:
            logger.warning(
                "enable_iter_perf_stats is ignored for non-generation models: they report "
                "no iteration statistics."
            )

        self._device_id = torch.cuda.current_device()
        self._execution_stream = torch.cuda.Stream()
        self._warmup(mapping)

    @classmethod
    def create(
        cls,
        llm_args: TorchLlmArgs,
        *,
        model_path: str,
        checkpoint_loader: BaseCheckpointLoader,
        mapping: Mapping,
        dist: Distributed,
    ) -> ScheduledEncoderExecutor:
        """Load and compile the model, build its runner, and warm it up on every rank.

        Args:
            llm_args: The arguments; scheduled-encoder constraints are applied to them first.
            model_path: The checkpoint directory.
            checkpoint_loader: The loader for the checkpoint's format.
            mapping: The parallel mapping of this rank.
            dist: The communicator across the model's ranks.

        Returns:
            The executor, warmed up; its loop starts with ``start_worker``.
        """
        llm_args = resolve_scheduled_encoder_args(llm_args)
        with build_encoder_runner(
            llm_args,
            model_path=model_path,
            checkpoint_loader=checkpoint_loader,
            mapping=mapping,
            dist=dist,
        ) as built:
            if built.model_config.is_generation:
                raise RuntimeError(
                    "The checkpoint holds a generation model; it runs on PyExecutor."
                )
            return cls(
                built.runner,
                model_loader=built.model_loader,
                llm_args=llm_args,
                mapping=mapping,
                dist=dist,
                max_batch_size=built.max_batch_size,
                max_num_tokens=built.max_num_tokens,
                max_seq_len=built.max_seq_len,
            )

    @property
    def model_loader(self) -> ModelLoader:
        """The loader that loaded the model; the worker reports its load metrics at startup."""
        return self._model_loader

    def _warmup(self, mapping: Mapping) -> None:
        self.dist.barrier()
        set_moe_comm_warmup(True)
        try:
            self._execution_stream.wait_stream(torch.cuda.current_stream())
            with (
                torch.cuda.stream(self._execution_stream),
                distributed_warmup_guard(self.dist, mapping),
            ):
                self._runner.warmup()
            torch.cuda.current_stream().wait_stream(self._execution_stream)
        finally:
            set_moe_comm_warmup(False)

    def _on_hang_detected(self) -> None:
        try:
            logger.error(
                f"Hang detected on rank {self.dist.rank} in ScheduledEncoderExecutor; "
                "hard-killing and propagating to peer ranks."
            )
        finally:
            propagate_hard_kill()

    # The members the worker drives.

    def enqueue_request(
        self, request: ExecutorRequest, result_wait_queue: ActorHandle | None = None
    ) -> int:
        """Queue a request for admission and batching.

        Args:
            request: The request.
            result_wait_queue: A Ray result queue that receives the request's responses.

        Returns:
            The request id the executor answers under.
        """
        request_id = self._request_queue.enqueue_request(request)
        if result_wait_queue is not None:
            self._response_channel.register([request_id], result_wait_queue)
        return request_id

    def await_responses(
        self,
        id: list[int] | int | None = None,
        timeout: datetime.timedelta | None = None,
    ) -> list[list[LlmResponse]] | list[LlmResponse]:
        """Wait for responses, as ``PyExecutor.await_responses`` does.

        Args:
            id: None for any request's responses, one request id, or a list of ids.
            timeout: How long to wait; None waits until a response arrives or shutdown.

        Returns:
            Every ready response for None; that request's responses for an id;
            one list per id for a list. Empty on timeout.

        Raises:
            RuntimeError: The loop ended before a requested id had a response.
        """
        timeout = timeout.total_seconds() if timeout is not None else None
        if id is None:
            return self._response_channel.await_any(timeout, is_shutdown=lambda: self.is_shutdown)
        if isinstance(id, int):
            return self._await_single(id, timeout)
        return [self._await_single(request_id, timeout) for request_id in id]

    def _await_single(self, request_id: int, timeout: float | None) -> list[LlmResponse]:
        return self._response_channel.await_single(
            request_id,
            timeout,
            is_shutdown=lambda: self.is_shutdown,
            read_error=lambda: self._event_loop_error,
            mark_error_delivered=self._event_loop_error_delivered.set,
        )

    def cancel_request(self, id: int) -> None:
        """Cancel a request; a queued one finishes as cancelled, a finished one is ignored.

        Args:
            id: The request id.
        """
        self._request_queue.enqueue_cancel_request(id)

    def start_profile(
        self,
        output_dir: str | None = None,
        num_steps: int | None = None,
        start_step: int = 0,
        activities: list[str] | None = None,
    ) -> None:
        """Profile a window of loop iterations on every rank; see ``ProfileController.start_profile``."""
        self._profiler.start_profile(
            output_dir=output_dir,
            num_steps=num_steps,
            start_step=start_step,
            activities=activities,
        )

    def stop_profile(self) -> None:
        """Stop the profile window at the next loop iteration and wait for its trace."""
        self._profiler.stop_profile()

    def can_enqueue_requests(self) -> bool:
        """Whether the request queue still accepts requests."""
        return self._request_queue.can_enqueue_request()

    def get_latest_iteration_stats(self) -> list:
        """Iteration statistics, which this executor does not report."""
        return []

    def get_latest_kv_cache_events(self) -> list:
        """KV cache events; there is no KV cache."""
        return []

    def start_worker(self) -> None:
        """Start the loop thread; later calls do nothing."""
        if self._worker_thread is None:
            # Loading the model can start threads after the worker set its affinity;
            # an explicit opt-out stays one.
            if os.environ.get("TLLM_NUMA_AWARE_WORKER_AFFINITY") in (None, "1"):
                _reapply_current_thread_affinity_to_all_threads()
            self._worker_thread = threading.Thread(
                target=self._loop_wrapper, daemon=True, name="scheduled_encoder_executor"
            )
            self._worker_thread.start()

    def wait_shutdown(self) -> None:
        """Block until the loop has stopped; non-root ranks wait here."""
        self.shutdown_event.wait()

    def shutdown(self) -> None:
        """Let queued requests finish, stop the loop, then release the loader, the graphs and the model."""
        if self._worker_thread is not None:
            self._request_queue.enqueue_shutdown_request()
            self.shutdown_event.wait()
            if self._hang_detector.detected():
                logger.error("Hang detected, shutting down immediately.")
                return
            self._worker_thread.join()
        self._profiler.cleanup()
        self._model_loader.cleanup()
        self._runner.release_graphs()
        del self._runner, self._model_loader
        release_gc()

    # The loop.

    def _loop_wrapper(self) -> None:
        crashed = False
        try:
            with (
                host_profiler_context(enable=bool(os.environ.get("TLLM_LINE_PROFILER_PATH"))),
                customized_gc_thresholds(self._gc_gen0_threshold),
            ):
                self._loop()
        except Exception as error:
            # A raise after the loop's own exit is teardown; peers are only
            # stranded by one before it.
            crashed = not self._loop_completed
            logger.error(f"Error in event loop: {error}")
            logger.error(traceback.format_exc())
            self._event_loop_error = error
            raise
        finally:
            # Armed before cleanup, which can block on a wedged device.
            watchdog = (
                start_rank_crash_kill_watchdog(
                    self.dist.world_size, error_delivered=self._event_loop_error_delivered
                )
                if crashed
                else None
            )
            try:
                with self._response_channel.response_cv:
                    self.is_shutdown = True
                    self._response_channel.response_cv.notify_all()
                self.shutdown_event.set()
            finally:
                if crashed:
                    deadline = None
                    if watchdog is not None:
                        watchdog.cancel()
                        deadline = watchdog.deadline
                    hard_kill_on_rank_crash(
                        self.dist.world_size,
                        deadline=deadline,
                        error_delivered=self._event_loop_error_delivered,
                    )

    def _loop(self) -> None:
        torch.cuda.set_device(self._device_id)
        # Create the context before any MPI call.
        CUASSERT(cudart.cudaSetDevice(self._device_id))
        with self._profiler.iterations(), self._hang_detector:
            while True:
                self._hang_detector.checkpoint()
                self._profiler.end_iteration()
                self._profiler.begin_iteration()
                self._receive()
                if self.is_shutdown and not self._waiting_queue:
                    self._loop_completed = True
                    break
                count = fit_batch(
                    self._waiting_queue.peek_requests(self.max_batch_size),
                    max_batch_size=self.max_batch_size,
                    max_num_tokens=self.max_num_tokens,
                )
                if count:
                    self._run([self._waiting_queue.pop_request() for _ in range(count)])
                # Every pass counts, idle ones too, so a stop requested while idle fires.
                self._iter_counter += 1

    def _receive(self) -> None:
        """Fetch on rank 0, broadcast, then apply shutdown, cancellations and admission."""
        # Every rank computes the same value: it picks the broadcast transport.
        idle = not self._waiting_queue and not self.is_shutdown
        items = []
        if self.dist.rank == 0:
            # Once the shutdown signal is consumed nothing would wake a blocking fetch.
            timeout = None if idle else datetime.timedelta(0)
            with self._hang_detector.pause():
                items = self._request_queue.get_from_request_queue(timeout)
        items = self._broadcaster.broadcast(items, prefer_cpu=idle)

        canceled_ids = set()
        rejected = []
        for item in items:
            if item.is_shutdown_request:
                self.is_shutdown = True
                break
            if item.is_canceled_request:
                canceled_ids.add(item.id)
            elif item.is_profile_start_request:
                self._profiler.apply_start_config(item.profile_config or {})
            elif item.is_profile_stop_request:
                self._profiler.apply_stop_config()
            elif not item.is_normal_request:
                raise RuntimeError(f"Unsupported request queue item with id {item.id}.")
            else:
                try:
                    self._check_admissible(item)
                except ValueError as error:
                    rejected.append((item, str(error)))
                else:
                    self._waiting_queue.add_request(item)

        if self.dist.rank == 0:
            self._respond_with_errors(rejected)
        if canceled_ids:
            self._cancel_waiting(canceled_ids)

    def _check_admissible(self, item: RequestQueueItem) -> None:
        """Raise ValueError if one forward cannot run this request."""
        request = item.request
        if (
            getattr(request, "py_multimodal_data", None) is not None
            or request.mrope_config is not None
        ):
            raise ValueError(_MULTIMODAL_INPUT_ERROR)
        check_prompt_capacity(request.num_input_tokens, min(self.max_seq_len, self.max_num_tokens))
        rows, tokens = prompt_cost(item)
        if rows > self.max_batch_size or tokens > self.max_num_tokens:
            raise ValueError(
                f"The request runs {rows} sequences of {request.num_input_tokens} tokens in "
                f"one forward, which exceeds max_batch_size ({self.max_batch_size}) or "
                f"max_num_tokens ({self.max_num_tokens})."
            )

    def _cancel_waiting(self, request_ids: set[int]) -> None:
        """Finish queued requests as cancelled; ids no longer queued have already completed."""
        items = [item for item in self._waiting_queue if item.id in request_ids]
        if not items:
            return
        self._waiting_queue.remove_by_ids({item.id for item in items})
        if self.dist.rank != 0:
            return
        responses = []
        for item in items:
            for request in merge_requests_to_llm_requests(
                [item], exclude_last_generation_logits=False
            ):
                request.finish_by_reason(FinishReason.CANCELLED)
                response = request.create_response()
                if response is not None:
                    responses.append((item.id, response))
        self._publish(responses, (item.id for item in items))

    def _run(self, items: list[RequestQueueItem]) -> None:
        """Run one forward over ``items`` and publish one final response per request."""
        requests = merge_requests_to_llm_requests(items, exclude_last_generation_logits=False)
        for request in requests:
            if request.return_perf_metrics:
                request.set_first_scheduled_time()
        batch = ScheduledRequests()
        batch.context_requests_last_chunk = requests
        inputs = ScheduledInputs(
            batch=batch,
            gather_context_logits=any(request.py_return_context_logits for request in requests),
        )
        gpu_forward_start, gpu_forward_end, _ = self._perf_manager.create_timing_events()
        try:
            with torch.cuda.stream(self._execution_stream):
                with self._perf_manager.record_perf_events(
                    gpu_forward_start, gpu_forward_end
                ) as timing:
                    outputs = self._runner.forward(inputs, resource_manager=None)
                if self._perf_manager.enabled:
                    # There is no sampling: report a zero-length sample phase.
                    self._perf_manager.save_timing_to_requests(
                        requests,
                        gpu_forward_start,
                        gpu_forward_end,
                        None,
                        timing.start_time,
                        timing.end_time,
                        timing.end_time,
                        timing.end_time,
                    )
                responses = self._complete(requests, outputs)
                # Responses carry device tensors that other threads and processes read.
                self._execution_stream.synchronize()
        except Exception as error:
            self._fail(requests, error)
            return
        if self.dist.rank == 0:
            self._publish(responses, (request.py_request_id for request in requests))

    def _complete(
        self, requests: list[LlmRequest], outputs: dict[str, torch.Tensor]
    ) -> list[tuple[int, LlmResponse]]:
        """Store each request's pooled row as its context logits and report it finished with reason LENGTH."""
        batch_token_time = self._perf_manager.get_timestamp()
        gpu_times_cache = {}
        responses = []
        for request, row in zip(requests, outputs["logits"]):
            if request.py_return_context_logits:
                request.py_result.append_context_logits(row.unsqueeze(0) if row.ndim == 1 else row)
            # The public result reports a finished request with no tokens and reason LENGTH.
            request.state = LlmRequestState.GENERATION_COMPLETE
            request.set_finished_reason(FinishReason.LENGTH, 0)
            self._perf_manager.append_step_metrics(
                request, self._iter_counter, batch_token_time=batch_token_time
            )
            if request.return_perf_metrics:
                self._perf_manager.compute_batch_gpu_times(
                    [request], gpu_times_cache=gpu_times_cache
                )
            response = request.create_response()
            if response is not None:
                responses.append((request.py_request_id, response))
        return responses

    def _fail(self, requests: list[LlmRequest], error: Exception) -> None:
        """Fail the batch; a fatal error also fails everything queued and shuts the loop down.

        With several ranks, a fatal error then raises out of the loop: the loop's crash path
        kills every rank, because the peers' queues still hold work for a forward this rank
        will never join.
        """
        traceback.print_exc()
        error_msg = str(error) or "error"
        logger.error(f"Encountered an error in forward function: {error_msg}")
        # Child requests share their parent's client, which takes one error.
        responses = [
            (
                request.py_request_id,
                LlmResponse(
                    request_id=request.py_request_id,
                    error_msg=error_msg,
                    client_id=request.py_client_id,
                ),
            )
            for request in requests
            if not request.is_child
        ]
        released = [request.py_request_id for request in requests]
        fatal = self._error_budget.consume(error_msg)
        if fatal:
            logger.error(f"Fatal error detected, initiating shutdown: {error_msg}")
            self.is_shutdown = True
            drained = [
                item
                for item in drain_queued_items(self._waiting_queue, self._request_queue)
                if item.request is not None
            ]
            responses.extend(
                (
                    item.id,
                    LlmResponse(
                        request_id=item.id,
                        error_msg=error_msg,
                        client_id=item.request.client_id,
                    ),
                )
                for item in drained
            )
            released.extend(item.id for item in drained)
            self._request_queue.enqueue_shutdown_request()
        if self.dist.rank == 0:
            self._publish(responses, released)
        if fatal and self.dist.world_size > 1:
            raise RuntimeError(f"Fatal error: {error_msg}") from error

    def _respond_with_errors(self, rejected: list[tuple[RequestQueueItem, str]]) -> None:
        responses = [
            (
                item.id,
                LlmResponse(request_id=item.id, error_msg=error, client_id=item.request.client_id),
            )
            for item, error in rejected
        ]
        self._publish(responses, (item.id for item, _ in rejected))

    def _publish(
        self, responses: list[tuple[int, LlmResponse]], finished_ids: Iterable[int]
    ) -> None:
        if responses:
            self._response_channel.publish(responses)
        for request_id in finished_ids:
            self._response_channel.release(request_id)
