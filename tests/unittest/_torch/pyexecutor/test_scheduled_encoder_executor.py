# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scheduled encoder executor behaviour, driven through the members the worker calls.

A fake runner answers each request with a row holding its request id. As a pooling model it
writes into one reused output buffer, as a captured graph does; as a multimodal encoder it
returns one CPU embedding per request, which the completion exports through shared memory.
It can hold or fail a forward. The executor's queue, admission, batching, completion and
delivery are real.
"""

import gc
import re
import threading
import time
from contextlib import contextmanager, nullcontext
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.pyexecutor import scheduled_encoder_executor
from tensorrt_llm._torch.pyexecutor.executor_request_queue import RequestQueueItem
from tensorrt_llm._torch.pyexecutor.llm_request import PyResult
from tensorrt_llm._torch.pyexecutor.scheduled_encoder_executor import (
    MultimodalEmbeddingCompletion,
    PoolingCompletion,
    ScheduledEncoderExecutor,
    fit_batch,
)
from tensorrt_llm._torch.shared_tensor import SharedTensorContainer
from tensorrt_llm.bindings.executor import FinishReason, OutputConfig, Request, SamplingConfig
from tensorrt_llm.executor.utils import is_context_length_exceeded_message
from tensorrt_llm.llmapi.llm_args import SchedulerConfig, WaitingQueuePolicy
from tensorrt_llm.mapping import Mapping

WAIT = timedelta(seconds=30)
NO_WAIT = timedelta(seconds=0.2)
WIDTH = 3


def _item(request_id: int, num_tokens: int, children: int = 0) -> RequestQueueItem:
    return RequestQueueItem(
        request_id,
        SimpleNamespace(num_input_tokens=num_tokens),
        child_req_ids=list(range(100, 100 + children)) or None,
    )


@pytest.mark.cpu_only
def test_fit_batch_takes_every_item_within_budget():
    items = [_item(1, 10), _item(2, 20), _item(3, 30)]
    assert fit_batch(items, max_batch_size=3, max_num_tokens=60) == 3


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("max_batch_size", "max_num_tokens", "expected"),
    [(2, 100, 2), (8, 59, 2), (8, 29, 1), (8, 9, 0)],
)
def test_fit_batch_stops_at_the_first_item_that_does_not_fit(
    max_batch_size, max_num_tokens, expected
):
    # The third item would fit the token budget alone after the second; it still waits.
    items = [_item(1, 10), _item(2, 20), _item(3, 30), _item(4, 1)]
    assert (
        fit_batch(items, max_batch_size=max_batch_size, max_num_tokens=max_num_tokens) == expected
    )


@pytest.mark.cpu_only
def test_fit_batch_charges_each_child_request_a_full_row():
    items = [_item(1, 10, children=2), _item(2, 10)]
    assert fit_batch(items, max_batch_size=3, max_num_tokens=100) == 1
    assert fit_batch(items, max_batch_size=4, max_num_tokens=39) == 1
    assert fit_batch(items, max_batch_size=4, max_num_tokens=40) == 2


class _Runner:
    """Answer each request with a row of its request id."""

    def __init__(self, max_batch_size: int, multimodal: bool):
        self.multimodal = multimodal
        self.embedding_device = "cpu"
        self.output = torch.zeros(max_batch_size, WIDTH, device="cuda")
        self.batches: list[list[int]] = []
        self.errors: list[Exception] = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()
        # GPU time to stage the inputs and to compute, so that a batch stays in flight after
        # its launch returns.
        self.staging_cycles = 0
        self.gpu_cycles = 0
        self.on_forward = None
        self.staged: list[torch.cuda.Event] = []

    def hold(self) -> None:
        self.entered.clear()
        self.release.clear()

    def warmup(self) -> None:
        pass

    def release_graphs(self) -> None:
        pass

    def forward(self, inputs, *, resource_manager):
        assert resource_manager is None
        requests = inputs.batch.context_requests
        self.batches.append([request.py_request_id for request in requests])
        if self.on_forward is not None:
            self.on_forward()
        self.entered.set()
        assert self.release.wait(timeout=30)
        if self.errors:
            raise self.errors.pop(0)
        if self.staging_cycles:
            torch.cuda._sleep(self.staging_cycles)
        if inputs.inputs_staged is not None:
            inputs.inputs_staged.record()
            self.staged.append(inputs.inputs_staged)
        if self.gpu_cycles:
            torch.cuda._sleep(self.gpu_cycles)
        if self.multimodal:
            return {
                "mm_embeddings": [
                    torch.full(
                        (1, WIDTH), float(request.py_request_id), device=self.embedding_device
                    )
                    for request in requests
                ],
                "mm_embedding_request_indices": list(range(len(requests))),
                "mm_embedding_lengths": [[1]] * len(requests),
            }
        for row, request in enumerate(requests):
            self.output[row].fill_(request.py_request_id)
        return {"logits": self.output[: len(requests)]}


class _Harness:
    def __init__(
        self,
        *,
        completion="pooling",
        max_batch_size=4,
        max_num_tokens=64,
        max_seq_len=32,
        print_iter_log=False,
        scheduler_config=None,
    ):
        self.multimodal = multimodal = completion == "multimodal"
        self.runner = _Runner(max_batch_size, multimodal)
        self.executor = ScheduledEncoderExecutor(
            self.runner,
            completion=(
                MultimodalEmbeddingCompletion(exports=True) if multimodal else PoolingCompletion()
            ),
            model_loader=Mock(),
            llm_args=SimpleNamespace(
                garbage_collection_gen0_threshold=20000,
                batch_wait_timeout_ms=0,
                return_perf_metrics=False,
                enable_iter_perf_stats=False,
                print_iter_log=print_iter_log,
                scheduler_config=scheduler_config or SchedulerConfig(),
            ),
            mapping=Mapping(),
            dist=SimpleNamespace(rank=0, world_size=1, has_pp=False, barrier=lambda: None),
            max_batch_size=max_batch_size,
            max_num_tokens=max_num_tokens,
            max_seq_len=max_seq_len,
        )
        self.closed = False

    def submit(
        self,
        num_tokens: int = 4,
        *,
        context_logits: bool | None = None,
        perf_metrics: bool = False,
        **attrs,
    ) -> int:
        request = Request(
            input_token_ids=list(range(1, num_tokens + 1)),
            max_tokens=1,
            output_config=OutputConfig(
                return_context_logits=(
                    not self.multimodal if context_logits is None else context_logits
                ),
                exclude_input_from_output=True,
                return_perf_metrics=perf_metrics,
            ),
        )
        for name, value in attrs.items():
            setattr(request, name, value)
        return self.executor.enqueue_request(request)

    def responses(self, request_id: int, timeout: timedelta = WAIT) -> list:
        responses = self.executor.await_responses(request_id, timeout=timeout)
        for response in responses:
            if response.result is not None:
                response.result.deserialize()
        return responses

    def result(self, request_id: int):
        (response,) = self.responses(request_id)
        assert response.error_msg is None
        assert response.result.is_final
        return response.result

    def complete_after_the_loop_takes_the_cancellation(self, request_id: int) -> None:
        """Hold each launched batch's completion until the loop has taken this cancellation."""
        executor = self.executor
        taken = threading.Event()
        receive, cancel_launched = executor._receive, executor._cancel_launched

        def receive_and_report():
            receive()
            if request_id in executor._cancelled_after_launch:
                taken.set()

        def cancel_once_taken(flight):
            assert taken.wait(timeout=30)
            return cancel_launched(flight)

        executor._receive = receive_and_report
        executor._cancel_launched = cancel_once_taken

    def shutdown(self) -> None:
        if not self.closed:
            self.closed = True
            self.runner.release.set()
            self.executor.shutdown()
            # A crashed executor loop leaves its completion thread unjoined.
            completion = self.executor._completion_thread
            if completion is not None and completion.is_alive():
                completion.join(timeout=30)


@contextmanager
def _harness(**kwargs):
    h = _Harness(**kwargs)
    try:
        yield h
    finally:
        h.shutdown()


@pytest.fixture(params=["pooling", "multimodal"])
def completion(request):
    return request.param


def test_batches_follow_the_budget_in_arrival_order(completion):
    with _harness(completion=completion, max_batch_size=2, max_num_tokens=64, max_seq_len=64) as h:
        ids = [h.submit(), h.submit(), h.submit(61), h.submit()]
        h.executor.start_worker()

        for request_id in ids:
            h.result(request_id)

        assert h.runner.batches == [ids[:2], ids[2:3], ids[3:]]


def test_results_are_pooled_rows_with_no_tokens_and_reason_length():
    with _harness(max_batch_size=1) as h:
        pooled, unrequested = h.submit(), h.submit(context_logits=False)
        h.executor.start_worker()

        first = h.result(pooled)
        second = h.result(unrequested)

        # The runner rewrote its output buffer for the second batch.
        assert torch.equal(first.context_logits.cpu(), torch.full((1, WIDTH), float(pooled)))
        assert second.context_logits is None
        for result in (first, second):
            assert result.output_token_ids == [[]]
            assert result.finish_reasons == [FinishReason.LENGTH]


def test_multimodal_results_are_exported_embeddings_with_no_tokens_and_reason_length():
    with _harness(completion="multimodal", max_batch_size=2) as h:
        ids = [h.submit(py_multimodal_data={"image": {}}) for _ in range(3)]
        h.executor.start_worker()

        for request_id in ids:
            result = h.result(request_id)
            (handle,) = result.mm_embedding_handles
            embedding = SharedTensorContainer.from_dict(handle).get_local_view()
            assert torch.equal(embedding, torch.full((1, WIDTH), float(request_id)))
            assert result.output_token_ids == [[]]
            assert result.finish_reasons == [FinishReason.LENGTH]


# Long enough that a launched batch is still running when the loop has launched the next one.
GPU_CYCLES = 1_000_000_000


def test_the_next_batch_is_launched_while_the_previous_one_runs_and_two_are_in_flight_at_most():
    with _harness(max_batch_size=1) as h:
        h.runner.gpu_cycles = GPU_CYCLES
        published, published_at_launch = [], []
        publish = h.executor._publish

        def record_publish(responses, finished_ids):
            finished_ids = list(finished_ids)
            publish(responses, finished_ids)
            published.extend(finished_ids)

        h.executor._publish = record_publish
        h.runner.on_forward = lambda: published_at_launch.append(list(published))
        ids = [h.submit(), h.submit(), h.submit()]
        h.executor.start_worker()

        for request_id in ids:
            h.result(request_id)

        assert h.runner.batches == [[request_id] for request_id in ids]
        # The second launch did not wait for the first batch; the third waited for its publish.
        assert published_at_launch == [[], [], [ids[0]]]


def test_the_next_forward_waits_until_the_previous_inputs_are_staged():
    with _harness(max_batch_size=1) as h:
        h.runner.staging_cycles = GPU_CYCLES
        staged_at_launch = []
        h.runner.on_forward = lambda: staged_at_launch.append(
            [event.query() for event in h.runner.staged]
        )
        ids = [h.submit(), h.submit()]
        h.executor.start_worker()

        for request_id in ids:
            h.result(request_id)

        assert staged_at_launch == [[], [True]]


def test_a_failure_after_the_launch_ends_the_loop_and_wakes_waiters():
    with _harness() as h:

        def fail(responses, finished_ids):
            # The loop also publishes an empty list each pass; only a batch's responses fail.
            if responses:
                raise RuntimeError("the device was lost")

        h.executor._publish = fail
        pending = h.submit()
        h.executor.start_worker()

        with pytest.raises(RuntimeError, match="Event loop terminated") as raised:
            h.executor.await_responses(pending, timeout=WAIT)
        assert "the device was lost" in str(raised.value)


def test_perf_metrics_record_when_the_request_was_scheduled():
    with _harness() as h:
        request_id = h.submit(perf_metrics=True)
        h.executor.start_worker()

        timing = h.result(request_id).request_perf_metrics.timing_metrics
        assert timedelta(0) < timing.arrival_time <= timing.first_scheduled_time


def test_inadmissible_requests_fail_alone():
    with _harness(max_seq_len=32, max_num_tokens=64) as h:
        too_long = h.submit(33)
        multimodal = h.submit(py_multimodal_data={"image": {}})
        fitting = h.submit(32)
        h.executor.start_worker()

        (long_response,) = h.responses(too_long)
        (multimodal_response,) = h.responses(multimodal)
        h.result(fitting)

        assert is_context_length_exceeded_message(long_response.error_msg)
        assert "multimodal" in multimodal_response.error_msg
        assert h.runner.batches == [[fitting]]


@pytest.mark.parametrize(
    ("max_seq_len", "max_num_tokens"), [(32, 64), (64, 32)], ids=["max_seq_len", "max_num_tokens"]
)
def test_a_multimodal_encoder_admits_multimodal_requests_and_fails_an_oversized_one_alone(
    max_seq_len, max_num_tokens
):
    with _harness(
        completion="multimodal", max_seq_len=max_seq_len, max_num_tokens=max_num_tokens
    ) as h:
        too_long = h.submit(33, py_multimodal_data={"image": {}})
        fitting = h.submit(32, py_multimodal_data={"image": {}})
        h.executor.start_worker()

        (long_response,) = h.responses(too_long)
        h.result(fitting)

        assert is_context_length_exceeded_message(long_response.error_msg)
        assert h.runner.batches == [[fitting]]


@pytest.mark.parametrize("max_wait_s", [None, 60.0], ids=["priority", "aging"])
def test_a_priority_queue_admits_the_higher_priority_request_first(max_wait_s):
    with _harness(
        max_batch_size=1,
        scheduler_config=SchedulerConfig(
            waiting_queue_policy=WaitingQueuePolicy.PRIORITY, waiting_queue_max_wait_s=max_wait_s
        ),
    ) as h:
        h.runner.hold()
        running = h.submit()
        h.executor.start_worker()
        assert h.runner.entered.wait(timeout=30)
        low, high = h.submit(priority=0.1), h.submit(priority=0.9)
        h.runner.release.set()

        for request_id in (running, low, high):
            h.result(request_id)
        assert h.runner.batches == [[running], [high], [low]]


def test_a_request_past_the_wait_cap_goes_ahead_of_a_higher_priority_one():
    max_wait_s = 0.2
    with _harness(
        max_batch_size=1,
        scheduler_config=SchedulerConfig(
            waiting_queue_policy=WaitingQueuePolicy.PRIORITY, waiting_queue_max_wait_s=max_wait_s
        ),
    ) as h:
        h.runner.hold()
        running = h.submit()
        h.executor.start_worker()
        assert h.runner.entered.wait(timeout=30)
        # Arrivals are stamped at enqueue, so the older request is past the cap when popped.
        old = h.submit(priority=0.1)
        time.sleep(2 * max_wait_s)
        new = h.submit(priority=0.9)
        h.runner.release.set()

        for request_id in (running, old, new):
            h.result(request_id)
        assert h.runner.batches == [[running], [old], [new]]
        # Admitted requests leave no bookkeeping behind in the queue.
        assert not h.executor._waiting_queue._seq


def test_cancelling_a_queued_request_finishes_it_once_as_cancelled(completion):
    with _harness(completion=completion, max_batch_size=1) as h:
        h.runner.hold()
        running, queued = h.submit(), h.submit()
        h.executor.start_worker()
        assert h.runner.entered.wait(timeout=30)

        h.executor.cancel_request(queued)
        h.runner.release.set()

        h.result(running)
        (cancelled,) = h.responses(queued)
        assert cancelled.result.is_final
        assert cancelled.result.finish_reasons == [FinishReason.CANCELLED]

        h.executor.cancel_request(running)
        later = h.submit()
        h.result(later)
        assert h.responses(running, NO_WAIT) == []
        assert h.responses(queued, NO_WAIT) == []
        assert h.runner.batches == [[running], [later]]


def test_cancelling_a_launched_request_finishes_it_once_as_cancelled(completion):
    with _harness(completion=completion, max_batch_size=1) as h:
        h.runner.hold()
        launched, later = h.submit(), h.submit()
        h.complete_after_the_loop_takes_the_cancellation(launched)
        h.executor.start_worker()
        assert h.runner.entered.wait(timeout=30)

        h.executor.cancel_request(launched)
        h.runner.release.set()

        (cancelled,) = h.responses(launched)
        assert cancelled.result.is_final
        assert cancelled.result.finish_reasons == [FinishReason.CANCELLED]
        assert cancelled.result.mm_embedding_handles is None
        assert cancelled.result.context_logits is None
        h.result(later)
        assert h.responses(launched, NO_WAIT) == []
        assert h.runner.batches == [[launched], [later]]


def test_cancelling_a_launched_request_with_two_sequences_delivers_both_and_ends_with_the_last(
    completion,
):
    with _harness(completion=completion) as h:
        h.runner.hold()
        launched = h.submit(sampling_config=SamplingConfig(num_return_sequences=2))
        h.complete_after_the_loop_takes_the_cancellation(launched)
        h.executor.start_worker()
        assert h.runner.entered.wait(timeout=30)

        h.executor.cancel_request(launched)
        h.runner.release.set()

        # The worker takes every response; a child's is stored under its own id.
        responses = []
        while len(responses) < 2 and (ready := h.executor.await_responses(timeout=WAIT)):
            responses.extend(ready)
        for response in responses:
            response.result.deserialize()
        assert [response.request_id for response in responses] == [launched, launched]
        assert [response.result.sequence_index for response in responses] == [0, 1]
        assert [response.result.is_final for response in responses] == [False, True]
        for response in responses:
            assert response.result.finish_reasons == [FinishReason.CANCELLED]


def test_a_cancellation_sent_while_the_loop_waits_for_a_slot_still_cancels_a_launched_request(
    completion,
):
    with _harness(completion=completion, max_batch_size=1) as h:
        slots, waiting = h.executor._slots, threading.Event()

        class Slots:
            def acquire(self):
                if not slots.acquire(blocking=False):
                    waiting.set()
                    slots.acquire()

            def release(self):
                slots.release()

        h.executor._slots = Slots()
        sent = threading.Event()
        cancel_launched = h.executor._cancel_launched

        def cancel_once_sent(flight):
            assert sent.wait(timeout=30)
            return cancel_launched(flight)

        h.executor._cancel_launched = cancel_once_sent
        first, second, third = h.submit(), h.submit(), h.submit()
        h.executor.start_worker()
        # Two batches are in flight, and the loop waits in the third launch, not in _receive.
        assert waiting.wait(timeout=30)

        h.executor.cancel_request(first)
        sent.set()

        (cancelled,) = h.responses(first)
        assert cancelled.result.finish_reasons == [FinishReason.CANCELLED]
        assert cancelled.result.mm_embedding_handles is None
        assert cancelled.result.context_logits is None
        h.result(second)
        h.result(third)
        assert h.runner.batches == [[first], [second], [third]]


@pytest.mark.parametrize("rank, exports", [(0, True), (1, False)], ids=["rank_0", "tp_follower"])
def test_only_rank_0_of_a_multimodal_encoder_exports(monkeypatch, rank, exports):
    """A follower's responses reach no client, so its exports would pin each batch's memory."""
    built = SimpleNamespace(
        runner=_Runner(1, multimodal=True),
        model_config=SimpleNamespace(is_generation=False),
        model_loader=Mock(),
        max_batch_size=1,
        max_num_tokens=64,
        max_seq_len=32,
    )
    monkeypatch.setattr(
        scheduled_encoder_executor, "resolve_scheduled_encoder_args", lambda llm_args: llm_args
    )
    monkeypatch.setattr(
        scheduled_encoder_executor,
        "build_mm_encoder_runner",
        lambda *args, **kwargs: nullcontext(built),
    )
    executor = ScheduledEncoderExecutor.create(
        SimpleNamespace(
            mm_encoder_only=True,
            garbage_collection_gen0_threshold=20000,
            batch_wait_timeout_ms=0,
            return_perf_metrics=False,
            enable_iter_perf_stats=False,
            print_iter_log=False,
            scheduler_config=SchedulerConfig(),
        ),
        model_path="",
        checkpoint_loader=Mock(),
        mapping=Mapping(world_size=2, rank=rank, tp_size=2),
        dist=SimpleNamespace(rank=rank, world_size=2, has_pp=False, barrier=lambda: None),
    )
    request = SimpleNamespace(py_result=PyResult(prompt_len=1, max_new_tokens=1))

    executor._completion.store(
        [request],
        {
            "mm_embeddings": [torch.ones(1, WIDTH)],
            "mm_embedding_request_indices": [0],
            "mm_embedding_lengths": [[1]],
        },
    )

    assert (request.py_result.mm_embedding_handles is not None) is exports
    request.py_result.release_exports()


def _cuda_memory_allocated() -> int:
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.ipc_collect()
    return torch.cuda.memory_allocated()


@pytest.mark.parametrize("cause", ["cancelled", "failed"])
def test_a_launched_request_whose_result_is_not_delivered_frees_its_exports(cause):
    with _harness(completion="multimodal", max_batch_size=1) as h:
        h.runner.embedding_device = "cuda"
        if cause == "failed":

            def fail_after_the_export(*args, **kwargs):
                raise RuntimeError("bad step")

            h.executor._perf_manager.append_step_metrics = fail_after_the_export
        allocated = _cuda_memory_allocated()
        h.runner.hold()
        launched = h.submit(py_multimodal_data={"image": {}})
        if cause == "cancelled":
            h.complete_after_the_loop_takes_the_cancellation(launched)
        h.executor.start_worker()
        assert h.runner.entered.wait(timeout=30)
        if cause == "cancelled":
            h.executor.cancel_request(launched)
        h.runner.release.set()

        (response,) = h.responses(launched)
        undelivered = (
            response.result.finish_reasons == [FinishReason.CANCELLED]
            if cause == "cancelled"
            else response.error_msg == "bad step"
        )
        del response

        assert _cuda_memory_allocated() == allocated
        assert undelivered


def test_a_survivable_forward_error_fails_only_its_batch(completion):
    with _harness(completion=completion, max_batch_size=1) as h:
        h.runner.errors.append(RuntimeError("bad batch"))
        failed, later = h.submit(), h.submit()
        h.executor.start_worker()

        (error,) = h.responses(failed)
        assert error.error_msg == "bad batch"
        h.result(later)
        assert h.executor.can_enqueue_requests()


def test_a_failed_forward_answers_a_request_with_children_once():
    with _harness() as h:
        h.runner.errors.append(RuntimeError("bad batch"))
        failed = h.submit(sampling_config=SamplingConfig(num_return_sequences=2))
        h.executor.start_worker()

        (error,) = h.executor.await_responses(timeout=WAIT)
        assert (error.request_id, error.error_msg) == (failed, "bad batch")


def test_a_fatal_forward_error_fails_everything_queued_and_shuts_down(completion):
    with _harness(completion=completion, max_batch_size=1) as h:
        h.runner.hold()
        h.runner.errors.append(RuntimeError("an illegal memory access was encountered"))
        running, waiting = h.submit(), h.submit()
        h.executor.start_worker()
        assert h.runner.entered.wait(timeout=30)
        incoming = h.submit()
        h.runner.release.set()

        for request_id in (running, waiting, incoming):
            (response,) = h.responses(request_id)
            assert "illegal memory access" in response.error_msg
        assert h.executor.shutdown_event.wait(timeout=30)
        assert not h.executor.can_enqueue_requests()
        assert h.runner.batches == [[running]]


def test_a_fatal_error_with_peers_fails_everything_queued_and_kills_every_rank(monkeypatch):
    kills = []
    monkeypatch.setattr(
        scheduled_encoder_executor, "start_rank_crash_kill_watchdog", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        scheduled_encoder_executor,
        "hard_kill_on_rank_crash",
        lambda world_size, **kwargs: kills.append(world_size),
    )
    with _harness(max_batch_size=1) as h:
        h.executor.dist.world_size = 2
        h.executor._broadcaster.broadcast = lambda items, prefer_cpu: items
        h.runner.errors.append(RuntimeError("an illegal memory access was encountered"))
        running, waiting = h.submit(), h.submit()
        h.executor.start_worker()

        for request_id in (running, waiting):
            (response,) = h.responses(request_id)
            assert "illegal memory access" in response.error_msg
        h.executor._worker_thread.join(timeout=30)
        assert kills == [2]
        assert h.runner.batches == [[running]]


def test_print_iter_log_logs_each_batch_with_its_size_and_tokens(monkeypatch):
    lines = []
    monkeypatch.setattr(scheduled_encoder_executor.logger, "info", lines.append)
    with _harness(max_batch_size=2, print_iter_log=True) as h:
        first, second = h.submit(4), h.submit(4)
        h.executor.start_worker()
        h.result(first)
        h.result(second)

    (line,) = [line for line in lines if "num_scheduled_requests" in line]
    assert "num_scheduled_requests = 2, num_tokens = 8, num_waiting = 0, " in line
    assert "host_step_time = " in line and "device_step_time = " in line


def test_shutdown_runs_queued_requests_first(completion):
    with _harness(completion=completion, max_batch_size=1) as h:
        h.runner.hold()
        ids = [h.submit(), h.submit(), h.submit()]
        h.executor.start_worker()
        assert h.runner.entered.wait(timeout=30)

        stopper = threading.Thread(target=h.shutdown)
        stopper.start()
        h.runner.release.set()
        stopper.join(timeout=30)

        assert not stopper.is_alive()
        for request_id in ids:
            h.result(request_id)


def test_profiling_writes_a_trace_of_the_window(tmp_path):
    with _harness() as h:
        h.executor.start_worker()
        h.executor.start_profile(output_dir=str(tmp_path), activities=["CPU"])
        deadline = time.monotonic() + 30
        while not h.executor._profiler.enabled:
            assert time.monotonic() < deadline, "the profile window never opened"
            time.sleep(0.01)

        h.result(h.submit())
        h.executor.stop_profile()

        assert not h.executor._profiler.enabled
        (trace,) = tmp_path.glob("trtllm-trace-*-rank-0.json")
        assert trace.stat().st_size > 0


def test_line_profiler_profiles_the_loop_by_default(tmp_path, monkeypatch):
    output = tmp_path / "line_profile.txt"
    monkeypatch.setenv("TLLM_LINE_PROFILER_PATH", str(output))
    with _harness() as h:
        h.executor.start_worker()
        h.result(h.submit())

    timed = re.search(
        r"Total time:\s*([\d.e+-]+)\s*s\nFile:.*?\nFunction:.*ScheduledEncoderExecutor\._run\s",
        output.read_text(),
    )
    assert timed and float(timed.group(1)) > 0


def test_a_crashed_loop_wakes_waiters_with_its_error():
    with _harness() as h:
        pending = h.submit()
        # A control item only comes from an entry point this executor does not offer.
        h.executor._request_queue.enqueue_control_request()
        h.executor.start_worker()

        with pytest.raises(RuntimeError, match="Event loop terminated") as raised:
            h.executor.await_responses(pending, timeout=WAIT)
        assert "Unsupported request queue item" in str(raised.value.__cause__)
        assert h.executor.await_responses(timeout=WAIT) == []
