# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scheduled encoder executor behaviour, driven through the members the worker calls.

A fake runner answers each request with a row holding its request id. It writes into one
reused output buffer, as a captured graph does, and can hold or fail a forward. The
executor's queue, admission, batching, completion and delivery are real.
"""

import re
import threading
import time
from contextlib import contextmanager
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.pyexecutor import scheduled_encoder_executor
from tensorrt_llm._torch.pyexecutor.executor_request_queue import RequestQueueItem
from tensorrt_llm._torch.pyexecutor.scheduled_encoder_executor import (
    ScheduledEncoderExecutor,
    fit_batch,
)
from tensorrt_llm.bindings.executor import FinishReason, OutputConfig, Request, SamplingConfig
from tensorrt_llm.executor.utils import is_context_length_exceeded_message
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
    """Answer each request with a row of its request id, through one reused output buffer."""

    def __init__(self, max_batch_size: int):
        self.output = torch.zeros(max_batch_size, WIDTH, device="cuda")
        self.batches: list[list[int]] = []
        self.errors: list[Exception] = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()

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
        self.entered.set()
        assert self.release.wait(timeout=30)
        if self.errors:
            raise self.errors.pop(0)
        for row, request in enumerate(requests):
            self.output[row].fill_(request.py_request_id)
        return {"logits": self.output[: len(requests)]}


class _Harness:
    def __init__(self, *, max_batch_size=4, max_num_tokens=64, max_seq_len=32):
        self.runner = _Runner(max_batch_size)
        self.executor = ScheduledEncoderExecutor(
            self.runner,
            model_loader=Mock(),
            llm_args=SimpleNamespace(
                garbage_collection_gen0_threshold=20000,
                batch_wait_timeout_ms=0,
                return_perf_metrics=False,
                enable_iter_perf_stats=False,
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
        context_logits: bool = True,
        perf_metrics: bool = False,
        **attrs,
    ) -> int:
        request = Request(
            input_token_ids=list(range(1, num_tokens + 1)),
            max_tokens=1,
            output_config=OutputConfig(
                return_context_logits=context_logits,
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

    def shutdown(self) -> None:
        if not self.closed:
            self.closed = True
            self.runner.release.set()
            self.executor.shutdown()


@contextmanager
def _harness(**kwargs):
    h = _Harness(**kwargs)
    try:
        yield h
    finally:
        h.shutdown()


def test_batches_follow_the_budget_in_arrival_order():
    with _harness(max_batch_size=2, max_num_tokens=64, max_seq_len=64) as h:
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


def test_cancelling_a_queued_request_finishes_it_once_as_cancelled():
    with _harness(max_batch_size=1) as h:
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


def test_a_survivable_forward_error_fails_only_its_batch():
    with _harness(max_batch_size=1) as h:
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


def test_a_fatal_forward_error_fails_everything_queued_and_shuts_down():
    with _harness(max_batch_size=1) as h:
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


def test_shutdown_runs_queued_requests_first():
    with _harness(max_batch_size=1) as h:
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
