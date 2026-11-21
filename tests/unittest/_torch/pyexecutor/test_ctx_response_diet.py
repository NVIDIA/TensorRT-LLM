# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parity tests for the context-server response diet in ``PyExecutor``.

Each trim (emit gate, empty-notify skip, connector page skip) must leave the
responses, the request lifecycle, the per-request perf metrics, the
collectives and the connector callbacks exactly as its kill switch does, and
must stay off on anything but a disaggregated context server.
"""

import copy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.pyexecutor import perf_metrics_manager, py_executor
from tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector import KvCacheConnectorScheduler
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.perf_metrics_manager import PerfMetricsManager
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.request_delivery import ResponseChannel

pytestmark = pytest.mark.cpu_only

KILL_SWITCHES = (
    "TRTLLM_DISABLE_CTX_RESPONSE_EMIT_GATE",
    "TRTLLM_DISABLE_EMPTY_RESPONSE_NOTIFY_SKIP",
    "TRTLLM_DISABLE_CONNECTOR_PAGE_SKIP",
)
TRIM_FLAGS = ("_resp_emit_gate", "_empty_resp_notify_skip", "_connector_page_skip")


@pytest.fixture(autouse=True)
def _clear_kill_switches(monkeypatch):
    for name in KILL_SWITCHES:
        monkeypatch.delenv(name, raising=False)


def _kv_cache_manager_v2(num_layer_groups=1):
    manager = object.__new__(KVCacheManagerV2)
    manager.impl = SimpleNamespace(layer_grouping=[[i] for i in range(num_layer_groups)])
    return manager


class _PageFreeScheduler:
    """A connector scheduler that declares it never reads the finished pages and keeps the base grouped callback."""

    uses_finished_block_ids = False
    request_finished_by_layer_group = KvCacheConnectorScheduler.request_finished_by_layer_group


def _page_free_scheduler(scheduler_cls=_PageFreeScheduler):
    return object.__new__(scheduler_cls)


def _armed_executor(
    *,
    context_only=True,
    model_context_only=False,
    scheduler=None,
    kv_cache_manager=None,
    capacity_only=False,
):
    """Run the real ``_init_ctx_response_diet`` on a minimal executor."""
    executor = object.__new__(PyExecutor)
    executor.model_engine = SimpleNamespace(
        model=SimpleNamespace(model=SimpleNamespace(disagg_context_only=model_context_only))
    )
    executor.kv_cache_manager = (
        _kv_cache_manager_v2() if kv_cache_manager is None else kv_cache_manager
    )
    executor.kv_connector_manager = SimpleNamespace(
        context_only=context_only,
        capacity_only=capacity_only,
        scheduler=_page_free_scheduler() if scheduler is None else scheduler,
    )
    PyExecutor._init_ctx_response_diet(executor)
    return executor


# ---------------------------------------------------------------------------
# Arming and kill switches
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("context_only", [True, False])
def test_only_a_context_server_arms_the_trims(context_only):
    executor = _armed_executor(context_only=context_only)
    assert [getattr(executor, flag) for flag in TRIM_FLAGS] == [context_only] * 3
    assert executor._response_gpu_times_cache() == ({} if context_only else None)


@pytest.mark.parametrize("switch, flag", list(zip(KILL_SWITCHES, TRIM_FLAGS)))
def test_each_kill_switch_disables_only_its_trim(monkeypatch, switch, flag):
    monkeypatch.setenv(switch, "1")
    executor = _armed_executor()
    for other in TRIM_FLAGS:
        assert getattr(executor, other) is (other != flag)


def test_model_context_flag_marks_a_context_server():
    executor = object.__new__(PyExecutor)
    executor.model_engine = SimpleNamespace(
        model=SimpleNamespace(model=SimpleNamespace(disagg_context_only=True))
    )
    executor.kv_connector_manager = None
    executor.kv_cache_manager = None
    PyExecutor._init_ctx_response_diet(executor)
    assert executor._resp_emit_gate and executor._empty_resp_notify_skip
    assert not executor._connector_page_skip


class _PageReadingScheduler(_PageFreeScheduler):
    uses_finished_block_ids = True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scheduler": _page_free_scheduler(_PageReadingScheduler)},
        {"scheduler": None, "kv_cache_manager": _kv_cache_manager_v2(num_layer_groups=2)},
    ],
    ids=["reads-pages", "two-layer-groups"],
)
def test_connector_page_skip_needs_proof_pages_are_unused(kwargs):
    executor = _armed_executor(**kwargs)
    assert not executor._connector_page_skip
    assert executor._resp_emit_gate


# ---------------------------------------------------------------------------
# W1: emit gate in _handle_responses
# ---------------------------------------------------------------------------


class _Event:
    """Ping-pong timing event; ``record`` overwrites what the last use measured."""

    def __init__(self):
        self.elapsed_ms = None
        self.elapsed_time_calls = 0

    def record(self, elapsed_ms):
        self.elapsed_ms = elapsed_ms

    def query(self):
        return True

    def synchronize(self):
        pass

    def elapsed_time(self, _end):
        self.elapsed_time_calls += 1
        return self.elapsed_ms


class _Request:
    """Request mirroring the result guard of ``LlmRequest::createResult``."""

    def __init__(self, request_id, *, chunks=1, streaming=False, dummy=False):
        self.py_request_id = request_id
        self.chunks_left = chunks
        self.streaming = streaming
        self.is_attention_dp_dummy = dummy
        self.is_finished = False
        self.in_generation = False
        self.py_kv_transfer_timed_out = False
        self.is_generation_only_request = False
        self.py_gen_first_emitted = False
        self.py_total_accepted_draft_tokens = 0
        self.state = None
        self.py_draft_tokens = []
        self.py_decoding_iter = 0
        self.return_perf_metrics = True
        self.py_perf_timing = None
        self.cached_tokens = 0
        self.py_per_pos_drafted = None
        self.py_per_pos_accepted = None
        self.py_total_draft_tokens = 0
        self.py_disaggregated_params = None
        self.is_disagg_context_complete_state = False
        self.is_disagg_context_transmission_state = False
        self.create_response_calls = 0

    @property
    def context_remaining_length(self):
        return self.chunks_left

    def update_perf_metrics(self, iter_counter):
        pass

    def run_step(self):
        if self.streaming:
            self.in_generation = True
            self.py_decoding_iter += 1
            return
        self.chunks_left -= 1
        if self.chunks_left == 0:
            self.py_decoding_iter = 1
            self.is_finished = True
            self.is_disagg_context_transmission_state = True

    def create_response(self, use_fast_logits=False, mpi_world_rank=0):
        self.create_response_calls += 1
        can_produce = self.is_finished or (self.streaming and self.in_generation)
        if not can_produce:
            return None
        timing = self.py_perf_timing
        return SimpleNamespace(
            request_id=self.py_request_id,
            result=SimpleNamespace(
                is_final=self.is_finished,
                decoding_iter=self.py_decoding_iter,
                time_breakdown=copy.deepcopy(
                    None
                    if timing is None
                    else (
                        timing.ctx_chunk_metrics,
                        timing.step_metrics,
                        timing.ctx_gpu_forward_time,
                        timing.ctx_gpu_sample_time,
                    )
                ),
            ),
        )


# Iteration -> request ids scheduled in that batch. Request 2 runs at
# iteration 0, idles at 1, and runs again at 2 and 3.
_SCHEDULE = {0: [1, 2, 4], 1: [1, 4], 2: [1, 2, 4], 3: [2, 3, 4], 4: [4], 5: [4]}


def _requests():
    return {
        1: _Request(1, chunks=3),
        2: _Request(2, chunks=3),
        3: _Request(3, chunks=1),
        4: _Request(4, streaming=True),
        5: _Request(5, dummy=True),
        6: _Request(6, chunks=2),  # preemption pending, never scheduled
    }


def _response_executor(requests, emit_gate):
    executor = object.__new__(PyExecutor)
    executor._resp_emit_gate = emit_gate
    executor.active_requests = list(requests.values())
    executor.perf_manager = PerfMetricsManager(enabled=True)
    executor.iter_counter = 0
    executor.stream_interval = 2
    executor.first_content_emit_iters = 0
    executor.dist = SimpleNamespace(rank=0, world_size=1)
    executor.force_terminate_ctx_for_partial_reuse = False
    executor.disable_overlap_scheduler = True
    executor._disagg_coordinator = SimpleNamespace(
        fail_timed_out=lambda requests: None,
        inflight_cancel_active=lambda: False,
        request_cancellation=lambda request: False,
    )
    executor._is_preemption_pending = lambda request: request.py_request_id == 6
    executor._maybe_attach_ctx_usage = lambda request, response: None
    executor.enqueued = []
    executor._enqueue_responses = lambda responses: executor.enqueued.append(
        [(req_id, copy.deepcopy(vars(resp.result))) for req_id, resp in responses]
    )
    executor.terminated = []
    executor._terminate_request = lambda request: executor.terminated.append(request.py_request_id)
    return executor


def _run_response_loop(emit_gate):
    """Drive executor iterations of schedule, forward, ``_handle_responses`` and GPU timing."""
    requests = _requests()
    executor = _response_executor(requests, emit_gate)
    event_sets = [(_Event(), _Event(), _Event()) for _ in range(2)]
    returned = []
    for iteration, scheduled_ids in _SCHEDULE.items():
        executor.iter_counter = iteration
        scheduled = [requests[i] for i in scheduled_ids if requests[i] in executor.active_requests]
        start, end, sample_end = event_sets[iteration % 2]
        start.record(1.0 + iteration)
        end.record(0.25 + iteration)
        executor.perf_manager.save_timing_to_requests(
            scheduled,
            start,
            end,
            sample_end,
            10.0 * iteration,
            10.0 * iteration + 1,
            10.0 * iteration + 2,
            10.0 * iteration + 3,
        )
        for request in scheduled:
            request.run_step()

        gpu_times_cache = executor._response_gpu_times_cache()
        done = PyExecutor._handle_responses(executor, gpu_times_cache=gpu_times_cache)
        returned.append(sorted(request.py_request_id for request in done))
        executor.perf_manager.compute_batch_gpu_times(scheduled, gpu_times_cache=gpu_times_cache)

    metrics = {
        req_id: copy.deepcopy(
            None
            if request.py_perf_timing is None
            else (
                request.py_perf_timing.ctx_chunk_metrics,
                request.py_perf_timing.step_metrics,
                request.py_perf_timing.ctx_gpu_forward_time,
                request.py_perf_timing.ctx_gpu_sample_time,
            )
        )
        for req_id, request in requests.items()
    }
    snapshot = {
        "enqueued": executor.enqueued,
        "returned": returned,
        "terminated": executor.terminated,
        "active": [request.py_request_id for request in executor.active_requests],
        "metrics": metrics,
    }
    calls = {req_id: request.create_response_calls for req_id, request in requests.items()}
    elapsed_calls = sum(event.elapsed_time_calls for events in event_sets for event in events)
    return snapshot, calls, elapsed_calls, executor


def test_emit_gate_is_bitwise_over_iterations(monkeypatch):
    # Both runs read one fixed nonzero clock, so the step metrics' token_time is comparable between them.
    for module in (perf_metrics_manager, py_executor):
        monkeypatch.setattr(module, "get_steady_clock_now_in_seconds", lambda: 123.5)
    gated, gated_calls, gated_elapsed, gated_executor = _run_response_loop(emit_gate=True)
    plain, plain_calls, plain_elapsed, plain_executor = _run_response_loop(emit_gate=False)

    assert gated == plain
    # The run exercises finished responses and chunk metrics with GPU times.
    assert any(resp["is_final"] for step in plain["enqueued"] for _, resp in step)
    assert plain["metrics"][2][2] == pytest.approx(1.0 + 3.0 + 4.0)
    assert plain["metrics"][2][0][0]["gpu_forward_time"] == 1.0
    # Only finished and streaming requests still create a response.
    assert gated_calls[1] == 1 and gated_calls[2] == 1 and gated_calls[3] == 1
    assert gated_calls[4] == plain_calls[4]
    assert sum(gated_calls.values()) < sum(plain_calls.values())
    assert gated_executor._resp_emit_gate_skips > 0
    assert plain_executor._resp_emit_gate_skips == 0
    # Sharing the cache keeps elapsed_time reads at one per batch.
    assert gated_elapsed <= plain_elapsed


def test_handle_responses_without_cache_keeps_the_gate_off():
    """Callers that pass no cache keep creating a response for every due request."""
    requests = _requests()
    executor = _response_executor(requests, emit_gate=True)
    PyExecutor._handle_responses(executor)
    assert requests[1].create_response_calls == 1
    assert executor._resp_emit_gate_skips == 0


# ---------------------------------------------------------------------------
# W2: empty-notify skip in _enqueue_responses
# ---------------------------------------------------------------------------


class _Condition:
    def __init__(self):
        self.notifies = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def notify_all(self):
        self.notifies += 1


class _Dist:
    def __init__(self, rank, gathered):
        self.rank = rank
        self.world_size = 2
        self.mapping = SimpleNamespace(tp_group=[0, 1])
        self.gathered = gathered
        self.collectives = []

    def tp_gather(self, responses):
        self.collectives.append(("tp_gather", list(responses)))
        return self.gathered if self.rank == 0 else None


def _enqueue_executor(rank, gathered, skip):
    executor = object.__new__(PyExecutor)
    executor._empty_resp_notify_skip = skip
    executor.dist = _Dist(rank, gathered)
    executor.enable_attention_dp = True
    executor.gather_all_responses = False
    executor._response_channel = ResponseChannel()
    executor._response_channel.response_cv = _Condition()
    return executor


@pytest.mark.parametrize(
    "rank, gathered",
    [(0, [[], []]), (0, [[], [(7, "resp")]]), (1, [[], [(7, "resp")]])],
    ids=["rank0-empty", "rank0-one", "rank1"],
)
def test_empty_notify_skip_is_bitwise(rank, gathered):
    runs = {}
    for skip in (True, False):
        executor = _enqueue_executor(rank, gathered, skip)
        local = [] if rank == 0 else [(9, "local")]
        PyExecutor._enqueue_responses(executor, local)
        runs[skip] = executor

    on, off = runs[True], runs[False]
    assert on.dist.collectives == off.dist.collectives
    assert on._response_channel.responses == off._response_channel.responses
    any_response = rank == 0 and any(gathered)
    assert on._response_channel.response_cv.notifies == int(any_response)
    assert off._response_channel.response_cv.notifies == int(rank == 0)
    assert on._empty_resp_notify_skips == int(rank == 0 and not any_response)


# ---------------------------------------------------------------------------
# W3: connector page skip in _start_connector_async_save
# ---------------------------------------------------------------------------


def _run_connector_save(page_skip):
    finished_calls = []

    def request_finished(request, block_ids, by_layer_group):
        finished_calls.append((request.request_id, list(block_ids), by_layer_group))
        return True

    manager = SimpleNamespace(
        has_outstanding_save=lambda request_id: False, request_finished=request_finished
    )
    kv_cache_manager = _kv_cache_manager_v2()
    kv_cache_manager.get_page_indices_by_layer_group = Mock(return_value=[[3, 4, 5]])
    executor = object.__new__(PyExecutor)
    executor._connector_page_skip = page_skip
    executor.kv_connector_manager = manager
    executor.kv_cache_manager = kv_cache_manager
    executor.async_transfer_manager = Mock()

    request = SimpleNamespace(request_id=11, py_request_id=11, is_dummy_request=False, state=None)
    PyExecutor._start_connector_async_save(executor, request)
    transfers = executor.async_transfer_manager.start_transfer.call_args_list
    return transfers, finished_calls, kv_cache_manager, executor


def test_connector_page_skip_is_bitwise():
    skipped, skipped_calls, skipped_cache, skipped_executor = _run_connector_save(True)
    gathered, gathered_calls, gathered_cache, gathered_executor = _run_connector_save(False)

    assert skipped == gathered and len(skipped) == 1
    # Both reach the flat callback; only the unused pages differ.
    assert [call[:2] for call in gathered_calls] == [(11, [3, 4, 5])]
    assert [call[:2] for call in skipped_calls] == [(11, [])]
    skipped_cache.get_page_indices_by_layer_group.assert_not_called()
    gathered_cache.get_page_indices_by_layer_group.assert_called_once()
    assert skipped_executor._connector_page_skips == 1
    assert gathered_executor._connector_page_skips == 0
