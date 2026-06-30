# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import threading
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.pyexecutor import py_executor as executor_module
from tensorrt_llm._torch.pyexecutor.executor_request_queue import (
    ExecutorRequestQueue,
    RequestQueueItem,
)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm.bindings.executor import FinishReason


class _WaitingQueue:
    def __init__(self, items=()):
        self._items = list(items)

    def __bool__(self):
        return bool(self._items)

    def pop_request(self):
        return self._items.pop(0)

    def remove_by_ids(self, request_ids):
        self._items = [item for item in self._items if item.id not in request_ids]


@pytest.fixture(autouse=True)
def _clear_unsafe_shutdown_quarantine():
    with PyExecutor._UNSAFE_TRANSFER_SHUTDOWN_QUARANTINE_LOCK:
        PyExecutor._UNSAFE_TRANSFER_SHUTDOWN_QUARANTINE.clear()
    yield
    with PyExecutor._UNSAFE_TRANSFER_SHUTDOWN_QUARANTINE_LOCK:
        PyExecutor._UNSAFE_TRANSFER_SHUTDOWN_QUARANTINE.clear()


def _request(request_id=7, state=LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS):
    response = SimpleNamespace(result=SimpleNamespace(cached_tokens=None))
    request = SimpleNamespace(
        py_request_id=request_id,
        py_client_id=17,
        is_child=False,
        state=state,
        py_decoding_iter=3,
        py_draft_tokens=[],
        cached_tokens=11,
        create_response=Mock(return_value=response),
        finish_by_reason=Mock(),
    )
    request.finish_by_reason.side_effect = lambda _: setattr(
        request, "state", LlmRequestState.GENERATION_COMPLETE
    )
    return request, response


def _cancel_executor(request):
    executor = object.__new__(PyExecutor)
    executor.active_requests = [request]
    executor.canceled_req_ids = [request.py_request_id]
    executor.previous_batch = None
    executor.waiting_queue = _WaitingQueue()
    executor.kv_cache_transceiver = Mock()
    executor.kv_cache_transceiver.supports_inflight_request_cancellation.return_value = True
    executor.async_transfer_manager = Mock()
    executor.async_transfer_manager.requests_in_transfer.return_value = {}
    executor.dist = SimpleNamespace(rank=0)
    executor._maybe_attach_ctx_usage = Mock()
    executor._enqueue_responses = Mock()
    executor._terminate_request = Mock()
    return executor


def _fatal_executor(active_requests=(), transfer_requests=(), *, should_store_blocks=False):
    executor = object.__new__(PyExecutor)
    executor.active_requests = list(active_requests)
    executor.waiting_queue = _WaitingQueue()
    raw_queue = Queue()
    executor.executor_request_queue = Mock()
    executor.executor_request_queue.get_request_queue.return_value = raw_queue
    executor.executor_request_queue.enqueue_shutdown_request.side_effect = lambda: raw_queue.put(
        RequestQueueItem(-1)
    )
    executor.previous_batch = None
    executor.request_accumulated = []
    executor.control_requests = []
    executor.control_request_barrier = Mock()
    executor.control_action_done = Mock()
    executor._pending_transfer_responses = []
    executor._fatal_transfer_cleanup_requests = {}
    executor._fatal_transfer_cleanup_skip_termination_ids = set()
    executor._fatal_transfer_shutdown = False
    executor._fatal_transfer_cleanup_complete = False
    executor._unsafe_transfer_shutdown = False
    executor._fatal_transfer_cleanup_deadline = None
    executor._fatal_transfer_cleanup_timer = None
    executor._fatal_transfer_cleanup_lock = threading.Lock()
    executor._start_fatal_transfer_cleanup_watchdog = Mock(
        side_effect=lambda: setattr(executor, "_fatal_transfer_cleanup_deadline", float("inf"))
    )
    executor._fatal_error = None
    executor._error_budget = Mock(budget=1.0)
    executor.is_shutdown = False
    executor.gather_all_responses = False
    executor.dist = SimpleNamespace(rank=0, world_size=1)
    executor.kv_cache_transceiver = Mock()
    executor.async_transfer_manager = Mock()
    executor.async_transfer_manager.should_store_blocks = should_store_blocks
    executor.async_transfer_manager.requests_in_transfer.return_value = dict(transfer_requests)
    executor._check_disagg_ctx_cache_transfer_status = Mock()
    executor._check_disagg_gen_cache_transfer_status = Mock()
    executor._enqueue_responses = Mock()
    executor._terminate_request = Mock()
    return executor


def test_context_success_waits_for_pending_user_cancel():
    request, response = _request(state=LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS)
    executor = _cancel_executor(request)
    executor._fatal_transfer_cleanup_requests = {}
    executor._pending_transfer_responses = []
    executor._is_disagg_inflight_cancel_active = Mock(return_value=True)

    def finish_transfer(_):
        request.state = LlmRequestState.DISAGG_CONTEXT_COMPLETE
        return True

    executor.async_transfer_manager.end_transfer.side_effect = finish_transfer

    PyExecutor._end_transfer_and_maybe_terminate(executor, request)

    request.create_response.assert_not_called()
    assert executor._pending_transfer_responses == []
    assert executor.active_requests == [request]
    executor._terminate_request.assert_not_called()

    executor.async_transfer_manager.requests_in_transfer.return_value = {}
    finished = PyExecutor._handle_canceled_requests(executor)
    PyExecutor._finish_canceled_requests_without_forward(executor, finished)

    request.finish_by_reason.assert_called_once_with(FinishReason.CANCELLED)
    request.create_response.assert_called_once_with(False, 0)
    assert response.result.cached_tokens == request.cached_tokens
    executor._enqueue_responses.assert_called_once_with([(request.py_request_id, response)])
    executor._terminate_request.assert_called_once_with(request)
    assert executor.active_requests == []
    assert executor.canceled_req_ids == []


def test_transfer_only_cancel_issues_once_then_finishes_after_poll():
    request, response = _request()
    executor = _cancel_executor(request)
    executor._fatal_error = None
    executor._is_disagg_inflight_cancel_active = Mock(return_value=True)
    executor._check_disagg_gen_cache_transfer_status = Mock(
        side_effect=lambda _: setattr(request, "state", LlmRequestState.DISAGG_TRANS_ERROR)
    )

    PyExecutor._check_disagg_gen_transfer_status(executor)

    executor.kv_cache_transceiver.cancel_request.assert_called_once_with(request)
    request.finish_by_reason.assert_called_once_with(FinishReason.CANCELLED)
    executor._enqueue_responses.assert_called_once_with([(request.py_request_id, response)])
    executor._terminate_request.assert_called_once_with(request)
    assert executor.canceled_req_ids == []


def test_transfer_only_cancel_does_not_terminate_previous_forward():
    request, _ = _request()
    executor = _cancel_executor(request)
    executor.previous_batch = SimpleNamespace(
        scheduled_requests=SimpleNamespace(all_requests=Mock(return_value=[request]))
    )

    finished = PyExecutor._handle_canceled_requests(executor, transfer_only=True)

    assert finished == []
    executor.kv_cache_transceiver.cancel_request.assert_not_called()
    request.finish_by_reason.assert_not_called()
    assert executor.canceled_req_ids == [request.py_request_id]


def test_cancel_waits_for_secondary_async_owner():
    request, _ = _request(state=LlmRequestState.DISAGG_TRANS_ERROR)
    executor = _cancel_executor(request)
    executor.async_transfer_manager.requests_in_transfer.return_value = {
        request.py_request_id: request
    }

    finished = PyExecutor._handle_canceled_requests(executor)

    assert finished == []
    request.finish_by_reason.assert_not_called()
    assert executor.canceled_req_ids == [request.py_request_id]


def test_active_timeout_is_owned_by_cpp_not_python():
    request, _ = _request()
    request.py_kv_transfer_start_time = 0.0
    request.py_kv_transfer_timed_out = False
    request.is_disagg_generation_transmission_in_progress = True
    executor = _cancel_executor(request)
    executor.kv_cache_transceiver.kv_transfer_timeout_ms = 1
    executor._is_disagg_inflight_cancel_active = Mock(return_value=True)

    PyExecutor._check_kv_transfer_timeout(executor)

    assert not request.py_kv_transfer_timed_out
    executor.async_transfer_manager.requests_in_transfer.assert_not_called()
    executor.kv_cache_transceiver.cancel_request.assert_not_called()


def test_poison_fatal_closes_and_drains_queues_without_freeing_active_requests():
    transfer_request, _ = _request()
    unrelated_transfer, _ = _request(request_id=8)
    waiting_request = SimpleNamespace(client_id=31)
    raw_request = SimpleNamespace(client_id=32)
    accumulated_request = SimpleNamespace(client_id=33)
    executor = object.__new__(PyExecutor)
    executor.active_requests = [transfer_request, unrelated_transfer]
    executor.waiting_queue = _WaitingQueue([RequestQueueItem(21, waiting_request)])
    raw_queue = Queue()
    raw_queue.put(RequestQueueItem(22, raw_request))
    executor.executor_request_queue = Mock()
    executor.executor_request_queue.get_request_queue.return_value = raw_queue
    executor.executor_request_queue.enqueue_shutdown_request.side_effect = lambda: raw_queue.put(
        RequestQueueItem(-1)
    )
    executor.request_accumulated = [RequestQueueItem(23, accumulated_request)]
    executor.control_requests = [RequestQueueItem(-2)]
    executor.control_request_barrier = Mock()
    executor._start_fatal_transfer_cleanup_watchdog = Mock()
    executor.previous_batch = None
    executor._pending_transfer_responses = []
    executor._fatal_transfer_cleanup_requests = {}
    executor._fatal_error = None
    executor.is_shutdown = False
    executor.gather_all_responses = False
    executor.dist = SimpleNamespace(rank=0)
    executor.kv_cache_transceiver = Mock()
    executor._enqueue_responses = Mock()
    executor._terminate_request = Mock()

    PyExecutor._handle_errors(
        executor, "poisoned", charge_budget=False, force_fatal=True, defer_transfer_cleanup=True
    )
    PyExecutor._handle_errors(
        executor, "poisoned", charge_budget=False, force_fatal=True, defer_transfer_cleanup=True
    )

    executor.executor_request_queue.enqueue_shutdown_request.assert_called_once_with()
    assert not executor.waiting_queue
    assert raw_queue.empty()
    assert executor.request_accumulated == []
    assert executor.control_requests == []
    executor.control_request_barrier.set.assert_called_once_with()
    assert executor.active_requests == []
    assert executor._fatal_transfer_cleanup_requests == {
        transfer_request.py_request_id: transfer_request,
        unrelated_transfer.py_request_id: unrelated_transfer,
    }
    executor._terminate_request.assert_not_called()
    executor.kv_cache_transceiver.cancel_request.assert_not_called()
    executor._enqueue_responses.assert_called_once()
    response_ids = {request_id for request_id, _ in executor._enqueue_responses.call_args.args[0]}
    assert response_ids == {
        transfer_request.py_request_id,
        unrelated_transfer.py_request_id,
        21,
        22,
        23,
    }


@pytest.mark.parametrize("should_store_blocks", [False, True])
def test_poison_tracks_detached_context_owner_until_quiescence(
    should_store_blocks,
):
    request, _ = _request(state=LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS)
    executor = _fatal_executor(
        transfer_requests={request.py_request_id: request},
        should_store_blocks=should_store_blocks,
    )

    PyExecutor._handle_errors(
        executor,
        "poisoned",
        charge_budget=False,
        force_fatal=True,
        defer_transfer_cleanup=True,
    )

    assert executor._fatal_transfer_cleanup_requests == {request.py_request_id: request}
    assert not PyExecutor._fatal_shutdown_complete(executor)
    executor.kv_cache_transceiver.cancel_request.assert_not_called()
    response_ids = {
        request_id
        for call_args in executor._enqueue_responses.call_args_list
        for request_id, _ in call_args.args[0]
    }
    assert request.py_request_id not in response_ids
    executor._terminate_request.assert_not_called()

    assert PyExecutor._progress_fatal_transfer_cleanup(executor)
    assert request.py_request_id in executor._fatal_transfer_cleanup_requests
    executor.kv_cache_transceiver.cancel_request.assert_called_once_with(request)
    assert not PyExecutor._fatal_shutdown_complete(executor)

    request.state = LlmRequestState.DISAGG_TRANS_ERROR
    executor.async_transfer_manager.requests_in_transfer.return_value = {}

    assert PyExecutor._progress_fatal_transfer_cleanup(executor)
    assert executor._fatal_transfer_cleanup_requests == {}
    assert PyExecutor._fatal_shutdown_complete(executor)
    if should_store_blocks:
        executor._terminate_request.assert_not_called()
    else:
        executor._terminate_request.assert_called_once_with(request)


def test_fatal_cleanup_reaps_only_after_native_owner_quiesces(monkeypatch):
    request, _ = _request()
    executor = _fatal_executor()
    executor._fatal_transfer_cleanup_requests = {request.py_request_id: request}
    executor._fatal_transfer_shutdown = True
    executor._fatal_transfer_cleanup_deadline = float("inf")
    executor._request_has_transfer_ownership = Mock(side_effect=[True, False])
    monkeypatch.setattr(executor_module.time, "sleep", Mock())

    assert PyExecutor._progress_fatal_transfer_cleanup(executor)
    assert request.py_request_id in executor._fatal_transfer_cleanup_requests
    executor._terminate_request.assert_not_called()

    assert PyExecutor._progress_fatal_transfer_cleanup(executor)
    assert executor._fatal_transfer_cleanup_requests == {}
    executor._terminate_request.assert_called_once_with(request)
    assert request.state == LlmRequestState.GENERATION_COMPLETE
    assert executor.kv_cache_transceiver.cancel_request.call_count == 2


def test_fatal_cleanup_deadline_marks_transfer_unsafe(monkeypatch):
    request, _ = _request()
    executor = _fatal_executor(
        active_requests=[request],
        transfer_requests={request.py_request_id: request},
    )
    monotonic = Mock(return_value=101.0)
    monkeypatch.setattr(PyExecutor, "FATAL_TRANSFER_CLEANUP_GRACE_PERIOD_S", 0.5)
    monkeypatch.setattr(executor_module.time, "monotonic", monotonic)

    PyExecutor._handle_errors(
        executor,
        "poisoned",
        charge_budget=False,
        force_fatal=True,
        defer_transfer_cleanup=True,
    )
    executor._fatal_transfer_cleanup_deadline = 100.0

    assert PyExecutor._progress_fatal_transfer_cleanup(executor)
    assert executor._unsafe_transfer_shutdown
    assert executor._fatal_transfer_cleanup_complete
    assert request.py_request_id in executor._fatal_transfer_cleanup_requests
    executor._terminate_request.assert_not_called()
    executor.kv_cache_transceiver.cancel_request.assert_called_once_with(request)
    executor._check_disagg_ctx_cache_transfer_status.assert_called_once_with(0)
    executor._check_disagg_gen_cache_transfer_status.assert_called_once_with(0)


def test_fatal_cleanup_watchdog_wakes_shutdown_while_cleanup_is_blocked(monkeypatch):
    request, _ = _request()
    executor = _fatal_executor(transfer_requests={request.py_request_id: request})
    executor._fatal_transfer_cleanup_requests = {request.py_request_id: request}
    executor._fatal_transfer_shutdown = True
    executor.shutdown_event = threading.Event()
    del executor._start_fatal_transfer_cleanup_watchdog

    timer = Mock()
    timer_factory = Mock(return_value=timer)
    monkeypatch.setattr(executor_module.threading, "Timer", timer_factory)
    monkeypatch.setattr(executor_module.time, "monotonic", Mock(return_value=100.0))

    entered = threading.Event()
    release = threading.Event()

    def block_cleanup():
        entered.set()
        release.wait()

    executor.previous_batch = SimpleNamespace(
        sample_state=SimpleNamespace(sampler_event=Mock(side_effect=block_cleanup))
    )
    PyExecutor._start_fatal_transfer_cleanup_watchdog(executor)
    watchdog_callback = timer_factory.call_args.args[1]
    cleanup_thread = threading.Thread(
        target=PyExecutor._progress_fatal_transfer_cleanup,
        args=(executor,),
        daemon=True,
    )
    cleanup_thread.start()
    assert entered.wait(timeout=1.0)

    watchdog_callback()

    assert executor.shutdown_event.wait(timeout=0.1)
    assert executor._unsafe_transfer_shutdown
    assert not executor._fatal_transfer_cleanup_complete
    assert not PyExecutor._fatal_shutdown_complete(executor)
    assert cleanup_thread.is_alive()
    executor._terminate_request.assert_not_called()
    assert any(
        quarantined is executor for quarantined in PyExecutor._UNSAFE_TRANSFER_SHUTDOWN_QUARANTINE
    )

    release.set()
    cleanup_thread.join(timeout=1.0)
    assert not cleanup_thread.is_alive()
    assert executor._fatal_transfer_cleanup_complete
    assert PyExecutor._fatal_shutdown_complete(executor)


@pytest.mark.parametrize("failure_site", ["cancel", "context_status"])
def test_fatal_cleanup_exception_marks_transfer_unsafe(failure_site):
    request, _ = _request()
    executor = _fatal_executor(transfer_requests={request.py_request_id: request})
    executor._fatal_transfer_cleanup_requests = {request.py_request_id: request}
    executor._fatal_transfer_shutdown = True
    executor._fatal_transfer_cleanup_complete = False
    executor._fatal_transfer_cleanup_deadline = float("inf")
    error = RuntimeError(f"{failure_site} failed")
    if failure_site == "cancel":
        executor.kv_cache_transceiver.cancel_request.side_effect = error
    else:
        executor._check_disagg_ctx_cache_transfer_status.side_effect = error

    assert PyExecutor._progress_fatal_transfer_cleanup(executor)

    assert executor._unsafe_transfer_shutdown
    assert executor._fatal_transfer_cleanup_complete
    assert request.py_request_id in executor._fatal_transfer_cleanup_requests
    executor._terminate_request.assert_not_called()
    executor._check_disagg_ctx_cache_transfer_status.assert_called_once_with(0)
    executor._check_disagg_gen_cache_transfer_status.assert_called_once_with(0)


def test_fatal_cleanup_empty_rank_participates_and_rejects_owner_mismatch():
    executor = _fatal_executor()
    executor._fatal_transfer_shutdown = True
    executor._fatal_transfer_cleanup_deadline = float("inf")
    executor.dist = SimpleNamespace(
        rank=0,
        world_size=2,
        tp_allgather=Mock(
            return_value=[
                (False, False, False, ()),
                (False, False, True, (7,)),
            ]
        ),
    )

    assert PyExecutor._progress_fatal_transfer_cleanup(executor)

    executor._check_disagg_ctx_cache_transfer_status.assert_called_once_with(0)
    executor._check_disagg_gen_cache_transfer_status.assert_called_once_with(0)
    executor.dist.tp_allgather.assert_called_once()
    assert executor._unsafe_transfer_shutdown
    assert executor._fatal_transfer_cleanup_complete


@pytest.mark.parametrize(
    ("unsafe", "has_pending"),
    [(True, False), (False, True)],
)
def test_unsafe_transfer_shutdown_skips_resource_teardown(unsafe, has_pending):
    executor = object.__new__(PyExecutor)
    executor.executor_request_queue = Mock()
    executor.shutdown_event = Mock()
    executor.hang_detector = Mock()
    executor.hang_detector.detected.return_value = False
    executor._fatal_transfer_shutdown = True
    executor._unsafe_transfer_shutdown = unsafe
    executor._fatal_transfer_cleanup_requests = {7: Mock()} if has_pending else {}
    executor.worker_thread = Mock()
    manager = Mock()
    executor.resource_manager = SimpleNamespace(resource_managers={"kv": manager})

    PyExecutor.shutdown(executor)

    executor.executor_request_queue.enqueue_shutdown_request.assert_called_once_with()
    executor.shutdown_event.wait.assert_called_once_with()
    executor.worker_thread.join.assert_not_called()
    manager.shutdown.assert_not_called()


def test_status_poll_uses_cpp_topology_poison_without_python_allreduce():
    executor = object.__new__(PyExecutor)
    executor.active_requests = []
    executor.canceled_req_ids = []
    executor.enable_attention_dp = False
    executor.dist = SimpleNamespace(world_size=2, tp_allreduce=Mock())
    executor.async_transfer_manager = Mock()
    executor.async_transfer_manager.requests_in_transfer.return_value = {}
    executor.kv_cache_transceiver = Mock()
    executor.kv_cache_transceiver.check_gen_transfer_status.return_value = None
    executor.kv_cache_transceiver.has_poisoned_transfer_buffer.return_value = False
    executor._is_disagg_inflight_cancel_active = Mock(return_value=True)

    PyExecutor._check_disagg_gen_cache_transfer_status(executor, 0)

    executor.kv_cache_transceiver.has_poisoned_transfer_buffer.assert_called_once_with()
    executor.dist.tp_allreduce.assert_not_called()


def test_poison_is_fatal_before_any_request_is_terminal():
    executor = object.__new__(PyExecutor)
    executor.active_requests = []
    executor.canceled_req_ids = []
    executor.kv_cache_transceiver = Mock()
    executor.kv_cache_transceiver.check_gen_transfer_status.return_value = None
    executor.kv_cache_transceiver.has_poisoned_transfer_buffer.return_value = True
    executor._is_disagg_inflight_cancel_active = Mock(return_value=True)
    executor._handle_errors = Mock()

    PyExecutor._check_disagg_gen_cache_transfer_status(executor, 0)

    executor._handle_errors.assert_called_once_with(
        "Error in kv cache transfer for generation requests; poisoned "
        "transfer buffer requires process restart",
        charge_budget=False,
        force_fatal=True,
        defer_transfer_cleanup=True,
    )


def test_fatal_poison_is_started_only_once():
    executor = object.__new__(PyExecutor)
    executor._fatal_error = RuntimeError("already fatal")
    executor._fatal_transfer_shutdown = True
    executor._error_budget = Mock()

    PyExecutor._handle_errors(
        executor, "poisoned", charge_budget=False, force_fatal=True, defer_transfer_cleanup=True
    )

    executor._error_budget.consume.assert_not_called()


def test_generic_fatal_preserves_overlap_previous_batch():
    request, _ = _request()
    previous_batch = SimpleNamespace(sample_state=SimpleNamespace(sampler_event=Mock()))
    raw_queue = Queue()
    executor = object.__new__(PyExecutor)
    executor.active_requests = [request]
    executor.waiting_queue = _WaitingQueue()
    executor.executor_request_queue = Mock()
    executor.executor_request_queue.get_request_queue.return_value = raw_queue
    executor.executor_request_queue.enqueue_shutdown_request.side_effect = lambda: raw_queue.put(
        RequestQueueItem(-1)
    )
    executor.previous_batch = previous_batch
    executor._pending_transfer_responses = []
    executor._fatal_transfer_cleanup_requests = {}
    executor._fatal_transfer_cleanup_skip_termination_ids = set()
    executor._fatal_transfer_shutdown = False
    executor._fatal_transfer_cleanup_complete = False
    executor._unsafe_transfer_shutdown = False
    executor._fatal_transfer_cleanup_deadline = None
    executor._fatal_error = None
    executor._error_budget = Mock(budget=0.0)
    executor._error_budget.consume.return_value = True
    executor.is_shutdown = False
    executor.gather_all_responses = False
    executor.dist = SimpleNamespace(rank=0)
    executor._enqueue_responses = Mock()
    executor._terminate_request = Mock()

    PyExecutor._handle_errors(executor, "sampling failed")

    assert executor.previous_batch is previous_batch
    previous_batch.sample_state.sampler_event.synchronize.assert_not_called()
    assert not PyExecutor._fatal_shutdown_complete(executor)
    executor.executor_request_queue.enqueue_shutdown_request.assert_called_with()
    assert executor.executor_request_queue.enqueue_shutdown_request.call_count == 2
    assert raw_queue.get_nowait().is_shutdown_request


def test_control_action_aborts_after_fatal_shutdown():
    executor = object.__new__(PyExecutor)
    executor.dist = SimpleNamespace(rank=0)
    executor.executor_request_queue = Mock()
    executor.control_request_barrier = Mock()
    executor.control_action_done = Mock()
    executor._fatal_error = RuntimeError("poisoned")
    yielded = False

    with pytest.raises(RuntimeError):
        with PyExecutor.control_action(executor):
            yielded = True

    assert not yielded
    executor.executor_request_queue.enqueue_control_request.assert_not_called()
    executor.control_request_barrier.wait.assert_not_called()
    executor.control_action_done.set.assert_not_called()


def test_control_action_rejects_ordinary_shutdown():
    executor = object.__new__(PyExecutor)
    executor.dist = SimpleNamespace(rank=0)
    executor.executor_request_queue = Mock()
    executor.control_request_barrier = Mock()
    executor.control_action_done = Mock()
    executor._fatal_error = None
    executor.is_shutdown = True
    yielded = False

    with pytest.raises(RuntimeError):
        with PyExecutor.control_action(executor):
            yielded = True

    assert not yielded
    executor.executor_request_queue.enqueue_control_request.assert_not_called()
    executor.control_request_barrier.wait.assert_not_called()
    executor.control_action_done.set.assert_not_called()


def test_control_action_rejects_queue_shutdown_race():
    executor = object.__new__(PyExecutor)
    executor.dist = SimpleNamespace(rank=0)
    executor.executor_request_queue = Mock()
    executor.executor_request_queue.enqueue_control_request.return_value = False
    executor.control_request_barrier = Mock()
    executor.control_action_done = Mock()
    executor._fatal_error = None
    executor.is_shutdown = False
    yielded = False

    with pytest.raises(RuntimeError):
        with PyExecutor.control_action(executor):
            yielded = True

    assert not yielded
    executor.executor_request_queue.enqueue_control_request.assert_called_once_with(drain=True)
    executor.control_request_barrier.wait.assert_not_called()
    executor.control_action_done.set.assert_not_called()


def test_control_enqueue_and_shutdown_are_lock_ordered():
    for _ in range(20):
        request_queue = ExecutorRequestQueue(
            dist=SimpleNamespace(rank=0),
            max_batch_size=8,
            enable_iter_perf_stats=False,
            batch_wait_timeout_ms=0,
        )
        start = threading.Barrier(3)
        control_accepted = []

        def enqueue_control():
            start.wait()
            control_accepted.append(request_queue.enqueue_control_request())

        def enqueue_shutdown():
            start.wait()
            request_queue.enqueue_shutdown_request()

        control_thread = threading.Thread(target=enqueue_control)
        shutdown_thread = threading.Thread(target=enqueue_shutdown)
        control_thread.start()
        shutdown_thread.start()
        start.wait()
        control_thread.join(timeout=1.0)
        shutdown_thread.join(timeout=1.0)

        assert not control_thread.is_alive()
        assert not shutdown_thread.is_alive()
        items = []
        raw_queue = request_queue.get_request_queue()
        while not raw_queue.empty():
            items.append(raw_queue.get_nowait())
        if control_accepted == [True]:
            assert [item.is_control_request for item in items] == [True, False]
            assert items[-1].is_shutdown_request
        else:
            assert control_accepted == [False]
            assert len(items) == 1
            assert items[0].is_shutdown_request


def test_waiting_control_action_aborts_when_fatal_shutdown_releases_barrier():
    executor = object.__new__(PyExecutor)
    executor.dist = SimpleNamespace(rank=0)
    executor.executor_request_queue = Mock()
    executor.control_request_barrier = Mock()
    executor.control_action_done = Mock()
    executor._fatal_error = None
    executor.control_request_barrier.wait.side_effect = lambda: setattr(
        executor, "_fatal_error", RuntimeError("poisoned")
    )
    yielded = False

    with pytest.raises(RuntimeError):
        with PyExecutor.control_action(executor):
            yielded = True

    assert not yielded
    executor.executor_request_queue.enqueue_control_request.assert_called_once_with(drain=True)
    executor.control_request_barrier.wait.assert_called_once_with()
    executor.control_request_barrier.clear.assert_called_once_with()
    executor.control_action_done.set.assert_not_called()
