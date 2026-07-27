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

from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.py_executor import (
    _PYTHON_NATIVE_TRANSCEIVER_OWNER,
    _SHUTDOWN_CONNECTOR_COMPLETION_POLL_INTERVAL_S,
    _SHUTDOWN_CONNECTOR_COMPLETION_POLLS,
    DisaggPPTerminationHandler,
    PyExecutor,
    _TransferTerminalOutcome,
)


class _ExactTransferManager:
    """Small exact-request owner ledger for executor lifecycle tests."""

    def __init__(self, request, *, transceiver=False, anonymous=0, events=None):
        self._request = request
        self._transceiver = transceiver
        self._anonymous = anonymous
        self._events = events
        self.begin_shutdown = Mock()
        self.has_pending_admission = Mock(return_value=False)
        self.has_transfer_owner = Mock(side_effect=self._has_transfer_owner)
        self.has_any_transfer_owner = Mock(side_effect=self._has_any_transfer_owner)
        self.is_final_transfer_owner = Mock(side_effect=self._is_final_transfer_owner)
        self.requests_in_transfer = Mock(side_effect=self._requests_in_transfer)
        self.requests_with_owner = Mock(side_effect=self._requests_with_owner)
        self.end_transfer = Mock(side_effect=self._end_transfer)
        self.has_any_inflight_requests = Mock(side_effect=lambda: self._has_any_owner())

    def _has_any_owner(self):
        return self._transceiver or self._anonymous > 0

    def _has_exact_request(self, request):
        return request is self._request and self._has_any_owner()

    def _has_transfer_owner(self, request, owner):
        return (
            self._has_exact_request(request)
            and owner == _PYTHON_NATIVE_TRANSCEIVER_OWNER
            and self._transceiver
        )

    def _has_any_transfer_owner(self, request):
        return self._has_exact_request(request)

    def _is_final_transfer_owner(self, request, owner=None):
        assert request is self._request
        if owner is None:
            assert self._anonymous > 0
        else:
            assert owner == _PYTHON_NATIVE_TRANSCEIVER_OWNER
            assert self._transceiver
        return int(self._transceiver) + self._anonymous == 1

    def _requests_in_transfer(self):
        if self._has_any_owner():
            return {self._request.py_request_id: self._request}
        return {}

    def _requests_with_owner(self, owner):
        if owner == _PYTHON_NATIVE_TRANSCEIVER_OWNER and self._transceiver:
            return {self._request.py_request_id: self._request}
        return {}

    def _end_transfer(self, request, owner=None):
        assert request is self._request
        if owner is None:
            assert self._anonymous > 0
            self._anonymous -= 1
            if self._events is not None:
                self._events.connector_owner_release(request)
        else:
            assert owner == _PYTHON_NATIVE_TRANSCEIVER_OWNER
            assert self._transceiver
            self._transceiver = False
            if self._events is not None:
                self._events.transceiver_owner_release(request)
        return not self._has_any_owner()


def _initialize_bare_executor_lifecycle_state(executor):
    """Install constructor-owned state needed by unbound lifecycle methods."""
    executor.enable_attention_dp = False
    executor.dist = SimpleNamespace(rank=1, world_size=1)
    executor._disagg_timed_out_ctx_cancelled_ids = set()
    executor._disagg_timed_out_gen_cancelled_ids = set()
    executor._pending_native_context_completions = {}
    executor._pending_transfer_terminals = {}
    return executor


def _make_error_executor(request, *, fatal):
    executor = _initialize_bare_executor_lifecycle_state(object.__new__(PyExecutor))
    executor.active_requests = [request]
    executor._deferred_transfer_terminations = {}
    executor._error_budget = Mock(budget=1.0)
    executor._error_budget.consume.return_value = fatal
    executor._fatal_error = None
    executor.is_shutdown = False
    executor.waiting_queue = []
    executor.gather_all_responses = False
    executor.dist = SimpleNamespace(rank=0, world_size=1)
    executor.enable_attention_dp = False
    executor._enqueue_responses = Mock()
    executor._terminate_request = Mock()
    executor.kv_cache_transceiver = Mock()
    executor.kv_cache_transceiver.requires_physical_drain_before_request_release = True
    executor.force_terminate_ctx_for_partial_reuse = False
    executor._pending_connector_completions = {}
    executor._terminated_transfer_requests = {}
    executor.executor_request_queue = Mock()
    executor.executor_request_queue.get_request_queue.return_value = Queue()
    return executor


@pytest.mark.parametrize("fatal", [False, True])
def test_error_response_precedes_transfer_resource_termination(fatal):
    request = Mock(
        py_request_id=7,
        py_client_id=None,
        state=LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS,
    )
    request.is_generation_only_request.return_value = True
    executor = _make_error_executor(request, fatal=fatal)
    executor.kv_cache_transceiver.cancel_request.side_effect = [False, True]

    if fatal:
        executor._handle_errors("transfer failure")
    else:
        executor._handle_errors("transfer failure", requests=[request], charge_budget=False)

    assert request.state == LlmRequestState.GENERATION_COMPLETE
    assert executor.active_requests == []
    executor._enqueue_responses.assert_called_once()
    executor._terminate_request.assert_not_called()
    assert executor._deferred_transfer_terminations == {id(request): request}

    # This hook runs at the top of every executor iteration. The second
    # cancellation attempt proves physical drain and releases the request.
    executor._handle_disagg_cache_errors_synced()

    assert executor.kv_cache_transceiver.cancel_request.call_args_list == [
        call(request),
        call(request),
    ]
    executor._terminate_request.assert_called_once_with(request)
    assert executor._deferred_transfer_terminations == {}


def test_error_without_transceiver_does_not_defer_stale_transfer_state():
    request = Mock(
        py_request_id=7,
        py_client_id=None,
        state=LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS,
    )
    request.is_generation_only_request.return_value = True
    executor = _make_error_executor(request, fatal=False)
    executor.kv_cache_transceiver = None

    executor._handle_errors("transfer failure", requests=[request], charge_budget=False)

    executor._terminate_request.assert_called_once_with(request)
    assert executor._deferred_transfer_terminations == {}


def test_cpp_transceiver_keeps_existing_error_termination_contract():
    request = Mock(
        py_request_id=7,
        py_client_id=None,
        state=LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS,
    )
    request.is_generation_only_request.return_value = True
    executor = _make_error_executor(request, fatal=False)
    executor.kv_cache_transceiver.requires_physical_drain_before_request_release = False

    executor._handle_errors("transfer failure", requests=[request], charge_budget=False)

    executor.kv_cache_transceiver.cancel_request.assert_not_called()
    executor._terminate_request.assert_called_once_with(request)
    assert executor._deferred_transfer_terminations == {}


def test_receive_setup_error_still_uses_physical_drain_gate():
    request = Mock(
        py_request_id=7,
        py_client_id=None,
        state=LlmRequestState.DISAGG_TRANS_ERROR,
    )
    request.is_generation_only_request.return_value = True
    executor = _make_error_executor(request, fatal=False)
    executor.kv_cache_transceiver.cancel_request.return_value = False

    executor._handle_errors("receive setup failed", requests=[request], charge_budget=False)

    executor._terminate_request.assert_not_called()
    assert executor._deferred_transfer_terminations == {id(request): request}


def test_error_skips_pending_admission_but_terminates_sibling():
    pending = Mock(
        py_request_id=7,
        py_client_id=None,
        state=LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS,
    )
    sibling = Mock(
        py_request_id=8,
        py_client_id=None,
        state=LlmRequestState.GENERATION_COMPLETE,
    )
    executor = _make_error_executor(pending, fatal=False)
    executor.active_requests = [pending, sibling]
    executor.kv_cache_transceiver.requires_physical_drain_before_request_release = False
    executor.async_transfer_manager = Mock()
    executor.async_transfer_manager.has_pending_admission.side_effect = (
        lambda request: request is pending
    )

    executor._handle_errors(
        "admission failed",
        requests=[pending, sibling],
        charge_budget=False,
    )

    executor._terminate_request.assert_called_once_with(sibling)
    assert executor.active_requests == []


def test_error_skips_pending_connector_completion_but_terminates_sibling():
    pending = Mock(
        py_request_id=7,
        py_client_id=None,
        state=LlmRequestState.GENERATION_COMPLETE,
    )
    sibling = Mock(
        py_request_id=8,
        py_client_id=None,
        state=LlmRequestState.GENERATION_COMPLETE,
    )
    executor = _make_error_executor(pending, fatal=False)
    executor.active_requests = [pending, sibling]
    executor.kv_cache_transceiver.requires_physical_drain_before_request_release = False
    executor._pending_connector_completions = {id(pending): SimpleNamespace(request=pending)}

    executor._handle_errors(
        "connector cleanup failed",
        requests=[pending, sibling],
        charge_budget=False,
    )

    executor._terminate_request.assert_called_once_with(sibling)
    assert executor.active_requests == []


def test_context_error_releases_async_manager_only_after_physical_drain():
    request = Mock(
        py_request_id=7,
        py_client_id=None,
        state=LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS,
    )
    request.is_generation_only_request.return_value = False
    executor = _make_error_executor(request, fatal=False)
    executor.kv_cache_transceiver.cancel_request.side_effect = [False, True]
    executor.async_transfer_manager = Mock()
    executor.async_transfer_manager.requests_in_transfer.return_value = {7: request}
    executor.async_transfer_manager.has_transfer_owner.return_value = True

    def end_transfer(req, *, owner):
        assert owner == _PYTHON_NATIVE_TRANSCEIVER_OWNER
        if req.state != LlmRequestState.DISAGG_TRANS_ERROR:
            req.state = LlmRequestState.DISAGG_CONTEXT_COMPLETE
        return True

    executor.async_transfer_manager.end_transfer.side_effect = end_transfer
    executor.force_terminate_ctx_for_partial_reuse = False

    executor._handle_errors("context transfer failed", requests=[request], charge_budget=False)

    executor.async_transfer_manager.end_transfer.assert_not_called()
    executor._terminate_request.assert_not_called()

    executor._handle_disagg_cache_errors_synced()

    executor.async_transfer_manager.end_transfer.assert_called_once_with(
        request, owner=_PYTHON_NATIVE_TRANSCEIVER_OWNER
    )
    executor._terminate_request.assert_called_once_with(request)
    assert request.state == LlmRequestState.DISAGG_TRANS_ERROR
    assert executor._deferred_transfer_terminations == {}


def test_context_error_preserves_error_state_for_remaining_transfer_owner():
    request = Mock(
        py_request_id=7,
        state=LlmRequestState.GENERATION_COMPLETE,
    )
    request.is_generation_only_request.return_value = False
    executor = _make_error_executor(request, fatal=False)
    executor._deferred_transfer_terminations = {id(request): request}
    executor.async_transfer_manager = Mock()
    executor.async_transfer_manager.requests_in_transfer.return_value = {7: request}
    executor.async_transfer_manager.has_transfer_owner.return_value = True
    executor.async_transfer_manager.end_transfer.return_value = False
    executor.force_terminate_ctx_for_partial_reuse = False

    assert executor._finalize_deferred_transfer_termination(request)

    assert request.state == LlmRequestState.DISAGG_TRANS_ERROR
    executor._terminate_request.assert_not_called()
    assert executor._deferred_transfer_terminations == {}


def _make_native_launch_executor(request):
    executor = _initialize_bare_executor_lifecycle_state(object.__new__(PyExecutor))
    executor.kv_cache_manager = SimpleNamespace(release_index_slot=Mock())
    executor.kv_cache_transceiver = Mock(
        requires_physical_drain_before_request_release=True,
        kv_transfer_timeout_ms=None,
    )
    executor.async_transfer_manager = Mock()
    executor.async_transfer_manager.requests_in_transfer.return_value = {
        request.py_request_id: request
    }
    executor.async_transfer_manager.has_transfer_owner.return_value = True
    executor.async_transfer_manager.has_any_transfer_owner.return_value = True
    executor.async_transfer_manager.end_transfer.return_value = True
    executor.kv_connector_manager = None
    executor._deferred_transfer_terminations = {}
    executor._deferred_transfer_terminations_already_terminated = set()
    executor._terminated_transfer_requests = {}
    executor.force_terminate_ctx_for_partial_reuse = False
    executor._terminate_request = Mock()
    return executor


def _make_native_launch_request():
    request = Mock(
        py_request_id=7,
        state=LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS,
        is_context_only_request=True,
        is_context_finished=True,
        is_finished_due_to_length=False,
        is_finished_due_to_cancellation=False,
    )
    request.is_generation_only_request.return_value = False
    return request


def _make_native_completion_executor(request):
    executor = _initialize_bare_executor_lifecycle_state(object.__new__(PyExecutor))
    executor.kv_cache_transceiver = Mock(requires_physical_drain_before_request_release=True)
    executor.kv_cache_transceiver.check_context_transfer_status.side_effect = [
        ([request.py_request_id], []),
        ([], []),
    ]
    executor.async_transfer_manager = _ExactTransferManager(request, transceiver=True)
    executor.active_requests = [request]
    executor._deferred_transfer_terminations = {}
    executor._terminated_transfer_requests = {}
    executor.force_terminate_ctx_for_partial_reuse = False
    executor._pending_transfer_responses = []
    executor._maybe_attach_ctx_usage = Mock()
    executor._terminate_request = Mock()
    executor._check_cache_transfer_errors = Mock()
    return executor


def _make_native_completion_request():
    request = Mock(
        py_request_id=7,
        py_kv_transfer_timed_out=False,
        state=LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS,
        cached_tokens=11,
    )
    request.is_generation_only_request.return_value = False
    request.create_response.return_value = Mock(result=SimpleNamespace())
    return request


@pytest.mark.parametrize("requires_physical_drain", [False, True])
def test_transceiver_owner_mode_boundary(requires_physical_drain):
    request = _make_native_launch_request()
    request.py_kv_transfer_timed_out = False
    executor = _make_native_launch_executor(request)
    executor.kv_cache_transceiver.requires_physical_drain_before_request_release = (
        requires_physical_drain
    )
    executor.kv_cache_transceiver.check_context_transfer_status.return_value = ([], [])
    executor.active_requests = []

    executor._send_kv_async([request])

    expected_call = (
        call(request, owner=_PYTHON_NATIVE_TRANSCEIVER_OWNER)
        if requires_physical_drain
        else call(request)
    )
    assert executor.async_transfer_manager.start_transfer.call_args == expected_call
    executor.kv_cache_transceiver.respond_and_send_async.assert_called_once_with(request)
    executor.kv_cache_transceiver.cancel_request.assert_not_called()


@pytest.mark.parametrize("requires_physical_drain", [False, True])
def test_context_status_owner_mode_boundary(requires_physical_drain):
    request = Mock(
        py_request_id=7,
        py_kv_transfer_timed_out=False,
        state=LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS,
    )
    executor = _initialize_bare_executor_lifecycle_state(object.__new__(PyExecutor))
    executor.kv_cache_transceiver = Mock(
        requires_physical_drain_before_request_release=requires_physical_drain
    )
    executor.kv_cache_transceiver.check_context_transfer_status.return_value = ([7], [])
    executor.async_transfer_manager = Mock()
    executor.async_transfer_manager.requests_in_transfer.return_value = {7: request}
    executor.async_transfer_manager.end_transfer.return_value = True
    executor.active_requests = []
    executor._deferred_transfer_terminations = {}
    executor._terminated_transfer_requests = {}
    executor.force_terminate_ctx_for_partial_reuse = False
    executor._terminate_request = Mock()
    executor._finalize_deferred_transfer_termination = Mock(return_value=False)
    executor._end_transfer_and_maybe_terminate = Mock()
    executor._check_cache_transfer_errors = Mock()

    executor._check_disagg_ctx_cache_transfer_status()

    if requires_physical_drain:
        executor._end_transfer_and_maybe_terminate.assert_not_called()
        executor.async_transfer_manager.end_transfer.assert_called_once_with(
            request, owner=_PYTHON_NATIVE_TRANSCEIVER_OWNER
        )
        executor._terminate_request.assert_called_once_with(request)
        assert executor._pending_native_context_completions == {}
    else:
        executor._end_transfer_and_maybe_terminate.assert_called_once_with(
            request, transfer_owner=None
        )


def test_native_completion_response_preparation_retries_exact_response():
    request = _make_native_completion_request()
    response = request.create_response.return_value
    executor = _make_native_completion_executor(request)
    executor._maybe_attach_ctx_usage.side_effect = [
        RuntimeError("usage unavailable"),
        None,
    ]

    executor._check_disagg_ctx_cache_transfer_status()

    progress = executor._pending_native_context_completions[id(request)]
    assert progress.request is request
    terminal = executor._pending_transfer_terminals[id(request)]
    assert terminal.response is response
    assert executor._pending_transfer_responses == []
    executor.async_transfer_manager.end_transfer.assert_not_called()

    executor._check_disagg_ctx_cache_transfer_status()

    assert executor._pending_native_context_completions == {}
    request.create_response.assert_called_once_with(False, 1)
    assert executor._pending_transfer_responses == [(7, response)]
    executor.async_transfer_manager.end_transfer.assert_called_once_with(
        request, owner=_PYTHON_NATIVE_TRANSCEIVER_OWNER
    )
    executor._terminate_request.assert_called_once_with(request)


def test_native_completion_unpin_failure_retries_named_owner_once_resolved():
    request = _make_native_completion_request()
    executor = _make_native_completion_executor(request)
    manager = executor.async_transfer_manager
    release_attempts = 0

    def release_owner(req, owner=None):
        nonlocal release_attempts
        release_attempts += 1
        if release_attempts == 1:
            raise RuntimeError("unpin failed")
        return manager._end_transfer(req, owner)

    manager.end_transfer.side_effect = release_owner

    executor._check_disagg_ctx_cache_transfer_status()

    progress = executor._pending_native_context_completions[id(request)]
    assert executor._pending_transfer_terminals[id(request)].response_buffered
    assert not progress.owner_released
    executor._terminate_request.assert_not_called()

    executor._check_disagg_ctx_cache_transfer_status()

    assert executor._pending_native_context_completions == {}
    assert manager.end_transfer.call_args_list == [
        call(request, owner=_PYTHON_NATIVE_TRANSCEIVER_OWNER),
        call(request, owner=_PYTHON_NATIVE_TRANSCEIVER_OWNER),
    ]
    request.create_response.assert_called_once()
    assert len(executor._pending_transfer_responses) == 1
    executor._terminate_request.assert_called_once_with(request)


def test_late_cancel_does_not_compete_with_consumed_native_completion():
    request = _make_native_completion_request()
    request.is_child = False
    request.parent_request_id = None
    executor = _make_native_completion_executor(request)
    manager = executor.async_transfer_manager
    release_attempts = 0

    def release_owner(req, owner=None):
        nonlocal release_attempts
        release_attempts += 1
        if release_attempts == 1:
            raise RuntimeError("unpin failed")
        return manager._end_transfer(req, owner)

    manager.end_transfer.side_effect = release_owner
    executor._check_disagg_ctx_cache_transfer_status()
    assert id(request) in executor._pending_native_context_completions

    executor.canceled_req_ids = [request.py_request_id]
    executor.waiting_queue = Mock()
    executor._handle_canceled_requests()

    assert executor.canceled_req_ids == []
    assert executor._pending_request_cancellations == {}
    executor.kv_cache_transceiver.cancel_request.assert_not_called()
    assert manager.end_transfer.call_count == 1
    request.finish_by_reason.assert_not_called()

    executor._check_disagg_ctx_cache_transfer_status()

    assert executor._pending_native_context_completions == {}
    assert manager.end_transfer.call_args_list == [
        call(request, owner=_PYTHON_NATIVE_TRANSCEIVER_OWNER),
        call(request, owner=_PYTHON_NATIVE_TRANSCEIVER_OWNER),
    ]
    assert len(executor._pending_transfer_responses) == 1
    executor._terminate_request.assert_called_once_with(request)


def test_native_status_hands_transport_drain_to_earlier_cancellation():
    request = _make_cancel_request(
        7,
        LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS,
        generation_only=False,
    )
    request.py_kv_transfer_timed_out = False
    executor = _make_cancel_executor([request])
    executor.async_transfer_manager = _ExactTransferManager(request, transceiver=True)
    executor.kv_cache_transceiver.cancel_request.return_value = False
    executor.kv_cache_transceiver.check_context_transfer_status.return_value = (
        [request.py_request_id],
        [],
    )
    executor._deferred_transfer_terminations = {}
    executor._terminated_transfer_requests = {}
    executor.force_terminate_ctx_for_partial_reuse = False
    executor._check_cache_transfer_errors = Mock()

    executor._handle_canceled_requests()
    progress = executor._pending_request_cancellations[id(request)]
    assert not progress.transport_drained

    executor._check_disagg_ctx_cache_transfer_status()

    assert progress.transport_drained
    assert executor._pending_native_context_completions == {}
    executor.kv_cache_transceiver.cancel_request.assert_called_once_with(request)

    executor._handle_canceled_requests()

    executor.kv_cache_transceiver.cancel_request.assert_called_once_with(request)
    executor.async_transfer_manager.end_transfer.assert_called_once_with(
        request, owner=_PYTHON_NATIVE_TRANSCEIVER_OWNER
    )
    request.finish_by_reason.assert_called_once()
    assert executor._pending_request_cancellations == {}
    assert executor.canceled_req_ids == []


def test_error_does_not_override_buffered_native_success():
    request = _make_native_completion_request()
    request.py_client_id = None
    executor = _make_native_completion_executor(request)
    manager = executor.async_transfer_manager
    release_attempts = 0

    def release_owner(req, owner=None):
        nonlocal release_attempts
        release_attempts += 1
        if release_attempts == 1:
            raise RuntimeError("unpin failed")
        return manager._end_transfer(req, owner)

    manager.end_transfer.side_effect = release_owner
    executor._check_disagg_ctx_cache_transfer_status()
    response = request.create_response.return_value
    assert executor._pending_transfer_responses == [(7, response)]

    executor._error_budget = Mock(budget=1.0)
    executor._error_budget.consume.return_value = False
    executor._fatal_error = None
    executor.is_shutdown = False
    executor.waiting_queue = []
    executor.gather_all_responses = False
    executor._enqueue_responses = Mock()
    executor.executor_request_queue = Mock()
    executor.executor_request_queue.get_request_queue.return_value = Queue()

    executor._handle_errors(
        "unrelated executor error",
        requests=[request],
        charge_budget=False,
    )

    assert executor.active_requests == [request]
    assert executor._pending_transfer_responses == [(7, response)]
    executor._enqueue_responses.assert_called_once_with([])
    executor.kv_cache_transceiver.cancel_request.assert_not_called()

    executor._check_disagg_ctx_cache_transfer_status()

    request.create_response.assert_called_once_with(False, 1)
    assert executor._pending_native_context_completions == {}
    assert executor.active_requests == []
    executor._terminate_request.assert_called_once_with(request)


def test_native_failure_emits_one_error_while_completion_ledger_retires():
    request = _make_native_completion_request()
    request.py_client_id = None
    request.is_context_only_request = True
    request.state = LlmRequestState.DISAGG_TRANS_ERROR
    executor = _make_native_completion_executor(request)
    executor.kv_cache_transceiver.check_context_transfer_status.side_effect = [
        ([], [request.py_request_id]),
        ([], []),
    ]
    executor.kv_cache_transceiver.cancel_request.return_value = True
    manager = executor.async_transfer_manager
    release_attempts = 0

    def release_owner(req, owner=None):
        nonlocal release_attempts
        release_attempts += 1
        if release_attempts == 1:
            raise RuntimeError("unpin failed")
        return manager._end_transfer(req, owner)

    manager.end_transfer.side_effect = release_owner
    executor._error_budget = Mock(budget=1.0)
    executor._error_budget.consume.return_value = False
    executor._fatal_error = None
    executor.is_shutdown = False
    executor.waiting_queue = []
    executor.gather_all_responses = False
    executor._enqueue_responses = Mock()
    executor.executor_request_queue = Mock()
    executor.executor_request_queue.get_request_queue.return_value = Queue()
    del executor._check_cache_transfer_errors

    executor._check_disagg_ctx_cache_transfer_status()
    assert id(request) in executor._pending_native_context_completions

    executor._handle_errors(
        "error observed while native cleanup is pending",
        requests=[request],
        charge_budget=False,
    )

    assert executor.active_requests == []
    executor._terminate_request.assert_not_called()
    assert executor._enqueue_responses.call_count == 1
    assert executor._enqueue_responses.call_args.args[0][0][0] == 7

    executor._check_disagg_ctx_cache_transfer_status()

    assert executor._pending_native_context_completions == {}
    assert executor.active_requests == []
    executor._terminate_request.assert_called_once_with(request)
    nonempty_response_calls = [
        args[0] for args, _kwargs in executor._enqueue_responses.call_args_list if args[0]
    ]
    assert len(nonempty_response_calls) == 1
    assert nonempty_response_calls[0][0][0] == request.py_request_id
    assert nonempty_response_calls[0][0][1].error_msg == (
        "error observed while native cleanup is pending"
    )
    request.create_response.assert_not_called()


def test_native_completion_termination_retry_does_not_release_twice():
    request = _make_native_completion_request()
    executor = _make_native_completion_executor(request)
    executor.resource_manager = Mock()
    executor._prefetched_request_ids = Mock()
    executor._prefetched_request_ids.discard.side_effect = [
        RuntimeError("post-release bookkeeping failed"),
        None,
    ]
    executor.gather_all_responses = False
    executor._request_resource_termination_progress = {}
    executor._terminate_request = executor._do_terminate_request

    executor._check_disagg_ctx_cache_transfer_status()

    progress = executor._pending_native_context_completions[id(request)]
    assert progress.owner_released
    executor.async_transfer_manager.end_transfer.assert_called_once_with(
        request, owner=_PYTHON_NATIVE_TRANSCEIVER_OWNER
    )
    executor.resource_manager.free_resources.assert_called_once_with(request)

    executor._check_disagg_ctx_cache_transfer_status()

    assert executor._pending_native_context_completions == {}
    executor.async_transfer_manager.end_transfer.assert_called_once_with(
        request, owner=_PYTHON_NATIVE_TRANSCEIVER_OWNER
    )
    executor.resource_manager.free_resources.assert_called_once_with(request)
    assert executor._request_resource_termination_progress == {}
    request.create_response.assert_called_once()


def test_native_launch_failure_before_session_retires_exact_owner():
    request = _make_native_launch_request()
    executor = _make_native_launch_executor(request)
    executor.kv_cache_transceiver.respond_and_send_async.side_effect = RuntimeError(
        "launch failed before session"
    )
    executor.kv_cache_transceiver.cancel_request.return_value = True

    with pytest.raises(RuntimeError, match="before session"):
        executor._send_kv_async([request])

    executor.async_transfer_manager.start_transfer.assert_called_once_with(
        request, owner=_PYTHON_NATIVE_TRANSCEIVER_OWNER
    )
    executor.kv_cache_transceiver.cancel_request.assert_called_once_with(request)
    executor.async_transfer_manager.end_transfer.assert_called_once_with(
        request, owner=_PYTHON_NATIVE_TRANSCEIVER_OWNER
    )
    executor._terminate_request.assert_called_once_with(request)
    assert executor._deferred_transfer_terminations == {}
    assert request.state == LlmRequestState.DISAGG_TRANS_ERROR


def test_native_launch_failure_after_drained_session_retires_exact_owner():
    request = _make_native_launch_request()
    executor = _make_native_launch_executor(request)
    session_created = Mock()

    def launch_then_fail(req):
        session_created(req)
        raise RuntimeError("launch failed after session")

    executor.kv_cache_transceiver.respond_and_send_async.side_effect = launch_then_fail
    executor.kv_cache_transceiver.cancel_request.return_value = True

    with pytest.raises(RuntimeError, match="after session"):
        executor._send_kv_async([request])

    session_created.assert_called_once_with(request)
    executor.kv_cache_transceiver.cancel_request.assert_called_once_with(request)
    executor.async_transfer_manager.end_transfer.assert_called_once_with(
        request, owner=_PYTHON_NATIVE_TRANSCEIVER_OWNER
    )
    executor._terminate_request.assert_called_once_with(request)
    assert executor._deferred_transfer_terminations == {}


def test_native_launch_failure_with_ambiguous_tasks_retains_exact_request():
    request = _make_native_launch_request()
    executor = _make_native_launch_executor(request)
    executor.kv_cache_transceiver.respond_and_send_async.side_effect = RuntimeError(
        "launch outcome ambiguous"
    )
    executor.kv_cache_transceiver.cancel_request.return_value = False

    with pytest.raises(RuntimeError, match="ambiguous"):
        executor._send_kv_async([request])

    assert executor._deferred_transfer_terminations == {id(request): request}
    executor.async_transfer_manager.end_transfer.assert_not_called()
    executor._terminate_request.assert_not_called()
    assert request.state == LlmRequestState.DISAGG_TRANS_ERROR


def _make_cancel_request(request_id, state, *, generation_only):
    request = Mock(
        py_request_id=request_id,
        parent_request_id=None,
        is_child=False,
        state=state,
        py_decoding_iter=3,
        is_finished=False,
        is_finished_due_to_cancellation=False,
    )
    request.is_generation_only_request.return_value = generation_only

    def finish_by_reason(_reason):
        request.is_finished = True
        request.is_finished_due_to_cancellation = True
        request.state = LlmRequestState.GENERATION_COMPLETE

    request.finish_by_reason.side_effect = finish_by_reason
    return request


def _make_cancel_executor(requests):
    executor = _initialize_bare_executor_lifecycle_state(object.__new__(PyExecutor))
    executor.canceled_req_ids = [request.py_request_id for request in requests]
    executor.waiting_queue = Mock()
    executor.active_requests = list(requests)
    executor.kv_cache_transceiver = Mock(requires_physical_drain_before_request_release=True)
    executor.kv_cache_transceiver.cancel_request.return_value = True
    executor.async_transfer_manager = Mock()
    executor.async_transfer_manager.has_pending_admission.return_value = False
    executor.async_transfer_manager.has_transfer_owner.return_value = False
    executor.async_transfer_manager.has_any_transfer_owner.return_value = False
    executor.async_transfer_manager.requests_in_transfer.return_value = {}
    executor._terminated_transfer_requests = {}
    return executor


def test_user_cancel_retires_native_context_owner_after_transport_drain():
    request = _make_cancel_request(
        7,
        LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS,
        generation_only=False,
    )
    executor = _make_cancel_executor([request])
    executor.async_transfer_manager.has_transfer_owner.return_value = True
    events = []

    def cancel_request(req):
        events.append(("transport", req))
        return True

    def end_transfer(req, *, owner):
        assert req.state == LlmRequestState.DISAGG_TRANS_ERROR
        events.append(("owner", req, owner))
        return True

    def finish_by_reason(reason):
        events.append(("finish", request, reason))
        request.is_finished = True
        request.is_finished_due_to_cancellation = True
        request.state = LlmRequestState.GENERATION_COMPLETE

    executor.kv_cache_transceiver.cancel_request.side_effect = cancel_request
    executor.async_transfer_manager.end_transfer.side_effect = end_transfer
    request.finish_by_reason.side_effect = finish_by_reason

    executor._handle_canceled_requests()

    assert [event[0] for event in events] == ["transport", "owner", "finish"]
    executor.async_transfer_manager.end_transfer.assert_called_once_with(
        request, owner=_PYTHON_NATIVE_TRANSCEIVER_OWNER
    )
    assert request.state == LlmRequestState.GENERATION_COMPLETE
    assert request.is_finished_due_to_cancellation
    assert executor.canceled_req_ids == []


def test_user_cancel_retries_native_owner_release_and_continues_siblings():
    retry_request = _make_cancel_request(
        7,
        LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS,
        generation_only=False,
    )
    sibling_request = _make_cancel_request(
        8,
        LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS,
        generation_only=True,
    )
    executor = _make_cancel_executor([retry_request, sibling_request])
    executor.async_transfer_manager.has_transfer_owner.side_effect = (
        lambda request, _owner: request is retry_request
    )
    executor.async_transfer_manager.end_transfer.side_effect = [
        RuntimeError("unpin failed"),
        True,
    ]

    executor._handle_canceled_requests()

    assert executor.canceled_req_ids == [7]
    assert retry_request.state == LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS
    retry_request.finish_by_reason.assert_not_called()
    sibling_request.finish_by_reason.assert_called_once()

    executor._handle_canceled_requests()

    assert executor.canceled_req_ids == []
    assert retry_request.state == LlmRequestState.GENERATION_COMPLETE
    retry_request.finish_by_reason.assert_called_once()
    assert executor.kv_cache_transceiver.cancel_request.call_args_list == [
        call(retry_request),
        call(sibling_request),
    ]
    assert executor.async_transfer_manager.end_transfer.call_args_list == [
        call(retry_request, owner=_PYTHON_NATIVE_TRANSCEIVER_OWNER),
        call(retry_request, owner=_PYTHON_NATIVE_TRANSCEIVER_OWNER),
    ]


def test_user_cancel_retries_transport_error_and_continues_siblings():
    retry_request = _make_cancel_request(
        7,
        LlmRequestState.DISAGG_TRANS_ERROR,
        generation_only=False,
    )
    sibling_request = _make_cancel_request(
        8,
        LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS,
        generation_only=True,
    )
    executor = _make_cancel_executor([retry_request, sibling_request])
    executor.async_transfer_manager.has_transfer_owner.side_effect = (
        lambda request, _owner: request is retry_request
    )
    executor.kv_cache_transceiver.cancel_request.side_effect = [
        RuntimeError("transport failed"),
        True,
        True,
    ]
    executor.async_transfer_manager.end_transfer.return_value = True

    executor._handle_canceled_requests()

    assert executor.canceled_req_ids == [7]
    assert retry_request.state == LlmRequestState.DISAGG_TRANS_ERROR
    retry_request.finish_by_reason.assert_not_called()
    sibling_request.finish_by_reason.assert_called_once()
    executor.async_transfer_manager.end_transfer.assert_not_called()

    executor._handle_canceled_requests()

    assert executor.canceled_req_ids == []
    retry_request.finish_by_reason.assert_called_once()
    executor.async_transfer_manager.end_transfer.assert_called_once_with(
        retry_request, owner=_PYTHON_NATIVE_TRANSCEIVER_OWNER
    )


def test_failed_cancellation_keeps_first_outcome_across_late_error():
    request = _make_cancel_request(
        7,
        LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS,
        generation_only=False,
    )
    request.py_client_id = None
    executor = _make_cancel_executor([request])
    executor.async_transfer_manager = _ExactTransferManager(request, transceiver=True)
    executor.kv_cache_transceiver.cancel_request.side_effect = [False, True]
    executor._deferred_transfer_terminations = {}
    executor._error_budget = Mock()
    executor._error_budget.consume.return_value = False
    executor._fatal_error = None
    executor.is_shutdown = False
    executor.gather_all_responses = False
    executor.dist = SimpleNamespace(rank=0, world_size=1)
    executor.enable_attention_dp = False
    executor._enqueue_responses = Mock()
    executor._terminate_request = Mock()
    executor.force_terminate_ctx_for_partial_reuse = False

    executor._handle_canceled_requests()

    progress = executor._pending_request_cancellations[id(request)]
    assert not progress.transport_drained
    executor._terminate_request.assert_not_called()

    executor._handle_errors(
        "request failed during cancellation",
        requests=[request],
        charge_budget=False,
    )

    assert executor.active_requests == [request]
    executor._terminate_request.assert_not_called()

    executor._handle_canceled_requests()

    assert executor._pending_request_cancellations == {}
    assert executor.async_transfer_manager.end_transfer.call_args_list == [
        call(request, owner=_PYTHON_NATIVE_TRANSCEIVER_OWNER)
    ]

    _configure_finished_response(executor, request)
    executor._handle_responses()

    executor._terminate_request.assert_called_once_with(request)


def test_native_owner_release_retry_survives_response_processing():
    request = _make_cancel_request(
        7,
        LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS,
        generation_only=False,
    )
    request.is_finished = True
    request.is_attention_dp_dummy = False
    executor = _make_cancel_executor([request])
    executor.async_transfer_manager.has_transfer_owner.return_value = True
    executor.async_transfer_manager.end_transfer.side_effect = [
        RuntimeError("unpin failed"),
        True,
    ]
    executor.perf_manager = Mock()
    executor.perf_manager.get_timestamp.return_value = 0
    executor.dist = SimpleNamespace(rank=0, world_size=1)
    executor.enable_attention_dp = False
    executor._enqueue_responses = Mock()
    executor._terminate_request = Mock()

    executor._handle_canceled_requests()

    assert executor.canceled_req_ids == [7]
    assert executor._pending_request_cancellations[id(request)].request is request

    # A normal response pass must neither emit the pre-cancellation success nor
    # drop the only active-list root before the owner release can retry.
    assert executor._handle_responses() == []
    assert executor.active_requests == [request]
    request.create_response.assert_not_called()
    executor._terminate_request.assert_not_called()

    executor._handle_canceled_requests()

    assert executor.canceled_req_ids == []
    assert executor._pending_request_cancellations == {}
    executor.kv_cache_transceiver.cancel_request.assert_called_once_with(request)
    assert executor.async_transfer_manager.end_transfer.call_count == 2
    request.finish_by_reason.assert_called_once()


def _make_connector_completion_executor(request):
    executor = _initialize_bare_executor_lifecycle_state(object.__new__(PyExecutor))
    executor.kv_connector_manager = Mock()
    executor.kv_cache_transceiver = Mock(requires_physical_drain_before_request_release=True)
    executor.active_requests = [request]
    executor.async_transfer_manager = Mock()
    executor.async_transfer_manager.has_pending_admission.return_value = False
    executor.async_transfer_manager.has_any_transfer_owner.return_value = False
    executor.async_transfer_manager.is_final_transfer_owner.return_value = True
    executor.async_transfer_manager.requests_in_transfer.return_value = {}
    executor.force_terminate_ctx_for_partial_reuse = False
    executor._terminate_request = Mock()
    executor.dist = SimpleNamespace(rank=0, world_size=1)
    executor._maybe_attach_ctx_usage = Mock()
    executor._pending_transfer_responses = []
    executor._pending_connector_completions = {}
    executor._terminated_transfer_requests = {}
    return executor


def _configure_finished_response(executor, request):
    request.is_attention_dp_dummy = False
    request.py_kv_transfer_timed_out = False
    request.py_draft_tokens = []
    request.py_num_accepted_draft_tokens = 0
    request.py_per_pos_drafted = []
    request.py_per_pos_accepted = []
    request.return_perf_metrics = False
    request.cached_tokens = 0
    request.is_disagg_context_complete_state = False
    request.is_disagg_context_transmission_state = False
    response = Mock(result=SimpleNamespace())
    request.create_response.return_value = response
    executor.perf_manager = Mock()
    executor.perf_manager.get_timestamp.return_value = 0
    executor.iter_counter = 1
    executor.stream_interval = 1
    executor.dist = SimpleNamespace(rank=0, world_size=1)
    executor.enable_attention_dp = False
    executor._maybe_attach_ctx_usage = Mock()
    executor._enqueue_responses = Mock()


def _configure_connector_poll(executor, request):
    executor.kv_connector_manager = _make_one_shot_connector_manager(request)
    executor._pending_connector_completions = {}
    executor._pending_transfer_responses = []
    executor._terminated_transfer_requests = {}
    executor.force_terminate_ctx_for_partial_reuse = False


def _make_one_shot_connector_manager(request, *, empty_polls=0):
    manager = Mock()
    poll_count = 0

    def get_finished():
        nonlocal poll_count
        poll_count += 1
        if poll_count == empty_polls + 1:
            return [request]
        return []

    manager.get_finished.side_effect = get_finished
    return manager


@pytest.mark.parametrize("first_provider", ["native", "connector"])
def test_mixed_provider_completion_publishes_and_terminates_once(first_provider):
    request = _make_native_completion_request()
    executor = _make_native_completion_executor(request)
    executor.async_transfer_manager = _ExactTransferManager(request, transceiver=True, anonymous=1)
    executor.kv_connector_manager = _make_one_shot_connector_manager(request)
    executor._enqueue_responses = Mock()
    executor._terminate_request.side_effect = executor._mark_transfer_terminal_teardown_complete

    if first_provider == "connector":
        executor._kv_connector_terminate_requests()
        # Connector success is not authoritative while native may still fail.
        assert executor._pending_transfer_terminals == {}
        request.create_response.assert_not_called()

    executor._check_disagg_ctx_cache_transfer_status()

    if first_provider == "native":
        executor._kv_connector_terminate_requests()

    response = request.create_response.return_value
    request.create_response.assert_called_once_with(False, 1)
    assert executor._pending_transfer_responses == [(7, response)]
    assert executor.async_transfer_manager.end_transfer.call_count == 2
    executor._terminate_request.assert_called_once_with(request)

    executor._flush_pending_transfer_responses()

    executor._enqueue_responses.assert_called_once_with([(7, response)])
    assert executor._pending_transfer_responses == []
    assert executor._pending_transfer_terminals == {}


def test_connector_first_does_not_mask_later_native_failure():
    request = _make_native_completion_request()
    request.py_client_id = None
    request.is_context_only_request = True
    executor = _make_native_completion_executor(request)
    executor.kv_cache_transceiver.check_context_transfer_status.side_effect = [
        ([], [request.py_request_id]),
        ([], []),
    ]
    executor.async_transfer_manager = _ExactTransferManager(request, transceiver=True, anonymous=1)
    executor.kv_connector_manager = _make_one_shot_connector_manager(request)
    executor._enqueue_responses = Mock()

    executor._kv_connector_terminate_requests()

    assert executor._pending_transfer_terminals == {}
    request.state = LlmRequestState.DISAGG_TRANS_ERROR
    executor._check_disagg_ctx_cache_transfer_status()

    terminal = executor._pending_transfer_terminals[id(request)]
    assert terminal.outcome.value == "failed"
    request.create_response.assert_not_called()

    executor._error_budget = Mock(budget=1.0)
    executor._error_budget.consume.return_value = False
    executor._fatal_error = None
    executor.is_shutdown = False
    executor.waiting_queue = []
    executor.gather_all_responses = False
    executor.executor_request_queue = Mock()
    executor.executor_request_queue.get_request_queue.return_value = Queue()
    executor._handle_errors("native transfer failed", requests=[request], charge_budget=False)

    responses = executor._enqueue_responses.call_args.args[0]
    assert len(responses) == 1
    assert responses[0][1].error_msg == "native transfer failed"
    executor._terminate_request.assert_called_once_with(request)
    assert executor.async_transfer_manager.end_transfer.call_count == 2


def test_connector_first_does_not_mask_later_cpp_failure():
    request = _make_native_completion_request()
    request.py_client_id = None
    request.is_context_only_request = True
    executor = _make_native_completion_executor(request)
    executor.kv_cache_transceiver.requires_physical_drain_before_request_release = False
    executor.kv_cache_transceiver.check_context_transfer_status.side_effect = [
        ([], [request.py_request_id]),
        ([], []),
    ]
    # The connector and legacy C++ transceiver are both anonymous owners.
    executor.async_transfer_manager = _ExactTransferManager(request, anonymous=2)
    executor.kv_connector_manager = _make_one_shot_connector_manager(request)
    executor._enqueue_responses = Mock()

    executor._kv_connector_terminate_requests()

    assert executor._pending_transfer_terminals == {}
    request.create_response.assert_not_called()
    request.state = LlmRequestState.DISAGG_TRANS_ERROR
    executor._check_disagg_ctx_cache_transfer_status()

    terminal = executor._pending_transfer_terminals[id(request)]
    assert terminal.outcome.value == "failed"
    request.create_response.assert_not_called()

    executor._error_budget = Mock(budget=1.0)
    executor._error_budget.consume.return_value = False
    executor._fatal_error = None
    executor.is_shutdown = False
    executor.waiting_queue = []
    executor.gather_all_responses = False
    executor.executor_request_queue = Mock()
    executor.executor_request_queue.get_request_queue.return_value = Queue()
    executor._handle_errors("C++ transfer failed", requests=[request], charge_budget=False)

    responses = executor._enqueue_responses.call_args.args[0]
    assert len(responses) == 1
    assert responses[0][1].error_msg == "C++ transfer failed"
    request.create_response.assert_not_called()
    executor._terminate_request.assert_called_once_with(request)
    assert executor.async_transfer_manager.end_transfer.call_count == 2


@pytest.mark.parametrize("late_event", ["error", "cancel"])
def test_native_success_wins_late_sibling_window_event(late_event):
    request = _make_native_completion_request()
    request.py_client_id = None
    request.is_child = False
    request.parent_request_id = None
    executor = _make_native_completion_executor(request)
    executor.async_transfer_manager = _ExactTransferManager(request, transceiver=True, anonymous=1)
    executor.kv_connector_manager = _make_one_shot_connector_manager(request)
    executor._enqueue_responses = Mock()

    executor._check_disagg_ctx_cache_transfer_status()
    terminal = executor._pending_transfer_terminals[id(request)]
    assert terminal.outcome.value == "succeeded"

    if late_event == "error":
        executor._error_budget = Mock(budget=1.0)
        executor._error_budget.consume.return_value = False
        executor._fatal_error = None
        executor.is_shutdown = False
        executor.waiting_queue = []
        executor.gather_all_responses = False
        executor.executor_request_queue = Mock()
        executor.executor_request_queue.get_request_queue.return_value = Queue()
        executor._handle_errors("late sibling error", requests=[request], charge_budget=False)
    else:
        executor.canceled_req_ids = [request.py_request_id]
        executor.waiting_queue = Mock()
        executor._handle_canceled_requests()
        request.finish_by_reason.assert_not_called()

    assert executor._pending_transfer_terminals[id(request)] is terminal
    assert executor.active_requests == [request]

    executor._kv_connector_terminate_requests()

    request.create_response.assert_called_once()
    executor._terminate_request.assert_called_once_with(request)
    assert executor.async_transfer_manager.end_transfer.call_count == 2


def test_connector_response_publication_failure_is_not_replayed():
    request = _make_native_completion_request()
    response = request.create_response.return_value
    executor = _make_connector_completion_executor(request)
    executor._terminate_request.side_effect = executor._mark_transfer_terminal_teardown_complete
    executor.async_transfer_manager = _ExactTransferManager(request, anonymous=1)
    executor.kv_connector_manager.get_finished.side_effect = [[request], []]
    executor._maybe_attach_ctx_usage.side_effect = [
        RuntimeError("usage unavailable"),
        None,
    ]

    executor._kv_connector_terminate_requests()

    terminal = executor._pending_transfer_terminals[id(request)]
    assert terminal.response is response
    assert not terminal.response_buffered
    executor.async_transfer_manager.end_transfer.assert_not_called()

    executor._kv_connector_terminate_requests()

    request.create_response.assert_called_once_with(False, 0)
    assert executor._pending_transfer_responses == [(7, response)]
    executor.async_transfer_manager.end_transfer.assert_called_once_with(request)
    executor._terminate_request.assert_called_once_with(request)
    assert executor._pending_transfer_terminals[id(request)] is terminal

    executor._enqueue_responses = Mock(side_effect=RuntimeError("enqueue outcome ambiguous"))
    with pytest.raises(RuntimeError, match="enqueue outcome ambiguous"):
        executor._flush_pending_transfer_responses()

    assert executor._pending_transfer_responses == [(7, response)]
    assert not terminal.response_published
    assert terminal.response_publication_in_doubt

    with pytest.raises(RuntimeError, match="refusing to replay"):
        executor._flush_pending_transfer_responses()

    executor._enqueue_responses.assert_called_once_with([(7, response)])
    executor._discard_pending_transfer_responses_after_shutdown()
    assert executor._pending_transfer_responses == []
    assert executor._pending_transfer_terminals == {}


def test_connector_response_creation_failure_is_not_replayed():
    request = _make_native_completion_request()
    executor = _make_connector_completion_executor(request)
    executor.async_transfer_manager = _ExactTransferManager(request, anonymous=1)
    executor.kv_connector_manager.get_finished.side_effect = [[request], []]
    serialized_side_effects = []

    def mutate_then_raise(*_args):
        serialized_side_effects.append("consumed")
        raise RuntimeError("response creation outcome ambiguous")

    request.create_response.side_effect = mutate_then_raise

    executor._kv_connector_terminate_requests()

    terminal = executor._pending_transfer_terminals[id(request)]
    assert terminal.response_creation_in_doubt
    assert serialized_side_effects == ["consumed"]
    executor.async_transfer_manager.end_transfer.assert_not_called()

    executor._kv_connector_terminate_requests()

    assert serialized_side_effects == ["consumed"]
    request.create_response.assert_called_once_with(False, 0)
    executor.async_transfer_manager.end_transfer.assert_not_called()
    assert executor._pending_connector_completions[id(request)].request is request


@pytest.mark.parametrize("transceiver_kind", ["connector_only", "capability_false"])
def test_connector_owned_cancel_waits_for_connector_completion(transceiver_kind):
    request = _make_cancel_request(
        7,
        LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS,
        generation_only=False,
    )
    executor = _make_cancel_executor([request])
    if transceiver_kind == "connector_only":
        executor.kv_cache_transceiver = None
    else:
        executor.kv_cache_transceiver = Mock(requires_physical_drain_before_request_release=False)
        executor.kv_cache_transceiver.cancel_request.return_value = True
    executor.async_transfer_manager = _ExactTransferManager(request, anonymous=1)
    executor._terminate_request = Mock()
    _configure_finished_response(executor, request)
    _configure_connector_poll(executor, request)

    executor._handle_canceled_requests()
    finished_requests = executor._handle_responses()

    assert finished_requests == []
    assert executor.active_requests == [request]
    request.create_response.assert_not_called()
    executor._terminate_request.assert_not_called()
    executor._enqueue_responses.assert_called_once_with([])
    executor._enqueue_responses.reset_mock()

    executor._kv_connector_terminate_requests()

    response = request.create_response.return_value
    assert executor._pending_transfer_responses == [(7, response)]
    executor._terminate_request.assert_called_once_with(request)
    executor.async_transfer_manager.end_transfer.assert_called_once_with(request)

    executor._flush_pending_transfer_responses()

    executor._enqueue_responses.assert_called_once_with([(7, response)])
    request.create_response.assert_called_once_with(False, 0)


def test_capability_false_cancel_keeps_anonymous_status_owner_after_connector():
    request = _make_cancel_request(
        7,
        LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS,
        generation_only=False,
    )
    executor = _make_cancel_executor([request])
    executor.kv_cache_transceiver.requires_physical_drain_before_request_release = False
    executor.kv_cache_transceiver.cancel_request.return_value = True
    # Default C++ transceiver and connector retain the legacy anonymous count.
    executor.async_transfer_manager = _ExactTransferManager(request, anonymous=2)
    executor._terminate_request = Mock()
    _configure_finished_response(executor, request)
    _configure_connector_poll(executor, request)

    executor._handle_canceled_requests()
    finished_requests = executor._handle_responses()

    assert finished_requests == []
    assert executor.active_requests == [request]
    executor.async_transfer_manager.end_transfer.assert_not_called()
    executor._terminate_request.assert_not_called()
    executor._enqueue_responses.assert_called_once_with([])
    executor._enqueue_responses.reset_mock()

    executor._kv_connector_terminate_requests()

    response = request.create_response.return_value
    assert executor._pending_transfer_responses == [(7, response)]
    assert executor.active_requests == [request]
    executor.async_transfer_manager.end_transfer.assert_called_once_with(request)
    executor._terminate_request.assert_not_called()

    executor._flush_pending_transfer_responses()

    executor._enqueue_responses.assert_called_once_with([(7, response)])

    executor._end_transfer_and_maybe_terminate(request)

    assert executor.async_transfer_manager.end_transfer.call_args_list == [
        call(request),
        call(request),
    ]
    executor._terminate_request.assert_called_once_with(request)


@pytest.mark.parametrize("transceiver_kind", ["connector_only", "capability_false"])
def test_connector_owned_error_waits_for_connector_completion(transceiver_kind):
    request = Mock(
        py_request_id=7,
        py_client_id=None,
        state=LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS,
    )
    request.is_generation_only_request.return_value = False
    executor = _make_error_executor(request, fatal=False)
    if transceiver_kind == "connector_only":
        executor.kv_cache_transceiver = None
    else:
        executor.kv_cache_transceiver.requires_physical_drain_before_request_release = False
    executor.async_transfer_manager = _ExactTransferManager(request, anonymous=1)
    _configure_connector_poll(executor, request)

    executor._handle_errors(
        "connector-owned request failed",
        requests=[request],
        charge_budget=False,
    )

    assert executor.active_requests == []
    executor._terminate_request.assert_not_called()

    executor._kv_connector_terminate_requests()

    executor._terminate_request.assert_called_once_with(request)
    executor.async_transfer_manager.end_transfer.assert_called_once_with(request)


def test_connector_unpin_failure_retries_without_replaying_response():
    request = _make_cancel_request(
        7,
        LlmRequestState.GENERATION_COMPLETE,
        generation_only=False,
    )
    request.is_finished = True
    request.is_finished_due_to_cancellation = True
    request.cached_tokens = 0
    request.create_response.return_value = Mock(result=SimpleNamespace())
    executor = _make_connector_completion_executor(request)
    executor.kv_connector_manager.get_finished.side_effect = [[request], []]
    executor.async_transfer_manager.end_transfer.side_effect = [
        RuntimeError("unpin failed"),
        True,
    ]

    executor._kv_connector_terminate_requests()

    assert executor._pending_connector_completions[id(request)].request is request
    assert request.state == LlmRequestState.GENERATION_COMPLETE
    request.create_response.assert_called_once()
    executor._terminate_request.assert_not_called()

    executor._kv_connector_terminate_requests()

    assert executor._pending_connector_completions == {}
    assert executor.active_requests == []
    request.create_response.assert_called_once()
    assert executor.async_transfer_manager.end_transfer.call_count == 2
    executor._terminate_request.assert_called_once_with(request)
    assert request.state == LlmRequestState.GENERATION_COMPLETE
    assert executor.kv_connector_manager.get_finished.call_args_list == [
        call(),
        call(),
    ]


def test_post_unpin_termination_retry_does_not_release_resources_twice():
    request = _make_cancel_request(
        7,
        LlmRequestState.GENERATION_COMPLETE,
        generation_only=False,
    )
    request.is_finished = True
    request.is_finished_due_to_cancellation = True
    request.cached_tokens = 0
    request.create_response.return_value = Mock(result=SimpleNamespace())
    executor = _make_connector_completion_executor(request)
    executor.kv_connector_manager.get_finished.side_effect = [[request], []]
    executor.async_transfer_manager.end_transfer.return_value = True
    executor.async_transfer_manager.has_transfer_owner.return_value = False
    executor.resource_manager = Mock()
    executor._prefetched_request_ids = Mock()
    executor._prefetched_request_ids.discard.side_effect = [
        RuntimeError("post-release bookkeeping failed"),
        None,
    ]
    executor.gather_all_responses = False
    executor.dist = SimpleNamespace(rank=1, world_size=1)
    executor._request_resource_termination_progress = {}
    executor._terminate_request = executor._do_terminate_request

    executor._kv_connector_terminate_requests()

    assert executor._pending_connector_completions[id(request)].owner_released
    executor.async_transfer_manager.end_transfer.assert_called_once_with(request)
    executor.resource_manager.free_resources.assert_called_once_with(request)

    executor._kv_connector_terminate_requests()

    assert executor._pending_connector_completions == {}
    executor.async_transfer_manager.end_transfer.assert_called_once_with(request)
    executor.resource_manager.free_resources.assert_called_once_with(request)
    assert executor._request_resource_termination_progress == {}
    request.create_response.assert_called_once()


@pytest.mark.parametrize("force_partial_reuse", [False, True])
def test_cancelled_context_connector_owner_terminates_exactly_once(force_partial_reuse):
    request = _make_cancel_request(
        7,
        LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS,
        generation_only=False,
    )
    executor = _make_cancel_executor([request])
    executor.async_transfer_manager = _ExactTransferManager(request, transceiver=True, anonymous=1)
    executor.force_terminate_ctx_for_partial_reuse = force_partial_reuse
    executor._terminate_request = Mock(side_effect=executor._record_terminated_transfer_request)

    executor._handle_canceled_requests()

    request.is_attention_dp_dummy = False
    request.py_kv_transfer_timed_out = False
    request.py_draft_tokens = []
    request.py_num_accepted_draft_tokens = 0
    request.py_per_pos_drafted = []
    request.py_per_pos_accepted = []
    request.return_perf_metrics = False
    request.cached_tokens = 0
    request.is_disagg_context_complete_state = False
    request.is_disagg_context_transmission_state = False
    response = Mock(result=SimpleNamespace())
    request.create_response.return_value = response
    executor.perf_manager = Mock()
    executor.perf_manager.get_timestamp.return_value = 0
    executor.iter_counter = 1
    executor.stream_interval = 1
    executor.dist = SimpleNamespace(rank=0, world_size=1)
    executor.enable_attention_dp = False
    executor._maybe_attach_ctx_usage = Mock()
    executor._enqueue_responses = Mock()

    finished_requests = executor._handle_responses()
    assert executor.active_requests == []
    assert finished_requests == [request]
    request.create_response.assert_called_once_with(False, 0)
    executor._enqueue_responses.assert_called_once_with([(7, response)])
    if force_partial_reuse:
        executor._terminate_request.assert_called_once_with(request)
    else:
        executor._terminate_request.assert_not_called()

    executor._end_transfer_and_maybe_terminate(request)

    executor._terminate_request.assert_called_once_with(request)
    assert request.state == LlmRequestState.GENERATION_COMPLETE
    assert request.is_finished_due_to_cancellation
    assert executor.async_transfer_manager.end_transfer.call_args_list == [
        call(request, owner=_PYTHON_NATIVE_TRANSCEIVER_OWNER),
        call(request),
    ]
    assert executor._terminated_transfer_requests == {}


@pytest.mark.parametrize(
    ("state", "generation_only"),
    [
        (LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS, True),
        (LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS, False),
    ],
)
def test_user_cancel_without_native_context_owner_is_unchanged(state, generation_only):
    request = _make_cancel_request(
        7,
        state,
        generation_only=generation_only,
    )
    executor = _make_cancel_executor([request])

    executor._handle_canceled_requests()

    executor.kv_cache_transceiver.cancel_request.assert_called_once_with(request)
    executor.async_transfer_manager.end_transfer.assert_not_called()
    request.finish_by_reason.assert_called_once()
    assert request.state == LlmRequestState.GENERATION_COMPLETE
    assert executor.canceled_req_ids == []


def test_successful_request_termination_records_live_transfer_owner():
    request = Mock(py_request_id=7)
    executor = _initialize_bare_executor_lifecycle_state(object.__new__(PyExecutor))
    executor.resource_manager = Mock()
    executor._prefetched_request_ids = {7}
    executor.gather_all_responses = False
    executor.dist = SimpleNamespace(rank=1, world_size=1)
    executor.kv_cache_transceiver = SimpleNamespace(
        requires_physical_drain_before_request_release=True
    )
    executor.async_transfer_manager = Mock()
    executor.async_transfer_manager.has_pending_admission.return_value = False
    executor.async_transfer_manager.has_any_transfer_owner.return_value = True
    executor.async_transfer_manager.requests_in_transfer.return_value = {7: request}
    executor._terminated_transfer_requests = {}

    executor._do_terminate_request(request)

    executor.resource_manager.free_resources.assert_called_once_with(request)
    assert executor._prefetched_request_ids == set()
    assert executor._terminated_transfer_requests == {id(request): request}


def test_failed_request_termination_is_not_recorded_as_complete():
    request = Mock(py_request_id=7)
    executor = _initialize_bare_executor_lifecycle_state(object.__new__(PyExecutor))
    executor.resource_manager = Mock()
    executor.resource_manager.free_resources.side_effect = RuntimeError("free failed")
    executor._prefetched_request_ids = {7}
    executor.gather_all_responses = False
    executor.dist = SimpleNamespace(rank=1, world_size=1)
    executor.kv_cache_transceiver = SimpleNamespace(
        requires_physical_drain_before_request_release=True
    )
    executor.async_transfer_manager = Mock()
    executor.async_transfer_manager.has_pending_admission.return_value = False
    executor.async_transfer_manager.has_any_transfer_owner.return_value = True
    executor.async_transfer_manager.requests_in_transfer.return_value = {7: request}
    executor._terminated_transfer_requests = {}

    with pytest.raises(RuntimeError, match="free failed"):
        executor._do_terminate_request(request)

    assert executor._prefetched_request_ids == {7}
    assert executor._terminated_transfer_requests == {}


def test_request_termination_retry_does_not_release_resources_twice():
    request = Mock(py_request_id=7)
    executor = _initialize_bare_executor_lifecycle_state(object.__new__(PyExecutor))
    executor.resource_manager = Mock()
    executor._prefetched_request_ids = Mock()
    executor._prefetched_request_ids.discard.side_effect = [
        RuntimeError("post-release bookkeeping failed"),
        None,
    ]
    executor.gather_all_responses = False
    executor.dist = SimpleNamespace(rank=1, world_size=1)
    executor.kv_cache_transceiver = None
    executor._request_resource_termination_progress = {}

    with pytest.raises(RuntimeError, match="bookkeeping failed"):
        executor._do_terminate_request(request)

    assert executor._request_resource_termination_progress == {id(request): request}

    executor._do_terminate_request(request)

    executor.resource_manager.free_resources.assert_called_once_with(request)
    assert executor._request_resource_termination_progress == {}


def test_request_termination_rejects_pending_transfer_admission():
    request = Mock(py_request_id=7)
    executor = _initialize_bare_executor_lifecycle_state(object.__new__(PyExecutor))
    executor.async_transfer_manager = Mock()
    executor.async_transfer_manager.has_pending_admission.return_value = True
    executor.resource_manager = Mock()

    with pytest.raises(RuntimeError, match="transfer admission is incomplete"):
        executor._do_terminate_request(request)

    executor.resource_manager.free_resources.assert_not_called()


def _make_shutdown_executor(events, request):
    executor = _initialize_bare_executor_lifecycle_state(object.__new__(PyExecutor))
    executor.executor_request_queue = Mock()
    executor.shutdown_event = Mock()
    executor.hang_detector = Mock()
    executor.hang_detector.detected.return_value = False
    executor.worker_thread = Mock()
    executor.dist = SimpleNamespace(pp_size=1, rank=1, world_size=1)
    executor._shutdown_sleep_wakeup_listeners = Mock()
    executor.worker_started = True
    executor.model_engine = SimpleNamespace()
    executor.draft_model_engine = None
    executor.kv_cache_transceiver = SimpleNamespace(
        shutdown=events.transceiver_shutdown,
        cancel_request=events.cancel_request,
        requires_physical_drain_before_request_release=True,
    )
    manager = SimpleNamespace(shutdown=events.manager_shutdown)
    executor.resource_manager = SimpleNamespace(resource_managers={"test": manager})
    executor._deferred_transfer_terminations = {id(request): request}
    executor._terminated_transfer_requests = {}
    executor._pending_transfer_responses = []
    executor._terminate_request = events.terminate_request
    executor.virtual_memory_pools = None
    executor.sampler = object()
    executor.dwdp_manager = None
    return executor


def test_shutdown_discards_post_loop_native_response_without_collective(monkeypatch):
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.py_executor.torch.cuda.is_available",
        lambda: False,
    )
    events = Mock()
    request = _make_native_completion_request()
    request.is_dummy_request = False
    executor = _make_shutdown_executor(events, request)
    events.terminate_request.side_effect = executor._mark_transfer_terminal_teardown_complete
    executor.active_requests = [request]
    executor._deferred_transfer_terminations = {}
    executor.async_transfer_manager = _ExactTransferManager(request, transceiver=True)
    executor.force_terminate_ctx_for_partial_reuse = False
    executor._maybe_attach_ctx_usage = Mock()
    executor._enqueue_responses = Mock()
    executor._ingest_native_context_completions(
        [request.py_request_id], [], executor.async_transfer_manager.requests_in_transfer()
    )
    events.transceiver_shutdown.return_value = True

    executor.shutdown()

    request.create_response.assert_called_once_with(False, 1)
    executor._enqueue_responses.assert_not_called()
    assert executor._pending_transfer_responses == []
    assert executor._pending_native_context_completions == {}
    events.terminate_request.assert_called_once_with(request)
    events.manager_shutdown.assert_called_once_with()


def test_capability_false_shutdown_does_not_call_legacy_transceiver(monkeypatch):
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.py_executor.torch.cuda.is_available",
        lambda: False,
    )
    events = Mock()
    request = Mock(py_request_id=7, is_dummy_request=False)
    request.is_generation_only_request.return_value = False
    executor = _make_shutdown_executor(events, request)
    executor.kv_cache_transceiver.requires_physical_drain_before_request_release = False
    executor.dist = SimpleNamespace(pp_size=1, rank=0, world_size=2, allreduce=Mock())
    executor._deferred_transfer_terminations = {}
    executor.async_transfer_manager = _ExactTransferManager(request)

    executor.shutdown()

    events.transceiver_shutdown.assert_not_called()
    events.cancel_request.assert_not_called()
    executor.dist.allreduce.assert_not_called()
    events.manager_shutdown.assert_called_once_with()


def test_capability_false_shutdown_preserves_legacy_owner_agnostic_teardown(monkeypatch):
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.py_executor.torch.cuda.is_available",
        lambda: False,
    )
    events = Mock()
    request = Mock(py_request_id=7, is_dummy_request=False)
    request.is_generation_only_request.return_value = False
    executor = _make_shutdown_executor(events, request)
    executor._deferred_transfer_terminations = {}
    executor.kv_cache_transceiver.requires_physical_drain_before_request_release = False
    executor.async_transfer_manager = _ExactTransferManager(request, anonymous=1)
    executor.dist = SimpleNamespace(pp_size=1, rank=0, world_size=2, allreduce=Mock())

    executor.shutdown()

    events.transceiver_shutdown.assert_not_called()
    executor.async_transfer_manager.begin_shutdown.assert_not_called()
    executor.async_transfer_manager.end_transfer.assert_not_called()
    events.terminate_request.assert_not_called()
    executor.dist.allreduce.assert_not_called()
    events.manager_shutdown.assert_called_once_with()


def test_shutdown_waits_for_final_connector_owner_after_native_cancel(monkeypatch):
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.py_executor.torch.cuda.is_available",
        lambda: False,
    )
    monkeypatch.setattr("tensorrt_llm._torch.pyexecutor.py_executor.time.sleep", lambda _: None)
    events = Mock()
    request = _make_cancel_request(
        7,
        LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS,
        generation_only=False,
    )
    request.is_dummy_request = False
    request.cached_tokens = 0
    request.create_response.return_value = Mock(result=SimpleNamespace())
    executor = _make_shutdown_executor(events, request)
    events.terminate_request.side_effect = executor._mark_transfer_terminal_teardown_complete
    executor.dist = SimpleNamespace(pp_size=1, rank=0, world_size=1)
    executor._deferred_transfer_terminations = {}
    executor.active_requests = [request]
    executor.canceled_req_ids = [request.py_request_id]
    executor.waiting_queue = Mock()
    executor.async_transfer_manager = _ExactTransferManager(
        request, transceiver=True, anonymous=1, events=events
    )
    executor.kv_connector_manager = _make_one_shot_connector_manager(request)
    executor._pending_connector_completions = {}
    executor._pending_transfer_responses = []
    executor._maybe_attach_ctx_usage = Mock()
    executor.force_terminate_ctx_for_partial_reuse = False

    events.cancel_request.return_value = False
    executor._handle_canceled_requests()
    assert executor._pending_request_cancellations[id(request)].request is request

    events.reset_mock()
    events.transceiver_shutdown.return_value = True
    events.cancel_request.return_value = True

    executor.shutdown()

    assert events.mock_calls == [
        call.transceiver_shutdown(),
        call.cancel_request(request),
        call.transceiver_owner_release(request),
        call.connector_owner_release(request),
        call.terminate_request(request),
        call.manager_shutdown(),
    ]
    assert executor._pending_request_cancellations == {}
    assert executor._pending_connector_completions == {}
    assert executor._terminated_transfer_requests == {}


def test_shutdown_releases_deferred_request_only_after_transceiver_drain(monkeypatch):
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.py_executor.torch.cuda.is_available",
        lambda: False,
    )
    events = Mock()
    request = Mock(py_request_id=7)
    executor = _make_shutdown_executor(events, request)
    events.transceiver_shutdown.return_value = True
    events.cancel_request.return_value = True

    executor.shutdown()

    assert events.mock_calls == [
        call.transceiver_shutdown(),
        call.cancel_request(request),
        call.terminate_request(request),
        call.manager_shutdown(),
    ]
    assert executor._deferred_transfer_terminations == {}


def test_non_drained_shutdown_preserves_deferred_request_and_managers(monkeypatch):
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.py_executor.torch.cuda.is_available",
        lambda: False,
    )
    events = Mock()
    request = Mock(py_request_id=7)
    executor = _make_shutdown_executor(events, request)
    events.transceiver_shutdown.return_value = False

    with pytest.raises(RuntimeError, match="still owns active transfer targets"):
        executor.shutdown()

    assert events.mock_calls == [call.transceiver_shutdown()]
    assert executor._deferred_transfer_terminations == {id(request): request}


def test_in_doubt_resource_release_vetoes_manager_shutdown(monkeypatch):
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.py_executor.torch.cuda.is_available",
        lambda: False,
    )
    events = Mock()
    request = Mock(py_request_id=7)
    executor = _make_shutdown_executor(events, request)
    executor._deferred_transfer_terminations = {}
    events.transceiver_shutdown.return_value = True
    executor.resource_manager.has_in_doubt_resource_releases = Mock(return_value=True)

    with pytest.raises(RuntimeError, match="release outcome is in doubt"):
        executor.shutdown()

    events.manager_shutdown.assert_not_called()


def test_pending_resource_release_vetoes_manager_shutdown(monkeypatch):
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.py_executor.torch.cuda.is_available",
        lambda: False,
    )
    events = Mock()
    request = Mock(py_request_id=7)
    executor = _make_shutdown_executor(events, request)
    executor._deferred_transfer_terminations = {}
    events.transceiver_shutdown.return_value = True
    executor.resource_manager.has_pending_resource_releases = Mock(return_value=True)

    with pytest.raises(RuntimeError, match="release is still pending"):
        executor.shutdown()

    events.manager_shutdown.assert_not_called()


def test_shutdown_retires_native_owner_but_vetoes_connector_owner(monkeypatch):
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.py_executor.torch.cuda.is_available",
        lambda: False,
    )
    events = Mock()
    request = Mock(py_request_id=7)
    request.is_generation_only_request.return_value = False
    executor = _make_shutdown_executor(events, request)
    events.transceiver_shutdown.return_value = True
    events.cancel_request.return_value = True
    executor.async_transfer_manager = _ExactTransferManager(request, transceiver=True, anonymous=1)
    executor.force_terminate_ctx_for_partial_reuse = False

    with pytest.raises(RuntimeError, match="Asynchronous transfer ownership is still active"):
        executor.shutdown()

    executor.async_transfer_manager.end_transfer.assert_called_once_with(
        request, owner=_PYTHON_NATIVE_TRANSCEIVER_OWNER
    )
    executor.async_transfer_manager.has_any_inflight_requests.assert_called_once_with()
    assert executor._deferred_transfer_terminations == {}
    events.terminate_request.assert_not_called()
    events.manager_shutdown.assert_not_called()


def test_shutdown_drain_polls_connector_after_initial_empty_sample(monkeypatch):
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.py_executor.torch.cuda.is_available",
        lambda: False,
    )
    sleep = Mock()
    monkeypatch.setattr("tensorrt_llm._torch.pyexecutor.py_executor.time.sleep", sleep)
    events = Mock()
    request = SimpleNamespace(
        py_request_id=7,
        is_finished_due_to_cancellation=False,
        state=LlmRequestState.GENERATION_COMPLETE,
    )
    executor = _make_shutdown_executor(events, request)
    executor._deferred_transfer_terminations = {}
    executor.active_requests = []
    executor.force_terminate_ctx_for_partial_reuse = False
    events.transceiver_shutdown.return_value = True
    executor.kv_connector_manager = _make_one_shot_connector_manager(request, empty_polls=1)
    executor.async_transfer_manager = _ExactTransferManager(request, anonymous=1)

    executor.shutdown()

    assert (
        executor.kv_connector_manager.get_finished.call_count
        == _SHUTDOWN_CONNECTOR_COMPLETION_POLLS
    )
    sleep.assert_called_once_with(_SHUTDOWN_CONNECTOR_COMPLETION_POLL_INTERVAL_S)
    executor.async_transfer_manager.end_transfer.assert_called_once_with(request)
    executor.async_transfer_manager.has_any_inflight_requests.assert_called_with()
    assert executor.async_transfer_manager.has_any_inflight_requests.call_count > 1
    events.terminate_request.assert_called_once_with(request)
    events.manager_shutdown.assert_called_once_with()


def test_shutdown_connector_poll_error_finishes_uniform_drain_without_replay():
    request = SimpleNamespace(
        py_request_id=7,
        is_finished_due_to_cancellation=False,
        state=LlmRequestState.GENERATION_COMPLETE,
    )
    executor = _initialize_bare_executor_lifecycle_state(object.__new__(PyExecutor))
    executor.dist = SimpleNamespace(world_size=2, allreduce=Mock(return_value=1))
    executor.kv_cache_transceiver = None
    executor.active_requests = []
    executor.async_transfer_manager = _ExactTransferManager(request, anonymous=1)
    executor._terminate_request_after_worker_shutdown = Mock()
    executor._terminated_transfer_requests = {}
    executor.kv_connector_manager = Mock()
    executor.kv_connector_manager.get_finished.side_effect = [
        [request],
        RuntimeError("local connector poll failed"),
    ] + [[]] * (_SHUTDOWN_CONNECTOR_COMPLETION_POLLS - 2)

    with pytest.raises(RuntimeError, match="local connector poll failed"):
        executor._drain_connector_completions_after_shutdown()

    assert (
        executor.kv_connector_manager.get_finished.call_count
        == _SHUTDOWN_CONNECTOR_COMPLETION_POLLS
    )
    executor.async_transfer_manager.end_transfer.assert_called_once_with(request)
    executor._terminate_request_after_worker_shutdown.assert_called_once_with(request)
    executor.dist.allreduce.assert_called_once()
    assert executor.dist.allreduce.call_args.args[0] == 1


def test_shutdown_connector_peer_error_raises_before_later_collectives():
    executor = _initialize_bare_executor_lifecycle_state(object.__new__(PyExecutor))
    executor.dist = SimpleNamespace(world_size=2, allreduce=Mock(return_value=1))
    executor.kv_cache_transceiver = None
    executor.active_requests = []
    executor.async_transfer_manager = None
    executor.kv_connector_manager = Mock()
    executor.kv_connector_manager.get_finished.return_value = []

    with pytest.raises(RuntimeError, match="another rank"):
        executor._drain_connector_completions_after_shutdown()

    assert (
        executor.kv_connector_manager.get_finished.call_count
        == _SHUTDOWN_CONNECTOR_COMPLETION_POLLS
    )
    executor.dist.allreduce.assert_called_once()
    assert executor.dist.allreduce.call_args.args[0] == 0


@pytest.mark.parametrize("local_pending", [True, False])
def test_shutdown_final_readiness_vote_precedes_manager_teardown(monkeypatch, local_pending):
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.py_executor.torch.cuda.is_available",
        lambda: False,
    )
    events = Mock()
    request = Mock(py_request_id=7, is_dummy_request=False)
    executor = _make_shutdown_executor(events, request)
    executor._deferred_transfer_terminations = {}
    events.transceiver_shutdown.return_value = True
    executor.async_transfer_manager = _ExactTransferManager(request)
    executor.kv_connector_manager = Mock()
    executor.kv_connector_manager.get_finished.return_value = []
    executor.dist = SimpleNamespace(
        pp_size=1,
        rank=0,
        world_size=2,
        allreduce=Mock(side_effect=[0, 1]),
    )
    executor.resource_manager.has_in_doubt_resource_releases = Mock(return_value=False)
    executor.resource_manager.has_pending_resource_releases = Mock(return_value=local_pending)

    error = "release is still pending" if local_pending else "another rank"
    with pytest.raises(RuntimeError, match=error):
        executor.shutdown()

    assert [entry.args[0] for entry in executor.dist.allreduce.call_args_list] == [
        0,
        int(local_pending),
    ]
    events.manager_shutdown.assert_not_called()


@pytest.mark.parametrize("local_manager_failure", [True, False])
def test_shutdown_post_manager_vote_precedes_engine_deletion(monkeypatch, local_manager_failure):
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.py_executor.torch.cuda.is_available",
        lambda: False,
    )
    events = Mock()
    request = Mock(py_request_id=7, is_dummy_request=False)
    executor = _make_shutdown_executor(events, request)
    executor._deferred_transfer_terminations = {}
    events.transceiver_shutdown.return_value = True
    executor.async_transfer_manager = _ExactTransferManager(request)
    executor.kv_connector_manager = Mock()
    executor.kv_connector_manager.get_finished.return_value = []
    executor.dist = SimpleNamespace(
        pp_size=1,
        rank=0,
        world_size=2,
        allreduce=Mock(side_effect=[0, 0, 1]),
    )
    if local_manager_failure:
        events.manager_shutdown.side_effect = RuntimeError("local manager shutdown failed")

    error = "local manager shutdown failed" if local_manager_failure else "another rank"
    with pytest.raises(RuntimeError, match=error):
        executor.shutdown()

    assert [entry.args[0] for entry in executor.dist.allreduce.call_args_list] == [
        0,
        0,
        int(local_manager_failure),
    ]
    events.manager_shutdown.assert_called_once_with()
    assert hasattr(executor, "model_engine")
    progress = executor._resource_manager_shutdown_progress
    assert len(progress) == 1
    assert progress[0].in_doubt is local_manager_failure
    assert progress[0].completed is not local_manager_failure


def test_ambiguous_manager_shutdown_is_not_replayed():
    manager = Mock()
    manager.shutdown.side_effect = RuntimeError("manager shutdown failed")
    executor = object.__new__(PyExecutor)
    executor.dist = SimpleNamespace(rank=0, world_size=1)
    executor.resource_manager = SimpleNamespace(resource_managers={"test": manager})

    for _attempt in range(2):
        with pytest.raises(RuntimeError, match="manager shutdown"):
            executor._shutdown_resource_managers_rank_uniform()

    manager.shutdown.assert_called_once_with()


def test_ambiguous_manager_shutdown_blocks_later_dependencies() -> None:
    first_manager = Mock()
    first_manager.shutdown.side_effect = RuntimeError("manager shutdown failed")
    later_manager = Mock()
    executor = object.__new__(PyExecutor)
    executor.dist = SimpleNamespace(rank=0, world_size=1)
    executor.resource_manager = SimpleNamespace(
        resource_managers={
            "dependent": first_manager,
            "kv_cache": later_manager,
        }
    )

    for _attempt in range(2):
        with pytest.raises(RuntimeError, match="manager shutdown"):
            executor._shutdown_resource_managers_rank_uniform()

    first_manager.shutdown.assert_called_once_with()
    later_manager.shutdown.assert_not_called()


def test_shutdown_discards_ownerless_unbuffered_terminal_before_teardown(
    monkeypatch,
):
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.py_executor.torch.cuda.is_available",
        lambda: False,
    )
    events = Mock()
    request = SimpleNamespace(py_request_id=7, is_dummy_request=False)
    executor = _make_shutdown_executor(events, request)
    executor._deferred_transfer_terminations = {}
    events.transceiver_shutdown.return_value = True
    executor.async_transfer_manager = _ExactTransferManager(request)
    executor.active_requests = [request]
    executor._claim_transfer_terminal(
        request, _TransferTerminalOutcome.FAILED, "native_transceiver"
    )
    events.terminate_request.side_effect = (
        lambda owner: executor._mark_transfer_terminal_teardown_complete(owner)
    )

    executor.shutdown()

    events.terminate_request.assert_called_once_with(request)
    assert executor.active_requests == []
    assert executor._pending_transfer_terminals == {}
    events.manager_shutdown.assert_called_once_with()


def test_shutdown_discard_retires_already_torn_down_terminal() -> None:
    request = SimpleNamespace(py_request_id=7)
    executor = _initialize_bare_executor_lifecycle_state(object.__new__(PyExecutor))
    executor.async_transfer_manager = None
    terminal = executor._claim_transfer_terminal(
        request, _TransferTerminalOutcome.FAILED, "native_transceiver"
    )
    terminal.teardown_complete = True

    executor._discard_unpublishable_transfer_terminals_after_shutdown()

    assert terminal.response_discarded
    assert executor._pending_transfer_terminals == {}


def test_shutdown_native_failure_still_enters_connector_consensus(monkeypatch):
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.py_executor.torch.cuda.is_available",
        lambda: False,
    )
    events = Mock()
    request = Mock(py_request_id=7, is_dummy_request=False)
    executor = _make_shutdown_executor(events, request)
    executor._deferred_transfer_terminations = {}
    executor.async_transfer_manager = _ExactTransferManager(request)
    executor.kv_connector_manager = Mock()
    executor.kv_connector_manager.get_finished.return_value = []
    executor.dist = SimpleNamespace(
        pp_size=1,
        rank=0,
        world_size=2,
        allreduce=Mock(return_value=1),
    )
    events.transceiver_shutdown.return_value = False

    with pytest.raises(RuntimeError, match="still owns active transfer targets"):
        executor.shutdown()

    assert (
        executor.kv_connector_manager.get_finished.call_count
        == _SHUTDOWN_CONNECTOR_COMPLETION_POLLS
    )
    executor.dist.allreduce.assert_called_once()
    assert executor.dist.allreduce.call_args.args[0] == 1
    events.manager_shutdown.assert_not_called()


@pytest.mark.parametrize("failure_site", ["cuda_graph", "cuda_synchronize"])
def test_shutdown_local_cleanup_failure_still_enters_connector_consensus(monkeypatch, failure_site):
    events = Mock()
    request = Mock(py_request_id=7, is_dummy_request=False)
    executor = _make_shutdown_executor(events, request)
    executor._deferred_transfer_terminations = {}
    events.transceiver_shutdown.return_value = True
    executor.async_transfer_manager = _ExactTransferManager(request)
    executor.kv_connector_manager = Mock()
    executor.kv_connector_manager.get_finished.return_value = []
    executor.dist = SimpleNamespace(
        pp_size=1,
        rank=0,
        world_size=2,
        allreduce=Mock(return_value=1),
    )

    if failure_site == "cuda_graph":
        executor.model_engine = SimpleNamespace(
            _release_cuda_graphs=Mock(side_effect=RuntimeError("CUDA graph cleanup failed"))
        )
        monkeypatch.setattr(
            "tensorrt_llm._torch.pyexecutor.py_executor.torch.cuda.is_available",
            lambda: False,
        )
        expected_error = "CUDA graph cleanup failed"
    else:
        monkeypatch.setattr(
            "tensorrt_llm._torch.pyexecutor.py_executor.torch.cuda.is_available",
            lambda: True,
        )
        monkeypatch.setattr(
            "tensorrt_llm._torch.pyexecutor.py_executor.torch.cuda.synchronize",
            Mock(side_effect=RuntimeError("CUDA synchronize failed")),
        )
        expected_error = "CUDA synchronize failed"

    with pytest.raises(RuntimeError, match=expected_error):
        executor.shutdown()

    assert (
        executor.kv_connector_manager.get_finished.call_count
        == _SHUTDOWN_CONNECTOR_COMPLETION_POLLS
    )
    executor.dist.allreduce.assert_called_once()
    assert executor.dist.allreduce.call_args.args[0] == 1
    events.manager_shutdown.assert_not_called()


def test_shutdown_retries_and_retires_unpolled_native_owner(monkeypatch):
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.py_executor.torch.cuda.is_available",
        lambda: False,
    )
    events = Mock()
    request = Mock(py_request_id=7, is_dummy_request=False)
    request.is_generation_only_request.return_value = False
    executor = _make_shutdown_executor(events, request)
    executor._deferred_transfer_terminations = {}
    events.transceiver_shutdown.side_effect = [False, True]
    events.cancel_request.return_value = True
    executor.async_transfer_manager = _ExactTransferManager(request, transceiver=True)
    executor.force_terminate_ctx_for_partial_reuse = False

    with pytest.raises(RuntimeError, match="still owns active transfer targets"):
        executor.shutdown()

    assert executor._deferred_transfer_terminations == {id(request): request}
    executor.async_transfer_manager.end_transfer.assert_not_called()
    events.manager_shutdown.assert_not_called()

    executor.shutdown()

    executor.async_transfer_manager.end_transfer.assert_called_once_with(
        request, owner=_PYTHON_NATIVE_TRANSCEIVER_OWNER
    )
    events.terminate_request.assert_called_once_with(request)
    events.manager_shutdown.assert_called_once_with()
    assert executor._deferred_transfer_terminations == {}


def test_shutdown_does_not_reterminate_partial_reuse_request(monkeypatch):
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.py_executor.torch.cuda.is_available",
        lambda: False,
    )
    events = Mock()
    request = Mock(py_request_id=7, is_dummy_request=False)
    request.is_generation_only_request.return_value = False
    executor = _make_shutdown_executor(events, request)
    executor._deferred_transfer_terminations = {}
    executor.force_terminate_ctx_for_partial_reuse = True
    executor._terminated_transfer_requests = {id(request): request}
    events.transceiver_shutdown.return_value = True
    events.cancel_request.return_value = True
    executor.async_transfer_manager = _ExactTransferManager(request, transceiver=True)

    executor.shutdown()

    executor.async_transfer_manager.end_transfer.assert_called_once_with(
        request, owner=_PYTHON_NATIVE_TRANSCEIVER_OWNER
    )
    events.terminate_request.assert_not_called()
    events.manager_shutdown.assert_called_once_with()
    assert executor._deferred_transfer_terminations == {}
    assert executor._deferred_transfer_terminations_already_terminated == set()
    assert executor._terminated_transfer_requests == {}


def test_shutdown_finalizes_pp_request_after_executor_loop_stops(monkeypatch):
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.py_executor.torch.cuda.is_available",
        lambda: False,
    )
    events = Mock()
    request = Mock(py_request_id=7, is_dummy_request=False)
    request.is_generation_only_request.return_value = False
    executor = _make_shutdown_executor(events, request)
    executor._deferred_transfer_terminations = {}
    executor._terminate_request = Mock()
    events.transceiver_shutdown.return_value = True
    events.cancel_request.return_value = True
    executor.async_transfer_manager = _ExactTransferManager(request, transceiver=True)
    executor.force_terminate_ctx_for_partial_reuse = False
    termination_handler = object.__new__(DisaggPPTerminationHandler)
    termination_handler._terminator_func = events.terminate_request
    termination_handler._pending_termination = {7: request}
    executor._disagg_pp_termination_handler = termination_handler

    executor.shutdown()

    executor._terminate_request.assert_not_called()
    events.terminate_request.assert_called_once_with(request)
    assert termination_handler._pending_termination == {}
    events.manager_shutdown.assert_called_once_with()


def test_shutdown_finalizes_pending_only_pp_request(monkeypatch):
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.py_executor.torch.cuda.is_available",
        lambda: False,
    )
    events = Mock()
    request = Mock(py_request_id=7, is_dummy_request=False)
    executor = _make_shutdown_executor(events, request)
    executor.dist = SimpleNamespace(pp_size=2)
    executor.executed_batch_queue = Mock()
    executor.broadcast_sample_state_handler = Mock()
    executor._deferred_transfer_terminations = {}
    events.transceiver_shutdown.return_value = True
    termination_handler = DisaggPPTerminationHandler(executor.dist, events.terminate_request)
    termination_handler._pending_termination = {7: request}
    termination_handler._send_handle = Mock()
    executor._disagg_pp_termination_handler = termination_handler

    executor.shutdown()

    assert events.mock_calls == [
        call.transceiver_shutdown(),
        call.terminate_request(request),
        call.manager_shutdown(),
    ]
    assert termination_handler._pending_termination == {}
    termination_handler._send_handle.wait.assert_not_called()


def test_pending_pp_termination_failure_retains_exact_request():
    request = Mock(py_request_id=7)
    terminator = Mock(side_effect=[RuntimeError("release failed"), None])
    handler = DisaggPPTerminationHandler(SimpleNamespace(), terminator)
    handler._pending_termination = {7: request}

    with pytest.raises(RuntimeError, match="request 7: release failed"):
        handler.terminate_all_after_shutdown()

    assert handler._pending_termination == {7: request}
    handler.terminate_all_after_shutdown()
    assert handler._pending_termination == {}
    assert terminator.call_count == 2


def test_direct_post_shutdown_termination_failure_keeps_pending_owner():
    request = Mock(py_request_id=7)
    terminator = Mock(side_effect=RuntimeError("release failed"))
    handler = DisaggPPTerminationHandler(SimpleNamespace(), terminator)
    handler._pending_termination = {7: request}

    with pytest.raises(RuntimeError, match="release failed"):
        handler.terminate_after_shutdown(request)

    assert handler._pending_termination == {7: request}
