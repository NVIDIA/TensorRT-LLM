# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The PyExecutor adapters must honor the same contract the fake does.

Coordinator tests run against ``FakeExecutorEffects``; these tests pin that
the production adapter routes each effect to the executor primitive with the
same meaning, so a coordinator proven against the fake behaves the same on
the real executor.
"""

from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
from fake_executor_effects import FakeExecutorEffects, FakeRequestRegistry

from tensorrt_llm._torch.pyexecutor.disagg_adapter import (
    PyExecutorEffects,
    PyExecutorRequestRegistry,
)
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

pytestmark = pytest.mark.cpu_only


def _executor() -> PyExecutor:
    executor = object.__new__(PyExecutor)
    executor._pending_transfer_responses = []
    executor._pending_response_terminations = []
    executor._terminate_request = Mock()
    executor._handle_errors = Mock()
    executor._fatal_error = None
    executor.is_shutdown = False
    executor.active_requests = []
    executor.canceled_req_ids = []
    return executor


def test_terminate_request_reaches_executor_termination() -> None:
    executor = _executor()
    request = Mock()

    PyExecutorEffects(executor).terminate_request(request)

    executor._terminate_request.assert_called_once_with(request)


def test_staged_response_waits_for_the_synchronized_flush() -> None:
    """A staged response must not be enqueued directly (that would enter the
    ADP tp_gather from one rank); it lands in the buffer the loop flushes at
    the rank-synchronized point, and the request terminates only after."""
    executor = _executor()
    executor._enqueue_responses = Mock()
    request, response = Mock(), Mock()

    PyExecutorEffects(executor).stage_transfer_response(7, response, request)

    executor._enqueue_responses.assert_not_called()
    executor._terminate_request.assert_not_called()
    assert executor._pending_transfer_responses == [(7, response)]
    assert executor._pending_response_terminations == [request]


def test_staged_response_without_termination_only_publishes() -> None:
    executor = _executor()
    response = Mock()

    PyExecutorEffects(executor).stage_transfer_response(7, response, None)

    assert executor._pending_transfer_responses == [(7, response)]
    assert executor._pending_response_terminations == []


def test_fail_requests_uses_the_executor_error_path() -> None:
    executor = _executor()
    requests = [Mock()]

    PyExecutorEffects(executor).fail_requests("boom", requests, charge_budget=False)

    executor._handle_errors.assert_called_once_with(
        error_msg="boom", requests=requests, charge_budget=False
    )


def test_fail_fatal_marks_the_executor_fatal_before_the_aligned_error_path_runs() -> None:
    """The coordinator calls this only after a world-wide collective agreed,
    so the executor may enter the collective-aligned fatal path. The fatal
    state must already be set when ``_handle_errors`` runs: with
    ``charge_budget=False`` it reads ``_fatal_error`` to decide whether to do
    the fatal cleanup, so the reverse order would take the plain error path."""
    executor = _executor()
    state_on_entry = {}

    def record_state_on_entry(*_args, **_kwargs):
        state_on_entry["fatal_error"] = executor._fatal_error
        state_on_entry["is_shutdown"] = executor.is_shutdown

    executor._handle_errors.side_effect = record_state_on_entry

    PyExecutorEffects(executor).fail_fatal("poisoned")

    executor._handle_errors.assert_called_once_with(
        "poisoned", requests=None, charge_budget=False, fatal_is_collective_aligned=True
    )
    assert isinstance(state_on_entry["fatal_error"], RuntimeError)
    assert str(state_on_entry["fatal_error"]) == "Fatal error: poisoned"
    assert state_on_entry["is_shutdown"] is True


def test_registry_reads_the_executor_lists_live() -> None:
    """The executor rebinds active_requests; the registry must not cache."""
    executor = _executor()
    registry = PyExecutorRequestRegistry(executor)
    first, second = Mock(), Mock()
    executor.active_requests = [first]
    assert list(registry.active_requests()) == [first]
    assert registry.contains(first)

    executor.active_requests = [second]
    executor.canceled_req_ids = [9]

    assert list(registry.active_requests()) == [second]
    assert not registry.contains(first)
    assert list(registry.canceled_request_ids()) == [9]


def test_registry_remove_is_the_only_way_a_request_leaves_active() -> None:
    """Removal goes through the registry, so the executor's list is the one
    that changes and the coordinator never touches the container itself."""
    executor = _executor()
    registry = PyExecutorRequestRegistry(executor)
    keep, gone = Mock(), Mock()
    executor.active_requests = [keep, gone]

    registry.remove(gone)

    assert executor.active_requests == [keep]
    assert not registry.contains(gone)


def test_fake_and_adapter_record_the_same_effects() -> None:
    """Drive both implementations with one script and compare outcomes."""
    request, late_request, response = Mock(), Mock(), Mock()
    script = [
        ("terminate_request", (request,), {}),
        ("stage_transfer_response", (7, response, late_request), {}),
        ("fail_requests", ("boom", [request]), {"charge_budget": False}),
        ("fail_fatal", ("poisoned",), {}),
    ]
    fake = FakeExecutorEffects()
    executor = _executor()
    adapter = PyExecutorEffects(executor)
    for name, args, kwargs in script:
        getattr(fake, name)(*args, **kwargs)
        getattr(adapter, name)(*args, **kwargs)

    assert fake.terminated == [request]
    assert [c.args[0] for c in executor._terminate_request.call_args_list] == [request]
    assert fake.staged_responses == [(7, response, late_request)]
    assert executor._pending_transfer_responses == [(7, response)]
    assert executor._pending_response_terminations == [late_request]
    assert fake.failed == [("boom", [request], False)]
    assert fake.fatal == ["poisoned"]
    assert executor.is_shutdown is True
    assert executor._handle_errors.call_args_list == [
        call(error_msg="boom", requests=[request], charge_budget=False),
        call("poisoned", requests=None, charge_budget=False, fatal_is_collective_aligned=True),
    ]


def test_fake_registry_reads_live_and_removes_like_the_adapter() -> None:
    active = []
    registry = FakeRequestRegistry(active, canceled_request_ids=[3])
    request = SimpleNamespace(py_request_id=1)
    active.append(request)

    assert list(registry.active_requests()) == [request]
    assert registry.contains(request)
    assert list(registry.canceled_request_ids()) == [3]

    registry.remove(request)

    assert active == []
    assert registry.removed == [request]
