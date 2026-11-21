# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Structural tests for the DisaggTransferCoordinator and its interfaces."""

import ast
import inspect
from unittest.mock import Mock

import pytest
from coordinator_harness import CoordinatorHarness, TransferRequest

from tensorrt_llm._torch.disaggregation.orchestration import coordinator as coordinator_module
from tensorrt_llm._torch.disaggregation.orchestration import interfaces as interfaces_module
from tensorrt_llm._torch.disaggregation.orchestration.cancellation import ReceiveCancellation
from tensorrt_llm._torch.disaggregation.orchestration.coordinator import (
    DisaggTransferCoordinator,
    NoopDisaggCoordinator,
)
from tensorrt_llm._torch.disaggregation.orchestration.interfaces import (
    ActiveRequestRegistry,
    ExecutorEffects,
)
from tensorrt_llm.bindings import LlmRequestState

pytestmark = pytest.mark.cpu_only

# Entry points the executor loops call every iteration; the no-op coordinator
# must accept all of them. release_transfer is reached only with a
# transceiver (the executor keeps the connector-only release).
_LOOP_ENTRY_POINTS_EXCLUDED = {"release_transfer"}


def _public_methods(cls) -> set:
    return {
        name
        for name, member in inspect.getmembers(cls, predicate=inspect.isfunction)
        if not name.startswith("_")
    }


@pytest.mark.parametrize("module", [coordinator_module, interfaces_module])
def test_orchestration_modules_do_not_depend_on_py_executor(module) -> None:
    """The coordinator must be constructible and testable without PyExecutor."""
    tree = ast.parse(inspect.getsource(module))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    assert not any("py_executor" in name or name == "PyExecutor" for name in imported)


def test_executor_facing_surface_is_a_closed_set() -> None:
    """The executor-owned behavior the coordinator can trigger is a closed set:
    six effects plus one registry mutation. Growing it is a design decision,
    not a convenience."""
    assert _public_methods(ExecutorEffects) == {
        "terminate_request",
        "stage_transfer_response",
        "fail_requests",
        "fail_fatal",
        "prepare_gen_resources",
        "revert_ctx_alloc",
    }
    assert _public_methods(ActiveRequestRegistry) == {
        "active_requests",
        "contains",
        "remove",
        "canceled_request_ids",
    }


def test_noop_coordinator_admits_everything_unchanged() -> None:
    """Without a transceiver, scheduler-fitting gen-init requests must pass
    through unfiltered and never report a transfer-budget block."""
    fitting = [object(), object()]
    assert NoopDisaggCoordinator().admit(fitting) == (fitting, False)


def test_noop_coordinator_has_no_completed_receives_to_finish() -> None:
    """Without a transceiver no receive can complete: the query is empty and
    finishing is declined, so the executor never runs its receive tail."""
    noop = NoopDisaggCoordinator()
    assert noop.completed_gen_receives(Mock(generation_requests=[Mock()])) == []
    assert noop.try_finish_gen_receive(Mock()) is False


def test_noop_coordinator_overrides_every_entry_point() -> None:
    """The no-op coordinator has no services; any inherited implementation
    would dereference None on the first loop iteration."""
    inherited = {
        name
        for name in _public_methods(DisaggTransferCoordinator)
        if name not in vars(NoopDisaggCoordinator)
    }
    assert inherited == set(), inherited


@pytest.mark.parametrize("use_keywords", [False, True], ids=["positional", "keyword"])
@pytest.mark.parametrize("include_defaults", [False, True], ids=["required-only", "all"])
def test_noop_coordinator_accepts_every_loop_call(
    use_keywords: bool, include_defaults: bool
) -> None:
    """Loops call the coordinator unconditionally, so the no-op variant must
    accept positional and keyword calls, including omitted optional arguments."""
    noop = NoopDisaggCoordinator()
    for name in (
        _public_methods(DisaggTransferCoordinator) - {"admit"} - _LOOP_ENTRY_POINTS_EXCLUDED
    ):
        params = [
            param
            for key, param in inspect.signature(
                getattr(DisaggTransferCoordinator, name)
            ).parameters.items()
            if key != "self" and (include_defaults or param.default is inspect.Parameter.empty)
        ]
        args = [
            Mock()
            for param in params
            if param.kind == inspect.Parameter.POSITIONAL_ONLY
            or (param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD and not use_keywords)
        ]
        kwargs = {
            param.name: Mock()
            for param in params
            if param.kind == inspect.Parameter.KEYWORD_ONLY
            or (param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD and use_keywords)
        }
        inspect.signature(getattr(noop, name)).bind(*args, **kwargs)
        getattr(noop, name)(*args, **kwargs)


def test_noop_coordinator_refuses_release_transfer() -> None:
    """Connector-only releases stay in the executor; routing one here would
    silently leak the request."""
    with pytest.raises(RuntimeError, match="no transceiver"):
        NoopDisaggCoordinator().release_transfer(Mock())


def test_unsupported_receive_cancellation_does_not_take_fence() -> None:
    harness = CoordinatorHarness()
    request = TransferRequest(
        1,
        is_context_only_request=False,
        state=LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS,
    )
    ready = Mock(spec=["query"])
    harness.coordinator._unpublished_gen_receives[1] = (request, ready)
    assert harness.coordinator.cancel_gen_logically(request) is ReceiveCancellation.NOT_HANDLED
    assert harness.coordinator._unpublished_gen_receives == {1: (request, ready)}
    ready.query.assert_not_called()
    assert NoopDisaggCoordinator().cancel_gen_logically(request) is ReceiveCancellation.NOT_HANDLED
    assert not NoopDisaggCoordinator().supports_group_receive_cancellation


@pytest.mark.parametrize("physical_entry", [False, True])
@pytest.mark.parametrize("prior_failure", [False, True])
def test_coordinator_passes_failure_without_writing_request_state(
    physical_entry: bool, prior_failure: bool
) -> None:
    harness = CoordinatorHarness()
    request = TransferRequest(
        1,
        is_context_only_request=False,
        state=LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS,
    )
    ready = Mock(spec=["query"])
    harness.coordinator._unpublished_gen_receives[1] = (request, ready)
    harness.transceiver.supports_group_receive_cancellation = Mock(return_value=True)
    harness.transceiver.cancel_gen_logically = Mock(return_value=ReceiveCancellation.PENDING)
    harness.transceiver.cancel_request = Mock(return_value=False)
    if physical_entry:
        assert not harness.coordinator.request_cancellation(request, failed=prior_failure)
    else:
        assert (
            harness.coordinator.cancel_gen_logically(request, prior_failure=prior_failure)
            is ReceiveCancellation.PENDING
        )
    harness.transceiver.cancel_gen_logically.assert_called_once_with(
        request, ready, prior_failure=prior_failure
    )
    assert request.state == LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
    assert not harness.coordinator._unpublished_gen_receives
    assert not harness.coordinator._retiring_gen_receives
    ready.query.assert_not_called()
