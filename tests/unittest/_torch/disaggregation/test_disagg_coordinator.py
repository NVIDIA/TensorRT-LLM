# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Structural tests for the DisaggTransferCoordinator and its interfaces."""

import ast
import inspect
from dataclasses import fields
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.disaggregation.orchestration import coordinator as coordinator_module
from tensorrt_llm._torch.disaggregation.orchestration import interfaces as interfaces_module
from tensorrt_llm._torch.disaggregation.orchestration.coordinator import (
    DisaggLoopDelegates,
    DisaggTransferCoordinator,
    NoopDisaggCoordinator,
)
from tensorrt_llm._torch.disaggregation.orchestration.interfaces import (
    ActiveRequestRegistry,
    ExecutorEffects,
)

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


def _delegating_coordinator(delegates: DisaggLoopDelegates) -> DisaggTransferCoordinator:
    return DisaggTransferCoordinator(
        transceiver=Mock(),
        transfer_manager=Mock(),
        kv_cache_manager=Mock(),
        dist=Mock(),
        effects=Mock(spec=ExecutorEffects),
        registry=Mock(),
        enable_attention_dp=False,
        force_terminate_ctx_for_partial_reuse=False,
        delegates=delegates,
    )


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
    three effects plus one registry mutation. Growing it is a design decision,
    not a convenience."""
    assert _public_methods(ExecutorEffects) == {
        "terminate_request",
        "stage_transfer_response",
        "fail_requests",
    }
    assert _public_methods(ActiveRequestRegistry) == {
        "active_requests",
        "contains",
        "remove",
        "canceled_request_ids",
    }


@pytest.mark.parametrize("name", [f.name for f in fields(DisaggLoopDelegates)])
def test_delegated_methods_forward_to_their_own_delegate(name: str) -> None:
    """Each still-delegated entry point must reach exactly its own delegate with
    the arguments unchanged; a cross-wired or dropped call changes loop behavior
    and may break rank symmetry for collective-sensitive entry points."""
    if name in ("check_transfer_errors", "requests_in_error_state"):
        pytest.skip("reached from inside the reaps, not a coordinator entry point")
    delegates = DisaggLoopDelegates(**{f.name: Mock() for f in fields(DisaggLoopDelegates)})
    coordinator = _delegating_coordinator(delegates)
    method = getattr(coordinator, name)
    args = [object() for _ in inspect.signature(method).parameters]

    result = method(*args)

    getattr(delegates, name).assert_called_once_with(*args)
    for other in fields(DisaggLoopDelegates):
        if other.name != name:
            getattr(delegates, other.name).assert_not_called()
    expected = getattr(delegates, name).return_value if name == "admit" else None
    assert result is expected


def test_noop_coordinator_admits_everything_unchanged() -> None:
    """Without a transceiver, scheduler-fitting gen-init requests must pass
    through unfiltered and never report a transfer-budget block."""
    fitting = [object(), object()]
    assert NoopDisaggCoordinator().admit(fitting) == (fitting, False)


def test_noop_coordinator_overrides_every_entry_point() -> None:
    """The no-op coordinator has no services; any inherited implementation
    would dereference None on the first loop iteration."""
    inherited = {
        name
        for name in _public_methods(DisaggTransferCoordinator)
        if name not in vars(NoopDisaggCoordinator)
    }
    assert inherited <= {"admit"} | set(f.name for f in fields(DisaggLoopDelegates)), inherited


def test_noop_coordinator_accepts_every_loop_call() -> None:
    """Loops call the coordinator unconditionally, so the no-op variant must
    accept every call the real one does."""
    noop = NoopDisaggCoordinator()
    for name in (
        _public_methods(DisaggTransferCoordinator) - {"admit"} - _LOOP_ENTRY_POINTS_EXCLUDED
    ):
        params = inspect.signature(getattr(noop, name)).parameters
        getattr(noop, name)(*[Mock() for _ in params])


def test_noop_coordinator_refuses_release_transfer() -> None:
    """Connector-only releases stay in the executor; routing one here would
    silently leak the request."""
    with pytest.raises(RuntimeError, match="no transceiver"):
        NoopDisaggCoordinator().release_transfer(Mock())
