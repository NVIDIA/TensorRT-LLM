# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the DisaggTransferCoordinator skeleton."""

import ast
import inspect
from dataclasses import fields
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.disaggregation.executor import coordinator as coordinator_module
from tensorrt_llm._torch.disaggregation.executor.coordinator import (
    DisaggLoopDelegates,
    DisaggTransferCoordinator,
    NoopDisaggCoordinator,
)

pytestmark = pytest.mark.cpu_only


def _public_methods(cls) -> set:
    return {
        name
        for name, member in inspect.getmembers(cls, predicate=inspect.isfunction)
        if not name.startswith("_")
    }


def test_coordinator_module_does_not_depend_on_py_executor() -> None:
    """The coordinator must be constructible and testable without PyExecutor."""
    tree = ast.parse(inspect.getsource(coordinator_module))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    assert not any("py_executor" in name or name == "PyExecutor" for name in imported)


def test_every_coordinator_method_has_a_delegate() -> None:
    """Every entry point must be backed by a delegate: a silently no-op method on
    the real coordinator would drop a rank-consensus collective and hang peers."""
    delegate_names = {f.name for f in fields(DisaggLoopDelegates)}
    assert _public_methods(DisaggTransferCoordinator) == delegate_names


@pytest.mark.parametrize("name", [f.name for f in fields(DisaggLoopDelegates)])
def test_real_coordinator_forwards_each_method_to_its_delegate(name: str) -> None:
    """Each entry point must reach exactly its own delegate with the arguments
    unchanged; a cross-wired or dropped call changes loop behavior and may break
    rank symmetry for collective-sensitive entry points."""
    delegates = DisaggLoopDelegates(**{f.name: Mock() for f in fields(DisaggLoopDelegates)})
    coordinator = DisaggTransferCoordinator(delegates)
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


def test_noop_coordinator_accepts_every_loop_call() -> None:
    """Loops call the coordinator unconditionally, so the no-op variant must
    accept every call the real one does."""
    noop = NoopDisaggCoordinator()
    for name in _public_methods(DisaggTransferCoordinator) - {"admit"}:
        params = inspect.signature(getattr(noop, name)).parameters
        getattr(noop, name)(*[Mock() for _ in params])
