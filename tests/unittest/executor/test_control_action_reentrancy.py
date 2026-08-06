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
"""Guard tests for PyExecutor.control_action() re-entrancy.

No GPU, MPI or model weights required: the executor is built with
object.__new__ and only the handful of attributes control_action() touches.
"""

import threading
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.cpu_only


def _make_executor():
    """Build a PyExecutor shell exercising only control_action()'s state.

    rank is 1 so the rank-0 enqueue path is skipped, and the barrier starts
    set so wait() returns immediately instead of blocking on a real executor
    loop.
    """
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    ex = object.__new__(PyExecutor)
    ex.dist = SimpleNamespace(rank=1)
    ex.executor_request_queue = None  # unreachable while rank != 0
    ex.control_request_barrier = threading.Event()
    ex.control_request_barrier.set()
    ex.control_action_done = threading.Event()
    ex._control_action_in_progress = False
    return ex


def test_nested_control_action_is_rejected():
    """A control action must not open another control action."""
    ex = _make_executor()

    with ex.control_action():
        assert ex._control_action_in_progress
        with pytest.raises(RuntimeError, match="not re-entrant"):
            with ex.control_action():
                pytest.fail("nested control_action() should not have yielded")


def test_rejected_nesting_leaves_the_outer_handshake_intact():
    """The rejection must not touch the events the outer action still owns.

    control_request_barrier / control_action_done are single-slot state: if a
    refused nested entry cleared the barrier or set done, the executor loop
    would resume while the outer body is still running -- the very corruption
    the guard exists to prevent.
    """
    ex = _make_executor()

    with ex.control_action():
        with pytest.raises(RuntimeError):
            with ex.control_action():
                pass
        # Still mid-outer-action: barrier held, completion not signalled.
        assert ex.control_request_barrier.is_set()
        assert not ex.control_action_done.is_set()
        assert ex._control_action_in_progress

    # Outer exit performs the handshake exactly once.
    assert ex.control_action_done.is_set()
    assert not ex.control_request_barrier.is_set()
    assert not ex._control_action_in_progress


def test_sequential_control_actions_are_allowed():
    """The guard rejects nesting, not repeated use."""
    ex = _make_executor()

    for _ in range(3):
        ex.control_request_barrier.set()
        ex.control_action_done.clear()
        with ex.control_action():
            assert ex._control_action_in_progress
        assert not ex._control_action_in_progress


def test_flag_is_cleared_when_the_body_raises():
    """An exception inside the body must not leave the executor wedged."""
    ex = _make_executor()

    with pytest.raises(ValueError):
        with ex.control_action():
            raise ValueError("boom")

    assert not ex._control_action_in_progress

    # A later control action still works.
    ex.control_request_barrier.set()
    with ex.control_action():
        assert ex._control_action_in_progress
