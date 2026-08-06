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
"""Guard tests for PyExecutor.control_action() admission.

control_request_barrier / control_action_done are a broadcast handshake, not a
mutex: set() releases every waiter. So the method has to provide the exclusion
itself -- serialising callers from different threads, and rejecting a nested
call from the thread that already holds it (blocking there would deadlock).

No GPU, MPI or model weights required: the executor is built with
object.__new__ and only the attributes control_action() touches.
"""

import threading
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.cpu_only

# Long enough that a wrongly-blocked thread is unambiguous, short enough that a
# regression does not stall the suite.
_TIMEOUT = 10.0


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
    ex._control_action_lock = threading.Lock()
    ex._control_action_owner = None
    return ex


# ---------------------------------------------------------------------------
# Nesting (same thread) -- must raise, because blocking would deadlock
# ---------------------------------------------------------------------------


def test_nested_control_action_is_rejected():
    ex = _make_executor()

    with ex.control_action():
        assert ex._control_action_owner == threading.get_ident()
        with pytest.raises(RuntimeError, match="not re-entrant"):
            with ex.control_action():
                pytest.fail("nested control_action() should not have yielded")


def test_rejected_nesting_leaves_the_outer_handshake_intact():
    """The rejection must not touch the events the outer action still owns.

    If a refused nested entry cleared the barrier or set done, the executor
    loop would resume while the outer body is still running -- the corruption
    the guard exists to prevent.
    """
    ex = _make_executor()

    with ex.control_action():
        with pytest.raises(RuntimeError):
            with ex.control_action():
                pass
        assert ex.control_request_barrier.is_set()
        assert not ex.control_action_done.is_set()
        assert ex._control_action_owner == threading.get_ident()

    assert ex.control_action_done.is_set()
    assert not ex.control_request_barrier.is_set()
    assert ex._control_action_owner is None
    assert not ex._control_action_lock.locked()


# ---------------------------------------------------------------------------
# Concurrency (different threads) -- must serialise, NOT raise
# ---------------------------------------------------------------------------


def test_concurrent_callers_serialise_and_do_not_raise():
    """A second thread waits its turn rather than being refused.

    Rejecting it would be wrong twice over: contention is not re-entrancy, and
    the existing callers (base_worker's _sleep_wakeup_lock sites) rely on
    concurrent control actions serialising.
    """
    ex = _make_executor()
    events = []
    errors = []
    first_inside = threading.Event()
    release_first = threading.Event()
    second_inside = threading.Event()

    def first():
        try:
            with ex.control_action():
                events.append("first-enter")
                first_inside.set()
                release_first.wait(timeout=_TIMEOUT)
            events.append("first-exit")
        except BaseException as exc:  # noqa: BLE001 - surfaced via assert below
            errors.append(("first", exc))

    def second():
        try:
            with ex.control_action():
                events.append("second-enter")
                second_inside.set()
        except BaseException as exc:  # noqa: BLE001
            errors.append(("second", exc))

    t1 = threading.Thread(target=first)
    t1.start()
    assert first_inside.wait(timeout=_TIMEOUT), "first thread never entered"

    t2 = threading.Thread(target=second)
    t2.start()

    # While the first action holds the lock the second must be blocked, not
    # raising and not running the body.
    assert not second_inside.wait(timeout=0.5), "second thread entered concurrently"
    assert errors == [], f"a concurrent caller was refused: {errors}"

    release_first.set()
    t1.join(timeout=_TIMEOUT)

    # The first action's cleanup cleared the barrier; the real executor loop
    # re-arms it for the next control request, so do that here.
    ex.control_request_barrier.set()
    assert second_inside.wait(timeout=_TIMEOUT), "second thread never got its turn"
    t2.join(timeout=_TIMEOUT)

    assert errors == []
    assert events == ["first-enter", "first-exit", "second-enter"]
    assert ex._control_action_owner is None
    assert not ex._control_action_lock.locked()


def test_owner_is_per_thread_so_a_sibling_thread_is_not_mistaken_for_nesting():
    """The nesting check keys on thread ident, not a global flag."""
    ex = _make_executor()
    seen = {}
    first_inside = threading.Event()
    release_first = threading.Event()

    def first():
        with ex.control_action():
            first_inside.set()
            release_first.wait(timeout=_TIMEOUT)

    t1 = threading.Thread(target=first)
    t1.start()
    assert first_inside.wait(timeout=_TIMEOUT)

    # From the main thread the owner is someone else, so this is contention,
    # not nesting -- the pre-check must not fire.
    seen["owner"] = ex._control_action_owner
    assert seen["owner"] not in (None, threading.get_ident())

    release_first.set()
    t1.join(timeout=_TIMEOUT)
    assert ex._control_action_owner is None


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def test_sequential_control_actions_are_allowed():
    """The guard rejects nesting, not repeated use."""
    ex = _make_executor()

    for _ in range(3):
        ex.control_request_barrier.set()
        ex.control_action_done.clear()
        with ex.control_action():
            assert ex._control_action_owner == threading.get_ident()
        assert ex._control_action_owner is None


def test_lock_and_owner_are_released_when_the_body_raises():
    """An exception inside the body must not leave the executor wedged."""
    ex = _make_executor()

    with pytest.raises(ValueError):
        with ex.control_action():
            raise ValueError("boom")

    assert ex._control_action_owner is None
    assert not ex._control_action_lock.locked()

    ex.control_request_barrier.set()
    with ex.control_action():
        assert ex._control_action_owner == threading.get_ident()
