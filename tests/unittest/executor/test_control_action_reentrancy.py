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
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

pytestmark = pytest.mark.cpu_only

# Long enough that a wrongly-blocked thread is unambiguous, short enough that a
# regression does not stall the suite.
_TIMEOUT = 10.0


def _make_executor(rank: int = 1, queue: object = None) -> "PyExecutor":
    """Build a PyExecutor shell exercising only control_action()'s state.

    rank defaults to 1 so the rank-0 enqueue path is skipped, and the barrier
    starts set so the wait returns immediately instead of blocking on a real
    executor loop.  Pass rank=0 with a recording queue to exercise enqueue
    ordering.
    """
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    ex = object.__new__(PyExecutor)
    ex.dist = SimpleNamespace(rank=rank)
    ex.executor_request_queue = queue  # unreachable while rank != 0
    ex.control_request_barrier = threading.Event()
    ex.control_request_barrier.set()
    ex.control_action_done = threading.Event()
    ex.shutdown_event = threading.Event()
    ex._control_action_lock = threading.Lock()
    ex._control_action_owner = None
    return ex


class _RecordingQueue:
    """Records enqueue_control_request() calls so ordering can be asserted."""

    def __init__(self) -> None:
        self.calls: list = []

    def enqueue_control_request(self, **kwargs: object) -> None:
        self.calls.append(kwargs)


# ---------------------------------------------------------------------------
# Nesting (same thread) -- must raise, because blocking would deadlock
# ---------------------------------------------------------------------------


def test_nested_control_action_is_rejected() -> None:
    ex = _make_executor()

    with ex.control_action():
        assert ex._control_action_owner == threading.get_ident()
        with pytest.raises(RuntimeError, match="not re-entrant"):
            with ex.control_action():
                pytest.fail("nested control_action() should not have yielded")


def test_rejected_nesting_leaves_the_outer_handshake_intact() -> None:
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


def test_rejected_nesting_enqueues_no_control_request() -> None:
    """On rank 0 the guard must fire BEFORE enqueue_control_request().

    This is the property the change exists for: a refused nesting attempt must
    not leave an orphaned sentinel in the queue for the executor loop to fire
    at a body that never runs.  Needs rank=0 -- with rank=1 the enqueue is
    skipped entirely, so moving the guard below it would go unnoticed.
    """
    queue = _RecordingQueue()
    ex = _make_executor(rank=0, queue=queue)

    with ex.control_action(control_id="outer"):
        assert len(queue.calls) == 1, "outer action should enqueue exactly once"
        with pytest.raises(RuntimeError, match="not re-entrant"):
            with ex.control_action(control_id="nested"):
                pytest.fail("nested control_action() should not have yielded")
        # The decisive assertion: still one call, so the rejected nested entry
        # enqueued nothing.
        assert len(queue.calls) == 1, (
            f"rejected nesting left an orphaned control request: {queue.calls}"
        )

    assert [c.get("control_id") for c in queue.calls] == ["outer"]


# ---------------------------------------------------------------------------
# Concurrency (different threads) -- must serialise, NOT raise
# ---------------------------------------------------------------------------


def test_concurrent_callers_serialise_and_do_not_raise() -> None:
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

    def first() -> None:
        try:
            with ex.control_action():
                events.append("first-enter")
                first_inside.set()
                release_first.wait(timeout=_TIMEOUT)
            events.append("first-exit")
        except BaseException as exc:  # noqa: BLE001 - surfaced via assert below
            errors.append(("first", exc))

    def second() -> None:
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


def test_owner_is_per_thread_so_a_sibling_thread_is_not_mistaken_for_nesting() -> None:
    """The nesting check keys on thread ident, not a global flag."""
    ex = _make_executor()
    seen = {}
    first_inside = threading.Event()
    release_first = threading.Event()

    def first() -> None:
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


def test_sequential_control_actions_are_allowed() -> None:
    """The guard rejects nesting, not repeated use."""
    ex = _make_executor()

    for _ in range(3):
        ex.control_request_barrier.set()
        ex.control_action_done.clear()
        with ex.control_action():
            assert ex._control_action_owner == threading.get_ident()
        assert ex._control_action_owner is None


# ---------------------------------------------------------------------------
# Dead executor loop -- must be reported, not waited on forever
#
# The wait is deliberately unbounded (a deadline would strand the already
# enqueued sentinel and hang the loop), so these are the ONLY two escapes.
# Each runs the call on a worker thread and joins with a timeout, so a
# regression surfaces as a test failure rather than a hung CI stage.
# ---------------------------------------------------------------------------


def _run_expecting_shutdown_error(ex: "PyExecutor", control_id: str) -> str:
    """Call control_action() off-thread; return "raised"/"yielded"/"hung"."""
    outcome = []

    def body() -> None:
        try:
            with ex.control_action(control_id=control_id):
                outcome.append("yielded")
        except RuntimeError as exc:
            outcome.append("raised" if "shut down" in str(exc) else str(exc))

    t = threading.Thread(target=body, daemon=True)
    t.start()
    t.join(timeout=_TIMEOUT)
    if t.is_alive():
        return "hung"
    return outcome[0] if outcome else "no-outcome"


def test_shutdown_is_reported_instead_of_waited_on() -> None:
    """shutdown_event set => the loop already exited, so nothing will fire."""
    from tensorrt_llm._torch.pyexecutor import py_executor

    ex = _make_executor()
    ex.control_request_barrier.clear()
    ex.shutdown_event.set()

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(py_executor, "_CONTROL_BARRIER_POLL_INTERVAL_S", 0.05)
        assert _run_expecting_shutdown_error(ex, "after-shutdown") == "raised"

    assert not ex._control_action_lock.locked()
    assert ex._control_action_owner is None


def test_dead_worker_thread_is_reported_instead_of_waited_on() -> None:
    """A crashed executor loop must be caught with ``shutdown_event`` clear.

    ``shutdown_event`` only covers the orderly path.  A worker that died
    without setting it is the other half of the fast-fail condition, and is
    reachable only through the liveness check -- ``_make_executor`` leaves
    ``worker_thread`` unset, so every other test evaluates that half as
    ``False`` and would not notice it breaking.
    """
    from tensorrt_llm._torch.pyexecutor import py_executor

    ex = _make_executor()
    ex.control_request_barrier.clear()
    # shutdown_event stays CLEAR: only the worker liveness check can catch this.
    ex.worker_thread = SimpleNamespace(is_alive=lambda: False)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(py_executor, "_CONTROL_BARRIER_POLL_INTERVAL_S", 0.05)
        assert _run_expecting_shutdown_error(ex, "dead-worker") == "raised"

    assert not ex._control_action_lock.locked()
    assert ex._control_action_owner is None


def test_live_loop_keeps_waiting_rather_than_stranding_the_sentinel() -> None:
    """With the loop alive, a missing barrier edge must NOT raise.

    Rank 0 has already enqueued the sentinel by this point.  Bailing out would
    leave the loop to pop it, set the barrier and block forever in the untimed
    control_action_done.wait() with no caller left -- hanging the executor.
    So the caller has to keep waiting, and resume once the edge arrives.
    """
    from tensorrt_llm._torch.pyexecutor import py_executor

    ex = _make_executor()
    ex.control_request_barrier.clear()  # edge not fired yet
    ex.worker_thread = SimpleNamespace(is_alive=lambda: True)  # loop is healthy
    entered = threading.Event()

    def body() -> None:
        with ex.control_action(control_id="slow-drain"):
            entered.set()

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(py_executor, "_CONTROL_BARRIER_POLL_INTERVAL_S", 0.05)
        t = threading.Thread(target=body, daemon=True)
        t.start()
        # Must still be waiting, not raising, well past several poll intervals.
        assert not entered.wait(timeout=1.0), "caller bailed out on a live loop"
        assert t.is_alive(), "caller must not have raised while the loop is alive"

        ex.control_request_barrier.set()  # the loop finally fires it
        assert entered.wait(timeout=_TIMEOUT), "caller never resumed"
        t.join(timeout=_TIMEOUT)

    assert not ex._control_action_lock.locked()
    assert ex._control_action_owner is None


def test_lock_and_owner_are_released_when_the_body_raises() -> None:
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
