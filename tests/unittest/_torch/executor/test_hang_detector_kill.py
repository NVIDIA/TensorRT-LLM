# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""HangDetector timer behavior and the hard-kill propagation mechanism (no GPU)."""

import asyncio
import contextlib
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import types

import pytest

from tensorrt_llm._torch.pyexecutor import hang_detector as hang_detector_module
from tensorrt_llm._torch.pyexecutor.hang_detector import (
    RANK_CRASH_KILL_GRACE_ENV,
    HangDetector,
    hard_kill_on_rank_crash,
    start_rank_crash_kill_watchdog,
)


def test_detector_fires_after_timeout():
    fired = []
    hd = HangDetector(timeout=2, on_detected=lambda: fired.append(time.monotonic()))
    with hd:
        hd.checkpoint()
        time.sleep(1.0)
        assert hd.detected() is False
        assert fired == []
        # Poll up to a generous deadline rather than asserting at the exact
        # timeout boundary -- the detector thread may wake a bit after the
        # configured ``timeout`` and a fixed sleep flakes in CI.
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and not hd.detected():
            time.sleep(0.05)
        assert hd.detected() is True
        assert len(fired) == 1


def test_checkpoint_resets_timer():
    """Repeated checkpoints before the timeout keep the detector quiet."""
    fired = []
    hd = HangDetector(timeout=2, on_detected=lambda: fired.append(1))
    with hd:
        for _ in range(6):
            hd.checkpoint()
            time.sleep(0.4)  # < timeout, so the timer keeps resetting
        assert fired == []
        assert hd.detected() is False


def test_pause_suppresses_detection():
    fired = []
    hd = HangDetector(timeout=1, on_detected=lambda: fired.append(1))
    with hd:
        hd.checkpoint()
        with hd.pause():
            time.sleep(2.0)  # would have fired if not paused
        assert fired == []
        assert hd.detected() is False


def test_status_provider_errors_are_logged(monkeypatch):
    events = []

    async def no_sleep(_timeout):
        pass

    def failing_provider():
        raise RuntimeError("provider failed")

    monkeypatch.setattr(hang_detector_module.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(
        hang_detector_module,
        "_best_effort_log_error",
        lambda message: events.append(("log", message)),
    )
    monkeypatch.setattr(
        hang_detector_module,
        "print_all_stacks",
        lambda: events.append(("stacks", None)),
    )

    detector = HangDetector(timeout=1, on_detected=lambda: events.append(("detected", None)))
    detector.register_status_provider(failing_provider)
    detector.register_status_provider(lambda: "transceiver status")

    asyncio.run(detector._detect_hang())

    messages = "\n".join(message for kind, message in events if kind == "log")
    assert "provider failed" in messages
    assert "transceiver status" in messages
    assert events[-2:] == [("stacks", None), ("detected", None)]


def test_propagate_hard_kill_self_sigkills_without_mpi():
    """With MPI disabled, propagate_hard_kill self-SIGKILLs the process.

    A SIGKILL'd process reports returncode -SIGKILL (== -9) to the parent.
    """
    script = (
        "from tensorrt_llm._torch.pyexecutor.hang_detector import propagate_hard_kill; "
        "propagate_hard_kill()"
    )
    env = {**os.environ, "TLLM_DISABLE_MPI": "1"}
    # Generous timeout: the subprocess pays a cold `import tensorrt_llm` (full
    # _torch init), which alone can take a minute on slower hosts, before it
    # ever reaches propagate_hard_kill().
    proc = subprocess.run([sys.executable, "-c", script], env=env, timeout=300, capture_output=True)
    assert proc.returncode == -signal.SIGKILL, (
        f"expected self-SIGKILL (-9), got {proc.returncode}; "
        f"stderr={proc.stderr.decode(errors='replace')[-500:]}"
    )


# --------------------------------------------------------------------------
# hard_kill_on_rank_crash: a rank whose executor loop crashed must kill the
# world (after a grace) instead of leaving peers to burn 300s in collectives.
# --------------------------------------------------------------------------


def test_rank_crash_kill_single_rank_is_noop(monkeypatch):
    """No peers to unblock: the worker's own death already carries the error."""
    kills = []
    monkeypatch.setattr(hang_detector_module, "propagate_hard_kill", lambda: kills.append(1))
    assert hard_kill_on_rank_crash(world_size=1) is False
    assert kills == []


def test_rank_crash_kill_fires_for_multi_rank(monkeypatch):
    kills = []
    monkeypatch.setattr(hang_detector_module, "propagate_hard_kill", lambda: kills.append(1))
    monkeypatch.setenv(RANK_CRASH_KILL_GRACE_ENV, "0")
    assert hard_kill_on_rank_crash(world_size=4) is True
    assert kills == [1]


def _assert_slept_then_killed(order, grace):
    """Assert the grace was slept out EXACTLY ONCE before EXACTLY ONE kill.

    Patching time.sleep is process-wide, so an unrelated background thread can
    append its own ("sleep", x) while this runs; asserting exact list equality
    would flake on that. But the counts must still be pinned: sleeping the
    grace twice before killing (i.e. crash + 2*grace) is precisely the bug
    class this PR series introduced with its two independent timers, and a
    membership-only check accepts it.
    """
    assert order.count(("sleep", grace)) == 1, order
    assert order.count("kill") == 1, order
    assert order.index(("sleep", grace)) < order.index("kill"), order


def test_rank_crash_kill_sleeps_grace_before_kill(monkeypatch):
    """The grace must elapse BEFORE the kill so cleaner error paths win the race."""
    order = []
    monkeypatch.setattr(hang_detector_module, "propagate_hard_kill", lambda: order.append("kill"))
    monkeypatch.setattr(hang_detector_module.time, "sleep", lambda s: order.append(("sleep", s)))
    monkeypatch.setenv(RANK_CRASH_KILL_GRACE_ENV, "2.5")
    assert hard_kill_on_rank_crash(world_size=2) is True
    _assert_slept_then_killed(order, 2.5)


def test_rank_crash_kill_disabled_by_negative_grace(monkeypatch):
    kills = []
    monkeypatch.setattr(hang_detector_module, "propagate_hard_kill", lambda: kills.append(1))
    monkeypatch.setenv(RANK_CRASH_KILL_GRACE_ENV, "-1")
    assert hard_kill_on_rank_crash(world_size=8) is False
    assert kills == []


def test_rank_crash_kill_invalid_grace_uses_default(monkeypatch):
    """A malformed env value must not disable the kill (fail-safe default)."""
    order = []
    monkeypatch.setattr(hang_detector_module, "propagate_hard_kill", lambda: order.append("kill"))
    monkeypatch.setattr(hang_detector_module.time, "sleep", lambda s: order.append(("sleep", s)))
    monkeypatch.setenv(RANK_CRASH_KILL_GRACE_ENV, "bogus")
    assert hard_kill_on_rank_crash(world_size=2) is True
    _assert_slept_then_killed(order, 10.0)


@pytest.mark.parametrize("raw", ["nan", "NaN", "inf", "-inf", "Infinity"])
def test_rank_crash_kill_non_finite_grace_uses_default(monkeypatch, raw):
    """float() accepts nan/inf, but neither survives the arithmetic downstream.

    ``nan`` slips past the ``grace < 0`` check (all nan comparisons are False)
    and then collapses to a ZERO grace -- ``max(0.0, nan)`` is ``0.0`` and
    ``remaining > 0`` is False -- so the kill fires instantly and destroys the
    window the grace exists to give the traceback. ``inf`` is the mirror case:
    a watchdog that never fires, silently equivalent to ``-1`` but costing a
    live thread. Both must fall back to the documented default.
    """
    order = []
    monkeypatch.setattr(hang_detector_module, "propagate_hard_kill", lambda: order.append("kill"))
    monkeypatch.setattr(hang_detector_module.time, "sleep", lambda s: order.append(("sleep", s)))
    monkeypatch.setenv(RANK_CRASH_KILL_GRACE_ENV, raw)
    assert hard_kill_on_rank_crash(world_size=2) is True
    _assert_slept_then_killed(order, 10.0)


def test_rank_crash_kill_never_raises(monkeypatch):
    """It runs in a `finally`: raising would mask the loop's original error."""

    def boom():
        raise RuntimeError("abort machinery broken")

    monkeypatch.setattr(hang_detector_module, "propagate_hard_kill", boom)
    monkeypatch.setenv(RANK_CRASH_KILL_GRACE_ENV, "0")
    assert hard_kill_on_rank_crash(world_size=2) is False


# --------------------------------------------------------------------------
# start_rank_crash_kill_watchdog: the kill must fire even when executor-loop
# cleanup never returns (e.g. blocked on a PP send handle wedged by the
# crash), so it is armed in a daemon thread BEFORE cleanup starts.
# --------------------------------------------------------------------------


def test_watchdog_kills_while_caller_blocks(monkeypatch):
    """The kill fires from the watchdog thread with no help from the caller."""
    killed = threading.Event()
    monkeypatch.setattr(hang_detector_module, "propagate_hard_kill", killed.set)
    monkeypatch.setenv(RANK_CRASH_KILL_GRACE_ENV, "0")

    watchdog = start_rank_crash_kill_watchdog(world_size=2)

    assert watchdog is not None
    assert watchdog.daemon  # must never block interpreter exit
    # The caller does nothing further (it would be blocked in cleanup);
    # the kill must fire regardless.
    assert killed.wait(timeout=30.0)
    watchdog.join(timeout=30.0)


def test_watchdog_not_armed_for_single_rank(monkeypatch):
    kills = []
    monkeypatch.setattr(hang_detector_module, "propagate_hard_kill", lambda: kills.append(1))
    monkeypatch.setenv(RANK_CRASH_KILL_GRACE_ENV, "0")
    assert start_rank_crash_kill_watchdog(world_size=1) is None
    assert kills == []


def test_watchdog_not_armed_when_disabled(monkeypatch):
    kills = []
    monkeypatch.setattr(hang_detector_module, "propagate_hard_kill", lambda: kills.append(1))
    monkeypatch.setenv(RANK_CRASH_KILL_GRACE_ENV, "-1")
    assert start_rank_crash_kill_watchdog(world_size=8) is None
    assert kills == []


def test_watchdog_cancel_disarms_this_timer(monkeypatch):
    """cancel() must break the grace wait immediately, not after it elapses.

    This is the handover primitive, NOT protection against a spurious kill:
    the only production caller cancels in order to take the same kill over on
    the same deadline. What decides whether a rank is killed at all is the
    `crashed` predicate in _event_loop_wrapper.
    """
    kills = []
    monkeypatch.setattr(hang_detector_module, "propagate_hard_kill", lambda: kills.append(1))
    monkeypatch.setenv(RANK_CRASH_KILL_GRACE_ENV, "30")

    watchdog = start_rank_crash_kill_watchdog(world_size=2)
    assert watchdog is not None
    try:
        watchdog.cancel()
        watchdog.join(timeout=10.0)
        assert not watchdog.is_alive()
        assert kills == []
        assert watchdog.cancelled is True
    finally:
        # Never let an armed killer thread outlive the stubbed
        # propagate_hard_kill: if cancel() ever regresses, the real one would
        # SIGKILL the pytest process once monkeypatch restores it.
        watchdog.cancel()
        watchdog.join(timeout=60.0)


def test_watchdog_deadline_is_grace_from_arming(monkeypatch):
    monkeypatch.setenv(RANK_CRASH_KILL_GRACE_ENV, "5")
    watchdog = hang_detector_module.RankCrashKillWatchdog(world_size=2, grace=5.0)
    assert watchdog.deadline == pytest.approx(time.monotonic() + 5.0, abs=0.5)


def test_kill_keeps_original_deadline_on_handover(monkeypatch):
    """Handing the kill over must not restart the grace clock.

    The caller cancels the watchdog once cleanup returns and carries the kill
    itself; passing the watchdog's deadline must make it sleep the REMAINING
    time, not a fresh grace. Asserted on the exact duration handed to sleep
    (with monotonic pinned) rather than on wall-clock, so the margin does not
    depend on scheduling luck on a loaded CI node.
    """
    order = []
    monkeypatch.setattr(hang_detector_module, "propagate_hard_kill", lambda: order.append("kill"))
    monkeypatch.setattr(hang_detector_module.time, "sleep", lambda s: order.append(("sleep", s)))
    monkeypatch.setattr(hang_detector_module.time, "monotonic", lambda: 1000.0)
    monkeypatch.setenv(RANK_CRASH_KILL_GRACE_ENV, "30")

    # 29.5s of the 30s grace has already been burned by the watchdog.
    assert hard_kill_on_rank_crash(world_size=2, deadline=1000.5) is True
    # Exactly one 0.5s sleep, then exactly one kill -- never a second grace.
    assert order.count(("sleep", 0.5)) == 1, order
    assert not any(s == ("sleep", 30.0) for s in order), order
    _assert_slept_then_killed(order, 0.5)


def test_kill_fires_immediately_when_deadline_already_passed(monkeypatch):
    """A deadline in the past must fire the kill now, and never sleep negative.

    time.sleep of a negative duration raises into hard_kill_on_rank_crash's
    blanket except, which would return False and silently skip the kill --
    exactly in the case the watchdog exists for (cleanup outlasted the grace).
    """
    order = []
    monkeypatch.setattr(hang_detector_module, "propagate_hard_kill", lambda: order.append("kill"))
    monkeypatch.setattr(hang_detector_module.time, "sleep", lambda s: order.append(("sleep", s)))
    monkeypatch.setenv(RANK_CRASH_KILL_GRACE_ENV, "30")

    t0 = time.monotonic()
    assert hard_kill_on_rank_crash(world_size=2, deadline=t0 - 100.0) is True
    assert order == ["kill"], order
    assert not [s for s in order if isinstance(s, tuple) and s[1] < 0], order


def test_wait_out_kill_grace_never_sleeps_negative(monkeypatch):
    """The `remaining > 0` guard, not the deadline clamp, is what protects here."""
    slept = []
    monkeypatch.setattr(hang_detector_module.time, "sleep", lambda s: slept.append(s))
    assert hang_detector_module._wait_out_kill_grace(-100.0, None) is True
    assert slept == []
    # The cancellable path must also return promptly, not wait forever.
    assert hang_detector_module._wait_out_kill_grace(-100.0, threading.Event()) is True


# --------------------------------------------------------------------------
# Wiring: PyExecutor._event_loop_wrapper must invoke the kill on the crash
# path only, and only after local cleanup has woken rank-local waiters.
# --------------------------------------------------------------------------


def _bare_executor(pe, monkeypatch, world_size, is_shutdown=False):
    # Neutralize the profiling/GC context managers: they are irrelevant to the
    # crash path and must not depend on env/GC state in a unit test.
    monkeypatch.setattr(pe, "host_profiler_context", lambda enable: contextlib.nullcontext())
    monkeypatch.setattr(pe, "customized_gc_thresholds", lambda threshold: contextlib.nullcontext())
    ex = pe.PyExecutor.__new__(pe.PyExecutor)
    ex.dist = types.SimpleNamespace(world_size=world_size)
    ex.garbage_collection_gen0_threshold = None
    # is_shutdown must NOT influence the crash decision -- _handle_errors sets
    # it rank-locally on a fatal error while peers are told nothing.
    ex.is_shutdown = is_shutdown
    ex._event_loop_completed = False
    # The real __init__ creates this; __new__ does not run it. The wrapper
    # reads it on every crash path, so a bare executor without it turns a
    # wiring test into an AttributeError.
    ex._event_loop_error_delivered = threading.Event()
    return ex


class _FakeWatchdog:
    """Stand-in for RankCrashKillWatchdog that records cancellation.

    ``cancelled`` is a read-only property, matching the real class: a wiring
    change that assigned to it would pass against a plain attribute here and
    raise AttributeError in production.
    """

    def __init__(self, events, world_size):
        self._events = events
        self._cancelled = False
        self.deadline = 1234.5
        events.append(("watchdog", world_size))

    @property
    def cancelled(self):
        return self._cancelled

    def cancel(self):
        self._cancelled = True
        self._events.append("cancel")


def _stub_kill_paths(pe, monkeypatch, events, arm_watchdog=True, seen=None):
    # ``error_delivered`` is keyword-only with NO default on both stubs: if the
    # wiring that threads the delivery gate through is ever dropped, these
    # raise TypeError instead of silently accepting the pre-gate signature.
    # Pass ``seen`` to capture the objects actually handed over.
    def _kill(world_size, deadline=None, *, error_delivered):
        if seen is not None:
            seen.append(("kill", error_delivered))
        events.append(("kill", world_size, deadline))

    monkeypatch.setattr(pe, "hard_kill_on_rank_crash", _kill)
    watchdogs = []

    def _start(world_size, *, error_delivered):
        if seen is not None:
            seen.append(("watchdog", error_delivered))
        if not arm_watchdog:
            events.append(("watchdog", world_size))
            return None
        wd = _FakeWatchdog(events, world_size)
        watchdogs.append(wd)
        return wd

    monkeypatch.setattr(pe, "start_rank_crash_kill_watchdog", _start)
    return watchdogs


def test_event_loop_wrapper_kills_world_on_crash(monkeypatch):
    """A genuine mid-loop crash (is_shutdown still False) must kill the world."""
    from tensorrt_llm._torch.pyexecutor import py_executor as pe

    events = []
    watchdogs = _stub_kill_paths(pe, monkeypatch, events)
    ex = _bare_executor(pe, monkeypatch, world_size=4, is_shutdown=False)
    ex._executor_loop_cleanup = lambda: events.append("cleanup")

    def crash():
        raise ValueError("boom")

    ex.event_loop = crash

    with pytest.raises(ValueError, match="boom"):
        ex._event_loop_wrapper()

    # The watchdog is armed BEFORE cleanup (cleanup can block forever);
    # cleanup wakes rank-local waiters (who read the stashed error) BEFORE
    # the direct kill tears the world down. Once cleanup returns, the
    # watchdog is disarmed and the kill is carried inline on the watchdog's
    # ORIGINAL deadline, so only one timer is ever live.
    assert events == [("watchdog", 4), "cleanup", "cancel", ("kill", 4, 1234.5)]
    assert watchdogs[0].cancelled is True
    assert isinstance(ex._event_loop_error, ValueError)


def test_event_loop_wrapper_hands_both_kill_paths_this_executors_gate(monkeypatch):
    """Both kill paths must receive THIS executor's delivery gate.

    Handing over a fresh Event, or a different executor's, would read as
    "the error never reached the client" and kill the world even on the
    path the grace exists to protect.
    """
    from tensorrt_llm._torch.pyexecutor import py_executor as pe

    events, seen = [], []
    _stub_kill_paths(pe, monkeypatch, events, seen=seen)
    ex = _bare_executor(pe, monkeypatch, world_size=4, is_shutdown=False)
    ex._executor_loop_cleanup = lambda: events.append("cleanup")

    def crash():
        raise ValueError("boom")

    ex.event_loop = crash

    with pytest.raises(ValueError, match="boom"):
        ex._event_loop_wrapper()

    assert [kind for kind, _ in seen] == ["watchdog", "kill"]
    assert all(gate is ex._event_loop_error_delivered for _, gate in seen)


def test_event_loop_wrapper_kills_world_when_cleanup_raises(monkeypatch):
    """The kill must not be skippable by a cleanup failure.

    Cleanup runs precisely when the process is already unhealthy; if its
    exception aborted the finally block before the kill, peers would burn
    300s in their HangDetectors — the worst case is exactly when the kill
    matters most.
    """
    from tensorrt_llm._torch.pyexecutor import py_executor as pe

    events = []
    _stub_kill_paths(pe, monkeypatch, events)
    ex = _bare_executor(pe, monkeypatch, world_size=4, is_shutdown=False)

    def broken_cleanup():
        events.append("cleanup")
        raise RuntimeError("cleanup exploded")

    ex._executor_loop_cleanup = broken_cleanup

    def crash():
        raise ValueError("boom")

    ex.event_loop = crash

    with pytest.raises(RuntimeError, match="cleanup exploded"):
        ex._event_loop_wrapper()

    assert events == [("watchdog", 4), "cleanup", "cancel", ("kill", 4, 1234.5)]
    # The original loop error stays reachable for rank-local consumers.
    assert isinstance(ex._event_loop_error, ValueError)


def test_event_loop_wrapper_kills_world_when_watchdog_cannot_arm(monkeypatch):
    """A watchdog that fails to start must not silently drop the escalation."""
    from tensorrt_llm._torch.pyexecutor import py_executor as pe

    events = []
    _stub_kill_paths(pe, monkeypatch, events, arm_watchdog=False)
    ex = _bare_executor(pe, monkeypatch, world_size=4, is_shutdown=False)
    ex._executor_loop_cleanup = lambda: events.append("cleanup")

    def crash():
        raise ValueError("boom")

    ex.event_loop = crash

    with pytest.raises(ValueError, match="boom"):
        ex._event_loop_wrapper()

    assert events == [("watchdog", 4), "cleanup", ("kill", 4, None)]


def test_event_loop_wrapper_no_kill_on_clean_exit(monkeypatch):
    from tensorrt_llm._torch.pyexecutor import py_executor as pe

    events = []
    _stub_kill_paths(pe, monkeypatch, events)
    ex = _bare_executor(pe, monkeypatch, world_size=4)
    ex._executor_loop_cleanup = lambda: events.append("cleanup")
    ex.event_loop = lambda: None

    ex._event_loop_wrapper()

    assert events == ["cleanup"]


# --------------------------------------------------------------------------
# The kill must stay scoped to crashes that actually strand peers, and the
# only signal that says so is _event_loop_completed -- set at the loops'
# normal-exit `break` sites and nowhere else. is_shutdown does NOT mean
# "peers were told": _handle_errors flips it rank-locally on a fatal error.
# --------------------------------------------------------------------------


def test_event_loop_wrapper_no_kill_when_loop_raises_after_completing(monkeypatch):
    """A raise after the loop's normal-exit break is a teardown error, not a crash."""
    from tensorrt_llm._torch.pyexecutor import py_executor as pe

    events = []
    _stub_kill_paths(pe, monkeypatch, events)
    ex = _bare_executor(pe, monkeypatch, world_size=4)
    ex._executor_loop_cleanup = lambda: events.append("cleanup")

    def late_raise():
        # The loop hit its normal-exit `break` and drained all work, then
        # something raised on the way out (e.g. a context manager's __exit__).
        ex._event_loop_completed = True
        raise RuntimeError("teardown hiccup")

    ex.event_loop = late_raise

    with pytest.raises(RuntimeError, match="teardown hiccup"):
        ex._event_loop_wrapper()

    # Logged and re-raised, but no watchdog and no kill: peers are not stranded.
    assert events == ["cleanup"]
    assert isinstance(ex._event_loop_error, RuntimeError)


def test_event_loop_wrapper_kills_world_on_rank_local_fatal(monkeypatch):
    """REGRESSION: a rank-local CUDA fatal sets is_shutdown but strands peers.

    _handle_errors classifies a device-side fault as immediate_fatal, sets
    is_shutdown=True on THIS rank and enqueues a shutdown into THIS process's
    own queue -- peers are told nothing and keep waiting in their collective.
    An exception raised after that point (e.g. the unguarded
    guided_decoder.execute(batch_outputs['logits']) on a None batch_outputs)
    must still hard-kill the world. Keying the decision off is_shutdown
    silently disabled the kill for this, the feature's most common trigger.
    """
    from tensorrt_llm._torch.pyexecutor import py_executor as pe

    events = []
    _stub_kill_paths(pe, monkeypatch, events)
    ex = _bare_executor(pe, monkeypatch, world_size=4)
    ex._executor_loop_cleanup = lambda: events.append("cleanup")

    def cuda_fatal_then_crash():
        ex.is_shutdown = True  # what _handle_errors does, rank-locally
        assert ex._event_loop_completed is False  # the loop never terminated
        raise TypeError("'NoneType' object is not subscriptable")

    ex.event_loop = cuda_fatal_then_crash

    with pytest.raises(TypeError):
        ex._event_loop_wrapper()

    assert events == [("watchdog", 4), "cleanup", "cancel", ("kill", 4, 1234.5)]


def test_event_loop_wrapper_kills_world_when_loop_never_started(monkeypatch):
    """A failure before the loop runs strands peers just as surely as one inside it."""
    from tensorrt_llm._torch.pyexecutor import py_executor as pe

    events = []
    _stub_kill_paths(pe, monkeypatch, events)

    @contextlib.contextmanager
    def failing_enter(**_kwargs):
        raise RuntimeError("profiler setup failed")
        yield  # pragma: no cover

    ex = _bare_executor(pe, monkeypatch, world_size=4)
    monkeypatch.setattr(pe, "host_profiler_context", lambda enable: failing_enter())
    ex._executor_loop_cleanup = lambda: events.append("cleanup")
    ex.event_loop = lambda: pytest.fail("event_loop must not be reached")

    with pytest.raises(RuntimeError, match="profiler setup failed"):
        ex._event_loop_wrapper()

    assert events == [("watchdog", 4), "cleanup", "cancel", ("kill", 4, 1234.5)]


def test_event_loop_wrapper_no_kill_when_enclosing_context_manager_raises(monkeypatch):
    """Teardown of the host-profiler / GC context managers after a completed loop.

    They wrap event_loop() but are not part of it; a failure while unwinding
    them once the loop has completed leaves no peer waiting on this rank.
    """
    from tensorrt_llm._torch.pyexecutor import py_executor as pe

    events = []
    _stub_kill_paths(pe, monkeypatch, events)

    @contextlib.contextmanager
    def exploding_ctx(**_kwargs):
        yield
        raise RuntimeError("profiler teardown failed")

    ex = _bare_executor(pe, monkeypatch, world_size=4)
    monkeypatch.setattr(pe, "host_profiler_context", lambda enable: exploding_ctx())
    ex._executor_loop_cleanup = lambda: events.append("cleanup")

    def completed_loop():
        ex._event_loop_completed = True

    ex.event_loop = completed_loop

    with pytest.raises(RuntimeError, match="profiler teardown failed"):
        ex._event_loop_wrapper()

    assert events == ["cleanup"]


# --------------------------------------------------------------------------
# The sentinel is only trustworthy if the real loops actually set it. Assert
# against the shipped source so a new normal-exit path (or a moved break)
# cannot silently make every clean shutdown look like a peer-stranding crash.
# --------------------------------------------------------------------------


def _executor_loop_ast_nodes():
    import ast
    import inspect

    from tensorrt_llm._torch.pyexecutor import py_executor as pe

    tree = ast.parse(inspect.getsource(pe))
    wanted = {"_executor_loop", "_executor_loop_pp", "_executor_loop_overlap"}
    return {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    }


def _outer_while(fn):
    """The `while True:` that IS the event loop.

    Selected by its `True` test, not by walk order: _executor_loop_pp contains
    three other `while`s, one of them (the Stage-5 drain) a SIBLING of the
    event loop, so relying on ast.walk ordering would silently point the guard
    at the wrong loop if the body were ever reordered.
    """
    import ast

    candidates = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.While)
        and isinstance(node.test, ast.Constant)
        and node.test.value is True
    ]
    assert len(candidates) == 1, (
        f"{fn.name}: expected exactly one `while True:` (the event loop), "
        f"found {len(candidates)} at lines {[c.lineno for c in candidates]}"
    )
    return candidates[0]


def _loop_terminating_breaks(loop):
    """(block, index, break_node) for every break that exits ``loop`` itself.

    Recurses through if/try/with, but stops at nested for/while/def: a break
    inside those binds to the inner construct, not to the event loop.
    """
    import ast

    found = []

    def visit(block):
        for i, stmt in enumerate(block):
            if isinstance(stmt, ast.Break):
                found.append((block, i, stmt))
            elif isinstance(
                stmt, (ast.For, ast.AsyncFor, ast.While, ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                continue  # binds to the inner construct
            else:
                for field in ("body", "orelse", "finalbody", "handlers"):
                    inner = getattr(stmt, field, None)
                    if isinstance(inner, list):
                        if field == "handlers":
                            for h in inner:
                                visit(h.body)
                        else:
                            visit(inner)

    visit(loop.body)
    return found


def _sets_sentinel_true(stmt):
    """Exactly `self._event_loop_completed = True` -- object and value both checked."""
    import ast

    if not isinstance(stmt, ast.Assign):
        return False
    if not (isinstance(stmt.value, ast.Constant) and stmt.value.value is True):
        return False
    return any(
        isinstance(t, ast.Attribute)
        and t.attr == "_event_loop_completed"
        and isinstance(t.value, ast.Name)
        and t.value.id == "self"
        for t in stmt.targets
    )


def test_loop_terminating_break_sets_the_completion_sentinel():
    """Only the break that exits the OUTER `while True` terminates the event loop.

    Deliberately not "every break": these loops contain inner `for` loops, and
    an inner break does not end the event loop. Demanding the sentinel there
    would instruct a contributor to set it while the loop is still running,
    which makes every later rank-local crash look like a clean shutdown and
    silently disables the kill -- the same class of bug this predicate has
    already regressed into twice.
    """

    loops = _executor_loop_ast_nodes()
    assert set(loops) == {"_executor_loop", "_executor_loop_pp", "_executor_loop_overlap"}, (
        f"executor loops renamed or removed: {sorted(loops)}"
    )

    for name, fn in loops.items():
        outer = _outer_while(fn)
        assert outer is not None, f"{name}: no `while` loop found -- did the loop shape change?"

        terminating = _loop_terminating_breaks(outer)
        assert len(terminating) == 1, (
            f"{name}: expected exactly 1 loop-terminating break, found "
            f"{len(terminating)} at lines {[b.lineno for _, _, b in terminating]}. "
            "A new normal-exit path must also set self._event_loop_completed = True."
        )

        block, idx, brk = terminating[0]
        prev = block[idx - 1] if idx else None
        assert _sets_sentinel_true(prev), (
            f"{name}: the loop-terminating `break` at line {brk.lineno} is not "
            "preceded by `self._event_loop_completed = True`. Without it "
            "_event_loop_wrapper treats a clean shutdown as a peer-stranding "
            "crash and SIGKILLs the job. (Inner-loop breaks must NOT set it.)"
        )


def test_completion_sentinel_is_reset_per_event_loop_run():
    """The reset in _event_loop_wrapper is load-bearing, not redundant with __init__.

    PyExecutor outlives a single loop run; without the reset a second run
    starts with the sentinel left True by the first, so a genuine crash in it
    is misread as a clean shutdown and no kill is armed.
    """
    import inspect

    from tensorrt_llm._torch.pyexecutor import py_executor as pe

    assert "self._event_loop_completed = False" in inspect.getsource(
        pe.PyExecutor._event_loop_wrapper
    )


def test_second_loop_run_still_kills_after_a_clean_first_run(monkeypatch):
    """Behavioral guard for the reset above: run clean, then crash, on ONE executor."""
    from tensorrt_llm._torch.pyexecutor import py_executor as pe

    events = []
    _stub_kill_paths(pe, monkeypatch, events)
    ex = _bare_executor(pe, monkeypatch, world_size=4)
    ex._executor_loop_cleanup = lambda: events.append("cleanup")

    # Run 1: reaches the normal-exit break, leaving the sentinel True.
    def clean():
        ex._event_loop_completed = True

    ex.event_loop = clean
    ex._event_loop_wrapper()
    assert events == ["cleanup"]
    assert ex._event_loop_completed is True

    # Run 2 on the SAME executor: a genuine crash must still arm the kill.
    events.clear()
    ex.event_loop = lambda: (_ for _ in ()).throw(ValueError("boom"))
    with pytest.raises(ValueError, match="boom"):
        ex._event_loop_wrapper()

    assert events == [("watchdog", 4), "cleanup", "cancel", ("kill", 4, 1234.5)]


# ---------------------------------------------------------------------------
# The delivery gate (review: symmetric crashes must not become exit 137).
# ---------------------------------------------------------------------------


def test_kill_is_skipped_once_the_error_reached_the_client(monkeypatch):
    """A reportable crash must not be converted into a bare exit 137.

    `crashed` means "the loop raised before its break", which is broader than
    "peers are stranded". In a symmetric crash every rank raises, nobody is
    stranded, and every rank arms this kill. If the stashed error already
    surfaced to the client the failure is diagnosable, so killing the world
    only destroys N tracebacks.
    """
    calls = []
    monkeypatch.setattr(
        hang_detector_module, "propagate_hard_kill", lambda *a, **k: calls.append(1)
    )
    monkeypatch.setenv(RANK_CRASH_KILL_GRACE_ENV, "0")

    delivered = threading.Event()
    delivered.set()
    fired = hard_kill_on_rank_crash(4, error_delivered=delivered)

    assert fired is False, "kill fired despite the error having been delivered"
    assert calls == [], "propagate_hard_kill must not run once the error is reportable"


def test_kill_still_fires_when_nothing_consumed_the_error(monkeypatch):
    """The stranded-peer case is unchanged: nothing consumes it, so kill."""
    calls = []
    monkeypatch.setattr(
        hang_detector_module, "propagate_hard_kill", lambda *a, **k: calls.append(1)
    )
    monkeypatch.setenv(RANK_CRASH_KILL_GRACE_ENV, "0")

    fired = hard_kill_on_rank_crash(4, error_delivered=threading.Event())

    assert fired is True
    assert calls == [1]


def test_kill_fires_when_no_delivery_event_is_supplied(monkeypatch):
    """Back-compat: callers that pass nothing get the old behaviour."""
    calls = []
    monkeypatch.setattr(
        hang_detector_module, "propagate_hard_kill", lambda *a, **k: calls.append(1)
    )
    monkeypatch.setenv(RANK_CRASH_KILL_GRACE_ENV, "0")

    assert hard_kill_on_rank_crash(4) is True
    assert calls == [1]


def test_delivery_is_checked_after_the_grace_not_before(monkeypatch):
    """The check must come after the wait, else it defeats its own purpose.

    The grace exists so the error can reach the client. Sampling the flag
    before waiting would read it while it is still False and kill anyway.
    """
    calls = []
    monkeypatch.setattr(
        hang_detector_module, "propagate_hard_kill", lambda *a, **k: calls.append(1)
    )
    monkeypatch.setenv(RANK_CRASH_KILL_GRACE_ENV, "0.5")

    delivered = threading.Event()

    def deliver_during_grace():
        time.sleep(0.15)
        delivered.set()

    t = threading.Thread(target=deliver_during_grace, daemon=True)
    t.start()
    fired = hard_kill_on_rank_crash(4, error_delivered=delivered)
    t.join(timeout=5)

    assert fired is False, "the flag was set during the grace window; the kill must observe it"
    assert calls == []


def test_watchdog_threads_the_delivery_event_through(monkeypatch):
    calls = []
    monkeypatch.setattr(
        hang_detector_module, "propagate_hard_kill", lambda *a, **k: calls.append(1)
    )
    monkeypatch.setenv(RANK_CRASH_KILL_GRACE_ENV, "0")

    delivered = threading.Event()
    delivered.set()
    wd = start_rank_crash_kill_watchdog(4, error_delivered=delivered)
    assert wd is not None
    wd.join(timeout=5)

    assert calls == [], "watchdog killed despite a delivered error"


# ---------------------------------------------------------------------------
# Real 2-rank MPI: the kill and the gate, with propagate_hard_kill NOT mocked.
#
# Every other kill-path test in this file monkeypatches propagate_hard_kill,
# so none of them exercises a real MPI_Abort (raised in review by @BowenFu).
# These two do: they launch a real 2-rank MPI job and assert on the exit
# status of the whole job.
#
# Scope, stated honestly: this proves the KILL MECHANISM and the delivery
# gate over a real communicator. It is not a full 2-rank LLM crash -- there
# is no engine here -- so it does not by itself prove the end-to-end claim
# that a client sees the original exception. It does close the "nothing
# exercises a real MPI_Abort" gap.
# ---------------------------------------------------------------------------

_MPI_2RANK_SCRIPT = """
import os, sys, time
from mpi4py import MPI
from tensorrt_llm._torch.pyexecutor.hang_detector import hard_kill_on_rank_crash

comm = MPI.COMM_WORLD
comm.Barrier()                      # both ranks up, imports done
# Printed only once imports and MPI init have succeeded. The assertions
# require it, so a setup failure (bad import, no MPI) cannot masquerade as
# a successful abort just by exiting non-zero.
if comm.Get_rank() == 0:
    print("RANK0_READY", flush=True)

if comm.Get_rank() == 0:
    import threading
    delivered = threading.Event()
    if os.environ["DELIVERED"] == "1":
        delivered.set()
    hard_kill_on_rank_crash(comm.Get_size(), error_delivered=delivered)
    # Only reached when the kill is skipped.
    print("RANK0_SURVIVED", flush=True)
else:
    # A peer that would otherwise sit in a collective forever.
    time.sleep(20)
    print("RANK1_SURVIVED", flush=True)

comm.Barrier()
sys.exit(0)
"""


def _run_two_rank(delivered: str):
    env = {
        **os.environ,
        "DELIVERED": delivered,
        hang_detector_module.RANK_CRASH_KILL_GRACE_ENV: "0",
    }
    return subprocess.run(
        ["mpirun", "--allow-run-as-root", "-n", "2", sys.executable, "-c", _MPI_2RANK_SCRIPT],
        env=env,
        timeout=600,
        capture_output=True,
    )


@pytest.mark.skipif(shutil.which("mpirun") is None, reason="mpirun not available")
def test_real_mpi_abort_takes_down_both_ranks():
    """Undelivered crash: the abort must reach the peer, not just rank 0.

    Cross-rank propagation is the load-bearing part of the whole feature. If
    MPI_Abort only killed rank 0, the peer would still burn to its own
    HangDetector -- exactly the failure this exists to prevent.
    """
    proc = _run_two_rank(delivered="0")
    out = (proc.stdout + proc.stderr).decode(errors="replace")

    assert "RANK0_READY" in out, (
        f"the job never reached the kill call -- this is a setup failure, not "
        f"an abort, and must not be read as a pass; out={out[-1500:]}"
    )
    assert proc.returncode != 0, f"job survived an undelivered crash kill; out={out[-800:]}"
    assert "RANK1_SURVIVED" not in out, (
        f"peer rank outlived the abort -- propagation failed; out={out[-800:]}"
    )


@pytest.mark.skipif(shutil.which("mpirun") is None, reason="mpirun not available")
def test_real_mpi_job_survives_when_the_error_was_delivered():
    """Delivered crash: no abort, so both ranks run to completion.

    This is the review point -- a symmetric crash whose error already reached
    the client must not have its tracebacks replaced by exit 137.
    """
    proc = _run_two_rank(delivered="1")
    out = (proc.stdout + proc.stderr).decode(errors="replace")

    assert "RANK0_READY" in out, (
        f"the job never reached the kill call -- setup failure; out={out[-1500:]}"
    )
    assert proc.returncode == 0, f"job died despite a delivered error; out={out[-800:]}"
    assert "RANK0_SURVIVED" in out, f"rank 0 was killed anyway; out={out[-800:]}"
    assert "RANK1_SURVIVED" in out, f"peer was killed anyway; out={out[-800:]}"
