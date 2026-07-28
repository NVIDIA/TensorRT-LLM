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
    monkeypatch.setattr(hd_module, "propagate_hard_kill", lambda: kills.append(1))
    assert hard_kill_on_rank_crash(world_size=1) is False
    assert kills == []


def test_rank_crash_kill_fires_for_multi_rank(monkeypatch):
    kills = []
    monkeypatch.setattr(hd_module, "propagate_hard_kill", lambda: kills.append(1))
    monkeypatch.setenv(RANK_CRASH_KILL_GRACE_ENV, "0")
    assert hard_kill_on_rank_crash(world_size=4) is True
    assert kills == [1]


def test_rank_crash_kill_sleeps_grace_before_kill(monkeypatch):
    """The grace must elapse BEFORE the kill so cleaner error paths win the race."""
    order = []
    monkeypatch.setattr(hd_module, "propagate_hard_kill", lambda: order.append("kill"))
    monkeypatch.setattr(hang_detector_module.time, "sleep", lambda s: order.append(("sleep", s)))
    monkeypatch.setenv(RANK_CRASH_KILL_GRACE_ENV, "2.5")
    assert hard_kill_on_rank_crash(world_size=2) is True
    assert order == [("sleep", 2.5), "kill"]


def test_rank_crash_kill_disabled_by_negative_grace(monkeypatch):
    kills = []
    monkeypatch.setattr(hd_module, "propagate_hard_kill", lambda: kills.append(1))
    monkeypatch.setenv(RANK_CRASH_KILL_GRACE_ENV, "-1")
    assert hard_kill_on_rank_crash(world_size=8) is False
    assert kills == []


def test_rank_crash_kill_invalid_grace_uses_default(monkeypatch):
    """A malformed env value must not disable the kill (fail-safe default)."""
    order = []
    monkeypatch.setattr(hd_module, "propagate_hard_kill", lambda: order.append("kill"))
    monkeypatch.setattr(hang_detector_module.time, "sleep", lambda s: order.append(("sleep", s)))
    monkeypatch.setenv(RANK_CRASH_KILL_GRACE_ENV, "bogus")
    assert hard_kill_on_rank_crash(world_size=2) is True
    assert order == [("sleep", 10.0), "kill"]


def test_rank_crash_kill_never_raises(monkeypatch):
    """It runs in a `finally`: raising would mask the loop's original error."""

    def boom():
        raise RuntimeError("abort machinery broken")

    monkeypatch.setattr(hd_module, "propagate_hard_kill", boom)
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
    monkeypatch.setattr(hd_module, "propagate_hard_kill", killed.set)
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
    monkeypatch.setattr(hd_module, "propagate_hard_kill", lambda: kills.append(1))
    monkeypatch.setenv(RANK_CRASH_KILL_GRACE_ENV, "0")
    assert start_rank_crash_kill_watchdog(world_size=1) is None
    assert kills == []


def test_watchdog_not_armed_when_disabled(monkeypatch):
    kills = []
    monkeypatch.setattr(hd_module, "propagate_hard_kill", lambda: kills.append(1))
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
    monkeypatch.setattr(hd_module, "propagate_hard_kill", lambda: kills.append(1))
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
    watchdog = hd_module.RankCrashKillWatchdog(world_size=2, grace=5.0)
    assert watchdog.deadline == pytest.approx(time.monotonic() + 5.0, abs=0.5)


def test_kill_keeps_original_deadline_on_handover(monkeypatch):
    """Handing the kill over must not restart the grace clock.

    The caller cancels the watchdog once cleanup returns and carries the kill
    itself; passing the watchdog's deadline keeps the kill at crash + grace
    instead of crash + 2*grace. Uses a real (short) sleep rather than patching
    time.sleep process-wide, so no background thread can busy-spin into the
    assertion.
    """
    kills = []
    monkeypatch.setattr(hd_module, "propagate_hard_kill", lambda: kills.append(1))
    monkeypatch.setenv(RANK_CRASH_KILL_GRACE_ENV, "30")

    # Nearly all of the 30s grace has already been burned by the watchdog.
    t0 = time.monotonic()
    assert hard_kill_on_rank_crash(world_size=2, deadline=t0 + 0.3) is True
    elapsed = time.monotonic() - t0
    assert kills == [1]
    # Slept out the REMAINING 0.3s, not a fresh 30s grace.
    assert elapsed == pytest.approx(0.3, abs=0.25)


def test_kill_fires_immediately_when_deadline_already_passed(monkeypatch):
    """A deadline in the past must fire now, not raise into the blanket except.

    Without the max(0.0, ...) clamp this sleeps a negative duration, raises,
    and hard_kill_on_rank_crash returns False -- silently skipping the kill in
    exactly the case the watchdog exists for (cleanup outlasted the grace).
    """
    kills = []
    monkeypatch.setattr(hd_module, "propagate_hard_kill", lambda: kills.append(1))
    monkeypatch.setenv(RANK_CRASH_KILL_GRACE_ENV, "30")

    t0 = time.monotonic()
    assert hard_kill_on_rank_crash(world_size=2, deadline=t0 - 100.0) is True
    assert kills == [1]
    assert time.monotonic() - t0 < 1.0


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


def _stub_kill_paths(pe, monkeypatch, events, arm_watchdog=True):
    monkeypatch.setattr(
        pe,
        "hard_kill_on_rank_crash",
        lambda world_size, deadline=None: events.append(("kill", world_size, deadline)),
    )
    watchdogs = []

    def _start(world_size):
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


def test_every_executor_loop_break_sets_the_completion_sentinel():
    import ast

    loops = _executor_loop_ast_nodes()
    assert set(loops) == {"_executor_loop", "_executor_loop_pp", "_executor_loop_overlap"}, (
        f"executor loops renamed or removed: {sorted(loops)}"
    )

    for name, fn in loops.items():
        # Only breaks belonging to THIS function (not to a nested def) end
        # the event loop.
        nested = {
            id(n)
            for d in ast.walk(fn)
            if isinstance(d, (ast.FunctionDef, ast.AsyncFunctionDef)) and d is not fn
            for n in ast.walk(d)
        }
        own_breaks = [n for n in ast.walk(fn) if isinstance(n, ast.Break) and id(n) not in nested]
        assert own_breaks, f"{name}: no break found -- did the loop exit change?"

        checked = 0
        for parent in ast.walk(fn):
            for field in ("body", "orelse", "finalbody"):
                block = getattr(parent, field, None)
                if not isinstance(block, list):
                    continue
                for i, stmt in enumerate(block):
                    if not isinstance(stmt, ast.Break) or id(stmt) in nested:
                        continue
                    checked += 1
                    prev = block[i - 1] if i else None
                    sets_sentinel = isinstance(prev, ast.Assign) and any(
                        isinstance(t, ast.Attribute) and t.attr == "_event_loop_completed"
                        for t in prev.targets
                    )
                    assert sets_sentinel, (
                        f"{name}: the `break` at line {stmt.lineno} is not preceded "
                        "by `self._event_loop_completed = True`. Every normal exit "
                        "must set it, or _event_loop_wrapper treats a clean "
                        "shutdown as a peer-stranding crash and SIGKILLs the job."
                    )
        assert checked == len(own_breaks)


def test_completion_sentinel_is_initialized_false():
    import inspect

    from tensorrt_llm._torch.pyexecutor import py_executor as pe

    assert "self._event_loop_completed = False" in inspect.getsource(pe.PyExecutor.__init__)
