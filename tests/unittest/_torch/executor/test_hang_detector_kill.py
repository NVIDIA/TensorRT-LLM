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


def test_watchdog_cancel_prevents_the_kill(monkeypatch):
    """A cancelled watchdog must not SIGKILL a rank that goes on to exit cleanly.

    Without a cancel path an armed watchdog fires at crash + grace no matter
    what happens afterwards, turning a would-be exit 0 into exit 137.
    """
    kills = []
    monkeypatch.setattr(hd_module, "propagate_hard_kill", lambda: kills.append(1))
    monkeypatch.setenv(RANK_CRASH_KILL_GRACE_ENV, "30")

    watchdog = start_rank_crash_kill_watchdog(world_size=2)
    assert watchdog is not None
    try:
        watchdog.cancel()
        # Cancel must break the grace wait immediately, not merely be observed
        # after it elapses -- otherwise the process still dies 30s later.
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


def test_kill_keeps_original_deadline_on_handover(monkeypatch):
    """Handing the kill over must not restart the grace clock.

    The caller cancels the watchdog once cleanup returns and carries the kill
    itself; passing the watchdog's deadline keeps the kill at crash + grace
    instead of crash + 2*grace.
    """
    slept = []
    monkeypatch.setattr(hd_module, "propagate_hard_kill", lambda: slept.append("kill"))
    monkeypatch.setattr(hd_module.time, "sleep", lambda s: slept.append(round(s, 1)))
    monkeypatch.setenv(RANK_CRASH_KILL_GRACE_ENV, "10")

    # 6s of the 10s grace has already been burned by the watchdog.
    deadline = hd_module.time.monotonic() + 4.0
    assert hard_kill_on_rank_crash(world_size=2, deadline=deadline) is True
    assert slept == [4.0, "kill"]


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
    ex.is_shutdown = is_shutdown
    return ex


class _FakeWatchdog:
    """Stand-in for RankCrashKillWatchdog that records cancellation."""

    def __init__(self, events, world_size):
        self._events = events
        self.deadline = 1234.5
        self.cancelled = False
        events.append(("watchdog", world_size))

    def cancel(self):
        self.cancelled = True
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
# The kill must stay scoped to crashes that actually strand peers. A raise
# on the way out of an already-shut-down loop happens after every rank has
# processed the shutdown broadcast and all work is done: escalating it turns
# a benign teardown error into a whole-job SIGKILL (exit 137 instead of 0).
# --------------------------------------------------------------------------


def test_event_loop_wrapper_no_kill_when_loop_raises_after_shutdown(monkeypatch):
    from tensorrt_llm._torch.pyexecutor import py_executor as pe

    events = []
    _stub_kill_paths(pe, monkeypatch, events)
    ex = _bare_executor(pe, monkeypatch, world_size=4)
    ex._executor_loop_cleanup = lambda: events.append("cleanup")

    def late_raise():
        # The loop processed its shutdown request and drained all work, then
        # something raised on the way out (e.g. a context manager's __exit__).
        ex.is_shutdown = True
        raise RuntimeError("teardown hiccup")

    ex.event_loop = late_raise

    with pytest.raises(RuntimeError, match="teardown hiccup"):
        ex._event_loop_wrapper()

    # Logged and re-raised, but no watchdog and no kill: peers are not stranded.
    assert events == ["cleanup"]
    assert isinstance(ex._event_loop_error, RuntimeError)


def test_event_loop_wrapper_no_kill_when_enclosing_context_manager_raises(monkeypatch):
    """Teardown of the host-profiler / GC context managers is not a crash.

    They wrap event_loop() but are not part of it; a failure while unwinding
    them leaves no peer waiting on this rank.
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
    ex.event_loop = lambda: None

    with pytest.raises(RuntimeError, match="profiler teardown failed"):
        ex._event_loop_wrapper()

    assert events == ["cleanup"]
