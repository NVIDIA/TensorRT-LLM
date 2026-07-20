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

import contextlib
import os
import signal
import subprocess
import sys
import time
import types

import pytest

from tensorrt_llm._torch.pyexecutor import hang_detector as hd_module
from tensorrt_llm._torch.pyexecutor.hang_detector import (
    RANK_CRASH_KILL_GRACE_ENV,
    HangDetector,
    hard_kill_on_rank_crash,
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
    monkeypatch.setattr(hd_module.time, "sleep", lambda s: order.append(("sleep", s)))
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
    monkeypatch.setattr(hd_module.time, "sleep", lambda s: order.append(("sleep", s)))
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
# Wiring: PyExecutor._event_loop_wrapper must invoke the kill on the crash
# path only, and only after local cleanup has woken rank-local waiters.
# --------------------------------------------------------------------------


def _bare_executor(pe, monkeypatch, world_size):
    # Neutralize the profiling/GC context managers: they are irrelevant to the
    # crash path and must not depend on env/GC state in a unit test.
    monkeypatch.setattr(pe, "host_profiler_context", lambda enable: contextlib.nullcontext())
    monkeypatch.setattr(pe, "customized_gc_thresholds", lambda threshold: contextlib.nullcontext())
    ex = pe.PyExecutor.__new__(pe.PyExecutor)
    ex.dist = types.SimpleNamespace(world_size=world_size)
    ex.garbage_collection_gen0_threshold = None
    return ex


def test_event_loop_wrapper_kills_world_on_crash(monkeypatch):
    from tensorrt_llm._torch.pyexecutor import py_executor as pe

    events = []
    monkeypatch.setattr(
        pe, "hard_kill_on_rank_crash", lambda world_size: events.append(("kill", world_size))
    )
    ex = _bare_executor(pe, monkeypatch, world_size=4)
    ex._executor_loop_cleanup = lambda: events.append("cleanup")

    def crash():
        raise ValueError("boom")

    ex.event_loop = crash

    with pytest.raises(ValueError, match="boom"):
        ex._event_loop_wrapper()

    # Cleanup wakes rank-local waiters (who read the stashed error) BEFORE
    # the world is torn down.
    assert events == ["cleanup", ("kill", 4)]
    assert isinstance(ex._event_loop_error, ValueError)


def test_event_loop_wrapper_no_kill_on_clean_exit(monkeypatch):
    from tensorrt_llm._torch.pyexecutor import py_executor as pe

    events = []
    monkeypatch.setattr(pe, "hard_kill_on_rank_crash", lambda world_size: events.append("kill"))
    ex = _bare_executor(pe, monkeypatch, world_size=4)
    ex._executor_loop_cleanup = lambda: events.append("cleanup")
    ex.event_loop = lambda: None

    ex._event_loop_wrapper()

    assert events == ["cleanup"]
