# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``PyExecutor.start_profile`` / ``stop_profile``.

Covers the runtime profiling handshake used by ``trtllm-serve``'s
``/start_profile`` and ``/stop_profile`` HTTP endpoints:

* ``start_profile`` tracks its scheduled iteration indices so a
  subsequent ``stop_profile`` can either cancel them (if the engine
  never reached them) or schedule a stop cleanly.
* ``stop_profile`` does NOT call ``torch.profiler.stop()`` directly
  because torch.profiler / Kineto require start and stop to happen on
  the same thread. Instead it schedules the stop at the next executor
  iteration; the HTTP layer is responsible for tickling the engine so
  the scheduled iteration actually runs when the server is otherwise
  idle.

The tests construct a ``PyExecutor`` via ``__new__`` and populate only
the attributes the handlers touch, so they run without GPUs, models,
or an MPI/RPC environment.
"""

import threading

from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor


def _bare_executor():
    """Build a minimal ``PyExecutor`` with only the state the profile
    handlers depend on. Avoids the heavy real __init__."""
    executor = PyExecutor.__new__(PyExecutor)
    executor.iter_counter = 0
    executor.profile_start_iters = set()
    executor.profile_stop_iters = set()
    executor.global_rank = 0
    executor._runtime_profile_trace_path = None
    executor._runtime_profile_activities = None
    executor._runtime_profile_cuda_only = False
    executor._runtime_profile_pending_start_iter = None
    executor._runtime_profile_pending_stop_iter = None
    executor._profile_state_lock = threading.Lock()
    executor._profile_enabled = False
    return executor


def test_start_profile_marks_pending_on_caller_thread(tmp_path):
    """``start_profile()`` must set ``_runtime_profile_pending_start_iter``
    immediately on the caller thread so a concurrent second call is
    rejected by the guard even before the broadcast is applied. The
    actual ``profile_start_iters`` update happens on every rank via the
    broadcast path (``_apply_profile_start_config``) — see
    ``test_apply_profile_start_config_uses_local_iter_counter``.
    """
    executor = _bare_executor()
    executor.iter_counter = 7
    executor.start_profile(output_dir=str(tmp_path), num_steps=100)

    # Local apply only marks a sentinel for the rejection guard.
    assert executor._runtime_profile_pending_start_iter is not None


def test_apply_profile_start_config_uses_local_iter_counter(tmp_path):
    """The broadcasted profile-start applies on every rank based on
    its *own* ``iter_counter``. Two ranks that run in lockstep compute
    the same ``start_iter``, so both fire on the same executor
    iteration.
    """
    executor = _bare_executor()
    executor.iter_counter = 7
    executor._apply_profile_start_config(
        {
            "output_dir": str(tmp_path),
            "activities": ["CPU", "GPU"],
            "start_step": 0,
            "num_steps": 100,
        }
    )

    # start_iter = iter_counter + 1 + start_step
    assert 8 in executor.profile_start_iters
    assert 108 in executor.profile_stop_iters
    assert executor._runtime_profile_pending_start_iter == 8
    assert executor._runtime_profile_pending_stop_iter == 108
    trace_path = executor._runtime_profile_trace_path
    assert trace_path.startswith(str(tmp_path))
    assert "rank-0" in trace_path and trace_path.endswith(".json")


def test_apply_profile_start_config_no_num_steps_leaves_stop_open(tmp_path):
    executor = _bare_executor()
    executor.iter_counter = 2
    executor._apply_profile_start_config(
        {
            "output_dir": str(tmp_path),
            "start_step": 0,
        }
    )

    assert 3 in executor.profile_start_iters
    assert executor._runtime_profile_pending_start_iter == 3
    # No num_steps => no auto stop scheduled.
    assert executor._runtime_profile_pending_stop_iter is None
    assert 3 not in executor.profile_stop_iters


def test_stop_profile_cancels_pending_start_before_firing():
    """If ``start_profile()`` was scheduled but the engine has not yet
    reached ``start_iter`` (idle server), ``stop_profile()`` must remove
    the pending start so profiling does not silently begin later."""
    executor = _bare_executor()
    executor.iter_counter = 3

    # Simulate the full ``start_profile`` path for a bare executor: the
    # broadcast apply is what populates ``profile_start_iters``.
    executor.start_profile(output_dir="/tmp/unused", num_steps=50)
    executor._apply_profile_start_config(
        {
            "output_dir": "/tmp/unused",
            "start_step": 0,
            "num_steps": 50,
        }
    )
    assert executor._runtime_profile_pending_start_iter == 4
    assert 4 in executor.profile_start_iters
    assert 54 in executor.profile_stop_iters

    # Engine has not iterated yet.
    assert executor._profile_enabled is False

    executor.stop_profile()

    assert 4 not in executor.profile_start_iters
    assert 54 not in executor.profile_stop_iters
    assert executor._runtime_profile_pending_start_iter is None
    assert executor._runtime_profile_pending_stop_iter is None


def test_stop_profile_schedules_next_iter_when_active():
    """While a profile window is live (``_profile_enabled`` is True) the
    stop call must schedule the next iteration for the in-loop flush;
    it must not attempt to call torch.profiler from this thread.

    ``stop_profile`` blocks until the executor loop clears
    ``_profile_enabled``; we simulate that on a side-thread so the test
    does not hit the 30s poll timeout.
    """
    import threading as _threading

    executor = _bare_executor()
    executor.iter_counter = 100
    executor._profile_enabled = True  # Simulate a running profile.

    def _simulate_in_loop_stop():
        # Mimic what profile_step() does when it reaches the scheduled
        # stop iteration: clears _profile_enabled so the caller's poll
        # loop in stop_profile() returns promptly.
        import time as _time

        _time.sleep(0.05)
        with executor._profile_state_lock:
            executor._profile_enabled = False

    sim = _threading.Thread(target=_simulate_in_loop_stop, daemon=True)
    sim.start()
    executor.stop_profile()
    sim.join(timeout=1.0)

    # stop_iter = iter_counter + 1 so the NEXT profile_step check (on
    # main's ``iter_counter in profile_stop_iters`` semantics) fires the
    # stop on the very next loop iteration whether stop_profile was
    # called mid-body or between iterations.
    assert 101 in executor.profile_stop_iters
    assert executor._runtime_profile_pending_stop_iter == 101


def test_stop_profile_without_pending_start_falls_back_to_iteration_stop():
    """With no runtime ``start_profile()`` call pending and no active
    profile, ``stop_profile()`` must still schedule a stop so env-var
    driven windows (``TLLM_PROFILE_START_STOP``) can still be torn down
    on the next iteration."""
    executor = _bare_executor()
    executor.iter_counter = 42

    executor.stop_profile()

    assert 43 in executor.profile_stop_iters
    assert executor._runtime_profile_pending_start_iter is None
    # Fallback path publishes the stop iteration too.
    assert executor._runtime_profile_pending_stop_iter == 43


def test_stop_profile_after_start_without_num_steps_cancels_pending(tmp_path):
    """When ``start_profile()`` was called without ``num_steps`` and
    the engine never iterated, ``stop_profile()`` must cancel the
    pending start rather than scheduling a stop."""
    executor = _bare_executor()
    executor.iter_counter = 10

    executor.start_profile(output_dir=str(tmp_path))
    executor._apply_profile_start_config(
        {
            "output_dir": str(tmp_path),
            "start_step": 0,
        }
    )
    assert 11 in executor.profile_start_iters
    assert executor._runtime_profile_pending_start_iter == 11

    executor.stop_profile()

    assert 11 not in executor.profile_start_iters
    assert executor._runtime_profile_pending_start_iter is None
