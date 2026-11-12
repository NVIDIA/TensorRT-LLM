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
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.pyexecutor import profiling
from tensorrt_llm._torch.pyexecutor.profiling import PyExecutorProfileManager
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm.metrics.perf import PerfMetricsCollector


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
    # The profile-state machine lives in PyExecutorProfileManager. The
    # real PyExecutor.__init__ wires this up; replicate that here for
    # the bare-bones test double so ``executor.start_profile``,
    # ``executor.stop_profile``, etc. delegate correctly.
    executor._perf_metrics = PerfMetricsCollector()
    executor._profile_manager = PyExecutorProfileManager(executor)
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


def test_stop_profile_cancels_pending_start_before_firing(tmp_path):
    """If ``start_profile()`` was scheduled but the engine has not yet
    reached ``start_iter`` (idle server), ``stop_profile()`` must remove
    the pending start so profiling does not silently begin later."""
    executor = _bare_executor()
    executor.iter_counter = 3

    # Simulate the full ``start_profile`` path for a bare executor: the
    # broadcast apply is what populates ``profile_start_iters``.
    executor.start_profile(output_dir=str(tmp_path), num_steps=50)
    executor._apply_profile_start_config(
        {
            "output_dir": str(tmp_path),
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


def test_start_profile_rejects_num_steps_zero(tmp_path):
    """``num_steps == 0`` would make ``stop_iter == start_iter``;
    ``profile_step()`` would discard the stop marker as stale and the
    profile window would run forever. The Pydantic schema rejects this
    at the HTTP layer, but programmatic callers must also see a clean
    error rather than a silently-stuck profile window.
    """
    import pytest as _pytest

    executor = _bare_executor()
    executor.iter_counter = 5

    with _pytest.raises(ValueError, match="num_steps must be >= 1"):
        executor.start_profile(output_dir=str(tmp_path), num_steps=0)

    # State must remain pristine — no pending markers, no scheduled
    # iterations — because the request was rejected before any side
    # effect happened.
    assert executor._runtime_profile_pending_start_iter is None
    assert not executor.profile_start_iters
    assert not executor.profile_stop_iters


def test_start_profile_rejects_num_steps_negative(tmp_path):
    """Negative ``num_steps`` is also rejected for the same reason."""
    import pytest as _pytest

    executor = _bare_executor()
    executor.iter_counter = 5

    with _pytest.raises(ValueError, match="num_steps must be >= 1"):
        executor.start_profile(output_dir=str(tmp_path), num_steps=-3)

    assert executor._runtime_profile_pending_start_iter is None


def test_concurrent_start_profile_admits_exactly_one_caller(tmp_path):
    """Only one of N concurrent ``start_profile()`` calls may win.

    The re-entrancy guard reads ``_profile_enabled`` and
    ``_runtime_profile_pending_start_iter`` and then claims the pending
    marker. If that check-then-set is not atomic, two callers both pass
    the guard and both broadcast a profile start, and the second
    ``_apply_profile_start_config`` overwrites the trace path and
    activities of the first window.

    This is reachable in production: ``OpenAIServer.start_profile``
    dispatches through ``asyncio.to_thread``, so two concurrent
    ``POST /start_profile`` requests run on two different worker
    threads, and the in-process / Ray paths call
    ``PyExecutor.start_profile`` directly without the single-owner
    thread that ``GenerationExecutorProxy`` interposes.
    """
    from tensorrt_llm.executor.utils import RequestError

    class _RecordingQueue:
        """Stand-in for ``executor_request_queue`` that counts broadcasts."""

        def __init__(self):
            self.start_configs = []
            self._lock = threading.Lock()

        def enqueue_profile_start_request(self, config):
            with self._lock:
                self.start_configs.append(config)

    num_threads = 8
    # The unguarded window is a couple of bytecodes wide, so repeat the
    # race rather than relying on a single interleaving. The pre-fix
    # code fails this within the first few trials.
    for trial in range(50):
        executor = _bare_executor()
        executor.iter_counter = 7
        queue = _RecordingQueue()
        executor.executor_request_queue = queue

        barrier = threading.Barrier(num_threads)
        accepted = []
        rejected = []
        accepted_lock = threading.Lock()

        def _worker():
            # Release all threads into the guard at the same moment.
            barrier.wait()
            try:
                executor.start_profile(output_dir=str(tmp_path), num_steps=100)
            except RequestError as e:
                with accepted_lock:
                    rejected.append(e)
            else:
                with accepted_lock:
                    accepted.append(True)

        threads = [threading.Thread(target=_worker) for _ in range(num_threads)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

        assert not any(t.is_alive() for t in threads), f"trial {trial}: start_profile deadlocked"
        assert len(accepted) == 1, (
            f"trial {trial}: {len(accepted)} callers were admitted into the "
            f"profile window, expected exactly 1 "
            f"({len(rejected)} rejected out of {num_threads})"
        )
        assert len(rejected) == num_threads - 1
        assert len(queue.start_configs) == 1, (
            f"trial {trial}: {len(queue.start_configs)} profile-start "
            "broadcasts were enqueued, expected exactly 1"
        )


def test_start_profile_rolls_back_claim_when_broadcast_fails(tmp_path):
    """A failed broadcast must not leave the window claimed.

    Otherwise the engine is wedged: every later ``start_profile()`` is
    rejected as "already pending" even though nothing is running and
    there is nothing for ``stop_profile()`` to cancel on the ranks that
    never saw a broadcast.
    """
    import pytest as _pytest

    class _FailingQueue:
        def enqueue_profile_start_request(self, config):
            raise RuntimeError("zmq send failed")

    executor = _bare_executor()
    executor.iter_counter = 7
    executor.executor_request_queue = _FailingQueue()

    with _pytest.raises(RuntimeError, match="failed to enqueue profile-start"):
        executor.start_profile(output_dir=str(tmp_path), num_steps=100)

    assert executor._runtime_profile_pending_start_iter is None
    assert executor._runtime_profile_activities is None

    # The next attempt must be admitted rather than rejected as pending.
    executor.executor_request_queue = None
    executor.start_profile(output_dir=str(tmp_path), num_steps=100)
    assert executor._runtime_profile_pending_start_iter is not None


def test_apply_profile_stop_config_reports_cancel_vs_schedule(tmp_path):
    """``_apply_profile_stop_config`` tells the caller which path it took.

    ``stop_profile()`` uses the return value to decide whether to wait
    for the executor loop to flush a trace. Cancelling a start that
    never fired produces no trace, so waiting would just burn the
    30s timeout.
    """
    # Cancel path: a pending start that has not fired yet.
    executor = _bare_executor()
    executor.iter_counter = 7
    executor._apply_profile_start_config(
        {
            "output_dir": str(tmp_path),
            "start_step": 0,
            "num_steps": 100,
        }
    )
    assert executor._runtime_profile_pending_start_iter == 8

    assert executor._apply_profile_stop_config() is True
    assert executor._runtime_profile_pending_start_iter is None
    assert executor._runtime_profile_pending_stop_iter is None
    assert 8 not in executor.profile_start_iters

    # Schedule path: the window is already active on the executor loop.
    executor = _bare_executor()
    executor.iter_counter = 7
    executor._profile_enabled = True

    assert executor._apply_profile_stop_config() is False
    assert executor._runtime_profile_pending_stop_iter == 8
    assert 8 in executor.profile_stop_iters


@pytest.mark.cpu_only
def test_rebind_resets_phase_metrics_without_stale_profile_reference() -> None:
    executor = _bare_executor()
    executor._closed = False
    executor._resources_released = True
    executor._event_loop_error = None
    executor._fatal_error = None
    collector = executor._perf_metrics
    collector.collect_rank({"kvCacheStats": {"maxNumBlocks": 10}}, capture_iter=8)
    values = {"iter": 8, "rank": 0, "globalRank": 0, "numFetchedRequestsTotal": 20}
    collector.collect_loop(values)
    executor._initialize_execution = Mock()
    executor.rebind_resources(Mock())
    assert executor._profile_manager.metrics is collector
    assert collector.latest_loop is None
    assert not collector.latest_rank
    new = collector.collect_loop({**values, "iter": 0, "numFetchedRequestsTotal": 2})
    assert new.per_iter["numFetchedRequests"] == 2
    assert new.per_iter["loopSequence"] == 0
    assert not new.per_rank
    executor._initialize_execution.assert_called_once()


@pytest.mark.cpu_only
@pytest.mark.parametrize("rank,stats_enabled", [(0, False), (1, False), (0, True), (1, True)])
def test_profile_step_collects_with_source_and_logging_gates(
    monkeypatch: pytest.MonkeyPatch, rank: int, stats_enabled: bool
) -> None:
    executor = _bare_executor()
    executor.dist = SimpleNamespace(rank=rank, pp_size=1)
    executor.model_engine = SimpleNamespace(model=None, iter_states="decode")
    executor.print_log = True
    executor.enable_iter_perf_stats = stats_enabled
    executor.is_warmup = False
    executor.num_scheduled_requests = 0
    executor.num_fetch_requests_cur_rank = 2
    executor.num_fetch_requests = 2
    executor.active_requests = []
    executor.num_dummy_requests = 0
    executor.num_unscheduled_requests = 0
    executor._iter_adp_dummy_ctx_tokens = 0
    executor._iter_adp_dummy_gen_tokens = 0
    reader = Mock(return_value=SimpleNamespace(max_num_blocks=10, free_num_blocks=4))
    executor.kv_cache_manager = SimpleNamespace(get_kv_cache_stats=reader)
    monkeypatch.setenv(profiling.PROFILE_LOG_RANKS_ENV_VAR_NAME, "0")
    monkeypatch.delenv(profiling.PROFILE_START_STOP_ENV_VAR_NAME, raising=False)
    monkeypatch.setattr(profiling.torch.cuda, "Event", Mock())
    monkeypatch.setattr(profiling, "get_calibrator", Mock(return_value=Mock()))
    monkeypatch.setattr(
        profiling,
        "runtime_hooks_of",
        Mock(return_value=SimpleNamespace(iteration_timing_one_event=False)),
    )
    messages = []
    executor._profile_manager.metrics_logger._write = messages.append
    with executor._profile_manager.profile_step() as step:
        step()
        executor.iter_counter = 1
        step()
    assert executor._perf_metrics.latest_loop is not None
    if stats_enabled:
        import json

        from tensorrt_llm.executor.base_worker import BaseWorker

        executor.enable_attention_dp = False
        executor.disable_overlap_scheduler = True
        executor._latest_kv_iter_stats = None
        executor.stats = []
        executor.stats_lock = threading.Lock()
        executor.max_stats_len = 10
        native_stats = Mock()
        native_stats.to_json_str.return_value = '{"iter": 0}'
        executor._append_iter_stats(native_stats)
        wire = json.loads(BaseWorker._stats_serializer(executor.stats[0]))
        assert wire["executorLoop"]["per_iter"]["iter"] == 1
        assert wire["iter"] == 0
    assert reader.call_count == int(rank == 0 and not stats_enabled)
    assert any("iter = " in message for message in messages) == (rank == 0)
