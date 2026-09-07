# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-iteration transcript of communication boundaries in each executor loop.

Each test drives one idle iteration (nothing schedulable) followed by shutdown
through a real loop body on a single rank and pins the sequence of coordinator
calls plus the executor-owned ADP-synchronized flushes. This protects the call
points against being dropped or reordered while disagg logic moves into the
coordinator; a changed sequence is a review signal, not necessarily a bug.

Rank symmetry is checked only in the narrow form that fits one process: the
first and a non-first PP rank must issue the same collective-sensitive calls in
the same order during an idle iteration. Real multi-rank blocking semantics are
covered elsewhere (Gloo tests; FakeDist arrives with CS-2). Regular disagg PP
termination advances from executed-batch handling; a recompute-pause fallback
can call the same termination handler from an idle iteration. Neither path is
covered here (nothing is pending in these iterations); both belong to the
executed-batch/lifecycle transcripts of PR-5.
"""

import inspect
import queue
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from tensorrt_llm._torch.disaggregation.executor.coordinator import (
    DisaggTransferCoordinator,
    NoopDisaggCoordinator,
)
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import (
    ScheduledRequests,
    SerializableSchedulerOutput,
)

pytestmark = pytest.mark.cpu_only

# Coordinator entry points whose delegates run a rank-consensus collective.
# Derived from the delegate targets in PyExecutor._build_disagg_coordinator;
# update alongside them.
_COLLECTIVE_COORDINATOR_CALLS = {
    "handle_errors_synced",  # dist.allreduce / tp_allgather under ADP
    "prepare_context_schedulable",  # transceiver.prepare_context_requests consensus
    "poll_gen_transfers",  # gen transfer status consensus
    "poll_progress_when_idle",  # ctx transfer status consensus
    "receive_gen_init",  # async receive polls gen status consensus
    "reap_context_sends",  # ctx transfer status consensus
}
# Executor-owned per-iteration collectives that must stay in lockstep with the
# coordinator calls under ADP.
_COLLECTIVE_EXECUTOR_EVENTS = {
    "flush_pending_transfer_responses",  # tp_gather in _enqueue_responses
    "handle_kv_transfer_timeouts_synced",  # tp_allgather
}
_COLLECTIVE_SENSITIVE = _COLLECTIVE_COORDINATOR_CALLS | _COLLECTIVE_EXECUTOR_EVENTS


def _entry_points() -> list:
    return sorted(
        name
        for name, member in inspect.getmembers(
            DisaggTransferCoordinator, predicate=inspect.isfunction
        )
        if not name.startswith("_")
    )


def _recording_coordinator(calls: list) -> DisaggTransferCoordinator:
    coordinator = NoopDisaggCoordinator()
    for name in _entry_points():

        def record(*args, _name=name):
            calls.append((_name, *args))

        setattr(coordinator, name, record)

    def admit(fitting):
        calls.append(("admit", fitting))
        return fitting, False

    coordinator.admit = admit
    return coordinator


def _collective_calls(calls: list) -> list:
    return [call for call in calls if call[0] in _COLLECTIVE_SENSITIVE]


def _idle_executor(monkeypatch, calls: list) -> PyExecutor:
    """Bare executor whose first iteration schedules nothing and whose second
    iteration observes shutdown. Executor-owned helpers are stubbed; the loop
    bodies and the communication boundaries are real call sites."""
    for target in ("torch.cuda.set_device", "cudart.cudaSetDevice", "CUASSERT"):
        monkeypatch.setattr(f"tensorrt_llm._torch.pyexecutor.py_executor.{target}", Mock())

    executor = object.__new__(PyExecutor)
    executor._disagg_coordinator = _recording_coordinator(calls)
    executor._flush_pending_transfer_responses = lambda: calls.append(
        ("flush_pending_transfer_responses",)
    )
    executor._handle_kv_transfer_timeouts_synced = lambda: calls.append(
        ("handle_kv_transfer_timeouts_synced",)
    )
    executor.kv_cache_transceiver = Mock()
    executor.async_transfer_manager = Mock()
    executor.async_transfer_manager.has_any_inflight_requests.return_value = False
    executor.kv_connector_manager = None

    executor.device_id = 0
    profiler = MagicMock()
    profiler.__enter__.return_value = Mock()
    executor._profiler = Mock(return_value=profiler)
    executor.hang_detector = MagicMock()
    executor.enable_iter_perf_stats = False
    executor.enable_attention_dp = False
    executor.is_benchmark_disagg = False
    executor.iter_counter = 0
    executor._resource_governor_enabled = False
    executor._is_kv_manager_v2 = False
    executor._mm_encoder_item_scheduling_enabled = False
    executor.enable_early_first_token_response = False
    executor.drafter = None
    executor.model_engine = None
    executor.previous_batch = None
    executor.active_requests = []
    executor.waiting_queue = []
    executor.inflight_req_ids = set()
    executor.is_shutdown = False

    def fetch_new_requests():
        if fetch_new_requests.called:
            executor.is_shutdown = True
        fetch_new_requests.called = True
        return []

    fetch_new_requests.called = False
    executor._fetch_and_activate_new_requests = fetch_new_requests
    executor._schedule = Mock(return_value=(ScheduledRequests(), [], 0))
    executor._can_queue = Mock(return_value=(False, False))
    executor._check_benchmark_disagg_gate = Mock(return_value=(True, False))
    executor._sync_gen_only_benchmark_has_insufficient_kv = Mock(return_value=False)
    for name in (
        "_poll_encoder_steps",
        "_handle_control_request",
        "_pad_attention_dp_dummy_request",
        "_prefetch_for_context_requests",
        "_pad_empty_attention_dp_batch",
        "_terminate_requests",
        "_pause_requests",
        "_revert_gen_alloc",
        "_finalize_adp_dummy_allocation",
        "_wait_for_model_engine_input_copy",
        "_enqueue_responses",
    ):
        setattr(executor, name, Mock())
    return executor


def _pp_executor(monkeypatch, calls: list, *, rank: int) -> PyExecutor:
    """Idle executor on a two-stage pipeline; rank 0 schedules, rank 1 receives
    the schedule from its predecessor and re-runs the scheduler locally."""
    executor = _idle_executor(monkeypatch, calls)
    executor.dist = Mock(
        rank=rank,
        pp_rank=rank,
        pp_size=2,
        tp_size=1,
        cp_size=1,
        world_size=2,
        is_first_pp_rank=rank == 0,
        is_last_pp_rank=rank == 1,
        next_pp_rank=1 - rank,
        prev_pp_rank=1 - rank,
    )
    executor.num_micro_batches = 1
    executor.micro_batches = [None]
    executor.send_handles = [None]
    executor.send_schedule_handles = [None]
    executor.send_expected_batch_num_handles = [None]
    executor.wait_on_pp_send_handles = Mock()
    executor.executed_batch_response_queue = queue.Queue()
    executor.unhandled_batch_counter = 0
    executor.pp_async_broadcast_sample_state = True
    executor._pp_rebalance_drain_iters = None
    executor._progress_recompute_pause_termination_if_idle = Mock()
    if rank != 0:
        empty_schedule = SerializableSchedulerOutput.from_scheduler_result(
            ScheduledRequests(), [], 0
        )
        # First recv is the schedule, second is the executed-batch count.
        executor.dist.recv_object = Mock(side_effect=[empty_schedule, 0])
        executor.scheduler = Mock()
        executor.scheduler.can_schedule.return_value = True
        executor.scheduler.schedule_request.return_value = SimpleNamespace(
            fitting_disagg_gen_init_requests=[]
        )
        executor.kv_cache_manager = Mock(spec=[])
    return executor


_SCHEDULE_HEAD = [
    ("handle_errors_synced",),
    ("prepare_context_schedulable", []),
    ("poll_gen_transfers",),
    ("check_transfer_timeouts",),
    ("admit", []),
    ("receive_gen_init", []),
    ("poll_progress_when_idle",),
]
# Non-PP loops flush at loop exit; the PP loop does not.
_SHUTDOWN_PASS = [("handle_errors_synced",), ("flush_pending_transfer_responses",)]
_PP_SHUTDOWN_PASS = [("handle_errors_synced",)]


def test_executor_loop_transcript(monkeypatch) -> None:
    """The timeout-consensus drain precedes the response flush in this loop."""
    calls = []
    executor = _idle_executor(monkeypatch, calls)
    executor.dist = Mock(tp_size=1, world_size=1)

    PyExecutor._executor_loop(executor)

    idle_pass = _SCHEDULE_HEAD + [
        ("handle_kv_transfer_timeouts_synced",),
        ("flush_pending_transfer_responses",),
        ("pace_idle",),
    ]
    assert calls == idle_pass + _SHUTDOWN_PASS


def test_executor_loop_overlap_transcript(monkeypatch) -> None:
    """The overlap loop flushes responses before the timeout-consensus drain,
    the reverse of the non-overlap loop; each order is consistent across ranks
    on its own."""
    calls = []
    executor = _idle_executor(monkeypatch, calls)
    executor.dist = Mock(tp_size=1, world_size=1)

    PyExecutor._executor_loop_overlap(executor)

    idle_pass = _SCHEDULE_HEAD + [
        ("flush_pending_transfer_responses",),
        ("handle_kv_transfer_timeouts_synced",),
        ("pace_idle",),
    ]
    assert calls == idle_pass + _SHUTDOWN_PASS


def test_executor_loop_pp_transcript_on_first_rank(monkeypatch) -> None:
    """The PP loop admits inside schedule propagation, checks transfer timeouts
    only on the retry and executed-batch paths, and flushes responses only from
    executed-batch handling, so an idle iteration has none of those."""
    calls = []
    PyExecutor._executor_loop_pp(_pp_executor(monkeypatch, calls, rank=0))

    assert (
        calls
        == [
            ("handle_errors_synced",),
            ("prepare_context_schedulable", []),
            ("poll_gen_transfers",),
            ("admit", []),
            ("receive_gen_init", []),
            ("poll_progress_when_idle",),
            ("pace_idle",),
        ]
        + _PP_SHUTDOWN_PASS
    )


def test_executor_loop_pp_transcript_on_non_first_rank(monkeypatch) -> None:
    """A non-first rank does not admit; it reverts KV for candidates its local
    scheduler picked but the first rank did not admit."""
    calls = []
    PyExecutor._executor_loop_pp(_pp_executor(monkeypatch, calls, rank=1))

    assert (
        calls
        == [
            ("handle_errors_synced",),
            ("prepare_context_schedulable", []),
            ("poll_gen_transfers",),
            ("revert_deferred_gen_init", [], []),
            ("receive_gen_init", []),
            ("poll_progress_when_idle",),
            ("pace_idle",),
        ]
        + _PP_SHUTDOWN_PASS
    )


def test_pp_ranks_issue_the_same_collective_sensitive_calls(monkeypatch) -> None:
    """First and non-first PP ranks take different local paths, but every
    collective-sensitive boundary must be reached the same number of times in
    the same order or the consensus inside it deadlocks."""
    assert _COLLECTIVE_COORDINATOR_CALLS <= set(_entry_points())
    first, other = [], []
    PyExecutor._executor_loop_pp(_pp_executor(monkeypatch, first, rank=0))
    PyExecutor._executor_loop_pp(_pp_executor(monkeypatch, other, rank=1))

    assert _collective_calls(first) == _collective_calls(other)
    assert _collective_calls(first)  # the comparison is not vacuous
