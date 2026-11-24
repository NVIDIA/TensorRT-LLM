# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scheduler-confirmed idle and queue-only snapshots without model execution."""

import datetime
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.cpu_only

# Index of the fields in a row of the export buffer (see _append_iter_stats).
_ROW_STATS, _ROW_REQ_STATS, _ROW_KV_ITER_STATS, _ROW_RANK, _ROW_HOST_STEP_MS = range(5)


def _bind(stub, *names):
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    for name in names:
        setattr(stub, name, getattr(PyExecutor, name).__get__(stub))


def _snapshot(queued_context_requests=0, queued_ctx_tokens=0):
    ifb = SimpleNamespace(
        num_queued_context_requests=queued_context_requests,
        num_queued_ctx_tokens=queued_ctx_tokens,
        num_queued_gen_requests=0,
        num_queued_gen_kv_tokens=0,
    )
    return SimpleNamespace(inflight_batching_stats=ifb, iter_latency_ms=0.0)


@pytest.fixture
def clock(monkeypatch):
    """A monotonic clock the test advances by hand; no test waits on wall time."""
    from tensorrt_llm._torch.pyexecutor import py_executor

    now = SimpleNamespace(value=1000.0)
    monkeypatch.setattr(
        py_executor, "time", SimpleNamespace(monotonic=lambda: now.value, time=time.time)
    )
    return now


@pytest.fixture
def executor(clock):
    """A rank-0 executor stub around the real snapshot/heartbeat/export methods."""
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    stub = SimpleNamespace(
        enable_iter_perf_stats=True,
        is_shutdown=False,
        iter_counter=7,
        iter_perf_stats_interval=1,
        _quiescent_stats_iter=None,
        _next_idle_heartbeat_time=0.0,
        _idle_heartbeat_rows=[],
        _idle_recheck_iter=None,
        _queue_snapshot_counters=None,
        _next_queue_snapshot_time=0.0,
        stats=[],
        stats_lock=threading.Lock(),
        max_stats_len=1000,
        disable_overlap_scheduler=True,
        _latest_kv_iter_stats={32: "measured-deltas"},
        _latest_host_step_time_ms=7.0,
        _latest_prev_device_step_time_ms=5.0,
        enable_attention_dp=False,
        dist=SimpleNamespace(rank=0, pp_size=1, tp_size=1),
        perf_manager=SimpleNamespace(take_encoder_gpu_times_ms=Mock(return_value=[])),
        control_requests=[],
        _stats_has_pending_work=False,
        _has_pending_iteration_stats_work=lambda: False,
        _get_init_iter_stats=Mock(side_effect=lambda *args: _snapshot()),
        _update_iter_stats=Mock(side_effect=lambda row, *args, **kwargs: row),
        _adp_iter_stats=SimpleNamespace(
            queue=Mock(),
            last_finalize_settled=False,
            last_idle_rank_states=None,
            last_fanout_is_busy_queue_snapshot=False,
            last_fanout_queued_counters=(),
            last_fanout_has_rank0_row=True,
            next_payload=lambda: None,
            make_idle_heartbeat_rows=Mock(return_value=[]),
        ),
    )
    # The real export, so rows have their real shape and side effects.
    stub._append_iter_stats = Mock(side_effect=PyExecutor._append_iter_stats.__get__(stub))
    _bind(
        stub,
        "_stats_scheduler_mode",
        "_zero_duration_stats_row",
        "_take_mm_encoder_stats",
        "_append_stats_rows",
        "_maybe_emit_quiescent_stats",
        "_build_quiescent_stats",
        "_publish_idle_heartbeat",
        "_append_queue_snapshot",
        "_export_adp_fanout",
        "_idle_heartbeat_tick",
        "_plan_idle_heartbeat_wait",
    )
    return stub


def _append_measured(executor):
    """Export a measured record; returns its row."""
    executor._append_iter_stats(_snapshot(), ["req"])
    return executor.stats[-1]


def _publish_heartbeat(executor):
    executor._publish_idle_heartbeat([executor._zero_duration_stats_row(_snapshot())])


def _adp_rows(tag):
    """Serialized rows of a two-rank fanout, as ADPIterStatsBuffer returns them."""
    return [json.dumps({"attentionDpRank": rank, "tag": tag}) for rank in range(2)]


def _queue_backlog(executor, queued_context_requests, queued_ctx_tokens=100):
    executor._get_init_iter_stats.side_effect = lambda *args: _snapshot(
        queued_context_requests, queued_ctx_tokens
    )


def test_idle_snapshot_has_no_fabricated_iteration_time(executor):
    executor._maybe_emit_quiescent_stats(engine_idle=True)
    call = executor._update_iter_stats.call_args
    assert call.args[1:3] == (0.0, 0)
    assert call.args[3].batch_size == 0
    # A snapshot must not consume the KV deltas that belong to measured records.
    assert call.kwargs == {"consume_deltas": False}
    (row,) = executor.stats
    assert row[_ROW_KV_ITER_STATS] is None  # Nor repeat the last measured record's.
    assert row[_ROW_HOST_STEP_MS] == 0.0
    assert row[_ROW_RANK] is None


def test_busy_empty_snapshot_is_not_idle(executor):
    executor._maybe_emit_quiescent_stats(engine_idle=False)
    assert executor.stats == []


def test_blocked_prefill_snapshot_keeps_queued_work(executor):
    _queue_backlog(executor, 2)
    executor._maybe_emit_quiescent_stats(engine_idle=False)
    (row,) = executor.stats
    assert row[_ROW_STATS].inflight_batching_stats.num_queued_context_requests == 2
    assert row[_ROW_KV_ITER_STATS] is None
    # A queue snapshot is not a heartbeat: a later heartbeat must not replace it.
    executor.iter_counter += 1
    executor._maybe_emit_quiescent_stats(engine_idle=True)
    assert len(executor.stats) == 2


def _spin(executor):
    """One forward-less loop pass of a busy engine; returns the exported row count."""
    executor.iter_counter += 1
    executor._maybe_emit_quiescent_stats(engine_idle=False)
    return len(executor.stats)


def test_unchanged_queue_snapshot_is_exported_once_per_period(executor, clock):
    """Forward-less loops spin; repeats must not flood the bounded export buffer."""
    _queue_backlog(executor, 2, 300)
    assert [_spin(executor), _spin(executor), _spin(executor)] == [1, 1, 1]
    _queue_backlog(executor, 2, 200)  # A changed backlog is always exported.
    assert [_spin(executor), _spin(executor)] == [2, 2]
    clock.value += 0.99
    assert _spin(executor) == 2
    clock.value += 0.02  # The period has elapsed.
    assert _spin(executor) == 3
    # Snapshots are still taken every pass; only the export is rate-limited.
    assert executor._update_iter_stats.call_count == 7


def test_queue_snapshot_is_news_again_after_any_other_row(executor):
    """A consumer's newest row must not stay a stale measured or idle row."""
    _queue_backlog(executor, 2)
    assert _spin(executor) == 1
    _append_measured(executor)
    assert _spin(executor) == 3
    _publish_heartbeat(executor)
    assert _spin(executor) == 5
    assert _spin(executor) == 5


@pytest.mark.parametrize("rank,engine_idle", [(0, False), (1, True)])
def test_adp_retains_the_global_idle_decision(executor, rank, engine_idle):
    executor.enable_attention_dp = True
    executor.dist.rank = rank
    executor._maybe_emit_quiescent_stats(engine_idle=engine_idle)
    executor._adp_iter_stats.queue.assert_called_once_with(
        executor._update_iter_stats.call_args.args[0],
        is_rank0=rank == 0,
        kv_iter_stats=None,
        host_step_time_ms=0.0,
        prev_device_step_time_ms=0.0,
        gpu_forward_time_ms=0.0,
        is_queue_snapshot=True,
        idle_heartbeat=engine_idle,
    )
    assert executor.stats == []


def test_idle_wakeup_and_no_forward_pass_do_not_duplicate_stats(executor, clock):
    executor._maybe_emit_quiescent_stats(engine_idle=True)
    executor._maybe_emit_quiescent_stats(engine_idle=False)
    executor._update_iter_stats.assert_called_once()
    executor.iter_counter += 1
    clock.value += 1.0  # The heartbeat period has elapsed.
    executor._maybe_emit_quiescent_stats(engine_idle=True)
    assert executor._update_iter_stats.call_count == 2


def test_queue_only_sampling_does_not_throttle_idle_heartbeats(executor):
    executor.iter_perf_stats_interval = 10
    executor._maybe_emit_quiescent_stats(engine_idle=False)
    executor._update_iter_stats.assert_not_called()
    executor._maybe_emit_quiescent_stats(engine_idle=True)
    assert len(executor.stats) == 1


@pytest.mark.parametrize("attention_dp", [False, True])
def test_idle_heartbeat_is_rate_limited_only_without_attention_dp(executor, clock, attention_dp):
    """The period clock is rank-local, so it must not desynchronize ADP snapshots."""
    executor.enable_attention_dp = attention_dp
    executor._maybe_emit_quiescent_stats(engine_idle=True)
    assert executor._next_idle_heartbeat_time == clock.value + 1.0
    executor.iter_counter += 1
    clock.value += 0.5
    executor._maybe_emit_quiescent_stats(engine_idle=True)
    if attention_dp:
        assert executor._adp_iter_stats.queue.call_count == 2
    else:
        assert len(executor.stats) == 1
        # The suppressed heartbeat still settles this pass: no second snapshot.
        executor._maybe_emit_quiescent_stats(engine_idle=False)
        executor._update_iter_stats.assert_called_once()


def test_idle_heartbeat_replaces_the_previous_heartbeat(executor, clock):
    """An idle engine keeps one heartbeat, however slowly its stats are read."""
    measured = _append_measured(executor)
    for _ in range(3):
        executor.iter_counter += 1
        clock.value += 1.0
        executor._maybe_emit_quiescent_stats(engine_idle=True)
    assert len(executor.stats) == 2 and executor.stats[0] is measured
    newest = executor.stats[1]

    # A measured record after the heartbeat ends the replaceable tail.
    _append_measured(executor)
    _publish_heartbeat(executor)
    assert len(executor.stats) == 4 and executor.stats[1] is newest

    # Rows a consumer already took are left alone.
    taken, executor.stats = executor.stats, []
    _publish_heartbeat(executor)
    assert len(taken) == 4 and len(executor.stats) == 1


def test_attention_dp_heartbeat_replaces_every_rank_row(executor):
    first, second = _adp_rows("first"), _adp_rows("second")
    executor._publish_idle_heartbeat(first)
    executor._publish_idle_heartbeat(second)
    assert executor.stats == second
    # No rows: nothing is published and the previous heartbeat stays.
    executor._publish_idle_heartbeat([])
    assert executor.stats == second


@pytest.mark.parametrize("kind", ["measured", "queue-snapshot", "idle-heartbeat"])
def test_attention_dp_fanout_export_follows_the_fanout_kind(executor, kind):
    executor.enable_attention_dp = True
    adp_stats = executor._adp_iter_stats
    adp_stats.last_fanout_is_busy_queue_snapshot = kind == "queue-snapshot"
    adp_stats.last_fanout_queued_counters = ((2, 200, 0, 0), (1, 100, 0, 0))
    if kind == "idle-heartbeat":
        adp_stats.last_idle_rank_states = [object(), object()]
    first, second, third = _adp_rows("first"), _adp_rows("second"), _adp_rows("third")
    fourth = _adp_rows("fourth")
    executor._export_adp_fanout(first)
    executor._export_adp_fanout(second)
    # An iteration that fans out no row exports nothing.
    executor._export_adp_fanout([])

    if kind == "measured":
        # Every measured iteration is exported.
        assert executor.stats == first + second
    elif kind == "queue-snapshot":
        # The repeat is dropped: the first snapshot stays.
        assert executor.stats == first
        # A changed backlog is exported.
        adp_stats.last_fanout_queued_counters = ((2, 200, 0, 0), (2, 200, 0, 0))
        executor._export_adp_fanout(third)
        assert executor.stats == first + third
        # So is the same backlog on other ranks.
        adp_stats.last_fanout_queued_counters = ((3, 300, 0, 0), (1, 100, 0, 0))
        executor._export_adp_fanout(fourth)
        executor._export_adp_fanout(_adp_rows("repeat"))
        assert executor.stats == first + third + fourth
    else:
        # The newer heartbeat replaces the first.
        assert executor.stats == second


def test_attention_dp_queue_snapshot_is_exported_again_when_the_backlog_moves(executor):
    """The fanout buffer's per-rank counters drive the export: a moved backlog is news."""
    from tensorrt_llm._torch.pyexecutor.adp_iter_stats import ADPIterStatsBuffer
    from tensorrt_llm._torch.pyexecutor.scheduler.adp_router import RankState
    from tensorrt_llm.bindings.executor import InflightBatchingStats, IterationStats

    executor.enable_attention_dp = True
    buffers = [ADPIterStatsBuffer(), ADPIterStatsBuffer()]
    executor._adp_iter_stats = buffers[0]

    def spin(iter_id, queued_requests_per_rank):
        """One forward-less loop pass on both ranks; returns the rows exported so far."""
        states = []
        for rank, (buffer, queued) in enumerate(zip(buffers, queued_requests_per_rank)):
            stats = IterationStats()
            stats.iter = iter_id
            stats.inflight_batching_stats = InflightBatchingStats()
            stats.inflight_batching_stats.num_queued_context_requests = queued
            stats.inflight_batching_stats.num_queued_ctx_tokens = 100 * queued
            buffer.queue(stats, is_rank0=rank == 0, is_queue_snapshot=True)
            states.append(RankState(rank=rank, iter_stats=buffer.next_payload()))
        assert buffers[1].finalize(states, is_rank0=False) == []
        executor._export_adp_fanout(buffers[0].finalize(states, is_rank0=True))
        return [
            (row["attentionDpRank"], row["numQueuedRequests"])
            for row in map(json.loads, executor.stats)
        ]

    assert spin(1, [2, 1]) == [(0, 2), (1, 1)]
    # A repeat is dropped.
    assert spin(2, [2, 1]) == [(0, 2), (1, 1)]
    # The same three requests, two of them now on rank 1.
    assert spin(3, [1, 2]) == [(0, 2), (1, 1), (0, 1), (1, 2)]
    assert spin(4, [1, 2]) == [(0, 2), (1, 1), (0, 1), (1, 2)]
    # The backlog left rank 0 altogether: the snapshot has rank 1's row only.
    assert spin(5, [0, 3]) == [(0, 2), (1, 1), (0, 1), (1, 2), (1, 3)]


def test_zero_duration_rows_leave_the_encoder_calls_for_a_measured_record(executor):
    """A heartbeat can be replaced and a snapshot dropped, so neither takes the encoder calls."""
    take = executor.perf_manager.take_encoder_gpu_times_ms
    take.return_value = [2.5]

    _publish_heartbeat(executor)
    _queue_backlog(executor, 2)
    executor._maybe_emit_quiescent_stats(engine_idle=False)
    assert len(executor.stats) == 2
    assert all(row.mm_encoder_stats is None and row.is_queue_snapshot for row in executor.stats)
    take.assert_not_called()

    measured = _append_measured(executor)
    assert measured.mm_encoder_stats == {"gpuTimesMS": [2.5]} and not measured.is_queue_snapshot
    take.assert_called_once_with()


@pytest.mark.parametrize(
    "kind, has_rank0_row, taken",
    [
        ("measured", True, True),
        # A zero-duration record that dropped rank 0's empty row: the first row is a peer's.
        ("measured", False, False),
        ("queue-snapshot", True, False),
        ("idle-heartbeat", True, False),
    ],
)
def test_attention_dp_encoder_calls_go_on_rank_zeros_measured_row(
    executor, kind, has_rank0_row, taken
):
    executor.enable_attention_dp = True
    executor.perf_manager.take_encoder_gpu_times_ms.return_value = [2.5]
    adp_stats = executor._adp_iter_stats
    adp_stats.last_fanout_is_busy_queue_snapshot = kind == "queue-snapshot"
    adp_stats.last_fanout_queued_counters = ((1, 100, 0, 0), (0, 0, 0, 0))
    adp_stats.last_fanout_has_rank0_row = has_rank0_row
    if kind == "idle-heartbeat":
        adp_stats.last_idle_rank_states = [object(), object()]
    rows = _adp_rows("first") if has_rank0_row else _adp_rows("first")[1:]

    executor._export_adp_fanout(rows)

    encoder_stats = [json.loads(row).get("multimodalEncoderStats") for row in executor.stats]
    if taken:
        assert encoder_stats == [{"gpuTimesMS": [2.5]}, None]
    else:
        assert executor.stats == rows
        executor.perf_manager.take_encoder_gpu_times_ms.assert_not_called()


def test_idle_tick_publishes_only_when_due(executor, clock):
    from tensorrt_llm._torch.pyexecutor.py_executor import IDLE_HEARTBEAT_PERIOD_S

    executor._next_idle_heartbeat_time = clock.value + 0.4
    assert executor._idle_heartbeat_tick() == pytest.approx(0.4)
    executor._update_iter_stats.assert_not_called()

    clock.value += 0.4
    assert executor._idle_heartbeat_tick() == IDLE_HEARTBEAT_PERIOD_S
    assert len(executor.stats) == 1
    assert executor._next_idle_heartbeat_time == clock.value + IDLE_HEARTBEAT_PERIOD_S
    # Ticks do not consume the per-iteration snapshot guard: an all-rank
    # exchange in the same iteration may still queue its snapshot.
    assert executor._quiescent_stats_iter is None


def test_attention_dp_idle_tick_repeats_one_row_per_rank(executor):
    executor.enable_attention_dp = True
    rows = _adp_rows("heartbeat")
    executor._adp_iter_stats.make_idle_heartbeat_rows.return_value = rows
    executor._idle_heartbeat_tick()
    executor._adp_iter_stats.make_idle_heartbeat_rows.assert_called_once_with(
        executor._update_iter_stats.call_args.args[0], "non_overlap"
    )
    assert executor.stats == rows


def test_parked_rank_zero_publishes_heartbeats_through_the_request_queue(executor, monkeypatch):
    """The real tick, driven by the real queue wait, on a short period."""
    from tensorrt_llm._torch.pyexecutor import py_executor
    from tensorrt_llm._torch.pyexecutor.executor_request_queue import (
        ExecutorRequestQueue,
        RequestQueueItem,
    )

    # Wall clock for this test: the queue wait is real.
    monkeypatch.setattr(py_executor, "time", time)
    monkeypatch.setattr(py_executor, "IDLE_HEARTBEAT_PERIOD_S", 0.01)
    request_queue = ExecutorRequestQueue(
        dist=executor.dist, max_batch_size=8, enable_iter_perf_stats=True, batch_wait_timeout_ms=0.0
    )
    measured = _append_measured(executor)
    timeout, idle_tick = executor._plan_idle_heartbeat_wait(None)
    assert timeout is None
    item = RequestQueueItem(7, Mock())

    def tick_then_request_after_three_heartbeats():
        # The request arrives after a fixed number of heartbeats, so the count
        # below does not depend on how the threads are scheduled.
        wait = idle_tick()
        if executor._update_iter_stats.call_count == 3 and request_queue.request_queue.empty():
            request_queue.request_queue.put(item)
        return wait

    # The timeout only keeps a broken wait from hanging the test.
    items = request_queue.get_from_request_queue(
        datetime.timedelta(seconds=60), idle_tick=tick_then_request_after_three_heartbeats
    )

    assert items == [item]
    # Three heartbeats were published while parked; the buffer kept the newest
    # one and the measured record, and the loop never ran.
    assert executor._update_iter_stats.call_count == 3
    assert len(executor.stats) == 2 and executor.stats[0] is measured
    assert executor.stats[1][_ROW_STATS] is executor._update_iter_stats.call_args.args[0]
    assert executor.iter_counter == 7


def test_idle_wait_parks_rank_zero_with_local_heartbeats(executor):
    timeout, tick = executor._plan_idle_heartbeat_wait(None)
    assert timeout is None and tick == executor._idle_heartbeat_tick
    # The Ray heartbeat timeout still bounds the whole wait.
    ray_timeout = datetime.timedelta(seconds=1200)
    assert executor._plan_idle_heartbeat_wait(ray_timeout) == (
        ray_timeout,
        executor._idle_heartbeat_tick,
    )


def test_idle_wait_is_unchanged_on_other_ranks(executor):
    executor.dist.rank = 1
    assert executor._plan_idle_heartbeat_wait(None) == (None, None)


@pytest.mark.parametrize("attention_dp", [False, True])
@pytest.mark.parametrize("source", ["pending-work", "control"])
def test_idle_wait_lets_a_loop_pass_settle_pending_work(executor, attention_dp, source):
    """A busy rank cannot be reported idle: one pass at once, then once per period."""
    from tensorrt_llm._torch.pyexecutor.py_executor import IDLE_HEARTBEAT_PERIOD_S

    executor.enable_attention_dp = attention_dp
    if source == "control":
        executor.control_requests = [object()]
    elif attention_dp:
        executor._stats_has_pending_work = True
    else:
        executor._has_pending_iteration_stats_work = lambda: True
    period = datetime.timedelta(seconds=IDLE_HEARTBEAT_PERIOD_S)
    poll = datetime.timedelta(milliseconds=200)

    # A pending model batch is harvested by the next loop pass: run it now.
    assert executor._plan_idle_heartbeat_wait(None) == (datetime.timedelta(0), None)
    # Still pending after that pass (a transfer, a control request): no spin.
    executor.iter_counter += 1
    assert executor._plan_idle_heartbeat_wait(None) == (period, None)
    executor.iter_counter += 1
    assert executor._plan_idle_heartbeat_wait(poll) == (datetime.timedelta(0), None)
    executor.iter_counter += 1
    assert executor._plan_idle_heartbeat_wait(poll) == (poll, None)


def test_attention_dp_idle_wait_exchanges_before_parking(executor, clock):
    """Without the peers' rows, the wait ends when the next heartbeat is due."""
    executor.enable_attention_dp = True
    assert executor._plan_idle_heartbeat_wait(None) == (datetime.timedelta(0), None)

    executor._next_idle_heartbeat_time = clock.value + 0.4
    timeout, tick = executor._plan_idle_heartbeat_wait(None)
    assert tick is None and timeout.total_seconds() == pytest.approx(0.4)
    # A shorter caller timeout wins.
    poll = datetime.timedelta(milliseconds=100)
    assert executor._plan_idle_heartbeat_wait(poll) == (poll, None)


def test_attention_dp_idle_wait_parks_once_peer_rows_are_current(executor):
    executor.enable_attention_dp = True
    adp_stats = executor._adp_iter_stats
    adp_stats.last_finalize_settled = True
    adp_stats.last_idle_rank_states = [object(), object()]
    assert executor._plan_idle_heartbeat_wait(None) == (None, executor._idle_heartbeat_tick)

    # Snapshots still queued behind the fanned-out one drain one per pass.
    adp_stats.next_payload = lambda: object()
    assert executor._plan_idle_heartbeat_wait(None) == (datetime.timedelta(0), None)
    # Ranks that are not aligned settle nothing: wait instead of spinning.
    adp_stats.last_finalize_settled = False
    adp_stats.last_idle_rank_states = None
    assert executor._plan_idle_heartbeat_wait(None) == (datetime.timedelta(seconds=1), None)


def _attention_dp_rank(rank, clock):
    """One DP rank: a real stats buffer behind the real snapshot/export methods."""
    from tensorrt_llm._torch.pyexecutor.adp_iter_stats import ADPIterStatsBuffer
    from tensorrt_llm.bindings.executor import InflightBatchingStats, IterationStats

    stub = SimpleNamespace(
        enable_iter_perf_stats=True,
        is_shutdown=False,
        iter_counter=11,
        iter_perf_stats_interval=1,
        _quiescent_stats_iter=None,
        _next_idle_heartbeat_time=0.0,
        _idle_heartbeat_rows=[],
        _idle_recheck_iter=None,
        _queue_snapshot_counters=None,
        _next_queue_snapshot_time=0.0,
        stats=[],
        stats_lock=threading.Lock(),
        max_stats_len=1000,
        disable_overlap_scheduler=False,
        enable_attention_dp=True,
        dist=SimpleNamespace(rank=rank, pp_size=1, tp_size=2),
        perf_manager=SimpleNamespace(take_encoder_gpu_times_ms=lambda: []),
        control_requests=[],
        _stats_has_pending_work=False,
        _adp_iter_stats=ADPIterStatsBuffer(),
        _update_iter_stats=lambda row, *args, **kwargs: row,
    )

    def make_stats(latency_ms=0.0, gen_requests=0):
        stats = IterationStats()
        stats.iter = stub.iter_counter
        stats.iter_latency_ms = latency_ms
        stats.inflight_batching_stats = InflightBatchingStats()
        stats.inflight_batching_stats.num_gen_requests = gen_requests
        return stats

    stub.make_stats = make_stats
    stub._get_init_iter_stats = lambda *args: make_stats()
    _bind(
        stub,
        "_stats_scheduler_mode",
        "_zero_duration_stats_row",
        "_take_mm_encoder_stats",
        "_append_stats_rows",
        "_maybe_emit_quiescent_stats",
        "_build_quiescent_stats",
        "_publish_idle_heartbeat",
        "_append_queue_snapshot",
        "_export_adp_fanout",
        "_publish_adp_iter_stats",
        "_idle_heartbeat_tick",
        "_plan_idle_heartbeat_wait",
    )
    return stub


class _RankExchange:
    """The DP ranks' allgather of iteration-stats payloads.

    A rank's call returns every rank's state once all ranks have called, so
    ranks that disagree on the number of exchanges time out instead of passing.
    """

    def __init__(self, num_ranks):
        self.count = 0
        self._payloads = [None] * num_ranks
        self._states = None
        self._barrier = threading.Barrier(num_ranks, action=self._gather, timeout=5)

    def _gather(self):
        from tensorrt_llm._torch.pyexecutor.scheduler.rank_state import RankState

        self._states = []
        for rank, payload in enumerate(self._payloads):
            state = RankState(rank=rank)
            state.copy_iter_stats_from(payload)
            self._states.append(state)
        self.count += 1

    def gather(self, rank, payload):
        self._payloads[rank] = payload
        self._barrier.wait()
        return self._states


def _attention_dp_idle_pass(ranks, clock):
    """Routing exchange, fanout and rank 0's wait plan, as _fetch_new_requests does.

    Returns rank 0's idle_tick once it parks, and the number of exchanges the
    pass took. Otherwise rank 0's wait ends empty and every rank runs the rest
    of an idle loop pass.
    """
    exchange = _RankExchange(len(ranks))

    def exchange_and_publish(stub):
        rank = stub.dist.rank
        stub.adp_router = SimpleNamespace(
            gather_iter_stats=lambda payload: exchange.gather(rank, payload)
        )
        states = exchange.gather(rank, stub._adp_iter_stats.next_payload())
        stub._publish_adp_iter_stats(states, idle_wait_follows=True)

    with ThreadPoolExecutor(len(ranks)) as pool:
        for done in [pool.submit(exchange_and_publish, stub) for stub in ranks]:
            done.result()
    timeout, idle_tick = ranks[0]._plan_idle_heartbeat_wait(None)
    if idle_tick is not None:
        return idle_tick, exchange.count
    clock.value += timeout.total_seconds()
    for stub in ranks:
        stub._maybe_emit_quiescent_stats(engine_idle=True)
        stub._maybe_emit_quiescent_stats(engine_idle=False)  # The loop tail.
        stub.iter_counter += 1
    return None, exchange.count


@pytest.mark.parametrize("overlap", [False, True])
def test_attention_dp_engine_parks_soon_after_going_idle(clock, overlap):
    """Rank 0 must reach the parked state, not exchange every heartbeat forever."""
    ranks = [_attention_dp_rank(rank, clock) for rank in range(2)]
    for stub in ranks:
        # The last busy pass queued the measured record of its batch ...
        stub.iter_counter = 10
        stub._adp_iter_stats.queue(
            stub.make_stats(latency_ms=5.0, gen_requests=1), is_rank0=stub.dist.rank == 0
        )
        stub.iter_counter = 11
        if overlap:
            # ... and under the overlap scheduler the forward-less pass that
            # harvested it also queued a snapshot of the still-busy engine.
            stub._maybe_emit_quiescent_stats(engine_idle=False)
            stub.iter_counter = 12
    rank0, started = ranks[0], clock.value

    idle_tick, exchanges = None, []
    for _ in range(5):
        idle_tick, count = _attention_dp_idle_pass(ranks, clock)
        exchanges.append(count)
        if idle_tick is not None:
            break
    assert idle_tick is not None, "rank 0 never parked"
    # Before an idle wait the ranks exchange until rank 0 has nothing pending:
    # the first pass fans out what the busy passes left, the second the idle
    # heartbeat every rank queued after it. Each ends with the exchange that
    # shows rank 0 has nothing left, which must not make rank 0 forget that
    # the heartbeat was fanned out.
    assert exchanges == ([3, 2] if overlap else [2, 2])
    assert clock.value == started  # No pass waited for a heartbeat period.
    # The measured record, then one fresh heartbeat row per rank.
    rows = [json.loads(row) for row in rank0.stats]
    assert [row["attentionDpRank"] for row in rows] == [0, 1, 0, 1]
    assert [row["iterLatencyMS"] for row in rows] == [5.0, 5.0, 0.0, 0.0]
    assert {row["schedulerMode"] for row in rows} == {"overlap"}
    assert rows[2]["iter"] == rank0.iter_counter - 1
    assert ranks[1].stats == [] and ranks[1]._adp_iter_stats.next_payload() is None

    # Parked: heartbeats come from the tick alone and replace one another.
    for _ in range(3):
        clock.value += 1.0
        assert idle_tick() == 1.0
    rows = [json.loads(row) for row in rank0.stats]
    assert [row["attentionDpRank"] for row in rows] == [0, 1, 0, 1]
    assert [row["iter"] for row in rows[2:]] == [rank0.iter_counter] * 2
    assert [row["hostStepTimeMS"] for row in rows[2:]] == [0.0, 0.0]
    assert rank0._adp_iter_stats.next_payload() is None


class _WaitingQueue(list):
    def add_requests(self, requests):
        self.extend(requests)


def _fetch_stub(**overrides):
    stub = SimpleNamespace(
        kv_connector_manager=None,
        control_requests=[],
        enable_attention_dp=True,
        is_shutdown=False,
        enable_iter_perf_stats=True,
        _disable_mpi=False,
        _benchmark_sweep=False,
        iter_counter=7,
        dist=SimpleNamespace(rank=0, pp_size=1),
        disagg=SimpleNamespace(has_retiring_gen_receives=False),
        canceled_req_ids=set(),
        kv_cache_transceiver=None,
        _has_pending_connector_transfers=lambda: False,
        _has_pending_context_retirement=lambda: False,
        _idle_kv_transfer_poll_timeout=lambda: None,
        _stats_has_pending_work=False,
        _has_pending_iteration_stats_work=lambda: False,
        _next_idle_heartbeat_time=0.0,
        _idle_recheck_iter=None,
        _adp_iter_stats=SimpleNamespace(
            last_finalize_settled=False, last_idle_rank_states=None, next_payload=lambda: None
        ),
        _idle_heartbeat_tick=Mock(return_value=1.0),
        _flush_staged_responses=lambda: None,
        _waiting_queue_needs_clock=False,
        request_accumulated=[],
        hang_detector=SimpleNamespace(pause=nullcontext),
        executor_request_queue=SimpleNamespace(get_from_request_queue=Mock(return_value=[])),
        request_broadcaster=SimpleNamespace(broadcast=Mock(side_effect=lambda items, **kw: items)),
        _handle_special_queue_items=lambda items: items,
        _maybe_emit_quiescent_stats=Mock(),
    )
    for name, value in overrides.items():
        setattr(stub, name, value)
    _bind(stub, "_plan_idle_heartbeat_wait")
    return stub


@pytest.mark.parametrize(
    "live,waiting,pending,retiring,new_items,control,stats_busy,expected_idle",
    [
        (0, False, False, False, False, False, False, True),
        (1, False, False, False, False, False, False, False),
        (0, True, False, False, False, False, False, False),
        (0, False, True, False, False, False, False, False),
        (0, False, False, True, False, False, False, False),
        (0, False, False, False, True, False, False, False),
        (0, False, False, False, False, True, False, False),
        (0, False, False, False, False, False, True, False),
    ],
    ids=[
        "all-ranks-idle",
        "peer-busy",
        "waiting",
        "transfer",
        "retiring-receive",
        "arrival",
        "control",
        "peer-pending-work",
    ],
)
def test_idle_fetch_requires_all_ranks_to_drain(
    clock, live, waiting, pending, retiring, new_items, control, stats_busy, expected_idle
):
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    queue = _WaitingQueue([object()] if waiting else [])
    stub = _fetch_stub(
        control_requests=[SimpleNamespace(control_requires_drain=True)] if control else [],
        disagg=SimpleNamespace(has_retiring_gen_receives=retiring),
        _has_pending_connector_transfers=lambda: pending,
        _stats_has_pending_work=stats_busy,
        executor_request_queue=SimpleNamespace(
            get_from_request_queue=Mock(return_value=[object()] if new_items else [])
        ),
    )
    PyExecutor._fetch_and_enqueue_requests(stub, queue, live)
    fetch = stub.executor_request_queue.get_from_request_queue
    if expected_idle:
        stub._maybe_emit_quiescent_stats.assert_called_once_with(engine_idle=True)
    else:
        stub._maybe_emit_quiescent_stats.assert_not_called()
    if not control:
        # No peer rows yet and a heartbeat is due, a peer still busy, or the
        # engine not idle at all: none of them waits, and none parks.
        fetch.assert_called_once_with(datetime.timedelta(0), wait_for_batch=not retiring)
    assert bool(queue) == (waiting or new_items)


@pytest.mark.parametrize("attention_dp", [False, True])
@pytest.mark.parametrize("blocker", ["control-arrival", "pending-work"])
def test_idle_fetch_reports_nothing_while_work_remains(clock, attention_dp, blocker):
    """The engine was idle before the wait; what the wait returned decides the report."""
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    stub = _fetch_stub(enable_attention_dp=attention_dp)
    if blocker == "control-arrival":
        # The fetched envelope held only a control request.
        def take_control(items):
            stub.control_requests.append(object())
            return []

        stub._handle_special_queue_items = take_control
    elif attention_dp:
        stub._stats_has_pending_work = True
    else:
        stub._has_pending_iteration_stats_work = lambda: True

    PyExecutor._fetch_and_enqueue_requests(stub, _WaitingQueue(), 0)

    stub.request_broadcaster.broadcast.assert_called_once_with([], prefer_cpu=True)
    stub._maybe_emit_quiescent_stats.assert_not_called()


@pytest.mark.parametrize("attention_dp", [False, True])
def test_idle_fetch_parks_with_local_heartbeats(clock, attention_dp):
    """A parked rank 0 passes its tick to the queue wait and skips nothing else."""
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    stub = _fetch_stub(
        enable_attention_dp=attention_dp,
        _adp_iter_stats=SimpleNamespace(
            last_finalize_settled=True,
            last_idle_rank_states=[object(), object()],
            next_payload=lambda: None,
        ),
    )
    PyExecutor._fetch_and_enqueue_requests(stub, _WaitingQueue(), 0)
    stub.executor_request_queue.get_from_request_queue.assert_called_once_with(
        None, wait_for_batch=True, idle_tick=stub._idle_heartbeat_tick
    )
    stub.request_broadcaster.broadcast.assert_called_once_with([], prefer_cpu=True)


@pytest.mark.parametrize(
    "previous,pp_pending,native_transfer,control,close_outs",
    [
        (False, False, False, False, False),
        (True, False, False, False, False),
        (False, True, False, False, False),
        (False, False, True, False, False),
        (False, False, False, True, False),
        (False, False, False, False, True),
    ],
    ids=["drained", "model-batch", "pipeline", "native-kv-send", "control", "close-outs"],
)
def test_pending_work_includes_model_batches_and_native_transfers(
    previous, pp_pending, native_transfer, control, close_outs
):
    """Also completions of requests finished at their first token that no record reports yet:
    an idle heartbeat must not come before them."""
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    stub = SimpleNamespace(
        previous_batch=object() if previous else None,
        dist=SimpleNamespace(pp_size=1 if previous else 2),
        _pp_ring_is_drained=lambda: not pp_pending,
        control_requests=[object()] if control else [],
        _has_close_outs_pending_iter_stats=close_outs,
        async_transfer_manager=SimpleNamespace(has_any_inflight_requests=lambda: native_transfer),
    )
    assert PyExecutor._has_pending_iteration_stats_work(stub) == (
        previous or pp_pending or native_transfer or control or close_outs
    )


def test_drained_pipeline_ignores_retained_previous_batch():
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    stub = SimpleNamespace(
        previous_batch=object(),
        dist=SimpleNamespace(pp_size=2),
        _pp_ring_is_drained=lambda: True,
        control_requests=[],
        _has_close_outs_pending_iter_stats=False,
        async_transfer_manager=None,
    )
    assert not PyExecutor._has_pending_iteration_stats_work(stub)
