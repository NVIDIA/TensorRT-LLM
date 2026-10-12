# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``iter_perf_stats_interval`` sampling in ``PyExecutor``.

With ``iter_perf_stats_interval = N`` only every Nth executor iteration builds
an ``IterationStats`` record. These tests call the real ``PyExecutor`` methods
unbound against a minimal fake ``self`` and check that:

* only iterations with ``iter_counter % N == 0`` get a record, and the
  decision only depends on ``iter_counter`` (so all ranks agree);
* ``numNewActiveRequests`` / ``numCompletedRequests`` from skipped iterations
  are folded into the next emitted record, so sums over records stay exact;
* without attention DP, an unsampled batch that drains the executor still
  emits one record, so the carried-over counters are not held back while the
  executor is idle;
* an interval of 1 still builds a record on every iteration;
* KV-cache iteration deltas are keyed on the record's construction iteration,
  so they still line up with sampled records under the overlap scheduler.
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import pytest

from tensorrt_llm._torch.pyexecutor.py_executor import BatchState, PyExecutor
from tensorrt_llm.bindings.executor import InflightBatchingStats, IterationStats

pytestmark = pytest.mark.cpu_only


def _build_fake_self(interval: int, *, enabled: bool = True):
    """Minimal ``self`` for the iteration-stats sampling methods."""
    fake = MagicMock()
    fake.enable_iter_perf_stats = enabled
    fake.enable_iter_req_stats = False
    fake.enable_attention_dp = False
    fake.iter_perf_stats_interval = interval
    fake.iter_counter = 0
    fake.active_requests = []
    fake._pending_num_new_active_requests = 0
    fake._pending_num_completed_requests = 0
    fake._latest_host_step_time_ms = None
    fake._latest_prev_device_step_time_ms = None
    # _get_init_iter_stats: no spec-decode resource manager / spec config.
    fake.resource_manager.resource_managers.get.return_value = None
    fake.model_engine.spec_config = None
    fake._get_new_active_requests_queue_latency.return_value = 0.0
    fake.perf_manager.try_compute_gpu_elapsed_time_ms.return_value = None

    # Bind the real methods under test; everything else stays a mock.
    for name in (
        "_get_init_iter_stats",
        "_init_iter_stats_if_sampled",
        "_take_pending_new_active_requests",
        "_should_flush_skipped_iter_stats",
    ):
        method = getattr(PyExecutor, name)
        setattr(fake, name, types.MethodType(method, fake))
    # Record what _update_iter_stats receives and pass the stats through.
    fake._update_iter_stats.side_effect = lambda stats, *args, **kwargs: stats
    return fake


def _run_iteration(fake, iter_counter: int, num_new: int):
    """Build the iteration's record and queue its batch."""
    fake.iter_counter = iter_counter
    return fake._take_pending_new_active_requests(fake._init_iter_stats_if_sampled(num_new))


def _process(fake, iter_stats, *, iter_id, finished=(), active=(), from_pool=False):
    batch_state = BatchState(
        scheduled_requests=MagicMock(),
        sample_state=MagicMock(),
        iter_start_time=0.0,
        iter_stats=iter_stats,
        iter_id=iter_id,
        gpu_forward_start_event=MagicMock() if from_pool else None,
        gpu_forward_end_event=MagicMock() if from_pool else None,
        gpu_forward_events_from_perf_pool=from_pool,
    )
    PyExecutor._process_iter_stats(fake, list(finished), list(active), batch_state)
    return batch_state


def _emitted(fake):
    """(stats, num_completed_requests) for every record passed downstream."""
    return [(call.args[0], call.args[2]) for call in fake._update_iter_stats.call_args_list]


# ---------------------------------------------------------------------------
# _init_iter_stats_if_sampled
# ---------------------------------------------------------------------------


def test_disabled_stats_never_sample_or_accumulate():
    fake = _build_fake_self(interval=1, enabled=False)
    assert _run_iteration(fake, 0, num_new=3) is None
    assert fake._pending_num_new_active_requests == 0


def test_interval_one_samples_every_iteration():
    fake = _build_fake_self(interval=1)
    for it, num_new in enumerate([2, 0, 5]):
        stats = _run_iteration(fake, it, num_new)
        assert stats is not None
        assert stats.iter == it
        assert stats.num_new_active_requests == num_new
    assert fake._pending_num_new_active_requests == 0


def test_interval_samples_every_nth_iteration_and_folds_new_requests():
    fake = _build_fake_self(interval=4)
    new_per_iter = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    sampled = {}
    for it, num_new in enumerate(new_per_iter):
        stats = _run_iteration(fake, it, num_new)
        if stats is not None:
            sampled[it] = stats.num_new_active_requests

    assert list(sampled) == [0, 4, 8]
    # Each record covers its own iteration plus the skipped ones before it.
    assert sampled == {0: 1, 4: 2 + 3 + 4 + 5, 8: 6 + 7 + 8 + 9}
    assert sum(sampled.values()) == sum(new_per_iter)
    assert fake._pending_num_new_active_requests == 0
    # A sampled record dropped with an unqueued batch leaves its count pending.
    fake.iter_counter = 12
    assert fake._init_iter_stats_if_sampled(10) is not None
    assert _run_iteration(fake, 16, num_new=1).num_new_active_requests == 10 + 1


# ---------------------------------------------------------------------------
# _process_iter_stats
# ---------------------------------------------------------------------------


def test_skipped_batches_fold_completed_requests_into_next_record():
    fake = _build_fake_self(interval=3)
    still_running = [MagicMock()]
    finished_per_iter = [1, 2, 0, 4, 1, 3]
    for it, num_finished in enumerate(finished_per_iter):
        iter_stats = _run_iteration(fake, it, num_new=0)
        _process(
            fake,
            iter_stats,
            iter_id=it,
            finished=[MagicMock()] * num_finished,
            active=still_running,
        )

    emitted = _emitted(fake)
    assert [stats.iter for stats, _ in emitted] == [0, 3]
    assert [completed for _, completed in emitted] == [1, 2 + 0 + 4]
    # Iterations 4 and 5 are still pending for the next record.
    assert fake._pending_num_completed_requests == 1 + 3
    assert fake._append_iter_stats.call_count == 2


def test_skipped_batch_releases_borrowed_forward_events():
    fake = _build_fake_self(interval=2)
    _process(fake, None, iter_id=1, active=[MagicMock()], from_pool=True)
    fake.perf_manager.release_forward_timing_events.assert_called_once()
    fake._update_iter_stats.assert_not_called()
    fake._append_iter_stats.assert_not_called()


def test_draining_skipped_batch_emits_record_with_carried_counters():
    fake = _build_fake_self(interval=4)
    # Iteration 0 is sampled; iterations 1-2 are skipped and admit/finish work.
    for it, (num_new, num_finished) in enumerate([(2, 0), (1, 1), (0, 1)]):
        iter_stats = _run_iteration(fake, it, num_new)
        _process(
            fake,
            iter_stats,
            iter_id=it,
            finished=[MagicMock()] * num_finished,
            active=[MagicMock()],
        )
    # Iteration 3 is skipped and finishes the last active request.
    iter_stats = _run_iteration(fake, 3, num_new=0)
    assert iter_stats is None
    _process(fake, iter_stats, iter_id=3, finished=[MagicMock()], active=[])

    emitted = _emitted(fake)
    assert [stats.iter for stats, _ in emitted] == [0, 3]
    drain_stats, drain_completed = emitted[-1]
    assert drain_completed == 1 + 1 + 1
    assert drain_stats.num_new_active_requests == 1
    assert drain_stats.num_active_requests == 0
    assert fake._pending_num_completed_requests == 0
    assert fake._pending_num_new_active_requests == 0


def test_draining_skipped_batch_without_pending_counters_emits_nothing():
    fake = _build_fake_self(interval=4)
    _process(fake, None, iter_id=5, finished=[], active=[])
    fake._update_iter_stats.assert_not_called()


def test_attention_dp_never_flushes_unsampled_batch():
    """Under attention DP the drain decision is rank-local, so it is skipped."""
    fake = _build_fake_self(interval=4)
    fake.enable_attention_dp = True
    _process(fake, None, iter_id=3, finished=[MagicMock()], active=[])
    fake._update_iter_stats.assert_not_called()
    assert fake._pending_num_completed_requests == 1


def test_interval_one_never_flushes_unsampled_batch():
    """With the default interval, a batch without stats keeps the old early return."""
    fake = _build_fake_self(interval=1)
    _process(fake, None, iter_id=7, finished=[MagicMock()], active=[])
    fake._update_iter_stats.assert_not_called()
    fake._append_iter_stats.assert_not_called()


# ---------------------------------------------------------------------------
# _update_iter_stats: KV-cache iteration stats interval
# ---------------------------------------------------------------------------


def _kv_update_fake(kv_interval: int):
    kv_cache_manager = MagicMock()
    kv_cache_manager.get_kv_cache_stats.return_value = types.SimpleNamespace(
        max_num_blocks=8,
        free_num_blocks=4,
        used_num_blocks=4,
        tokens_per_block=32,
        alloc_total_blocks=4,
        alloc_new_blocks=4,
        reused_blocks=0,
        missed_blocks=4,
        cache_hit_rate=0.0,
    )
    kv_cache_manager.get_iteration_stats.return_value = {32: "deltas"}

    fake = MagicMock()
    fake.max_num_active_requests = 8
    fake.enable_attention_dp = False
    fake.executor_request_queue.get_request_queue_size.return_value = 0
    fake.executor_request_queue.get_request_queue.return_value.queue = []
    fake.resource_manager.resource_managers.get.return_value = kv_cache_manager
    fake._kv_iter_stats_interval = kv_interval
    fake._last_kv_iter_stats_fetch_iter = None
    fake._is_stats_dummy_request = PyExecutor._is_stats_dummy_request
    return fake, kv_cache_manager


def _update(fake, stats_iter: int):
    stats = IterationStats()
    stats.inflight_batching_stats = InflightBatchingStats()
    stats.iter = stats_iter
    scheduled_batch = types.SimpleNamespace(
        context_requests=[],
        generation_requests=[],
        paused_requests=[],
        recompute_paused_requests=[],
        num_context_requests=0,
        num_generation_requests=0,
    )
    with patch(
        "tensorrt_llm._torch.pyexecutor.py_executor.torch.cuda.mem_get_info",
        return_value=(1 << 30, 1 << 30),
    ):
        PyExecutor._update_iter_stats(
            fake,
            stats,
            iter_latency_ms=1.0,
            num_completed_requests=0,
            scheduled_batch=scheduled_batch,
            micro_batch_id=0,
        )
    return fake._latest_kv_iter_stats


def test_kv_iteration_stats_follow_construction_iter_not_live_counter():
    """Overlap consumes batch k in loop k+1; the KV interval must match k."""
    fake, kv_cache_manager = _kv_update_fake(kv_interval=4)
    fetched = {}
    for stats_iter in (0, 4, 8):
        # Live counter one ahead, as under the overlap scheduler.
        fake.iter_counter = stats_iter + 1
        fetched[stats_iter] = _update(fake, stats_iter)
    assert fetched == {0: {32: "deltas"}, 4: {32: "deltas"}, 8: {32: "deltas"}}
    assert kv_cache_manager.get_iteration_stats.call_count == 3


def test_kv_iteration_stats_skip_non_interval_records_and_duplicates():
    fake, kv_cache_manager = _kv_update_fake(kv_interval=4)
    fake.iter_counter = 100
    assert _update(fake, 6) is None
    assert _update(fake, 8) == {32: "deltas"}
    # A second record for the same iteration must not drain the deltas again.
    assert _update(fake, 8) is None
    assert kv_cache_manager.get_iteration_stats.call_count == 1
