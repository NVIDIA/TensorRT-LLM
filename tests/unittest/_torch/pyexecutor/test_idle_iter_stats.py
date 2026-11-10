# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scheduler-confirmed idle and queue-only snapshots without model execution."""

import datetime
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def executor():
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    ifb = SimpleNamespace(num_queued_context_requests=0, num_queued_gen_requests=0)
    stats = SimpleNamespace(inflight_batching_stats=ifb, iter_latency_ms=0.0)
    stub = SimpleNamespace(
        enable_iter_perf_stats=True,
        is_shutdown=False,
        iter_counter=7,
        iter_perf_stats_interval=1,
        _quiescent_stats_iter=None,
        _latest_kv_iter_stats={1: object()},
        enable_attention_dp=False,
        dist=SimpleNamespace(rank=0, pp_size=1),
        _get_init_iter_stats=Mock(return_value=stats),
        _update_iter_stats=Mock(side_effect=lambda row, *args: row),
        _append_iter_stats=Mock(),
        _adp_iter_stats=SimpleNamespace(queue=Mock()),
    )
    stub._maybe_emit_quiescent_stats = PyExecutor._maybe_emit_quiescent_stats.__get__(stub)
    return stub


def test_idle_snapshot_has_no_fabricated_iteration_time(executor):
    executor._maybe_emit_quiescent_stats(engine_idle=True)
    args = executor._update_iter_stats.call_args.args
    assert args[1:3] == (0.0, 0)
    assert args[3].batch_size == 0
    executor._append_iter_stats.assert_called_once()


def test_busy_empty_snapshot_is_not_idle(executor):
    executor._maybe_emit_quiescent_stats(engine_idle=False)
    executor._append_iter_stats.assert_not_called()


def test_blocked_prefill_snapshot_keeps_queued_work(executor):
    stats = executor._get_init_iter_stats.return_value
    stats.inflight_batching_stats.num_queued_context_requests = 2
    executor._maybe_emit_quiescent_stats(engine_idle=False)
    executor._append_iter_stats.assert_called_once()


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("engine_idle", [False, True])
def test_adp_retains_the_global_idle_decision(executor, rank, engine_idle):
    executor.enable_attention_dp = True
    executor.dist.rank = rank
    executor._maybe_emit_quiescent_stats(engine_idle=engine_idle)
    executor._adp_iter_stats.queue.assert_called_once_with(
        executor._get_init_iter_stats.return_value,
        is_rank0=rank == 0,
        kv_iter_stats=executor._latest_kv_iter_stats,
        host_step_time_ms=0.0,
        prev_device_step_time_ms=0.0,
        gpu_forward_time_ms=0.0,
        allow_empty_heartbeat=engine_idle,
    )
    executor._append_iter_stats.assert_not_called()


def test_idle_wakeup_and_no_forward_pass_do_not_duplicate_stats(executor):
    executor._maybe_emit_quiescent_stats(engine_idle=True)
    executor._maybe_emit_quiescent_stats(engine_idle=False)
    executor._update_iter_stats.assert_called_once()
    executor.iter_counter += 1
    executor._maybe_emit_quiescent_stats(engine_idle=True)
    assert executor._update_iter_stats.call_count == 2


@pytest.mark.parametrize("enabled,shutdown", [(False, False), (True, True)])
def test_disabled_or_shutdown_does_not_emit(executor, enabled, shutdown):
    executor.enable_iter_perf_stats = enabled
    executor.is_shutdown = shutdown
    executor._maybe_emit_quiescent_stats(engine_idle=True)
    executor._update_iter_stats.assert_not_called()


def test_queue_only_sampling_does_not_throttle_idle_heartbeats(executor):
    executor.iter_perf_stats_interval = 10
    executor._maybe_emit_quiescent_stats(engine_idle=False)
    executor._update_iter_stats.assert_not_called()
    executor._maybe_emit_quiescent_stats(engine_idle=True)
    executor._append_iter_stats.assert_called_once()


class _WaitingQueue(list):
    def add_requests(self, requests):
        self.extend(requests)


@pytest.mark.parametrize(
    "live,waiting,pending,new_items,control,stats_busy,expected_idle",
    [
        (0, False, False, False, False, False, True),
        (1, False, False, False, False, False, False),
        (0, True, False, False, False, False, False),
        (0, False, True, False, False, False, False),
        (0, False, False, True, False, False, False),
        (0, False, False, False, True, False, False),
        (0, False, False, False, False, True, False),
    ],
    ids=[
        "all-ranks-idle",
        "peer-busy",
        "waiting",
        "transfer",
        "arrival",
        "control",
        "peer-pending-work",
    ],
)
def test_idle_fetch_requires_all_ranks_to_drain(
    live, waiting, pending, new_items, control, stats_busy, expected_idle
):
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    queue = _WaitingQueue([object()] if waiting else [])
    incoming = [object()] if new_items else []
    stub = SimpleNamespace(
        kv_connector_manager=None,
        control_requests=[object()] if control else [],
        enable_attention_dp=True,
        is_shutdown=False,
        enable_iter_perf_stats=True,
        _disable_mpi=False,
        dist=SimpleNamespace(rank=0, pp_size=1),
        _has_pending_connector_transfers=lambda: pending,
        _stats_has_pending_work=stats_busy,
        request_accumulated=[],
        hang_detector=SimpleNamespace(pause=nullcontext),
        executor_request_queue=SimpleNamespace(get_from_request_queue=Mock(return_value=incoming)),
        request_broadcaster=SimpleNamespace(broadcast=Mock(side_effect=lambda items, **kw: items)),
        _handle_special_queue_items=lambda items: items,
        _maybe_emit_quiescent_stats=Mock(),
    )
    PyExecutor._fetch_and_enqueue_requests(stub, queue, live)
    if expected_idle:
        stub._maybe_emit_quiescent_stats.assert_called_once_with(engine_idle=True)
        stub.executor_request_queue.get_from_request_queue.assert_called_once_with(
            datetime.timedelta(seconds=0.5)
        )
    else:
        stub._maybe_emit_quiescent_stats.assert_not_called()
    assert bool(queue) == (waiting or new_items)


@pytest.mark.parametrize(
    "previous,pp_pending,native_transfer,control",
    [
        (False, False, False, False),
        (True, False, False, False),
        (False, True, False, False),
        (False, False, True, False),
        (False, False, False, True),
    ],
    ids=["drained", "model-batch", "pipeline", "native-kv-send", "control"],
)
def test_pending_work_includes_model_batches_and_native_transfers(
    previous, pp_pending, native_transfer, control
):
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    stub = SimpleNamespace(
        previous_batch=object() if previous else None,
        dist=SimpleNamespace(pp_size=1 if previous else 2),
        _pp_ring_is_drained=lambda: not pp_pending,
        control_requests=[object()] if control else [],
        async_transfer_manager=SimpleNamespace(has_any_inflight_requests=lambda: native_transfer),
    )
    assert PyExecutor._has_pending_iteration_stats_work(stub) == (
        previous or pp_pending or native_transfer or control
    )


def test_drained_pipeline_ignores_retained_previous_batch():
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    stub = SimpleNamespace(
        previous_batch=object(),
        dist=SimpleNamespace(pp_size=2),
        _pp_ring_is_drained=lambda: True,
        control_requests=[],
        async_transfer_manager=None,
    )
    assert not PyExecutor._has_pending_iteration_stats_work(stub)
