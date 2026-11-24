# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where the non-overlap executor loop commits a context batch's V2 KV.

The context commit must be issued after the request states advance and before
``_update_requests`` waits for the forward, behind the input-copy wait, as the
overlap loop already does; the generation update stays after
``_update_requests``.

The source-order tests read ``py_executor.py`` without importing it, so they
also run where ``tensorrt_llm`` is not built. The loop tests drive one executed
iteration of the real ``PyExecutor._executor_loop`` body (one context request,
then shutdown) with every helper around the forward stubbed; they need an
importable ``tensorrt_llm`` and are skipped without one.
"""

import ast
import pathlib
from unittest.mock import MagicMock, Mock

import pytest

pytestmark = pytest.mark.cpu_only

_PY_EXECUTOR = (
    pathlib.Path(__file__).resolve().parents[4] / "tensorrt_llm/_torch/pyexecutor/py_executor.py"
)


def _self_calls(method: str) -> list:
    """Names of ``self.<name>(...)`` calls in a PyExecutor method, in source order."""
    if not _PY_EXECUTOR.is_file():
        pytest.skip(f"{_PY_EXECUTOR} not found")
    tree = ast.parse(_PY_EXECUTOR.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PyExecutor")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method)
    calls = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
    ]
    calls.sort(key=lambda node: (node.lineno, node.col_offset))
    return [node.func.attr for node in calls]


def test_non_overlap_loop_commits_context_kv_before_the_forward_wait() -> None:
    calls = _self_calls("_executor_loop")
    assert "_update_v2_context_resources" not in calls
    assert calls.count("_commit_ctx_kv_before_sync") == 1
    commit = calls.index("_commit_ctx_kv_before_sync")
    assert calls[commit - 1] == "_update_request_states"
    assert calls[commit + 1] == "_update_requests"
    assert calls.index("_update_v2_generation_resources") > commit + 1


def test_commit_before_sync_waits_for_the_input_copy_first() -> None:
    calls = _self_calls("_commit_ctx_kv_before_sync")
    assert calls == ["_wait_for_model_engine_input_copy", "_update_v2_context_resources"]


def test_overlap_loop_keeps_its_own_context_commit() -> None:
    calls = _self_calls("_executor_loop_overlap")
    assert "_commit_ctx_kv_before_sync" not in calls
    assert "_update_v2_context_resources" in calls


_RECORDED = (
    "_update_request_states",
    "_wait_for_model_engine_input_copy",
    "_update_v2_context_resources",
    "_update_requests",
    "_update_v2_generation_resources",
    "_send_kv_async",
    "_handle_responses",
)


@pytest.fixture
def engine():
    """The modules the loop tests import; skipped where tensorrt_llm cannot be imported."""
    py_executor = pytest.importorskip("tensorrt_llm._torch.pyexecutor.py_executor")
    coordinator = pytest.importorskip(
        "tensorrt_llm._torch.disaggregation.orchestration.coordinator"
    )
    scheduler = pytest.importorskip("tensorrt_llm._torch.pyexecutor.scheduler.scheduler")
    return py_executor, coordinator, scheduler


def _batch(engine, num_context: int):
    batch = engine[2].ScheduledRequests()
    for _ in range(num_context):
        batch.append_context_request(Mock(is_last_context_chunk=True))
    return batch


def _executor(monkeypatch, engine, calls: list, batch, *, v2: bool):
    py_executor, coordinator, _ = engine
    for target in ("torch.cuda.set_device", "cudart.cudaSetDevice", "CUASSERT"):
        monkeypatch.setattr(f"{py_executor.__name__}.{target}", Mock())

    executor = object.__new__(py_executor.PyExecutor)
    executor._disagg_coordinator = coordinator.NoopDisaggCoordinator()
    executor.device_id = 0
    profiler = MagicMock()
    profiler.__enter__.return_value = Mock()
    executor._profiler = Mock(return_value=profiler)
    executor.hang_detector = MagicMock()
    executor.enable_iter_perf_stats = False
    executor.enable_kv_cache_events = False
    executor.enable_attention_dp = False
    executor.is_benchmark_disagg = False
    executor.iter_counter = 0
    executor._resource_governor_enabled = False
    executor._is_kv_manager_v2 = v2
    executor._mm_encoder_item_scheduling_enabled = False
    executor._is_warmup = True
    executor.kv_cache_transceiver = None
    executor.kv_connector_manager = None
    executor.guided_decoder = None
    executor.drafter = None
    executor.speculation_gate = None
    executor.dwdp_manager = None
    executor.active_requests = []
    executor.perf_manager = MagicMock(enabled=False)
    executor.perf_manager.create_timing_events.return_value = (None, None, None)
    executor.resource_manager = MagicMock()
    executor.model_engine = MagicMock()

    schedule = iter([(batch, None), (None, None)])
    executor._prepare_and_schedule_batch = lambda: next(schedule)
    executor._check_benchmark_disagg_gate = Mock(return_value=(True, False))
    executor._can_pause_for_rebalance = Mock(return_value=False)
    executor._can_queue = Mock(return_value=(True, False))
    executor._kv_connector_start_batch = Mock(return_value=(True, False))
    executor._step_scope = MagicMock()
    executor._forward_step = Mock(return_value={"logits": None})
    executor._sample_async = Mock(return_value=Mock())
    for name in (
        "_terminate_recompute_paused_requests",
        "_pause_recompute_paused_requests",
        "_terminate_requests",
        "_pause_requests",
        "_release_unused_connector_reservations",
        "_prepare_disagg_gen_transmission_complete",
        "_handle_dynamic_draft_len",
        "_maybe_record_hang_diagnostic_phase",
        "_revert_gen_alloc",
        "_finalize_adp_dummy_allocation",
        "_commit_kv_cache_stats",
        "_maybe_prefetch_next_iter_mm_encoders",
        "_handle_guided_decoder_errors",
        "_handle_canceled_requests",
        "_kv_connector_terminate_requests",
        "_flush_pending_transfer_responses",
        "_flush_iter_stats_synced",
    ):
        setattr(executor, name, Mock())

    def recorder(name, result=None):
        def record(*args, **kwargs):
            calls.append(name)
            return result

        return record

    for name in _RECORDED:
        setattr(executor, name, recorder(name, [] if name == "_handle_responses" else None))
    return executor


def test_loop_issues_the_context_commit_before_the_forward_wait(monkeypatch, engine) -> None:
    calls = []
    executor = _executor(monkeypatch, engine, calls, _batch(engine, 1), v2=True)

    engine[0].PyExecutor._executor_loop(executor)

    assert calls == [
        "_update_request_states",
        "_wait_for_model_engine_input_copy",
        "_update_v2_context_resources",
        "_update_requests",
        "_update_v2_generation_resources",
        "_send_kv_async",
        "_handle_responses",
    ]


def test_loop_without_context_requests_skips_the_commit_and_the_input_wait(
    monkeypatch, engine
) -> None:
    calls = []
    executor = _executor(monkeypatch, engine, calls, _batch(engine, 0), v2=True)

    engine[0].PyExecutor._executor_loop(executor)

    assert calls == [
        "_update_request_states",
        "_update_requests",
        "_update_v2_generation_resources",
        "_send_kv_async",
        "_handle_responses",
    ]


def test_loop_with_the_v1_kv_manager_keeps_its_sequence(monkeypatch, engine) -> None:
    calls = []
    executor = _executor(monkeypatch, engine, calls, _batch(engine, 1), v2=False)

    engine[0].PyExecutor._executor_loop(executor)

    assert calls == [
        "_update_request_states",
        "_update_requests",
        "_send_kv_async",
        "_handle_responses",
    ]


def test_commit_before_sync_logs_its_witness_once(monkeypatch, engine) -> None:
    calls = []
    executor = _executor(monkeypatch, engine, calls, _batch(engine, 1), v2=True)
    seen = []
    monkeypatch.setattr(
        engine[0].logger, "info_once", lambda *msg, key: seen.append((" ".join(msg), key))
    )

    engine[0].PyExecutor._commit_ctx_kv_before_sync(executor, _batch(engine, 2))
    engine[0].PyExecutor._commit_ctx_kv_before_sync(executor, _batch(engine, 0))

    assert calls == ["_wait_for_model_engine_input_copy", "_update_v2_context_resources"]
    assert seen == [
        (
            "V2 context KV commit issued before the step sync",
            "ctx_kv_before_sync",
        )
    ]
