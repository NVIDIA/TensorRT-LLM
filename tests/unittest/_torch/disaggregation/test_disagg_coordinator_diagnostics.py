# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Telemetry through the progress/error paths moved into the coordinator.

Use the real coordinator and transfer manager with the existing contract fakes.
FakeDist checks collective ordering and payloads, not real GPU communication.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from coordinator_harness import CoordinatorHarness, TransferRequest
from fake_dist import FakeDistGroup

from tensorrt_llm._torch.disaggregation import diagnostics as disagg_diagnostics
from tensorrt_llm._torch.disaggregation.orchestration import coordinator as coordinator_module
from tensorrt_llm._torch.disaggregation.orchestration.admission import (
    DisaggTransferAdmissionController,
)
from tensorrt_llm.bindings import LlmRequestState

pytestmark = pytest.mark.cpu_only


@pytest.fixture(params=["disabled", "enabled", "raises"])
def diagnostics(request, monkeypatch) -> tuple[bool, Mock]:
    enabled = request.param != "disabled"
    emit = Mock(
        side_effect=RuntimeError("diagnostic sink failed") if request.param == "raises" else None
    )
    monkeypatch.setattr(disagg_diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", enabled)
    monkeypatch.setattr(disagg_diagnostics, "emit_event", emit)
    monkeypatch.setattr(coordinator_module, "is_disagg_inflight_cancel_enabled", lambda: False)
    monkeypatch.delenv("TRTLLM_DISAGG_BENCHMARK_GEN_ONLY", raising=False)
    monkeypatch.delenv("TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP", raising=False)
    return enabled, emit


def _harness(group: FakeDistGroup, rank: int = 0, **kwargs) -> CoordinatorHarness:
    dist = group.rank(rank)
    dist.pp_rank = dist.cp_rank = 0
    h = CoordinatorHarness(dist=dist, **kwargs)
    h.kv_cache_manager.mapping = SimpleNamespace(rank=rank)
    # Match the real KV manager's block-ID collection, rather than the harness's
    # scalar placeholder, so the unpin diagnostic must actually reach emit_event.
    h.kv_cache_manager.store_blocks_for_reuse.side_effect = lambda req, _: [req.py_request_id]
    return h


def _request(h: CoordinatorHarness) -> TransferRequest:
    req = TransferRequest(
        7, prompt_len=128, py_disaggregated_params=SimpleNamespace(disagg_request_id=7007)
    )
    h.active.append(req)
    return req


def _assert_trace(diagnostics: tuple[bool, Mock], expected: list[tuple[str, int]]) -> None:
    enabled, emit = diagnostics
    if not enabled:
        emit.assert_not_called()
        return
    calls = emit.call_args_list
    assert [(call.args[0], call.kwargs["rank"]) for call in calls] == expected
    for call in calls:
        assert call.kwargs["request_id"] == 7007
        assert call.kwargs["request_id_scope"] == "run"
        assert call.kwargs["local_request_id"] == 7
        assert call.kwargs["side"] == "ctx"


@pytest.mark.parametrize("synchronous", [False, True])
def test_idle_progress_unpins_and_terminates_once(diagnostics, monkeypatch, synchronous) -> None:
    """Both single-rank idle paths reach the unpin edge, even if emitting fails."""
    if synchronous:
        monkeypatch.setenv("TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP", "1")
    h = _harness(FakeDistGroup(world_size=1, tp_size=1))
    req = _request(h)
    h.send(req)
    h.transceiver.finish_send(req)

    h.coordinator.poll_progress_when_idle()
    h.coordinator.poll_progress_when_idle()

    assert not h.in_transfer(req)
    assert req.state == LlmRequestState.DISAGG_CONTEXT_COMPLETE
    assert h.active == []
    assert h.effects.terminated == [req]
    assert h.effects.failed == []
    h.kv_cache_manager.store_blocks_for_reuse.assert_called_once_with(req, True)
    h.kv_cache_manager.unpin_blocks_by_id.assert_called_once_with([7])
    assert h.dist.calls == []
    _assert_trace(diagnostics, [("ctx_send_ready", 0), ("ctx_source_unpinned", 0)])
    enabled, emit = diagnostics
    if enabled:
        unpinned = emit.call_args.kwargs
        assert unpinned["source_kv_reuse_pinned"] is False
        assert unpinned["source_kv_reuse_block_count"] == 1
        assert unpinned["state"] == "DISAGG_CONTEXT_COMPLETE"


def test_idle_error_cleanup_waits_for_last_owner(diagnostics) -> None:
    """No unpin event or error cleanup is allowed while the connector owns KV."""
    h = _harness(FakeDistGroup(world_size=1, tp_size=1))
    req = _request(h)
    h.send(req)
    h.transfers.start_transfer(req)  # A second claim held by the KV connector.
    h.transceiver.finish_send(req, outcome="error")

    h.coordinator.poll_progress_when_idle()

    assert h.in_transfer(req)
    assert req.state == LlmRequestState.DISAGG_TRANS_ERROR
    assert h.effects.failed == []
    h.kv_cache_manager.unpin_blocks_by_id.assert_not_called()
    _assert_trace(diagnostics, [("ctx_send_ready", 0)])

    h.coordinator.release_transfer(req)
    h.coordinator._check_transfer_errors("context requests")

    assert not h.in_transfer(req)
    assert req.state == LlmRequestState.DISAGG_TRANS_ERROR
    assert h.effects.failed == [("Error in kv cache transfer for context requests", [req], False)]
    assert h.effects.terminated == []
    h.kv_cache_manager.unpin_blocks_by_id.assert_called_once_with([7])
    _assert_trace(diagnostics, [("ctx_send_ready", 0), ("ctx_source_unpinned", 0)])
    enabled, emit = diagnostics
    if enabled:
        assert emit.call_args.kwargs["state"] == "DISAGG_TRANS_ERROR"


def test_rank_skew_preserves_error_vote_and_unpin_edges(diagnostics) -> None:
    """Diagnostic failures must not bypass the cross-rank last-owner barrier."""
    group = FakeDistGroup(world_size=2, tp_size=2)
    ranks = [_harness(group, rank, enable_attention_dp=True) for rank in range(2)]
    requests = [_request(h) for h in ranks]
    for h, req in zip(ranks, requests):
        h.send(req)
    requests[0].state = LlmRequestState.DISAGG_TRANS_ERROR
    ranks[1].transceiver.finish_send(requests[1], outcome="error")

    group.run(lambda rank: ranks[rank].coordinator.poll_progress_when_idle())
    group.run(lambda rank: ranks[rank].coordinator.handle_errors_synced())

    assert ranks[0].in_transfer(requests[0])
    assert not ranks[1].in_transfer(requests[1])
    assert all(h.effects.failed == [] for h in ranks)
    ranks[0].kv_cache_manager.unpin_blocks_by_id.assert_not_called()
    assert ranks[0].dist.calls == [("tp_allgather", {"error_ids": [7], "blocked_ids": [7]})]
    assert ranks[1].dist.calls == [("tp_allgather", {"error_ids": [7], "blocked_ids": []})]
    _assert_trace(
        diagnostics, [("ctx_send_ready", 0), ("ctx_send_ready", 1), ("ctx_source_unpinned", 1)]
    )

    ranks[0].transceiver.finish_send(requests[0], outcome="error")
    group.run(lambda rank: ranks[rank].coordinator.poll_progress_when_idle())
    group.run(lambda rank: ranks[rank].coordinator.handle_errors_synced())

    for h, req in zip(ranks, requests):
        assert not h.in_transfer(req)
        assert req.state == LlmRequestState.DISAGG_TRANS_ERROR
        assert h.effects.failed == [("Disagg KV cache transfer error", [req], False)]
        assert h.effects.terminated == []
        h.kv_cache_manager.unpin_blocks_by_id.assert_called_once_with([7])
        assert len(h.dist.calls) == 2
        assert h.dist.calls[1] == ("tp_allgather", {"error_ids": [7], "blocked_ids": []})
    _assert_trace(
        diagnostics,
        [
            ("ctx_send_ready", 0),
            ("ctx_send_ready", 1),
            ("ctx_source_unpinned", 1),
            ("ctx_source_unpinned", 0),
        ],
    )


def test_timeout_observation_and_idle_cancellation_preserve_trace(diagnostics, clock) -> None:
    """The recorded timer start survives timeout observation and idle cleanup."""
    h = _harness(FakeDistGroup(world_size=1, tp_size=1), kv_transfer_timeout_ms=1000)
    req = _request(h)
    h.send(req)
    start = req.py_kv_transfer_start_time
    assert start == clock["t"]
    clock["t"] += 2.0

    h.coordinator.check_transfer_timeouts()
    h.coordinator.check_transfer_timeouts()  # Do not report the same timeout twice.
    h.coordinator.poll_progress_when_idle()
    h.coordinator.poll_progress_when_idle()

    assert req.py_kv_transfer_timed_out
    assert req.py_kv_transfer_start_time is None
    assert not h.in_transfer(req)
    assert req.state == LlmRequestState.DISAGG_CONTEXT_COMPLETE
    assert h.transceiver.call_log.count("cancel_request:7") == 1
    h.kv_cache_manager.unpin_blocks_by_id.assert_called_once_with([7])
    assert h.effects.terminated == [req]
    assert h.dist.calls == []
    _assert_trace(
        diagnostics,
        [
            ("ctx_send_ready", 0),
            ("transfer_timeout_started", 0),
            ("transfer_timeout_observed", 0),
            ("ctx_source_unpinned", 0),
        ],
    )
    enabled, emit = diagnostics
    if enabled:
        started = emit.call_args_list[1].kwargs
        observed = emit.call_args_list[2].kwargs
        assert started["timer_start_monotonic_ns"] == int(start * 1_000_000_000)
        assert observed["timer_start_monotonic_ns"] == started["timer_start_monotonic_ns"]
        assert started["timeout_owner"] == observed["timeout_owner"] == "pyexecutor"
        assert started["timeout_ms"] == observed["timeout_ms"] == 1000
        assert observed["elapsed_ms"] == 2000.0


@pytest.mark.parametrize("bypass", [False, True])
def test_admission_diagnostics_preserve_admitted_batch_and_kv_rollback(
    diagnostics, bypass: bool
) -> None:
    controller = DisaggTransferAdmissionController(max_tokens_in_buffer=32, tokens_per_block=32)
    h = _harness(
        FakeDistGroup(world_size=1, tp_size=1),
        admission_controller=controller,
        is_kv_manager_v2=True,
        consumes_transfer_buffer=not bypass,
    )
    candidates = [
        TransferRequest(
            rid,
            prompt_len=32,
            is_context_only_request=False,
            state=LlmRequestState.DISAGG_GENERATION_INIT,
            py_disaggregated_params=SimpleNamespace(disagg_request_id=7000 + rid),
        )
        for rid in (7, 8)
    ]
    h.active.extend(candidates)

    admitted, waiting_for_progress = h.coordinator.admit(candidates)

    assert admitted == (candidates if bypass else candidates[:1])
    assert waiting_for_progress is False
    assert h.effects.reverted == ([] if bypass else [candidates[1:]])
    assert h.effects.prepared == []
    assert h.transceiver.call_log == []
    assert all(req.state == LlmRequestState.DISAGG_GENERATION_INIT for req in candidates)
    enabled, emit = diagnostics
    if not enabled:
        emit.assert_not_called()
        return
    window_events = [
        call for call in emit.call_args_list if call.args[0] == "gen_transfer_window_result"
    ]
    assert window_events
    assert window_events[0].kwargs["policy"] == ("bypassed" if bypass else "enforced")
    assert window_events[0].kwargs["request_id"] == 7007
    assert window_events[0].kwargs["request_id_scope"] == "run"
    rollback_events = [call for call in emit.call_args_list if call.args[0] == "gen_kv_rollback"]
    assert len(rollback_events) == (0 if bypass else 1)
    if rollback_events:
        assert rollback_events[0].kwargs["request_id"] == 7008
        assert rollback_events[0].kwargs["reason"] == "transfer_window"


@pytest.mark.parametrize("synchronous", [False, True])
def test_receive_diagnostics_preserve_preparation_poll_and_timeout_boundaries(
    diagnostics, monkeypatch, clock, synchronous: bool
) -> None:
    if synchronous:
        monkeypatch.setenv("TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP", "1")
    h = _harness(FakeDistGroup(world_size=1, tp_size=1), kv_transfer_timeout_ms=1000)
    request = TransferRequest(
        7,
        is_context_only_request=False,
        state=LlmRequestState.DISAGG_GENERATION_INIT,
        py_disaggregated_params=SimpleNamespace(disagg_request_id=7007),
    )
    h.active.append(request)

    h.coordinator.receive_gen_init([request])

    assert h.effects.prepared == [[request]]
    assert h.effects.reverted == []
    assert h.effects.failed == []
    if synchronous:
        assert h.transceiver.call_log == ["request_and_receive_sync:7"]
        assert request.state == LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE
        assert request.py_kv_transfer_start_time is None
    else:
        assert h.transceiver.call_log == [
            "request_and_receive_async:7",
            "check_gen_transfer_status:0",
        ]
        assert request.state == LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
        assert request.py_kv_transfer_start_time == clock["t"]
    enabled, emit = diagnostics
    if synchronous or not enabled:
        emit.assert_not_called()
    else:
        assert emit.call_count == 1
        assert emit.call_args.args == ("transfer_timeout_started",)
        assert emit.call_args.kwargs["request_id"] == 7007
        assert emit.call_args.kwargs["request_id_scope"] == "run"
        assert emit.call_args.kwargs["side"] == "gen"
        assert emit.call_args.kwargs["timer_start_monotonic_ns"] == int(clock["t"] * 1_000_000_000)
