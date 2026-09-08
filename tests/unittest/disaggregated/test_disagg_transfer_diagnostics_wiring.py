# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU-only wiring tests for disaggregated-transfer diagnostic events."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import tensorrt_llm._torch.disaggregation.native.transfer as transfer_module
from tensorrt_llm import DisaggregatedParams
from tensorrt_llm._torch.disaggregation import diagnostics
from tensorrt_llm._torch.disaggregation.base.transfer import KVSlice
from tensorrt_llm._torch.disaggregation.native.transfer import (
    AgentResult,
    Receiver,
    RxSession,
    TaskStatus,
)
from tensorrt_llm._torch.disaggregation.orchestration import coordinator as coordinator_module
from tensorrt_llm._torch.disaggregation.orchestration.admission import (
    DisaggTransferAdmissionController,
)
from tensorrt_llm._torch.disaggregation.orchestration.coordinator import DisaggTransferCoordinator
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import KVCacheV2Scheduler, ScheduleAction

pytestmark = pytest.mark.cpu_only


def _disagg_request(local_id: int, canonical_id: int, prompt_len: int = 65):
    return SimpleNamespace(
        py_request_id=local_id,
        request_id=local_id,
        py_disaggregated_params=SimpleNamespace(disagg_request_id=canonical_id),
        prompt_len=prompt_len,
    )


def test_scheduler_emits_admitted_and_deferred_kv_admission_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admitted = _disagg_request(11, 1011)
    deferred = _disagg_request(12, 1012)
    results = iter((True, False))
    manager = SimpleNamespace(
        prepare_disagg_gen_init=lambda _request: next(results),
        kv_cache_map={
            admitted.py_request_id: SimpleNamespace(capacity=65, history_length=64),
        },
        mapping=SimpleNamespace(rank=3),
    )
    scheduler = object.__new__(KVCacheV2Scheduler)
    scheduler.kv_cache_manager = manager
    scheduler.tokens_per_block = 32

    emit_event = Mock()
    monkeypatch.setattr(diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", True)
    monkeypatch.setattr(diagnostics, "emit_event", emit_event)

    admitted_result = scheduler._try_schedule_disagg_gen_init(admitted, None)
    deferred_result = scheduler._try_schedule_disagg_gen_init(deferred, None)

    assert admitted_result == (ScheduleAction.SCHEDULED, 0)
    assert deferred_result == (ScheduleAction.SKIP, 0)
    assert emit_event.call_count == 2

    admitted_event = emit_event.call_args_list[0]
    assert admitted_event.args == ("gen_kv_admission_result",)
    assert admitted_event.kwargs == {
        "side": "gen",
        "request_id": 1011,
        "local_request_id": 11,
        "rank": 3,
        "outcome": "admitted",
        "reason": None,
        "prompt_tokens": 65,
        "tokens_per_block": 32,
        "cache_present": True,
        "capacity_tokens": 65,
        "history_tokens": 64,
        "capacity_block_equivalent": 3,
    }

    deferred_event = emit_event.call_args_list[1]
    assert deferred_event.args == ("gen_kv_admission_result",)
    assert deferred_event.kwargs["request_id"] == 1012
    assert deferred_event.kwargs["local_request_id"] == 12
    assert deferred_event.kwargs["outcome"] == "deferred"
    assert deferred_event.kwargs["reason"] == "kv_or_index_capacity"
    assert deferred_event.kwargs["cache_present"] is False
    assert deferred_event.kwargs["capacity_tokens"] is None
    assert deferred_event.kwargs["capacity_block_equivalent"] is None


def test_scheduler_kv_pool_snapshot_reports_pressure_and_skips_irrelevant_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    get_stats = Mock(
        return_value=SimpleNamespace(
            max_num_blocks=100,
            free_num_blocks=40,
            used_num_blocks=60,
        ),
    )
    manager = SimpleNamespace(
        get_kv_cache_stats=get_stats,
        mapping=SimpleNamespace(rank=3, tp_rank=1, pp_rank=0, cp_rank=0),
        index_mapper=SimpleNamespace(num_free_slots=lambda: 7),
    )
    scheduler = object.__new__(KVCacheV2Scheduler)
    scheduler.kv_cache_manager = manager
    emit_event = Mock()
    monkeypatch.setattr(diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", True)
    monkeypatch.setattr(diagnostics, "emit_event", emit_event)

    decode = SimpleNamespace(
        is_disagg_generation_init_state=False,
        is_disagg_generation_transmission_in_progress=False,
        is_disagg_generation_transmission_complete=False,
        state=LlmRequestState.GENERATION_IN_PROGRESS,
    )
    scheduler._emit_disagg_kv_pool_snapshot([decode], [])

    get_stats.assert_not_called()
    emit_event.assert_not_called()

    pending = SimpleNamespace(
        is_disagg_generation_init_state=True,
        is_disagg_generation_transmission_in_progress=False,
        is_disagg_generation_transmission_complete=False,
        state=LlmRequestState.DISAGG_GENERATION_INIT,
    )
    transferring = SimpleNamespace(
        is_disagg_generation_init_state=False,
        is_disagg_generation_transmission_in_progress=True,
        is_disagg_generation_transmission_complete=False,
        state=LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS,
    )
    transferred = SimpleNamespace(
        is_disagg_generation_init_state=False,
        is_disagg_generation_transmission_in_progress=False,
        is_disagg_generation_transmission_complete=True,
        state=LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE,
    )
    scheduler._emit_disagg_kv_pool_snapshot(
        [pending, transferring, transferred, decode],
        [pending],
    )

    get_stats.assert_called_once_with()
    event = emit_event.call_args
    assert event.args == ("gen_kv_pool_snapshot",)
    assert event.kwargs == {
        "side": "gen",
        "request_id": None,
        "rank": 3,
        "init_requests": 1,
        "transfers_in_progress": 1,
        "transfers_complete": 1,
        "kv_admitted_this_iteration": 1,
        "decode_requests": 1,
        "kv_pool_max_blocks": 100,
        "kv_pool_free_blocks": 40,
        "kv_pool_used_blocks": 60,
        "index_free_slots": 7,
        "tp_rank": 1,
        "pp_rank": 0,
        "cp_rank": 0,
    }


def test_gen_timeout_start_and_observation_share_request_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _disagg_request(21, 2021)
    request.state = LlmRequestState.DISAGG_GENERATION_INIT
    request.py_kv_transfer_start_time = None
    request.py_kv_transfer_timed_out = False
    request.is_disagg_generation_transmission_in_progress = False

    def start_receive(req) -> None:
        req.state = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
        req.is_disagg_generation_transmission_in_progress = True

    transceiver = Mock()
    transceiver.kv_transfer_timeout_ms = 100
    transceiver.request_and_receive_async.side_effect = start_receive

    executor = object.__new__(PyExecutor)
    executor.kv_cache_transceiver = transceiver
    executor.global_rank = 4
    executor.dist = SimpleNamespace(tp_rank=1, pp_rank=0, cp_rank=0)
    executor._is_disagg_gen_only_no_context_benchmark = Mock(return_value=False)
    executor._uses_async_disagg_gen_transfer = Mock(return_value=True)
    executor._disagg_coordinator = SimpleNamespace(reap_gen_receives=Mock())

    coordinator = DisaggTransferCoordinator(
        transceiver=transceiver,
        transfer_manager=SimpleNamespace(requests_in_transfer=lambda: {}),
        kv_cache_manager=None,
        dist=SimpleNamespace(rank=4, tp_rank=1, pp_rank=0, cp_rank=0),
        effects=None,
        registry=SimpleNamespace(active_requests=lambda: [request]),
        enable_attention_dp=False,
        force_terminate_ctx_for_partial_reuse=False,
        delegates=None,
    )

    emit_event = Mock()
    monkeypatch.setattr(diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", True)
    monkeypatch.setattr(diagnostics, "emit_event", emit_event)
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.py_executor.time.monotonic",
        lambda: 10.0,
    )
    monkeypatch.setattr(coordinator_module.time, "monotonic", lambda: 10.2)
    monkeypatch.setattr(
        coordinator_module,
        "is_disagg_inflight_cancel_enabled",
        lambda: False,
    )

    executor._recv_disagg_gen_cache([request])
    coordinator.check_transfer_timeouts()

    assert request.py_kv_transfer_start_time == 10.0
    assert request.py_kv_transfer_timed_out
    assert [entry.args[0] for entry in emit_event.call_args_list] == [
        "transfer_timeout_started",
        "transfer_timeout_observed",
    ]
    started = emit_event.call_args_list[0].kwargs
    observed = emit_event.call_args_list[1].kwargs
    assert started["side"] == observed["side"] == "gen"
    assert started["request_id"] == observed["request_id"] == 2021
    assert started["local_request_id"] == observed["local_request_id"] == 21
    assert started["timeout_owner"] == observed["timeout_owner"] == "pyexecutor"
    assert started["timer_start_monotonic_ns"] == 10_000_000_000
    assert observed["timer_start_monotonic_ns"] == 10_000_000_000
    assert observed["elapsed_ms"] == pytest.approx(200.0)
    assert observed["cancellation_requested"] is False


def test_ctx_send_ready_and_timeout_start_follow_transfer_handoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operations = []
    request = SimpleNamespace(
        is_context_only_request=True,
        is_finished_due_to_cancellation=False,
        is_child=False,
        py_request_id=31,
        request_id=31,
        py_disaggregated_params=SimpleNamespace(disagg_request_id=3031),
        is_context_finished=True,
        is_finished_due_to_length=False,
        prompt_len=128,
        state=SimpleNamespace(name="CONTEXT_IN_PROGRESS"),
        py_kv_transfer_start_time=None,
    )
    transfer_manager = SimpleNamespace(
        start_transfer=lambda req: operations.append(("start_transfer", req)),
        should_store_blocks=True,
    )
    transceiver = SimpleNamespace(
        has_retired_send_session=lambda _req: False,
        respond_and_send_async=lambda req: operations.append(("respond", req)),
        kv_transfer_timeout_ms=100,
        pipeline_transfer_enabled=False,
    )
    coordinator = DisaggTransferCoordinator(
        transceiver=transceiver,
        transfer_manager=transfer_manager,
        kv_cache_manager=None,
        dist=SimpleNamespace(rank=4, tp_rank=1, pp_rank=0, cp_rank=0),
        effects=None,
        registry=SimpleNamespace(canceled_request_ids=lambda: []),
        enable_attention_dp=False,
        force_terminate_ctx_for_partial_reuse=False,
        delegates=None,
    )

    def emit_event(event: str, **kwargs) -> None:
        operations.append((event, kwargs))

    def monotonic() -> float:
        operations.append(("monotonic", None))
        return 12.5

    monkeypatch.setattr(diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", True)
    monkeypatch.setattr(diagnostics, "emit_event", emit_event)
    monkeypatch.setattr(coordinator_module.time, "monotonic", monotonic)

    coordinator.send_completed_context([request])

    assert [operation[0] for operation in operations] == [
        "start_transfer",
        "ctx_send_ready",
        "respond",
        "monotonic",
        "transfer_timeout_started",
    ]
    send_ready = operations[1][1]
    timeout_started = operations[4][1]
    assert send_ready["request_id"] == timeout_started["request_id"] == 3031
    assert send_ready["source_kv_request_owned"] is True
    assert send_ready["source_kv_reuse_pinned"] is True
    assert timeout_started["timeout_ms"] == 100
    assert timeout_started["timer_start_monotonic_ns"] == 12_500_000_000
    assert request.py_kv_transfer_start_time == 12.5


def test_bypassed_transfer_window_reports_legacy_budget_counterfactual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = _disagg_request(20, 2020, prompt_len=32)
    active.is_disagg_generation_transmission_in_progress = True
    candidates = [
        _disagg_request(21, 2021, prompt_len=64),
        _disagg_request(22, 2022, prompt_len=32),
    ]
    controller = DisaggTransferAdmissionController(
        max_tokens_in_buffer=64,
        tokens_per_block=32,
    )
    executor = object.__new__(PyExecutor)
    executor.active_requests = [active]
    executor.global_rank = 4
    executor.dist = SimpleNamespace(tp_rank=1, pp_rank=0, cp_rank=0)
    executor._is_disagg_gen_only_no_context_benchmark = Mock(return_value=False)
    executor._get_disagg_transfer_admission_controller = Mock(return_value=controller)
    executor._disagg_transfer_window_is_active = Mock(return_value=False)
    executor._is_disagg_transfer_window_bypass_eligible = Mock(return_value=True)

    emit_event = Mock()
    monkeypatch.setattr(diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", True)
    monkeypatch.setattr(diagnostics, "emit_event", emit_event)

    admitted, wait_for_progress = executor._apply_disagg_transfer_admission(candidates)

    assert admitted == candidates
    assert wait_for_progress is False
    assert emit_event.call_count == 2
    for event, request in zip(emit_event.call_args_list, candidates):
        assert event.args == ("gen_transfer_window_result",)
        assert event.kwargs["request_id"] == request.py_disaggregated_params.disagg_request_id
        assert event.kwargs["outcome"] == "admitted"
        assert event.kwargs["policy"] == "bypassed"
        assert event.kwargs["legacy_budget_outcome"] == "deferred"
        assert event.kwargs["legacy_active_transfer_blocks"] == 1
        assert event.kwargs["legacy_admitted_transfer_blocks"] == 0
        assert event.kwargs["legacy_limited_by_budget"] is True
        assert event.kwargs["transfer_block_budget"] == 2


def test_disabled_diagnostics_do_not_evaluate_bypass_counterfactual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _disagg_request(21, 2021)
    controller = Mock()
    controller.enabled.return_value = True
    controller.select.side_effect = AssertionError(
        "disabled diagnostics evaluated the legacy transfer window"
    )
    executor = object.__new__(PyExecutor)
    executor.active_requests = []
    executor._is_disagg_gen_only_no_context_benchmark = Mock(return_value=False)
    executor._get_disagg_transfer_admission_controller = Mock(return_value=controller)
    executor._disagg_transfer_window_is_active = Mock(return_value=False)
    monkeypatch.setattr(diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", False)

    admitted, wait_for_progress = executor._apply_disagg_transfer_admission([candidate])

    assert admitted == [candidate]
    assert wait_for_progress is False
    controller.select.assert_not_called()


def test_native_writer_result_precedes_local_destination_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_id = 3031
    writer_rank = 7
    receiver = object.__new__(Receiver)
    receiver._enforce_physical_ownership = True
    receiver._sessions = {}
    receiver._sessions_lock = threading.Lock()
    receiver._pre_cancelled_rids = set()
    receiver._shutdown = True
    receiver._bounce = Mock()
    receiver._bounce.is_bounced.return_value = False
    receiver._registrar = SimpleNamespace(
        self_rank_info=SimpleNamespace(instance_name="gen", instance_rank=2),
    )

    session = RxSession(
        request_id=31,
        params=DisaggregatedParams(disagg_request_id=request_id),
        receiver=receiver,
    )

    def dispatch(task) -> None:
        task.expected_transfers = 1
        session.mark_transferring(task.slice_id, writer_cohort={writer_rank})

    receiver.dispatch_task = dispatch
    emit_event = Mock()
    monkeypatch.setattr(diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", True)
    monkeypatch.setattr(diagnostics, "emit_event", emit_event)
    monkeypatch.setattr(
        transfer_module.tensorrt_llm.bindings,
        "global_steady_clock_now",
        lambda: 0,
    )

    session.receive(KVSlice(is_last_slice=True))
    destination_timestamp_captured = threading.Event()
    destination_timestamp = (123_000, 456_000)

    def capture_timestamp() -> tuple[int, int]:
        destination_timestamp_captured.set()
        return destination_timestamp

    task = session._kv_tasks[0]
    original_complete = task.complete

    def complete_after_timestamp_capture() -> None:
        assert destination_timestamp_captured.is_set()
        original_complete()

    monkeypatch.setattr(diagnostics, "capture_timestamp", capture_timestamp)
    monkeypatch.setattr(task, "complete", complete_after_timestamp_capture)
    message = transfer_module._make_kv_result_msg(
        writer_rank,
        request_id,
        0,
        True,
        AgentResult.SUCCESS,
        transfer_size=4096,
    )
    receiver._process_kv_agent_result(b"sender", message)

    assert session._kv_tasks[0].status is TaskStatus.TRANSFERRED
    assert session.resources_drained()
    assert [entry.args[0] for entry in emit_event.call_args_list] == [
        "gen_writer_result_received",
        "gen_destination_complete",
    ]
    writer_event = emit_event.call_args_list[0].kwargs
    destination_event = emit_event.call_args_list[1].kwargs
    assert writer_event["request_id"] == destination_event["request_id"] == request_id
    assert writer_event["slice_id"] == destination_event["slice_id"] == 0
    assert writer_event["peer_rank"] == destination_event["peer_rank"] == writer_rank
    assert writer_event["outcome"] == "success"
    assert writer_event["transfer_bytes"] == 4096
    assert destination_event["outcome"] == "completed"
    assert destination_event["timestamp"] == destination_timestamp
    assert session.close()
