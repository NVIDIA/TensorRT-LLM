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

import queue
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import tensorrt_llm._torch.disaggregation.native.transfer as transfer_module
from tensorrt_llm import DisaggregatedParams
from tensorrt_llm._torch.disaggregation import diagnostics
from tensorrt_llm._torch.disaggregation.base.transfer import KVSlice, SessionStatus, WaitResult
from tensorrt_llm._torch.disaggregation.native.transfer import (
    AgentResult,
    Receiver,
    RxSession,
    Sender,
    TaskStatus,
)
from tensorrt_llm._torch.disaggregation.orchestration import coordinator as coordinator_module
from tensorrt_llm._torch.disaggregation.orchestration.admission import (
    DisaggTransferAdmissionController,
)
from tensorrt_llm._torch.disaggregation.orchestration.coordinator import DisaggTransferCoordinator
from tensorrt_llm._torch.disaggregation.orchestration.transfer_manager import AsyncTransferManager
from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import KVCacheV2Scheduler, ScheduleAction

pytestmark = pytest.mark.cpu_only


def _disagg_request(local_id: int, canonical_id: int, prompt_len: int = 65):
    return SimpleNamespace(
        py_request_id=local_id,
        request_id=local_id,
        py_disaggregated_params=SimpleNamespace(disagg_request_id=canonical_id),
        prompt_len=prompt_len,
    )


def _diagnostic_transceiver() -> KvCacheTransceiverV2:
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._mapping = SimpleNamespace(rank=4, tp_rank=1, pp_rank=0, cp_rank=0)
    transceiver._instance_name = "diagnostic-test"
    transceiver._dp_rank = 2
    return transceiver


def test_sender_reports_worker_dequeue_before_transfer_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_queue = queue.Queue()
    write_meta = SimpleNamespace(
        meta_type=transfer_module.WriteMetaType.KV,
        src_ptrs=SimpleNamespace(size=1),
        sizes=SimpleNamespace(sum=lambda: 4096),
        unique_rid=1010,
        slice_id=2,
        peer_rank=3,
        receiver_slice_id=4,
        is_last_slice=True,
    )
    task_queue.put(write_meta)
    task_queue.put(None)
    sender = object.__new__(Sender)
    sender._device_id = 0
    sender._send_task_queues = [task_queue]
    sender._registrar = SimpleNamespace(self_rank_info=SimpleNamespace())
    sender._thread_local = threading.local()
    operations = []

    def emit_event(event: str, **kwargs) -> None:
        operations.append((event, kwargs))

    sender._deliver_kv_to_agent = Mock(
        side_effect=lambda meta: operations.append(("deliver", meta))
    )
    monkeypatch.setattr(diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", True)
    monkeypatch.setattr(diagnostics, "emit_event", emit_event)
    monkeypatch.setattr(transfer_module.torch.cuda, "set_device", Mock())
    monkeypatch.setattr(transfer_module, "CUASSERT", Mock())
    monkeypatch.setattr(
        transfer_module,
        "cudart",
        SimpleNamespace(cudaSetDevice=Mock(return_value=0)),
    )

    sender._process_task_queue(0)

    assert [operation[0] for operation in operations] == [
        "ctx_worker_dequeued",
        "deliver",
    ]
    event = operations[0][1]
    assert event["request_id"] == 1010
    assert event["slice_id"] == 2
    assert event["peer_rank"] == 3
    assert event["worker_queue_index"] == 0
    assert event["transfer_bytes"] == 4096


def test_gen_ingress_uses_full_cp_prompt_length(monkeypatch: pytest.MonkeyPatch) -> None:
    request = _disagg_request(10, 1010, prompt_len=1)
    request.total_input_len_cp = 257
    request.is_disagg_generation_init_state = True
    request.state = LlmRequestState.DISAGG_GENERATION_INIT
    executor = object.__new__(PyExecutor)
    executor.waiting_queue = []
    executor.active_requests = []
    executor._fetch_new_requests = Mock(return_value=[request])
    executor._validate_request = Mock()
    executor._mm_encoder_item_scheduling_enabled = False
    executor.global_rank = 4
    executor.dist = SimpleNamespace(tp_rank=1, pp_rank=0, cp_rank=3)

    emit_event = Mock()
    monkeypatch.setattr(diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", True)
    monkeypatch.setattr(diagnostics, "emit_event", emit_event)

    assert executor._fetch_and_activate_new_requests() == [request]
    assert executor.active_requests == [request]
    event = emit_event.call_args
    assert event.args == ("gen_ingress",)
    assert event.kwargs["prompt_tokens"] == 257
    assert event.kwargs["cp_rank"] == 3


def test_scheduler_emits_admitted_and_deferred_kv_admission_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admitted = _disagg_request(11, 1011)
    admitted.total_input_len_cp = 257
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
        "prompt_tokens": 257,
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
    assert deferred_event.kwargs["prompt_tokens"] == 65
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


def test_gen_decode_ready_uses_full_cp_prompt_length(monkeypatch: pytest.MonkeyPatch) -> None:
    request = _disagg_request(22, 2022, prompt_len=1)
    request.total_input_len_cp = 257
    request.is_disagg_generation_transmission_complete = True
    request.state = LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE
    request.context_phase_params = SimpleNamespace(first_gen_tokens=[7], draft_tokens=None)
    request.py_beam_width = 1
    request.add_new_token = Mock()
    seq_slot_manager = SimpleNamespace(prepare_resources=Mock())
    executor = object.__new__(PyExecutor)
    executor.resource_manager = SimpleNamespace(
        resource_managers={ResourceManagerType.SEQ_SLOT_MANAGER: seq_slot_manager},
    )
    executor._setup_sampler_step = Mock()
    executor.model_engine = SimpleNamespace(enable_spec_decode=False)
    executor.kv_cache_transceiver = None
    executor._update_sampler_state_for_disagg_gen_request = Mock(return_value=True)
    executor._maybe_prepend_logprobs_and_logits = Mock()
    executor.global_rank = 4
    executor.dist = SimpleNamespace(tp_rank=1, pp_rank=0, cp_rank=3)

    emit_event = Mock()
    monkeypatch.setattr(diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", True)
    monkeypatch.setattr(diagnostics, "emit_event", emit_event)

    executor._prepare_disagg_gen_transmission_complete(
        SimpleNamespace(generation_requests=[request]),
    )

    event = emit_event.call_args
    assert event.args == ("gen_decode_ready",)
    assert event.kwargs["prompt_tokens"] == 257
    assert event.kwargs["cp_rank"] == 3
    request.add_new_token.assert_called_once_with(7, 0)


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
    assert send_ready["timeout_expected"] is True
    assert timeout_started["timeout_ms"] == 100
    assert timeout_started["timer_start_monotonic_ns"] == 12_500_000_000
    assert request.py_kv_transfer_start_time == 12.5


def test_ctx_send_continues_when_diagnostic_preparation_fails(
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

    monkeypatch.setattr(diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", True)
    monkeypatch.setattr(
        diagnostics,
        "emit_event",
        Mock(side_effect=RuntimeError("diagnostics failed")),
    )
    monkeypatch.setattr(coordinator_module.time, "monotonic", lambda: 12.5)

    coordinator.send_completed_context([request])

    assert [operation[0] for operation in operations] == ["start_transfer", "respond"]
    assert request.py_kv_transfer_start_time == 12.5


def test_bridge_validation_rejection_emits_failed_ctx_settlement_after_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operations = []
    request = SimpleNamespace(
        is_context_only_request=True,
        is_finished_due_to_cancellation=False,
        is_child=False,
        py_request_id=32,
        request_id=32,
        py_disaggregated_params=SimpleNamespace(disagg_request_id=3032),
        is_context_finished=True,
        is_finished_due_to_length=False,
        prompt_len=128,
        state=LlmRequestState.CONTEXT_INIT,
        py_kv_transfer_start_time=None,
    )
    transfer_manager = SimpleNamespace(
        start_transfer=lambda req: operations.append(("start_transfer", req)),
        should_store_blocks=True,
    )

    def reject(req) -> None:
        operations.append(("respond", req))
        req.state = LlmRequestState.DISAGG_TRANS_ERROR

    transceiver = SimpleNamespace(
        _fp4_mla_bridge_enabled=True,
        _instance_name="ctx-test",
        has_retired_send_session=lambda _req: False,
        respond_and_send_async=reject,
        has_inflight_transfer=lambda _req: False,
        kv_transfer_timeout_ms=100,
        pipeline_transfer_enabled=False,
    )
    coordinator = DisaggTransferCoordinator(
        transceiver=transceiver,
        transfer_manager=transfer_manager,
        kv_cache_manager=None,
        dist=SimpleNamespace(rank=4, tp_rank=1, pp_rank=0, cp_rank=0, dp_rank=2),
        effects=None,
        registry=SimpleNamespace(canceled_request_ids=lambda: []),
        enable_attention_dp=False,
        force_terminate_ctx_for_partial_reuse=False,
        delegates=None,
    )
    coordinator.release_transfer = lambda req: operations.append(("release", req))

    def emit_event(event: str, **kwargs) -> None:
        operations.append((event, kwargs))

    monkeypatch.setattr(diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", True)
    monkeypatch.setattr(diagnostics, "emit_event", emit_event)

    coordinator.send_completed_context([request])

    assert [operation[0] for operation in operations] == [
        "start_transfer",
        "ctx_send_ready",
        "respond",
        "release",
        "ctx_transfer_settled",
    ]
    settlement = operations[4][1]
    assert settlement["request_id"] == 3032
    assert settlement["local_request_id"] == 32
    assert settlement["outcome"] == "failed"
    assert settlement["session_status"] is None
    assert settlement["resources_drained"] is True
    assert request.py_kv_transfer_start_time is None


@pytest.mark.parametrize(
    ("wait_result", "session_status", "outcome", "request_state"),
    [
        (
            WaitResult.COMPLETED,
            SessionStatus.FULLY_TRANSFERRED,
            "completed",
            LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE,
        ),
        (
            WaitResult.FAILED,
            SessionStatus.ERROR,
            "failed",
            LlmRequestState.DISAGG_TRANS_ERROR,
        ),
    ],
)
def test_sync_receive_emits_start_and_terminal_settlement(
    monkeypatch: pytest.MonkeyPatch,
    wait_result: WaitResult,
    session_status: SessionStatus,
    outcome: str,
    request_state: LlmRequestState,
) -> None:
    operations = []
    session = SimpleNamespace(
        status=session_status,
        receive=Mock(side_effect=lambda _slice: operations.append("receive")),
        wait_complete=Mock(side_effect=lambda blocking: operations.append("wait") or wait_result),
        has_transferring_tasks=Mock(return_value=False),
        close=Mock(side_effect=lambda: operations.append("close") or True),
    )
    request = _disagg_request(41, 4041)
    request.state = LlmRequestState.DISAGG_GENERATION_INIT
    request.set_kv_cache_size = Mock()
    transceiver = _diagnostic_transceiver()
    transceiver._validate_bridge_req = Mock(return_value=True)
    transceiver._recv_sessions = {}
    transceiver._recv_reqs = {}
    transceiver._transfer_worker = SimpleNamespace(create_rx_session=Mock(return_value=session))
    transceiver._create_kv_slice = Mock(return_value=KVSlice(is_last_slice=True))
    transceiver._slice_num_bytes = Mock(return_value=64)
    transceiver._kv_size_rank_factor = 2
    transceiver._need_aux_transfer = Mock(return_value=False)
    transceiver._assert_disagg_history_declared = Mock()

    def emit_event(event: str, **kwargs) -> None:
        if event == "gen_transfer_settled":
            assert transceiver._recv_sessions == {}
            assert transceiver._recv_reqs == {}
        operations.append(event)
        emitted.append((event, kwargs))

    emitted = []
    monkeypatch.setattr(diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", True)
    monkeypatch.setattr(diagnostics, "emit_event", emit_event)

    transceiver.request_and_receive_sync(request)

    assert operations == [
        "gen_receive_start",
        "receive",
        "wait",
        "close",
        "gen_transfer_settled",
    ]
    assert [event for event, _kwargs in emitted] == [
        "gen_receive_start",
        "gen_transfer_settled",
    ]
    receive_start = emitted[0][1]
    settlement = emitted[1][1]
    assert receive_start["request_id"] == settlement["request_id"] == 4041
    assert receive_start["local_request_id"] == settlement["local_request_id"] == 41
    assert receive_start["transfer_bytes"] == 128
    assert receive_start["timeout_expected"] is False
    assert settlement["outcome"] == outcome
    assert settlement["session_status"] == session_status.value
    assert settlement["resources_drained"] is True
    assert request.state == request_state
    assert transceiver._recv_sessions == {}
    assert transceiver._recv_reqs == {}
    transceiver._validate_bridge_req.assert_called_once_with(request, synchronous=True)
    session.wait_complete.assert_called_once_with(blocking=True)
    if wait_result == WaitResult.COMPLETED:
        request.set_kv_cache_size.assert_called_once_with(128)
        transceiver._assert_disagg_history_declared.assert_called_once_with(request)
    else:
        request.set_kv_cache_size.assert_not_called()
        transceiver._assert_disagg_history_declared.assert_not_called()


def test_sync_receive_does_not_report_settlement_while_close_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = SimpleNamespace(
        status=SessionStatus.ERROR,
        receive=Mock(side_effect=RuntimeError("receive failed")),
        has_transferring_tasks=Mock(return_value=True),
        close=Mock(return_value=False),
    )
    request = _disagg_request(44, 4044)
    request.state = LlmRequestState.DISAGG_GENERATION_INIT
    transceiver = _diagnostic_transceiver()
    transceiver._validate_bridge_req = Mock(return_value=True)
    transceiver._recv_sessions = {}
    transceiver._recv_reqs = {}
    transceiver._transfer_worker = SimpleNamespace(create_rx_session=Mock(return_value=session))
    transceiver._create_kv_slice = Mock(return_value=KVSlice(is_last_slice=True))
    transceiver._slice_num_bytes = Mock(return_value=64)
    transceiver._kv_size_rank_factor = 2

    emit_event = Mock()
    monkeypatch.setattr(diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", True)
    monkeypatch.setattr(diagnostics, "emit_event", emit_event)

    with pytest.raises(RuntimeError, match="receive failed"):
        transceiver.request_and_receive_sync(request)

    assert [call.args[0] for call in emit_event.call_args_list] == ["gen_receive_start"]
    assert transceiver._recv_sessions == {4044: session}
    assert transceiver._recv_reqs == {4044: request}


def test_sync_receive_disabled_diagnostics_preserves_receive_failure_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = SimpleNamespace(
        receive=Mock(side_effect=RuntimeError("receive failed")),
        close=Mock(return_value=True),
    )
    request = _disagg_request(45, 4045)
    request.state = LlmRequestState.DISAGG_GENERATION_INIT
    transceiver = _diagnostic_transceiver()
    transceiver._validate_bridge_req = Mock(return_value=True)
    transceiver._recv_sessions = {}
    transceiver._recv_reqs = {}
    transceiver._transfer_worker = SimpleNamespace(create_rx_session=Mock(return_value=session))
    kv_slice = KVSlice(is_last_slice=True)
    transceiver._create_kv_slice = Mock(return_value=kv_slice)
    transceiver._slice_num_bytes = Mock(side_effect=AssertionError("must not size before receive"))

    monkeypatch.setattr(diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", False)

    with pytest.raises(RuntimeError, match="receive failed"):
        transceiver.request_and_receive_sync(request)

    session.receive.assert_called_once_with(kv_slice)
    transceiver._slice_num_bytes.assert_not_called()
    assert request.state == LlmRequestState.DISAGG_TRANS_ERROR
    assert transceiver._recv_sessions == {}
    assert transceiver._recv_reqs == {}


@pytest.mark.parametrize("side", ("ctx", "gen"))
def test_fast_cancel_emits_one_terminal_settlement(
    monkeypatch: pytest.MonkeyPatch,
    side: str,
) -> None:
    request = _disagg_request(42, 4042)
    request.py_kv_send_session_retired = False
    session = SimpleNamespace(
        status=SessionStatus.READY,
        has_transferring_tasks=Mock(return_value=False),
        close=Mock(return_value=True),
    )
    session.cancel = Mock(side_effect=lambda: setattr(session, "status", SessionStatus.CANCELLED))
    transceiver = _diagnostic_transceiver()
    transceiver._wait_reqs = {}
    transceiver._send_sessions = {4042: session} if side == "ctx" else {}
    transceiver._send_reqs = {4042: request} if side == "ctx" else {}
    transceiver._recv_sessions = {4042: session} if side == "gen" else {}
    transceiver._recv_reqs = {4042: request} if side == "gen" else {}

    emitted = []

    def emit_event(event: str, **kwargs) -> None:
        if event.endswith("_transfer_settled"):
            assert 4042 not in transceiver._send_sessions
            assert 4042 not in transceiver._recv_sessions
        emitted.append((event, kwargs))

    monkeypatch.setattr(diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", True)
    monkeypatch.setattr(diagnostics, "emit_event", emit_event)

    assert transceiver.cancel_request(request) is True
    assert transceiver.cancel_request(request) is True

    assert [event for event, _kwargs in emitted] == [
        "transfer_cancel_requested",
        f"{side}_transfer_settled",
    ]
    settlement = emitted[1][1]
    assert settlement["outcome"] == "cancelled"
    assert settlement["session_status"] == SessionStatus.CANCELLED.value
    assert settlement["resources_drained"] is True
    session.close.assert_called_once_with()
    sessions = transceiver._send_sessions if side == "ctx" else transceiver._recv_sessions
    requests = transceiver._send_reqs if side == "ctx" else transceiver._recv_reqs
    assert sessions == {}
    assert requests == {}
    if side == "ctx":
        assert request.py_kv_send_session_retired is True


@pytest.mark.parametrize("side", ("ctx", "gen"))
def test_active_cancel_does_not_emit_terminal_settlement(
    monkeypatch: pytest.MonkeyPatch,
    side: str,
) -> None:
    request = _disagg_request(43, 4043)
    session = SimpleNamespace(
        status=SessionStatus.TRANSFERRING,
        cancel=Mock(),
        has_transferring_tasks=Mock(return_value=True),
        close=Mock(),
    )
    transceiver = _diagnostic_transceiver()
    transceiver._wait_reqs = {}
    transceiver._send_sessions = {4043: session} if side == "ctx" else {}
    transceiver._send_reqs = {4043: request} if side == "ctx" else {}
    transceiver._recv_sessions = {4043: session} if side == "gen" else {}
    transceiver._recv_reqs = {4043: request} if side == "gen" else {}

    emit_event = Mock()
    monkeypatch.setattr(diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", True)
    monkeypatch.setattr(diagnostics, "emit_event", emit_event)

    assert transceiver.cancel_request(request) is False

    assert [call.args[0] for call in emit_event.call_args_list] == ["transfer_cancel_requested"]
    session.close.assert_not_called()


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


def test_transfer_window_result_uses_full_cp_prompt_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _disagg_request(23, 2023, prompt_len=1)
    request.total_input_len_cp = 257
    controller = DisaggTransferAdmissionController(
        max_tokens_in_buffer=512,
        tokens_per_block=32,
    )
    executor = object.__new__(PyExecutor)
    executor.global_rank = 4
    executor.dist = SimpleNamespace(tp_rank=1, pp_rank=0, cp_rank=3)

    emit_event = Mock()
    monkeypatch.setattr(diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", True)
    monkeypatch.setattr(diagnostics, "emit_event", emit_event)

    executor._emit_disagg_transfer_window_results(
        [request],
        [request],
        policy="enforced",
        controller=controller,
        active_transfer_blocks=0,
        admitted_transfer_blocks=9,
    )

    event = emit_event.call_args
    assert event.args == ("gen_transfer_window_result",)
    assert event.kwargs["prompt_tokens"] == 257
    assert event.kwargs["request_blocks"] == 9
    assert event.kwargs["cp_rank"] == 3


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


def test_diagnostic_preparation_failure_does_not_change_bypassed_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _BrokenLegacyResult:
        @property
        def admitted_requests(self):
            raise RuntimeError("diagnostic counterfactual inspection failed")

    candidate = _disagg_request(21, 2021)
    controller = Mock()
    controller.enabled.return_value = True
    controller.select.return_value = _BrokenLegacyResult()
    executor = object.__new__(PyExecutor)
    executor.active_requests = []
    executor._is_disagg_gen_only_no_context_benchmark = Mock(return_value=False)
    executor._get_disagg_transfer_admission_controller = Mock(return_value=controller)
    executor._disagg_transfer_window_is_active = Mock(return_value=False)
    executor._is_disagg_transfer_window_bypass_eligible = Mock(return_value=True)
    monkeypatch.setattr(diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", True)

    admitted, wait_for_progress = executor._apply_disagg_transfer_admission([candidate])

    assert admitted == [candidate]
    assert wait_for_progress is False
    executor._is_disagg_transfer_window_bypass_eligible.assert_called_once_with()
    controller.select.assert_called_once_with([], [candidate])


def test_transfer_window_helper_contains_diagnostic_preparation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _OpaqueCandidate:
        @property
        def py_request_id(self) -> int:
            raise RuntimeError("diagnostic request inspection failed")

    candidate = _OpaqueCandidate()
    executor = object.__new__(PyExecutor)
    monkeypatch.setattr(diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", True)

    executor._emit_disagg_transfer_window_results(
        [candidate],
        [candidate],
        policy="bypassed",
    )


def test_admission_rollback_continues_when_diagnostic_emission_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = [
        _disagg_request(31, 3031, prompt_len=32),
        _disagg_request(32, 3032, prompt_len=32),
    ]
    controller = DisaggTransferAdmissionController(
        max_tokens_in_buffer=32,
        tokens_per_block=32,
    )
    executor = object.__new__(PyExecutor)
    executor.active_requests = []
    executor._is_disagg_gen_only_no_context_benchmark = Mock(return_value=False)
    executor._get_disagg_transfer_admission_controller = Mock(return_value=controller)
    executor._disagg_transfer_window_is_active = Mock(return_value=True)
    executor._emit_disagg_transfer_window_results = Mock(
        side_effect=RuntimeError("diagnostics failed")
    )
    executor._revert_deferred_disagg_gen_init_alloc = Mock()
    monkeypatch.setattr(diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", True)

    admitted, wait_for_progress = executor._apply_disagg_transfer_admission(candidates)

    assert admitted == [candidates[0]]
    assert wait_for_progress is False
    executor._revert_deferred_disagg_gen_init_alloc.assert_called_once_with(
        candidates,
        [candidates[0]],
        reason="transfer_window",
    )


def test_pp_reconciliation_emits_rollback_after_releasing_kv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admitted = _disagg_request(33, 3033)
    deferred = _disagg_request(34, 3034)
    operations = []
    executor = object.__new__(PyExecutor)
    executor._is_kv_manager_v2 = True
    executor._revert_ctx_alloc = lambda requests: operations.append(("revert", requests))
    executor.global_rank = 4
    executor.dist = SimpleNamespace(tp_rank=1, pp_rank=2, cp_rank=3)

    def emit_event(event: str, **kwargs) -> None:
        operations.append((event, kwargs))

    monkeypatch.setattr(diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", True)
    monkeypatch.setattr(diagnostics, "emit_event", emit_event)

    executor._revert_deferred_disagg_gen_init_alloc(
        [admitted, deferred],
        [admitted],
    )

    assert operations[0] == ("revert", [deferred])
    assert operations[1][0] == "gen_kv_rollback"
    event = operations[1][1]
    assert event["request_id"] == 3034
    assert event["reason"] == "pp_reconciliation"
    assert event["pp_rank"] == 2
    assert event["cp_rank"] == 3


def test_ctx_settlement_continues_when_diagnostic_emission_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_id = 41
    session = SimpleNamespace(
        wait_complete=Mock(return_value=WaitResult.COMPLETED),
        status=SessionStatus.READY,
        has_transferring_tasks=Mock(return_value=False),
    )
    request = SimpleNamespace(py_request_id=4)
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._ever_had_send_session = True
    transceiver._ctx_need_tp_sync = False
    transceiver._ctx_need_pp_sync = False
    transceiver._send_sessions = {request_id: session}
    transceiver._send_reqs = {request_id: request}
    transceiver._collect_done = Mock(return_value=([request_id], []))
    transceiver._ctx_consensus = Mock(side_effect=lambda request_ids: request_ids)
    transceiver._build_to_process = Mock(return_value=[request_id])
    transceiver._ctx_consensus_outcome = Mock(return_value=([], [], [request_id], [request_id]))

    def retire_send_session(rid: int, **_kwargs) -> None:
        transceiver._send_sessions.pop(rid, None)
        transceiver._send_reqs.pop(rid, None)

    transceiver._retire_send_session = Mock(side_effect=retire_send_session)
    transceiver._close_failed_sessions = Mock()
    transceiver._transfer_worker = SimpleNamespace(sweep_stale_req_infos=Mock())
    transceiver._mapping = SimpleNamespace(rank=1, tp_rank=0, pp_rank=0, cp_rank=0)
    transceiver._instance_name = "ctx"
    transceiver._dp_rank = 0

    monkeypatch.setattr(diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", True)
    emitted_events = []

    def fail_after_ctx_release(_event: str, **_kwargs) -> None:
        emitted_events.append(
            (
                _event,
                tuple(transceiver._send_sessions),
                tuple(transceiver._send_reqs),
            )
        )
        raise RuntimeError("diagnostics failed")

    monkeypatch.setattr(diagnostics, "emit_event", fail_after_ctx_release)

    status = transceiver.check_context_transfer_status(at_least_request_num=0)

    assert status.completed_request_ids == [request_id]
    assert status.error_request_ids == []
    assert emitted_events == [("ctx_transfer_settled", (), ())]
    transceiver._retire_send_session.assert_called_once_with(
        request_id,
        outcome="completed",
    )
    transceiver._transfer_worker.sweep_stale_req_infos.assert_called_once_with()


def test_ctx_settlement_continues_when_diagnostic_session_snapshot_is_stale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_id = 43
    session = SimpleNamespace(
        wait_complete=Mock(return_value=WaitResult.COMPLETED),
        status=SessionStatus.READY,
        has_transferring_tasks=Mock(return_value=False),
    )
    request = SimpleNamespace(py_request_id=6)
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._ever_had_send_session = True
    transceiver._ctx_need_tp_sync = False
    transceiver._ctx_need_pp_sync = False
    transceiver._send_sessions = {request_id: session}
    transceiver._send_reqs = {request_id: request}
    transceiver._collect_done = Mock(return_value=([request_id], []))
    transceiver._ctx_consensus = Mock(side_effect=lambda request_ids: request_ids)
    transceiver._build_to_process = Mock(return_value=[request_id])

    def retire_before_diagnostic_snapshot(*_args):
        transceiver._send_sessions.pop(request_id)
        return [], [], [request_id], [request_id]

    transceiver._ctx_consensus_outcome = Mock(side_effect=retire_before_diagnostic_snapshot)
    transceiver._retire_send_session = Mock(
        side_effect=lambda rid, **_kwargs: transceiver._send_reqs.pop(rid, None)
    )
    transceiver._close_failed_sessions = Mock()
    transceiver._transfer_worker = SimpleNamespace(sweep_stale_req_infos=Mock())
    transceiver._mapping = SimpleNamespace(rank=1, tp_rank=0, pp_rank=0, cp_rank=0)
    transceiver._instance_name = "ctx"
    transceiver._dp_rank = 0
    emit_event = Mock()

    monkeypatch.setattr(diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", True)
    monkeypatch.setattr(diagnostics, "emit_event", emit_event)

    status = transceiver.check_context_transfer_status(at_least_request_num=0)

    assert status.completed_request_ids == [request_id]
    assert status.error_request_ids == []
    transceiver._retire_send_session.assert_called_once_with(
        request_id,
        outcome="completed",
    )
    emit_event.assert_not_called()


def test_gen_settlement_continues_when_diagnostic_emission_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_id = 42
    session = SimpleNamespace(
        wait_complete=Mock(return_value=WaitResult.COMPLETED),
        status=SessionStatus.READY,
        transfer_end_time=None,
        kv_cache_size_bytes=0,
        has_transferring_tasks=Mock(return_value=False),
    )
    request = SimpleNamespace(
        py_request_id=5,
        set_kv_cache_size=Mock(),
    )
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._ever_had_recv_session = True
    transceiver._gen_need_sync = False
    transceiver._recv_sessions = {request_id: session}
    transceiver._recv_reqs = {request_id: request}
    transceiver._collect_done = Mock(return_value=([request_id], []))
    transceiver._gen_consensus = Mock(side_effect=lambda request_ids: request_ids)
    transceiver._build_to_process = Mock(return_value=[request_id])
    transceiver._gen_consensus_outcome = Mock(return_value=([], [], [request_id]))
    transceiver._need_aux_transfer = Mock(return_value=False)
    transceiver._assert_disagg_history_declared = Mock()
    transceiver._close_session_or_raise = Mock()
    transceiver._close_failed_sessions = Mock()
    transceiver._mapping = SimpleNamespace(rank=1, tp_rank=0, pp_rank=0, cp_rank=0)
    transceiver._instance_name = "gen"
    transceiver._dp_rank = 0
    transceiver._dist = SimpleNamespace(rank=1)

    monkeypatch.setattr(diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", True)
    emitted_events = []

    def fail_after_gen_release(_event: str, **_kwargs) -> None:
        emitted_events.append(
            (
                _event,
                tuple(transceiver._recv_sessions),
                tuple(transceiver._recv_reqs),
            )
        )
        raise RuntimeError("diagnostics failed")

    monkeypatch.setattr(diagnostics, "emit_event", fail_after_gen_release)

    status = transceiver.check_gen_transfer_status(at_least_request_num=0)

    assert status.completed_request_ids == [request_id]
    assert status.error_request_ids == []
    assert status.cancelled_requests == []
    assert emitted_events == [("gen_transfer_settled", (), ())]
    request.set_kv_cache_size.assert_called_once_with(0)
    transceiver._close_session_or_raise.assert_called_once_with(
        session,
        request_id,
        "completed",
    )
    assert request.state == LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE
    assert transceiver._recv_sessions == {}
    assert transceiver._recv_reqs == {}


@pytest.mark.parametrize("enforce_physical_ownership", [False, True])
def test_backend_wait_continues_when_submission_diagnostic_callback_fails(
    enforce_physical_ownership: bool,
) -> None:
    status = SimpleNamespace(wait=Mock(return_value=True))
    sender = object.__new__(Sender)
    sender._enforce_physical_ownership = enforce_physical_ownership
    sender._agent = SimpleNamespace(submit_transfer_requests=Mock(return_value=status))
    sender._ownership_poison_lock = threading.Lock()
    sender._ownership_poisoned = None
    task = Mock()
    request = Mock()
    callback = Mock(side_effect=RuntimeError("diagnostics failed"))

    result = sender._submit_transfer(
        task,
        7,
        request,
        on_submitted=callback,
    )

    assert result == (True, None)
    callback.assert_called_once_with()
    status.wait.assert_called_once_with()
    if enforce_physical_ownership:
        task.begin_backend_submission.assert_called_once_with(7, request)
        task.record_backend_submission.assert_called_once_with(7, status)
        task.retire_backend_done_physical_operation.assert_called_once_with(7)


def test_request_cleanup_continues_when_diagnostic_emission_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = SimpleNamespace(
        is_context_only_request=True,
        py_request_id=51,
        request_id=51,
        py_disaggregated_params=SimpleNamespace(disagg_request_id=5051),
        prompt_len=128,
        state=SimpleNamespace(name="DISAGG_CONTEXT_COMPLETE"),
    )
    executor = object.__new__(PyExecutor)
    executor.resource_manager = SimpleNamespace(free_resources=Mock())
    executor._prefetched_request_ids = {request.py_request_id}
    executor._disagg_coordinator = SimpleNamespace(forget_request=Mock())
    executor.global_rank = 4
    executor.dist = SimpleNamespace(tp_rank=1, pp_rank=0, cp_rank=0)
    monkeypatch.setattr(diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", True)
    monkeypatch.setattr(
        diagnostics,
        "emit_event",
        Mock(side_effect=RuntimeError("diagnostics failed")),
    )

    executor._free_request_resources(request)

    executor.resource_manager.free_resources.assert_called_once_with(request)
    assert request.py_request_id not in executor._prefetched_request_ids
    executor._disagg_coordinator.forget_request.assert_called_once_with(request.py_request_id)


def test_source_unpin_continues_when_diagnostic_inspection_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _KVCacheManager:
        def __init__(self) -> None:
            self.unpin_blocks_by_id = Mock()

        @property
        def mapping(self):
            raise RuntimeError("diagnostic mapping inspection failed")

    request = SimpleNamespace(
        is_context_only_request=True,
        py_request_id=61,
        state=LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS,
    )
    block_ids = [9]
    metadata = AsyncTransferManager.RequestTransferMetadata(block_id=block_ids)
    metadata.start_transfer()
    manager = object.__new__(AsyncTransferManager)
    manager.should_store_blocks = True
    manager.kv_cache_manager = _KVCacheManager()
    manager._requests_in_transfer = {request.py_request_id: request}
    manager._request_transfer_metadata = {request.py_request_id: metadata}
    monkeypatch.setattr(diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", True)

    should_terminate = manager.end_transfer(request)

    assert should_terminate is True
    manager.kv_cache_manager.unpin_blocks_by_id.assert_called_once_with(block_ids)
    assert request.state == LlmRequestState.DISAGG_CONTEXT_COMPLETE
    assert manager._requests_in_transfer == {}
    assert manager._request_transfer_metadata == {}


def test_source_unpin_summarizes_real_block_id_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    block_ids = [9, 10, 11]
    request = SimpleNamespace(
        is_context_only_request=True,
        py_request_id=62,
        py_disaggregated_params=SimpleNamespace(disagg_request_id=6062),
        state=LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS,
    )
    metadata = AsyncTransferManager.RequestTransferMetadata(block_id=block_ids)
    metadata.start_transfer()
    manager = object.__new__(AsyncTransferManager)
    manager.should_store_blocks = True
    manager.kv_cache_manager = SimpleNamespace(
        mapping=SimpleNamespace(rank=4),
        unpin_blocks_by_id=Mock(),
    )
    manager._requests_in_transfer = {request.py_request_id: request}
    manager._request_transfer_metadata = {request.py_request_id: metadata}
    emit_event = Mock()
    monkeypatch.setattr(diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", True)
    monkeypatch.setattr(diagnostics, "emit_event", emit_event)

    assert manager.end_transfer(request) is True

    manager.kv_cache_manager.unpin_blocks_by_id.assert_called_once_with(block_ids)
    event = emit_event.call_args
    assert event.args == ("ctx_source_unpinned",)
    assert event.kwargs["source_kv_reuse_block_count"] == 3


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
