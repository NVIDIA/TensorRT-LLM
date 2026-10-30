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
import tensorrt_llm._torch.disaggregation.transceiver as python_transceiver_module
from tensorrt_llm import DisaggregatedParams
from tensorrt_llm._torch.disaggregation import diagnostics
from tensorrt_llm._torch.disaggregation.base import TokenRange
from tensorrt_llm._torch.disaggregation.base.transfer import SessionStatus, WaitResult
from tensorrt_llm._torch.disaggregation.native.extent import CacheExtent, Chunk
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


def _disagg_request(
    local_id: int, canonical_id: int | None, prompt_len: int = 65
) -> SimpleNamespace:
    return SimpleNamespace(
        py_request_id=local_id,
        request_id=local_id,
        py_disaggregated_params=DisaggregatedParams(disagg_request_id=canonical_id),
        prompt_len=prompt_len,
    )


def _diagnostic_transceiver() -> KvCacheTransceiverV2:
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._mapping = SimpleNamespace(rank=4, tp_rank=1, pp_rank=0, cp_rank=0)
    transceiver._instance_name = "diagnostic-test"
    transceiver._dp_rank = 2
    return transceiver


def _extent(request_id: int) -> CacheExtent:
    return CacheExtent(
        name=request_id,
        local=Chunk(
            block_ids_per_layer_groups=[],
            kind_per_layer_group=[],
            token_range=TokenRange(start=0, end=0),
            is_last=True,
        ),
    )


def _coordinator(
    *,
    controller=None,
    active_requests=(),
    consumes_transfer_buffer: bool = False,
) -> DisaggTransferCoordinator:
    return DisaggTransferCoordinator(
        transceiver=SimpleNamespace(consumes_transfer_buffer=consumes_transfer_buffer),
        transfer_manager=None,
        kv_cache_manager=None,
        dist=SimpleNamespace(rank=4, tp_rank=1, pp_rank=0, cp_rank=0, pp_size=1),
        effects=SimpleNamespace(revert_ctx_alloc=Mock()),
        registry=SimpleNamespace(active_requests=lambda: list(active_requests)),
        enable_attention_dp=False,
        force_terminate_ctx_for_partial_reuse=False,
        admission_controller=controller,
        is_kv_manager_v2=True,
    )


def test_sender_reports_worker_dequeue_before_transfer_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_queue = queue.Queue()
    write_meta = SimpleNamespace(
        task=SimpleNamespace(_params=DisaggregatedParams(disagg_request_id=1010)),
        meta_type=transfer_module.WriteMetaType.KV,
        src_ptrs=SimpleNamespace(size=1),
        sizes=SimpleNamespace(sum=Mock(side_effect=AssertionError("must reuse descriptor size"))),
        transfer_bytes=4096,
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
    sender._pending_settlements = [{}]
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
    assert event["request_id_scope"] == "run"
    assert event["slice_id"] == 2
    assert event["peer_rank"] == 3
    assert event["worker_queue_index"] == 0
    assert event["transfer_bytes"] == 4096
    write_meta.sizes.sum.assert_not_called()


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


def test_scheduler_reports_only_first_failed_kv_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admitted = _disagg_request(11, 1011)
    admitted.total_input_len_cp = 257
    deferred = _disagg_request(12, 1012)
    results = iter((True, False, False))
    manager = SimpleNamespace(
        prepare_disagg_gen_init=lambda _request: next(results),
        kv_cache_map={
            admitted.py_request_id: SimpleNamespace(capacity=65, history_length=64),
        },
        mapping=SimpleNamespace(rank=3),
        tokens_per_block=32,
    )
    scheduler = object.__new__(KVCacheV2Scheduler)
    scheduler.kv_cache_manager = manager
    scheduler.tokens_per_block = 32

    emit_event = Mock()
    monkeypatch.setattr(diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", True)
    monkeypatch.setattr(diagnostics, "emit_event", emit_event)

    admitted_result = scheduler._try_schedule_disagg_gen_init(admitted, None)
    deferred_result = scheduler._try_schedule_disagg_gen_init(deferred, None)
    assert scheduler._try_schedule_disagg_gen_init(deferred, None) == (ScheduleAction.SKIP, 0)

    assert admitted_result == (ScheduleAction.SCHEDULED, 0)
    assert deferred_result == (ScheduleAction.SKIP, 0)
    emit_event.assert_called_once()
    deferred_event = emit_event.call_args
    assert deferred_event.args == ("gen_kv_admission_result",)
    assert deferred_event.kwargs["request_id"] == 1012
    assert deferred_event.kwargs["local_request_id"] == 12
    assert deferred_event.kwargs["outcome"] == "deferred"
    assert deferred_event.kwargs["reason"] == "kv_or_index_capacity"
    assert deferred_event.kwargs["prompt_tokens"] == 65
    assert deferred_event.kwargs["cache_present"] is False
    assert deferred_event.kwargs["capacity_tokens"] is None
    assert deferred_event.kwargs["capacity_block_equivalent"] is None
    assert deferred_event.kwargs["kv_deferral_count"] == 1
    assert deferred._disagg_diagnostics_kv_deferral_count == 2


def _gen_admission_setup(
    request: SimpleNamespace,
) -> tuple[KVCacheV2Scheduler, DisaggTransferCoordinator]:
    """Use real admission paths with fake external KV and receive operations."""
    manager = SimpleNamespace(
        prepare_disagg_gen_init=Mock(return_value=True),
        kv_cache_map={request.py_request_id: SimpleNamespace(capacity=65, history_length=64)},
        mapping=SimpleNamespace(rank=4),
        tokens_per_block=32,
    )
    scheduler = object.__new__(KVCacheV2Scheduler)
    scheduler.kv_cache_manager = manager
    scheduler.tokens_per_block = 32
    coordinator = _coordinator()
    coordinator._kv_cache_manager = manager
    coordinator._effects.prepare_gen_resources = Mock()
    coordinator._transceiver.kv_transfer_timeout_ms = None
    coordinator._transceiver.request_and_receive_async = Mock(
        side_effect=lambda req: setattr(
            req, "state", LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
        )
    )
    coordinator.reap_gen_receives = Mock()
    request.state = LlmRequestState.DISAGG_GENERATION_INIT
    return scheduler, coordinator


@pytest.mark.parametrize("selection", ["transfer_window", "pp_reconciliation", "bypass"])
def test_gen_admission_is_reported_once_after_final_selection(
    monkeypatch: pytest.MonkeyPatch, selection: str
) -> None:
    """Tentative scheduler success and repeated rollback are not admission."""
    request = _disagg_request(13, 1013, prompt_len=32)
    request.total_input_len_cp = 257
    scheduler, coordinator = _gen_admission_setup(request)
    active = _disagg_request(14, 1014, prompt_len=32)
    active.is_disagg_generation_transmission_in_progress = True
    active_requests = [active]
    coordinator._registry.active_requests = lambda: active_requests
    coordinator._admission_controller = DisaggTransferAdmissionController(32, 32)
    if selection != "bypass":
        coordinator._dist.pp_size = 2
        coordinator._dist.pp_rank = 1
    emit_event = Mock()
    monkeypatch.setattr(diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", True)
    monkeypatch.setattr(diagnostics, "emit_event", emit_event)
    monkeypatch.delenv("TRTLLM_DISAGG_BENCHMARK_GEN_ONLY", raising=False)
    monkeypatch.delenv("TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP", raising=False)

    for _ in range(3 if selection != "bypass" else 0):
        assert scheduler._try_schedule_disagg_gen_init(request, None) == (
            ScheduleAction.SCHEDULED,
            0,
        )
        if selection == "transfer_window":
            admitted, blocked = coordinator.admit([request])
            assert admitted == []
            assert blocked
        else:
            coordinator.revert_deferred_gen_init([request], [])
            admitted = []
        coordinator.receive_gen_init(admitted)
    assert not any(call.kwargs.get("outcome") == "admitted" for call in emit_event.call_args_list)
    coordinator._effects.prepare_gen_resources.assert_not_called()
    if selection != "bypass":
        emit_event.assert_called_once()
        assert emit_event.call_args.args == ("gen_kv_rollback",)
        assert emit_event.call_args.kwargs["reason"] == selection
        assert emit_event.call_args.kwargs["rollback_count"] == 1
        assert coordinator._effects.revert_ctx_alloc.call_count == 3
        active_requests.clear()

    assert scheduler._try_schedule_disagg_gen_init(request, None) == (ScheduleAction.SCHEDULED, 0)
    if selection == "pp_reconciliation":
        coordinator.revert_deferred_gen_init([request], [request])
    admitted, blocked = coordinator.admit([request])
    assert admitted == [request]
    assert not blocked
    coordinator.receive_gen_init(admitted)

    event = emit_event.call_args
    assert event.args == ("gen_kv_admission_result",)
    assert event.kwargs["outcome"] == "admitted"
    assert event.kwargs["request_id"] == 1013
    assert event.kwargs["prompt_tokens"] == 257
    assert event.kwargs["capacity_tokens"] == 65
    assert event.kwargs["capacity_block_equivalent"] == 3
    assert event.kwargs["kv_deferral_count"] == 0
    assert event.kwargs["transfer_window_rollback_count"] == (
        3 if selection == "transfer_window" else 0
    )
    assert event.kwargs["pp_reconciliation_rollback_count"] == (
        3 if selection == "pp_reconciliation" else 0
    )
    assert emit_event.call_count == (1 if selection == "bypass" else 2)
    coordinator._effects.prepare_gen_resources.assert_called_once_with([request])
    coordinator._transceiver.request_and_receive_async.assert_called_once_with(request)


def test_gen_admission_precedes_readiness_wait_and_receive_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An admitted request may still wait for the destination readiness fence."""
    request = _disagg_request(15, 1015)
    _, coordinator = _gen_admission_setup(request)
    ready = SimpleNamespace(query=Mock(return_value=False))
    coordinator._defer_unready_gen_receives = True
    coordinator._kv_cache_manager.take_disagg_receive_ready = Mock(return_value=ready)
    operations = []
    coordinator._effects.prepare_gen_resources.side_effect = lambda _requests: operations.append(
        "prepare"
    )
    coordinator._transceiver.request_and_receive_async.side_effect = lambda _request: (
        operations.append("receive")
    )
    emit_event = Mock(side_effect=lambda event, **_fields: operations.append(event))
    monkeypatch.setattr(diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", True)
    monkeypatch.setattr(diagnostics, "emit_event", emit_event)
    monkeypatch.delenv("TRTLLM_DISAGG_BENCHMARK_GEN_ONLY", raising=False)
    monkeypatch.delenv("TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP", raising=False)

    coordinator.receive_gen_init([request])

    assert operations == ["prepare", "gen_kv_admission_result"]
    assert request.state == LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
    assert request.py_request_id in coordinator._unpublished_gen_receives
    coordinator._transceiver.request_and_receive_async.assert_not_called()
    ready.query.return_value = True
    coordinator._publish_ready_gen_receives()
    assert operations == ["prepare", "gen_kv_admission_result", "receive"]
    assert not coordinator._unpublished_gen_receives
    emit_event.assert_called_once()


@pytest.mark.parametrize("is_kv_manager_v2", [False, True])
@pytest.mark.parametrize("diagnostic_mode", ["disabled", "enabled", "failing"])
def test_gen_admission_diagnostics_preserve_preparation_and_receive_behavior(
    monkeypatch: pytest.MonkeyPatch, is_kv_manager_v2: bool, diagnostic_mode: str
) -> None:
    """Final admission is V2-only and optional reporting never stops receiving."""
    request = _disagg_request(16, 1016)
    _, coordinator = _gen_admission_setup(request)
    coordinator._is_kv_manager_v2 = is_kv_manager_v2
    emit_event = Mock(
        side_effect=RuntimeError("diagnostics failed") if diagnostic_mode == "failing" else None
    )
    monkeypatch.setattr(
        diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", diagnostic_mode != "disabled"
    )
    monkeypatch.setattr(diagnostics, "emit_event", emit_event)
    monkeypatch.delenv("TRTLLM_DISAGG_BENCHMARK_GEN_ONLY", raising=False)
    monkeypatch.delenv("TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP", raising=False)

    coordinator.receive_gen_init([request])

    coordinator._effects.prepare_gen_resources.assert_called_once_with([request])
    coordinator._transceiver.request_and_receive_async.assert_called_once_with(request)
    assert request.state == LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
    assert emit_event.call_count == int(is_kv_manager_v2 and diagnostic_mode != "disabled")


def test_gen_resource_preparation_failure_is_not_reported_as_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A throwing resource preparation cannot complete the admission interval."""
    request = _disagg_request(17, 1017)
    scheduler, coordinator = _gen_admission_setup(request)
    coordinator._effects.prepare_gen_resources.side_effect = RuntimeError("preparation failed")
    emit_event = Mock()
    monkeypatch.setattr(diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", True)
    monkeypatch.setattr(diagnostics, "emit_event", emit_event)

    assert scheduler._try_schedule_disagg_gen_init(request, None) == (ScheduleAction.SCHEDULED, 0)
    with pytest.raises(RuntimeError, match="preparation failed"):
        coordinator.receive_gen_init([request])

    emit_event.assert_not_called()
    coordinator._transceiver.request_and_receive_async.assert_not_called()
    assert not coordinator._unpublished_gen_receives


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

    coordinator = DisaggTransferCoordinator(
        transceiver=transceiver,
        transfer_manager=SimpleNamespace(requests_in_transfer=lambda: {}),
        kv_cache_manager=None,
        dist=SimpleNamespace(rank=4, tp_rank=1, pp_rank=0, cp_rank=0),
        effects=SimpleNamespace(prepare_gen_resources=Mock()),
        registry=SimpleNamespace(active_requests=lambda: [request]),
        enable_attention_dp=False,
        force_terminate_ctx_for_partial_reuse=False,
    )
    coordinator.reap_gen_receives = Mock()

    emit_event = Mock()
    monkeypatch.setattr(diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", True)
    monkeypatch.setattr(diagnostics, "emit_event", emit_event)
    monkeypatch.setattr(
        coordinator_module.time,
        "monotonic",
        lambda: 10.0,
    )
    monkeypatch.setattr(
        coordinator_module,
        "is_disagg_inflight_cancel_enabled",
        lambda: False,
    )

    monkeypatch.delenv("TRTLLM_DISAGG_BENCHMARK_GEN_ONLY", raising=False)
    monkeypatch.delenv("TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP", raising=False)
    coordinator.receive_gen_init([request])
    coordinator._effects.prepare_gen_resources.assert_called_once_with([request])
    coordinator.reap_gen_receives.assert_called_once_with(0)
    monkeypatch.setattr(coordinator_module.time, "monotonic", lambda: 10.2)
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


@pytest.mark.parametrize("preparation_succeeds", [False, True])
def test_gen_decode_ready_requires_successful_handoff_and_uses_full_cp_prompt_length(
    monkeypatch: pytest.MonkeyPatch, preparation_succeeds: bool
) -> None:
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

    def prepare_sampler(_requests) -> None:
        if not preparation_succeeds:
            request.state = LlmRequestState.DISAGG_TRANS_ERROR
            request.is_disagg_generation_transmission_complete = False

    executor._setup_sampler_step = Mock(side_effect=prepare_sampler)
    executor.enable_spec_decode = False
    executor.model_engine = SimpleNamespace()
    executor.draft_model_engine = None
    executor.kv_cache_transceiver = SimpleNamespace(commit_blocks_for_reuse=Mock())
    executor._update_sampler_state_for_disagg_gen_request = Mock(return_value=True)
    executor._maybe_prepend_logprobs_and_logits = Mock()
    executor.global_rank = 4
    executor.dist = SimpleNamespace(tp_rank=1, pp_rank=0, cp_rank=3)
    executor._disagg_coordinator = _coordinator()
    executor._disagg_coordinator._transceiver = executor.kv_cache_transceiver

    emit_event = Mock()
    monkeypatch.setattr(diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", True)
    monkeypatch.setattr(diagnostics, "emit_event", emit_event)

    executor._prepare_disagg_gen_transmission_complete(
        SimpleNamespace(generation_requests=[request]),
    )

    executor._setup_sampler_step.assert_called_once()
    if not preparation_succeeds:
        emit_event.assert_not_called()
        executor.kv_cache_transceiver.commit_blocks_for_reuse.assert_not_called()
        request.add_new_token.assert_not_called()
        assert request.state == LlmRequestState.DISAGG_TRANS_ERROR
        return
    event = emit_event.call_args
    assert event.args == ("gen_decode_ready",)
    assert event.kwargs["prompt_tokens"] == 257
    assert event.kwargs["cp_rank"] == 3
    executor.kv_cache_transceiver.commit_blocks_for_reuse.assert_called_once_with(request)
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
    )

    monkeypatch.setattr(diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", True)
    failing_emit = Mock(side_effect=RuntimeError("diagnostics failed"))
    monkeypatch.setattr(diagnostics, "emit_event", failing_emit)
    monkeypatch.setattr(coordinator_module.time, "monotonic", lambda: 12.5)

    coordinator.send_completed_context([request])

    assert failing_emit.call_count > 0
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
    assert settlement["resources_drained"] is True
    assert request.py_kv_transfer_start_time is None


@pytest.mark.parametrize(
    ("wait_result", "session_status", "outcome", "request_state"),
    [
        (
            WaitResult.COMPLETED,
            SessionStatus.TRANSFERRED,
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
        _kv_tasks=[],
        wait_complete=Mock(side_effect=lambda blocking: operations.append("wait") or wait_result),
        has_transferring_tasks=Mock(return_value=False),
        close=Mock(side_effect=lambda: operations.append("close") or True),
        kv_write_verified=Mock(return_value=True),
    )

    def receive(_chunk: Chunk, *, expected_write_bytes: int) -> None:
        assert expected_write_bytes == 64
        operations.append("receive")
        session._kv_tasks.append(SimpleNamespace(status=TaskStatus.TRANSFERRED))

    session.receive = Mock(side_effect=receive)
    request = _disagg_request(41, 4041)
    request.state = LlmRequestState.DISAGG_GENERATION_INIT
    request.set_kv_cache_size = Mock()
    transceiver = _diagnostic_transceiver()
    transceiver._validate_bridge_req = Mock(return_value=True)
    transceiver._recv_sessions = {}
    transceiver._recv_reqs = {}
    transceiver._transfer_worker = SimpleNamespace(create_rx_session=Mock(return_value=session))
    transceiver._create_cache_extent = Mock(return_value=_extent(4041))
    transceiver._chunk_num_bytes = Mock(return_value=64)
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
    assert settlement["session_retired"] is True
    assert "resources_drained" not in settlement
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
        _kv_tasks=[],
        wait_complete=Mock(side_effect=RuntimeError("receive failed")),
        has_transferring_tasks=Mock(return_value=True),
        close=Mock(return_value=False),
    )
    session.receive = Mock(
        side_effect=lambda _chunk, **_kwargs: session._kv_tasks.append(
            SimpleNamespace(status=TaskStatus.TRANSFERRING)
        )
    )
    request = _disagg_request(44, 4044)
    request.state = LlmRequestState.DISAGG_GENERATION_INIT
    transceiver = _diagnostic_transceiver()
    transceiver._validate_bridge_req = Mock(return_value=True)
    transceiver._recv_sessions = {}
    transceiver._recv_reqs = {}
    transceiver._transfer_worker = SimpleNamespace(create_rx_session=Mock(return_value=session))
    transceiver._create_cache_extent = Mock(return_value=_extent(4044))
    transceiver._chunk_num_bytes = Mock(return_value=64)
    transceiver._kv_size_rank_factor = 2

    emit_event = Mock()
    monkeypatch.setattr(diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", True)
    monkeypatch.setattr(diagnostics, "emit_event", emit_event)

    with pytest.raises(RuntimeError, match="receive failed"):
        transceiver.request_and_receive_sync(request)

    assert [call.args[0] for call in emit_event.call_args_list] == ["gen_receive_start"]
    session.wait_complete.assert_called_once_with(blocking=True)
    session.close.assert_called_once_with()
    assert transceiver._recv_sessions == {4044: session}
    assert transceiver._recv_reqs == {4044: request}


def test_sync_receive_disabled_diagnostics_preserves_receive_failure_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = SimpleNamespace(
        receive=Mock(side_effect=RuntimeError("receive failed")),
        fail_admission=Mock(),
        _kv_tasks=[],
        close=Mock(return_value=True),
    )
    request = _disagg_request(45, 4045)
    request.state = LlmRequestState.DISAGG_GENERATION_INIT
    transceiver = _diagnostic_transceiver()
    transceiver._validate_bridge_req = Mock(return_value=True)
    transceiver._recv_sessions = {}
    transceiver._recv_reqs = {}
    transceiver._transfer_worker = SimpleNamespace(create_rx_session=Mock(return_value=session))
    extent = _extent(4045)
    transceiver._create_cache_extent = Mock(return_value=extent)
    transceiver._chunk_num_bytes = Mock(return_value=64)

    monkeypatch.setattr(diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", False)

    with pytest.raises(RuntimeError, match="receive failed"):
        transceiver.request_and_receive_sync(request)

    transceiver._create_cache_extent.assert_called_once_with(request)
    session.receive.assert_called_once_with(extent.local, expected_write_bytes=64)
    session.fail_admission.assert_called_once()
    transceiver._chunk_num_bytes.assert_called_once_with(extent.local)
    assert request.state == LlmRequestState.DISAGG_TRANS_ERROR
    assert transceiver._recv_sessions == {}
    assert transceiver._recv_reqs == {}


@pytest.mark.parametrize("enabled", [False, True])
def test_async_receive_extent_failure_does_not_start_or_report_transfer(
    monkeypatch: pytest.MonkeyPatch, enabled: bool
) -> None:
    request = _disagg_request(46, 4046)
    request.state = LlmRequestState.DISAGG_GENERATION_INIT
    request.set_kv_cache_transfer_start = Mock()
    transceiver = _diagnostic_transceiver()
    transceiver._validate_bridge_req = Mock(return_value=True)
    transceiver._recv_sessions = {}
    transceiver._recv_reqs = {}
    transceiver._create_cache_extent = Mock(side_effect=RuntimeError("extent build failed"))
    transceiver._open_peer_source = Mock(side_effect=AssertionError("must not open a peer source"))
    emit_event = Mock()
    monkeypatch.setattr(diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", enabled)
    monkeypatch.setattr(diagnostics, "emit_event", emit_event)
    monkeypatch.setattr(
        python_transceiver_module.tensorrt_llm.bindings, "global_steady_clock_now", lambda: 0
    )

    with pytest.raises(RuntimeError, match="extent build failed"):
        transceiver.request_and_receive_async(request)

    assert request.state == LlmRequestState.DISAGG_GENERATION_INIT
    assert transceiver._recv_sessions == {}
    assert transceiver._recv_reqs == {}
    transceiver._open_peer_source.assert_not_called()
    emit_event.assert_not_called()


def test_fallback_identity_matches_session_without_changing_local_table_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _disagg_request(41, None)
    request.py_disaggregated_params.ctx_request_id = 7
    request.state = LlmRequestState.DISAGG_GENERATION_INIT
    request.set_kv_cache_transfer_start = Mock()
    manager = SimpleNamespace(
        prepare_disagg_gen_init=Mock(return_value=True),
        kv_cache_map={41: SimpleNamespace(capacity=65, history_length=64)},
        mapping=SimpleNamespace(rank=4),
        tokens_per_block=32,
    )
    scheduler = object.__new__(KVCacheV2Scheduler)
    scheduler.kv_cache_manager = manager
    scheduler.tokens_per_block = 32
    coordinator = _coordinator()
    session = SimpleNamespace(
        disagg_request_id=7,
        _base_args=SimpleNamespace(params=request.py_disaggregated_params),
        _kv_tasks=[],
        status=SessionStatus.INIT,
        has_transferring_tasks=Mock(return_value=False),
        close=Mock(return_value=True),
    )
    session.receive = Mock(
        side_effect=lambda _chunk, **_kwargs: session._kv_tasks.append(
            SimpleNamespace(status=TaskStatus.TRANSFERRING)
        )
    )
    session.cancel = Mock(side_effect=lambda: setattr(session, "status", SessionStatus.CANCELLED))
    transceiver = _diagnostic_transceiver()
    transceiver._validate_bridge_req = Mock(return_value=True)
    transceiver._send_sessions = {}
    transceiver._recv_sessions = {}
    transceiver._recv_reqs = {}
    transceiver._wait_reqs = {}
    transceiver._transfer_worker = SimpleNamespace(create_rx_session=Mock(return_value=session))
    transceiver._create_cache_extent = Mock(return_value=_extent(41))
    transceiver._chunk_num_bytes = Mock(return_value=64)
    transceiver._kv_size_rank_factor = 2
    transceiver.kv_transfer_timeout_ms = None
    coordinator._transceiver = transceiver
    coordinator._kv_cache_manager = manager
    coordinator._effects.prepare_gen_resources = Mock()
    coordinator.reap_gen_receives = Mock()
    emit_event = Mock()
    monkeypatch.setattr(diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", True)
    monkeypatch.setattr(diagnostics, "emit_event", emit_event)
    monkeypatch.delenv("TRTLLM_DISAGG_BENCHMARK_GEN_ONLY", raising=False)
    monkeypatch.delenv("TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP", raising=False)
    monkeypatch.setattr(
        python_transceiver_module.tensorrt_llm.bindings, "global_steady_clock_now", lambda: 0
    )

    assert scheduler._try_schedule_disagg_gen_init(request, None) == (ScheduleAction.SCHEDULED, 0)
    assert coordinator.admit([request]) == ([request], False)
    coordinator.receive_gen_init([request])

    manager.prepare_disagg_gen_init.assert_called_once_with(request)
    transceiver._create_cache_extent.assert_called_once_with(request)
    assert transceiver._recv_sessions == {41: session}
    assert transceiver._recv_reqs == {41: request}
    assert transceiver.cancel_request(request) is True
    assert transceiver._recv_sessions == {}
    assert transceiver._recv_reqs == {}
    session.cancel.assert_called_once_with()
    session.close.assert_called_once_with()
    assert [call.args[0] for call in emit_event.call_args_list] == [
        "gen_kv_admission_result",
        "gen_receive_start",
        "transfer_cancel_requested",
        "gen_transfer_settled",
    ]
    for call in emit_event.call_args_list:
        assert call.kwargs["request_id"] == 7
        assert call.kwargs["local_request_id"] == 41
        assert call.kwargs["request_id_scope"] == "process"
    assert emit_event.call_args_list[0].kwargs["capacity_tokens"] == 65
    assert emit_event.call_args_list[-1].kwargs["session_retired"] is True


@pytest.mark.parametrize("enforce_physical_ownership", [False, True])
@pytest.mark.parametrize(
    ("disagg_request_id", "ctx_request_id", "expected_id", "expected_scope"),
    [(7007, 7, 7007, "run"), (None, 7, 7, "process")],
)
def test_session_only_settlement_uses_session_identity_and_scope(
    monkeypatch: pytest.MonkeyPatch,
    enforce_physical_ownership: bool,
    disagg_request_id: int | None,
    ctx_request_id: int,
    expected_id: int,
    expected_scope: str,
) -> None:
    session = SimpleNamespace(
        disagg_request_id=expected_id,
        _enforce_physical_ownership=enforce_physical_ownership,
        resources_drained=Mock(side_effect=AssertionError("diagnostics polled physical drain")),
        _base_args=SimpleNamespace(
            params=DisaggregatedParams(
                disagg_request_id=disagg_request_id, ctx_request_id=ctx_request_id
            )
        ),
        status=SessionStatus.TRANSFERRED,
        has_transferring_tasks=Mock(return_value=False),
    )
    emit_event = Mock()
    monkeypatch.setattr(diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", True)
    monkeypatch.setattr(diagnostics, "emit_event", emit_event)

    _diagnostic_transceiver()._emit_transfer_settled("gen", 41, None, session, "completed")

    emit_event.assert_called_once()
    assert emit_event.call_args.args == ("gen_transfer_settled",)
    fields = emit_event.call_args.kwargs
    assert fields["request_id"] == expected_id
    assert fields["request_id_scope"] == expected_scope
    assert fields["local_request_id"] is None
    assert fields["session_retired"] is True
    assert "resources_drained" not in fields
    session.resources_drained.assert_not_called()


def test_settlement_without_request_or_session_reports_unknown_identity_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emit_event = Mock()
    monkeypatch.setattr(diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", True)
    monkeypatch.setattr(diagnostics, "emit_event", emit_event)

    _diagnostic_transceiver()._emit_transfer_settled("gen", 41, None, None, "failed")

    emit_event.assert_called_once()
    assert emit_event.call_args.args == ("gen_transfer_settled",)
    assert emit_event.call_args.kwargs["request_id"] == 41
    assert emit_event.call_args.kwargs["request_id_scope"] == "unknown"
    assert emit_event.call_args.kwargs["local_request_id"] is None


@pytest.mark.parametrize("side", ("ctx", "gen"))
@pytest.mark.parametrize("inflight", [False, True])
def test_cancel_diagnostics_do_not_poll_session_status(
    monkeypatch: pytest.MonkeyPatch, side: str, inflight: bool
) -> None:
    """Observe cancellation without advancing state through a status getter."""
    status_poll = Mock(side_effect=AssertionError("diagnostics polled session status"))

    class Session(SimpleNamespace):
        @property
        def status(self) -> SessionStatus:
            return status_poll()

    request = _disagg_request(42, 4042)
    request.py_kv_send_session_retired = False
    session = Session(
        cancel=Mock(),
        has_transferring_tasks=Mock(return_value=inflight),
        close=Mock(return_value=True),
    )
    transceiver = _diagnostic_transceiver()
    transceiver._wait_reqs = {}
    transceiver._send_sessions = {4042: session} if side == "ctx" else {}
    transceiver._send_reqs = {4042: request} if side == "ctx" else {}
    transceiver._recv_sessions = {4042: session} if side == "gen" else {}
    transceiver._recv_reqs = {4042: request} if side == "gen" else {}

    settlement_cleanup: list[bool] = []

    def record_settlement_cleanup(event: str, **_fields: object) -> None:
        """Capture cleanup ordering for assertions outside diagnostic suppression."""
        if event.endswith("_transfer_settled"):
            settlement_cleanup.append(
                not any(
                    (
                        transceiver._send_sessions,
                        transceiver._recv_sessions,
                        transceiver._send_reqs,
                        transceiver._recv_reqs,
                    )
                )
            )

    emit = Mock(side_effect=record_settlement_cleanup)
    monkeypatch.setattr(diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", True)
    monkeypatch.setattr(diagnostics, "emit_event", emit)

    assert transceiver.cancel_request(request) is not inflight
    session.cancel.assert_called_once_with()
    session.has_transferring_tasks.assert_called_once_with()
    status_poll.assert_not_called()
    expected = ["transfer_cancel_requested"]
    if not inflight:
        assert transceiver.cancel_request(request) is True
        expected.append(f"{side}_transfer_settled")
        session.close.assert_called_once_with()
        settlement = emit.call_args_list[-1].kwargs
        assert settlement["outcome"] == "cancelled"
        assert settlement["session_retired"] is True
        assert "resources_drained" not in settlement
        assert not transceiver._send_sessions
        assert not transceiver._recv_sessions
        if side == "ctx":
            assert request.py_kv_send_session_retired is True
    else:
        session.close.assert_not_called()
        sessions = transceiver._send_sessions if side == "ctx" else transceiver._recv_sessions
        requests = transceiver._send_reqs if side == "ctx" else transceiver._recv_reqs
        assert sessions == {4042: session}
        assert requests == {4042: request}
    assert [call.args[0] for call in emit.call_args_list] == expected
    assert settlement_cleanup == ([] if inflight else [True])
    for call in emit.call_args_list:
        assert call.kwargs["request_id"] == 4042
        assert call.kwargs["local_request_id"] == 42
        assert call.kwargs["request_id_scope"] == "run"
        assert "session_status" not in call.kwargs


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
    coordinator = _coordinator(controller=controller, consumes_transfer_buffer=True)
    monkeypatch.delenv("TRTLLM_DISAGG_BENCHMARK_GEN_ONLY", raising=False)
    emit_event = Mock(side_effect=RuntimeError("diagnostics failed"))
    monkeypatch.setattr(diagnostics, "emit_event", emit_event)
    monkeypatch.setattr(diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", True)

    admitted, wait_for_progress = coordinator.admit(candidates)

    assert admitted == [candidates[0]]
    assert wait_for_progress is False
    coordinator._effects.revert_ctx_alloc.assert_called_once_with([candidates[1]])
    assert emit_event.call_args_list[-1].args == ("gen_kv_rollback",)
    assert emit_event.call_args_list[-1].kwargs["reason"] == "transfer_window"


def test_pp_reconciliation_emits_rollback_after_releasing_kv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admitted = _disagg_request(33, 3033)
    deferred = _disagg_request(34, 3034)
    operations = []
    coordinator = _coordinator()
    coordinator._effects.revert_ctx_alloc = lambda requests: operations.append(("revert", requests))
    coordinator._dist.pp_rank = 2
    coordinator._dist.cp_rank = 3

    def emit_event(event: str, **kwargs) -> None:
        operations.append((event, kwargs))

    monkeypatch.setattr(diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", True)
    monkeypatch.setattr(diagnostics, "emit_event", emit_event)

    coordinator.revert_deferred_gen_init(
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
    request = _disagg_request(4, request_id)
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
    request = _disagg_request(6, request_id)
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._ever_had_send_session = True
    transceiver._ctx_need_tp_sync = False
    transceiver._ctx_need_pp_sync = False
    transceiver._send_sessions = {request_id: session}
    transceiver._send_reqs = {request_id: request}
    transceiver._collect_done = Mock(return_value=([request_id], []))
    transceiver._ctx_consensus = Mock(side_effect=lambda request_ids: request_ids)
    transceiver._build_to_process = Mock(return_value=[request_id])

    def retire_before_diagnostic_snapshot(*_args) -> tuple[list, list, list, list]:
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
        kv_write_verified=Mock(return_value=True),
        has_transferring_tasks=Mock(return_value=False),
    )
    request = _disagg_request(5, request_id)
    request.set_kv_cache_size = Mock()
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._ever_had_recv_session = True
    transceiver._gen_need_sync = False
    transceiver._recv_sessions = {request_id: session}
    transceiver._recv_reqs = {request_id: request}
    transceiver._collect_done = Mock(return_value=([request_id], []))
    transceiver._gen_consensus = Mock(side_effect=lambda request_ids: request_ids)
    transceiver._build_to_process = Mock(return_value=[request_id])
    transceiver._gen_consensus_outcome = Mock(return_value=([], [], [request_id], [request_id]))
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
    session.kv_write_verified.assert_called_once_with()
    assert request.py_kv_transfer_verified is True
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
    else:
        task.begin_backend_submission.assert_not_called()
        task.record_backend_submission.assert_not_called()
        task.retire_backend_done_physical_operation.assert_not_called()


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
    executor.sampler = SimpleNamespace(release_request=Mock())
    executor.resource_manager = SimpleNamespace(free_resources=Mock())
    executor._prefetched_request_ids = {request.py_request_id}
    executor._disagg_coordinator = SimpleNamespace(forget_request=Mock())
    executor.model_engine = SimpleNamespace(model=SimpleNamespace(release_request_state=Mock()))
    executor.draft_model_engine = SimpleNamespace(
        model=SimpleNamespace(release_request_state=Mock())
    )
    executor.global_rank = 4
    executor.dist = SimpleNamespace(tp_rank=1, pp_rank=0, cp_rank=0)
    monkeypatch.setattr(diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", True)
    failing_emit = Mock(side_effect=RuntimeError("diagnostics failed"))
    monkeypatch.setattr(diagnostics, "emit_event", failing_emit)

    executor._free_request_resources(request)

    failing_emit.assert_called_once()
    executor.sampler.release_request.assert_called_once_with(request)
    executor.resource_manager.free_resources.assert_called_once_with(request)
    assert request.py_request_id not in executor._prefetched_request_ids
    executor._disagg_coordinator.forget_request.assert_called_once_with(request.py_request_id)
    for engine in (executor.model_engine, executor.draft_model_engine):
        engine.model.release_request_state.assert_called_once_with(request.py_request_id)


def test_source_unpin_continues_when_diagnostic_inspection_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _KVCacheManager:
        def __init__(self) -> None:
            self.unpin_blocks_by_id = Mock()
            self.mapping_accessed = False

        @property
        def mapping(self) -> SimpleNamespace:
            self.mapping_accessed = True
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
    assert manager.kv_cache_manager.mapping_accessed
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
    receiver._pre_cancelled_rids = {}
    receiver._shutdown = False
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

    session.receive(_extent(request_id).local)
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


@pytest.mark.parametrize("mode", ["disabled", "enabled", "failing"])
@pytest.mark.parametrize("anchor", ["send", "fallback"])
def test_native_deadline_diagnostics_preserve_fixed_nonterminal_timeout(
    monkeypatch: pytest.MonkeyPatch, mode: str, anchor: str
) -> None:
    """Observe both existing anchors without rearming or retiring the session."""
    now = [10.0]
    sender = SimpleNamespace(
        setup_session=Mock(),
        clear_session=Mock(),
        dispatch_task=Mock(),
        _get_req_info=lambda _rid: {},
        _enforce_physical_ownership=False,
        _registrar=SimpleNamespace(self_rank_info=SimpleNamespace(instance_rank=0)),
    )
    emit = Mock(side_effect=RuntimeError("diagnostics failed") if mode == "failing" else None)
    monkeypatch.setattr(diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", mode != "disabled")
    monkeypatch.setattr(diagnostics, "emit_event", emit)
    monkeypatch.setattr(transfer_module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(transfer_module.tensorrt_llm.bindings, "global_steady_clock_now", lambda: 0)
    session = transfer_module.TxSession(
        request_id=7,
        params=DisaggregatedParams(disagg_request_id=7007),
        sender=sender,
        timeout_s=0.1,
        overall_timeout_s=1.0,
    )
    session.send(_extent(7).local)
    session.send(_extent(7).local)
    assert session._deadline_monotonic_s == 11.0
    assert sender.dispatch_task.call_count == 2

    if anchor == "fallback":
        session._deadline_monotonic_s = None
        emit.reset_mock()
        clock = iter((10.0, 12.0, 12.0))
        monkeypatch.setattr(transfer_module.time, "monotonic", lambda: next(clock))
    else:
        now[0] = 12.0
    assert session.wait_complete() is WaitResult.TIMEOUT
    assert session.wait_complete() is WaitResult.TIMEOUT
    assert session._deadline_monotonic_s == 11.0
    assert session._terminal_status is None
    assert not session._closed
    assert all(task.status is TaskStatus.INIT for task in session.kv_tasks)
    assert session.close()
    if mode == "disabled":
        emit.assert_not_called()
        return
    assert [call.args[0] for call in emit.call_args_list] == [
        "ctx_native_deadline_started",
        "ctx_native_deadline_observed",
        "ctx_native_deadline_observed",
    ]
    for call in emit.call_args_list:
        assert call.kwargs["request_id"] == 7007
        assert call.kwargs["request_id_scope"] == "run"
        assert call.kwargs["timeout_owner"] == "native_tx_session"
        assert call.kwargs["deadline_monotonic_ns"] == 11_000_000_000
    assert emit.call_args_list[0].kwargs["timer_start_monotonic_ns"] == 10_000_000_000


@pytest.mark.parametrize("mode", ["disabled", "enabled", "failing"])
def test_native_profile_diagnostics_do_not_affect_worker_setup(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """Declare only native-path identity after successful worker setup."""
    rank_info = SimpleNamespace(instance_rank=0)
    monkeypatch.setattr(transfer_module, "_make_aux_buffer", lambda *_args: None)
    monkeypatch.setattr(transfer_module.RankInfo, "from_kv_cache_manager", lambda *_args: rank_info)
    setup_peer = Mock()
    setup_engine = Mock()
    monkeypatch.setattr(transfer_module.TransferWorker, "_setup_peer_infrastructure", setup_peer)
    monkeypatch.setattr(transfer_module.TransferWorker, "_setup_transfer_engine", setup_engine)
    emit = Mock(side_effect=RuntimeError("diagnostics failed") if mode == "failing" else None)
    monkeypatch.setattr(diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", mode != "disabled")
    monkeypatch.setattr(diagnostics, "emit_event", emit)
    config = transfer_module.TransferWorkerConfig(
        kv_cache_manager=object(),
        max_concurrent_sessions=1,
        max_draft_len=0,
        instance_name="test-native",
        device_id=0,
    )
    worker = transfer_module.TransferWorker(config)
    assert worker._rank_info is rank_info
    setup_peer.assert_called_once_with(config.kv_cache_manager)
    setup_engine.assert_called_once_with()
    if mode == "disabled":
        emit.assert_not_called()
    else:
        emit.assert_called_once_with(
            "diagnostic_profile",
            side="runtime",
            request_id=None,
            rank_info=rank_info,
            profile="native_v1",
        )
