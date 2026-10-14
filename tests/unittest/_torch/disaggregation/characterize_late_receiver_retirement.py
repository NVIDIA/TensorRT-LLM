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
"""Explicit QA-only characterization of late context-first receiver arrival.

Invoke this file by its full pytest path. Its name intentionally excludes it
from ordinary test discovery because its characterization assertions describe
current undesirable behavior, not a permanent regression contract.

The coordinator, transfer manager, transceiver, publication adapter, session,
and sender are production implementations. Memory descriptions and peer
endpoints are CPU substitutes; the early control uses a zero-byte send through
the real result path. No network threads or GPU transfers run. Normal
TensorRT-LLM imports and native bindings are still required.

A passing characterization observes an unanswered late request. It is not a
fix assertion, a GPU reproduction, or evidence of what delayed the receiver.
"""

import queue
import threading
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Literal
from unittest.mock import Mock

import numpy as np
import pytest
from coordinator_harness import TransferRequest
from fake_executor_effects import FakeExecutorEffects, FakeRequestRegistry

import tensorrt_llm.bindings
from tensorrt_llm._torch.disaggregation.base import CacheExtent, Chunk, TokenRange
from tensorrt_llm._torch.disaggregation.base.transfer import SessionStatus
from tensorrt_llm._torch.disaggregation.native import transfer as transfer_mod
from tensorrt_llm._torch.disaggregation.native.bounce.impl import NoBounceTransport
from tensorrt_llm._torch.disaggregation.orchestration import coordinator as coordinator_mod
from tensorrt_llm._torch.disaggregation.orchestration.coordinator import DisaggTransferCoordinator
from tensorrt_llm._torch.disaggregation.orchestration.transfer_manager import AsyncTransferManager
from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType
from tensorrt_llm.bindings import LlmRequestState
from tensorrt_llm.disaggregated_params import DisaggregatedParams, DisaggScheduleStyle

pytestmark = pytest.mark.cpu_only

_REQUEST_ID = 41
_TIMEOUT_MS = 60_000


def _write_meta(
    task: transfer_mod.KVSendTask, info: transfer_mod.RecvReqInfo
) -> transfer_mod.WriteMeta:
    """Describe a synthetic backend operation without allocating device memory."""
    task.expected_transfers = 1
    return transfer_mod.WriteMeta(
        task=task,
        expected_transfers=1,
        peer_name="gen0",
        peer_rank=info.instance_rank,
        peer_endpoint="tcp://gen:1234",
        unique_rid=info.unique_rid,
        src_ptrs=np.array([], dtype=np.int64),
        dst_ptrs=np.array([], dtype=np.int64),
        sizes=np.array([], dtype=np.int64),
        slice_id=task.slice_id,
        receiver_slice_id=0,
        is_last_slice=True,
    )


def _sender() -> transfer_mod.Sender:
    """Keep real sender bookkeeping and dispatch, replacing transport setup."""
    sender = object.__new__(transfer_mod.Sender)
    sender._enforce_physical_ownership = False
    sender._sessions_lock, sender._sessions = threading.Lock(), {}
    sender._pre_cancelled_rids = {}
    sender._peer_requests_lock = threading.Lock()
    sender._peer_requests, sender._peer_requests_timestamps = {}, {}
    # No listener/worker threads were started, so destructor shutdown is a no-op.
    sender._shutdown, sender._shutdown_requested = True, False
    sender._retirement_watchdog = None
    sender._instance_rank = 0
    sender._num_threads = 1
    sender._send_task_queues = [queue.Queue()]
    sender._agent = Mock()
    sender._registrar = Mock()
    sender._registrar.get_peer_rank_info.return_value = SimpleNamespace(
        instance_name="gen", instance_rank=0, dp_rank=0, self_endpoint="tcp://gen:1234"
    )
    sender._registrar.get_peer_overlap.return_value = SimpleNamespace(ranks=[0])
    sender._registrar.self_rank_info = SimpleNamespace(instance_name="ctx", instance_rank=0)
    dealer = Mock()
    sender._get_or_connect_dealer = Mock(return_value=dealer)
    sender._get_or_connect_thread_dealer = Mock(return_value=dealer)
    sender._build_kv_write_meta = Mock(side_effect=_write_meta)
    return sender


class _ContextFirstCase:
    """One local sender with real lifecycle code and controlled external I/O."""

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        clock: dict[str, float],
        *,
        request_id: int = _REQUEST_ID,
        sender: transfer_mod.Sender | None = None,
    ) -> None:
        """Start one context send, optionally reusing an existing sender."""
        self.clock = clock
        self.request_id = request_id
        monkeypatch.setattr(coordinator_mod, "is_disagg_inflight_cancel_enabled", lambda: False)
        monkeypatch.setattr(tensorrt_llm.bindings, "global_steady_clock_now", lambda: 0)
        self.sender = sender if sender is not None else _sender()
        self.request = TransferRequest(
            request_id,
            py_kv_send_session_retired=False,
            py_disaggregated_params=DisaggregatedParams(
                disagg_request_id=request_id,
                schedule_style=DisaggScheduleStyle.CONTEXT_FIRST,
            ),
            set_kv_cache_transfer_start=Mock(),
        )
        self.extent = CacheExtent(
            name=request_id,
            local=Chunk(
                block_ids_per_layer_groups=[],
                kind_per_layer_group=[],
                token_range=TokenRange(start=0, end=1),
                is_last=True,
            ),
        )
        self.cache = Mock(spec=["store_blocks_for_reuse", "unpin_blocks_by_id"])
        self.cache.store_blocks_for_reuse.return_value = request_id
        self.transfers = AsyncTransferManager(
            SimpleNamespace(resource_managers={ResourceManagerType.KV_CACHE_MANAGER: self.cache})
        )
        self.effects = FakeExecutorEffects()
        self.registry = FakeRequestRegistry([self.request])
        self.dist = SimpleNamespace(
            rank=0,
            tp_size=1,
            pp_size=1,
            world_size=1,
            tp_allgather=Mock(side_effect=AssertionError("unexpected collective")),
        )
        transceiver = object.__new__(KvCacheTransceiverV2)
        transceiver._fp4_mla_bridge_enabled = False
        transceiver._enable_pipelined_transfer = False
        transceiver.kv_transfer_timeout_ms = _TIMEOUT_MS
        transceiver._sender_future_timeout_ms = 1000
        transceiver._ever_had_send_session = False
        transceiver._ever_had_recv_session = False
        transceiver._ctx_need_tp_sync = False
        transceiver._ctx_need_pp_sync = False
        transceiver._dist = self.dist
        transceiver._mapping = SimpleNamespace(
            enable_attention_dp=True, rank=0, cp_size=1, pp_size=1, world_size=1
        )
        transceiver._gen_need_sync = False
        transceiver._gen_allgather = Mock(side_effect=AssertionError("unexpected collective"))
        transceiver._dp_rank = 0
        transceiver._context_info_endpoint = "tcp://ctx:1234"
        transceiver._send_sessions = {}
        transceiver._send_reqs = {}
        transceiver._recv_sessions = {}
        transceiver._recv_reqs = {}
        transceiver._wait_reqs = {}
        transceiver._create_cache_extent = Mock(return_value=self.extent)
        transceiver._transfer_worker = SimpleNamespace(
            create_tx_session=self._create_tx_session,
            sweep_stale_req_infos=self.sender.sweep_stale_req_infos,
        )
        self.transceiver = transceiver
        self.coordinator = DisaggTransferCoordinator(
            transceiver=transceiver,
            transfer_manager=self.transfers,
            kv_cache_manager=self.cache,
            dist=self.dist,
            effects=self.effects,
            registry=self.registry,
            enable_attention_dp=True,
            force_terminate_ctx_for_partial_reuse=False,
        )
        self.coordinator.send_completed_context([self.request])
        self.session = transceiver._send_sessions[request_id]
        self.started_at = self.request.py_kv_transfer_start_time
        assert self.started_at == clock["t"]
        assert self.session.status is SessionStatus.INIT
        assert self.sender._send_task_queues[0].empty()

    def _create_tx_session(self, request: TransferRequest) -> transfer_mod.TxSession:
        """Replace only the hardware-backed worker's session factory."""
        return transfer_mod.TxSession(
            request_id=request.py_request_id,
            params=request.py_disaggregated_params,
            sender=self.sender,
            timeout_s=1.0,
            overall_timeout_s=_TIMEOUT_MS / 1000,
            prompt_len=1,
        )

    def request_data(self) -> None:
        """Deliver real wire-format receiver metadata to the sender handler."""
        info = transfer_mod.RecvReqInfo(
            sender_req_id=self.request_id,
            instance_name="gen",
            instance_rank=0,
            block_ids_per_layer_groups=[],
            unique_rid=self.request_id,
            slice_id=0,
        )
        self.sender._respond_with_kv(
            b"gen", [transfer_mod.MessageType.REQUEST_DATA, info.to_bytes()]
        )

    def advance_and_reap(self, elapsed_seconds: float) -> None:
        """Run production timeout detection and send cleanup at a chosen time."""
        self.clock["t"] = self.started_at + elapsed_seconds
        self.coordinator.check_transfer_timeouts()
        self.coordinator.reap_context_sends()

    def assert_retired_once(self) -> None:
        """Check real session removal and exactly-once transfer-claim release."""
        assert self.session._closed
        assert self.request.py_kv_send_session_retired
        assert not self.transceiver._send_sessions
        assert not self.transceiver._send_reqs
        assert self.sender._get_session(self.request_id) is None
        assert not self.transfers.requests_in_transfer()
        self.cache.unpin_blocks_by_id.assert_called_once_with(self.request_id)
        assert self.effects.terminated == [self.request]
        assert self.registry.removed == [self.request]
        assert not self.effects.failed

    def close(self) -> None:
        """Release synthetic resources even when an assertion fails."""
        for session in self.transceiver._recv_sessions.values():
            session.close()
        self.transceiver._recv_sessions.clear()
        self.transceiver._recv_reqs.clear()
        self.session.close()
        self.transceiver._send_sessions.clear()
        self.transceiver._send_reqs.clear()
        if self.request_id in self.transfers.requests_in_transfer():
            self.coordinator.release_transfer(self.request)


@pytest.fixture
def context_first_case(
    monkeypatch: pytest.MonkeyPatch, clock: dict[str, float]
) -> Iterator[_ContextFirstCase]:
    """Close every synthetic session, including non-expired boundary cases."""
    case = _ContextFirstCase(monkeypatch, clock)
    try:
        yield case
    finally:
        case.close()


@pytest.mark.parametrize("elapsed_seconds", [59.999, 60.0, 60.001])
def test_context_sender_timeout_boundary(
    context_first_case: _ContextFirstCase, elapsed_seconds: float
) -> None:
    """Cancellation starts strictly after, never at, the configured deadline."""
    case = context_first_case
    case.advance_and_reap(elapsed_seconds)
    if elapsed_seconds <= 60.0:
        assert not case.request.py_kv_transfer_timed_out
        assert not case.session._closed
        assert case.sender._get_session(_REQUEST_ID) is case.session
        assert _REQUEST_ID in case.transfers.requests_in_transfer()
        case.cache.unpin_blocks_by_id.assert_not_called()
    else:
        assert case.request.py_kv_transfer_timed_out
        assert case.session.status is SessionStatus.CANCELLED
        case.assert_retired_once()


def test_early_receiver_dispatches_and_completed_sender_retires_once(
    context_first_case: _ContextFirstCase,
) -> None:
    """An early receiver gets a real zero-byte success response and cleanup."""
    case = context_first_case
    case.clock["t"] = case.started_at + 59.0
    case.request_data()
    queued = case.sender._send_task_queues[0].get_nowait()
    assert queued.task is case.session.kv_tasks[0]
    assert queued.task.expected_transfers == 1
    assert queued.task.reports_outstanding
    assert case.session.receiver_ready
    case.sender._build_kv_write_meta.assert_called_once()
    # A zero-byte send exercises the real success/report path without a GPU.
    case.sender._deliver_kv_to_agent(queued)
    dealer = case.sender._get_or_connect_thread_dealer.return_value
    dealer.send.assert_called_once()
    message = dealer.send.call_args.args[0]
    assert message[0] == transfer_mod.MessageType.KV_AGENT_RESULT
    assert transfer_mod._KV_RESULT_PREFIX.unpack(message[1]) == (0, _REQUEST_ID, 0, True, 0, 0)
    case.sender._agent.submit_transfer_requests.assert_not_called()
    case.coordinator.reap_context_sends()
    assert case.session.status is SessionStatus.TRANSFERRED
    assert not case.request.py_kv_transfer_timed_out
    case.assert_retired_once()
    case.coordinator.reap_context_sends()
    case.assert_retired_once()


def test_characterization_late_receiver_metadata_is_cached_without_terminal_response(
    context_first_case: _ContextFirstCase,
) -> None:
    """Observe unanswered metadata after genuine timeout-driven retirement."""
    case = context_first_case
    case.advance_and_reap(60.001)
    case.assert_retired_once()
    assert case.request.state == LlmRequestState.DISAGG_CONTEXT_COMPLETE
    assert case.sender._get_req_info(_REQUEST_ID) is None

    case.clock["t"] = case.started_at + 84.0
    case.request_data()
    assert case.sender._get_req_info(_REQUEST_ID)[0].unique_rid == _REQUEST_ID
    assert case.sender._get_session(_REQUEST_ID) is None
    assert case.sender._send_task_queues[0].empty()
    case.sender._build_kv_write_meta.assert_not_called()
    case.sender._get_or_connect_dealer.return_value.send.assert_not_called()
    case.sender._get_or_connect_thread_dealer.return_value.send.assert_not_called()
    case.sender._agent.submit_transfer_requests.assert_not_called()

    # Duplicate publication/cancellation must not recreate the retired session
    # or release the transfer claim twice. This is not a fresh router attempt.
    case.request_data()
    assert case.coordinator.request_cancellation(case.request)
    case.coordinator.send_completed_context([case.request])
    case.coordinator.reap_context_sends()
    assert len(case.sender._get_req_info(_REQUEST_ID)) == 1
    assert case.sender._send_task_queues[0].empty()
    case.assert_retired_once()


@pytest.mark.parametrize("blocker", ["active_write", "close_refused"])
def test_context_timeout_retains_sender_until_cleanup_is_safe(
    context_first_case: _ContextFirstCase,
    monkeypatch: pytest.MonkeyPatch,
    blocker: Literal["active_write", "close_refused"],
) -> None:
    """Neither a timeout flag nor cancellation alone permits source release."""
    case = context_first_case
    with monkeypatch.context() as patch:
        if blocker == "active_write":
            # External backend state: the real cancellation must retain it.
            case.session.kv_tasks[0].status = transfer_mod.TaskStatus.TRANSFERRING
        else:
            patch.setattr(case.session, "close", Mock(return_value=False))
        case.advance_and_reap(60.001)
        assert case.request.py_kv_transfer_timed_out
        assert case.session.status is SessionStatus.CANCELLED
        assert not case.session._closed
        assert not case.request.py_kv_send_session_retired
        assert case.transceiver._send_sessions[_REQUEST_ID] is case.session
        assert case.sender._get_session(_REQUEST_ID) is case.session
        assert _REQUEST_ID in case.transfers.requests_in_transfer()
        case.cache.unpin_blocks_by_id.assert_not_called()
        assert not case.effects.terminated
    if blocker == "active_write":
        # The substitute backend has now stopped accessing the synthetic source.
        case.session.kv_tasks[0].fail(RuntimeError("controlled backend has drained"))
    assert case.coordinator.request_cancellation(case.request)
    case.coordinator.reap_context_sends()
    case.assert_retired_once()


def test_fresh_request_id_can_dispatch_after_old_sender_retirement(
    context_first_case: _ContextFirstCase,
    monkeypatch: pytest.MonkeyPatch,
    clock: dict[str, float],
) -> None:
    """An expired request does not disable unrelated work on the same sender."""
    old = context_first_case
    old.advance_and_reap(60.001)
    old.request_data()
    old.assert_retired_once()
    fresh = _ContextFirstCase(monkeypatch, clock, request_id=42, sender=old.sender)
    try:
        fresh.request_data()
        queued = fresh.sender._send_task_queues[0].get_nowait()
        assert queued.unique_rid == 42
        assert queued.task is fresh.session.kv_tasks[0]
        fresh.sender._deliver_kv_to_agent(queued)
        fresh.coordinator.reap_context_sends()
        assert fresh.session.status is SessionStatus.TRANSFERRED
        fresh.assert_retired_once()
        assert old.sender._get_session(_REQUEST_ID) is None
        old.assert_retired_once()
    finally:
        fresh.close()


def test_characterization_late_generation_starts_an_independent_timeout(
    context_first_case: _ContextFirstCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real receive published after sender retirement waits its own deadline."""
    case = context_first_case
    case.advance_and_reap(60.001)
    case.assert_retired_once()
    case.clock["t"] = case.started_at + 84.0
    generation = TransferRequest(
        _REQUEST_ID,
        is_context_only_request=False,
        is_disagg_generation_transmission_in_progress=True,
        state=LlmRequestState.DISAGG_GENERATION_INIT,
        py_disaggregated_params=case.request.py_disaggregated_params,
        set_kv_cache_transfer_start=Mock(),
    )
    receiver = SimpleNamespace(
        _enforce_physical_ownership=False,
        _bounce=NoBounceTransport(),
        setup_session=Mock(),
        clear_session=Mock(),
        send_cancel_to_senders=Mock(return_value=set()),
    )
    receive_session = transfer_mod.RxSession(
        request_id=_REQUEST_ID,
        params=generation.py_disaggregated_params,
        receiver=receiver,
        timeout_s=_TIMEOUT_MS / 1000,
        prompt_len=1,
    )

    def publish(task: transfer_mod.KVRecvTask) -> None:
        """Substitute the receiver's network publication with direct delivery."""
        task.expected_transfers = 1
        receive_session.mark_transferring(task.slice_id)
        case.request_data()

    receiver.dispatch_task = publish
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._fp4_mla_bridge_enabled = False
    transceiver.kv_transfer_timeout_ms = _TIMEOUT_MS
    transceiver._ever_had_recv_session = False
    transceiver._gen_need_sync = False
    transceiver._gen_allgather = Mock(side_effect=AssertionError("unexpected collective"))
    transceiver._mapping = case.transceiver._mapping
    transceiver._recv_sessions, transceiver._recv_reqs = {}, {}
    transceiver._create_cache_extent = Mock(return_value=case.extent)
    transceiver._chunk_num_bytes = Mock(return_value=0)
    transceiver._kv_size_rank_factor = 1
    transceiver._transfer_worker = SimpleNamespace(
        create_rx_session=Mock(return_value=receive_session)
    )
    effects = FakeExecutorEffects()
    coordinator = DisaggTransferCoordinator(
        transceiver=transceiver,
        transfer_manager=AsyncTransferManager(SimpleNamespace(resource_managers={})),
        kv_cache_manager=None,
        dist=case.dist,
        effects=effects,
        registry=FakeRequestRegistry([generation]),
        enable_attention_dp=True,
        force_terminate_ctx_for_partial_reuse=False,
    )
    monkeypatch.setattr(coordinator_mod, "uses_async_gen_transfer", lambda: True)
    monkeypatch.setattr(coordinator_mod, "is_gen_only_no_context_benchmark", lambda: False)
    try:
        coordinator.receive_gen_init([generation])
        assert generation.py_kv_transfer_start_time == case.started_at + 84.0
        assert receive_session.status is SessionStatus.TRANSFERRING
        assert case.sender._get_session(_REQUEST_ID) is None
        assert case.sender._get_req_info(_REQUEST_ID)[0].unique_rid == _REQUEST_ID
        case.sender._get_or_connect_dealer.return_value.send.assert_not_called()
        assert not effects.failed
        for elapsed_seconds in (59.999, 60.0):
            case.clock["t"] = generation.py_kv_transfer_start_time + elapsed_seconds
            coordinator.check_transfer_timeouts()
            assert not generation.py_kv_transfer_timed_out
        case.clock["t"] = generation.py_kv_transfer_start_time + 60.001
        coordinator.check_transfer_timeouts()
        assert generation.py_kv_transfer_timed_out
        coordinator.fail_timed_out([generation])
        assert effects.failed == [(f"Request {_REQUEST_ID} timed out", [generation], False)]
    finally:
        # This fixture publishes no memory to a backend; close its CPU substitute.
        receive_session.close()
