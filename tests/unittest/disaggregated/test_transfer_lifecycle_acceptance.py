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
"""Component acceptance for cancellation, retirement, and allocation reuse.

Native sessions, logical handles, status polling, and the coordinator are real.
The transport, request registry, executor effects, and one-slot allocation pools
are controlled CPU substitutes; these tests do not qualify GPU/NIXL deployments.
"""

from types import SimpleNamespace
from typing import Literal
from unittest.mock import Mock, call

import pytest

import tensorrt_llm._torch.disaggregation.transceiver as transceiver_mod
from tensorrt_llm._torch.disaggregation.base import Cancelled, Chunk, TokenRange
from tensorrt_llm._torch.disaggregation.base.transfer import SessionStatus
from tensorrt_llm._torch.disaggregation.native.bounce.impl import NoBounceTransport
from tensorrt_llm._torch.disaggregation.native.handle import TaskHandle
from tensorrt_llm._torch.disaggregation.native.transfer import AgentResult, KVRecvTask, RxSession
from tensorrt_llm._torch.disaggregation.orchestration import coordinator as coordinator_mod
from tensorrt_llm._torch.disaggregation.orchestration.coordinator import DisaggTransferCoordinator
from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.resource_manager import CacheTypeCpp
from tensorrt_llm.bindings import DataType, LlmRequestState
from tensorrt_llm.disaggregated_params import DisaggregatedParams, DisaggScheduleStyle

pytestmark = pytest.mark.cpu_only

_OLD_REQUEST = 41
_NEW_REQUEST = 42
_SENTINEL = b"new-data"


class _PrematureReuse(AssertionError):
    """The pending destination was exposed to a subsequent allocation."""


class _Slot:
    def __init__(self) -> None:
        self.owner: int | None = None
        self.data = bytearray(len(_SENTINEL))
        self.release_count = 0

    def allocate(self, owner: int) -> bool:
        if self.owner is not None:
            return False
        self.owner = owner
        self.data[:] = _SENTINEL
        return True

    def release(self) -> None:
        assert self.owner == _OLD_REQUEST, "duplicate or wrong-generation release"
        self.owner = None
        self.release_count += 1


class _AuxBuffer:
    def __init__(self) -> None:
        self.slot = _Slot()

    def alloc_slot(self) -> SimpleNamespace:
        assert self.slot.allocate(_OLD_REQUEST)
        return SimpleNamespace(id=0)

    def free_slot(self, slot_id: int) -> None:
        assert slot_id == 0
        self.slot.release()


def _profile(monkeypatch: pytest.MonkeyPatch, enabled: bool) -> tuple[SimpleNamespace, bool]:
    mapping = SimpleNamespace(enable_attention_dp=True, pp_size=1, cp_size=1, world_size=1)
    manager = Mock(spec=KVCacheManagerV2)
    manager.is_disagg = True
    manager.dtype = DataType.NVFP4
    manager.kv_cache_type = CacheTypeCpp.SELFKONLY
    config = SimpleNamespace(
        backend="NIXL",
        transceiver_runtime="PYTHON",
        kv_transfer_timeout_ms=1000,
        kv_cache_bounce_size_mb=0,
        enable_pipelined_transfer=False,
    )
    monkeypatch.setenv(transceiver_mod._FP4_MLA_OWNERSHIP_BRIDGE_ENV, "1" if enabled else "0")
    monkeypatch.setenv("TRTLLM_DISAGG_NO_RETRY", "1")
    monkeypatch.delenv("TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP", raising=False)
    monkeypatch.delenv("TRTLLM_DISAGG_LAYERWISE", raising=False)
    monkeypatch.setattr(transceiver_mod, "use_pure_python_transfer_agent", lambda: False)
    ownership = transceiver_mod._validate_fp4_mla_bridge_profile(mapping, manager, config)
    assert ownership is enabled
    return mapping, ownership


class _ReceiveCase:
    def __init__(self, monkeypatch: pytest.MonkeyPatch, *, bridge_enabled: bool) -> None:
        mapping, ownership = _profile(monkeypatch, bridge_enabled)
        monkeypatch.setattr(coordinator_mod, "is_disagg_inflight_cancel_enabled", lambda: False)
        self.now = 10.0
        monkeypatch.setattr(coordinator_mod.time, "monotonic", lambda: self.now)
        self.kv = _Slot()
        assert self.kv.allocate(_OLD_REQUEST)
        self.aux = _AuxBuffer()
        self.request = SimpleNamespace(
            request_id=_OLD_REQUEST,
            py_request_id=_OLD_REQUEST,
            is_child=False,
            is_context_only_request=False,
            is_disagg_generation_transmission_in_progress=True,
            py_kv_transfer_start_time=self.now,
            py_kv_transfer_timed_out=False,
            py_disaggregated_params=DisaggregatedParams(
                disagg_request_id=_OLD_REQUEST,
                schedule_style=DisaggScheduleStyle.GENERATION_FIRST,
            ),
            state=LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS,
        )
        self.active = [self.request]
        self.cancelled_ids: set[int] = set()
        self.cleanup_count = 0
        self.receiver = SimpleNamespace(
            _enforce_physical_ownership=ownership,
            _bounce=NoBounceTransport(),
            _registrar=SimpleNamespace(
                self_rank_info=SimpleNamespace(instance_name="gen", instance_rank=0)
            ),
            setup_session=Mock(),
            clear_session=Mock(),
            send_cancel_to_senders=Mock(side_effect=lambda _rid, endpoints: set(endpoints)),
            dispatch_task=self._publish,
        )
        self.session = RxSession(
            request_id=_OLD_REQUEST,
            params=self.request.py_disaggregated_params,
            receiver=self.receiver,
            aux_buffer=self.aux,
        )
        self.session.receive(
            Chunk(
                block_ids_per_layer_groups=[],
                kind_per_layer_group=[],
                token_range=TokenRange(start=0, end=1),
                is_last=True,
            )
        )
        self.handle = TaskHandle(self.session, self.session._kv_tasks[0], token_end=1)
        assert self.handle.poll() is None
        self.transceiver = object.__new__(KvCacheTransceiverV2)
        self.transceiver._fp4_mla_bridge_enabled = ownership
        self.transceiver.kv_transfer_timeout_ms = 1000
        self.transceiver._ever_had_recv_session = True
        self.transceiver._gen_need_sync = False
        self.transceiver._gen_allgather = Mock(side_effect=AssertionError("unexpected collective"))
        self.transceiver._mapping = mapping
        self.transceiver._wait_reqs = {}
        self.transceiver._send_sessions = {}
        self.transceiver._send_reqs = {}
        self.transceiver._recv_sessions = {_OLD_REQUEST: self.session}
        self.transceiver._recv_reqs = {_OLD_REQUEST: self.request}
        assert self.transceiver._validate_bridge_req(self.request)
        self.coordinator = DisaggTransferCoordinator(
            transceiver=self.transceiver,
            transfer_manager=SimpleNamespace(requests_in_transfer=lambda: {}),
            kv_cache_manager=None,
            dist=SimpleNamespace(rank=0, tp_size=1, world_size=1),
            effects=SimpleNamespace(fail_requests=self._fail_requests),
            registry=SimpleNamespace(
                active_requests=lambda: tuple(self.active),
                canceled_request_ids=lambda: self.cancelled_ids,
            ),
            enable_attention_dp=True,
            force_terminate_ctx_for_partial_reuse=False,
        )

    def _publish(self, task: KVRecvTask) -> None:
        task.expected_transfers = 1
        if self.session._enforce_physical_ownership:
            published: set[int] = set()
            assert self.session.try_begin_transfer(
                task.slice_id,
                {"ctx"},
                writer_cohort={0},
                publish=lambda: published.add(0),
                published_writers=published,
            )
        else:
            self.session.mark_transferring(task.slice_id)

    def _release_request(self) -> None:
        self.active.remove(self.request)
        self.kv.release()
        self.cleanup_count += 1

    def _fail_requests(
        self, error_msg: str, requests: list[SimpleNamespace], *, charge_budget: bool
    ) -> None:
        assert error_msg and not charge_budget
        assert requests == [self.request]
        self._release_request()

    def cancel(self, cause: Literal["user", "timeout"]) -> None:
        if cause == "user":
            self.cancelled_ids.add(_OLD_REQUEST)
        else:
            self.coordinator.check_transfer_timeouts()
            assert not self.request.py_kv_transfer_timed_out
            self.now += 1.001
            self.coordinator.check_transfer_timeouts()
            assert self.request.py_kv_transfer_timed_out
        assert not self.coordinator.request_cancellation(self.request)
        if self.session._enforce_physical_ownership:
            self.receiver.send_cancel_to_senders.assert_any_call(_OLD_REQUEST, {"ctx"})
        self.assert_cancelled()

    def poll_and_cleanup(self) -> None:
        self.coordinator.reap_gen_receives()
        # Executor-effect seam: user cancellation and legacy timeout cleanup
        # release only after request_cancellation authorizes reclamation.
        if self.active and self.coordinator.request_cancellation(self.request):
            self._release_request()

    def assert_cancelled(self) -> None:
        assert self.session.status is SessionStatus.CANCELLED
        outcome = self.handle.poll()
        assert isinstance(outcome, Cancelled)
        assert not outcome.by_peer

    def assert_not_reusable(self) -> None:
        self.poll_and_cleanup()
        kv_reused = self.kv.allocate(_NEW_REQUEST)
        aux_reused = self.aux.slot.allocate(_NEW_REQUEST)
        if kv_reused or aux_reused:
            raise _PrematureReuse("status polling released a destination with an active writer")
        assert self.cleanup_count == self.kv.release_count == self.aux.slot.release_count == 0
        assert self.transceiver._recv_sessions[_OLD_REQUEST] is self.session
        assert self.transceiver._recv_reqs[_OLD_REQUEST] is self.request
        self.receiver.clear_session.assert_not_called()
        if self.session._enforce_physical_ownership:
            notifications = self.receiver.send_cancel_to_senders.call_args_list
            assert notifications[0] == call(_OLD_REQUEST, {"ctx"})
            assert all(item == call(_OLD_REQUEST, set()) for item in notifications[1:])
        self.assert_cancelled()

    def complete(self, resource: Literal["kv", "aux"]) -> None:
        if resource == "kv":
            self.kv.data[:] = b"old-data"
            self.session.process_kv_agent_result(0, 0, True, AgentResult.SUCCESS)
        else:
            self.aux.slot.data[:] = b"old-data"
            self.session.process_aux_agent_result(0, AgentResult.SUCCESS)


@pytest.mark.parametrize("cause", ["user", "timeout"])
@pytest.mark.parametrize("last_resource", ["kv", "aux"])
def test_cancelled_receive_retains_destinations_until_all_accessors_settle(
    monkeypatch: pytest.MonkeyPatch,
    cause: Literal["user", "timeout"],
    last_resource: Literal["kv", "aux"],
) -> None:
    case = _ReceiveCase(monkeypatch, bridge_enabled=True)
    case.cancel(cause)
    for _ in range(2):
        case.assert_not_reusable()
    case.complete("aux" if last_resource == "kv" else "kv")
    case.assert_not_reusable()
    case.complete(last_resource)
    case.assert_cancelled()
    case.poll_and_cleanup()
    assert case.cleanup_count == case.kv.release_count == case.aux.slot.release_count == 1
    assert case.active == []
    assert not case.transceiver._recv_sessions and not case.transceiver._recv_reqs
    case.receiver.clear_session.assert_called_once_with(_OLD_REQUEST)
    assert case.kv.allocate(_NEW_REQUEST)
    assert case.aux.slot.allocate(_NEW_REQUEST)
    # Duplicate control-plane completion reports carry no new memory access.
    case.session.process_kv_agent_result(0, 0, True, AgentResult.SUCCESS)
    case.session.process_aux_agent_result(0, AgentResult.SUCCESS)
    for _ in range(2):
        case.poll_and_cleanup()
        assert case.coordinator.request_cancellation(case.request)
        case.assert_cancelled()
    assert case.cleanup_count == case.kv.release_count == case.aux.slot.release_count == 1
    assert case.kv.data == case.aux.slot.data == _SENTINEL


@pytest.mark.parametrize("cause", ["user", "timeout"])
@pytest.mark.xfail(
    strict=True,
    raises=_PrematureReuse,
    reason="default Python receive retirement does not enforce physical ownership",
)
def test_default_receive_cancellation_retains_pending_destinations(
    monkeypatch: pytest.MonkeyPatch, cause: Literal["user", "timeout"]
) -> None:
    case = _ReceiveCase(monkeypatch, bridge_enabled=False)
    case.cancel(cause)
    case.assert_not_reusable()
