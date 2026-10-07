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

from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import Mock
from weakref import WeakSet

import pytest
import torch

from tensorrt_llm._torch.alltoall_watchdog import ActiveRankMaskSnapshot
from tensorrt_llm._torch.mnnvl_alltoall_workspace import _MnnvlAlltoAllWorkspaceLifecycle
from tensorrt_llm._torch.moe.fused_moe.communication.moe_alltoall import MoeAlltoAll
from tensorrt_llm._torch.moe.fused_moe.communication.nvlink_one_sided import NVLinkOneSided
from tensorrt_llm._torch.moe.fused_moe.ep_group_health import EPGroupHealth
from tensorrt_llm._torch.moe_a2a_execution_control import (
    MoeA2AExecutionAbortStatus,
    MoeA2AExecutionControl,
    validate_execution_mode,
)
from tensorrt_llm.mapping import Mapping


def _pack_status(
    *,
    execution_epoch: int,
    phase_code: int,
    reason_code: int,
    waiting_peer: int | None,
) -> int:
    peer_code = 0 if waiting_peer is None else waiting_peer + 1
    return (
        ((execution_epoch & ((1 << 38) - 1)) << 25)
        | ((peer_code & 0x1FF) << 16)
        | ((phase_code & 0xFF) << 8)
        | (reason_code & 0xFF)
    )


def test_execution_abort_status_zero_means_no_abort() -> None:
    assert MoeA2AExecutionAbortStatus.from_raw(0) is None


@pytest.mark.parametrize(
    "phase_code,reason_code,phase,reason,waiting_peer",
    [
        (1, 1, "dispatch", "host_requested", 2),
        (2, 2, "combine", "timeout", None),
    ],
)
def test_execution_abort_status_decodes_packed_fields(
    phase_code: int,
    reason_code: int,
    phase: str,
    reason: str,
    waiting_peer: int | None,
) -> None:
    raw_status = _pack_status(
        execution_epoch=1234,
        phase_code=phase_code,
        reason_code=reason_code,
        waiting_peer=waiting_peer,
    )

    status = MoeA2AExecutionAbortStatus.from_raw(raw_status)

    assert status == MoeA2AExecutionAbortStatus(
        execution_epoch=1234,
        phase=phase,
        reason=reason,
        waiting_peer=waiting_peer,
        raw_status=raw_status,
    )


def test_execution_abort_status_preserves_unknown_codes() -> None:
    raw_status = _pack_status(
        execution_epoch=(1 << 38) - 1,
        phase_code=17,
        reason_code=23,
        waiting_peer=255,
    )

    status = MoeA2AExecutionAbortStatus.from_raw(raw_status)

    assert status is not None
    assert status.execution_epoch == (1 << 38) - 1
    assert status.phase == "unknown(17)"
    assert status.reason == "unknown(23)"
    assert status.waiting_peer == 255


@pytest.mark.parametrize(
    "rank_mask_enabled,can_use_cft", [(False, False), (False, True), (True, False)]
)
def test_execution_mode_accepts_qualified_combinations(
    rank_mask_enabled: bool, can_use_cft: bool
) -> None:
    validate_execution_mode(rank_mask_enabled, can_use_cft)


def test_execution_mode_rejects_rank_mask_with_cft() -> None:
    with pytest.raises(RuntimeError, match="supports only non-CFT transport"):
        validate_execution_mode(rank_mask_enabled=True, can_use_cft=True)


@pytest.mark.parametrize("wrapper_type", [MoeAlltoAll, NVLinkOneSided])
def test_wrapper_rejects_resolved_cft_before_workspace_allocation(
    wrapper_type: type[MoeAlltoAll] | type[NVLinkOneSided],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(f"{wrapper_type.__module__}.resolve_can_use_cft", lambda _requested: True)
    monkeypatch.setattr(f"{wrapper_type.__module__}.get_force_cft", lambda: True)
    monkeypatch.setattr(f"{wrapper_type.__module__}.MnnvlMemory.initialize", lambda: None)
    mapping = Mapping(world_size=1, rank=0, tp_size=1, moe_ep_size=1)
    if wrapper_type is MoeAlltoAll:
        monkeypatch.setattr(MoeAlltoAll, "_init_constants", staticmethod(lambda: None))
        kwargs = {"max_num_tokens": 1, "workspace_size_per_rank": 4096}
    else:
        monkeypatch.setattr(NVLinkOneSided, "is_platform_supported", staticmethod(lambda: True))
        kwargs = {"max_num_tokens_per_rank": 1}

    with pytest.raises(RuntimeError, match="supports only non-CFT transport"):
        wrapper_type(mapping=mapping, num_slots=1, top_k=1, ep_group_health=object(), **kwargs)


@pytest.mark.parametrize("rank_mask_enabled", [False, True])
def test_moe_alltoall_allocates_shared_execution_control_only_for_ft(
    rank_mask_enabled: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = torch.zeros((1, 4096), dtype=torch.uint8)
    metainfo = torch.tensor([0, 4, 8], dtype=torch.int64)
    controls: list[object] = []

    class FakeMemory:
        mapped = True

        def __init__(self, mapping: Mapping, size: int) -> None:
            assert mapping.moe_ep_size == 1
            assert size == 4096

        @staticmethod
        def initialize() -> None:
            pass

        def as_torch_strided_tensor(self, dtype: torch.dtype) -> torch.Tensor:
            assert dtype == torch.uint8
            return workspace

    def create_control(workspace_arg: torch.Tensor, ep_rank: int) -> object:
        assert workspace_arg is workspace
        assert ep_rank == 0
        control = object()
        controls.append(control)
        return control

    monkeypatch.delenv("TRTLLM_MOE_A2A_WORKSPACE_MB", raising=False)
    monkeypatch.setattr(f"{MoeAlltoAll.__module__}.MnnvlMemory", FakeMemory)
    monkeypatch.setattr(f"{MoeAlltoAll.__module__}.resolve_can_use_cft", lambda _requested: False)
    monkeypatch.setattr(f"{MoeAlltoAll.__module__}.get_force_cft", lambda: False)
    monkeypatch.setattr(f"{MoeAlltoAll.__module__}.MoeA2AExecutionControl", create_control)
    monkeypatch.setattr(MoeAlltoAll, "_WORKSPACES", {})
    monkeypatch.setattr(MoeAlltoAll, "_init_constants", staticmethod(lambda: None))
    monkeypatch.setattr(
        MoeAlltoAll,
        "_METAINFO_INDEX",
        {
            "FLAG_VAL_OFFSET_INDEX": 0,
            "DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX": 1,
            "COMBINE_COMPLETION_FLAGS_OFFSET_INDEX": 2,
        },
    )
    monkeypatch.setattr(torch.ops.trtllm, "moe_a2a_initialize", lambda *_args: metainfo)
    mapping = Mapping(world_size=1, rank=0, tp_size=1, moe_ep_size=1)
    health = EPGroupHealth(1) if rank_mask_enabled else None
    wrappers = []
    try:
        for _ in range(2):
            wrappers.append(
                MoeAlltoAll(
                    mapping,
                    max_num_tokens=1,
                    top_k=1,
                    num_slots=1,
                    workspace_size_per_rank=4096,
                    ep_group_health=health,
                )
            )
        assert len(controls) == int(rank_mask_enabled)
        expected_control = controls[0] if rank_mask_enabled else None
        assert all(wrapper._execution_control is expected_control for wrapper in wrappers)
        assert MoeAlltoAll._WORKSPACES[False]["execution_control"] is expected_control
    finally:
        for wrapper in wrappers:
            wrapper.destroy()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_native_ops_reject_mixed_cuda_devices() -> None:
    workspace = torch.empty((1, 256), dtype=torch.uint8, device="cuda:0")
    metainfo = torch.empty(0, dtype=torch.int64)

    token_selected_experts = torch.empty((0, 1), dtype=torch.int32, device="cuda:1")
    dispatch_payload = torch.empty((0, 1), dtype=torch.float16, device="cuda:0")
    with pytest.raises(RuntimeError, match="same CUDA device as workspace"):
        torch.ops.trtllm.moe_a2a_dispatch(
            token_selected_experts,
            [dispatch_payload],
            workspace,
            metainfo,
            1,
            0,
            1,
            1,
            1,
        )

    combine_payload = torch.empty((1, 1, 1), dtype=torch.float16, device="cuda:1")
    with pytest.raises(RuntimeError, match="same CUDA device as workspace"):
        torch.ops.trtllm.moe_a2a_combine(
            combine_payload,
            0,
            workspace,
            metainfo,
            1,
            0,
            1,
            1,
            0,
            False,
        )


def test_execution_control_lifecycle_delegates_to_host_ops(monkeypatch: pytest.MonkeyPatch) -> None:
    control_tensor = torch.zeros(32, dtype=torch.uint64)
    workspace = torch.empty(0, dtype=torch.uint8)
    state = {"live_epoch": 0, "raw_status": 0}
    begin_calls: list[tuple[torch.Tensor, int, torch.Tensor, int]] = []
    release_calls: list[torch.Tensor] = []

    def create_control(workspace_arg: torch.Tensor, ep_rank_arg: int) -> torch.Tensor:
        assert workspace_arg is workspace
        assert ep_rank_arg == 3
        return control_tensor

    def get_state(control: torch.Tensor) -> tuple[int, int]:
        assert control is control_tensor
        return state["live_epoch"], state["raw_status"]

    def request_abort(control: torch.Tensor) -> int:
        assert control is control_tensor
        state["live_epoch"] += 1
        return state["live_epoch"]

    def begin_epoch(
        workspace_arg: torch.Tensor,
        ep_rank: int,
        control: torch.Tensor,
        execution_epoch: int,
    ) -> None:
        assert control is control_tensor
        begin_calls.append((workspace_arg, ep_rank, control, execution_epoch))
        state["raw_status"] = 0

    def release_control(control: torch.Tensor) -> None:
        assert control is control_tensor
        release_calls.append(control)

    replacements: dict[str, Callable[..., object]] = {
        "moe_a2a_create_execution_control": create_control,
        "moe_a2a_get_execution_abort_state": get_state,
        "moe_a2a_request_execution_abort": request_abort,
        "moe_a2a_begin_execution_epoch": begin_epoch,
        "moe_a2a_release_execution_control": release_control,
    }
    for name, replacement in replacements.items():
        monkeypatch.setattr(torch.ops.trtllm, name, replacement, raising=False)

    control = MoeA2AExecutionControl(workspace, ep_rank=3)
    assert control.tensor is control_tensor
    assert control.capture_epoch() == 0
    assert control.requested_epoch() == 0
    assert control.status() is None
    assert not control.has_pending_abort()
    with pytest.raises(ValueError, match="newly requested execution epoch"):
        control.begin_epoch()

    requested_epoch = control.request_abort()
    assert requested_epoch == 1
    assert control.requested_epoch() == 1
    # An abort request invalidates running work but does not admit a new epoch.
    assert control.capture_epoch() == 0
    assert control.status() is None
    assert control.has_pending_abort()

    raw_status = _pack_status(
        execution_epoch=0,
        phase_code=1,
        reason_code=1,
        waiting_peer=2,
    )
    state["raw_status"] = raw_status
    assert control.status() == MoeA2AExecutionAbortStatus(
        execution_epoch=0,
        phase="dispatch",
        reason="host_requested",
        waiting_peer=2,
        raw_status=raw_status,
    )

    with pytest.raises(ValueError, match="latest requested epoch"):
        control.begin_epoch(0)
    assert begin_calls == []

    assert control.begin_epoch() == 1
    assert len(begin_calls) == 1
    workspace_arg, ep_rank, control_arg, execution_epoch = begin_calls[0]
    assert workspace_arg is workspace
    assert ep_rank == 3
    assert control_arg is control_tensor
    assert execution_epoch == 1
    assert control.capture_epoch() == 1
    assert control.status() is None
    assert not control.has_pending_abort()

    # A second wrapper sharing this workspace-owned control can acknowledge the
    # same already-reset epoch without issuing a duplicate device reset.
    assert control.begin_epoch(1) == 1
    assert len(begin_calls) == 1

    # A native kernel timeout latches status without advancing the mapped host
    # epoch. It must not be mistaken for an already-reset idempotent call.
    state["raw_status"] = _pack_status(
        execution_epoch=1,
        phase_code=2,
        reason_code=2,
        waiting_peer=0,
    )
    assert control.has_pending_abort()
    with pytest.raises(ValueError, match=r"call request_abort\(\) first"):
        control.begin_epoch()
    assert len(begin_calls) == 1
    assert control.status() is not None

    requested_epoch = control.request_abort()
    assert requested_epoch == 2
    assert control.begin_epoch(requested_epoch) == requested_epoch
    assert len(begin_calls) == 2
    assert control.status() is None
    assert not control.has_pending_abort()

    control.close()
    control.close()
    assert len(release_calls) == 1
    assert release_calls[0] is control_tensor
    with pytest.raises(RuntimeError, match="has been released"):
        control.capture_epoch()
    with pytest.raises(RuntimeError, match="has been released"):
        control.has_pending_abort()


def test_nvlink_workspace_cache_closes_each_execution_control_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeExecutionControl:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    first_control = FakeExecutionControl()
    second_control = FakeExecutionControl()
    first_workspace = torch.zeros((1, 32), dtype=torch.uint8)
    first_state = {
        "workspace": first_workspace,
        "execution_control": first_control,
        "instances": WeakSet(),
    }
    second_state = {
        "workspace": torch.empty(0, dtype=torch.uint8),
        "execution_control": second_control,
    }
    lifecycle = _MnnvlAlltoAllWorkspaceLifecycle.get_or_create(
        workspace_state=first_state,
        memory=SimpleNamespace(mapped=True),
        workspace=first_workspace,
        metainfo=torch.tensor([0, 4, 8], dtype=torch.int64),
        metainfo_index={
            "FLAG_VAL_OFFSET_INDEX": 0,
            "DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX": 1,
            "COMBINE_COMPLETION_FLAGS_OFFSET_INDEX": 2,
        },
        ep_rank=0,
        ep_size=1,
        health=None,
    )
    wrapper = object.__new__(NVLinkOneSided)
    wrapper._workspace_lifecycle = lifecycle
    lifecycle.register(
        wrapper,
        watchdog_timeout_s=1.0,
        watchdog_poll_interval_s=0.1,
        watchdog_on_timeout=None,
    )
    watchdog = wrapper._alltoall_watchdog
    assert watchdog is not None
    first_state["instances"].add(wrapper)
    monkeypatch.setattr(
        NVLinkOneSided,
        "_WORKSPACES",
        {
            ("first",): first_state,
            ("first-alias",): first_state,
            ("second",): second_state,
        },
    )
    monkeypatch.setattr(
        NVLinkOneSided,
        "_WORKSPACE_REFCOUNTS",
        {("first",): 2, ("second",): 1},
    )
    monkeypatch.setattr(NVLinkOneSided, "_WORKSPACE", second_state)

    NVLinkOneSided._clear_workspace_cache()

    assert first_control.close_calls == 1
    assert second_control.close_calls == 1
    assert wrapper._alltoall_watchdog is None
    with pytest.raises(RuntimeError, match="cannot start a stopped"):
        watchdog.start()
    assert NVLinkOneSided._WORKSPACES == {}
    assert NVLinkOneSided._WORKSPACE_REFCOUNTS == {}
    assert NVLinkOneSided._WORKSPACE is None


@pytest.mark.parametrize("wrapper_type", [MoeAlltoAll, NVLinkOneSided])
def test_wrapper_rejects_recoverable_abort_outside_rank_mask_mode(
    wrapper_type: type[MoeAlltoAll] | type[NVLinkOneSided],
) -> None:
    wrapper = object.__new__(wrapper_type)
    wrapper._rank_mask_enabled = False
    wrapper._execution_control = None

    with pytest.raises(RuntimeError, match="requires WideEP FT rank-mask mode"):
        wrapper.request_execution_abort()
    with pytest.raises(RuntimeError, match="requires WideEP FT rank-mask mode"):
        wrapper.begin_execution_epoch()
    assert wrapper.get_execution_abort_status() is None


@pytest.mark.parametrize("rank_mask_enabled", [False, True])
def test_moe_alltoall_wires_one_epoch_and_resets_all_shared_wrappers(
    monkeypatch: pytest.MonkeyPatch,
    rank_mask_enabled: bool,
) -> None:
    control_tensor = torch.zeros(32, dtype=torch.uint64)
    committed_mask = torch.tensor([1, 0, 0, 0], dtype=torch.uint64) if rank_mask_enabled else None
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    class FakeExecutionControl:
        def __init__(self) -> None:
            self.tensor = control_tensor
            self.begin_calls: list[int | None] = []
            self.capture_calls = 0

        def capture_epoch(self) -> int:
            self.capture_calls += 1
            return 7

        def begin_epoch(self, execution_epoch: int | None = None) -> int:
            self.begin_calls.append(execution_epoch)
            return 8 if execution_epoch is None else execution_epoch

    class FakeWatchdogCoordinator:
        def capture_active_rank_mask(
            self, active_rank_mask: torch.Tensor | None
        ) -> ActiveRankMaskSnapshot:
            assert active_rank_mask is None
            return ActiveRankMaskSnapshot(committed_mask, None)

        def active_rank_mask_for_combine(
            self,
            snapshot: ActiveRankMaskSnapshot,
            active_rank_mask: torch.Tensor | None,
        ) -> torch.Tensor | None:
            assert active_rank_mask is None
            return snapshot.active_rank_mask

        def watch_collective(
            self,
            watchdog: object | None,
            phase: str,
            active_rank_mask: torch.Tensor | None,
        ) -> None:
            assert watchdog is None
            assert phase in ("dispatch", "combine")
            assert active_rank_mask is committed_mask

    fake_control = FakeExecutionControl()
    workspace = torch.empty((1, 256), dtype=torch.uint8)
    metainfo = torch.zeros(10, dtype=torch.int64)
    workspace_state = {"instances": WeakSet()}
    lifecycle = SimpleNamespace(
        metainfo=metainfo,
        coordinator=FakeWatchdogCoordinator(),
        watchdog_for=lambda _wrapper: None,
    )

    def make_wrapper() -> MoeAlltoAll:
        wrapper = object.__new__(MoeAlltoAll)
        wrapper.workspace = workspace
        wrapper.mnnvl_mem = SimpleNamespace(mapped=True)
        wrapper._workspace_state = workspace_state
        wrapper._workspace_lifecycle = lifecycle
        wrapper.max_num_tokens = 8
        wrapper.ep_rank = 0
        wrapper.ep_size = 1
        wrapper.top_k = 1
        wrapper.num_experts = 1
        wrapper.enable_eplb = False
        wrapper.eplb_stats_num_experts = None
        wrapper.can_use_cft_counted_writes = False
        wrapper._force_cft = None
        wrapper.cft_max_batch_for_dispatch = None
        wrapper.cft_max_batch_for_combine = None
        wrapper._rank_mask_enabled = rank_mask_enabled
        wrapper._execution_control = fake_control if rank_mask_enabled else None
        wrapper.reset_state()
        workspace_state["instances"].add(wrapper)
        return wrapper

    dispatch_calls: list[tuple[object, ...]] = []
    combine_calls: list[tuple[object, ...]] = []

    def dispatch_op(
        token_selected_experts: torch.Tensor,
        input_payloads: list[torch.Tensor],
        workspace_arg: torch.Tensor,
        metainfo_arg: torch.Tensor,
        runtime_max_tokens_per_rank: int,
        ep_rank: int,
        ep_size: int,
        top_k: int,
        num_experts: int,
        eplb_local_stats: torch.Tensor | None,
        use_cft_counted_writes: bool,
        expert_id_payload_index: int | None,
        invalid_token_expert_id: int | None,
        enable_rank_mask: bool,
        active_rank_mask: torch.Tensor | None,
        execution_control: torch.Tensor | None,
        expected_execution_epoch: int,
    ) -> tuple[list[torch.Tensor], int, torch.Tensor]:
        dispatch_calls.append(
            (
                token_selected_experts,
                input_payloads,
                workspace_arg,
                metainfo_arg,
                runtime_max_tokens_per_rank,
                ep_rank,
                ep_size,
                top_k,
                num_experts,
                eplb_local_stats,
                use_cft_counted_writes,
                expert_id_payload_index,
                invalid_token_expert_id,
                enable_rank_mask,
                active_rank_mask,
                execution_control,
                expected_execution_epoch,
            )
        )
        return input_payloads, 64, torch.empty(0, dtype=torch.int32)

    def combine_op(
        payload: torch.Tensor,
        local_num_tokens: int,
        workspace_arg: torch.Tensor,
        metainfo_arg: torch.Tensor,
        runtime_max_tokens_per_rank: int,
        ep_rank: int,
        ep_size: int,
        top_k: int,
        combine_payload_offset: int,
        payload_in_workspace: bool,
        use_low_precision: bool,
        use_cft_counted_writes: bool,
        enable_rank_mask: bool,
        active_rank_mask: torch.Tensor | None,
        execution_control: torch.Tensor | None,
        expected_execution_epoch: int,
    ) -> torch.Tensor:
        combine_calls.append(
            (
                payload,
                local_num_tokens,
                workspace_arg,
                metainfo_arg,
                runtime_max_tokens_per_rank,
                ep_rank,
                ep_size,
                top_k,
                combine_payload_offset,
                payload_in_workspace,
                use_low_precision,
                use_cft_counted_writes,
                enable_rank_mask,
                active_rank_mask,
                execution_control,
                expected_execution_epoch,
            )
        )
        return payload

    monkeypatch.setattr(torch.ops.trtllm, "moe_a2a_dispatch", dispatch_op, raising=False)
    monkeypatch.setattr(torch.ops.trtllm, "moe_a2a_combine", combine_op, raising=False)

    first = make_wrapper()
    second = make_wrapper()
    token_selected_experts = torch.zeros((2, 1), dtype=torch.int32)
    payload = torch.ones((2, 4), dtype=torch.float32)

    recv_payloads = first.dispatch(token_selected_experts, [payload], 2)
    assert len(recv_payloads) == 1
    assert recv_payloads[0] is payload
    expected_epoch = 7 if rank_mask_enabled else 0
    expected_control = control_tensor if rank_mask_enabled else None
    assert first._state.execution_epoch == expected_epoch
    first.combine(payload.view(1, 2, 4), 2)
    assert first._state.phase == "idle"
    assert dispatch_calls[0][-7:-4] == (False, None, None)
    assert dispatch_calls[0][-4] is rank_mask_enabled
    assert dispatch_calls[0][-3] is committed_mask
    assert dispatch_calls[0][-2] is expected_control
    assert dispatch_calls[0][-1] == expected_epoch
    assert combine_calls[0][-4] is rank_mask_enabled
    assert combine_calls[0][-5] is False
    assert combine_calls[0][-3] is committed_mask
    assert combine_calls[0][-2] is expected_control
    assert combine_calls[0][-1] == expected_epoch
    assert fake_control.capture_calls == int(rank_mask_enabled)

    first._state.phase = "dispatched"
    second._state.phase = "dispatched"
    if rank_mask_enabled:
        assert first.begin_execution_epoch(8) == 8
        assert fake_control.begin_calls == [8]
        assert first._state.phase == "idle"
        assert second._state.phase == "idle"
    else:
        with pytest.raises(RuntimeError, match="requires WideEP FT rank-mask mode"):
            first.begin_execution_epoch(8)
        assert fake_control.begin_calls == []
        assert first._state.phase == "dispatched"
        assert second._state.phase == "dispatched"


@pytest.mark.parametrize("wrapper_type", [MoeAlltoAll, NVLinkOneSided])
@pytest.mark.parametrize("phase", ["idle", "dispatched"])
@pytest.mark.parametrize("pending_abort", [None, False, True])
def test_checkpoint_readiness_requires_idle_frontend_without_pending_abort(
    wrapper_type: type[MoeAlltoAll] | type[NVLinkOneSided],
    phase: str,
    pending_abort: bool | None,
) -> None:
    wrapper = object.__new__(wrapper_type)
    wrapper._execution_control = (
        None if pending_abort is None else SimpleNamespace(has_pending_abort=lambda: pending_abort)
    )
    if wrapper_type is MoeAlltoAll:
        wrapper._state = SimpleNamespace(phase=phase)
    else:
        wrapper._dispatch_state = {"phase": phase}

    assert wrapper._mnnvl_checkpoint_is_idle() is (phase == "idle" and not pending_abort)


@pytest.mark.parametrize("wrapper_type", [MoeAlltoAll, NVLinkOneSided])
@pytest.mark.parametrize("abort_after_prepare", [False, True])
def test_checkpoint_restore_fails_closed_on_abort_after_detachment(
    wrapper_type: type[MoeAlltoAll] | type[NVLinkOneSided],
    abort_after_prepare: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = SimpleNamespace(has_pending_abort=Mock(return_value=False))
    wrapper = object.__new__(wrapper_type)
    wrapper._execution_control = control
    wrapper.reset_state()
    state_attribute = "_state" if wrapper_type is MoeAlltoAll else "_dispatch_state"
    initial_state = getattr(wrapper, state_attribute)
    comm = SimpleNamespace(allgather=Mock(side_effect=lambda ready: [ready]))
    memory = Mock(mapped=True, comm=comm)
    memory.checkpoint_prepare.side_effect = lambda: setattr(memory, "mapped", False)
    memory.checkpoint_restore.return_value = True
    metainfo = torch.tensor([0, 4, 8], dtype=torch.int64)
    lifecycle = _MnnvlAlltoAllWorkspaceLifecycle.get_or_create(
        workspace_state={},
        memory=memory,
        workspace=torch.zeros((1, 32), dtype=torch.uint8),
        metainfo=metainfo,
        metainfo_index={
            "FLAG_VAL_OFFSET_INDEX": 0,
            "DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX": 1,
            "COMBINE_COMPLETION_FLAGS_OFFSET_INDEX": 2,
        },
        ep_rank=0,
        ep_size=1,
        health=None,
    )
    lifecycle.register(
        wrapper,
        watchdog_timeout_s=None,
        watchdog_poll_interval_s=0.1,
        watchdog_on_timeout=None,
    )
    monkeypatch.setattr(torch.cuda, "synchronize", Mock())
    try:
        lifecycle.checkpoint_prepare()
        memory.checkpoint_prepare.assert_called_once_with()
        assert not memory.mapped
        control.has_pending_abort.return_value = abort_after_prepare

        if abort_after_prepare:
            with pytest.raises(RuntimeError, match="unacknowledged execution abort"):
                lifecycle.checkpoint_restore(comm, lambda: metainfo)
            memory._checkpoint_restore_failed.assert_called_once_with()
            memory._checkpoint_restore_complete.assert_not_called()
            assert getattr(wrapper, state_attribute) is initial_state
        else:
            lifecycle.checkpoint_restore(comm, lambda: metainfo)
            memory._checkpoint_restore_complete.assert_called_once_with()
            memory._checkpoint_restore_failed.assert_not_called()
            assert getattr(wrapper, state_attribute) is not initial_state
        assert comm.allgather.call_count == 2
        comm.allgather.assert_called_with(not abort_after_prepare)
    finally:
        lifecycle.unregister(wrapper)


def test_nvlink_begin_epoch_resets_all_shared_workspace_wrappers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeExecutionControl:
        def __init__(self) -> None:
            self.begin_calls: list[int | None] = []

        def begin_epoch(self, execution_epoch: int | None = None) -> int:
            self.begin_calls.append(execution_epoch)
            return 3 if execution_epoch is None else execution_epoch

    workspace_key = ("shared-test-workspace",)
    workspace_state = {"instances": WeakSet()}
    monkeypatch.setattr(NVLinkOneSided, "_WORKSPACES", {workspace_key: workspace_state})
    control = FakeExecutionControl()

    def make_wrapper() -> NVLinkOneSided:
        wrapper = object.__new__(NVLinkOneSided)
        wrapper._rank_mask_enabled = True
        wrapper._workspace_key = workspace_key
        wrapper._execution_control = control
        wrapper._dispatch_state = {"phase": "dispatched"}
        workspace_state["instances"].add(wrapper)
        return wrapper

    first = make_wrapper()
    second = make_wrapper()

    assert first.begin_execution_epoch(3) == 3
    assert control.begin_calls == [3]
    assert first._dispatch_state == {"phase": "idle"}
    assert second._dispatch_state == {"phase": "idle"}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "layout,message",
    [("expanded_bytes", "byte-contiguous"), ("overlapping_rows", "rank rows must not overlap")],
)
def test_native_control_rejects_unsafe_workspace_layout(layout: str, message: str) -> None:
    if layout == "expanded_bytes":
        workspace = torch.zeros((2, 1), dtype=torch.uint8, device="cuda").expand(2, 256)
    else:
        workspace = torch.zeros((1, 256), dtype=torch.uint8, device="cuda").expand(2, 256)
    with pytest.raises(RuntimeError, match=message):
        torch.ops.trtllm.moe_a2a_create_execution_control(workspace, 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_native_control_accepts_padded_workspace_rows() -> None:
    workspace = torch.zeros((2, 512), dtype=torch.uint8, device="cuda")[:, :256]
    assert not workspace.is_contiguous()
    control = torch.ops.trtllm.moe_a2a_create_execution_control(workspace, 1)
    try:
        execution_epoch = torch.ops.trtllm.moe_a2a_request_execution_abort(control)
        torch.ops.trtllm.moe_a2a_begin_execution_epoch(workspace, 1, control, execution_epoch)
        assert torch.ops.trtllm.moe_a2a_get_execution_abort_state(control) == (execution_epoch, 0)
    finally:
        torch.ops.trtllm.moe_a2a_release_execution_control(control)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_native_begin_rejects_released_execution_control() -> None:
    workspace = torch.zeros((1, 256), dtype=torch.uint8, device="cuda")
    control = torch.ops.trtllm.moe_a2a_create_execution_control(workspace, 0)
    execution_epoch = torch.ops.trtllm.moe_a2a_request_execution_abort(control)
    torch.ops.trtllm.moe_a2a_release_execution_control(control)

    with pytest.raises(RuntimeError, match="not registered or was already released"):
        torch.ops.trtllm.moe_a2a_begin_execution_epoch(
            workspace,
            0,
            control,
            execution_epoch,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_native_control_rejects_different_workspace_on_same_device() -> None:
    workspace = torch.zeros((2, 256), dtype=torch.uint8, device="cuda")
    other_workspace = torch.zeros_like(workspace)
    control = torch.ops.trtllm.moe_a2a_create_execution_control(workspace, 0)
    try:
        with pytest.raises(RuntimeError, match="already has a registered execution_control"):
            torch.ops.trtllm.moe_a2a_create_execution_control(workspace, 0)
        execution_epoch = torch.ops.trtllm.moe_a2a_request_execution_abort(control)
        with pytest.raises(RuntimeError, match="belongs to a different workspace"):
            torch.ops.trtllm.moe_a2a_begin_execution_epoch(
                other_workspace,
                0,
                control,
                execution_epoch,
            )
        with pytest.raises(RuntimeError, match="belongs to ep_rank 0"):
            torch.ops.trtllm.moe_a2a_begin_execution_epoch(
                workspace,
                1,
                control,
                execution_epoch,
            )
        torch.ops.trtllm.moe_a2a_begin_execution_epoch(
            workspace,
            0,
            control,
            execution_epoch,
        )
    finally:
        torch.ops.trtllm.moe_a2a_release_execution_control(control)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_native_no_mask_ops_keep_legacy_arity() -> None:
    workspace = torch.zeros((1, 4096), dtype=torch.uint8, device="cuda")
    metainfo = torch.ops.trtllm.moe_a2a_initialize(workspace, 0, 1, 1)
    token_selected_experts = torch.zeros((1, 1), dtype=torch.int32, device="cuda")
    payload = torch.arange(16, dtype=torch.float32, device="cuda").to(torch.bfloat16).view(1, 16)

    recv_tensors, combine_payload_offset, _ = torch.ops.trtllm.moe_a2a_dispatch(
        token_selected_experts,
        [payload],
        workspace,
        metainfo,
        1,
        0,
        1,
        1,
        1,
    )
    output = torch.ops.trtllm.moe_a2a_combine(
        recv_tensors[0],
        1,
        workspace,
        metainfo,
        1,
        0,
        1,
        1,
        combine_payload_offset,
        False,
    )
    torch.cuda.synchronize(workspace.device)

    assert recv_tensors[0].shape == (1, 1, 16)
    assert output.shape == payload.shape
    torch.testing.assert_close(output, payload)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_native_rank_mask_ops_reject_missing_execution_control() -> None:
    workspace = torch.zeros((1, 4096), dtype=torch.uint8, device="cuda")
    metainfo = torch.ops.trtllm.moe_a2a_initialize(workspace, 0, 1, 1)
    token_selected_experts = torch.zeros((1, 1), dtype=torch.int32, device="cuda")
    payload = torch.zeros((1, 16), dtype=torch.bfloat16, device="cuda")
    combine_payload = payload.view(1, 1, 16)
    active_rank_mask = torch.tensor([1, 0, 0, 0], dtype=torch.uint64)

    with pytest.raises(RuntimeError, match="execution_control is required"):
        torch.ops.trtllm.moe_a2a_dispatch(
            token_selected_experts,
            [payload],
            workspace,
            metainfo,
            1,
            0,
            1,
            1,
            1,
            enable_rank_mask=True,
            active_rank_mask=active_rank_mask,
        )
    with pytest.raises(RuntimeError, match="execution_control is required"):
        torch.ops.trtllm.moe_a2a_combine(
            combine_payload,
            1,
            workspace,
            metainfo,
            1,
            0,
            1,
            1,
            0,
            False,
            enable_rank_mask=True,
            active_rank_mask=active_rank_mask,
        )


@pytest.mark.parametrize("phase", ["dispatch", "combine"])
@pytest.mark.parametrize(
    "device",
    [
        "meta",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
        ),
    ],
)
def test_rank_mask_ops_reject_cft(phase: str, device: str) -> None:
    workspace = torch.empty((1, 4096), dtype=torch.uint8, device=device)
    metainfo = torch.empty((10,), dtype=torch.int64)
    payload = torch.empty((1, 16), dtype=torch.bfloat16, device=device)
    active_rank_mask = torch.tensor([1, 0, 0, 0], dtype=torch.uint64)
    execution_control = torch.zeros(32, dtype=torch.uint64)

    with pytest.raises(RuntimeError, match="supports only non-CFT transport"):
        if phase == "dispatch":
            torch.ops.trtllm.moe_a2a_dispatch(
                torch.empty((1, 1), dtype=torch.int32, device=device),
                [payload],
                workspace,
                metainfo,
                1,
                0,
                1,
                1,
                1,
                use_cft_counted_writes=True,
                enable_rank_mask=True,
                active_rank_mask=active_rank_mask,
                execution_control=execution_control,
            )
        else:
            torch.ops.trtllm.moe_a2a_combine(
                payload.view(1, 1, 16),
                1,
                workspace,
                metainfo,
                1,
                0,
                1,
                1,
                0,
                False,
                use_cft_counted_writes=True,
                enable_rank_mask=True,
                active_rank_mask=active_rank_mask,
                execution_control=execution_control,
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("execution_epoch,message", [(-1, "non-negative"), (1 << 38, "38-bit")])
@pytest.mark.parametrize("operation", ["dispatch", "combine", "begin"])
def test_native_ops_reject_unrepresentable_execution_epoch(
    execution_epoch: int, message: str, operation: str
) -> None:
    workspace = torch.zeros((1, 4096), dtype=torch.uint8, device="cuda")
    metainfo = torch.ops.trtllm.moe_a2a_initialize(workspace, 0, 1, 1)
    control = torch.ops.trtllm.moe_a2a_create_execution_control(workspace, 0)
    payload = torch.zeros((1, 16), dtype=torch.bfloat16, device="cuda")
    active_rank_mask = torch.tensor([1, 0, 0, 0], dtype=torch.uint64)
    try:
        with pytest.raises(RuntimeError, match=message):
            if operation == "begin":
                torch.ops.trtllm.moe_a2a_begin_execution_epoch(
                    workspace, 0, control, execution_epoch
                )
            elif operation == "dispatch":
                torch.ops.trtllm.moe_a2a_dispatch(
                    torch.zeros((1, 1), dtype=torch.int32, device="cuda"),
                    [payload],
                    workspace,
                    metainfo,
                    1,
                    0,
                    1,
                    1,
                    1,
                    enable_rank_mask=True,
                    active_rank_mask=active_rank_mask,
                    execution_control=control,
                    expected_execution_epoch=execution_epoch,
                )
            else:
                torch.ops.trtllm.moe_a2a_combine(
                    payload.view(1, 1, 16),
                    1,
                    workspace,
                    metainfo,
                    1,
                    0,
                    1,
                    1,
                    0,
                    False,
                    enable_rank_mask=True,
                    active_rank_mask=active_rank_mask,
                    execution_control=control,
                    expected_execution_epoch=execution_epoch,
                )
    finally:
        torch.ops.trtllm.moe_a2a_release_execution_control(control)


def test_fake_ops_match_rank_mask_execution_control_contract() -> None:
    workspace = torch.empty((1, 4096), dtype=torch.uint8, device="meta")
    metainfo = torch.empty((10,), dtype=torch.int64, device="meta")
    token_selected_experts = torch.empty((1, 1), dtype=torch.int32, device="meta")
    payload = torch.empty((1, 16), dtype=torch.bfloat16, device="meta")
    combine_payload = payload.view(1, 1, 16)
    active_rank_mask = torch.tensor([1, 0, 0, 0], dtype=torch.uint64)

    recv_tensors, combine_payload_offset, _ = torch.ops.trtllm.moe_a2a_dispatch(
        token_selected_experts,
        [payload],
        workspace,
        metainfo,
        1,
        0,
        1,
        1,
        1,
    )
    output = torch.ops.trtllm.moe_a2a_combine(
        combine_payload,
        1,
        workspace,
        metainfo,
        1,
        0,
        1,
        1,
        combine_payload_offset,
        False,
    )

    assert recv_tensors[0].shape == (1, 1, 16)
    assert output.shape == (1, 16)

    with pytest.raises(RuntimeError, match="execution_control is required"):
        torch.ops.trtllm.moe_a2a_dispatch(
            token_selected_experts,
            [payload],
            workspace,
            metainfo,
            1,
            0,
            1,
            1,
            1,
            enable_rank_mask=True,
            active_rank_mask=active_rank_mask,
        )
    with pytest.raises(RuntimeError, match="execution_control is required"):
        torch.ops.trtllm.moe_a2a_combine(
            combine_payload,
            1,
            workspace,
            metainfo,
            1,
            0,
            1,
            1,
            0,
            False,
            enable_rank_mask=True,
            active_rank_mask=active_rank_mask,
        )
