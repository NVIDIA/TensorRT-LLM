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
from weakref import WeakSet

import pytest
import torch

from tensorrt_llm._torch.alltoall_watchdog import ActiveRankMaskSnapshot
from tensorrt_llm._torch.distributed.moe_alltoall import MoeAlltoAll
from tensorrt_llm._torch.modules.fused_moe.communication.nvlink_one_sided import NVLinkOneSided
from tensorrt_llm._torch.moe_a2a_execution_control import (
    MoeA2AExecutionAbortStatus,
    MoeA2AExecutionControl,
)


def _pack_status(
    *,
    execution_epoch: int,
    phase_code: int,
    reason_code: int,
    waiting_peer: int | None,
) -> int:
    peer_code = 0 if waiting_peer is None else waiting_peer + 1
    return (
        ((execution_epoch & ((1 << 39) - 1)) << 24)
        | ((peer_code & 0xFF) << 16)
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
        execution_epoch=(1 << 39) - 1,
        phase_code=17,
        reason_code=23,
        waiting_peer=127,
    )

    status = MoeA2AExecutionAbortStatus.from_raw(raw_status)

    assert status is not None
    assert status.execution_epoch == (1 << 39) - 1
    assert status.phase == "unknown(17)"
    assert status.reason == "unknown(23)"
    assert status.waiting_peer == 127


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
    with pytest.raises(ValueError, match="newly requested execution epoch"):
        control.begin_epoch()

    requested_epoch = control.request_abort()
    assert requested_epoch == 1
    assert control.requested_epoch() == 1
    # An abort request invalidates running work but does not admit a new epoch.
    assert control.capture_epoch() == 0

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
    with pytest.raises(ValueError, match=r"call request_abort\(\) first"):
        control.begin_epoch()
    assert len(begin_calls) == 1
    assert control.status() is not None

    requested_epoch = control.request_abort()
    assert requested_epoch == 2
    assert control.begin_epoch(requested_epoch) == requested_epoch
    assert len(begin_calls) == 2
    assert control.status() is None

    control.close()
    control.close()
    assert len(release_calls) == 1
    assert release_calls[0] is control_tensor
    with pytest.raises(RuntimeError, match="has been released"):
        control.capture_epoch()


def test_moe_alltoall_wires_one_epoch_and_resets_all_shared_wrappers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control_tensor = torch.zeros(32, dtype=torch.uint64)

    class FakeExecutionControl:
        def __init__(self) -> None:
            self.tensor = control_tensor
            self.begin_calls: list[int | None] = []

        def capture_epoch(self) -> int:
            return 7

        def begin_epoch(self, execution_epoch: int | None = None) -> int:
            self.begin_calls.append(execution_epoch)
            return 8 if execution_epoch is None else execution_epoch

    class FakeWatchdogCoordinator:
        def capture_active_rank_mask(
            self, active_rank_mask: torch.Tensor | None
        ) -> ActiveRankMaskSnapshot:
            return ActiveRankMaskSnapshot(active_rank_mask, None)

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
            assert active_rank_mask is None

    fake_control = FakeExecutionControl()
    workspace = torch.empty((1, 256), dtype=torch.uint8)
    metainfo = torch.zeros(10, dtype=torch.int64)
    workspace_state = {"instances": WeakSet()}
    monkeypatch.setattr(MoeAlltoAll, "_WORKSPACE", workspace_state)

    def make_wrapper() -> MoeAlltoAll:
        wrapper = object.__new__(MoeAlltoAll)
        wrapper.workspace = workspace
        wrapper.metainfo = metainfo
        wrapper.max_num_tokens = 8
        wrapper.ep_rank = 0
        wrapper.ep_size = 1
        wrapper.top_k = 1
        wrapper.num_experts = 1
        wrapper.enable_eplb = False
        wrapper.eplb_stats_num_experts = None
        wrapper._execution_control = fake_control
        wrapper._watchdog_coordinator = FakeWatchdogCoordinator()
        wrapper._alltoall_watchdog = None
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
        active_rank_mask: torch.Tensor | None,
        execution_control: torch.Tensor,
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
        active_rank_mask: torch.Tensor | None,
        execution_control: torch.Tensor,
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
    assert first._state.execution_epoch == 7
    first.combine(payload.view(1, 2, 4), 2)
    assert first._state.phase == "idle"
    assert dispatch_calls[0][-3] is None
    assert dispatch_calls[0][-2] is control_tensor
    assert dispatch_calls[0][-1] == 7
    assert combine_calls[0][-3] is None
    assert combine_calls[0][-2] is control_tensor
    assert combine_calls[0][-1] == 7

    first._state.phase = "dispatched"
    second._state.phase = "dispatched"
    assert first.begin_execution_epoch(8) == 8
    assert fake_control.begin_calls == [8]
    assert first._state.phase == "idle"
    assert second._state.phase == "idle"


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
def test_native_dispatch_rejects_missing_execution_control() -> None:
    workspace = torch.zeros((1, 4096), dtype=torch.uint8, device="cuda")
    metainfo = torch.ops.trtllm.moe_a2a_initialize(workspace, 0, 1, 1)
    token_selected_experts = torch.zeros((1, 1), dtype=torch.int32, device="cuda")
    payload = torch.zeros((1, 16), dtype=torch.bfloat16, device="cuda")

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
        )
