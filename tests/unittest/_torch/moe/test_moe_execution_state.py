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
"""MoeExecutionState, and the NVLinkOneSided layers bound to it at model load."""

import re
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from _torch.moe.moe_a2a_test_utils import a2a_combine, a2a_dispatch, a2a_workspace

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.distributed import Distributed
from tensorrt_llm._torch.distributed.startup_record import StartupRecord
from tensorrt_llm._torch.moe.fused_moe import ConfigurableMoE, TritonFusedMoE, VanillaMoE
from tensorrt_llm._torch.moe.fused_moe.communication import AllGatherReduceScatter
from tensorrt_llm._torch.moe.fused_moe.communication.execution_state import MoeExecutionState
from tensorrt_llm._torch.moe.fused_moe.communication.nvlink_one_sided import NVLinkOneSided
from tensorrt_llm._torch.moe.fused_moe.ep_group_health import EPGroupHealth
from tensorrt_llm._torch.pyexecutor import rank_failure_model_load
from tensorrt_llm._torch.pyexecutor.rank_failure_model_load import moe_comms, prepare_model
from tensorrt_llm._wide_ep_fault_tolerance import TURN_FAULT_TOLERANCE_OFF
from tensorrt_llm.bindings import internal
from tensorrt_llm.mapping import Mapping

_EP_SIZE, _TOKENS, _HIDDEN = 2, 8, 128
# Word of the abort state holding the round flag awaited from peer 0.
_WAITED_FLAGS = int(internal.thop.MOE_A2A_ABORT_WAITED_FLAG_INDEX)
_JOIN_TIMEOUT_S = 10.0


def _bindable_state(**attributes) -> SimpleNamespace:
    """Return a stand-in for an execution state that records the layers bound to it."""
    return SimpleNamespace(register_layer=Mock(), register_workspace=Mock(), **attributes)


def _collective_round(state: MoeExecutionState, workspace, meta, experts, payload) -> torch.Tensor:
    """Run dispatch and combine as rank 0 under ``state``, and return the combined output."""
    controls = {
        "enable_rank_mask": True,
        "active_rank_mask": state.active_rank_mask,
        "abort_state": state.abort,
    }
    received, offset, _ = a2a_dispatch(workspace, meta, experts, controls, payload)
    return a2a_combine(workspace, meta, received[0], offset, controls)


def _wait_for(condition, state: MoeExecutionState, failure: str, timeout_s: float = 10) -> None:
    """Wait until ``condition()`` holds. At the deadline, abort the collectives and fail."""
    deadline = time.monotonic() + timeout_s
    while not condition():
        if time.monotonic() > deadline:
            state.request_abort()
            torch.cuda.synchronize()
            pytest.fail(failure)
        time.sleep(0.01)


def _queued_work_done():
    done = torch.cuda.Event()
    done.record()
    return done.query


def test_mask_holds_the_active_ranks() -> None:
    """Rank 70 of 72 is bit 6 of the second mask word."""
    health = EPGroupHealth(72)
    health.mark_failed(70)
    state = MoeExecutionState(health, 0, torch.device("cuda"))
    assert state.active_rank_mask.is_cuda
    assert state.active_rank_mask.cpu().tolist() == [(1 << 64) - 1, 0xBF, 0, 0]


def test_install_mask_rewrites_the_mask_in_place() -> None:
    """The device mask changes only with the generation of the EP group health, at one address."""
    health = EPGroupHealth(4)
    state = MoeExecutionState(health, 0, torch.device("cuda"))
    address = state.active_rank_mask.data_ptr()

    def install() -> list[int]:
        state.install_mask()
        return state.active_rank_mask.cpu().tolist()

    assert install() == [0b1111, 0, 0, 0]
    health.mark_failed(3)
    assert install() == [0b0111, 0, 0, 0]
    # The kernels treat this rank as active, so a mask without it is refused.
    health.mark_failed(0)
    with pytest.raises(RuntimeError, match="must mark the local ep_rank"):
        install()
    assert state.active_rank_mask.data_ptr() == address
    assert state.active_rank_mask.cpu().tolist() == [0b0111, 0, 0, 0]


def test_state_requires_active_local_rank() -> None:
    excluded = EPGroupHealth(2)
    excluded.mark_failed(0)
    with pytest.raises(RuntimeError, match="must mark the local ep_rank"):
        MoeExecutionState(excluded, 0, torch.device("cuda"))


def test_abort_and_rearm_after_peer_failure() -> None:
    """A dispatch stuck on a failed peer aborts from another thread, and runs again after rearm."""
    workspace, meta, experts, payload = a2a_workspace(_EP_SIZE)
    health = EPGroupHealth(_EP_SIZE)
    state = MoeExecutionState(health, 0, workspace.device)
    resets = []

    def reset_workspace() -> None:
        resets.append(1)
        torch.ops.trtllm.moe_a2a_initialize(workspace, meta, 0, _EP_SIZE, barrier=False)

    state.register_workspace(object(), reset_workspace, Mock(), workspace)

    _collective_round(state, workspace, meta, experts, payload)
    done = _queued_work_done()
    # A long wait names the flag value it expects from rank 1. It is not an abort.
    _wait_for(state.waited_flags, state, "the dispatch published no waited flag")
    assert state.waited_flags() == {1: 1}
    assert not state.aborted()
    requester = threading.Thread(target=state.request_abort)
    requester.start()
    requester.join(_JOIN_TIMEOUT_S)
    assert not requester.is_alive(), "abort request queued behind the spinning kernel"
    _wait_for(done, state, "collective kept waiting after the abort request")
    assert state.aborted()
    assert state.waited_flags() == {}

    health.mark_failed(1)
    with pytest.raises(RuntimeError, match="abort state"):
        state.install_mask()
    assert state.active_rank_mask.cpu().tolist() == [3, 0, 0, 0]
    state.rearm()
    assert resets == [1]
    assert not state.aborted()
    state.install_mask()
    assert state.active_rank_mask.cpu().tolist() == [1, 0, 0, 0]
    for _ in range(2):
        output = _collective_round(state, workspace, meta, experts, payload)
        _wait_for(_queued_work_done(), state, "the round without rank 1 waited for it")
        torch.testing.assert_close(output, payload, rtol=0, atol=0)
    assert not state.aborted()


def test_waited_flags_map_peer_to_flag() -> None:
    """waited_flags() maps each peer a kernel waits for to the round flag value it waits for."""
    state = MoeExecutionState(EPGroupHealth(4), 0, torch.device("cuda"))
    assert state.waited_flags() == {}
    state.abort[_WAITED_FLAGS + 1] = 7  # the dispatch of round 3
    state.abort[_WAITED_FLAGS + 3] = 8  # the combine of round 3
    state.abort[_WAITED_FLAGS + 2] = -(1 << 31)  # uint32 flag 1 << 31, past the int32 range
    assert state.waited_flags() == {1: 7, 2: 1 << 31, 3: 8}


def test_incomplete_round_word_marks_the_state_aborted() -> None:
    """check_incomplete_rounds checks each bound workspace once, and a set word cancels the state."""
    state = MoeExecutionState(EPGroupHealth(_EP_SIZE), 0, torch.device("cuda"))
    checks = []
    bound = []
    for name in ("first", "second"):
        workspace, meta, _, _ = a2a_workspace(_EP_SIZE)
        incomplete_round = int(meta[int(internal.thop.MOE_A2A_INCOMPLETE_ROUND_OFFSET_INDEX)])
        bound.append(workspace[0, incomplete_round : incomplete_round + 4].view(torch.int32))

        def check(name=name, workspace=workspace, meta=meta) -> None:
            checks.append(name)
            torch.ops.trtllm.moe_a2a_check_incomplete_round(workspace, meta, 0, state.abort)

        key = object()
        # Layers that share a workspace register it again.
        for _ in range(2):
            state.register_workspace(key, Mock(), check, workspace)

    state.check_incomplete_rounds()
    torch.cuda.synchronize()
    assert checks == ["first", "second"]
    assert not state.aborted()

    bound[1].fill_(1)
    state.check_incomplete_rounds()
    torch.cuda.synchronize()
    assert state.aborted()
    assert state.abort[:2].tolist() == [0, 1]


def test_rearm_resets_bound_layers_and_shared_workspaces() -> None:
    """Rearm resets each shared workspace once and every layer an abort left between dispatch and combine."""
    state = MoeExecutionState(EPGroupHealth(2), 0, torch.device("cuda"))
    resets = []
    shared = object()
    for name in ("shared", "shared again"):
        state.register_workspace(
            shared,
            lambda name=name: resets.append(name),
            Mock(),
            torch.empty(2, 1, dtype=torch.uint8),
        )
    comm = _frontend()
    comm._reset_local_workspace = lambda: resets.append("layer workspace")
    comm.bind_execution_state(state)
    comm._dispatch_state = {"phase": "dispatched"}
    state.rearm()
    assert resets == ["shared", "layer workspace"]
    assert comm._dispatch_state == {"phase": "idle"}


def test_graph_replay_follows_installed_mask() -> None:
    """A CUDA graph captured with every rank active replays without the rank masked out later."""
    workspace, meta, experts, payload = a2a_workspace(_EP_SIZE)
    health = EPGroupHealth(_EP_SIZE)
    state = MoeExecutionState(health, 0, workspace.device)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = _collective_round(state, workspace, meta, experts, payload)

    health.mark_failed(1)
    state.install_mask()
    for _ in range(2):
        graph.replay()
        _wait_for(_queued_work_done(), state, "the replay waited for the masked-out rank")
        torch.testing.assert_close(output, payload, rtol=0, atol=0)
    assert not state.aborted()


def _workspace(device_bytes: int, fresh_rows: frozenset[int]) -> Mock:
    """Return a mock membership workspace."""
    return Mock(
        local_device_bytes=device_bytes, **{"rows_needing_fresh_pages.return_value": fresh_rows}
    )


def test_bound_layer_remaps_its_memory() -> None:
    """Binding a layer registers its MNNVL memory as the workspace that a change remaps."""
    state = MoeExecutionState(EPGroupHealth(2), 0, torch.device("cuda"))
    comm = _frontend()
    comm.mnnvl_mem = _workspace(6 << 20, frozenset({1}))
    comm.bind_execution_state(state)
    store_group = Mock()

    assert state.workspace_bytes() == 6 << 20
    assert state.workspace_rows_needing_fresh_pages(store_group) == frozenset({1})
    state.remap_workspaces(store_group, frozenset({1}))
    comm.mnnvl_mem.remap.assert_called_once_with(store_group, frozenset({1}))


def test_shared_workspace_counted_once() -> None:
    """Layers that share a workspace remap, count, check and reset it once. Only rearm resets them."""
    state = MoeExecutionState(EPGroupHealth(4), 0, torch.device("cuda"))
    shared = _workspace(6 << 20, frozenset())
    other = _workspace(2 << 20, frozenset({0, 3}))
    resets = []
    checks = []
    rows = {
        shared: torch.empty(4, 6, dtype=torch.uint8),
        other: torch.empty(4, 2, dtype=torch.uint8),
    }
    for workspace, name in ((shared, "shared"), (shared, "shared again"), (other, "other")):
        state.register_workspace(
            workspace,
            lambda name=name: resets.append(name),
            lambda name=name: checks.append(name),
            rows[workspace],
        )
    comm = Mock()

    state.check_incomplete_rounds()
    assert checks == ["shared", "other"]

    # Restores stage expert weights through the largest workspace.
    assert state.staging_workspace() is rows[shared]
    assert state.workspace_bytes() == 8 << 20
    assert state.workspace_rows_needing_fresh_pages(comm) == frozenset({0, 3})
    state.remap_workspaces(comm, frozenset({1}))
    shared.remap.assert_called_once_with(comm, frozenset({1}))
    other.remap.assert_called_once_with(comm, frozenset({1}))
    comm.allgather.assert_not_called()
    assert resets == []
    state.rearm()
    assert resets == ["shared", "other"]


def _frontend(*, fault_tolerance: bool = True, ep_rank: int = 0) -> NVLinkOneSided:
    """Build an NVLinkOneSided with only the state that dispatch and combine read."""
    comm = NVLinkOneSided.__new__(NVLinkOneSided)
    comm._rank_mask_enabled = fault_tolerance
    comm.execution_state = None
    comm._dispatch_state = {"phase": "idle"}
    comm._workspace_lifecycle = Mock()
    comm.mnnvl_mem = Mock(mapped=True)
    comm._sentinel_row = None
    comm.enable_eplb = False
    comm.workspace = torch.zeros(1, dtype=torch.uint8)
    comm.ep_rank, comm.ep_size, comm.top_k, comm.num_experts = ep_rank, _EP_SIZE, 1, _EP_SIZE
    comm.invalid_token_expert_id = -1
    comm.payload_in_workspace = False
    comm.use_low_precision_combine = False
    return comm


@pytest.mark.cpu_only
def test_bind_refuses_unsupported_layers() -> None:
    state = _bindable_state(ep_rank=0)
    comm = _frontend()
    comm.bind_execution_state(state)
    with pytest.raises(RuntimeError, match="already bound"):
        comm.bind_execution_state(state)
    with pytest.raises(ValueError, match="built with enable_wide_ep_fault_tolerance"):
        _frontend(fault_tolerance=False).bind_execution_state(state)
    with pytest.raises(ValueError, match="cannot run EP rank 1"):
        _frontend(ep_rank=1).bind_execution_state(state)


def test_bind_refused_after_membership_change() -> None:
    """The remaps of a membership change leave out the workspace of a layer bound afterwards."""
    health = EPGroupHealth(2)
    state = MoeExecutionState(health, 0, torch.device("cuda"))
    _frontend().bind_execution_state(state)
    health.mark_failed(1)
    with pytest.raises(RuntimeError, match="only before the EP membership changes"):
        _frontend().bind_execution_state(state)


@pytest.mark.cpu_only
def test_bound_layer_passes_the_device_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pass the bound state's device mask and abort state to the ops, also during CUDA graph capture."""
    calls = []
    payload = torch.zeros(_EP_SIZE, _TOKENS, _HIDDEN)
    received = [payload, torch.zeros(_EP_SIZE, _TOKENS, 1, dtype=torch.int32)]

    def dispatch_op(*args, **kwargs):
        calls.append(kwargs)
        return received, 0, torch.empty(0)

    def combine_op(*args, **kwargs):
        calls.append(kwargs)
        return torch.zeros(_TOKENS, _HIDDEN)

    monkeypatch.setattr(torch.ops.trtllm, "moe_a2a_dispatch", dispatch_op, raising=False)
    monkeypatch.setattr(torch.ops.trtllm, "moe_a2a_combine", combine_op, raising=False)
    monkeypatch.setattr(torch.ops.trtllm, "moe_a2a_sanitize_expert_ids", Mock(), raising=False)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    state = _bindable_state(
        ep_rank=0,
        active_rank_mask=torch.zeros(4, dtype=torch.uint64),
        abort=torch.zeros(4, dtype=torch.int32),
    )
    hidden, experts = torch.zeros(_TOKENS, _HIDDEN), torch.zeros(_TOKENS, 1, dtype=torch.int32)
    inputs = (hidden, None, experts, None, [_TOKENS] * _EP_SIZE)
    without_mask = _frontend(fault_tolerance=False)
    bound = _frontend()
    with pytest.raises(RuntimeError, match="dispatches only once bound"):
        bound.dispatch(*inputs)
    bound.bind_execution_state(state)
    for comm in (without_mask, bound):
        comm.dispatch(*inputs)
        comm.combine(payload)

    # Without fault tolerance the ops get no rank mask and no abort state.
    fence = {"enable_rank_mask": False, "active_rank_mask": None, "use_sentinel": False}
    assert calls[:2] == [
        {"expert_id_payload_index": None, "invalid_token_expert_id": None, **fence},
        {"use_low_precision": False, **fence},
    ]
    for kwargs in calls[2:]:
        assert kwargs["enable_rank_mask"] is True and kwargs["use_sentinel"] is False
        assert kwargs["active_rank_mask"] is state.active_rank_mask
        assert kwargs["abort_state"] is state.abort
    state.register_layer.assert_called_once_with(bound.reset_state)
    state.register_workspace.assert_called_once_with(
        bound.mnnvl_mem,
        bound._reset_local_workspace,
        bound._check_incomplete_round,
        bound.workspace,
    )


def _moe_layer(cls: type, **attributes) -> torch.nn.Module:
    """Build a MoE layer of class ``cls`` without its ``__init__`` and set ``attributes`` on it.

    Like a layer built without EPLB, it has no load balancer.
    """
    layer = cls.__new__(cls)
    torch.nn.Module.__init__(layer)
    layer.layer_load_balancer = None
    for name, value in attributes.items():
        setattr(layer, name, value)
    return layer


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "kind, uses",
    [
        ("all_gather", "(ConfigurableMoE) uses AllGatherReduceScatter"),
        ("no_communication", "(ConfigurableMoE) uses no communication strategy"),
        ("triton", "(TritonFusedMoE) uses no communication strategy"),
        ("vanilla", "(VanillaMoE) uses no communication strategy"),
    ],
)
def test_unbindable_moe_layer_is_refused(kind: str, uses: str) -> None:
    """Every MoE layer must bind, whatever its class."""
    layer = {
        "all_gather": lambda: _moe_layer(ConfigurableMoE, comm=AllGatherReduceScatter(Mapping())),
        "no_communication": lambda: _moe_layer(ConfigurableMoE, comm=None),
        "triton": lambda: _moe_layer(TritonFusedMoE),
        "vanilla": lambda: _moe_layer(VanillaMoE),
    }[kind]()
    bound = _moe_layer(ConfigurableMoE, comm=_frontend())
    with pytest.raises(ValueError, match=re.escape(f"MoE layer 1 {uses}")) as error:
        moe_comms(torch.nn.ModuleList([bound, layer]))
    assert "enable_wide_ep_fault_tolerance=False" in str(error.value)


@pytest.mark.cpu_only
def test_model_without_moe_layers_refused() -> None:
    with pytest.raises(ValueError, match="requires a model with MoE layers"):
        moe_comms(torch.nn.Linear(1, 1))


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "architectures",
    [["LlamaForCausalLM"], ["DeepseekV3ForCausalLM", "LlamaForCausalLM"], [], None],
    ids=["other", "mixed", "empty", "none"],
)
def test_unsupported_architecture_refused(architectures) -> None:
    model = torch.nn.Linear(1, 1)
    model.model_config = SimpleNamespace(
        pretrained_config=SimpleNamespace(architectures=architectures)
    )
    with pytest.raises(ValueError, match=re.escape(f"not {architectures}")) as error:
        prepare_model(model)
    assert TURN_FAULT_TOLERANCE_OFF in str(error.value)


@pytest.mark.cpu_only
@pytest.mark.parametrize("replacement", [False, True], ids=["original", "replacement"])
def test_layers_bound_to_one_execution_state(
    monkeypatch: pytest.MonkeyPatch, replacement: bool
) -> None:
    """prepare_model binds every MoE layer to one execution state; a replacement starts alone."""
    monkeypatch.setattr(
        rank_failure_model_load,
        "MoeExecutionState",
        lambda health, ep_rank, device: SimpleNamespace(ep_group_health=health, ep_rank=ep_rank),
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(Distributed, "startup_record", StartupRecord())
    monkeypatch.setattr(StartupRecord, "replacement", replacement)
    comms = [Mock(spec=["bind_execution_state"]) for _ in range(2)]
    model = torch.nn.ModuleList([_moe_layer(ConfigurableMoE, comm=comm) for comm in comms])
    model.model_config = SimpleNamespace(
        pretrained_config=SimpleNamespace(architectures=["DeepseekV3ForCausalLM"]),
        mapping=SimpleNamespace(moe_ep_size=4, moe_ep_rank=1),
    )
    prepare_model(model)
    (state,), (other,) = (comm.bind_execution_state.call_args.args for comm in comms)
    assert state is other
    assert state.ep_rank == 1
    assert state.ep_group_health.get_mask() == (0b0010 if replacement else 0b1111)
