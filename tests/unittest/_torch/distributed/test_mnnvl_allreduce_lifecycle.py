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

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from tensorrt_llm._torch.distributed import ops as ops_module
from tensorrt_llm._torch.distributed.ops import (
    MNNVLAllReduce,
    get_or_scale_allreduce_mnnvl_workspace,
)
from tensorrt_llm.mapping import Mapping


@pytest.fixture(autouse=True)
def mpi_mode(monkeypatch):
    """Pin the checkpoint hooks to their MPI branch.

    These tests assert MPI ownership semantics -- the workspace duplicates the communicator it is
    handed and frees its own copy. The hooks pick that branch on mpi_disabled(), which reads the
    environment, so pin it rather than inherit whatever the runner happens to export.
    """
    monkeypatch.setattr(ops_module, "mpi_disabled", lambda: False)


class _FakeComm:
    def __init__(self) -> None:
        self.allreduce_count = 0
        self.dup_count = 0
        self.free_count = 0
        self.duplicates = []

    def py2f(self) -> int:
        return 42

    def allreduce(self, value: int) -> int:
        self.allreduce_count += 1
        return value

    def Get_size(self) -> int:
        return 1

    def Dup(self):
        self.dup_count += 1
        duplicate = _FakeComm()
        self.duplicates.append(duplicate)
        return duplicate

    def Free(self) -> None:
        self.free_count += 1


class _FailingDupComm(_FakeComm):
    def Dup(self):
        raise RuntimeError("injected communicator duplication failure")


class _FakeMcastBuffer:
    is_available = True
    reason = ""

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.mapped = True
        self.restore_pending = False
        self.prepare_count = 0
        self.restore_count = 0
        self.complete_count = 0

    def get_workspace_size(self) -> int:
        return 3 * 1024 * 8192 * torch.float32.itemsize

    def get_local_buffer(self) -> torch.Tensor:
        return torch.zeros(1, dtype=torch.float32)

    def get_uc_buffer(self, *args: object, **kwargs: object) -> torch.Tensor:
        return torch.zeros(1, dtype=torch.float32)

    def is_mapped(self) -> bool:
        return self.mapped

    def checkpoint_prepare(self) -> None:
        if self.mapped:
            self.prepare_count += 1
            self.mapped = False

    def checkpoint_restore(self, mpi_comm_fortran_handle: int) -> bool:
        assert mpi_comm_fortran_handle == 42
        if not self.mapped:
            self.restore_count += 1
            self.restore_pending = True
            return True
        return False

    def checkpoint_restore_complete(self, local_protocol_reset_succeeded: bool) -> None:
        assert self.restore_pending
        self.complete_count += 1
        self.restore_pending = False
        if not local_protocol_reset_succeeded:
            raise RuntimeError("protocol reset failed on one or more ranks")
        self.mapped = True


class _FakeWorld:
    def __init__(self) -> None:
        self.comm = _FakeComm()
        self.split_count = 0

    def Split(self, color: int, key: int) -> _FakeComm:
        self.split_count += 1
        return self.comm


def _patch_workspace_dependencies(monkeypatch: pytest.MonkeyPatch) -> _FakeWorld:
    world = _FakeWorld()
    cpu_device = torch.device("cpu")
    monkeypatch.setattr(MNNVLAllReduce, "allreduce_mnnvl_workspaces", {})
    monkeypatch.setattr(MNNVLAllReduce, "allreduce_mnnvl_pending_comms", {})
    monkeypatch.setattr(MNNVLAllReduce, "_allreduce_mnnvl_workspace_locks", {})
    monkeypatch.setattr(ops_module, "mpi_comm", lambda: world)
    monkeypatch.setattr(ops_module.torch, "device", lambda *args, **kwargs: cpu_device)
    monkeypatch.setattr(ops_module.torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(ops_module, "_initialize_allreduce_mnnvl_protocol", lambda workspace: None)
    return world


def test_workspace_retry_reuses_pending_communicator(monkeypatch: pytest.MonkeyPatch) -> None:
    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    world = _patch_workspace_dependencies(monkeypatch)
    construction_count = 0

    def fail_once(*args: object, **kwargs: object) -> _FakeMcastBuffer:
        nonlocal construction_count
        construction_count += 1
        if construction_count == 1:
            raise RuntimeError("injected workspace construction failure")
        return _FakeMcastBuffer()

    monkeypatch.setattr(ops_module, "MnnvlWorkspace", fail_once)

    with pytest.raises(RuntimeError, match="workspace construction failed"):
        get_or_scale_allreduce_mnnvl_workspace(mapping, torch.float32)

    assert world.split_count == 1
    assert MNNVLAllReduce.allreduce_mnnvl_pending_comms[mapping] is world.comm

    get_or_scale_allreduce_mnnvl_workspace(mapping, torch.float32)

    assert construction_count == 2
    assert world.split_count == 1
    assert MNNVLAllReduce.allreduce_mnnvl_pending_comms == {}


def test_concurrent_workspace_construction_is_serialized(monkeypatch: pytest.MonkeyPatch) -> None:
    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    world = _patch_workspace_dependencies(monkeypatch)
    construction_barrier = threading.Barrier(2)
    construction_count = 0
    construction_count_lock = threading.Lock()

    def construct(*args: object, **kwargs: object) -> _FakeMcastBuffer:
        nonlocal construction_count
        with construction_count_lock:
            construction_count += 1
        try:
            construction_barrier.wait(timeout=0.2)
        except threading.BrokenBarrierError:
            pass
        return _FakeMcastBuffer()

    monkeypatch.setattr(ops_module, "MnnvlWorkspace", construct)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(get_or_scale_allreduce_mnnvl_workspace, mapping, torch.float32)
            for _ in range(2)
        ]
        workspaces = [future.result() for future in futures]

    assert construction_count == 1
    assert world.split_count == 1
    assert workspaces[0] is workspaces[1]


def test_checkpoint_restore_resets_inference_protocol_state(monkeypatch) -> None:
    mapping = object()
    stale_comm = _FakeComm()
    comm = _FakeComm()
    redundant_comm = _FakeComm()
    handle = _FakeMcastBuffer()
    with torch.inference_mode():
        uc_buffer = torch.ones(8, dtype=torch.float32)
        buffer_flags = torch.ones(9, dtype=torch.uint32)
    workspace = {
        "handle": handle,
        "uc_buffer": uc_buffer,
        "buffer_flags": buffer_flags,
        "buffer_size_bytes": 1024,
        "comm": stale_comm,
    }
    allreduce = object.__new__(MNNVLAllReduce)
    torch.nn.Module.__init__(allreduce)
    allreduce.mapping = mapping
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(MNNVLAllReduce, "allreduce_mnnvl_workspaces", {mapping: workspace})

    assert not torch.is_inference_mode_enabled()
    allreduce.checkpoint_prepare()
    allreduce.checkpoint_prepare()
    with pytest.raises(TypeError):
        allreduce.checkpoint_restore()
    allreduce.checkpoint_restore(comm)
    allreduce.checkpoint_restore(redundant_comm)

    assert workspace["comm"] is comm.duplicates[0]
    assert stale_comm.allreduce_count == 0
    assert stale_comm.free_count == 1
    assert handle.prepare_count == 1
    assert handle.restore_count == 1
    assert handle.complete_count == 1
    assert comm.allreduce_count == 0
    assert comm.dup_count == 1
    assert redundant_comm.allreduce_count == 0
    assert redundant_comm.dup_count == 0
    assert redundant_comm.duplicates == []
    assert not torch.is_inference_mode_enabled()
    assert torch.all(uc_buffer == 0)
    assert torch.all(torch.signbit(uc_buffer))
    assert torch.equal(
        buffer_flags,
        torch.tensor([0, 2, 1024, 0, 0, 0, 0, 0, 0], dtype=torch.uint32),
    )


def test_checkpoint_restore_protocol_failure_is_terminal(monkeypatch) -> None:
    mapping = object()
    comm = _FakeComm()
    handle = _FakeMcastBuffer()
    workspace = {
        "handle": handle,
        "uc_buffer": torch.ones(8, dtype=torch.float32),
        "buffer_flags": torch.ones(9, dtype=torch.uint32),
        "buffer_size_bytes": 1024,
        "comm": _FakeComm(),
    }
    allreduce = object.__new__(MNNVLAllReduce)
    torch.nn.Module.__init__(allreduce)
    allreduce.mapping = mapping
    monkeypatch.setattr(MNNVLAllReduce, "allreduce_mnnvl_workspaces", {mapping: workspace})
    monkeypatch.setattr(
        ops_module,
        "_initialize_allreduce_mnnvl_protocol",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("injected protocol reset failure")
        ),
    )

    allreduce.checkpoint_prepare()
    with pytest.raises(RuntimeError, match="injected protocol reset failure"):
        allreduce.checkpoint_restore(comm)

    assert not handle.is_mapped()
    assert not handle.restore_pending
    assert handle.complete_count == 1
    assert workspace["comm"] is None
    assert comm.duplicates[0].free_count == 1


def test_checkpoint_restore_communicator_duplication_failure_is_terminal(monkeypatch) -> None:
    mapping = object()
    handle = _FakeMcastBuffer()
    workspace = {
        "handle": handle,
        "uc_buffer": torch.ones(8, dtype=torch.float32),
        "buffer_flags": torch.ones(9, dtype=torch.uint32),
        "buffer_size_bytes": 1024,
        "comm": _FakeComm(),
    }
    allreduce = object.__new__(MNNVLAllReduce)
    torch.nn.Module.__init__(allreduce)
    allreduce.mapping = mapping
    monkeypatch.setattr(MNNVLAllReduce, "allreduce_mnnvl_workspaces", {mapping: workspace})

    allreduce.checkpoint_prepare()
    with pytest.raises(RuntimeError, match="injected communicator duplication failure"):
        allreduce.checkpoint_restore(_FailingDupComm())

    assert not handle.is_mapped()
    assert not handle.restore_pending
    assert handle.complete_count == 1
    assert workspace["comm"] is None


def test_workspace_uses_alignment_slack_without_reallocation(monkeypatch) -> None:
    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    world = _patch_workspace_dependencies(monkeypatch)
    initial_capacity = 1024 * 8192 * torch.float32.itemsize

    class SlackWorkspace(_FakeMcastBuffer):
        def get_workspace_size(self) -> int:
            return 3 * (initial_capacity + 1024) + 16

    monkeypatch.setattr(ops_module, "MnnvlWorkspace", SlackWorkspace)
    workspace = get_or_scale_allreduce_mnnvl_workspace(mapping, torch.float32)
    reused = get_or_scale_allreduce_mnnvl_workspace(mapping, torch.float32, initial_capacity + 512)
    assert reused is workspace
    assert workspace["buffer_size_bytes"] == initial_capacity + 1024
    assert world.split_count == 1


def test_workspace_growth_during_capture_preserves_existing_workspace(monkeypatch) -> None:
    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    _patch_workspace_dependencies(monkeypatch)
    monkeypatch.setattr(ops_module, "MnnvlWorkspace", _FakeMcastBuffer)
    workspace = get_or_scale_allreduce_mnnvl_workspace(mapping, torch.float32)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(
        RuntimeError, match="workspace creation or growth during CUDA graph capture"
    ):
        get_or_scale_allreduce_mnnvl_workspace(
            mapping, torch.float32, workspace["buffer_size_bytes"] + 32
        )
    assert MNNVLAllReduce.allreduce_mnnvl_workspaces[mapping] is workspace


def test_preflight_unavailability_does_not_publish_workspace(monkeypatch) -> None:
    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    world = _patch_workspace_dependencies(monkeypatch)

    class UnavailableWorkspace(_FakeMcastBuffer):
        is_available = False
        reason = "fabric handles unavailable"

    monkeypatch.setattr(ops_module, "MnnvlWorkspace", UnavailableWorkspace)
    with pytest.raises(ops_module._MnnvlUnavailableError, match="fabric handles unavailable"):
        get_or_scale_allreduce_mnnvl_workspace(mapping, torch.float32)
    assert mapping not in MNNVLAllReduce.allreduce_mnnvl_workspaces
    assert MNNVLAllReduce.allreduce_mnnvl_pending_comms[mapping] is world.comm


def test_checkpoint_restore_retains_ray_process_group(monkeypatch) -> None:
    monkeypatch.setattr(ops_module, "mpi_disabled", lambda: True)
    monkeypatch.setattr(ops_module, "torch_pybind11_abi", lambda: "abi")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    mapping = object()
    process_group = object()

    class RayWorkspace(_FakeMcastBuffer):
        def checkpoint_restore(self, *, process_group, pybind11_abi) -> bool:
            self.process_group = process_group
            assert pybind11_abi == "abi"
            return super().checkpoint_restore(42)

    handle = RayWorkspace()
    workspace = {
        "handle": handle,
        "uc_buffer": torch.ones(8),
        "buffer_flags": torch.ones(9, dtype=torch.uint32),
        "buffer_size_bytes": 1024,
        "comm": object(),
    }
    monkeypatch.setattr(MNNVLAllReduce, "allreduce_mnnvl_workspaces", {mapping: workspace})
    allreduce = object.__new__(MNNVLAllReduce)
    torch.nn.Module.__init__(allreduce)
    allreduce.mapping = mapping
    allreduce.checkpoint_prepare()
    allreduce.checkpoint_restore(process_group)
    assert handle.process_group is process_group
    assert workspace["comm"] is process_group
    assert handle.is_mapped()
    assert torch.all(torch.signbit(workspace["uc_buffer"]))


def test_rms_norm_without_residual_returns_single_tensor(monkeypatch) -> None:
    mapping = Mapping(world_size=2, tp_size=2, rank=0)
    allreduce = object.__new__(MNNVLAllReduce)
    torch.nn.Module.__init__(allreduce)
    allreduce.mapping = mapping
    allreduce.dtype = torch.bfloat16
    allreduce.transport = ops_module.MnnvlTransport.POSIX_FD
    workspace = {
        "uc_buffer": torch.zeros(48),
        "buffer_flags": torch.zeros(9, dtype=torch.uint32),
    }
    monkeypatch.setattr(
        ops_module, "get_or_scale_allreduce_mnnvl_workspace", lambda *args, **kwargs: workspace
    )
    expected = torch.ones((2, 8), dtype=torch.bfloat16)
    calls = []

    def fusion(*args):
        calls.append(args)
        return [expected]

    monkeypatch.setattr(torch.ops.trtllm, "mnnvl_fusion_allreduce", fusion, raising=False)
    params = ops_module.AllReduceParams(
        fusion_op=ops_module.AllReduceFusionOp.RMS_NORM,
        norm_weight=torch.ones(8, dtype=torch.bfloat16),
    )
    assert allreduce(expected, params) is expected
    assert calls[0][2] is None
    assert calls[0][-1] == int(ops_module.AllReduceFusionOp.RMS_NORM)


def test_dtype_mismatch_falls_back_before_workspace_access() -> None:
    allreduce = object.__new__(MNNVLAllReduce)
    torch.nn.Module.__init__(allreduce)
    allreduce.dtype = torch.bfloat16
    params = ops_module.AllReduceParams()
    assert allreduce(torch.ones((2, 8), dtype=torch.float32), params) is None
