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
"""Communicator selection for the MNNVL AllReduce workspace.

The MNNVL workspace needs a TP-wide host collective to exchange CUDA memory handles. Under MPI
that is a split of the session communicator; under Ray there is no MPI communicator and the TP
ProcessGroup takes its place. These tests pin that dispatch without needing GPUs or a real
multi-rank job -- calling into MPI under Ray silently produces a size-1 communicator, which used
to make the MNNVL workspace allocation fail and fall back to NCCL.
"""

import pytest

from tensorrt_llm._torch.distributed import ops


class _FakeWork:
    def __init__(self):
        self.waits = 0

    def wait(self):
        self.waits += 1


class _FakeBackend:
    def __init__(self, group_size):
        self.group_size = group_size
        self.allreduces = 0
        self.work = _FakeWork()

    def allreduce(self, tensors):
        self.allreduces += 1
        # Stand in for the peers by having every rank report what this one did.
        for tensor in tensors:
            tensor.mul_(self.group_size)
        return self.work


class _FakeProcessGroup:
    """Records the backend lookups the workspace collectives make."""

    GROUP_SIZE = 4

    def __init__(self):
        self.backend = _FakeBackend(self.GROUP_SIZE)
        self.requested_devices = []

    def size(self):
        return self.GROUP_SIZE

    def _get_backend(self, device):
        self.requested_devices.append(device)
        return self.backend


class _FakeMapping:
    """Just enough of Mapping for the workspace helpers."""

    def __init__(self, tp_group_pg=None):
        self.tp_size = 4
        self.tp_rank = 1
        self.pp_rank = 0
        self.cp_size = 1
        self.cp_rank = 0
        self.local_rank = 3
        self.tp_group_pg = tp_group_pg
        self.multi_node = False

    def has_cp(self):
        return False

    def is_multi_node(self):
        return self.multi_node


@pytest.fixture
def ray_mode(monkeypatch):
    monkeypatch.setattr(ops, "mpi_disabled", lambda: True)


@pytest.fixture
def mpi_mode(monkeypatch):
    monkeypatch.setattr(ops, "mpi_disabled", lambda: False)


def test_workspace_comm_is_tp_process_group_under_ray(ray_mode, monkeypatch):
    pg = _FakeProcessGroup()

    def _no_mpi():
        pytest.fail("mpi_comm() must not be used when MPI is disabled")

    monkeypatch.setattr(ops, "mpi_comm", _no_mpi)

    assert ops._get_mnnvl_workspace_comm(_FakeMapping(pg)) is pg


def test_workspace_comm_rejects_missing_process_group(ray_mode):
    with pytest.raises(AssertionError):
        ops._get_mnnvl_workspace_comm(_FakeMapping(tp_group_pg=None))


def test_workspace_comm_splits_mpi_comm_by_tp_rank(mpi_mode, monkeypatch):
    recorded = {}

    class _FakeMpiComm:
        def Split(self, color, key):
            recorded["color"] = color
            recorded["key"] = key
            return "split-comm"

    monkeypatch.setattr(ops, "mpi_comm", lambda: _FakeMpiComm())

    mapping = _FakeMapping()
    assert ops._get_mnnvl_workspace_comm(mapping) == "split-comm"
    # Ranks sharing a PP/CP slice land in the same group, ordered by TP rank.
    assert recorded == {"color": 0, "key": mapping.tp_rank}


def test_convergence_uses_process_group_cpu_backend_under_ray(ray_mode, monkeypatch):
    """Convergence must reach the group's CPU backend without going through the dispatcher.

    torch.distributed's collectives are c10d operators that build their tensors with at::empty.
    Workspaces are set up while the model is still under MetaInitMode, which redirects that
    allocation to the meta device and then rejects the operator with MetaInitException.
    """
    monkeypatch.setattr(
        ops.torch.distributed,
        "all_reduce",
        lambda *args, **kwargs: pytest.fail(
            "workspace convergence must not use the dispatched torch.distributed.all_reduce"
        ),
    )

    pg = _FakeProcessGroup()
    assert ops._mnnvl_workspace_all_succeeded(pg, True)

    assert pg.requested_devices == [ops.torch.device("cpu")]
    assert pg.backend.allreduces == 1
    # The reduction is what fences the handle exchange, so it has to be waited on, not just issued.
    assert pg.backend.work.waits == 1


def test_convergence_reports_a_failed_rank_under_ray(ray_mode):
    """One rank failing has to be visible to all of them, not just to itself."""
    pg = _FakeProcessGroup()
    assert not ops._mnnvl_workspace_all_succeeded(pg, False)


def test_convergence_uses_mpi_comm_under_mpi(mpi_mode):
    class _FakeMpiComm:
        def __init__(self):
            self.reduced = []

        def allreduce(self, value):
            self.reduced.append(value)
            return value * 4

        def Get_size(self):
            return 4

    comm = _FakeMpiComm()
    assert ops._mnnvl_workspace_all_succeeded(comm, True)
    assert comm.reduced == [1]
    assert not ops._mnnvl_workspace_all_succeeded(comm, False)


def test_mcast_buffer_receives_process_group_under_ray(ray_mode, monkeypatch):
    pg = _FakeProcessGroup()
    captured = {}

    def _fake_buffer(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "buffer"

    monkeypatch.setattr(ops, "McastGPUBuffer", _fake_buffer)
    monkeypatch.setattr(ops, "torch_pybind11_abi", lambda: "abi")
    monkeypatch.setattr(ops.torch.cuda, "current_device", lambda: 0)

    mapping = _FakeMapping(pg)
    assert ops._make_mnnvl_mcast_buffer(pg, 4096, mapping, True) == "buffer"

    assert captured["kwargs"]["process_group"] is pg
    assert captured["kwargs"]["pybind11_abi"] == "abi"
    # No MPI Fortran handle is passed on this path.
    assert captured["args"] == (4096, mapping.tp_size, mapping.tp_rank, 0, True)


def test_mcast_buffer_receives_fortran_handle_under_mpi(mpi_mode, monkeypatch):
    captured = {}

    def _fake_buffer(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "buffer"

    class _FakeMpiComm:
        def py2f(self):
            return 42

    monkeypatch.setattr(ops, "McastGPUBuffer", _fake_buffer)

    mapping = _FakeMapping()
    ops._make_mnnvl_mcast_buffer(_FakeMpiComm(), 4096, mapping, True)

    assert captured["kwargs"] == {}
    assert captured["args"] == (
        4096,
        mapping.tp_size,
        mapping.tp_rank,
        mapping.local_rank,
        True,
        42,
    )


def test_device_index_prefers_current_device_under_ray(ray_mode, monkeypatch):
    # A Ray worker may run under a remapped CUDA_VISIBLE_DEVICES, so its device is not
    # necessarily rank % gpus_per_node.
    monkeypatch.setattr(ops.torch.cuda, "current_device", lambda: 0)
    assert ops._mnnvl_device_index(_FakeMapping()) == 0


def test_device_index_uses_local_rank_under_mpi(mpi_mode):
    mapping = _FakeMapping()
    assert ops._mnnvl_device_index(mapping) == mapping.local_rank


@pytest.fixture
def mnnvl_capable_hardware(monkeypatch):
    """Make every hardware-level precondition of is_mnnvl() pass."""
    import tensorrt_llm._torch.distributed.mnnvl_memory as mnnvl_utils

    monkeypatch.setattr(ops.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(mnnvl_utils.MnnvlMemory, "supports_mnnvl", staticmethod(lambda: True))
    monkeypatch.delenv("TLLM_TEST_MNNVL", raising=False)


def test_auto_does_not_pick_mnnvl_on_a_single_node(mnnvl_capable_hardware):
    import torch

    mapping = _FakeMapping()
    mapping.multi_node = False
    assert not ops.MNNVLAllReduce.is_mnnvl(mapping, torch.bfloat16)


def test_explicit_request_enables_mnnvl_on_a_single_node(mnnvl_capable_hardware):
    import torch

    mapping = _FakeMapping()
    mapping.multi_node = False
    assert ops.MNNVLAllReduce.is_mnnvl(mapping, torch.bfloat16, explicitly_requested=True)


def test_auto_picks_mnnvl_across_nodes(mnnvl_capable_hardware):
    import torch

    mapping = _FakeMapping()
    mapping.multi_node = True
    assert ops.MNNVLAllReduce.is_mnnvl(mapping, torch.bfloat16)


def test_explicit_request_still_respects_hardware(mnnvl_capable_hardware, monkeypatch):
    """An explicit request relaxes the single-node heuristic, not the dtype check."""
    import torch

    mapping = _FakeMapping()
    mapping.multi_node = True
    assert not ops.MNNVLAllReduce.is_mnnvl(mapping, torch.int8, explicitly_requested=True)
