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


class _FakeProcessGroup:
    pass


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


def test_barrier_uses_process_group_under_ray(ray_mode, monkeypatch):
    pg = _FakeProcessGroup()
    seen = {}

    monkeypatch.setattr(
        ops.torch.distributed, "barrier", lambda group: seen.setdefault("group", group)
    )

    ops._mnnvl_workspace_barrier(pg)
    assert seen["group"] is pg


def test_barrier_uses_mpi_comm_under_mpi(mpi_mode):
    class _FakeMpiComm:
        def __init__(self):
            self.barriers = 0

        def Barrier(self):
            self.barriers += 1

    comm = _FakeMpiComm()
    ops._mnnvl_workspace_barrier(comm)
    assert comm.barriers == 1


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
    import tensorrt_llm._mnnvl_utils as mnnvl_utils

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
