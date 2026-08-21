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
"""Communicator selection for MnnvlMemory (the MoE / MNNVL workspace allocator).

Sibling of test_mnnvl_workspace_comm.py, which pins the same dispatch for the
AllReduce workspace in _torch/distributed/ops.py. _mnnvl_utils.py used
mpi_comm().Split() unconditionally, so under Ray -- where workers are not
launched by mpirun and each process is its own MPI singleton -- Get_size()
returned 1 while the MoE workspace was sized and indexed with moe_ep_size. The
strided workspace tensor then had one segment instead of ep_size, and
FusedMoeWorkspace::initializeLocalWorkspace memset past the end of the
allocation: CUDA_ERROR_ILLEGAL_ADDRESS, a poisoned CUDA context, and an
apparently unrelated illegal-access failure at the next CUDA call.

These tests need neither GPUs nor a multi-rank job.
"""

import pytest
import torch

from tensorrt_llm import _mnnvl_utils
from tensorrt_llm._mnnvl_utils import (HelixCpMnnvlMemory, MnnvlMemory,
                                       ProcessGroupComm)
from tensorrt_llm._torch.models.modeling_utils import (MetaInitException,
                                                       MetaInitMode)


class _FakeProcessGroup:
    """Stands in for c10d.ProcessGroup; identity is all the dispatch needs."""

    def __init__(self, name="pg"):
        self.name = name


class _FakeMapping:
    """Just enough of Mapping for the communicator helpers."""

    def __init__(self, tp_group_pg=None, cp_group_pg=None):
        self.tp_size = 4
        self.tp_rank = 1
        self.pp_rank = 0
        self.cp_size = 1
        self.cp_rank = 0
        self.moe_tp_size = 1
        self.moe_tp_rank = 0
        self.tp_group_pg = tp_group_pg
        self.cp_group_pg = cp_group_pg


@pytest.fixture(autouse=True)
def _reset_cached_comms():
    """get_comm() memoizes into class state; keep tests independent."""
    MnnvlMemory.comm = None
    HelixCpMnnvlMemory.comm = None
    yield
    MnnvlMemory.comm = None
    HelixCpMnnvlMemory.comm = None


@pytest.fixture
def ray_mode(monkeypatch):
    monkeypatch.setattr(_mnnvl_utils, "mpi_disabled", lambda: True)


@pytest.fixture
def mpi_mode(monkeypatch):
    monkeypatch.setattr(_mnnvl_utils, "mpi_disabled", lambda: False)


def _forbid_mpi(monkeypatch):
    def _no_mpi():
        pytest.fail("mpi_comm() must not be used when MPI is disabled")

    monkeypatch.setattr(_mnnvl_utils, "mpi_comm", _no_mpi)


def test_get_comm_uses_tp_process_group_under_ray(ray_mode, monkeypatch):
    _forbid_mpi(monkeypatch)
    pg = _FakeProcessGroup("tp")

    comm = MnnvlMemory.get_comm(_FakeMapping(tp_group_pg=pg))

    assert isinstance(comm, ProcessGroupComm)
    assert comm._pg is pg


def test_helix_cp_get_comm_uses_cp_process_group_under_ray(
        ray_mode, monkeypatch):
    _forbid_mpi(monkeypatch)
    pg = _FakeProcessGroup("cp")

    comm = HelixCpMnnvlMemory.get_comm(_FakeMapping(cp_group_pg=pg))

    assert isinstance(comm, ProcessGroupComm)
    assert comm._pg is pg


def test_get_comm_rejects_missing_process_group(ray_mode, monkeypatch):
    _forbid_mpi(monkeypatch)
    with pytest.raises(AssertionError):
        MnnvlMemory.get_comm(_FakeMapping(tp_group_pg=None))


def test_get_comm_still_splits_mpi_comm_under_mpi(mpi_mode, monkeypatch):
    """The MPI path must be untouched: same split key and ordering as before."""
    recorded = {}

    class _FakeMpiComm:

        def Split(self, color, key):
            recorded["color"] = color
            recorded["key"] = key
            return "mpi-split-comm"

    monkeypatch.setattr(_mnnvl_utils, "mpi_comm", lambda: _FakeMpiComm())

    mapping = _FakeMapping()
    comm = MnnvlMemory.get_comm(mapping)

    assert comm == "mpi-split-comm"
    # (pp_rank * cp_size + cp_rank) * moe_tp_size + moe_tp_rank, keyed by tp_rank
    assert recorded["color"] == 0
    assert recorded["key"] == mapping.tp_rank


def test_get_comm_is_cached(ray_mode, monkeypatch):
    _forbid_mpi(monkeypatch)
    mapping = _FakeMapping(tp_group_pg=_FakeProcessGroup())

    first = MnnvlMemory.get_comm(mapping)
    second = MnnvlMemory.get_comm(mapping)

    assert first is second


def test_process_group_comm_delegates_size_and_rank(monkeypatch):
    pg = _FakeProcessGroup()
    monkeypatch.setattr(torch.distributed, "get_world_size",
                        lambda group=None: 4 if group is pg else -1)
    monkeypatch.setattr(torch.distributed, "get_rank",
                        lambda group=None: 2 if group is pg else -1)

    comm = ProcessGroupComm(pg)

    assert comm.Get_size() == 4
    assert comm.Get_rank() == 2


def test_process_group_comm_allgather_returns_one_entry_per_rank(monkeypatch):
    pg = _FakeProcessGroup()
    monkeypatch.setattr(torch.distributed, "get_world_size",
                        lambda group=None: 3)

    def _fake_all_gather_object(out_list, obj, group=None):
        assert group is pg
        for i in range(len(out_list)):
            out_list[i] = (i, obj)

    monkeypatch.setattr(torch.distributed, "all_gather_object",
                        _fake_all_gather_object)

    assert ProcessGroupComm(pg).allgather("handle") == [(0, "handle"),
                                                        (1, "handle"),
                                                        (2, "handle")]


def test_process_group_comm_allgather_survives_meta_init_mode(monkeypatch):
    """The regression this adapter exists for.

    MNNVL workspaces are built while the model is under MetaInitMode, which
    redirects aten.empty to the meta device and then rejects any non-random-init
    op that touches a meta tensor. all_gather_object materializes a real byte
    tensor and calls aten.set_.source_Storage on it, so without popping the
    dispatch modes the handle exchange dies with
    "Meta tensor used in unsupported function".
    """
    pg = _FakeProcessGroup()
    monkeypatch.setattr(torch.distributed, "get_world_size",
                        lambda group=None: 2)

    def _all_gather_object_like_torch(out_list, obj, group=None):
        # Mirror _object_to_tensor: allocate, then set_ from a storage. Under an
        # active MetaInitMode the allocation becomes meta and set_ raises.
        buf = torch.empty(8, dtype=torch.uint8)
        buf.set_(torch.empty(8, dtype=torch.uint8).untyped_storage())
        for i in range(len(out_list)):
            out_list[i] = obj

    monkeypatch.setattr(torch.distributed, "all_gather_object",
                        _all_gather_object_like_torch)

    comm = ProcessGroupComm(pg)

    # Sanity: the same call pattern really is rejected when modes stay active.
    with MetaInitMode():
        with pytest.raises(MetaInitException):
            buf = torch.empty(8, dtype=torch.uint8)
            buf.set_(torch.empty(8, dtype=torch.uint8).untyped_storage())

    with MetaInitMode():
        assert comm.allgather("handle") == ["handle", "handle"]


def test_process_group_comm_barrier_uses_cpu_backend(monkeypatch):
    """Barrier must go straight to the CPU backend.

    The public ProcessGroup.barrier() is a c10d operator, so MetaInitMode
    intercepts it before it reaches Gloo; ops.py::_mnnvl_workspace_barrier takes
    the same shortcut.
    """
    waited = []

    class _Work:

        def wait(self):
            waited.append(True)

    class _Backend:

        def barrier(self):
            return _Work()

    class _Pg:

        def __init__(self):
            self.requested_devices = []

        def _get_backend(self, device):
            self.requested_devices.append(device)
            return _Backend()

    pg = _Pg()
    ProcessGroupComm(pg).barrier()

    assert pg.requested_devices == [torch.device("cpu")]
    assert waited == [True]
