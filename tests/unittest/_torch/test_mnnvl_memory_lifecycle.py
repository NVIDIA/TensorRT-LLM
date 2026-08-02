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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import tensorrt_llm._mnnvl_utils as mnnvl
from tensorrt_llm._torch.distributed.moe_alltoall import MoeAlltoAll, _A2AState


class _FakeComm:
    def __init__(self, rank=0, size=2):
        self.rank = rank
        self.size = size
        self.barrier_count = 0

    def Get_rank(self):
        return self.rank

    def Get_size(self):
        return self.size

    def barrier(self):
        self.barrier_count += 1


class _TestMnnvlMemory(mnnvl.MnnvlMemory):
    pass


@pytest.fixture
def memory(monkeypatch):
    comm = _FakeComm()
    record = mnnvl._MnnvlAllocationRecord(
        comm=comm,
        comm_size=2,
        comm_rank=0,
        aligned_size=64,
        mem_handles=[11, 22],
        start_address=1000,
        rank_stride=256,
        address_offset=32,
    )
    obj = _TestMnnvlMemory.__new__(_TestMnnvlMemory)
    obj.ptr = 1032
    _TestMnnvlMemory.allocated_map = {obj.ptr: record}
    _TestMnnvlMemory.address_refcnt = {record.start_address: 1}
    _TestMnnvlMemory.current_start_address = record.start_address
    _TestMnnvlMemory.current_rank_stride = record.rank_stride
    _TestMnnvlMemory.current_mem_offset = record.address_offset + record.aligned_size

    monkeypatch.setattr(mnnvl.torch.cuda, "synchronize", Mock())
    monkeypatch.setattr(mnnvl, "_check_cu_result", lambda result: result)
    monkeypatch.setattr(mnnvl.cuda, "cuMemUnmap", Mock(return_value=None))
    monkeypatch.setattr(mnnvl.cuda, "cuMemRelease", Mock(return_value=None))
    yield obj, record
    _TestMnnvlMemory.allocated_map = {}
    _TestMnnvlMemory.address_refcnt = {}
    if hasattr(obj, "ptr"):
        del obj.ptr


def test_checkpoint_prepare_preserves_va_and_is_idempotent(memory):
    obj, record = memory

    obj.checkpoint_prepare()

    assert not obj.mapped
    assert record.start_address == 1000
    assert record.rank_stride == 256
    assert record.address_offset == 32
    assert record.mem_handles == [None, None]
    assert record.comm.barrier_count == 2
    assert [call.args[0] for call in mnnvl.cuda.cuMemUnmap.call_args_list] == [1032, 1288]

    obj.checkpoint_prepare()
    assert record.comm.barrier_count == 2


def test_checkpoint_restore_reuses_layout_with_fresh_handles(memory, monkeypatch):
    obj, record = memory
    obj.checkpoint_prepare()
    restored_comm = _FakeComm()
    create_and_map = Mock(return_value=[33, 44])
    monkeypatch.setattr(_TestMnnvlMemory, "_create_and_map_handles", create_and_map)

    obj.checkpoint_restore(restored_comm)

    create_and_map.assert_called_once_with(restored_comm, 64, 1000, 256, 32)
    assert obj.ptr == 1032
    assert obj.mapped
    assert record.mem_handles == [33, 44]
    assert record.comm is restored_comm
    assert _TestMnnvlMemory.comm is restored_comm


def test_checkpoint_restore_rejects_changed_rank_layout(memory):
    obj, _ = memory
    obj.checkpoint_prepare()

    with pytest.raises(RuntimeError, match="rank/size 1/2 != 0/2"):
        obj.checkpoint_restore(_FakeComm(rank=1))

    assert not obj.mapped


def test_close_detached_memory_only_frees_va(memory, monkeypatch):
    obj, record = memory
    obj.checkpoint_prepare()
    address_free = Mock(return_value=None)
    monkeypatch.setattr(mnnvl.cuda, "CUdeviceptr", lambda value: value)
    monkeypatch.setattr(mnnvl.cuda, "cuMemAddressFree", address_free)

    _TestMnnvlMemory.close_mnnvl_memory(obj.ptr)

    address_free.assert_called_once_with(
        record.start_address, record.comm_size * record.rank_stride
    )
    assert obj.ptr not in _TestMnnvlMemory.allocated_map
    assert record.start_address not in _TestMnnvlMemory.address_refcnt
    del obj.ptr


def test_create_and_map_handles_cleans_partial_allocation(monkeypatch):
    comm = Mock()
    comm.Get_rank.return_value = 0
    comm.Get_size.return_value = 2
    comm.allgather.return_value = [b"local", b"remote"]
    allocation_prop = SimpleNamespace(
        requestedHandleTypes=mnnvl.cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC,
        location=object(),
    )
    access_desc = SimpleNamespace(location=None, flags=None)

    monkeypatch.setattr(mnnvl.MnnvlMemory, "dev_id", 0)
    monkeypatch.setattr(
        mnnvl.MnnvlMemory,
        "get_allocation_prop",
        Mock(return_value=allocation_prop),
    )
    monkeypatch.setattr(mnnvl, "_check_cu_result", lambda result: result)
    monkeypatch.setattr(mnnvl.cuda, "cuCtxGetDevice", Mock(return_value=0))
    monkeypatch.setattr(mnnvl.cuda, "cuMemCreate", Mock(return_value=11))
    monkeypatch.setattr(
        mnnvl.cuda,
        "cuMemExportToShareableHandle",
        Mock(return_value=SimpleNamespace(data=b"local")),
    )
    monkeypatch.setattr(mnnvl.cuda, "CUmemAccessDesc", Mock(return_value=access_desc))
    monkeypatch.setattr(mnnvl.cuda, "cuMemImportFromShareableHandle", Mock(return_value=22))
    map_memory = Mock(side_effect=[None, RuntimeError("map failed")])
    unmap_memory = Mock(return_value=None)
    release_memory = Mock(return_value=None)
    monkeypatch.setattr(mnnvl.cuda, "cuMemMap", map_memory)
    monkeypatch.setattr(mnnvl.cuda, "cuMemSetAccess", Mock(return_value=None))
    monkeypatch.setattr(mnnvl.cuda, "cuMemUnmap", unmap_memory)
    monkeypatch.setattr(mnnvl.cuda, "cuMemRelease", release_memory)

    with pytest.raises(RuntimeError, match="map failed"):
        _TestMnnvlMemory._create_and_map_handles(comm, 64, 1000, 256, 32)

    unmap_memory.assert_called_once_with(1032, 64)
    assert [call.args[0] for call in release_memory.call_args_list] == [11, 22]


def test_create_and_map_handles_releases_local_handle_on_export_failure(monkeypatch):
    comm = Mock()
    comm.Get_rank.return_value = 0
    comm.Get_size.return_value = 2
    allocation_prop = SimpleNamespace(
        requestedHandleTypes=mnnvl.cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC,
        location=object(),
    )
    release_memory = Mock(return_value=None)

    monkeypatch.setattr(mnnvl.MnnvlMemory, "dev_id", 0)
    monkeypatch.setattr(
        mnnvl.MnnvlMemory,
        "get_allocation_prop",
        Mock(return_value=allocation_prop),
    )
    monkeypatch.setattr(mnnvl, "_check_cu_result", lambda result: result)
    monkeypatch.setattr(mnnvl.cuda, "cuCtxGetDevice", Mock(return_value=0))
    monkeypatch.setattr(mnnvl.cuda, "cuMemCreate", Mock(return_value=11))
    monkeypatch.setattr(
        mnnvl.cuda,
        "cuMemExportToShareableHandle",
        Mock(side_effect=RuntimeError("export failed")),
    )
    monkeypatch.setattr(mnnvl.cuda, "cuMemRelease", release_memory)

    with pytest.raises(RuntimeError, match="export failed"):
        _TestMnnvlMemory._create_and_map_handles(comm, 64, 1000, 256, 32)

    release_memory.assert_called_once_with(11)


def _make_moe_alltoall_for_lifecycle():
    obj = MoeAlltoAll.__new__(MoeAlltoAll)
    obj._destroyed = True
    obj._state = _A2AState()
    obj._alltoall_watchdog = None
    obj.mnnvl_mem = Mock(mapped=True)
    return obj


def test_moe_alltoall_checkpoint_prepare_delegates_with_shared_owners():
    first = _make_moe_alltoall_for_lifecycle()
    second = _make_moe_alltoall_for_lifecycle()
    shared_memory = Mock(mapped=True)
    first.mnnvl_mem = shared_memory
    second.mnnvl_mem = shared_memory

    first.checkpoint_prepare()
    second.checkpoint_prepare()

    assert shared_memory.checkpoint_prepare.call_count == 2


def test_moe_alltoall_checkpoint_prepare_rejects_active_phase():
    obj = _make_moe_alltoall_for_lifecycle()
    obj._state.phase = "dispatched"
    with pytest.raises(RuntimeError, match="active MoE All-to-All phase"):
        obj.checkpoint_prepare()


def test_moe_alltoall_rejects_unmapped_workspace():
    obj = _make_moe_alltoall_for_lifecycle()
    obj.mnnvl_mem.mapped = False

    with pytest.raises(RuntimeError, match="workspace handles are unmapped"):
        obj._require_mapped()
