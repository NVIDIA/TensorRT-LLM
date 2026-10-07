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
import torch

import tensorrt_llm._mnnvl_utils as mnnvl
from tensorrt_llm._torch.moe.fused_moe.communication.moe_alltoall import MoeAlltoAll
from tensorrt_llm._torch.moe.fused_moe.communication.nvlink_two_sided import NVLinkTwoSided
from tensorrt_llm.mapping import Mapping


class _FakeComm:
    def __init__(self, rank=0, size=2, membership=(0, 1)):
        self.rank = rank
        self.size = size
        self.membership = membership
        self.barrier_count = 0

    def Get_rank(self):
        return self.rank

    def Get_size(self):
        return self.size

    def barrier(self):
        self.barrier_count += 1

    def allgather(self, value):
        if isinstance(value, int) and not isinstance(value, bool):
            return list(self.membership)
        return [value] * self.size


class _TestMnnvlMemory(mnnvl.MnnvlMemory):
    pass


def test_checkpoint_allgather_timeout_is_bounded(monkeypatch):
    orphaned_requests = []
    monkeypatch.setattr(mnnvl, "_MNNVL_CHECKPOINT_ORPHANED_REQUESTS", orphaned_requests)
    receive_request = SimpleNamespace(
        test=lambda: (False, None),
        Cancel=Mock(),
        Free=Mock(),
    )
    send_request = SimpleNamespace(
        test=lambda: (False, None),
        Cancel=Mock(),
        Free=Mock(),
    )
    comm = SimpleNamespace(
        Get_rank=lambda: 0,
        Get_size=lambda: 2,
        irecv=lambda **kwargs: receive_request,
        isend=lambda value, **kwargs: send_request,
    )
    monkeypatch.setattr(mnnvl, "_MNNVL_CHECKPOINT_COLLECTIVE_TIMEOUT_S", 0.0)
    monkeypatch.setattr(mnnvl, "_MNNVL_CHECKPOINT_COLLECTIVE_POLL_INTERVAL_S", 0.0)

    with pytest.raises(TimeoutError, match="mapping readiness"):
        mnnvl._checkpoint_allgather(
            comm,
            None,
            operation="mapping readiness",
        )
    receive_request.Cancel.assert_called_once_with()
    receive_request.Free.assert_not_called()
    send_request.Cancel.assert_called_once_with()
    send_request.Free.assert_not_called()
    assert orphaned_requests == [receive_request, send_request]


def test_checkpoint_request_cleanup_retains_request_when_cancel_fails(monkeypatch):
    orphaned_requests = []
    monkeypatch.setattr(mnnvl, "_MNNVL_CHECKPOINT_ORPHANED_REQUESTS", orphaned_requests)
    request = SimpleNamespace(
        test=Mock(return_value=(False, None)),
        Cancel=Mock(side_effect=RuntimeError("cancel failed")),
        Free=Mock(),
    )
    monkeypatch.setattr(mnnvl, "_MNNVL_CHECKPOINT_COLLECTIVE_TIMEOUT_S", 0.0)

    mnnvl._cancel_checkpoint_requests([request])

    request.Cancel.assert_called_once_with()
    request.test.assert_called_once_with()
    request.Free.assert_not_called()
    assert orphaned_requests == [request]


def test_checkpoint_allgather_uses_bounded_object_requests():
    class _ReceiveRequest:
        def __init__(self, value):
            self.value = value

        def test(self):
            return True, self.value

    comm = SimpleNamespace(
        Get_rank=lambda: 1,
        Get_size=lambda: 3,
        irecv=lambda source, tag: _ReceiveRequest(f"rank-{source}"),
        isend=lambda value, dest, tag: SimpleNamespace(test=lambda: (True, None)),
        allgather=Mock(side_effect=AssertionError("blocking fallback used")),
    )

    result = mnnvl._checkpoint_allgather(comm, "rank-1", operation="test")

    assert result == ["rank-0", "rank-1", "rank-2"]
    comm.allgather.assert_not_called()


def test_checkpoint_allgather_post_failure_retires_partial_requests():
    requests = [
        SimpleNamespace(test=Mock(return_value=(True, None)), Cancel=Mock(), Free=Mock()),
        SimpleNamespace(test=Mock(return_value=(True, None)), Cancel=Mock(), Free=Mock()),
        SimpleNamespace(test=Mock(return_value=(True, None)), Cancel=Mock(), Free=Mock()),
    ]
    receive_requests = iter(requests[:2])
    send_requests = iter([requests[2], RuntimeError("post failed")])

    def isend(value, dest, tag):
        result = next(send_requests)
        if isinstance(result, Exception):
            raise result
        return result

    comm = SimpleNamespace(
        Get_rank=lambda: 0,
        Get_size=lambda: 3,
        irecv=lambda source, tag: next(receive_requests),
        isend=isend,
    )

    with pytest.raises(RuntimeError, match="post failed"):
        mnnvl._checkpoint_allgather(comm, None, operation="test")

    for request in requests:
        request.Cancel.assert_called_once_with()
        request.test.assert_called_once_with()
        request.Free.assert_not_called()


@pytest.fixture
def memory(monkeypatch):
    comm = _FakeComm()
    record = mnnvl._MnnvlAllocationRecord(
        comm=comm,
        comm_size=2,
        comm_rank=0,
        comm_membership=(0, 1),
        aligned_size=64,
        mem_handles=[11, 22],
        start_address=1000,
        rank_stride=256,
        address_offset=32,
    )
    obj = _TestMnnvlMemory.__new__(_TestMnnvlMemory)
    obj.ptr = 1032
    obj.mapping = SimpleNamespace(rank=0)
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
    assert record.comm.barrier_count == 0
    assert [call.args[0] for call in mnnvl.cuda.cuMemUnmap.call_args_list] == [1032, 1288]

    obj.checkpoint_prepare()
    assert record.comm.barrier_count == 0


def test_checkpoint_restore_reuses_layout_with_fresh_handles(memory, monkeypatch):
    obj, record = memory
    obj.checkpoint_prepare()
    restored_comm = _FakeComm()
    create_and_map = Mock(return_value=[33, 44])
    monkeypatch.setattr(_TestMnnvlMemory, "_create_and_map_handles", create_and_map)

    assert obj.checkpoint_restore(restored_comm)

    create_and_map.assert_called_once_with(restored_comm, 64, 1000, 256, 32)
    assert obj.ptr == 1032
    assert not obj.mapped
    assert record.state is mnnvl._MnnvlAllocationState.RESTORING
    assert record.mem_handles == [33, 44]
    assert record.comm is not restored_comm
    assert record.pending_comm is restored_comm
    obj._checkpoint_restore_complete()
    assert obj.mapped
    assert record.comm is restored_comm
    assert _TestMnnvlMemory.comm is restored_comm


def test_checkpoint_restore_rejects_changed_ordered_membership(memory):
    obj, record = memory
    obj.checkpoint_prepare()

    with pytest.raises(RuntimeError, match="ordered membership differs"):
        obj.checkpoint_restore(_FakeComm(membership=(1, 0)))

    assert record.state is mnnvl._MnnvlAllocationState.UNMAPPED
    assert record.pending_comm is None


def test_checkpoint_prepare_failure_is_terminal_and_fails_closed(memory):
    obj, record = memory
    mnnvl.cuda.cuMemUnmap.side_effect = [None, RuntimeError("unmap failed")]

    with pytest.raises(RuntimeError, match="unmap failed"):
        obj.checkpoint_prepare()

    assert not obj.mapped
    assert record.state is mnnvl._MnnvlAllocationState.BROKEN
    with pytest.raises(RuntimeError, match="broken state"):
        obj.checkpoint_prepare()


def test_checkpoint_restore_failure_is_terminal_and_fails_closed(memory, monkeypatch):
    obj, record = memory
    obj.checkpoint_prepare()
    monkeypatch.setattr(
        _TestMnnvlMemory,
        "_create_and_map_handles",
        Mock(side_effect=RuntimeError("restore failed")),
    )

    with pytest.raises(RuntimeError, match="restore failed"):
        obj.checkpoint_restore(_FakeComm())

    assert not obj.mapped
    assert record.state is mnnvl._MnnvlAllocationState.BROKEN
    with pytest.raises(RuntimeError, match="broken state"):
        obj.checkpoint_restore(_FakeComm())


def test_checkpoint_restore_membership_timeout_is_terminal(memory, monkeypatch):
    obj, record = memory
    obj.checkpoint_prepare()
    monkeypatch.setattr(mnnvl, "_MNNVL_CHECKPOINT_COLLECTIVE_TIMEOUT_S", 0.0)
    monkeypatch.setattr(mnnvl, "_MNNVL_CHECKPOINT_COLLECTIVE_POLL_INTERVAL_S", 0.0)

    receive_request = SimpleNamespace(
        test=lambda: (False, None),
        Cancel=Mock(),
        Free=Mock(),
    )
    comm = SimpleNamespace(
        Get_rank=lambda: 0,
        Get_size=lambda: 2,
        irecv=lambda source, tag: receive_request,
        isend=lambda value, dest, tag: SimpleNamespace(test=lambda: (True, None)),
    )
    with pytest.raises(TimeoutError, match="communicator membership"):
        obj.checkpoint_restore(comm)

    assert record.state is mnnvl._MnnvlAllocationState.BROKEN
    assert record.pending_comm is None
    with pytest.raises(RuntimeError, match="broken state"):
        obj.checkpoint_restore(comm)


def test_checkpoint_restore_timeout_cleans_locally_mapped_handles(memory, monkeypatch):
    obj, record = memory
    obj.checkpoint_prepare()
    monkeypatch.setattr(
        _TestMnnvlMemory,
        "_create_and_map_handles",
        Mock(return_value=[33, 44]),
    )
    monkeypatch.setattr(mnnvl, "_MNNVL_CHECKPOINT_COLLECTIVE_TIMEOUT_S", 0.0)
    monkeypatch.setattr(mnnvl, "_MNNVL_CHECKPOINT_COLLECTIVE_POLL_INTERVAL_S", 0.0)

    pending = object()

    class Request:
        def __init__(self, result=pending):
            self.result = result

        def test(self):
            return (self.result is not pending, None if self.result is pending else self.result)

        def Cancel(self):
            pass

        def Free(self):
            pass

    class TimeoutComm(_FakeComm):
        def __init__(self):
            super().__init__()
            self.receive_requests = iter([Request(1), Request()])

        def irecv(self, source, tag):
            return next(self.receive_requests)

        def isend(self, value, dest, tag):
            return Request(None)

    with pytest.raises(TimeoutError, match="mapping readiness"):
        obj.checkpoint_restore(TimeoutComm())

    assert record.state is mnnvl._MnnvlAllocationState.BROKEN
    assert record.mem_handles == [None, None]
    assert [call.args[0] for call in mnnvl.cuda.cuMemUnmap.call_args_list[-2:]] == [1032, 1288]


def test_handle_exchange_timeout_does_not_start_second_collective(memory, monkeypatch):
    obj, record = memory
    obj.checkpoint_prepare()
    monkeypatch.setattr(
        _TestMnnvlMemory,
        "_create_and_map_handles",
        Mock(side_effect=TimeoutError("handle exchange timed out")),
    )

    class CountingComm(_FakeComm):
        def __init__(self):
            super().__init__()
            self.allgather_count = 0

        def allgather(self, value):
            self.allgather_count += 1
            return super().allgather(value)

    comm = CountingComm()
    with pytest.raises(TimeoutError, match="handle exchange timed out"):
        obj.checkpoint_restore(comm)

    assert comm.allgather_count == 1
    assert record.state is mnnvl._MnnvlAllocationState.BROKEN


def test_failed_frontend_restore_releases_unpublished_handles(memory, monkeypatch):
    obj, record = memory
    obj.checkpoint_prepare()
    monkeypatch.setattr(
        _TestMnnvlMemory,
        "_create_and_map_handles",
        Mock(return_value=[33, 44]),
    )

    assert obj.checkpoint_restore(_FakeComm())
    obj._checkpoint_restore_failed()

    assert record.state is mnnvl._MnnvlAllocationState.BROKEN
    assert record.mem_handles == [None, None]
    assert record.pending_comm is None
    assert [call.args[0] for call in mnnvl.cuda.cuMemUnmap.call_args_list[-2:]] == [1032, 1288]
    assert [call.args[0] for call in mnnvl.cuda.cuMemRelease.call_args_list[-2:]] == [33, 44]


def test_failed_frontend_restore_cleanup_continues_after_cuda_errors(memory, monkeypatch):
    obj, record = memory
    obj.checkpoint_prepare()
    monkeypatch.setattr(
        _TestMnnvlMemory,
        "_create_and_map_handles",
        Mock(return_value=[33, 44]),
    )

    assert obj.checkpoint_restore(_FakeComm())
    mnnvl.cuda.cuMemUnmap.reset_mock()
    mnnvl.cuda.cuMemRelease.reset_mock()
    mnnvl.cuda.cuMemUnmap.side_effect = [RuntimeError("unmap failed"), None]
    mnnvl.cuda.cuMemRelease.side_effect = [RuntimeError("release failed"), None]

    obj._checkpoint_restore_failed()

    assert record.state is mnnvl._MnnvlAllocationState.BROKEN
    assert record.mem_handles == [33, None]
    assert record.pending_comm is None
    assert [call.args[0] for call in mnnvl.cuda.cuMemUnmap.call_args_list] == [1032, 1288]
    assert [call.args[0] for call in mnnvl.cuda.cuMemRelease.call_args_list] == [33, 44]


def test_checkpoint_restore_rejects_changed_rank_layout(memory):
    obj, _ = memory
    obj.checkpoint_prepare()

    with pytest.raises(RuntimeError, match="rank/size 1/2 != 0/2"):
        obj.checkpoint_restore(_FakeComm(rank=1))

    assert not obj.mapped


def test_mnnvl_moe_restore_publishes_all_workspaces_after_frontend_ready(monkeypatch):
    first = Mock()
    first.checkpoint_restore.return_value = True
    second = Mock()
    second.checkpoint_restore.return_value = True
    workspace_tensor = Mock()
    mapping = SimpleNamespace(moe_ep_rank=0, moe_ep_size=2)
    initialize_workspace = Mock()
    synchronize = Mock()
    comm = _FakeComm()
    monkeypatch.setattr(mnnvl.MnnvlMoe, "moe_workspace", first)
    monkeypatch.setattr(mnnvl.MnnvlMoe, "moe_prepare_workspace", second)
    monkeypatch.setattr(mnnvl.MnnvlMoe, "moe_workspace_tensor", workspace_tensor)
    monkeypatch.setattr(mnnvl.MnnvlMoe, "moe_mapping", mapping)
    monkeypatch.setattr(
        mnnvl.torch.ops.trtllm,
        "moe_initialize_workspace",
        initialize_workspace,
    )
    monkeypatch.setattr(mnnvl.torch.cuda, "synchronize", synchronize)

    mnnvl.MnnvlMoe.checkpoint_restore(comm)

    first.checkpoint_restore.assert_called_once_with(comm)
    second.checkpoint_restore.assert_called_once_with(comm)
    initialize_workspace.assert_called_once_with(workspace_tensor, 0, 2)
    synchronize.assert_called_once_with()
    assert comm.barrier_count == 0
    first._checkpoint_restore_complete.assert_called_once_with()
    second._checkpoint_restore_complete.assert_called_once_with()


def test_mnnvl_moe_restore_prepare_only_skips_main_workspace_initialization(monkeypatch):
    main_workspace = Mock()
    main_workspace.checkpoint_restore.return_value = False
    prepare_workspace = Mock()
    prepare_workspace.checkpoint_restore.return_value = True
    workspace_tensor = Mock()
    initialize_workspace = Mock()
    synchronize = Mock()
    comm = _FakeComm()
    monkeypatch.setattr(mnnvl.MnnvlMoe, "moe_workspace", main_workspace)
    monkeypatch.setattr(mnnvl.MnnvlMoe, "moe_prepare_workspace", prepare_workspace)
    monkeypatch.setattr(mnnvl.MnnvlMoe, "moe_workspace_tensor", workspace_tensor)
    monkeypatch.setattr(
        mnnvl.torch.ops.trtllm,
        "moe_initialize_workspace",
        initialize_workspace,
    )
    monkeypatch.setattr(mnnvl.torch.cuda, "synchronize", synchronize)

    mnnvl.MnnvlMoe.checkpoint_restore(comm)

    initialize_workspace.assert_not_called()
    synchronize.assert_called_once_with()
    assert comm.barrier_count == 0
    main_workspace._checkpoint_restore_complete.assert_not_called()
    prepare_workspace._checkpoint_restore_complete.assert_called_once_with()


def test_mnnvl_moe_restore_failure_marks_earlier_workspace_broken(monkeypatch):
    first = Mock()
    first.checkpoint_restore.return_value = True
    second = Mock()
    second.checkpoint_restore.side_effect = RuntimeError("second restore failed")
    monkeypatch.setattr(mnnvl.MnnvlMoe, "moe_workspace", first)
    monkeypatch.setattr(mnnvl.MnnvlMoe, "moe_prepare_workspace", second)

    with pytest.raises(RuntimeError, match="second restore failed"):
        mnnvl.MnnvlMoe.checkpoint_restore(_FakeComm())

    first._checkpoint_restore_failed.assert_called_once_with()
    first._checkpoint_restore_complete.assert_not_called()
    second._checkpoint_restore_complete.assert_not_called()


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
    comm.isend = None
    comm.irecv = None
    comm.Get_rank.return_value = 0
    comm.Get_size.return_value = 2
    comm.allgather.side_effect = lambda value: [
        value,
        {
            "error": None,
            "handle": b"remote",
            "is_fabric": True,
            "pid": 1001,
        },
    ]
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
    comm.isend = None
    comm.irecv = None
    comm.Get_rank.return_value = 0
    comm.Get_size.return_value = 2
    comm.allgather.side_effect = lambda value: [
        value,
        {
            "error": None,
            "handle": b"remote",
            "is_fabric": True,
            "pid": 1001,
        },
    ]
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


@pytest.mark.parametrize(
    ("errno_value", "expected_hint"),
    [(mnnvl.errno.EPERM, "--cap-add=SYS_PTRACE"), (mnnvl.errno.ENOSYS, "requires Linux 5.6+")],
)
def test_create_and_map_handles_preserves_pidfd_error_hint(monkeypatch, errno_value, expected_hint):
    comm = Mock()
    comm.isend = None
    comm.irecv = None
    comm.Get_rank.return_value = 0
    comm.Get_size.return_value = 2
    comm.allgather.side_effect = lambda value: (
        [
            value,
            {
                "error": None,
                "handle": 101,
                "is_fabric": False,
                "pid": 1001,
            },
        ]
        if isinstance(value, dict)
        else [value, value]
    )
    allocation_prop = SimpleNamespace(
        requestedHandleTypes=(
            mnnvl.cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        ),
        location=object(),
    )
    syscall = Mock(side_effect=[10, 11, -1])

    monkeypatch.setattr(mnnvl.MnnvlMemory, "dev_id", 0)
    monkeypatch.setattr(
        mnnvl.MnnvlMemory,
        "get_allocation_prop",
        Mock(return_value=allocation_prop),
    )
    monkeypatch.setattr(mnnvl, "_check_cu_result", lambda result: result)
    monkeypatch.setattr(mnnvl.cuda, "cuCtxGetDevice", Mock(return_value=0))
    monkeypatch.setattr(mnnvl.cuda, "cuMemCreate", Mock(return_value=11))
    monkeypatch.setattr(mnnvl.cuda, "cuMemExportToShareableHandle", Mock(return_value=100))
    monkeypatch.setattr(mnnvl.cuda, "cuMemRelease", Mock(return_value=None))
    monkeypatch.setattr(mnnvl.ctypes, "CDLL", Mock(return_value=SimpleNamespace(syscall=syscall)))
    monkeypatch.setattr(mnnvl.ctypes, "get_errno", Mock(return_value=errno_value))
    close_fd = Mock()
    monkeypatch.setattr(mnnvl.os, "close", close_fd)

    with pytest.raises(RuntimeError, match=expected_hint):
        _TestMnnvlMemory._create_and_map_handles(comm, 64, 1000, 256, 32)

    assert comm.allgather.call_count == 2
    assert [call.args[0] for call in close_fd.call_args_list] == [10, 11, 100]


def test_create_and_map_handles_close_failure_does_not_mask_original_error(monkeypatch):
    comm = Mock()
    comm.isend = None
    comm.irecv = None
    comm.Get_rank.return_value = 0
    comm.Get_size.return_value = 2
    comm.allgather.side_effect = lambda value: (
        [
            value,
            {
                "error": None,
                "handle": 101,
                "is_fabric": False,
                "pid": 1001,
            },
        ]
        if isinstance(value, dict)
        else [value, value]
    )
    allocation_prop = SimpleNamespace(
        requestedHandleTypes=(
            mnnvl.cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        ),
        location=object(),
    )
    access_desc = SimpleNamespace(location=None, flags=None)
    syscall = Mock(side_effect=[10, 11, 20, 21])

    monkeypatch.setattr(mnnvl.MnnvlMemory, "dev_id", 0)
    monkeypatch.setattr(
        mnnvl.MnnvlMemory,
        "get_allocation_prop",
        Mock(return_value=allocation_prop),
    )
    monkeypatch.setattr(mnnvl, "_check_cu_result", lambda result: result)
    monkeypatch.setattr(mnnvl.cuda, "cuCtxGetDevice", Mock(return_value=0))
    monkeypatch.setattr(mnnvl.cuda, "cuMemCreate", Mock(return_value=11))
    monkeypatch.setattr(mnnvl.cuda, "cuMemExportToShareableHandle", Mock(return_value=100))
    monkeypatch.setattr(mnnvl.cuda, "CUmemAccessDesc", Mock(return_value=access_desc))
    monkeypatch.setattr(mnnvl.cuda, "cuMemMap", Mock(side_effect=RuntimeError("map failed")))
    monkeypatch.setattr(mnnvl.cuda, "cuMemRelease", Mock(return_value=None))
    monkeypatch.setattr(mnnvl.ctypes, "CDLL", Mock(return_value=SimpleNamespace(syscall=syscall)))
    close_fd = Mock(side_effect=[OSError("close failed"), None, None, None, None])
    monkeypatch.setattr(
        mnnvl,
        "os",
        SimpleNamespace(close=close_fd, getpid=mnnvl.os.getpid, strerror=mnnvl.os.strerror),
    )

    with pytest.raises(RuntimeError, match="map failed"):
        _TestMnnvlMemory._create_and_map_handles(comm, 64, 1000, 256, 32)

    assert comm.allgather.call_count == 2
    assert [call.args[0] for call in close_fd.call_args_list] == [10, 11, 20, 21, 100]


def _make_moe_alltoall_for_lifecycle():
    obj = MoeAlltoAll.__new__(MoeAlltoAll)
    obj._destroyed = True
    obj.mnnvl_mem = Mock(mapped=True)
    return obj


def test_moe_alltoall_rejects_unmapped_workspace():
    obj = _make_moe_alltoall_for_lifecycle()
    obj.mnnvl_mem.mapped = False

    with pytest.raises(RuntimeError, match="workspace handles are unmapped"):
        obj._require_mapped()


def test_two_sided_combine_requires_new_prepare_before_next_dispatch(monkeypatch):
    communication = NVLinkTwoSided.__new__(NVLinkTwoSided)
    communication._dispatch_state = {
        "alltoall_info": object(),
        "original_token_count": 1,
    }
    communication.alltoall_workspace = object()
    communication.ep_rank = 0
    communication.ep_size = 1
    communication.top_k = 1
    communication.use_low_precision_combine = False
    communication.alltoall_result_do_sum = True
    monkeypatch.setattr(mnnvl.MnnvlMoe, "require_mapped", Mock())
    monkeypatch.setattr(
        mnnvl.MnnvlMoe,
        "mnnvl_moe_alltoallv_combine",
        Mock(return_value=torch.ones(1, 1)),
    )

    communication.combine(torch.ones(1, 1))

    with pytest.raises(ValueError, match=r"requires prepare_dispatch\(\) to be called first"):
        communication.dispatch(None, None, None, None, [1])


class _FakeSplitComm:
    """Sub-communicator returned by _FakeWorld.Split()."""

    def __init__(self, size):
        self._size = size
        self.freed = False

    def Get_size(self):
        return self._size

    def Free(self):
        self.freed = True


class _FakeWorld:
    """Minimal MPI world standing in for MPI_COMM_WORLD.

    group_sizes maps a split color to the sub-communicator size MPI would report for it,
    stated by the test instead of derived from the split rule under test: a get_comm() that
    computed an unexpected color would raise here rather than quietly agree with itself.
    """

    def __init__(self, group_sizes):
        self.group_sizes = group_sizes
        self.split_calls = []

    def Split(self, color, key):
        self.split_calls.append((color, key))
        return _FakeSplitComm(self.group_sizes[color])


class _CommCacheMemory(mnnvl.MnnvlMemory):
    """Isolated comm state, courtesy of MnnvlMemory.__init_subclass__."""


def _moe_mapping(rank, *, tp_size, pp_size, moe_ep_size, world_size=4):
    return Mapping(
        world_size=world_size,
        rank=rank,
        gpus_per_node=world_size,
        tp_size=tp_size,
        pp_size=pp_size,
        moe_tp_size=1,
        moe_ep_size=moe_ep_size,
        enable_attention_dp=True,
    )


def _install_world(monkeypatch, group_sizes):
    world = _FakeWorld(group_sizes)
    monkeypatch.setattr(mnnvl, "mpi_comm", lambda: world)
    return world


class _FakeCuda:
    """Stands in for the driver module, faking only the one call this file reaches.

    new_mnnvl_memory_address() reserves virtual address space; everything else,
    _check_cu_result's CUresult lookup included, is served by the real module.
    """

    def __init__(self, real, address):
        self._real = real
        self._address = address
        self.reserved_sizes = []

    def __getattr__(self, name):
        return getattr(self._real, name)

    def cuMemAddressReserve(self, size, alignment, addr, flags):
        self.reserved_sizes.append(size)
        return (self._real.CUresult.CUDA_SUCCESS, self._address)


@pytest.fixture
def comm_cache_cls():
    def reset():
        _CommCacheMemory.comm = None
        _CommCacheMemory.comm_signature = None
        _CommCacheMemory.allocated_map = {}
        _CommCacheMemory.current_start_address = 0
        _CommCacheMemory.current_rank_stride = 0
        _CommCacheMemory.current_mem_offset = 0
        _CommCacheMemory.current_address_comm_size = 0

    reset()
    yield _CommCacheMemory
    reset()


def test_get_comm_rebuilds_when_parallel_layout_changes(comm_cache_cls, monkeypatch):
    """A reused MPI worker must not inherit the previous test's communicator.

    The CI worker pool is keyed on world size alone, so a tp4/ep4 test and a tp2/pp2/ep2
    test are served by the same process. The cached communicator's size becomes the
    leading dimension of the MoE workspace (as_torch_strided_tensor), so a stale one used
    to fail the epSize check inside the all-to-all kernel instead of being rejected here.
    """

    def polluter(rank):
        return _moe_mapping(rank, tp_size=4, pp_size=1, moe_ep_size=4)

    def victim(rank):
        return _moe_mapping(rank, tp_size=2, pp_size=2, moe_ep_size=2)

    # Rank 0 lands in the first PP group either way; that group spans the whole world
    # under tp4/pp1 and only half of it under tp2/pp2.
    _install_world(monkeypatch, {0: 4})
    stale = comm_cache_cls.get_comm(polluter(0))
    assert stale.Get_size() == polluter(0).moe_ep_size == 4

    _install_world(monkeypatch, {0: 2})
    fresh = comm_cache_cls.get_comm(victim(0))

    assert fresh is not stale
    assert fresh.Get_size() == victim(0).moe_ep_size == 2
    # Dropped, never freed. MPI_Comm_free is collective and allocated_map empties by
    # per-rank object lifetime, so no local answer to "is anyone still using it" may gate
    # the free; see drop_cached_comm().
    assert not stale.freed


def test_get_comm_reuses_communicator_within_one_layout(comm_cache_cls, monkeypatch):
    """An equal-but-distinct mapping must hit the cache: Split() is collective.

    Rebuilding unconditionally would satisfy the test above just as well, while putting a
    collective on a path whose callers expect a lookup.
    """

    def layout(rank):
        return _moe_mapping(rank, tp_size=4, pp_size=1, moe_ep_size=4)

    world = _install_world(monkeypatch, {0: 4})

    first = comm_cache_cls.get_comm(layout(0))
    second = comm_cache_cls.get_comm(layout(0))

    assert second is first
    assert len(world.split_calls) == 1


def test_comm_layout_signature_is_rank_invariant():
    """Every rank must reach the same cache verdict, or Split() deadlocks.

    If the signature embedded a per-rank field, one rank could reuse the cached
    communicator while the others blocked in the collective Split().
    """
    signatures = {
        mnnvl.MnnvlMemory.comm_layout_signature(
            _moe_mapping(rank, tp_size=2, pp_size=2, moe_ep_size=2)
        )
        for rank in range(4)
    }

    assert len(signatures) == 1


def test_allocation_keeps_its_own_layout_after_the_comm_is_rebuilt(memory, monkeypatch):
    """An allocation is described by the layout it was mapped with, not the current one.

    Rebuilding the class-level communicator lets an allocation outlive the communicator it
    was created with, so its segment count and its handle index have to keep coming from
    its own record. The hazard exists only because get_comm() can now rebuild at all.
    """
    obj, record = memory
    obj.segment_size = record.aligned_size
    obj.rank_stride = record.rank_stride

    seen = {}

    def fake_pack(ptr, segment_size, rank_stride, num_segments, dtype, dev_id):
        seen["num_segments"] = num_segments
        return "tensor"

    monkeypatch.setattr(mnnvl, "pack_strided_memory", fake_pack)
    # A communicator for a different, larger layout, as get_comm() would leave behind.
    monkeypatch.setattr(
        _TestMnnvlMemory, "comm", _FakeComm(rank=1, size=4, membership=(0, 1, 2, 3))
    )

    assert obj.as_torch_strided_tensor(torch.uint64) == "tensor"
    assert seen["num_segments"] == record.comm_size == 2
    assert obj.local_mem_handle == record.mem_handles[record.comm_rank] == 11


def test_reserved_address_range_is_not_shared_across_communicator_sizes(
    comm_cache_cls, monkeypatch
):
    """A reserved range belongs to the rank count it was reserved for.

    The range is one slot of rank_stride bytes per rank, so free bytes inside a slot do not
    make it reusable on their own. A larger communicator would map its extra ranks past the
    end of the range, and a range shared by two rank counts would be handed to
    cuMemAddressFree with a length matching neither, since close_mnnvl_memory() reads that
    length off whichever allocation closes last. Before get_comm() could rebuild, the rank
    count could not change within a process and the byte check alone was sufficient.
    """
    stride = mnnvl.MnnvlMemory.fabric_page_size
    quarter = stride // 4
    _install_world(monkeypatch, {0: 2})
    cuda_stub = _FakeCuda(mnnvl.cuda, address=1 << 40)
    monkeypatch.setattr(mnnvl, "cuda", cuda_stub)

    comm_cache_cls.new_mnnvl_memory_address(
        _moe_mapping(0, tp_size=2, pp_size=2, moe_ep_size=2), quarter
    )
    # What open_mnnvl_memory() would leave behind after placing one allocation.
    comm_cache_cls.current_mem_offset = quarter

    assert cuda_stub.reserved_sizes == [stride * 2]
    assert comm_cache_cls.current_address_comm_size == 2
    # Three quarters of the slot are free, so only the rank count separates these.
    assert comm_cache_cls._address_window_fits(quarter, 2)
    assert not comm_cache_cls._address_window_fits(quarter, 4)
    assert not comm_cache_cls._address_window_fits(quarter, 1)
    # A matching rank count does not excuse an allocation that overruns the slot.
    assert not comm_cache_cls._address_window_fits(stride, 2)
