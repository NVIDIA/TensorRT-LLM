# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import ctypes
import errno
import functools
import os
import platform
import sys
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional, Protocol, Union

import pynvml
import torch

try:
    from cuda.bindings import driver as cuda
except ImportError:
    from cuda import cuda

from ._dlpack_utils import pack_strided_memory
from ._utils import get_sm_version, mpi_comm
from .logger import logger
from .mapping import Mapping

_MNNVL_CHECKPOINT_COLLECTIVE_TIMEOUT_S = 20.0
_MNNVL_CHECKPOINT_COLLECTIVE_POLL_INTERVAL_S = 0.01
_MNNVL_CHECKPOINT_REQUEST_CLEANUP_TIMEOUT_S = 0.1
_MNNVL_CHECKPOINT_ALLGATHER_TAG = 31415
_MNNVL_CHECKPOINT_ORPHANED_REQUESTS: list[Any] = []


class MnnvlCheckpointCommunicator(Protocol):
    """Structural contract required by bounded MNNVL checkpoint collectives."""

    def Get_rank(self) -> int:
        """Return this process's rank in the communicator."""
        ...

    def Get_size(self) -> int:
        """Return the communicator size."""
        ...

    def barrier(self) -> None:
        """Synchronize all communicator members."""
        ...

    def allgather(self, value: Any) -> list[Any]:
        """Gather a Python object from every communicator member."""
        ...

    def isend(self, value: Any, *, dest: int, tag: int) -> Any:
        """Start a nonblocking Python-object send."""
        ...

    def irecv(self, *, source: int, tag: int) -> Any:
        """Start a nonblocking Python-object receive."""
        ...


def _cancel_checkpoint_requests(requests: list[Any]) -> None:
    """Bound cancellation cleanup so timeout handling cannot hang in MPI."""
    for request in requests:
        try:
            request.Cancel()
        except Exception as error:
            logger.warning(f"Failed to cancel MNNVL checkpoint request: {error}")

    pending_requests = list(requests)
    cleanup_timeout_s = min(
        _MNNVL_CHECKPOINT_REQUEST_CLEANUP_TIMEOUT_S,
        _MNNVL_CHECKPOINT_COLLECTIVE_TIMEOUT_S,
    )
    deadline = time.monotonic() + cleanup_timeout_s
    while pending_requests:
        incomplete_requests = []
        for request in pending_requests:
            try:
                ready, _ = request.test()
            except Exception as error:
                logger.warning(f"Failed to poll MNNVL checkpoint request cleanup: {error}")
                incomplete_requests.append(request)
                continue
            if not ready:
                incomplete_requests.append(request)
        pending_requests = incomplete_requests
        if not pending_requests or time.monotonic() >= deadline:
            break
        time.sleep(_MNNVL_CHECKPOINT_COLLECTIVE_POLL_INTERVAL_S)

    if pending_requests:
        # An active receive must not be freed: mpi4py owns its receive buffer,
        # and MPI may still write into it. Keep the wrappers alive until the
        # enclosing fail-closed path terminates the worker.
        _MNNVL_CHECKPOINT_ORPHANED_REQUESTS.extend(pending_requests)
        logger.error(
            f"Retaining {len(pending_requests)} active MNNVL checkpoint requests "
            "until worker termination"
        )


def _checkpoint_allgather(
    comm: MnnvlCheckpointCommunicator,
    value: Any,
    *,
    operation: str,
) -> list[Any]:
    """Run a bounded object allgather over nonblocking point-to-point requests.

    Production mpi4py communicators provide ``isend`` and ``irecv`` but no
    nonblocking object allgather. The blocking fallback is retained only for
    lightweight test communicators that do not implement point-to-point APIs.
    """
    isend = getattr(comm, "isend", None)
    irecv = getattr(comm, "irecv", None)
    if isend is None or irecv is None:
        return comm.allgather(value)

    rank = comm.Get_rank()
    size = comm.Get_size()
    results: list[Any] = [None] * size
    results[rank] = value
    receive_requests: dict[int, Any] = {}
    send_requests: list[Any] = []
    try:
        for peer in range(size):
            if peer != rank:
                receive_requests[peer] = irecv(
                    source=peer,
                    tag=_MNNVL_CHECKPOINT_ALLGATHER_TAG,
                )
        for peer in range(size):
            if peer != rank:
                send_requests.append(
                    isend(
                        value,
                        dest=peer,
                        tag=_MNNVL_CHECKPOINT_ALLGATHER_TAG,
                    )
                )
        if not receive_requests:
            return results
        deadline = time.monotonic() + _MNNVL_CHECKPOINT_COLLECTIVE_TIMEOUT_S
        while receive_requests or send_requests:
            for peer, request in list(receive_requests.items()):
                ready, result = request.test()
                if ready:
                    results[peer] = result
                    del receive_requests[peer]
            send_requests = [request for request in send_requests if not request.test()[0]]
            if not receive_requests and not send_requests:
                return results
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for MNNVL checkpoint {operation} allgather")
            time.sleep(_MNNVL_CHECKPOINT_COLLECTIVE_POLL_INTERVAL_S)
        return results
    except Exception:
        _cancel_checkpoint_requests([*receive_requests.values(), *send_requests])
        raise


def _check_cu_result(cu_func_ret):
    if isinstance(cu_func_ret, tuple):
        cu_result, *others = cu_func_ret
        if cu_result != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(cu_result)
        if len(others) == 1:
            return others[0]
        elif len(others) > 1:
            return tuple(others)
        else:
            return None
    else:
        if cu_func_ret != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(cu_func_ret)
        return None


class _MnnvlAllocationState(Enum):
    MAPPED = "mapped"
    PREPARING = "preparing"
    UNMAPPED = "unmapped"
    RESTORING = "restoring"
    BROKEN = "broken"


@dataclass
class _MnnvlAllocationRecord:
    comm: Any
    comm_size: int
    comm_rank: int
    comm_membership: tuple[int, ...]
    aligned_size: int
    mem_handles: List[Any]
    start_address: int
    rank_stride: int
    address_offset: int
    state: _MnnvlAllocationState = _MnnvlAllocationState.MAPPED
    pending_comm: Any = None


class MnnvlMemory:
    """MNNVL memory management for tensor parallel (TP) operations."""

    # Shared across all subclasses (global/device state).
    initialized: bool = False
    allocation_granularity: int = 0
    fabric_page_size: int = 1 << 29  # 512 MB.
    dev_id: int = None

    # Per-class state attributes. These will be auto-initialized for each subclass
    # to avoid polluting the parent class's state. Use callable (e.g., dict) for mutable defaults.
    _per_class_attrs = {
        "current_mem_offset": 0,
        "current_rank_stride": 0,  # stride for ranks and also address space size.
        "current_start_address": 0,
        "comm": None,  # MPI communicator.
        "allocated_map": dict,  # callable for fresh dict.
        "address_refcnt": dict,  # callable for fresh dict.
    }

    # Initialize per-class state for the base class.
    current_mem_offset: int = 0
    current_rank_stride: int = 0
    current_start_address: int = 0
    comm = None
    allocated_map = {}
    address_refcnt = {}

    def __init_subclass__(cls, **kwargs):
        """Auto-initialize per-class attributes for each subclass to avoid sharing state with parent."""
        super().__init_subclass__(**kwargs)
        for attr, default in cls._per_class_attrs.items():
            if callable(default):
                setattr(cls, attr, default())  # e.g., dict() creates a fresh dict.
            else:
                setattr(cls, attr, default)

    def __init__(self, mapping: Mapping, size: int):
        self.mapping = mapping
        self.segment_size = size
        self.ptr, self.rank_stride = type(self).open_mnnvl_memory(self.mapping, size)

    def __del__(self):
        if not sys.is_finalizing():
            if hasattr(self, "ptr"):
                type(self).close_mnnvl_memory(self.ptr)

    @property
    def mapped(self) -> bool:
        """Whether the allocation is mapped and ready for data-path access."""
        return type(self).allocated_map[self.ptr].state is _MnnvlAllocationState.MAPPED

    def as_torch_strided_tensor(self, dtype):
        num_segments = type(self).comm.Get_size()
        return pack_strided_memory(
            self.ptr, self.segment_size, self.rank_stride, num_segments, dtype, MnnvlMemory.dev_id
        )

    @staticmethod
    def initialize():
        if not MnnvlMemory.initialized:
            # use a dummy torch CUDA tensor to trigger CUDA context initialization
            _ = torch.empty(1, device="cuda")
            MnnvlMemory._ensure_nvml_initialized()
            MnnvlMemory.initialized = True

    @staticmethod
    def _ensure_nvml_initialized() -> None:
        """Initialize NVML when it has not already been initialized."""
        try:
            pynvml.nvmlDeviceGetCount()
        except pynvml.NVMLError_Uninitialized:
            pynvml.nvmlInit()

    @classmethod
    def get_comm(cls, mapping: Mapping):
        """Get TP-based communicator (ranks grouped by PP+CP+MOE_TP, ordered by TP rank)."""
        if cls.comm is not None:
            return cls.comm
        comm = mpi_comm().Split(
            (mapping.pp_rank * mapping.cp_size + mapping.cp_rank) * mapping.moe_tp_size
            + mapping.moe_tp_rank,
            mapping.tp_rank,
        )
        cls.comm = comm
        return comm

    @staticmethod
    def get_allocation_prop(dev_id: int):
        location = cuda.CUmemLocation()
        location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        location.id = dev_id
        allocation_prop = cuda.CUmemAllocationProp()
        allocation_prop.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED

        # TODO: We differentiate FABRIC for GB200 (aarch64) and POSIX_FILE_DESCRIPTOR for BB200 (x86_64).
        # May need to find a better way to handle this.
        arch = platform.machine().lower()
        is_on_aarch64 = "aarch64" in arch
        if is_on_aarch64:
            allocation_prop.requestedHandleTypes = (
                cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
            )
        else:
            allocation_prop.requestedHandleTypes = (
                cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
            )
        allocation_prop.location = location
        return allocation_prop

    @staticmethod
    def get_allocation_granularity(dev_id: int):
        if MnnvlMemory.allocation_granularity != 0:
            return MnnvlMemory.allocation_granularity
        allocation_prop = MnnvlMemory.get_allocation_prop(dev_id)
        option = cuda.CUmemAllocationGranularity_flags(
            cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
        )
        granularity = _check_cu_result(
            cuda.cuMemGetAllocationGranularity(prop=allocation_prop, option=option)
        )
        MnnvlMemory.allocation_granularity = granularity
        return MnnvlMemory.allocation_granularity

    @classmethod
    def new_mnnvl_memory_address(cls, mapping: Mapping, size: int):
        page_count = (size + MnnvlMemory.fabric_page_size - 1) // MnnvlMemory.fabric_page_size
        current_rank_stride = page_count * MnnvlMemory.fabric_page_size
        logger.info(f"[{cls.__name__}] creating address with stride={current_rank_stride}")
        comm = cls.get_comm(mapping)
        comm_size = comm.Get_size()
        address_size = current_rank_stride * comm_size
        ptr = _check_cu_result(
            cuda.cuMemAddressReserve(address_size, MnnvlMemory.fabric_page_size, 0, 0)
        )
        cls.current_start_address = int(ptr)
        cls.current_rank_stride = current_rank_stride
        cls.current_mem_offset = 0

    @classmethod
    def _create_and_map_handles(
        cls,
        comm,
        aligned_size: int,
        start_address: int,
        rank_stride: int,
        address_offset: int,
    ) -> List[Any]:
        local_handle = None
        exported_handle = None
        pidfds = []
        remote_fds = []
        mem_handles = [None] * comm.Get_size()
        mapped_rank_ptrs = []
        is_fabric = False
        try:
            local_error = None
            local_handle_data = None
            local_pid = None
            try:
                dev_id = int(_check_cu_result(cuda.cuCtxGetDevice()))
                assert dev_id == MnnvlMemory.dev_id, (
                    f"Different dev_id found dev_id={dev_id} but "
                    f"MnnvlMemory.dev_id={MnnvlMemory.dev_id}"
                )
                allocation_prop = MnnvlMemory.get_allocation_prop(dev_id)
                is_fabric = (
                    allocation_prop.requestedHandleTypes
                    == cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
                )
                local_handle = _check_cu_result(
                    cuda.cuMemCreate(aligned_size, allocation_prop, flags=0)
                )
                exported_handle = _check_cu_result(
                    cuda.cuMemExportToShareableHandle(
                        local_handle, allocation_prop.requestedHandleTypes, 0
                    )
                )
                local_handle_data = exported_handle.data if is_fabric else int(exported_handle)
                local_pid = os.getpid()
            except Exception as error:
                local_error = f"{type(error).__name__}: {error}"

            exported_by_rank = _checkpoint_allgather(
                comm,
                {
                    "error": local_error,
                    "handle": local_handle_data,
                    "is_fabric": is_fabric,
                    "pid": local_pid,
                },
                operation="handle export",
            )
            export_errors = [
                f"rank {rank}: {payload['error']}"
                for rank, payload in enumerate(exported_by_rank)
                if payload["error"] is not None
            ]
            if export_errors:
                raise RuntimeError(
                    "MNNVL handle export failed before mapping:\n" + "\n".join(export_errors)
                )
            fabric_modes = {payload["is_fabric"] for payload in exported_by_rank}
            if len(fabric_modes) != 1:
                raise RuntimeError("MNNVL ranks selected inconsistent shareable handle types")
            is_fabric = fabric_modes.pop()
            if is_fabric:
                all_handles_data = [payload["handle"] for payload in exported_by_rank]
            else:
                all_exported_fds = [payload["handle"] for payload in exported_by_rank]
                all_pids = [payload["pid"] for payload in exported_by_rank]
                syscall = ctypes.CDLL(None, use_errno=True).syscall
                fd_import_error = None
                try:
                    for pid in all_pids:
                        pidfd = syscall(434, pid, 0)
                        if pidfd < 0:
                            err = ctypes.get_errno()
                            raise RuntimeError(
                                f"pidfd_open({pid}) failed with errno {err}: {os.strerror(err)}"
                            )
                        pidfds.append(pidfd)
                    for pidfd, fd in zip(pidfds, all_exported_fds):
                        remote_fd = syscall(438, pidfd, fd, 0)
                        if remote_fd < 0:
                            err = ctypes.get_errno()
                            error_msg = (
                                f"pidfd_getfd(pidfd={pidfd}, fd={fd}) failed with errno "
                                f"{err}: {os.strerror(err)}."
                            )
                            if err == errno.EPERM:
                                error_msg += (
                                    " Permission denied. If running in a container, try adding "
                                    "--cap-add=SYS_PTRACE to your docker run command."
                                )
                            elif err == errno.ENOSYS:
                                error_msg += (
                                    " This may be due to kernel version (requires Linux 5.6+)."
                                )
                            raise RuntimeError(error_msg)
                        remote_fds.append(remote_fd)
                except Exception as error:
                    fd_import_error = f"{type(error).__name__}: {error}"

                fd_import_errors = _checkpoint_allgather(
                    comm,
                    fd_import_error,
                    operation="POSIX file descriptor import readiness",
                )
                failed_ranks = [
                    f"rank {rank}: {error}"
                    for rank, error in enumerate(fd_import_errors)
                    if error is not None
                ]
                if failed_ranks:
                    raise RuntimeError(
                        "MNNVL POSIX file descriptor import failed on one or more ranks:\n"
                        + "\n".join(failed_ranks)
                    )
                all_handles_data = remote_fds

            access_desc = cuda.CUmemAccessDesc()
            access_desc.location = allocation_prop.location
            access_desc.flags = cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
            for rank, handle_data in enumerate(all_handles_data):
                rank_ptr = start_address + rank_stride * rank + address_offset
                handle = (
                    local_handle
                    if rank == comm.Get_rank()
                    else _check_cu_result(
                        cuda.cuMemImportFromShareableHandle(
                            handle_data, allocation_prop.requestedHandleTypes
                        )
                    )
                )
                mem_handles[rank] = handle
                _check_cu_result(cuda.cuMemMap(rank_ptr, aligned_size, 0, handle, 0))
                mapped_rank_ptrs.append(rank_ptr)
                _check_cu_result(cuda.cuMemSetAccess(rank_ptr, aligned_size, [access_desc], 1))
            return mem_handles
        except Exception:
            for rank_ptr in reversed(mapped_rank_ptrs):
                try:
                    _check_cu_result(cuda.cuMemUnmap(rank_ptr, aligned_size))
                except RuntimeError as error:
                    logger.warning(f"Failed to unmap incomplete MNNVL allocation: {error}")

            handles_to_release = [handle for handle in mem_handles if handle is not None]
            if local_handle is not None and mem_handles[comm.Get_rank()] is None:
                handles_to_release.append(local_handle)
            for handle in handles_to_release:
                try:
                    _check_cu_result(cuda.cuMemRelease(handle))
                except RuntimeError as error:
                    logger.warning(f"Failed to release incomplete MNNVL allocation: {error}")
            raise
        finally:
            for pidfd in pidfds:
                try:
                    os.close(pidfd)
                except OSError as error:
                    logger.warning(f"Failed to close MNNVL pidfd: {error}")
            for remote_fd in remote_fds:
                try:
                    os.close(remote_fd)
                except OSError as error:
                    logger.warning(f"Failed to close imported MNNVL file descriptor: {error}")
            if not is_fabric and exported_handle is not None:
                try:
                    os.close(int(exported_handle))
                except OSError as error:
                    logger.warning(f"Failed to close exported MNNVL file descriptor: {error}")

    @classmethod
    def open_mnnvl_memory(cls, mapping: Mapping, size: int):
        # Ensure MnnvlMemory is initialized (for dev_id and allocation_granularity)
        MnnvlMemory.initialize()

        dev = _check_cu_result(cuda.cuCtxGetDevice())
        dev_id = int(dev)
        if MnnvlMemory.dev_id is None:
            MnnvlMemory.dev_id = dev_id
        assert dev_id == MnnvlMemory.dev_id, (
            f"Different dev_id found dev_id={dev_id} but MnnvlMemory.dev_id={MnnvlMemory.dev_id}"
        )
        comm = cls.get_comm(mapping)
        comm_rank = comm.Get_rank()
        comm_size = comm.Get_size()
        comm_membership = tuple(int(rank) for rank in comm.allgather(mapping.rank))
        if len(comm_membership) != comm_size:
            raise RuntimeError(
                "MNNVL communicator membership size does not match its rank count: "
                f"{len(comm_membership)} != {comm_size}"
            )
        all_rank_allocate_sizes = comm.allgather(size)
        assert len(all_rank_allocate_sizes) == comm_size
        assert all(x == size for x in all_rank_allocate_sizes), "Not all rank allocating same size."
        granularity = MnnvlMemory.get_allocation_granularity(dev_id)
        aligned_size = (size + granularity - 1) // granularity * granularity

        previous_address_state = (
            cls.current_start_address,
            cls.current_rank_stride,
            cls.current_mem_offset,
        )
        reserved_new_address = cls.current_mem_offset + aligned_size > cls.current_rank_stride
        if reserved_new_address:
            cls.new_mnnvl_memory_address(mapping, aligned_size)

        assert cls.current_mem_offset + aligned_size <= cls.current_rank_stride

        try:
            mem_handles = cls._create_and_map_handles(
                comm,
                aligned_size,
                cls.current_start_address,
                cls.current_rank_stride,
                cls.current_mem_offset,
            )
        except Exception:
            if reserved_new_address:
                try:
                    device_ptr = cuda.CUdeviceptr(cls.current_start_address)
                    _check_cu_result(
                        cuda.cuMemAddressFree(device_ptr, comm_size * cls.current_rank_stride)
                    )
                except RuntimeError as error:
                    logger.warning(f"cuMemAddressFree failed during error cleanup: {error}")
                else:
                    (
                        cls.current_start_address,
                        cls.current_rank_stride,
                        cls.current_mem_offset,
                    ) = previous_address_state
            raise
        ptr = cls.current_start_address + cls.current_mem_offset
        stride = cls.current_rank_stride
        cls.allocated_map[ptr] = _MnnvlAllocationRecord(
            comm=comm,
            comm_size=comm_size,
            comm_rank=comm_rank,
            comm_membership=comm_membership,
            aligned_size=aligned_size,
            mem_handles=mem_handles,
            start_address=cls.current_start_address,
            rank_stride=cls.current_rank_stride,
            address_offset=cls.current_mem_offset,
        )
        cls.address_refcnt[cls.current_start_address] = (
            cls.address_refcnt.get(cls.current_start_address, 0) + 1
        )

        cls.current_mem_offset += aligned_size
        return ptr, stride

    @classmethod
    def close_mnnvl_memory(cls, ptr: int):
        record = cls.allocated_map[ptr]
        if record.state not in (
            _MnnvlAllocationState.MAPPED,
            _MnnvlAllocationState.UNMAPPED,
        ):
            logger.warning(
                f"Skipping cleanup of MNNVL allocation in terminal state {record.state.value}"
            )
            return
        cls.allocated_map.pop(ptr)
        if record.state is _MnnvlAllocationState.MAPPED:
            cls._unmap_and_release_handles(record)
        cls.address_refcnt[record.start_address] -= 1

        if cls.address_refcnt[record.start_address] == 0:
            cls.address_refcnt.pop(record.start_address)
            device_ptr = cuda.CUdeviceptr(record.start_address)
            _check_cu_result(
                cuda.cuMemAddressFree(device_ptr, record.comm_size * record.rank_stride)
            )
            if record.start_address == cls.current_start_address:
                cls.current_start_address = 0
                cls.current_rank_stride = 0
                cls.current_mem_offset = 0

    @classmethod
    def _unmap_and_release_handles(cls, record: _MnnvlAllocationRecord) -> None:
        first_error = None
        for rank in range(record.comm_size):
            rank_ptr = record.start_address + rank * record.rank_stride + record.address_offset
            try:
                _check_cu_result(cuda.cuMemUnmap(rank_ptr, record.aligned_size))
            except RuntimeError as error:
                if first_error is None:
                    first_error = error
                continue
            try:
                _check_cu_result(cuda.cuMemRelease(record.mem_handles[rank]))
            except RuntimeError as error:
                if first_error is None:
                    first_error = error
            else:
                record.mem_handles[rank] = None
        if first_error is not None:
            raise first_error

    def checkpoint_prepare(self) -> None:
        """Detach local backing handles while retaining graph-visible VA.

        The engine checkpoint coordinator is responsible for quiescing and
        invoking this operation on every rank. No collective follows the CUDA
        mutation, so a local failure cannot strand peers in a trailing barrier.
        """
        cls = type(self)
        record = cls.allocated_map[self.ptr]
        if record.state is _MnnvlAllocationState.UNMAPPED:
            return
        if record.state is not _MnnvlAllocationState.MAPPED:
            raise RuntimeError(f"Cannot prepare MNNVL allocation in {record.state.value} state")
        record.state = _MnnvlAllocationState.PREPARING
        try:
            torch.cuda.synchronize()
            cls._unmap_and_release_handles(record)
            record.mem_handles = [None] * record.comm_size
        except Exception:
            record.state = _MnnvlAllocationState.BROKEN
            raise
        record.state = _MnnvlAllocationState.UNMAPPED

    def checkpoint_fail_closed(self) -> None:
        """Make a timed-out checkpoint allocation terminal.

        A timeout means ranks may have reached different checkpoint phases.
        The allocation must not be reused by a later checkpoint attempt.
        """
        record = type(self).allocated_map[self.ptr]
        record.state = _MnnvlAllocationState.BROKEN
        record.pending_comm = None

    def checkpoint_restore(self, comm: MnnvlCheckpointCommunicator) -> bool:
        """Remap fresh handles while keeping data-path access disabled."""
        cls = type(self)
        record = cls.allocated_map[self.ptr]
        if record.state is _MnnvlAllocationState.MAPPED:
            return False
        if record.state is not _MnnvlAllocationState.UNMAPPED:
            raise RuntimeError(f"Cannot restore MNNVL allocation in {record.state.value} state")
        comm_size = comm.Get_size()
        comm_rank = comm.Get_rank()
        if comm_size != record.comm_size or comm_rank != record.comm_rank:
            raise RuntimeError(
                "Cannot restore MNNVL memory with a communicator that differs from "
                "the graph-visible allocation layout: "
                f"rank/size {comm_rank}/{comm_size} != "
                f"{record.comm_rank}/{record.comm_size}"
            )
        try:
            comm_membership = tuple(
                int(rank)
                for rank in _checkpoint_allgather(
                    comm,
                    self.mapping.rank,
                    operation="communicator membership",
                )
            )
        except Exception:
            self.checkpoint_fail_closed()
            raise
        if comm_membership != record.comm_membership:
            raise RuntimeError(
                "Cannot restore MNNVL memory with a communicator whose ordered "
                "membership differs from the graph-visible allocation layout: "
                f"{comm_membership} != {record.comm_membership}"
            )
        record.state = _MnnvlAllocationState.RESTORING
        local_error = None
        try:
            torch.cuda.synchronize()
            record.mem_handles = cls._create_and_map_handles(
                comm,
                record.aligned_size,
                record.start_address,
                record.rank_stride,
                record.address_offset,
            )
        except TimeoutError:
            record.state = _MnnvlAllocationState.BROKEN
            raise
        except Exception as error:
            local_error = f"{type(error).__name__}: {error}"

        try:
            restore_errors = _checkpoint_allgather(
                comm,
                local_error,
                operation="mapping readiness",
            )
        except Exception:
            self._checkpoint_restore_failed()
            raise
        failed_ranks = [
            f"rank {rank}: {error}"
            for rank, error in enumerate(restore_errors)
            if error is not None
        ]
        if failed_ranks:
            self._checkpoint_restore_failed()
            raise RuntimeError(
                "MNNVL checkpoint restore failed on one or more ranks:\n" + "\n".join(failed_ranks)
            )
        record.pending_comm = comm
        return True

    def _checkpoint_restore_complete(self) -> None:
        """Publish a restored allocation after frontend protocol readiness."""
        record = type(self).allocated_map[self.ptr]
        if record.state is not _MnnvlAllocationState.RESTORING:
            raise RuntimeError(f"Cannot complete MNNVL restore in {record.state.value} state")
        if record.pending_comm is None:
            raise RuntimeError("Cannot complete MNNVL restore without a replacement communicator")
        record.comm = record.pending_comm
        type(self).comm = record.pending_comm
        record.pending_comm = None
        record.state = _MnnvlAllocationState.MAPPED

    def _checkpoint_restore_failed(self) -> None:
        """Make a failed frontend restore terminal and fail closed."""
        record = type(self).allocated_map[self.ptr]
        if record.state is _MnnvlAllocationState.RESTORING:
            for rank, handle in enumerate(record.mem_handles):
                if handle is None:
                    continue
                rank_ptr = record.start_address + rank * record.rank_stride + record.address_offset
                try:
                    _check_cu_result(cuda.cuMemUnmap(rank_ptr, record.aligned_size))
                except RuntimeError as error:
                    logger.warning(
                        f"Failed to unmap unpublished MNNVL restore for rank {rank}: {error}"
                    )
                try:
                    _check_cu_result(cuda.cuMemRelease(handle))
                except RuntimeError as error:
                    logger.warning(
                        "Failed to release unpublished MNNVL restore handle "
                        f"for rank {rank}: {error}"
                    )
                else:
                    record.mem_handles[rank] = None
            record.state = _MnnvlAllocationState.BROKEN
            record.pending_comm = None

    @staticmethod
    @functools.cache
    def support_nvlink(dev_id: int, need_all_up: bool = True):
        # Do not rely on other modules having initialized NVML as an import side effect.
        MnnvlMemory._ensure_nvml_initialized()
        handle = pynvml.nvmlDeviceGetHandleByIndex(dev_id)
        link_count = pynvml.NVML_NVLINK_MAX_LINKS
        active_links = 0
        available_links = 0
        probed_links = link_count
        for link_idx in range(link_count):
            try:
                if pynvml.nvmlDeviceGetNvLinkCapability(
                    handle, link_idx, pynvml.NVML_NVLINK_CAP_P2P_SUPPORTED
                ):
                    available_links += 1
                    is_active = pynvml.nvmlDeviceGetNvLinkState(handle, link_idx)
                    if is_active:
                        active_links += 1
            except (pynvml.NVMLError_NotSupported, pynvml.NVMLError_InvalidArgument):
                continue
            except pynvml.NVMLError_InvalidArgument:
                # NVML_NVLINK_MAX_LINKS (36) is an upper bound over all architectures;
                # the driver rejects indices past this GPU's link count (18 on GB200).
                probed_links = link_idx
                break
        supported = (
            active_links == available_links and available_links > 0
            if need_all_up
            else available_links > 0
        )
        logger.info(
            f"[MnnvlMemory] dev {dev_id} NVLink: {active_links}/{available_links} links up "
            f"({probed_links} of {link_count} link indices accepted by the driver), "
            f"need_all_up={need_all_up}, supported={supported}"
        )
        return supported

    @staticmethod
    @functools.cache
    def _is_pcie_nvl_sku(dev_id: int) -> bool:
        """Return whether visible H100/H200 GPUs form PCIe-connected NVLink islands."""
        # H100/H200 NVL PCIe SKUs bond GPUs into local NVLink islands joined
        # only through PCIe/SYS. Per-device NVLink state therefore cannot
        # distinguish them from an NVSwitch fabric.
        device_name = torch.cuda.get_device_name(dev_id).upper()
        # NVML may report SYSTEM between peers on later NVSwitch platforms, so
        # use this fallback only for the affected Hopper SKUs.
        if not any(sku in device_name for sku in ("H100", "H200")):
            return False

        if " NVL" in device_name:
            return True

        try:
            MnnvlMemory._ensure_nvml_initialized()
            self_handle = pynvml.nvmlDeviceGetHandleByIndex(dev_id)
            for peer_id in range(pynvml.nvmlDeviceGetCount()):
                if peer_id == dev_id:
                    continue
                peer_handle = pynvml.nvmlDeviceGetHandleByIndex(peer_id)
                if (
                    pynvml.nvmlDeviceGetTopologyCommonAncestor(self_handle, peer_handle)
                    == pynvml.NVML_TOPOLOGY_SYSTEM
                ):
                    # SYSTEM is only a distance classification. A dual-socket
                    # HGX can still provide NVLink P2P to such a peer through
                    # NVSwitch. Split islands instead have local NVLink but no
                    # NVLink P2P path to the SYSTEM peer.
                    p2p_status = pynvml.nvmlDeviceGetP2PStatus(
                        self_handle,
                        peer_handle,
                        pynvml.NVML_P2P_CAPS_INDEX_NVLINK,
                    )
                    if p2p_status != pynvml.NVML_P2P_STATUS_OK:
                        return MnnvlMemory.support_nvlink(dev_id, need_all_up=False)
        except pynvml.NVMLError:
            return False
        return False

    @staticmethod
    def supports_mnnvl() -> bool:
        # TODO:
        # We check if it has all NVLink up now.
        # But it is not equivalent to MNNVL support.
        # May need better support check.
        # SM120/121 (RTX PRO 6000 Blackwell) lack NVSwitch fabric; MNNVL-class
        # all-to-all kernels deadlock there even when local NVLink bridges
        # report up.
        if get_sm_version() in (120, 121):
            return False
        dev_id = torch.cuda.current_device()
        if MnnvlMemory._is_pcie_nvl_sku(dev_id):
            return False
        support_nvlink_and_all_up = MnnvlMemory.support_nvlink(dev_id, True)
        return support_nvlink_and_all_up


class HelixCpMnnvlMemory(MnnvlMemory):
    """MNNVL memory management for Helix context parallel (CP) operations.

    Per-class state (current_mem_offset, comm, allocated_map, etc.) is automatically
    initialized via __init_subclass__ in the parent class, ensuring this class has
    its own isolated state separate from MnnvlMemory.
    """

    @classmethod
    def get_comm(cls, mapping: Mapping):
        """Get CP-based communicator (ranks grouped by PP+TP+MOE_TP, ordered by CP rank)."""
        if cls.comm is not None:
            return cls.comm
        comm = mpi_comm().Split(
            mapping.pp_rank * mapping.tp_size + mapping.tp_rank,
            mapping.cp_rank,
        )
        cls.comm = comm
        return comm


def init_helix_cp_comm(mapping: Mapping) -> None:
    """Pre-initialize the Helix CP communicator.

    This function MUST be called during model initialization when all ranks
    are synchronized (before any PP pipeline divergence). The MPI Split operation
    is collective and requires all ranks in the communicator to participate.

    In PP (pipeline parallel) mode, different PP stages execute different parts
    of the model at different times. If the communicator is initialized lazily
    during the first forward pass, ranks in different PP stages may not reach
    the Split operation at the same time, causing a deadlock.

    Args:
        mapping: The mapping object containing parallelism configuration.
    """
    if mapping.has_cp_helix() and not mapping.cp_config.get("use_nccl_for_alltoall", True):
        HelixCpMnnvlMemory.get_comm(mapping)


@dataclass
class MoEAlltoallInfo:
    local_gather_indices: torch.Tensor
    send_rank_count_cumsum: torch.Tensor
    send_rank_local_indices: torch.Tensor
    recv_rank_count_cumsum: torch.Tensor
    recv_rank_local_indices: torch.Tensor
    backward_recv_rank_local_indices: torch.Tensor
    local_token_allocation_count: int


class MnnvlMoe:
    moe_workspace: MnnvlMemory = None
    moe_prepare_workspace: MnnvlMemory = None
    moe_workspace_tensor: torch.Tensor = None
    moe_prepare_workspace_tensor: torch.Tensor = None
    moe_mapping: Mapping = None

    @staticmethod
    def get_moe_workspaces(mapping: Mapping):
        if MnnvlMoe.moe_workspace is not None:
            assert mapping == MnnvlMoe.moe_mapping, "only one moe mapping supported now"
            return MnnvlMoe.moe_workspace_tensor

        MnnvlMoe.moe_mapping = mapping
        workspace_size_per_rank = torch.ops.trtllm.get_moe_commworkspace_size_per_rank(
            mapping.moe_ep_size
        )
        MnnvlMoe.moe_workspace = MnnvlMemory(mapping, workspace_size_per_rank)
        MnnvlMoe.moe_workspace_tensor = MnnvlMoe.moe_workspace.as_torch_strided_tensor(torch.uint64)
        torch.ops.trtllm.moe_initialize_workspace(
            MnnvlMoe.moe_workspace_tensor, mapping.moe_ep_rank, mapping.moe_ep_size
        )
        torch.cuda.synchronize()
        MnnvlMoe.moe_workspace.comm.barrier()
        return MnnvlMoe.moe_workspace_tensor

    @staticmethod
    def get_moe_prepare_workspace(mapping: Mapping):
        if MnnvlMoe.moe_prepare_workspace_tensor is not None:
            assert mapping == MnnvlMoe.moe_mapping, "only one moe mapping supported now"
            return MnnvlMoe.moe_prepare_workspace_tensor
        workspace_size_per_rank = torch.ops.trtllm.get_moe_prepare_workspace_size_per_rank(
            mapping.moe_ep_size
        )
        MnnvlMoe.moe_prepare_workspace = MnnvlMemory(mapping, workspace_size_per_rank)
        MnnvlMoe.moe_prepare_workspace_tensor = (
            MnnvlMoe.moe_prepare_workspace.as_torch_strided_tensor(torch.uint64)
        )
        return MnnvlMoe.moe_prepare_workspace_tensor

    @staticmethod
    def checkpoint_prepare() -> None:
        """Detach TRT-native two-sided MoE workspaces for checkpointing."""
        for workspace in (MnnvlMoe.moe_workspace, MnnvlMoe.moe_prepare_workspace):
            if workspace is not None:
                workspace.checkpoint_prepare()

    @staticmethod
    def checkpoint_restore(comm: MnnvlCheckpointCommunicator) -> None:
        """Restore TRT-native two-sided MoE workspaces at their original virtual addresses."""
        workspaces = (MnnvlMoe.moe_workspace, MnnvlMoe.moe_prepare_workspace)
        restored_workspaces = []
        try:
            for workspace in workspaces:
                if workspace is not None and workspace.checkpoint_restore(comm):
                    restored_workspaces.append(workspace)
            if not restored_workspaces:
                return
            restored_main_workspace = any(
                workspace is MnnvlMoe.moe_workspace for workspace in restored_workspaces
            )
            local_error = None
            try:
                if restored_main_workspace and MnnvlMoe.moe_workspace_tensor is not None:
                    assert MnnvlMoe.moe_mapping is not None
                    torch.ops.trtllm.moe_initialize_workspace(
                        MnnvlMoe.moe_workspace_tensor,
                        MnnvlMoe.moe_mapping.moe_ep_rank,
                        MnnvlMoe.moe_mapping.moe_ep_size,
                    )
                torch.cuda.synchronize()
            except Exception as error:
                local_error = f"{type(error).__name__}: {error}"
            readiness_errors = _checkpoint_allgather(
                comm,
                local_error,
                operation="two-sided frontend readiness",
            )
            failed_ranks = [
                f"rank {rank}: {error}"
                for rank, error in enumerate(readiness_errors)
                if error is not None
            ]
            if failed_ranks:
                raise RuntimeError(
                    "Native two-sided MoE restore failed on one or more ranks:\n"
                    + "\n".join(failed_ranks)
                )
        except Exception:
            for workspace in restored_workspaces:
                workspace._checkpoint_restore_failed()
            raise
        for workspace in restored_workspaces:
            workspace._checkpoint_restore_complete()

    @staticmethod
    def require_mapped() -> None:
        """Reject kernel access while either native MoE workspace is detached."""
        for workspace in (MnnvlMoe.moe_workspace, MnnvlMoe.moe_prepare_workspace):
            if workspace is not None and not workspace.mapped:
                raise RuntimeError("Native MoE All-to-All workspace handles are unmapped")

    @staticmethod
    def compute_target_rank_id(
        token_selected_experts: torch.Tensor, expert_count: int, ep_size: int
    ):
        assert expert_count % ep_size == 0, "expert_count should be divisible by ep_size"
        expert_per_rank = expert_count // ep_size
        token_target_rank_ids = token_selected_experts // expert_per_rank
        return token_target_rank_ids

    @staticmethod
    def mnnvl_moe_alltoallv_prepare_without_allgather(
        expert_ids: torch.Tensor,
        expert_statics: Optional[torch.Tensor],
        workspace: torch.Tensor,
        max_token_count_per_rank: int,
        ep_rank: int,
        ep_size: int,
        expert_count: int,
        slot_count: int,
        top_k: int,
    ):
        (
            local_send_rank_count_cumsum,
            local_send_rank_indices,
            local_recv_rank_count_cumsum,
            local_recv_rank_indices,
            backward_local_recv_rank_indices,
            gathered_expert_statics,
        ) = torch.ops.trtllm.mnnvl_moe_alltoallv_prepare_without_allgather(
            expert_ids,
            expert_statics,
            workspace,
            max_token_count_per_rank,
            ep_rank,
            ep_size,
            expert_count,
            slot_count,
            top_k,
        )

        local_token_allocation_count = max_token_count_per_rank * ep_size
        # Looks like we don't need this.
        local_gather_indices = None

        alltoall_info = MoEAlltoallInfo(
            local_gather_indices,
            local_send_rank_count_cumsum,
            local_send_rank_indices,
            local_recv_rank_count_cumsum,
            local_recv_rank_indices,
            backward_local_recv_rank_indices,
            local_token_allocation_count,
        )

        return alltoall_info, gathered_expert_statics

    @staticmethod
    def mnnvl_moe_expert_static_allgather(
        expert_ids: torch.Tensor,
        workspace: torch.Tensor,
        ep_rank: int,
        ep_size: int,
        expert_count: int,
    ):
        gathered_expert_ids = torch.ops.trtllm.mnnvl_moe_expert_static_allgather(
            expert_ids, workspace, ep_rank, ep_size, expert_count
        )
        return gathered_expert_ids

    @staticmethod
    def mnnvl_moe_alltoallv_prepare(
        gathered_target_rank_ids: torch.Tensor,
        real_rank_token_count_cumsum: Optional[torch.Tensor],
        gathered_expert_ids: torch.Tensor,
        gathered_scales: Optional[torch.Tensor],
        max_token_count_per_rank: int,
        expert_count: int,
        top_k: int,
        ep_rank: int,
        ep_size: int,
    ):
        (
            local_gather_indices,
            send_rank_count_cumsum,
            send_rank_local_indices,
            recv_rank_count_cumsum,
            recv_rank_local_indices,
            backward_recv_rank_local_indices,
        ) = torch.ops.trtllm.moe_comm_prepare_indices(
            gathered_target_rank_ids,
            real_rank_token_count_cumsum,
            max_token_count_per_rank,
            expert_count,
            top_k,
            ep_rank,
            ep_size,
        )

        local_token_allocation_count = max_token_count_per_rank * ep_size

        local_expert_ids = torch.empty(
            local_token_allocation_count, top_k, dtype=torch.int32, device=torch.device("cuda")
        )
        if gathered_scales is None:
            local_scales = None
        else:
            local_scales = torch.empty(
                local_token_allocation_count,
                top_k,
                dtype=torch.float32,
                device=torch.device("cuda"),
            )

        torch.ops.trtllm.moe_local_gather(
            recv_rank_count_cumsum,
            local_gather_indices,
            gathered_expert_ids,
            gathered_scales,
            local_expert_ids,
            local_scales,
            max_token_count_per_rank,
            expert_count,
            top_k,
            ep_rank,
            ep_size,
        )

        alltoall_info = MoEAlltoallInfo(
            local_gather_indices,
            send_rank_count_cumsum,
            send_rank_local_indices,
            recv_rank_count_cumsum,
            recv_rank_local_indices,
            backward_recv_rank_local_indices,
            local_token_allocation_count,
        )
        return alltoall_info, local_expert_ids, local_scales

    @staticmethod
    def mnnvl_moe_alltoallv(
        x: Union[torch.Tensor, List[Optional[torch.Tensor]]],
        alltoall_info: MoEAlltoallInfo,
        workspace: torch.Tensor,
        ep_rank: int,
        ep_size: int,
    ) -> Union[torch.Tensor, List[Optional[torch.Tensor]]]:
        # Convert single tensor to list for unified handling
        is_single_tensor = not isinstance(x, list)
        if is_single_tensor:
            assert x.dim() == 2, "only 2D tensor supported, please reshape."
            x = [x]

        assert len(x) > 0, "Empty tensor list not supported"

        # Filter out None values
        valid_list = [tensor is not None for tensor in x]
        valid_tensors = [tensor for tensor in x if tensor is not None]

        if len(valid_tensors) == 0:
            # All tensors are None, return list of None
            result = [None] * len(x)
        else:
            first_dim = None
            for tensor in valid_tensors:
                # Validate dimensions of valid tensors
                assert tensor.dim() == 2, "only 2D tensor supported, please reshape."
                if first_dim is None:
                    first_dim = tensor.shape[0]
                else:
                    assert tensor.shape[0] == first_dim, (
                        f"All tensors must have the same first dimension, got {tensor.shape[0]} vs {first_dim}"
                    )

            # Process only valid tensors
            output_tensors = torch.ops.trtllm.moe_comm(
                valid_tensors,
                alltoall_info.send_rank_count_cumsum,
                alltoall_info.send_rank_local_indices,
                alltoall_info.recv_rank_count_cumsum,
                alltoall_info.recv_rank_local_indices,
                workspace,
                alltoall_info.local_token_allocation_count,
                ep_rank,
                ep_size,
            )

            # Restore None positions in output
            idx = 0
            result = []
            for is_valid in valid_list:
                if is_valid:
                    result.append(output_tensors[idx])
                    idx += 1
                else:
                    result.append(None)

        # If input was a single tensor, return a single tensor
        if is_single_tensor:
            result = result[0]

        return result

    @staticmethod
    def mnnvl_moe_alltoallv_combine(
        x: torch.Tensor,
        alltoall_info: MoEAlltoallInfo,
        workspace: torch.Tensor,
        ep_rank: int,
        ep_size: int,
        top_k: int,
        token_count: int,
        use_low_precision_combine: bool = False,
        do_reduce: bool = True,
    ):
        assert x.dim() == 2, "2D tensor supported, please reshape."
        output_tensors = torch.ops.trtllm.moe_comm(
            [x],
            alltoall_info.recv_rank_count_cumsum,
            alltoall_info.recv_rank_local_indices,
            alltoall_info.send_rank_count_cumsum,
            alltoall_info.backward_recv_rank_local_indices,
            workspace,
            token_count * top_k,
            ep_rank,
            ep_size,
            [True],
            use_low_precision_combine,
        )
        output_tensor = output_tensors[0].reshape(token_count, top_k, x.shape[1])
        if do_reduce:
            return torch.sum(output_tensor, dim=1, keepdim=False)
        else:
            return output_tensor
