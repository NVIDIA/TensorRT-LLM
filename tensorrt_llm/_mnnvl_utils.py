# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os
import platform
import sys
from dataclasses import dataclass
from typing import List, Optional, Union

import pynvml
import torch

try:
    from cuda.bindings import driver as cuda
except ImportError:
    from cuda import cuda

from ._dlpack_utils import pack_strided_memory
from ._utils import mpi_comm
from .logger import logger
from .mapping import Mapping


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


class MnnvlMemory:
    initialized: bool = False

    current_mem_offset: int = 0
    current_rank_stride: int = 0  # stride for ranks and also address space size.
    current_start_address: int = 0

    # allocation granularity
    allocation_granularity: int = 0

    # fabric address page size (512 MB)
    fabric_page_size: int = 1 << 29

    # MPI communicator
    comm = None

    dev_id: int = None

    allocated_map = {}
    address_refcnt = {}

    def __init__(self, mapping: Mapping, size: int):
        self.mapping = mapping
        self.segment_size = size
        self.ptr, self.rank_stride = MnnvlMemory.open_mnnvl_memory(self.mapping, size)

    def __del__(self):
        if not sys.is_finalizing():
            if hasattr(self, "ptr"):
                MnnvlMemory.close_mnnvl_memory(self.ptr)

    def as_torch_strided_tensor(self, dtype):
        num_segments = MnnvlMemory.comm.Get_size()
        return pack_strided_memory(
            self.ptr, self.segment_size, self.rank_stride, num_segments, dtype, MnnvlMemory.dev_id
        )

    @staticmethod
    def initialize():
        if not MnnvlMemory.initialized:
            # use a dummy torch CUDA tensor to trigger CUDA context initialization
            _ = torch.empty(1, device="cuda")
            # ensure nvml is initialized.
            try:
                pynvml.nvmlDeviceGetCount()
            except pynvml.NVMLError_Uninitialized:
                pynvml.nvmlInit()
            MnnvlMemory.initialized = True

    @staticmethod
    def get_comm(mapping: Mapping):
        if MnnvlMemory.comm is not None:
            return MnnvlMemory.comm
        comm = mpi_comm().Split(
            (mapping.pp_rank * mapping.cp_size + mapping.cp_rank) * mapping.moe_tp_size
            + mapping.moe_tp_rank,
            mapping.tp_rank,
        )
        MnnvlMemory.comm = comm
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

    @staticmethod
    def new_mnnvl_memory_address(mapping: Mapping, size: int):
        page_count = (size + MnnvlMemory.fabric_page_size - 1) // MnnvlMemory.fabric_page_size
        current_rank_stride = page_count * MnnvlMemory.fabric_page_size
        logger.info(f"[MnnvlMemory] creating address with stride={current_rank_stride}")
        comm = MnnvlMemory.get_comm(mapping)
        comm_size = comm.Get_size()
        address_size = current_rank_stride * comm_size
        ptr = _check_cu_result(
            cuda.cuMemAddressReserve(address_size, MnnvlMemory.fabric_page_size, 0, 0)
        )
        MnnvlMemory.current_start_address = int(ptr)
        MnnvlMemory.current_rank_stride = current_rank_stride
        MnnvlMemory.current_mem_offset = 0

    @staticmethod
    def open_mnnvl_memory(mapping: Mapping, size: int):
        dev = _check_cu_result(cuda.cuCtxGetDevice())
        dev_id = int(dev)
        if MnnvlMemory.dev_id is None:
            MnnvlMemory.dev_id = dev_id
        assert dev_id == MnnvlMemory.dev_id, (
            f"Different dev_id found dev_id={dev_id} but MnnvlMemory.dev_id={MnnvlMemory.dev_id}"
        )
        comm = MnnvlMemory.get_comm(mapping)
        comm_rank = comm.Get_rank()
        comm_size = comm.Get_size()
        all_rank_allocate_sizes = comm.allgather(size)
        assert len(all_rank_allocate_sizes) == comm_size
        assert all(x == size for x in all_rank_allocate_sizes), "Not all rank allocating same size."
        granularity = MnnvlMemory.get_allocation_granularity(dev_id)
        aligned_size = (size + granularity - 1) // granularity * granularity

        if MnnvlMemory.current_mem_offset + aligned_size > MnnvlMemory.current_rank_stride:
            MnnvlMemory.new_mnnvl_memory_address(mapping, aligned_size)

        assert MnnvlMemory.current_mem_offset + aligned_size <= MnnvlMemory.current_rank_stride

        allocation_prop = MnnvlMemory.get_allocation_prop(dev_id)
        allocated_mem_handle = _check_cu_result(
            cuda.cuMemCreate(aligned_size, allocation_prop, flags=0)
        )
        exported_fabric_handle = _check_cu_result(
            cuda.cuMemExportToShareableHandle(
                allocated_mem_handle, allocation_prop.requestedHandleTypes, 0
            )
        )
        if (
            allocation_prop.requestedHandleTypes
            == cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
        ):
            all_handles_data = comm.allgather(exported_fabric_handle.data)
        else:
            all_handles_data = comm.allgather(exported_fabric_handle)
            all_pids = comm.allgather(os.getpid())
            libc = ctypes.CDLL(None, use_errno=True)
            syscall = libc.syscall
            SYS_pidfd_open = 434
            SYS_pidfd_getfd = 438
            pidfds = []
            for i, pid in enumerate(all_pids):
                pidfd = syscall(SYS_pidfd_open, pid, 0)
                if pidfd < 0:
                    err = ctypes.get_errno()
                    raise RuntimeError(
                        f"pidfd_open({pid}) failed with errno {err}: {os.strerror(err)}"
                    )
                pidfds.append(pidfd)

            remote_fds = []
            for i, (pidfd, fd) in enumerate(zip(pidfds, all_handles_data)):
                remote_fd = syscall(SYS_pidfd_getfd, pidfd, fd, 0)
                if remote_fd < 0:
                    err = ctypes.get_errno()
                    error_msg = f"pidfd_getfd(pidfd={pidfd}, fd={fd}) failed with errno {err}: {os.strerror(err)}."
                    if err == 1:  # EPERM
                        error_msg += (
                            " Permission denied. If running in a container, try adding --cap-add=SYS_PTRACE "
                            "to your docker run command."
                        )
                    else:
                        error_msg += " This may be due to kernel version (requires Linux 5.6+)."
                    raise RuntimeError(error_msg)
                remote_fds.append(remote_fd)

            all_handles_data = remote_fds
        # all_handles_data like b'\x00\x00\x00 \x00\x00\x00\x00\x8f\xec\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\t\x00\x00\x00\x00\x00\x1d\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'  # noqa: E501
        # can use buf = memoryview(data) to import if using plain buffer for data.

        madesc = cuda.CUmemAccessDesc()
        madesc.location = allocation_prop.location
        madesc.flags = cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE

        mem_handles = [None] * comm_size

        for i, remote_handle_data in enumerate(all_handles_data):
            rank_ptr = (
                MnnvlMemory.current_start_address
                + MnnvlMemory.current_rank_stride * i
                + MnnvlMemory.current_mem_offset
            )
            if i == comm_rank:
                # Local memory mapping
                mem_handles[i] = allocated_mem_handle
                _check_cu_result(cuda.cuMemMap(rank_ptr, aligned_size, 0, allocated_mem_handle, 0))
            else:
                # Fabric memory mapping
                imported_mem_handle = _check_cu_result(
                    cuda.cuMemImportFromShareableHandle(
                        remote_handle_data, allocation_prop.requestedHandleTypes
                    )
                )
                mem_handles[i] = imported_mem_handle
                _check_cu_result(cuda.cuMemMap(rank_ptr, aligned_size, 0, imported_mem_handle, 0))

            _check_cu_result(cuda.cuMemSetAccess(rank_ptr, aligned_size, [madesc], 1))

        ptr = MnnvlMemory.current_start_address + MnnvlMemory.current_mem_offset
        stride = MnnvlMemory.current_rank_stride
        MnnvlMemory.allocated_map[ptr] = (
            mapping,
            aligned_size,
            mem_handles,
            MnnvlMemory.current_start_address,
            MnnvlMemory.current_rank_stride,
            MnnvlMemory.current_mem_offset,
        )
        MnnvlMemory.address_refcnt[MnnvlMemory.current_start_address] = (
            MnnvlMemory.address_refcnt.get(MnnvlMemory.current_start_address, 0) + 1
        )

        MnnvlMemory.current_mem_offset += aligned_size
        return ptr, stride

    @staticmethod
    def close_mnnvl_memory(ptr: int):
        mapping, aligned_size, mem_handles, start_address, rank_stride, address_offset = (
            MnnvlMemory.allocated_map.pop(ptr)
        )
        comm = MnnvlMemory.get_comm(mapping)
        comm_size = comm.Get_size()
        for i in range(comm_size):
            rank_ptr = start_address + i * rank_stride + address_offset
            _check_cu_result(cuda.cuMemUnmap(rank_ptr, aligned_size))
            _check_cu_result(cuda.cuMemRelease(mem_handles[i]))
        MnnvlMemory.address_refcnt[start_address] -= 1

        if MnnvlMemory.address_refcnt[start_address] == 0:
            MnnvlMemory.address_refcnt.pop(start_address)
            device_ptr = cuda.CUdeviceptr(start_address)
            _check_cu_result(cuda.cuMemAddressFree(device_ptr, comm_size * rank_stride))
            if start_address == MnnvlMemory.current_start_address:
                MnnvlMemory.current_start_address = 0
                MnnvlMemory.current_rank_stride = 0
                MnnvlMemory.current_mem_offset = 0

    @staticmethod
    def support_nvlink(need_all_up: bool = True):
        dev_id = torch.cuda.current_device()
        handle = pynvml.nvmlDeviceGetHandleByIndex(dev_id)
        link_count = pynvml.NVML_NVLINK_MAX_LINKS
        active_links = 0
        available_links = 0
        for link_idx in range(link_count):
            try:
                if pynvml.nvmlDeviceGetNvLinkCapability(
                    handle, link_idx, pynvml.NVML_NVLINK_CAP_P2P_SUPPORTED
                ):
                    available_links += 1
                    is_active = pynvml.nvmlDeviceGetNvLinkState(handle, link_idx)
                    if is_active:
                        active_links += 1
            except pynvml.NVMLError_NotSupported:
                continue
        return (
            active_links == available_links and available_links > 0
            if need_all_up
            else available_links > 0
        )

    @staticmethod
    def supports_mnnvl() -> bool:
        # TODO:
        # We check if it has all NVLink up now.
        # But it is not equivalent to MNNVL support.
        # May need better support check.
        support_nvlink_and_all_up = MnnvlMemory.support_nvlink(True)
        return support_nvlink_and_all_up


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
