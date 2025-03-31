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
import sys

import torch
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
        self.ptr, self.rank_stride = MnnvlMemory.open_mnnvl_memory(
            self.mapping, size)

    def __del__(self):
        if not sys.is_finalizing():
            MnnvlMemory.close_mnnvl_memory(self.ptr)

    def as_torch_strided_tensor(self, dtype):
        num_segments = MnnvlMemory.comm.Get_size()
        return pack_strided_memory(self.ptr, self.segment_size,
                                   self.rank_stride, num_segments, dtype,
                                   MnnvlMemory.dev_id)

    @staticmethod
    def initialize():
        # use a dummy torch CUDA tensor to trigger CUDA context initialization
        _ = torch.empty(1, device='cuda')

    @staticmethod
    def get_comm(mapping: Mapping):
        if MnnvlMemory.comm is not None:
            return MnnvlMemory.comm
        comm = mpi_comm().Split(
            mapping.pp_rank * mapping.cp_size + mapping.cp_rank,
            mapping.tp_rank)
        MnnvlMemory.comm = comm
        return comm

    @staticmethod
    def get_allocation_prop(dev_id: int):
        location = cuda.CUmemLocation()
        location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        location.id = dev_id
        allocation_prop = cuda.CUmemAllocationProp()
        allocation_prop.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        allocation_prop.requestedHandleTypes = cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
        allocation_prop.location = location
        return allocation_prop

    @staticmethod
    def get_allocation_granularity(dev_id: int):
        if MnnvlMemory.allocation_granularity != 0:
            return MnnvlMemory.allocation_granularity
        allocation_prop = MnnvlMemory.get_allocation_prop(dev_id)
        option = cuda.CUmemAllocationGranularity_flags(
            cuda.CUmemAllocationGranularity_flags.
            CU_MEM_ALLOC_GRANULARITY_RECOMMENDED)
        granularity = _check_cu_result(
            cuda.cuMemGetAllocationGranularity(prop=allocation_prop,
                                               option=option))
        MnnvlMemory.allocation_granularity = granularity
        return MnnvlMemory.allocation_granularity

    @staticmethod
    def new_mnnvl_memory_address(mapping: Mapping, size: int):
        page_count = (size + MnnvlMemory.fabric_page_size -
                      1) // MnnvlMemory.fabric_page_size
        current_rank_stride = page_count * MnnvlMemory.fabric_page_size
        logger.info(
            f"[MnnvlMemory] creating address with stride={current_rank_stride}")
        comm = MnnvlMemory.get_comm(mapping)
        comm_size = comm.Get_size()
        address_size = current_rank_stride * comm_size
        ptr = _check_cu_result(
            cuda.cuMemAddressReserve(address_size, MnnvlMemory.fabric_page_size,
                                     0, 0))
        MnnvlMemory.current_start_address = int(ptr)
        MnnvlMemory.current_rank_stride = current_rank_stride
        MnnvlMemory.current_mem_offset = 0

    @staticmethod
    def open_mnnvl_memory(mapping: Mapping, size: int):
        dev = _check_cu_result(cuda.cuCtxGetDevice())
        dev_id = int(dev)
        if MnnvlMemory.dev_id is None:
            MnnvlMemory.dev_id = dev_id
        assert dev_id == MnnvlMemory.dev_id,\
            f"Different dev_id found dev_id={dev_id} but MnnvlMemory.dev_id={MnnvlMemory.dev_id}"
        comm = MnnvlMemory.get_comm(mapping)
        comm_rank = comm.Get_rank()
        comm_size = comm.Get_size()
        all_rank_allocate_sizes = comm.allgather(size)
        assert len(all_rank_allocate_sizes) == comm_size
        assert all(
            x == size for x in
            all_rank_allocate_sizes), "Not all rank allocating same size."
        granularity = MnnvlMemory.get_allocation_granularity(dev_id)
        aligned_size = (size + granularity - 1) // granularity * granularity

        if MnnvlMemory.current_mem_offset + aligned_size > MnnvlMemory.current_rank_stride:
            MnnvlMemory.new_mnnvl_memory_address(mapping, aligned_size)

        assert MnnvlMemory.current_mem_offset + aligned_size <= MnnvlMemory.current_rank_stride

        allocation_prop = MnnvlMemory.get_allocation_prop(dev_id)
        allocated_mem_handle = _check_cu_result(
            cuda.cuMemCreate(aligned_size, allocation_prop, flags=0))
        exported_fabric_handle = _check_cu_result(
            cuda.cuMemExportToShareableHandle(
                allocated_mem_handle,
                cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC, 0))
        all_handles_data = comm.allgather(exported_fabric_handle.data)
        # all_handles_data like b'\x00\x00\x00 \x00\x00\x00\x00\x8f\xec\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\t\x00\x00\x00\x00\x00\x1d\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
        # can use buf = memoryview(data) to import if using plain buffer for data.

        madesc = cuda.CUmemAccessDesc()
        madesc.location = allocation_prop.location
        madesc.flags = cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE

        mem_handles = [None] * comm_size

        for i, remote_handle_data in enumerate(all_handles_data):
            rank_ptr = MnnvlMemory.current_start_address + MnnvlMemory.current_rank_stride * i + MnnvlMemory.current_mem_offset
            if i == comm_rank:
                # Local memory mapping
                mem_handles[i] = allocated_mem_handle
                _check_cu_result(
                    cuda.cuMemMap(rank_ptr, aligned_size, 0,
                                  allocated_mem_handle, 0))
            else:
                # Fabric memory mapping
                imported_mem_handle = _check_cu_result(
                    cuda.cuMemImportFromShareableHandle(
                        remote_handle_data, cuda.CUmemAllocationHandleType.
                        CU_MEM_HANDLE_TYPE_FABRIC))
                mem_handles[i] = imported_mem_handle
                _check_cu_result(
                    cuda.cuMemMap(rank_ptr, aligned_size, 0,
                                  imported_mem_handle, 0))

            _check_cu_result(
                cuda.cuMemSetAccess(rank_ptr, aligned_size, [madesc], 1))

        ptr = MnnvlMemory.current_start_address + MnnvlMemory.current_mem_offset
        stride = MnnvlMemory.current_rank_stride
        MnnvlMemory.allocated_map[ptr] = (mapping, aligned_size, mem_handles,
                                          MnnvlMemory.current_start_address,
                                          MnnvlMemory.current_rank_stride,
                                          MnnvlMemory.current_mem_offset)
        MnnvlMemory.address_refcnt[
            MnnvlMemory.current_start_address] = MnnvlMemory.address_refcnt.get(
                MnnvlMemory.current_start_address, 0) + 1

        MnnvlMemory.current_mem_offset += aligned_size
        return ptr, stride

    @staticmethod
    def close_mnnvl_memory(ptr: int):
        mapping, aligned_size, mem_handles, start_address, rank_stride, address_offset = MnnvlMemory.allocated_map.pop(
            ptr)
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
            _check_cu_result(
                cuda.cuMemAddressFree(device_ptr, comm_size * rank_stride))
            if start_address == MnnvlMemory.current_start_address:
                MnnvlMemory.current_start_address = 0
                MnnvlMemory.current_rank_stride = 0
                MnnvlMemory.current_mem_offset = 0
