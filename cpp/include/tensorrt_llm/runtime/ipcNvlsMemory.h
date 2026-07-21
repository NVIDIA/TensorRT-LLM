/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <set>
#include <vector>

namespace tensorrt_llm::runtime
{
struct IpcNvlsHandle
{
    // Begin internal kernel visible fields
    // Changes to these fields must sync with ipcNvlsMemory.h in internal kernel repo
    size_t size = 0;
    // Device pointers used by kernels
    uintptr_t uc_ptr = 0;
    uintptr_t mc_ptr = 0;
    // End internal kernel visible fields
    std::vector<uintptr_t> ipc_uc_ptrs;
    // Device pointers
    CUdeviceptr uc_va;
    CUdeviceptr mc_va;
    std::vector<CUdeviceptr> ipc_uc_vas;
    // Device allocation handles
    CUmemGenericAllocationHandle uc_handle;
    CUmemGenericAllocationHandle mc_handle;
    std::vector<CUmemGenericAllocationHandle> ipc_uc_handles;
};

void MPI_group_barrier(std::set<int> ranks);

//! \brief Whether NVLS (NVLink SHARP) multicast memory can be allocated on this
//! node. Checks only the static capability (CUDA driver >= 12010 and
//! CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED). This is the precondition for
//! ipcNvlsAllocate(): the allocator itself selects a fabric or POSIX-FD handle
//! via getMemHandleType(), so single-node NVLS works over POSIX-FD even when the
//! fabric/IMEX plane is not provisioned. The result is cached.
bool ipcNvlsSupported();

//! \brief Whether the NVLink fabric/IMEX plane is provisioned so that NVLS
//! multicast memory can actually be *bound* (not merely statically supported).
//! Extends ipcNvlsSupported() with a live fabric probe (getMemHandleType() must
//! resolve to CU_MEM_HANDLE_TYPE_FABRIC). Use this to decide whether NCCL may
//! attempt NVLS: on an unprovisioned node the static multicast attribute is a
//! false positive and NCCL aborts during init, so callers should disable
//! NCCL_NVLS when this returns false. The (heavy) result is cached.
bool ipcNvlsFabricUsable();

IpcNvlsHandle* ipcNvlsAllocate(size_t size, std::set<int> ranks);

void ipcNvlsFree(IpcNvlsHandle* handle);

template <typename T>
class DeviceAllocationNvls
{
public:
    DeviceAllocationNvls() = default;

    ~DeviceAllocationNvls()
    {
        this->free();
    }

    void reset(size_t size, std::set<int> ranks)
    {
        this->free();
        _handle = ipcNvlsAllocate(size * sizeof(T), ranks);
        _capacity = size;
    }

    // Return device pointer to multicast memory
    [[nodiscard]] T* getMulticastPointer() const
    {
        return reinterpret_cast<T*>(_handle->mc_ptr);
    }

    // Return device pointer for current rank
    [[nodiscard]] T* getUnicastPointer() const
    {
        return reinterpret_cast<T*>(_handle->uc_ptr);
    }

    // Return host list of device pointers to memory on each rank
    [[nodiscard]] T** getIpcUnicastPointers()
    {
        return reinterpret_cast<T**>(_handle->ipc_uc_ptrs.data());
    }

    [[nodiscard]] size_t getCapacity() const
    {
        return _capacity;
    }

    void free()
    {
        if (_capacity > 0)
        {
            ipcNvlsFree(_handle);
            _capacity = 0;
        }
    }

private:
    size_t _capacity = 0;
    IpcNvlsHandle* _handle;
};
} // namespace tensorrt_llm::runtime
