/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/runtime/ipcNvlsMemory.h"
#include "ipcSocket.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"

#include <unistd.h>

#define CUCHECK(cmd)                                                                                                   \
    do                                                                                                                 \
    {                                                                                                                  \
        CUresult retval = cmd;                                                                                         \
        if (retval != CUDA_SUCCESS)                                                                                    \
        {                                                                                                              \
            const char* error_string;                                                                                  \
            cuGetErrorString(retval, &error_string);                                                                   \
            printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, error_string);                               \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

#define ALIGN_SIZE(x, align) x = ((x + align - 1) / align) * align;

namespace tensorrt_llm::runtime
{
using namespace tensorrt_llm::mpi;

void MPI_group_rank(std::set<int> group, int* groupRank)
{
#if ENABLE_MULTI_DEVICE
    int rank = COMM_SESSION.getRank();
    auto it = std::find(group.begin(), group.end(), rank);
    TLLM_CHECK_WITH_INFO(
        it != group.end(), "Incorrect group specified - rank " + std::to_string(rank) + " not found in group");
    *groupRank = std::distance(group.begin(), it);
#else
    TLLM_THROW("MPI_group_rank needs to be compiled with ENABLE_MULTI_DEVICE");
#endif
}

/**
 * @brief MPI_Barrier when subset of ranks present
 */
void MPI_group_barrier(std::set<int> group)
{
#if ENABLE_MULTI_DEVICE
    std::vector<int> ranks(group.begin(), group.end());
    int size = group.size();
    int group_rank;
    MPI_group_rank(group, &group_rank);

    int root = 0;

    if (group_rank == root)
    {
        int dummy = 0;
        // Root receives messages from all other processes
        for (int i = 1; i < size; i++)
        {
            COMM_SESSION.recv(&dummy, 1, MpiType::kINT32, ranks[i], MpiTag::kDefault);
        }
        // Root sends messages back to all other processes
        for (int i = 1; i < size; i++)
        {
            COMM_SESSION.send(&dummy, 1, MpiType::kINT32, ranks[i], MpiTag::kDefault);
        }
    }
    else
    {
        int dummy = 0;
        // Non-root processes send a message to root
        COMM_SESSION.send(&dummy, 1, MpiType::kINT32, ranks[root], MpiTag::kDefault);
        // Non-root processes receive a message from root
        COMM_SESSION.recv(&dummy, 1, MpiType::kINT32, ranks[root], MpiTag::kDefault);
    }
#else
    TLLM_THROW("MPI_group_barrier needs to be compiled with ENABLE_MULTI_DEVICE");
#endif
}

/**
 * @brief MPI_Bcast when subset of ranks present
 */
void MPI_group_bcast(std::set<int> group, void* buffer, int count, MpiType datatype, int root)
{
#if ENABLE_MULTI_DEVICE
    int group_rank;
    MPI_group_rank(group, &group_rank);
    std::vector<int> ranks(group.begin(), group.end());

    if (group_rank == root)
    {
        // Root sends message to all other processes
        for (size_t i = 1; i < ranks.size(); ++i)
        {
            COMM_SESSION.send(buffer, count, datatype, ranks[i], MpiTag::kDefault);
        }
    }
    else
    {
        // Non-root processes receive a message from root
        COMM_SESSION.recv(buffer, count, datatype, ranks[root], MpiTag::kDefault);
    }
    MPI_group_barrier(group);
#else
    TLLM_THROW("MPI_group_bcast needs to be compiled with ENABLE_MULTI_DEVICE");
#endif
}

bool ipcNvlsSupported()
{
    CUdevice current_dev;
    int cuda_dev = -1;
    int cuda_driver_version = -1;
    int dev_count = 0;

    TLLM_CUDA_CHECK(cudaDriverGetVersion(&cuda_driver_version));
    if (cuda_driver_version < 12010)
    {
        return false;
    }

    TLLM_CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    for (int i = 0; i < dev_count; ++i)
    {
        TLLM_CUDA_CHECK(cudaGetDevice(&cuda_dev));
        CUCHECK(cuDeviceGet(&current_dev, cuda_dev));

        int mc_support = 0;
        CUCHECK(cuDeviceGetAttribute(
            &mc_support, static_cast<CUdevice_attribute>(CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED), current_dev));
        if (mc_support == 0)
        {
            return false;
        }
    }

    return true;
}

IpcNvlsHandle* ipcNvlsAllocate(size_t size, std::set<int> group)
{
#if ENABLE_MULTI_DEVICE
    TLLM_CHECK(size > 0);

    std::vector<int> ranks(group.begin(), group.end());

    int rank = COMM_SESSION.getRank();

    int group_rank;
    MPI_group_rank(group, &group_rank);
    int device_id = ranks[group_rank];

    cudaSetDevice(device_id);

    CUmemAllocationProp ucprop;
    CUmulticastObjectProp mcprop;
    size_t uc_align = 0;
    size_t mc_align = 0;

    CUmemAccessDesc uc_mc_access;
    memset(&uc_mc_access, 0, sizeof(CUmemAccessDesc));
    uc_mc_access.location.id = device_id;
    uc_mc_access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    uc_mc_access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    memset(&ucprop, 0, sizeof(CUmemAllocationProp));
    ucprop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    ucprop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    ucprop.location.id = device_id;
    ucprop.allocFlags.gpuDirectRDMACapable = 1;
    ucprop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    CUCHECK(cuMemGetAllocationGranularity(&uc_align, &ucprop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    ALIGN_SIZE(size, uc_align);

    memset(&mcprop, 0, sizeof(CUmulticastObjectProp));
    mcprop.numDevices = ranks.size();
    mcprop.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    mcprop.flags = 0;
    mcprop.size = size;
    CUCHECK(cuMulticastGetGranularity(&mc_align, &mcprop, CU_MULTICAST_GRANULARITY_MINIMUM));
    ALIGN_SIZE(size, mc_align);
    mcprop.size = size;

    // Init NVLS handle
    IpcNvlsHandle handle;
    handle.size = mcprop.size;

    // Get time
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    // High res time down to nanosec
    unsigned long seed = ts.tv_sec * 1000000000L + ts.tv_nsec;
    // Initialize with rand seed.
    srand(seed);
    int root = 0;
    uint64_t unique_op_id = (uint64_t) (rand()) ^ ((uint64_t) (rand()) << 32);
    MPI_group_bcast(group, &unique_op_id, sizeof(unique_op_id), MpiType::kBYTE, root);

    uint32_t volatile abort_flag = 0;
    std::shared_ptr<NcclIpcSocket> socket = ncclIpcSocketInit(rank, unique_op_id, &abort_flag);
    MPI_group_barrier(group);

    int fd;
    if (group_rank == root)
    {
        CUCHECK(cuMulticastCreate(&handle.mc_handle, &mcprop));
        CUCHECK(
            cuMemExportToShareableHandle(&fd, handle.mc_handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0 /*flags*/));
        // Root to send fd to all other processes
        for (size_t i = 1; i < group.size(); ++i)
        {
            ncclIpcSocketSendFd(socket, fd, ranks[i], unique_op_id);
        }
        MPI_group_barrier(group);
    }
    else
    {
        MPI_group_barrier(group);
        fd = ncclIpcSocketRecvFd(socket);
        CUCHECK(cuMemImportFromShareableHandle(
            &handle.mc_handle, (void*) (uintptr_t) fd, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
    }

    MPI_group_barrier(group);
    close(fd);

    // Add device to multicast object
    CUdevice dev;
    CUCHECK(cuDeviceGet(&dev, device_id));
    CUCHECK(cuMulticastAddDevice(handle.mc_handle, dev));

    // This step needs to be completed on all processes controlling devices that should participate in a Multicast Team before memory on any device is bound to the Multicast Object.
    MPI_group_barrier(group);

    // Create multicast VA
    CUCHECK(cuMemAddressReserve(&handle.mc_va, size, mc_align, 0U, 0));
    CUCHECK(cuMemMap(handle.mc_va, size, 0, handle.mc_handle, 0));
    CUCHECK(cuMemSetAccess(handle.mc_va, size, &uc_mc_access, 1 /* count */));

    // Allocate unicast VA
    CUCHECK(cuMemCreate(&handle.uc_handle, size, &ucprop, 0));
    CUCHECK(cuMemAddressReserve(&handle.uc_va, size, uc_align, 0U, 0));
    CUCHECK(cuMemMap(handle.uc_va, size, 0, handle.uc_handle, 0));

    // set access on UC address, for all GPUs so that UVA works
    for (int gpu_id : group)
    {
        uc_mc_access.location.id = gpu_id;
        CUCHECK(cuMemSetAccess(handle.uc_va, size, &uc_mc_access, 1 /* count */));
    }

    // Bind unicast memory to multicast group
    CUCHECK(cuMulticastBindMem(handle.mc_handle, 0 /*mcOffset*/, handle.uc_handle, 0 /*memOffset*/, size, 0 /*flags*/));

    handle.mc_ptr = reinterpret_cast<uintptr_t>((void*) handle.mc_va);
    handle.uc_ptr = reinterpret_cast<uintptr_t>((void*) handle.uc_va);

    printf("Rank %d nvlsAllocated %zu bytes successfully %p %p\n", rank, size, (void*) handle.uc_ptr,
        (void*) handle.mc_ptr);

    // Export to unicast VA to shareable handle
    int fd_uc;
    CUCHECK(cuMemExportToShareableHandle(
        (void*) &fd_uc, handle.uc_handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0 /*flags*/));

    handle.ipc_uc_ptrs.resize(ranks.size());
    handle.ipc_uc_vas.resize(ranks.size());
    handle.ipc_uc_handles.resize(ranks.size());

    // Allgather unicast shareable handles
    std::vector<int> peer_fds_uc(ranks.size());
    peer_fds_uc[group_rank] = fd_uc;
    for (size_t i = 1; i < ranks.size(); ++i)
    {
        MPI_group_barrier(group);
        int send_rank = (group_rank + i) % ranks.size();
        int recv_rank = (group_rank + ranks.size() - i) % ranks.size();
        ncclIpcSocketSendFd(socket, fd_uc, ranks[send_rank], unique_op_id);
        peer_fds_uc[recv_rank] = ncclIpcSocketRecvFd(socket);
    }
    ncclIpcSocketClose(socket);

    // Import unicast shareable handles
    for (size_t i = 0; i < ranks.size(); ++i)
    {
        if (ranks[i] == rank)
        {
            handle.ipc_uc_ptrs[i] = handle.uc_ptr;
            handle.ipc_uc_vas[i] = handle.uc_va;
            handle.ipc_uc_handles[i] = handle.uc_handle;
        }
        else
        {
            CUCHECK(cuMemImportFromShareableHandle(&handle.ipc_uc_handles[i], (void*) (uintptr_t) peer_fds_uc[i],
                CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
            CUCHECK(cuMemAddressReserve(&handle.ipc_uc_vas[i], size, uc_align, 0U, 0));
            CUCHECK(cuMemMap(handle.ipc_uc_vas[i], size, 0, handle.ipc_uc_handles[i], 0));
            // set access on UC address, for all GPUs so that UVA works
            for (int gpu_id : group)
            {
                uc_mc_access.location.id = gpu_id;
                CUCHECK(cuMemSetAccess(handle.ipc_uc_vas[i], size, &uc_mc_access, 1 /* count */));
            }

            handle.ipc_uc_ptrs[i] = reinterpret_cast<uintptr_t>((void*) handle.ipc_uc_vas[i]);
        }
        // close FD UC
        close(peer_fds_uc[i]);
    }

    MPI_group_barrier(group);

    printf("Rank %d imported IPC handles successfully\n", rank);

    return new IpcNvlsHandle(std::move(handle));
#else
    TLLM_THROW("ipcNvlsAllocate needs to be compiled with ENABLE_MULTI_DEVICE");
#endif
}

void ipcNvlsFree(IpcNvlsHandle* handle)
{
#if ENABLE_MULTI_DEVICE
    if (handle == nullptr)
    {
        return;
    }

    // Unmap and release MC VA
    CUCHECK(cuMemUnmap(handle->mc_va, handle->size));
    CUCHECK(cuMemRelease(handle->mc_handle));
    CUCHECK(cuMemAddressFree(handle->mc_va, handle->size));
    // Unmap and release UC VA
    for (size_t i = 0; i < handle->ipc_uc_vas.size(); ++i)
    {
        CUCHECK(cuMemUnmap(handle->ipc_uc_vas[i], handle->size));
        CUCHECK(cuMemRelease(handle->ipc_uc_handles[i]));
        CUCHECK(cuMemAddressFree(handle->ipc_uc_vas[i], handle->size));
    }

    delete handle;
#else
    TLLM_THROW("ipcNvlsFree needs to be compiled with ENABLE_MULTI_DEVICE");
#endif
}

} // namespace tensorrt_llm::runtime
