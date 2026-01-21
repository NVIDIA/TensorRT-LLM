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
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/runtime/ipcNvlsMemory.h"
#include "tensorrt_llm/runtime/ipcSocket.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"

#if ENABLE_NVSHMEM
#include <nvshmem/nvshmem.h>
#include <nvshmem/nvshmemx.h>
#endif
#if ENABLE_MULTI_DEVICE
#include <nvml.h>
#endif
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

#define NVMLCHECK(cmd)                                                                                                 \
    do                                                                                                                 \
    {                                                                                                                  \
        nvmlReturn_t retval = cmd;                                                                                     \
        if (retval != NVML_SUCCESS)                                                                                    \
        {                                                                                                              \
            printf("Failed: NVML error %s:%d '%s'\n", __FILE__, __LINE__, nvmlErrorString(retval));                    \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

// if n is already a multiple of "multiple", n is returned unchanged, otherwise round up to next multiple.
#define ROUND_UP(n, multiple) (((n + multiple - 1) / multiple) * multiple)

namespace tensorrt_llm::runtime
{
using namespace tensorrt_llm::mpi;

#if ENABLE_MULTI_DEVICE && !ENABLE_NVSHMEM
union IpcMemHandle
{
    uint64_t fd;
    CUmemFabricHandle fh;
};

class IpcCommunicator
{
public:
    virtual ~IpcCommunicator() = default;
    virtual void bcastMemHandle(IpcMemHandle* handle, int root) = 0;
};

class IpcSocketCommunicator : public IpcCommunicator
{
public:
    IpcSocketCommunicator(int world_rank, int group_rank, std::vector<int> group_ranks, MPI_Comm group_comm)
        : mGroupRank(group_rank)
        , mGroupRanks(group_ranks)
        , mGroupComm(group_comm)
    {
        timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        unsigned long seed = ts.tv_sec * 1000000000L + ts.tv_nsec;
        srand(seed);
        uint64_t unique_op_id = (uint64_t) (rand()) ^ ((uint64_t) (rand()) << 32);
        MPI_Bcast(&unique_op_id, sizeof(unique_op_id), MPI_BYTE, 0, group_comm);

        uint32_t volatile abort_flag = 0;
        mSocket = ncclIpcSocketInit(world_rank, unique_op_id, &abort_flag);
        MPI_Barrier(group_comm);
    }

    ~IpcSocketCommunicator()
    {
        ncclIpcSocketClose(mSocket);
    }

    void bcastMemHandle(IpcMemHandle* handle, int root) override
    {
        if (mGroupRank == root)
        {
            for (size_t i = 0; i < mGroupRanks.size(); ++i)
            {
                if (i != root)
                {
                    ncclIpcSocketSendFd(mSocket, handle->fd, mGroupRanks[i]);
                }
            }
            MPI_Barrier(mGroupComm);
        }
        else
        {
            MPI_Barrier(mGroupComm);
            handle->fd = ncclIpcSocketRecvFd(mSocket);
        }
        MPI_Barrier(mGroupComm);
    }

private:
    int mGroupRank;
    std::vector<int> mGroupRanks;
    MPI_Comm mGroupComm;
    std::shared_ptr<NcclIpcSocket> mSocket;
};

class IpcFabricCommunicator : public IpcCommunicator
{
public:
    IpcFabricCommunicator(MPI_Comm group_comm)
        : mGroupComm(group_comm)
    {
    }

    ~IpcFabricCommunicator() = default;

    void bcastMemHandle(IpcMemHandle* handle, int root) override
    {
        MPI_Bcast(handle, sizeof(CUmemFabricHandle), MPI_BYTE, root, mGroupComm);
    }

private:
    MPI_Comm mGroupComm;
};

class NVLSCudaAllocator
{
public:
    static IpcNvlsHandle* allocate(size_t size, std::vector<int> ranks)
    {
        auto nvls_handle = new IpcNvlsHandle();

        // Create a new communicator for the subset of ranks.
        MPI_Group world_group, new_group;
        MPI_Comm new_comm;
        // Get the group of the world communicator.
        MPI_Comm_group(COMM_SESSION, &world_group);
        // Create a new group containing only the ranks we want.
        MPI_Group_incl(world_group, ranks.size(), ranks.data(), &new_group);
        // Create a new communicator from the group.
        MPI_Comm_create_group(COMM_SESSION, new_group, 0, &new_comm);

        // Get rank and group rank.
        int world_rank;
        int group_rank;
        MPI_Comm_rank(COMM_SESSION, &world_rank);
        MPI_Comm_rank(new_comm, &group_rank);

        // Get runtime and driver device IDs.
        int device_id;
        int CU_dev;
        TLLM_CUDA_CHECK(cudaGetDevice(&device_id));
        CUCHECK(cuDeviceGet(&CU_dev, device_id));

        // Get handle type used to share memory handles between devices.
        auto handle_type = getMemHandleType();

        // Define allocation access permissions (same for unicast and multicast).
        CUmemAccessDesc access_desc;
        memset(&access_desc, 0, sizeof(CUmemAccessDesc));
        access_desc.location.id = device_id;
        access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

        // Define unicast allocation properties.
        CUmemAllocationProp prop;
        memset(&prop, 0, sizeof(CUmemAllocationProp));
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device_id;
        prop.requestedHandleTypes = handle_type;

        // Define multicast allocation properties.
        CUmulticastObjectProp mcprop;
        memset(&mcprop, 0, sizeof(CUmulticastObjectProp));
        mcprop.numDevices = ranks.size();
        mcprop.handleTypes = handle_type;
        mcprop.flags = 0;

        // Round up allocation size to the nearest multiple of the unicast allocation granularity.
        size_t granularity = 0;
        CUCHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
        size = ROUND_UP(size, granularity);

        // Round up allocation size to the nearest multiple of the multicast allocation granularity.
        size_t mc_granularity = 0;
        CUCHECK(cuMulticastGetGranularity(&mc_granularity, &mcprop, CU_MULTICAST_GRANULARITY_MINIMUM));
        size = ROUND_UP(size, mc_granularity);
        mcprop.size = size;
        nvls_handle->size = size;

        // Allocate physical pages of memory on GPU.
        CUCHECK(cuMemCreate(&nvls_handle->uc_handle, size, &prop, 0));
        // Reserve unicast virtual address space for the memory.
        CUCHECK(cuMemAddressReserve(&nvls_handle->uc_va, size, granularity, 0U, 0));
        // Map the unicast virtual address space to the physical pages.
        CUCHECK(cuMemMap(nvls_handle->uc_va, size, 0, nvls_handle->uc_handle, 0));
        // Set the access permissions for the unicast memory.
        CUCHECK(cuMemSetAccess(nvls_handle->uc_va, size, &access_desc, 1));
        nvls_handle->uc_ptr = reinterpret_cast<uintptr_t>((void*) nvls_handle->uc_va);

        // Setup communicator needed for multicast and unicast pointer exchange.
        std::shared_ptr<IpcCommunicator> ipc_communicator;
        if (handle_type == CU_MEM_HANDLE_TYPE_FABRIC)
        {
            ipc_communicator = std::make_shared<IpcFabricCommunicator>(new_comm);
        }
        else
        {
            ipc_communicator = std::make_shared<IpcSocketCommunicator>(world_rank, group_rank, ranks, new_comm);
        }

        // Unicast pointer exchange between ranks.
        IpcMemHandle ipc_handle;
        CUCHECK(cuMemExportToShareableHandle((void*) &ipc_handle, nvls_handle->uc_handle, handle_type, 0 /*flags*/));

        nvls_handle->ipc_uc_ptrs.resize(ranks.size());
        nvls_handle->ipc_uc_vas.resize(ranks.size());
        nvls_handle->ipc_uc_handles.resize(ranks.size());

        for (int i = 0; i < ranks.size(); i++)
        {
            IpcMemHandle peer_ipc_handle = ipc_handle;
            ipc_communicator->bcastMemHandle(&peer_ipc_handle, i);
            if (i != group_rank)
            {
                void* os_handle
                    = handle_type == CU_MEM_HANDLE_TYPE_FABRIC ? (void*) &peer_ipc_handle : (void*) peer_ipc_handle.fd;
                CUCHECK(cuMemImportFromShareableHandle(&nvls_handle->ipc_uc_handles[i], os_handle, handle_type));
                // Reserve peer unicast virtual address space for the memory.
                CUCHECK(cuMemAddressReserve(&nvls_handle->ipc_uc_vas[i], size, granularity, 0U, 0));
                // Map the peer unicast virtual address space to the physical pages.
                CUCHECK(cuMemMap(nvls_handle->ipc_uc_vas[i], size, 0, nvls_handle->ipc_uc_handles[i], 0));
                // Set the access permissions for the peer unicast memory.
                CUCHECK(cuMemSetAccess(nvls_handle->ipc_uc_vas[i], size, &access_desc, 1));
                nvls_handle->ipc_uc_ptrs[i] = reinterpret_cast<uintptr_t>((void*) nvls_handle->ipc_uc_vas[i]);
            }
            else
            {
                nvls_handle->ipc_uc_ptrs[i] = nvls_handle->uc_ptr;
                nvls_handle->ipc_uc_vas[i] = nvls_handle->uc_va;
                nvls_handle->ipc_uc_handles[i] = nvls_handle->uc_handle;
            }
        }

        // Initialize multicast object for all ranks.
        if (group_rank == 0)
        {
            CUCHECK(cuMulticastCreate(&nvls_handle->mc_handle, &mcprop));
            // Export the allocation for the importing process.
            CUCHECK(cuMemExportToShareableHandle(&ipc_handle, nvls_handle->mc_handle, handle_type, 0 /*flags*/));
            ipc_communicator->bcastMemHandle(&ipc_handle, 0);
        }
        else
        {
            ipc_communicator->bcastMemHandle(&ipc_handle, 0);
            void* os_handle = handle_type == CU_MEM_HANDLE_TYPE_FABRIC ? (void*) &ipc_handle : (void*) ipc_handle.fd;
            CUCHECK(cuMemImportFromShareableHandle(&nvls_handle->mc_handle, os_handle, handle_type));
        }

        // Add device to multicast object
        CUCHECK(cuMulticastAddDevice(nvls_handle->mc_handle, CU_dev));
        // Bind physical memory to the Multicast group.
        // Note: It will block until all ranks have been added to the group.
        CUCHECK(cuMulticastBindMem(nvls_handle->mc_handle, 0, nvls_handle->uc_handle, 0, size, 0));
        // Reserve multicast virtual address space for the memory.
        CUCHECK(cuMemAddressReserve(&nvls_handle->mc_va, size, mc_granularity, 0U, 0));
        // Map the multicast virtual address space to the physical pages.
        CUCHECK(cuMemMap(nvls_handle->mc_va, size, 0, nvls_handle->mc_handle, 0));
        // Set the access permissions for the multicast memory.
        CUCHECK(cuMemSetAccess(nvls_handle->mc_va, size, &access_desc, 1 /* count */));
        nvls_handle->mc_ptr = reinterpret_cast<uintptr_t>((void*) nvls_handle->mc_va);

        // Clean up
        MPI_Group_free(&new_group);
        MPI_Group_free(&world_group);
        MPI_Comm_free(&new_comm);

        return nvls_handle;
    }

    static void free(IpcNvlsHandle* nvls_handle)
    {
        CUCHECK(cuMemUnmap(nvls_handle->mc_va, nvls_handle->size));
        CUCHECK(cuMemRelease(nvls_handle->mc_handle));
        CUCHECK(cuMemAddressFree(nvls_handle->mc_va, nvls_handle->size));
        for (size_t i = 0; i < nvls_handle->ipc_uc_vas.size(); ++i)
        {
            CUCHECK(cuMemUnmap(nvls_handle->ipc_uc_vas[i], nvls_handle->size));
            CUCHECK(cuMemRelease(nvls_handle->ipc_uc_handles[i]));
            CUCHECK(cuMemAddressFree(nvls_handle->ipc_uc_vas[i], nvls_handle->size));
        }
    }

private:
    static CUmemAllocationHandleType getMemHandleType()
    {
        int device_id;
        TLLM_CUDA_CHECK(cudaGetDevice(&device_id));

        // Check if fabric handle support is available.
        int fabric_supported = 0;
        CUCHECK(cuDeviceGetAttribute(&fabric_supported, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, device_id));
        if (!fabric_supported)
        {
            TLLM_LOG_TRACE(
                "checking fabric support... CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED not supported.");
            return CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
        }

        nvmlDevice_t nvml_device;
        nvmlGpuFabricInfo_t fabric_info;
        NVMLCHECK(nvmlInit_v2());
        NVMLCHECK(nvmlDeviceGetHandleByIndex(device_id, &nvml_device));
        NVMLCHECK(nvmlDeviceGetGpuFabricInfo(nvml_device, &fabric_info));
        NVMLCHECK(nvmlShutdown());

        // Check if the fabric is fully initialized.
        if (fabric_info.state != NVML_GPU_FABRIC_STATE_COMPLETED || fabric_info.status != NVML_SUCCESS)
        {
            TLLM_LOG_TRACE("checking fabric support... fabric state is NOT COMPLETE: state=%u status=%u.",
                fabric_info.state, fabric_info.status);
            return CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
        }

        // Check that fabric handles can be created.
        CUmemAllocationProp prop;
        memset(&prop, 0, sizeof(CUmemAllocationProp));
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device_id;
        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;

        size_t alloc_size = 1024; // anything > 0
        size_t min_gran = 0;
        CUCHECK(cuMemGetAllocationGranularity(&min_gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
        alloc_size = ROUND_UP(alloc_size, min_gran);

        CUmemGenericAllocationHandle handle;
        CUresult err = cuMemCreate(&handle, alloc_size, &prop, 0);
        if (err == CUDA_ERROR_NOT_PERMITTED || err == CUDA_ERROR_NOT_SUPPORTED)
        {
            TLLM_LOG_TRACE("checking fabric support... cuMemCreate failed with not %s.",
                err == CUDA_ERROR_NOT_PERMITTED ? "permitted" : "supported");
            return CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
        }
        else
        {
            CUCHECK(err);
        }

        // Check if fabric handles can be exported & imported by IMEX (Internode Memory Exchange)
        CUmemFabricHandle fh;
        err = cuMemExportToShareableHandle(&fh, handle, CU_MEM_HANDLE_TYPE_FABRIC, 0);
        if (err != CUDA_SUCCESS
            || (err = cuMemImportFromShareableHandle(&handle, &fh, CU_MEM_HANDLE_TYPE_FABRIC)) != CUDA_SUCCESS)
        {
            TLLM_LOG_TRACE("checking fabric support... cuMemExport/cuMemImport failed.");
            CUCHECK(cuMemRelease(handle));
            return CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
        }

        TLLM_LOG_TRACE("fabric status: state=%u status=%u clique=%u", device_id, fabric_info.state, fabric_info.status,
            fabric_info.cliqueId);

        CUCHECK(cuMemRelease(handle));
        // If we get here, fabric handles are supported.
        return CU_MEM_HANDLE_TYPE_FABRIC;
    }
};
#endif

/**
 * @brief MPI_Barrier when subset of ranks present
 */
void MPI_group_barrier(std::set<int> group)
{
#if ENABLE_MULTI_DEVICE
    // Create a new communicator for the subset of ranks
    MPI_Group world_group, new_group;
    MPI_Comm new_comm;

    // Get the group of the world communicator
    MPI_Comm_group(COMM_SESSION, &world_group);

    // Create a new group containing only the ranks we want
    std::vector<int> ranks(group.begin(), group.end());
    MPI_Group_incl(world_group, ranks.size(), ranks.data(), &new_group);

    // Create a new communicator from the group
    MPI_Comm_create_group(COMM_SESSION, new_group, 0, &new_comm);

    // Use the new communicator for the barrier
    MPI_Barrier(new_comm);

    // Clean up
    MPI_Group_free(&new_group);
    MPI_Group_free(&world_group);
    MPI_Comm_free(&new_comm);
#else
    TLLM_THROW("MPI_group_barrier needs to be compiled with ENABLE_MULTI_DEVICE");
#endif
}

bool ipcNvlsSupported()
{
#if ENABLE_MULTI_DEVICE
    CUdevice current_dev;
    int cuda_dev = -1;
    int cuda_driver_version = -1;
    int dev_count = 0;

    TLLM_CUDA_CHECK(cudaDriverGetVersion(&cuda_driver_version));
    if (cuda_driver_version < 12010)
    {
        TLLM_LOG_ERROR("CUDA Driver version < 12010");
        return false;
    }

    TLLM_CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    for (int i = 0; i < dev_count; ++i)
    {
        TLLM_CUDA_CHECK(cudaGetDevice(&cuda_dev));
        CUCHECK(cuDeviceGet(&current_dev, cuda_dev));

        int multicast_supported = 0;
        CUCHECK(cuDeviceGetAttribute(&multicast_supported, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, current_dev));
        if (!multicast_supported)
        {
            TLLM_LOG_ERROR("CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED not supported on GPU%d.", cuda_dev);
            return false;
        }
    }

    return true;
#else
    return false;
#endif
}

IpcNvlsHandle* ipcNvlsAllocate(size_t size, std::set<int> group)
{
#if ENABLE_MULTI_DEVICE
    TLLM_CHECK_WITH_INFO(ipcNvlsSupported(), "Switch multicast is not supported on this system.");
    TLLM_CHECK(size > 0);
    TLLM_CHECK(group.size() >= 2);

    std::vector<int> ranks(group.begin(), group.end());
    int group_size = ranks.size();

    MPI_Comm mpi_comm = COMM_SESSION;

    // Create a new communicator with only the ranks in the group
    MPI_Group world_group, new_group;
    MPI_Comm_group(mpi_comm, &world_group);
    MPI_Group_incl(world_group, group_size, ranks.data(), &new_group);

    MPI_Comm new_comm;
    MPI_Comm_create_group(mpi_comm, new_group, 0, &new_comm);

#if ENABLE_NVSHMEM
    // Initialize NVSHMEM with the new communicator
    nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
    attr.mpi_comm = &new_comm;
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

    // Allocate NVSHMEM memory
    void* ptr = nvshmem_malloc(size);

    // Create handle to return
    auto handle = new IpcNvlsHandle();

    handle->size = size;
    handle->uc_ptr = reinterpret_cast<uintptr_t>(ptr);
    handle->mc_ptr = reinterpret_cast<uintptr_t>(nvshmemx_mc_ptr(NVSHMEM_TEAM_WORLD, ptr));
    for (int i = 0; i < ranks.size(); i++)
    {
        handle->ipc_uc_ptrs.push_back(reinterpret_cast<uintptr_t>(nvshmem_ptr(ptr, i)));
    }
#else // !ENABLE_NVSHMEM
    auto handle = NVLSCudaAllocator::allocate(size, ranks);
#endif

    TLLM_LOG_INFO("Rank %d NVLS allocate %zu bytes, uc_ptr:%p mc_ptr:%p", COMM_SESSION.getRank(), size,
        (void*) handle->uc_ptr, (void*) handle->mc_ptr);

    // Cleanup
    MPI_Group_free(&new_group);
    MPI_Group_free(&world_group);

    MPI_Barrier(new_comm);

    MPI_Comm_free(&new_comm);

    return handle;
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
#if ENABLE_NVSHMEM
    nvshmem_free((void*) handle->uc_ptr);
#else
    NVLSCudaAllocator::free(handle);
#endif
    delete handle;
#else
    TLLM_THROW("ipcNvlsFree needs to be compiled with ENABLE_MULTI_DEVICE");
#endif
}

} // namespace tensorrt_llm::runtime
