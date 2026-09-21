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
#include "ipcNvlsMemoryImpl.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/runtime/ipcNvlsMemory.h"
#include "tensorrt_llm/runtime/ipcSocket.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"

#if ENABLE_NVSHMEM
#include <nvshmem/nvshmem.h>
#include <nvshmem/nvshmemx.h>
#endif
#if ENABLE_MULTI_DEVICE
#include "tensorrt_llm/common/nvmlWrapper.h"
#endif
#include <unistd.h>

#include <functional>
#include <iterator>
#include <sstream>

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
            printf("Failed: NVML error %s:%d '%s'\n", __FILE__, __LINE__,                                              \
                tensorrt_llm::common::NVMLWrapper::getInstance()->nvmlErrorString(retval));                            \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

// if n is already a multiple of "multiple", n is returned unchanged, otherwise round up to next multiple.
#define ROUND_UP(n, multiple) (((n + multiple - 1) / multiple) * multiple)

namespace tensorrt_llm::runtime
{
using namespace tensorrt_llm::mpi;

#if ENABLE_MULTI_DEVICE && !ENABLE_NVSHMEM
using detail::IpcCommunicator;
using detail::IpcMemHandle;

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

        mSocket = ncclIpcSocketInit(world_rank, unique_op_id, &mAbortFlag);
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

    [[nodiscard]] int getGroupRank() const override
    {
        return mGroupRank;
    }

    [[nodiscard]] int getGroupSize() const override
    {
        return static_cast<int>(mGroupRanks.size());
    }

    [[nodiscard]] CUmemAllocationHandleType getMemHandleType() const override
    {
        return CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    }

private:
    int mGroupRank;
    std::vector<int> mGroupRanks;
    MPI_Comm mGroupComm;
    uint32_t volatile mAbortFlag{0};
    std::shared_ptr<NcclIpcSocket> mSocket;
};

class IpcFabricCommunicator : public IpcCommunicator
{
public:
    IpcFabricCommunicator(MPI_Comm groupComm, int groupRank, int groupSize)
        : mGroupComm(groupComm)
        , mGroupRank(groupRank)
        , mGroupSize(groupSize)
    {
    }

    ~IpcFabricCommunicator() = default;

    void bcastMemHandle(IpcMemHandle* handle, int root) override
    {
        MPI_Bcast(handle, sizeof(CUmemFabricHandle), MPI_BYTE, root, mGroupComm);
    }

    [[nodiscard]] int getGroupRank() const override
    {
        return mGroupRank;
    }

    [[nodiscard]] int getGroupSize() const override
    {
        return mGroupSize;
    }

    [[nodiscard]] CUmemAllocationHandleType getMemHandleType() const override
    {
        return CU_MEM_HANDLE_TYPE_FABRIC;
    }

private:
    MPI_Comm mGroupComm;
    int mGroupRank;
    int mGroupSize;
};

// Returns CU_MEM_HANDLE_TYPE_FABRIC when fabric-handle memory can actually be
// allocated and exported on the current device (i.e. the fabric/IMEX plane is
// provisioned), else CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR. Used both to pick
// the NVLS allocation handle type and by ipcNvlsFabricUsable() to gauge usability.
CUmemAllocationHandleType detail::getIpcNvlsMemHandleType()
{
    int device_id;
    TLLM_CUDA_CHECK(cudaGetDevice(&device_id));

    // Check if fabric handle support is available.
    int fabric_supported = 0;
    CUCHECK(cuDeviceGetAttribute(&fabric_supported, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, device_id));
    if (!fabric_supported)
    {
        TLLM_LOG_TRACE("checking fabric support... CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED not supported.");
        return CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    }

    auto nvml = tensorrt_llm::common::NVMLWrapper::getInstance();
    tensorrt_llm::common::NvmlManager nvmlManager;

    nvmlDevice_t nvml_device;
    NVMLCHECK(nvml->nvmlDeviceGetHandleByIndex(device_id, &nvml_device));

    nvmlGpuFabricState_t fabric_state;
    nvmlReturn_t fabric_status;
    if (nvml->hasGpuFabricInfoV())
    {
        nvmlGpuFabricInfoV_t fabric_info_v;
        memset(&fabric_info_v, 0, sizeof(fabric_info_v));
        fabric_info_v.version = nvmlGpuFabricInfo_v2;
        NVMLCHECK(nvml->nvmlDeviceGetGpuFabricInfoV(nvml_device, &fabric_info_v));
        fabric_state = fabric_info_v.state;
        fabric_status = fabric_info_v.status;
    }
    else if (nvml->hasGpuFabricInfo())
    {
        nvmlGpuFabricInfo_t fabric_info;
        NVMLCHECK(nvml->nvmlDeviceGetGpuFabricInfo(nvml_device, &fabric_info));
        fabric_state = fabric_info.state;
        fabric_status = fabric_info.status;
    }
    else
    {
        TLLM_LOG_TRACE("checking fabric support... NVML fabric info APIs not available.");
        return CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    }

    // Check if the fabric is fully initialized.
    if (fabric_state != NVML_GPU_FABRIC_STATE_COMPLETED || fabric_status != NVML_SUCCESS)
    {
        TLLM_LOG_TRACE("checking fabric support... fabric state is NOT COMPLETE: state=%u status=%u.", fabric_state,
            fabric_status);
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

    // Check if fabric handles can be exported & imported by IMEX (Internode Memory Exchange).
    CUmemFabricHandle fh;
    CUmemGenericAllocationHandle imported_handle;
    err = cuMemExportToShareableHandle(&fh, handle, CU_MEM_HANDLE_TYPE_FABRIC, 0);
    if (err != CUDA_SUCCESS
        || (err = cuMemImportFromShareableHandle(&imported_handle, &fh, CU_MEM_HANDLE_TYPE_FABRIC)) != CUDA_SUCCESS)
    {
        TLLM_LOG_TRACE("checking fabric support... cuMemExport/cuMemImport failed.");
        CUCHECK(cuMemRelease(handle));
        return CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    }

    TLLM_LOG_TRACE("fabric status: device=%d, state=%u status=%u", device_id, fabric_state, fabric_status);

    CUCHECK(cuMemRelease(imported_handle));
    CUCHECK(cuMemRelease(handle));
    // If we get here, fabric handles are supported.
    return CU_MEM_HANDLE_TYPE_FABRIC;
}

class NVLSCudaAllocator
{
public:
    static IpcNvlsHandle* allocate(size_t size, std::shared_ptr<IpcCommunicator> const& communicator)
    {
        TLLM_CHECK_WITH_INFO(communicator != nullptr, "IPC communicator must not be null");
        auto nvlsHandle = new IpcNvlsHandle();

        int deviceId;
        int cuDevice;
        TLLM_CUDA_CHECK(cudaGetDevice(&deviceId));
        CUCHECK(cuDeviceGet(&cuDevice, deviceId));

        auto const handleType = communicator->getMemHandleType();
        auto const groupRank = communicator->getGroupRank();
        auto const groupSize = communicator->getGroupSize();
        TLLM_CHECK_WITH_INFO(groupSize >= 2, "NVLS allocation requires at least two ranks");

        CUmemAccessDesc accessDesc{};
        accessDesc.location.id = deviceId;
        accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

        CUmemAllocationProp prop{};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = deviceId;
        prop.requestedHandleTypes = handleType;

        CUmulticastObjectProp multicastProp{};
        multicastProp.numDevices = groupSize;
        multicastProp.handleTypes = handleType;

        size_t granularity = 0;
        CUCHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
        size = ROUND_UP(size, granularity);

        size_t multicastGranularity = 0;
        CUCHECK(cuMulticastGetGranularity(&multicastGranularity, &multicastProp, CU_MULTICAST_GRANULARITY_MINIMUM));
        size = ROUND_UP(size, multicastGranularity);
        multicastProp.size = size;
        nvlsHandle->size = size;

        CUCHECK(cuMemCreate(&nvlsHandle->uc_handle, size, &prop, 0));
        CUCHECK(cuMemAddressReserve(&nvlsHandle->uc_va, size, granularity, 0U, 0));
        CUCHECK(cuMemMap(nvlsHandle->uc_va, size, 0, nvlsHandle->uc_handle, 0));
        CUCHECK(cuMemSetAccess(nvlsHandle->uc_va, size, &accessDesc, 1));
        nvlsHandle->uc_ptr = reinterpret_cast<uintptr_t>((void*) nvlsHandle->uc_va);

        IpcMemHandle ipcHandle;
        CUCHECK(cuMemExportToShareableHandle((void*) &ipcHandle, nvlsHandle->uc_handle, handleType, 0));

        nvlsHandle->ipc_uc_ptrs.resize(groupSize);
        nvlsHandle->ipc_uc_vas.resize(groupSize);
        nvlsHandle->ipc_uc_handles.resize(groupSize);

        for (int rank = 0; rank < groupSize; ++rank)
        {
            IpcMemHandle peerHandle = ipcHandle;
            communicator->bcastMemHandle(&peerHandle, rank);
            if (rank != groupRank)
            {
                void* osHandle = handleType == CU_MEM_HANDLE_TYPE_FABRIC ? (void*) &peerHandle : (void*) peerHandle.fd;
                CUCHECK(cuMemImportFromShareableHandle(&nvlsHandle->ipc_uc_handles[rank], osHandle, handleType));
                CUCHECK(cuMemAddressReserve(&nvlsHandle->ipc_uc_vas[rank], size, granularity, 0U, 0));
                CUCHECK(cuMemMap(nvlsHandle->ipc_uc_vas[rank], size, 0, nvlsHandle->ipc_uc_handles[rank], 0));
                CUCHECK(cuMemSetAccess(nvlsHandle->ipc_uc_vas[rank], size, &accessDesc, 1));
                nvlsHandle->ipc_uc_ptrs[rank] = reinterpret_cast<uintptr_t>((void*) nvlsHandle->ipc_uc_vas[rank]);
            }
            else
            {
                nvlsHandle->ipc_uc_ptrs[rank] = nvlsHandle->uc_ptr;
                nvlsHandle->ipc_uc_vas[rank] = nvlsHandle->uc_va;
                nvlsHandle->ipc_uc_handles[rank] = nvlsHandle->uc_handle;
            }
        }

        if (groupRank == 0)
        {
            CUCHECK(cuMulticastCreate(&nvlsHandle->mc_handle, &multicastProp));
            CUCHECK(cuMemExportToShareableHandle(&ipcHandle, nvlsHandle->mc_handle, handleType, 0));
            communicator->bcastMemHandle(&ipcHandle, 0);
        }
        else
        {
            communicator->bcastMemHandle(&ipcHandle, 0);
            void* osHandle = handleType == CU_MEM_HANDLE_TYPE_FABRIC ? (void*) &ipcHandle : (void*) ipcHandle.fd;
            CUCHECK(cuMemImportFromShareableHandle(&nvlsHandle->mc_handle, osHandle, handleType));
        }

        CUCHECK(cuMulticastAddDevice(nvlsHandle->mc_handle, cuDevice));
        CUCHECK(cuMulticastBindMem(nvlsHandle->mc_handle, 0, nvlsHandle->uc_handle, 0, size, 0));
        CUCHECK(cuMemAddressReserve(&nvlsHandle->mc_va, size, multicastGranularity, 0U, 0));
        CUCHECK(cuMemMap(nvlsHandle->mc_va, size, 0, nvlsHandle->mc_handle, 0));
        CUCHECK(cuMemSetAccess(nvlsHandle->mc_va, size, &accessDesc, 1));
        nvlsHandle->mc_ptr = reinterpret_cast<uintptr_t>((void*) nvlsHandle->mc_va);

        return nvlsHandle;
    }

    static void free(IpcNvlsHandle* nvlsHandle)
    {
        CUCHECK(cuMemUnmap(nvlsHandle->mc_va, nvlsHandle->size));
        CUCHECK(cuMemRelease(nvlsHandle->mc_handle));
        CUCHECK(cuMemAddressFree(nvlsHandle->mc_va, nvlsHandle->size));
        for (size_t i = 0; i < nvlsHandle->ipc_uc_vas.size(); ++i)
        {
            CUCHECK(cuMemUnmap(nvlsHandle->ipc_uc_vas[i], nvlsHandle->size));
            CUCHECK(cuMemRelease(nvlsHandle->ipc_uc_handles[i]));
            CUCHECK(cuMemAddressFree(nvlsHandle->ipc_uc_vas[i], nvlsHandle->size));
        }
    }
};

IpcNvlsHandle* detail::ipcNvlsAllocateWithCommunicator(size_t size, std::shared_ptr<IpcCommunicator> communicator)
{
    return NVLSCudaAllocator::allocate(size, communicator);
}
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
    // Static capability check only (driver version + the multicast attribute on
    // every device). This is the precondition for allocating NVLS multicast
    // memory: ipcNvlsAllocate() selects a fabric or POSIX-FD handle itself, so
    // single-node NVLS works without a provisioned fabric/IMEX plane. The result
    // is static for the process, so compute it once and cache it.
    static bool const supported = []() -> bool
    {
        int cuda_driver_version = -1;
        TLLM_CUDA_CHECK(cudaDriverGetVersion(&cuda_driver_version));
        if (cuda_driver_version < 12010)
        {
            TLLM_LOG_DEBUG("CUDA Driver version < 12010");
            return false;
        }
        int dev_count = 0;
        TLLM_CUDA_CHECK(cudaGetDeviceCount(&dev_count));
        for (int i = 0; i < dev_count; ++i)
        {
            int cuda_dev = -1;
            CUdevice current_dev;
            TLLM_CUDA_CHECK(cudaGetDevice(&cuda_dev));
            CUCHECK(cuDeviceGet(&current_dev, cuda_dev));
            int multicast_supported = 0;
            CUCHECK(cuDeviceGetAttribute(&multicast_supported, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, current_dev));
            if (!multicast_supported)
            {
                TLLM_LOG_DEBUG("CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED not supported on GPU%d.", cuda_dev);
                return false;
            }
        }
        return true;
    }();
    return supported;
#else
    return false;
#endif
}

bool ipcNvlsFabricUsable()
{
#if ENABLE_MULTI_DEVICE
    // Extends ipcNvlsSupported() with a live fabric probe. The probe is
    // relatively heavy (it allocates and exports fabric-handle memory), so
    // compute it once and cache it.
    static bool const usable = []() -> bool
    {
        if (!ipcNvlsSupported())
        {
            return false;
        }
#if !ENABLE_NVSHMEM
        // The multicast attribute is a false positive when the fabric/IMEX plane
        // is not provisioned (e.g. nvidia-imex not running): NVLS multicast
        // cannot actually be bound there. detail::getIpcNvlsMemHandleType() does the real test
        // -- it resolves to FABRIC only if fabric-handle memory can be allocated
        // and exported. On any probe error, assume usable so an unrelated
        // failure never disables a healthy NVLS setup.
        try
        {
            if (detail::getIpcNvlsMemHandleType() != CU_MEM_HANDLE_TYPE_FABRIC)
            {
                TLLM_LOG_WARNING(
                    "\n"
                    "**************************************************************************\n"
                    "* NVLS (NVLink SHARP) DISABLED for NCCL -- falling back to NVLink P2P     *\n"
                    "**************************************************************************\n"
                    "* The GPU advertises multicast support, but the NVLink fabric/IMEX plane *\n"
                    "* is NOT provisioned on this node, so NVLS multicast memory cannot be    *\n"
                    "* bound. Collectives will still work over NVLink P2P, but NVLS-           *\n"
                    "* accelerated NCCL is off and performance may be reduced. (Single-node   *\n"
                    "* NVLS over POSIX-FD, e.g. MNNVL allreduce, is unaffected.)              *\n"
                    "*                                                                        *\n"
                    "* To enable NVLS: start nvidia-imex and expose                           *\n"
                    "* /dev/nvidia-caps-imex-channels to the container. Set                   *\n"
                    "* NCCL_NVLS_ENABLE=1 explicitly to override this fallback.               *\n"
                    "**************************************************************************");
                return false;
            }
        }
        catch (std::exception const& e)
        {
            TLLM_LOG_DEBUG("NVLS fabric probe could not run, assuming usable: %s", e.what());
        }
#endif // !ENABLE_NVSHMEM
        return true;
    }();
    return usable;
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
    int worldRank;
    int groupRank;
    MPI_Comm_rank(COMM_SESSION, &worldRank);
    MPI_Comm_rank(new_comm, &groupRank);

    auto const handleType = detail::getIpcNvlsMemHandleType();
    std::shared_ptr<IpcCommunicator> communicator;
    if (handleType == CU_MEM_HANDLE_TYPE_FABRIC)
    {
        communicator = std::make_shared<IpcFabricCommunicator>(new_comm, groupRank, group_size);
    }
    else
    {
        communicator = std::make_shared<IpcSocketCommunicator>(worldRank, groupRank, ranks, new_comm);
    }
    auto handle = detail::ipcNvlsAllocateWithCommunicator(size, std::move(communicator));
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

namespace
{
class MpiIpcNvlsRendezvous final : public IpcNvlsRendezvous
{
public:
    explicit MpiIpcNvlsRendezvous(std::set<int> ranks)
        : mRanks(std::move(ranks))
    {
        TLLM_CHECK_WITH_INFO(!mRanks.empty(), "MPI NVLS rendezvous group must not be empty");
    }

    IpcNvlsHandle* allocate(size_t bytes) const override
    {
        return ipcNvlsAllocate(bytes, mRanks);
    }

    void barrier() const override
    {
        if (mRanks.size() > 1)
        {
            MPI_group_barrier(mRanks);
        }
    }

    [[nodiscard]] int rank() const override
    {
        auto const worldRank = COMM_SESSION.getRank();
        auto const it = mRanks.find(worldRank);
        TLLM_CHECK_WITH_INFO(it != mRanks.end(), "MPI rank %d is not in the NVLS rendezvous group", worldRank);
        return static_cast<int>(std::distance(mRanks.begin(), it));
    }

    [[nodiscard]] int size() const override
    {
        return static_cast<int>(mRanks.size());
    }

    [[nodiscard]] uintptr_t identity() const override
    {
        uintptr_t seed = 0;
        for (auto const rank : mRanks)
        {
            seed ^= std::hash<int>{}(rank) + 0x9e3779b9U + (seed << 6) + (seed >> 2);
        }
        return seed << 1;
    }

    [[nodiscard]] IpcNvlsRendezvousKind kind() const override
    {
        return IpcNvlsRendezvousKind::kMPI;
    }

    [[nodiscard]] std::string describe() const override
    {
        std::stringstream stream;
        stream << "MPI[";
        for (auto const rank : mRanks)
        {
            stream << rank << ',';
        }
        stream << ']';
        return stream.str();
    }

private:
    std::set<int> mRanks;
};
} // namespace

IpcNvlsRendezvousPtr makeMpiIpcNvlsRendezvous(std::set<int> ranks)
{
    return std::make_shared<MpiIpcNvlsRendezvous>(std::move(ranks));
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
