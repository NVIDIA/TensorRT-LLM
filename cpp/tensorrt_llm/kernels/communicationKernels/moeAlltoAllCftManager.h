/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

// ============================================================================
// CFT (Compute Fabric Transport) LE Manager for MoE AlltoAll
//
// Manages the lifecycle of Logical Endpoints (LEs) for CFT handle-based
// counted writes. Each rank creates one unicast LE per peer rank, bound to
// its recv buffer in MNNVL workspace. LE IDs are exchanged across ranks so
// that the dispatch kernel can use fabric.try_put.counted to write directly
// to peer LEs.
//
// LE lifecycle:
//   1. loadApis()           — cuGetProcAddress for all LE APIs
//   2. createEndpoints()    — reserve IDs, create LEs, alloc fabric mem, bind
//   3. exchangeEndpoints()  — export local LE, import all peers' LEs
//   4. getLeId(target_rank) — kernel uses this to address puts
//   5. destroy()            — unbind, destroy, release
//
// Requirements:
//   - CUDA 13.4+ headers (LE type definitions)
//   - CUDA driver with LE support
//   - IMEX daemon running (fabric memory)
//   - NVSwitch fabric (Fabric State: Completed)
// ============================================================================

#include "tensorrt_llm/common/config.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cuda.h>
#include <thread>
#include <vector>

#include "tensorrt_llm/kernels/communicationKernels/moeAlltoAllCftSupport.h"

#if TLLM_CFT_HAS_CUDA_13_4_SUPPORT
// Function pointer types matching real CUDA 13.4 LE API signatures
// (loaded at runtime via cuGetProcAddress for driver compatibility)
typedef CUresult (*PFN_cuLeIdReserve)(CUlogicalEndpointId*, cuuint32_t);
typedef CUresult (*PFN_cuLeIdRelease)(CUlogicalEndpointId, cuuint32_t);
typedef CUresult (*PFN_cuLeCreate)(CUlogicalEndpointId, CUlogicalEndpointProp const*);
typedef CUresult (*PFN_cuLeDestroy)(CUlogicalEndpointId);
typedef CUresult (*PFN_cuLeBindMem)(CUlogicalEndpointId, CUdevice, cuuint64_t, CUmemGenericAllocationHandle, cuuint64_t,
    cuuint64_t, unsigned long long);
typedef CUresult (*PFN_cuLeUnbind)(CUlogicalEndpointId, CUdevice, cuuint64_t, cuuint64_t);
typedef CUresult (*PFN_cuLeQuery)(CUlogicalEndpointId, cuuint32_t, int*);
typedef CUresult (*PFN_cuLeExport)(void*, CUlogicalEndpointId, CUlogicalEndpointIpcHandleType);
typedef CUresult (*PFN_cuLeImport)(CUlogicalEndpointId, void const*, CUlogicalEndpointIpcHandleType);
#endif

// Helper macro for driver API calls — must be defined before use in class methods
#define CU_MUST(call)                                                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        CUresult _r = (call);                                                                                          \
        if (_r != CUDA_SUCCESS)                                                                                        \
        {                                                                                                              \
            char const *_name = nullptr, *_str = nullptr;                                                              \
            cuGetErrorName(_r, &_name);                                                                                \
            cuGetErrorString(_r, &_str);                                                                               \
            fprintf(stderr, "CftLeManager: %s:%d CUDA Error: %s (%s)\n", __FILE__, __LINE__, _name ? _name : "?",      \
                _str ? _str : "");                                                                                     \
            return false;                                                                                              \
        }                                                                                                              \
    } while (0)

TRTLLM_NAMESPACE_BEGIN

namespace kernels::moe_comm
{

#if TLLM_CFT_HAS_CUDA_13_4_SUPPORT

// Per-rank LE state: one unicast LE representing this rank's recv buffer
struct RankLE
{
    CUlogicalEndpointId leId = 0;               //!< Logical endpoint ID for this rank's recv buffer.
    CUmemGenericAllocationHandle memHandle = 0; //!< Generic allocation handle backing the bound fabric memory.
    CUdeviceptr backingPtr = 0;                 //!< Device pointer for the memory bound to the LE.
    size_t allocSize = 0;                       //!< Size in bytes of the LE-backed allocation.
    bool ownsMemory = true;                     //!< False when bound to external workspace memory.
    bool leCreated = false;                     //!< True once the LE exists and must be destroyed.
    bool memBound = false;                      //!< True once memory is bound and must be unbound.
    bool valid = false;                         //!< True once the LE reports ready.
};

struct LeIdBlock
{
    CUlogicalEndpointId base = 0; //!< First logical endpoint ID reserved for this rank.
    unsigned int count = 0;       //!< Number of consecutive logical endpoint IDs in the reservation.
    bool reserved = false;        //!< True while the ID block must be released during cleanup.
};

// ============================================================================
// CftLeManager — manages LE lifecycle for MoE AlltoAll
// ============================================================================
class CftLeManager
{
public:
    CftLeManager() = default;

    ~CftLeManager()
    {
        destroy();
    }

    // Step 1: Load LE APIs from driver via cuGetProcAddress.
    // Returns false if driver doesn't support LE APIs.
    bool loadApis()
    {
        struct
        {
            char const* name;
            void** ptr;
        } apis[] = {
            {"cuLogicalEndpointIdReserve", (void**) &pfnIdReserve_},
            {"cuLogicalEndpointIdRelease", (void**) &pfnIdRelease_},
            {"cuLogicalEndpointCreate", (void**) &pfnCreate_},
            {"cuLogicalEndpointDestroy", (void**) &pfnDestroy_},
            {"cuLogicalEndpointBindMem", (void**) &pfnBindMem_},
            {"cuLogicalEndpointUnbind", (void**) &pfnUnbind_},
            {"cuLogicalEndpointQuery", (void**) &pfnQuery_},
            {"cuLogicalEndpointExport", (void**) &pfnExport_},
            {"cuLogicalEndpointImport", (void**) &pfnImport_},
        };

        for (auto& a : apis)
        {
            CUdriverProcAddressQueryResult status;
            CUresult r = cuGetProcAddress(a.name, a.ptr, 13030, CU_GET_PROC_ADDRESS_DEFAULT, &status);
            if (r != CUDA_SUCCESS || *a.ptr == nullptr)
            {
                fprintf(stderr, "CftLeManager: MISSING API: %s\n", a.name);
                return false;
            }
        }
        apisLoaded_ = true;
        return true;
    }

    // Step 2: Create this rank's unicast LE and bind it to fabric memory.
    //
    // The LE represents this rank's recv buffer region. Other ranks will
    // write into it using fabric.try_put.counted with the exported LE ID.
    //
    // Args:
    //   deviceIdx: CUDA device index for this rank
    //   totalSize: total LE-backed allocation size in bytes
    //   epRank, epSize: EP topology
    bool createEndpoint(int deviceIdx, size_t totalSize, int epRank, int epSize)
    {
        if (!apisLoaded_)
            return false;

        deviceIdx_ = deviceIdx;
        epRank_ = epRank;
        epSize_ = epSize;

        CUdevice cuDevice;
        CU_MUST(cuDeviceGet(&cuDevice, deviceIdx));
        cuDevice_ = cuDevice;

        CUcontext ctx;
        CU_MUST(cuDevicePrimaryCtxRetain(&ctx, cuDevice));
        CU_MUST(cuCtxSetCurrent(ctx));

        // Allocate fabric memory
        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = deviceIdx;
        prop.requestedHandleTypes = (CUmemAllocationHandleType) CU_MEM_HANDLE_TYPE_FABRIC;

        size_t granularity = 0;
        CU_MUST(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
        localLe_.allocSize = ((totalSize + granularity - 1) / granularity) * granularity;

        CU_MUST(cuMemCreate(&localLe_.memHandle, localLe_.allocSize, &prop, 0));
        CU_MUST(cuMemAddressReserve(&localLe_.backingPtr, localLe_.allocSize, granularity, 0, 0));
        CU_MUST(cuMemMap(localLe_.backingPtr, localLe_.allocSize, 0, localLe_.memHandle, 0));

        CUmemAccessDesc access = {};
        access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access.location.id = deviceIdx;
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        CU_MUST(cuMemSetAccess(localLe_.backingPtr, localLe_.allocSize, &access, 1));

        CU_MUST(pfnIdReserve_(&leIdBlock_.base, static_cast<unsigned int>(epSize)));
        leIdBlock_.count = static_cast<unsigned int>(epSize);
        leIdBlock_.reserved = true;

        localLe_.leId = leIdBlock_.base + static_cast<unsigned int>(epRank);

        CUlogicalEndpointProp leProp = {};
        leProp.type = CU_LOGICAL_ENDPOINT_TYPE_UNICAST;
        leProp.size = localLe_.allocSize;
        leProp.unicast.device = cuDevice;
        leProp.ipcHandleTypes = (unsigned int) CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_FABRIC;
        leProp.flags = CU_LOGICAL_ENDPOINT_FLAG_COUNTED_OPS;

        CU_MUST(pfnCreate_(localLe_.leId, &leProp));
        localLe_.leCreated = true;

        // Wait for the local LE before binding memory to it.
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (std::chrono::steady_clock::now() < deadline)
        {
            int ready = 0;
            CU_MUST(pfnQuery_(localLe_.leId, 1, &ready));
            if (ready)
            {
                localLe_.valid = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        if (!localLe_.valid)
        {
            fprintf(stderr, "CftLeManager: LE not ready after 10s on device %d\n", deviceIdx);
            return false;
        }

        CU_MUST(pfnBindMem_(localLe_.leId, cuDevice, 0, localLe_.memHandle, 0, localLe_.allocSize, 0));
        localLe_.memBound = true;

        return true;
    }

    // Step 2b (alternative): Create LE bound to external workspace memory.
    //
    // Instead of allocating separate fabric memory, bind the LE to the MNNVL
    // workspace's memory handle. The LE then addresses the workspace directly —
    // fabric.try_put.counted writes land in workspace recv_buffers, eliminating
    // the duplicate allocation and the need to know payload layout at init time.
    //
    // Args:
    //   deviceIdx: CUDA device index for this rank
    //   externalHandle: CUmemGenericAllocationHandle of the workspace (from cuMemCreate)
    //   externalPtr: VA pointer to this rank's workspace region
    //   totalSize: size of the workspace per rank (will be rounded up to granularity)
    //   epRank, epSize: EP topology
    bool createEndpointExternal(int deviceIdx, CUmemGenericAllocationHandle externalHandle, CUdeviceptr externalPtr,
        size_t totalSize, int epRank, int epSize)
    {
        if (!apisLoaded_)
            return false;

        deviceIdx_ = deviceIdx;
        epRank_ = epRank;
        epSize_ = epSize;

        CUdevice cuDevice;
        CU_MUST(cuDeviceGet(&cuDevice, deviceIdx));
        cuDevice_ = cuDevice;

        CUcontext ctx;
        CU_MUST(cuDevicePrimaryCtxRetain(&ctx, cuDevice));
        CU_MUST(cuCtxSetCurrent(ctx));

        // Use external memory — no allocation needed.
        // Round up to granularity for LE size (same as cuMemCreate granularity).
        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = deviceIdx;
        prop.requestedHandleTypes = (CUmemAllocationHandleType) CU_MEM_HANDLE_TYPE_FABRIC;

        size_t granularity = 0;
        CU_MUST(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
        localLe_.allocSize = ((totalSize + granularity - 1) / granularity) * granularity;
        localLe_.backingPtr = externalPtr;
        localLe_.memHandle = 0; // not owned
        localLe_.ownsMemory = false;

        CU_MUST(pfnIdReserve_(&leIdBlock_.base, static_cast<unsigned int>(epSize)));
        leIdBlock_.count = static_cast<unsigned int>(epSize);
        leIdBlock_.reserved = true;

        localLe_.leId = leIdBlock_.base + static_cast<unsigned int>(epRank);

        CUlogicalEndpointProp leProp = {};
        leProp.type = CU_LOGICAL_ENDPOINT_TYPE_UNICAST;
        leProp.size = localLe_.allocSize;
        leProp.unicast.device = cuDevice;
        leProp.ipcHandleTypes = (unsigned int) CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_FABRIC;
        leProp.flags = CU_LOGICAL_ENDPOINT_FLAG_COUNTED_OPS;

        CU_MUST(pfnCreate_(localLe_.leId, &leProp));
        localLe_.leCreated = true;

        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (std::chrono::steady_clock::now() < deadline)
        {
            int ready = 0;
            CU_MUST(pfnQuery_(localLe_.leId, 1, &ready));
            if (ready)
            {
                localLe_.valid = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        if (!localLe_.valid)
        {
            fprintf(stderr, "CftLeManager: LE not ready after 10s on device %d\n", deviceIdx);
            return false;
        }

        CU_MUST(pfnBindMem_(localLe_.leId, cuDevice, 0, externalHandle, 0, localLe_.allocSize, 0));
        localLe_.memBound = true;

        return true;
    }

    // Step 3: Exchange LE handles across all ranks.
    //
    // Each rank exports its local LE handle, then all ranks do an allgather
    // to exchange handles. Each rank then imports all peer handles to get
    // LE IDs it can use in fabric.try_put.
    //
    // Args:
    //   epRank: this rank's index
    //   epSize: total number of ranks
    //   allgatherFn: callback to perform MPI/NCCL allgather of IPC handles
    //                signature: void(const void* sendBuf, void* recvBuf, size_t bytesPerRank)
    template <typename AllgatherFn>
    bool exchangeEndpoints(AllgatherFn allgatherFn)
    {
        peerLeIds_.resize(epSize_);

        // Export local LE handle (fabric type)
        CUlogicalEndpointFabricHandle localHandle = {};
        CU_MUST(pfnExport_(&localHandle, localLe_.leId, CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_FABRIC));

        // Allgather handles across all ranks
        std::vector<CUlogicalEndpointFabricHandle> allHandles(epSize_);
        allgatherFn(&localHandle, allHandles.data(), sizeof(CUlogicalEndpointFabricHandle));

        // Import each peer handle into its contiguous slot and wait for the full block.
        for (int r = 0; r < epSize_; r++)
        {
            if (r == epRank_)
            {
                peerLeIds_[r] = localLe_.leId;
            }
            else
            {
                CUlogicalEndpointId peerLeId = leIdBlock_.base + static_cast<unsigned int>(r);
                CU_MUST(pfnImport_(peerLeId, &allHandles[r], CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_FABRIC));
                peerLeIds_[r] = peerLeId;
            }
        }

        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        bool allLeReady = false;
        while (std::chrono::steady_clock::now() < deadline)
        {
            int ready = 0;
            CU_MUST(pfnQuery_(leIdBlock_.base, leIdBlock_.count, &ready));
            if (ready)
            {
                allLeReady = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (!allLeReady)
        {
            fprintf(stderr, "CftLeManager: Not all LEs ready after 10s (rank %d, %d LEs)\n", epRank_, epSize_);
            return false;
        }

        initialized_ = true;
        return true;
    }

    // Get LE ID for writing to a target rank.
    // Used by the dispatch kernel: fabric.try_put.counted(getLeId(target), offset, ...)
    CUlogicalEndpointId getLeId(int targetRank) const
    {
        return peerLeIds_[targetRank];
    }

    // Get all LE IDs as a contiguous array (for passing to kernel).
    CUlogicalEndpointId const* getAllLeIds() const
    {
        return peerLeIds_.data();
    }

    // Get the backing pointer for this rank's LE memory (for local reads).
    CUdeviceptr getLocalBackingPtr() const
    {
        return localLe_.backingPtr;
    }

    size_t getAllocSize() const
    {
        return localLe_.allocSize;
    }

    bool isInitialized() const
    {
        return initialized_;
    }

    void destroy()
    {
        // Imported peer LEs reuse the reserved contiguous ID block.
        for (size_t r = 0; r < peerLeIds_.size(); r++)
        {
            if (static_cast<int>(r) != epRank_ && peerLeIds_[r] != 0)
            {
                pfnDestroy_(peerLeIds_[r]);
            }
        }
        peerLeIds_.clear();

        // Destroy local LE
        if (localLe_.memBound)
        {
            pfnUnbind_(localLe_.leId, cuDevice_, 0, localLe_.allocSize);
        }
        if (localLe_.leCreated)
        {
            pfnDestroy_(localLe_.leId);
        }
        if (leIdBlock_.reserved)
        {
            pfnIdRelease_(leIdBlock_.base, leIdBlock_.count);
            leIdBlock_.reserved = false;
        }

        // Free fabric memory only if we own it (not for external workspace binding)
        if (localLe_.ownsMemory)
        {
            if (localLe_.backingPtr)
            {
                cuMemUnmap(localLe_.backingPtr, localLe_.allocSize);
                cuMemAddressFree(localLe_.backingPtr, localLe_.allocSize);
            }
            if (localLe_.memHandle)
                cuMemRelease(localLe_.memHandle);
        }

        localLe_ = RankLE{};
        initialized_ = false;
    }

private:
    bool apisLoaded_ = false;
    bool initialized_ = false;
    int deviceIdx_ = -1;
    int epRank_ = -1;
    int epSize_ = 0;
    CUdevice cuDevice_;

    RankLE localLe_;
    LeIdBlock leIdBlock_;
    std::vector<CUlogicalEndpointId> peerLeIds_;

    // Function pointers loaded via cuGetProcAddress
    PFN_cuLeIdReserve pfnIdReserve_ = nullptr;
    PFN_cuLeIdRelease pfnIdRelease_ = nullptr;
    PFN_cuLeCreate pfnCreate_ = nullptr;
    PFN_cuLeDestroy pfnDestroy_ = nullptr;
    PFN_cuLeBindMem pfnBindMem_ = nullptr;
    PFN_cuLeUnbind pfnUnbind_ = nullptr;
    PFN_cuLeQuery pfnQuery_ = nullptr;
    PFN_cuLeExport pfnExport_ = nullptr;
    PFN_cuLeImport pfnImport_ = nullptr;
};

#else

class CftLeManager
{
public:
    bool loadApis()
    {
        fprintf(stderr, "CftLeManager: CUDA logical endpoint APIs require CUDA 13.4+ headers.\n");
        return false;
    }

    bool createEndpoint(int, size_t, int, int)
    {
        return false;
    }

    bool createEndpointExternal(int, CUmemGenericAllocationHandle, CUdeviceptr, size_t, int, int)
    {
        return false;
    }

    template <typename AllgatherFn>
    bool exchangeEndpoints(AllgatherFn)
    {
        return false;
    }

    uint32_t getLeId(int targetRank) const
    {
        return peerLeIds_[targetRank];
    }

    uint32_t const* getAllLeIds() const
    {
        return peerLeIds_.data();
    }

    CUdeviceptr getLocalBackingPtr() const
    {
        return 0;
    }

    size_t getAllocSize() const
    {
        return 0;
    }

    bool isInitialized() const
    {
        return false;
    }

    void destroy() {}

private:
    std::vector<uint32_t> peerLeIds_;
};

#endif // TLLM_CFT_HAS_CUDA_13_4_SUPPORT

} // namespace kernels::moe_comm

TRTLLM_NAMESPACE_END

#undef CU_MUST
