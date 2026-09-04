/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
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
#include <cuda.h>

// Rest of includes
#include "mcastDeviceMemory.h"
#include "tensorrt_llm/common/cudaDriverWrapper.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <exception>
#include <string>
#include <utility>

namespace tensorrt_llm::runtime
{

namespace
{
// An efficient implementation assuming gran is a power of 2
inline size_t roundUp(size_t val, size_t gran)
{
    return (val + gran - 1) & ~(gran - 1);
}

template <typename Cleanup>
bool cleanupNoThrow(char const* resource, Cleanup cleanup) noexcept
{
    try
    {
        cleanup();
        return true;
    }
    catch (std::exception const& error)
    {
        try
        {
            TLLM_LOG_WARNING("Failed to release %s during McastDeviceMemory cleanup: %s", resource, error.what());
        }
        catch (...)
        {
            // Logging must not make destruction terminate during static teardown.
        }
        return false;
    }
    catch (...)
    {
        try
        {
            TLLM_LOG_WARNING("Failed to release %s during McastDeviceMemory cleanup", resource);
        }
        catch (...)
        {
            // Logging must not make destruction terminate during static teardown.
        }
        return false;
    }
}

template <typename Action>
void runCollectivePhase(
    tensorrt_llm::mpi::MpiComm const& comm, char const* phase, Action action, bool failureIsTerminal = true)
{
    std::exception_ptr localError;
    try
    {
        action();
    }
    catch (...)
    {
        localError = std::current_exception();
    }

    int32_t const localSuccess{localError == nullptr ? 1 : 0};
    int32_t allSucceeded{0};
    comm.allreduce(&localSuccess, &allSucceeded, 1, mpi::MpiType::kINT32, mpi::MpiOp::MIN);
    if (allSucceeded != 0)
    {
        return;
    }

    if (localError != nullptr)
    {
        try
        {
            std::rethrow_exception(localError);
        }
        catch (std::exception const& error)
        {
            TLLM_LOG_ERROR("[McastDeviceMemory] %s failed on group rank %d: %s", phase, comm.getRank(), error.what());
        }
        catch (...)
        {
            TLLM_LOG_ERROR("[McastDeviceMemory] %s failed on group rank %d", phase, comm.getRank());
        }
    }
    if (failureIsTerminal)
    {
        TLLM_THROW(
            "[McastDeviceMemory] %s failed on one or more ranks; the object is unusable and the process must be "
            "torn down.",
            phase);
    }
    TLLM_THROW(
        "[McastDeviceMemory] %s failed on one or more ranks before restore mutation; the object remains "
        "unmapped and the caller may retry with a valid communicator.",
        phase);
}
} // namespace

McastDeviceMemory::McastDeviceMemory(
    size_t bufSize, uint32_t groupSize, uint32_t groupRank, int deviceIdx, bool mnNvlink, int64_t mpiCommFortranHandle)
    : mIsMNNvlink(mnNvlink)
    , mDeviceIdx(deviceIdx)
    , mGroupSize(groupSize)
    , mGroupRank(groupRank)
    , mBufSize(bufSize)
    , mSignalPadOffset(0)
    , mAllocationSize(0)
    , mMcPtr(0)
    , mUcBasePtr(0)
    , mMcHandle(0)
#if ENABLE_MULTI_DEVICE
    , mGroupComm(std::in_place, MPI_Comm_f2c(mpiCommFortranHandle), false)
#else
    , mGroupComm(std::in_place, nullptr, false)
#endif
    , mGroupWorldRanks(tensorrt_llm::mpi::getWorldRanks(*mGroupComm))
    , mUcPtrsDev(nullptr)
    , mSignalPadsDev(nullptr)
    , mState(State::kUnmapped)
    , mMcMapped(false)
    , mMcBound(false)
    , mNvlsHandle(nullptr)
{
    runCollectivePhase(*mGroupComm, "device capability validation",
        [this]()
        {
            TLLM_CHECK_WITH_INFO(mGroupRank < mGroupSize && mGroupComm->getRank() == static_cast<int>(mGroupRank)
                    && mGroupComm->getSize() == static_cast<int>(mGroupSize),
                "[McastDeviceMemory] Constructor communicator does not match the supplied rank or group size.");
            TLLM_CUDA_CHECK(cudaSetDevice(mDeviceIdx));
            int multicastSupported{0};
            TLLM_CU_CHECK(
                cuDeviceGetAttribute(&multicastSupported, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, mDeviceIdx));
            TLLM_CHECK_WITH_INFO(multicastSupported != 0, "[McastDeviceMemory] Device does not support multicasting.");
            if (mIsMNNvlink)
            {
                int fabricHandleSupported{0};
                TLLM_CU_CHECK(cuDeviceGetAttribute(
                    &fabricHandleSupported, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, mDeviceIdx));
                TLLM_CHECK_WITH_INFO(
                    fabricHandleSupported != 0, "[McastDeviceMemory] Device does not support fabric handle.");
            }
        });

    // From pytorch implementation for alignment
    constexpr size_t kSignalPadAlignment = 16UL;
    mSignalPadOffset = roundUp(mBufSize, kSignalPadAlignment);
    int const world_rank{tensorrt_llm::mpi::MpiComm::session().getRank()};

    TLLM_LOG_DEBUG(
        "[McastDeviceMemory] World Rank: %u, Group Rank: %u, Group size: %u, isMultiNode: %d, "
        "device_idx: %d, Signal pad offset: %zu",
        world_rank, mGroupRank, mGroupSize, mIsMNNvlink, mDeviceIdx, mSignalPadOffset);

    if (mIsMNNvlink)
    {
        mState = State::kTransitioning;
        try
        {
            createAndMapMnMcastMem(mBufSize);
            runCollectivePhase(*mGroupComm, "pointer-table initialization", [this]() { initializePointerTables(); });
            mState = State::kMapped;
        }
        catch (...)
        {
            unmapAndReleaseMnMcastMem();
            if (mSignalPadsDev != nullptr
                && cleanupNoThrow(
                    "signal-pad pointer table", [this]() { TLLM_CUDA_CHECK_FREE_RESOURCE(cudaFree(mSignalPadsDev)); }))
            {
                mSignalPadsDev = nullptr;
            }
            if (mUcPtrsDev != nullptr
                && cleanupNoThrow(
                    "unicast pointer table", [this]() { TLLM_CUDA_CHECK_FREE_RESOURCE(cudaFree(mUcPtrsDev)); }))
            {
                mUcPtrsDev = nullptr;
            }
            if (mUcBasePtr != 0
                && cleanupNoThrow("unicast virtual address reservation",
                    [this]()
                    { TLLM_CU_CHECK_FREE_RESOURCE(cuMemAddressFree(mUcBasePtr, mAllocationSize * mGroupSize)); }))
            {
                mUcBasePtr = 0;
            }
            if (mMcPtr != 0
                && cleanupNoThrow("multicast virtual address reservation",
                    [this]() { TLLM_CU_CHECK_FREE_RESOURCE(cuMemAddressFree(mMcPtr, mAllocationSize)); }))
            {
                mMcPtr = 0;
            }
            mState = State::kBroken;
            throw;
        }
    }
    else
    {
        allocNvlsMcastMem(mSignalPadOffset + kSIGNAL_PAD_SIZE);
        initializePointerTables();
    }
}

void McastDeviceMemory::initializePointerTables()
{
    // Initialize signal pads and rebuild graph-independent peer pointer tables.
    mSignalPads.resize(mGroupSize);
    for (size_t i = 0; i < mGroupSize; i++)
    {
        mSignalPads[i] = mUcPtrs[i] + mSignalPadOffset;
        if (i == mGroupRank)
        {
            TLLM_CU_CHECK(cuMemsetD8(mSignalPads[i], 0, kSIGNAL_PAD_SIZE));
        }
    }
    if (mSignalPadsDev == nullptr)
    {
        TLLM_CUDA_CHECK(cudaMalloc(&mSignalPadsDev, mGroupSize * sizeof(CUdeviceptr)));
    }
    if (mUcPtrsDev == nullptr)
    {
        TLLM_CUDA_CHECK(cudaMalloc(&mUcPtrsDev, mGroupSize * sizeof(CUdeviceptr)));
    }
    TLLM_CUDA_CHECK(
        cudaMemcpy(mSignalPadsDev, mSignalPads.data(), mGroupSize * sizeof(CUdeviceptr), cudaMemcpyHostToDevice));
    TLLM_CUDA_CHECK(cudaMemcpy(mUcPtrsDev, mUcPtrs.data(), mGroupSize * sizeof(CUdeviceptr), cudaMemcpyHostToDevice));
}

McastDeviceMemory::~McastDeviceMemory() noexcept
{
    cleanupNoThrow(
        "multicast buffer registration", [this]() { tensorrt_llm::common::unregisterMcastDevMemBuffer(this); });
    if (mIsMNNvlink)
    {
        if (mState != State::kUnmapped)
        {
            unmapAndReleaseMnMcastMem();
        }
        if (mUcBasePtr != 0 && std::none_of(mUcMapped.begin(), mUcMapped.end(), [](bool mapped) { return mapped; }))
        {
            cleanupNoThrow("unicast virtual address reservation",
                [this]() { TLLM_CU_CHECK_FREE_RESOURCE(cuMemAddressFree(mUcBasePtr, mAllocationSize * mGroupSize)); });
        }
        if (mMcPtr != 0 && !mMcMapped)
        {
            cleanupNoThrow("multicast virtual address reservation",
                [this]() { TLLM_CU_CHECK_FREE_RESOURCE(cuMemAddressFree(mMcPtr, mAllocationSize)); });
        }
    }
    else
    {
        // The nvlsfree function will free the handle pointer as well
        cleanupNoThrow("NVLS multicast memory", [this]() { tensorrt_llm::runtime::ipcNvlsFree(mNvlsHandle); });
    }

    if (mSignalPadsDev != nullptr)
    {
        cleanupNoThrow("signal-pad pointer table", [this]() { TLLM_CUDA_CHECK(cudaFree(mSignalPadsDev)); });
    }
    if (mUcPtrsDev != nullptr)
    {
        cleanupNoThrow("unicast pointer table", [this]() { TLLM_CUDA_CHECK(cudaFree(mUcPtrsDev)); });
    }
}

void McastDeviceMemory::createAndMapMnMcastMem(size_t bufSize)
{
    CUmemAllocationHandleType const handleType = CU_MEM_HANDLE_TYPE_FABRIC;
    CUmemAllocationProp prop = {};
    prop.requestedHandleTypes = handleType;
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = mDeviceIdx;
    prop.allocFlags.gpuDirectRDMACapable = 1;

    size_t allocGranularity{0}, mcGranularity{0};
    size_t allocationSize{0};
    CUmulticastObjectProp mcProp{};
    runCollectivePhase(*mGroupComm, "allocation-layout validation",
        [&]()
        {
            TLLM_CU_CHECK(cuMemGetAllocationGranularity(&allocGranularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
            allocationSize = roundUp(bufSize + kSIGNAL_PAD_SIZE, allocGranularity);
            mcProp = {.numDevices = mGroupSize, .size = allocationSize, .handleTypes = handleType};
            TLLM_CU_CHECK(cuMulticastGetGranularity(&mcGranularity, &mcProp, CU_MULTICAST_GRANULARITY_RECOMMENDED));
            allocationSize = roundUp(allocationSize, mcGranularity);
            mcProp.size = allocationSize;
            if (mAllocationSize != 0)
            {
                TLLM_CHECK_WITH_INFO(mAllocationSize == allocationSize,
                    "[McastDeviceMemory] Restored allocation layout differs from the retained virtual-address "
                    "layout.");
            }
        });
    mAllocationSize = allocationSize;
    runCollectivePhase(*mGroupComm, "restore precondition validation",
        [this]()
        {
            TLLM_CHECK_WITH_INFO(mMcHandle == 0
                    && std::all_of(mUcHandles.begin(), mUcHandles.end(), [](auto handle) { return handle == 0; }),
                "[McastDeviceMemory] Cannot restore over unreleased allocation handles.");
            mUcHandles.assign(mGroupSize, 0);
            mUcPtrs.resize(mGroupSize);
            mUcMapped.assign(mGroupSize, false);
            mMcMapped = false;
            mMcBound = false;
        });

    CUmemFabricHandle* exportedHandles{nullptr};
    CUmemFabricHandle* multicastFabricHandle{nullptr};

    try
    {
        CUmemFabricHandle myHandle;
        runCollectivePhase(*mGroupComm, "unicast handle creation",
            [&]()
            {
                TLLM_CU_CHECK(cuMemCreate(&mUcHandles[mGroupRank], mAllocationSize, &prop, 0));
                TLLM_CU_CHECK(
                    cuMemExportToShareableHandle(&myHandle, mUcHandles[mGroupRank], CU_MEM_HANDLE_TYPE_FABRIC, 0));
                TLLM_CUDA_CHECK(cudaMallocHost(&exportedHandles, mGroupSize * sizeof(CUmemFabricHandle)));
            });
        mGroupComm->allgather(&myHandle, exportedHandles, sizeof(CUmemFabricHandle), mpi::MpiType::kCHAR);
        runCollectivePhase(*mGroupComm, "unicast handle import",
            [&]()
            {
                TLLM_CUDA_CHECK(cudaDeviceSynchronize());
                for (uint32_t rank = 0; rank < mGroupSize; rank++)
                {
                    if (rank != mGroupRank)
                    {
                        TLLM_CU_CHECK(cuMemImportFromShareableHandle(&mUcHandles[rank],
                            reinterpret_cast<void*>(&exportedHandles[rank]), CU_MEM_HANDLE_TYPE_FABRIC));
                    }
                }
                TLLM_CUDA_CHECK(cudaFreeHost(exportedHandles));
                exportedHandles = nullptr;
            });

        runCollectivePhase(*mGroupComm, "multicast handle creation",
            [&]()
            {
                TLLM_CUDA_CHECK(cudaMallocHost(&multicastFabricHandle, sizeof(CUmemFabricHandle)));
                if (mGroupRank == 0)
                {
                    TLLM_CU_CHECK(cuMulticastCreate(&mMcHandle, &mcProp));
                    TLLM_CU_CHECK(
                        cuMemExportToShareableHandle(multicastFabricHandle, mMcHandle, CU_MEM_HANDLE_TYPE_FABRIC, 0));
                }
            });
        mGroupComm->bcast(multicastFabricHandle, sizeof(CUmemFabricHandle), mpi::MpiType::kCHAR, 0);
        runCollectivePhase(*mGroupComm, "multicast handle import",
            [&]()
            {
                TLLM_CUDA_CHECK(cudaDeviceSynchronize());
                if (mGroupRank != 0)
                {
                    TLLM_CU_CHECK(
                        cuMemImportFromShareableHandle(&mMcHandle, multicastFabricHandle, CU_MEM_HANDLE_TYPE_FABRIC));
                }
                TLLM_CU_CHECK(cuMulticastAddDevice(mMcHandle, mDeviceIdx));
                TLLM_CUDA_CHECK(cudaFreeHost(multicastFabricHandle));
                multicastFabricHandle = nullptr;
            });

        runCollectivePhase(*mGroupComm, "virtual-address mapping",
            [&]()
            {
                if (mUcBasePtr == 0)
                {
                    TLLM_CU_CHECK(
                        cuMemAddressReserve(&mUcBasePtr, mAllocationSize * mGroupSize, mcGranularity, 0ULL, 0));
                }
                CUmemAccessDesc accessDesc = {};
                accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                accessDesc.location.id = mDeviceIdx;

                for (uint32_t rank = 0; rank < mGroupSize; rank++)
                {
                    mUcPtrs[rank] = mUcBasePtr + (mAllocationSize * rank);
                    TLLM_CU_CHECK(cuMemMap(mUcPtrs[rank], mAllocationSize, 0, mUcHandles[rank], 0));
                    mUcMapped[rank] = true;
                }
                TLLM_CU_CHECK(cuMemSetAccess(mUcBasePtr, mAllocationSize * mGroupSize, &accessDesc, 1));

                if (mMcPtr == 0)
                {
                    TLLM_CU_CHECK(cuMemAddressReserve(&mMcPtr, mAllocationSize, mcGranularity, 0ULL, 0));
                }
                TLLM_CU_CHECK(cuMemMap(mMcPtr, mAllocationSize, 0, mMcHandle, 0));
                mMcMapped = true;
                TLLM_CU_CHECK(cuMemSetAccess(mMcPtr, mAllocationSize, &accessDesc, 1));
                TLLM_CU_CHECK(
                    cuMulticastBindMem(mMcHandle, 0, mUcHandles[mGroupRank], 0 /*memOffset*/, mAllocationSize, 0));
                mMcBound = true;
            });
    }
    catch (...)
    {
        bool rollbackSucceeded = cleanupNoThrow("exported unicast fabric handles",
            [&]()
            {
                if (exportedHandles != nullptr)
                {
                    TLLM_CUDA_CHECK_FREE_RESOURCE(cudaFreeHost(exportedHandles));
                }
            });
        rollbackSucceeded &= cleanupNoThrow("exported multicast fabric handle",
            [&]()
            {
                if (multicastFabricHandle != nullptr)
                {
                    TLLM_CUDA_CHECK_FREE_RESOURCE(cudaFreeHost(multicastFabricHandle));
                }
            });
        rollbackSucceeded &= unmapAndReleaseMnMcastMem();
        if (!rollbackSucceeded)
        {
            mState = State::kBroken;
            try
            {
                TLLM_LOG_ERROR(
                    "[McastDeviceMemory] Restore rollback was incomplete; the object is unusable and the process "
                    "must be torn down.");
            }
            catch (...)
            {
                // Preserve the original restore exception if logging fails.
            }
        }
        throw;
    }
}

bool McastDeviceMemory::unmapAndReleaseMnMcastMem() noexcept
{
    bool succeeded{true};
    if (mMcBound)
    {
        bool const unbound = cleanupNoThrow("multicast binding",
            [this]() { TLLM_CU_CHECK_FREE_RESOURCE(cuMulticastUnbind(mMcHandle, mDeviceIdx, 0, mAllocationSize)); });
        mMcBound = !unbound;
        succeeded &= unbound;
    }
    if (mMcMapped)
    {
        bool const unmapped = cleanupNoThrow(
            "multicast mapping", [this]() { TLLM_CU_CHECK_FREE_RESOURCE(cuMemUnmap(mMcPtr, mAllocationSize)); });
        mMcMapped = !unmapped;
        succeeded &= unmapped;
    }
    for (uint32_t rank = 0; rank < mUcMapped.size(); rank++)
    {
        if (mUcMapped[rank])
        {
            bool const unmapped = cleanupNoThrow("unicast mapping",
                [this, rank]() { TLLM_CU_CHECK_FREE_RESOURCE(cuMemUnmap(mUcPtrs[rank], mAllocationSize)); });
            mUcMapped[rank] = !unmapped;
            succeeded &= unmapped;
        }
    }
    if (mMcHandle != 0 && !mMcBound && !mMcMapped)
    {
        bool const released = cleanupNoThrow(
            "multicast allocation", [this]() { TLLM_CU_CHECK_FREE_RESOURCE(cuMemRelease(mMcHandle)); });
        if (released)
        {
            mMcHandle = 0;
        }
        succeeded &= released;
    }
    for (uint32_t rank = 0; rank < mUcHandles.size(); rank++)
    {
        bool const canRelease = rank < mUcMapped.size() && !mUcMapped[rank] && (rank != mGroupRank || !mMcBound);
        if (mUcHandles[rank] != 0 && canRelease)
        {
            bool const released = cleanupNoThrow(
                "unicast allocation", [this, rank]() { TLLM_CU_CHECK_FREE_RESOURCE(cuMemRelease(mUcHandles[rank])); });
            if (released)
            {
                mUcHandles[rank] = 0;
            }
            succeeded &= released;
        }
    }
    return succeeded && mMcHandle == 0
        && std::all_of(mUcHandles.begin(), mUcHandles.end(), [](auto handle) { return handle == 0; });
}

void McastDeviceMemory::checkpointPrepare()
{
    TLLM_CHECK_WITH_INFO(
        mIsMNNvlink, "[McastDeviceMemory] Stable-VA checkpointing is only supported for fabric-backed MNNVL memory.");
    if (mState == State::kUnmapped)
    {
        TLLM_CHECK_WITH_INFO(!mGroupComm.has_value(),
            "[McastDeviceMemory] Unmapped checkpoint state unexpectedly retained an MPI communicator.");
        return;
    }
    TLLM_CHECK_WITH_INFO(
        mGroupComm.has_value(), "[McastDeviceMemory] Mapped checkpoint state is missing its MPI communicator.");
    int32_t const localState{static_cast<int32_t>(mState)};
    int32_t minState{0};
    int32_t maxState{0};
    mGroupComm->allreduce(&localState, &minState, 1, mpi::MpiType::kINT32, mpi::MpiOp::MIN);
    mGroupComm->allreduce(&localState, &maxState, 1, mpi::MpiType::kINT32, mpi::MpiOp::MAX);
    if (minState != static_cast<int32_t>(State::kMapped) || maxState != minState)
    {
        mState = State::kBroken;
        TLLM_THROW(
            "[McastDeviceMemory] Checkpoint prepare observed inconsistent or unusable rank state; process "
            "teardown is required.");
    }
    mState = State::kTransitioning;
    try
    {
        runCollectivePhase(*mGroupComm, "checkpoint quiescence",
            [this]()
            {
                TLLM_CUDA_CHECK(cudaSetDevice(mDeviceIdx));
                TLLM_CUDA_CHECK(cudaDeviceSynchronize());
            });
        runCollectivePhase(*mGroupComm, "checkpoint detach",
            [this]()
            {
                TLLM_CHECK_WITH_INFO(unmapAndReleaseMnMcastMem(),
                    "[McastDeviceMemory] Failed to detach one or more local fabric resources.");
            });
        mState = State::kUnmapped;
        // Release an owned post-restore duplicate while it still belongs to
        // the current MPI runtime. Restore will install a new duplicate.
        mGroupComm.reset();
    }
    catch (...)
    {
        mState = State::kBroken;
        throw;
    }
}

bool McastDeviceMemory::checkpointRestore(int64_t mpiCommFortranHandle)
{
    TLLM_CHECK_WITH_INFO(
        mIsMNNvlink, "[McastDeviceMemory] Stable-VA checkpointing is only supported for fabric-backed MNNVL memory.");
#if ENABLE_MULTI_DEVICE
    auto restoredGroupComm = tensorrt_llm::mpi::MpiComm(MPI_Comm_f2c(mpiCommFortranHandle), false);
#else
    auto restoredGroupComm = tensorrt_llm::mpi::MpiComm(nullptr, false);
#endif
    runCollectivePhase(
        restoredGroupComm, "restore communicator validation",
        [this, &restoredGroupComm]()
        {
            TLLM_CHECK_WITH_INFO(restoredGroupComm.getRank() == static_cast<int>(mGroupRank)
                    && restoredGroupComm.getSize() == static_cast<int>(mGroupSize),
                "[McastDeviceMemory] Restore communicator does not match the original rank or group size.");
            TLLM_CHECK_WITH_INFO(tensorrt_llm::mpi::getWorldRanks(restoredGroupComm) == mGroupWorldRanks,
                "[McastDeviceMemory] Restore communicator does not match the original ordered world-rank "
                "membership.");
        },
        false);
    int32_t const localState{static_cast<int32_t>(mState)};
    int32_t minState{0};
    int32_t maxState{0};
    restoredGroupComm.allreduce(&localState, &minState, 1, mpi::MpiType::kINT32, mpi::MpiOp::MIN);
    restoredGroupComm.allreduce(&localState, &maxState, 1, mpi::MpiType::kINT32, mpi::MpiOp::MAX);
    if (minState == static_cast<int32_t>(State::kMapped) && maxState == minState)
    {
        return false;
    }
    if (minState != static_cast<int32_t>(State::kUnmapped) || maxState != minState)
    {
        mState = State::kBroken;
        TLLM_THROW(
            "[McastDeviceMemory] Checkpoint restore observed inconsistent or unusable rank state; process "
            "teardown is required.");
    }
    runCollectivePhase(
        restoredGroupComm, "restore communicator ownership validation",
        [this]()
        {
            TLLM_CHECK_WITH_INFO(!mGroupComm.has_value(),
                "[McastDeviceMemory] Restore found a communicator retained past checkpoint prepare.");
        },
        false);
#if ENABLE_MULTI_DEVICE
    MPI_Comm ownedGroupComm{MPI_COMM_NULL};
    runCollectivePhase(restoredGroupComm, "restore communicator duplication",
        [&]() { TLLM_MPI_CHECK(MPI_Comm_dup(restoredGroupComm, &ownedGroupComm)); });
    mGroupComm.emplace(ownedGroupComm, true);
#else
    mGroupComm.emplace(std::move(restoredGroupComm));
#endif
    mState = State::kTransitioning;
    try
    {
        runCollectivePhase(
            *mGroupComm, "restore device selection", [this]() { TLLM_CUDA_CHECK(cudaSetDevice(mDeviceIdx)); });
        createAndMapMnMcastMem(mBufSize);
        runCollectivePhase(*mGroupComm, "pointer-table initialization", [this]() { initializePointerTables(); });
        return true;
    }
    catch (...)
    {
        unmapAndReleaseMnMcastMem();
        mState = State::kBroken;
        throw;
    }
}

void McastDeviceMemory::checkpointRestoreComplete(bool localProtocolResetSucceeded)
{
    TLLM_CHECK_WITH_INFO(
        mIsMNNvlink, "[McastDeviceMemory] Stable-VA checkpointing is only supported for fabric-backed MNNVL memory.");
    TLLM_CHECK_WITH_INFO(mGroupComm.has_value(),
        "[McastDeviceMemory] Checkpoint restore completion does not have an active MPI communicator.");
    int32_t const localState{static_cast<int32_t>(mState)};
    int32_t minState{0};
    int32_t maxState{0};
    mGroupComm->allreduce(&localState, &minState, 1, mpi::MpiType::kINT32, mpi::MpiOp::MIN);
    mGroupComm->allreduce(&localState, &maxState, 1, mpi::MpiType::kINT32, mpi::MpiOp::MAX);
    if (minState != static_cast<int32_t>(State::kTransitioning) || maxState != minState)
    {
        mState = State::kBroken;
        TLLM_THROW(
            "[McastDeviceMemory] Restore completion observed inconsistent or unusable rank state; process "
            "teardown is required.");
    }
    try
    {
        runCollectivePhase(*mGroupComm, "all-reduce protocol reset",
            [localProtocolResetSucceeded]()
            {
                TLLM_CHECK_WITH_INFO(localProtocolResetSucceeded,
                    "[McastDeviceMemory] The owning rank failed to reset its all-reduce protocol state.");
            });
        mState = State::kMapped;
    }
    catch (...)
    {
        unmapAndReleaseMnMcastMem();
        mState = State::kBroken;
        throw;
    }
}

void McastDeviceMemory::allocNvlsMcastMem(size_t bufSize)
{
    // Get the world ranks for ranks in this group
    auto ranks_ = tensorrt_llm::mpi::getWorldRanks(*mGroupComm);
    std::set<int> ranks(ranks_.begin(), ranks_.end());
    // Reuse existing implementation
    mNvlsHandle = tensorrt_llm::runtime::ipcNvlsAllocate(bufSize, ranks);
    mMcHandle = mNvlsHandle->mc_handle;
    mMcPtr = mNvlsHandle->mc_va;
    mUcPtrs = mNvlsHandle->ipc_uc_vas;
    mUcHandles = mNvlsHandle->ipc_uc_handles;
    mState = State::kMapped;
}

} // namespace tensorrt_llm::runtime
