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
#pragma once

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/mcastDevMemUtils.h"
#include "tensorrt_llm/runtime/ipcNvlsMemory.h"
#include "tensorrt_llm/runtime/mcastDeviceMemory.h"
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace tensorrt_llm::runtime
{

//! \brief Wrapper class for McastDeviceMemory to facilitate PyTorch tensor creation.
//! It manages a buffer accessible via unicast or multicast for multi-node communication.
class McastGPUBuffer
{
public:
    // Disallow copy construction and assignment
    McastGPUBuffer(McastGPUBuffer const&) = delete;
    McastGPUBuffer& operator=(McastGPUBuffer const&) = delete;

    //! \brief Constructor for McastGpuBuffer.
    //! \param bufSize The total size of the buffer in bytes.
    //! \param groupSize The number of ranks in the communication group.
    //! \param groupRank The rank of the local process within the group.
    //! \param deviceIdx The CUDA device for buffer allocation.
    //! \param mnNvlink Flag indicating if multi-node NVLink is used.
    //! \param mpiCommFortranHandle The Fortran handle for the MPI communicator (from Python mpi4py).
    McastGPUBuffer(size_t bufSize, uint32_t groupSize, uint32_t groupRank, uint32_t deviceIdx, bool mnNvlink,
        int64_t mpiCommFortranHandle)
        : mMcastDeviceMemory(std::make_shared<McastDeviceMemory>(
            bufSize, groupSize, groupRank, deviceIdx, mnNvlink, mpiCommFortranHandle))
        , mBufSize(mMcastDeviceMemory->getBufferSize())
        , mLocalDevice(at::Device(at::DeviceType::CUDA, deviceIdx))
    {
        for (uint32_t rank = 0; rank < groupSize; ++rank)
        {
            tensorrt_llm::common::registerMcastDevMemBuffer(
                mMcastDeviceMemory->getUnicastPtr(rank), mMcastDeviceMemory);
        }
        tensorrt_llm::common::registerMcastDevMemBuffer(mMcastDeviceMemory->getMulticastPtr(), mMcastDeviceMemory);
    }

    //! \brief Returns the usable logical size after fabric-allocation rounding and signal-pad reservation.
    [[nodiscard]] size_t getBufferSize() const
    {
        return mBufSize;
    }

    //! \brief Returns a PyTorch tensor view of the unicast buffer portion for a specific rank.
    //! \param rank The target rank for the unicast pointer.
    //! \param sizes The desired shape (dimensions) of the tensor.
    //! \param dtype The data type of the tensor elements.
    //! \param storageOffset The offset in elements from the start of the buffer.
    //! \return An ATen tensor wrapping the unicast buffer section.
    at::Tensor getUCBuffer(uint32_t rank, std::vector<long int> sizes, torch::ScalarType dtype, int64_t storageOffset)
    {
        TORCH_CHECK(rank < mMcastDeviceMemory->getWorldSize(), "McastGPUBuffer::getUCBuffer: rank ", rank,
            " is outside world size ", mMcastDeviceMemory->getWorldSize());
        return makeTensor(
            mMcastDeviceMemory->getUnicastPtr(rank), sizes, dtype, storageOffset, "McastGPUBuffer::getUCBuffer");
    }

    //! \brief Returns a PyTorch tensor view of the multicast buffer portion.
    //! \param sizes The desired shape (dimensions) of the tensor.
    //! \param dtype The data type of the tensor elements.
    //! \param storageOffset The offset in elements from the start of the buffer.
    //! \return An ATen tensor wrapping the multicast buffer section.
    at::Tensor getMCBuffer(std::vector<long int> sizes, torch::ScalarType dtype, int64_t storageOffset)
    {
        return makeTensor(
            mMcastDeviceMemory->getMulticastPtr(), sizes, dtype, storageOffset, "McastGPUBuffer::getMCBuffer");
    }

private:
    using DeviceMemoryOwner = std::shared_ptr<tensorrt_llm::runtime::McastDeviceMemory>;

    static void deleteDeviceMemoryOwner(void* context)
    {
        delete static_cast<DeviceMemoryOwner*>(context);
    }

    at::Tensor makeTensor(void* basePtr, std::vector<long int> const& sizes, torch::ScalarType dtype,
        int64_t storageOffset, char const* caller)
    {
        TORCH_CHECK(storageOffset >= 0, caller, ": storage offset must be nonnegative, got ", storageOffset);

        size_t numel{1};
        for (long int const size : sizes)
        {
            TORCH_CHECK(size >= 0, caller, ": dimensions must be nonnegative, got ", size);
            size_t const unsignedSize = static_cast<size_t>(size);
            TORCH_CHECK(unsignedSize == 0 || numel <= std::numeric_limits<size_t>::max() / unsignedSize, caller,
                ": tensor element count overflows size_t");
            numel *= unsignedSize;
        }

        size_t const unsignedOffset = static_cast<size_t>(storageOffset);
        TORCH_CHECK(numel <= std::numeric_limits<size_t>::max() - unsignedOffset, caller,
            ": tensor storage extent overflows size_t");
        size_t const storageElements = numel + unsignedOffset;
        size_t const elementSize = c10::elementSize(dtype);
        TORCH_CHECK(elementSize == 0 || storageElements <= std::numeric_limits<size_t>::max() / elementSize, caller,
            ": tensor storage size overflows size_t");
        size_t const requiredSize = storageElements * elementSize;
        TORCH_CHECK(requiredSize <= mBufSize, caller, ": the requested size (", requiredSize,
            " bytes) exceeds the allocated size (", mBufSize, " bytes)");

        auto* dataPtr = static_cast<uint8_t*>(basePtr) + unsignedOffset * elementSize;
        auto* owner = new DeviceMemoryOwner(mMcastDeviceMemory);
        auto const options = at::TensorOptions().dtype(dtype).device(mLocalDevice);
        return at::for_blob(dataPtr, c10::IntArrayRef(sizes))
            .context(owner, deleteDeviceMemoryOwner)
            .options(options)
            .target_device(mLocalDevice)
            .make_tensor();
    }

    //!< Underlying memory manager for multi-node communication.
    DeviceMemoryOwner mMcastDeviceMemory;
    size_t mBufSize;         //!< Total size of the managed buffer.
    at::Device mLocalDevice; //!< The local CUDA device.
};

// MNNVL all-reduce runtime qualification.
enum class MnnvlTransport
{
    kPosixFd,
    kFabric,
};

namespace mnnvl_workspace_detail
{

inline constexpr size_t kCollectiveErrorMessageSize = 768;

enum class PreflightStatus : int32_t
{
    kAvailable,
    kUnavailable,
    kError,
};

struct PreflightResult
{
    PreflightStatus status{PreflightStatus::kAvailable};
    std::string message;
};

struct PreflightWire
{
    PreflightStatus status{PreflightStatus::kAvailable};
    std::array<char, kCollectiveErrorMessageSize> message{};
};

inline PreflightWire makePreflightWire(PreflightResult const& result)
{
    PreflightWire wire{};
    wire.status = result.status;
    size_t const length = std::min(result.message.size(), wire.message.size() - 1);
    std::copy_n(result.message.data(), length, wire.message.data());
    return wire;
}

inline PreflightResult getCollectivePreflightResult(mpi::MpiComm const& comm, PreflightResult const& localResult)
{
    PreflightWire const local = makePreflightWire(localResult);
    std::vector<PreflightWire> results(static_cast<size_t>(comm.getSize()));
    comm.allgather(&local, results.data(), sizeof(PreflightWire), mpi::MpiType::kBYTE);

    for (PreflightStatus const target : {PreflightStatus::kError, PreflightStatus::kUnavailable})
    {
        for (size_t rank = 0; rank < results.size(); ++rank)
        {
            if (results[rank].status == target)
            {
                std::ostringstream message;
                message << "[MNNVL] Runtime preflight failed on rank " << rank << ": " << results[rank].message.data();
                return {target, message.str()};
            }
        }
    }
    return {};
}

inline bool isGroupLocal(mpi::MpiComm const& groupComm)
{
    auto const groupRanks = mpi::getWorldRanks(groupComm);
    auto const localRanks = mpi::getWorldRanks(mpi::MpiComm::localSession());
    std::unordered_set<int> const localRankSet(localRanks.begin(), localRanks.end());
    return std::all_of(groupRanks.begin(), groupRanks.end(),
        [&localRankSet](int rank) { return localRankSet.find(rank) != localRankSet.end(); });
}

//! Format a fabric-probe failure after its raw CUresult has been classified.
[[nodiscard]] inline std::string formatFabricProbeError(CUresult result, char const* operation)
{
    char const* errorName = nullptr;
    char const* errorDescription = nullptr;
    cuGetErrorName(result, &errorName);
    cuGetErrorString(result, &errorDescription);
    std::ostringstream message;
    message << operation << " failed with " << (errorName == nullptr ? "unknown CUDA error" : errorName);
    if (errorDescription != nullptr)
    {
        message << " (" << errorDescription << ")";
    }
    return message.str();
}

inline PreflightResult probeFabricHandleSupport(int deviceIdx)
{
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = deviceIdx;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;

    size_t granularity{0};
    TLLM_CU_CHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

    CUmemGenericAllocationHandle allocation{0};
    CUmemGenericAllocationHandle importedAllocation{0};
    bool allocationAcquired{false};
    bool importedAllocationAcquired{false};

    // Cleanup is secondary to the probe result and must not throw.
    auto cleanup = [&]() noexcept
    {
        if (importedAllocationAcquired)
        {
            CUresult const result = cuMemRelease(importedAllocation);
            if (result != CUDA_SUCCESS)
            {
                TLLM_LOG_WARNING("[MNNVL] Failed to release the imported fabric probe allocation");
            }
        }
        if (allocationAcquired)
        {
            CUresult const result = cuMemRelease(allocation);
            if (result != CUDA_SUCCESS)
            {
                TLLM_LOG_WARNING("[MNNVL] Failed to release the fabric probe allocation");
            }
        }
    };

    CUresult result = cuMemCreate(&allocation, granularity, &prop, 0);
    if (result != CUDA_SUCCESS)
    {
        // These errors are expected when the fabric/IMEX plane is not provisioned.
        bool const isExpectedUnavailable = result == CUDA_ERROR_NOT_PERMITTED || result == CUDA_ERROR_NOT_SUPPORTED;
        return {isExpectedUnavailable ? PreflightStatus::kUnavailable : PreflightStatus::kError,
            formatFabricProbeError(result, "cuMemCreate(fabric probe)")};
    }
    allocationAcquired = true;

    // Match the existing IPC NVLS handle selection: export or import failure makes fabric unavailable.
    CUmemFabricHandle fabricHandle{};
    result = cuMemExportToShareableHandle(&fabricHandle, allocation, CU_MEM_HANDLE_TYPE_FABRIC, 0);
    if (result != CUDA_SUCCESS)
    {
        cleanup();
        return {PreflightStatus::kUnavailable,
            formatFabricProbeError(result, "cuMemExportToShareableHandle(fabric probe)")};
    }
    result = cuMemImportFromShareableHandle(
        &importedAllocation, static_cast<void*>(&fabricHandle), CU_MEM_HANDLE_TYPE_FABRIC);
    if (result != CUDA_SUCCESS)
    {
        cleanup();
        return {PreflightStatus::kUnavailable,
            formatFabricProbeError(result, "cuMemImportFromShareableHandle(fabric probe)")};
    }
    importedAllocationAcquired = true;

    cleanup();
    return {};
}

inline PreflightResult runPosixFdPreflight(mpi::MpiComm const& groupComm)
{
    if (!isGroupLocal(groupComm))
    {
        return {PreflightStatus::kUnavailable,
            "the POSIX-FD transport requires every communicator rank to be on the same host"};
    }
    if (!ipcNvlsSupported())
    {
        return {PreflightStatus::kUnavailable, "the IPC NVLS allocator reports that multicast memory is unavailable"};
    }
    return {};
}

inline PreflightResult runFabricPreflight(int deviceIdx, CUdevice device)
{
    int driverVersion{-1};
    TLLM_CUDA_CHECK(cudaDriverGetVersion(&driverVersion));
    if (driverVersion < 12010)
    {
        return {PreflightStatus::kUnavailable, "CUDA driver 12.1 or newer is required"};
    }

    int multicastSupported{0};
    TLLM_CU_CHECK(cuDeviceGetAttribute(&multicastSupported, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, device));
    if (multicastSupported == 0)
    {
        return {PreflightStatus::kUnavailable, "CUDA multicast memory is not supported by the selected device"};
    }

    int fabricHandleSupported{0};
    TLLM_CU_CHECK(
        cuDeviceGetAttribute(&fabricHandleSupported, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, device));
    if (fabricHandleSupported == 0)
    {
        return {PreflightStatus::kUnavailable, "CUDA fabric handles are not supported by the selected device"};
    }

    return probeFabricHandleSupport(deviceIdx);
}

inline PreflightResult runLocalPreflight(
    MnnvlTransport transport, int deviceIdx, mpi::MpiComm const& groupComm, CUuuid& deviceUuid)
{
    try
    {
        TLLM_CUDA_CHECK(cudaSetDevice(deviceIdx));

        CUdevice device{};
        TLLM_CU_CHECK(cuDeviceGet(&device, deviceIdx));
        TLLM_CU_CHECK(cuDeviceGetUuid(&deviceUuid, device));

        int computeCapabilityMajor{0};
        TLLM_CU_CHECK(
            cuDeviceGetAttribute(&computeCapabilityMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
        if (computeCapabilityMajor < 9)
        {
            return {
                PreflightStatus::kUnavailable, "the MNNVL all-reduce kernel requires compute capability 9.0 or newer"};
        }

        if (transport == MnnvlTransport::kPosixFd)
        {
            return runPosixFdPreflight(groupComm);
        }
        return runFabricPreflight(deviceIdx, device);
    }
    catch (std::exception const& error)
    {
        return {PreflightStatus::kError, error.what()};
    }
    catch (...)
    {
        return {PreflightStatus::kError, "unknown error during the CUDA runtime preflight"};
    }
}

} // namespace mnnvl_workspace_detail

//! Runtime-qualified MNNVL workspace used by the PyTorch all-reduce integration.
//!
//! Mapping policy stays in Python. This object validates the actual MPI placement and CUDA runtime,
//! coordinates preflight availability across the group, and owns the multicast allocation. Allocation itself
//! deliberately retains McastDeviceMemory's existing failure behavior.
class MnnvlWorkspace
{
public:
    MnnvlWorkspace(size_t workspaceSize, MnnvlTransport transport, int deviceIdx, int64_t mpiCommFortranHandle)
    {
#if ENABLE_MULTI_DEVICE
        mpi::MpiComm groupComm(MPI_Comm_f2c(mpiCommFortranHandle), false);
#else
        mpi::MpiComm groupComm(nullptr, false);
#endif
        int const groupSize = groupComm.getSize();
        int const groupRank = groupComm.getRank();

        CUuuid localDeviceUuid{};
        auto const localPreflight
            = mnnvl_workspace_detail::runLocalPreflight(transport, deviceIdx, groupComm, localDeviceUuid);
        // This agreement covers dispatch qualification only. McastDeviceMemory retains its existing allocation and
        // failure behavior, including its own request-size agreement.
        auto const collectivePreflight
            = mnnvl_workspace_detail::getCollectivePreflightResult(groupComm, localPreflight);
        if (collectivePreflight.status == mnnvl_workspace_detail::PreflightStatus::kError)
        {
            throw std::runtime_error(collectivePreflight.message);
        }
        if (collectivePreflight.status == mnnvl_workspace_detail::PreflightStatus::kUnavailable)
        {
            mReason = collectivePreflight.message;
            return;
        }

        std::vector<CUuuid> deviceUuids(static_cast<size_t>(groupSize));
        groupComm.allgather(&localDeviceUuid, deviceUuids.data(), sizeof(CUuuid), mpi::MpiType::kBYTE);
        for (size_t lhs = 0; lhs < deviceUuids.size(); ++lhs)
        {
            for (size_t rhs = lhs + 1; rhs < deviceUuids.size(); ++rhs)
            {
                if (std::memcmp(deviceUuids[lhs].bytes, deviceUuids[rhs].bytes, sizeof(deviceUuids[lhs].bytes)) == 0)
                {
                    throw std::invalid_argument("[MNNVL] Communicator ranks " + std::to_string(lhs) + " and "
                        + std::to_string(rhs) + " selected the same physical CUDA device");
                }
            }
        }

        mBuffer = std::make_shared<McastGPUBuffer>(workspaceSize, static_cast<uint32_t>(groupSize),
            static_cast<uint32_t>(groupRank), static_cast<uint32_t>(deviceIdx), transport == MnnvlTransport::kFabric,
            mpiCommFortranHandle);

        mGroupRank = static_cast<uint32_t>(groupRank);
    }

    MnnvlWorkspace(MnnvlWorkspace const&) = delete;
    MnnvlWorkspace& operator=(MnnvlWorkspace const&) = delete;

    [[nodiscard]] bool isAvailable() const
    {
        return mBuffer != nullptr;
    }

    [[nodiscard]] std::string const& getReason() const
    {
        return mReason;
    }

    //! Return the allocator's total usable workspace size in bytes.
    [[nodiscard]] size_t getWorkspaceSize() const
    {
        return mBuffer == nullptr ? 0 : mBuffer->getBufferSize();
    }

    //! Return a flat FP32 view of the allocator's total usable local workspace.
    [[nodiscard]] at::Tensor getLocalBuffer()
    {
        TORCH_CHECK(mBuffer != nullptr, "MnnvlWorkspace::getLocalBuffer: workspace is unavailable: ", mReason);
        return mBuffer->getUCBuffer(
            mGroupRank, {static_cast<long int>(mBuffer->getBufferSize() / sizeof(float))}, at::kFloat, 0);
    }

private:
    std::string mReason;
    uint32_t mGroupRank{0};
    std::shared_ptr<McastGPUBuffer> mBuffer;
};

} // namespace tensorrt_llm::runtime
