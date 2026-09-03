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

#include "tensorrt_llm/runtime/ipcNvlsMemory.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include <cstddef>
#include <cstdint>
#include <cuda.h>
#include <vector>

namespace tensorrt_llm::runtime
{

//! \brief A class that manages multicast device memory for efficient communication between GPUs.
//!
//! This class uses fabric allocation if mnNvlink is true, otherwise it uses IPC-based allocation.
//! The fabric allocation can also be used for single-node/intra-node-only communication, but the machine
//! must properly configure IMEX services. See:
//! https://docs.nvidia.com/multi-node-nvlink-systems/imex-guide/gettingstarted.html
//!
//! The class manages both unicast pointers (one per rank) and a single multicast pointer,
//! along with signal pads used for synchronization between devices.
class McastDeviceMemory
{
public:
    // Disallow copy construction
    McastDeviceMemory(McastDeviceMemory const&) = delete;
    McastDeviceMemory& operator=(McastDeviceMemory const&) = delete;

    McastDeviceMemory(size_t bufSize, uint32_t groupSize, uint32_t groupRank, int deviceIdx, bool mnNvlink,
        int64_t mpiCommFortranHandle);

    // McastGPUBuffer registers these pointers once shared ownership has been established.
    //! Get the raw array of signal pad pointers to all ranks (including self)
    [[nodiscard]] void** getSignalPadPtrsDev() const
    {
        return mSignalPadsDev;
    }

    //! Get the raw array of unicast pointers to all ranks (including self)
    [[nodiscard]] void** getBufferPtrsDev() const
    {
        return mUcPtrsDev;
    }

    //! Get the raw unicast pointer to a given rank
    [[nodiscard]] void* getUnicastPtr(uint32_t rank) const;

    //! Get the raw multicast pointer
    [[nodiscard]] void* getMulticastPtr() const;

    [[nodiscard]] size_t getRank() const
    {
        return mGroupRank;
    }

    [[nodiscard]] size_t getWorldSize() const
    {
        return mGroupSize;
    }

    //! Get the usable logical buffer size. Fabric allocations include reusable multicast-alignment slack.
    [[nodiscard]] size_t getBufferSize() const
    {
        return mBufSize;
    }

    ~McastDeviceMemory() noexcept;

private:
    bool mIsMNNvlink;
    int mDeviceIdx;
    uint32_t mGroupSize, mGroupRank;
    int mCommSize{-1};
    int mCommRank{-1};
    int mWorldRank{-1};
    size_t mBufSize;
    size_t mSignalPadOffset{0};
    size_t mAllocationSize{0};
    size_t mAllocationGranularity{0};
    size_t mMulticastRecommendedGranularity{0};

    CUdeviceptr mMcPtr{0};
    CUdeviceptr mUcPtrBase{0};
    size_t mUcReservationSize{0};
    CUmemGenericAllocationHandle mMcHandle{0};
    bool mMcHandleAcquired{false};
    bool mMcAddressReserved{false};
    bool mMcMapped{false};
    bool mMcBound{false};
    std::vector<CUmemGenericAllocationHandle> mUcHandles;
    std::vector<uint8_t> mUcHandleAcquired;
    std::vector<uint8_t> mUcMapped;

    tensorrt_llm::mpi::MpiComm mGroupComm; //!< The MPI communicator for the group

    // Host array of pointers
    std::vector<CUdeviceptr> mUcPtrs;
    std::vector<CUdeviceptr> mSignalPads;

    // Device array of pointers
    void** mUcPtrsDev{nullptr};
    void** mSignalPadsDev{nullptr};

    // For intra-node mcast
    tensorrt_llm::runtime::IpcNvlsHandle* mNvlsHandle{nullptr};

    void allocMnMcastMem(size_t bufSize);
    void allocNvlsMcastMem(size_t bufSize);
    void initializePointerArrays();
    void cleanup() noexcept;
};

constexpr size_t kSIGNAL_PAD_SIZE = 2048;

} // namespace tensorrt_llm::runtime
