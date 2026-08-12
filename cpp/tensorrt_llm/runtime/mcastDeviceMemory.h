/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

#include "tensorrt_llm/common/mcastDevMemUtils.h"
#include "tensorrt_llm/runtime/ipcNvlsMemory.h"
#include "tensorrt_llm/runtime/mcastGroupComm.h"
#include <cstddef>
#include <cstdint>
#include <cuda.h>
#include <memory>
#include <vector>

namespace tensorrt_llm::runtime
{

//! \brief A class that manages multicast device memory for efficient communication between GPUs.
//!
//! When mnNvlink is true the group may span nodes, so the memory is shared through fabric handles;
//! that requires the machine to have IMEX services properly configured. Otherwise the intra-node
//! NVLS path is used, which shares memory through fabric handles where they are available and
//! POSIX file descriptors otherwise. See:
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

    //! \brief Constructor taking an arbitrary host collective backend.
    //! \param groupComm Collective used to exchange the CUDA memory handles. Its size must match
    //! \p groupSize and its rank must match \p groupRank.
    McastDeviceMemory(size_t bufSize, uint32_t groupSize, uint32_t groupRank, int deviceIdx, bool mnNvlink,
        std::shared_ptr<McastGroupComm> groupComm);

    // We don't register the pointer in these two functions since we don't expect any python-level code would call
    // to obtain the raw pointers.
    //! Get the raw array of signal pad pointers to all ranks (including self)
    void** getSignalPadPtrsDev()
    {
        return mSignalPadsDev;
    }

    //! Get the raw array of unicast pointers to all ranks (including self)
    void** getBufferPtrsDev()
    {
        return mUcPtrsDev;
    }

    //! Get the raw unicast pointer to a given rank
    void* getUnicastPtr(uint32_t rank)
    {
        auto* data_ptr = reinterpret_cast<void*>(mUcPtrs[rank]);
        tensorrt_llm::common::registerMcastDevMemBuffer(data_ptr, this);
        return data_ptr;
    }

    //! Get the raw multicast pointer
    void* getMulticastPtr()
    {
        auto* data_ptr = reinterpret_cast<void*>(mMcPtr);
        tensorrt_llm::common::registerMcastDevMemBuffer(data_ptr, this);
        return data_ptr;
    }

    [[nodiscard]] size_t getRank() const
    {
        return mGroupRank;
    }

    [[nodiscard]] size_t getWorldSize() const
    {
        return mGroupSize;
    }

    ~McastDeviceMemory();

private:
    bool mIsMNNvlink;
    int mDeviceIdx;
    uint32_t mGroupSize, mGroupRank;
    size_t mBufSize;
    size_t mSignalPadOffset;
    size_t mAllocationSize;

    CUdeviceptr mMcPtr;
    CUmemGenericAllocationHandle mMcHandle;
    std::vector<CUmemGenericAllocationHandle> mUcHandles;

    std::shared_ptr<McastGroupComm> mGroupComm; //!< Host collective used to exchange memory handles

    // Host array of pointers
    std::vector<CUdeviceptr> mUcPtrs;
    std::vector<CUdeviceptr> mSignalPads;

    // Device array of pointers
    void** mUcPtrsDev;
    void** mSignalPadsDev;

    // For intra-node mcast
    tensorrt_llm::runtime::IpcNvlsHandle* mNvlsHandle;

    void allocMnMcastMem(size_t bufSize);
    void allocNvlsMcastMem(size_t bufSize);
};

constexpr size_t kSIGNAL_PAD_SIZE = 2048;

} // namespace tensorrt_llm::runtime
