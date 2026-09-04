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

#include "tensorrt_llm/common/mcastDevMemUtils.h"
#include "tensorrt_llm/runtime/ipcNvlsMemory.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include <cstddef>
#include <cstdint>
#include <cuda.h>
#include <memory>
#include <optional>
#include <vector>

namespace tensorrt_llm::runtime
{

//! \brief A class that manages multicast device memory for efficient communication between GPUs.
//!
//! This class uses fabric-backed allocation if mnNvlink is true, otherwise it uses intra-node NVLS allocation.
//! The fabric allocation can also be used for single-node/intra-node-only communication, but the machine
//! must properly configure IMEX services. See:
//! https://docs.nvidia.com/multi-node-nvlink-systems/imex-guide/gettingstarted.html
//!
//! The class manages both unicast pointers (one per rank) and a single multicast pointer,
//! along with signal pads used for synchronization between devices.
class McastDeviceMemory
{
    enum class State : int32_t
    {
        kUnmapped,
        kMapped,
        kTransitioning,
        kBroken,
    };

public:
    // Disallow copy construction
    McastDeviceMemory(McastDeviceMemory const&) = delete;
    McastDeviceMemory& operator=(McastDeviceMemory const&) = delete;

    McastDeviceMemory(size_t bufSize, uint32_t groupSize, uint32_t groupRank, int deviceIdx, bool mnNvlink,
        int64_t mpiCommFortranHandle);

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

    //! Release fabric-backed physical memory while retaining virtual-address reservations.
    //!
    //! \warning This is an internal, experimental resource hook. The caller must stop admission, drain every
    //! in-flight request and collective on all participating ranks, and invoke the hook collectively. It is not
    //! sufficient by itself for live-serving checkpointing. The current owned communicator duplicate is released
    //! before this method returns, while it still belongs to the pre-checkpoint MPI runtime.
    void checkpointPrepare();

    //! Recreate fabric-backed physical memory and remap it at the retained virtual addresses.
    //!
    //! \param mpiCommFortranHandle A communicator created after process restore. Its ordered world-rank membership,
    //! rank, and size must match the original communicator exactly. A successful restore retains an owned duplicate;
    //! the caller may release the supplied communicator after this method returns.
    //! \return True when new mappings were created and checkpointRestoreComplete() must be called; false when the
    //! object was already mapped and the supplied communicator was not retained.
    //! \warning Multi-node MPI, NCCL, and RDMA process-restore semantics are not supported without an external
    //! engine-wide coordinator that keeps every participating rank quiescent through restore. Communicator validation
    //! rejects before mutation and may be retried; any failure after mutation starts leaves the object unusable and
    //! requires process teardown.
    [[nodiscard]] bool checkpointRestore(int64_t mpiCommFortranHandle);

    //! Publish or abort mappings created by checkpointRestore after the owner resets its protocol state.
    void checkpointRestoreComplete(bool localProtocolResetSucceeded);

    [[nodiscard]] bool isMapped() const
    {
        return mState == State::kMapped;
    }

    ~McastDeviceMemory() noexcept;

private:
    bool mIsMNNvlink;
    int mDeviceIdx;
    uint32_t mGroupSize, mGroupRank;
    size_t mBufSize;
    size_t mSignalPadOffset;
    size_t mAllocationSize;

    CUdeviceptr mMcPtr;
    CUdeviceptr mUcBasePtr;
    CUmemGenericAllocationHandle mMcHandle;
    std::vector<CUmemGenericAllocationHandle> mUcHandles;

    std::optional<tensorrt_llm::mpi::MpiComm> mGroupComm; //!< Present only while the current MPI runtime is valid
    std::vector<int> mGroupWorldRanks;                    //!< Ordered world-rank membership retained across restore

    // Host array of pointers
    std::vector<CUdeviceptr> mUcPtrs;
    std::vector<CUdeviceptr> mSignalPads;

    // Device array of pointers
    void** mUcPtrsDev;
    void** mSignalPadsDev;
    State mState;
    std::vector<bool> mUcMapped;
    bool mMcMapped;
    bool mMcBound;

    // For intra-node mcast
    tensorrt_llm::runtime::IpcNvlsHandle* mNvlsHandle;

    void createAndMapMnMcastMem(size_t bufSize);
    bool unmapAndReleaseMnMcastMem() noexcept;
    void allocNvlsMcastMem(size_t bufSize);
    void initializePointerTables();
};

constexpr size_t kSIGNAL_PAD_SIZE = 2048;

} // namespace tensorrt_llm::runtime
