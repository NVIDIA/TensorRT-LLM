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

#include "tensorrt_llm/runtime/cudaStream.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include "tensorrt_llm/runtime/worldConfig.h"

#include <cuda_runtime_api.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

struct ncclComm;
typedef struct ncclComm* ncclComm_t;

namespace tensorrt_llm::runtime
{

class NcclUniqueIdRendezvousComm;

class NcclCommunicator
{
public:
    //! Wrap an existing communicator.
    //!
    //! In fault-tolerance mode the caller is responsible for creating `comm`
    //! in non-blocking mode. The default-off path preserves the legacy
    //! blocking communicator contract.
    explicit NcclCommunicator(ncclComm_t comm);

    //! Create a communicator bootstrapped by `mpiComm`.
    //!
    //! Construction establishes and retains a dedicated MPI control channel while
    //! the initial group is healthy. Recovery reuses that channel without an
    //! MPI collective; `mpiComm` therefore need not outlive this object.
    explicit NcclCommunicator(int worldSize, int rank, mpi::MpiComm const& mpiComm = COMM_SESSION);

    explicit NcclCommunicator(WorldConfig const& worldConfig, mpi::MpiComm const& mpiComm = COMM_SESSION)
        : NcclCommunicator{worldConfig.getSize(), worldConfig.getRank(), mpiComm} {};

    ~NcclCommunicator();

    // no copy
    NcclCommunicator(NcclCommunicator const&) = delete;
    NcclCommunicator& operator=(NcclCommunicator const&) = delete;

    void send(IBuffer const& buf, int peer, CudaStream const& stream) const
    {
        send(buf.data(), buf.getSize(), buf.getDataType(), peer, stream);
    }

    void receive(IBuffer& buf, int peer, CudaStream const& stream) const
    {
        receive(buf.data(), buf.getSize(), buf.getDataType(), peer, stream);
    }

    //! Abort the current communicator. Idempotent.
    void abort();

    //! Abort and replace the current communicator with a fresh communicator
    //! containing only `activeRanks`.
    //!
    //! Ranks are the original world-rank IDs supplied to the constructor. The
    //! local rank must be present, and previously removed ranks cannot be added
    //! again. Every survivor must call this method with the same rank set. The
    //! positive `rendezvousId` must identify the same logical recovery attempt
    //! on every survivor; zero is reserved for the initial bootstrap. The NCCL
    //! unique ID is exchanged point-to-point over the dedicated control
    //! channel, so failed ranks do not participate in recovery.
    //!
    //! Communicators created by the raw-handle constructor cannot be rebuilt
    //! because that constructor has no MPI bootstrap communicator.
    void abortAndReinit(std::vector<int> const& activeRanks, std::uint64_t rendezvousId);

    //! Return the latched NCCL abort/error reason, or an empty string.
    [[nodiscard]] std::string getAsyncError() const;

    //! Return the original world-rank IDs in the current communicator.
    [[nodiscard]] std::vector<int> getActiveRanks() const;

private:
    void send(
        void const* sendbuff, size_t count, nvinfer1::DataType dataType, int peer, CudaStream const& stream) const;

    void receive(void* sendbuff, size_t count, nvinfer1::DataType dataType, int peer, CudaStream const& stream) const;

    static ncclComm_t createComm(std::vector<int> const& activeRanks, int worldRank,
        NcclUniqueIdRendezvousComm const& controlComm, std::uint64_t rendezvousId,
        std::chrono::milliseconds readyTimeout);
    static ncclComm_t createLegacyComm(int worldSize, int rank, mpi::MpiComm const& mpiComm);
    static void waitUntilReady(ncclComm_t comm, char const* operation, std::chrono::steady_clock::time_point deadline);

    void startWatcher();
    void stopWatcher() noexcept;
    void watchAsyncErrors() noexcept;
    [[nodiscard]] bool abortLocked(std::string reason) const noexcept;
    void checkUsableLocked() const;
    [[nodiscard]] int getCommPeerRankLocked(int worldRank) const;
    [[nodiscard]] uint64_t beginOperationLocked(cudaStream_t stream, char const* operation) const;
    void finishOperationLocked(uint64_t token, cudaStream_t stream) const;
    [[nodiscard]] cudaEvent_t acquireEventLocked() const;
    void recycleEventLocked(cudaEvent_t event) const noexcept;
    void quarantinePendingOperationsLocked() const noexcept;
    void destroyCompletedWatchdogEventsLocked() const noexcept;
    void destroyPooledEventsLocked() const noexcept;

    int mInitialWorldSize{0};
    int mWorldRank{0};
    bool mFaultToleranceEnabled{false};
    std::vector<int> mActiveRanks;
    std::shared_ptr<NcclUniqueIdRendezvousComm> mControlComm;

    mutable std::mutex mCommMutex;
    mutable ncclComm_t mComm{nullptr};
    mutable std::string mAsyncError;

    struct PendingOperation
    {
        uint64_t token;
        cudaEvent_t start;
        cudaEvent_t completion;
        std::chrono::steady_clock::time_point deadline;
        bool armed;
        std::string name;
    };

    mutable std::vector<PendingOperation> mPendingOperations;
    mutable std::vector<cudaEvent_t> mEventPool;
    mutable uint64_t mNextOperationToken{1};
    static constexpr size_t kMaxPendingOperations = 4096;
    static constexpr size_t kMaxPooledEvents = 256;

    std::atomic<bool> mStopWatcher{false};
    std::mutex mWatcherWaitMutex;
    mutable std::condition_variable mWatcherWakeup;
    std::thread mWatcherThread;
};

} // namespace tensorrt_llm::runtime
