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

#include "tensorrt_llm/runtime/ncclCommunicator.h"

#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/runtime/utils/multiDeviceUtils.h"
#include "tensorrt_llm/runtime/utils/ncclHostApi.h"
#include "tensorrt_llm/runtime/utils/ncclUniqueIdRendezvous.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <initializer_list>
#include <numeric>
#include <set>
#include <sstream>
#include <utility>

#if ENABLE_MULTI_DEVICE
#include <nccl.h>
#endif // ENABLE_MULTI_DEVICE

using namespace tensorrt_llm::runtime;

namespace
{
using namespace std::chrono_literals;

constexpr auto kInitialReadyTimeout = 120s;
#if ENABLE_MULTI_DEVICE
constexpr auto kReadyPollInterval = 1ms;
constexpr auto kRecoveryReadyTimeout = 5s;
constexpr auto kWatcherPollInterval = 100ms;
#endif

#if ENABLE_MULTI_DEVICE

ncclDataType_t toNcclType(nvinfer1::DataType dataType)
{
    switch (dataType)
    {
    case nvinfer1::DataType::kFLOAT: return ncclFloat32;
    case nvinfer1::DataType::kHALF: return ncclHalf;
    case nvinfer1::DataType::kINT8: return ncclInt8;
    case nvinfer1::DataType::kINT32: return ncclInt32;
    case nvinfer1::DataType::kUINT8: return ncclUint8;
    case nvinfer1::DataType::kINT64: return ncclInt64;
    case nvinfer1::DataType::kFP8: return ncclUint8;
#if ENABLE_BF16
    case nvinfer1::DataType::kBF16: return ncclBfloat16;
#endif // ENABLE_BF16
    default: TLLM_THROW("NCCL error: unsupported data type: %d", static_cast<int>(dataType));
    }
}

std::string ncclErrorMessage(char const* operation, ncclResult_t result)
{
    std::ostringstream message;
    message << "NCCL error during " << operation << ": " << ncclGetErrorString(result);
    return message.str();
}

std::chrono::milliseconds getOperationTimeout()
{
    static auto const timeout = []
    {
        constexpr int64_t defaultTimeoutMs = 5000;
        auto const* value = std::getenv("TRTLLM_NCCL_NONBLOCKING_TIMEOUT_MS");
        if (value != nullptr)
        {
            try
            {
                auto const parsed = std::stoll(value);
                if (parsed > 0)
                {
                    return std::chrono::milliseconds{parsed};
                }
            }
            catch (...)
            {
            }
            TLLM_LOG_WARNING(
                "Ignoring invalid TRTLLM_NCCL_NONBLOCKING_TIMEOUT_MS=%s; expected positive milliseconds", value);
        }
        return std::chrono::milliseconds{defaultTimeoutMs};
    }();
    return timeout;
}

#endif // ENABLE_MULTI_DEVICE
} // namespace

NcclCommunicator::NcclCommunicator(ncclComm_t comm)
    : mFaultToleranceEnabled{mpi::isFaultToleranceModeEnabled()}
    , mComm{comm}
{
#if ENABLE_MULTI_DEVICE
    TLLM_CHECK_WITH_INFO(mComm != nullptr, "NCCL error: cannot wrap a null communicator");

    auto const hostApiLock = acquireNcclHostApiLock();
    try
    {
        if (mFaultToleranceEnabled)
        {
            waitUntilReady(mComm, "wrapping communicator", std::chrono::steady_clock::now() + kInitialReadyTimeout);
        }
        TLLM_NCCL_CHECK(ncclCommCount(mComm, &mInitialWorldSize));
        TLLM_NCCL_CHECK(ncclCommUserRank(mComm, &mWorldRank));
        mActiveRanks.resize(mInitialWorldSize);
        std::iota(mActiveRanks.begin(), mActiveRanks.end(), 0);
    }
    catch (...)
    {
        ncclCommAbort(mComm);
        mComm = nullptr;
        throw;
    }
    startWatcher();
#else
    (void) comm;
#endif // ENABLE_MULTI_DEVICE
}

NcclCommunicator::NcclCommunicator(int worldSize, int rank, mpi::MpiComm const& mpiComm)
    : mInitialWorldSize{worldSize}
    , mWorldRank{rank}
    , mFaultToleranceEnabled{mpi::isFaultToleranceModeEnabled()}
{
    TLLM_CHECK_WITH_INFO(worldSize > 0, "NCCL error: communicator world size must be positive, got %d", worldSize);
    TLLM_CHECK_WITH_INFO(
        rank >= 0 && rank < worldSize, "NCCL error: communicator rank must be in [0, %d), got %d", worldSize, rank);

    mActiveRanks.resize(worldSize);
    std::iota(mActiveRanks.begin(), mActiveRanks.end(), 0);
#if ENABLE_MULTI_DEVICE
    TLLM_CHECK_WITH_INFO(mpiComm.getRank() == rank, "NCCL error: world rank %d does not match MPI bootstrap rank %d",
        rank, mpiComm.getRank());
    TLLM_CHECK_WITH_INFO(mpiComm.getSize() >= worldSize, "NCCL error: world size %d exceeds MPI bootstrap size %d",
        worldSize, mpiComm.getSize());
    if (mFaultToleranceEnabled)
    {
        mControlComm = createNcclUniqueIdRendezvousComm(
            mActiveRanks, rank, mpiComm, static_cast<int>(mpi::MpiTag::kNcclPpControl));
        mComm = createComm(mActiveRanks, rank, *mControlComm, /*rendezvousId=*/0, kInitialReadyTimeout);
    }
    else
    {
        // Preserve the legacy blocking MPI-broadcast bootstrap when FT is
        // disabled. It creates no control communicator or watcher thread.
        mComm = createLegacyComm(worldSize, rank, mpiComm);
    }
#else
    mComm = nullptr;
#endif // ENABLE_MULTI_DEVICE
    startWatcher();
}

void NcclCommunicator::send(
    void const* sendbuff, size_t count, nvinfer1::DataType dataType, int peer, CudaStream const& stream) const
{
#if ENABLE_MULTI_DEVICE
    if (!mFaultToleranceEnabled)
    {
        TLLM_NCCL_CHECK(ncclSend(sendbuff, count, toNcclType(dataType), peer, mComm, stream.get()));
        return;
    }
    std::lock_guard<std::mutex> lock(mCommMutex);
    auto const hostApiLock = acquireNcclHostApiLock();
    checkUsableLocked();
    int const commPeer = getCommPeerRankLocked(peer);
    auto const watchdogToken = beginOperationLocked(stream.get(), "ncclSend");
    ncclResult_t const result = ncclSend(sendbuff, count, toNcclType(dataType), commPeer, mComm, stream.get());
    if (result == ncclInProgress)
    {
        try
        {
            waitUntilReady(mComm, "ncclSend enqueue", std::chrono::steady_clock::now() + kRecoveryReadyTimeout);
        }
        catch (std::exception const& error)
        {
            static_cast<void>(abortLocked(
                std::string{"NCCL error: communicator was aborted after ncclSend failed: "} + error.what()));
            TLLM_THROW("%s", mAsyncError.c_str());
        }
        catch (...)
        {
            static_cast<void>(abortLocked("NCCL error: communicator was aborted after ncclSend failed"));
            TLLM_THROW("%s", mAsyncError.c_str());
        }
    }
    else if (result != ncclSuccess)
    {
        static_cast<void>(
            abortLocked("NCCL error: communicator was aborted after " + ncclErrorMessage("ncclSend", result)));
        TLLM_THROW("%s", mAsyncError.c_str());
    }
    finishOperationLocked(watchdogToken, stream.get());
#else
    TLLM_THROW("NCCL error: multi device support is disabled.");
#endif // ENABLE_MULTI_DEVICE
}

void NcclCommunicator::receive(
    void* sendbuff, size_t count, nvinfer1::DataType dataType, int peer, CudaStream const& stream) const
{
#if ENABLE_MULTI_DEVICE
    if (!mFaultToleranceEnabled)
    {
        TLLM_NCCL_CHECK(ncclRecv(sendbuff, count, toNcclType(dataType), peer, mComm, stream.get()));
        return;
    }
    std::lock_guard<std::mutex> lock(mCommMutex);
    auto const hostApiLock = acquireNcclHostApiLock();
    checkUsableLocked();
    int const commPeer = getCommPeerRankLocked(peer);
    auto const watchdogToken = beginOperationLocked(stream.get(), "ncclRecv");
    ncclResult_t const result = ncclRecv(sendbuff, count, toNcclType(dataType), commPeer, mComm, stream.get());
    if (result == ncclInProgress)
    {
        try
        {
            waitUntilReady(mComm, "ncclRecv enqueue", std::chrono::steady_clock::now() + kRecoveryReadyTimeout);
        }
        catch (std::exception const& error)
        {
            static_cast<void>(abortLocked(
                std::string{"NCCL error: communicator was aborted after ncclRecv failed: "} + error.what()));
            TLLM_THROW("%s", mAsyncError.c_str());
        }
        catch (...)
        {
            static_cast<void>(abortLocked("NCCL error: communicator was aborted after ncclRecv failed"));
            TLLM_THROW("%s", mAsyncError.c_str());
        }
    }
    else if (result != ncclSuccess)
    {
        static_cast<void>(
            abortLocked("NCCL error: communicator was aborted after " + ncclErrorMessage("ncclRecv", result)));
        TLLM_THROW("%s", mAsyncError.c_str());
    }
    finishOperationLocked(watchdogToken, stream.get());
#else
    TLLM_THROW("NCCL error: multi device support is disabled.");
#endif // ENABLE_MULTI_DEVICE
}

ncclComm_t NcclCommunicator::createComm(std::vector<int> const& activeRanks, int worldRank,
    NcclUniqueIdRendezvousComm const& controlComm, std::uint64_t rendezvousId, std::chrono::milliseconds readyTimeout)
{
#if ENABLE_MULTI_DEVICE
    TLLM_CHECK_WITH_INFO(!activeRanks.empty(), "NCCL error: communicator active-rank set must not be empty");
    auto const rankIt = std::find(activeRanks.begin(), activeRanks.end(), worldRank);
    TLLM_CHECK_WITH_INFO(
        rankIt != activeRanks.end(), "NCCL error: world rank %d is not present in the active-rank set", worldRank);

    // Do not use an MPI collective here. During recovery, failed ranks remain
    // members of the pre-failure control communicator. Only the survivors
    // exchange the fresh NCCL ID point-to-point.
    auto const deadline = std::chrono::steady_clock::now() + readyTimeout;
    TLLM_CHECK_WITH_INFO(controlComm.worldRank() == worldRank,
        "NCCL error: world rank %d does not match rendezvous control world rank %d", worldRank,
        controlComm.worldRank());
    ncclUniqueId const id = exchangeNcclUniqueId(activeRanks, controlComm,
        {mpi::MpiTag::kNcclPpReady, mpi::MpiTag::kNcclPpUniqueId, mpi::MpiTag::kNcclPpAck}, rendezvousId, deadline);

    ncclComm_t comm{nullptr};
// Need static connection initialization for accurate KV cache size estimation
#if defined(_WIN32)
    if (getenv("NCCL_RUNTIME_CONNECT") == nullptr)
        _putenv_s("NCCL_RUNTIME_CONNECT", "0");
#else
    setenv("NCCL_RUNTIME_CONNECT", "0", 0);
#endif // _WIN32

    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.blocking = 0;
    int const commRank = static_cast<int>(std::distance(activeRanks.begin(), rankIt));
    int const commSize = static_cast<int>(activeRanks.size());
    auto const hostApiLock = acquireNcclHostApiLock();
    ncclResult_t const result = ncclCommInitRankConfig(&comm, commSize, id, commRank, &config);
    if (result != ncclSuccess && result != ncclInProgress)
    {
        if (comm != nullptr)
        {
            ncclCommAbort(comm);
        }
        TLLM_THROW("%s", ncclErrorMessage("ncclCommInitRankConfig", result).c_str());
    }

    try
    {
        waitUntilReady(comm, "ncclCommInitRankConfig", deadline);
    }
    catch (...)
    {
        if (comm != nullptr)
        {
            ncclCommAbort(comm);
        }
        throw;
    }
    return comm;
#else
    (void) activeRanks;
    (void) worldRank;
    (void) controlComm;
    (void) rendezvousId;
    (void) readyTimeout;
    // Python runtime requires instantiation of a communicator even though it may never be used to enable
    // pipeline parallel code-path. To enable this, have an empty communicator with uninitialized state.
    return nullptr;
#endif // ENABLE_MULTI_DEVICE
}

ncclComm_t NcclCommunicator::createLegacyComm(int worldSize, int rank, mpi::MpiComm const& mpiComm)
{
#if ENABLE_MULTI_DEVICE
    ncclUniqueId id;
    if (rank == 0)
    {
        auto const hostApiLock = acquireNcclHostApiLock();
        TLLM_NCCL_CHECK(ncclGetUniqueId(&id));
    }
    mpiComm.bcastValue(id, 0);

// Need static connection initialization for accurate KV cache size estimation
#if defined(_WIN32)
    if (getenv("NCCL_RUNTIME_CONNECT") == nullptr)
        _putenv_s("NCCL_RUNTIME_CONNECT", "0");
#else
    setenv("NCCL_RUNTIME_CONNECT", "0", 0);
#endif // _WIN32

    ncclComm_t comm{nullptr};
    auto const hostApiLock = acquireNcclHostApiLock();
    TLLM_NCCL_CHECK(ncclCommInitRank(&comm, worldSize, id, rank));
    return comm;
#else
    (void) worldSize;
    (void) rank;
    (void) mpiComm;
    return nullptr;
#endif // ENABLE_MULTI_DEVICE
}

void NcclCommunicator::waitUntilReady(
    ncclComm_t comm, char const* operation, std::chrono::steady_clock::time_point deadline)
{
#if ENABLE_MULTI_DEVICE
    auto const hostApiLock = acquireNcclHostApiLock();
    TLLM_CHECK_WITH_INFO(comm != nullptr, "NCCL error: communicator was aborted while waiting for %s", operation);
    while (true)
    {
        ncclResult_t asyncError = ncclInProgress;
        ncclResult_t const queryResult = ncclCommGetAsyncError(comm, &asyncError);
        if (queryResult != ncclSuccess && queryResult != ncclInProgress)
        {
            TLLM_THROW("%s", ncclErrorMessage("ncclCommGetAsyncError", queryResult).c_str());
        }
        if (queryResult == ncclSuccess && asyncError == ncclSuccess)
        {
            return;
        }
        if (queryResult == ncclSuccess && asyncError != ncclInProgress)
        {
            TLLM_THROW("%s", ncclErrorMessage(operation, asyncError).c_str());
        }
        if (std::chrono::steady_clock::now() >= deadline)
        {
            TLLM_THROW("NCCL error: timed out waiting for %s", operation);
        }
        std::this_thread::sleep_for(kReadyPollInterval);
    }
#else
    (void) comm;
    (void) operation;
    (void) deadline;
#endif // ENABLE_MULTI_DEVICE
}

void NcclCommunicator::checkUsableLocked() const
{
    TLLM_CHECK_WITH_INFO(mAsyncError.empty(), "%s", mAsyncError.c_str());
    TLLM_CHECK_WITH_INFO(mComm != nullptr, "NCCL error: communicator was aborted");
}

bool NcclCommunicator::abortLocked(std::string reason) const noexcept
{
#if ENABLE_MULTI_DEVICE
    auto const hostApiLock = acquireNcclHostApiLock();
    if (mComm == nullptr)
    {
        if (mAsyncError.empty())
        {
            mAsyncError = std::move(reason);
        }
        return true;
    }

    ncclComm_t const comm = mComm;
    ncclResult_t const result = ncclCommAbort(comm);
    if (result == ncclSuccess)
    {
        mComm = nullptr;
        mAsyncError = std::move(reason);
        quarantinePendingOperationsLocked();
        return true;
    }

    bool const firstAbortFailure = reason.find("ncclCommAbort failed") == std::string::npos;
    mAsyncError = std::move(reason);
    if (firstAbortFailure)
    {
        try
        {
            mAsyncError += "; ncclCommAbort failed: ";
            mAsyncError += ncclGetErrorString(result);
        }
        catch (...)
        {
        }
    }
    if (firstAbortFailure)
    {
        TLLM_LOG_ERROR("Failed to abort NCCL communicator: %s", ncclGetErrorString(result));
    }
    return false;
#else
    (void) reason;
    return true;
#endif // ENABLE_MULTI_DEVICE
}

int NcclCommunicator::getCommPeerRankLocked(int worldRank) const
{
    auto const peer = std::find(mActiveRanks.begin(), mActiveRanks.end(), worldRank);
    TLLM_CHECK_WITH_INFO(peer != mActiveRanks.end(),
        "NCCL error: peer world rank %d is not active in the current communicator", worldRank);
    return static_cast<int>(std::distance(mActiveRanks.begin(), peer));
}

uint64_t NcclCommunicator::beginOperationLocked(cudaStream_t stream, char const* operation) const
{
#if ENABLE_MULTI_DEVICE
    checkUsableLocked();
    if (!mFaultToleranceEnabled)
    {
        return 0;
    }
    cudaStreamCaptureStatus captureStatus = cudaStreamCaptureStatusNone;
    auto const captureResult = cudaStreamIsCapturing(stream, &captureStatus);
    if (captureResult != cudaSuccess)
    {
        static_cast<void>(abortLocked(std::string{"NCCL error: "} + operation
            + " failed to query CUDA stream capture state: " + cudaGetErrorString(captureResult)));
        TLLM_THROW("%s", mAsyncError.c_str());
    }
    if (captureStatus != cudaStreamCaptureStatusNone)
    {
        TLLM_CHECK_WITH_INFO(!mFaultToleranceEnabled,
            "NCCL error: CUDA graph capture of PP NCCL operations is disabled while "
            "TLLM_FAULT_TOLERANCE_MODE=1; captured graph nodes retain the old communicator across recovery");
        TLLM_LOG_DEBUG("Skipping eager NCCL completion deadline for %s during CUDA graph capture", operation);
        return 0;
    }
    if (mPendingOperations.size() >= kMaxPendingOperations)
    {
        static_cast<void>(abortLocked(
            std::string{"NCCL error: "} + operation + " exceeded the bounded NCCL watchdog operation queue"));
        TLLM_THROW("%s", mAsyncError.c_str());
    }

    auto const token = mNextOperationToken++;
    auto const start = acquireEventLocked();
    mPendingOperations.push_back({token, start, nullptr, {}, false, operation});
    auto const result = cudaEventRecord(start, stream);
    if (result != cudaSuccess)
    {
        static_cast<void>(abortLocked(std::string{"NCCL error: "} + operation
            + " failed to record its start marker: " + cudaGetErrorString(result)));
        TLLM_THROW("%s", mAsyncError.c_str());
    }
    mWatcherWakeup.notify_all();
    return token;
#else
    (void) stream;
    (void) operation;
    return 0;
#endif // ENABLE_MULTI_DEVICE
}

void NcclCommunicator::finishOperationLocked(uint64_t token, cudaStream_t stream) const
{
#if ENABLE_MULTI_DEVICE
    checkUsableLocked();
    if (token == 0)
    {
        return;
    }
    auto const operation = std::find_if(mPendingOperations.begin(), mPendingOperations.end(),
        [token](PendingOperation const& pending) { return pending.token == token; });
    TLLM_CHECK_WITH_INFO(operation != mPendingOperations.end(), "NCCL error: unknown watchdog operation token %llu",
        static_cast<unsigned long long>(token));
    TLLM_CHECK_WITH_INFO(operation->completion == nullptr,
        "NCCL error: watchdog operation token %llu was already completed", static_cast<unsigned long long>(token));

    operation->completion = acquireEventLocked();
    auto const result = cudaEventRecord(operation->completion, stream);
    if (result != cudaSuccess)
    {
        static_cast<void>(abortLocked(std::string{"NCCL error: "} + operation->name
            + " failed to record its completion marker: " + cudaGetErrorString(result)));
        TLLM_THROW("%s", mAsyncError.c_str());
    }
    mWatcherWakeup.notify_all();
#else
    (void) token;
    (void) stream;
#endif // ENABLE_MULTI_DEVICE
}

cudaEvent_t NcclCommunicator::acquireEventLocked() const
{
#if ENABLE_MULTI_DEVICE
    if (!mEventPool.empty())
    {
        auto const event = mEventPool.back();
        mEventPool.pop_back();
        return event;
    }
    cudaEvent_t event = nullptr;
    auto const result = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
    if (result != cudaSuccess)
    {
        static_cast<void>(
            abortLocked(std::string{"NCCL error: failed to allocate a watchdog event: "} + cudaGetErrorString(result)));
        TLLM_THROW("%s", mAsyncError.c_str());
    }
    return event;
#else
    return nullptr;
#endif // ENABLE_MULTI_DEVICE
}

void NcclCommunicator::recycleEventLocked(cudaEvent_t event) const noexcept
{
#if ENABLE_MULTI_DEVICE
    if (event == nullptr)
    {
        return;
    }
    if (mEventPool.size() < kMaxPooledEvents)
    {
        mEventPool.push_back(event);
    }
    else
    {
        static_cast<void>(cudaEventDestroy(event));
    }
#else
    (void) event;
#endif // ENABLE_MULTI_DEVICE
}

void NcclCommunicator::quarantinePendingOperationsLocked() const noexcept
{
    if (!mPendingOperations.empty())
    {
        // The failed communicator may still have device work referencing
        // these events. Drop host ownership without destroying their handles;
        // process restart reclaims the deliberately quarantined resources.
        TLLM_LOG_WARNING(
            "Quarantining %zu PP NCCL completion markers after communicator abort", mPendingOperations.size());
    }
    mPendingOperations.clear();
}

void NcclCommunicator::destroyCompletedWatchdogEventsLocked() const noexcept
{
#if ENABLE_MULTI_DEVICE
    size_t quarantined = 0;
    for (auto const& operation : mPendingOperations)
    {
        for (auto const event : {operation.start, operation.completion})
        {
            if (event == nullptr)
            {
                continue;
            }
            if (cudaEventQuery(event) == cudaSuccess)
            {
                static_cast<void>(cudaEventDestroy(event));
            }
            else
            {
                ++quarantined;
            }
        }
    }
    mPendingOperations.clear();
    if (quarantined != 0)
    {
        TLLM_LOG_WARNING("Quarantining %zu incomplete PP NCCL watchdog events during healthy teardown", quarantined);
    }
    destroyPooledEventsLocked();
#endif // ENABLE_MULTI_DEVICE
}

void NcclCommunicator::destroyPooledEventsLocked() const noexcept
{
#if ENABLE_MULTI_DEVICE
    for (auto const event : mEventPool)
    {
        static_cast<void>(cudaEventDestroy(event));
    }
    mEventPool.clear();
#endif // ENABLE_MULTI_DEVICE
}

void NcclCommunicator::abort()
{
#if ENABLE_MULTI_DEVICE
    std::lock_guard<std::mutex> lock(mCommMutex);
    bool const abortSucceeded = abortLocked("NCCL error: communicator was aborted by the fault-tolerance control path");
    TLLM_CHECK_WITH_INFO(abortSucceeded, "%s", mAsyncError.c_str());
#endif // ENABLE_MULTI_DEVICE
}

void NcclCommunicator::abortAndReinit(std::vector<int> const& activeRanks, std::uint64_t rendezvousId)
{
#if ENABLE_MULTI_DEVICE
    std::lock_guard<std::mutex> lock(mCommMutex);
    TLLM_CHECK_WITH_INFO(mFaultToleranceEnabled,
        "NCCL error: communicator reinitialization requires TLLM_FAULT_TOLERANCE_MODE=1; otherwise captured CUDA "
        "graphs may retain the old communicator");
    TLLM_CHECK_WITH_INFO(
        rendezvousId != 0, "NCCL error: recovery rendezvous ID must be positive; zero is reserved for bootstrap");
    TLLM_CHECK_WITH_INFO(mControlComm != nullptr,
        "NCCL error: communicator created from a raw handle has no MPI bootstrap for reinitialization");
    TLLM_CHECK_WITH_INFO(!activeRanks.empty(), "NCCL error: active-rank set must not be empty");
    std::set<int> const requested(activeRanks.begin(), activeRanks.end());
    TLLM_CHECK_WITH_INFO(requested.size() == activeRanks.size(), "NCCL error: active-rank set contains duplicates");
    TLLM_CHECK_WITH_INFO(requested.find(mWorldRank) != requested.end(),
        "NCCL error: survivor rank %d must be present in active_ranks", mWorldRank);

    for (int const rank : requested)
    {
        TLLM_CHECK_WITH_INFO(rank >= 0 && rank < mInitialWorldSize,
            "NCCL error: active world rank %d is outside [0, %d)", rank, mInitialWorldSize);
        TLLM_CHECK_WITH_INFO(std::find(mActiveRanks.begin(), mActiveRanks.end(), rank) != mActiveRanks.end(),
            "NCCL error: cannot re-add world rank %d after it was removed", rank);
    }

    std::vector<int> nextActiveRanks;
    nextActiveRanks.reserve(requested.size());
    for (int const worldRank : mActiveRanks)
    {
        if (requested.find(worldRank) != requested.end())
        {
            nextActiveRanks.push_back(worldRank);
        }
    }

    TLLM_CHECK_WITH_INFO(mControlComm->worldRank() == mWorldRank,
        "NCCL error: world rank %d does not match rendezvous control world rank %d", mWorldRank,
        mControlComm->worldRank());
    // This is deliberately a fresh-ID rebuild rather than ncclCommShrink. The
    // watcher may already have destroyed the poisoned communicator, and the
    // failed rank must not be required by any recovery collective.
    bool const abortSucceeded
        = abortLocked("NCCL error: communicator was aborted before survivor-only reinitialization");
    TLLM_CHECK_WITH_INFO(abortSucceeded,
        "NCCL error: communicator reinitialization refused because ncclCommAbort did not succeed: %s",
        mAsyncError.c_str());

    ncclComm_t nextComm{nullptr};
    try
    {
        nextComm = createComm(nextActiveRanks, mWorldRank, *mControlComm, rendezvousId, kRecoveryReadyTimeout);
    }
    catch (std::exception const& error)
    {
        mAsyncError = std::string{"NCCL error: communicator reinitialization failed after abort: "} + error.what();
        TLLM_THROW("%s", mAsyncError.c_str());
    }
    catch (...)
    {
        mAsyncError = "NCCL error: communicator reinitialization failed after abort";
        TLLM_THROW("%s", mAsyncError.c_str());
    }

    mComm = nextComm;
    mActiveRanks = std::move(nextActiveRanks);
    mAsyncError.clear();
#else
    (void) activeRanks;
    (void) rendezvousId;
    TLLM_THROW("NCCL error: multi device support is disabled.");
#endif // ENABLE_MULTI_DEVICE
}

std::string NcclCommunicator::getAsyncError() const
{
    std::lock_guard<std::mutex> lock(mCommMutex);
    return mAsyncError;
}

std::vector<int> NcclCommunicator::getActiveRanks() const
{
    std::lock_guard<std::mutex> lock(mCommMutex);
    return mActiveRanks;
}

void NcclCommunicator::startWatcher()
{
#if ENABLE_MULTI_DEVICE
    if (mFaultToleranceEnabled && mComm != nullptr)
    {
        try
        {
            mWatcherThread = std::thread(&NcclCommunicator::watchAsyncErrors, this);
        }
        catch (...)
        {
            auto const hostApiLock = acquireNcclHostApiLock();
            ncclCommAbort(mComm);
            mComm = nullptr;
            throw;
        }
    }
#endif // ENABLE_MULTI_DEVICE
}

void NcclCommunicator::stopWatcher() noexcept
{
    mStopWatcher.store(true, std::memory_order_release);
    mWatcherWakeup.notify_all();
    if (mWatcherThread.joinable())
    {
        mWatcherThread.join();
    }
}

void NcclCommunicator::watchAsyncErrors() noexcept
{
#if ENABLE_MULTI_DEVICE
    try
    {
        while (!mStopWatcher.load(std::memory_order_acquire))
        {
            {
                // NCCL communicators are not generally thread-safe. Serialize
                // the monitor with send/receive, abort, and reinit host calls.
                std::lock_guard<std::mutex> lock(mCommMutex);
                if (mComm != nullptr && !mAsyncError.empty())
                {
                    // A previous abort attempt failed. Keep the handle so the
                    // watcher can retry; a rebuild is forbidden until abort
                    // succeeds.
                    static_cast<void>(abortLocked(mAsyncError));
                }
                else if (mComm != nullptr)
                {
                    std::string error;
                    auto const now = std::chrono::steady_clock::now();
                    for (auto operation = mPendingOperations.begin(); operation != mPendingOperations.end();)
                    {
                        if (operation->start != nullptr)
                        {
                            auto const startResult = cudaEventQuery(operation->start);
                            if (startResult == cudaSuccess)
                            {
                                recycleEventLocked(operation->start);
                                operation->start = nullptr;
                                operation->deadline = now + getOperationTimeout();
                                operation->armed = true;
                            }
                            else if (startResult != cudaErrorNotReady)
                            {
                                error = operation->name + " start marker failed: " + cudaGetErrorString(startResult);
                                break;
                            }
                            else
                            {
                                ++operation;
                                continue;
                            }
                        }
                        if (operation->completion == nullptr)
                        {
                            if (operation->armed && now >= operation->deadline)
                            {
                                error = operation->name + " timed out before its completion marker was recorded";
                                break;
                            }
                            ++operation;
                            continue;
                        }

                        auto const eventResult = cudaEventQuery(operation->completion);
                        if (eventResult == cudaSuccess)
                        {
                            recycleEventLocked(operation->completion);
                            operation = mPendingOperations.erase(operation);
                            continue;
                        }
                        if (eventResult != cudaErrorNotReady)
                        {
                            error = operation->name + " completion marker failed: " + cudaGetErrorString(eventResult);
                            break;
                        }
                        if (operation->armed && now >= operation->deadline)
                        {
                            error = operation->name + " timed out before its CUDA completion marker";
                            break;
                        }
                        ++operation;
                    }

                    if (error.empty())
                    {
                        auto const hostApiLock = acquireNcclHostApiLock();
                        ncclResult_t asyncError = ncclSuccess;
                        ncclResult_t const queryResult = ncclCommGetAsyncError(mComm, &asyncError);
                        if (queryResult != ncclSuccess && queryResult != ncclInProgress)
                        {
                            error = ncclErrorMessage("ncclCommGetAsyncError watcher", queryResult);
                        }
                        else if (queryResult == ncclSuccess && asyncError != ncclSuccess
                            && asyncError != ncclInProgress)
                        {
                            error = ncclErrorMessage("asynchronous communicator operation", asyncError);
                        }
                    }

                    if (!error.empty())
                    {
                        static_cast<void>(abortLocked(
                            "NCCL error: communicator was aborted by the async-error watcher after " + error));
                        TLLM_LOG_ERROR("%s", mAsyncError.c_str());
                    }
                }
            }

            std::unique_lock<std::mutex> waitLock(mWatcherWaitMutex);
            mWatcherWakeup.wait_for(
                waitLock, kWatcherPollInterval, [this]() { return mStopWatcher.load(std::memory_order_acquire); });
        }
    }
    catch (...)
    {
        // Never allow a diagnostic side thread to terminate the process.
    }
#endif // ENABLE_MULTI_DEVICE
}

NcclCommunicator::~NcclCommunicator()
{
#if ENABLE_MULTI_DEVICE
    if (!mFaultToleranceEnabled)
    {
        if (mComm != nullptr && ncclCommDestroy(mComm) != ncclSuccess)
        {
            TLLM_LOG_WARNING("Failed to destroy NCCL communicator.");
        }
        return;
    }
#endif // ENABLE_MULTI_DEVICE
    stopWatcher();
#if ENABLE_MULTI_DEVICE
    std::lock_guard<std::mutex> lock(mCommMutex);
    auto const hostApiLock = acquireNcclHostApiLock();
    if (mComm != nullptr)
    {
        ncclResult_t asyncError = ncclSuccess;
        ncclResult_t const queryResult = ncclCommGetAsyncError(mComm, &asyncError);
        bool const healthy = mAsyncError.empty() && queryResult == ncclSuccess && asyncError == ncclSuccess;
        // ncclCommDestroy is an intra-node collective. Python model-engine
        // reference counts are process-local, so their final release is not a
        // cross-rank teardown barrier; after a peer failure, the one healthy
        // sample above is also inherently racy. In FT mode use the local abort
        // path and never stop the watcher only to enter an unbounded graceful
        // destroy while holding the process-wide NCCL gate.
        bool const useAbort = !healthy || mFaultToleranceEnabled;
        ncclResult_t const result = useAbort ? ncclCommAbort(mComm) : ncclCommDestroy(mComm);
        if (result != ncclSuccess)
        {
            TLLM_LOG_WARNING("Failed to release NCCL communicator: %s", ncclGetErrorString(result));
        }
        mComm = nullptr;
        if (!useAbort && result == ncclSuccess)
        {
            destroyCompletedWatchdogEventsLocked();
        }
        else
        {
            quarantinePendingOperationsLocked();
            destroyPooledEventsLocked();
        }
    }
    else
    {
        destroyPooledEventsLocked();
    }
#endif // ENABLE_MULTI_DEVICE
}
