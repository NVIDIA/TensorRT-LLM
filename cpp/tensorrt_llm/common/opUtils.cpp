/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/common/ncclUtils.h"
#include "tensorrt_llm/runtime/ipcNvlsMemory.h"
#include "tensorrt_llm/runtime/utils/mpiTags.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include "tensorrt_llm/runtime/utils/ncclHostApi.h"
#include "tensorrt_llm/runtime/utils/ncclUniqueIdRendezvous.h"

#include "cuda.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <mutex>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <utility>

TRTLLM_NAMESPACE_BEGIN
#if ENABLE_MULTI_DEVICE

std::unordered_map<nvinfer1::DataType, ncclDataType_t>* getDtypeMap()
{
    static std::unordered_map<nvinfer1::DataType, ncclDataType_t> dtypeMap = {
        {nvinfer1::DataType::kFLOAT, ncclFloat32},
        {nvinfer1::DataType::kHALF, ncclFloat16},
        {nvinfer1::DataType::kBF16, ncclBfloat16},
        {nvinfer1::DataType::kFP8, ncclInt8},
        {nvinfer1::DataType::kBOOL, ncclInt8},
        {nvinfer1::DataType::kINT32, ncclInt32},
        {nvinfer1::DataType::kINT64, ncclInt64},
        {nvinfer1::DataType::kUINT8, ncclUint8},
        {nvinfer1::DataType::kINT8, ncclInt8},
    };
    return &dtypeMap;
}

namespace
{

using Clock = std::chrono::steady_clock;

std::chrono::milliseconds getPositiveDurationFromEnv(char const* name, int64_t defaultMs)
{
    auto const* value = std::getenv(name);
    if (value == nullptr)
    {
        return std::chrono::milliseconds{defaultMs};
    }
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
    TLLM_LOG_WARNING("Ignoring invalid %s=%s; expected a positive integer in milliseconds", name, value);
    return std::chrono::milliseconds{defaultMs};
}

std::string groupToString(std::set<int> const& group)
{
    std::ostringstream oss;
    for (auto it = group.begin(); it != group.end(); ++it)
    {
        if (it != group.begin())
        {
            oss << ',';
        }
        oss << *it;
    }
    return oss.str();
}

ncclUniqueId getLegacyUniqueId(std::set<int> const& group)
{
    auto const rank = COMM_SESSION.getRank();
    ncclUniqueId id{};
    if (rank == *group.begin())
    {
        {
            auto hostApiLock = tensorrt_llm::runtime::acquireNcclHostApiLock();
            NCCLCHECK_THROW(ncclGetUniqueId(&id));
        }
        for (auto it = std::next(group.begin()); it != group.end(); ++it)
        {
            COMM_SESSION.sendValue(id, *it, tensorrt_llm::mpi::MpiTag::kDefault);
        }
    }
    else
    {
        COMM_SESSION.recvValue(id, *group.begin(), tensorrt_llm::mpi::MpiTag::kDefault);
    }
    return id;
}

void configureNcclEnvironment()
{
#if defined(_WIN32)
    // Need static connection initialization for accurate KV cache size estimation
    if (getenv("NCCL_RUNTIME_CONNECT") == nullptr)
        _putenv_s("NCCL_RUNTIME_CONNECT", "0");
    // Disable graph register to avoid startup hangs
    if (getenv("NCCL_GRAPH_REGISTER") == nullptr)
        _putenv_s("NCCL_GRAPH_REGISTER", "0");
#else
    setenv("NCCL_RUNTIME_CONNECT", "0", 0);
    setenv("NCCL_GRAPH_REGISTER", "0", 0);
    // NCCL aborts during init if it tries NVLS multicast but the fabric/IMEX
    // plane can't bind it. Disable NVLS when it isn't actually usable so NCCL
    // falls back to NVLink P2P. No-overwrite preserves an explicit user setting.
    if (!tensorrt_llm::runtime::ipcNvlsSupported())
    {
        setenv("NCCL_NVLS_ENABLE", "0", 0);
    }
#endif // _WIN32
}

class NcclCommState
{
public:
    explicit NcclCommState(std::set<int> group,
        std::shared_ptr<tensorrt_llm::runtime::NcclUniqueIdRendezvousComm> controlComm = nullptr, bool recovery = false,
        std::uint64_t rendezvousId = 0)
        : mGroup(std::move(group))
        , mControlComm(std::move(controlComm))
        , mWorldRank(mControlComm != nullptr ? mControlComm->worldRank() : COMM_SESSION.getRank())
        , mFaultToleranceEnabled(tensorrt_llm::isNcclFaultToleranceEnabled())
        , mPollInterval(mFaultToleranceEnabled
                  ? getPositiveDurationFromEnv("TRTLLM_NCCL_WATCHDOG_POLL_INTERVAL_MS", 100)
                  : std::chrono::milliseconds{100})
        , mOperationTimeout(mFaultToleranceEnabled
                  ? getPositiveDurationFromEnv("TRTLLM_NCCL_NONBLOCKING_TIMEOUT_MS", 5000)
                  : std::chrono::milliseconds{5000})
        , mInitTimeout(mFaultToleranceEnabled
                  ? (recovery ? getPositiveDurationFromEnv("TRTLLM_NCCL_RECOVERY_TIMEOUT_MS", 5000)
                              : getPositiveDurationFromEnv("TRTLLM_NCCL_INIT_TIMEOUT_MS", 120000))
                  : std::chrono::milliseconds{120000})
    {
        TLLM_CHECK_WITH_INFO(!recovery || mFaultToleranceEnabled,
            "NCCL error: recovery communicator construction requires TLLM_FAULT_TOLERANCE_MODE=1");
        if (mFaultToleranceEnabled && mControlComm == nullptr)
        {
            std::vector<int> const initialRanks(mGroup.begin(), mGroup.end());
            mControlComm = tensorrt_llm::runtime::createNcclUniqueIdRendezvousComm(
                initialRanks, mWorldRank, COMM_SESSION, static_cast<int>(tensorrt_llm::mpi::MpiTag::kNcclCommControl));
        }
        initialize(rendezvousId);
        if (!mFaultToleranceEnabled)
        {
            return;
        }
        try
        {
            mWatchdog = std::thread([this]() { watchdogLoop(); });
        }
        catch (...)
        {
            ncclComm_t abortedComm = nullptr;
            if (mComm != nullptr)
            {
                auto const comm = mComm;
                ncclResult_t abortResult;
                {
                    auto const hostApiLock = tensorrt_llm::runtime::acquireNcclHostApiLock();
                    abortResult = ncclCommAbort(comm);
                    if (abortResult == ncclSuccess)
                    {
                        common::nccl_util::NcclCommResourceManager::getInstance().beginAbortCleanup(comm);
                    }
                }
                if (abortResult == ncclSuccess)
                {
                    mComm = nullptr;
                    abortedComm = comm;
                }
                else
                {
                    TLLM_LOG_ERROR("NCCL communicator leaked after watchdog thread creation and abort failed: %s",
                        ncclGetErrorString(abortResult));
                }
            }
            if (abortedComm != nullptr)
            {
                static_cast<void>(
                    common::nccl_util::NcclCommResourceManager::getInstance().cleanupResources(abortedComm, true));
            }
            throw;
        }
    }

    ~NcclCommState()
    {
        {
            std::lock_guard<std::mutex> lock(mMutex);
            mStop = true;
        }
        mWakeup.notify_all();
        if (mWatchdog.joinable())
        {
            mWatchdog.join();
        }

        cleanupAbortedResources();

        std::lock_guard<std::mutex> lock(mMutex);
        if (mComm != nullptr)
        {
            auto const comm = mComm;
            mComm = nullptr;
            if (mError.has_value() || mFaultToleranceEnabled)
            {
                ncclResult_t abortResult;
                {
                    auto const hostApiLock = tensorrt_llm::runtime::acquireNcclHostApiLock();
                    abortResult = ncclCommAbort(comm);
                    if (abortResult == ncclSuccess)
                    {
                        common::nccl_util::NcclCommResourceManager::getInstance().beginAbortCleanup(comm);
                    }
                }
                if (abortResult == ncclSuccess)
                {
                    static_cast<void>(
                        common::nccl_util::NcclCommResourceManager::getInstance().cleanupResources(comm, true));
                    quarantinePendingOperationsLocked();
                }
                else
                {
                    // Do not release registered buffers while a failed abort
                    // may have left kernels referencing them.
                    try
                    {
                        TLLM_LOG_WARNING("Leaking failed NCCL communicator for group (%s) after abort error: %s",
                            groupToString(mGroup).c_str(), ncclGetErrorString(abortResult));
                    }
                    catch (...)
                    {
                    }
                }
                if (abortResult != ncclSuccess)
                {
                    quarantinePendingOperationsLocked();
                }
                destroyPooledEventsLocked();
                return;
            }
            bool const resourcesCleaned
                = common::nccl_util::NcclCommResourceManager::getInstance().cleanupResources(comm);
            if (!resourcesCleaned)
            {
                ncclResult_t abortResult;
                {
                    auto const hostApiLock = tensorrt_llm::runtime::acquireNcclHostApiLock();
                    abortResult = ncclCommAbort(comm);
                }
                if (abortResult != ncclSuccess)
                {
                    try
                    {
                        TLLM_LOG_WARNING(
                            "NCCL resource cleanup did not complete and ncclCommAbort failed for group (%s): %s",
                            groupToString(mGroup).c_str(), ncclGetErrorString(abortResult));
                    }
                    catch (...)
                    {
                    }
                }
                quarantinePendingOperationsLocked();
                destroyPooledEventsLocked();
                return;
            }
            ncclResult_t result;
            {
                auto const hostApiLock = tensorrt_llm::runtime::acquireNcclHostApiLock();
                ncclResult_t asyncResult = ncclInProgress;
                auto const queryResult = ncclCommGetAsyncError(comm, &asyncResult);
                if (queryResult != ncclSuccess || asyncResult != ncclSuccess)
                {
                    // The watcher is already stopped. Re-sample immediately
                    // before destroy so a late asynchronous error cannot turn
                    // ncclCommDestroy into an unbounded blocking call.
                    result = ncclCommAbort(comm);
                    if (result != ncclSuccess)
                    {
                        try
                        {
                            TLLM_LOG_WARNING("Final NCCL health check failed and abort failed for group (%s): %s",
                                groupToString(mGroup).c_str(), ncclGetErrorString(result));
                        }
                        catch (...)
                        {
                        }
                    }
                    quarantinePendingOperationsLocked();
                    destroyPooledEventsLocked();
                    return;
                }
                result = ncclCommDestroy(comm);
            }
            if (result != ncclSuccess)
            {
                try
                {
                    TLLM_LOG_WARNING("ncclCommDestroy failed for group (%s): %s", groupToString(mGroup).c_str(),
                        ncclGetErrorString(result));
                }
                catch (...)
                {
                }
            }
            if (result == ncclSuccess)
            {
                destroyCompletedWatchdogEventsLocked();
            }
            else
            {
                quarantinePendingOperationsLocked();
                destroyPooledEventsLocked();
            }
            return;
        }
        destroyPooledEventsLocked();
    }

    NcclCommState(NcclCommState const&) = delete;
    NcclCommState& operator=(NcclCommState const&) = delete;

    ncclComm_t requireCommLocked() const
    {
        if (mError.has_value())
        {
            TLLM_THROW("NCCL error: communicator was aborted: %s", mError->c_str());
        }
        TLLM_CHECK_WITH_INFO(mComm != nullptr, "NCCL error: communicator was aborted");
        return mComm;
    }

    void checkResultLocked(ncclResult_t result, char const* operation)
    {
        if (result == ncclSuccess)
        {
            return;
        }

        auto const deadline = Clock::now() + mOperationTimeout;
        while (result == ncclInProgress && Clock::now() < deadline)
        {
            ncclResult_t asyncResult = ncclInProgress;
            auto const queryResult = ncclCommGetAsyncError(requireCommLocked(), &asyncResult);
            if (queryResult != ncclSuccess && queryResult != ncclInProgress)
            {
                result = queryResult;
                break;
            }
            if (queryResult == ncclSuccess && asyncResult == ncclSuccess)
            {
                return;
            }
            if (queryResult == ncclSuccess && asyncResult != ncclInProgress)
            {
                result = asyncResult;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds{1});
        }

        std::string message;
        if (result == ncclInProgress)
        {
            message = std::string(operation) + " timed out while the NCCL communicator was in progress";
        }
        else
        {
            message = std::string(operation) + " failed with NCCL error: " + ncclGetErrorString(result);
        }
        setErrorAndAbortLocked(message);
        TLLM_THROW("NCCL error: communicator was aborted: %s", message.c_str());
    }

    ncclResult_t checkOptionalResultLocked(ncclResult_t result, char const* operation)
    {
        if (result == ncclSuccess || result == ncclInvalidArgument)
        {
            return result;
        }
        if (result != ncclInProgress)
        {
            checkResultLocked(result, operation);
        }

        auto const deadline = Clock::now() + mOperationTimeout;
        while (Clock::now() < deadline)
        {
            ncclResult_t asyncResult = ncclInProgress;
            auto const queryResult = ncclCommGetAsyncError(requireCommLocked(), &asyncResult);
            if (queryResult != ncclSuccess && queryResult != ncclInProgress)
            {
                checkResultLocked(queryResult, "ncclCommGetAsyncError(optional NCCL operation)");
            }
            if (queryResult == ncclSuccess)
            {
                if (asyncResult == ncclSuccess)
                {
                    return asyncResult;
                }
                if (asyncResult != ncclInProgress)
                {
                    // ncclInvalidArgument is a benign capability fallback only
                    // when the optional API returns it synchronously. Once the
                    // call reports ncclInProgress, every eventual non-success
                    // result is an asynchronous communicator failure.
                    checkResultLocked(asyncResult, operation);
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds{1});
        }

        auto const message = std::string(operation) + " timed out while the optional NCCL operation was in progress";
        setErrorAndAbortLocked(message);
        TLLM_THROW("NCCL error: communicator was aborted: %s", message.c_str());
    }

    void checkEnqueueResultLocked(ncclResult_t result, char const* operation)
    {
        if (result == ncclSuccess || result == ncclInProgress)
        {
            return;
        }
        auto const message = std::string(operation) + " failed with NCCL error: " + ncclGetErrorString(result);
        setErrorAndAbortLocked(message);
        TLLM_THROW("NCCL error: communicator was aborted: %s", message.c_str());
    }

    void waitOperation(std::uint64_t token, char const* operation)
    {
        TLLM_CHECK_WITH_INFO(token != 0, "NCCL error: watchdog operation token must not be zero in FT mode");

        while (true)
        {
            std::string pendingError;
            {
                std::lock_guard<std::mutex> lock(mMutex);
                if (mError.has_value())
                {
                    pendingError = *mError;
                }
                else
                {
                    TLLM_CHECK_WITH_INFO(token < mNextOperationToken,
                        "NCCL error: unknown watchdog operation token %llu", static_cast<unsigned long long>(token));
                    auto const pending = std::find_if(mPendingOperations.begin(), mPendingOperations.end(),
                        [token](PendingOperation const& candidate) { return candidate.token == token; });
                    // The watchdog may have observed and retired a fast
                    // operation between track() releasing its lease and this
                    // waiter acquiring the state mutex.
                    if (pending == mPendingOperations.end())
                    {
                        return;
                    }
                    TLLM_CHECK_WITH_INFO(pending->completion != nullptr,
                        "NCCL error: watchdog operation token %llu has not been tracked",
                        static_cast<unsigned long long>(token));

                    auto const now = Clock::now();
                    if (pending->start != nullptr)
                    {
                        auto const startResult = cudaEventQuery(pending->start);
                        if (startResult == cudaSuccess)
                        {
                            recycleEventLocked(pending->start);
                            pending->start = nullptr;
                            pending->deadline = now + mOperationTimeout;
                            pending->armed = true;
                        }
                        else if (startResult != cudaErrorNotReady)
                        {
                            pendingError = pending->name + " start marker failed: " + cudaGetErrorString(startResult);
                        }
                    }

                    bool completionReady = false;
                    if (pendingError.empty() && pending->start == nullptr)
                    {
                        auto const completionResult = cudaEventQuery(pending->completion);
                        if (completionResult == cudaSuccess)
                        {
                            completionReady = true;
                        }
                        else if (completionResult != cudaErrorNotReady)
                        {
                            pendingError
                                = pending->name + " failed with CUDA error: " + cudaGetErrorString(completionResult);
                        }
                        else if (pending->armed && now >= pending->deadline)
                        {
                            pendingError = pending->name + " timed out before its CUDA completion marker";
                        }
                    }

                    bool communicatorReady = false;
                    if (pendingError.empty())
                    {
                        auto const hostApiLock = tensorrt_llm::runtime::acquireNcclHostApiLock();
                        ncclResult_t asyncResult = ncclInProgress;
                        auto const queryResult = ncclCommGetAsyncError(requireCommLocked(), &asyncResult);
                        if (queryResult != ncclSuccess && queryResult != ncclInProgress)
                        {
                            pendingError = std::string(operation)
                                + " failed while querying NCCL: " + ncclGetErrorString(queryResult);
                        }
                        else if (queryResult == ncclSuccess && asyncResult != ncclSuccess
                            && asyncResult != ncclInProgress)
                        {
                            pendingError = std::string(operation)
                                + " failed with NCCL error: " + ncclGetErrorString(asyncResult);
                        }
                        else
                        {
                            communicatorReady = queryResult == ncclSuccess && asyncResult == ncclSuccess;
                        }
                    }

                    if (pendingError.empty() && completionReady && communicatorReady)
                    {
                        recycleEventLocked(pending->completion);
                        mPendingOperations.erase(pending);
                        return;
                    }
                    if (!pendingError.empty())
                    {
                        setErrorAndAbortLocked(pendingError);
                        pendingError = mError.value_or(pendingError);
                    }
                }
            }

            if (!pendingError.empty())
            {
                cleanupAbortedResources();
                TLLM_THROW("NCCL error: communicator was aborted: %s", pendingError.c_str());
            }
            std::this_thread::sleep_for(std::chrono::milliseconds{1});
        }
    }

    uint64_t beginOperationLocked(cudaStream_t stream, char const* operation)
    {
        requireCommLocked();
        if (!mFaultToleranceEnabled)
        {
            return 0;
        }

        cudaStreamCaptureStatus captureStatus = cudaStreamCaptureStatusNone;
        auto const captureResult = cudaStreamIsCapturing(stream, &captureStatus);
        if (captureResult != cudaSuccess)
        {
            auto const message = std::string(operation)
                + " failed to query CUDA stream capture state: " + cudaGetErrorString(captureResult);
            setErrorAndAbortLocked(message);
            TLLM_THROW("NCCL error: communicator was aborted: %s", message.c_str());
        }
        if (captureStatus != cudaStreamCaptureStatusNone)
        {
            TLLM_CHECK_WITH_INFO(!mFaultToleranceEnabled,
                "NCCL error: CUDA graph capture of managed NCCL operations is disabled while "
                "TLLM_FAULT_TOLERANCE_MODE=1; captured graph nodes retain the old communicator across recovery");
            // An event recorded while capturing belongs to the graph and may
            // not execute until long after capture ends. Arming a wall-clock
            // deadline here would therefore report a false timeout before the
            // graph is launched. Non-FT graph launches retain legacy async
            // error monitoring; FT mode above requires eager launches so every
            // operation receives the completion deadline below.
            TLLM_LOG_DEBUG("Skipping eager NCCL completion deadline for %s during CUDA graph capture", operation);
            return 0;
        }

        if (mPendingOperations.size() >= kMaxPendingOperations)
        {
            auto const message = std::string(operation) + " exceeded the bounded NCCL watchdog operation queue";
            setErrorAndAbortLocked(message);
            TLLM_THROW("NCCL error: communicator was aborted: %s", message.c_str());
        }

        auto const token = mNextOperationToken++;
        cudaEvent_t start = acquireEventLocked();
        mPendingOperations.push_back({token, start, nullptr, {}, false, operation});
        auto const result = cudaEventRecord(start, stream);
        if (result != cudaSuccess)
        {
            auto const message
                = std::string(operation) + " failed to record its start marker: " + cudaGetErrorString(result);
            setErrorAndAbortLocked(message);
            TLLM_THROW("NCCL error: communicator was aborted: %s", message.c_str());
        }

        mWakeup.notify_all();
        return token;
    }

    void finishOperationLocked(uint64_t token, cudaStream_t stream)
    {
        requireCommLocked();
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
            auto const message
                = operation->name + " failed to record its completion marker: " + cudaGetErrorString(result);
            setErrorAndAbortLocked(message);
            TLLM_THROW("NCCL error: communicator was aborted: %s", message.c_str());
        }
        mWakeup.notify_all();
    }

    void abort(std::string const& reason)
    {
        {
            std::lock_guard<std::mutex> lock(mMutex);
            auto hostApiLock = tensorrt_llm::runtime::acquireNcclHostApiLock();
            setErrorAndAbortLocked(reason);
            TLLM_CHECK_WITH_INFO(mComm == nullptr,
                "NCCL error: communicator abort failed; refusing to initialize a replacement while the old handle "
                "is live");
        }
        cleanupAbortedResources();
    }

    std::optional<std::string> getError() const
    {
        std::lock_guard<std::mutex> lock(mMutex);
        return mError;
    }

    std::shared_ptr<tensorrt_llm::runtime::NcclUniqueIdRendezvousComm> const& controlComm() const noexcept
    {
        return mControlComm;
    }

    int worldRank() const noexcept
    {
        return mWorldRank;
    }

    ncclComm_t* slot() noexcept
    {
        return &mComm;
    }

    std::mutex& mutex() noexcept
    {
        return mMutex;
    }

    void cleanupAbortedResources() noexcept
    {
        std::vector<ncclComm_t> abortedComms;
        {
            std::lock_guard<std::mutex> lock(mMutex);
            abortedComms.swap(mPendingAbortCleanup);
        }
        for (auto const comm : abortedComms)
        {
            static_cast<void>(common::nccl_util::NcclCommResourceManager::getInstance().cleanupResources(comm, true));
        }
    }

private:
    void initialize(std::uint64_t rendezvousId)
    {
        configureNcclEnvironment();
        auto const rank = mWorldRank;
        TLLM_CHECK_WITH_INFO(!mGroup.empty(), "NCCL communicator group must not be empty");
        auto const rankIt = mGroup.find(rank);
        TLLM_CHECK_WITH_INFO(rankIt != mGroup.end(), "Global rank %d is not in NCCL communicator group (%s)", rank,
            groupToString(mGroup).c_str());

        auto const groupRank = static_cast<int>(std::distance(mGroup.begin(), rankIt));
        if (!mFaultToleranceEnabled)
        {
            auto const id = getLegacyUniqueId(mGroup);
            auto hostApiLock = tensorrt_llm::runtime::acquireNcclHostApiLock();
            ncclComm_t comm = nullptr;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 0)
            ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
            config.graphUsageMode = 1;
            auto const result = ncclCommInitRankConfig(&comm, static_cast<int>(mGroup.size()), id, groupRank, &config);
#else
            auto const result = ncclCommInitRank(&comm, static_cast<int>(mGroup.size()), id, groupRank);
#endif // NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 0)
            if (result != ncclSuccess)
            {
                if (comm != nullptr)
                {
                    static_cast<void>(ncclCommAbort(comm));
                }
                TLLM_THROW("NCCL communicator initialization failed for group (%s) with NCCL error: %s",
                    groupToString(mGroup).c_str(), ncclGetErrorString(result));
            }
            hostApiLock.unlock();
            common::nccl_util::NcclCommResourceManager::getInstance().waitForAbortCleanup(comm);
            mComm = comm;
            return;
        }

        TLLM_CHECK_WITH_INFO(mControlComm != nullptr, "NCCL rendezvous control channel is not initialized");
        TLLM_CHECK_WITH_INFO(rank == mControlComm->worldRank(),
            "NCCL error: MPI rank %d does not match rendezvous world rank %d", rank, mControlComm->worldRank());
        auto const deadline = Clock::now() + mInitTimeout;
        std::vector<int> const group(mGroup.begin(), mGroup.end());
        auto const id = tensorrt_llm::runtime::exchangeNcclUniqueId(group, *mControlComm,
            {tensorrt_llm::mpi::MpiTag::kNcclCommReady, tensorrt_llm::mpi::MpiTag::kNcclCommUniqueId,
                tensorrt_llm::mpi::MpiTag::kNcclCommAck},
            rendezvousId, deadline);
        ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
        config.blocking = 0;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 0)
        config.graphUsageMode = 1;
#endif // NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 0)

        auto hostApiLock = tensorrt_llm::runtime::acquireNcclHostApiLock();
        ncclComm_t comm = nullptr;
        auto result = ncclCommInitRankConfig(&comm, static_cast<int>(mGroup.size()), id, groupRank, &config);
        while (result == ncclInProgress && Clock::now() < deadline)
        {
            ncclResult_t asyncResult = ncclInProgress;
            auto const queryResult = ncclCommGetAsyncError(comm, &asyncResult);
            if (queryResult != ncclSuccess && queryResult != ncclInProgress)
            {
                result = queryResult;
                break;
            }
            if (queryResult == ncclSuccess && asyncResult == ncclSuccess)
            {
                result = ncclSuccess;
                break;
            }
            if (queryResult == ncclSuccess && asyncResult != ncclInProgress)
            {
                result = asyncResult;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds{1});
        }

        if (result != ncclSuccess)
        {
            if (comm != nullptr)
            {
                ncclCommAbort(comm);
            }
            if (result == ncclInProgress)
            {
                TLLM_THROW(
                    "NCCL error: communicator initialization timed out for group (%s)", groupToString(mGroup).c_str());
            }
            TLLM_THROW("NCCL communicator initialization failed for group (%s) with NCCL error: %s",
                groupToString(mGroup).c_str(), ncclGetErrorString(result));
        }
        hostApiLock.unlock();
        common::nccl_util::NcclCommResourceManager::getInstance().waitForAbortCleanup(comm);
        mComm = comm;
    }

    void setErrorAndAbortLocked(std::string const& reason)
    {
        auto const hostApiLock = tensorrt_llm::runtime::acquireNcclHostApiLock();
        if (!mError.has_value())
        {
            mError = reason;
        }
        if (mComm == nullptr)
        {
            return;
        }

        auto const comm = mComm;
        // Abort first: graceful window cleanup synchronizes the entire device
        // and can hang forever while a failed NCCL kernel is still resident.
        auto const abortResult = ncclCommAbort(comm);
        if (abortResult == ncclSuccess)
        {
            common::nccl_util::NcclCommResourceManager::getInstance().beginAbortCleanup(comm);
            mComm = nullptr;
            quarantinePendingOperationsLocked();
            // Cleanup is deferred until the operation lease and any allocator
            // lock have unwound. Running callbacks here can recursively enter
            // NCCLWindowAllocator and deadlock its non-recursive mutex.
            mPendingAbortCleanup.push_back(comm);
        }
        else
        {
            // Keep both the handle and registered buffers alive. Releasing
            // them while NCCL kernels may still reference them is unsafe; a
            // later control-path call can retry the idempotent abort.
            TLLM_LOG_WARNING("ncclCommAbort failed for group (%s): %s", groupToString(mGroup).c_str(),
                ncclGetErrorString(abortResult));
        }
    }

    void quarantinePendingOperationsLocked() noexcept
    {
        if (!mPendingOperations.empty())
        {
            // cudaEventDestroy is asynchronous, but the event may still be
            // referenced by work from the failed communicator. Keep the
            // handles alive for the rest of the process instead of making
            // assumptions about CUDA progress after abort.
            TLLM_LOG_WARNING(
                "Quarantining %zu NCCL completion events after communicator abort", mPendingOperations.size());
        }
        mPendingOperations.clear();
    }

    void destroyCompletedWatchdogEventsLocked() noexcept
    {
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
            TLLM_LOG_WARNING("Quarantining %zu incomplete NCCL watchdog events during healthy teardown", quarantined);
        }
        for (auto const event : mEventPool)
        {
            static_cast<void>(cudaEventDestroy(event));
        }
        mEventPool.clear();
    }

    void destroyPooledEventsLocked() noexcept
    {
        for (auto const event : mEventPool)
        {
            static_cast<void>(cudaEventDestroy(event));
        }
        mEventPool.clear();
    }

    cudaEvent_t acquireEventLocked()
    {
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
            auto const message
                = std::string("failed to allocate an NCCL watchdog event: ") + cudaGetErrorString(result);
            setErrorAndAbortLocked(message);
            TLLM_THROW("NCCL error: communicator was aborted: %s", message.c_str());
        }
        return event;
    }

    void recycleEventLocked(cudaEvent_t event) noexcept
    {
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
            // The event has completed, so releasing pool overflow cannot race
            // device work. Keeping the pool bounded avoids unbounded steady-
            // state driver resources under bursty workloads.
            static_cast<void>(cudaEventDestroy(event));
        }
    }

    void watchdogLoop() noexcept
    {
        while (true)
        {
            std::unique_lock<std::mutex> lock(mMutex);
            if (mWakeup.wait_for(lock, mPollInterval, [this]() { return mStop; }))
            {
                return;
            }
            if (mComm == nullptr)
            {
                if (mError.has_value())
                {
                    lock.unlock();
                    cleanupAbortedResources();
                    return;
                }
                continue;
            }
            if (mError.has_value())
            {
                {
                    auto const hostApiLock = tensorrt_llm::runtime::acquireNcclHostApiLock();
                    setErrorAndAbortLocked(*mError);
                }
                lock.unlock();
                cleanupAbortedResources();
                continue;
            }

            std::string pendingError;
            auto const now = Clock::now();
            for (auto operation = mPendingOperations.begin(); operation != mPendingOperations.end();)
            {
                if (operation->start != nullptr)
                {
                    auto const startResult = cudaEventQuery(operation->start);
                    if (startResult == cudaSuccess)
                    {
                        recycleEventLocked(operation->start);
                        operation->start = nullptr;
                        operation->deadline = now + mOperationTimeout;
                        operation->armed = true;
                    }
                    else if (startResult != cudaErrorNotReady)
                    {
                        pendingError = operation->name + " start marker failed: " + cudaGetErrorString(startResult);
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
                        pendingError = operation->name + " timed out before its completion marker was recorded";
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
                    pendingError = operation->name + " failed with CUDA error: " + cudaGetErrorString(eventResult);
                    break;
                }
                if (operation->armed && now >= operation->deadline)
                {
                    pendingError = operation->name + " timed out before its CUDA completion marker";
                    break;
                }
                ++operation;
            }

            {
                auto const hostApiLock = tensorrt_llm::runtime::acquireNcclHostApiLock();
                if (!pendingError.empty())
                {
                    setErrorAndAbortLocked(pendingError);
                }
                else
                {
                    ncclResult_t asyncResult = ncclSuccess;
                    auto const queryResult = ncclCommGetAsyncError(mComm, &asyncResult);
                    if (queryResult != ncclSuccess && queryResult != ncclInProgress)
                    {
                        setErrorAndAbortLocked(std::string("ncclCommGetAsyncError failed with NCCL error: ")
                            + ncclGetErrorString(queryResult));
                    }
                    else if (asyncResult != ncclSuccess && asyncResult != ncclInProgress)
                    {
                        setErrorAndAbortLocked(std::string("NCCL watchdog observed asynchronous NCCL error: ")
                            + ncclGetErrorString(asyncResult));
                    }
                }
            }
            lock.unlock();
            cleanupAbortedResources();
        }
    }

    std::set<int> mGroup;
    std::shared_ptr<tensorrt_llm::runtime::NcclUniqueIdRendezvousComm> mControlComm;
    int const mWorldRank;
    bool mFaultToleranceEnabled;
    ncclComm_t mComm{nullptr};
    mutable std::mutex mMutex;
    std::condition_variable mWakeup;
    bool mStop{false};
    std::optional<std::string> mError;
    std::vector<ncclComm_t> mPendingAbortCleanup;

    struct PendingOperation
    {
        uint64_t token;
        cudaEvent_t start;
        cudaEvent_t completion;
        Clock::time_point deadline;
        bool armed;
        std::string name;
    };

    std::vector<PendingOperation> mPendingOperations;
    std::vector<cudaEvent_t> mEventPool;
    uint64_t mNextOperationToken{1};
    static constexpr size_t kMaxPendingOperations = 4096;
    static constexpr size_t kMaxPooledEvents = 256;
    std::chrono::milliseconds mPollInterval;
    std::chrono::milliseconds mOperationTimeout;
    std::chrono::milliseconds mInitTimeout;
    std::thread mWatchdog;
};

class NcclCommRegistry
{
public:
    static NcclCommRegistry& instance()
    {
        // Raw communicators, watchdogs, and their CUDA/MPI dependencies are
        // process-lifetime resources. Avoid cross-translation-unit static
        // destruction where the resource manager or allocator could disappear
        // before a live watchdog/state attempts final cleanup.
        static auto* registry = new NcclCommRegistry;
        return *registry;
    }

    std::shared_ptr<ncclComm_t> get(std::set<int> const& group)
    {
        std::lock_guard<std::mutex> lock(mMutex);
        auto it = mComms.find(group);
        if (it == mComms.end())
        {
            auto state = std::make_shared<NcclCommState>(group);
            mSlots[state->slot()] = state;
            it = mComms.emplace(group, std::move(state)).first;
        }
        return std::shared_ptr<ncclComm_t>{it->second, it->second->slot()};
    }

    std::shared_ptr<NcclCommState> find(std::shared_ptr<ncclComm_t> const& comm)
    {
        TLLM_CHECK_WITH_INFO(comm != nullptr, "NCCL communicator holder is null");
        std::lock_guard<std::mutex> lock(mMutex);
        auto const it = mSlots.find(comm.get());
        TLLM_CHECK_WITH_INFO(it != mSlots.end(), "NCCL communicator is not managed by the TRT-LLM registry");
        auto state = it->second.lock();
        TLLM_CHECK_WITH_INFO(state != nullptr, "NCCL communicator registry entry has expired");
        return state;
    }

    std::shared_ptr<NcclCommState> find(std::set<int> const& group)
    {
        std::lock_guard<std::mutex> lock(mMutex);
        auto const it = mComms.find(group);
        TLLM_CHECK_WITH_INFO(
            it != mComms.end(), "No cached NCCL communicator exists for group (%s)", groupToString(group).c_str());
        return it->second;
    }

    void abortAndReinit(std::set<int> const& oldGroup, std::set<int> const& activeGroup, std::uint64_t rendezvousId)
    {
        std::lock_guard<std::mutex> const recoveryLock(mRecoveryMutex);
        TLLM_CHECK_WITH_INFO(rendezvousId != 0,
            "NCCL error: recovery rendezvous ID must be nonzero; zero is reserved for initial bootstrap");
        TLLM_CHECK_WITH_INFO(!activeGroup.empty(), "active NCCL rank group must not be empty");
        TLLM_CHECK_WITH_INFO(std::includes(oldGroup.begin(), oldGroup.end(), activeGroup.begin(), activeGroup.end()),
            "active NCCL rank group (%s) must be a subset of old group (%s)", groupToString(activeGroup).c_str(),
            groupToString(oldGroup).c_str());
        TLLM_CHECK_WITH_INFO(tensorrt_llm::isNcclFaultToleranceEnabled(),
            "NCCL error: communicator reinitialization requires TLLM_FAULT_TOLERANCE_MODE=1; otherwise captured "
            "CUDA graphs may retain the old communicator");
        std::shared_ptr<NcclCommState> oldState;
        std::shared_ptr<NcclCommState> existingTargetState;
        {
            std::lock_guard<std::mutex> lock(mMutex);
            auto const old = mComms.find(oldGroup);
            if (old != mComms.end())
            {
                oldState = old->second;
            }
            auto const existingTarget = mComms.find(activeGroup);
            if (existingTarget != mComms.end())
            {
                existingTargetState = existingTarget->second;
            }
        }
        TLLM_CHECK_WITH_INFO(
            oldState != nullptr, "No cached NCCL communicator exists for group (%s)", groupToString(oldGroup).c_str());
        auto const& controlComm = oldState->controlComm();
        TLLM_CHECK_WITH_INFO(controlComm != nullptr, "NCCL rendezvous control channel is not initialized");
        auto const worldRank = controlComm->worldRank();
        TLLM_CHECK_WITH_INFO(
            activeGroup.count(worldRank) != 0, "Failed rank %d must not call NCCL abort_and_reinit", worldRank);
        oldState->abort("NCCL error: communicator was aborted for survivor reinitialization");
        if (existingTargetState != nullptr && existingTargetState != oldState)
        {
            // Explicit recovery is a distributed generation change. Never let
            // one rank reuse a locally healthy cached target communicator
            // while another rank enters the fresh-ID rendezvous.
            existingTargetState->abort("NCCL error: cached target communicator was retired for reinitialization");
        }

        // Do not mutate oldState: existing operations retain it and must fail
        // immediately instead of silently switching rank numbering underneath
        // an in-flight call. Subsequent operations request activeGroup.
        std::lock_guard<std::mutex> lock(mMutex);
        auto state = std::make_shared<NcclCommState>(activeGroup, controlComm, true, rendezvousId);
        mSlots[state->slot()] = state;
        mComms[activeGroup] = std::move(state);
    }

    std::optional<std::string> getError(std::set<int> const& group)
    {
        std::shared_ptr<NcclCommState> state;
        {
            std::lock_guard<std::mutex> lock(mMutex);
            auto const it = mComms.find(group);
            if (it == mComms.end())
            {
                return std::nullopt;
            }
            state = it->second;
        }
        return state->getError();
    }

private:
    std::mutex mRecoveryMutex;
    std::mutex mMutex;
    std::map<std::set<int>, std::shared_ptr<NcclCommState>> mComms;
    std::unordered_map<ncclComm_t*, std::weak_ptr<NcclCommState>> mSlots;
};

} // namespace

struct NcclCommLease::Impl
{
    explicit Impl(std::shared_ptr<NcclCommState> state_)
        : state(std::move(state_))
        , lock(state->mutex())
    {
        // Complete the call_once-protected capability probe before taking the
        // host gate. The probe itself briefly takes that gate for
        // ncclGetVersion; reversing the order can deadlock with a concurrent
        // first-time capability query.
        static_cast<void>(common::nccl_util::isNcclWindowSupported());
        hostApiLock = tensorrt_llm::runtime::acquireNcclHostApiLock();
        state->requireCommLocked();
    }

    ~Impl()
    {
        if (groupActive)
        {
            auto const result = ncclGroupEnd();
            groupActive = false;
            try
            {
                state->checkResultLocked(result, "ncclGroupEnd(exception cleanup)");
            }
            catch (...)
            {
                // Destructors cannot propagate. checkResultLocked has already
                // latched the classifier-friendly error and aborted the comm.
            }
        }
        if (hostApiLock.owns_lock())
        {
            hostApiLock.unlock();
        }
        if (lock.owns_lock())
        {
            lock.unlock();
        }
        state->cleanupAbortedResources();
    }

    std::shared_ptr<NcclCommState> state;
    std::unique_lock<std::mutex> lock;
    tensorrt_llm::runtime::NcclHostApiLock hostApiLock;
    bool groupActive{false};
};

NcclCommLease::NcclCommLease(std::unique_ptr<Impl> impl)
    : mImpl(std::move(impl))
{
}

NcclCommLease::NcclCommLease(ncclComm_t legacyComm)
    : mLegacyComm(legacyComm)
{
}

NcclCommLease::~NcclCommLease()
{
    if (mLegacyGroupActive)
    {
        auto const result = ncclGroupEnd();
        mLegacyGroupActive = false;
        if (result != ncclSuccess)
        {
            TLLM_LOG_WARNING(
                "Failed to close legacy NCCL group during exception cleanup: %s", ncclGetErrorString(result));
        }
    }
}

NcclCommLease::NcclCommLease(NcclCommLease&& other) noexcept
    : mImpl(std::move(other.mImpl))
    , mLegacyComm(std::exchange(other.mLegacyComm, nullptr))
    , mLegacyGroupActive(std::exchange(other.mLegacyGroupActive, false))
{
}

NcclCommLease& NcclCommLease::operator=(NcclCommLease&& other) noexcept
{
    if (this != &other)
    {
        if (mLegacyGroupActive)
        {
            static_cast<void>(ncclGroupEnd());
        }
        mImpl = std::move(other.mImpl);
        mLegacyComm = std::exchange(other.mLegacyComm, nullptr);
        mLegacyGroupActive = std::exchange(other.mLegacyGroupActive, false);
    }
    return *this;
}

ncclComm_t NcclCommLease::get() const
{
    if (mImpl == nullptr)
    {
        TLLM_CHECK_WITH_INFO(mLegacyComm != nullptr, "NCCL communicator lease was moved from");
        return mLegacyComm;
    }
    return mImpl->state->requireCommLocked();
}

void NcclCommLease::check(ncclResult_t result, char const* operation) const
{
    if (mImpl == nullptr)
    {
        (void) operation;
        TLLM_CHECK_WITH_INFO(mLegacyComm != nullptr, "NCCL communicator lease was moved from");
        NCCLCHECK_THROW(result);
        return;
    }
    mImpl->state->checkResultLocked(result, operation);
}

ncclResult_t NcclCommLease::checkOptional(ncclResult_t result, char const* operation) const
{
    if (mImpl == nullptr)
    {
        (void) operation;
        TLLM_CHECK_WITH_INFO(mLegacyComm != nullptr, "NCCL communicator lease was moved from");
        if (result != ncclInvalidArgument)
        {
            NCCLCHECK_THROW(result);
        }
        return result;
    }
    return mImpl->state->checkOptionalResultLocked(result, operation);
}

void NcclCommLease::checkEnqueue(ncclResult_t result, char const* operation) const
{
    if (mImpl == nullptr)
    {
        check(result, operation);
        return;
    }
    if (result != ncclSuccess && result != ncclInProgress && mImpl->groupActive)
    {
        // Close the thread-local group before aborting. Otherwise the first
        // collective on a replacement communicator can be nested inside the
        // failed group.
        static_cast<void>(ncclGroupEnd());
        mImpl->groupActive = false;
    }
    mImpl->state->checkEnqueueResultLocked(result, operation);
}

uint64_t NcclCommLease::begin(cudaStream_t stream, char const* operation) const
{
    if (mImpl == nullptr)
    {
        (void) stream;
        (void) operation;
        TLLM_CHECK_WITH_INFO(mLegacyComm != nullptr, "NCCL communicator lease was moved from");
        return 0;
    }
    return mImpl->state->beginOperationLocked(stream, operation);
}

void NcclCommLease::track(uint64_t token, cudaStream_t stream) const
{
    if (mImpl == nullptr)
    {
        (void) token;
        (void) stream;
        TLLM_CHECK_WITH_INFO(mLegacyComm != nullptr, "NCCL communicator lease was moved from");
        return;
    }
    mImpl->state->finishOperationLocked(token, stream);
}

void NcclCommLease::groupStart(char const* operation) const
{
    if (mImpl == nullptr)
    {
        TLLM_CHECK_WITH_INFO(mLegacyComm != nullptr, "NCCL communicator lease was moved from");
        TLLM_CHECK_WITH_INFO(!mLegacyGroupActive, "NCCL error: nested NCCL groups are not supported by this lease");
        (void) operation;
        NCCLCHECK_THROW(ncclGroupStart());
        mLegacyGroupActive = true;
        return;
    }
    TLLM_CHECK_WITH_INFO(!mImpl->groupActive, "NCCL error: nested NCCL groups are not supported by this lease");
    mImpl->state->checkResultLocked(ncclGroupStart(), operation);
    mImpl->groupActive = true;
}

void NcclCommLease::groupEnd(char const* operation) const
{
    if (mImpl == nullptr)
    {
        TLLM_CHECK_WITH_INFO(mLegacyComm != nullptr, "NCCL communicator lease was moved from");
        TLLM_CHECK_WITH_INFO(mLegacyGroupActive, "NCCL error: no NCCL group is active on this lease");
        auto const result = ncclGroupEnd();
        mLegacyGroupActive = false;
        (void) operation;
        NCCLCHECK_THROW(result);
        return;
    }
    TLLM_CHECK_WITH_INFO(mImpl->groupActive, "NCCL error: no NCCL group is active on this lease");
    auto const result = ncclGroupEnd();
    mImpl->groupActive = false;
    mImpl->state->checkResultLocked(result, operation);
}

std::shared_ptr<ncclComm_t> getComm(std::set<int> const& group)
{
    auto const groupString = groupToString(group);
    // Do not query COMM_SESSION here: after survivor reinitialization that
    // communicator may include a dead process and retain its fatal handler.
    TLLM_LOG_TRACE("Get NCCL comm for group(%s)", groupString.c_str());
    return NcclCommRegistry::instance().get(group);
}

int getCommWorldRank(std::shared_ptr<ncclComm_t> const& comm)
{
    return NcclCommRegistry::instance().find(comm)->worldRank();
}

bool isNcclFaultToleranceEnabled()
{
    // Communicator mode cannot change after the first managed communicator is
    // used: its blocking configuration and watchdog ownership are fixed.
    static bool const enabled = tensorrt_llm::mpi::isFaultToleranceModeEnabled();
    return enabled;
}

NcclCommLease acquireComm(std::shared_ptr<ncclComm_t> const& comm)
{
    // FT mode is a startup-only process setting. Keep the default-off path
    // equivalent to the legacy blocking communicator: no registry lookup,
    // heap allocation, communicator mutex, or process-wide NCCL gate per op.
    if (!isNcclFaultToleranceEnabled())
    {
        TLLM_CHECK_WITH_INFO(comm != nullptr && *comm != nullptr, "NCCL communicator is null");
        return NcclCommLease{*comm};
    }
    auto state = NcclCommRegistry::instance().find(comm);
    return NcclCommLease{std::make_unique<NcclCommLease::Impl>(std::move(state))};
}

void waitCommOperation(std::shared_ptr<ncclComm_t> const& comm, std::uint64_t token, char const* operation)
{
    if (!isNcclFaultToleranceEnabled())
    {
        TLLM_CHECK_WITH_INFO(token == 0, "NCCL error: legacy communicator received a watchdog operation token");
        return;
    }
    auto state = NcclCommRegistry::instance().find(comm);
    state->waitOperation(token, operation);
}

void abortAndReinitComm(std::set<int> const& oldGroup, std::set<int> const& activeGroup, std::uint64_t rendezvousId)
{
    NcclCommRegistry::instance().abortAndReinit(oldGroup, activeGroup, rendezvousId);
}

std::optional<std::string> getCommAsyncError(std::set<int> const& group)
{
    return NcclCommRegistry::instance().getError(group);
}

std::optional<std::string> getCommAsyncError(std::shared_ptr<ncclComm_t> const& comm)
{
    auto state = NcclCommRegistry::instance().find(comm);
    return state->getError();
}
#endif // ENABLE_MULTI_DEVICE

void const* tensorrt_llm::common::op::getCommSessionHandle()
{
#if ENABLE_MULTI_DEVICE
    return &COMM_SESSION;
#else
    return nullptr;
#endif // ENABLE_MULTI_DEVICE
}

namespace
{
using tensorrt_llm::common::op::hash;

// Get current cuda context, a default context will be created if there is no context.
inline CUcontext getCurrentCudaCtx()
{
    CUcontext ctx{};
    CUresult err = cuCtxGetCurrent(&ctx);
    if (err == CUDA_ERROR_NOT_INITIALIZED || ctx == nullptr)
    {
        TLLM_CUDA_CHECK(cudaFree(nullptr));
        err = cuCtxGetCurrent(&ctx);
    }
    TLLM_CHECK(err == CUDA_SUCCESS);
    return ctx;
}

// Helper to create per-cuda-context and per-thread singleton managed by std::shared_ptr.
// Unlike conventional singletons, singleton created with this will be released
// when not needed, instead of on process exit.
// Objects of this class shall always be declared static / global, and shall never own CUDA
// resources.
template <typename T>
class PerCudaCtxPerThreadSingletonCreator
{
public:
    using CreatorFunc = std::function<std::unique_ptr<T>()>;
    using DeleterFunc = std::function<void(T*)>;

    // creator returning std::unique_ptr is by design.
    // It forces separation of memory for T and memory for control blocks.
    // So when T is released, but we still have observer weak_ptr in mObservers, the T mem block can be released.
    // creator itself must not own CUDA resources. Only the object it creates can.
    PerCudaCtxPerThreadSingletonCreator(CreatorFunc creator, DeleterFunc deleter)
        : mCreator{std::move(creator)}
        , mDeleter{std::move(deleter)}
        , mObservers{new std::unordered_map<CacheKey, std::weak_ptr<T>, hash<CacheKey>>()}
    {
    }

    ~PerCudaCtxPerThreadSingletonCreator()
    {
        std::lock_guard<std::mutex> lk{mMutex};
        delete mObservers;
        mObservers = nullptr;
    }

    std::shared_ptr<T> operator()()
    {
        std::lock_guard<std::mutex> lk{mMutex};
        CUcontext ctx{getCurrentCudaCtx()};
        std::thread::id thread = std::this_thread::get_id();
        auto const key = std::make_tuple(ctx, thread);
        std::shared_ptr<T> result = (*mObservers)[key].lock();
        if (result == nullptr)
        {
            TLLM_LOG_TRACE("creating singleton instance for CUDA context %lu and thread %lu", ctx, thread);
            // Create the resource and register with an observer.
            result = std::shared_ptr<T>{mCreator().release(),
                [this, key](T* obj)
                {
                    if (obj == nullptr)
                    {
                        return;
                    }
                    mDeleter(obj);

                    if (mObservers == nullptr)
                    {
                        return;
                    }

                    // Clears observer to avoid growth of mObservers, in case users creates/destroys cuda contexts
                    // frequently.
                    std::shared_ptr<T> observedObjHolder; // Delay destroy to avoid dead lock.
                    std::lock_guard<std::mutex> lk{mMutex};
                    // Must check observer again because another thread may created new instance for this ctx and this
                    // thread just before we lock mMutex. We can't infer that the observer is stale from the fact that
                    // obj is destroyed, because shared_ptr ref-count checking and observer removing are not in one
                    // atomic operation, and the observer may be changed to observe another instance.
                    auto it = mObservers->find(key);
                    if (it == mObservers->end())
                    {
                        return;
                    }
                    observedObjHolder = it->second.lock();
                    if (observedObjHolder == nullptr)
                    {
                        mObservers->erase(it);
                    }
                }};
            (*mObservers)[key] = result;
        }
        else
        {
            TLLM_LOG_TRACE("singleton instance for CUDA context %d and thread %d is cached", ctx, thread);
        }
        return result;
    }

private:
    CreatorFunc mCreator;
    DeleterFunc mDeleter;
    mutable std::mutex mMutex;
    // CUDA resources are per-context and per-thread.
    using CacheKey = std::tuple<CUcontext, std::thread::id>;
    std::unordered_map<CacheKey, std::weak_ptr<T>, hash<CacheKey>>* mObservers;
};

// Structure to hold memory information
struct MemoryInfo
{
    size_t free_mb;
    size_t total_mb;
    float free_percent;
};

// Helper function to get current memory information
MemoryInfo getMemoryInfo()
{
    size_t free_mem = 0, total_mem = 0;
    TLLM_CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

    size_t const free_mb = free_mem / (1024 * 1024);
    size_t const total_mb = total_mem / (1024 * 1024);
    float const free_percent = (total_mem > 0) ? (static_cast<float>(free_mem) / total_mem * 100.0f) : 0.0f;

    return {free_mb, total_mb, free_percent};
}

// Helper function to log current memory usage
void logMemoryUsage(char const* operation, CUcontext ctx)
{
    auto const mem = getMemoryInfo();
    TLLM_LOG_DEBUG("%s: Context=%p, Free Memory=%zu MB (%.1f%%), Total=%zu MB", operation, ctx, mem.free_mb,
        mem.free_percent, mem.total_mb);
}

// Helper function to throw
void throwCublasErrorWithMemInfo(char const* operation, CUcontext ctx, cublasStatus_t status)
{
    auto const mem = getMemoryInfo();
    TLLM_THROW(
        "Failed to create %s. "
        "Status: %d, Context: %p, Free Memory: %zu MB (%.1f%%), Total: %zu MB. "
        "Consider reducing kv_cache_config.free_gpu_memory_fraction.",
        operation, status, ctx, mem.free_mb, mem.free_percent, mem.total_mb);
}

} // namespace

std::shared_ptr<cublasHandle_t> getCublasHandle()
{
    static PerCudaCtxPerThreadSingletonCreator<cublasHandle_t> creator(
        []() -> auto
        {
            CUcontext ctx = getCurrentCudaCtx();
            logMemoryUsage("Creating cublas handle", ctx);

            auto handle = std::make_unique<cublasHandle_t>();
            auto status = cublasCreate(handle.get());

            if (status != CUBLAS_STATUS_SUCCESS)
            {
                throwCublasErrorWithMemInfo("cublas handle", ctx, status);
            }

            return handle;
        },
        [](cublasHandle_t* handle)
        {
            auto status = cublasDestroy(*handle);
            if (status != CUBLAS_STATUS_SUCCESS)
            {
                TLLM_LOG_WARNING("Failed to destroy cublas handle. Status: %d", status);
            }
            delete handle;
            handle = nullptr;
        });
    return creator();
}

std::shared_ptr<cublasLtHandle_t> getCublasLtHandle()
{
    static PerCudaCtxPerThreadSingletonCreator<cublasLtHandle_t> creator(
        []() -> auto
        {
            CUcontext ctx = getCurrentCudaCtx();
            logMemoryUsage("Creating cublasLt handle", ctx);

            auto handle = std::make_unique<cublasLtHandle_t>();
            auto status = cublasLtCreate(handle.get());

            if (status != CUBLAS_STATUS_SUCCESS)
            {
                throwCublasErrorWithMemInfo("cublasLt handle", ctx, status);
            }

            return handle;
        },
        [](cublasLtHandle_t* handle)
        {
            auto status = cublasLtDestroy(*handle);
            if (status != CUBLAS_STATUS_SUCCESS)
            {
                TLLM_LOG_WARNING("Failed to destroy cublasLt handle. Status: %d", status);
            }
            delete handle;
            handle = nullptr;
        });
    return creator();
}

TRTLLM_NAMESPACE_END
