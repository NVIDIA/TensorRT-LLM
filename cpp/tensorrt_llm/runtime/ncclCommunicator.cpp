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

#if ENABLE_MULTI_DEVICE
#include <nccl.h>
#endif // ENABLE_MULTI_DEVICE

#include <array>
#include <chrono>
#include <cstdlib>
#include <string>
#include <thread>

using namespace tensorrt_llm::runtime;

namespace
{
#if ENABLE_MULTI_DEVICE
constexpr int kDefaultNcclCommInitTimeoutMs = 60'000;
constexpr int kNcclCommInitPollIntervalMs = 20;
constexpr char const* kNcclCommInitTimeoutEnv = "TLLM_NCCL_COMM_INIT_TIMEOUT_MS";
constexpr char const* kNcclNvlsEnableEnv = "NCCL_NVLS_ENABLE";

struct NcclInitResult
{
    ncclComm_t comm{nullptr};
    ncclResult_t result{ncclSuccess};
    bool timedOut{false};

    [[nodiscard]] bool isSuccess() const
    {
        return result == ncclSuccess;
    }
};

struct NcclInitStatus
{
    bool failed{false};
    bool timedOut{false};
};

int getNcclCommInitTimeoutMs()
{
    auto const* env = std::getenv(kNcclCommInitTimeoutEnv);
    int const timeoutMs = env == nullptr ? 0 : std::atoi(env);
    return timeoutMs > 0 ? timeoutMs : kDefaultNcclCommInitTimeoutMs;
}

bool canSuggestNvlsDisable()
{
    auto const* nvlsEnable = std::getenv(kNcclNvlsEnableEnv);
    return nvlsEnable == nullptr || std::string{nvlsEnable} == "2";
}

void setRuntimeConnectIfUnset()
{
    // Need static connection initialization for accurate KV cache size estimation.
#if defined(_WIN32)
    if (getenv("NCCL_RUNTIME_CONNECT") == nullptr)
    {
        _putenv_s("NCCL_RUNTIME_CONNECT", "0");
    }
#else
    setenv("NCCL_RUNTIME_CONNECT", "0", 0);
#endif // _WIN32
}

void abortNcclComm(ncclComm_t comm)
{
    if (comm == nullptr)
    {
        return;
    }

    auto const result = ncclCommAbort(comm);
    if (result != ncclSuccess)
    {
        TLLM_LOG_WARNING("Failed to abort NCCL communicator: %s.", ncclGetErrorString(result));
    }
}

NcclInitResult initNcclCommWithTimeout(ncclUniqueId const& id, int worldSize, int rank, int timeoutMs)
{
    NcclInitResult initResult;
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.blocking = 0;

    auto result = ncclCommInitRankConfig(&initResult.comm, worldSize, id, rank, &config);
    if (result != ncclSuccess && result != ncclInProgress)
    {
        initResult.result = result;
        return initResult;
    }
    if (result == ncclSuccess)
    {
        initResult.result = ncclSuccess;
        return initResult;
    }
    if (initResult.comm == nullptr)
    {
        initResult.result = result;
        return initResult;
    }

    auto const deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds{timeoutMs};
    while (true)
    {
        ncclResult_t asyncResult = ncclSuccess;
        result = ncclCommGetAsyncError(initResult.comm, &asyncResult);
        if (result != ncclSuccess)
        {
            initResult.result = result;
            return initResult;
        }
        if (asyncResult != ncclInProgress)
        {
            initResult.result = asyncResult;
            return initResult;
        }
        if (std::chrono::steady_clock::now() >= deadline)
        {
            initResult.result = ncclInProgress;
            initResult.timedOut = true;
            return initResult;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds{kNcclCommInitPollIntervalMs});
    }
}

NcclInitStatus getNcclInitStatus(NcclInitResult const& result, tensorrt_llm::mpi::MpiComm const& mpiComm)
{
    std::array<int, 2> localStatus{result.isSuccess() ? 0 : 1, result.timedOut ? 1 : 0};
    std::array<int, 2> globalStatus{};
    mpiComm.allreduce(
        localStatus.data(), globalStatus.data(), 2, tensorrt_llm::mpi::MpiType::kINT32, tensorrt_llm::mpi::MpiOp::MAX);
    return {globalStatus[0] != 0, globalStatus[1] != 0};
}

bool allRanksCanUseNvlsDisableWorkaround(tensorrt_llm::mpi::MpiComm const& mpiComm)
{
    int const localCanDisable = canSuggestNvlsDisable() ? 1 : 0;
    int globalCanDisable = 0;
    mpiComm.allreduce(
        &localCanDisable, &globalCanDisable, 1, tensorrt_llm::mpi::MpiType::kINT32, tensorrt_llm::mpi::MpiOp::MIN);

    return globalCanDisable != 0;
}

void checkNcclResult(ncclComm_t comm, ncclResult_t result, char const* operation)
{
    if (result == ncclSuccess)
    {
        return;
    }
    if (result != ncclInProgress)
    {
        TLLM_NCCL_CHECK(result);
    }

    while (true)
    {
        ncclResult_t asyncResult = ncclSuccess;
        result = ncclCommGetAsyncError(comm, &asyncResult);
        if (result != ncclSuccess)
        {
            TLLM_THROW("NCCL %s failed while polling communicator status: %s.", operation, ncclGetErrorString(result));
        }
        if (asyncResult == ncclSuccess)
        {
            return;
        }
        if (asyncResult != ncclInProgress)
        {
            TLLM_THROW("NCCL %s failed asynchronously: %s.", operation, ncclGetErrorString(asyncResult));
        }
        std::this_thread::sleep_for(std::chrono::milliseconds{kNcclCommInitPollIntervalMs});
    }
}

ncclUniqueId createAndBroadcastNcclId(int rank, tensorrt_llm::mpi::MpiComm const& mpiComm)
{
    ncclUniqueId id;
    if (rank == 0)
    {
        TLLM_NCCL_CHECK(ncclGetUniqueId(&id));
    }
    mpiComm.bcastValue(id, 0);
    return id;
}

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
    default: TLLM_THROW("Unsupported data type: %d", static_cast<int>(dataType));
    }
}
#endif // ENABLE_MULTI_DEVICE
} // namespace

void NcclCommunicator::send(
    void const* sendbuff, size_t count, nvinfer1::DataType dataType, int peer, CudaStream const& stream) const
{
#if ENABLE_MULTI_DEVICE
    checkNcclResult(mComm, ncclSend(sendbuff, count, toNcclType(dataType), peer, mComm, stream.get()), "send");
#else
    TLLM_THROW("Multi device support is disabled.");
#endif // ENABLE_MULTI_DEVICE
}

void NcclCommunicator::receive(
    void* sendbuff, size_t count, nvinfer1::DataType dataType, int peer, CudaStream const& stream) const
{
#if ENABLE_MULTI_DEVICE
    checkNcclResult(mComm, ncclRecv(sendbuff, count, toNcclType(dataType), peer, mComm, stream.get()), "receive");
#else
    TLLM_THROW("Multi device support is disabled.");
#endif // ENABLE_MULTI_DEVICE
}

ncclComm_t NcclCommunicator::createComm(int worldSize, int rank, mpi::MpiComm const& mpiComm)
{
#if ENABLE_MULTI_DEVICE

    setRuntimeConnectIfUnset();
    auto const timeoutMs = getNcclCommInitTimeoutMs();

    auto id = createAndBroadcastNcclId(rank, mpiComm);
    auto initResult = initNcclCommWithTimeout(id, worldSize, rank, timeoutMs);
    auto const initStatus = getNcclInitStatus(initResult, mpiComm);

    if (initStatus.failed)
    {
        if (initStatus.timedOut)
        {
            if (allRanksCanUseNvlsDisableWorkaround(mpiComm))
            {
                TLLM_THROW(
                    "NCCL communicator initialization timed out after %d ms on at least one rank. This may indicate "
                    "an NVLS multicast resource setup failure in Fabric Manager. Set NCCL_NVLS_ENABLE=0 before "
                    "process startup and retry. TensorRT-LLM does not retry in-process because NCCL may not recover "
                    "after a timed-out NVLS initialization.",
                    timeoutMs);
            }
            TLLM_THROW(
                "NCCL communicator initialization timed out after %d ms on at least one rank. NCCL_NVLS_ENABLE is "
                "explicitly set, so TensorRT-LLM will not override it.",
                timeoutMs);
        }
        if (!initResult.isSuccess())
        {
            abortNcclComm(initResult.comm);
        }
        mpiComm.barrier();
        if (!initResult.isSuccess())
        {
            TLLM_THROW(
                "NCCL communicator initialization failed on rank %d: %s.", rank, ncclGetErrorString(initResult.result));
        }
        TLLM_THROW("NCCL communicator initialization failed on at least one peer rank.");
    }

    return initResult.comm;
#else
    // Python runtime requires instantiation of a communicator even though it may never be used to enable
    // pipeline parallel code-path. To enable this, have an empty communicator with uninitialized state.
    return nullptr;
#endif // ENABLE_MULTI_DEVICE
}

NcclCommunicator::~NcclCommunicator()
{
#if ENABLE_MULTI_DEVICE
    if (mComm && ncclCommDestroy(mComm) != ncclSuccess)
    {
        TLLM_LOG_WARNING("Failed to destroy NCCL communicator.");
    }
#endif // ENABLE_MULTI_DEVICE
}
