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

#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/runtime/ipcNvlsMemory.h"
#include "tensorrt_llm/runtime/utils/multiDeviceUtils.h"

#if ENABLE_MULTI_DEVICE
#include <nccl.h>
#endif // ENABLE_MULTI_DEVICE

#include <chrono>
#include <cstdlib>
#include <thread>

using namespace tensorrt_llm::runtime;

namespace
{
#if ENABLE_MULTI_DEVICE
constexpr int kDefaultNcclCommInitTimeoutSec = 60;
constexpr int kNcclCommInitPollIntervalMs = 20;
constexpr char const* kNcclCommInitTimeoutEnv = "TRTLLM_NCCL_COMM_INIT_TIMEOUT_SEC";

int getNcclCommInitTimeoutSec()
{
    auto const timeoutSec
        = tensorrt_llm::common::getIntEnv(kNcclCommInitTimeoutEnv).value_or(kDefaultNcclCommInitTimeoutSec);
    return timeoutSec > 0 ? timeoutSec : kDefaultNcclCommInitTimeoutSec;
}

void waitForNcclAsyncCompletion(ncclComm_t comm)
{
    while (true)
    {
        ncclResult_t asyncResult = ncclSuccess;
        auto const result = ncclCommGetAsyncError(comm, &asyncResult);
        TLLM_NCCL_CHECK(result);
        if (asyncResult == ncclSuccess)
        {
            return;
        }
        if (asyncResult != ncclInProgress)
        {
            TLLM_NCCL_CHECK(asyncResult);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds{kNcclCommInitPollIntervalMs});
    }
}

void checkNcclAsyncResult(ncclComm_t comm, ncclResult_t result)
{
    if (result == ncclInProgress)
    {
        waitForNcclAsyncCompletion(comm);
        return;
    }
    TLLM_NCCL_CHECK(result);
}

ncclComm_t initNcclCommWithTimeout(ncclUniqueId const& id, int worldSize, int rank, int timeoutSec)
{
    ncclComm_t comm{nullptr};
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.blocking = 0;

    auto result = ncclCommInitRankConfig(&comm, worldSize, id, rank, &config);
    if (result == ncclSuccess)
    {
        return comm;
    }
    if (result != ncclInProgress)
    {
        TLLM_THROW("NCCL communicator initialization failed on rank %d of %d: %s.", rank, worldSize,
            ncclGetErrorString(result));
    }
    if (comm == nullptr)
    {
        TLLM_THROW(
            "NCCL communicator initialization is in progress on rank %d of %d but returned a null "
            "communicator.",
            rank, worldSize);
    }

    auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds{timeoutSec};
    while (true)
    {
        ncclResult_t asyncResult = ncclSuccess;
        result = ncclCommGetAsyncError(comm, &asyncResult);
        if (result != ncclSuccess)
        {
            TLLM_THROW("NCCL communicator initialization failed while polling rank %d of %d: %s.", rank, worldSize,
                ncclGetErrorString(result));
        }
        if (asyncResult == ncclSuccess)
        {
            return comm;
        }
        if (asyncResult != ncclInProgress)
        {
            TLLM_THROW("NCCL communicator initialization failed asynchronously on rank %d of %d: %s.", rank, worldSize,
                ncclGetErrorString(asyncResult));
        }
        if (std::chrono::steady_clock::now() >= deadline)
        {
            TLLM_THROW(
                "NCCL communicator initialization timed out after %d seconds on rank %d of %d in "
                "NcclCommunicator::createComm. This indicates NCCL init is hung. TensorRT-LLM will not retry "
                "in-process. Set %s to adjust the timeout.",
                timeoutSec, rank, worldSize, kNcclCommInitTimeoutEnv);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds{kNcclCommInitPollIntervalMs});
    }
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
    checkNcclAsyncResult(mComm, ncclSend(sendbuff, count, toNcclType(dataType), peer, mComm, stream.get()));
#else
    TLLM_THROW("Multi device support is disabled.");
#endif // ENABLE_MULTI_DEVICE
}

void NcclCommunicator::receive(
    void* sendbuff, size_t count, nvinfer1::DataType dataType, int peer, CudaStream const& stream) const
{
#if ENABLE_MULTI_DEVICE
    checkNcclAsyncResult(mComm, ncclRecv(sendbuff, count, toNcclType(dataType), peer, mComm, stream.get()));
#else
    TLLM_THROW("Multi device support is disabled.");
#endif // ENABLE_MULTI_DEVICE
}

ncclComm_t NcclCommunicator::createComm(int worldSize, int rank, mpi::MpiComm const& mpiComm)
{
#if ENABLE_MULTI_DEVICE

// Need static connection initialization for accurate KV cache size estimation.
#if defined(_WIN32)
    if (getenv("NCCL_RUNTIME_CONNECT") == nullptr)
        _putenv_s("NCCL_RUNTIME_CONNECT", "0");
#else
    setenv("NCCL_RUNTIME_CONNECT", "0", 0);
    if (getenv("NCCL_NVLS_ENABLE") == nullptr && !ipcNvlsSupported())
    {
        setenv("NCCL_NVLS_ENABLE", "0", 0);
    }
#endif // _WIN32
    ncclUniqueId id;
    if (rank == 0)
    {
        ncclGetUniqueId(&id);
    }
    mpiComm.bcastValue(id, 0);
    return initNcclCommWithTimeout(id, worldSize, rank, getNcclCommInitTimeoutSec());
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
