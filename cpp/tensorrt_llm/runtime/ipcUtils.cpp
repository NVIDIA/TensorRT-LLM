/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/runtime/ipcUtils.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/customAllReduceUtils.h"
#include "tensorrt_llm/common/workspace.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"

#include <NvInferRuntimeBase.h>
#include <cstddef>

namespace tensorrt_llm::runtime
{

bool canAccessPeer(WorldConfig const& worldConfig)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto const srcDevice = worldConfig.getDevice();

    for (SizeType32 rank : worldConfig.getTensorParallelGroup())
    {
        SizeType32 destDevice = worldConfig.getDeviceOf(rank);
        if (worldConfig.getNodeRankOf(rank) != worldConfig.getNodeRank())
        {
            TLLM_LOG_INFO("Detect inter-node TP between rank %d and rank %d, fail to access peer GPU memory",
                worldConfig.getRank(), rank);
            TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
            return false;
        }
        if (destDevice == srcDevice)
        {
            continue;
        }

        int canAccessPeer{0};
        TLLM_CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccessPeer, srcDevice, destDevice));
        if (canAccessPeer == 0)
        {
            TLLM_LOG_INFO("cudaDeviceCanAccessPeer failed for device: %d peerDevice: %d", srcDevice, destDevice);
            TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
            return false;
        }
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return true;
}

IpcMemory::IpcMemory(std::size_t bufferSize, BufferManager const& manager, WorldConfig const& worldConfig, bool openIpc)
    : mTpRank(worldConfig.getTensorParallelRank())
    , mCommPtrs(worldConfig.getTensorParallelism())
{
    mOpenIpc = openIpc && worldConfig.getTensorParallelism() <= worldConfig.getGpusPerNode();
    if (mOpenIpc)
    {
        allocateIpcMemory(bufferSize, manager, worldConfig);
    }
}

void IpcMemory::allocateIpcMemory(std::size_t bufferSize, BufferManager const& manager, WorldConfig const& worldConfig)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    // Note: cudaIpcGet/OpenMemHandle does not work well with tiny allocations. The risk is that two small
    // allocations will get 'packed' together, and that we will get the same IPC handle for both using
    // cudaIpcGetMemHandle. We then try to open this handle twice on the receiving end, resulting in an 'already mapped'
    // error. This manual alignment is a WAR for this behavior, until we move to using cuMemMap and OS handles to share
    // these allocations, which is the recommended way. On top of that, we use gpuSync here (relying on cudaMalloc)
    // instead of gpu (relying on cudaMallocAsync), because the default memory pool for cudaMallocAsync does not expose
    // IPC handles. If we want to support stream-ordered allocations here, we need to create another pool with the
    // correct handle type.
    auto const ipcAlignedBufferSize = common::alignSize(bufferSize, 1LU << 21);
    mBuffer = BufferManager::gpuSync(ipcAlignedBufferSize, nvinfer1::DataType::kUINT8);
    manager.setZero(*mBuffer);
    auto* bufferPtr = mBuffer->data();

    cudaIpcMemHandle_t localHandle;
    TLLM_CUDA_CHECK(cudaIpcGetMemHandle(&localHandle, bufferPtr));

    auto const tpGroupId = worldConfig.getContextParallelRank()
        + worldConfig.getContextParallelism() * worldConfig.getPipelineParallelRank();
    auto const comm = COMM_SESSION.split(tpGroupId, mTpRank);
    std::vector<char> serialHandles(CUDA_IPC_HANDLE_SIZE * worldConfig.getTensorParallelism(), 0);
    comm.allgather(&localHandle.reserved, serialHandles.data(), CUDA_IPC_HANDLE_SIZE, mpi::MpiType::kBYTE);

    std::vector<cudaIpcMemHandle_t> handles(worldConfig.getTensorParallelism());
    for (size_t i = 0; i < handles.size(); ++i)
    {
        memcpy(handles[i].reserved, &serialHandles[i * CUDA_IPC_HANDLE_SIZE], CUDA_IPC_HANDLE_SIZE);
    }

    for (std::size_t nodeId = 0; nodeId < handles.size(); nodeId++)
    {
        if (nodeId == static_cast<std::size_t>(mTpRank))
        {
            mCommPtrs.at(nodeId) = bufferPtr;
        }
        else
        {
            uint8_t* foreignBuffer{nullptr};
            TLLM_CUDA_CHECK(cudaIpcOpenMemHandle(
                reinterpret_cast<void**>(&foreignBuffer), handles[nodeId], cudaIpcMemLazyEnablePeerAccess));
            mCommPtrs.at(nodeId) = foreignBuffer;
        }
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

IpcMemory::~IpcMemory()
{
    if (mOpenIpc)
    {
        destroyIpcMemory();
    }
}

void IpcMemory::destroyIpcMemory()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    for (std::size_t nodeId = 0; nodeId < mCommPtrs.size(); ++nodeId)
    {
        if (nodeId != static_cast<std::size_t>(mTpRank))
        {
            TLLM_CUDA_CHECK(cudaIpcCloseMemHandle(mCommPtrs.at(nodeId)));
        }
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

AllReduceBuffers::AllReduceBuffers(SizeType32 maxBatchSize, SizeType32 maxBeamWidth, SizeType32 maxSequenceLength,
    SizeType32 hiddenSize, BufferManager const& manager, WorldConfig const& worldConfig, bool const fakeBuffers)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    if (fakeBuffers)
    {
        auto const tpSize = worldConfig.getTensorParallelism();
        mAllReduceCommPtrs = BufferManager::cpu(
            ITensor::makeShape({static_cast<SizeType32>(7) * tpSize + 3}), nvinfer1::DataType::kINT64);
    }
    else
    {
        auto const isP2pSupported = canAccessPeer(worldConfig);

        auto const tpSize = worldConfig.getTensorParallelism();
        bool const forceDeterministic = common::getEnvForceDeterministicAllReduce();
        // Force pull mode and disable lamport when force deterministic is enabled, for reducing device memory usage.
        auto const bufferSize = (forceDeterministic ? 1 : tpSize)
            * std::min(
                static_cast<std::size_t>(maxBatchSize) * maxBeamWidth * maxSequenceLength * hiddenSize * sizeof(float),
                utils::customAllReduceUtils::getMaxRequiredWorkspaceSize(tpSize));
        size_t realHiddenSize = tpSize * hiddenSize;
        // PUSH_MODE need TP_SIZE times the activation tensor size
        auto const lamportBufferSize = forceDeterministic
            ? 1 // zero size is not allowed for IpcMemory::allocateIpcMemory.
            : (tpSize * tensorrt_llm::kernels::reduce_fusion::details::kLamportTokenNumThreshold * realHiddenSize
                * sizeof(half));
        auto const flagsSize = IpcMemory::FLAGS_SIZE * tpSize * 2 * tpSize;

        for (auto size :
            {bufferSize, bufferSize, flagsSize, flagsSize, lamportBufferSize, lamportBufferSize, lamportBufferSize})
        {
            mIpcMemoryHandles.emplace_back(size, manager, worldConfig, isP2pSupported);
        }

        mAllReduceCommPtrs
            = BufferManager::cpu(ITensor::makeShape({static_cast<SizeType32>(mIpcMemoryHandles.size()) * tpSize + 3}),
                nvinfer1::DataType::kINT64);
        auto commPtrs = BufferRange<void*>(*mAllReduceCommPtrs);
        // Start from 1 since 0 represents released state for barrier at the beginning of the all_reduce.
        // The last element is the barrier flag counter.
        mFlagPtrs = manager.copyFrom(std::vector<int64_t>{1, 1, 0}, ITensor::makeShape({3}), MemoryType::kGPU);
        commPtrs[mIpcMemoryHandles.size() * tpSize] = mFlagPtrs->data();
        commPtrs[mIpcMemoryHandles.size() * tpSize + 1] = mFlagPtrs->data(1);
        commPtrs[mIpcMemoryHandles.size() * tpSize + 2] = mFlagPtrs->data(2);

        for (std::size_t memIdx = 0; memIdx < mIpcMemoryHandles.size(); memIdx++)
        {
            auto const& memCommPtrs = mIpcMemoryHandles[memIdx].getCommPtrs();
            TLLM_CHECK(memCommPtrs.size() == static_cast<std::size_t>(tpSize));
            std::copy(memCommPtrs.begin(), memCommPtrs.end(), commPtrs.begin() + memIdx * tpSize);
        }
#if ENABLE_MULTI_DEVICE
        auto tp_rank = worldConfig.getTensorParallelRank();
        // When p2p is not supported all the mIpcMemoryHandles are
        // null
        if (isP2pSupported)
        {
            lamportInitializeAll(mIpcMemoryHandles[4].getCommPtrs()[tp_rank],
                mIpcMemoryHandles[5].getCommPtrs()[tp_rank], mIpcMemoryHandles[6].getCommPtrs()[tp_rank],
                lamportBufferSize);
        }
#endif
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void lamportInitializeAll(void* buffer_0, void* buffer_1, void* buffer_2, size_t size)
{
#if ENABLE_MULTI_DEVICE
    tensorrt_llm::kernels::lamportInitialize(buffer_0, size / sizeof(half), nvinfer1::DataType::kHALF, 0);
    tensorrt_llm::kernels::lamportInitialize(buffer_1, size / sizeof(half), nvinfer1::DataType::kHALF, 0);
    tensorrt_llm::kernels::lamportInitialize(buffer_2, size / sizeof(half), nvinfer1::DataType::kHALF, 0);
    cudaDeviceSynchronize();
#endif
}

} // namespace tensorrt_llm::runtime
