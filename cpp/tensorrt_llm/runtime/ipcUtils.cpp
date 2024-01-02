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
#include "tensorrt_llm/common/mpiUtils.h"

namespace tensorrt_llm::runtime
{

void setPeerAccess(WorldConfig worldConfig, bool enable)
{
    const auto srcNode = worldConfig.getTensorParallelRank();

    for (SizeType destNode = 0; destNode < worldConfig.getTensorParallelism(); destNode++)
    {
        if (destNode == srcNode)
        {
            continue;
        }

        int canAccessPeer;
        TLLM_CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccessPeer, srcNode, destNode));

        if (enable)
        {
            cudaDeviceEnablePeerAccess(destNode, 0);
        }
        else
        {
            cudaDeviceDisablePeerAccess(destNode);
        }
        const auto error = cudaGetLastError();
        if (error != cudaErrorPeerAccessAlreadyEnabled && error != cudaErrorPeerAccessNotEnabled)
        {
            TLLM_CUDA_CHECK(error);
        }
    }
}

IpcMemory::IpcMemory(WorldConfig worldConfig, std::size_t bufferSize)
    : mWorldConfig(worldConfig)
    , mCommPtrs(worldConfig.getTensorParallelism())
    , mBufferSize(bufferSize)
{
    allocateIpcMemory();
}

void IpcMemory::allocateIpcMemory()
{
    TLLM_CUDA_CHECK(cudaMalloc(&mBufferPtr, mBufferSize));
    TLLM_CUDA_CHECK(cudaMemset(mBufferPtr, 0, mBufferSize));

    cudaIpcMemHandle_t localHandle;
    TLLM_CUDA_CHECK(cudaIpcGetMemHandle(&localHandle, mBufferPtr));

    const auto tpRank = mWorldConfig.getTensorParallelRank();
    const auto ppRank = mWorldConfig.getPipelineParallelRank();
    auto const comm = COMM_SESSION.split(ppRank, tpRank);
    std::vector<char> serialHandles(CUDA_IPC_HANDLE_SIZE * mWorldConfig.getTensorParallelism(), 0);
    comm.allgather(&localHandle.reserved, serialHandles.data(), CUDA_IPC_HANDLE_SIZE, mpi::MpiType::kBYTE);

    std::vector<cudaIpcMemHandle_t> handles(mWorldConfig.getTensorParallelism());
    for (size_t i = 0; i < handles.size(); ++i)
    {
        memcpy(handles[i].reserved, &serialHandles[i * CUDA_IPC_HANDLE_SIZE], CUDA_IPC_HANDLE_SIZE);
    }

    for (size_t nodeId = 0; nodeId < handles.size(); nodeId++)
    {
        if ((int) nodeId == mWorldConfig.getTensorParallelRank())
        {
            mCommPtrs[nodeId] = mBufferPtr;
        }
        else
        {
            uint8_t* foreignBuffer;
            TLLM_CUDA_CHECK(cudaIpcOpenMemHandle(
                reinterpret_cast<void**>(&foreignBuffer), handles[nodeId], cudaIpcMemLazyEnablePeerAccess));
            mCommPtrs[nodeId] = foreignBuffer;
        }
    }
}

IpcMemory::~IpcMemory()
{
    destroyIpcMemory();
}

void IpcMemory::destroyIpcMemory()
{
    for (SizeType nodeId = 0; nodeId < mWorldConfig.getTensorParallelism(); ++nodeId)
    {
        if ((int) nodeId == mWorldConfig.getTensorParallelRank())
        {
            TLLM_CUDA_CHECK(cudaFree(mCommPtrs[nodeId]));
        }
        else
        {
            TLLM_CUDA_CHECK(cudaIpcCloseMemHandle(mCommPtrs[nodeId]));
        }
    }
    cudaFree(mBufferPtr);
}

} // namespace tensorrt_llm::runtime
