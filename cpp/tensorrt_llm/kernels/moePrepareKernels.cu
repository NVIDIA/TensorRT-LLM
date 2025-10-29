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

#include "moePrepareKernels.h"

#include <stdio.h>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>

namespace cg = cooperative_groups;

namespace tensorrt_llm::kernels
{

namespace moe_prepare
{

using tensorrt_llm::common::launchWithPdlWhenEnabled;

__device__ __forceinline__ void st_release_sys_global(uint64_t volatile* ptr, uint64_t val)
{
    asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(ptr), "l"(val) : "memory");
}

__device__ __forceinline__ uint64_t ld_acquire_sys_global(uint64_t volatile* ptr)
{
    uint64_t ret;
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(ret) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ int ld_acquire_sys_global_int(int volatile* ptr)
{
    int ret;
    asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

class CounterCommunicator
{
public:
    __device__ __inline__ CounterCommunicator(MoeCommFifoConnInfo* fifoConnInfo)
        : fifoConnInfo(fifoConnInfo)
    {
    }

    __forceinline__ __device__ void releaseValue(uint64_t value, int index)
    {
        // Avoid block on 0
        while (fifoConnInfo->values[index] != 0)
        {
            // loop wait until value reset
        }
        fifoConnInfo->values[index] = value + 1;
    }

    __forceinline__ __device__ uint64_t acquireValue(int index)
    {
        uint64_t localValue = 0;
        do
        {
            localValue = fifoConnInfo->values[index];
        } while (localValue == 0);

        fifoConnInfo->values[index] = 0; // reset the value

        return localValue - 1;
    }

protected:
    MoeCommFifoConnInfo* fifoConnInfo;
};

template <int kThreadsGroupSize>
__device__ __forceinline__ void computeCountAndSendStatics(int* experts, int tokenCount, int* sharedSendRecvRankCount,
    int* sendCounts, int* sendIndiceWorkspace, int* backwardIndiceWorkspace, int* expertStatics,
    MoeCommWorkspace workspace, int maxTokenCountPerRank, int slotCount, int expertCount, int topK, int epRank,
    int epSize)
{
    cg::thread_block_tile<kThreadsGroupSize> tile = cg::tiled_partition<kThreadsGroupSize>(cg::this_thread_block());
    int laneInTile = tile.thread_rank();
    int tileId = threadIdx.x / kThreadsGroupSize;
    int tileCountPerBlock = blockDim.x / kThreadsGroupSize;
    int expertCountPerRank = slotCount / epSize;

    if (threadIdx.x == 0)
    {
        *sharedSendRecvRankCount = 0;
    }
    __syncthreads();
    int targetRankId = blockIdx.x;
    int readRankTokenCount = tokenCount;
    if (targetRankId >= epSize)
    {
        return;
    }

    int* localSendIndice = sendIndiceWorkspace + targetRankId * maxTokenCountPerRank;
    int* localBackwardIndice = backwardIndiceWorkspace + targetRankId * maxTokenCountPerRank;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
    cudaTriggerProgrammaticLaunchCompletion();
#endif

    for (int i = tileId; i < readRankTokenCount; i += tileCountPerBlock)
    {
        int expertRankId = laneInTile < topK ? experts[i * topK + laneInTile] / expertCountPerRank : epSize;
        bool rankMatched = (expertRankId == targetRankId);
        bool hasRankMatched = tile.any(rankMatched);
        int mask = tile.ballot(rankMatched);
        int firstMatchLane = __ffs(mask) - 1; // only valid if hasRankMatched is true
        if (hasRankMatched && laneInTile == 0)
        {
            int index = atomicAdd_block(sharedSendRecvRankCount, 1);
            localSendIndice[index] = i;
            localBackwardIndice[index] = i * topK + firstMatchLane;
        }
        tile.sync();
    }
    __syncthreads();

    CounterCommunicator counter(workspace.getFifoConnInfo(true, epRank, targetRankId, 0, epSize, 1));

    int communicationCount = expertStatics == nullptr ? 1 : expertCount + 1;
    for (int i = threadIdx.x; i < communicationCount; i += blockDim.x)
    {
        int value = i == 0 ? *(sharedSendRecvRankCount) : *(expertStatics + i - 1);
        counter.releaseValue(value, i);
        if (i == 0)
        {
            *(sendCounts + targetRankId) = value;
        }
    }
}

__device__ __forceinline__ void recvCountAndStatics(int* recvIndiceWorkspace, int* recvCounts, int* sharedCountsBase,
    int* gatheredExpertStatics, MoeCommWorkspace workspace, int expertCount, int maxTokenCountPerRank, int rankId,
    int rankCount)
{
    int rankOffset = threadIdx.x / THREADS_PER_PIPELINE;
    if (rankOffset >= PIPELINE_PER_CTA)
    {
        return;
    }
    int* sharedCountsThisRank = sharedCountsBase + rankOffset;
    int targetRankId = (blockIdx.x - rankCount) * PIPELINE_PER_CTA + rankOffset;
    if (targetRankId >= rankCount)
    {
        return;
    }
    int unitId = threadIdx.x % UNIT_PER_PIPELINE;
    cg::thread_block_tile<THREADS_PER_PIPELINE> rankTile
        = cg::tiled_partition<THREADS_PER_PIPELINE>(cg::this_thread_block());
    int* localRecvIndice = recvIndiceWorkspace + targetRankId * maxTokenCountPerRank;

    CounterCommunicator counter(workspace.getFifoConnInfo(false, rankId, targetRankId, 0, rankCount, 1));
    int communicationCount = gatheredExpertStatics == nullptr ? 1 : expertCount + 1;

    for (int i = rankTile.thread_rank(); i < communicationCount; i += THREADS_PER_PIPELINE)
    {
        int recvValue = counter.acquireValue(i);
        if (i == 0)
        {
            *(recvCounts + targetRankId) = recvValue;
            *(sharedCountsThisRank) = recvValue;
        }
        else
        {
            *(gatheredExpertStatics + targetRankId * expertCount + i - 1) = recvValue;
        }
    }
    rankTile.sync();

    int rankRecvCount = *(sharedCountsThisRank);
    for (int tokenId = unitId; tokenId < rankRecvCount; tokenId += UNIT_PER_PIPELINE)
    {
        *(localRecvIndice + tokenId) = tokenId;
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <int kThreadsGroupSize>
__global__ void computeCountAndIndiceDevice(int* experts, int* sendCounts, int* recvCounts, int* sendIndiceWorkspace,
    int* backwardIndiceWorkspace, int* recvIndiceWorkspace, int* expertStatics, int* gatheredExpertStatics,
    MoeCommWorkspace workspace, int tokenCount, int maxTokenCountPerRank, int topK, int slotCount, int expertCount,
    int rankId, int rankCount)
{
    __shared__ int sharedCounts[PIPELINE_PER_CTA];
    bool isSender = blockIdx.x < rankCount;
    if (isSender)
    {
        computeCountAndSendStatics<kThreadsGroupSize>(experts, tokenCount, &sharedCounts[0], sendCounts,
            sendIndiceWorkspace, backwardIndiceWorkspace, expertStatics, workspace, maxTokenCountPerRank, slotCount,
            expertCount, topK, rankId, rankCount);
    }
    else
    {
        recvCountAndStatics(recvIndiceWorkspace, recvCounts, &sharedCounts[0], gatheredExpertStatics, workspace,
            expertCount, maxTokenCountPerRank, rankId, rankCount);
    }
}

__global__ void moveIndiceDevice(int* sendCountsCumsum, int* recvCountsCumsum, int* sendIndice, int* gatherSendIndice,
    int* backwardIndice, int* gatherBackwardIndice, int* recvIndice, int* gatherRecvIndice, int maxTokenCountPerRank)
{
    int targetRankId = blockIdx.x;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
    cudaTriggerProgrammaticLaunchCompletion();
#endif
    if (blockIdx.y == 0)
    {
        // sendIndice and backwardIndice CTA
        int startIndex = targetRankId == 0 ? 0 : sendCountsCumsum[targetRankId - 1];
        int endIndex = sendCountsCumsum[targetRankId];
        int count = endIndex - startIndex;
        int* localSendIndice = sendIndice + targetRankId * maxTokenCountPerRank;
        int* localBackwardIndice = backwardIndice + targetRankId * maxTokenCountPerRank;
        for (int localIdx = threadIdx.x; localIdx < count; localIdx += blockDim.x)
        {
            gatherSendIndice[startIndex + localIdx] = localSendIndice[localIdx];
            gatherBackwardIndice[startIndex + localIdx] = localBackwardIndice[localIdx];
        }
    }
    else
    {
        // recvIndice CTA
        int startIndex = targetRankId == 0 ? 0 : recvCountsCumsum[targetRankId - 1];
        int endIndex = recvCountsCumsum[targetRankId];
        int count = endIndex - startIndex;
        for (int localIdx = threadIdx.x; localIdx < count; localIdx += blockDim.x)
        {
            gatherRecvIndice[startIndex + localIdx] = startIndex + localIdx;
        }
    }
}

__global__ void computeCumsumDevice(int* sendCountsCumsum, int* recvCountsCumsum, int rankId, int rankCount)
{
    int* inputOutputPtr = blockIdx.x == 0 ? sendCountsCumsum : recvCountsCumsum;

    // Use 2 block to comuteCumsum
    typedef cub::BlockScan<int, CUMSUM_THREADS_PER_BLOCK> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
    cudaTriggerProgrammaticLaunchCompletion();
#endif

    int tid = threadIdx.x;
    int threadData = tid < rankCount ? inputOutputPtr[tid] : 0;
    __syncthreads();

    BlockScan(temp_storage).InclusiveSum(threadData, threadData);
    if (tid < rankCount)
    {
        inputOutputPtr[tid] = threadData;
    }
}

__global__ void memsetExpertIdsDevice(
    int* expertIds, int* recvCountsCumsum, int maxTokenCountPerRank, int topK, int slotCount, int rankCount)
{
    int maxTokenCount = maxTokenCountPerRank * rankCount;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
    cudaTriggerProgrammaticLaunchCompletion();
#endif
    int totalRecvTokenCount = *(recvCountsCumsum + rankCount - 1);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i + totalRecvTokenCount * topK < maxTokenCount * topK;
         i += gridDim.x * blockDim.x)
    {
        *(expertIds + i + totalRecvTokenCount * topK) = slotCount;
    }
}

void computeCountAndIndice(int* experts, int* sendCounts, int* recvCounts, int* sendIndiceWorkspace,
    int* backwardIndiceWorkspace, int* recvIndiceWorkspace, int* expertStatics, int* gatheredExpertStatics,
    MoeCommWorkspace workspace, int tokenCount, int maxTokenCountPerRank, int topK, int slotCount, int expertCount,
    int rankId, int rankCount, cudaStream_t stream)
{
    // first rankCount CTAs for count and send, then rankCount / PIPELINE_PER_CTA CTAs only for receive
    int grid_x = rankCount + (rankCount + PIPELINE_PER_CTA - 1) / PIPELINE_PER_CTA;
    int block_size = 1024;
    dim3 block(block_size);
    dim3 grid(grid_x);
    TLLM_CHECK_WITH_INFO(topK >= 1 && topK <= 32, "Only 1 <= topK <= 32 is supported now.");
    auto* kernelFn = computeCountAndIndiceDevice<1>;
    if (topK > 16)
    {
        kernelFn = computeCountAndIndiceDevice<32>;
    }
    else if (topK > 8)
    {
        kernelFn = computeCountAndIndiceDevice<16>;
    }
    else if (topK > 4)
    {
        kernelFn = computeCountAndIndiceDevice<8>;
    }
    else if (topK > 2)
    {
        kernelFn = computeCountAndIndiceDevice<4>;
    }
    else if (topK > 1)
    {
        kernelFn = computeCountAndIndiceDevice<2>;
    }

    launchWithPdlWhenEnabled("computeCountAndIndice", kernelFn, grid, block, 0, stream, experts, sendCounts, recvCounts,
        sendIndiceWorkspace, backwardIndiceWorkspace, recvIndiceWorkspace, expertStatics, gatheredExpertStatics,
        workspace, tokenCount, maxTokenCountPerRank, topK, slotCount, expertCount, rankId, rankCount);
}

void computeCumsum(int* sendCountsCumsum, int* recvCountsCumsum, int rankId, int rankCount, cudaStream_t stream)
{
    int block_size = CUMSUM_THREADS_PER_BLOCK;
    dim3 block(block_size);
    dim3 grid(2);

    launchWithPdlWhenEnabled("computeCumsum", computeCumsumDevice, grid, block, 0, stream, sendCountsCumsum,
        recvCountsCumsum, rankId, rankCount);
}

void moveIndice(int* sendCountsCumsum, int* recvCountsCumsum, int* sendIndice, int* gatherSendIndice,
    int* backwardIndice, int* gatherBackwardIndice, int* recvIndice, int* gatherRecvIndice, int rankId, int rankCount,
    int maxTokenCountPerRank, cudaStream_t stream)
{
    dim3 block(512);
    dim3 grid(rankCount, 2);

    launchWithPdlWhenEnabled("moveIndice", moveIndiceDevice, grid, block, 0, stream, sendCountsCumsum, recvCountsCumsum,
        sendIndice, gatherSendIndice, backwardIndice, gatherBackwardIndice, recvIndice, gatherRecvIndice,
        maxTokenCountPerRank);
}

void memsetExpertIds(int* expertIds, int* recvCountsCumsum, int maxTokenCountPerRank, int topK, int slotCount,
    int rankCount, cudaStream_t stream)
{
    int smCount = tensorrt_llm::common::getMultiProcessorCount();
    int block_size = 256;
    dim3 block(block_size);
    dim3 grid(smCount);

    launchWithPdlWhenEnabled("memsetExpertIds", memsetExpertIdsDevice, grid, block, 0, stream, expertIds,
        recvCountsCumsum, maxTokenCountPerRank, topK, slotCount, rankCount);
}

size_t getMoePrepareWorkspaceSize(int epSize)
{
    return sizeof(MoeCommFifoConnInfo) * epSize;
}

} // namespace moe_prepare

} // namespace tensorrt_llm::kernels
