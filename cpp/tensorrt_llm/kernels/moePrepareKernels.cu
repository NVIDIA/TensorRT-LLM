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

class StepCommunicatorBase
{
public:
    static constexpr int META_SIZE = sizeof(MoeCommFifoConnInfo);

    __device__ __inline__ StepCommunicatorBase(MoeCommFifoConnInfo* fifoConnInfo)
        : fifoConnInfo(fifoConnInfo)
        , localCachedHead(0)
        , localCachedTail(0)
    {
    }

    __forceinline__ __device__ void reset()
    {
        fifoConnInfo->head = 0;
        fifoConnInfo->tail = 0;
    }

    __forceinline__ __device__ void releaseSendStep()
    {
        localCachedHead += 1;
        st_release_sys_global(&(fifoConnInfo->head), uint64_t(localCachedHead));
    }

    __forceinline__ __device__ void releaseRecvStep()
    {
        localCachedTail += 1;
        st_release_sys_global(&(fifoConnInfo->tail), uint64_t(localCachedTail));
    }

    __forceinline__ __device__ uint64_t acquireTail()
    {
        uint64_t tail = ld_acquire_sys_global(&(fifoConnInfo->tail));
        localCachedTail = tail;
        return tail;
    }

    __forceinline__ __device__ uint64_t acquireHead()
    {
        uint64_t head = ld_acquire_sys_global(&(fifoConnInfo->head));
        localCachedHead = head;
        return head;
    }

    __forceinline__ __device__ int acquireNewSendStep()
    {

        int64_t tail;
        do
        {
            tail = acquireTail();
        } while (localCachedHead >= tail + STEP_DEPTH);
        // depth = 2, head = 1, tail = 0 , ok
        // depth = 2, head = 2, tail = 0, should wait

        return localCachedHead % STEP_DEPTH;
    }

    __forceinline__ __device__ int acquireNewRecvStep()
    {
        int64_t head = 0;
        do
        {
            head = acquireHead();
        } while (localCachedTail >= head);

        return localCachedTail % STEP_DEPTH;
    }

public:
    MoeCommFifoConnInfo* fifoConnInfo;
    uint64_t localCachedHead;
    uint64_t localCachedTail;
    int rank;
    int targetRank;
};

// Use MoeCommFifoConnInfo as media to transfer a counter number.
// Use the "head" field as flag.
// Use the "tail" field to transfer the counter number.
class CounterCommunicator
{
public:
    __device__ __inline__ CounterCommunicator(MoeCommFifoConnInfo* fifoConnInfo)
        : fifoConnInfo(fifoConnInfo)
    {
    }

    __forceinline__ __device__ void releaseValue(uint64_t value)
    {
        // Avoid block on 0
        st_release_sys_global(&(fifoConnInfo->count), value + 1);
    }

    __forceinline__ __device__ uint64_t acquireValue()
    {
        uint64_t localCount = 0;
        do
        {
            localCount = ld_acquire_sys_global(&(fifoConnInfo->count));
        } while (localCount == 0);

        fifoConnInfo->count = 0; // reset the count

        return localCount - 1;
    }

protected:
    MoeCommFifoConnInfo* fifoConnInfo;
};

template <int kThreadsGroupSize>
__device__ __forceinline__ void computeCountAndSend(int* experts, int tokenCount, int* sharedSendRecvRankCount,
    int* sendCounts, int* sendIndiceWorkspace, int* backwardIndiceWorkspace, MoeCommWorkspace workspace,
    int maxTokenCountPerRank, int expertCount, int topK, int epRank, int epSize)
{
    cg::thread_block_tile<kThreadsGroupSize> tile = cg::tiled_partition<kThreadsGroupSize>(cg::this_thread_block());
    int laneInTile = tile.thread_rank();
    int tileId = threadIdx.x / kThreadsGroupSize;
    int tileCountPerBlock = blockDim.x / kThreadsGroupSize;
    int expertCountPerRank = expertCount / epSize;
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
    if (threadIdx.x == 0)
    {
        CounterCommunicator counter(workspace.getFifoConnInfo(true, epRank, targetRankId, 0, epSize, 1));
        int count = *(sharedSendRecvRankCount);
        // printf("sendRecvCount: %d, rankId: %d, targetRankId: %d\n", count, rankId, targetRankId);
        counter.releaseValue(uint64_t(count));
        *(sendCounts + targetRankId) = count;
    }
}

__device__ __forceinline__ void recvCount(int* recvIndiceWorkspace, int* recvCounts, int* sharedCountsBase,
    MoeCommWorkspace workspace, int maxTokenCountPerRank, int rankId, int rankCount)
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
    int rankRecvCount;
    if (rankTile.thread_rank() == 0)
    {
        CounterCommunicator counter(workspace.getFifoConnInfo(false, rankId, targetRankId, 0, rankCount, 1));
        rankRecvCount = int(counter.acquireValue());
        // printf("rankRecvCount: %d, rankId: %d, targetRankId: %d\n", rankRecvCount, rankId, targetRankId);
        *(recvCounts + targetRankId) = rankRecvCount;
        *(sharedCountsThisRank) = rankRecvCount;
    }
    rankTile.sync();

    rankRecvCount = *(sharedCountsThisRank);
    for (int tokenId = unitId; tokenId < rankRecvCount; tokenId += UNIT_PER_PIPELINE)
    {
        *(localRecvIndice + tokenId) = tokenId;
    }
}

template <int kThreadsGroupSize>
__global__ void computeCountAndIndiceDevice(int* experts, int* sendCounts, int* recvCounts, int* sendIndiceWorkspace,
    int* backwardIndiceWorkspace, int* recvIndiceWorkspace, MoeCommWorkspace workspace, int tokenCount,
    int maxTokenCountPerRank, int topK, int expertCount, int rankId, int rankCount)
{
    __shared__ int sharedCounts[PIPELINE_PER_CTA];
    bool isSender = blockIdx.x < rankCount;
    if (isSender)
    {
        computeCountAndSend<kThreadsGroupSize>(experts, tokenCount, &sharedCounts[0], sendCounts, sendIndiceWorkspace,
            backwardIndiceWorkspace, workspace, maxTokenCountPerRank, expertCount, topK, rankId, rankCount);
    }
    else
    {
        recvCount(
            recvIndiceWorkspace, recvCounts, &sharedCounts[0], workspace, maxTokenCountPerRank, rankId, rankCount);
    }
}

__global__ void moveIndiceDevice(int* sendCountsCumsum, int* recvCountsCumsum, int* sendIndice, int* gatherSendIndice,
    int* backwardIndice, int* gatherBackwardIndice, int* recvIndice, int* gatherRecvIndice, int maxTokenCountPerRank)
{
    int targetRankId = blockIdx.x;
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

    int tid = threadIdx.x;
    int threadData = tid < rankCount ? inputOutputPtr[tid] : 0;
    int count = threadData;
    __syncthreads();

    BlockScan(temp_storage).InclusiveSum(threadData, threadData);
    if (tid < rankCount)
    {
        inputOutputPtr[tid] = threadData;
        // printf("cumsum, send? : %d, rankId:%d, tid:%d, threadData:%d, count:%d\n", blockIdx.x == 0, rankId, tid,
        // threadData, count);
    }
}

template <typename STEP_COMMUNICATOR_TYPE>
class PacketPipeline
{
public:
    __device__ __inline__ PacketPipeline(
        void* bufferBase, STEP_COMMUNICATOR_TYPE* stepCommunicator, int* sharedNewStepPtr, bool isSender)
        : bufferBase(bufferBase)
        , stepCommunicator(stepCommunicator)
        , shared_new_step(sharedNewStepPtr)
    {
        step = 0;
        needRelease = false;
        packetId = isSender ? 0 : PACKET_PER_STEP - 1;
    }

    __device__ __forceinline__ void* getFirstSendPacket()
    {
        return bufferBase;
    }

    __device__ __inline__ void* finishSendPacket(bool acquireNewStep)
    {

        packetId++;
        if (packetId < PACKET_PER_STEP)
        {
            return acquireNewStep ? bufferBase + step * PACKET_PER_STEP * PACKET_SIZE + packetId * PACKET_SIZE
                                  : nullptr;
        }

        __syncthreads();
        if (threadIdx.x == 0)
        {
            stepCommunicator->releaseSendStep();
            if (acquireNewStep)
            {
                step = stepCommunicator->acquireNewSendStep();
                *(shared_new_step) = step;
            }
        }
        __syncthreads();

        if (acquireNewStep)
        {
            step = *(shared_new_step);
            packetId = 0;
            return bufferBase + step * PACKET_SIZE * PACKET_PER_STEP;
        }

        return nullptr;
    }

    __device__ __forceinline__ void* sendFinalize()
    {
        if (packetId > 0 && threadIdx.x == 0)
        {
            stepCommunicator->releaseSendStep();
        }
    }

    __device__ __inline__ void* getNewRecvPacket()
    {
        packetId++;
        if (packetId < PACKET_PER_STEP)
        {
            return bufferBase + step * PACKET_PER_STEP * PACKET_SIZE + packetId * PACKET_SIZE;
        }

        __syncthreads();
        if (threadIdx.x == 0)
        {
            if (needRelease)
            {
                stepCommunicator->releaseRecvStep();
            }
            step = stepCommunicator->acquireNewRecvStep();
            needRelease = true;
            *(shared_new_step) = step;
        }
        __syncthreads();
        packetId = 0;
        step = *(shared_new_step);
        void* packetPtr = bufferBase + step * PACKET_SIZE * PACKET_PER_STEP;

        return packetPtr;
    }

    __device__ __forceinline__ void reset()
    {
        if (threadIdx.x == 0)
        {
            stepCommunicator->reset();
        }
    }

    void* bufferBase;
    STEP_COMMUNICATOR_TYPE* stepCommunicator;
    int step;
    int packetId;
    bool needRelease;
    int* shared_new_step;
};

template <typename STEP_COMMUNICATOR_TYPE>
__global__ void allToAllMetadataDevice(int* sendExperts, int* recvExperts, float* sendScales, float* recvScales,
    int* localExpertStatics, int* gatheredExpertStatics, MoeCommWorkspace workspace, int* sendCountsCumsum,
    int* localSendIndice, int* recvCountsCumsum, int* localRecvIndice, int tokenCount, int maxTokenCountPerRank,
    int topK, int expertCount, int slotCount, int rankId, int rankCount)
{
    bool isSender = (blockIdx.y == 0);
    int targetRankId = blockIdx.x;
    int slotCountPerRank = slotCount / rankCount;
    int groupSize = topK / UNIT_SIZE;
    int groupId = threadIdx.x % groupSize;

    __shared__ int sharedNewStep;
    __align__(16) int experts[UNIT_SIZE];
    __align__(16) float scales[UNIT_SIZE];

    uint8_t* bufferBase = (uint8_t*) (workspace.getFifoBasePtr(isSender, rankId, targetRankId, 0, 1));
    STEP_COMMUNICATOR_TYPE stepCommunicator(workspace.getFifoConnInfo(isSender, rankId, targetRankId, 0, rankCount, 1));
    PacketPipeline<STEP_COMMUNICATOR_TYPE> pipeline(bufferBase, &stepCommunicator, &sharedNewStep, isSender);

    if (isSender)
    {
        int baseCumsum = targetRankId == 0 ? 0 : *(sendCountsCumsum + targetRankId - 1);
        int sendTokenCount = *(sendCountsCumsum + targetRankId) - baseCumsum;
        int unitCount = sendTokenCount * topK / UNIT_SIZE;

        void* packPtr = pipeline.getFirstSendPacket();
        int indexBase = 0;
        int staticCopyBase = 0;
        bool acquireNewStep = unitCount > 0 || (localExpertStatics != nullptr && expertCount > 0);
        while (acquireNewStep)
        {
            if (threadIdx.x < UNIT_PER_ITER)
            {
                int index = indexBase + threadIdx.x;
                if (index < unitCount)
                {
                    int tokenId = *(localSendIndice + maxTokenCountPerRank * targetRankId + (index / groupSize));
                    *((int4*) (experts)) = *(int4*) (sendExperts + tokenId * topK + groupId * UNIT_SIZE);
                    *((float4*) (scales)) = *(float4*) (sendScales + tokenId * topK + groupId * UNIT_SIZE);

#pragma unroll
                    for (int j = 0; j < UNIT_SIZE; j++)
                    {
                        int expertId = experts[j];
                        if (expertId / slotCountPerRank != targetRankId)
                        {
                            experts[j] = slotCount;
                            scales[j] = 0.0f;
                        }
                    }

                    int* expertsPtr = (int*) (packPtr) + threadIdx.x * UNIT_SIZE;
                    float* scaleBasePtr = (float*) (packPtr + SCALE_OFFSET);
                    float* scalesPtr = (float*) (scaleBasePtr) + threadIdx.x * UNIT_SIZE;
                    *((int4*) (expertsPtr)) = *((int4*) (experts));
                    *((float4*) (scalesPtr)) = *((float4*) (scales));
                }
            }
            else if (localExpertStatics != nullptr)
            {
                int staticCopyIdx = threadIdx.x - UNIT_PER_ITER;
                if (staticCopyBase + staticCopyIdx * 4 < expertCount)
                {
                    int4* staticBasePtr = (int4*) (packPtr + STATIC_COPY_OFFSET);
                    int4 staticData = *(int4*) (localExpertStatics + staticCopyBase + staticCopyIdx * 4);
                    *(staticBasePtr + staticCopyIdx) = staticData;
                }
            }

            indexBase += UNIT_PER_ITER;
            staticCopyBase += STATIC_COPY_PER_ITER * 4;
            acquireNewStep = indexBase < unitCount || staticCopyBase < expertCount;
            packPtr = pipeline.finishSendPacket(acquireNewStep);
        }

        pipeline.sendFinalize();
    }
    else
    {
        int baseCumsum = targetRankId == 0 ? 0 : *(recvCountsCumsum + targetRankId - 1);
        int recvTokenCount = *(recvCountsCumsum + targetRankId) - baseCumsum;
        int recvUnitCount = recvTokenCount * groupSize;

        int unitIdBase = 0;
        int staticCopyBase = 0;
        while (unitIdBase < recvUnitCount || (localExpertStatics != nullptr && staticCopyBase < expertCount))
        {
            void* packetPtr = pipeline.getNewRecvPacket();
            int packetUnitCount
                = unitIdBase + UNIT_PER_ITER < recvUnitCount ? UNIT_PER_ITER : recvUnitCount - unitIdBase;
            packetUnitCount = max(packetUnitCount, 0);
            if (threadIdx.x < UNIT_PER_ITER)
            {
                if (threadIdx.x < packetUnitCount)
                {
                    int* expertsPtr = (int*) (packetPtr) + threadIdx.x * UNIT_SIZE;
                    float* scaleBasePtr = (float*) (packetPtr + SCALE_OFFSET);
                    float* scalesPtr = scaleBasePtr + threadIdx.x * UNIT_SIZE;
                    *((int4*) (experts)) = *((int4*) (expertsPtr));
                    *((float4*) (scales)) = *((float4*) (scalesPtr));

                    int tokenId = baseCumsum + (unitIdBase + threadIdx.x) / groupSize;

                    int4* dstExpertsPtr = (int4*) (recvExperts + tokenId * topK + groupId * UNIT_SIZE);
                    float4* dstScalesPtr = (float4*) (recvScales + tokenId * topK + groupId * UNIT_SIZE);
                    *dstExpertsPtr = *((int4*) (experts));
                    *dstScalesPtr = *((float4*) (scales));
                }
            }
            else if (localExpertStatics != nullptr)
            {
                int staticCopyIdx = threadIdx.x - UNIT_PER_ITER;
                if (staticCopyBase + staticCopyIdx * 4 < expertCount)
                {
                    int4* staticBasePtr = (int4*) (packetPtr + STATIC_COPY_OFFSET);
                    int4 staticData = *(staticBasePtr + staticCopyIdx);
                    *(int4*) (gatheredExpertStatics + targetRankId * expertCount + staticCopyBase + staticCopyIdx * 4)
                        = staticData;
                }
            }

            unitIdBase += packetUnitCount;
            staticCopyBase += STATIC_COPY_PER_ITER * 4;
        }

        pipeline.reset();
    }
}

void computeCountAndIndice(int* experts, int* sendCounts, int* recvCounts, int* sendIndiceWorkspace,
    int* backwardIndiceWorkspace, int* recvIndiceWorkspace, MoeCommWorkspace workspace, int tokenCount,
    int maxTokenCountPerRank, int topK, int expert_count, int rankId, int rankCount, cudaStream_t stream)
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
    kernelFn<<<grid, block, 0, stream>>>(experts, sendCounts, recvCounts, sendIndiceWorkspace, backwardIndiceWorkspace,
        recvIndiceWorkspace, workspace, tokenCount, maxTokenCountPerRank, topK, expert_count, rankId, rankCount);
}

void computeCumsum(int* sendCountsCumsum, int* recvCountsCumsum, int rankId, int rankCount, cudaStream_t stream)
{
    int block_size = CUMSUM_THREADS_PER_BLOCK;
    dim3 block(block_size);
    dim3 grid(2);
    computeCumsumDevice<<<grid, block, 0, stream>>>(sendCountsCumsum, recvCountsCumsum, rankId, rankCount);
}

void moveIndice(int* sendCountsCumsum, int* recvCountsCumsum, int* sendIndice, int* gatherSendIndice,
    int* backwardIndice, int* gatherBackwardIndice, int* recvIndice, int* gatherRecvIndice, int rankId, int rankCount,
    int maxTokenCountPerRank, cudaStream_t stream)
{
    dim3 block(512);
    dim3 grid(rankCount, 2);
    moveIndiceDevice<<<grid, block, 0, stream>>>(sendCountsCumsum, recvCountsCumsum, sendIndice, gatherSendIndice,
        backwardIndice, gatherBackwardIndice, recvIndice, gatherRecvIndice, maxTokenCountPerRank);
}

void allToAllMetadata(int* sendExperts, int* recvExperts, float* sendScales, float* recvScales, int* localExpertStatics,
    int* gatheredExpertStatics, MoeCommWorkspace workspace, int* sendCountsCumsum, int* localSendIndice,
    int* recvCountsCumsum, int* localRecvIndice, int tokenCount, int maxTokenCountPerRank, int topK, int expertCount,
    int slotCount, int rankId, int rankCount, cudaStream_t stream)
{
    int block_size = localExpertStatics == nullptr ? UNIT_PER_ITER : UNIT_PER_ITER + STATIC_COPY_PER_ITER;
    dim3 block(block_size);
    dim3 grid(rankCount, 2);
    assert(topK == 8);
    allToAllMetadataDevice<StepCommunicatorBase><<<grid, block, 0, stream>>>(sendExperts, recvExperts, sendScales,
        recvScales, localExpertStatics, gatheredExpertStatics, workspace, sendCountsCumsum, localSendIndice,
        recvCountsCumsum, localRecvIndice, tokenCount, maxTokenCountPerRank, topK, expertCount, slotCount, rankId,
        rankCount);
}

size_t getMoePrepareWorkspaceSize(int epSize)
{
    return (STEP_DEPTH * PACKET_PER_STEP * PACKET_SIZE + StepCommunicatorBase::META_SIZE) * epSize;
}

} // namespace moe_prepare

} // namespace tensorrt_llm::kernels
