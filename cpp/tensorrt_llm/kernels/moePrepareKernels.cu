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
        st_release_sys_global(&(fifoConnInfo->tail), uint64_t(localCachedTail));
        localCachedTail += 1;
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
        } while (localCachedHead >= tail + PACKET_DEPTH);
        // depth = 2, head = 1, tail = 0 , ok
        // depth = 2, head = 2, tail = 0, should wait

        return localCachedHead % PACKET_DEPTH;
    }

    __forceinline__ __device__ int acquireNewRecvStep()
    {
        int64_t head = 0;
        do
        {
            head = acquireHead();
        } while (localCachedTail >= head);

        return localCachedTail % PACKET_DEPTH;
    }

protected:
    MoeCommFifoConnInfo* fifoConnInfo;
    uint64_t localCachedHead;
    uint64_t localCachedTail;
};

struct IndexInfo
{
    int unitIdBase;
    int cachedUnitId;
    int cachedFirstValidThread;

    __device__ __inline__ IndexInfo()
        : unitIdBase(0)
        , cachedUnitId(0)
        , cachedFirstValidThread(0)
    {
    }
};

template <typename STEP_COMMUNICATOR_TYPE, bool IS_SENDER>
class PipelineBase
{
public:
    static constexpr int BUFFER_SIZE_PER_PIPELINE = PACKET_DEPTH * PACKET_SIZE + STEP_COMMUNICATOR_TYPE::META_SIZE;

    static_assert(THREADS_PER_UNIT <= 32, "THREADS_PER_UNIT must be less than or equal to 32");
    static_assert(UNIT_BYTES_SIZE % THREADS_PER_UNIT == 0, "UNIT_BYTES_SIZE must be divisible by THREADS_PER_UNIT");

    static constexpr int MAX_UNIT_PER_ITERATION = THREADS_PER_PIPELINE / THREADS_PER_UNIT;
    static_assert(UNIT_COUNT_PER_PACKET % MAX_UNIT_PER_ITERATION == 0,
        "UNIT_COUNT_PER_PACKET must be divisible by MAX_UNIT_PER_ITERATION");

    __device__ __inline__ PipelineBase(
        uint8_t* bufferBase, STEP_COMMUNICATOR_TYPE* stepCommunicator, int* tempStorage, int* shared_new_step, int rank)
        : bufferBase(bufferBase)
        , stepCommunicator(stepCommunicator)
        , tempStorage(tempStorage)
        , pipelineTile(cg::tiled_partition<THREADS_PER_PIPELINE>(cg::this_thread_block()))
        , unitTile(cg::tiled_partition<THREADS_PER_UNIT>(pipelineTile))
        , shared_new_step(shared_new_step)
        , packetPtr(bufferBase)
        , packetCounterPtr(bufferBase + UNIT_BYTES_SIZE * UNIT_COUNT_PER_PACKET)
        , allocatedUnit(0)
        , unitIndex(pipelineTile.thread_rank() / THREADS_PER_UNIT)
        , recivedUnit(0)
        , totalUnit(0)
        ,
        // for debug
        rank(rank)
    {
        if (!IS_SENDER)
        {
            acquireNewPacket();
        }
    }

    __device__ __forceinline__ void reset()
    {
        stepCommunicator->reset();
    }

    __device__ __inline__ void finishPacket()
    {
        if (pipelineTile.thread_rank() == 0)
        {
            if (IS_SENDER)
            {
                int rankOffset = threadIdx.x / (THREADS_PER_UNIT * UNIT_PER_PIPELINE);
                int* packetCounterI32Ptr = (int*) (packetCounterPtr);
                *packetCounterI32Ptr = allocatedUnit;
                stepCommunicator->releaseSendStep();
            }
            else
            {
                stepCommunicator->releaseRecvStep();
            }
        }

        pipelineTile.sync();
    }

    __device__ __inline__ void acquireNewPacket()
    {
        int step;
        if (pipelineTile.thread_rank() == 0)
        {
            int step;
            if (IS_SENDER)
            {
                step = stepCommunicator->acquireNewSendStep();
            }
            else
            {
                step = stepCommunicator->acquireNewRecvStep();
            }

            *(shared_new_step) = step;
        }

        pipelineTile.sync();
        step = *(shared_new_step);
        packetPtr = bufferBase + step * PACKET_SIZE;
        allocatedUnit = 0;
        packetCounterPtr = packetPtr + UNIT_BYTES_SIZE * UNIT_COUNT_PER_PACKET;
        pipelineTile.sync();

        if (!IS_SENDER)
        {
            recivedUnit = 0;
            packetCounterPtr = packetPtr + UNIT_BYTES_SIZE * UNIT_COUNT_PER_PACKET;
            allocatedUnit = *((int*) packetCounterPtr);
        }
    }

    __device__ __inline__ uint8_t* conditionalAcquireSendBuffer(bool needBuffer = true)
    {
        bool unitNeedAlloc = unitTile.any(needBuffer);
        int isFirstThreadOnValidUnit = unitTile.thread_rank() == 0 && unitNeedAlloc ? 1 : 0;

        int allocUnit = cg::reduce(pipelineTile, isFirstThreadOnValidUnit, cg::plus<int>());

        if (allocUnit == 0)
        {
            return nullptr;
        }

        bool packetNeedAlloc = allocUnit + allocatedUnit > UNIT_COUNT_PER_PACKET;
        if (packetNeedAlloc)
        {
            finishPacket();
            acquireNewPacket();
        }

        // InclusiveSum the unit index on pipeline
        int unitId = pipelineTile.thread_rank() / THREADS_PER_UNIT;
        if (unitTile.thread_rank() == 0)
        {
            tempStorage[unitId] = isFirstThreadOnValidUnit;
        }
        pipelineTile.sync();

        if (unitTile.thread_rank() == 0)
        {
            unitIndex = 0;
            for (int i = 0; i < unitId; i++)
            {
                unitIndex += tempStorage[i];
            }
        }
        pipelineTile.sync();

        if (unitTile.thread_rank() == 0)
        {
            tempStorage[unitId] = unitIndex;
        }
        pipelineTile.sync();

        unitIndex = tempStorage[unitId];
        uint8_t* retPtr = nullptr;
        if (unitNeedAlloc)
        {
            retPtr
                = packetPtr + (unitIndex + allocatedUnit) * UNIT_BYTES_SIZE + unitTile.thread_rank() * BYTES_PER_THREAD;
        }

        indexInfo.unitIdBase = totalUnit;
        allocatedUnit += allocUnit;
        totalUnit += allocUnit;
        if (unitNeedAlloc)
        {
            indexInfo.cachedUnitId = unitIndex;
            int mask = unitTile.ballot(needBuffer);
            indexInfo.cachedFirstValidThread = __ffs(mask) - 1;
        }

        return retPtr;
    }

    __device__ __forceinline__ bool const recvEnd(int expectedTotalRecivedUnit)
    {
        return totalUnit == expectedTotalRecivedUnit;
    }

    bool flag;

    __device__ __inline__ uint8_t* conditionalAcquireRecvBuffer()
    {
        if (recivedUnit >= allocatedUnit)
        {
            finishPacket();
            acquireNewPacket();
        }

        uint8_t* retPtr = nullptr;
        if (recivedUnit + MAX_UNIT_PER_ITERATION <= allocatedUnit)
        {
            // fully received
            retPtr
                = packetPtr + (recivedUnit + unitIndex) * UNIT_BYTES_SIZE + unitTile.thread_rank() * BYTES_PER_THREAD;
            totalUnit += MAX_UNIT_PER_ITERATION;
            recivedUnit += MAX_UNIT_PER_ITERATION;
        }
        else
        {
            if (recivedUnit + unitIndex < allocatedUnit)
            {
                retPtr = packetPtr + (recivedUnit + unitIndex) * UNIT_BYTES_SIZE
                    + unitTile.thread_rank() * BYTES_PER_THREAD;
            }

            totalUnit += (allocatedUnit - recivedUnit);
            recivedUnit = allocatedUnit;
        }

        return retPtr;
    }

    __device__ __inline__ uint8_t* acquireSendBuffer()
    {
        if (allocatedUnit + MAX_UNIT_PER_ITERATION > UNIT_COUNT_PER_PACKET)
        {
            finishPacket();
            acquireNewPacket();
        }

        uint8_t* retPtr
            = packetPtr + (allocatedUnit + unitIndex) * UNIT_BYTES_SIZE + unitTile.thread_rank() * BYTES_PER_THREAD;

        allocatedUnit += MAX_UNIT_PER_ITERATION;

        return retPtr;
    }

    // we only need dense require now
    __device__ __inline__ uint8_t* acquireRecvBuffer()
    {
        if (recivedUnit >= allocatedUnit)
        {
            finishPacket();
            acquireNewPacket();
        }

        uint8_t* retPtr = nullptr;
        if (recivedUnit + unitIndex < allocatedUnit)
        {
            retPtr
                = packetPtr + (recivedUnit + unitIndex) * UNIT_BYTES_SIZE + unitTile.thread_rank() * BYTES_PER_THREAD;
        }

        recivedUnit += MAX_UNIT_PER_ITERATION;

        return retPtr;
    }

    __device__ __forceinline__ IndexInfo& getIndexInfo()
    {
        return indexInfo;
    }

    __device__ __forceinline__ int getTotalRecivedUnit()
    {
        return totalUnit;
    }

protected:
    uint8_t* bufferBase;
    STEP_COMMUNICATOR_TYPE* stepCommunicator;
    int* tempStorage;
    int* shared_new_step;
    uint8_t* packetPtr;
    uint8_t* packetCounterPtr;
    int allocatedUnit;
    int recivedUnit;
    int totalUnit;
    int rank;
    // used for indices calculation
    IndexInfo indexInfo;
    int unitIndex;
    cg::thread_block_tile<THREADS_PER_PIPELINE> pipelineTile;
    cg::thread_block_tile<THREADS_PER_UNIT> unitTile;
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

__global__ void rankCountDevice(int* experts, int* sendCountsCumsum, int* recvCountsCumsum, MoeCommWorkspace workspace,
    int tokenCount, int topK, int expert_count, int rank_id, int rank_count, int* sendCountReady)
{
    bool isSender = (blockIdx.x < gridDim.x - 1);
    using BlockScan = cub::BlockScan<int, THREADS_PER_UNIT * UNIT_PER_PIPELINE * PIPELINE_PER_CTA>;
    __shared__ typename BlockScan::TempStorage tempStorage;
    __shared__ int sharedCountsPerRank[PIPELINE_PER_CTA];
    int expertCountPerRank = (expert_count + rank_count - 1) / rank_count;

    if (isSender)
    {
        int laneID = threadIdx.x % THREADS_PER_UNIT;
        int unitId = ((threadIdx.x - laneID) / THREADS_PER_UNIT) % UNIT_PER_PIPELINE;
        int rankOffset = threadIdx.x / (THREADS_PER_UNIT * UNIT_PER_PIPELINE);
        int target_rankId = blockIdx.x * PIPELINE_PER_CTA + rankOffset;
        cg::thread_block_tile<THREADS_PER_PIPELINE> rankTile
            = cg::tiled_partition<THREADS_PER_PIPELINE>(cg::this_thread_block());
        cg::thread_block_tile<THREADS_PER_UNIT> unitTile = cg::tiled_partition<THREADS_PER_UNIT>(rankTile);
        CounterCommunicator counter(workspace.getFifoConnInfo(true, rank_id, target_rankId, 0, rank_count, 1));

        int* sharedCountsThisRank = sharedCountsPerRank + rankOffset;
        if (rankTile.thread_rank() == 0)
        {
            *sharedCountsThisRank = 0;
        }
        rankTile.sync();

        for (int token_id = unitId; token_id < tokenCount; token_id += UNIT_PER_PIPELINE)
        {
            bool is_valid = false;
            if (laneID < topK)
            {
                int expert_id = *(experts + token_id * topK + laneID);
                is_valid |= ((expert_id / expertCountPerRank) == target_rankId);
            }
            bool token_is_valid = unitTile.any(is_valid);

            if (unitTile.thread_rank() == 0 && token_is_valid)
            {
                atomicAdd_block(sharedCountsThisRank, 1);
            }
        }
        rankTile.sync();

        // send the count of this rank
        if (rankTile.thread_rank() == 0 && target_rankId < rank_count)
        {
            int countThisRank = *(sharedCountsThisRank);
            counter.releaseValue(uint64_t(countThisRank));
        }

        __syncthreads();

        if (threadIdx.x < PIPELINE_PER_CTA)
        {
            int rank = PIPELINE_PER_CTA * blockIdx.x + threadIdx.x;
            if (rank < rank_count)
            {
                *(sendCountsCumsum + rank) = sharedCountsThisRank[threadIdx.x];
            }
        }

        // block 0 compute send cumsum
        if (blockIdx.x == 0)
        {
            int volatile value = -1;

            if (threadIdx.x < rank_count)
            {
                int volatile* sendCountPtr = sendCountsCumsum + threadIdx.x;
                do
                {
                    value = *(sendCountPtr);
                } while (value < 0);
            }

            __syncthreads();

            int cumsum;
            BlockScan(tempStorage).InclusiveSum(value, cumsum);
            if (threadIdx.x < rank_count)
            {
                *(sendCountsCumsum + threadIdx.x) = cumsum;
            }
        }
    }
    else // receiver
    {
        int value = 0;
        CounterCommunicator counter(workspace.getFifoConnInfo(false, rank_id, threadIdx.x, 0, rank_count, 1));
        if (threadIdx.x < rank_count)
        {
            value = int(counter.acquireValue());
        }

        __syncthreads();

        int cumsum;
        BlockScan(tempStorage).InclusiveSum(value, cumsum);
        if (threadIdx.x < rank_count)
        {
            *(recvCountsCumsum + threadIdx.x) = cumsum;
        }
    }
}

template <typename STEP_COMMUNICATOR_TYPE>
__global__ void generateIndiceAndLocalDataDevice(int* experts, float* scales, int* localExperts, float* localScales,
    int* sendCountsCumsum, int* recvCountsCumsum, int* sendIndice, int* backwardIndice, int* recvIndice,
    MoeCommWorkspace workspace, int tokenCount, int topK, int expertCount, int rank, int rankCount)
{
    bool isSender = (blockIdx.y == 0);
    int laneId = threadIdx.x % THREADS_PER_UNIT;
    int unitId = ((threadIdx.x - laneId) / THREADS_PER_UNIT) % UNIT_PER_PIPELINE;
    int rankOffset = threadIdx.x / (THREADS_PER_UNIT * UNIT_PER_PIPELINE);
    int targetRankId = blockIdx.x * PIPELINE_PER_CTA + rankOffset;
    int expertCountPerRank = expertCount / rankCount;

    __shared__ int tempStorage[UNIT_PER_PIPELINE * PIPELINE_PER_CTA];
    __shared__ int sharedNewStep[PIPELINE_PER_CTA];

    if (targetRankId >= rankCount)
    {
        return;
    }

    if (isSender)
    {
        int cumsum = targetRankId == 0 ? 0 : *(sendCountsCumsum + targetRankId - 1);
        uint8_t* bufferBase = (uint8_t*) (workspace.getFifoBasePtr(true, rank, targetRankId, 0, 1));
        STEP_COMMUNICATOR_TYPE stepCommunicator(workspace.getFifoConnInfo(true, rank, targetRankId, 0, rankCount, 1));
        PipelineBase<STEP_COMMUNICATOR_TYPE, true> pipeline(bufferBase, &stepCommunicator,
            tempStorage + rankOffset * UNIT_PER_PIPELINE, &(sharedNewStep[rankOffset]), rank);

        int alignTokenCount = (tokenCount + UNIT_PER_PIPELINE - 1) / UNIT_PER_PIPELINE * UNIT_PER_PIPELINE;

        for (int tokenId = unitId; tokenId < alignTokenCount; tokenId += UNIT_PER_PIPELINE)
        {
            uint8_t* ptr = nullptr;
            int rankId = -1;
            int expertId;
            if (tokenId < tokenCount && laneId < topK)
            {
                int offset = tokenId * topK + laneId;
                int* source_ptr = (int*) (experts + offset);
                expertId = *source_ptr;
                rankId = expertId / expertCountPerRank;
                ptr = pipeline.conditionalAcquireSendBuffer(rankId == targetRankId);
            }
            else
            {
                ptr = pipeline.conditionalAcquireSendBuffer(false);
            }

            if (ptr)
            {
                float scale = *(scales + tokenId * topK + laneId);
                if (rankId != targetRankId)
                {
                    scale = 0.0f;
                    expertId = expertCount;
                }
                // TODO: pack expertId and scale
                *(int*) ptr = expertId;
                *(float*) (ptr + sizeof(int)) = scale;

                if (laneId == 0)
                {
                    int sendTokenId = pipeline.getIndexInfo().unitIdBase + pipeline.getIndexInfo().cachedUnitId;
                    int sendUnitId = pipeline.getIndexInfo().cachedFirstValidThread;
                    *(sendIndice + cumsum + sendTokenId) = tokenId;
                    *(backwardIndice + cumsum + sendTokenId) = tokenId * topK + sendUnitId;
                }
            }
        }
        pipeline.finishPacket();
    }
    else
    { // recv
        int cumsum = targetRankId == 0 ? 0 : *(recvCountsCumsum + targetRankId - 1);
        int recvTokenCount = *(recvCountsCumsum + targetRankId) - cumsum;
        uint8_t* bufferBase = (uint8_t*) (workspace.getFifoBasePtr(false, rank, targetRankId, 0, 1));
        STEP_COMMUNICATOR_TYPE stepCommunicator(workspace.getFifoConnInfo(false, rank, targetRankId, 0, rankCount, 1));
        PipelineBase<STEP_COMMUNICATOR_TYPE, false> pipeline(bufferBase, &stepCommunicator,
            tempStorage + rankOffset * UNIT_PER_PIPELINE, &(sharedNewStep[rankOffset]), rank);

        int tokenIdBase = 0;
        while (!pipeline.recvEnd(recvTokenCount))
        {
            uint8_t* ptr = pipeline.conditionalAcquireRecvBuffer();
            if (ptr)
            {
                int expertId;
                float scale;
                expertId = *(int*) ptr;
                scale = *(float*) (ptr + sizeof(int));

                int tokenId = tokenIdBase + unitId;
                tokenIdBase = pipeline.getTotalRecivedUnit();

                if (laneId < topK)
                {
                    *(localExperts + (cumsum + tokenId) * topK + laneId) = expertId;
                    *(localScales + (cumsum + tokenId) * topK + laneId) = scale;
                }
                if (laneId == 0)
                {
                    *(recvIndice + cumsum + tokenId) = cumsum + tokenId;
                }
            }
            else
            {
                tokenIdBase = pipeline.getTotalRecivedUnit();
            }
        }
        // Reset workspace value for next use
        pipeline.reset();
    }
}

void rankCount(int* experts, int* sendCountsCumsum, int* recvCountsCumsum, MoeCommWorkspace workspace, int tokenCount,
    int topK, int expert_count, int rank_id, int rankCount, int* sendCountReady, cudaStream_t stream)
{
    TLLM_CUDA_CHECK(cudaMemsetAsync(sendCountsCumsum, -1, sizeof(int) * rankCount, stream));
    // One for recv, others for send.
    int cta_count = (rankCount + PIPELINE_PER_CTA - 1) / PIPELINE_PER_CTA + 1;
    int block_size = THREADS_PER_CTA;
    dim3 block(block_size);
    dim3 grid(cta_count);
    rankCountDevice<<<grid, block, 0, stream>>>(experts, sendCountsCumsum, recvCountsCumsum, workspace, tokenCount,
        topK, expert_count, rank_id, rankCount, sendCountReady);
}

void generateIndiceAndLocalData(int* experts, float* scales, int* localExperts, float* localScales,
    int* sendCountsCumsum, int* recvCountsCumsum, int* sendIndice, int* backwardIndice, int* recvIndice,
    MoeCommWorkspace workspace, int tokenCount, int topK, int expertCount, int rank, int rankCount, cudaStream_t stream)
{
    int grid_x = (rankCount + PIPELINE_PER_CTA - 1) / PIPELINE_PER_CTA;
    int block_size = THREADS_PER_CTA;
    dim3 block(block_size);
    dim3 grid(grid_x, 2);
    generateIndiceAndLocalDataDevice<StepCommunicatorBase><<<grid, block, 0, stream>>>(experts, scales, localExperts,
        localScales, sendCountsCumsum, recvCountsCumsum, sendIndice, backwardIndice, recvIndice, workspace, tokenCount,
        topK, expertCount, rank, rankCount);
}

size_t getMoePrepareWorkspaceSize(int epSize)
{
    return (PACKET_DEPTH * PACKET_SIZE + StepCommunicatorBase::META_SIZE) * epSize;
}

} // namespace moe_prepare

} // namespace tensorrt_llm::kernels
