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
#include <cub/cub.cuh>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

namespace tensorrt_llm::kernels
{

namespace moe_prepare
{

__device__  __forceinline__ void st_release_sys_global(volatile uint64_t *ptr, uint64_t val) {
    asm volatile("st.release.sys.global.u64 [%0], %1;"::"l"(ptr), "l"(val) : "memory");
}

__device__ __forceinline__ uint64_t ld_acquire_sys_global(volatile uint64_t *ptr) {
    uint64_t ret;
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(ret) : "l"(ptr));
    return ret;
}

class StepCommunicatorBase
{
public:
    static constexpr int META_SIZE = sizeof(MoeCommFifoConnInfo);

    __device__ __inline__ StepCommunicatorBase(MoeCommFifoConnInfo* fifoConnInfo)
    : fifoConnInfo(fifoConnInfo),   
    localCachedHead(0),
    localCachedTail(0)
    {
    }

    __forceinline__ __device__ void releaseSendStep()
    {
        localCachedHead += 1;
        //printf("Release send step, threadIdx.x = %d, localCachedHead = %lu\n", threadIdx.x, localCachedHead);
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
        do {
            tail = acquireTail();
        } while(localCachedHead >= tail + PACKET_DEPTH);
        // depth = 2, head = 1, tail = 0 , ok
        // depth = 2, head = 2, tail = 0, should wait

        //printf("Acquire new send step, threadIdx.x = %d, localCachedHead = %lu, tail = %lu\n", threadIdx.x, localCachedHead, tail);
        return localCachedHead % PACKET_DEPTH;
    }

    __forceinline__ __device__ int acquireNewRecvStep()
    {
        int64_t head = 0;
        do {
            head = acquireHead();
        } while(localCachedTail >= head);

        //localCachedTail += 1;
        //printf("Acquire new recv step, threadIdx.x = %d, localCachedTail = %lu\n", threadIdx.x, localCachedTail);
        return localCachedTail % PACKET_DEPTH;
    }   

protected:
    MoeCommFifoConnInfo* fifoConnInfo;
    uint64_t localCachedHead;
    uint64_t localCachedTail;
};

struct IndexInfo
{
    int32_t unitIdBase;
    int32_t cachedUnitId;
    int32_t cachedFirstValidThread;

    __device__ __inline__ IndexInfo()
    : unitIdBase(0),
    cachedUnitId(0),
    cachedFirstValidThread(0)
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
    static_assert(UNIT_COUNT_PER_PACKET % MAX_UNIT_PER_ITERATION == 0, "UNIT_COUNT_PER_PACKET must be divisible by MAX_UNIT_PER_ITERATION");
    
    __device__ __inline__ PipelineBase(
        uint8_t* bufferBase, 
        STEP_COMMUNICATOR_TYPE* stepCommunicator,
        typename cub::BlockScan<int, THREADS_PER_PIPELINE>::TempStorage& tempStorage,
        int* shared_new_step)
    : bufferBase(bufferBase), 
    stepCommunicator(stepCommunicator), 
    tempStorage(tempStorage),
    pipelineTile(cg::tiled_partition<THREADS_PER_PIPELINE>(cg::this_thread_block())),
    unitTile(cg::tiled_partition<THREADS_PER_UNIT>(pipelineTile)),
    shared_new_step(shared_new_step),
    packetPtr(bufferBase),
    packetCounterPtr(bufferBase + UNIT_BYTES_SIZE * UNIT_COUNT_PER_PACKET),
    allocatedUnit(0),
    unitIndex(pipelineTile.thread_rank() / THREADS_PER_UNIT),
    recivedUnit(0),
    totalRecivedUnit(0)
    {
        if (!IS_SENDER) {
            acquireNewPacket();
        }
    }

    __device__ __inline__ void finishPacket()
    {
        if (pipelineTile.thread_rank() == 0) {
            if (IS_SENDER) {
                //*(packetCounterPtr) = allocatedUnit;
                int32_t* packetCounterI32Ptr = (int32_t*)(packetCounterPtr);
                *packetCounterI32Ptr = allocatedUnit;
                assert(allocatedUnit > 0);
                int rankOffset = threadIdx.x / (THREADS_PER_UNIT * UNIT_PER_PIPELINE);
                printf("Sender finish packet, allocatedUnit = %d, rankOffset = %d, packetCounterI32Ptr = %p\n", allocatedUnit, rankOffset, packetCounterI32Ptr);
                stepCommunicator->releaseSendStep();
            } else {
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
                //printf("Acquire new send step, threadIdx.x = %d, step = %d\n", threadIdx.x, step);
            } else {
                step = stepCommunicator->acquireNewRecvStep();
                //printf("Acquire new recv step, threadIdx.x = %d, step = %d\n", threadIdx.x, step);
            }

            *(shared_new_step) = step;
        }

        pipelineTile.sync();
        step = *(shared_new_step);
        packetPtr = bufferBase + step * PACKET_SIZE;
        allocatedUnit = 0;
        packetCounterPtr = packetPtr + UNIT_BYTES_SIZE * UNIT_COUNT_PER_PACKET;
        pipelineTile.sync();

        if (!IS_SENDER) {
            recivedUnit = 0;
            packetCounterPtr = packetPtr + UNIT_BYTES_SIZE * UNIT_COUNT_PER_PACKET;
            allocatedUnit = *((int32_t*)packetCounterPtr);
            if (pipelineTile.thread_rank() == 0) {
                int rankOffset = threadIdx.x / (THREADS_PER_UNIT * UNIT_PER_PIPELINE);
                printf("Recvicer Acquire new packet, rankOffset = %d, allocatedUnit = %d, packetCounterPtr = %p\n", rankOffset, allocatedUnit, packetCounterPtr);
            }
        } else {
            if (pipelineTile.thread_rank() == 0) {
                int rankOffset = threadIdx.x / (THREADS_PER_UNIT * UNIT_PER_PIPELINE);
                printf("Sender Acquire new packet, rankOffset = %d, step = %d, allocatedUnit = %d, packetCounterPtr = %p\n", rankOffset, step, allocatedUnit, packetCounterPtr);
            }
        }
    }

    __device__ __inline__ uint8_t* conditionalAcquireSendBuffer(bool needBuffer = true)
    {
        bool unitNeedAlloc = unitTile.any(needBuffer);
        int isFirstThreadOnValidUnit = unitTile.thread_rank() == 0 && unitNeedAlloc ? 1 : 0;
        
        int allocUnit = cg::reduce(pipelineTile, isFirstThreadOnValidUnit, cg::plus<int>());

        if (allocUnit == 0) {
            return nullptr;
        }

        bool packetNeedAlloc = allocUnit + allocatedUnit > UNIT_COUNT_PER_PACKET;
        if (packetNeedAlloc) 
        {
            finishPacket();
            acquireNewPacket();
        }

        unitIndex = 0;
        typedef cub::BlockScan<int, THREADS_PER_PIPELINE> BlockScan;
        BlockScan(tempStorage).InclusiveSum(isFirstThreadOnValidUnit, unitIndex);
        unitIndex -= 1;

        uint8_t* retPtr = nullptr;
        if (needBuffer) {
            retPtr = packetPtr + (unitIndex + allocatedUnit) * UNIT_BYTES_SIZE + unitTile.thread_rank() * BYTES_PER_THREAD;
            //uint64_t addr = (uint64_t)retPtr;
            //assert(addr % 8 == 0);
        }


        allocatedUnit += allocUnit;
        indexInfo.unitIdBase += allocUnit;
        if (unitNeedAlloc) {
            indexInfo.cachedUnitId = unitIndex;
            int mask = unitTile.ballot(needBuffer);
            indexInfo.cachedFirstValidThread = __ffs(mask) - 1;
        }

        return retPtr;
    }

    __device__ __forceinline__ const bool recvEnd(int32_t expectedTotalRecivedUnit)
    {
        return totalRecivedUnit >= expectedTotalRecivedUnit;
    }

    __device__ __inline__ uint8_t* conditionalAcquireRecvBuffer()
    {
        if (recivedUnit >= allocatedUnit)
        {
            finishPacket();
            acquireNewPacket();
            //printf("Acquire new recv packet, threadIdx.x = %d, allocatedUnit = %d\n", threadIdx.x, allocatedUnit);
        }


        uint8_t* retPtr = nullptr;  
        if (recivedUnit + MAX_UNIT_PER_ITERATION <= allocatedUnit) {
            // fully recived
            retPtr = packetPtr + (recivedUnit + unitIndex) * UNIT_BYTES_SIZE + unitTile.thread_rank() * BYTES_PER_THREAD;
            totalRecivedUnit += MAX_UNIT_PER_ITERATION;
            recivedUnit += MAX_UNIT_PER_ITERATION;
        } else {
            if (recivedUnit + unitIndex < allocatedUnit) {
                retPtr = packetPtr + (recivedUnit + unitIndex) * UNIT_BYTES_SIZE + unitTile.thread_rank() * BYTES_PER_THREAD;
            } 
            totalRecivedUnit += (allocatedUnit - recivedUnit);
            recivedUnit = allocatedUnit;
        }

        return retPtr;
    }

    __device__ __inline__ uint8_t* acquireSendBuffer()
    {
        if (allocatedUnit + MAX_UNIT_PER_ITERATION > UNIT_COUNT_PER_PACKET) {
            finishPacket();
            acquireNewPacket();
        }

        uint8_t*  retPtr = packetPtr + (allocatedUnit + unitIndex) * UNIT_BYTES_SIZE + unitTile.thread_rank() * BYTES_PER_THREAD;
        
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
            //printf("Acquire new recv packet, threadIdx.x = %d, allocatedUnit = %d\n", threadIdx.x, allocatedUnit);
        }

        uint8_t* retPtr = nullptr;  
        if (recivedUnit + unitIndex < allocatedUnit) {
            retPtr = packetPtr + (recivedUnit + unitIndex) * UNIT_BYTES_SIZE + unitTile.thread_rank() * BYTES_PER_THREAD;
        } else {
            if (pipelineTile.thread_rank() == 0) {
                //printf("Acquire recv buffer faild, recivedUnit=%d, allocatedUnit=%d\n", recivedUnit, allocatedUnit);
            }
        }
        //if (threadIdx.x == 1) {
        //printf("Acquire recv buffer, threadIdx.x = %d, bufferBase = %p, packetPtr = %p, retPtr = %p, allocatedUnit = %d, MAX_UNIT_PER_ITERATION = %d, unitIndex = %d\n", threadIdx.x, bufferBase, packetPtr, retPtr, allocatedUnit, MAX_UNIT_PER_ITERATION, unitIndex);
        //}

        recivedUnit += MAX_UNIT_PER_ITERATION;

        return retPtr;
    }

    __device__ __forceinline__ IndexInfo& getIndexInfo()
    {
        return indexInfo;
    }

    __device__ __forceinline__ int32_t getTotalRecivedUnit()
    {
        return totalRecivedUnit;
    }
    
protected:
    uint8_t* bufferBase;
    STEP_COMMUNICATOR_TYPE* stepCommunicator;
    typename cub::BlockScan<int, THREADS_PER_PIPELINE>::TempStorage& tempStorage;
    int* shared_new_step;
    uint8_t* packetPtr;
    uint8_t* packetCounterPtr;
    int32_t allocatedUnit;
    int32_t recivedUnit;
    int32_t totalRecivedUnit;
    // used for indices calculation
    IndexInfo indexInfo;
    int32_t unitIndex;
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
        fifoConnInfo->tail = value;
        st_release_sys_global(&(fifoConnInfo->head), uint64_t(1));
    }

    __forceinline__ __device__ uint64_t acquireValue()
    {
        uint64_t head = 0;
        do {
            head = ld_acquire_sys_global(&(fifoConnInfo->head));
        } while(head == 0);

        uint64_t value = fifoConnInfo->tail;
        return value;
    }

protected:
    MoeCommFifoConnInfo* fifoConnInfo;
};


__global__ void rankCountDevice(uint32_t* experts, int* sendCountsCumsum, int* recvCountsCumsum, MoeCommWorkspace workspace, int tokenCount, int topK, int expert_count, int rank_id, int rank_count, int* sendCountReady)
{
    bool isSender = (blockIdx.x < gridDim.x - 1);
    using BlockScan = cub::BlockScan<int, THREADS_PER_UNIT * UNIT_PER_PIPELINE * PIPELINE_PER_CTA>;
    __shared__ typename BlockScan::TempStorage tempStorage;
    __shared__ int sharedCountsPerRank[PIPELINE_PER_CTA];
    int expertCountPerRank = (expert_count + rank_count - 1) / rank_count;

    if (isSender) {
        int laneID = threadIdx.x % THREADS_PER_UNIT;
        int unitId = ((threadIdx.x - laneID) / THREADS_PER_UNIT) % UNIT_PER_PIPELINE;
        int rankOffset = threadIdx.x / (THREADS_PER_UNIT * UNIT_PER_PIPELINE);
        int target_rankId = blockIdx.x  * PIPELINE_PER_CTA + rankOffset;
        cg::thread_block_tile<THREADS_PER_PIPELINE> rankTile = cg::tiled_partition<THREADS_PER_PIPELINE>(cg::this_thread_block());
        cg::thread_block_tile<THREADS_PER_UNIT> unitTile = cg::tiled_partition<THREADS_PER_UNIT>(rankTile);
        CounterCommunicator counter(workspace.getFifoConnInfo(true, rank_id, target_rankId, 0, rank_count, 1));

        int* sharedCountsThisRank = sharedCountsPerRank + rankOffset;
        if (rankTile.thread_rank() == 0) {
            *sharedCountsThisRank = 0;
        }

        for (int token_id = unitId; token_id < tokenCount; token_id += UNIT_PER_PIPELINE) {
            bool is_valid = false;
            if (laneID < topK)
            {
                int32_t expert_id = *(experts + token_id * topK + laneID);
                //int32_t rank_id = expert_id / expert_count_per_rank;
                is_valid |= ((expert_id / expertCountPerRank) == target_rankId);

                //if (blockIdx.x == 0 && threadIdx.x == 0) {
                //    printf("token_id = %d, expert_id = %d, rank_id=%d\n", token_id, expert_id, rank_id);
                //}
            }
            bool token_is_valid = unitTile.any(is_valid);
            if (unitTile.thread_rank() == 0 && token_is_valid) {
                atomicAdd_block(sharedCountsThisRank, 1);
                if (blockIdx.x == 0 && threadIdx.x == 0) {
                    printf("Rank %d, token_id = %d\n", target_rankId, token_id);
                }
            }
        }
        rankTile.sync();

        // send the count of this rank
        if (rankTile.thread_rank() == 0 && target_rankId < rank_count) {
            int countThisRank = *(sharedCountsThisRank);
            counter.releaseValue(uint64_t(countThisRank));
        }

        __syncthreads();

        if (threadIdx.x < PIPELINE_PER_CTA) {
            int rank = PIPELINE_PER_CTA * blockIdx.x + threadIdx.x;
            if (rank < rank_count) {
                *(sendCountsCumsum + rank) = sharedCountsThisRank[threadIdx.x]; 
            }
        }

        if (threadIdx.x == 0) {
            int rank_this_cta = blockIdx.x < gridDim.x - 2 ? PIPELINE_PER_CTA : rank_count - PIPELINE_PER_CTA * (gridDim.x - 2);
            atomicAdd_system(sendCountReady, rank_this_cta);

            //printf("Send count ready = %d, blockIdx.x = %d, rank_this_cta = %d, gridDim.x = %d\n", *sendCountReady, blockIdx.x, rank_this_cta, gridDim.x);
        }

        // block 0 compute send cumsum
        if (blockIdx.x == 0) {
            if (threadIdx.x == 0) {
                do {} while (*sendCountReady < rank_count);
            }

            __syncthreads();
            int value = 0;
            if (threadIdx.x < rank_count) {
                value = int(*(sendCountsCumsum + threadIdx.x));
            }

            __syncthreads();

            int cumsum;
            BlockScan(tempStorage).InclusiveSum(value, cumsum);
            if (threadIdx.x < rank_count) {
                *(sendCountsCumsum + threadIdx.x) = uint32_t(cumsum);
                //printf("Rank %d, send cumsum = %d, value = %d\n", threadIdx.x, cumsum, value);
            }
        }
    }
    else // receiver
    {
        // assert(gridDim.x >= 1);
        int value = 0;
        CounterCommunicator counter(workspace.getFifoConnInfo(false, rank_id, threadIdx.x, 0, rank_count, 1));
        if (threadIdx.x < rank_count) {
            value = int(counter.acquireValue());
        }

        __syncthreads();

        int cumsum;
        BlockScan(tempStorage).ExclusiveSum(value, cumsum);
        if (threadIdx.x < rank_count) {
            *(recvCountsCumsum + threadIdx.x) = cumsum;
            printf("Rank %d, recv cumsum = %d\n", threadIdx.x, cumsum);
            //printf("Revc CTA end: %d, threadIdx.x = %d, \n", blockIdx.x, threadIdx.x);
        }

        //printf("Revc CTA end: %d, threadIdx.x = %d\n", blockIdx.x, threadIdx.x);
    }
}

__device__ __forceinline__ uint64_t packExpertAndScale(uint32_t expertId, float scale)
{
    return (uint64_t(expertId) << 32) | *reinterpret_cast<uint32_t*>(&scale);
}

__device__ __forceinline__ void unPackExpertAndScale(uint64_t value, uint32_t& expertId, float& scale)
{
    expertId = uint32_t(value >> 32);
    scale = *reinterpret_cast<float*>(&value);
}


template <typename STEP_COMMUNICATOR_TYPE>
__global__ void generateIndiceAndLocalDataDevice(uint32_t* experts, float* scales, uint32_t* localExperts, float* localScales, int* sendCountsCumsum, int* recvCountsCumsum, 
    int* sendIndice, int* backwardIndice, int* recvIndice, MoeCommWorkspace workspace, int tokenCount, int topK, int expertCount, int rank, int rankCount)
{
    bool isSender = (blockIdx.y == 0);
    int laneId = threadIdx.x % THREADS_PER_UNIT;
    int unitId = ((threadIdx.x - laneId) / THREADS_PER_UNIT) % UNIT_PER_PIPELINE;
    int rankOffset = threadIdx.x / (THREADS_PER_UNIT * UNIT_PER_PIPELINE);
    int targetRankId = blockIdx.x  * PIPELINE_PER_CTA + rankOffset;
    int expertCountPerRank = expertCount / rankCount;
    //static constexpr int BYTES_PER_THREAD = UNIT_BYTES_SIZE / THREADS_PER_UNIT;
    //static constexpr int BUFFER_SIZE_PER_PIPELINE = PACKET_DEPTH * UNIT_BYTES_SIZE * UNIT_COUNT_PER_PACKET + STEP_COMMUNICATOR_TYPE::META_SIZE;

    __shared__ typename cub::BlockScan<int, THREADS_PER_PIPELINE>::TempStorage tempStorage;
    __shared__ int sharedNewStep[PIPELINE_PER_CTA];
#if DEBUG_PIPELINE
    //Used for debug
    __shared__ int tokenCounter[PIPELINE_PER_CTA];
#endif

    if (targetRankId >= rankCount) {
        return;
    }

    if (isSender) {
        int cumsum = targetRankId == 0 ? 0 : *(sendCountsCumsum + targetRankId - 1);
#if DEBUG_PIPELINE
        uint8_t* bufferBase = (uint8_t*)(workspace.getFifoBasePtrDebug(true, rank, targetRankId, 0, 1));
        STEP_COMMUNICATOR_TYPE stepCommunicator(workspace.getFifoConnInfoDebug(true, rank, targetRankId, 0, rankCount, 1));
#else
        uint8_t* bufferBase = (uint8_t*)(workspace.getFifoBasePtr(true, rank, targetRankId, 0, 1));
        STEP_COMMUNICATOR_TYPE stepCommunicator(workspace.getFifoConnInfo(true, rank, targetRankId, 0, rankCount, 1));
#endif
        PipelineBase<STEP_COMMUNICATOR_TYPE, true> pipeline(bufferBase, &stepCommunicator, tempStorage, &(sharedNewStep[rankOffset]));

        int alignTokenCount = (tokenCount + UNIT_PER_PIPELINE - 1) / UNIT_PER_PIPELINE * UNIT_PER_PIPELINE;

        for (int tokenId = unitId; tokenId < alignTokenCount; tokenId += UNIT_PER_PIPELINE) {
            uint8_t* ptr;
            int rankId = -1;
            int expertId;
            if (tokenId < tokenCount && laneId < topK)
            {
                int offset = tokenId * topK + laneId;
                int* source_ptr = (int*)(experts + offset);
                expertId = *source_ptr;
                int32_t rank_id = expertId / expertCountPerRank;
                //rankId = expertId % rankCount;
                ptr = pipeline.conditionalAcquireSendBuffer(rankId == targetRankId);
            } else {
               ptr = pipeline.conditionalAcquireSendBuffer(false); 
            }

            //printf("11111 ptr = %p, threadIdx.x = %d\n", ptr, threadIdx.x);
            //return;

            if (ptr) {
                //printf("222222 ptr = %p, threadIdx.x = %d\n", ptr, threadIdx.x);
                float scale = *(scales + tokenId * topK + laneId);
                uint64_t value;
                if (rankId == targetRankId) {
                    value = packExpertAndScale(expertId, scale);
                } else {
                    value = packExpertAndScale(expertCount, 0.0f);
                }
                *((uint64_t*)ptr) = value;

                if (laneId == 0) {
                    int sendTokenId = pipeline.getIndexInfo().unitIdBase + pipeline.getIndexInfo().cachedUnitId;
                    int sendUnitId = pipeline.getIndexInfo().cachedFirstValidThread;
                    *(sendIndice + cumsum + sendTokenId) = tokenId;
                    *(backwardIndice + cumsum + sendTokenId) = tokenId * topK + sendUnitId;
                }
            }
        }
        pipeline.finishPacket();
    } else { // recv
        int cumsum = targetRankId == 0 ? 0 : *(recvCountsCumsum + targetRankId - 1);
        int recvTokenCount = *(recvCountsCumsum + targetRankId) - cumsum;
#if DEBUG_PIPELINE
        uint8_t* bufferBase = (uint8_t*)(workspace.getFifoBasePtrDebug(false, rank, targetRankId, 0, 1));
        STEP_COMMUNICATOR_TYPE stepCommunicator(workspace.getFifoConnInfoDebug(false, rank, targetRankId, 0, rankCount, 1));
#else
        uint8_t* bufferBase = (uint8_t*)(workspace.getFifoBasePtr(false, rank, targetRankId, 0, 1));
        STEP_COMMUNICATOR_TYPE stepCommunicator(workspace.getFifoConnInfo(false, rank, targetRankId, 0, rankCount, 1));
#endif
        PipelineBase<STEP_COMMUNICATOR_TYPE, false> pipeline(bufferBase, &stepCommunicator, tempStorage, &(sharedNewStep[rankOffset]));

        //int alignRecvTokenCount = (recvTokenCount + UNIT_PER_PIPELINE - 1) / UNIT_PER_PIPELINE * UNIT_PER_PIPELINE;
        //for (int tokenId = unitId; tokenId < alignRecvTokenCount; tokenId += UNIT_PER_PIPELINE) {
        int tokenIdBase = 0;
        while (!pipeline.recvEnd(recvTokenCount)) {
            uint8_t* ptr = pipeline.conditionalAcquireRecvBuffer();
            if (ptr) {
                //assert(tokenId < recvTokenCount);
                int64_t value = *((int64_t*)ptr);
                uint32_t expertId;
                float scale;
                unPackExpertAndScale(value, expertId, scale);

                int tokenId = tokenIdBase + unitId;
#if DEBUG_PIPELINE
                assert(tokenId < recvTokenCount);
#endif
                tokenIdBase = pipeline.getTotalRecivedUnit();

                if (laneId < topK) {
                    *(localExperts + (cumsum + tokenId) * topK + laneId) = expertId;
                    *(localScales + (cumsum + tokenId) * topK + laneId) = scale;
                }
                if (laneId == 0) {
                    *(recvIndice + cumsum + tokenId) = cumsum + tokenId;
#if DEBUG_PIPELINE
                    atomicAdd_block(&(tokenCounter[rankOffset]), 1);
#endif
                }
            }
        }
#if DEBUG_PIPELINE
        if (laneId == 0 && unitId == 0) {
            printf("Recv end: %d, recvTokenCount = %d, targetRankId = %d, recvCounter = %d\n", blockIdx.x, recvTokenCount, targetRankId, tokenCounter[rankOffset]);
        }
#endif
    }
}

void rankCount(uint32_t* experts, int* sendCountsCumsum, int* recvCountsCumsum, MoeCommWorkspace workspace, int tokenCount, int topK, int expert_count, int rank_id, int rankCount, int* sendCountReady, cudaStream_t stream)
{
    // One for recv, others for send.
    int cta_count = (rankCount + PIPELINE_PER_CTA - 1) / PIPELINE_PER_CTA + 1;
    int block_size = THREADS_PER_CTA;
    dim3 block(block_size);
    dim3 grid(cta_count);
    rankCountDevice<<<grid, block, 0, stream>>>(experts, sendCountsCumsum, recvCountsCumsum, workspace, tokenCount, topK, expert_count, rank_id, rankCount, sendCountReady);
}

void generateIndiceAndLocalData(uint32_t* experts, float* scales, uint32_t* localExperts, float* localScales, int* sendCountsCumsum, int* recvCountsCumsum, 
    int* sendIndice, int* backwardIndice, int* recvIndice, MoeCommWorkspace workspace, int tokenCount, int topK, int expertCount, int rank, int rankCount, cudaStream_t stream)
{
    int grid_x = (rankCount + PIPELINE_PER_CTA - 1) / PIPELINE_PER_CTA + 1;
    int block_size = THREADS_PER_CTA;
    dim3 block(block_size);
    dim3 grid(grid_x, 2);
    generateIndiceAndLocalDataDevice<StepCommunicatorBase><<<grid, block, 0, stream>>>(experts, scales, localExperts, localScales, sendCountsCumsum, recvCountsCumsum, sendIndice, backwardIndice, recvIndice, workspace, tokenCount, topK, expertCount, rank, rankCount);
}

} // namespace moe_prepare

} // namespace tensorrt_llm::kernels