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

#include "moeCommKernels.h"

#include <stdio.h>

#include <cooperative_groups.h>
#include <cub/cub.cuh>

namespace cg = cooperative_groups;

namespace tensorrt_llm::kernels
{

__device__ inline void barrier_sync(int name, int nThreads)
{
    asm volatile("barrier.sync.aligned %0, %1;" ::"r"(name), "r"(nThreads) : "memory");
}

inline __device__ void load128(uint64_t const* ptr, uint64_t& v0, uint64_t& v1)
{
    asm volatile("ld.volatile.global.v2.u64 {%0,%1}, [%2];" : "=l"(v0), "=l"(v1) : "l"(ptr) : "memory");
}

inline __device__ void store128(uint64_t* ptr, uint64_t v0, uint64_t v1)
{
    asm volatile("st.volatile.global.v2.u64 [%2], {%0,%1};" ::"l"(v0), "l"(v1), "l"(ptr) : "memory");
}

template <bool isSender>
class AllToAllChannelCommunicator : public AllToAllChannelCommunicatorBase
{
private:
    int const tid;      // thread index in primitives group
    int const nthreads; // number of threads in primitives group
    int const wid;      // lane index in warp
    int const warp;     // warp index in primitives group
    const MoeEpWorldInfo worldInfo;
    const MoeCommWorkspace workspace;
    const SendRecvDataInfo sendRecvDataInfo;
    const SendRecvDispls dataDispls;
    int peerRank;      // peer rank index
    bool const flagThread;
    int const group;   // primitives group index
    int const channel; // channel index

    MoeCommFifoConnInfo* fifoConnInfoPtr;
    uint64_t* fifoBasePtr; // pointer to fifo base address
    uint64_t step;
    uint64_t tailStepCache;
    uint64_t regs[U64_DATA_REG_PER_THREAD];
    GroupSharedBuffer* groupSharedBuffer;

    int groupStartIndice;
    int groupEndIndice;

    int sliceStartIndice;
    int sliceEndIndice;

    uint64_t* stepFifoEntryPtr;

public:
    __inline__ __device__ uint64_t getFlag()
    {
        return step + 1;
    }

    __inline__ __device__ AllToAllChannelCommunicator(MoeEpWorldInfo const& worldInfo, MoeCommWorkspace workspace,
        SendRecvDataInfo sendRecvDataInfo, SendRecvDispls dataDispls, GroupSharedBuffer* groupSharedBuffer)
        : worldInfo(worldInfo)
        , nthreads(blockDim.x)
        , tid(threadIdx.x)
        , workspace(workspace)
        , sendRecvDataInfo(sendRecvDataInfo)
        , dataDispls(dataDispls)
        , wid(threadIdx.x % WARP_SIZE)
        , warp(threadIdx.x / WARP_SIZE)
        , peerRank(blockIdx.x * GROUP_COUNT_PER_BLOCK + threadIdx.y)
        , group(threadIdx.y)
        , channel(blockIdx.y)
        , flagThread(threadIdx.x % 8 == 7)
        , fifoConnInfoPtr(nullptr)
        , fifoBasePtr(nullptr)
        , step(0)
        , tailStepCache(0)
        , groupSharedBuffer(groupSharedBuffer)
    {
    }

    __inline__ __device__ void init()
    {
        fifoBasePtr = workspace.getFifoBasePtr(isSender, worldInfo.epRank, peerRank, channel);
        fifoConnInfoPtr = workspace.getFifoConnInfo(isSender, worldInfo.epRank, peerRank, channel, worldInfo.epSize);
        step = isSender ? fifoConnInfoPtr->head : fifoConnInfoPtr->tail;
        tailStepCache = isSender ? fifoConnInfoPtr->tail : 0;
    }

    __inline__ __device__ void computeGroupTransferRange()
    {
        if (tid == 0)
        {
            int rankCount = dataDispls.getCount(peerRank);
            int rankStart = dataDispls.getRankStart(peerRank);
            int countPerChannel = (rankCount + CHANNEL_COUNT - 1) / CHANNEL_COUNT;
            int groupEnd = min(rankStart + (channel + 1) * countPerChannel, rankStart + rankCount);
            int groupStart = min(rankStart + channel * countPerChannel, rankStart + rankCount);
            groupSharedBuffer->groupStartIndice = groupStart;
            groupSharedBuffer->groupEndIndice = groupEnd;
        }
        barrier();
        groupStartIndice = groupSharedBuffer->groupStartIndice;
        groupEndIndice = groupSharedBuffer->groupEndIndice;
    }

    __inline__ __device__ void loadTransferIndices()
    {
        sliceStartIndice = groupStartIndice;
        sliceEndIndice = min(groupStartIndice + sendRecvDataInfo.vectorCountPerFifoEntry, groupEndIndice);
        for (int i = groupStartIndice + tid; i < sliceEndIndice; i += WARP_SIZE * WARP_PER_GROUP)
        {
            groupSharedBuffer->groupIndiceBuffer[i - groupStartIndice] = dataDispls.getRealVectorIndice(i);
        }
        groupStartIndice = sliceEndIndice;
        barrier();
    }

    __inline__ __device__ void computeSlicePtr()
    {
        stepFifoEntryPtr = fifoBasePtr + RECV_FIFO_ENTRY_U64 * (step % RECV_FIFO_DEPTH);
    }

    __inline__ __device__ void sendSlice()
    {
        waitSend();
        int EltPer16B = 2;
        int eltN = sendRecvDataInfo.vectorSizeInU64;
        for (int vecId = warp + sliceStartIndice; vecId < sliceEndIndice; vecId += WARP_PER_GROUP)
        {
            int idxInSlice = vecId - sliceStartIndice;
            int vecRealIdx = groupSharedBuffer->groupIndiceBuffer[idxInSlice];
            uint64_t* src = dataDispls.getVectorDataPtr(vecRealIdx);
            uint64_t* slicePtr = stepFifoEntryPtr
                + idxInSlice * sendRecvDataInfo.dataPacketCountPerVector * PACKET_SIZE_IN_U64 + 2 * wid;
            for (int packetId = 0; packetId < sendRecvDataInfo.dataPacketCountPerVector; packetId++)
            {
                int vecOff = packetId * DATA_PAYLOAD_SIZE_PER_PACKET_IN_U64;
#pragma unroll
                for (int g = 0; g < U64_DATA_REG_PER_THREAD / 2; g++)
                {
                    int ix = g * WARP_SIZE - 4 * (g / 2) + wid - (g % 2) * (wid / 8);
                    __syncwarp();
                    if (!flagThread || g % 2 == 0)
                    {
                        if (ix * EltPer16B + vecOff < eltN)
                        {
                            load128((uint64_t*) (src + ix * EltPer16B + vecOff), regs[2 * g + 0], regs[2 * g + 1]);
                        }
                    }
                    __syncwarp();
                }
#pragma unroll
                for (int g = 1; g < U64_DATA_REG_PER_THREAD / 2; g += 2)
                {
                    if (flagThread)
                        regs[2 * g] = regs[2 * g - 1];
                }

                uint64_t flag = getFlag();
                uint64_t* packetPtr = slicePtr + packetId * PACKET_SIZE_IN_U64;
                __syncwarp();
#pragma unroll
                for (int u = 0; u < U64_DATA_REG_PER_THREAD; u += 2)
                {
                    store128(packetPtr + u * WARP_SIZE, regs[u], flagThread ? flag : regs[u + 1]);
                }
            }
        }
        updateSend();
    }

    __inline__ __device__ void recvSlice()
    {
        // receiver don't need to wait since we have flag.
        int EltPer16B = 2;
        int eltN = sendRecvDataInfo.vectorSizeInU64;
        for (int vecId = warp + sliceStartIndice; vecId < sliceEndIndice; vecId += WARP_PER_GROUP)
        {
            int idxInSlice = vecId - sliceStartIndice;
            int vecRealIdx = groupSharedBuffer->groupIndiceBuffer[idxInSlice];

            uint64_t* dst = dataDispls.getVectorDataPtr(vecRealIdx);
            uint64_t* slicePtr = stepFifoEntryPtr
                + idxInSlice * sendRecvDataInfo.dataPacketCountPerVector * PACKET_SIZE_IN_U64 + 2 * wid;
            for (int packetId = 0; packetId < sendRecvDataInfo.dataPacketCountPerVector; packetId++)
            {
                uint64_t* packetPtr = slicePtr + packetId * PACKET_SIZE_IN_U64;
                int vecOff = packetId * DATA_PAYLOAD_SIZE_PER_PACKET_IN_U64;

                bool needReload;
                uint64_t flag = getFlag();
                __syncwarp();
                do
                {
                    needReload = false;
#pragma unroll
                    for (int u = 0; u < U64_DATA_REG_PER_THREAD; u += 2)
                    {
                        load128(packetPtr + u * WARP_SIZE, regs[u], regs[u + 1]);
                        needReload |= flagThread && (regs[u + 1] != flag);
                    }
                } while (__any_sync(WARP_MASK, needReload));
#pragma unroll
                for (int g = 1; g < U64_DATA_REG_PER_THREAD / 2; g += 2)
                {
                    if (flagThread)
                        regs[2 * g - 1] = regs[2 * g];
                }

#pragma unroll
                for (int g = 0; g < U64_DATA_REG_PER_THREAD / 2; g++)
                {
                    int ix = g * WARP_SIZE - 4 * (g / 2) + wid - (g % 2) * (wid / 8);
                    __syncwarp();
                    if (!flagThread || g % 2 == 0)
                    {
                        if (ix * EltPer16B + vecOff < eltN)
                        {
                            store128((uint64_t*) (dst + ix * EltPer16B + vecOff), regs[2 * g + 0], regs[2 * g + 1]);
                        }
                    }
                    __syncwarp();
                }
            }
        }
        updateRecv();
    }

    __inline__ __device__ void run()
    {
        if (peerRank >= worldInfo.epSize)
        {
            return;
        }
        init();
        computeGroupTransferRange();
        while (groupStartIndice < groupEndIndice)
        {
            loadTransferIndices();
            computeSlicePtr();
            if (isSender)
            {
                sendSlice();
            }
            else
            {
                recvSlice();
            }
        }
    }

    __inline__ __device__ ~AllToAllChannelCommunicator() {}

    __inline__ __device__ void barrier()
    {
        barrier_sync(15 - group, nthreads);
    }

    __inline__ __device__ void waitSend()
    {
        barrier();
        while (tailStepCache + RECV_FIFO_DEPTH < step + 1)
        {
            tailStepCache = fifoConnInfoPtr->tail;
        }
        barrier();
    }

    __inline__ __device__ void updateSend()
    {
        barrier();
        if (tid == 0)
        {
            atomicAdd_system((unsigned long long*) &fifoConnInfoPtr->head, 1);
        }
        barrier();
        step++;
    }

    __inline__ __device__ void updateRecv()
    {
        barrier();
        if (tid == 0)
        {
            atomicAdd_system((unsigned long long*) &fifoConnInfoPtr->tail, 1);
        }
        barrier();
        step++;
    }
};

__global__ void moeAllToAllKernel(MoeEpWorldInfo worldInfo, MoeCommWorkspace workspace,
    SendRecvDataInfo sendRecvDataInfo, SendRecvDispls sendDispls, SendRecvDispls recvDispls)
{
    __shared__ AllToAllChannelCommunicatorBase::GroupSharedBuffer
        allGroupSharedBuffer[AllToAllChannelCommunicatorBase::GROUP_COUNT_PER_BLOCK];
    bool isSender = blockIdx.z == 0;
    int group = threadIdx.y;
    SendRecvDispls dataDispls = isSender ? sendDispls : recvDispls;
    AllToAllChannelCommunicatorBase::GroupSharedBuffer* groupSharedBuffer = &allGroupSharedBuffer[group];
    if (isSender)
    {
        AllToAllChannelCommunicator<true> comm(worldInfo, workspace, sendRecvDataInfo, dataDispls, groupSharedBuffer);
        comm.run();
    }
    else
    {
        AllToAllChannelCommunicator<false> comm(worldInfo, workspace, sendRecvDataInfo, dataDispls, groupSharedBuffer);
        comm.run();
    }
}

void moeAllToAll(MoeEpWorldInfo worldInfo, SendRecvDataInfo sendRecvDataInfo, SendRecvDispls sendDispls,
    SendRecvDispls recvDispls, MoeCommWorkspace workspace, cudaStream_t stream)
{
    sendRecvDataInfo.DoPreCompute();
    TLLM_CHECK_WITH_INFO(
        reinterpret_cast<uintptr_t>(sendDispls.dataPtr) % 16 == 0, "sendDispls.dataPtr must be 16-byte aligned");
    TLLM_CHECK_WITH_INFO(
        reinterpret_cast<uintptr_t>(recvDispls.dataPtr) % 16 == 0, "recvDispls.dataPtr must be 16-byte aligned");
    dim3 block = AllToAllChannelCommunicatorBase::getLaunchBlockDim();
    dim3 grid = AllToAllChannelCommunicatorBase::getLaunchGridDim(worldInfo.epSize);
    moeAllToAllKernel<<<grid, block, 0, stream>>>(worldInfo, workspace, sendRecvDataInfo, sendDispls, recvDispls);
}

template <bool isSend, int kThreadsGroupSize>
__inline__ __device__ void computeSendRecvRankCountDevice(MoeEpWorldInfo worldInfo,
    MoeExpertParallelInfo expertParallelInfo, int maxTokenCountPerRank, int const* realRankTokenCountCumSum,
    int const* gatheredTargetRankIds, int* sharedSendRecvRankCount, int* sendRecvRankCount)
{
    cg::thread_block_tile<kThreadsGroupSize> tile = cg::tiled_partition<kThreadsGroupSize>(cg::this_thread_block());
    int laneInTile = tile.thread_rank();
    int tileId = threadIdx.x / kThreadsGroupSize;
    int tileCountPerBlock = blockDim.x / kThreadsGroupSize;

    int topK = expertParallelInfo.topK;
    int epRank = worldInfo.epRank;
    int epSize = worldInfo.epSize;

    if (threadIdx.x == 0)
    {
        *sharedSendRecvRankCount = 0;
    }

    __syncthreads();
    int readRank = isSend ? epRank : blockIdx.x;
    int compareRankId = isSend ? blockIdx.x : epRank;
    int const* readRankTargetRankIds = gatheredTargetRankIds + readRank * maxTokenCountPerRank * topK;
    int readRankTokenCount = maxTokenCountPerRank;
    if (realRankTokenCountCumSum != nullptr)
    {
        int readRankStart = readRank == 0 ? 0 : realRankTokenCountCumSum[readRank - 1];
        readRankTargetRankIds = gatheredTargetRankIds + readRankStart * topK;
        readRankTokenCount = realRankTokenCountCumSum[readRank] - readRankStart;
    }

    for (int i = tileId + blockIdx.z * tileCountPerBlock; i < readRankTokenCount; i += tileCountPerBlock * gridDim.z)
    {
        int targetRankId = laneInTile < topK ? readRankTargetRankIds[i * topK + laneInTile] : epSize;
        bool rankMatched = (targetRankId == compareRankId);
        bool hasRankMatched = tile.any(rankMatched);
        if (hasRankMatched && laneInTile == 0)
        {
            atomicAdd_block(sharedSendRecvRankCount, 1);
        }
        tile.sync();
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        atomicAdd_system(sendRecvRankCount + blockIdx.x, *sharedSendRecvRankCount);
    }
}

template <int kThreadsGroupSize>
__global__ void computeSendRecvRankCountKernel(MoeEpWorldInfo worldInfo, MoeExpertParallelInfo expertParallelInfo,
    int maxTokenCountPerRank, int const* realRankTokenCountCumSum, int const* gatheredTargetRankIds, int* sendRankCount,
    int* recvRankCount)
{
    static_assert(kThreadsGroupSize == 1 || kThreadsGroupSize == 2 || kThreadsGroupSize == 4 || kThreadsGroupSize == 8
            || kThreadsGroupSize == 16 || kThreadsGroupSize == 32,
        "Only 1, 2, 4, 8, 16, 32 threads group size supported now.");
    __shared__ int sharedSendRecvRankCount;
    if (blockIdx.y == 0)
    {
        // compute send rank count
        computeSendRecvRankCountDevice<true, kThreadsGroupSize>(worldInfo, expertParallelInfo, maxTokenCountPerRank,
            realRankTokenCountCumSum, gatheredTargetRankIds, &sharedSendRecvRankCount, sendRankCount);
    }
    else
    {
        // compute recv rank count
        computeSendRecvRankCountDevice<false, kThreadsGroupSize>(worldInfo, expertParallelInfo, maxTokenCountPerRank,
            realRankTokenCountCumSum, gatheredTargetRankIds, &sharedSendRecvRankCount, recvRankCount);
    }
}

void computeSendRecvRankCount(MoeEpWorldInfo worldInfo, MoeExpertParallelInfo expertParallelInfo,
    int maxTokenCountPerRank, int const* realRankTokenCountCumSum, int const* gatheredTargetRankIds, int* sendRankCount,
    int* recvRankCount, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(expertParallelInfo.topK <= 32, "Only topK less than or equal to 32 supported now.");
    int threadsPerBlock = 1024;
    auto* kernelPtr = computeSendRecvRankCountKernel<32>;
    if (expertParallelInfo.topK <= 1)
    {
        kernelPtr = computeSendRecvRankCountKernel<1>;
    }
    else if (expertParallelInfo.topK <= 2)
    {
        kernelPtr = computeSendRecvRankCountKernel<2>;
    }
    else if (expertParallelInfo.topK <= 4)
    {
        kernelPtr = computeSendRecvRankCountKernel<4>;
    }
    else if (expertParallelInfo.topK <= 8)
    {
        kernelPtr = computeSendRecvRankCountKernel<8>;
    }
    else if (expertParallelInfo.topK <= 16)
    {
        kernelPtr = computeSendRecvRankCountKernel<16>;
    }
    dim3 block(worldInfo.epSize, 2, 1);
    kernelPtr<<<block, threadsPerBlock, 0, stream>>>(worldInfo, expertParallelInfo, maxTokenCountPerRank,
        realRankTokenCountCumSum, gatheredTargetRankIds, sendRankCount, recvRankCount);
}

template <int kThreadsPerBlock>
__global__ void inplaceSendRecvRankCumSumKernel(MoeEpWorldInfo worldInfo, int* sendRankCount, int* recvRankCount)
{
    int* inputOutputPtr = blockIdx.x == 0 ? sendRankCount : recvRankCount;
    typedef cub::BlockScan<int, kThreadsPerBlock> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;

    int tid = threadIdx.x;
    int threadData = tid < worldInfo.epSize ? inputOutputPtr[tid] : 0;

    BlockScan(temp_storage).InclusiveSum(threadData, threadData);
    if (tid < worldInfo.epSize)
    {
        inputOutputPtr[tid] = threadData;
    }
}

void inplaceSendRecvRankCumSum(MoeEpWorldInfo worldInfo, int* sendRankCount, int* recvRankCount, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(worldInfo.epSize <= 1024, "Only worldInfo.epSize less than or equal to 1024 supported now.");
    auto* kernelPtr = inplaceSendRecvRankCumSumKernel<1024>;
    int blockSize = 1024;
    if (worldInfo.epSize <= 32)
    {
        kernelPtr = inplaceSendRecvRankCumSumKernel<32>;
        blockSize = 32;
    }
    else if (worldInfo.epSize <= 64)
    {
        kernelPtr = inplaceSendRecvRankCumSumKernel<64>;
        blockSize = 64;
    }
    else if (worldInfo.epSize <= 128)
    {
        kernelPtr = inplaceSendRecvRankCumSumKernel<128>;
        blockSize = 128;
    }
    else if (worldInfo.epSize <= 256)
    {
        kernelPtr = inplaceSendRecvRankCumSumKernel<256>;
        blockSize = 256;
    }
    else if (worldInfo.epSize <= 512)
    {
        kernelPtr = inplaceSendRecvRankCumSumKernel<512>;
        blockSize = 512;
    }
    kernelPtr<<<2, blockSize, 0, stream>>>(worldInfo, sendRankCount, recvRankCount);
}

template <bool isSend, int kThreadsGroupSize, int kThreadsPerBlock>
__inline__ __device__ void computeSendRecvIndicesDevice(MoeEpWorldInfo worldInfo,
    MoeExpertParallelInfo expertParallelInfo, int maxTokenCountPerRank, int const* realRankTokenCountCumSum,
    int const* gatheredTargetRankIds, int const* sendRecvCumSum,
    int* sendRecvIndices,              // send or receive
    int* localGatherIndices,           // receive only
    int* backwardRecvRankLocalIndices, // send only
    int* sharedSendRecvRankStart, typename cub::BlockScan<int, kThreadsPerBlock>::TempStorage& tempStorage)
{
    cg::thread_block_tile<kThreadsGroupSize> tile = cg::tiled_partition<kThreadsGroupSize>(cg::this_thread_block());
    int laneInTile = tile.thread_rank();
    int tileId = threadIdx.x / kThreadsGroupSize;
    int tileCountPerBlock = blockDim.x / kThreadsGroupSize;

    int topK = expertParallelInfo.topK;
    int epRank = worldInfo.epRank;
    int epSize = worldInfo.epSize;

    if (threadIdx.x == 0)
    {
        *sharedSendRecvRankStart = blockIdx.x == 0 ? 0 : sendRecvCumSum[blockIdx.x - 1];
    }

    __syncthreads();
    int readRank = isSend ? epRank : blockIdx.x;
    int compareRankId = isSend ? blockIdx.x : epRank;
    int readRankStart = readRank * maxTokenCountPerRank;
    int const* readRankTargetRankIds = gatheredTargetRankIds + readRankStart * topK;
    int readRankTokenCount = maxTokenCountPerRank;
    if (realRankTokenCountCumSum != nullptr)
    {
        readRankStart = readRank == 0 ? 0 : realRankTokenCountCumSum[readRank - 1];
        readRankTargetRankIds = gatheredTargetRankIds + readRankStart * topK;
        readRankTokenCount = realRankTokenCountCumSum[readRank] - readRankStart;
    }

    for (int blockStartId = blockIdx.z * tileCountPerBlock; blockStartId < readRankTokenCount;
         blockStartId += tileCountPerBlock * gridDim.z)
    {
        int stepStartIndice = *sharedSendRecvRankStart;
        int i = blockStartId + tileId;
        int targetRankId
            = (laneInTile < topK && i < readRankTokenCount) ? readRankTargetRankIds[i * topK + laneInTile] : epSize;
        bool rankMatched = (targetRankId == compareRankId);
        bool hasRankMatched = tile.any(rankMatched);
        unsigned int laneMask = tile.ballot(rankMatched);
        int lowestLane = __ffs(laneMask) - 1;
        int isMatchedLane = (hasRankMatched && laneInTile == lowestLane) ? 1 : 0;
        int indice;
        typedef cub::BlockScan<int, kThreadsPerBlock> BlockScan;
        BlockScan(tempStorage).ExclusiveSum(isMatchedLane, indice);
        indice += stepStartIndice;
        __syncthreads();

        if (isMatchedLane == 1)
        {
            atomicAdd_block(sharedSendRecvRankStart, 1);
            if (isSend)
            {
                sendRecvIndices[indice] = i;
                backwardRecvRankLocalIndices[indice] = i * topK + lowestLane;
            }
            else
            {
                sendRecvIndices[indice] = indice;
                localGatherIndices[indice] = readRankStart + i;
            }
        }
        __syncthreads();
    }
}

template <int kThreadsGroupSize, int kThreadsPerBlock>
__global__ void computeSendRecvIndicesKernel(MoeEpWorldInfo worldInfo, MoeExpertParallelInfo expertParallelInfo,
    int maxTokenCountPerRank, int const* realRankTokenCountCumSum, int const* gatheredTargetRankIds,
    int const* sendRankCountCumSum, int const* recvRankCountCumSum, int* localGatherIndices, int* sendRankLocalIndices,
    int* recvRankLocalIndices, int* backwardRecvRankLocalIndices)
{
    static_assert(kThreadsGroupSize == 1 || kThreadsGroupSize == 2 || kThreadsGroupSize == 4 || kThreadsGroupSize == 8
            || kThreadsGroupSize == 16 || kThreadsGroupSize == 32,
        "Only 1, 2, 4, 8, 16, 32 threads group size supported now.");
    __shared__ int sharedSendRecvRankStart;
    __shared__ typename cub::BlockScan<int, kThreadsPerBlock>::TempStorage tempStorage;
    if (blockIdx.y == 0)
    {
        // compute send rank count
        computeSendRecvIndicesDevice<true, kThreadsGroupSize, kThreadsPerBlock>(worldInfo, expertParallelInfo,
            maxTokenCountPerRank, realRankTokenCountCumSum, gatheredTargetRankIds, sendRankCountCumSum,
            sendRankLocalIndices, localGatherIndices, backwardRecvRankLocalIndices, &sharedSendRecvRankStart,
            tempStorage);
    }
    else
    {
        // compute recv rank count
        computeSendRecvIndicesDevice<false, kThreadsGroupSize, kThreadsPerBlock>(worldInfo, expertParallelInfo,
            maxTokenCountPerRank, realRankTokenCountCumSum, gatheredTargetRankIds, recvRankCountCumSum,
            recvRankLocalIndices, localGatherIndices, backwardRecvRankLocalIndices, &sharedSendRecvRankStart,
            tempStorage);
    }
}

void computeSendRecvIndices(MoeEpWorldInfo worldInfo, MoeExpertParallelInfo expertParallelInfo,
    int maxTokenCountPerRank, int const* realRankTokenCountCumSum, int const* gatheredTargetRankIds,
    int const* sendRankCountCumSum, int const* recvRankCountCumSum, int* localGatherIndices, int* sendRankLocalIndices,
    int* recvRankLocalIndices, int* backwardRecvRankLocalIndices, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(expertParallelInfo.topK <= 32, "Only topK less than or equal to 32 supported now.");
    int threadsPerBlock = 1024;
    auto* kernelPtr = computeSendRecvIndicesKernel<32, 1024>;
    if (expertParallelInfo.topK <= 1)
    {
        kernelPtr = computeSendRecvIndicesKernel<1, 1024>;
    }
    else if (expertParallelInfo.topK <= 2)
    {
        kernelPtr = computeSendRecvIndicesKernel<2, 1024>;
    }
    else if (expertParallelInfo.topK <= 4)
    {
        kernelPtr = computeSendRecvIndicesKernel<4, 1024>;
    }
    else if (expertParallelInfo.topK <= 8)
    {
        kernelPtr = computeSendRecvIndicesKernel<8, 1024>;
    }
    else if (expertParallelInfo.topK <= 16)
    {
        kernelPtr = computeSendRecvIndicesKernel<16, 1024>;
    }
    else if (expertParallelInfo.topK <= 32)
    {
        kernelPtr = computeSendRecvIndicesKernel<32, 1024>;
    }
    dim3 block(worldInfo.epSize, 2, 1);
    kernelPtr<<<block, threadsPerBlock, 0, stream>>>(worldInfo, expertParallelInfo, maxTokenCountPerRank,
        realRankTokenCountCumSum, gatheredTargetRankIds, sendRankCountCumSum, recvRankCountCumSum, localGatherIndices,
        sendRankLocalIndices, recvRankLocalIndices, backwardRecvRankLocalIndices);
}

void moeAllToAllPrepareIndices(MoeEpWorldInfo worldInfo, MoeExpertParallelInfo expertParallelInfo,
    int maxTokenCountPerRank, int const* gatheredTargetRankIds, int const* realRankTokenCountCumSum,
    // indices of gatheredTargetRankIds that has the local rank in topK
    int* localGatherIndices,   // max length = maxTokenCountPerRank * worldInfo.epSize when all ranks send to current
                               // rank
    int* sendRankCountCumSum,  // max length = worldInfo.epSize
    int* sendRankLocalIndices, // max length = maxTokenCountPerRank * expertParallelInfo.expertCount when current rank
                               // has maxTokenCountPerRank tokens to send and all has expertCount dest
    int* recvRankCountCumSum,  // max length = worldInfo.epSize
    int* recvRankLocalIndices, // max length = maxTokenCountPerRank * worldInfo.epSize when all ranks send to current
                               // rank
    // the rankCountCumSum of combineRecv should be the same as sendRankCountCumSum
    int*
        backwardRecvRankLocalIndices, // max length = maxTokenCountPerRank * expertParallelInfo.expertCount when current
                                      // rank has maxTokenCountPerRank tokens to send and all has expertCount dest
    cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(worldInfo.epSize <= 1024, "Only worldInfo.epSize less than or equal to 1024 supported now.");
    TLLM_CUDA_CHECK(cudaMemsetAsync(sendRankCountCumSum, 0, sizeof(int) * worldInfo.epSize, stream));
    TLLM_CUDA_CHECK(cudaMemsetAsync(recvRankCountCumSum, 0, sizeof(int) * worldInfo.epSize, stream));
    int maxSendRanksPerToken = std::max(worldInfo.epSize, expertParallelInfo.topK);

    TLLM_CUDA_CHECK(
        cudaMemsetAsync(localGatherIndices, -1, maxTokenCountPerRank * worldInfo.epSize * sizeof(int), stream));
    TLLM_CUDA_CHECK(
        cudaMemsetAsync(sendRankLocalIndices, -1, maxTokenCountPerRank * maxSendRanksPerToken * sizeof(int), stream));
    TLLM_CUDA_CHECK(
        cudaMemsetAsync(recvRankLocalIndices, -1, maxTokenCountPerRank * worldInfo.epSize * sizeof(int), stream));
    TLLM_CUDA_CHECK(cudaMemsetAsync(
        backwardRecvRankLocalIndices, -1, maxTokenCountPerRank * maxSendRanksPerToken * sizeof(int), stream));
    computeSendRecvRankCount(worldInfo, expertParallelInfo, maxTokenCountPerRank, realRankTokenCountCumSum,
        gatheredTargetRankIds, sendRankCountCumSum, recvRankCountCumSum, stream);
    inplaceSendRecvRankCumSum(worldInfo, sendRankCountCumSum, recvRankCountCumSum, stream);
    computeSendRecvIndices(worldInfo, expertParallelInfo, maxTokenCountPerRank, realRankTokenCountCumSum,
        gatheredTargetRankIds, sendRankCountCumSum, recvRankCountCumSum, localGatherIndices, sendRankLocalIndices,
        recvRankLocalIndices, backwardRecvRankLocalIndices, stream);
}

template <int kThreadsGroupSize>
__global__ void moeLocalGatherDevice(MoeEpWorldInfo worldInfo, MoeExpertParallelInfo expertParallelInfo,
    int maxTokenCountPerRank, int localMaxTokenCount, int const* recvRankCountCumSum, int const* localGatherIndices,
    int const* gatheredExpertIds, float const* gatheredScales, int* localExpertIds, float* localScales)
{
    cg::thread_block_tile<kThreadsGroupSize> tile = cg::tiled_partition<kThreadsGroupSize>(cg::this_thread_block());
    int laneInTile = tile.thread_rank();
    int tileId = threadIdx.x / kThreadsGroupSize;
    int tileCountPerBlock = blockDim.x / kThreadsGroupSize;

    int epSize = worldInfo.epSize;
    int rankTokenCount = recvRankCountCumSum[epSize - 1];
    bool needLoad = laneInTile < expertParallelInfo.topK;

    for (int index = tileId + blockIdx.x * tileCountPerBlock; index < localMaxTokenCount;
         index += tileCountPerBlock * gridDim.x)
    {
        int localTokenIndice = localGatherIndices[index];
        int expertId = needLoad && (index < rankTokenCount)
            ? gatheredExpertIds[localTokenIndice * expertParallelInfo.topK + laneInTile]
            : expertParallelInfo.expertCount;
        float scale = needLoad && (index < rankTokenCount)
            ? gatheredScales[localTokenIndice * expertParallelInfo.topK + laneInTile]
            : 0.0f;
        if (needLoad)
        {
            localExpertIds[index * expertParallelInfo.topK + laneInTile] = expertId;
            localScales[index * expertParallelInfo.topK + laneInTile] = scale;
        }
    }
}

void moeLocalGather(MoeEpWorldInfo worldInfo, MoeExpertParallelInfo expertParallelInfo, int maxTokenCountPerRank,
    int localMaxTokenCount, int const* recvRankCountCumSum, int const* localGatherIndices, int const* gatheredExpertIds,
    float const* gatheredScales, int* localExpertIds, float* localScales, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(expertParallelInfo.topK <= 32, "Only topK less than or equal to 32 supported now.");
    auto* kernelPtr = moeLocalGatherDevice<32>;
    int paddedTopK = 32;
    if (expertParallelInfo.topK <= 1)
    {
        paddedTopK = 1;
        kernelPtr = moeLocalGatherDevice<1>;
    }
    else if (expertParallelInfo.topK <= 2)
    {
        paddedTopK = 2;
        kernelPtr = moeLocalGatherDevice<2>;
    }
    else if (expertParallelInfo.topK <= 4)
    {
        paddedTopK = 4;
        kernelPtr = moeLocalGatherDevice<4>;
    }
    else if (expertParallelInfo.topK <= 8)
    {
        paddedTopK = 8;
        kernelPtr = moeLocalGatherDevice<8>;
    }
    else if (expertParallelInfo.topK <= 16)
    {
        paddedTopK = 16;
        kernelPtr = moeLocalGatherDevice<16>;
    }

    int threadsPerBlock = 512;
    int tokenPerBlock = threadsPerBlock / paddedTopK;
    int blockCount = (localMaxTokenCount + tokenPerBlock - 1) / tokenPerBlock * 2;

    kernelPtr<<<blockCount, threadsPerBlock, 0, stream>>>(worldInfo, expertParallelInfo, maxTokenCountPerRank,
        localMaxTokenCount, recvRankCountCumSum, localGatherIndices, gatheredExpertIds, gatheredScales, localExpertIds,
        localScales);
}

} // namespace tensorrt_llm::kernels
