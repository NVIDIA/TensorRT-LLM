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

#pragma once

#include "tensorrt_llm/common/cudaUtils.h"

namespace tensorrt_llm::kernels
{

#ifdef __CUDACC__
#define ALIGN_256 __align__(256)
#else
#define ALIGN_256 alignas(256)
#endif

struct ALIGN_256 MoeCommFifoConnInfo
{
    volatile uint64_t head; // write position
    volatile uint64_t tail; // read position
};

constexpr int WARP_SIZE = 32;
constexpr uint32_t WARP_MASK = 0xffffffff;

constexpr int RECV_FIFO_DEPTH = 8;
constexpr int RECV_FIFO_ENTRY_BYTES = 256 * 1024;
constexpr int RECV_FIFO_ENTRY_U64 = RECV_FIFO_ENTRY_BYTES / sizeof(uint64_t);
constexpr int RECV_FIFO_TOTAL_BYTES = RECV_FIFO_DEPTH * RECV_FIFO_ENTRY_BYTES;
constexpr int RECV_FIFO_TOTAL_U64 = RECV_FIFO_TOTAL_BYTES / sizeof(uint64_t);

constexpr int CHANNEL_COUNT = 8;

inline size_t getMoeCommWorkspaceSize(int epSize)
{
    return RECV_FIFO_TOTAL_BYTES * epSize * CHANNEL_COUNT + sizeof(MoeCommFifoConnInfo) * epSize * CHANNEL_COUNT;
}

class AllToAllChannelCommunicatorBase
{
public:
    static constexpr int GROUP_COUNT_PER_BLOCK = 8;
    static_assert(GROUP_COUNT_PER_BLOCK <= 8, "GROUP_COUNT_PER_BLOCK must be less than or equal to 8");
    static constexpr int WARP_PER_GROUP = 2;
    static constexpr int U64_DATA_REG_PER_THREAD = 8;
    // A packet is a warp-sized chunk of data that is sent or received in one go,
    // but may be split into multiple 64-bit registers, the number of which is U64_DATA_REG_PER_THREAD.
    static constexpr int PACKET_SIZE_IN_U64 = WARP_SIZE * U64_DATA_REG_PER_THREAD;
    static constexpr int PACKET_SIZE_IN_BYTES = PACKET_SIZE_IN_U64 * sizeof(uint64_t);
    static constexpr int DATA_PAYLOAD_SIZE_PER_PACKET_IN_U64 = (WARP_SIZE - 2) * U64_DATA_REG_PER_THREAD;
    static constexpr int DATA_PAYLOAD_SIZE_PER_PACKET = DATA_PAYLOAD_SIZE_PER_PACKET_IN_U64 * sizeof(uint64_t);
    static constexpr int U64_ELT_COUNT_PER_PACKET = PACKET_SIZE_IN_BYTES / sizeof(uint64_t);

    static constexpr int PACKET_COUNT_PER_FIFO_ENTRY = RECV_FIFO_ENTRY_BYTES / PACKET_SIZE_IN_BYTES;

    static constexpr int GROUP_MAX_INDICE_COUNT
        = RECV_FIFO_ENTRY_BYTES / sizeof(uint64_t) / (WARP_SIZE * U64_DATA_REG_PER_THREAD);

    struct GroupSharedBuffer
    {
        int groupIndiceBuffer[GROUP_MAX_INDICE_COUNT];
        int groupStartIndice;
        int groupEndIndice;
    };

    static dim3 getLaunchBlockDim()
    {
        return dim3(WARP_SIZE * WARP_PER_GROUP, GROUP_COUNT_PER_BLOCK);
    }

    static dim3 getLaunchGridDim(int epSize)
    {
        return dim3((epSize + GROUP_COUNT_PER_BLOCK - 1) / GROUP_COUNT_PER_BLOCK, CHANNEL_COUNT, 2);
    }
};

struct MoeEpWorldInfo
{
    int epSize;
    int epRank;
};

struct MoeExpertParallelInfo
{
    int expertCount = -1;
    int topK = 1;
};

struct SendRecvDataInfo
{
    int vectorSizeInU64;
    // pre-computed at host side for GPU kernel
    int dataPacketCountPerVector;
    int vectorCountPerFifoEntry;

    void ComputeDataPacketCountPerVector()
    {
        dataPacketCountPerVector
            = (vectorSizeInU64 * sizeof(uint64_t) + AllToAllChannelCommunicatorBase::DATA_PAYLOAD_SIZE_PER_PACKET - 1)
            / AllToAllChannelCommunicatorBase::DATA_PAYLOAD_SIZE_PER_PACKET;
    }

    void ComputeVectorCountPerFifoEntry()
    {
        ComputeDataPacketCountPerVector();
        vectorCountPerFifoEntry
            = AllToAllChannelCommunicatorBase::PACKET_COUNT_PER_FIFO_ENTRY / dataPacketCountPerVector;
    }

    void DoPreCompute()
    {
        ComputeDataPacketCountPerVector();
        ComputeVectorCountPerFifoEntry();
        assert(vectorCountPerFifoEntry <= AllToAllChannelCommunicatorBase::GROUP_MAX_INDICE_COUNT);
    }
};

struct SendRecvDispls
{
    uint64_t* dataPtr;
    int const* rankCountCumSum;  // length = epSize
    int const* rankLocalIndices; // length = rankCountCumSum[epRank] - rankCountCumSum[epRank - 1] if epRank > 0 else
                                 // rankCountCumSum[epRank]
    int vectorStrideInU64;

#ifdef __CUDACC__
    __inline__ __device__ int getCount(int rank) const
    {
        return rank == 0 ? rankCountCumSum[rank] : rankCountCumSum[rank] - rankCountCumSum[rank - 1];
    }

    __inline__ __device__ int getRankStart(int rank) const
    {
        return rank == 0 ? 0 : rankCountCumSum[rank - 1];
    }

    __inline__ __device__ int getRealVectorIndice(int globalVectorIndex) const
    {
        return rankLocalIndices[globalVectorIndex];
    }

    __inline__ __device__ uint64_t* getVectorDataPtr(int realVectorIndex) const
    {
        return dataPtr + realVectorIndex * vectorStrideInU64;
    }
#endif
};

struct MoeCommWorkspace
{
    uint64_t* workspacePtr;
    size_t rankStrideInU64;
#ifdef __CUDACC__
    __inline__ __device__ uint64_t* getFifoBasePtr(bool isSender, int epRank, int peerRank, int channel) const
    {
        // fifo itself is in receiver's side.
        if (isSender)
        {
            return workspacePtr + peerRank * rankStrideInU64 + (epRank * CHANNEL_COUNT + channel) * RECV_FIFO_TOTAL_U64;
        }
        else
        {
            return workspacePtr + epRank * rankStrideInU64 + (peerRank * CHANNEL_COUNT + channel) * RECV_FIFO_TOTAL_U64;
        }
    }

    __inline__ __device__ MoeCommFifoConnInfo* getFifoConnInfo(
        bool isSender, int epRank, int peerRank, int channel, int epSize) const
    {
        // fifoInfo is in sender's side.
        uint64_t* fifoInfoPtrU64 = workspacePtr + RECV_FIFO_TOTAL_U64 * CHANNEL_COUNT * epSize;
        int strideIndice = isSender ? epRank : peerRank;
        int fifoInfoIndice = isSender ? peerRank : epRank;
        fifoInfoPtrU64 += strideIndice * rankStrideInU64;
        MoeCommFifoConnInfo* fifoInfoPtr = (MoeCommFifoConnInfo*) fifoInfoPtrU64;
        return fifoInfoPtr + fifoInfoIndice * CHANNEL_COUNT + channel;
    }
#endif
};

void moeAllToAll(MoeEpWorldInfo worldInfo, SendRecvDataInfo sendRecvDataInfo, SendRecvDispls sendDispls,
    SendRecvDispls recvDispls, MoeCommWorkspace workspace, cudaStream_t stream);

void moeAllToAllPrepareIndices(MoeEpWorldInfo worldInfo, MoeExpertParallelInfo expertParallelInfo,
    int maxTokenCountPerRank, int const* gatheredTargetRankIds, int const* realRankTokenCountCumSum,
    int* localGatheredIndices, // indices of gatheredTargetRankIds that has the local rank in topK
    int* sendRankCountCumSum, int* sendRankLocalIndices, int* recvRankCountCumSum, int* recvRankLocalIndices,
    // the rankCountCumSum of combineRecv should be the same as sendRankCountCumSum
    int* backwardRecvRankLocalIndices, cudaStream_t stream);

void moeLocalGather(MoeEpWorldInfo worldInfo, MoeExpertParallelInfo expertParallelInfo, int maxTokenCountPerRank,
    int localMaxTokenCount, int const* recvRankCountCumSum, int const* localGatherIndices, int const* gatheredExpertIds,
    float const* gatheredScales, int* localExpertIds, float* localScales, cudaStream_t stream);

} // namespace tensorrt_llm::kernels
