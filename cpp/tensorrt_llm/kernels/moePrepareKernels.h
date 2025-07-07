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

#include <map>

#include "tensorrt_llm/common/cudaUtils.h"

#define DEBUG_PIPELINE 0

namespace tensorrt_llm::kernels
{

namespace moe_prepare
{

#define STEP_DEPTH 2
#define PACKET_PER_STEP 16
#define THREADS_PER_UNIT 1
#define UNIT_PER_PIPELINE 128
#define PIPELINE_PER_CTA 4
#define EXPERT_BYTES_PER_UNIT 32
#define SCALE_BYTES_PER_UNIT 32
#define UNIT_COUNT_PER_PACKET 1024
#define BYTES_COUNTER 8
#define CUMSUM_THREADS_PER_BLOCK 128

#define UNIT_SIZE 4
#define UNIT_PER_ITER 256
#define STATIC_COPY_PER_ITER 128
#define MAX_TOKEN_SIZE 8192

static constexpr int UNIT_BYTES_SIZE = EXPERT_BYTES_PER_UNIT + SCALE_BYTES_PER_UNIT;
static constexpr int THREADS_PER_PIPELINE = THREADS_PER_UNIT * UNIT_PER_PIPELINE;
static constexpr int THREADS_PER_CTA = THREADS_PER_PIPELINE * PIPELINE_PER_CTA;

static constexpr int SCALE_OFFSET = UNIT_SIZE * UNIT_PER_ITER * sizeof(int);
static constexpr int STATIC_COPY_OFFSET = UNIT_SIZE * UNIT_PER_ITER * (sizeof(int) + sizeof(float));
static constexpr int PACKET_SIZE
    = UNIT_SIZE * UNIT_PER_ITER * (sizeof(int) + sizeof(float)) + STATIC_COPY_PER_ITER * 4 * sizeof(int);
static constexpr int PACKET_SIZE_IN_U64 = (PACKET_SIZE / 8);
static constexpr int FIFO_SIZE_IN_U64 = PACKET_SIZE_IN_U64 * PACKET_PER_STEP * STEP_DEPTH;

#ifdef __CUDACC__
#define ALIGN_256 __align__(256)
#else
#define ALIGN_256 alignas(256)
#endif

struct ALIGN_256 MoeCommFifoConnInfo
{
    volatile uint64_t head;  // write position
    volatile uint64_t tail;  // read position
    volatile uint64_t count; // for counter
};

struct MoeCommWorkspace
{
    uint64_t* workspacePtr;
    size_t rankStrideInU64;
#ifdef __CUDACC__
    __inline__ __device__ uint64_t* getFifoBasePtr(
        bool isSender, int epRank, int peerRank, int channel, int channelCount) const
    {
        // fifo itself is in receiver's side.
        if (isSender)
        {
            return workspacePtr + peerRank * rankStrideInU64 + (epRank * channelCount + channel) * FIFO_SIZE_IN_U64;
        }
        else
        {
            return workspacePtr + epRank * rankStrideInU64 + (peerRank * channelCount + channel) * FIFO_SIZE_IN_U64;
        }
    }

    __inline__ __device__ MoeCommFifoConnInfo* getFifoConnInfo(
        bool isSender, int epRank, int peerRank, int channel, int epSize, int channelCount) const
    {
        // fifoInfo is in sender's side.
        uint64_t* fifoInfoPtrU64 = workspacePtr + FIFO_SIZE_IN_U64 * channelCount * epSize;
        int strideIndice = isSender ? epRank : peerRank;
        int fifoInfoIndice = isSender ? peerRank : epRank;
        fifoInfoPtrU64 += strideIndice * rankStrideInU64;
        MoeCommFifoConnInfo* fifoInfoPtr = (MoeCommFifoConnInfo*) fifoInfoPtrU64;
        MoeCommFifoConnInfo* result = fifoInfoPtr + fifoInfoIndice * channelCount + channel;
        return result;
    }

#endif
};

void computeCountAndIndice(int* experts, int* sendCounts, int* recvCounts, int* sendIndiceWorkspace,
    int* backwardIndiceWorkspace, int* recvIndiceWorkspace, MoeCommWorkspace workspace, int tokenCount,
    int maxTokenCountPerRank, int topK, int expert_count, int rankId, int rankCount, cudaStream_t stream);

void computeCumsum(int* sendCountsCumsum, int* recvCountsCumsum, int rankId, int rankCount, cudaStream_t stream);

void moveIndice(int* sendCountsCumsum, int* recvCountsCumsum, int* sendIndice, int* gatherSendIndice,
    int* backwardIndice, int* gatherBackwardIndice, int* recvIndice, int* gatherRecvIndice, int rankId, int rankCount,
    int maxTokenCountPerRank, cudaStream_t stream);

void allToAllMetadata(int* sendExperts, int* recvExperts, float* sendScales, float* recvScales, int* localExpertStatics,
    int* gatheredExpertStatics, MoeCommWorkspace workspace, int* sendCountsCumsum, int* localSendIndice,
    int* recvCountsCumsum, int* localRecvIndice, int tokenCount, int maxTokenCountPerRank, int topK, int expertCount,
    int slotCount, int rankId, int rankCount, cudaStream_t stream);

size_t getMoePrepareWorkspaceSize(int epSize);

} // namespace moe_prepare

} // namespace tensorrt_llm::kernels
