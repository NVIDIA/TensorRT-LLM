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
#include "tensorrt_llm/common/envUtils.h"

#define DEBUG_PIPELINE 0

namespace tensorrt_llm::kernels
{

namespace moe_prepare
{

#define UNIT_PER_PIPELINE 128
#define PIPELINE_PER_CTA 4
#define CUMSUM_THREADS_PER_BLOCK 128

static constexpr int THREADS_PER_PIPELINE = UNIT_PER_PIPELINE;

#ifdef __CUDACC__
#define ALIGN_256 __align__(256)
#else
#define ALIGN_256 alignas(256)
#endif

struct ALIGN_256 MoeCommFifoConnInfo
{
    volatile uint64_t head;   // write position
    volatile uint64_t tail;   // read position
    int volatile values[512]; // for values
};

struct MoeCommWorkspace
{
    uint64_t* workspacePtr;
    size_t rankStrideInU64;
#ifdef __CUDACC__
    __inline__ __device__ MoeCommFifoConnInfo* getFifoConnInfo(
        bool isSender, int epRank, int peerRank, int channel, int epSize, int channelCount) const
    {
        // fifoInfo is in sender's side.
        uint64_t* fifoInfoPtrU64 = workspacePtr;
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
    int* backwardIndiceWorkspace, int* recvIndiceWorkspace, int* expertStatics, int* gatheredExpertStatics,
    MoeCommWorkspace workspace, int tokenCount, int maxTokenCountPerRank, int topK, int slotCount, int expertCount,
    int rankId, int rankCount, cudaStream_t stream);

void computeCumsum(int* sendCountsCumsum, int* recvCountsCumsum, int rankId, int rankCount, cudaStream_t stream);

void moveIndice(int* sendCountsCumsum, int* recvCountsCumsum, int* sendIndice, int* gatherSendIndice,
    int* backwardIndice, int* gatherBackwardIndice, int* recvIndice, int* gatherRecvIndice, int rankId, int rankCount,
    int maxTokenCountPerRank, cudaStream_t stream);

void memsetExpertIds(int* expertIds, int* recvCountsCumsum, int maxTokenCountPerRank, int topK, int slotCount,
    int epSize, cudaStream_t stream);

size_t getMoePrepareWorkspaceSize(int epSize);

} // namespace moe_prepare

} // namespace tensorrt_llm::kernels
