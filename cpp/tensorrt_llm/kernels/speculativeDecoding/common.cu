/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/kernels/speculativeDecoding/common.h"
#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#elif (CUDART_VERSION >= 11050)
#include <cub/cub.cuh>
#else
#include "3rdparty/cub/cub.cuh"
#endif

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::runtime;

namespace tensorrt_llm::kernels::speculative_decoding
{
template <int32_t BLOCK_SIZE>
__global__ void packAcceptedPaths(SizeType32* acceptedLengthsCumSum, SizeType32* pathsOffsets,
    SizeType32 const* acceptedLengths, SizeType32 const* bestPathIds, SizeType32 const* paths,
    SizeType32 const* batchSlots, SizeType32 batchSize, SizeType32 numPaths, SizeType32 maxPathLen,
    bool isPathsLinearBatchIdx)
{
    // Specialize BlockScan for a 1D block of 128 threads of type int
    typedef cub::BlockScan<SizeType32, BLOCK_SIZE> BlockScan;

    // Allocate shared memory for BlockScan
    __shared__ typename BlockScan::TempStorage tempStorage;
    auto const batchSizeRounded = ((batchSize + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
    __shared__ SizeType32 currentCumSum;
    if (threadIdx.x == 0)
    {
        currentCumSum = 0;
    }

    __syncthreads();

    for (auto bi = static_cast<SizeType32>(threadIdx.x); bi < batchSizeRounded;
         bi += static_cast<SizeType32>(blockDim.x))
    {
        auto const valid = bi < batchSize;
        auto const batchSlot = valid ? batchSlots[bi] : 0;
        auto const acceptedLen = valid ? acceptedLengths[batchSlot] - 1 : 0;
        SizeType32 cumSum;
        BlockScan(tempStorage).ExclusiveSum(acceptedLen + currentCumSum, cumSum);
        if (threadIdx.x == blockDim.x - 1)
        {
            currentCumSum = cumSum;
        }
        __syncthreads();

        if (valid)
        {
            acceptedLengthsCumSum[bi] = cumSum;
            auto const pathBatchIdx = isPathsLinearBatchIdx ? bi : batchSlot;
            auto const bestPathIdx = bestPathIds[pathBatchIdx];
            auto const pathIdx = flat_index3(pathBatchIdx, bestPathIdx, 0, numPaths, maxPathLen);
            for (SizeType32 ti = 0; ti < acceptedLen; ++ti)
            {
                pathsOffsets[cumSum + ti] = paths[pathIdx + ti + 1] - 1;
            }
        }
    }
    if (threadIdx.x == 0)
    {
        acceptedLengthsCumSum[batchSize] = currentCumSum;
    }
}

void invokePackAcceptedPaths(SizeType32* acceptedLengthsCumSum, SizeType32* pathsOffsets,
    SizeType32 const* acceptedLengths, SizeType32 const* bestPathIds, SizeType32 const* paths,
    SizeType32 const* batchSlots, SizeType32 batchSize, SizeType32 numPaths, SizeType32 maxPathLen,
    bool isPathsLinearBatchIdx, cudaStream_t stream)
{
    constexpr SizeType32 BLOCK_SIZE = 1024;
    packAcceptedPaths<BLOCK_SIZE><<<1, BLOCK_SIZE, 0, stream>>>(acceptedLengthsCumSum, pathsOffsets, acceptedLengths,
        bestPathIds, paths, batchSlots, batchSize, numPaths, maxPathLen, isPathsLinearBatchIdx);
}

namespace
{
__device__ __forceinline__ int4 reduceMaxInt4(int4 const& a, int4 const& b)
{
    return a.x >= b.x ? a : b;
}

template <typename T, SizeType32 BLOCK_SIZE>
__global__ void acceptDraftTokensByIdsWithPaths(TokenIdType* outputIds, TokenIdType const* draftIds,
    TokenIdType const* targetIds, SizeType32* sequenceLengths, SizeType32* acceptedLengths,
    FinishedState* finishedFinal, SizeType32 const* batchSlots, SizeType32 const* paths, TokenIdType const* endIds,
    T const** medusaLogits, T const** logitsPtrs, SizeType32* curTokensPerStep, SizeType32 const* targetTokensPerStep,
    SizeType32* bestPathIds, SizeType32 batchSize, SizeType32 vocabSize, SizeType32 maxBatchSize, SizeType32 maxSeqLen,
    SizeType32 maxDraftPathLen, SizeType32 maxDecodingTokens)
{
    auto const batchIdx = static_cast<SizeType32>(blockIdx.x);
    auto const batchSlot = batchSlots == nullptr ? batchIdx : batchSlots[batchIdx];
    auto const inputLength = sequenceLengths == nullptr ? 0 : sequenceLengths[batchSlot];
    auto const endId = endIds == nullptr ? -1 : endIds[batchSlot];
    auto const numTokensPerStep = curTokensPerStep == nullptr ? maxDecodingTokens : curTokensPerStep[batchSlot];
    auto const maxPathLen = maxDraftPathLen + 1;

    int4 partialMax{-1, -1, 0, 0};
    // Go over different paths and construct implicit sequences
    for (auto pathIdx = static_cast<SizeType32>(threadIdx.x); pathIdx < maxDecodingTokens;
         pathIdx += static_cast<SizeType32>(blockDim.x))
    {
        auto acceptedLength = maxPathLen;
        auto const pathOffset = flat_index3(batchSlot, pathIdx, 0, maxDecodingTokens, maxPathLen);
        bool hasEnd = false;

        auto const tokenId = paths[pathOffset];
        // Continue if path does not exist
        if (tokenId == -1)
        {
            continue;
        }
        auto const targetTokenIdx = batchSlot * maxDecodingTokens + tokenId;
        auto targetToken = targetIds[targetTokenIdx];
        auto nextIdx = tokenId;

        // Go along the path
        for (SizeType32 ti = 1; ti < maxPathLen; ++ti)
        {
            auto const tokenId = paths[pathOffset + ti];
            // Break if path terminates
            if (tokenId == -1)
            {
                hasEnd = endIds == nullptr ? false
                                           : targetToken == endId; // check if last token is EOS when path terminates.
                acceptedLength = hasEnd ? ti - 1 : ti;
                break;
            }
            auto const targetTokenIdx = batchSlot * maxDecodingTokens + tokenId;
            auto const draftTokenIdx = batchSlot * (maxDecodingTokens - 1) + tokenId - 1;
            // In context phase, no draft tokens are given. Set draft token to -1 to get guaranteed rejection
            auto const draftToken = tokenId >= numTokensPerStep ? -1 : draftIds[draftTokenIdx];
            // Check if draft tokens are the same as target tokens
            bool const accepted = draftToken == targetToken;
            hasEnd = endIds == nullptr ? false : targetToken == endId;
            if (!accepted || hasEnd)
            {
                acceptedLength = hasEnd ? ti - 1 : ti;
                break;
            }
            targetToken = targetIds[targetTokenIdx];
            nextIdx = tokenId;
        }
        // Get longest path of the thread
        if (partialMax.x < acceptedLength)
        {
            partialMax.x = acceptedLength;
            partialMax.y = pathIdx;
            partialMax.z = hasEnd;
            partialMax.w = nextIdx;
        }
    }

    // Get the longest path of the block (request)
    typedef cub::BlockReduce<int4, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage tempStorage;
    int4 total = BlockReduce(tempStorage).Reduce(partialMax, reduceMaxInt4);

    __shared__ int4 totalShared;
    if (threadIdx.x == 0)
    {
        totalShared = total;
    }

    __syncthreads();

    auto const acceptedLength = totalShared.x;
    auto const bestPathIdx = totalShared.y;
    auto const bestNextIdx = numTokensPerStep == 1 ? 0 : totalShared.w;
    auto const pathOffset = flat_index3(batchSlot, bestPathIdx, 0, maxDecodingTokens, maxPathLen);
    for (auto ti = static_cast<SizeType32>(threadIdx.x); ti < acceptedLength; ti += static_cast<SizeType32>(blockDim.x))
    {
        auto const tokenId = paths[pathOffset + ti];
        auto const targetSrcTokenIdx = batchSlot * maxDecodingTokens + tokenId;
        auto const outputTokenIdx = batchSlot * maxSeqLen + inputLength + ti;
        auto const targetToken = targetIds[targetSrcTokenIdx];
        // Copy accepted tokens to the sequence with draft tokens (outputIds === outputIds)
        outputIds[outputTokenIdx] = targetToken;
    }

    // Leading thread reconstructs winning path and sets new data
    if (threadIdx.x == 0)
    {
        auto const hasEnd = totalShared.z;
        // Set end condition
        if (hasEnd && finishedFinal)
        {
            finishedFinal[batchSlot].setFinishedEOS();
        }
        // Make correction to the sequence length
        if (sequenceLengths)
        {
            sequenceLengths[batchSlot] += acceptedLength;
        }
        acceptedLengths[batchSlot] = acceptedLength;
        // In Medusa decoding step, number of draft tokens is 0 and must be updated for the next steps
        if (curTokensPerStep && targetTokensPerStep && numTokensPerStep == 1)
        {
            curTokensPerStep[batchSlot] = targetTokensPerStep[batchSlot];
        }
        bestPathIds[batchSlot] = bestPathIdx;
    }

    // Prepare logits pointers to respective logits from Medusa Heads for the all-top-K sampling kernel
    if (medusaLogits && logitsPtrs)
    {
        for (auto hi = static_cast<SizeType32>(threadIdx.x); hi < maxDraftPathLen;
             hi += static_cast<SizeType32>(blockDim.x))
        {
            logitsPtrs[batchIdx * maxDraftPathLen + hi]
                = medusaLogits[batchSlot * maxDraftPathLen + hi] + flat_index2(bestNextIdx, 0, vocabSize);
        }
    }
}
} // namespace

template <typename T>
void acceptDraftTokensByIdsWithPaths(AcceptDraftTokensByIdsWithPathsParams<T> const& params)
{
    constexpr SizeType32 BLOCK_SIZE = 256;
    dim3 block(BLOCK_SIZE);
    dim3 grid(params.batchSize);
    acceptDraftTokensByIdsWithPaths<T, BLOCK_SIZE><<<grid, block, 0, params.stream>>>(params.outputIds, params.draftIds,
        params.targetIds, params.sequenceLengths, params.acceptedLengths, params.finishedFinal, params.batchSlots,
        params.paths, params.endIds, params.medusaLogits, params.logitsPtrs, params.curTokensPerStep,
        params.targetTokensPerStep, params.bestPathIds, params.batchSize, params.vocabSize, params.maxBatchSize,
        params.maxSeqLen, params.maxDraftPathLen, params.maxDecodingTokens);
}

template void acceptDraftTokensByIdsWithPaths(AcceptDraftTokensByIdsWithPathsParams<float> const& params);
template void acceptDraftTokensByIdsWithPaths(AcceptDraftTokensByIdsWithPathsParams<__half> const& params);

} // namespace tensorrt_llm::kernels::speculative_decoding
