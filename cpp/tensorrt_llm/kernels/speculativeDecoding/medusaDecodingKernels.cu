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
#include "tensorrt_llm/kernels/speculativeDecoding/medusaDecodingKernels.h"
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
    SizeType32 maxNumHeads, SizeType32 maxDecodingTokens)
{
    auto const batchIdx = static_cast<SizeType32>(blockIdx.x);
    auto const batchSlot = batchSlots[batchIdx];
    auto const inputLength = sequenceLengths[batchSlot];
    auto const endId = endIds[batchSlot];
    auto const numTokensPerStep = curTokensPerStep[batchSlot];
    auto const maxNumDraftTokens = maxNumHeads + 1;

    int4 partialMax{-1, -1, 0, 0};
    // Go over different paths and construct implicit sequences
    for (auto pathIdx = static_cast<SizeType32>(threadIdx.x); pathIdx < maxDecodingTokens;
         pathIdx += static_cast<SizeType32>(blockDim.x))
    {
        auto acceptedLength = maxNumDraftTokens;
        auto const pathOffset = flat_index3(batchSlot, pathIdx, 0, maxDecodingTokens, maxNumDraftTokens);
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
        for (SizeType32 ti = 1; ti < maxNumDraftTokens; ++ti)
        {
            auto const tokenId = paths[pathOffset + ti];
            // Break if path terminates
            if (tokenId == -1)
            {
                hasEnd = targetToken == endId; // check if last token is EOS when path terminates.
                acceptedLength = hasEnd ? ti - 1 : ti;
                break;
            }
            auto const targetTokenIdx = batchSlot * maxDecodingTokens + tokenId;
            auto const draftTokenIdx = batchSlot * (maxDecodingTokens - 1) + tokenId - 1;
            // In context phase, no draft tokens are given. Set draft token to -1 to get guaranteed rejection
            auto const draftToken = tokenId >= numTokensPerStep ? -1 : draftIds[draftTokenIdx];
            // Check if draft tokens are the same as target tokens
            bool const accepted = draftToken == targetToken;
            hasEnd = targetToken == endId;
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
    auto const pathOffset = flat_index3(batchSlot, bestPathIdx, 0, maxDecodingTokens, maxNumDraftTokens);
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
        if (hasEnd)
        {
            finishedFinal[batchSlot].setFinishedEOS();
        }
        // Make correction to the sequence length
        sequenceLengths[batchSlot] += acceptedLength;
        acceptedLengths[batchSlot] = acceptedLength;
        // In Medusa decoding step, number of draft tokens is 0 and must be updated for the next steps
        if (numTokensPerStep == 1)
        {
            curTokensPerStep[batchSlot] = targetTokensPerStep[batchSlot];
        }
        bestPathIds[batchSlot] = bestPathIdx;
    }

    // Prepare logits pointers to respective logits from Medusa Heads for the all-top-K sampling kernel
    for (auto hi = static_cast<SizeType32>(threadIdx.x); hi < maxNumHeads; hi += static_cast<SizeType32>(blockDim.x))
    {
        logitsPtrs[batchIdx * maxNumHeads + hi]
            = medusaLogits[batchSlot * maxNumHeads + hi] + flat_index2(bestNextIdx, 0, vocabSize);
    }
}
} // namespace

template <typename T>
void acceptDraftTokensByIdsWithPaths(TokenIdType* outputIds, TokenIdType const* draftIds, TokenIdType const* targetIds,
    SizeType32* sequenceLengths, SizeType32* acceptedLengths, FinishedState* finishedFinal,
    SizeType32 const* batchSlots, SizeType32 const* paths, TokenIdType const* endIds, T const** medusaLogits,
    T const** logitsPtrs, SizeType32* curTokensPerStep, SizeType32 const* targetTokensPerStep, SizeType32* bestPathIds,
    SizeType32 batchSize, SizeType32 vocabSize, SizeType32 maxBatchSize, SizeType32 maxSeqLen, SizeType32 maxNumHeads,
    SizeType32 maxDecodingTokens, cudaStream_t stream)
{
    constexpr SizeType32 BLOCK_SIZE = 256;
    dim3 block(BLOCK_SIZE);
    dim3 grid(batchSize);
    acceptDraftTokensByIdsWithPaths<T, BLOCK_SIZE><<<grid, block, 0, stream>>>(outputIds, draftIds, targetIds,
        sequenceLengths, acceptedLengths, finishedFinal, batchSlots, paths, endIds, medusaLogits, logitsPtrs,
        curTokensPerStep, targetTokensPerStep, bestPathIds, batchSize, vocabSize, maxBatchSize, maxSeqLen, maxNumHeads,
        maxDecodingTokens);
}

template void acceptDraftTokensByIdsWithPaths(TokenIdType* outputIds, TokenIdType const* draftIds,
    TokenIdType const* targetIds, SizeType32* sequenceLengths, SizeType32* acceptedLengths,
    FinishedState* finishedFinal, SizeType32 const* batchSlots, SizeType32 const* paths, TokenIdType const* endIds,
    float const** medusaLogits, float const** logitsPtrs, SizeType32* curTokensPerStep,
    SizeType32 const* targetTokensPerStep, SizeType32* bestPathIds, SizeType32 batchSize, SizeType32 vocabSize,
    SizeType32 maxBatchSize, SizeType32 maxSeqLen, SizeType32 maxNumHeads, SizeType32 maxDecodingTokens,
    cudaStream_t stream);
template void acceptDraftTokensByIdsWithPaths(TokenIdType* outputIds, TokenIdType const* draftIds,
    TokenIdType const* targetIds, SizeType32* sequenceLengths, SizeType32* acceptedLengths,
    FinishedState* finishedFinal, SizeType32 const* batchSlots, SizeType32 const* paths, TokenIdType const* endIds,
    half const** medusaLogits, half const** logitsPtrs, SizeType32* curTokensPerStep,
    SizeType32 const* targetTokensPerStep, SizeType32* bestPathIds, SizeType32 batchSize, SizeType32 vocabSize,
    SizeType32 maxBatchSize, SizeType32 maxSeqLen, SizeType32 maxNumHeads, SizeType32 maxDecodingTokens,
    cudaStream_t stream);

namespace
{
__global__ void scatterMedusaDraftTokens(TokenIdType* treeDraftIds, TokenIdType const* sourceDraftIds,
    SizeType32 const* treeIds, SizeType32 const* tokensPerStepData, SizeType32 const* batchSlots,
    SizeType32 maxDecodingTokens)
{
    auto const batchIdx = static_cast<SizeType32>(blockIdx.x);
    auto const batchSlot = batchSlots[batchIdx];
    auto const tokensPerStep = tokensPerStepData[batchSlot];
    auto const maxDecodingDraftTokens = maxDecodingTokens - 1;
    for (auto index = static_cast<SizeType32>(threadIdx.x); index < tokensPerStep - 1;
         index += static_cast<SizeType32>(blockDim.x))
    {
        auto const indexInTree = treeIds[batchSlot * maxDecodingDraftTokens + index];
        auto const treeDraftIdx = batchSlot * maxDecodingDraftTokens + index;
        auto const sourceDraftIdx = batchSlot * maxDecodingTokens + indexInTree;
        treeDraftIds[treeDraftIdx] = sourceDraftIds[sourceDraftIdx];
    }
}
} // namespace

void scatterMedusaDraftTokens(TokenIdType* treeDraftIds, TokenIdType const* sourceDraftIds, SizeType32 const* treeIds,
    SizeType32 const* tokensPerStep, SizeType32 const* batchSlots, SizeType32 maxDecodingTokens, SizeType32 batchSize,
    cudaStream_t stream)
{
    constexpr SizeType32 BLOCK_SIZE = 256;
    scatterMedusaDraftTokens<<<batchSize, BLOCK_SIZE, 0, stream>>>(
        treeDraftIds, sourceDraftIds, treeIds, tokensPerStep, batchSlots, maxDecodingTokens);
}
} // namespace tensorrt_llm::kernels::speculative_decoding
