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
#include "tensorrt_llm/kernels/speculativeDecoding/eagleDecodingKernels.h"
#include "tensorrt_llm/kernels/speculativeDecoding/explicitDraftTokensKernels.h"
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
template <typename T, int BLOCK_SIZE>
__global__ void assembleTargetLogitsOffsets(T const** logitsPtrs, SizeType32* decodingTokens, T const* logits,
    SizeType32 const* draftDecodingTokens, SizeType32 batchSize, SizeType32 maxDecodingTokens,
    SizeType32 vocabSizePadded)
{
    typedef cub::BlockScan<SizeType32, BLOCK_SIZE> BlockScan;
    __shared__ typename BlockScan::TempStorage tempStorage;

    auto const bid = static_cast<SizeType32>(threadIdx.x);

    SizeType32 numDecodingTokens{0};
    // if valid request
    if (bid < batchSize)
    {
        // Get how many logits are there per request
        numDecodingTokens = draftDecodingTokens[bid] + 1;
        // Save number of decoding tokens
        decodingTokens[bid] = numDecodingTokens;
    }

    // Get offsets to the logits of each request
    SizeType32 logitsOffset{0};
    BlockScan(tempStorage).ExclusiveSum(numDecodingTokens, logitsOffset);

    // if valid request
    if (bid < batchSize)
    {
        for (SizeType32 ti = 0; ti < numDecodingTokens; ++ti)
        {
            // Assemble logits pointers
            logitsPtrs[bid * maxDecodingTokens + ti] = logits + (logitsOffset + ti) * vocabSizePadded;
        }
    }
}
} // namespace

template <typename T>
void invokeAssembleTargetLogitsOffsets(T const** logitsPtrs, SizeType32* decodingTokens, T const* logits,
    SizeType32 const* draftDecodingTokens, SizeType32 batchSize, SizeType32 maxDecodingTokens,
    SizeType32 vocabSizePadded, cudaStream_t stream)
{
    SizeType32 constexpr BLOCK_SIZE = 512;
    TLLM_CHECK_WITH_INFO(
        batchSize <= BLOCK_SIZE, "Batch size larger than %d is not supported for EAGLE yet", batchSize);
    assembleTargetLogitsOffsets<T, BLOCK_SIZE><<<1, BLOCK_SIZE, 0, stream>>>(
        logitsPtrs, decodingTokens, logits, draftDecodingTokens, batchSize, maxDecodingTokens, vocabSizePadded);

    sync_check_cuda_error();
}

template void invokeAssembleTargetLogitsOffsets(float const** logitsPtrs, SizeType32* decodingTokens,
    float const* logits, SizeType32 const* draftDecodingTokens, SizeType32 batchSize, SizeType32 maxDecodingTokens,
    SizeType32 vocabSizePadded, cudaStream_t stream);
template void invokeAssembleTargetLogitsOffsets(__half const** logitsPtrs, SizeType32* decodingTokens,
    __half const* logits, SizeType32 const* draftDecodingTokens, SizeType32 batchSize, SizeType32 maxDecodingTokens,
    SizeType32 vocabSizePadded, cudaStream_t stream);

namespace
{
template <int BLOCK_SIZE>
__global__ void prepareCtxEagleNetInputsKernel(SizeType32* eagleNetSequenceLengths, SizeType32* eagleNetContextLengths,
    TokenIdType* outputIds, SizeType32* positionIds, SizeType32* hiddenStatesIndices, SizeType32* lastTokenIndices,
    SizeType32* numLastTokenIndices, SizeType32* hiddenSizeBatchLevelStarts, TokenIdType const* inputIds,
    SizeType32 const* baseNetSequenceLengths, SizeType32 const* baseNetContextLengths,
    TokenIdType const* acceptedTokens, SizeType32 const* acceptedLens, SizeType32 const* prevDraftLens,
    SizeType32 const* prevPaths, SizeType32 const* bestPathIds, SizeType32 batchSize, SizeType32 maxPathLen,
    SizeType32 maxDecodingTokens, SizeType32 maxNonLeavesPerLayer)
{
    typedef cub::BlockScan<SizeType32, BLOCK_SIZE> BlockScan;
    __shared__ typename BlockScan::TempStorage tempStorage;

    auto const bid = static_cast<SizeType32>(threadIdx.x);

    bool const isValid{bid < batchSize};
    bool isContextRequest{false};
    SizeType32 numDecodingTokens{0};
    SizeType32 numInputTokens{0};
    SizeType32 prevDraftLen{0};
    // if valid request
    if (isValid)
    {
        prevDraftLen = prevDraftLens[bid];
        isContextRequest = (prevDraftLen == 0);
        if (isContextRequest)
        {
            // If context request, number of input and output tokens equal to prompt len.
            numInputTokens = baseNetContextLengths[bid];
            numDecodingTokens = numInputTokens;
        }
        else
        {
            // If gen request
            // Number of input tokens is draft len + 1.
            numInputTokens = prevDraftLen + 1;
            // Number of output tokens for EagleNet is accepted len.
            numDecodingTokens = acceptedLens[bid];
        }
    }

    for (SizeType32 ii = bid; ii < maxNonLeavesPerLayer * batchSize; ii += BLOCK_SIZE)
    {
        lastTokenIndices[ii] = 1;
    }

    SizeType32 outputStartPos{0};
    SizeType32 inputIndexBase{0};
    SizeType32 lastTokenIndex{0};
    // Get offset for input ids in inputIds.
    BlockScan(tempStorage).ExclusiveSum(numInputTokens, inputIndexBase);
    // Sync since tempStorage is reused later.
    __syncthreads();
    // Get offset for output ids in outputIds.
    BlockScan(tempStorage).ExclusiveSum(numDecodingTokens, outputStartPos);
    // Sync since tempStorage is reused later.
    __syncthreads();
    // Get offset for last logit in outputIds.
    BlockScan(tempStorage).InclusiveSum(numDecodingTokens, lastTokenIndex);

    // if valid request
    if (isValid)
    {
        // Sequence length of the base model (without draft len for gen requests and all prompt tokens for ctx request)
        auto const oldSequenceLength = baseNetSequenceLengths[bid] - numInputTokens;
        for (SizeType32 ti = 0; ti < numDecodingTokens; ++ti)
        {
            TokenIdType token;
            if (isContextRequest)
            {
                if (ti == numDecodingTokens - 1)
                {
                    // Last token is newly sampled token
                    token = acceptedTokens[bid * maxPathLen + 0];
                }
                else
                {
                    // Skip the first token
                    token = inputIds[inputIndexBase + ti + 1];
                }
            }
            else
            {
                // Get accepted tokens
                token = acceptedTokens[bid * maxPathLen + ti];
            }
            outputIds[outputStartPos + ti] = token;
            positionIds[outputStartPos + ti] = oldSequenceLength + ti;
        }
        // EagleNet0 expects context or chunked context.
        // Context len equals to the context len for context req or acceptedLen for gen request.
        eagleNetContextLengths[bid] = numDecodingTokens;
        // EagleNet0' sequence length equals to
        // old sequence len + (context len for context req or acceptedLen for gen request).
        eagleNetSequenceLengths[bid] = oldSequenceLength + numDecodingTokens;

        auto const bestPathId = bestPathIds[bid];
        // For all output indices of this request
        for (SizeType32 ii = 0; ii < numDecodingTokens; ++ii)
        {
            SizeType32 index;
            if (isContextRequest)
            {
                // If context, just take all hidden states one after another
                index = inputIndexBase + ii;
            }
            else
            {
                // If generation, take all hidden states of the tokens on the accepted path
                auto const pathIdx = flat_index3(bid, bestPathId, ii, maxDecodingTokens, maxPathLen);
                auto const lastTokenId = prevPaths[pathIdx];
                index = inputIndexBase + lastTokenId;
            }
            // Save index.
            hiddenStatesIndices[outputStartPos + ii] = index;
        }
        // After the first EagleNet we predict exactly one set of logits per request.
        lastTokenIndices[bid] = lastTokenIndex;

        // EagleNet0 produces 1 relevant hidden state per request.
        hiddenSizeBatchLevelStarts[bid] = bid;
    }

    // The last thread writes number of flattened tokens.
    if (bid == BLOCK_SIZE - 1)
    {
        // After the first EagleNet we predict exactly one set of logits per request.
        numLastTokenIndices[0] = batchSize;

        // Set last hiddenSizeBatchLevelStarts.
        hiddenSizeBatchLevelStarts[batchSize] = batchSize;
    }
}
} // namespace

void invokePrepareCtxEagleNetInputs(SizeType32* eagleNetSequenceLengths, SizeType32* eagleNetContextLengths,
    TokenIdType* outputIds, SizeType32* positionIds, SizeType32* hiddenStatesIndices, SizeType32* lastTokenIndices,
    SizeType32* numLastTokenIndices, SizeType32* hiddenSizeBatchLevelStarts, TokenIdType const* inputIds,
    SizeType32 const* baseNetSequenceLengths, SizeType32 const* baseNetContextLengths,
    TokenIdType const* acceptedTokens, SizeType32 const* acceptedLens, SizeType32 const* prevDraftLens,
    SizeType32 const* prevPaths, SizeType32 const* bestPathIds, SizeType32 batchSize, SizeType32 maxPathLen,
    SizeType32 maxDecodingTokens, SizeType32 maxNonLeavesPerLayer, cudaStream_t stream)
{
    SizeType32 constexpr BLOCK_SIZE = 512;
    TLLM_CHECK_WITH_INFO(
        batchSize <= BLOCK_SIZE, "Batch size larger than %d is not supported for EAGLE yet", batchSize);
    prepareCtxEagleNetInputsKernel<BLOCK_SIZE><<<1, BLOCK_SIZE, 0, stream>>>(eagleNetSequenceLengths,
        eagleNetContextLengths, outputIds, positionIds, hiddenStatesIndices, lastTokenIndices, numLastTokenIndices,
        hiddenSizeBatchLevelStarts, inputIds, baseNetSequenceLengths, baseNetContextLengths, acceptedTokens,
        acceptedLens, prevDraftLens, prevPaths, bestPathIds, batchSize, maxPathLen, maxDecodingTokens,
        maxNonLeavesPerLayer);
}

namespace
{
__global__ void buildLeafMask(
    int8_t* isLeafMask, SizeType32 const* paths, SizeType32 maxDecodingTokens, SizeType32 maxPathLen)
{
    auto const bid = static_cast<SizeType32>(blockIdx.x);
    auto const level = static_cast<SizeType32>(blockIdx.y);

    // For all paths
    for (auto pathIdx = static_cast<SizeType32>(threadIdx.x); pathIdx < maxDecodingTokens;
         pathIdx += static_cast<SizeType32>(blockDim.x))
    {
        // Get next level offset in paths buffer
        auto const tokenNextLevelOffset = flat_index3(bid, pathIdx, level + 1, maxDecodingTokens, maxPathLen);
        // Get current level offset in paths buffer
        auto const tokenCurLevelOffset = flat_index3(bid, pathIdx, level, maxDecodingTokens, maxPathLen);
        // Get token idx in the flattened draft tokens for the of the current level
        auto const curNodeTokenIdx = paths[tokenCurLevelOffset];
        // If token idx is not -1 (not terminated path) And the next
        // level token is not -1 -- path is not terminating and token at current level has child.
        if (curNodeTokenIdx != -1 && paths[tokenNextLevelOffset] != -1)
        {
            // Mark mask to 0.
            isLeafMask[bid * maxDecodingTokens + curNodeTokenIdx] = 0;
        }
    }
}

__global__ void getNonLeafEndingSubtree(SizeType32* selectedDraftIndices, SizeType32* selectedPosOffsets, bool* mask,
    SizeType32* numSelectedDraftIndices, SizeType32* parentNonLeafInLevelOffset, SizeType32* nonLeavesInLevelOffsets,
    int8_t const* isLeafMask, SizeType32 const* paths, SizeType32 levelIdx, SizeType32 maxDecodingTokens,
    SizeType32 maxPathLen)
{
    auto const bid = static_cast<SizeType32>(blockIdx.x);
    auto const maxDecodingDraftTokens = maxDecodingTokens - 1;

    extern __shared__ char smemBuf[];

    SizeType32* histogramSmem = reinterpret_cast<SizeType32*>(smemBuf);
    SizeType32* posOffsetSmem = reinterpret_cast<SizeType32*>(smemBuf + maxDecodingTokens * sizeof(SizeType32));
    SizeType32* selectedPathsSmem = reinterpret_cast<SizeType32*>(smemBuf + 2 * maxDecodingTokens * sizeof(SizeType32));
    SizeType32* tokenPosSmem = reinterpret_cast<SizeType32*>(smemBuf + 3 * maxDecodingTokens * sizeof(SizeType32));

    for (auto ii = static_cast<SizeType32>(threadIdx.x); ii < maxDecodingTokens;
         ii += static_cast<SizeType32>(blockDim.x))
    {
        // Init histogram.
        histogramSmem[ii] = 0;
        // Init pos offsets for CAS.
        posOffsetSmem[ii] = -1;
        // Init selected paths for CAS.
        selectedPathsSmem[ii] = -1;
    }

    __syncthreads();

    // For all paths
    for (auto pi = static_cast<SizeType32>(threadIdx.x); pi < maxDecodingTokens;
         pi += static_cast<SizeType32>(blockDim.x))
    {
        auto const tokenCurLevelOffset = flat_index3(bid, pi, levelIdx, maxDecodingTokens, maxPathLen);
        // Get token idx at current level for given path
        auto const tokenIdxLevel = paths[tokenCurLevelOffset];
        // Check if this path is not terminated yet.
        // And check if this node is not leaf.
        if (tokenIdxLevel >= 0 && !isLeafMask[bid * maxDecodingTokens + tokenIdxLevel])
        {
            // Set this path as selected for this token.
            atomicCAS(&selectedPathsSmem[tokenIdxLevel], -1, pi);
            // For all nodes before current in this path
            // Skip position 0 as it is golden token
            for (SizeType32 li = 1; li <= levelIdx; ++li)
            {
                auto const tokenOffset = flat_index3(bid, pi, li, maxDecodingTokens, maxPathLen);
                auto const tokenIdx = paths[tokenOffset];
                // Mark them as needed in the histogram.
                // FIXME: how bad is that atomic in shared?
                atomicAdd(&histogramSmem[tokenIdx], 1);
                // Store position offset.
                // li-1 here because golden token is already at kv cache at this point.
                atomicCAS(&posOffsetSmem[tokenIdx], -1, li - 1);
            }
        }
    }

    __syncthreads();

    // FIXME: do with more than 1 thread?
    if (threadIdx.x == 0)
    {
        SizeType32 selectedCount{0};
        SizeType32 prevPosOffset{-1};
        SizeType32 nonLeavesInLevelCounter{0};

        // First node (golden token) is always at index 0 at its level.
        nonLeavesInLevelOffsets[bid * maxDecodingTokens + 0] = 0;

        // Check which tokens are selected in the histogram.
        // Skip position 0 as it is golden token
        for (SizeType32 ti = 1; ti < maxDecodingTokens; ++ti)
        {
            // Check if id is selected.
            if (histogramSmem[ti] > 0)
            {
                // Save it.
                selectedDraftIndices[bid * maxDecodingDraftTokens + selectedCount] = ti;
                auto const posOffset = posOffsetSmem[ti];
                selectedPosOffsets[bid * maxDecodingDraftTokens + selectedCount] = posOffset;

                // When jumped to next level, reset non-leaves-in-level counter.
                if (posOffset != prevPosOffset)
                {
                    nonLeavesInLevelCounter = 0;
                    prevPosOffset = posOffset;
                }

                // If node is not leaf
                if (!isLeafMask[bid * maxDecodingTokens + ti])
                {
                    // Save its in-level index for output hidden state indices calculation.
                    nonLeavesInLevelOffsets[bid * maxDecodingTokens + ti] = nonLeavesInLevelCounter;
                    nonLeavesInLevelCounter++;
                }

                // Save position where the token is written to in selectedDraftIndices.
                tokenPosSmem[ti] = selectedCount;

                selectedCount++;
            }
        }

        // FIXME it is too inefficient.
        for (SizeType32 pi = 0; pi < maxDecodingTokens; ++pi)
        {
            for (SizeType32 li = 1; li <= levelIdx; ++li)
            {
                auto const tokenCurLevelOffset = flat_index3(bid, pi, li, maxDecodingTokens, maxPathLen);
                auto const tokenPrevLevelOffset = flat_index3(bid, pi, li - 1, maxDecodingTokens, maxPathLen);
                // Get token idx at current level for given path
                auto const tokenIdxCurLevel = paths[tokenCurLevelOffset];
                // Get token idx at previous level for given path
                auto const tokenIdxPrevLevel = paths[tokenPrevLevelOffset];
                // Propagate non leaf indices down to the tree.
                parentNonLeafInLevelOffset[bid * maxDecodingTokens + tokenIdxCurLevel]
                    = nonLeavesInLevelOffsets[bid * maxDecodingTokens + tokenIdxPrevLevel];
            }
        }

        numSelectedDraftIndices[bid] = selectedCount;
    }

    __syncthreads();

    // For all tokens
    for (auto ti = static_cast<SizeType32>(threadIdx.x); ti < maxDecodingTokens;
         ti += static_cast<SizeType32>(blockDim.x))
    {
        auto const pathId = selectedPathsSmem[ti];
        // This path was selected
        if (pathId >= 0)
        {
            // For all nodes in this path to levelIdx
            for (SizeType32 li = 1; li <= levelIdx; ++li)
            {
                // FIXME (paths is shared)?
                auto const tokenOffsetI = flat_index3(bid, pathId, li, maxDecodingTokens, maxPathLen);
                auto const tokenIdxI = paths[tokenOffsetI];
                // Get position of the token in the selectedDraftIndices.
                auto const tokenPosI = tokenPosSmem[tokenIdxI];
                // For all nodes before current in this path
                for (SizeType32 lj = 1; lj <= li; ++lj)
                {
                    auto const tokenOffsetJ = flat_index3(bid, pathId, lj, maxDecodingTokens, maxPathLen);
                    auto const tokenIdxJ = paths[tokenOffsetJ];
                    // Get position of the token in the selectedDraftIndices.
                    auto const tokenPosJ = tokenPosSmem[tokenIdxJ];
                    // Fill mask
                    auto const maskOffset
                        = flat_index3(bid, tokenPosI, tokenPosJ, maxDecodingTokens, maxDecodingTokens);
                    mask[maskOffset] = true;
                }
            }
        }
    }
}

template <int BLOCK_SIZE>
__global__ void prepareGenEagleNetInputsKernel(SizeType32* nextSequenceLengths, SizeType32* nextContextLengths,
    TokenIdType* outputIds, SizeType32* positionIds, SizeType32* specDecodingGenLengths,
    SizeType32* specDecodingPositionOffsets, SizeType32* specDecodingPackedMasks, SizeType32* hiddenStatesIndices,
    SizeType32* lastTokenIndices, SizeType32* numLastTokenIndices, SizeType32* outputHiddenSizeBatchStartsPerLevel,
    SizeType32* cumSumGenerationLengths, SizeType32* maxGenerationLength, TokenIdType const* nextDraftIds,
    SizeType32 const* selectedDraftIndices, SizeType32 const* selectedDraftPosIds,
    SizeType32 const* numSelectedDraftIndices, SizeType32 const* eagleNet0SequenceLengths,
    SizeType32 const* prevContextLengths, SizeType32 const* inputHiddenSizeBatchStartsPerLevel,
    SizeType32 const* parentNonLeafInLevelOffset, SizeType32 levelIdx, SizeType32 batchSize, SizeType32 maxPathLen,
    SizeType32 maxDecodingTokens, SizeType32 maxNonLeavesPerLayer)
{
    typedef cub::BlockScan<SizeType32, BLOCK_SIZE> BlockScan;
    typedef cub::BlockReduce<SizeType32, BLOCK_SIZE> BlockReduce;

    __shared__ union
    {
        typename BlockScan::TempStorage scan;
        typename BlockReduce::TempStorage reduce;
    } tempStorage;

    auto const bid = static_cast<SizeType32>(threadIdx.x);
    auto const maxDecodingDraftTokens{maxDecodingTokens - 1};

    bool const isValid{bid < batchSize};
    SizeType32 nextDraftLen{0};
    SizeType32 numNextLogits{0};

    if (isValid)
    {
        nextDraftLen = numSelectedDraftIndices[bid];
        for (SizeType32 ti = 0; ti < nextDraftLen; ++ti)
        {
            auto const posOffset = selectedDraftPosIds[bid * maxDecodingDraftTokens + ti];
            if (posOffset == levelIdx - 1)
            {
                numNextLogits++;
            }
        }
    }

    SizeType32 outputIndexBase{0};
    SizeType32 genLengthCumSum{0};
    SizeType32 lastIndices{0};
    SizeType32 outputLastIndicesBase{0};
    // Get offset for output ids in outputIds.
    BlockScan(tempStorage.scan).ExclusiveSum(nextDraftLen, outputIndexBase);
    // Sync because tempStorage is reused.
    __syncthreads();
    // Get offset for output ids in outputIds.
    BlockScan(tempStorage.scan).InclusiveSum(nextDraftLen, genLengthCumSum);
    // Sync because tempStorage is reused.
    __syncthreads();
    BlockScan(tempStorage.scan).InclusiveSum(numNextLogits, lastIndices);
    // Sync because tempStorage is reused.
    __syncthreads();
    BlockScan(tempStorage.scan).ExclusiveSum(numNextLogits, outputLastIndicesBase);
    // Sync because tempStorage is reused.
    __syncthreads();
    auto const maxGenLength = BlockReduce(tempStorage.reduce).Reduce(nextDraftLen, cub::Max());

    // Thread 0 has the result.
    if (bid == 0)
    {
        // Save max draft length for the mask packing kernel.
        maxGenerationLength[0] = maxGenLength;
    }

    if (isValid)
    {
        // Fill spec decoding gen length.
        specDecodingGenLengths[bid] = nextDraftLen;
        // Simply copy context len.
        nextContextLengths[bid] = prevContextLengths[bid];
        auto const sequenceLen = eagleNet0SequenceLengths[bid];
        // Next sequence len is base seqlen + draft len.
        nextSequenceLengths[bid] = sequenceLen + nextDraftLen;

        // Fill cumulative sum for the mask packing kernel.
        cumSumGenerationLengths[bid] = genLengthCumSum;

        // Pos id is Ctx EagleNet seqLen (prompt + all accepted).
        positionIds[bid] = sequenceLen;

        SizeType32 lastTokenIdx{0};
        for (SizeType32 ti = 0; ti < nextDraftLen; ++ti)
        {
            // Copy next draft ids.
            // Minus -1 here to account for path indexing. 0 is golden token, which is not present in nextDraftIds.
            auto const draftIdx = selectedDraftIndices[bid * maxDecodingDraftTokens + ti] - 1;
            outputIds[outputIndexBase + ti] = nextDraftIds[bid * maxDecodingDraftTokens + draftIdx];

            // Get draft pos offset.
            auto const posOffset = selectedDraftPosIds[bid * maxDecodingDraftTokens + ti];
            specDecodingPositionOffsets[bid * maxDecodingTokens + ti] = posOffset;

            // hiddenStatesIndex is constructed having hidden states layout in mind.
            // Hidden states are placed in memory as [maxPathLen - 1, batchSize, numOutputTokens] (this tensor is
            // flattaned and padding is removed) E.g. with BS=2, r0 and r1 have 1 hidden state at level 0 (golden
            // token). r0 has 2 hidden states and r1 has 3 hidden states at level 1. [h_0_0_0, h_0_0_1, h_0_1_0,
            // h_1_1_0, h_0_1_1, h_1_1_1, h_2_1_1], where h_i_j_k means ith hidden state of request k at level j.
            auto const inLevelTokenOffset = parentNonLeafInLevelOffset[bid * maxDecodingTokens + draftIdx + 1];
            hiddenStatesIndices[outputIndexBase + ti]
                = inputHiddenSizeBatchStartsPerLevel[posOffset * batchSize + bid] + inLevelTokenOffset;

            // If pos offset is equal to the layer idx, it means that this token is in the last available layer of the
            // tree. We have never sampled tokens from it and thus we need its logits.
            if (posOffset == levelIdx - 1)
            {
                // +1 here as gather_last_token_logits expects indices starting from 1
                lastTokenIndices[outputLastIndicesBase + lastTokenIdx] = outputIndexBase + ti + 1;
                lastTokenIdx++;
            }
        }

        // Copy existing inputHiddenSizeBatchStartsPerLevel.
        for (SizeType32 li = 0; li < levelIdx; ++li)
        {
            outputHiddenSizeBatchStartsPerLevel[li * batchSize + bid]
                = inputHiddenSizeBatchStartsPerLevel[li * batchSize + bid];
        }

        auto const lastStart = inputHiddenSizeBatchStartsPerLevel[(levelIdx - 1) * batchSize + batchSize];
        // Set new layer idx.
        outputHiddenSizeBatchStartsPerLevel[levelIdx * batchSize + bid] = lastStart + bid * maxNonLeavesPerLayer;
    }

    __syncthreads();

    // The last valid thread fills the number of tokens.
    if (bid == batchSize - 1)
    {
        // Set the total number of logits needed after the next iteration.
        numLastTokenIndices[0] = lastIndices;
        // Set last outputHiddenSizeBatchStartsPerLevel.
        outputHiddenSizeBatchStartsPerLevel[levelIdx * batchSize + batchSize]
            = outputHiddenSizeBatchStartsPerLevel[levelIdx * batchSize + batchSize - 1] + maxNonLeavesPerLayer;
    }
}

template <typename T>
inline __device__ __host__ T divUp(T m, T n)
{
    return (m + n - 1) / n;
}

__device__ SizeType32 positivePowerOfTwo(SizeType32 n)
{
    return 1 << n;
}

//! @brief Takes mask of size [maxGenerationLength] filled with 1s and 0s defined in the shared memory
//! and packs it to bitmask of size [numPackedMasks] written to outputPtr.
//! numPackedMasks = ceil(maxGenerationLength / 32);
__device__ __forceinline__ void maskToPackedMask(
    SizeType32* outputPtr, char const* shMask, SizeType32 maxGenerationLength, SizeType32 numPackedMasks)
{
    for (SizeType32 maskId = 0; maskId < numPackedMasks; ++maskId)
    {
        if (maskId * 32 >= maxGenerationLength)
        {
            outputPtr[maskId] = 0;
            return;
        }
        else
        {
            auto const shMaskIndexStart
                = ((maxGenerationLength - (maskId + 1) * 32) < 0) ? 0 : (maxGenerationLength - (maskId + 1) * 32);
            auto const shMaskIndexEnd = maxGenerationLength - maskId * 32;
            auto const validNumBits = shMaskIndexEnd - shMaskIndexStart;

            auto const firstBit1 = (shMask[shMaskIndexStart] == '1') ? true : false;
            SizeType32 mask31bits = 0;
            if (validNumBits != 1)
            {
                for (auto i = shMaskIndexStart + 1; i < shMaskIndexEnd; i++)
                {
                    auto const index = (validNumBits - 1) - (i - shMaskIndexStart);
                    mask31bits += (shMask[i] == '1') ? positivePowerOfTwo(index) : 0;
                }
            }
            SizeType32 mask32bits;
            if (validNumBits == 32)
            {
                mask32bits = firstBit1 ? mask31bits - positivePowerOfTwo(validNumBits - 1) : mask31bits;
            }
            else
            {
                mask32bits = firstBit1 ? mask31bits + positivePowerOfTwo(validNumBits - 1) : mask31bits;
            }
            outputPtr[maskId] = mask32bits;
        }
    }
}

__global__ void getPackedMask(SizeType32 const* __restrict__ cumGenerationLengths,
    SizeType32 const* __restrict__ maxGenerationLengths, bool const* __restrict__ mask,
    SizeType32 maxDecodingDraftTokens, SizeType32* __restrict__ packedMask)
{
    auto const batchIdx = static_cast<SizeType32>(blockIdx.y);
    auto const tokenIdx = static_cast<SizeType32>(blockIdx.x);

    auto const numTokens = (batchIdx == 0) ? cumGenerationLengths[0]
                                           : cumGenerationLengths[batchIdx] - cumGenerationLengths[batchIdx - 1];
    if (tokenIdx >= numTokens)
    {
        return;
    }

    auto const maxGenerationLength = maxGenerationLengths[0];
    auto const maxDecodingTokens = maxDecodingDraftTokens + 1;
    auto const numPackedMasks = divUp(maxDecodingTokens, 32);

    auto const outputStartId = ((batchIdx == 0) ? 0 : cumGenerationLengths[batchIdx - 1]);
    auto* outputPtr = packedMask + (outputStartId + tokenIdx) * numPackedMasks;
    if (tokenIdx == 0)
    {
        for (auto maskId = static_cast<SizeType32>(threadIdx.x); maskId < numPackedMasks;
             maskId += static_cast<SizeType32>(blockDim.x))
        {
            outputPtr[maskId] = maskId == 0 ? 1 : 0;
        }
        return;
    }
    else
    {
        bool const* maskPtr = mask + batchIdx * maxDecodingTokens * maxDecodingTokens + tokenIdx * maxDecodingTokens;
        extern __shared__ char shMask[];
        for (auto ti = static_cast<SizeType32>(threadIdx.x); ti < maxGenerationLength;
             ti += static_cast<SizeType32>(blockDim.x))
        {
            auto const shIndex = maxGenerationLength - 1 - ti;
            shMask[shIndex] = maskPtr[ti] ? '1' : '0';
        }
        __syncthreads();

        if (threadIdx.x == 0)
        {
            maskToPackedMask(outputPtr, shMask, maxGenerationLength, numPackedMasks);
        }
    }
}

__global__ void getPackedMaskFromPath(SizeType32* __restrict__ packedMask, SizeType32 const* __restrict__ batchSlots,
    SizeType32 const* __restrict__ paths, SizeType32 maxDecodingTokens, SizeType32 maxPathLen)
{
    extern __shared__ char adjacencyMatrix[];

    // The request Id that this block process
    auto const batchIdx = static_cast<SizeType32>(blockIdx.x);
    auto const batchSlot = batchSlots[batchIdx];

    // Initialize
    for (auto tix = static_cast<SizeType32>(threadIdx.x); tix < maxDecodingTokens * maxDecodingTokens;
         tix += static_cast<SizeType32>(blockDim.x))
    {
        // Set adjacency matrix to 0
        adjacencyMatrix[tix] = '0';
    }
    // Set token mask
    if (threadIdx.x == 0)
    {
        adjacencyMatrix[0 * maxDecodingTokens + (maxDecodingTokens - 1)] = '1';
    }

    __syncthreads();

    // Get the offset of the path (tree)
    auto const curPath = paths + batchSlot * maxDecodingTokens * maxPathLen;

    // Traverse the path and update adjacency matrix
    for (auto tix = static_cast<SizeType32>(threadIdx.x); tix < maxDecodingTokens;
         tix += static_cast<SizeType32>(blockDim.x))
    {
        for (SizeType32 ti = 1; ti < maxPathLen; ++ti)
        {
            auto const pathOffset = tix * maxPathLen;
            auto const toIndex = curPath[pathOffset + ti];
            if (toIndex == -1)
            {
                break;
            }
            adjacencyMatrix[toIndex * maxDecodingTokens + (maxDecodingTokens - 1 - toIndex)] = '1';
            for (SizeType32 fi = 0; fi < ti; ++fi)
            {
                auto const fromIndex = maxDecodingTokens - 1 - curPath[pathOffset + fi];
                // adjacencyMatrix[fromIndex][toIndex] = 1
                // TODO atomic cas
                adjacencyMatrix[toIndex * maxDecodingTokens + fromIndex] = '1';
            }
        }
    }

    __syncthreads();

    auto const numPackedMasks = divUp(maxDecodingTokens, 32);
    for (auto ti = static_cast<SizeType32>(threadIdx.x); ti < maxDecodingTokens;
         ti += static_cast<SizeType32>(blockDim.x))
    {
        auto outputPtr = packedMask + batchSlot * maxDecodingTokens * numPackedMasks + ti * numPackedMasks;
        maskToPackedMask(outputPtr, adjacencyMatrix + ti * maxDecodingTokens, maxDecodingTokens, numPackedMasks);
    }
}
} // namespace

void invokePrepareGenEagleNetInputs(PrepareGenEagleNetInputsParams const& params)
{
    SizeType32 constexpr BLOCK_SIZE = 512;
    TLLM_CHECK_WITH_INFO(
        params.batchSize <= BLOCK_SIZE, "Batch size larger than %d is not supported for EAGLE yet", params.batchSize);

    // Build mask to distinguish leaf and not-leaf nodes in the tree.
    {
        // TODO: computing it at each layer_idx is redundant.
        // Mask shape [batchSize, maxDecodingTokens], where 1 means that the node is the leaf, 0 means that node is not
        // leaf.
        dim3 grid(params.batchSize, params.maxPathLen - 1);
        buildLeafMask<<<grid, BLOCK_SIZE, 0, params.stream>>>(
            params.isLeafMask, params.nextPaths, params.maxDecodingTokens, params.maxPathLen);

        sync_check_cuda_error();
    }

    // Select all no-leaf ending paths for given level idx.
    // E.g. given paths [[0, 1, 4, 6], [0, 1, 4, 7], [0, 2, -1, -1], [0, 3, 5, -1]],
    // for levelIdx = 0 it returns [0], for levelIdx = 1 it returns [0, 1, 3], for levelIdx = 2 it returns [0, 1, 4],
    // for levelIdx = 3 [] parentNonLeafInLevelOffset is the index (from left to right) of parent node among all other
    // non-leaf nodes at parent's level. For the above example it is [0, 0, 0, 0, 1, 0, 0]. Here the first 0, 0, 0 are
    // for level=1, for tokens 1, 2 and 3 as their parent as 0th at its level (golden token). Next 0, 1, are for tokens
    // 4, 5 on the level 2. Token 4 depends on the 1 which is 0th non leaf at its level. Token 5 depends on 3 which is
    // the 1st non leaf at its level, etc.
    {
        auto const smemSize = 4 * params.maxDecodingTokens * sizeof(SizeType32);
        getNonLeafEndingSubtree<<<params.batchSize, BLOCK_SIZE, smemSize, params.stream>>>(params.selectedDraftIndices,
            params.selectedDraftPosOffsets, params.selectedMasks, params.numSelectedDraftIndices,
            params.parentNonLeafInLevelOffset, params.nonLeavesInLevelOffsets, params.isLeafMask, params.nextPaths,
            params.levelIdx, params.maxDecodingTokens, params.maxPathLen);

        sync_check_cuda_error();
    }

    // Use selected tokens and prepare data for gen iteration of EagleNet.
    {
        prepareGenEagleNetInputsKernel<BLOCK_SIZE><<<1, BLOCK_SIZE, 0, params.stream>>>(params.nextSequenceLengths,
            params.nextContextLengths, params.outputIds, params.positionIds, params.specDecodingGenLengths,
            params.specDecodingPositionOffsets, params.specDecodingPackedMasks, params.hiddenStatesIndices,
            params.lastTokenIndices, params.numLastTokenIndices, params.outputHiddenSizeBatchStartsPerLevel,
            params.cumSumGenerationLengths, params.maxGenerationLength, params.nextDraftIds,
            params.selectedDraftIndices, params.selectedDraftPosOffsets, params.numSelectedDraftIndices,
            params.eagleNet0SequenceLengths, params.prevContextLengths, params.inputHiddenSizeBatchStartsPerLevel,
            params.parentNonLeafInLevelOffset, params.levelIdx, params.batchSize, params.maxPathLen,
            params.maxDecodingTokens, params.maxNonLeavesPerLayer);

        sync_check_cuda_error();
    }

    {
        dim3 block(32);
        dim3 grid(params.maxDecodingTokens, params.batchSize);
        size_t shmSize = params.maxDecodingTokens * sizeof(char);
        getPackedMask<<<grid, block, shmSize, params.stream>>>(params.cumSumGenerationLengths,
            params.maxGenerationLength, params.selectedMasks, params.maxDecodingTokens - 1,
            params.specDecodingPackedMasks);

        sync_check_cuda_error();
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace
{

template <typename T>
__global__ void assembleDraftLogitsOffsets(T const** logitsPtrs, T const* logits, TokenIdType** outputIdsPtrs,
    TokenIdType* outputIds, bool* skipDecode, runtime::SizeType32 const* numValidLogits, SizeType32 batchSize,
    SizeType32 maxDecodingDraftTokens, SizeType32 vocabSizePadded)
{
    auto const tix = static_cast<SizeType32>(blockIdx.x * blockDim.x + threadIdx.x);
    auto const isValid{tix < numValidLogits[0]};

    if (isValid)
    {
        // logits: [numInputLogits, vocab_size]
        // logitsPtrs: [numInputLogits][1, vocab_size]
        logitsPtrs[tix] = logits + tix * vocabSizePadded;

        // outputIds: [numInputLogits * maxDecodingDraftTokens]
        // outputIdsPtrs: [numInputLogits][maxDecodingDraftTokens]
        outputIdsPtrs[tix] = outputIds + tix * maxDecodingDraftTokens;
    }

    skipDecode[tix] = !isValid;
}

} // namespace

template <typename T>
void invokeAssembleDraftLogitsOffsets(T const** logitsPtrs, T const* logits, runtime::TokenIdType** outputIdsPtrs,
    runtime::TokenIdType* outputIds, bool* skipDecode, runtime::SizeType32 const* numValidLogits,
    runtime::SizeType32 numInputLogits, runtime::SizeType32 batchSize, runtime::SizeType32 maxDecodingDraftTokens,
    runtime::SizeType32 vocabSizePadded, cudaStream_t stream)
{
    SizeType32 constexpr BLOCK_SIZE = 512;

    assembleDraftLogitsOffsets<T><<<divUp(numInputLogits, BLOCK_SIZE), BLOCK_SIZE, 0, stream>>>(logitsPtrs, logits,
        outputIdsPtrs, outputIds, skipDecode, numValidLogits, batchSize, maxDecodingDraftTokens, vocabSizePadded);

    sync_check_cuda_error();
}

template void invokeAssembleDraftLogitsOffsets(float const** logitsPtrs, float const* logits,
    runtime::TokenIdType** outputIdsPtrs, runtime::TokenIdType* outputIds, bool* skipDecode,
    runtime::SizeType32 const* numValidLogits, runtime::SizeType32 numInputLogits, runtime::SizeType32 batchSize,
    runtime::SizeType32 maxDecodingDraftTokens, runtime::SizeType32 vocabSizePadded, cudaStream_t stream);

template void invokeAssembleDraftLogitsOffsets(__half const** logitsPtrs, __half const* logits,
    runtime::TokenIdType** outputIdsPtrs, runtime::TokenIdType* outputIds, bool* skipDecode,
    runtime::SizeType32 const* numValidLogits, runtime::SizeType32 numInputLogits, runtime::SizeType32 batchSize,
    runtime::SizeType32 maxDecodingDraftTokens, runtime::SizeType32 vocabSizePadded, cudaStream_t stream);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace
{

// Extract the number of successors for each node from path
__global__ void extractNumSuccessorsFromPath(SizeType32 const* paths, SizeType32* numSuccessorsForEachNode,
    SizeType32 layerId, SizeType32 batchSize, SizeType32 maxDecodingTokens, SizeType32 maxPathLen)
{
    // paths: shape [batchSize, maxDecodingTokens, maxPathLen]
    // numSuccessorsForEachNode: shape [batchSize, maxDecodingTokens]

    // Adjacency matrix, shape: [maxDecodingTokens * maxDecodingTokens]
    extern __shared__ bool adjMatrix[];

    // The request Id that this block process
    auto const batchIdx = static_cast<SizeType32>(blockIdx.x);

    auto tix = static_cast<SizeType32>(threadIdx.x);

    // Initialize
    while (tix < maxDecodingTokens * maxDecodingTokens)
    {
        if (tix < maxDecodingTokens)
        {
            // Set numSuccessorsForEachNode to 0
            numSuccessorsForEachNode[batchIdx * maxDecodingTokens + tix] = 0;
        }
        // Set adjacency matrix to 0
        adjMatrix[tix] = 0;
        tix += blockDim.x;
    }
    __syncthreads();
    tix = static_cast<SizeType32>(threadIdx.x); // Reset tix

    // Get the offset of the path (tree)
    SizeType32 const* curPath = paths + batchIdx * maxDecodingTokens * maxPathLen;

    // Traverse a specific level (i.e. layerId) of the path and update adjacency matrix
    while (tix < maxDecodingTokens)
    {
        SizeType32 const* subPath = curPath + tix * maxPathLen + layerId;
        SizeType32 fromIndex = subPath[0];
        SizeType32 toIndex = subPath[1];

        if (fromIndex != -1 && toIndex != -1)
        {
            // adjMatrix[fromIndex][toIndex] = 1
            adjMatrix[fromIndex * maxDecodingTokens + toIndex] = 1;
        }
        tix += blockDim.x;
    }

    __syncthreads();

    // Fill the numSuccessorsForEachNode array according to the adjacency matrix
    tix = static_cast<SizeType32>(threadIdx.x); // Reset tix
    while (tix < maxDecodingTokens)
    {
        // For each tix row
        SizeType32 numSuccessors = 0;
        // Loop all the maxDecodingTokens column
        for (SizeType32 ii = 0; ii < maxDecodingTokens; ii++)
        {
            // adjMatrix[tix][ii]
            if (adjMatrix[tix * maxDecodingTokens + ii])
            {
                numSuccessors++;
            }
        }
        numSuccessorsForEachNode[batchIdx * maxDecodingTokens + tix] = numSuccessors;
        tix += blockDim.x;
    }
}

template <int BLOCK_SIZE>
__global__ void extracTopKsFromSuccessorsArray(SizeType32* topKs, SizeType32* topKOffset,
    SizeType32 const* numSuccessorsForEachNode, SizeType32 batchSize, SizeType32 maxDecodingTokens)
{
    // topKs: shape [numInputLogits]
    // topKOffset: shape [batchSize]
    // numSuccessorsForEachNode: shape [batchSize * maxDecodingTokens]

    typedef cub::BlockScan<SizeType32, BLOCK_SIZE> BlockScan;
    __shared__ typename BlockScan::TempStorage tempStorage;

    auto const tix = static_cast<SizeType32>(threadIdx.x); // batchIdx
    SizeType32 const* curNumSuccessorsForEachNode = numSuccessorsForEachNode + tix * maxDecodingTokens;

    SizeType32 numNodesHaveSuccessors{0};
    if (tix < batchSize)
    {
        for (SizeType32 ii = 0; ii < maxDecodingTokens; ii++)
        {
            if (curNumSuccessorsForEachNode[ii] > 0)
            {
                numNodesHaveSuccessors++;
            }
        }
    }

    SizeType32 curTopKOffset{0};
    BlockScan(tempStorage).ExclusiveSum(numNodesHaveSuccessors, curTopKOffset);

    if (tix < batchSize)
    {
        // Update topKOffset for this request
        topKOffset[tix] = curTopKOffset;

        // Fill the topK array according to the numSuccessorsForEachNode array.
        // The index values ​​go from small to large,
        // Corresponding to the tree node from left to right and from top to bottom.
        for (SizeType32 ii = 0; ii < maxDecodingTokens; ii++)
        {
            if (curNumSuccessorsForEachNode[ii] > 0)
            {
                topKs[curTopKOffset] = curNumSuccessorsForEachNode[ii];
                curTopKOffset++;
            }
        }
    }
}

} // namespace

// Extract topKs from paths and layerId
void invokeExtractTopKsFromPath(runtime::SizeType32 const* paths, runtime::SizeType32* topKs,
    runtime::SizeType32* topKOffset, runtime::SizeType32* numSuccessorsForEachNode, runtime::SizeType32 layerId,
    runtime::SizeType32 batchSize, runtime::SizeType32 maxDecodingTokens, runtime::SizeType32 maxPathLen,
    cudaStream_t stream)
{

    TLLM_CHECK_WITH_INFO(
        layerId < maxPathLen, "layerId (%d) larger than maxPathLen (%d) is not allow", layerId, maxPathLen);

    SizeType32 constexpr BLOCK_SIZE = 512;
    SizeType32 dynamicSmemSize
        = maxDecodingTokens * maxDecodingTokens * sizeof(bool); // shape: [maxDecodingTokens * maxDecodingTokens]
    // Each thread block corresponds to a request.
    // The shared memory in a block is the adjacency matrix for this request's path
    extractNumSuccessorsFromPath<<<batchSize, BLOCK_SIZE, dynamicSmemSize, stream>>>(
        paths, numSuccessorsForEachNode, layerId, batchSize, maxDecodingTokens, maxPathLen);

    sync_check_cuda_error();

    TLLM_CHECK_WITH_INFO(
        batchSize <= BLOCK_SIZE, "Batch size larger than %d is not supported for EAGLE yet", BLOCK_SIZE);

    // Extract topKs from numSuccessorsForEachNode array
    extracTopKsFromSuccessorsArray<BLOCK_SIZE>
        <<<1, BLOCK_SIZE, 0, stream>>>(topKs, topKOffset, numSuccessorsForEachNode, batchSize, maxDecodingTokens);
}

namespace
{
__global__ void copyOutputTokensIds(TokenIdType** tmpOutputIdsPtrs, SizeType32 const* topKs,
    SizeType32 const* topKOffset, TokenIdType const* pluginInputDraftIdsPtrs, SizeType32 const* pluginInputDraftLens,
    SizeType32 const* numValidLogits, TokenIdType* pluginOutputDraftIdsPtrs, SizeType32* pluginOutputDraftLens,
    SizeType32 layerId, SizeType32 batchSize, SizeType32 maxDecodingDraftTokens)
{
    // tmpOutputIdsPtrs: shape [numInputLogits][maxDecodingDraftTokens]
    // topKs: shape [numInputLogits]
    // topKOffset: shape [batchSize]
    // pluginInputDraftIdsPtrs: shape [batchSize][maxDecodingDraftTokens]
    // pluginInputDraftLens: shape [batchSize]
    // pluginOutputDraftIdsPtrs: shape [batchSize][maxDecodingDraftTokens]
    // pluginOutputDraftLens: shape [batchSize]

    auto const tix = static_cast<SizeType32>(blockIdx.x * blockDim.x + threadIdx.x);
    if (tix < batchSize)
    {
        // Output draft token ids offset
        TokenIdType* curPluginOutputDraftIdsPtrs = pluginOutputDraftIdsPtrs + tix * maxDecodingDraftTokens;
        TokenIdType const* curPluginInputDraftIddsPtrs = pluginInputDraftIdsPtrs + tix * maxDecodingDraftTokens;

        // The length of the existing draft token
        SizeType32 prevLen = layerId == 0 ? 0 : pluginInputDraftLens[tix];

        // Copy exist tokens
        for (SizeType32 ii = 0; ii < prevLen; ii++)
        {
            curPluginOutputDraftIdsPtrs[ii] = curPluginInputDraftIddsPtrs[ii];
        }

        SizeType32 curLen = prevLen;

        // Compute the topK offset
        SizeType32 startTopKOffset = topKOffset[tix];
        SizeType32 endTopkOffset = tix + 1 < batchSize ? topKOffset[tix + 1] : numValidLogits[0];

        for (SizeType32 ii = startTopKOffset; ii < endTopkOffset; ii++)
        {
            for (SizeType32 jj = 0; jj < topKs[ii]; jj++)
            {
                curPluginOutputDraftIdsPtrs[curLen] = tmpOutputIdsPtrs[ii][jj];
                curLen++;
            }
        }

        // Update the output draft token length of this request
        pluginOutputDraftLens[tix] = curLen;
    }
}

} // namespace

// Copy output draft token ids from temporary buffer to plugin output buffer, also update the draft token length
void invokeCopyOutputTokensIds(runtime::TokenIdType** tmpOutputIdsPtrs, runtime::SizeType32 const* topKs,
    runtime::SizeType32 const* topKOffset, runtime::TokenIdType const* pluginInputDraftIdsPtrs,
    runtime::SizeType32 const* pluginInputDraftLens, runtime::SizeType32 const* numValidLogits,
    runtime::TokenIdType* pluginOutputDraftIdsPtrs, runtime::SizeType32* pluginOutputDraftLens,
    runtime::SizeType32 layerId, runtime::SizeType32 batchSize, runtime::SizeType32 maxDecodingDraftTokens,
    cudaStream_t stream)
{
    SizeType32 constexpr BLOCK_SIZE = 512;

    copyOutputTokensIds<<<divUp(batchSize, BLOCK_SIZE), BLOCK_SIZE, 0, stream>>>(tmpOutputIdsPtrs, topKs, topKOffset,
        pluginInputDraftIdsPtrs, pluginInputDraftLens, numValidLogits, pluginOutputDraftIdsPtrs, pluginOutputDraftLens,
        layerId, batchSize, maxDecodingDraftTokens);
}

namespace
{
__global__ void packEagleGenerationLengths(PackEagleParams params)
{
    auto const batchIdx = static_cast<SizeType32>(blockIdx.x);
    auto const batchSlot = params.batchSlots[batchIdx];

    auto const isGenerationRequest = batchIdx >= params.numContextRequests;
    auto const genIdx = batchIdx - params.numContextRequests;

    if (threadIdx.x == 0 && isGenerationRequest)
    {
        params.outputSpecDecodingGenerationLengths[genIdx] = params.inputSpecDecodingGenerationLengths[batchSlot];
    }
}
} // namespace

void invokePackEagleGenerationLengths(PackEagleParams const& params, cudaStream_t stream)
{
    SizeType32 constexpr BLOCK_SIZE = 32;
    packEagleGenerationLengths<<<params.batchSize, BLOCK_SIZE, 0, stream>>>(params);
}

namespace
{
__global__ void packEagleTensors(PackEagleParams params)
{
    auto const batchIdx = static_cast<SizeType32>(blockIdx.x);
    auto const batchSlot = params.batchSlots[batchIdx];

    auto const isGenerationRequest = batchIdx >= params.numContextRequests;
    auto const genIdx = batchIdx - params.numContextRequests;

    // Copy data that is 1 elem per request
    if (threadIdx.x == 0)
    {
        params.outputRandomDataSample[batchIdx] = params.inputRandomDataSample[batchSlot];
        params.outputTemperatures[batchIdx] = params.inputTemperatures[batchSlot];
        // FIXME we need 1 value per draft token
        params.outputRandomDataValidation[batchIdx] = params.inputRandomDataValidation[batchSlot];

        // 0 for ctx request and actual draft len for gen requests.
        params.outputNextDraftLens[batchIdx]
            = isGenerationRequest ? params.inputSpecDecodingGenerationLengths[batchSlot] - 1 : 0;
    }

    // Copy draft paths
    auto const numPathElts = params.maxNumPaths * params.maxPathLength;
    auto outputNextDraftPaths = params.outputNextDraftPaths + batchIdx * numPathElts;
    auto const inputNextDraftPaths = params.inputNextDraftPaths + batchSlot * numPathElts;
    for (auto ti = static_cast<SizeType32>(threadIdx.x); ti < numPathElts; ti += static_cast<SizeType32>(blockDim.x))
    {
        outputNextDraftPaths[ti] = inputNextDraftPaths[ti];
    }

    if (isGenerationRequest)
    {
        // Copy draft tokens. We do it only for gen requests as for ctx requests outputNextDraftLens is 0.
        auto const maxDecodingDraftTokens = params.maxDecodingTokens - 1;
        auto outputNextDraftTokens = params.outputNextDraftTokens + batchIdx * maxDecodingDraftTokens;
        auto const inputNextDraftTokens = params.inputNextDraftTokens + batchSlot * maxDecodingDraftTokens;

        for (auto ti = static_cast<SizeType32>(threadIdx.x); ti < maxDecodingDraftTokens;
             ti += static_cast<SizeType32>(blockDim.x))
        {
            outputNextDraftTokens[ti] = inputNextDraftTokens[ti];
        }

        auto const maxGenerationLength = params.maxGenerationLength[0];
        auto const numPackedMasks = divUp(params.maxDecodingTokens, 32);
        auto const outputStartId = (genIdx == 0) ? 0 : params.cumSumGenerationLengths[genIdx - 1];
        auto const numTokens = (genIdx == 0)
            ? params.cumSumGenerationLengths[0]
            : params.cumSumGenerationLengths[genIdx] - params.cumSumGenerationLengths[genIdx - 1];
        // Copy packed masks.
        // Masks are placed next to each other with offsets of cumSumGenerationLengths[bi-1]
        auto const inputPackedMask
            = params.inputSpecDecodingPackedMasks + batchSlot * numPackedMasks * params.maxDecodingTokens;
        auto outputPackedMask = params.outputSpecDecodingPackedMasks + outputStartId * numPackedMasks;
        for (auto ti = static_cast<SizeType32>(threadIdx.x); ti < numTokens * numPackedMasks;
             ti += static_cast<SizeType32>(blockDim.x))
        {
            outputPackedMask[ti] = inputPackedMask[ti];
        }

        // Copy pos offsets. Copy only for maxGenerationLength
        auto const inputPositionOffsets
            = params.inputSpecDecodingPositionOffsets + batchSlot * params.maxDecodingTokens;
        auto outputPositionOffsets = params.outputSpecDecodingPositionOffsets + genIdx * maxGenerationLength;
        for (auto ti = static_cast<SizeType32>(threadIdx.x); ti < maxGenerationLength;
             ti += static_cast<SizeType32>(blockDim.x))
        {
            outputPositionOffsets[ti] = inputPositionOffsets[ti];
        }
    }
}
} // namespace

void invokePackEagle(PackEagleParams const& params, cudaStream_t stream)
{
    SizeType32 constexpr BLOCK_SIZE = 128;
    packEagleTensors<<<params.batchSize, BLOCK_SIZE, 0, stream>>>(params);
}

namespace
{
__global__ void unpackEagleData(UnpackEagleDataParams params)
{
    auto const bid = static_cast<SizeType32>(blockIdx.x);
    auto const batchSlot = params.batchSlots[bid];

    auto const currentSequenceLength = params.outputSequenceLengths[batchSlot];
    auto const acceptedLength = params.inputAcceptedLens[bid];

    for (auto ti = static_cast<SizeType32>(threadIdx.x); ti < acceptedLength; ti += static_cast<SizeType32>(blockDim.x))
    {
        // Copy accepted tokens to the output sequence.
        params.outputIds[batchSlot * params.maxSeqLen + currentSequenceLength + ti]
            = params.inputAcceptedTokens[bid * params.maxPathLength + ti];
    }

    auto const maxDecodingDraftTokens = params.maxDecodingTokens - 1;
    for (auto ti = static_cast<SizeType32>(threadIdx.x); ti < maxDecodingDraftTokens;
         ti += static_cast<SizeType32>(blockDim.x))
    {
        // Copy next draft tokens to the slots.
        params.outputNextDraftTokens[batchSlot * maxDecodingDraftTokens + ti]
            = params.inputNextDraftTokens[bid * maxDecodingDraftTokens + ti];
        params.outputUnpackedNextDraftTokens[batchSlot * maxDecodingDraftTokens + ti]
            = params.inputNextDraftTokens[bid * maxDecodingDraftTokens + ti];
    }

    for (auto ti = static_cast<SizeType32>(threadIdx.x); ti < params.maxDecodingTokens * params.maxPathLength;
         ti += static_cast<SizeType32>(blockDim.x))
    {
        // Copy next draft paths to the slots.
        params.outputNextDraftPaths[batchSlot * params.maxDecodingTokens * params.maxPathLength + ti]
            = params.inputNextDraftPaths[bid * params.maxDecodingTokens * params.maxPathLength + ti];
    }

    if (threadIdx.x == 0)
    {
        // One thread updates sequence length.
        params.outputSequenceLengths[batchSlot] = currentSequenceLength + acceptedLength;
        // One thread sets number of accepted tokens.
        params.outputNumNewTokens[batchSlot] = acceptedLength;
        // One thread copies next draft len to slot
        params.outputNextDraftLengths[batchSlot] = params.inputNextDraftLens[bid];
        // Set next gen len to slot.
        params.outputNextGenerationLength[batchSlot] = params.inputNextDraftLens[bid] + 1;
        // Set prev draft lengths needed for kv cache rewind in variable draft len.
        params.outputPrevDraftLengths[batchSlot] = params.inputLastDraftLens[bid];
        // Set random data for draft sampling kernels.
        params.outputRandDataSample[batchSlot]
            = static_cast<float>(curand_uniform(params.inputCurandState + batchSlot));
        // Set random data for draft verification kernels.
        params.outputRandDataVerification[batchSlot]
            = static_cast<float>(curand_uniform(params.inputCurandState + batchSlot));
        // Copy temperature.
        params.outputTemperatures[batchSlot] = params.inputTemperatures[batchSlot];

        // Set ctx request type to 0.
        params.outputEagleNetCtxRequestTypes[batchSlot] = 0;
        // Set gen request type to 1.
        params.outputEagleNetGenRequestTypes[batchSlot] = 1;

        // EagleNet0 context length is at most the input that we pass to EagleNet0, which is the size of the first chunk
        // (prompt len) or chunk (max accepted tokens == maxPathLength). As this kernel is called before the generation
        // stages, it is always maxPathLength.
        params.outputEagleNetCtxContextLengths[batchSlot] = params.maxPathLength;

        auto const nextSequenceLength = currentSequenceLength + acceptedLength + params.maxPathLength;
        // EagleNetX context length is the same as the base model context length (prompt len) + number of accepted
        // tokens (at most maxPathLength).
        params.outputEagleNetGenContextLengths[batchSlot] = nextSequenceLength;

        // EagleNet0 past kv length is sequence length, which is at most nextSequenceLength;
        params.outputEagleNetCtxPastKeyValueLengths[batchSlot] = nextSequenceLength;
        // EagleNetX past kv length is sequence length - 1, which is at most nextSequenceLength - 1;
        params.outputEagleNetGenPastKeyValueLengths[batchSlot] = nextSequenceLength - 1;

        // FIXME single thread filling those might be too slow
        // Fill output ids
        params.outputPositionIds[batchSlot * params.maxDecodingTokens + 0] = 0;
        for (SizeType32 pi = 0; pi < params.maxDecodingTokens; ++pi)
        {
            for (SizeType32 li = 1; li < params.maxPathLength; ++li)
            {
                auto const index = flat_index3(bid, pi, li, params.maxDecodingTokens, params.maxPathLength);
                auto const pathIdx = params.inputNextDraftPaths[index];
                if (pathIdx != -1)
                {
                    params.outputPositionIds[batchSlot * params.maxDecodingTokens + pathIdx] = li;
                }
            }
        }
    }
}
} // namespace

void invokeUnpackEagleData(UnpackEagleDataParams const& params, cudaStream_t stream)
{
    SizeType32 constexpr BLOCK_SIZE = 128;
    unpackEagleData<<<params.batchSize, BLOCK_SIZE, 0, stream>>>(params);

    sync_check_cuda_error();
}

namespace
{
__global__ void fillContextEagleData(FillContextEagleParams params)
{
    auto const bid = static_cast<SizeType32>(blockIdx.x * blockDim.x + threadIdx.x);
    auto const batchSlot = params.batchSlots[bid];

    if (bid < params.batchSize)
    {
        // Set random data for draft sampling kernels.
        params.outputRandDataSample[batchSlot]
            = static_cast<float>(curand_uniform(params.inputCurandState + batchSlot));
        // Copy temperature.
        params.outputTemperatures[batchSlot] = params.inputTemperatures[batchSlot];
    }
}
} // namespace

void invokeFillContextEagleData(FillContextEagleParams const& params, cudaStream_t stream)
{
    SizeType32 constexpr BLOCK_SIZE = 128;
    fillContextEagleData<<<params.batchSize, BLOCK_SIZE, 0, stream>>>(params);

    sync_check_cuda_error();
}

void invokeGetPackedMaskFromPath(int32_t* specDecodingPackedMasks, SizeType32 const* batchSlots,
    SizeType32 const* nextDraftPaths, SizeType32 batchSize, SizeType32 maxDecodingTokens, SizeType32 maxPathLen,
    cudaStream_t stream)
{
    dim3 block(128);
    dim3 grid(batchSize);
    size_t shmSize = maxDecodingTokens * maxDecodingTokens * sizeof(char);
    getPackedMaskFromPath<<<grid, block, shmSize, stream>>>(
        specDecodingPackedMasks, batchSlots, nextDraftPaths, maxDecodingTokens, maxPathLen);

    sync_check_cuda_error();
}

} // namespace tensorrt_llm::kernels::speculative_decoding
