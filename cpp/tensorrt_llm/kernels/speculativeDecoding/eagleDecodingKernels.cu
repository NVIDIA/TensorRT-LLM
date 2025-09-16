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

    sync_check_cuda_error(stream);
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
    TokenIdType const* chunkedContextNextTokens, SizeType32 const* baseNetSequenceLengths,
    SizeType32 const* baseNetContextLengths, TokenIdType const* acceptedTokens, SizeType32 const* acceptedLens,
    SizeType32 const* prevDraftLens, SizeType32 const* prevPaths, SizeType32 const* bestPathIds, SizeType32 batchSize,
    SizeType32 maxPathLen, SizeType32 maxDecodingTokens, SizeType32 maxNonLeavesPerLayer)
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
        // Get next token of the chunk if not the last chunk in chunked context. Otherwise, -1.
        auto const chunkedContextNextToken = chunkedContextNextTokens[bid];

        // Sequence length of the base model (without draft len for gen requests and all prompt tokens for ctx request)
        auto const oldSequenceLength = baseNetSequenceLengths[bid] - numInputTokens;
        for (SizeType32 ti = 0; ti < numDecodingTokens; ++ti)
        {
            TokenIdType token;
            if (isContextRequest)
            {
                if (ti == numDecodingTokens - 1)
                {
                    if (chunkedContextNextToken >= 0)
                    {
                        // Last token is not newly predicted token, but the first token from the next chunk in the
                        // prompt.
                        token = chunkedContextNextToken;
                    }
                    else
                    {
                        // Last token is newly sampled token
                        token = acceptedTokens[bid * maxPathLen + 0];
                    }
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
    TokenIdType const* chunkedContextNextTokens, SizeType32 const* baseNetSequenceLengths,
    SizeType32 const* baseNetContextLengths, TokenIdType const* acceptedTokens, SizeType32 const* acceptedLens,
    SizeType32 const* prevDraftLens, SizeType32 const* prevPaths, SizeType32 const* bestPathIds, SizeType32 batchSize,
    SizeType32 maxPathLen, SizeType32 maxDecodingTokens, SizeType32 maxNonLeavesPerLayer, cudaStream_t stream)
{
    SizeType32 constexpr BLOCK_SIZE = 512;
    TLLM_CHECK_WITH_INFO(
        batchSize <= BLOCK_SIZE, "Batch size larger than %d is not supported for EAGLE yet", batchSize);
    prepareCtxEagleNetInputsKernel<BLOCK_SIZE><<<1, BLOCK_SIZE, 0, stream>>>(eagleNetSequenceLengths,
        eagleNetContextLengths, outputIds, positionIds, hiddenStatesIndices, lastTokenIndices, numLastTokenIndices,
        hiddenSizeBatchLevelStarts, inputIds, chunkedContextNextTokens, baseNetSequenceLengths, baseNetContextLengths,
        acceptedTokens, acceptedLens, prevDraftLens, prevPaths, bestPathIds, batchSize, maxPathLen, maxDecodingTokens,
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
    auto const maxGenLength = BlockReduce(tempStorage.reduce).Reduce(nextDraftLen, cuda::maximum());

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
    // Considering that the Eagle-2 tree is dynamically changing,
    // we need the logits of the newly expanded nodes instead of treating them as leaves.
    // This value is use to distinguish non-leaf nodes for Eagle-2
    auto const nonLeafSignal = maxDecodingTokens + 1;

    if (batchSlot < 0)
    {
        return;
    }

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
            if (toIndex == -1 || toIndex == nonLeafSignal)
            {
                // nonLeafSignal is just an auxiliary value and has no practical meaning.
                // There is no need to calculate a mask for this.
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

        sync_check_cuda_error(params.stream);
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

        sync_check_cuda_error(params.stream);
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

        sync_check_cuda_error(params.stream);
    }

    {
        dim3 block(32);
        dim3 grid(params.maxDecodingTokens, params.batchSize);
        size_t shmSize = params.maxDecodingTokens * sizeof(char);
        getPackedMask<<<grid, block, shmSize, params.stream>>>(params.cumSumGenerationLengths,
            params.maxGenerationLength, params.selectedMasks, params.maxDecodingTokens - 1,
            params.specDecodingPackedMasks);

        sync_check_cuda_error(params.stream);
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
    // logitsPtrs: shape [numInputLogits], each points to a [vocabSizePadded] buffer
    // logits: shape [numInputLogits, vocabSizePadded]
    // outputIdsPtrs: shape [numInputLogits], each points to a [maxDecodingDraftTokens] buffer
    // outputIds: shape [numInputLogits * maxDecodingDraftTokens]

    auto const tix = static_cast<SizeType32>(blockIdx.x * blockDim.x + threadIdx.x);
    auto const isValid{tix < numValidLogits[0]};

    if (isValid)
    {
        logitsPtrs[tix] = logits + tix * vocabSizePadded;
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

    sync_check_cuda_error(stream);
}

template void invokeAssembleDraftLogitsOffsets(float const** logitsPtrs, float const* logits,
    runtime::TokenIdType** outputIdsPtrs, runtime::TokenIdType* outputIds, bool* skipDecode,
    runtime::SizeType32 const* numValidLogits, runtime::SizeType32 numInputLogits, runtime::SizeType32 batchSize,
    runtime::SizeType32 maxDecodingDraftTokens, runtime::SizeType32 vocabSizePadded, cudaStream_t stream);

template void invokeAssembleDraftLogitsOffsets(__half const** logitsPtrs, __half const* logits,
    runtime::TokenIdType** outputIdsPtrs, runtime::TokenIdType* outputIds, bool* skipDecode,
    runtime::SizeType32 const* numValidLogits, runtime::SizeType32 numInputLogits, runtime::SizeType32 batchSize,
    runtime::SizeType32 maxDecodingDraftTokens, runtime::SizeType32 vocabSizePadded, cudaStream_t stream);

namespace
{

// Extract the number of successors for each node from path
__global__ void extractNumSuccessorsFromPath(SizeType32 const* paths, SizeType32* numSuccessorsForEachNode,
    SizeType32 layerId, SizeType32 maxDecodingTokens, SizeType32 maxPathLen)
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
        paths, numSuccessorsForEachNode, layerId, maxDecodingTokens, maxPathLen);

    sync_check_cuda_error(stream);

    TLLM_CHECK_WITH_INFO(
        batchSize <= BLOCK_SIZE, "Batch size larger than %d is not supported for EAGLE yet", BLOCK_SIZE);

    // Extract topKs from numSuccessorsForEachNode array
    extracTopKsFromSuccessorsArray<BLOCK_SIZE>
        <<<1, BLOCK_SIZE, 0, stream>>>(topKs, topKOffset, numSuccessorsForEachNode, batchSize, maxDecodingTokens);
}

namespace
{
__global__ void copyOutputTokensIds(TokenIdType const* const* tmpOutputIdsPtrs, SizeType32 const* topKs,
    SizeType32 const* topKOffset, TokenIdType const* pluginInputDraftIdsPtrs, SizeType32 const* pluginInputDraftLens,
    SizeType32 const* numValidLogits, TokenIdType* pluginOutputDraftIdsPtrs, SizeType32* pluginOutputDraftLens,
    SizeType32 layerId, SizeType32 batchSize, SizeType32 maxDecodingDraftTokens, SizeType32 const* inputPaths,
    SizeType32* outputPaths, SizeType32 maxPathLen)
{
    // tmpOutputIdsPtrs: shape [numInputLogits][maxDecodingDraftTokens]
    // topKs: shape [numInputLogits]
    // topKOffset: shape [batchSize]
    // pluginInputDraftIdsPtrs: shape [batchSize][maxDecodingDraftTokens]
    // pluginInputDraftLens: shape [batchSize]
    // pluginOutputDraftIdsPtrs: shape [batchSize][maxDecodingDraftTokens]
    // pluginOutputDraftLens: shape [batchSize]
    // inputPaths: shape [batchSize][maxDecodingTokens][maxPathLen]
    // outputPaths: shape [batchSize][maxDecodingTokens][maxPathLen]

    auto const tix = static_cast<SizeType32>(blockIdx.x * blockDim.x + threadIdx.x);
    if (tix < batchSize)
    {
        // Output draft token ids offset
        TokenIdType* curPluginOutputDraftIdsPtrs = pluginOutputDraftIdsPtrs + tix * maxDecodingDraftTokens;
        TokenIdType const* indicescurPluginInputDraftIdsPtrs = pluginInputDraftIdsPtrs + tix * maxDecodingDraftTokens;

        // The length of the existing draft token
        SizeType32 prevLen = layerId == 0 ? 0 : pluginInputDraftLens[tix];

        // Copy exist tokens
        for (SizeType32 ii = 0; ii < prevLen; ii++)
        {
            curPluginOutputDraftIdsPtrs[ii] = indicescurPluginInputDraftIdsPtrs[ii];
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

        // Copy paths from input to output
        auto const maxDecodingTokens = maxDecodingDraftTokens + 1;
        auto inputPathsPtr = inputPaths + tix * maxDecodingTokens * maxPathLen;
        auto outputPathsPtr = outputPaths + tix * maxDecodingTokens * maxPathLen;
        for (SizeType32 ii = 0; ii < maxDecodingTokens * maxPathLen; ii++)
        {
            outputPathsPtr[ii] = inputPathsPtr[ii];
        }
    }
}

} // namespace

// Copy output draft token ids from temporary buffer to plugin output buffer, also update the draft token length
void invokeCopyOutputTokensIds(runtime::TokenIdType const* const* tmpOutputIdsPtrs, runtime::SizeType32 const* topKs,
    runtime::SizeType32 const* topKOffset, runtime::TokenIdType const* pluginInputDraftIdsPtrs,
    runtime::SizeType32 const* pluginInputDraftLens, runtime::SizeType32 const* numValidLogits,
    runtime::TokenIdType* pluginOutputDraftIdsPtrs, runtime::SizeType32* pluginOutputDraftLens,
    runtime::SizeType32 layerId, runtime::SizeType32 batchSize, runtime::SizeType32 maxDecodingDraftTokens,
    runtime::SizeType32 const* inputPaths, runtime::SizeType32* outputPaths, runtime::SizeType32 maxPathLen,
    cudaStream_t stream)
{
    SizeType32 constexpr BLOCK_SIZE = 512;

    copyOutputTokensIds<<<divUp(batchSize, BLOCK_SIZE), BLOCK_SIZE, 0, stream>>>(tmpOutputIdsPtrs, topKs, topKOffset,
        pluginInputDraftIdsPtrs, pluginInputDraftLens, numValidLogits, pluginOutputDraftIdsPtrs, pluginOutputDraftLens,
        layerId, batchSize, maxDecodingDraftTokens, inputPaths, outputPaths, maxPathLen);
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

        // 0 for ctx request and actual draft len for gen requests.
        params.outputNextDraftLens[batchIdx]
            = isGenerationRequest ? params.inputSpecDecodingGenerationLengths[batchSlot] - 1 : 0;
    }

    for (auto ti = static_cast<SizeType32>(threadIdx.x); ti < params.maxDecodingTokens;
         ti += static_cast<SizeType32>(blockDim.x))
    {
        params.outputRandomDataValidation[batchIdx * params.maxDecodingTokens + ti]
            = params.inputRandomDataValidation[batchSlot * params.maxDecodingTokens + ti];
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

    if (batchSlot < 0)
    {
        return;
    }

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
        for (SizeType32 ti = 0; ti < params.maxDecodingTokens; ++ti)
        {
            // Set random data for draft verification kernels.
            params.outputRandDataVerification[batchSlot * params.maxDecodingTokens + ti]
                = static_cast<float>(curand_uniform(params.inputCurandState + batchSlot));
        }
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

    sync_check_cuda_error(stream);
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

    sync_check_cuda_error(stream);
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

    sync_check_cuda_error(stream);
}

namespace
{
template <int BLOCK_SIZE>
__global__ void augmentBatchSlotsKernel(SizeType32* augmentedSeqSlots, SizeType32 const* chunkedContextNextTokens,
    SizeType32 const* lastDraftLens, SizeType32 const* seqSlots, SizeType32 engineBatchSize)
{
    auto const batchIdx = static_cast<SizeType32>(threadIdx.x);
    auto const valid = batchIdx < engineBatchSize;

    if (valid)
    {
        auto const draftLen = lastDraftLens[batchIdx];
        auto const needDecoding = (draftLen == 0 && chunkedContextNextTokens[batchIdx] == -1) || (draftLen > 0);
        augmentedSeqSlots[batchIdx] = needDecoding ? seqSlots[batchIdx] : -1;
    }
}
} // namespace

void invokeAugmentBatchSlots(SizeType32* augmentedSeqSlots, runtime::SizeType32 const* chunkedContextNextTokens,
    runtime::SizeType32 const* lastDraftLens, SizeType32 const* seqSlots, SizeType32 engineBatchSize,
    SizeType32 batchSize, cudaStream_t stream)
{
    SizeType32 constexpr BLOCK_SIZE = 512;
    TLLM_CHECK_WITH_INFO(
        engineBatchSize <= BLOCK_SIZE, "Batch size larger than %d is not supported for EAGLE yet", batchSize);
    augmentBatchSlotsKernel<BLOCK_SIZE><<<1, BLOCK_SIZE, 0, stream>>>(
        augmentedSeqSlots, chunkedContextNextTokens, lastDraftLens, seqSlots, engineBatchSize);
}

namespace
{
__global__ void setTopKsFromDyanmicTreeMaxTopK(SizeType32 layerIdx, SizeType32 batchSize, SizeType32* topKs,
    SizeType32* topKOffset, SizeType32 const dynamicTreeMaxTopK, SizeType32 const* numValidLogits)
{
    // topKs: shape [numInputLogits]
    // topKOffset: shape [batchSize]

#ifdef TLLM_DEBUG_MODE
    // Check the value
    if (layerIdx == 0 && !(batchSize == numValidLogits[0]))
    {
        printf("When layerIdx == 0, batchsize(%d) should be the same as numValidLogits(%d)\n", batchSize,
            numValidLogits[0]);
        asm volatile("brkpt;\n");
    }
    else if (layerIdx > 0 && !(batchSize * dynamicTreeMaxTopK == numValidLogits[0]))
    {
        printf("When layerIdx > 0, batchsize(%d) * dynamicTreeMaxTopK(%d) should be the same as numValidLogits(%d)\n",
            batchSize, dynamicTreeMaxTopK, numValidLogits[0]);
        asm volatile("brkpt;\n");
    }
#endif // TLLM_DEBUG_MODE

    auto const tix = static_cast<SizeType32>(blockIdx.x * blockDim.x + threadIdx.x);
    if (tix < numValidLogits[0])
    {
        // Set topKs
        // In Eagle-2, all logits have the same topK
        topKs[tix] = dynamicTreeMaxTopK;
    }

    if (tix < batchSize)
    {
        // Set topKOffset
        topKOffset[tix] = layerIdx == 0 ? tix : tix * dynamicTreeMaxTopK;
    }
}
} // namespace

void invokeSetTopKsFromDyanmicTreeMaxTopK(SizeType32 layerIdx, SizeType32 batchSize, SizeType32 numInputLogits,
    SizeType32* topKs, SizeType32* topKOffset, SizeType32 const dynamicTreeMaxTopK, SizeType32 const* numValidLogits,
    cudaStream_t stream)
{
    SizeType32 constexpr BLOCK_SIZE = 128;
    setTopKsFromDyanmicTreeMaxTopK<<<divUp(numInputLogits, BLOCK_SIZE), BLOCK_SIZE, 0, stream>>>(
        layerIdx, batchSize, topKs, topKOffset, dynamicTreeMaxTopK, numValidLogits);

    sync_check_cuda_error(stream);
}

namespace
{
__global__ void copyScoresAndDraftTokenIds(SizeType32 layerIdx, SizeType32 mNumEagleLayers,
    SizeType32 maxDecodingDraftTokens, SizeType32 batchSize, SizeType32 const dynamicTreeMaxTopK,
    TokenIdType const* pluginInputCurrentExpandIndices, float const* pluginInputAllLayersScores,
    TokenIdType const* pluginInputAllLayersDraftTokenIds,
    TokenIdType const* pluginInputAllLayersDraftTokenIdsPredecessor, float* pluginOutputAllLayersScores,
    TokenIdType* pluginOutputAllLayersDraftTokenIds, TokenIdType* pluginOutputAllLayersDraftTokenIdsPredecessor,
    float* firstTopKOutputLogProbs, TokenIdType* firstTopKOutputIds)
{
    // topKOffset: [batchSize]
    // pluginInputCurrentExpandIndices: [batchSize, maxDecodingDraftTokens]
    // pluginInputAllLayersScores: [batchSize, mNumEagleLayers, maxDecodingDraftTokens x maxDecodingDraftTokens]
    // pluginInputAllLayersDraftTokenIds: [batchSize, mNumEagleLayers, maxDecodingDraftTokens x maxDecodingDraftTokens]
    // pluginInputAllLayersDraftTokenIdsPredecessor: [batchSize, mNumEagleLayers, maxDecodingDraftTokens x
    // maxDecodingDraftTokens]

    // pluginOutputAllLayersScores: [batchSize, mNumEagleLayers, maxDecodingDraftTokens x maxDecodingDraftTokens]
    // pluginOutputAllLayersDraftTokenIds: [batchSize, mNumEagleLayers, maxDecodingDraftTokens x maxDecodingDraftTokens]
    // pluginOutputAllLayersDraftTokenIdsPredecessor: [batchSize, mNumEagleLayers, maxDecodingDraftTokens x
    // maxDecodingDraftTokens]

    // firstTopKOutputLogProbs: [numInputLogits, maxDecodingDraftTokens]
    // firstTopKOutputIds: [numInputLogits, maxDecodingDraftTokens]

    auto const bix = static_cast<SizeType32>(blockIdx.x * blockDim.x + threadIdx.x);
    if (bix < batchSize)
    {
        auto pluginInputCurrentExpandIndicesPtr = pluginInputCurrentExpandIndices + bix * maxDecodingDraftTokens;
        auto pluginInputAllLayersScoresPtr
            = pluginInputAllLayersScores + bix * mNumEagleLayers * maxDecodingDraftTokens * maxDecodingDraftTokens;
        auto pluginInputAllLayersDraftTokenIdsPtr = pluginInputAllLayersDraftTokenIds
            + bix * mNumEagleLayers * maxDecodingDraftTokens * maxDecodingDraftTokens;
        auto pluginInputAllLayersDraftTokenIdsPredecessorPtr = pluginInputAllLayersDraftTokenIdsPredecessor
            + bix * mNumEagleLayers * maxDecodingDraftTokens * maxDecodingDraftTokens;

        auto pluginOutputAllLayersScoresPtr
            = pluginOutputAllLayersScores + bix * mNumEagleLayers * maxDecodingDraftTokens * maxDecodingDraftTokens;
        auto pluginOutputAllLayersDraftTokenIdsPtr = pluginOutputAllLayersDraftTokenIds
            + bix * mNumEagleLayers * maxDecodingDraftTokens * maxDecodingDraftTokens;
        auto pluginOutputAllLayersDraftTokenIdsPredecessorPtr = pluginOutputAllLayersDraftTokenIdsPredecessor
            + bix * mNumEagleLayers * maxDecodingDraftTokens * maxDecodingDraftTokens;

        // When layerIdx == 0, firstTopKOutputLogProbs/firstTopKOutputIds shape: [batchSize, maxDecodingDraftTokens]
        // When layerIdx > 0, firstTopKOutputLogProbs/firstTopKOutputIds shape: [batchSize * dynamicTreeMaxTopK,
        // maxDecodingDraftTokens]
        auto firstTopKOutputOffset = layerIdx == 0 ? 1 : dynamicTreeMaxTopK;
        auto firstTopKOutputLogProbsPtr
            = firstTopKOutputLogProbs + bix * firstTopKOutputOffset * maxDecodingDraftTokens;
        auto firstTopKOutputIdsPtr = firstTopKOutputIds + bix * firstTopKOutputOffset * maxDecodingDraftTokens;

        // We save the scores and draft tokensIds continuously
        auto startOffset
            = layerIdx == 0 ? 0 : (layerIdx - 1) * (dynamicTreeMaxTopK * dynamicTreeMaxTopK) + dynamicTreeMaxTopK;

        // 1) Copy all the previous scores and draft tokenIds from plugin input to plugin output
        for (SizeType32 ii = 0; ii < startOffset; ++ii)
        {
            pluginOutputAllLayersScoresPtr[ii] = pluginInputAllLayersScoresPtr[ii];
            pluginOutputAllLayersDraftTokenIdsPtr[ii] = pluginInputAllLayersDraftTokenIdsPtr[ii];
            pluginOutputAllLayersDraftTokenIdsPredecessorPtr[ii] = pluginInputAllLayersDraftTokenIdsPredecessorPtr[ii];
        }

        // 2) Copy this layer's scores and draft tokenIds
        // When layerIdx == 0, we only need to save dynamicTreeMaxTopK scores/draft tokens
        // When layerIdx > 0, we need to save dynamicTreeMaxTopK * dynamicTreeMaxTopK scores/draft tokens
        auto numExpandTokens = layerIdx == 0 ? 1 : dynamicTreeMaxTopK;
        for (SizeType32 ii = 0; ii < numExpandTokens; ++ii)
        {
            for (SizeType32 jj = 0; jj < dynamicTreeMaxTopK; ++jj)
            {
                pluginOutputAllLayersScoresPtr[startOffset]
                    = firstTopKOutputLogProbsPtr[ii * maxDecodingDraftTokens + jj];
                pluginOutputAllLayersDraftTokenIdsPtr[startOffset]
                    = firstTopKOutputIdsPtr[ii * maxDecodingDraftTokens + jj];

                // Update the predecessor of this draft tokens
                pluginOutputAllLayersDraftTokenIdsPredecessorPtr[startOffset]
                    = layerIdx == 0 ? 0 : pluginInputCurrentExpandIndicesPtr[ii];
                startOffset++;
            }
        }
    }
}

} // namespace

void invokeCopyScoresAndDraftTokenIds(SizeType32 layerIdx, SizeType32 mNumEagleLayers,
    SizeType32 maxDecodingDraftTokens, SizeType32 batchSize, SizeType32 const dynamicTreeMaxTopK,
    TokenIdType const* pluginInputCurrentExpandIndices, float const* pluginInputAllLayersScores,
    TokenIdType const* pluginInputAllLayersDraftTokenIds,
    TokenIdType const* pluginInputAllLayersDraftTokenIdsPredecessor, float* pluginOutputAllLayersScores,
    TokenIdType* pluginOutputAllLayersDraftTokenIds, TokenIdType* pluginOutputAllLayersDraftTokenIdsPredecessor,
    float* firstTopKOutputLogProbs, TokenIdType* firstTopKOutputIds, cudaStream_t stream)
{
    SizeType32 constexpr BLOCK_SIZE = 128;
    copyScoresAndDraftTokenIds<<<divUp(batchSize, BLOCK_SIZE), BLOCK_SIZE, 0, stream>>>(layerIdx, mNumEagleLayers,
        maxDecodingDraftTokens, batchSize, dynamicTreeMaxTopK, pluginInputCurrentExpandIndices,
        pluginInputAllLayersScores, pluginInputAllLayersDraftTokenIds, pluginInputAllLayersDraftTokenIdsPredecessor,
        pluginOutputAllLayersScores, pluginOutputAllLayersDraftTokenIds, pluginOutputAllLayersDraftTokenIdsPredecessor,
        firstTopKOutputLogProbs, firstTopKOutputIds);

    sync_check_cuda_error(stream);
}

namespace
{
__global__ void updateScores(SizeType32 batchSize, SizeType32 const dynamicTreeMaxTopK,
    SizeType32 maxDecodingDraftTokens, float* curLogProbs, float const* prevLayerScores)
{
    // We update the current scores (curLogProbs, shape: [batchSize * dynamicTreeMaxTopK, maxDecodingDraftTokens])
    // with the previous layer's scores (prevLayerScores, shape: [batchSize, maxDecodingDraftTokens])
    // 'cu_scores = topk_p + scores[:, None]'
    // Example: when topk_p = [[1, 2], [3, 4]], scores = [10, 20].
    //          cu_scores = [[11, 12], [23, 24]]

    // curLogProbs [numInputLogits(batchSize * dynamicTreeMaxTopK), maxDecodingDraftTokens]
    // prevLayerScores [batchSize, maxDecodingDraftTokens]. for each request, only top 'dynamicTreeMaxTopK' is valuable.
    auto const bix = static_cast<SizeType32>(blockIdx.x * blockDim.x + threadIdx.x);
    if (bix < batchSize)
    {
        // This request's buffer
        auto prevLayerScoresPtr = prevLayerScores + bix * maxDecodingDraftTokens;
        auto curLogProbsPtr = curLogProbs + bix * dynamicTreeMaxTopK * maxDecodingDraftTokens;

        for (SizeType32 ii = 0; ii < dynamicTreeMaxTopK; ++ii)
        {
            auto curDraftTokenLogProbsPtr = curLogProbsPtr + ii * maxDecodingDraftTokens;
            auto scoreValue = prevLayerScoresPtr[ii];
            for (SizeType32 jj = 0; jj < maxDecodingDraftTokens; ++jj)
            {
                if (jj < dynamicTreeMaxTopK)
                {
                    curDraftTokenLogProbsPtr[jj] += scoreValue;
                }
                else
                {
                    curDraftTokenLogProbsPtr[jj] = -std::numeric_limits<float>::infinity();
                }
            }
        }
    }
}

} // namespace

void invokeUpdateScores(SizeType32 batchSize, SizeType32 const dynamicTreeMaxTopK, SizeType32 maxDecodingDraftTokens,
    float* curLogProbs, float const* prevLayerScores, cudaStream_t stream)
{
    SizeType32 constexpr BLOCK_SIZE = 128;
    updateScores<<<divUp(batchSize, BLOCK_SIZE), BLOCK_SIZE, 0, stream>>>(
        batchSize, dynamicTreeMaxTopK, maxDecodingDraftTokens, curLogProbs, prevLayerScores);

    sync_check_cuda_error(stream);
}

namespace
{
__global__ void assembleSecondTopKSamplingInputs(SizeType32 batchSize, SizeType32 const dynamicTreeMaxTopK,
    SizeType32 maxDecodingDraftTokens, float* firstTopKOutputLogProbs, float** secondTopKInputScoresPtrs,
    TokenIdType* secondTopKOutputIdsFlatten, TokenIdType** secondTopKOutputIdsPtrs)
{
    // firstTopKOutputLogProbs: shape [numInputLogits(batchSize * dynamicTreeMaxTopK), maxDecodingDraftTokens]
    // secondTopKInputScoresPtrs: shape [batchSize]
    // secondTopKOutputIdsFlatten: shape [batchSize, maxDecodingDraftTokens]
    // secondTopKOutputIdsPtrs: shape [batchSize]
    auto const bix = static_cast<SizeType32>(blockIdx.x * blockDim.x + threadIdx.x);
    if (bix < batchSize)
    {
        secondTopKInputScoresPtrs[bix] = firstTopKOutputLogProbs + bix * dynamicTreeMaxTopK * maxDecodingDraftTokens;
        secondTopKOutputIdsPtrs[bix] = secondTopKOutputIdsFlatten + bix * maxDecodingDraftTokens;
    }
}

} // namespace

void invokeAssembleSecondTopKSamplingInputs(SizeType32 batchSize, SizeType32 const dynamicTreeMaxTopK,
    SizeType32 maxDecodingDraftTokens, float* firstTopKOutputLogProbs, float** secondTopKInputScoresPtrs,
    TokenIdType* secondTopKOutputIdsFlatten, TokenIdType** secondTopKOutputIdsPtrs, cudaStream_t stream)
{

    SizeType32 constexpr BLOCK_SIZE = 128;
    assembleSecondTopKSamplingInputs<<<divUp(batchSize, BLOCK_SIZE), BLOCK_SIZE, 0, stream>>>(batchSize,
        dynamicTreeMaxTopK, maxDecodingDraftTokens, firstTopKOutputLogProbs, secondTopKInputScoresPtrs,
        secondTopKOutputIdsFlatten, secondTopKOutputIdsPtrs);

    sync_check_cuda_error(stream);
}

// The outputIds are almost ascending
inline __device__ void insertionSortOutputIds(TokenIdType* outputIds, SizeType32 n)
{
    for (SizeType32 ii = 1; ii < n; ++ii)
    {
        TokenIdType key = outputIds[ii];
        SizeType32 jj = ii - 1;

        while (jj >= 0 && outputIds[jj] > key)
        {
            outputIds[jj + 1] = outputIds[jj];
            jj--;
        }
        outputIds[jj + 1] = key;
    }
}

inline __device__ SizeType32 findAncestorPathIndex(SizeType32 const* prevPaths, SizeType32 ancestorId,
    SizeType32 ancestorLayerIdx, SizeType32 maxDecodingTokens, SizeType32 maxPathLen)
{
    if (ancestorLayerIdx == -1)
    {
        // Find the root layer
        return 0;
    }

    // prevPaths: [maxDecodingTokens, maxPathLen]
    for (SizeType32 ii = 0; ii < maxDecodingTokens; ++ii)
    {
        // '+1' because of the root node
        if (prevPaths[ii * maxPathLen + ancestorLayerIdx + 1] == ancestorId)
        {
            return ii;
        }
    }
    return -1;
}

namespace
{

__global__ void updatePath(SizeType32 layerIdx, SizeType32 batchSize, SizeType32 dynamicTreeMaxTopK,
    SizeType32 maxDecodingTokens, SizeType32 maxPathLen, SizeType32 const* prevPaths, SizeType32* newPaths,
    TokenIdType** secondTopKOutputIdsPtrs, TokenIdType* pluginOutputNextExpandIndices)
{
    // prevPaths: [batchSize, maxDecodingTokens, maxPathLen]
    // newPaths: [batchSize, maxDecodingTokens, maxPathLen]
    // secondTopKOutputIdsPtrs: [batchSize, maxDecodingDraftTokens]
    // pluginOutputNextExpandIndices: [batchSize, maxDecodingDraftTokens]

    auto const maxDecodingDraftTokens = maxDecodingTokens - 1;
    auto const bix = static_cast<SizeType32>(blockIdx.x * blockDim.x + threadIdx.x);
    // Considering that the Eagle-2 tree is dynamically changing,
    // we need the logits of the newly expanded nodes instead of treating them as leaves.
    // This value is use to distinguish non-leaf nodes for Eagle-2.
    auto const nonLeafSignal = maxDecodingTokens + 1;

    if (bix < batchSize)
    {
        auto const prevPathPtr = prevPaths + bix * maxDecodingTokens * maxPathLen;
        auto const newPathsPtr = newPaths + bix * maxDecodingTokens * maxPathLen;
        auto const pluginOutputNextExpandIndicesPtr = pluginOutputNextExpandIndices + bix * maxDecodingDraftTokens;

        // Init, all set to -1
        for (SizeType32 ii = 0; ii < maxDecodingTokens * maxPathLen; ++ii)
        {
            newPathsPtr[ii] = -1;
        }

        if (layerIdx == 0)
        {
            // layer 0 is simple
            // Example new paths: [[0, 1, -1, -1], [0, 2, -1, -1], ..., [0, dynamicTreeMaxTopK, -1, -1]]
            for (SizeType32 ii = 0; ii < dynamicTreeMaxTopK; ++ii)
            {
                newPathsPtr[ii * maxPathLen + 0] = 0;
                newPathsPtr[ii * maxPathLen + 1] = ii + 1;
                // Append nonLeafSignal
                newPathsPtr[ii * maxPathLen + 2] = nonLeafSignal;

                // When layerIdx == 0, only expand 'dynamicTreeMaxTopK' draft tokens
                // We '+1' here because we take the root node into consideration
                pluginOutputNextExpandIndicesPtr[ii] = ii + 1;
            }
        }
        else
        {
            // Find how many paths in the previous path
            SizeType32 prevLayerNumPaths = 0;
            for (SizeType32 ii = 0; ii < maxDecodingTokens; ++ii)
            {
                // Check the first value of each paths
                if (prevPathPtr[ii * maxPathLen + 0] != -1)
                {
                    prevLayerNumPaths++;
                }
                else
                {
                    break;
                }
            }

            auto secondTopKOutputIdsPtr = secondTopKOutputIdsPtrs[bix];

            // For each request, we will generate 'dynamicTreeMaxTopK' new draft tokens
            // auto newDraftTokensIdsPtr = secondTopKOutputIdsPtrs[bix];
            // Sort the outputIds, ascending
            insertionSortOutputIds(secondTopKOutputIdsPtr, dynamicTreeMaxTopK);

            // Update the selected draft tokens to the pluginOutputNextExpandIndices
            // Exclude the root node
            SizeType32 offsetToTheFinalTree = layerIdx == 1
                ? dynamicTreeMaxTopK + 1
                : (layerIdx - 1) * dynamicTreeMaxTopK * dynamicTreeMaxTopK + dynamicTreeMaxTopK + 1;
            for (SizeType32 ii = 0; ii < dynamicTreeMaxTopK; ++ii)
            {
                SizeType32 rowIdx = secondTopKOutputIdsPtr[ii] / maxDecodingDraftTokens;
                SizeType32 columnIdx = secondTopKOutputIdsPtr[ii] % maxDecodingDraftTokens;

                pluginOutputNextExpandIndicesPtr[ii] = rowIdx * dynamicTreeMaxTopK + columnIdx + offsetToTheFinalTree;
            }

            // The start index of the node in this layer
            auto const startIndexOfCurrentLayer = layerIdx * dynamicTreeMaxTopK + 1;
            // The start index of the node in previous layer
            auto const startIndexOfPreviousLayer = startIndexOfCurrentLayer - dynamicTreeMaxTopK;

            // Record the index of path that had been used to expand in this layer
            SizeType32 usedPrevLayerPathsIndex = -1;

            SizeType32 numNewPath = 0;
            for (SizeType32 ii = 0; ii < dynamicTreeMaxTopK; ++ii)
            {
                // This draft token's new index in the whole tree
                SizeType32 newIndex = ii + startIndexOfCurrentLayer;
                // Find this draft token's ancestor node
                SizeType32 ancestorIndex
                    = secondTopKOutputIdsPtr[ii] / maxDecodingDraftTokens + startIndexOfPreviousLayer;

                // Find the path index in the previous path that take this ancestor node as the leaf node
                SizeType32 ancestorPathIdxInPrevPaths
                    = findAncestorPathIndex(prevPathPtr, ancestorIndex, layerIdx - 1, maxDecodingTokens, maxPathLen);

                // The correct 'ancestorPathIdxInPrevPaths' must be:
                // 1) ancestorPathIdxInPrevPaths == usedPrevLayerPathsIndex: continue to expand this path
                // 2) ancestorPathIdxInPrevPaths > usedPrevLayerPathsIndex:
                //       2.1) ancestorPathIdxInPrevPaths == usedPrevLayerPathsIndex + 1: expand a new path,
                //            next to the previous one.
                //       2.2) ancestorPathIdxInPrevPaths > usedPrevLayerPathsIndex + 1: there are multiple
                //            path that do not have leaf at this layer, but we need to include as well.
#ifdef TLLM_DEBUG_MODE
                if (ancestorPathIdxInPrevPaths == -1 || ancestorPathIdxInPrevPaths < usedPrevLayerPathsIndex)
                {
                    // Throw error when can not find ancestor's path.
                    // Or ancestorPath had been finish expand.
                    printf(
                        "Throw error from updatePath kernel: bix: %d can not find the correct ancestorPath of "
                        "ancestorIndex: %d in layerIdx: %d, usedPrevLayerPathsIndex: %d, "
                        "ancestorPathIdxInPrevPaths:%d\n",
                        bix, ancestorIndex, layerIdx - 1, usedPrevLayerPathsIndex, ancestorPathIdxInPrevPaths);
                    asm volatile("brkpt;\n");
                }
#endif // TLLM_DEBUG_MODE

                if (ancestorPathIdxInPrevPaths == usedPrevLayerPathsIndex + 1)
                {
                    // Expand a new path, just behind the previous one.
                    usedPrevLayerPathsIndex++;
                }
                else if (ancestorPathIdxInPrevPaths > usedPrevLayerPathsIndex + 1)
                {
                    // There are multiple path that will not be expand in this layer.
                    // But we also need to include them, since they are part of the tree.
                    while (ancestorPathIdxInPrevPaths > usedPrevLayerPathsIndex + 1)
                    {
                        // Insert the paths that do not have leaf in this layer
                        usedPrevLayerPathsIndex++;
                        // We do not need to copy the whole paths (i.e., maxPathLen) that do not expand in this layer.
                        // We only need to copy top 'layerIdx + 1' value for these path.
                        // The paths that do not expand will have 'layerIdx + 1' valid steps at most.
                        // '+1' is for the root node.
                        // This prevents us from copying nonLeafSignal as well.
                        for (SizeType32 jj = 0; jj <= layerIdx; jj++)
                        {
                            newPathsPtr[numNewPath * maxPathLen + jj]
                                = prevPathPtr[usedPrevLayerPathsIndex * maxPathLen + jj];
                        }
                        numNewPath++;
                    }
                    usedPrevLayerPathsIndex++; // Point to the path that we will expand this time.
                }

                // Expand the path
                // Copy the original path
                for (SizeType32 jj = 0; jj <= layerIdx; ++jj)
                {
                    newPathsPtr[numNewPath * maxPathLen + jj]
                        = prevPathPtr[ancestorPathIdxInPrevPaths * maxPathLen + jj];
                }
                // Add this layer's new draft token
                newPathsPtr[numNewPath * maxPathLen + layerIdx + 1] = newIndex;
                // Append the nonLeafSignal
                // 'layerIdx + 1 + 1' is always small than maxPathLen,
                // because the last layer of EagleNet will not execute the logic here.
                // Example: numEagleNets = 4, maxPathLen = 5, layerIdx [0, 3]
                // The layerIdx range that will execute this logic is [1, 2]
                newPathsPtr[numNewPath * maxPathLen + layerIdx + 1 + 1] = nonLeafSignal;
                numNewPath++;
            }

            // Insert the paths that do not have leaf in this layer
            while (usedPrevLayerPathsIndex < prevLayerNumPaths)
            {
                usedPrevLayerPathsIndex++;
                for (SizeType32 jj = 0; jj <= layerIdx; jj++)
                {
                    newPathsPtr[numNewPath * maxPathLen + jj] = prevPathPtr[usedPrevLayerPathsIndex * maxPathLen + jj];
                }
                numNewPath++;
            }
        }
    }
}

} // namespace

void invokeUpdatePath(SizeType32 layerIdx, SizeType32 batchSize, SizeType32 dynamicTreeMaxTopK,
    SizeType32 maxDecodingTokens, SizeType32 maxPathLen, SizeType32 const* prevPaths, SizeType32* newPaths,
    TokenIdType** secondTopKOutputIdsPtrs, TokenIdType* pluginOutputNextExpandIndices, cudaStream_t stream)
{
    SizeType32 constexpr BLOCK_SIZE = 128;

    updatePath<<<divUp(batchSize, BLOCK_SIZE), BLOCK_SIZE, 0, stream>>>(layerIdx, batchSize, dynamicTreeMaxTopK,
        maxDecodingTokens, maxPathLen, prevPaths, newPaths, secondTopKOutputIdsPtrs, pluginOutputNextExpandIndices);

    sync_check_cuda_error(stream);
}

namespace
{

__global__ void updateDraftTokensAndLensAndCurScores(SizeType32 layerIdx, SizeType32 batchSize,
    SizeType32 dynamicTreeMaxTopK, SizeType32 maxDecodingDraftTokens, TokenIdType const* const* curDraftIds,
    TokenIdType const* pluginInputDraftIds, SizeType32 const* pluginInputDraftLens, TokenIdType* pluginOutputDraftIds,
    SizeType32* pluginOutputDraftLens, float const* curLayerScores, float* pluginOutputCurrentScores)
{
    // curDraftIds: shape [batchSize][maxDecodingDraftTokens]
    // pluginInputDraftIds: shape [batchSize, maxDecodingDraftTokens]
    // pluginInputDraftLens: shape [batchSize]
    // pluginOutputDraftIds: shape [batchSize, maxDecodingDraftTokens]
    // pluginOutputDraftLens: shape [batchSize]
    // curLayerScores: shape [batchSize, maxDecodingDraftTokens]
    // pluginOutputCurrentScores: shape [batchSize, maxDecodingDraftTokens]

    auto const bix = static_cast<SizeType32>(blockIdx.x * blockDim.x + threadIdx.x);
    if (bix < batchSize)
    {
        // 1) Update draft tokenIds and draft lengths
        // Output draft token ids offset
        TokenIdType* curPluginOutputDraftIdsPtr = pluginOutputDraftIds + bix * maxDecodingDraftTokens;
        TokenIdType const* indicescurPluginInputDraftIdsPtr = pluginInputDraftIds + bix * maxDecodingDraftTokens;

        // The length of the existing draft token
        SizeType32 prevLen = layerIdx == 0 ? 0 : pluginInputDraftLens[bix];

        // Copy exist tokens
        for (SizeType32 ii = 0; ii < prevLen; ii++)
        {
            curPluginOutputDraftIdsPtr[ii] = indicescurPluginInputDraftIdsPtr[ii];
        }

        SizeType32 curLen = prevLen;

        SizeType32 startTopKOffset = bix;
        SizeType32 endTopkOffset = bix + 1;

        for (SizeType32 ii = startTopKOffset; ii < endTopkOffset; ii++)
        {
            for (SizeType32 jj = 0; jj < dynamicTreeMaxTopK; jj++)
            {
                curPluginOutputDraftIdsPtr[curLen] = curDraftIds[ii][jj];
                curLen++;
            }
        }

        // Update the output draft token length of this request
        pluginOutputDraftLens[bix] = curLen;

        // 2) Update this layer's scores
        auto const* curLayerScoresPtr = curLayerScores + bix * maxDecodingDraftTokens;
        auto pluginOutputCurrentScoresPtr = pluginOutputCurrentScores + bix * maxDecodingDraftTokens;
        for (SizeType32 ii = 0; ii < maxDecodingDraftTokens; ii++)
        {
            pluginOutputCurrentScoresPtr[ii] = curLayerScoresPtr[ii];
        }
    }
}
} // namespace

void invokeUpdateDraftTokensAndLensAndCurScores(SizeType32 layerIdx, SizeType32 batchSize,
    SizeType32 dynamicTreeMaxTopK, SizeType32 maxDecodingDraftTokens, TokenIdType const* const* curDraftIds,
    TokenIdType const* pluginInputDraftIds, SizeType32 const* pluginInputDraftLens, TokenIdType* pluginOutputDraftIds,
    SizeType32* pluginOutputDraftLens, float const* curLayerScores, float* pluginOutputCurrentScores,
    cudaStream_t stream)
{
    SizeType32 constexpr BLOCK_SIZE = 128;
    updateDraftTokensAndLensAndCurScores<<<divUp(batchSize, BLOCK_SIZE), BLOCK_SIZE, 0, stream>>>(layerIdx, batchSize,
        dynamicTreeMaxTopK, maxDecodingDraftTokens, curDraftIds, pluginInputDraftIds, pluginInputDraftLens,
        pluginOutputDraftIds, pluginOutputDraftLens, curLayerScores, pluginOutputCurrentScores);

    sync_check_cuda_error(stream);
}

namespace
{
__global__ void extractScoresAndRealDraftTokensIds(SizeType32 batchSize, SizeType32 dynamicTreeMaxTopK,
    SizeType32 maxDecodingDraftTokens, float const* const* secondTopKInputScoresPtrs,
    TokenIdType* const* secondTopKOutputIdsPtrs, TokenIdType* firstTopKOutputIds, float* secondTopKOutputLogProbs)
{
    // secondTopKInputScoresPtrs: shape [batchSize][dynamicTreeMaxTopK * maxDecodingDraftTokens]
    // secondTopKOutputIdsPtrs: shape [batchSize][maxDecodingDraftTokens]
    // firstTopKOutputIds: shape [batchSize * dynamicTreeMaxTopK * maxDecodingDraftTokens]
    // secondTopKOutputLogProbs: shape [batchSize, maxDecodingDraftTokens]

    auto const bix = static_cast<SizeType32>(blockIdx.x * blockDim.x + threadIdx.x);
    if (bix < batchSize)
    {

        auto secondTopKOutputLogProbsPtr = secondTopKOutputLogProbs + bix * maxDecodingDraftTokens;
        auto firstTopKOutputIdsPtr = firstTopKOutputIds + bix * dynamicTreeMaxTopK * maxDecodingDraftTokens;
        for (SizeType32 ii = 0; ii < dynamicTreeMaxTopK; ii++)
        {
            // auto selectIndex = secondTopKOutputIdsPtrs[bix][ii];
            // auto selectScore = secondTopKInputScoresPtrs[bix][selectIndex];
            // secondTopKOutputLogProbsPtr[ii] = selectScore;

            auto row = secondTopKOutputIdsPtrs[bix][ii] / maxDecodingDraftTokens;
            auto column = secondTopKOutputIdsPtrs[bix][ii] % maxDecodingDraftTokens;

            // Extract scores
            secondTopKOutputLogProbsPtr[ii] = secondTopKInputScoresPtrs[bix][row * maxDecodingDraftTokens + column];
            // Extract real draft tokenIds
            secondTopKOutputIdsPtrs[bix][ii] = firstTopKOutputIdsPtr[row * maxDecodingDraftTokens + column];
        }
    }
}

} // namespace

void invokeExtractScoresAndRealDraftTokensIds(SizeType32 batchSize, SizeType32 dynamicTreeMaxTopK,
    SizeType32 maxDecodingDraftTokens, float const* const* secondTopKInputScoresPtrs,
    TokenIdType* const* secondTopKOutputIdsPtrs, TokenIdType* firstTopKOutputIds, float* secondTopKOutputLogProbs,
    cudaStream_t stream)
{
    SizeType32 constexpr BLOCK_SIZE = 128;
    extractScoresAndRealDraftTokensIds<<<divUp(batchSize, BLOCK_SIZE), BLOCK_SIZE, 0, stream>>>(batchSize,
        dynamicTreeMaxTopK, maxDecodingDraftTokens, secondTopKInputScoresPtrs, secondTopKOutputIdsPtrs,
        firstTopKOutputIds, secondTopKOutputLogProbs);

    sync_check_cuda_error(stream);
}

namespace
{

__global__ void assembleThridTopKSamplingInputs(SizeType32 batchSize, SizeType32 maxDecodingDraftTokens,
    SizeType32 mNumEagleLayers, SizeType32 const maxNodesOnFinalTree, SizeType32* thirdTopKs,
    float* pluginOutputAllLayersScores, float** thirdTopKInputScoresPtrs, TokenIdType* thirdTopKOutputIds,
    TokenIdType** thirdTopKOutputIdsPtrs)
{
    // pluginOutputAllLayersScores: [batchSize, mNumEagleLayers, maxDecodingDraftTokens x maxDecodingDraftTokens]
    // thirdTopKInputScoresPtrs: [batchSize]
    // thirdTopKOutputIds: [batchSize, maxDecodingDraftTokens]
    // thirdTopKOutputIdsPtrs: [batchSize]

    auto const bix = static_cast<SizeType32>(blockIdx.x * blockDim.x + threadIdx.x);
    if (bix < batchSize)
    {
        thirdTopKInputScoresPtrs[bix]
            = pluginOutputAllLayersScores + bix * mNumEagleLayers * maxDecodingDraftTokens * maxDecodingDraftTokens;
        thirdTopKOutputIdsPtrs[bix] = thirdTopKOutputIds + bix * maxDecodingDraftTokens;
        thirdTopKs[bix] = maxNodesOnFinalTree;
    }
}

} // namespace

void invokeAssembleThridTopKSamplingInputs(SizeType32 batchSize, SizeType32 maxDecodingDraftTokens,
    SizeType32 mNumEagleLayers, SizeType32 const maxNodesOnFinalTree, SizeType32* thirdTopKs,
    float* pluginOutputAllLayersScores, float** thirdTopKInputScoresPtrs, TokenIdType* thirdTopKOutputIds,
    TokenIdType** thirdTopKOutputIdsPtrs, cudaStream_t stream)
{
    SizeType32 constexpr BLOCK_SIZE = 128;
    assembleThridTopKSamplingInputs<<<divUp(batchSize, BLOCK_SIZE), BLOCK_SIZE, 0, stream>>>(batchSize,
        maxDecodingDraftTokens, mNumEagleLayers, maxNodesOnFinalTree, thirdTopKs, pluginOutputAllLayersScores,
        thirdTopKInputScoresPtrs, thirdTopKOutputIds, thirdTopKOutputIdsPtrs);

    sync_check_cuda_error(stream);
}

namespace
{

__device__ SizeType32 findIndexInPaths(
    SizeType32* indexMap, SizeType32 maxDecodingTokens, SizeType32 indexAmongAllDraftTokens)
{
    for (SizeType32 ii = 0; ii < maxDecodingTokens; ++ii)
    {
        if (indexMap[ii] == indexAmongAllDraftTokens)
        {
            return ii;
        }
    }
    return -1;
}

__global__ void reconstructFinalPath(SizeType32 batchSize, SizeType32 const dynamicTreeMaxTopK,
    SizeType32 maxDecodingDraftTokens, SizeType32 maxDecodingTokens, SizeType32 maxPathLen, SizeType32 mNumEagleLayers,
    SizeType32 const maxNodesOnFinalTree, TokenIdType* const* thirdTopKOutputIdsPtrs,
    TokenIdType* pluginOutputAllLayersDraftTokenIdsPredecessor, SizeType32* finalOutputPaths)
{
    // thirdTopKOutputIdsPtrs: shape [batchSize], each element points to a [maxDecodingDraftTokens] buffer
    // pluginOutputAllLayersDraftTokenIdsPredecessor:
    // [batchSize, mNumEagleLayers, maxDecodingDraftTokens * maxDecodingDraftTokens]
    // finalOutputPaths: [batchSize, maxDecodingTokens, maxPathLen]

    auto const bix = static_cast<SizeType32>(blockIdx.x * blockDim.x + threadIdx.x);

    extern __shared__ SizeType32 totalSmemPtr[];

    // shape: [2 * batchSize * maxDecodingTokens, maxPathLen]
    // '2' means ping-pong buffers
    SizeType32* tempPingPongPaths = totalSmemPtr;

    // shape: [batchSize * maxDecodingTokens]
    SizeType32* indexMapPtr = totalSmemPtr + 2 * batchSize * maxDecodingTokens * maxPathLen;

    SizeType32* tempSmemPtr[2];

    if (bix < batchSize)
    {
        SizeType32* curIndexMap = indexMapPtr + bix * maxDecodingDraftTokens;
        // Init, all set to '-1'
        for (SizeType32 ii = 0; ii < maxDecodingTokens; ++ii)
        {
            curIndexMap[ii] = -1;
        }

        // Store the pointers to buffer
        tempSmemPtr[0] = tempPingPongPaths + bix * maxDecodingTokens * maxPathLen;
        tempSmemPtr[1] = tempPingPongPaths + (batchSize + bix) * maxDecodingTokens * maxPathLen;

        // Init, all set to '-1'
        for (SizeType32 ii = 0; ii < maxDecodingTokens * maxPathLen; ++ii)
        {
            tempSmemPtr[0][ii] = -1;
            tempSmemPtr[1][ii] = -1;
        }

        // Offset to batch index
        auto thirdTopKOutputIdsPtr = thirdTopKOutputIdsPtrs[bix];
        auto pluginOutputAllLayersDraftTokenIdsPredecessorPtr = pluginOutputAllLayersDraftTokenIdsPredecessor
            + bix * mNumEagleLayers * maxDecodingDraftTokens * maxDecodingDraftTokens;
        auto finalOutputPathsPtr = finalOutputPaths + bix * maxDecodingTokens * maxPathLen;

        // Sort the output draft token indices in ascending order
        insertionSortOutputIds(thirdTopKOutputIdsPtr, maxNodesOnFinalTree);

        // Init the root node
        SizeType32* rooPathPtr = tempSmemPtr[0];
        SizeType32 curLayerSmemPtrIndex = 0;
        rooPathPtr[0] = 0;
        SizeType32 curLayerNumPaths = 1; // The path number of this layer
        curIndexMap[0] = 0;

        SizeType32 curTopKIndex = 0; // The index of the thirdTopKOutputIdsPtr
        // Update the path layer by layer
        for (SizeType32 li = 0; li < mNumEagleLayers; li++)
        {
            // Update previous layer's path information
            SizeType32* prevLayerPathsPtr = tempSmemPtr[curLayerSmemPtrIndex];
            SizeType32 prevLayerNumPaths = curLayerNumPaths;

            curLayerSmemPtrIndex = (curLayerSmemPtrIndex + 1) % 2;            // Update curLayerSmemPtrIndex
            SizeType32* curLayerPathsPtr = tempSmemPtr[curLayerSmemPtrIndex]; // Point to this layer's path buffer
            curLayerNumPaths = 0; // Reset the number of path of current layer

            // Record the index of path that had been used to expand in this layer
            SizeType32 usedPrevLayerPathsIndex = -1;

            // The index boundary of this layer.
            // The 'index' is the output index after top-maxDecodingDraftTokens sampling.
            SizeType32 curLayerStartIndex
                = li == 0 ? 0 : (li - 1) * dynamicTreeMaxTopK * dynamicTreeMaxTopK + dynamicTreeMaxTopK;
            SizeType32 curLayerEndIndex = li == 0 ? curLayerStartIndex + dynamicTreeMaxTopK
                                                  : curLayerStartIndex + dynamicTreeMaxTopK * dynamicTreeMaxTopK;

            // curProcessNode is the node index select from top-maxDecodingDraftTokens sampling (i.e., the third
            // sampling)
            SizeType32 curProcessNode = thirdTopKOutputIdsPtr[curTopKIndex];

            while (curProcessNode < curLayerEndIndex) // This node belong to this layer
            {
                // The pluginOutputAllLayersDraftTokenIdsPredecessorPtr does not consider root node, so we need to '-1'
                // Find the ancestor index of the current process node among all the history draft tokens
                SizeType32 ancestorIdxAmongAllDraftTokens
                    = pluginOutputAllLayersDraftTokenIdsPredecessorPtr[curProcessNode];
                // Map the index from the index among all the history draft tokens to the index in path
                SizeType32 ancestorIdxInPath
                    = findIndexInPaths(curIndexMap, maxDecodingTokens, ancestorIdxAmongAllDraftTokens);

                // Find the path that end with 'ancestorIdxInPath'
                SizeType32 ancestorPathIdxInPrevPaths = findAncestorPathIndex(
                    prevLayerPathsPtr, ancestorIdxInPath, li - 1, maxDecodingTokens, maxPathLen);

#ifdef TLLM_DEBUG_MODE
                if (ancestorPathIdxInPrevPaths == -1)
                {
                    printf(
                        "Throw error from reconstructFinalPath kernel: bix: %d can not find the correct ancestorPath "
                        "of ancestorIdxAmongAllDraftTokens: %d, ancestorIdxInPath: %d in layerIdx: %d, "
                        "usedPrevLayerPathsIndex: %d, ancestorPathIdxInPrevPaths: %d\n",
                        bix, ancestorIdxAmongAllDraftTokens, ancestorIdxInPath, li - 1, usedPrevLayerPathsIndex,
                        ancestorPathIdxInPrevPaths);
                    asm volatile("brkpt;\n");
                }
#endif // TLLM_DEBUG_MODE

                if (ancestorPathIdxInPrevPaths == usedPrevLayerPathsIndex + 1)
                {
                    usedPrevLayerPathsIndex++;
                }
                else if (ancestorPathIdxInPrevPaths > usedPrevLayerPathsIndex + 1)
                {
                    // There are some paths that do not expand in this layer
                    while (ancestorPathIdxInPrevPaths > usedPrevLayerPathsIndex + 1)
                    {
                        // Insert the paths that do not have leaf in this layer
                        usedPrevLayerPathsIndex++;
                        for (SizeType32 jj = 0; jj < maxPathLen; jj++)
                        {
                            curLayerPathsPtr[curLayerNumPaths * maxPathLen + jj]
                                = prevLayerPathsPtr[usedPrevLayerPathsIndex * maxPathLen + jj];
                        }
                        curLayerNumPaths++;
                    }
                    usedPrevLayerPathsIndex++;
                }

                // Expand this node
                for (SizeType32 jj = 0; jj < maxPathLen; jj++)
                {
                    curLayerPathsPtr[curLayerNumPaths * maxPathLen + jj]
                        = prevLayerPathsPtr[ancestorPathIdxInPrevPaths * maxPathLen + jj];
                }
                curLayerPathsPtr[curLayerNumPaths * maxPathLen + li + 1]
                    = curTopKIndex + 1; // '+1' is because the root node
                curLayerNumPaths++;

                // Update indexMap
                curIndexMap[curTopKIndex + 1]
                    = curProcessNode + 1; // '+1' because we will take root node into consideration

                // Point to next top-maxDecodingDraftTokens node
                curTopKIndex++;
                if (curTopKIndex >= maxNodesOnFinalTree)
                {
                    break;
                }
                curProcessNode = thirdTopKOutputIdsPtr[curTopKIndex];
            } // Finish the expand of this layer

            // Insert the paths that do not have leaf in this layer
            while (usedPrevLayerPathsIndex < prevLayerNumPaths)
            {
                usedPrevLayerPathsIndex++;
                for (SizeType32 jj = 0; jj < maxPathLen; jj++)
                {
                    curLayerPathsPtr[curLayerNumPaths * maxPathLen + jj]
                        = prevLayerPathsPtr[usedPrevLayerPathsIndex * maxPathLen + jj];
                }
                curLayerNumPaths++;
            }

            if (curTopKIndex >= maxNodesOnFinalTree)
            {
                break;
            }
        } // Finish all the layers

#ifdef TLLM_DEBUG_MODE
        if (curTopKIndex != maxNodesOnFinalTree)
        {
            printf(
                "Throw error from reconstructFinalPath kernel: curTopKIndex(%d) is not the same as "
                "maxNodesOnFinalTree - 1(%d - 1)",
                curTopKIndex, maxNodesOnFinalTree - 1);
            asm volatile("brkpt;\n");
        }
#endif // TLLM_DEBUG_MODE

        // Copy the final paths from shared memory to global memory
        SizeType32* smemPathsPtr = tempSmemPtr[curLayerSmemPtrIndex];
        for (SizeType32 ii = 0; ii < maxDecodingTokens * maxPathLen; ii++)
        {
            finalOutputPathsPtr[ii] = smemPathsPtr[ii];
        }
    }
}

} // namespace

void invokeReconstructFinalPath(SizeType32 batchSize, SizeType32 const dynamicTreeMaxTopK,
    SizeType32 maxDecodingDraftTokens, SizeType32 maxDecodingTokens, SizeType32 maxPathLen, SizeType32 mNumEagleLayers,
    SizeType32 const maxNodesOnFinalTree, TokenIdType* const* thirdTopKOutputIdsPtrs,
    TokenIdType* pluginOutputAllLayersDraftTokenIdsPredecessor, SizeType32* newPaths, cudaStream_t stream)
{
    SizeType32 constexpr BLOCK_SIZE = 32;
    // Use ping-pong temporary buffers to update path
    SizeType32 pingPongBufferSize = 2 * batchSize * maxDecodingTokens * maxPathLen * sizeof(SizeType32);

    // Although we have pluginOutputAllLayersDraftTokenIdsPredecessor, but the indices are correspond to all the history
    // draft tokens. During we select draft tokens into the path, the tokens' index starts from 1 and increases. We need
    // a map from the index among all the history to the index in the constructed path.
    SizeType32 indexMapSize = batchSize * maxDecodingTokens * sizeof(SizeType32);

    SizeType32 smemSize = pingPongBufferSize + indexMapSize;

    reconstructFinalPath<<<divUp(batchSize, BLOCK_SIZE), BLOCK_SIZE, smemSize, stream>>>(batchSize, dynamicTreeMaxTopK,
        maxDecodingDraftTokens, maxDecodingTokens, maxPathLen, mNumEagleLayers, maxNodesOnFinalTree,
        thirdTopKOutputIdsPtrs, pluginOutputAllLayersDraftTokenIdsPredecessor, newPaths);

    sync_check_cuda_error(stream);
}

namespace
{
__global__ void copyFinalDraftTokens(SizeType32 batchSize, SizeType32 maxDecodingDraftTokens,
    SizeType32 mNumEagleLayers, SizeType32 const maxNodesOnFinalTree, TokenIdType const* const* thirdTopKOutputIdsPtrs,
    TokenIdType* pluginOutputAllLayersDraftTokenIds, TokenIdType* pluginOutputDraftTokenIds,
    SizeType32* pluginOutputDraftLens)
{
    // thirdTopKOutputIdsPtrs: shape [batchSize], each points to [maxDecodingDraftTokens]
    // pluginOutputAllLayersDraftTokenIds: shape [batchSize, mNumEagleLayers, maxDecodingDraftTokens x
    // maxDecodingDraftTokens] pluginOutputDraftTokenIds: [batchSize, maxDecodingDraftTokens] pluginOutputDraftLens:
    // [batchSize]
    auto const bix = static_cast<SizeType32>(blockIdx.x * blockDim.x + threadIdx.x);

    if (bix < batchSize)
    {
        // Update final selected draft tokenIds
        auto thirdTopKOutputIdsPtr = thirdTopKOutputIdsPtrs[bix];
        auto pluginOutputAllLayersDraftTokenIdsPtr = pluginOutputAllLayersDraftTokenIds
            + bix * mNumEagleLayers * maxDecodingDraftTokens * maxDecodingDraftTokens;
        auto pluginOutputDraftTokenIdsPtr = pluginOutputDraftTokenIds + bix * maxDecodingDraftTokens;

        for (SizeType32 ii = 0; ii < maxNodesOnFinalTree; ++ii)
        {
            SizeType32 selectedNodeIndex = thirdTopKOutputIdsPtr[ii];
            TokenIdType realDraftTokenId = pluginOutputAllLayersDraftTokenIdsPtr[selectedNodeIndex];
            pluginOutputDraftTokenIdsPtr[ii] = realDraftTokenId;
        }

        // Update draft length according to the 'maxNodesOnFinalTree'
        pluginOutputDraftLens[bix] = maxNodesOnFinalTree;
    }
}

} // namespace

void invokeCopyFinalDraftTokens(SizeType32 batchSize, SizeType32 maxDecodingDraftTokens, SizeType32 mNumEagleLayers,
    SizeType32 const maxNodesOnFinalTree, TokenIdType const* const* thirdTopKOutputIdsPtrs,
    TokenIdType* pluginOutputAllLayersDraftTokenIds, TokenIdType* pluginOutputDraftTokenIds,
    SizeType32* pluginOutputDraftLens, cudaStream_t stream)
{
    SizeType32 constexpr BLOCK_SIZE = 128;
    copyFinalDraftTokens<<<divUp(batchSize, BLOCK_SIZE), BLOCK_SIZE, 0, stream>>>(batchSize, maxDecodingDraftTokens,
        mNumEagleLayers, maxNodesOnFinalTree, thirdTopKOutputIdsPtrs, pluginOutputAllLayersDraftTokenIds,
        pluginOutputDraftTokenIds, pluginOutputDraftLens);

    sync_check_cuda_error(stream);
}

} // namespace tensorrt_llm::kernels::speculative_decoding
