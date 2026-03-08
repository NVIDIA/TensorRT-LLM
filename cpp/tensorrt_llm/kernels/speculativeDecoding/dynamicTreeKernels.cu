/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
 * Portions Copyright (c) 2025 by SGLang team (original implementation).
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

#include "dynamicTreeKernels.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/memoryUtils.h"

TRTLLM_NAMESPACE_BEGIN

using namespace tensorrt_llm::runtime;

namespace kernels::speculative_decoding
{
// Build dynamic tree: construct treeMask, positions, and left-child-right-sibling linked list.
// tid==0 builds the linked list; tid>0 traces ancestors to compute depth and mask.
__global__ void buildDynamicTreeKernel(int64_t const* parentList, int64_t const* selectedIndex,
    SizeType32 const* verifiedSeqLen, int32_t* treeMask, int32_t* positions, int32_t* retrieveIndex,
    int32_t* retrieveNextToken, int32_t* retrieveNextSibling, SizeType32 topK, SizeType32 depth,
    SizeType32 draftTokenNum)
{
    int32_t bid = blockIdx.x;
    int32_t tid = threadIdx.x;

    if (tid >= draftTokenNum)
    {
        return;
    }

    int32_t seqLen = verifiedSeqLen[bid];

    int32_t tokenTreeIdx = draftTokenNum * draftTokenNum * bid + draftTokenNum * tid + 1;

    // Init mask: self-visible (root column), clear rest
    treeMask[tokenTreeIdx - 1] = 1;
    for (int32_t i = 0; i < draftTokenNum - 1; i++)
    {
        treeMask[tokenTreeIdx + i] = 0;
    }

    int32_t position = 0;

    if (tid == 0)
    {
        // Root thread: build left-child-right-sibling linked list (reverse order for correct sibling ordering)
        positions[bid * draftTokenNum] = seqLen;

        for (int32_t i = draftTokenNum - 1; i > 0; --i)
        {
            retrieveIndex[bid * draftTokenNum + i] = i;

            int64_t parentTbIdx = selectedIndex[bid * (draftTokenNum - 1) + i - 1] / topK;
            int32_t parentPosition = 0;

            if (parentTbIdx > 0)
            {
                int64_t parentTokenIdx = parentList[bid * (topK * (depth - 1) + 1) + parentTbIdx];
                for (; parentPosition < draftTokenNum; ++parentPosition)
                {
                    if (selectedIndex[bid * (draftTokenNum - 1) + parentPosition] == parentTokenIdx)
                    {
                        ++parentPosition;
                        break;
                    }
                }
            }

            if (parentPosition == draftTokenNum)
            {
                printf(
                    "WARNING: Invalid dynamic tree! Detected a token with no parent token selected. "
                    "Please check if the logprob has nan. The token will be ignored.\n");
                continue;
            }

            if (retrieveNextToken[bid * draftTokenNum + parentPosition] == -1)
            {
                retrieveNextToken[bid * draftTokenNum + parentPosition] = i;
            }
            else
            {
                int32_t originNextToken = retrieveNextToken[bid * draftTokenNum + parentPosition];
                retrieveNextToken[bid * draftTokenNum + parentPosition] = i;
                retrieveNextSibling[bid * draftTokenNum + i] = originNextToken;
            }
        }
        retrieveIndex[bid * draftTokenNum] = 0;
    }
    else
    {
        // Non-root threads: trace ancestors to compute depth and set mask bits
        int32_t curPosition = tid - 1;
        while (true)
        {
            position += 1;
            treeMask[tokenTreeIdx + curPosition] = 1;

            int64_t parentTbIdx = selectedIndex[bid * (draftTokenNum - 1) + curPosition] / topK;
            if (parentTbIdx == 0)
            {
                break;
            }

            int64_t tokenIdx = parentList[bid * (topK * (depth - 1) + 1) + parentTbIdx];
            for (curPosition = 0; curPosition < draftTokenNum; ++curPosition)
            {
                if (selectedIndex[bid * (draftTokenNum - 1) + curPosition] == tokenIdx)
                {
                    break;
                }
            }
        }
        positions[bid * draftTokenNum + tid] = position + seqLen;
    }
}

// Bit-packed version of buildDynamicTreeKernel.
// treeMask: [bs, draftTokenNum, ceil(draftTokenNum/32)] with each int32 storing 32 mask bits.
__global__ void buildDynamicTreeKernelPacked(int64_t const* parentList, int64_t const* selectedIndex,
    SizeType32 const* verifiedSeqLen, int32_t* treeMask, int32_t* positions, int32_t* retrieveIndex,
    int32_t* retrieveNextToken, int32_t* retrieveNextSibling, SizeType32 topK, SizeType32 depth,
    SizeType32 draftTokenNum, SizeType32 numInt32PerRow)
{
    int32_t bid = blockIdx.x;
    int32_t tid = threadIdx.x;

    if (tid >= draftTokenNum)
    {
        return;
    }

    int32_t seqLen = verifiedSeqLen[bid];

    int32_t rowBaseIdx = (bid * draftTokenNum + tid) * numInt32PerRow;
    treeMask[rowBaseIdx] = 1; // Every node sees root (bit 0)

    int32_t position = 0;

    if (tid == 0)
    {
        // Root thread: build linked list (same logic as unpacked version)
        positions[bid * draftTokenNum] = seqLen;

        for (int32_t i = draftTokenNum - 1; i > 0; --i)
        {
            retrieveIndex[bid * draftTokenNum + i] = i;

            int64_t parentTbIdx = selectedIndex[bid * (draftTokenNum - 1) + i - 1] / topK;
            int32_t parentPosition = 0;

            if (parentTbIdx > 0)
            {
                int64_t parentTokenIdx = parentList[bid * (topK * (depth - 1) + 1) + parentTbIdx];
                for (; parentPosition < draftTokenNum; ++parentPosition)
                {
                    if (selectedIndex[bid * (draftTokenNum - 1) + parentPosition] == parentTokenIdx)
                    {
                        ++parentPosition;
                        break;
                    }
                }
            }

            if (parentPosition == draftTokenNum)
            {
                printf("WARNING: Invalid dynamic tree! Detected a token with no parent token selected.\n");
                continue;
            }

            if (retrieveNextToken[bid * draftTokenNum + parentPosition] == -1)
            {
                retrieveNextToken[bid * draftTokenNum + parentPosition] = i;
            }
            else
            {
                int32_t originNextToken = retrieveNextToken[bid * draftTokenNum + parentPosition];
                retrieveNextToken[bid * draftTokenNum + parentPosition] = i;
                retrieveNextSibling[bid * draftTokenNum + i] = originNextToken;
            }
        }
        retrieveIndex[bid * draftTokenNum] = 0;
    }
    else
    {
        // Non-root threads: trace ancestors, set mask bits via atomicOr
        int32_t curPosition = tid - 1;
        while (true)
        {
            position += 1;

            int32_t bitPosition = curPosition + 1; // bit 0 = root, already set
            int32_t int32Idx = bitPosition / 32;
            int32_t bitIdx = bitPosition % 32;
            if (int32Idx < numInt32PerRow)
            {
                atomicOr(&treeMask[rowBaseIdx + int32Idx], 1 << bitIdx);
            }

            int64_t parentTbIdx = selectedIndex[bid * (draftTokenNum - 1) + curPosition] / topK;
            if (parentTbIdx == 0)
            {
                break;
            }

            int64_t tokenIdx = parentList[bid * (topK * (depth - 1) + 1) + parentTbIdx];
            for (curPosition = 0; curPosition < draftTokenNum; ++curPosition)
            {
                if (selectedIndex[bid * (draftTokenNum - 1) + curPosition] == tokenIdx)
                {
                    break;
                }
            }
        }
        positions[bid * draftTokenNum + tid] = position + seqLen;
    }
}

void invokeBuildDynamicTree(int64_t const* parentList, int64_t const* selectedIndex, SizeType32 const* verifiedSeqLen,
    void* treeMask, int32_t* positions, int32_t* retrieveIndex, int32_t* retrieveNextToken,
    int32_t* retrieveNextSibling, SizeType32 batchSize, SizeType32 topK, SizeType32 depth, SizeType32 numDraftTokens,
    TreeMaskMode treeMaskMode, cudaStream_t stream)
{
    dim3 grid(batchSize);
    dim3 block(numDraftTokens);

    if (treeMaskMode == TreeMaskMode::QLEN_ONLY_BITPACKING)
    {
        SizeType32 numInt32PerRow = (numDraftTokens + 31) / 32;

        buildDynamicTreeKernelPacked<<<grid, block, 0, stream>>>(parentList, selectedIndex, verifiedSeqLen,
            static_cast<int32_t*>(treeMask), positions, retrieveIndex, retrieveNextToken, retrieveNextSibling, topK,
            depth, numDraftTokens, numInt32PerRow);
    }
    else
    {
        buildDynamicTreeKernel<<<grid, block, 0, stream>>>(parentList, selectedIndex, verifiedSeqLen,
            static_cast<int32_t*>(treeMask), positions, retrieveIndex, retrieveNextToken, retrieveNextSibling, topK,
            depth, numDraftTokens);
    }

    sync_check_cuda_error(stream);
}

// Greedy verification: traverse the left-child-right-sibling tree layer by layer,
// accept draft tokens matching target predictions, output longest accepted path + bonus token.
// Single-threaded per batch element (block(1)).
__global__ void verifyDynamicTreeGreedyKernel(int64_t* predicts, int64_t* acceptIndex, int64_t* acceptTokenNum,
    int64_t const* candidates, int32_t const* retrieveIndex, int32_t const* retrieveNextToken,
    int32_t const* retrieveNextSibling, int64_t const* targetPredict, uint32_t batchSize, uint32_t numSpeculativeTokens,
    uint32_t numDraftTokens)
{
    uint32_t bx = blockIdx.x;

    uint32_t batchOffset = bx * numDraftTokens;

    int32_t lastAcceptedLocalIdx = retrieveIndex[batchOffset];
    acceptIndex[bx * numSpeculativeTokens] = lastAcceptedLocalIdx;
    uint32_t numAcceptedTokens = 0;
    int32_t curIndex = 0;

    for (uint32_t j = 1; j < numSpeculativeTokens; ++j)
    {
        curIndex = retrieveNextToken[batchOffset + curIndex];
        while (curIndex != -1)
        {
            int32_t draftLocalIdx = retrieveIndex[batchOffset + curIndex];
            int64_t draftTokenId = candidates[batchOffset + curIndex];
            int64_t targetTokenId = targetPredict[batchOffset + lastAcceptedLocalIdx];
            if (draftTokenId == targetTokenId)
            {
                predicts[batchOffset + lastAcceptedLocalIdx] = targetTokenId;
                ++numAcceptedTokens;
                acceptIndex[bx * numSpeculativeTokens + numAcceptedTokens] = draftLocalIdx;
                lastAcceptedLocalIdx = draftLocalIdx;
                break;
            }
            else
            {
                curIndex = retrieveNextSibling[batchOffset + curIndex];
            }
        }
        if (curIndex == -1)
            break;
    }

    acceptTokenNum[bx] = numAcceptedTokens;
    predicts[batchOffset + lastAcceptedLocalIdx] = targetPredict[batchOffset + lastAcceptedLocalIdx]; // bonus token
}

void invokeVerifyDynamicTreeGreedy(int64_t* predicts, int64_t* acceptIndex, int64_t* acceptTokenNum,
    int64_t const* candidates, int32_t const* retrieveIndex, int32_t const* retrieveNextToken,
    int32_t const* retrieveNextSibling, int64_t const* targetPredict, SizeType32 batchSize, SizeType32 numDraftTokens,
    SizeType32 numSpecStep, cudaStream_t stream)
{
    dim3 grid(batchSize);
    dim3 block(1);

    verifyDynamicTreeGreedyKernel<<<grid, block, 0, stream>>>(predicts, acceptIndex, acceptTokenNum, candidates,
        retrieveIndex, retrieveNextToken, retrieveNextSibling, targetPredict, batchSize, numSpecStep, numDraftTokens);

    sync_check_cuda_error(stream);
}

} // namespace kernels::speculative_decoding

TRTLLM_NAMESPACE_END
