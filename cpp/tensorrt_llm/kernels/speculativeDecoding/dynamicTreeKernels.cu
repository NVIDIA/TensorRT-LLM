/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
TRTLLM_NAMESPACE_BEGIN

using namespace tensorrt_llm::runtime;

namespace kernels::speculative_decoding
{

//! \param parentList           [in]  layer-wise parent indices [bs, topK*(depth-1)+1]
//! \param selectedIndex        [in]  resampled history buffer indices [bs, draftTokenNum-1]
//! \param treeMask             [out] attention mask (which nodes each node can see)
//! \param positions            [out] position id per node [bs, draftTokenNum]
//! \param retrieveIndex        [out] tree node -> local index mapping [bs, draftTokenNum]
//! \param retrieveNextToken    [out] first-child pointer [bs, draftTokenNum], -1=none
//! \param retrieveNextSibling  [out] next-sibling pointer [bs, draftTokenNum], -1=none
//! \param topK                 top-K value per layer
//! \param depth                max tree depth (number of draft layers)
//! \param draftTokenNum        total tree nodes per batch (including root)
__global__ void buildDynamicTreeKernel(int64_t const* parentList, int64_t const* selectedIndex, int32_t* treeMask,
    int32_t* positions, int32_t* retrieveIndex, int32_t* retrieveNextToken, int32_t* retrieveNextSibling,
    SizeType32 topK, SizeType32 depth, SizeType32 draftTokenNum)
{
    int32_t bid = blockIdx.x;
    int32_t tid = threadIdx.x;

    if (tid >= draftTokenNum)
    {
        return;
    }

    // treeMask layout: [batchSize, draftTokenNum, draftTokenNum] (QLEN_ONLY mode)
    int32_t tokenTreeIdx = draftTokenNum * draftTokenNum * bid + draftTokenNum * tid + 1;

    treeMask[tokenTreeIdx - 1] = 1; // self-attention diagonal
    for (int32_t i = 0; i < draftTokenNum - 1; i++)
    {
        treeMask[tokenTreeIdx + i] = 0;
    }

    int32_t position = 0;

    if (tid == 0)
    {
        positions[bid * draftTokenNum] = 0;

        // Reverse iteration: inserting at list head produces forward sibling order
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
                        ++parentPosition; // +1 because position 0 is root
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
        // Walk up to root, setting treeMask ancestor bits and counting depth
        int32_t curPosition = tid - 1;
        while (position < depth + 1)
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
            if (curPosition == draftTokenNum)
            {
                break;
            }
        }
        positions[bid * draftTokenNum + tid] = position;
    }
}

//! Bit-packed variant of buildDynamicTreeKernel.
//! \param numInt32PerRow  int32 count per treeMask row (buffer stride; >= ceil(draftTokenNum/32) if padded)
__global__ void buildDynamicTreeKernelPacked(int64_t const* parentList, int64_t const* selectedIndex, int32_t* treeMask,
    int32_t* positions, int32_t* retrieveIndex, int32_t* retrieveNextToken, int32_t* retrieveNextSibling,
    SizeType32 topK, SizeType32 depth, SizeType32 draftTokenNum, SizeType32 numInt32PerRow)
{
    int32_t bid = blockIdx.x;
    int32_t tid = threadIdx.x;

    if (tid >= draftTokenNum)
    {
        return;
    }

    int32_t rowBaseIdx = (bid * draftTokenNum + tid) * numInt32PerRow;

    treeMask[rowBaseIdx] = 1; // bit 0 = root, always visible

    int32_t position = 0;

    if (tid == 0)
    {
        positions[bid * draftTokenNum] = 0;

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
        int32_t curPosition = tid - 1;
        while (position < depth + 1)
        {
            position += 1;

            int32_t bitPosition = curPosition + 1; // +1 because bit 0 is root
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
            if (curPosition == draftTokenNum)
            {
                break;
            }
        }
        positions[bid * draftTokenNum + tid] = position;
    }
}

void invokeBuildDynamicTree(int64_t const* parentList, int64_t const* selectedIndex, void* treeMask, int32_t* positions,
    int32_t* retrieveIndex, int32_t* retrieveNextToken, int32_t* retrieveNextSibling, SizeType32 batchSize,
    SizeType32 topK, SizeType32 depth, SizeType32 numDraftTokens, TreeMaskMode treeMaskMode, cudaStream_t stream,
    SizeType32 numInt32PerRow)
{
    dim3 grid(batchSize);
    dim3 block(numDraftTokens);

    if (treeMaskMode == TreeMaskMode::QLEN_ONLY_BITPACKING)
    {
        TLLM_CHECK_WITH_INFO(
            numInt32PerRow > 0, "numInt32PerRow must be the packed treeMask row stride in int32s (from buffer shape).");
        buildDynamicTreeKernelPacked<<<grid, block, 0, stream>>>(parentList, selectedIndex,
            static_cast<int32_t*>(treeMask), positions, retrieveIndex, retrieveNextToken, retrieveNextSibling, topK,
            depth, numDraftTokens, numInt32PerRow);
    }
    else
    {
        buildDynamicTreeKernel<<<grid, block, 0, stream>>>(parentList, selectedIndex, static_cast<int32_t*>(treeMask),
            positions, retrieveIndex, retrieveNextToken, retrieveNextSibling, topK, depth, numDraftTokens);
    }

    sync_check_cuda_error(stream);
}

//! \param predicts             [out] accepted token ids + bonus token [bs * numDraftTokens]
//! \param acceptIndex          [out] accepted path as local tree positions [bs, numSpeculativeTokens]
//! \param acceptTokenNum       [out] number of accepted draft tokens per batch [bs]
//! \param candidates           [in]  candidate token id per tree node [bs, numDraftTokens]
//! \param retrieveIndex        [in]  tree node -> local index [bs, numDraftTokens]
//! \param retrieveNextToken    [in]  first-child pointer [bs, numDraftTokens], -1=none
//! \param retrieveNextSibling  [in]  next-sibling pointer [bs, numDraftTokens], -1=none
//! \param targetPredict        [in]  target model prediction per position [bs * numDraftTokens]
//! \param batchSize            batch size
//! \param numSpeculativeTokens second dim of acceptIndex/acceptToken
//!                             (= numSpecStep = max_path_len, >= max possible accepts + 1)
//! \param numDraftTokens       total tree nodes per batch (including root)
__global__ void verifyDynamicTreeGreedyKernel(int64_t* predicts, int64_t* acceptIndex, int64_t* acceptTokenNum,
    int64_t* acceptToken, int64_t const* candidates, int32_t const* retrieveIndex, int32_t const* retrieveNextToken,
    int32_t const* retrieveNextSibling, int64_t const* targetPredict, bool const* treeValid, uint32_t batchSize,
    uint32_t numSpeculativeTokens, uint32_t numDraftTokens)
{
    uint32_t bx = blockIdx.x;
    uint32_t batchOffset = bx * numDraftTokens;

    // First-gen or dummy request: no valid tree, accept only the bonus token
    if (treeValid != nullptr && !treeValid[bx])
    {
        acceptTokenNum[bx] = 0;
        acceptIndex[bx * numSpeculativeTokens] = 0;
        acceptToken[bx * numSpeculativeTokens] = targetPredict[batchOffset];
        predicts[batchOffset] = targetPredict[batchOffset];
        return;
    }

    int32_t lastAcceptedLocalIdx = retrieveIndex[batchOffset];
    acceptIndex[bx * numSpeculativeTokens] = lastAcceptedLocalIdx;
    uint32_t numAcceptedTokens = 0;
    int32_t curIndex = 0;

    // Root token: target prediction at root position
    acceptToken[bx * numSpeculativeTokens] = targetPredict[batchOffset + lastAcceptedLocalIdx];

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
                // Accepted token: target prediction at accepted draft position
                acceptToken[bx * numSpeculativeTokens + numAcceptedTokens] = targetPredict[batchOffset + draftLocalIdx];
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
    // Bonus token from target model at the last accepted position
    predicts[batchOffset + lastAcceptedLocalIdx] = targetPredict[batchOffset + lastAcceptedLocalIdx];
}

void invokeVerifyDynamicTreeGreedy(int64_t* predicts, int64_t* acceptIndex, int64_t* acceptTokenNum,
    int64_t* acceptToken, int64_t const* candidates, int32_t const* retrieveIndex, int32_t const* retrieveNextToken,
    int32_t const* retrieveNextSibling, int64_t const* targetPredict, bool const* treeValid, SizeType32 batchSize,
    SizeType32 numDraftTokens, SizeType32 numSpecStep, cudaStream_t stream)
{
    dim3 grid(batchSize);
    dim3 block(1);

    verifyDynamicTreeGreedyKernel<<<grid, block, 0, stream>>>(predicts, acceptIndex, acceptTokenNum, acceptToken,
        candidates, retrieveIndex, retrieveNextToken, retrieveNextSibling, targetPredict, treeValid, batchSize,
        numSpecStep, numDraftTokens);

    sync_check_cuda_error(stream);
}

// ------------------------------------------------------------
// Background: Speculative Sampling Theory
// ------------------------------------------------------------
//
// Goal: reuse draft model samples to speed up generation while keeping the
// final output distribution strictly equal to the target distribution q.
//
// For a given token x:
//   p(x) = draft_probs[x]   (draft model probability)
//   q(x) = target_probs[x]  (target model probability)
//
// Step 1 - The draft model proposes token x sampled from p.
// Step 2 - Accept x with probability min(1, q(x)/p(x)).
//          Equivalently: accept when u * p(x) < q(x), where u ~ Uniform(0,1).
//
// Why does this work?
//   x is proposed with probability p(x) and then accepted with probability
//   min(1, q(x)/p(x)), so its total probability mass reaching the output is:
//     p(x) * min(1, q(x)/p(x)) = min(p(x), q(x))
//
//   This covers only the min(p, q) portion of the target mass.
//   The remaining portion q - min(p, q) = relu(q - p) is not yet covered.
//
//   Therefore, if the draft token is rejected, we must resample from the
//   residual distribution relu(q - p) (normalised) to fill the gap and
//   restore the full target distribution.
//
// Example:
//   p = [0.6, 0.3, 0.1]   tokens [A, B, C]
//   q = [0.2, 0.5, 0.3]
//
//   Accept probabilities:
//     A: min(1, 0.2/0.6) = 1/3     B: min(1, 0.5/0.3) = 1     C: min(1, 0.3/0.1) = 1
//
//   Case 1 - draft proposes A (prob 0.6):
//     Accept (1/3):  contributes 0.6 * 1/3 = 0.2 to output A.
//     Reject (2/3):  total rejected mass = 0.6 * 2/3 = 0.4.
//       relu(q-p) = [0, 0.2, 0.2]  ->  normalised [0, 0.5, 0.5]
//       contributes 0.4*0.5 = 0.2 to B and 0.4*0.5 = 0.2 to C.
//   Case 2 - draft proposes B (prob 0.3): always accepted -> 0.3 to B.
//   Case 3 - draft proposes C (prob 0.1): always accepted -> 0.1 to C.
//
//   Final output distribution:
//     A = 0.2,  B = 0.3 + 0.2 = 0.5,  C = 0.1 + 0.2 = 0.3  ->  exactly q.
//
// Tree extension:
//   The same logic applies depth-by-depth along the draft tree. At each
//   depth the kernel tries siblings in score order; the first accepted
//   sibling extends the current path. If every sibling at a depth is
//   rejected the kernel samples a correction token from relu(q-p) and
//   terminates traversal for that request.
// ------------------------------------------------------------

#include <curand_kernel.h>

__device__ int64_t sampleFromDistribution(curandStatePhilox4_32_10_t& state, float const* probs, uint32_t vocabSize)
{
    float r = curand_uniform(&state);
    float cumsum = 0.0f;
    int64_t sampledTok = static_cast<int64_t>(vocabSize) - 1; // fallback: last vocab token

    for (uint32_t v = 0; v < vocabSize; ++v)
    {
        cumsum += probs[v];
        if (r <= cumsum)
        {
            sampledTok = static_cast<int64_t>(v);
            break;
        }
    }

    return sampledTok;
}

//! \param acceptIndex          [out] accepted path as tree positions [bs, numSpecStep]. int64.
//! \param acceptTokenNum       [out] number of accepted draft tokens (excl. root) [bs]. int64.
//! \param acceptToken          [out] emitted token ids [bs, numSpecStep]. int64.
//! \param candidates           [in]  candidate token ids [bs, numDraftTokens]; col 0 = root. int64.
//! \param draftProbs           [in]  draft probs [bs, numDraftTokens-1, vocabSize]; index 0 = tree pos 1. float32.
//! \param targetProbs          [in]  target probs [bs, numDraftTokens, vocabSize]; index 0 = root. float32.
//! \param retrieveNextToken    [in]  first-child pointer [bs, numDraftTokens], -1=none. int32.
//! \param retrieveNextSibling  [in]  next-sibling pointer [bs, numDraftTokens], -1=none. int32.
//! \param batchSize            batch size.
//! \param numSpecStep          second dim of acceptIndex/acceptToken
//!                             (= max_path_len = max_draft_len + 1).
//! \param numDraftTokens       total tree nodes per batch (including root).
//! \param vocabSize            vocabulary size.
//! \param seed                 [1] int64 on GPU. Philox RNG seed.
//! \param offset               [1] int64 on GPU. Philox RNG offset.
__global__ void verifyDynamicTreeRejectionKernel(int64_t* acceptIndex, int64_t* acceptTokenNum, int64_t* acceptToken,
    int64_t const* candidates, float const* draftProbs, float const* targetProbs, int32_t const* retrieveNextToken,
    int32_t const* retrieveNextSibling, uint32_t batchSize, uint32_t numSpeculativeTokens, uint32_t numDraftTokens,
    uint32_t vocabSize, int64_t const* seed, int64_t const* offset)
{
    uint32_t bx = blockIdx.x;
    if (bx >= batchSize)
    {
        return;
    }

    uint32_t batchOffset = bx * numDraftTokens;

    curandStatePhilox4_32_10_t state;
    curand_init(static_cast<uint64_t>(seed[0]), static_cast<uint64_t>(bx), static_cast<uint64_t>(offset[0]), &state);

    // Root (depth 0): initialize path state at tree position 0.
    //
    // Example tree used in code review discussions:
    //   root: E
    //   children of E:   F1, F2, F3
    //   children of F1:  G1, G2
    //   children of F2:  G3
    //
    // In that example the per-request inputs are conceptually:
    //   candidates   = [E, F1, F2, F3, G1, G2, G3]
    //   draftProbs   = [p(.|E), p(.|E), p(.|E), p(.|F1), p(.|F1), p(.|F2)]
    //   targetProbs  = [q(.|E), q(.|F1), q(.|F2), q(.|F3), q(.|G1), q(.|G2), q(.|G3)]
    //
    // Note that draftProbs is aligned to non-root tree nodes (tree position i maps
    // to draftProbs slot i-1), while targetProbs is aligned to all tree positions,
    // including the root at slot 0.
    //
    // Output convention:
    //   - acceptIndex stores the accepted draft path as tree positions, with slot 0
    //     reserved for the root position.
    //   - acceptToken stores the emitted token sequence, matching the greedy kernel:
    //       slot 0                  = first emitted token
    //       slot numAcceptedTokens  = final bonus/correction token
    //   - acceptTokenNum stores the number of accepted draft tokens only. The caller
    //     adds 1 to obtain the total number of emitted tokens.
    int32_t lastAcceptedLocalIdx = 0;
    acceptIndex[bx * numSpeculativeTokens] = lastAcceptedLocalIdx;
    uint32_t numAcceptedTokens = 0;
    int32_t curIndex = 0;
    bool hasTerminalToken = false;

    for (uint32_t j = 1; j < numSpeculativeTokens; ++j)
    {
        // Advance to the first child of the last accepted node.
        // Continuing the example above:
        //   j = 1, curIndex = 0 (E)  -> firstChild = F1
        //   j = 2, curIndex = 1 (F1) -> firstChild = G1
        int32_t firstChild = retrieveNextToken[batchOffset + curIndex];
        curIndex = firstChild;

        while (curIndex != -1)
        {
            int32_t draftLocalIdx = curIndex; // retrieveIndex is identity: draftLocalIdx == curIndex
            int64_t draftTokenId = candidates[batchOffset + curIndex];
            // draftProbs: tree position curIndex maps to draft index curIndex-1 (root has no draft prob).
            // For the example:
            //   curIndex = 1 (F1) -> compare F1 using draftProbs slot 0, i.e. p(F1|E)
            //   curIndex = 2 (F2) -> compare F2 using draftProbs slot 1, i.e. p(F2|E)
            //   curIndex = 4 (G1) -> compare G1 using draftProbs slot 3, i.e. p(G1|F1)
            uint64_t dpIdx = static_cast<uint64_t>(curIndex) - 1u;
            float pDraft
                = draftProbs[(static_cast<uint64_t>(bx) * (numDraftTokens - 1) + dpIdx) * vocabSize + draftTokenId];
            // Rejection sampling compares draft siblings under the target
            // distribution of the currently accepted parent node.
            float pTarget = targetProbs[(static_cast<uint64_t>(bx) * numDraftTokens + lastAcceptedLocalIdx)
                    * vocabSize
                + draftTokenId];

            // Rejection test for the current sibling:
            //   accept with probability min(1, q(x) / p(x))
            // where:
            //   p(x) = pDraft  from the draft model proposal
            //   q(x) = pTarget from the target model verification distribution
            float acceptProb = fminf(1.0f, pTarget / (pDraft + 1e-10f));

            float u = curand_uniform(&state);

            if (u < acceptProb)
            {
                // Accepted sibling: extend the path and continue with this node's children.
                // Example:
                //   if F1 is accepted at j = 1, then on the next iteration we descend to
                //   F1's first child (G1) and compare G1/G2 in sibling order.
                // Emit the accepted draft token immediately. If we later stop at a leaf
                // (or hit max depth), the final bonus token from the target distribution
                // will be written at slot numAcceptedTokens.
                acceptToken[bx * numSpeculativeTokens + numAcceptedTokens] = draftTokenId;
                ++numAcceptedTokens;
                acceptIndex[bx * numSpeculativeTokens + numAcceptedTokens] = draftLocalIdx;
                lastAcceptedLocalIdx = draftLocalIdx;
                break;
            }
            else
            {
                // Reject this sibling and try the next sibling at the same depth.
                // Example:
                //   reject F1 -> move to F2
                //   reject F2 -> move to F3
                curIndex = retrieveNextSibling[batchOffset + curIndex];
            }
        }

        if (curIndex == -1)
        {
            // All siblings exhausted. Two sub-cases:
            // (a) firstChild == -1: leaf node, no draft tokens at this depth.
            //     Emit the final bonus token sampled from q(.|lastAcceptedLocalIdx).
            // (b) firstChild != -1: every sibling was rejected -> sample correction token
            //     from relu(q - p) at firstChild's position to restore the target distribution.
            if (firstChild == -1)
            {
                float const* tProbs
                    = targetProbs + (static_cast<uint64_t>(bx) * numDraftTokens + lastAcceptedLocalIdx) * vocabSize;
                acceptToken[bx * numSpeculativeTokens + numAcceptedTokens]
                    = sampleFromDistribution(state, tProbs, vocabSize);
                hasTerminalToken = true;
            }
            else
            {
                uint64_t dpIdx = static_cast<uint64_t>(firstChild) - 1u;
                float const* tProbs
                    = targetProbs + (static_cast<uint64_t>(bx) * numDraftTokens + lastAcceptedLocalIdx) * vocabSize;
                float const* dProbs
                    = draftProbs + (static_cast<uint64_t>(bx) * (numDraftTokens - 1) + dpIdx) * vocabSize;

                // Pass 1: compute diffSum = sum of relu(q - p)
                float diffSum = 0.0f;
                for (uint32_t v = 0; v < vocabSize; ++v)
                {
                    float diff = tProbs[v] - dProbs[v];
                    if (diff > 0.0f)
                        diffSum += diff;
                }

                // Pass 2: CDF inversion over the normalised residual distribution.
                // Falls back to sampling directly from target when diffSum ~= 0.
                //
                // Example:
                //   if F1/F2/F3 are all rejected under parent E, then we sample a correction
                //   token from relu(q(.|E) - p(.|E)). The sampled token terminates traversal:
                //   it is emitted as the next output token, and there is no further descent.
                int64_t corrTok = static_cast<int64_t>(vocabSize) - 1; // fallback: last vocab token
                bool useDiff = (diffSum > 1e-10f);

                if (useDiff)
                {
                    float r = curand_uniform(&state);
                    float cumsum = 0.0f;
                    for (uint32_t v = 0; v < vocabSize; ++v)
                    {
                        float diff = tProbs[v] - dProbs[v];
                        float prob = (diff > 0.0f) ? diff / diffSum : 0.0f;
                        cumsum += prob;
                        if (r <= cumsum)
                        {
                            corrTok = static_cast<int64_t>(v);
                            break;
                        }
                    }
                }
                else
                {
                    corrTok = sampleFromDistribution(state, tProbs, vocabSize);
                }

                acceptToken[bx * numSpeculativeTokens + numAcceptedTokens] = corrTok;
                // acceptIndex at correction depth left as 0 (zero-initialized by caller)
                hasTerminalToken = true;
            }
            break;
        }
    }

    if (!hasTerminalToken)
    {
        // Reached max speculative depth while continuing to accept the draft path.
        // Emit the final bonus token from the last accepted position.
        float const* tProbs
            = targetProbs + (static_cast<uint64_t>(bx) * numDraftTokens + lastAcceptedLocalIdx) * vocabSize;
        acceptToken[bx * numSpeculativeTokens + numAcceptedTokens] = sampleFromDistribution(state, tProbs, vocabSize);
    }

    acceptTokenNum[bx] = numAcceptedTokens;
}

void invokeVerifyDynamicTreeRejection(int64_t* acceptIndex, int64_t* acceptTokenNum, int64_t* acceptToken,
    int64_t const* candidates, float const* draftProbs, float const* targetProbs, int32_t const* retrieveNextToken,
    int32_t const* retrieveNextSibling, SizeType32 batchSize, SizeType32 numDraftTokens, SizeType32 numSpecStep,
    SizeType32 vocabSize, int64_t const* seed, int64_t const* offset, cudaStream_t stream)
{
    dim3 grid(batchSize);
    dim3 block(1);

    verifyDynamicTreeRejectionKernel<<<grid, block, 0, stream>>>(acceptIndex, acceptTokenNum, acceptToken, candidates,
        draftProbs, targetProbs, retrieveNextToken, retrieveNextSibling, batchSize, numSpecStep, numDraftTokens,
        vocabSize, seed, offset);

    sync_check_cuda_error(stream);
}

} // namespace kernels::speculative_decoding

TRTLLM_NAMESPACE_END
