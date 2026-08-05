/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/kernels/minimaxM3SelectBlocks.h"

#include "tensorrt_llm/kernels/moeTopKFuncs.cuh"

#include <cmath>
#include <cooperative_groups.h>
#include <cstdint>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace
{

namespace cg = cooperative_groups;

constexpr int kTopK = 16;
constexpr int kWarpSize = 32;
constexpr int kThreadsPerBlock = 256;
constexpr int kWarpsPerBlock = kThreadsPerBlock / kWarpSize;
constexpr int kSmallMaxBlocks = 128;
constexpr float kInitScore = 1.0e30F;
constexpr float kLocalScore = 1.0e29F;
constexpr uint32_t kFullWarpMask = 0xFFFFFFFFU;

__forceinline__ __device__ bool candidateGreater(
    uint32_t lhsScoreKey, int32_t lhsBlockId, uint32_t rhsScoreKey, int32_t rhsBlockId)
{
    return lhsScoreKey > rhsScoreKey || (lhsScoreKey == rhsScoreKey && lhsBlockId < rhsBlockId);
}

__forceinline__ __device__ void warpBitonicSortDesc64(
    uint32_t& scoreKey0, int32_t& blockId0, uint32_t& scoreKey1, int32_t& blockId1, int32_t lane)
{
#pragma unroll
    for (int32_t size = 2; size <= 2 * kWarpSize; size *= 2)
    {
#pragma unroll
        for (int32_t stride = size / 2; stride > 0; stride /= 2)
        {
            if (stride == kWarpSize)
            {
                bool const firstGreater = candidateGreater(scoreKey0, blockId0, scoreKey1, blockId1);
                uint32_t const greaterScoreKey = firstGreater ? scoreKey0 : scoreKey1;
                int32_t const greaterBlockId = firstGreater ? blockId0 : blockId1;
                uint32_t const lesserScoreKey = firstGreater ? scoreKey1 : scoreKey0;
                int32_t const lesserBlockId = firstGreater ? blockId1 : blockId0;
                scoreKey0 = greaterScoreKey;
                blockId0 = greaterBlockId;
                scoreKey1 = lesserScoreKey;
                blockId1 = lesserBlockId;
            }
            else
            {
                uint32_t const partnerScoreKey0 = __shfl_xor_sync(kFullWarpMask, scoreKey0, stride);
                int32_t const partnerBlockId0 = __shfl_xor_sync(kFullWarpMask, blockId0, stride);
                uint32_t const partnerScoreKey1 = __shfl_xor_sync(kFullWarpMask, scoreKey1, stride);
                int32_t const partnerBlockId1 = __shfl_xor_sync(kFullWarpMask, blockId1, stride);
                bool const takeGreater0 = ((lane & size) == 0) == ((lane & stride) == 0);
                int32_t const secondIndex = lane + kWarpSize;
                bool const takeGreater1 = ((secondIndex & size) == 0) == ((secondIndex & stride) == 0);
                bool const partnerGreater0 = candidateGreater(partnerScoreKey0, partnerBlockId0, scoreKey0, blockId0);
                bool const currentGreater0 = candidateGreater(scoreKey0, blockId0, partnerScoreKey0, partnerBlockId0);
                bool const partnerGreater1 = candidateGreater(partnerScoreKey1, partnerBlockId1, scoreKey1, blockId1);
                bool const currentGreater1 = candidateGreater(scoreKey1, blockId1, partnerScoreKey1, partnerBlockId1);
                if ((takeGreater0 && partnerGreater0) || (!takeGreater0 && currentGreater0))
                {
                    scoreKey0 = partnerScoreKey0;
                    blockId0 = partnerBlockId0;
                }
                if ((takeGreater1 && partnerGreater1) || (!takeGreater1 && currentGreater1))
                {
                    scoreKey1 = partnerScoreKey1;
                    blockId1 = partnerBlockId1;
                }
            }
        }
    }
}

__forceinline__ __device__ void warpBitonicSortDesc128(uint32_t (&scoreKeys)[4], int32_t (&blockIds)[4], int32_t lane)
{
    // Virtual item slot * 32 + lane stays in registers. Short strides exchange
    // lanes with shuffles; strides 32 and 64 exchange slots within each lane.
#pragma unroll
    for (int32_t size = 2; size <= kSmallMaxBlocks; size *= 2)
    {
#pragma unroll
        for (int32_t stride = size / 2; stride > 0; stride /= 2)
        {
            uint32_t previousScoreKeys[4];
            int32_t previousBlockIds[4];
#pragma unroll
            for (int32_t slot = 0; slot < 4; ++slot)
            {
                previousScoreKeys[slot] = scoreKeys[slot];
                previousBlockIds[slot] = blockIds[slot];
            }

#pragma unroll
            for (int32_t slot = 0; slot < 4; ++slot)
            {
                uint32_t partnerScoreKey;
                int32_t partnerBlockId;
                if (stride < kWarpSize)
                {
                    partnerScoreKey = __shfl_xor_sync(kFullWarpMask, previousScoreKeys[slot], stride);
                    partnerBlockId = __shfl_xor_sync(kFullWarpMask, previousBlockIds[slot], stride);
                }
                else
                {
                    int32_t const partnerSlot = slot ^ (stride / kWarpSize);
                    partnerScoreKey = previousScoreKeys[partnerSlot];
                    partnerBlockId = previousBlockIds[partnerSlot];
                }

                int32_t const item = lane + slot * kWarpSize;
                bool const takeGreater = ((item & size) == 0) == ((item & stride) == 0);
                bool const partnerGreater = candidateGreater(
                    partnerScoreKey, partnerBlockId, previousScoreKeys[slot], previousBlockIds[slot]);
                bool const currentGreater = candidateGreater(
                    previousScoreKeys[slot], previousBlockIds[slot], partnerScoreKey, partnerBlockId);
                if ((takeGreater && partnerGreater) || (!takeGreater && currentGreater))
                {
                    scoreKeys[slot] = partnerScoreKey;
                    blockIds[slot] = partnerBlockId;
                }
            }
        }
    }
}

template <bool HeadMajorOutput>
__forceinline__ __device__ int64_t outputOffset(
    int32_t query, int32_t kvHead, int32_t totalQueries, int32_t numKvHeads, int32_t rank)
{
    int64_t const outputRow = HeadMajorOutput ? static_cast<int64_t>(kvHead) * totalQueries + query
                                              : static_cast<int64_t>(query) * numKvHeads + kvHead;
    return outputRow * kTopK + rank;
}

template <bool HeadMajorOutput, int NumCandidates>
__forceinline__ __device__ void selectFromCandidates(cg::thread_block_tile<kWarpSize> const& warp,
    float (&localScores)[NumCandidates], int32_t (&localIndices)[NumCandidates], int32_t* output, int32_t query,
    int32_t kvHead, int32_t totalQueries, int32_t numKvHeads, int32_t numBlocks)
{
    float selectedScores[kTopK];
    int32_t selectedIndices[kTopK];
    reduce_topk::reduceTopK<kTopK>(warp, selectedScores, selectedIndices, localScores, localIndices, -INFINITY);

    if (warp.thread_rank() == 0)
    {
#pragma unroll
        for (int32_t rank = 0; rank < kTopK; ++rank)
        {
            if (selectedScores[rank] == -INFINITY)
            {
                selectedIndices[rank] = -1;
            }
        }

        // MSA consumes block IDs in ascending order. Sort the sixteen selected
        // IDs in registers, treating -1 padding as greater than every valid ID.
#pragma unroll
        for (int32_t rank = 1; rank < kTopK; ++rank)
        {
            int32_t const candidate = selectedIndices[rank];
            int32_t const candidateKey = candidate < 0 ? numBlocks : candidate;
            int32_t insertion = rank;
            while (insertion > 0)
            {
                int32_t const previous = selectedIndices[insertion - 1];
                int32_t const previousKey = previous < 0 ? numBlocks : previous;
                if (previousKey <= candidateKey)
                {
                    break;
                }
                selectedIndices[insertion] = previous;
                --insertion;
            }
            selectedIndices[insertion] = candidate;
        }

#pragma unroll
        for (int32_t rank = 0; rank < kTopK; ++rank)
        {
            output[outputOffset<HeadMajorOutput>(query, kvHead, totalQueries, numKvHeads, rank)]
                = selectedIndices[rank];
        }
    }
}

template <int NumCandidates, bool HeadMajorOutput>
__global__ void minimaxM3SelectBlocksSmallKernel(float const* __restrict__ scores, int64_t headStride,
    int64_t blockStride, int64_t queryStride, int32_t const* __restrict__ nValidBlocks, int32_t* __restrict__ output,
    int32_t numKvHeads, int32_t numBlocks, int32_t totalQueries, int32_t initBlocks, int32_t localBlocks)
{
    static_assert(NumCandidates * kWarpSize <= kSmallMaxBlocks);
    auto const warp = cg::tiled_partition<kWarpSize>(cg::this_thread_block());
    int32_t const warpInBlock = threadIdx.x / kWarpSize;
    int32_t const outputRow = blockIdx.x * kWarpsPerBlock + warpInBlock;
    int32_t const numOutputRows = totalQueries * numKvHeads;
    if (outputRow >= numOutputRows)
    {
        return;
    }

    int32_t const query = outputRow / numKvHeads;
    int32_t const kvHead = outputRow % numKvHeads;
    int32_t const rawValidBlocks = nValidBlocks[query];
    int32_t const validBlocks = max(0, min(rawValidBlocks, numBlocks));
    int64_t const localStart
        = max(static_cast<int64_t>(rawValidBlocks) - static_cast<int64_t>(localBlocks), static_cast<int64_t>(0));

    using RedType = reduce_topk::TopKRedType<float>;
    float localScores[NumCandidates];
    int32_t localIndices[NumCandidates];
#pragma unroll
    for (int32_t slot = 0; slot < NumCandidates; ++slot)
    {
        int32_t const block = warp.thread_rank() + slot * kWarpSize;
        RedType candidate{-INFINITY, RedType::kMaxIdx};
        if (block < validBlocks)
        {
            int64_t const offset = static_cast<int64_t>(kvHead) * headStride + static_cast<int64_t>(block) * blockStride
                + static_cast<int64_t>(query) * queryStride;
            float score = scores[offset];
            if (block < initBlocks)
            {
                score = kInitScore;
            }
            // Match the PyTorch reference's second torch.where: local forcing
            // overwrites init forcing if the two ranges overlap.
            if (block >= localStart)
            {
                score = kLocalScore;
            }
            candidate = RedType{score, block};
        }
        RedType::unpack(localScores[slot], localIndices[slot], candidate.compValIdx);
    }

    selectFromCandidates<HeadMajorOutput>(
        warp, localScores, localIndices, output, query, kvHead, totalQueries, numKvHeads, numBlocks);
}

template <bool HeadMajorOutput>
__global__ void minimaxM3SelectBlocks64Kernel(float const* __restrict__ scores, int64_t headStride, int64_t blockStride,
    int64_t queryStride, int32_t const* __restrict__ nValidBlocks, int32_t* __restrict__ output, int32_t numKvHeads,
    int32_t numBlocks, int32_t totalQueries, int32_t initBlocks, int32_t localBlocks)
{
    int32_t const warpInBlock = threadIdx.x / kWarpSize;
    int32_t const outputRow = blockIdx.x * kWarpsPerBlock + warpInBlock;
    int32_t const lane = threadIdx.x % kWarpSize;
    int32_t const numOutputRows = totalQueries * numKvHeads;
    if (outputRow >= numOutputRows)
    {
        return;
    }

    int32_t const query = outputRow / numKvHeads;
    int32_t const kvHead = outputRow % numKvHeads;
    int32_t const rawValidBlocks = nValidBlocks[query];
    int32_t const validBlocks = max(0, min(rawValidBlocks, numBlocks));
    int64_t const localStart
        = max(static_cast<int64_t>(rawValidBlocks) - static_cast<int64_t>(localBlocks), static_cast<int64_t>(0));

    using RedType = reduce_topk::TopKRedType<float>;
    RedType candidates[2]{{-INFINITY, RedType::kMaxIdx}, {-INFINITY, RedType::kMaxIdx}};
#pragma unroll
    for (int32_t slot = 0; slot < 2; ++slot)
    {
        int32_t const block = lane + slot * kWarpSize;
        if (block < validBlocks)
        {
            int64_t const offset = static_cast<int64_t>(kvHead) * headStride + static_cast<int64_t>(block) * blockStride
                + static_cast<int64_t>(query) * queryStride;
            float score = scores[offset];
            if (block < initBlocks)
            {
                score = kInitScore;
            }
            // Match the PyTorch reference's second torch.where: local forcing
            // overwrites init forcing if the two ranges overlap.
            if (block >= localStart)
            {
                score = kLocalScore;
            }
            candidates[slot] = RedType{score, block};
        }
    }

    uint32_t scoreKey0 = static_cast<uint32_t>(candidates[0].compValIdx >> RedType::kMoveBits);
    int32_t blockId0
        = RedType::kMaxIdx - static_cast<int32_t>(static_cast<uint32_t>(candidates[0].compValIdx) & 0xFFFFU);
    uint32_t scoreKey1 = static_cast<uint32_t>(candidates[1].compValIdx >> RedType::kMoveBits);
    int32_t blockId1
        = RedType::kMaxIdx - static_cast<int32_t>(static_cast<uint32_t>(candidates[1].compValIdx) & 0xFFFFU);
    warpBitonicSortDesc64(scoreKey0, blockId0, scoreKey1, blockId1, lane);

    if (lane < kTopK)
    {
        RedType const negativeInfinity{-INFINITY, 0};
        uint32_t const negativeInfinityScoreKey
            = static_cast<uint32_t>(negativeInfinity.compValIdx >> RedType::kMoveBits);
        if (scoreKey0 == negativeInfinityScoreKey)
        {
            blockId0 = numBlocks;
        }

        // MSA consumes block IDs in ascending order. numBlocks is the sentinel
        // so padding naturally follows every valid ID.
#pragma unroll
        for (int32_t size = 2; size <= kTopK; size *= 2)
        {
#pragma unroll
            for (int32_t stride = size / 2; stride > 0; stride /= 2)
            {
                int32_t const partnerBlockId = __shfl_xor_sync(0xFFFFU, blockId0, stride);
                bool const takeMin = ((lane & size) == 0) == ((lane & stride) == 0);
                blockId0 = takeMin ? min(blockId0, partnerBlockId) : max(blockId0, partnerBlockId);
            }
        }

        output[outputOffset<HeadMajorOutput>(query, kvHead, totalQueries, numKvHeads, lane)]
            = blockId0 == numBlocks ? -1 : blockId0;
    }
}

template <bool HeadMajorOutput>
__global__ void minimaxM3SelectBlocks128Kernel(float const* __restrict__ scores, int64_t headStride,
    int64_t blockStride, int64_t queryStride, int32_t const* __restrict__ nValidBlocks, int32_t* __restrict__ output,
    int32_t numKvHeads, int32_t numBlocks, int32_t totalQueries, int32_t initBlocks, int32_t localBlocks)
{
    int32_t const warpInBlock = threadIdx.x / kWarpSize;
    int32_t const outputRow = blockIdx.x * kWarpsPerBlock + warpInBlock;
    int32_t const lane = threadIdx.x % kWarpSize;
    int32_t const numOutputRows = totalQueries * numKvHeads;
    if (outputRow >= numOutputRows)
    {
        return;
    }

    int32_t const query = outputRow / numKvHeads;
    int32_t const kvHead = outputRow % numKvHeads;
    int32_t const rawValidBlocks = nValidBlocks[query];
    int32_t const validBlocks = max(0, min(rawValidBlocks, numBlocks));
    int64_t const localStart
        = max(static_cast<int64_t>(rawValidBlocks) - static_cast<int64_t>(localBlocks), static_cast<int64_t>(0));

    using RedType = reduce_topk::TopKRedType<float>;
    RedType candidates[4];
#pragma unroll
    for (int32_t slot = 0; slot < 4; ++slot)
    {
        candidates[slot] = RedType{-INFINITY, RedType::kMaxIdx};
        int32_t const block = lane + slot * kWarpSize;
        if (block < validBlocks)
        {
            int64_t const offset = static_cast<int64_t>(kvHead) * headStride + static_cast<int64_t>(block) * blockStride
                + static_cast<int64_t>(query) * queryStride;
            float score = scores[offset];
            if (block < initBlocks)
            {
                score = kInitScore;
            }
            // Match the PyTorch reference's second torch.where: local forcing
            // overwrites init forcing if the two ranges overlap.
            if (block >= localStart)
            {
                score = kLocalScore;
            }
            candidates[slot] = RedType{score, block};
        }
    }

    uint32_t scoreKeys[4];
    int32_t blockIds[4];
#pragma unroll
    for (int32_t slot = 0; slot < 4; ++slot)
    {
        scoreKeys[slot] = static_cast<uint32_t>(candidates[slot].compValIdx >> RedType::kMoveBits);
        blockIds[slot]
            = RedType::kMaxIdx - static_cast<int32_t>(static_cast<uint32_t>(candidates[slot].compValIdx) & 0xFFFFU);
    }
    warpBitonicSortDesc128(scoreKeys, blockIds, lane);

    if (lane < kTopK)
    {
        RedType const negativeInfinity{-INFINITY, 0};
        uint32_t const negativeInfinityScoreKey
            = static_cast<uint32_t>(negativeInfinity.compValIdx >> RedType::kMoveBits);
        if (scoreKeys[0] == negativeInfinityScoreKey)
        {
            blockIds[0] = numBlocks;
        }

        // MSA consumes block IDs in ascending order. numBlocks is the sentinel
        // so padding naturally follows every valid ID.
#pragma unroll
        for (int32_t size = 2; size <= kTopK; size *= 2)
        {
#pragma unroll
            for (int32_t stride = size / 2; stride > 0; stride /= 2)
            {
                int32_t const partnerBlockId = __shfl_xor_sync(0xFFFFU, blockIds[0], stride);
                bool const takeMin = ((lane & size) == 0) == ((lane & stride) == 0);
                blockIds[0] = takeMin ? min(blockIds[0], partnerBlockId) : max(blockIds[0], partnerBlockId);
            }
        }

        output[outputOffset<HeadMajorOutput>(query, kvHead, totalQueries, numKvHeads, lane)]
            = blockIds[0] == numBlocks ? -1 : blockIds[0];
    }
}

template <bool HeadMajorOutput>
__global__ void minimaxM3SelectBlocksKernel(float const* __restrict__ scores, int64_t headStride, int64_t blockStride,
    int64_t queryStride, int32_t const* __restrict__ nValidBlocks, int32_t* __restrict__ output, int32_t numKvHeads,
    int32_t numBlocks, int32_t totalQueries, int32_t initBlocks, int32_t localBlocks)
{
    auto const warp = cg::tiled_partition<kWarpSize>(cg::this_thread_block());
    int32_t const warpInBlock = threadIdx.x / kWarpSize;
    int32_t const outputRow = blockIdx.x * kWarpsPerBlock + warpInBlock;
    int32_t const numOutputRows = totalQueries * numKvHeads;
    if (outputRow >= numOutputRows)
    {
        return;
    }

    int32_t const query = outputRow / numKvHeads;
    int32_t const kvHead = outputRow % numKvHeads;
    int32_t const rawValidBlocks = nValidBlocks[query];
    int32_t const validBlocks = max(0, min(rawValidBlocks, numBlocks));
    int64_t const localStart
        = max(static_cast<int64_t>(rawValidBlocks) - static_cast<int64_t>(localBlocks), static_cast<int64_t>(0));

    using RedType = reduce_topk::TopKRedType<float>;
    RedType localTopK[kTopK];
#pragma unroll
    for (int32_t rank = 0; rank < kTopK; ++rank)
    {
        localTopK[rank] = RedType{-INFINITY, RedType::kMaxIdx};
    }

    for (int32_t block = warp.thread_rank(); block < validBlocks; block += kWarpSize)
    {
        int64_t const offset = static_cast<int64_t>(kvHead) * headStride + static_cast<int64_t>(block) * blockStride
            + static_cast<int64_t>(query) * queryStride;
        float score = scores[offset];
        if (block < initBlocks)
        {
            score = kInitScore;
        }
        // Match the PyTorch reference's second torch.where: local forcing
        // overwrites init forcing if the two ranges overlap.
        if (block >= localStart)
        {
            score = kLocalScore;
        }

        RedType const candidate{score, block};
        if (candidate.compValIdx > localTopK[kTopK - 1].compValIdx)
        {
            int32_t insertion = kTopK - 1;
#pragma unroll
            for (int32_t rank = kTopK - 2; rank >= 0; --rank)
            {
                if (candidate.compValIdx > localTopK[rank].compValIdx)
                {
                    localTopK[rank + 1] = localTopK[rank];
                    insertion = rank;
                }
            }
            localTopK[insertion] = candidate;
        }
    }

    float localScores[kTopK];
    int32_t localIndices[kTopK];
#pragma unroll
    for (int32_t rank = 0; rank < kTopK; ++rank)
    {
        RedType::unpack(localScores[rank], localIndices[rank], localTopK[rank].compValIdx);
    }

    selectFromCandidates<HeadMajorOutput>(
        warp, localScores, localIndices, output, query, kvHead, totalQueries, numKvHeads, numBlocks);
}

template <bool HeadMajorOutput>
void launchMinimaxM3SelectBlocks(float const* scores, int64_t headStride, int64_t blockStride, int64_t queryStride,
    int32_t const* nValidBlocks, int32_t* output, int32_t numKvHeads, int32_t numBlocks, int32_t totalQueries,
    int32_t initBlocks, int32_t localBlocks, cudaStream_t stream)
{
    int32_t const numOutputRows = totalQueries * numKvHeads;
    if (numOutputRows == 0)
    {
        return;
    }
    int32_t const gridSize = (numOutputRows + kWarpsPerBlock - 1) / kWarpsPerBlock;
    if (numBlocks <= kWarpSize)
    {
        minimaxM3SelectBlocksSmallKernel<1, HeadMajorOutput><<<gridSize, kThreadsPerBlock, 0, stream>>>(scores,
            headStride, blockStride, queryStride, nValidBlocks, output, numKvHeads, numBlocks, totalQueries, initBlocks,
            localBlocks);
    }
    else if (numBlocks <= 2 * kWarpSize)
    {
        minimaxM3SelectBlocks64Kernel<HeadMajorOutput><<<gridSize, kThreadsPerBlock, 0, stream>>>(scores, headStride,
            blockStride, queryStride, nValidBlocks, output, numKvHeads, numBlocks, totalQueries, initBlocks,
            localBlocks);
    }
    else if (numBlocks <= kSmallMaxBlocks)
    {
        minimaxM3SelectBlocks128Kernel<HeadMajorOutput><<<gridSize, kThreadsPerBlock, 0, stream>>>(scores, headStride,
            blockStride, queryStride, nValidBlocks, output, numKvHeads, numBlocks, totalQueries, initBlocks,
            localBlocks);
    }
    else
    {
        minimaxM3SelectBlocksKernel<HeadMajorOutput><<<gridSize, kThreadsPerBlock, 0, stream>>>(scores, headStride,
            blockStride, queryStride, nValidBlocks, output, numKvHeads, numBlocks, totalQueries, initBlocks,
            localBlocks);
    }
}

} // namespace

void invokeMinimaxM3SelectBlocks(float const* scores, int64_t headStride, int64_t blockStride, int64_t queryStride,
    int32_t const* nValidBlocks, int32_t* output, int32_t numKvHeads, int32_t numBlocks, int32_t totalQueries,
    int32_t initBlocks, int32_t localBlocks, bool headMajorOutput, cudaStream_t stream)
{
    if (headMajorOutput)
    {
        launchMinimaxM3SelectBlocks<true>(scores, headStride, blockStride, queryStride, nValidBlocks, output,
            numKvHeads, numBlocks, totalQueries, initBlocks, localBlocks, stream);
    }
    else
    {
        launchMinimaxM3SelectBlocks<false>(scores, headStride, blockStride, queryStride, nValidBlocks, output,
            numKvHeads, numBlocks, totalQueries, initBlocks, localBlocks, stream);
    }
}

} // namespace kernels

TRTLLM_NAMESPACE_END
