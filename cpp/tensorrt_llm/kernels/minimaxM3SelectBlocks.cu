/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2021 NAVER Corp. Authored by CLOVA.
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

// The histogram select in the second half of this file is derived from
// TRT-LLM's own kernels/indexerTopK.cu (topKPerRowJob and its helpers), which
// carries the NAVER/CLOVA copyright above; see the comment on that section for
// what this port changes.

#include "tensorrt_llm/kernels/minimaxM3SelectBlocks.h"

#include "tensorrt_llm/kernels/moeTopKFuncs.cuh"

#include <cmath>
#include <cooperative_groups.h>
#include <cstdint>
#include <cub/cub.cuh>
#include <cuda_fp16.h>
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

// Apply the init / local forcing to a raw score. Local forcing wins where the
// two ranges overlap, matching the second torch.where in the PyTorch reference.
__forceinline__ __device__ float applyForcing(float raw, int32_t block, int32_t initBlocks, int64_t localStart)
{
    if (block >= localStart)
    {
        return kLocalScore;
    }
    if (block < initBlocks)
    {
        return kInitScore;
    }
    return raw;
}

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

// =============================================================================
// Histogram (radix) select — for rows too long for the bitonic paths
// =============================================================================
//
// The bitonic paths above stop at 128 blocks because that is the widest row
// four register slots per lane can hold across a warp. Wider rows come here.
//
// This replaced a kernel that gave each row one warp and walked it 32 blocks at
// a time, maintaining a 16-deep sorted array in registers, so its cost grew
// with the row width times the insertion depth. The histogram select instead
// makes a small number of whole-CTA passes over the row.
//
// Measured on a B200 under CUDA graph replay, full rows, per selector call: the
// histogram is flat at ~6 us from 32 blocks all the way to 4096 for batches up
// to 128 rows, while the warp-strided kernel it replaced grew from ~10 us at
// 160 blocks to ~140 us at 8192. The histogram is ahead from 160 blocks up at
// every row count from 1 to 2048, and at 128 blocks and below the bitonic paths
// beat it by up to 2x on the widest batches, so 128 is both the structural and
// the measured place to switch.
//
// The algorithm is TRT-LLM's own indexerTopK (kernels/indexerTopK.cu,
// topKPerRowJob): a step-0 fp16 histogram as a fast first cut, then up to three
// exact fp32 refinement steps over whatever landed in the threshold bin,
// finished by an insertion sort over the staged boundary candidates. The
// single-block-per-row, non-merging shape used here matches the port in
// 3rdparty/MSA (python/fmha_sm100/csrc/include/sparse_topk_select.cuh).
//
// Differences from both, all forced by this op's contract:
//   - The row is read through blockStride instead of being transposed to a
//     contiguous workspace first, so there is no transpose kernel and no
//     workspace. blockStride == 1 still takes a float4 path.
//   - rowEnd is per row (nValidBlocks[query]) rather than a scalar, so a mixed
//     length decode batch needs neither -inf padding nor a uniform bound.
//   - Scores are forced to init / local sentinels on the fly, at every read.
//   - The boundary insertion sort breaks ties by smaller block index rather
//     than by staging slot, which makes the result deterministic and matches
//     the tie-break of the bitonic paths and of the torch.topk reference.
//   - Selected blocks whose forced score is -inf are emitted as -1, and the
//     output is sorted ascending by block index.

// The number of boundary candidates that can be staged for the final sort.
// A threshold bin wider than this forces another refinement step.
constexpr int kNumFinalItems = 2048;

// Radix bin extraction. kNumBins picks the bin width: 2048 gives 11-bit bins
// (indexerTopK's canonical config, and the only width whose step 3 is exact),
// 1024 gives 10-bit bins. The transform maps float bits to an unsigned key that
// orders descending, so bin 0 holds the largest values and NaN sorts above
// +inf, matching torch.topk on CUDA.
template <int step, int kNumBins>
__forceinline__ __device__ uint32_t extractBinIdx(float x)
{
    static_assert(kNumBins == 1024 || kNumBins == 2048);
    constexpr int kBinBits = kNumBins == 2048 ? 11 : 10;
    if constexpr (step == 0)
    {
        uint16_t bits = __half_as_ushort(__float2half(x));
        bits = (bits & 0x8000) ? bits : ~bits & 0x7fff;
        return bits >> (16 - kBinBits);
    }
    else
    {
        uint32_t bits = __float_as_uint(x);
        bits = (bits & 0x80000000) ? bits : ~bits & 0x7fffffff;
        if constexpr (step == 1)
        {
            return bits >> (32 - kBinBits);
        }
        else if constexpr (step == 2)
        {
            return (bits >> (32 - 2 * kBinBits)) & (kNumBins - 1);
        }
        else
        {
            return (bits >> (32 - 2 * kBinBits - 10)) & 0x3ff;
        }
    }
}

template <int shift>
__forceinline__ __device__ bool isPartialMatch(float x, uint32_t pattern)
{
    if constexpr (shift == 0)
    {
        return true;
    }
    uint32_t bits = __float_as_uint(x);
    bits = (bits & 0x80000000) ? bits : ~bits & 0x7fffffff;
    return (bits ^ pattern) >> shift == 0;
}

// Reads one row of scores, applying the init / local forcing, and hands each
// (score, block) pair to f. blockStride == 1 goes through float4 loads.
template <typename Func>
__device__ void forEachScore(float const* __restrict__ rowBase, int64_t blockStride, int32_t rowEnd, int32_t initBlocks,
    int64_t localStart, int numThreads, Func f)
{
    auto forced = [&](float raw, int32_t block) { f(applyForcing(raw, block, initBlocks, localStart), block); };

    if (blockStride != 1)
    {
        for (int32_t block = static_cast<int32_t>(threadIdx.x); block < rowEnd; block += numThreads)
        {
            forced(rowBase[block * blockStride], block);
        }
        return;
    }

    constexpr int kItemsPerVec = 4;
    // Lead-in scalars until the row base is float4 aligned. There are fewer
    // than kItemsPerVec of them, and fewer than kItemsPerVec in the tail, so
    // both edges are one guarded call per thread rather than a loop.
    int32_t skipCnt = (reinterpret_cast<size_t>(rowBase) % sizeof(float4))
        ? static_cast<int32_t>((sizeof(float4) - reinterpret_cast<size_t>(rowBase) % sizeof(float4)) / sizeof(float))
        : 0;
    skipCnt = min(skipCnt, rowEnd);
    float4 const* rowVec = reinterpret_cast<float4 const*>(rowBase + skipCnt);
    int32_t const numVec = (rowEnd - skipCnt) / kItemsPerVec;

    for (int32_t i = static_cast<int32_t>(threadIdx.x); i < numVec; i += numThreads)
    {
        float4 const wide = rowVec[i];
        float const values[kItemsPerVec] = {wide.x, wide.y, wide.z, wide.w};
        int32_t const block = skipCnt + i * kItemsPerVec;
#pragma unroll
        for (int j = 0; j < kItemsPerVec; ++j)
        {
            forced(values[j], block + j);
        }
    }

    static_assert(kWarpSize >= kItemsPerVec);
    int32_t const threadRank = static_cast<int32_t>(threadIdx.x);
    if (threadRank < skipCnt)
    {
        forced(rowBase[threadRank], threadRank);
    }
    int32_t const tailBlock = skipCnt + numVec * kItemsPerVec + threadRank;
    if (tailBlock < rowEnd)
    {
        forced(rowBase[tailBlock], tailBlock);
    }
}

// One refinement step. Returns true if the threshold bin overflowed the staging
// buffer and the caller must refine further.
template <int step, int kNumThreadsPerBlock, int kNumBins, typename SmemFinalType>
__device__ bool processHistogramStep(float const* __restrict__ rowBase, int64_t blockStride, int32_t rowEnd,
    int32_t initBlocks, int64_t localStart, uint32_t& logitPattern, int& thresholdBinIdx, int32_t* smemOutput,
    int* smemThresholdBinIdx, int* smemFinalDstIdx, int* smemFinalBinSize, int* smemFoundTopKValues,
    SmemFinalType& smemFinal)
{
    // Step 0 is only an fp16 approximation. If it could not resolve the top-k
    // the fp32 radix restarts from scratch, so discard what step 0 emitted
    // rather than double-counting entries that pass both thresholds.
    if constexpr (step == 1)
    {
        if (threadIdx.x == 0)
        {
            smemFoundTopKValues[0] = 0;
            smemFinalDstIdx[0] = 0;
        }
    }

    for (int idx = threadIdx.x; idx < kNumBins; idx += kNumThreadsPerBlock)
    {
        smemFinal.histo.data[idx] = 0;
    }
    __syncthreads();

    constexpr int kBinBits = kNumBins == 2048 ? 11 : 10;
    constexpr int patternShift = step < 2 ? 0 : step == 2 ? 32 - kBinBits : 32 - 2 * kBinBits;
    if constexpr (step == 2)
    {
        logitPattern = static_cast<uint32_t>(thresholdBinIdx & (kNumBins - 1)) << patternShift;
    }
    else if constexpr (step == 3)
    {
        logitPattern |= static_cast<uint32_t>(thresholdBinIdx & (kNumBins - 1)) << patternShift;
    }

    forEachScore(rowBase, blockStride, rowEnd, initBlocks, localStart, kNumThreadsPerBlock,
        [&](float score, int32_t /* block */)
        {
            if (isPartialMatch<patternShift>(score, logitPattern))
            {
                atomicAdd(&smemFinal.histo.data[extractBinIdx<step, kNumBins>(score)], 1);
            }
        });
    __syncthreads();

    // Scan the histogram to find the bin the top-k boundary falls into.
    int lastValue = smemFoundTopKValues[0];
    for (int round = 0; round < kNumBins / kNumThreadsPerBlock; ++round)
    {
        int const idx = static_cast<int>(threadIdx.x) + kNumThreadsPerBlock * round;
        int const binCount = smemFinal.histo.data[idx];
        __syncthreads();

        int prefixSum{0}, totalSum{0};
        using Scan = cub::BlockScan<int, kNumThreadsPerBlock>;
        Scan(smemFinal.histo.scan).ExclusiveSum(binCount, prefixSum, totalSum);

        prefixSum += lastValue;
        totalSum += lastValue;
        smemFinal.histo.data[idx] = prefixSum;
        __syncthreads();

        bool foundThreshold = false;
        if (prefixSum < kTopK)
        {
            int const nextPrefixSum
                = static_cast<int>(threadIdx.x) == kNumThreadsPerBlock - 1 ? totalSum : smemFinal.histo.data[idx + 1];
            if (nextPrefixSum >= kTopK)
            {
                smemThresholdBinIdx[0] = idx;
                smemFinalBinSize[0] = nextPrefixSum - prefixSum;
                foundThreshold = true;
            }
        }

        if (__syncthreads_or(foundThreshold))
        {
            break;
        }
        lastValue = totalSum;
    }
    __syncthreads();

    thresholdBinIdx = smemThresholdBinIdx[0];

    forEachScore(rowBase, blockStride, rowEnd, initBlocks, localStart, kNumThreadsPerBlock,
        [&](float score, int32_t block)
        {
            if (!isPartialMatch<patternShift>(score, logitPattern))
            {
                return;
            }
            uint32_t const binIdx = extractBinIdx<step, kNumBins>(score);
            if (binIdx < thresholdBinIdx)
            {
                // Strictly above the boundary, so it is in the top-k outright.
                // The threshold bin is picked so fewer than kTopK elements can
                // land here; the guard only keeps a malformed threshold from
                // running off the end of smemOutput.
                int const dstIdx = atomicAdd(&smemFoundTopKValues[0], 1);
                if (dstIdx < kTopK)
                {
                    smemOutput[dstIdx] = block;
                }
            }
            if constexpr (step < 3)
            {
                if (binIdx == thresholdBinIdx && smemFinalBinSize[0] <= kNumFinalItems)
                {
                    int const dstIdx = atomicAdd(&smemFinalDstIdx[0], 1);
                    smemFinal.items.logits[dstIdx] = score;
                    smemFinal.items.indices[dstIdx] = block;
                }
            }
            else if (binIdx == thresholdBinIdx)
            {
                // Everything left in the threshold bin shares all 32 bits, so
                // the remaining slots are filled in whatever order the atomics
                // resolve. Reaching here needs more than kNumFinalItems
                // bit-identical scores in one row, which the ties the caller
                // can actually produce (the init / local sentinels) never hit.
                int const dstIdx = atomicAdd(&smemFinal.histo.data[binIdx], 1);
                if (dstIdx < kTopK)
                {
                    smemOutput[dstIdx] = block;
                }
            }
        });
    __syncthreads();

    return smemFinalBinSize[0] > kNumFinalItems;
}

// Ascending 32-element bitonic sort, one key per lane. Lanes with no real data
// pass ~0u, which sorts to the tail.
__forceinline__ __device__ void warpBitonicSortAsc32(uint32_t& key, uint32_t lane)
{
#pragma unroll
    for (int size = 2; size <= kWarpSize; size *= 2)
    {
#pragma unroll
        for (int stride = size / 2; stride > 0; stride /= 2)
        {
            uint32_t const partner = __shfl_xor_sync(kFullWarpMask, key, stride);
            bool const ascending = (lane & size) == 0;
            bool const isLower = (lane & stride) == 0;
            if (isLower == ascending)
            {
                key = min(key, partner);
            }
            else
            {
                key = max(key, partner);
            }
        }
    }
}

template <int kNumThreadsPerBlock, int kNumBins, bool HeadMajorOutput>
__global__ __launch_bounds__(kNumThreadsPerBlock) void minimaxM3SelectBlocksHistogramKernel(
    float const* __restrict__ scores, int64_t headStride, int64_t blockStride, int64_t queryStride,
    int32_t const* __restrict__ nValidBlocks, int32_t* __restrict__ output, int32_t numKvHeads, int32_t numBlocks,
    int32_t totalQueries, int32_t initBlocks, int32_t localBlocks)
{
    static_assert(kNumBins % kNumThreadsPerBlock == 0);
    static_assert(kNumFinalItems >= kTopK);

    using Scan = cub::BlockScan<int, kNumThreadsPerBlock>;

    struct FinalItems
    {
        int indices[kNumFinalItems];
        float logits[kNumFinalItems];
    };

    struct Histogram
    {
        typename Scan::TempStorage scan;
        int data[kNumBins];
    };

    __shared__ union
    {
        FinalItems items;
        Histogram histo;
    } smemFinal;

    __shared__ int32_t smemOutput[kTopK];
    __shared__ int smemThresholdBinIdx[1];
    __shared__ int smemFinalDstIdx[1];
    __shared__ int smemFinalBinSize[1];
    __shared__ int smemFoundTopKValues[1];

    int32_t const outputRow = blockIdx.x;
    int32_t const query = outputRow / numKvHeads;
    int32_t const kvHead = outputRow % numKvHeads;
    int32_t const rawValidBlocks = nValidBlocks[query];
    int32_t const validBlocks = max(0, min(rawValidBlocks, numBlocks));
    int64_t const localStart
        = max(static_cast<int64_t>(rawValidBlocks) - static_cast<int64_t>(localBlocks), static_cast<int64_t>(0));
    float const* __restrict__ rowBase
        = scores + static_cast<int64_t>(kvHead) * headStride + static_cast<int64_t>(query) * queryStride;

    for (int i = threadIdx.x; i < kTopK; i += kNumThreadsPerBlock)
    {
        smemOutput[i] = -1;
    }
    if (threadIdx.x == 0)
    {
        smemFinalDstIdx[0] = 0;
        smemFoundTopKValues[0] = 0;
    }
    __syncthreads();

    if (validBlocks > kTopK)
    {
        int thresholdBinIdx = -1;
        uint32_t logitPattern = 0;
        bool continueToNextStep = processHistogramStep<0, kNumThreadsPerBlock, kNumBins>(rowBase, blockStride,
            validBlocks, initBlocks, localStart, logitPattern, thresholdBinIdx, smemOutput, smemThresholdBinIdx,
            smemFinalDstIdx, smemFinalBinSize, smemFoundTopKValues, smemFinal);
        if (continueToNextStep)
        {
            continueToNextStep = processHistogramStep<1, kNumThreadsPerBlock, kNumBins>(rowBase, blockStride,
                validBlocks, initBlocks, localStart, logitPattern, thresholdBinIdx, smemOutput, smemThresholdBinIdx,
                smemFinalDstIdx, smemFinalBinSize, smemFoundTopKValues, smemFinal);
        }
        if (continueToNextStep)
        {
            continueToNextStep = processHistogramStep<2, kNumThreadsPerBlock, kNumBins>(rowBase, blockStride,
                validBlocks, initBlocks, localStart, logitPattern, thresholdBinIdx, smemOutput, smemThresholdBinIdx,
                smemFinalDstIdx, smemFinalBinSize, smemFoundTopKValues, smemFinal);
        }
        if (continueToNextStep)
        {
            processHistogramStep<3, kNumThreadsPerBlock, kNumBins>(rowBase, blockStride, validBlocks, initBlocks,
                localStart, logitPattern, thresholdBinIdx, smemOutput, smemThresholdBinIdx, smemFinalDstIdx,
                smemFinalBinSize, smemFoundTopKValues, smemFinal);
        }

        if (!continueToNextStep)
        {
            // The threshold bin fit the staging buffer: rank the staged
            // boundary candidates by score, breaking ties towards the smaller
            // block index, and emit the slots the steps above did not fill.
            int const baseIdx = smemFoundTopKValues[0];
            int const finalCount = smemFinalDstIdx[0];
            for (int i = threadIdx.x; i < finalCount; i += kNumThreadsPerBlock)
            {
                float const logit = smemFinal.items.logits[i];
                int const block = smemFinal.items.indices[i];
                int outIndex = 0;
                for (int j = 0; j < finalCount; ++j)
                {
                    float const otherLogit = smemFinal.items.logits[j];
                    if (logit < otherLogit || (logit == otherLogit && block > smemFinal.items.indices[j]))
                    {
                        ++outIndex;
                    }
                }
                if (outIndex + baseIdx < kTopK)
                {
                    smemOutput[outIndex + baseIdx] = block;
                }
            }
            __syncthreads();
        }
    }
    else
    {
        // Every valid block is selected; the tail stays at the -1 fill.
        for (int i = threadIdx.x; i < validBlocks; i += kNumThreadsPerBlock)
        {
            smemOutput[i] = i;
        }
    }

    __syncthreads();

    // One warp finishes: re-read each selected score so that a block forced to
    // nothing better than -inf is dropped (the reference emits -1 for those),
    // then sort ascending by block index with -1 padding at the tail.
    if (threadIdx.x >= kWarpSize)
    {
        return;
    }
    uint32_t const lane = threadIdx.x;
    uint32_t key = ~0U;
    if (lane < kTopK)
    {
        int32_t const block = smemOutput[lane];
        if (block >= 0
            && applyForcing(rowBase[static_cast<int64_t>(block) * blockStride], block, initBlocks, localStart)
                != -INFINITY)
        {
            key = static_cast<uint32_t>(block);
        }
    }
    warpBitonicSortAsc32(key, lane);
    if (lane < kTopK)
    {
        output[outputOffset<HeadMajorOutput>(query, kvHead, totalQueries, numKvHeads, static_cast<int32_t>(lane))]
            = key == ~0U ? -1 : static_cast<int32_t>(key);
    }
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
    if (numBlocks > kSmallMaxBlocks)
    {
        constexpr int kHistogramThreads = 512;
        constexpr int kHistogramBins = 2048;
        minimaxM3SelectBlocksHistogramKernel<kHistogramThreads, kHistogramBins, HeadMajorOutput>
            <<<numOutputRows, kHistogramThreads, 0, stream>>>(scores, headStride, blockStride, queryStride,
                nValidBlocks, output, numKvHeads, numBlocks, totalQueries, initBlocks, localBlocks);
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
    else
    {
        minimaxM3SelectBlocks128Kernel<HeadMajorOutput><<<gridSize, kThreadsPerBlock, 0, stream>>>(scores, headStride,
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
