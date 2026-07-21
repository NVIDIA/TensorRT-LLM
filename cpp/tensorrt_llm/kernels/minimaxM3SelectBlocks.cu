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
constexpr float kInitScore = 1.0e30F;
constexpr float kLocalScore = 1.0e29F;

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
            output[static_cast<int64_t>(outputRow) * kTopK + rank] = selectedIndices[rank];
        }
    }
}

} // namespace

void invokeMinimaxM3SelectBlocks(float const* scores, int64_t headStride, int64_t blockStride, int64_t queryStride,
    int32_t const* nValidBlocks, int32_t* output, int32_t numKvHeads, int32_t numBlocks, int32_t totalQueries,
    int32_t initBlocks, int32_t localBlocks, cudaStream_t stream)
{
    int32_t const numOutputRows = totalQueries * numKvHeads;
    if (numOutputRows == 0)
    {
        return;
    }
    int32_t const gridSize = (numOutputRows + kWarpsPerBlock - 1) / kWarpsPerBlock;
    minimaxM3SelectBlocksKernel<<<gridSize, kThreadsPerBlock, 0, stream>>>(scores, headStride, blockStride, queryStride,
        nValidBlocks, output, numKvHeads, numBlocks, totalQueries, initBlocks, localBlocks);
}

} // namespace kernels

TRTLLM_NAMESPACE_END
