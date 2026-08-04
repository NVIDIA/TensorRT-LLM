/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tensorrt_llm/kernels/deepseekV4TokenPositions.h"

#include <algorithm>
#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{
constexpr int32_t kThreadsPerBlock = 256;

// Phase 1: padded exclusive scan of seq_lens into cu_seq_lens.
// batchSize is the scheduler batch (a few hundred), so a single-block
// shared-memory scan avoids a separate cumsum launch.
template <int kMaxBatch>
__global__ void computeCuSeqLensKernel(
    int32_t const* __restrict__ seqLens, int32_t* __restrict__ cuSeqLens, int32_t batchSize)
{
    __shared__ int32_t buffers[2][kMaxBatch];

    int32_t const stride = static_cast<int32_t>(blockDim.x);
    int32_t const rounded = ((batchSize + stride - 1) / stride) * stride;

    for (int32_t i = static_cast<int32_t>(threadIdx.x); i < rounded; i += stride)
    {
        if (i < batchSize)
        {
            buffers[0][i] = seqLens[i];
        }
    }
    __syncthreads();

    int32_t src = 0;
    for (int32_t offset = 1; offset < batchSize; offset <<= 1)
    {
        int32_t const dst = src ^ 1;
        for (int32_t i = static_cast<int32_t>(threadIdx.x); i < rounded; i += stride)
        {
            if (i < batchSize)
            {
                buffers[dst][i] = buffers[src][i] + (i >= offset ? buffers[src][i - offset] : 0);
            }
        }
        __syncthreads();
        src = dst;
    }

    if (threadIdx.x == 0)
    {
        cuSeqLens[0] = 0;
    }
    for (int32_t i = static_cast<int32_t>(threadIdx.x); i < batchSize; i += stride)
    {
        cuSeqLens[i + 1] = buffers[src][i];
    }
}

// Phase 2: per-token request index and absolute position.
// Replaces the CPU repeat_interleave + pinned H2D memcpy on the prepare() path
// and the arange + searchsorted + two gathers on the update path.
__global__ void computeTokenPositionsKernel(int32_t const* __restrict__ cuSeqLens,
    int32_t const* __restrict__ cachedTokens, int32_t* __restrict__ reqIdxPerToken,
    int32_t* __restrict__ tokenPositions, int32_t batchSize, int32_t numTokens)
{
    for (int32_t t = blockIdx.x * blockDim.x + threadIdx.x; t < numTokens;
         t += gridDim.x * blockDim.x)
    {
        // searchsorted(cu_seq_lens[1:], t, right=True): largest j with cu[j] <= t.
        int32_t lo = 0;
        int32_t hi = batchSize;
        while (lo < hi)
        {
            int32_t const mid = lo + ((hi - lo) >> 1);
            if (cuSeqLens[mid + 1] <= t)
            {
                lo = mid + 1;
            }
            else
            {
                hi = mid;
            }
        }
        int32_t const reqIdx = min(lo, batchSize - 1);
        reqIdxPerToken[t] = reqIdx;
        if (tokenPositions != nullptr)
        {
            tokenPositions[t] = cachedTokens[reqIdx] + (t - cuSeqLens[reqIdx]);
        }
    }
}
} // namespace

void invokeDeepseekV4ComputeTokenPositions(int32_t const* seqLens, int32_t const* cachedTokens,
    int32_t* cuSeqLens, int32_t* reqIdxPerToken, int32_t* tokenPositions, int32_t batchSize,
    int32_t numTokens, bool computeCuSeqLens, cudaStream_t stream)
{
    if (batchSize <= 0)
    {
        return;
    }
    if (computeCuSeqLens)
    {
        dim3 const grid(1);
        dim3 const block(static_cast<uint32_t>(kThreadsPerBlock));
        if (batchSize <= 512)
        {
            computeCuSeqLensKernel<512><<<grid, block, 0, stream>>>(seqLens, cuSeqLens, batchSize);
        }
        else if (batchSize <= 2048)
        {
            computeCuSeqLensKernel<2048><<<grid, block, 0, stream>>>(seqLens, cuSeqLens, batchSize);
        }
        else
        {
            computeCuSeqLensKernel<kMaxTokenPositionScanBatch>
                <<<grid, block, 0, stream>>>(seqLens, cuSeqLens, batchSize);
        }
    }
    if (numTokens > 0)
    {
        int32_t const blocks
            = std::min((numTokens + kThreadsPerBlock - 1) / kThreadsPerBlock, 2048);
        computeTokenPositionsKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
            cuSeqLens, cachedTokens, reqIdxPerToken, tokenPositions, batchSize, numTokens);
    }
}

} // namespace kernels

TRTLLM_NAMESPACE_END
