/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tensorrt_llm/kernels/attentionMetadataKernels.h"

#include <algorithm>
#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{
constexpr int32_t kThreadsPerBlock = 256;
constexpr int32_t kVecThreadsPerBlock = 128;
// Mirrors kv_cache_manager_v2::kBadPageIndex; kept local so this file does not
// pull the batch_manager headers into device code.
constexpr int32_t kBadPageIndex = -1;

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

// ---------------------------------------------------------------------------
// One shared-page block table: gather block_offsets[poolId, copyIdx, 0, :] and
// map it with where(base == kBadPageIndex, kBadPageIndex, base * scale).
//
// Keeping it on the GPU removes a host gather of a few hundred KB plus the
// subsequent host->device staging copy from the decode critical path; callers
// that build several tables per iteration pay that cost once per table.
// ---------------------------------------------------------------------------
__global__ void computeSharedBlockTableKernel(int32_t const* __restrict__ blockOffsets,
    int32_t const* __restrict__ copyIdx, int32_t* __restrict__ output, int32_t poolId, int32_t scale,
    int32_t copyIdxCapacity, int32_t numTables, int32_t maxBlocksPerSeq)
{
    int32_t const tableId = static_cast<int32_t>(blockIdx.y);
    if (tableId >= numTables)
    {
        return;
    }

    int64_t const outputOffset = static_cast<int64_t>(tableId) * maxBlocksPerSeq;
    int32_t const mappedTableId = copyIdx[tableId];
    bool const validTable = mappedTableId >= 0 && mappedTableId < copyIdxCapacity;

    // blockOffsets layout is [numPools, copyIdxCapacity, 2, maxBlocksPerSeq];
    // the CPU path reads index 0 of the K/V dimension.
    int64_t const baseOffset
        = ((static_cast<int64_t>(poolId) * copyIdxCapacity + mappedTableId) * 2) * maxBlocksPerSeq;

    for (int32_t blockId = static_cast<int32_t>(blockIdx.x) * static_cast<int32_t>(blockDim.x)
             + static_cast<int32_t>(threadIdx.x);
         blockId < maxBlocksPerSeq;
         blockId += static_cast<int32_t>(gridDim.x) * static_cast<int32_t>(blockDim.x))
    {
        int32_t value = kBadPageIndex;
        if (validTable)
        {
            int32_t const base = blockOffsets[baseOffset + blockId];
            value = base == kBadPageIndex ? kBadPageIndex : base * scale;
        }
        output[outputOffset + blockId] = value;
    }
}
} // namespace

void invokeComputeTokenPositions(int32_t const* seqLens, int32_t const* cachedTokens,
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

void invokeComputeSharedBlockTable(int32_t const* blockOffsets, int32_t const* copyIdx, int32_t* output,
    int32_t poolId, int32_t scale, int32_t copyIdxCapacity, int32_t numTables, int32_t maxBlocksPerSeq,
    cudaStream_t stream)
{
    if (numTables <= 0 || maxBlocksPerSeq <= 0)
    {
        return;
    }

    int32_t const threadsPerBlock = kVecThreadsPerBlock;
    int32_t const blocksPerRow = std::min((maxBlocksPerSeq + threadsPerBlock - 1) / threadsPerBlock, 64);
    dim3 const block(static_cast<uint32_t>(threadsPerBlock));
    dim3 const grid(static_cast<uint32_t>(blocksPerRow), static_cast<uint32_t>(numTables));
    computeSharedBlockTableKernel<<<grid, block, 0, stream>>>(
        blockOffsets, copyIdx, output, poolId, scale, copyIdxCapacity, numTables, maxBlocksPerSeq);
}


} // namespace kernels

TRTLLM_NAMESPACE_END
