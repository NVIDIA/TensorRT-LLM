/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "convertReqIndexToGlobal.h"

#include <cstdint>

TRTLLM_NAMESPACE_BEGIN
namespace kernels
{

// Maps a request-local token index to a global index into the layer-interleaved
// KV pool: blockTable[req][tok / blockSize] * strideFactor + tok % blockSize + layerId * blockSize.
// Negative tokens, out-of-range blocks and padding block-table entries map to -1.
__device__ __forceinline__ int32_t convertOne(int32_t tok, int32_t const* __restrict__ btRow, int32_t maxNumBlocksPerReq,
    int32_t blockSize, int32_t strideFactor, int64_t btStride1, int32_t layerOffset)
{
    if (tok < 0)
    {
        return -1;
    }
    int32_t const blockId = tok / blockSize;
    if (blockId >= maxNumBlocksPerReq)
    {
        return -1;
    }
    int32_t const base = btRow[blockId * btStride1];
    if (base < 0)
    {
        return -1;
    }
    return base * strideFactor + (tok - blockId * blockSize) + layerOffset;
}

// Generic (scalar) kernel: one element per thread. Grid: (num_tokens, ceil(numTopkTokens / blockDim.x)).
// Used when the row layout is not 16-byte vectorizable.
__global__ void convertReqIndexToGlobalKernel(int32_t const* __restrict__ reqId, int32_t const* __restrict__ blockTable,
    int32_t const* __restrict__ tokenIndices, int32_t* __restrict__ output, int32_t numTopkTokens,
    int32_t maxNumBlocksPerReq, int32_t blockSize, int32_t strideFactor, int32_t layerId, int64_t btStride0,
    int64_t btStride1, int64_t tiStride0, int64_t tiStride1, int64_t outStride0, int64_t outStride1)
{
    int32_t const tokenId = blockIdx.x;
    int32_t const col = blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= numTopkTokens)
    {
        return;
    }
    int32_t const req = reqId[tokenId];
    int32_t const* btRow = blockTable + req * btStride0;
    int32_t const tok = tokenIndices[tokenId * tiStride0 + col * tiStride1];
    output[tokenId * outStride0 + col * outStride1]
        = convertOne(tok, btRow, maxNumBlocksPerReq, blockSize, strideFactor, btStride1, layerId * blockSize);
}

// Vectorized kernels: each thread moves int4 (4 indices) per step.
//  - Prefill shape (many rows, e.g. 16k x 2048): kRowsPerBlock rows per block, threads stride over
//    the row -> few large blocks, reqId / block-table row hoisted per row.
//  - Decode shape (a handful of rows): one row per block and the row split across gridDim.y blocks,
//    so the ~2k indices of a single token still spread over several SMs. A 4-rows-per-block launch
//    put a whole decode row on one block and cost +0.24 ms ITL (78 layers) on GLM 5.2 pareto01.
// The op is pure data movement (16k x 2048 int32 = 134 MB read+write per call on GLM 5.2 16k prefill), and the
// one-element-per-thread version above ran ~6x off DRAM bandwidth (130 us vs ~20 us) because every block only
// moved 1 KB and every element paid a dependent reqId + block-table load. Here reqId and the block-table row
// pointer are loaded once per row, and the indices stream through as 16-byte vectors.
template <int kThreads, int kRowsPerBlock>
__global__ __launch_bounds__(kThreads) void convertReqIndexToGlobalVecKernel(int32_t const* __restrict__ reqId,
    int32_t const* __restrict__ blockTable, int4 const* __restrict__ tokenIndices, int4* __restrict__ output,
    int32_t numTokens, int32_t numVecPerRow, int32_t maxNumBlocksPerReq, int32_t blockSize, int32_t strideFactor,
    int32_t layerOffset, int64_t btStride0, int64_t btStride1)
{
    int32_t const rowBase = blockIdx.x * kRowsPerBlock;
#pragma unroll 1
    for (int32_t r = 0; r < kRowsPerBlock; ++r)
    {
        int32_t const row = rowBase + r;
        if (row >= numTokens)
        {
            return;
        }
        int32_t const req = reqId[row];
        int32_t const* btRow = blockTable + static_cast<int64_t>(req) * btStride0;
        int4 const* src = tokenIndices + static_cast<int64_t>(row) * numVecPerRow;
        int4* dst = output + static_cast<int64_t>(row) * numVecPerRow;
        for (int32_t v = threadIdx.x; v < numVecPerRow; v += kThreads)
        {
            int4 const in = src[v];
            int4 out;
            out.x = convertOne(in.x, btRow, maxNumBlocksPerReq, blockSize, strideFactor, btStride1, layerOffset);
            out.y = convertOne(in.y, btRow, maxNumBlocksPerReq, blockSize, strideFactor, btStride1, layerOffset);
            out.z = convertOne(in.z, btRow, maxNumBlocksPerReq, blockSize, strideFactor, btStride1, layerOffset);
            out.w = convertOne(in.w, btRow, maxNumBlocksPerReq, blockSize, strideFactor, btStride1, layerOffset);
            dst[v] = out;
        }
    }
}

template <int kThreads>
__global__ __launch_bounds__(kThreads) void convertReqIndexToGlobalVecRowSplitKernel(int32_t const* __restrict__ reqId,
    int32_t const* __restrict__ blockTable, int4 const* __restrict__ tokenIndices, int4* __restrict__ output,
    int32_t numVecPerRow, int32_t maxNumBlocksPerReq, int32_t blockSize, int32_t strideFactor, int32_t layerOffset,
    int64_t btStride0, int64_t btStride1)
{
    int32_t const row = blockIdx.x;
    int32_t const v = blockIdx.y * kThreads + threadIdx.x;
    if (v >= numVecPerRow)
    {
        return;
    }
    int32_t const req = reqId[row];
    int32_t const* btRow = blockTable + static_cast<int64_t>(req) * btStride0;
    int4 const in = tokenIndices[static_cast<int64_t>(row) * numVecPerRow + v];
    int4 out;
    out.x = convertOne(in.x, btRow, maxNumBlocksPerReq, blockSize, strideFactor, btStride1, layerOffset);
    out.y = convertOne(in.y, btRow, maxNumBlocksPerReq, blockSize, strideFactor, btStride1, layerOffset);
    out.z = convertOne(in.z, btRow, maxNumBlocksPerReq, blockSize, strideFactor, btStride1, layerOffset);
    out.w = convertOne(in.w, btRow, maxNumBlocksPerReq, blockSize, strideFactor, btStride1, layerOffset);
    output[static_cast<int64_t>(row) * numVecPerRow + v] = out;
}

void invokeConvertReqIndexToGlobal(int32_t const* reqId, int32_t const* blockTable, int32_t const* tokenIndices,
    int32_t* output, int32_t numTokens, int32_t numTopkTokens, int32_t maxNumBlocksPerReq, int32_t blockSize,
    int32_t strideFactor, int32_t layerId, int64_t btStride0, int64_t btStride1, int64_t tiStride0, int64_t tiStride1,
    int64_t outStride0, int64_t outStride1, cudaStream_t stream)
{
    if (numTokens == 0 || numTopkTokens == 0)
    {
        return;
    }
    bool const vectorizable = tiStride1 == 1 && outStride1 == 1 && tiStride0 == numTopkTokens
        && outStride0 == numTopkTokens && (numTopkTokens % 4) == 0
        && (reinterpret_cast<uintptr_t>(tokenIndices) % 16) == 0 && (reinterpret_cast<uintptr_t>(output) % 16) == 0;
    if (vectorizable)
    {
        constexpr int32_t kThreads = 256;
        constexpr int32_t kRowsPerBlock = 4;
        // Below this many rows (decode / small batches) a 4-rows-per-block launch cannot fill the GPU:
        // split each row across blocks instead. 132 SMs x 4 rows keeps the prefill path for >= 528 rows.
        constexpr int32_t kMinRowsForRowBlocks = 512;
        int32_t const numVecPerRow = numTopkTokens / 4;
        if (numTokens >= kMinRowsForRowBlocks)
        {
            dim3 const grid((numTokens + kRowsPerBlock - 1) / kRowsPerBlock);
            convertReqIndexToGlobalVecKernel<kThreads, kRowsPerBlock><<<grid, kThreads, 0, stream>>>(reqId, blockTable,
                reinterpret_cast<int4 const*>(tokenIndices), reinterpret_cast<int4*>(output), numTokens, numVecPerRow,
                maxNumBlocksPerReq, blockSize, strideFactor, layerId * blockSize, btStride0, btStride1);
        }
        else
        {
            dim3 const grid(numTokens, (numVecPerRow + kThreads - 1) / kThreads);
            convertReqIndexToGlobalVecRowSplitKernel<kThreads><<<grid, kThreads, 0, stream>>>(reqId, blockTable,
                reinterpret_cast<int4 const*>(tokenIndices), reinterpret_cast<int4*>(output), numVecPerRow,
                maxNumBlocksPerReq, blockSize, strideFactor, layerId * blockSize, btStride0, btStride1);
        }
        return;
    }
    constexpr int32_t kThreadsPerBlock = 256;
    int32_t const tilesPerRow = (numTopkTokens + kThreadsPerBlock - 1) / kThreadsPerBlock;
    dim3 const grid(numTokens, tilesPerRow);
    dim3 const block(kThreadsPerBlock);
    convertReqIndexToGlobalKernel<<<grid, block, 0, stream>>>(reqId, blockTable, tokenIndices, output, numTopkTokens,
        maxNumBlocksPerReq, blockSize, strideFactor, layerId, btStride0, btStride1, tiStride0, tiStride1, outStride0,
        outStride1);
}

// Grouped (cross-layer fan-out) variant: one launch produces the remap output for a whole
// full+shared indexer group. blockIdx.z selects the group member; layerIds[z] is that member's
// layer offset. The req_id / block_table / token_indices inputs are shared across all members
// (identical top-k within an indexer-share group); only the additive `layerId * blockSize` term
// differs per member. The per-member output is therefore bit-identical to calling the single-layer
// kernel with that member's layer offset. MLA-only (kv_factor=1, num_kv_heads=1), mirroring the
// single-layer op.
// Grid: (num_tokens, ceil(numTopkTokens / blockDim.x), group_size)
__global__ void convertReqIndexToGlobalGroupedKernel(int32_t const* __restrict__ reqId,
    int32_t const* __restrict__ blockTable, int32_t const* __restrict__ tokenIndices,
    int32_t const* __restrict__ layerIds, int32_t* __restrict__ output, int32_t numTopkTokens,
    int32_t maxNumBlocksPerReq, int32_t blockSize, int32_t strideFactor, int64_t btStride0, int64_t btStride1,
    int64_t tiStride0, int64_t tiStride1, int64_t outStrideG, int64_t outStride0, int64_t outStride1)
{
    int32_t const tokenId = blockIdx.x;
    int32_t const col = blockIdx.y * blockDim.x + threadIdx.x;
    int32_t const g = blockIdx.z;

    if (col >= numTopkTokens)
    {
        return;
    }

    int64_t const outIdx = static_cast<int64_t>(g) * outStrideG + tokenId * outStride0 + col * outStride1;

    // Load request id and token index (shared across group members)
    int32_t const req = reqId[tokenId];
    int32_t const tok = tokenIndices[tokenId * tiStride0 + col * tiStride1];

    // Invalid token → output -1
    if (tok < 0)
    {
        output[outIdx] = -1;
        return;
    }

    int32_t const layerId = layerIds[g];
    int32_t const blockId = tok / blockSize;
    int32_t const inblockOff = tok % blockSize + layerId * blockSize;

    // Guard block_table access
    if (blockId >= maxNumBlocksPerReq)
    {
        output[outIdx] = -1;
        return;
    }

    int32_t const base = blockTable[req * btStride0 + blockId * btStride1];

    // Padding entry in block table
    if (base < 0)
    {
        output[outIdx] = -1;
        return;
    }

    output[outIdx] = base * strideFactor + inblockOff;
}

void invokeConvertReqIndexToGlobalGrouped(int32_t const* reqId, int32_t const* blockTable, int32_t const* tokenIndices,
    int32_t const* layerIds, int32_t* output, int32_t numTokens, int32_t numTopkTokens, int32_t groupSize,
    int32_t maxNumBlocksPerReq, int32_t blockSize, int32_t strideFactor, int64_t btStride0, int64_t btStride1,
    int64_t tiStride0, int64_t tiStride1, int64_t outStrideG, int64_t outStride0, int64_t outStride1,
    cudaStream_t stream)
{
    if (numTokens == 0 || numTopkTokens == 0 || groupSize == 0)
    {
        return;
    }

    constexpr int32_t kThreadsPerBlock = 256;
    int32_t const tilesPerRow = (numTopkTokens + kThreadsPerBlock - 1) / kThreadsPerBlock;
    dim3 const grid(numTokens, tilesPerRow, groupSize);
    dim3 const block(kThreadsPerBlock);

    convertReqIndexToGlobalGroupedKernel<<<grid, block, 0, stream>>>(reqId, blockTable, tokenIndices, layerIds, output,
        numTopkTokens, maxNumBlocksPerReq, blockSize, strideFactor, btStride0, btStride1, tiStride0, tiStride1,
        outStrideG, outStride0, outStride1);
}

} // namespace kernels
TRTLLM_NAMESPACE_END
