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

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

// Each thread handles one element at (token_id, col).
// Grid: (num_tokens, ceil(numTopkTokens / blockDim.x))
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

    // Load request id for this token
    int32_t const req = reqId[tokenId];

    // Load token index
    int32_t const tok = tokenIndices[tokenId * tiStride0 + col * tiStride1];

    // Invalid token → output -1
    if (tok < 0)
    {
        output[tokenId * outStride0 + col * outStride1] = -1;
        return;
    }

    // Compute block id and in-block offset
    int32_t const blockId = tok / blockSize;
    int32_t const inblockOff = tok % blockSize + layerId * blockSize;

    // Guard block_table access
    if (blockId >= maxNumBlocksPerReq)
    {
        output[tokenId * outStride0 + col * outStride1] = -1;
        return;
    }

    int32_t const base = blockTable[req * btStride0 + blockId * btStride1];

    // Padding entry in block table
    if (base < 0)
    {
        output[tokenId * outStride0 + col * outStride1] = -1;
        return;
    }

    output[tokenId * outStride0 + col * outStride1] = base * strideFactor + inblockOff;
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
