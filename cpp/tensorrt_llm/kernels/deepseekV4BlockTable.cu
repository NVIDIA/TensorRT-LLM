/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/kernels/deepseekV4BlockTable.h"

#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace
{

constexpr int32_t kBadPageIndex = -1;
constexpr int32_t kThreadsPerBlock = 256;
constexpr int32_t kVecThreadsPerBlock = 128;
constexpr int32_t kRowKernelMinBlocks = 256;
constexpr int32_t kVecRowsPerBlock = 8;

__device__ __forceinline__ int32_t computeBasePageIndex(int32_t const* __restrict__ blockOffsets,
    int32_t const* __restrict__ copyIdx, int64_t const* __restrict__ poolIds, bool const* __restrict__ validPool,
    int32_t const* __restrict__ scales, int32_t const* __restrict__ layerOffsets, int32_t numPools,
    int32_t copyIdxCapacity, int32_t numAttnTypes, int32_t maxBlocksPerSeq, int32_t layerId, int32_t attnTypeId,
    int32_t tableId, int32_t blockId)
{
    int32_t const layerAttnOffset = layerId * numAttnTypes + attnTypeId;
    int64_t const poolId64 = poolIds[layerAttnOffset];
    bool const isValidPool = validPool[layerAttnOffset] && poolId64 >= 0 && poolId64 < numPools;
    if (!isValidPool)
    {
        return kBadPageIndex;
    }

    int32_t const mappedTableId = copyIdx[tableId];
    if (mappedTableId < 0 || mappedTableId >= copyIdxCapacity)
    {
        return kBadPageIndex;
    }

    auto const poolId = static_cast<int32_t>(poolId64);
    int64_t const blockOffsetsIndex
        = (((static_cast<int64_t>(poolId) * copyIdxCapacity + mappedTableId) * 2) * maxBlocksPerSeq) + blockId;
    int32_t const base = blockOffsets[blockOffsetsIndex];
    if (base == kBadPageIndex)
    {
        return kBadPageIndex;
    }

    return base * scales[layerAttnOffset] + layerOffsets[layerAttnOffset];
}

__device__ __forceinline__ int32_t applyScaleAndOffset(int32_t base, int32_t scale, int32_t layerOffset)
{
    return base == kBadPageIndex ? kBadPageIndex : base * scale + layerOffset;
}

__device__ __forceinline__ void fillBadSlidingBlockTableRow(int32_t* outputRow, int32_t maxBlocksPerSeq, bool useVec4)
{
    if (useVec4)
    {
        int4 const bad = {kBadPageIndex, kBadPageIndex, kBadPageIndex, kBadPageIndex};
        auto* outputVec = reinterpret_cast<int4*>(outputRow);
        int32_t const vecsPerRow = maxBlocksPerSeq / 4;
        for (int32_t vecId = threadIdx.x; vecId < vecsPerRow; vecId += blockDim.x)
        {
            outputVec[vecId] = bad;
        }
        return;
    }

    for (int32_t blockId = threadIdx.x; blockId < maxBlocksPerSeq; blockId += blockDim.x)
    {
        outputRow[blockId] = kBadPageIndex;
    }
}

__global__ void computeSlidingBlockTablesRowsTiledKernel(int32_t const* __restrict__ blockOffsets,
    int32_t const* __restrict__ copyIdx, int64_t const* __restrict__ poolIds, bool const* __restrict__ validPool,
    int32_t const* __restrict__ scales, int32_t const* __restrict__ layerOffsets, int32_t* __restrict__ output,
    int32_t numPools, int32_t copyIdxCapacity, int32_t numLayerAttn, int32_t numTables, int32_t maxBlocksPerSeq)
{
    bool const useVec4 = maxBlocksPerSeq % 4 == 0;
    int32_t const vecsPerRow = maxBlocksPerSeq / 4;
    int32_t const firstTableId = static_cast<int32_t>(blockIdx.x) * kVecRowsPerBlock;
    int32_t const layerAttnOffset = static_cast<int32_t>(blockIdx.y);
    if (layerAttnOffset >= numLayerAttn)
    {
        return;
    }

    int64_t const poolId64 = poolIds[layerAttnOffset];
    bool const isValidPool = validPool[layerAttnOffset] && poolId64 >= 0 && poolId64 < numPools;
    if (!isValidPool)
    {
#pragma unroll
        for (int32_t localRow = 0; localRow < kVecRowsPerBlock; ++localRow)
        {
            int32_t const tableId = firstTableId + localRow;
            if (tableId >= numTables)
            {
                continue;
            }

            int64_t const outputOffset
                = (static_cast<int64_t>(layerAttnOffset) * numTables + tableId) * maxBlocksPerSeq;
            fillBadSlidingBlockTableRow(output + outputOffset, maxBlocksPerSeq, useVec4);
        }
        return;
    }

    auto const poolId = static_cast<int32_t>(poolId64);
    int32_t const scale = scales[layerAttnOffset];
    int32_t const layerOffset = layerOffsets[layerAttnOffset];

#pragma unroll
    for (int32_t localRow = 0; localRow < kVecRowsPerBlock; ++localRow)
    {
        int32_t const tableId = firstTableId + localRow;
        if (tableId >= numTables)
        {
            continue;
        }

        int64_t const outputOffset = (static_cast<int64_t>(layerAttnOffset) * numTables + tableId) * maxBlocksPerSeq;
        auto* outputRow = output + outputOffset;
        int32_t const mappedTableId = copyIdx[tableId];
        bool const isValidTable = mappedTableId >= 0 && mappedTableId < copyIdxCapacity;
        if (!isValidTable)
        {
            fillBadSlidingBlockTableRow(outputRow, maxBlocksPerSeq, useVec4);
            continue;
        }

        int64_t const blockOffsetsOffset
            = ((static_cast<int64_t>(poolId) * copyIdxCapacity + mappedTableId) * 2) * maxBlocksPerSeq;
        auto const* blockOffsetsRow = blockOffsets + blockOffsetsOffset;
        if (useVec4)
        {
            auto const* blockOffsetsVec = reinterpret_cast<int4 const*>(blockOffsetsRow);
            auto* outputVec = reinterpret_cast<int4*>(outputRow);
            for (int32_t vecId = threadIdx.x; vecId < vecsPerRow; vecId += blockDim.x)
            {
                int4 const base = blockOffsetsVec[vecId];
                int4 const value = {applyScaleAndOffset(base.x, scale, layerOffset),
                    applyScaleAndOffset(base.y, scale, layerOffset), applyScaleAndOffset(base.z, scale, layerOffset),
                    applyScaleAndOffset(base.w, scale, layerOffset)};
                outputVec[vecId] = value;
            }
            continue;
        }

        for (int32_t blockId = threadIdx.x; blockId < maxBlocksPerSeq; blockId += blockDim.x)
        {
            int32_t const base = blockOffsetsRow[blockId];
            outputRow[blockId] = applyScaleAndOffset(base, scale, layerOffset);
        }
    }
}

__global__ void computeSlidingBlockTablesWithScratchKernel(int32_t const* __restrict__ blockOffsets,
    int32_t const* __restrict__ copyIdx, int64_t const* __restrict__ poolIds, bool const* __restrict__ validPool,
    int32_t const* __restrict__ scales, int32_t const* __restrict__ layerOffsets,
    int32_t const* __restrict__ scratchPages, int32_t const* __restrict__ scratchBegs,
    int32_t const* __restrict__ scratchEnds, int32_t const* __restrict__ scratchSlots,
    int32_t const* __restrict__ numContexts, int32_t* __restrict__ output, int64_t totalElements, int32_t numPools,
    int32_t copyIdxCapacity, int32_t numAttnTypes, int32_t numTables, int32_t maxBlocksPerSeq, int32_t scratchCapacity,
    int32_t maxScratchSlots)
{
    int64_t const linearIdx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linearIdx >= totalElements)
    {
        return;
    }

    int64_t remaining = linearIdx;
    int32_t const blockId = static_cast<int32_t>(remaining % maxBlocksPerSeq);
    remaining /= maxBlocksPerSeq;
    int32_t const tableId = static_cast<int32_t>(remaining % numTables);
    remaining /= numTables;
    int32_t const attnTypeId = static_cast<int32_t>(remaining % numAttnTypes);
    int32_t const layerId = static_cast<int32_t>(remaining / numAttnTypes);

    int32_t const layerAttnOffset = layerId * numAttnTypes + attnTypeId;
    int32_t const basePageIndex = computeBasePageIndex(blockOffsets, copyIdx, poolIds, validPool, scales, layerOffsets,
        numPools, copyIdxCapacity, numAttnTypes, maxBlocksPerSeq, layerId, attnTypeId, tableId, blockId);

    int64_t const poolId64 = poolIds[layerAttnOffset];
    bool const isValidPool = validPool[layerAttnOffset] && poolId64 >= 0 && poolId64 < numPools;
    int32_t const activeContexts = numContexts[0];
    bool const canUseScratch = isValidPool && tableId < scratchCapacity && tableId < activeContexts;

    if (!canUseScratch)
    {
        output[linearIdx] = basePageIndex;
        return;
    }

    auto const poolId = static_cast<int32_t>(poolId64);
    int64_t const scratchRangeOffset = static_cast<int64_t>(poolId) * scratchCapacity + tableId;
    int32_t const scratchBeg = scratchBegs[scratchRangeOffset];
    int32_t const scratchEnd = scratchEnds[scratchRangeOffset];
    bool const inScratchRange = blockId >= scratchBeg && blockId < scratchEnd;
    if (!inScratchRange)
    {
        output[linearIdx] = basePageIndex;
        return;
    }

    int32_t const scale = scales[layerAttnOffset];
    int32_t const rangeIndex = blockId - scratchBeg;
    int32_t const totalOffset = rangeIndex * scratchPages[layerAttnOffset];
    int32_t slotIdx = totalOffset / scale;
    if (slotIdx >= maxScratchSlots)
    {
        slotIdx = maxScratchSlots - 1;
    }

    int64_t const slotOffset = scratchRangeOffset * maxScratchSlots + slotIdx;
    int32_t const slotId = scratchSlots[slotOffset];
    int32_t const offset = totalOffset % scale;
    output[linearIdx] = slotId * scale + ((offset + layerOffsets[layerAttnOffset]) % scale);
}

__global__ void computeSlidingBlockTablesWithScratchRowsKernel(int32_t const* __restrict__ blockOffsets,
    int32_t const* __restrict__ copyIdx, int64_t const* __restrict__ poolIds, bool const* __restrict__ validPool,
    int32_t const* __restrict__ scales, int32_t const* __restrict__ layerOffsets,
    int32_t const* __restrict__ scratchPages, int32_t const* __restrict__ scratchBegs,
    int32_t const* __restrict__ scratchEnds, int32_t const* __restrict__ scratchSlots,
    int32_t const* __restrict__ numContexts, int32_t* __restrict__ output, int32_t numPools, int32_t copyIdxCapacity,
    int32_t numAttnTypes, int32_t numTables, int32_t maxBlocksPerSeq, int32_t scratchCapacity, int32_t maxScratchSlots)
{
    int32_t const rowIdx = static_cast<int32_t>(blockIdx.x);
    int32_t const tableId = rowIdx % numTables;
    int32_t const layerAttnIdx = rowIdx / numTables;
    int32_t const attnTypeId = layerAttnIdx % numAttnTypes;
    int32_t const layerId = layerAttnIdx / numAttnTypes;
    int32_t const layerAttnOffset = layerId * numAttnTypes + attnTypeId;
    int32_t const outputOffset = rowIdx * maxBlocksPerSeq;

    int64_t const poolId64 = poolIds[layerAttnOffset];
    bool const isValidPool = validPool[layerAttnOffset] && poolId64 >= 0 && poolId64 < numPools;
    if (!isValidPool)
    {
        for (int32_t blockId = threadIdx.x; blockId < maxBlocksPerSeq; blockId += blockDim.x)
        {
            output[outputOffset + blockId] = kBadPageIndex;
        }
        return;
    }

    auto const poolId = static_cast<int32_t>(poolId64);
    int32_t const scale = scales[layerAttnOffset];
    int32_t const layerOffset = layerOffsets[layerAttnOffset];
    int32_t const activeContexts = numContexts[0];
    bool const canUseScratch = tableId < scratchCapacity && tableId < activeContexts;
    int64_t const scratchRangeOffset = static_cast<int64_t>(poolId) * scratchCapacity + tableId;
    int32_t const scratchBeg = canUseScratch ? scratchBegs[scratchRangeOffset] : 0;
    int32_t const scratchEnd = canUseScratch ? scratchEnds[scratchRangeOffset] : 0;
    int32_t const scratchPageCount = scratchPages[layerAttnOffset];

    int32_t const mappedTableId = copyIdx[tableId];
    bool const isValidTable = mappedTableId >= 0 && mappedTableId < copyIdxCapacity;
    int64_t const blockOffsetsOffset
        = ((static_cast<int64_t>(poolId) * copyIdxCapacity + mappedTableId) * 2) * maxBlocksPerSeq;

    for (int32_t blockId = threadIdx.x; blockId < maxBlocksPerSeq; blockId += blockDim.x)
    {
        bool const inScratchRange = canUseScratch && blockId >= scratchBeg && blockId < scratchEnd;
        if (inScratchRange)
        {
            int32_t const rangeIndex = blockId - scratchBeg;
            int32_t const totalOffset = rangeIndex * scratchPageCount;
            int32_t slotIdx = totalOffset / scale;
            if (slotIdx >= maxScratchSlots)
            {
                slotIdx = maxScratchSlots - 1;
            }

            int64_t const slotOffset = scratchRangeOffset * maxScratchSlots + slotIdx;
            int32_t const slotId = scratchSlots[slotOffset];
            int32_t const offset = totalOffset % scale;
            output[outputOffset + blockId] = slotId * scale + ((offset + layerOffset) % scale);
            continue;
        }

        if (!isValidTable)
        {
            output[outputOffset + blockId] = kBadPageIndex;
            continue;
        }

        int32_t const base = blockOffsets[blockOffsetsOffset + blockId];
        output[outputOffset + blockId] = base == kBadPageIndex ? kBadPageIndex : base * scale + layerOffset;
    }
}

} // namespace

void invokeDeepseekV4ComputeSlidingBlockTables(int32_t const* blockOffsets, int32_t const* copyIdx,
    int64_t const* poolIds, bool const* validPool, int32_t const* scales, int32_t const* layerOffsets, int32_t* output,
    int32_t numPools, int32_t copyIdxCapacity, int32_t numLayers, int32_t numAttnTypes, int32_t numTables,
    int32_t maxBlocksPerSeq, cudaStream_t stream)
{
    int64_t const totalElements = static_cast<int64_t>(numLayers) * numAttnTypes * numTables * maxBlocksPerSeq;
    if (totalElements == 0)
    {
        return;
    }

    int32_t const numLayerAttn = numLayers * numAttnTypes;
    int32_t const itemsPerRow = maxBlocksPerSeq % 4 == 0 ? maxBlocksPerSeq / 4 : maxBlocksPerSeq;
    int32_t threadsPerBlock = itemsPerRow >= kVecThreadsPerBlock ? kVecThreadsPerBlock : itemsPerRow;
    if (threadsPerBlock < 64)
    {
        threadsPerBlock = 64;
    }

    dim3 const block(static_cast<uint32_t>(threadsPerBlock));
    dim3 const grid(static_cast<uint32_t>((numTables + kVecRowsPerBlock - 1) / kVecRowsPerBlock),
        static_cast<uint32_t>(numLayerAttn));
    computeSlidingBlockTablesRowsTiledKernel<<<grid, block, 0, stream>>>(blockOffsets, copyIdx, poolIds, validPool,
        scales, layerOffsets, output, numPools, copyIdxCapacity, numLayerAttn, numTables, maxBlocksPerSeq);
}

void invokeDeepseekV4ComputeSlidingBlockTablesWithScratch(int32_t const* blockOffsets, int32_t const* copyIdx,
    int64_t const* poolIds, bool const* validPool, int32_t const* scales, int32_t const* layerOffsets,
    int32_t const* scratchPages, int32_t const* scratchBegs, int32_t const* scratchEnds, int32_t const* scratchSlots,
    int32_t const* numContexts, int32_t* output, int32_t numPools, int32_t copyIdxCapacity, int32_t numLayers,
    int32_t numAttnTypes, int32_t numTables, int32_t maxBlocksPerSeq, int32_t scratchCapacity, int32_t maxScratchSlots,
    cudaStream_t stream)
{
    int64_t const totalElements = static_cast<int64_t>(numLayers) * numAttnTypes * numTables * maxBlocksPerSeq;
    if (totalElements == 0)
    {
        return;
    }

    if (maxBlocksPerSeq >= kRowKernelMinBlocks)
    {
        int32_t const numRows = numLayers * numAttnTypes * numTables;
        dim3 const block(kThreadsPerBlock);
        dim3 const grid(static_cast<uint32_t>(numRows));
        computeSlidingBlockTablesWithScratchRowsKernel<<<grid, block, 0, stream>>>(blockOffsets, copyIdx, poolIds,
            validPool, scales, layerOffsets, scratchPages, scratchBegs, scratchEnds, scratchSlots, numContexts, output,
            numPools, copyIdxCapacity, numAttnTypes, numTables, maxBlocksPerSeq, scratchCapacity, maxScratchSlots);
        return;
    }

    dim3 const block(kThreadsPerBlock);
    dim3 const grid(static_cast<uint32_t>((totalElements + kThreadsPerBlock - 1) / kThreadsPerBlock));
    computeSlidingBlockTablesWithScratchKernel<<<grid, block, 0, stream>>>(blockOffsets, copyIdx, poolIds, validPool,
        scales, layerOffsets, scratchPages, scratchBegs, scratchEnds, scratchSlots, numContexts, output, totalElements,
        numPools, copyIdxCapacity, numAttnTypes, numTables, maxBlocksPerSeq, scratchCapacity, maxScratchSlots);
}

} // namespace kernels

TRTLLM_NAMESPACE_END
