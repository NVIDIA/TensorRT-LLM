/*
 * Copyright (c) 2020-2026, NVIDIA CORPORATION. All rights reserved.
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

#include "prepareCustomMask.h"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"
#include <cstdint>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
////////////////////////////////////////////////////////////////////////////////////////////////////

constexpr int32_t kCustomMaskOffsetScanBlockSize = 256;

__device__ __host__ inline int32_t ceilDiv(int32_t a, int32_t b)
{
    return (a + b - 1) / b;
}

__device__ inline int64_t computeKeepsMmaAbCustomMaskSize(int32_t seqLenQ, int32_t seqLenKv,
    int32_t firstSparseMaskOffsetKv, int32_t numHeadsQPerKv, int32_t stepQ, int32_t stepKv)
{
    int32_t const numTilesQ = ceilDiv(seqLenQ * numHeadsQPerKv, stepQ);
    int32_t const firstSparseTile = firstSparseMaskOffsetKv / stepKv;
    int32_t const numCustomMaskTilesKv = ceilDiv(seqLenKv, stepKv) - firstSparseTile;

    return static_cast<int64_t>(numTilesQ) * numCustomMaskTilesKv * stepQ * stepKv / 32;
}

__device__ inline int64_t computeSwapsMmaAbCustomMaskSize(int32_t seqLenQ, int32_t seqLenKv,
    int32_t firstSparseMaskOffsetKv, int32_t numHeadsQPerKv, int32_t stepQ, int32_t stepKv, int32_t tileSizeQRaw,
    int32_t tileSizeKv)
{
    int32_t const numInstsQ = stepQ / tileSizeQRaw;
    int32_t const numInstsKv = stepKv / tileSizeKv;
    int32_t const tileSizeKvPerCta = stepKv;
    int32_t const tileSizeQ = ceilDiv(tileSizeQRaw, 32) * 32;
    int32_t const tileSizeQPerCta = tileSizeQ * numInstsQ;
    int32_t const numTilesQPerToken = ceilDiv(numHeadsQPerKv, tileSizeQPerCta);
    int32_t const numTilesQ = seqLenQ * numTilesQPerToken;
    int32_t const firstSparseTile = firstSparseMaskOffsetKv / tileSizeKvPerCta;
    int32_t const numCustomMaskTilesKv = ceilDiv(seqLenKv, tileSizeKvPerCta) - firstSparseTile;
    int32_t const perTileSize = numInstsQ * numInstsKv * (tileSizeQ * tileSizeKv) / 32;

    return static_cast<int64_t>(numTilesQ) * numCustomMaskTilesKv * perTileSize;
}

__global__ void computeCustomMaskOffsetsParallelKernel(
    TllmGenFmhaRunnerParams runnerParams, int32_t stepQ, int32_t stepKv)
{
    // One CTA computes deterministic batch-prefix offsets.
    __shared__ int64_t threadSums[kCustomMaskOffsetScanBlockSize];

    int32_t const batchSize = runnerParams.mBatchSize;
    int32_t const numHeadsQPerKv = runnerParams.mNumHeadsQPerKv;
    int32_t const* seqLensKvPtr = runnerParams.seqLensKvPtr;
    int32_t const* firstSparseMaskOffsetsKvPtr = runnerParams.firstSparseMaskOffsetsKvPtr;
    int32_t const itemsPerThread = ceilDiv(batchSize, static_cast<int32_t>(blockDim.x));
    int32_t const threadIdxX = static_cast<int32_t>(threadIdx.x);
    int32_t const startIdx = threadIdxX * itemsPerThread;
    int32_t endIdx = startIdx + itemsPerThread;
    if (endIdx > batchSize)
    {
        endIdx = batchSize;
    }

    int64_t threadSum = 0;
    for (int32_t idx = startIdx; idx < endIdx; ++idx)
    {
        int32_t const seqLenQ = runnerParams.seqLensQPtr[idx];
        int32_t const seqLenKv = seqLensKvPtr[idx];
        int32_t const firstSparseMaskOffsetKv = firstSparseMaskOffsetsKvPtr[idx];
        threadSum += computeKeepsMmaAbCustomMaskSize(
            seqLenQ, seqLenKv, firstSparseMaskOffsetKv, numHeadsQPerKv, stepQ, stepKv);
    }

    threadSums[threadIdxX] = threadSum;
    __syncthreads();

    for (int32_t stride = 1; stride < blockDim.x; stride <<= 1)
    {
        int64_t partialSum = 0;
        if (threadIdxX >= stride)
        {
            partialSum = threadSums[threadIdxX - stride];
        }
        __syncthreads();
        threadSums[threadIdxX] += partialSum;
        __syncthreads();
    }

    int64_t localOffset = threadSums[threadIdxX] - threadSum;
    for (int32_t idx = startIdx; idx < endIdx; ++idx)
    {
        runnerParams.customMaskOffsetsPtr[idx] = localOffset;

        int32_t const seqLenQ = runnerParams.seqLensQPtr[idx];
        int32_t const seqLenKv = seqLensKvPtr[idx];
        int32_t const firstSparseMaskOffsetKv = firstSparseMaskOffsetsKvPtr[idx];
        localOffset += computeKeepsMmaAbCustomMaskSize(
            seqLenQ, seqLenKv, firstSparseMaskOffsetKv, numHeadsQPerKv, stepQ, stepKv);
    }
}

void launchComputeCustomMaskOffsetsKernel(
    TllmGenFmhaRunnerParams const& runnerParams, int32_t stepQ, int32_t stepKv, cudaStream_t stream)
{
    computeCustomMaskOffsetsParallelKernel<<<1, kCustomMaskOffsetScanBlockSize, 0, stream>>>(
        runnerParams, stepQ, stepKv);
}

// Input: customMaskInput (generalPackedCustoMaskPtr) shape: [batch_size, seqLenQ, ceilDiv(seqLenKv-firstSparse, 32)]
// Output: customMaskInput shape:[batch_size,numTilesQ, numTilesKv, numInstsQ, numInstsKv, tileSizeQ, tileSizeKv]
// Output: customMaskOffsets shape:[batch_size]
// Output: firstSparseMaskOffsetsKv shape:[batch_size]
__global__ void prepareCustomMaskBuffersKernelForKeepsMmaAb(
    TllmGenFmhaRunnerParams runnerParams, int32_t stepQ, int32_t stepKv, int32_t tileSizeQ, int32_t tileSizeKv)
{
    int32_t const batchSize = runnerParams.mBatchSize;
    int32_t const numHeadsQPerKv = runnerParams.mNumHeadsQPerKv;
    int32_t const numInstsQ = stepQ / tileSizeQ;
    int32_t const numInstsKv = stepKv / tileSizeKv;
    int32_t const tileSizeQPerCta = stepQ;
    int32_t const tileSizeKvPerCta = stepKv;

    int32_t const* seqLensKvPtr = runnerParams.seqLensKvPtr;
    int64_t* customMaskOffsetsPtr = runnerParams.customMaskOffsetsPtr;
    uint32_t* customMaskPtr = runnerParams.customMaskPtr;
    int32_t const* customMaskInputPtr = runnerParams.generalPackedCustoMaskPtr;
    int32_t* firstSparseMaskOffsetsKvPtr = runnerParams.firstSparseMaskOffsetsKvPtr;

    int32_t const batchIdx = static_cast<int32_t>(blockIdx.x);
    int32_t const qThreadIdx = static_cast<int32_t>(threadIdx.x);
    int32_t const qGroupIdx = static_cast<int32_t>(blockIdx.y);
    int32_t const kvThreadIdx = static_cast<int32_t>(threadIdx.y);
    int32_t const kvGroupIdx = static_cast<int32_t>(blockIdx.z);

    if (batchIdx >= batchSize)
    {
        return;
    }
    // The first sparseMask offset in the Kv sequence dimension.
    int32_t const firstSparseMaskOffsetKv = firstSparseMaskOffsetsKvPtr[batchIdx];
    int32_t const firstSparseMaskTileOffsetKv = firstSparseMaskOffsetKv / tileSizeKvPerCta;
    int32_t const adjustedFirstSparseMaskOffsetKv = firstSparseMaskTileOffsetKv * tileSizeKvPerCta;

    // The sequence length of tensor Q.
    int32_t const seqLenQ = runnerParams.seqLensQPtr[batchIdx];
    // The sequence length of tensor KV.
    int32_t const seqLenKv = seqLensKvPtr[batchIdx];

    // Use the padded mask row stride from the Python buffer.
    int32_t const packedMaskMaxSeqLenQ
        = runnerParams.mPackedMaskMaxSeqLenQ > 0 ? runnerParams.mPackedMaskMaxSeqLenQ : seqLenQ;
    int32_t const packedMaskNumBlocks = ceilDiv(packedMaskMaxSeqLenQ, 32);

    // Calculate global Q token index (flattened across heads)
    int32_t const qTokensPerBlock = static_cast<int32_t>(blockDim.x);
    int32_t const flattenedQIdx = qGroupIdx * qTokensPerBlock + qThreadIdx;
    int32_t const totalQTokens = seqLenQ * numHeadsQPerKv;

    if (flattenedQIdx >= totalQTokens)
    {
        return;
    }

    int32_t const tokenIdxQ = flattenedQIdx / numHeadsQPerKv;
    int32_t const headIdxInGrp = flattenedQIdx % numHeadsQPerKv;

    // Iterate from adjustedFirstSparseMaskOffsetKv to seqLenKv
    int32_t const kvTokensPerBlock = static_cast<int32_t>(blockDim.y);
    int32_t const globalKvIdx = kvGroupIdx * kvTokensPerBlock + kvThreadIdx;
    int32_t const tokenIdxKv = adjustedFirstSparseMaskOffsetKv + globalKvIdx;

    // Check KV bounds
    if (tokenIdxKv >= seqLenKv)
    {
        return;
    }

    // Get the mask value for this (Q, KV) pair
    int32_t randomMask = 0;
    if (tokenIdxKv < firstSparseMaskOffsetKv)
    {
        // Dense region: always attend
        randomMask = 1;
    }
    else
    {
        // Sparse region: check the input mask
        // The KV dimension in the mask corresponds to Q positions (tree mask)
        int32_t const qPosInTree = tokenIdxKv - firstSparseMaskOffsetKv;
        if (qPosInTree < seqLenQ)
        {
            // Use padded mask stride; FMHA cumSeqLensQ does not describe mask storage.
            int32_t const rowOffset = batchIdx * packedMaskMaxSeqLenQ + tokenIdxQ;
            int32_t const qMaskBaseIdx = rowOffset * packedMaskNumBlocks;
            int32_t const packedMaskIdx = qMaskBaseIdx + (qPosInTree >> 5);
            int32_t const bitPos = qPosInTree & 0x1F;
            randomMask = (customMaskInputPtr[packedMaskIdx] >> bitPos) & 1;
        }
    }

    if (randomMask)
    {
        int32_t const numCustomMaskTilesKv = ceilDiv(seqLenKv, tileSizeKvPerCta) - firstSparseMaskTileOffsetKv;
        int64_t const customMaskOffset = customMaskOffsetsPtr[batchIdx];
        uint32_t* localCustomMaskPtr = customMaskPtr + customMaskOffset;

        // Calculate Q indices in the custom mask
        int32_t const customMaskTokenIdxQ = tokenIdxQ * numHeadsQPerKv + headIdxInGrp;
        int32_t const tileIdxQ = customMaskTokenIdxQ / tileSizeQPerCta;
        int32_t const instIdxQ = (customMaskTokenIdxQ % tileSizeQPerCta) / tileSizeQ;
        int32_t const tokenIdxInTileQ = (customMaskTokenIdxQ % tileSizeQPerCta) % tileSizeQ;

        // Calculate KV indices in the custom mask
        int32_t const customMaskTokenIdxKv = tokenIdxKv - adjustedFirstSparseMaskOffsetKv;
        int32_t const tileIdxKv = customMaskTokenIdxKv / tileSizeKvPerCta;
        int32_t const instIdxKv = (customMaskTokenIdxKv % tileSizeKvPerCta) / tileSizeKv;
        int32_t const tokenIdxInTileKv = (customMaskTokenIdxKv % tileSizeKvPerCta) % tileSizeKv;

        // Calculate final mask offset
        int64_t const tileBase = static_cast<int64_t>(tileIdxQ) * numCustomMaskTilesKv;
        int64_t const tileOffset = tileBase + tileIdxKv;
        int64_t const instOffset = tileOffset * numInstsQ * numInstsKv + (instIdxQ * numInstsKv + instIdxKv);
        int64_t const maskOffset
            = instOffset * tileSizeQ * tileSizeKv + (tokenIdxInTileQ * tileSizeKv + tokenIdxInTileKv);
        // The offset of uint32_t custom mask
        int64_t const offsetAsUInt32 = maskOffset >> 5;
        int32_t const bitPosInUInt32 = maskOffset & 0x1F;
        // Set the bit in uint32_t custom mask
        atomicOr(&localCustomMaskPtr[offsetAsUInt32], (1U << bitPosInUInt32));
    }
}

__global__ void adjustFirstSparseMaskOffsetsKernel(TllmGenFmhaRunnerParams runnerParams, int32_t stepKv)
{
    int32_t const batchSize = runnerParams.mBatchSize;
    int32_t const tileSizeKvPerCta = stepKv;
    int32_t const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batchSize)
        return;
    int32_t* firstSparseMaskOffsetsKvPtr = runnerParams.firstSparseMaskOffsetsKvPtr;
    int32_t const firstSparseMaskOffsetKv = firstSparseMaskOffsetsKvPtr[idx];
    // It needs to be adjusted to multiple of tileSizeKvPerCta
    int32_t const adjusted = (firstSparseMaskOffsetKv / tileSizeKvPerCta) * tileSizeKvPerCta;
    firstSparseMaskOffsetsKvPtr[idx] = adjusted;
}

void launchPrepareCustomMaskBuffersKernelForKeepsMmaAb(TllmGenFmhaRunnerParams const& runnerParams, int32_t stepQ,
    int32_t stepKv, int32_t tileSizeQ, int32_t tileSizeKv, cudaStream_t stream)
{
    int32_t const batchSize = runnerParams.mBatchSize;
    int32_t const maxSeqLenQ = runnerParams.mMaxSeqLenQ;
    int32_t const numHeadsQPerKv = runnerParams.mNumHeadsQPerKv;
    int32_t const tileSizeKvPerCta = stepKv;

    // Total Q tokens (flattened across heads)
    int32_t const maxTotalQTokens = maxSeqLenQ * numHeadsQPerKv;

    // Calculate the maximum KV range to process
    // The actual range is [adjustedFirstSparseMaskOffsetKv, seqLenKv)
    // adjustedFirstSparseMaskOffsetKv <= firstSparseMaskOffsetKv = seqLenKv - seqLenQ
    // So the maximum range length is: seqLenKv - adjustedFirstSparseMaskOffsetKv <= maxSeqLenQ + (tileSizeKvPerCta - 1)
    int32_t const maxKvRangeLength = maxSeqLenQ + (tileSizeKvPerCta - 1);

    int32_t const qTokensPerBlock = 64;
    int32_t const kvTokensPerBlock = 4;

    int32_t const numBlocksY = ceilDiv(maxTotalQTokens, qTokensPerBlock);
    int32_t const numBlocksZ = ceilDiv(maxKvRangeLength, kvTokensPerBlock);

    dim3 gridDim(batchSize, numBlocksY, numBlocksZ);
    dim3 blockDim(qTokensPerBlock, kvTokensPerBlock, 1);

    prepareCustomMaskBuffersKernelForKeepsMmaAb<<<gridDim, blockDim, 0, stream>>>(
        runnerParams, stepQ, stepKv, tileSizeQ, tileSizeKv);
    // Ensure adjusted firstSparse offsets are written only after all blocks finish
    {
        int const blockSize = 128;
        int const gridSize = (batchSize + blockSize - 1) / blockSize;
        adjustFirstSparseMaskOffsetsKernel<<<gridSize, blockSize, 0, stream>>>(runnerParams, stepKv);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// SwapsMmaAb custom mask for groupsTokensHeadsQ=false.
__global__ void prepareCustomMaskBuffersKernelForSwapsMmaAb(
    TllmGenFmhaRunnerParams runnerParams, int32_t stepQ, int32_t stepKv, int32_t tileSizeQRaw, int32_t tileSizeKv)
{
    int32_t const batchSize = runnerParams.mBatchSize;
    int32_t const numHeadsQPerKv = runnerParams.mNumHeadsQPerKv;
    int32_t const numInstsQ = stepQ / tileSizeQRaw;
    int32_t const numInstsKv = stepKv / tileSizeKv;
    int32_t const tileSizeKvPerCta = stepKv;
    // Pad tileSizeQ for uint32 packing.
    int32_t const tileSizeQ = ((tileSizeQRaw + 31) / 32) * 32;
    int32_t const tileSizeQPerCta = tileSizeQ * numInstsQ;
    int32_t const* seqLensKvPtr = runnerParams.seqLensKvPtr;
    int64_t* customMaskOffsetsPtr = runnerParams.customMaskOffsetsPtr;
    uint32_t* customMaskPtr = runnerParams.customMaskPtr;
    int32_t const* customMaskInputPtr = runnerParams.generalPackedCustoMaskPtr;
    int32_t* firstSparseMaskOffsetsKvPtr = runnerParams.firstSparseMaskOffsetsKvPtr;

    int32_t const batchIdx = static_cast<int32_t>(blockIdx.x);
    int32_t const tokenThreadIdx = static_cast<int32_t>(threadIdx.x);
    int32_t const tokenGroupIdx = static_cast<int32_t>(blockIdx.y);
    int32_t const kvThreadIdx = static_cast<int32_t>(threadIdx.y);
    int32_t const kvGroupIdx = static_cast<int32_t>(blockIdx.z);

    if (batchIdx >= batchSize)
    {
        return;
    }

    int32_t const firstSparseMaskOffsetKv = firstSparseMaskOffsetsKvPtr[batchIdx];
    int32_t const firstSparseMaskTileOffsetKv = firstSparseMaskOffsetKv / tileSizeKvPerCta;
    int32_t const adjustedFirstSparseMaskOffsetKv = firstSparseMaskTileOffsetKv * tileSizeKvPerCta;

    int32_t const seqLenQ = runnerParams.seqLensQPtr[batchIdx];
    int32_t const seqLenKv = seqLensKvPtr[batchIdx];

    // Use padded mask row stride; see KeepsMmaAb path.
    int32_t const packedMaskMaxSeqLenQ
        = runnerParams.mPackedMaskMaxSeqLenQ > 0 ? runnerParams.mPackedMaskMaxSeqLenQ : seqLenQ;
    int32_t const packedMaskNumBlocks = ceilDiv(packedMaskMaxSeqLenQ, 32);

    int32_t const tokensPerBlock = static_cast<int32_t>(blockDim.x);
    int32_t const tokenIdxQ = tokenGroupIdx * tokensPerBlock + tokenThreadIdx;
    if (tokenIdxQ >= seqLenQ)
    {
        return;
    }

    int32_t const kvTokensPerBlock = static_cast<int32_t>(blockDim.y);
    int32_t const globalKvIdx = kvGroupIdx * kvTokensPerBlock + kvThreadIdx;
    int32_t const tokenIdxKv = adjustedFirstSparseMaskOffsetKv + globalKvIdx;
    if (tokenIdxKv >= seqLenKv)
    {
        return;
    }

    int32_t randomMask = 0;
    if (tokenIdxKv < firstSparseMaskOffsetKv)
    {
        randomMask = 1;
    }
    else
    {
        int32_t const qPosInTree = tokenIdxKv - firstSparseMaskOffsetKv;
        if (qPosInTree < seqLenQ)
        {
            // Use padded mask row stride.
            int32_t const rowOffset = batchIdx * packedMaskMaxSeqLenQ + tokenIdxQ;
            int32_t const qMaskBaseIdx = rowOffset * packedMaskNumBlocks;
            int32_t const packedMaskIdx = qMaskBaseIdx + (qPosInTree >> 5);
            int32_t const bitPos = qPosInTree & 0x1F;
            randomMask = (customMaskInputPtr[packedMaskIdx] >> bitPos) & 1;
        }
    }

    if (randomMask)
    {
        int32_t const numCustomMaskTilesKv = ceilDiv(seqLenKv, tileSizeKvPerCta) - firstSparseMaskTileOffsetKv;
        int64_t const customMaskOffset = customMaskOffsetsPtr[batchIdx];
        uint32_t* localCustomMaskPtr = customMaskPtr + customMaskOffset;

        // One Q tile per token when heads fit in the padded tile.
        int32_t const numTilesQPerToken = ceilDiv(numHeadsQPerKv, tileSizeQPerCta);

        int32_t const customMaskKvIdx = tokenIdxKv - adjustedFirstSparseMaskOffsetKv;
        int32_t const tileIdxKv = customMaskKvIdx / tileSizeKvPerCta;
        int32_t const instIdxKv = (customMaskKvIdx % tileSizeKvPerCta) / tileSizeKv;
        int32_t const kvInTile = customMaskKvIdx % tileSizeKv;

        // Match trtllm-gen SwapsMmaAb LDTM bit layout.
        for (int32_t headIdxInGrp = 0; headIdxInGrp < numHeadsQPerKv; ++headIdxInGrp)
        {
            int32_t const customMaskTokenIdxQ = headIdxInGrp;
            int32_t tileIdxQ = customMaskTokenIdxQ / tileSizeQPerCta;
            tileIdxQ += tokenIdxQ * numTilesQPerToken;
            int32_t const instIdxQ = (customMaskTokenIdxQ % tileSizeQPerCta) / tileSizeQ;
            int32_t const tokenIdxInTileQ = (customMaskTokenIdxQ % tileSizeQPerCta) % tileSizeQ;

            int64_t const tileOffset = static_cast<int64_t>(tileIdxQ) * numCustomMaskTilesKv + tileIdxKv;
            int64_t const instOffset = tileOffset * numInstsQ * numInstsKv + (instIdxQ * numInstsKv + instIdxKv);
            int64_t maskOffset = instOffset * tileSizeQ * tileSizeKv;

            int32_t const tokenIdxInTileKv = kvInTile;
            int32_t const threadIdxQ = (tokenIdxInTileQ % 8) / 2;
            int32_t const threadIdxKv = (tokenIdxInTileKv % 8) + (tokenIdxInTileKv / 32) * 8;
            int32_t const tokenIdxInWarpTileKv = tokenIdxInTileKv % 32;
            int32_t const eltIdxInThread = (tokenIdxInTileQ % 2) + ((tokenIdxInWarpTileKv / 8) % 2) * 2
                + (tokenIdxInTileQ / 8) * 4 + (tokenIdxInWarpTileKv / 16) * 4 * (tileSizeQRaw / 8);
            maskOffset += (threadIdxKv * 4 + threadIdxQ) * 32 + eltIdxInThread;

            int64_t const offsetAsUInt32 = maskOffset / 32;
            int32_t const bitPosInUInt32 = maskOffset % 32;
            atomicOr(&localCustomMaskPtr[offsetAsUInt32], (1U << bitPosInUInt32));
        }
    }
}

void launchPrepareCustomMaskBuffersKernelForSwapsMmaAb(TllmGenFmhaRunnerParams const& runnerParams, int32_t stepQ,
    int32_t stepKv, int32_t tileSizeQ, int32_t tileSizeKv, cudaStream_t stream)
{
    int32_t const batchSize = runnerParams.mBatchSize;
    int32_t const maxSeqLenQ = runnerParams.mMaxSeqLenQ;
    int32_t const tileSizeKvPerCta = stepKv;

    int32_t const maxKvRangeLength = maxSeqLenQ + (tileSizeKvPerCta - 1);

    int32_t const tokensPerBlock = 64;
    int32_t const kvTokensPerBlock = 4;

    int32_t const numBlocksY = ceilDiv(maxSeqLenQ, tokensPerBlock);
    int32_t const numBlocksZ = ceilDiv(maxKvRangeLength, kvTokensPerBlock);

    dim3 gridDim(batchSize, numBlocksY, numBlocksZ);
    dim3 blockDim(tokensPerBlock, kvTokensPerBlock, 1);

    prepareCustomMaskBuffersKernelForSwapsMmaAb<<<gridDim, blockDim, 0, stream>>>(
        runnerParams, stepQ, stepKv, tileSizeQ, tileSizeKv);
    {
        int const blockSize = 128;
        int const gridSize = (batchSize + blockSize - 1) / blockSize;
        adjustFirstSparseMaskOffsetsKernel<<<gridSize, blockSize, 0, stream>>>(runnerParams, stepKv);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void computeCustomMaskOffsetsParallelKernelForSwapsMmaAb(
    TllmGenFmhaRunnerParams runnerParams, int32_t stepQ, int32_t stepKv, int32_t tileSizeQRaw, int32_t tileSizeKv)
{
    // One CTA computes deterministic batch-prefix offsets.
    __shared__ int64_t threadSums[kCustomMaskOffsetScanBlockSize];

    int32_t const batchSize = runnerParams.mBatchSize;
    int32_t const numHeadsQPerKv = runnerParams.mNumHeadsQPerKv;
    int32_t const* seqLensKvPtr = runnerParams.seqLensKvPtr;
    int32_t const* firstSparseMaskOffsetsKvPtr = runnerParams.firstSparseMaskOffsetsKvPtr;
    int32_t const itemsPerThread = ceilDiv(batchSize, static_cast<int32_t>(blockDim.x));
    int32_t const threadIdxX = static_cast<int32_t>(threadIdx.x);
    int32_t const startIdx = threadIdxX * itemsPerThread;
    int32_t endIdx = startIdx + itemsPerThread;
    if (endIdx > batchSize)
    {
        endIdx = batchSize;
    }

    int64_t threadSum = 0;
    for (int32_t idx = startIdx; idx < endIdx; ++idx)
    {
        int32_t const seqLenQ = runnerParams.seqLensQPtr[idx];
        int32_t const seqLenKv = seqLensKvPtr[idx];
        int32_t const firstSparseMaskOffsetKv = firstSparseMaskOffsetsKvPtr[idx];
        threadSum += computeSwapsMmaAbCustomMaskSize(
            seqLenQ, seqLenKv, firstSparseMaskOffsetKv, numHeadsQPerKv, stepQ, stepKv, tileSizeQRaw, tileSizeKv);
    }

    threadSums[threadIdxX] = threadSum;
    __syncthreads();

    for (int32_t stride = 1; stride < blockDim.x; stride <<= 1)
    {
        int64_t partialSum = 0;
        if (threadIdxX >= stride)
        {
            partialSum = threadSums[threadIdxX - stride];
        }
        __syncthreads();
        threadSums[threadIdxX] += partialSum;
        __syncthreads();
    }

    int64_t localOffset = threadSums[threadIdxX] - threadSum;
    for (int32_t idx = startIdx; idx < endIdx; ++idx)
    {
        runnerParams.customMaskOffsetsPtr[idx] = localOffset;

        int32_t const seqLenQ = runnerParams.seqLensQPtr[idx];
        int32_t const seqLenKv = seqLensKvPtr[idx];
        int32_t const firstSparseMaskOffsetKv = firstSparseMaskOffsetsKvPtr[idx];
        localOffset += computeSwapsMmaAbCustomMaskSize(
            seqLenQ, seqLenKv, firstSparseMaskOffsetKv, numHeadsQPerKv, stepQ, stepKv, tileSizeQRaw, tileSizeKv);
    }
}

void launchComputeCustomMaskOffsetsKernelForSwapsMmaAb(TllmGenFmhaRunnerParams const& runnerParams, int32_t stepQ,
    int32_t stepKv, int32_t tileSizeQ, int32_t tileSizeKv, cudaStream_t stream)
{
    computeCustomMaskOffsetsParallelKernelForSwapsMmaAb<<<1, kCustomMaskOffsetScanBlockSize, 0, stream>>>(
        runnerParams, stepQ, stepKv, tileSizeQ, tileSizeKv);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void runPrepareCustomMask(TllmGenFmhaRunnerParams const& runnerParams, FmhaKernelType kernelType, int32_t stepQ,
    int32_t stepKv, int32_t tileSizeQ, int32_t tileSizeKv, cudaStream_t stream)
{
    if (isKeepsMmaAbForGenerationKernel(static_cast<FmhaKernelType>(kernelType)))
    {
        int cta_tile_size = stepQ * stepKv;
        if (cta_tile_size > 128 * 128 * 2)
        {
            TLLM_LOG_ERROR(
                "TRTLLM-GEN needs larger buffer for custom mask preparation please enlarge it according to the "
                "formula: tile_size_q * tile_size_k * num_instances_q * num_instances_k");
            return;
        }
        launchComputeCustomMaskOffsetsKernel(runnerParams, stepQ, stepKv, stream);
        launchPrepareCustomMaskBuffersKernelForKeepsMmaAb(runnerParams, stepQ, stepKv, tileSizeQ, tileSizeKv, stream);
        TLLM_CUDA_CHECK(cudaGetLastError());
    }
    else if (isSwapsMmaAbForGenerationKernel(static_cast<FmhaKernelType>(kernelType)))
    {
        launchComputeCustomMaskOffsetsKernelForSwapsMmaAb(runnerParams, stepQ, stepKv, tileSizeQ, tileSizeKv, stream);
        launchPrepareCustomMaskBuffersKernelForSwapsMmaAb(runnerParams, stepQ, stepKv, tileSizeQ, tileSizeKv, stream);
        TLLM_CUDA_CHECK(cudaGetLastError());
    }
    else
    {
        TLLM_LOG_ERROR("TRTLLM-GEN does not support kernel type: %d for custom mask preparation", kernelType);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernels

TRTLLM_NAMESPACE_END
