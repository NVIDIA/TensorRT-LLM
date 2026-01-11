/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION. All rights reserved.
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
#include <cstdint>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ __host__ inline int32_t ceilDiv(int32_t a, int32_t b)
{
    return (a + b - 1) / b;
}

// Input: customMaskInput (generalPackedCustoMaskPtr) shape: [batch_size, seqLenQ, ceilDiv(seqLenKv-firstSparse, 32)]
// Output: customMaskInput shape:[batch_size,numTilesQ, numTilesKv, numInstsQ, numInstsKv, tileSizeQ, tileSizeKv]
// Output: customMaskOffsets shape:[batch_size]
// Output: firstSparseMaskOffsetsKv shape:[batch_size]
__global__ void prepareCustomMaskBuffersKernelForKeepsMmaAb(
    TllmGenFmhaRunnerParams runnerParams, TllmGenFmhaKernelMetaInfo kernelMeta)
{
    int32_t const batchSize = runnerParams.mBatchSize;
    int32_t const numHeadsQPerKv = runnerParams.mNumHeadsQPerKv;
    int32_t const tileSizeQ = kernelMeta.mTileSizeQ;
    int32_t const tileSizeKv = kernelMeta.mTileSizeKv;
    int32_t const numInstsQ = kernelMeta.mStepQ / kernelMeta.mTileSizeQ;
    int32_t const numInstsKv = kernelMeta.mStepKv / kernelMeta.mTileSizeKv;
    int32_t const tileSizeQPerCta = kernelMeta.mStepQ;
    int32_t const tileSizeKvPerCta = kernelMeta.mStepKv;

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
    int32_t const seqLenQ = runnerParams.seqlensQPtr[batchIdx];
    // The sequence length of tensor KV.
    int32_t const seqLenKv = seqLensKvPtr[batchIdx];

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
        // Input mask shape: [bs, seqLenQ, ceilDiv(seqLenQ, 32)]
        // The KV dimension in the mask corresponds to Q positions (tree mask)
        int32_t const qPosInTree = tokenIdxKv - firstSparseMaskOffsetKv;
        if (qPosInTree < seqLenQ)
        {
            int32_t const qMaskBaseIdx = (batchIdx * seqLenQ + tokenIdxQ) * ceilDiv(seqLenQ, 32);
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

__global__ void computeCustomMaskOffsetsKernel(
    TllmGenFmhaKernelMetaInfo kernelMeta, TllmGenFmhaRunnerParams runnerParams, unsigned long long* globalCounter)
{
    int32_t batchSize = runnerParams.mBatchSize;
    int32_t numHeadsQPerKv = runnerParams.mNumHeadsQPerKv;
    int32_t tileSizeQPerCta = kernelMeta.mStepQ;
    int32_t tileSizeKvPerCta = kernelMeta.mStepKv;
    int32_t const* seqLensKvPtr = runnerParams.seqLensKvPtr;
    int32_t const* firstSparseMaskOffsetsKvPtr = runnerParams.firstSparseMaskOffsetsKvPtr;

    typedef cub::BlockScan<int64_t, 128> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t maskSize = 0;

    if (idx < batchSize)
    {

        int32_t seqLenQ = runnerParams.seqlensQPtr[idx];
        int32_t seqLenKv = seqLensKvPtr[idx];
        int32_t firstSparseMaskOffsetKv = firstSparseMaskOffsetsKvPtr[idx];

        int32_t numTilesQ = (seqLenQ * numHeadsQPerKv + tileSizeQPerCta - 1) / tileSizeQPerCta;
        int32_t firstSparseTile = firstSparseMaskOffsetKv / tileSizeKvPerCta;
        int32_t numCustomMaskTilesKv = (seqLenKv + tileSizeKvPerCta - 1) / tileSizeKvPerCta - firstSparseTile;

        maskSize = static_cast<int64_t>(numTilesQ * numCustomMaskTilesKv * kernelMeta.mStepQ * kernelMeta.mStepKv / 32);
    }

    int64_t prefixOffset;
    int64_t blockSum;
    BlockScan(temp_storage).ExclusiveSum(maskSize, prefixOffset, blockSum);

    __shared__ unsigned long long blockBase;
    if (threadIdx.x == 0)
        blockBase = atomicAdd(globalCounter, (unsigned long long) blockSum);
    __syncthreads();

    if (idx < batchSize)
        runnerParams.customMaskOffsetsPtr[idx] = static_cast<int64_t>(blockBase) + prefixOffset;
}

void launchComputeCustomMaskOffsetsKernel(
    TllmGenFmhaKernelMetaInfo const& kernelMeta, TllmGenFmhaRunnerParams const& runnerParams, cudaStream_t stream)
{

    int32_t batchSize = runnerParams.mBatchSize;

    unsigned long long* d_globalCounter;
    cudaMallocAsync(&d_globalCounter, sizeof(unsigned long long), stream);
    cudaMemsetAsync(d_globalCounter, 0, sizeof(unsigned long long), stream);

    int blockSize = 128;
    int gridSize = (batchSize + blockSize - 1) / blockSize;
    computeCustomMaskOffsetsKernel<<<gridSize, blockSize, 0, stream>>>(kernelMeta, runnerParams, d_globalCounter);

    cudaFreeAsync(d_globalCounter, stream);
}

// Post-processing kernel to write adjusted firstSparseMaskOffsetsKv after all work is done
__global__ void adjustFirstSparseMaskOffsetsKernel(
    TllmGenFmhaRunnerParams runnerParams, TllmGenFmhaKernelMetaInfo kernelMeta)
{
    int32_t const batchSize = runnerParams.mBatchSize;
    int32_t const tileSizeKvPerCta = kernelMeta.mStepKv;
    int32_t const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batchSize)
        return;
    int32_t* firstSparseMaskOffsetsKvPtr = runnerParams.firstSparseMaskOffsetsKvPtr;
    int32_t const firstSparseMaskOffsetKv = firstSparseMaskOffsetsKvPtr[idx];
    // It needs to be adjusted to multiple of tileSizeKvPerCta
    int32_t const adjusted = (firstSparseMaskOffsetKv / tileSizeKvPerCta) * tileSizeKvPerCta;
    firstSparseMaskOffsetsKvPtr[idx] = adjusted;
}

void launchPrepareCustomMaskBuffersKernelForKeepsMmaAb(
    TllmGenFmhaRunnerParams const& runnerParams, TllmGenFmhaKernelMetaInfo const& kernelMeta, cudaStream_t stream)
{
    int32_t const batchSize = runnerParams.mBatchSize;
    int32_t const maxSeqLenQ = runnerParams.mMaxSeqLenQ;
    int32_t const numHeadsQPerKv = runnerParams.mNumHeadsQPerKv;
    int32_t const tileSizeKvPerCta = kernelMeta.mStepKv;

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

    prepareCustomMaskBuffersKernelForKeepsMmaAb<<<gridDim, blockDim, 0, stream>>>(runnerParams, kernelMeta);
    // Ensure adjusted firstSparse offsets are written only after all blocks finish
    {
        int const blockSize = 128;
        int const gridSize = (batchSize + blockSize - 1) / blockSize;
        adjustFirstSparseMaskOffsetsKernel<<<gridSize, blockSize, 0, stream>>>(runnerParams, kernelMeta);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void runPrepareCustomMask(
    TllmGenFmhaKernelMetaInfo const& kernelMeta, TllmGenFmhaRunnerParams const& runnerParams, cudaStream_t stream)
{
    if (isKeepsMmaAbForGenerationKernel(static_cast<FmhaKernelType>(kernelMeta.mKernelType)))
    {
        int cta_tile_size = kernelMeta.mStepQ * kernelMeta.mStepKv;
        if (cta_tile_size > 128 * 128 * 2)
        {
            TLLM_LOG_ERROR(
                "TRTLLM-GEN needs larger buffer for  custom mask preparation please enlarge it according to the "
                "formula: tile_size_q * tile_size_k * num_instances_q * num_instances_k");
            return;
        }
        // Step 1: Compute offsets on GPU using prefix sum
        launchComputeCustomMaskOffsetsKernel(kernelMeta, runnerParams, stream);
        // Step 2: Compute custom mask buffers
        launchPrepareCustomMaskBuffersKernelForKeepsMmaAb(runnerParams, kernelMeta, stream);
        TLLM_CUDA_CHECK(cudaGetLastError());
    }
    else
    {
        TLLM_LOG_ERROR(
            "TRTLLM-GEN does not support kernel type: %d for custom mask preparation", runnerParams.mKernelType);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernels

TRTLLM_NAMESPACE_END
