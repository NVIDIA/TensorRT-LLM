/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.
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

#include "fmhaRunnerParams.h"
#include "prepareCustomMask.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include <cstdint>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

namespace tensorrt_llm
{
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
    int32_t const tileSizeQPerCta = tileSizeQ * numInstsQ;
    int32_t const tileSizeKvPerCta = tileSizeKv * numInstsKv;

    int32_t const* seqLensKvPtr = runnerParams.seqLensKvPtr;
    int64_t* customMaskOffsetsPtr = runnerParams.customMaskOffsetsPtr;
    uint32_t* customMaskPtr = runnerParams.customMaskPtr;
    int32_t const* customMaskInputPtr = runnerParams.generalPackedCustoMaskPtr;
    int32_t* firstSparseMaskOffsetsKvPtr = runnerParams.firstSparseMaskOffsetsKvPtr;

    int32_t const batchIdx = blockIdx.x;
    int32_t const flattenedThreadIdx = blockIdx.y * blockDim.x + threadIdx.x;

    if (batchIdx >= batchSize)
        return;

    int32_t const seqLenQ = runnerParams.spec_decoding_generation_lengths[batchIdx];
    int32_t const seqLenKv = seqLensKvPtr[batchIdx];
    int32_t const totalQTokens = seqLenQ * numHeadsQPerKv;

    if (flattenedThreadIdx >= totalQTokens)
        return;

    int32_t const tokenIdxQ = flattenedThreadIdx / numHeadsQPerKv;
    int32_t const headIdxInGrp = flattenedThreadIdx % numHeadsQPerKv;

    int32_t const firstSparseMaskOffsetKv = firstSparseMaskOffsetsKvPtr[batchIdx];
    int32_t const firstSparseMaskTileOffsetKv = firstSparseMaskOffsetKv / tileSizeKvPerCta;
    int32_t const adjustedFirstSparseMaskOffsetKv = firstSparseMaskTileOffsetKv * tileSizeKvPerCta;

    if (flattenedThreadIdx == 0)
    {
        firstSparseMaskOffsetsKvPtr[batchIdx] = adjustedFirstSparseMaskOffsetKv;
    }

    int32_t const numCustomMaskTilesKv = ceilDiv(seqLenKv, tileSizeKvPerCta) - firstSparseMaskTileOffsetKv;
    int64_t const customMaskOffset = customMaskOffsetsPtr[batchIdx];
    uint32_t* localCustomMaskPtr = customMaskPtr + customMaskOffset;

    int32_t const qMaskBaseIdx = (batchIdx * seqLenQ + tokenIdxQ) * ceilDiv(seqLenKv - firstSparseMaskOffsetKv, 32);

    int32_t const customMaskTokenIdxQ = tokenIdxQ * numHeadsQPerKv + headIdxInGrp;
    int32_t const tileIdxQ = customMaskTokenIdxQ / tileSizeQPerCta;
    int32_t const instIdxQ = (customMaskTokenIdxQ % tileSizeQPerCta) / tileSizeQ;
    int32_t const tokenIdxInTileQ = (customMaskTokenIdxQ % tileSizeQPerCta) % tileSizeQ;

    int64_t const tileBase = static_cast<int64_t>(tileIdxQ) * numCustomMaskTilesKv;
    int32_t const instQBase = instIdxQ * numInstsKv;
    int64_t const maskRowBase = static_cast<int64_t>(tokenIdxInTileQ) * tileSizeKv;

    for (int32_t tokenIdxKv = adjustedFirstSparseMaskOffsetKv; tokenIdxKv < seqLenKv; ++tokenIdxKv)
    {

        int32_t randomMask = 0;
        if (tokenIdxKv < firstSparseMaskOffsetKv)
        {
            randomMask = 1;
        }
        else
        {
            int32_t const packedMaskIdx = qMaskBaseIdx + ((tokenIdxKv - firstSparseMaskOffsetKv) >> 5);
            int32_t const bitPos = (tokenIdxKv - firstSparseMaskOffsetKv) & 0x1F;
            randomMask = (customMaskInputPtr[packedMaskIdx] >> bitPos) & 1;
        }

        if (randomMask)
        {
            int32_t const customMaskTokenIdxKv = tokenIdxKv - adjustedFirstSparseMaskOffsetKv;
            int32_t const tileIdxKv = customMaskTokenIdxKv / tileSizeKvPerCta;
            int32_t const instIdxKv = (customMaskTokenIdxKv % tileSizeKvPerCta) / tileSizeKv;
            int32_t const tokenIdxInTileKv = (customMaskTokenIdxKv % tileSizeKvPerCta) % tileSizeKv;

            int64_t const tileOffset = tileBase + tileIdxKv;
            int64_t const instOffset = tileOffset * numInstsQ * numInstsKv + (instQBase + instIdxKv);
            int64_t const maskOffset = instOffset * tileSizeQ * tileSizeKv + (maskRowBase + tokenIdxInTileKv);

            int64_t const offsetAsUInt32 = maskOffset >> 5;
            int32_t const bitPosInUInt32 = maskOffset & 0x1F;

            atomicOr(&localCustomMaskPtr[offsetAsUInt32], (1U << bitPosInUInt32));
        }
    }
}

__global__ void computeCustomMaskOffsetsKernel(
    TllmGenFmhaKernelMetaInfo kernelMeta, TllmGenFmhaRunnerParams runnerParams, unsigned long long* globalCounter)
{
    int32_t batchSize = runnerParams.mBatchSize;
    int32_t numHeadsQPerKv = runnerParams.mNumHeadsQPerKv;
    int32_t tileSizeQ = kernelMeta.mTileSizeQ;
    int32_t tileSizeKv = kernelMeta.mTileSizeKv;
    int32_t numInstsQ = kernelMeta.mStepQ / kernelMeta.mTileSizeQ;
    int32_t numInstsKv = kernelMeta.mStepKv / kernelMeta.mTileSizeKv;
    int32_t tileSizeQPerCta = tileSizeQ * numInstsQ;
    int32_t tileSizeKvPerCta = tileSizeKv * numInstsKv;
    int32_t const* seqLensKvPtr = runnerParams.seqLensKvPtr;
    int32_t const* firstSparseMaskOffsetsKvPtr = runnerParams.firstSparseMaskOffsetsKvPtr;

    typedef cub::BlockScan<int64_t, 128> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t maskSize = 0;

    if (idx < batchSize)
    {

        int32_t seqLenQ = runnerParams.spec_decoding_generation_lengths[idx];
        int32_t seqLenKv = seqLensKvPtr[idx];
        int32_t firstSparseMaskOffsetKv = firstSparseMaskOffsetsKvPtr[idx];

        int32_t numTilesQ = (seqLenQ * numHeadsQPerKv + tileSizeQPerCta - 1) / tileSizeQPerCta;
        int32_t firstSparseTile = firstSparseMaskOffsetKv / tileSizeKvPerCta;
        int32_t numCustomMaskTilesKv = (seqLenKv + tileSizeKvPerCta - 1) / tileSizeKvPerCta - firstSparseTile;

        maskSize = static_cast<int64_t>(
            numTilesQ * numCustomMaskTilesKv * numInstsQ * numInstsKv * (tileSizeQ * tileSizeKv) / 32);
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

void launchPrepareCustomMaskBuffersKernelForKeepsMmaAb(
    TllmGenFmhaRunnerParams const& runnerParams, TllmGenFmhaKernelMetaInfo const& kernelMeta, cudaStream_t stream)
{
    int32_t batchSize = runnerParams.mBatchSize;
    int32_t maxSeqLenQ = runnerParams.mMaxSeqLenQ;
    int32_t numHeadsQPerKv = runnerParams.mNumHeadsQPerKv;

    int32_t maxThreadsPerQ = maxSeqLenQ * numHeadsQPerKv;

    int32_t blockSize = 128;
    int32_t numBlocksY = ceilDiv(maxThreadsPerQ, blockSize);

    dim3 gridDim(batchSize, numBlocksY);
    dim3 blockDim(blockSize);

    prepareCustomMaskBuffersKernelForKeepsMmaAb<<<gridDim, blockDim, 0, stream>>>(runnerParams, kernelMeta);
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
} // namespace tensorrt_llm
