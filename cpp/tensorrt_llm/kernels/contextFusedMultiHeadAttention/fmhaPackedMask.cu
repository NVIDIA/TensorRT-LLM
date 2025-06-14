/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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
#include "fmhaPackedMask.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaBf16Wrapper.h"
#include "tensorrt_llm/common/cudaFp8Utils.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/mathUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include <cub/cub.cuh>

using namespace tensorrt_llm::common;

namespace tensorrt_llm
{
namespace kernels
{

////////////////////////////////////////////////////////////////////////////////////////////////////

// A stateful callback functor that maintains the running sum between consecutive scans.
struct BlockPrefixCallbackOp
{
    // Running prefix
    int mRunningTotal;

    // Constructor
    __device__ BlockPrefixCallbackOp(int runningTotal)
        : mRunningTotal(runningTotal)
    {
    }

    // Thread-0 is responsible for returning a value for seeding the block-wide scan.
    __device__ int operator()(int blockAggregate)
    {
        int oldPrefix = mRunningTotal;
        mRunningTotal += blockAggregate;
        return oldPrefix;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int THREADS_PER_BLOCK>
__global__ __launch_bounds__(THREADS_PER_BLOCK) void buildCuMaskRows(
    int batchSize, int const* qSeqLens, int* cuQSeqLens, int* cuMaskRows)
{

    // The implementation of the parallel scan in the thread block (see CUB for details).
    using BlockScan = cub::BlockScan<int, THREADS_PER_BLOCK>;

    // Allocate storage in shared memory to do the scan.
    __shared__ typename BlockScan::TempStorage tempStorage;
    __shared__ typename BlockScan::TempStorage tempStorageForMask;

    // This prefixOp operator keeps a running sum for when we need multiple iterations of the loop.
    BlockPrefixCallbackOp prefixOp(0);
    BlockPrefixCallbackOp prefixOpForMask(0);

    // Iterate over the sequences in the batch.
    //
    // The loop index does not depend on the thread index to make sure all the threads enter the
    // loop as we have __syncthreads in it (and we need all threads to participate to avoid
    // deadlocks).
    // Only the last block computes the full sequence offsets.
    bool const storeOffsets = blockIdx.x == (batchSize - 1);
    int const batchSizeBound = blockIdx.x + 1;
    for (int batchOffset = 0; batchOffset <= batchSizeBound; batchOffset += THREADS_PER_BLOCK)
    {
        // The index of the batch.
        int batchIdx = batchOffset + threadIdx.x;

        // Threads that correspond to valid sequences read the length.
        int maskRows = 0;
        int qSeqLen = 0;
        if (batchIdx < batchSizeBound)
        {
            qSeqLen = qSeqLens[batchIdx];
            // Need to pad to multiple of 128.
            maskRows = divUp(qSeqLens[batchIdx], int(FLASH_ATTEN_PACKED_MASK_M_ALIGNMENT))
                * FLASH_ATTEN_PACKED_MASK_M_ALIGNMENT;
        }

        // Do the prefix-scan (it calls syncthreads internally).
        int qSeqLenOffset;
        int maskRowOffset;
        BlockScan(tempStorage).ExclusiveSum(qSeqLen, qSeqLenOffset, prefixOp);
        BlockScan(tempStorageForMask).ExclusiveSum(maskRows, maskRowOffset, prefixOpForMask);

        // Store the result.
        if (batchIdx <= batchSizeBound && storeOffsets)
        {
            if (cuQSeqLens)
            {
                cuQSeqLens[batchIdx] = qSeqLenOffset;
            }
            cuMaskRows[batchIdx] = maskRowOffset;
        }

        // Make sure the shared memory can be reused for the next iteration of the loop.
        __syncthreads();
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename MaskInputDataType, ContextAttentionMaskType MaskType>
__global__ void packFlashAttentionMask(PackedMaskParams<MaskInputDataType> params)
{

    // Each core MMA_N = 8, and each thread holds 32bits as one packed mask (2 rows, 16 cols).
    // All packed mask units of one warp group are coalesced, and then repeated along the
    // col dimension, which means there will be 128 (num of threads) * 32 bits (one packed mask)
    // stride for each 16 cols. This is designed to have coalesced memory access for each
    // warp.
    // Layout:
    //  0 ~ 15 cols: t0, t1, t2, t3, ...., t127, t0,...,t127,....
    // 16 ~ 31 cols: t0, t1, t2, t3, ...., t127, t0,...,t127,....
    // ....

    // The batch index.
    int batchIdx = blockIdx.y;
    // The MMAS_N index.
    int mmasNIdx = blockIdx.x;
    // The number of MMAS_N.
    int mmasN = gridDim.x;

    // The upper bound of q and kv sequence length.
    int qSeqLenBound = params.actualQSeqLens[batchIdx];
    int kvSeqLenBound = params.actualKvSeqLens[batchIdx];

    // The mask input offset for batchIdx.
    size_t maskInputBatchOffset
        = (params.cuQSeqLens ? params.cuQSeqLens[batchIdx] : batchIdx * params.maxQSeqLen) * params.maxKvSeqLen;

    // The actual mask rows in the sequence.
    int actualMaskRows = params.cuMaskRows[batchIdx + 1] - params.cuMaskRows[batchIdx];
    // The actual mmasM for this sequence.
    // Note all maskSeqLens have been rounded up to multiple of 128.
    int mmasM = actualMaskRows / (FLASH_ATTEN_WARPS_M * 16);
    // The cumulative mmasM.
    int cuMmasM = params.cuMaskRows[batchIdx] / (FLASH_ATTEN_WARPS_M * 16);
    // Iterate over the mmasM, threads.
    for (size_t mi = threadIdx.x; mi < mmasM; mi += blockDim.x)
    {
        for (size_t tidx = 0; tidx < NUM_THREADS_PER_WARP_GROUP; ++tidx)
        {

            // The warp position.
            size_t warp = tidx / 32;
            size_t lane = tidx % 32;

            // The warp index.
            size_t warpM = warp % FLASH_ATTEN_WARPS_M;
            size_t warpN = warp / FLASH_ATTEN_WARPS_M;

            // The row/col of the 1st element for that MMA.
            size_t row = warpM * 16 + lane / 4;
            size_t col = warpN * 16 + lane % 4 * 2;

            // Take the mmas_m, mmas_n into account.
            row += mi * FLASH_ATTEN_WARPS_M * 16;
            col += mmasNIdx * NUM_CORE_MMAS_N * 8;

            // The offset to the 1st element computed by that thread in the mask.
            size_t offset = maskInputBatchOffset + row * params.maxKvSeqLen + col;

            // The mask for each row of MMAs.
            uint32_t mask = 0u;

// Iterate over the core mmas in the N dimension.
#pragma unroll
            for (size_t ni = 0; ni < NUM_CORE_MMAS_N;
                 ++ni, offset += 8 * FLASH_ATTEN_WARPS_N, col += 8 * FLASH_ATTEN_WARPS_N)
            {

                bool validMasks[4] = {row < qSeqLenBound && col < kvSeqLenBound,
                    row < qSeqLenBound && (col + 1) < kvSeqLenBound, (row + 8) < qSeqLenBound && col < kvSeqLenBound,
                    (row + 8) < qSeqLenBound && (col + 1) < kvSeqLenBound};

                if constexpr (MaskType == ContextAttentionMaskType::CUSTOM_MASK)
                {
                    validMasks[0] = validMasks[0]
                        && (params.maskInput[offset + 0 * params.maxKvSeqLen + 0] == params.validPosVal);
                    validMasks[1] = validMasks[1]
                        && (params.maskInput[offset + 0 * params.maxKvSeqLen + 1] == params.validPosVal);
                    validMasks[2] = validMasks[2]
                        && (params.maskInput[offset + 8 * params.maxKvSeqLen + 0] == params.validPosVal);
                    validMasks[3] = validMasks[3]
                        && (params.maskInput[offset + 8 * params.maxKvSeqLen + 1] == params.validPosVal);
                }
                else if constexpr (MaskType == ContextAttentionMaskType::CAUSAL)
                {
                    validMasks[0] &= (col <= row);
                    validMasks[1] &= ((col + 1) <= row);
                    validMasks[2] &= (col <= (row + 8));
                    validMasks[3] &= ((col + 1) <= (row + 8));
                }
                else if constexpr (MaskType == ContextAttentionMaskType::SLIDING_OR_CHUNKED_CAUSAL)
                {
                    validMasks[0] &= (col <= row) && (col > (row - params.slidingWindowSize));
                    validMasks[1] &= ((col + 1) <= row && (col + 1) > (row - params.slidingWindowSize));
                    validMasks[2] &= (col <= (row + 8) && col > ((row + 8) - params.slidingWindowSize));
                    validMasks[3] &= ((col + 1) <= (row + 8) && (col + 1) > ((row + 8) - params.slidingWindowSize));
                }

                mask |= (validMasks[0] ? 1u : 0u) << (4 * ni + 0);
                mask |= (validMasks[1] ? 1u : 0u) << (4 * ni + 1);
                mask |= (validMasks[2] ? 1u : 0u) << (4 * ni + 2);
                mask |= (validMasks[3] ? 1u : 0u) << (4 * ni + 3);
            }

            // The offset of uint32_t packed mask.
            size_t mOffset = (cuMmasM + mi) * mmasN * NUM_THREADS_PER_WARP_GROUP;
            size_t nOffset = mmasNIdx * NUM_THREADS_PER_WARP_GROUP;
            params.packedMask[mOffset + nOffset + tidx] = mask;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename MaskInputDataType>
void invokeBuildPackedMask(PackedMaskParams<MaskInputDataType> const& params, cudaStream_t stream)
{
    // Calculate the cuMaskRows.
    buildCuMaskRows<256><<<params.batchSize, 256, 0, stream>>>(
        params.batchSize, params.actualQSeqLens, params.cuQSeqLens, params.cuMaskRows);
    sync_check_cuda_error(stream);
    // The number of mmas in the N dimension (MMA_N = 64).
    size_t mmasN
        = (divUp(params.maxKvSeqLen, size_t(FLASH_ATTEN_PACKED_MASK_N_ALIGNMENT)) * FLASH_ATTEN_PACKED_MASK_N_ALIGNMENT)
        / FLASH_ATTEN_PACKED_MASK_MMA_N;
    // The grid.
    dim3 grid(mmasN, params.batchSize);
    // Launch the kernel.
    if (params.attentionMaskType == ContextAttentionMaskType::PADDING)
    {
        packFlashAttentionMask<MaskInputDataType, ContextAttentionMaskType::PADDING><<<grid, 256, 0, stream>>>(params);
    }
    else if (params.attentionMaskType == ContextAttentionMaskType::CAUSAL)
    {
        packFlashAttentionMask<MaskInputDataType, ContextAttentionMaskType::CAUSAL><<<grid, 256, 0, stream>>>(params);
    }
    else if (params.attentionMaskType == ContextAttentionMaskType::CUSTOM_MASK)
    {
        packFlashAttentionMask<MaskInputDataType, ContextAttentionMaskType::CUSTOM_MASK>
            <<<grid, 256, 0, stream>>>(params);
    }
    else if (params.attentionMaskType == ContextAttentionMaskType::SLIDING_OR_CHUNKED_CAUSAL)
    {
        packFlashAttentionMask<MaskInputDataType, ContextAttentionMaskType::SLIDING_OR_CHUNKED_CAUSAL>
            <<<grid, 256, 0, stream>>>(params);
    }
    else
    {
        TLLM_CHECK_WITH_INFO(false, "The attention mask type is not supported.");
    }
    sync_check_cuda_error(stream);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Instantiations.
template void invokeBuildPackedMask(PackedMaskParams<float> const&, cudaStream_t);
template void invokeBuildPackedMask(PackedMaskParams<half> const&, cudaStream_t);
template void invokeBuildPackedMask(PackedMaskParams<bool> const&, cudaStream_t);
template void invokeBuildPackedMask(PackedMaskParams<int> const&, cudaStream_t);
#ifdef ENABLE_BF16
template void invokeBuildPackedMask(PackedMaskParams<__nv_bfloat16> const&, cudaStream_t);
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernels
} // namespace tensorrt_llm
