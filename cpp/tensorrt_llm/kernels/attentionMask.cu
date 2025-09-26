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
#include "attentionMask.h"
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
__global__ __launch_bounds__(THREADS_PER_BLOCK) void buildCuQSeqLens(
    int batchSize, int const* qSeqLens, int* cuQSeqLens)
{

    // The implementation of the parallel scan in the thread block (see CUB for details).
    using BlockScan = cub::BlockScan<int, THREADS_PER_BLOCK>;

    // Allocate storage in shared memory to do the scan.
    __shared__ typename BlockScan::TempStorage tempStorage;

    // This prefixOp operator keeps a running sum for when we need multiple iterations of the loop.
    BlockPrefixCallbackOp prefixOp(0);

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
        int qSeqLen = 0;
        if (batchIdx < batchSizeBound)
        {
            qSeqLen = qSeqLens[batchIdx];
        }

        // Do the prefix-scan (it calls syncthreads internally).
        int qSeqLenOffset;
        BlockScan(tempStorage).ExclusiveSum(qSeqLen, qSeqLenOffset, prefixOp);

        // Store the result.
        if (batchIdx <= batchSizeBound && storeOffsets)
        {
            if (cuQSeqLens)
            {
                cuQSeqLens[batchIdx] = qSeqLenOffset;
            }
        }
        // Make sure the shared memory can be reused for the next iteration of the loop.
        __syncthreads();
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename MaskDataType, AttentionMaskType MaskType>
__global__ void prepareAttentionMask(AttentionMaskParams<MaskDataType> params)
{
    // The batch idx.
    int batchIdx = blockIdx.y;
    // Are paddings removed in the row dimension ?
    bool const paddingsRemoved = params.cuQSeqLens;
    // The row (Q) sequence offset.
    int qSeqOffset = (paddingsRemoved ? params.cuQSeqLens[batchIdx] : batchIdx * params.maxQSeqLen);
    // The actual sequence length.
    int qSeqLen = params.actualQSeqLens[batchIdx];
    int kvSeqLen = params.actualKvSeqLens[batchIdx];
    // The mask sequence length.
    int maskQSeqLen = paddingsRemoved ? params.actualQSeqLens[batchIdx] : params.maxQSeqLen;
    // Assume that the paddings are kept in the col dimension.
    int maskKvSeqLen = params.maxKvSeqLen;

    // The mask offset.
    size_t maskOffset = static_cast<size_t>(qSeqOffset) * params.maxKvSeqLen;

    // The attention mask row.
    for (int row = blockIdx.x; row < maskQSeqLen; row += gridDim.x)
    {
        // The attention mask col;
        for (int col = threadIdx.x; col < maskKvSeqLen; col += blockDim.x)
        {
            size_t localMaskOffset = static_cast<size_t>(row) * params.maxKvSeqLen + col;
            bool valid = false;
            if constexpr (MaskType == AttentionMaskType::PADDING)
            {
                valid = row < qSeqLen && col < kvSeqLen;
            }
            if constexpr (MaskType == AttentionMaskType::CAUSAL)
            {
                valid = row < qSeqLen && col < kvSeqLen && col <= row;
            }
            else if constexpr (MaskType == AttentionMaskType::SLIDING_WINDOW_CAUSAL)
            {
                valid = (col > (row - params.slidingWindowSize));
            }
            else if constexpr (MaskType == AttentionMaskType::BIDIRECTIONAL)
            {
                valid = (row < (qSeqLen - 1) && col < (kvSeqLen - 1)) || (row == qSeqLen - 1 && col < kvSeqLen);
            }
            else if constexpr (MaskType == AttentionMaskType::BIDIRECTIONALGLM)
            {
                valid = (col < (kvSeqLen - 1)) || (row == (qSeqLen - 1) && col == (kvSeqLen - 1));
            }
            else if constexpr (MaskType == AttentionMaskType::BLOCKSPARSE)
            {
                valid
                    = params.blockSparseParams.computeMask(row, col, qSeqLen, kvSeqLen, 1 /*num_heads*/, 0 /*head_id*/);
            }

            // Store it to mask.
            params.mask[maskOffset + localMaskOffset] = static_cast<MaskDataType>(valid);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename MaskDataType>
void invokeBuildAttentionMask(AttentionMaskParams<MaskDataType> const& params, cudaStream_t stream)
{
    // Calculate the cuQSeqLens.
    if (params.cuQSeqLens)
    {
        buildCuQSeqLens<256>
            <<<params.batchSize, 256, 0, stream>>>(params.batchSize, params.actualQSeqLens, params.cuQSeqLens);
        sync_check_cuda_error(stream);
    }

    // Set the attention mask.
    dim3 grid(std::min(1024, params.maxQSeqLen), params.batchSize);
    // Launch the kernel.
    if (params.attentionMaskType == AttentionMaskType::PADDING)
    {
        prepareAttentionMask<MaskDataType, AttentionMaskType::PADDING><<<grid, 256, 0, stream>>>(params);
    }
    else if (params.attentionMaskType == AttentionMaskType::CAUSAL)
    {
        prepareAttentionMask<MaskDataType, AttentionMaskType::CAUSAL><<<grid, 256, 0, stream>>>(params);
    }
    else if (params.attentionMaskType == AttentionMaskType::SLIDING_WINDOW_CAUSAL)
    {
        prepareAttentionMask<MaskDataType, AttentionMaskType::SLIDING_WINDOW_CAUSAL><<<grid, 256, 0, stream>>>(params);
    }
    else if (params.attentionMaskType == AttentionMaskType::BIDIRECTIONAL)
    {
        prepareAttentionMask<MaskDataType, AttentionMaskType::BIDIRECTIONAL><<<grid, 256, 0, stream>>>(params);
    }
    else if (params.attentionMaskType == AttentionMaskType::BIDIRECTIONALGLM)
    {
        prepareAttentionMask<MaskDataType, AttentionMaskType::BIDIRECTIONALGLM><<<grid, 256, 0, stream>>>(params);
    }
    else if (params.attentionMaskType == AttentionMaskType::BLOCKSPARSE)
    {
        prepareAttentionMask<MaskDataType, AttentionMaskType::BLOCKSPARSE><<<grid, 256, 0, stream>>>(params);
    }
    else
    {
        TLLM_CHECK_WITH_INFO(false, "The attention mask type is not supported.");
    }
    sync_check_cuda_error(stream);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Instantiations.
template void invokeBuildAttentionMask(AttentionMaskParams<float> const&, cudaStream_t);
template void invokeBuildAttentionMask(AttentionMaskParams<half> const&, cudaStream_t);
template void invokeBuildAttentionMask(AttentionMaskParams<bool> const&, cudaStream_t);
template void invokeBuildAttentionMask(AttentionMaskParams<int> const&, cudaStream_t);
#ifdef ENABLE_BF16
template void invokeBuildAttentionMask(AttentionMaskParams<__nv_bfloat16> const&, cudaStream_t);
#endif
#ifdef ENABLE_FP8
template void invokeBuildAttentionMask(AttentionMaskParams<__nv_fp8_e4m3> const&, cudaStream_t);
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernels
} // namespace tensorrt_llm
