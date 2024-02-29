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
#include "tensorrt_llm/common/cudaBf16Wrapper.h"
#include "tensorrt_llm/common/cudaFp8Utils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/kernels/gptKernels.h"
#include <cub/cub.cuh>

using namespace tensorrt_llm::common;

namespace tensorrt_llm
{
namespace kernels
{

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

// Given an array of sequence lengths, with batchSize elements, that kernel computes the exclusive
// prefix-sums of the sequence lengths. There are (batchSize+1) elements in seqOffsets.
//
// seqOffsets[ 0]        = 0
// seqOffsets[ii]        = seqLengths[0] + .. + seqLengths[ii-1],
// seqOffsets[batchSize] = seqLengths[0] + .. + seqLengths[batchSize-1]
//
// This kernel uses a single thread block of THREADS_PER_BLOCK threads.

template <int THREADS_PER_BLOCK>
__global__ __launch_bounds__(THREADS_PER_BLOCK) void computeSeqOffsets(
    int* seqOffsets, const int* seqLengths, int batchSize)
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
    for (int batchOffset = 0; batchOffset <= batchSize; batchOffset += THREADS_PER_BLOCK)
    {
        // The index of the batch.
        int batchIdx = batchOffset + threadIdx.x;

        // Threads that correspond to valid sequences read the length.
        int seqLength = 0;
        if (batchIdx < batchSize)
        {
            seqLength = seqLengths[batchIdx];
        }

        // Do the prefix-scan (it calls syncthreads internally).
        int seqOffset;
        BlockScan(tempStorage).ExclusiveSum(seqLength, seqOffset, prefixOp);

        // Store the result.
        if (batchIdx <= batchSize)
        {
            seqOffsets[batchIdx] = seqOffset;
        }

        // Make sure the shared memory can be reused for the next iteration of the loop.
        __syncthreads();
    }
}

// This kernel computes the padding offsets: Given the index (idx) of a token in a ragged tensor,
// we need the index of the token in the corresponding tensor with padding. We compute an array
// of numTokens elements, called the paddingOffsets, such that the position in the padded tensor
// of the token "idx" in the ragged tensor is given by idx + paddingOffset[idx].
//
// That kernel uses a grid of batchSize blocks.

__global__ void computePaddingOffsets(int* paddingOffsets, const int* seqOffsets, int maxSeqLength)
{
    // The index of the sequence in the batch.
    int batchIdx = blockIdx.x;

    // The beginning of the sequence.
    int seqBegin = seqOffsets[batchIdx];
    // The offset to the 1st element of the next sequence.
    int seqEnd = seqOffsets[batchIdx + 1];
    // The length of the sequence.
    int seqLength = seqEnd - seqBegin;

    // The number of padded tokens in the previous sequences.
    int paddingOffset = batchIdx * maxSeqLength - seqBegin;

    // Iterate over the tokens to update the number of padded elements.
    for (int tokenIdx = threadIdx.x; tokenIdx < seqLength; tokenIdx += blockDim.x)
    {
        paddingOffsets[seqBegin + tokenIdx] = paddingOffset;
    }
}

// This kernel computes the attention mask. We must compute this on-the-fly in the future.

template <typename AttentionMaskDataType>
__global__ void computeAttentionMask(AttentionMaskDataType* attentionMask, const int* seqOffsets, int maxSeqLength,
    int attentionWindowSize, AttentionMaskType attentionMaskType)
{
    // The index of the sequence in the batch.
    int batchIdx = blockIdx.y;

    // The number of items in the mask for each sequence.
    int maskSize = maxSeqLength * maxSeqLength;
    // The offset to the 1st element of the mask for that particular sequence.
    int batchOffset = batchIdx * maskSize;

    // The beginning of the sequence.
    int seqBegin = seqOffsets[batchIdx];
    // The offset to the 1st element of the next sequence.
    int seqEnd = seqOffsets[batchIdx + 1];
    // The length of the sequence.
    int seqLength = seqEnd - seqBegin;

    // Iterate over the tokens to update the number of padded elements.
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < maskSize; idx += gridDim.x * blockDim.x)
    {
        // The position in the matrix.
        int rowIdx = idx / maxSeqLength;
        int colIdx = idx % maxSeqLength;

        // Is it a valid token?
        bool isValid = true;
        switch (attentionMaskType)
        {
        case AttentionMaskType::PADDING:
            isValid = rowIdx < seqLength && colIdx < seqLength;
            // seq_length==4, max_seq_len==5
            // 1 1 1 1 0
            // 1 1 1 1 0
            // 1 1 1 1 0
            // 1 1 1 1 0
            // 0 0 0 0 0
            break;
        case AttentionMaskType::CAUSAL:
            isValid = rowIdx < seqLength && colIdx < seqLength && colIdx <= rowIdx;
            // Sliding_window_causal when there are not enough kv cache.
            isValid = isValid && colIdx >= max(0, rowIdx - attentionWindowSize);
            // seq_length==4, max_seq_len==5
            // 1 0 0 0 0
            // 1 1 0 0 0
            // 1 1 1 0 0
            // 1 1 1 1 0
            // 0 0 0 0 0

            // seq_length==6, max_seq_len==6, max_attention_window_size = 2
            // 1 0 0 0 0 0
            // 1 1 0 0 0 0
            // 1 1 1 0 0 0
            // 0 1 1 1 0 0
            // 0 0 1 1 1 0
            // 0 0 0 1 1 1
            break;
        case AttentionMaskType::BIDIRECTIONAL:
            // clang-format off
            isValid = (rowIdx <  seqLength - 1 && colIdx < seqLength - 1) ||
                      (rowIdx == seqLength - 1 && colIdx < seqLength);
            // clang-format on
            // seq_length==4, max_seq_len==5
            // 1 1 1 0 0
            // 1 1 1 0 0
            // 1 1 1 0 0
            // 1 1 1 1 0
            // 0 0 0 0 0
        case AttentionMaskType::BIDIRECTIONALGLM:
            // clang-format off
            isValid = (colIdx < seqLength - 1) ||
                      (rowIdx == seqLength - 1 && colIdx == seqLength - 1);
            // clang-format on
            // seq_length==4, max_seq_len==5
            // 1 1 1 1 0
            // 1 1 1 1 0
            // 1 1 1 1 0
            // 1 1 1 1 0
            // 1 1 1 1 1
            break;
        }

        // Store the mask.
        attentionMask[batchOffset + idx] = isValid ? AttentionMaskDataType(1.f) : AttentionMaskDataType(0.f);
    }
}

template <typename T>
void invokeBuildDecoderInfo(const BuildDecoderInfoParams<T>& params, cudaStream_t stream)
{
    // Compute the sequence offsets.
    const int THREADS_PER_BLOCK = 256;
    computeSeqOffsets<THREADS_PER_BLOCK>
        <<<1, THREADS_PER_BLOCK, 0, stream>>>(params.seqQOffsets, params.seqQLengths, params.batchSize);
    if (params.seqKVLengths)
    {
        computeSeqOffsets<THREADS_PER_BLOCK>
            <<<1, THREADS_PER_BLOCK, 0, stream>>>(params.seqKVOffsets, params.seqKVLengths, params.batchSize);
    }

    // Compute the padding offsets.
    computePaddingOffsets<<<params.batchSize, THREADS_PER_BLOCK, 0, stream>>>(
        params.paddingOffsets, params.seqQOffsets, params.maxSeqLength);

    // Compute the attention mask, if needed.
    if (params.attentionMask != nullptr)
    {
        const int MIN_BLOCKS = 512;
        int blocksPerSeq = 16;
        while (blocksPerSeq * params.batchSize < MIN_BLOCKS)
        {
            blocksPerSeq *= 2;
        }
        dim3 grid(blocksPerSeq, params.batchSize);
        computeAttentionMask<<<grid, THREADS_PER_BLOCK, 0, stream>>>(params.attentionMask, params.seqQOffsets,
            params.maxSeqLength, params.attentionWindowSize, params.attentionMaskType);
    }
}

template void invokeBuildDecoderInfo(const BuildDecoderInfoParams<float>&, cudaStream_t);
template void invokeBuildDecoderInfo(const BuildDecoderInfoParams<half>&, cudaStream_t);
#ifdef ENABLE_BF16
template void invokeBuildDecoderInfo(const BuildDecoderInfoParams<__nv_bfloat16>&, cudaStream_t);
#endif
#ifdef ENABLE_FP8
template void invokeBuildDecoderInfo(const BuildDecoderInfoParams<__nv_fp8_e4m3>&, cudaStream_t);
#endif

__global__ void updatePaddingCountKernel(int* paddingPerSeq, const int* seqLengths, int maxSeqLength, int batchSize)
{

    for (int ii = threadIdx.x; ii < batchSize; ii += blockDim.x)
    {
        paddingPerSeq[ii] = maxSeqLength - seqLengths[ii];
    }
}

} // namespace kernels
} // namespace tensorrt_llm
