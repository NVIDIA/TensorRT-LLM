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
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaBf16Wrapper.h"
#include "tensorrt_llm/common/cudaFp8Utils.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/mathUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttentionUtils.h"
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

// This kernel also computes the padding offsets: Given the index (idx) of a token in a ragged tensor,
// we need the index of the token in the corresponding tensor with padding. We compute an array
// of numTokens elements, called the paddingOffsets, such that the position in the padded tensor
// of the token "idx" in the ragged tensor is given by idx + paddingOffset[idx].
//
// That kernel uses a grid of batchSize blocks.

template <typename T, int THREADS_PER_BLOCK>
__global__ __launch_bounds__(THREADS_PER_BLOCK) void computeSeqAndPaddingOffsets(BuildDecoderInfoParams<T> params)
{
    // Dynamic shared memory for storing seqOffsets.
    extern __shared__ int smem[];
    int* smemSeqQOffsets = (int*) (smem);

    // Fixed Q sequence lengths.
    bool const fixed_q_seqlen = params.seqQLengths == nullptr;

    // Whether to calculate cumulative KV sequence lengths.
    bool const calculate_kv_offsets = params.seqKVOffsets != nullptr;

    // Whether to calculate cumulative packed mask rows.
    bool const calculate_packed_mask_row_offsets = params.packedMaskRowOffsets != nullptr;

    // Compute the padding offsets for Encoder Inputs.
    bool const need_encoder_padding_offsets = (params.encoderPaddingOffsets != nullptr) && calculate_kv_offsets;
    [[maybe_unused]] int* smemEncoderSeqQOffsets;

    // The implementation of the parallel scan in the thread block (see CUB for details).
    using BlockScan = cub::BlockScan<int, THREADS_PER_BLOCK>;

    // Allocate storage in shared memory to do the scan.
    __shared__ typename BlockScan::TempStorage tempQStorage;
    [[maybe_unused]] __shared__ typename BlockScan::TempStorage tempMaskStorage;
    [[maybe_unused]] __shared__ typename BlockScan::TempStorage tempKVStorage;

    // This prefixOp operator keeps a running sum for when we need multiple iterations of the loop.
    BlockPrefixCallbackOp prefixQOp(0);
    BlockPrefixCallbackOp prefixMaskOp(0);
    BlockPrefixCallbackOp prefixKVOp(0);

    if (need_encoder_padding_offsets)
    {
        smemEncoderSeqQOffsets = (int*) (&smemSeqQOffsets[params.batchSize + 1]);
    }

    // Iterate over the sequences in the batch.
    //
    // The loop index does not depend on the thread index to make sure all the threads enter the
    // loop as we have __syncthreads in it (and we need all threads to participate to avoid
    // deadlocks).
    // Only the last block computes the full sequence offsets.
    bool const storeSeqOffsets = blockIdx.x == (params.batchSize - 1);
    int const batchSizeBound = blockIdx.x + 1;
    for (int batchOffset = 0; batchOffset <= batchSizeBound; batchOffset += THREADS_PER_BLOCK)
    {
        // The index of the batch.
        int batchIdx = batchOffset + threadIdx.x;

        // Threads that correspond to valid sequences read the length.
        int seqQLength = 0;
        [[maybe_unused]] int packedMaskRows = 0;
        [[maybe_unused]] int seqKVLength = 0;
        if (batchIdx < batchSizeBound)
        {
            seqQLength = fixed_q_seqlen ? params.maxQSeqLength : params.seqQLengths[batchIdx];
            // Need to pad mask rows to multiple of 128 for each sequence in the batch.
            packedMaskRows = calculate_packed_mask_row_offsets
                ? divUp(seqQLength, int(FLASH_ATTEN_PACKED_MASK_M_ALIGNMENT)) * FLASH_ATTEN_PACKED_MASK_M_ALIGNMENT
                : 0;
            seqKVLength = calculate_kv_offsets ? params.seqKVLengths[batchIdx] : 0;
        }

        // Do the prefix-scan (it calls syncthreads internally).
        int seqQOffset;
        [[maybe_unused]] int packedMaskRowOffset;
        [[maybe_unused]] int seqKVOffset;
        BlockScan(tempQStorage).ExclusiveSum(seqQLength, seqQOffset, prefixQOp);
        if (calculate_packed_mask_row_offsets)
        {
            BlockScan(tempMaskStorage).ExclusiveSum(packedMaskRows, packedMaskRowOffset, prefixMaskOp);
        }
        if (calculate_kv_offsets)
        {
            BlockScan(tempKVStorage).ExclusiveSum(seqKVLength, seqKVOffset, prefixKVOp);
        }

        // Store the result to smem.
        if (batchIdx <= batchSizeBound)
        {
            smemSeqQOffsets[batchIdx] = seqQOffset;
            if (need_encoder_padding_offsets)
            {
                smemEncoderSeqQOffsets[batchIdx] = seqKVOffset;
            }
        }

        // Store the result.
        if (batchIdx <= batchSizeBound && storeSeqOffsets)
        {
            params.seqQOffsets[batchIdx] = seqQOffset;
            if (calculate_packed_mask_row_offsets)
            {
                params.packedMaskRowOffsets[batchIdx] = packedMaskRowOffset;
            }
            if (calculate_kv_offsets)
            {
                params.seqKVOffsets[batchIdx] = seqKVOffset;
            }
        }

        // Make sure the shared memory can be reused for the next iteration of the loop.
        __syncthreads();
    }

    int batchIdx = blockIdx.x;

    // Compute the padding offsets.
    auto compute_padding_offset = [&](int* smem_offset, int maxSeqLength, int* paddingOffsets)
    {
        // Block x dimension is the batch dimension, while threads iterate all tokens in the sequence.
        int seqBegin = smem_offset[batchIdx];
        // The offset to the 1st element of the next sequence.
        int seqEnd = smem_offset[batchIdx + 1];
        // The length of the sequence.
        int seqLength = seqEnd - seqBegin;
        // The number of padded tokens in the previous sequences.
        int paddingOffset = batchIdx * maxSeqLength - seqBegin;

        // Iterate over the tokens to update the number of padded elements.
        for (int tokenIdx = threadIdx.x; tokenIdx < seqLength; tokenIdx += blockDim.x)
        {
            paddingOffsets[seqBegin + tokenIdx] = paddingOffset;
        }
    };

    if (params.paddingOffsets != nullptr)
    {
        compute_padding_offset(smemSeqQOffsets, params.maxQSeqLength, params.paddingOffsets);
    }

    if (need_encoder_padding_offsets)
    {
        compute_padding_offset(smemEncoderSeqQOffsets, params.maxEncoderQSeqLength, params.encoderPaddingOffsets);
    }

    // Each block generates the rotary embedding inv_freq tensor for the corresponding sequence.
    int zid = 2 * threadIdx.x;
    int halfRotaryEmbeddingDim = params.rotaryEmbeddingDim / 2;
    if (params.rotaryEmbeddingDim > 0 && zid < params.rotaryEmbeddingDim)
    {
        mmha::update_rotary_base_n_scale(params.rotaryEmbeddingBase, params.rotaryEmbeddingScale,
            params.rotaryScalingType, params.rotaryEmbeddingDim, params.rotaryEmbeddingMaxPositions,
            params.seqKVLengths[batchIdx]);
        // Recompute the rotary scales when it is dynamic scaling.
        if (params.rotaryScalingType == RotaryScalingType::kDYNAMIC || params.rotaryEmbeddingInvFreqCache == nullptr)
        {
            float const invFreq = params.rotaryEmbeddingScale
                / powf(params.rotaryEmbeddingBase, zid / (float) params.rotaryEmbeddingDim);
            params.rotaryEmbeddingInvFreq[batchIdx * halfRotaryEmbeddingDim + threadIdx.x] = invFreq;
        }
        else
        {
            // Otherwise, expand the inv freq cache to batch size.
            float const invFreqCache = params.rotaryEmbeddingInvFreqCache[threadIdx.x];
            params.rotaryEmbeddingInvFreq[batchIdx * halfRotaryEmbeddingDim + threadIdx.x] = invFreqCache;
        }
    }

    // Perpare values for fmha.
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        // Reset fmha tile counter to 0 before launching fmha kernels.
        if (params.fmhaTileCounter)
        {
            params.fmhaTileCounter[0] = 0u;
        }
        // Take the quantization scales into consideration.
        if (params.fmhaBmm1Scale)
        {
            // The scale after fmha bmm1.
            params.fmhaBmm1Scale[0] = params.dequantScaleQkv[0] * params.dequantScaleQkv[0] * params.fmhaHostBmm1Scale;
            // The scale prepared for log2 optimization.
            constexpr float kLog2e = 1.4426950408889634074f;
            params.fmhaBmm1Scale[1] = params.fmhaBmm1Scale[0] * kLog2e;
        }
        if (params.fmhaBmm2Scale)
        {
            // The scale after fmha bmm2.
            params.fmhaBmm2Scale[0] = params.quantScaleO[0] * params.dequantScaleQkv[0];
        }
    }
}

// This kernel computes the attention mask. We must compute this on-the-fly in the future.

template <typename AttentionMaskDataType>
__global__ void computeAttentionMask(AttentionMaskDataType* attentionMask, int const* seqLengths, int maxQSeqLength,
    int attentionWindowSize, AttentionMaskType attentionMaskType, BlockSparseParams blockSparseParams)
{
    // The index of the sequence in the batch.
    int batchIdx = blockIdx.y;

    // The number of items in the mask for each sequence.
    int maskSize = maxQSeqLength * maxQSeqLength;
    // The offset to the 1st element of the mask for that particular sequence.
    int batchOffset = batchIdx * maskSize;

    // The length of the sequence.
    int seqLength = seqLengths[batchIdx];

    // Iterate over the tokens to update the number of padded elements.
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < maskSize; idx += gridDim.x * blockDim.x)
    {
        // The position in the matrix.
        int rowIdx = idx / maxQSeqLength;
        int colIdx = idx % maxQSeqLength;

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
        case AttentionMaskType::BLOCKSPARSE:
            isValid = blockSparseParams.computeMask(rowIdx, colIdx, seqLength, 1 /*num_heads*/, 0 /*head_id*/);
            break;
        }

        // Store the mask.
        attentionMask[batchOffset + idx] = isValid ? AttentionMaskDataType(1.f) : AttentionMaskDataType(0.f);
    }
}

template <typename T>
void invokeBuildDecoderInfo(BuildDecoderInfoParams<T> const& params, cudaStream_t stream)
{
    // Compute the sequence and padding offsets.
    int const THREADS_PER_BLOCK = 256;
    TLLM_CHECK_WITH_INFO(params.rotaryEmbeddingDim / 2 <= 256 && params.rotaryEmbeddingDim % 2 == 0,
        "Rotary embedding dim is assumed to be smaller than 512 and multiple of 2.");
    TLLM_CHECK_WITH_INFO(
        !(params.seqKVLengths == nullptr && params.rotaryEmbeddingDim > 0), "KV sequence lengths buffer is invalid.");
    bool const need_encoder_padding_offsets
        = (params.encoderPaddingOffsets != nullptr) && (params.seqKVOffsets != nullptr);
    const size_t smem_size
        = (need_encoder_padding_offsets ? (params.batchSize + 1) * 2 : (params.batchSize + 1)) * sizeof(int);
    computeSeqAndPaddingOffsets<T, THREADS_PER_BLOCK>
        <<<params.batchSize, THREADS_PER_BLOCK, smem_size, stream>>>(params);

    // Compute the attention mask, if needed.
    if (params.attentionMask != nullptr)
    {
        TLLM_CHECK_WITH_INFO(params.seqQLengths != nullptr, "Q sequence lengths buffer is invalid.");
        int const MIN_BLOCKS = 512;
        int blocksPerSeq = 16;
        while (blocksPerSeq * params.batchSize < MIN_BLOCKS)
        {
            blocksPerSeq *= 2;
        }
        dim3 grid(blocksPerSeq, params.batchSize);
        computeAttentionMask<<<grid, THREADS_PER_BLOCK, 0, stream>>>(params.attentionMask, params.seqQLengths,
            params.maxQSeqLength, params.attentionWindowSize, params.attentionMaskType, params.blockSparseParams);
    }
}

template void invokeBuildDecoderInfo(BuildDecoderInfoParams<float> const&, cudaStream_t);
template void invokeBuildDecoderInfo(BuildDecoderInfoParams<half> const&, cudaStream_t);
#ifdef ENABLE_BF16
template void invokeBuildDecoderInfo(BuildDecoderInfoParams<__nv_bfloat16> const&, cudaStream_t);
#endif
#ifdef ENABLE_FP8
template void invokeBuildDecoderInfo(BuildDecoderInfoParams<__nv_fp8_e4m3> const&, cudaStream_t);
#endif

__global__ void updatePaddingCountKernel(int* paddingPerSeq, int const* seqLengths, int maxQSeqLength, int batchSize)
{

    for (int ii = threadIdx.x; ii < batchSize; ii += blockDim.x)
    {
        paddingPerSeq[ii] = maxQSeqLength - seqLengths[ii];
    }
}

} // namespace kernels
} // namespace tensorrt_llm
