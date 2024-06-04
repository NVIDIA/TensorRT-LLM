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
#pragma once

#include "tensorrt_llm/runtime/iTensor.h"
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <sstream>

namespace tensorrt_llm
{
namespace kernels
{

enum class AttentionMaskType
{
    // Mask the padded tokens.
    PADDING = 0,
    // Mask the padded tokens and all the tokens that come after in a sequence.
    CAUSAL = 1,
    // See ChatGLM-6B mask.
    BIDIRECTIONAL = 2,
    // See GLM-10B mask.
    // TODO: merge this mask into BIDIRECTIONAL
    BIDIRECTIONALGLM = 3,
    // For Phi-3-small model
    BLOCKSPARSE = 4,
};

enum class PositionEmbeddingType : int8_t
{
    kLEARNED_ABSOLUTE = 0,
    kROPE_GPTJ = 1,
    kROPE_GPT_NEOX = 2,
    kLONG_ROPE = 3,
    // Workflow: (bmm1_output * scale_bmm1 + alibi).
    kALIBI = 4,
    // Workflow: (bmm1_output + alibi) * scale_bmm1.
    kALIBI_WITH_SCALE = 5,
    kRELATIVE = 6,
};

enum class RotaryScalingType : int8_t
{
    kNONE = 0,
    kLINEAR = 1,
    kDYNAMIC = 2,
};

struct BlockSparseParams
{
    int block_size;
    int homo_head_pattern;
    int num_local_blocks; // Sliding window blocks
    int vertical_stride;

    __device__ bool computeMask(int row_idx, int col_idx, int seq_length, int num_heads, int head_idx) const
    {
        bool causal_mask = row_idx < seq_length && col_idx < seq_length && col_idx <= row_idx;

        // Mask 1/0 decision is made at block_size granularity
        int block_row_idx = row_idx / block_size;
        int block_col_idx = col_idx / block_size;

        bool block_local_mask = (block_row_idx - block_col_idx) < num_local_blocks;

        int head_sliding_step = homo_head_pattern ? 0 : std::max(1, int(vertical_stride / num_heads));
        bool block_vertical_stride_mask = ((block_col_idx + head_idx * head_sliding_step + 1) % vertical_stride) == 0;

        bool is_valid = causal_mask && (block_local_mask || block_vertical_stride_mask);
        return is_valid;
    }
};

template <typename AttentionMaskDataType>
struct BuildDecoderInfoParams
{
    // The offsets to the 1st token in each sequence of Q buffer. Shape: [batchSize+1].
    int* seqQOffsets;
    // The offsets to the 1st token in each sequence of KV buffer. Shape: [batchSize+1].
    int* seqKVOffsets;
    // The number of padded tokens in the corresponding padded tensor before the current token. Shape: [numTokens].
    int* paddingOffsets;

    // The mask to mark invalid tokens in Attention - that's not used by the plugins as it can be
    // computed on-the-fly. When it's not needed, simply use nullptr.
    // Shape: [batchSize, maxSeqLength, maxSeqLength].
    AttentionMaskDataType* attentionMask;

    // The Q length of each sequence in the batch. Shape: [batchSize].
    int const* seqQLengths;
    // The KV length of each sequence in the batch. Shape: [batchSize].
    int const* seqKVLengths;

    // The fmha tile counter ptr (set to 0 before fmha).
    uint32_t* fmhaTileCounter;

    // The number of sequences in the batch.
    int batchSize;
    // The maximum query length of a sequence; it includes input and output.
    int maxQSeqLength;
    // Whether remove the input padding or not.
    bool removePadding;
    // The kv cache capacity.
    // We will apply the limited_length_causal mask when there are not enough kv cache.
    int attentionWindowSize;
    // The number of sink tokens in the kv cache.
    int sinkTokenLength;
    // The number of tokens in total. It's \sum_{ii=0}^{batchSize} seqLengths[ii].
    int numTokens;
    // The type of attention.
    AttentionMaskType attentionMaskType;
    // Params for block sparse pattern
    BlockSparseParams blockSparseParams;

    // Rotary Embedding inv_freq.
    // [batch_size, halfRotaryDim] variable across different requests due to dynamic scaling.
    float rotaryEmbeddingScale;
    float rotaryEmbeddingBase;
    int rotaryEmbeddingDim;
    RotaryScalingType rotaryScalingType;
    float* rotaryEmbeddingInvFreq;
    float2* rotaryEmbeddingCoeffCache;
    // Dynamic scaling;
    int rotaryEmbeddingMaxPositions;

    std::string toString() const
    {
        std::stringstream ss;
        ss << "BuildDecoderInfoParams ====================" << std::endl;
        ss << "seqQOffsets: "
           << *(runtime::ITensor::wrap(
                  (void*) seqQOffsets, nvinfer1::DataType::kINT32, runtime::ITensor::makeShape({batchSize})))
           << std::endl;
        ss << "seqKVOffsets: "
           << *(runtime::ITensor::wrap(
                  (void*) seqKVOffsets, nvinfer1::DataType::kINT32, runtime::ITensor::makeShape({batchSize})))
           << std::endl;
        ss << "paddingOffsets: "
           << *(runtime::ITensor::wrap(
                  (void*) paddingOffsets, nvinfer1::DataType::kINT32, runtime::ITensor::makeShape({batchSize})))
           << std::endl;
        ss << "attentionMask: " << static_cast<void*>(attentionMask) << std::endl;
        ss << "seqQLengths: " << seqQLengths << std::endl;
        ss << "seqKVLengths: " << seqKVLengths << std::endl;
        ss << "fmhaTileCounter: " << fmhaTileCounter << std::endl;
        ss << "batchSize: " << batchSize << std::endl;
        ss << "maxQSeqLength: " << maxQSeqLength << std::endl;
        ss << "removePadding: " << std::boolalpha << removePadding << std::endl;
        ss << "attentionWindowSize: " << attentionWindowSize << std::endl;
        ss << "sinkTokenLength: " << sinkTokenLength << std::endl;
        ss << "numTokens: " << numTokens << std::endl;
        ss << "attentionMaskType: " << static_cast<int>(attentionMaskType) << std::endl;
        ss << "rotaryEmbeddingScale: " << rotaryEmbeddingScale << std::endl;
        ss << "rotaryEmbeddingBase: " << rotaryEmbeddingBase << std::endl;
        ss << "rotaryEmbeddingDim: " << rotaryEmbeddingDim << std::endl;
        ss << "rotaryScalingType: " << static_cast<int>(rotaryScalingType) << std::endl;
        ss << "rotaryEmbeddingInvFreq: " << rotaryEmbeddingInvFreq << std::endl;
        ss << "rotaryEmbeddingCoeffCache: " << rotaryEmbeddingCoeffCache << std::endl;
        ss << "rotaryEmbeddingMaxPositions: " << rotaryEmbeddingMaxPositions << std::endl;

        return ss.str();
    }
};

template <typename T>
void invokeBuildDecoderInfo(BuildDecoderInfoParams<T> const& params, cudaStream_t stream);

} // namespace kernels
} // namespace tensorrt_llm
