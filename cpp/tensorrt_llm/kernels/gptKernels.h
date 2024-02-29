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

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

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
    BIDIRECTIONALGLM = 3
};

enum class PositionEmbeddingType : int8_t
{
    kLEARNED_ABSOLUTE = 0,
    kROPE_GPTJ = 1,
    kROPE_GPT_NEOX = 2,
    // Workflow: (bmm1_output * scale_bmm1 + alibi).
    kALIBI = 3,
    // Workflow: (bmm1_output + alibi) * scale_bmm1.
    kALIBI_WITH_SCALE = 4,
    kRELATIVE = 5
};

enum class RotaryScalingType : int8_t
{
    kNONE = 0,
    kLINEAR = 1,
    kDYNAMIC = 2,
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
    const int* seqQLengths;
    // The KV length of each sequence in the batch. Shape: [batchSize].
    const int* seqKVLengths;

    // The number of sequences in the batch.
    int batchSize;
    // The maximum length of a sequence; it includes input and output.
    int maxSeqLength;
    // The kv cache capacity.
    // We will apply the limited_length_causal mask when there are not enough kv cache.
    int attentionWindowSize;
    // The number of sink tokens in the kv cache.
    int sinkTokenLength;
    // The number of tokens in total. It's \sum_{ii=0}^{batchSize} seqLengths[ii].
    int numTokens;
    // The type of attention.
    AttentionMaskType attentionMaskType;
};

template <typename T>
void invokeBuildDecoderInfo(const BuildDecoderInfoParams<T>& params, cudaStream_t stream);

} // namespace kernels
} // namespace tensorrt_llm
