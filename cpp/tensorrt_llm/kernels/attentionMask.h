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

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/gptKernels.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <sstream>

namespace tc = tensorrt_llm::common;

namespace tensorrt_llm
{
namespace kernels
{

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename MaskDataType>
struct AttentionMaskParams
{
    // The attention mask's shape is [batchSize, maxQSeqLen, maxKvSeqLen] when cuQSeqLens is nullptr,
    // otherwise [batchSize, maxQSeqLen] is packed as numTokens.
    MaskDataType* mask = nullptr;
    // The cumulative sequence lengths of Q.
    int* cuQSeqLens = nullptr;
    // The actual q sequence lengths (used to create cuMaskRows).
    int const* actualQSeqLens = nullptr;
    // The actual kv sequence lengths (used to construct the packed mask when full mask is not given).
    int const* actualKvSeqLens = nullptr;
    // The attention mask type.
    AttentionMaskType attentionMaskType = AttentionMaskType::PADDING;
    // Params for block sparse pattern
    BlockSparseParams blockSparseParams;
    // The batch size.
    int batchSize;
    // The maximum q sequence length.
    int maxQSeqLen;
    // The maximum kv sequence length.
    int maxKvSeqLen;
    // The sliding window size.
    int slidingWindowSize;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename MaskDataType>
void invokeBuildAttentionMask(AttentionMaskParams<MaskDataType> const& params, cudaStream_t stream);

} // namespace kernels
} // namespace tensorrt_llm
