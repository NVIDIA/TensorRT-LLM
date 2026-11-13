/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/kernels/contextFusedMultiHeadAttention/fused_multihead_attention_common.h"
#include "tensorrt_llm/kernels/trtllmGenKernels/fmha/fmhaRunnerParams.h"

#include <climits>
#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace detail
{

/// The trtllm-gen mask of a context launch and the window fields that go with it.
struct TrtllmGenMaskSelection
{
    TrtllmGenAttentionMaskType maskType;
    int32_t leftSlidingWindow = -1;
    int32_t rightSlidingWindow = -1;
    int32_t chunkedAttentionSize = 0;
};

/// Maps a context mask type and this launch's window extents onto the trtllm-gen mask.
inline TrtllmGenMaskSelection selectTrtllmGenMask(ContextAttentionMaskType const attentionMaskType,
    int32_t const slidingWindowSize, int32_t const chunkedAttentionSize, int32_t const kvSeqLen)
{
    if (attentionMaskType == ContextAttentionMaskType::VARIABLE_WINDOW)
    {
        return {.maskType = TrtllmGenAttentionMaskType::VariableWindow};
    }

    TllmGenFmhaRunnerParams params{};
    params.setAttentionMaskType(static_cast<std::int8_t>(attentionMaskType));
    TrtllmGenMaskSelection selection{.maskType = params.mMaskType};

    bool const usesChunkedAttention
        = chunkedAttentionSize > 0 && chunkedAttentionSize != INT_MAX && chunkedAttentionSize < kvSeqLen;
    bool const usesSlidingWindow = slidingWindowSize > 0 && kvSeqLen > slidingWindowSize;
    TLLM_CHECK_WITH_INFO(!(usesChunkedAttention && usesSlidingWindow),
        "Chunked attention size and sliding window size should not be used together.");

    if (usesChunkedAttention)
    {
        selection.maskType = TrtllmGenAttentionMaskType::SlidingOrChunkedCausal;
        selection.chunkedAttentionSize = chunkedAttentionSize;
    }
    else if (usesSlidingWindow)
    {
        selection.maskType = TrtllmGenAttentionMaskType::SlidingOrChunkedCausal;
        selection.leftSlidingWindow = slidingWindowSize - 1;
        selection.rightSlidingWindow = 0;
    }
    else if (attentionMaskType == ContextAttentionMaskType::SLIDING_OR_CHUNKED_CAUSAL)
    {
        selection.maskType = TrtllmGenAttentionMaskType::Causal;
    }
    return selection;
}

} // namespace detail

/// Writes \p maskSelection into \p params.
inline void applyTrtllmGenMaskSelection(
    TllmGenFmhaRunnerParams& params, detail::TrtllmGenMaskSelection const& maskSelection)
{
    params.mMaskType = maskSelection.maskType;
    params.mLeftSlidingWindow = maskSelection.leftSlidingWindow;
    params.mRightSlidingWindow = maskSelection.rightSlidingWindow;
    params.mChunkedAttentionSize = maskSelection.chunkedAttentionSize;
}

} // namespace kernels

TRTLLM_NAMESPACE_END
