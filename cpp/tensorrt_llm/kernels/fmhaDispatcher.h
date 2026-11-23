/*
 * Copyright (c) 2020-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/kernels/contextFusedMultiHeadAttention/fmhaRunner.h"
#include "tensorrt_llm/kernels/contextFusedMultiHeadAttention/fused_multihead_attention_common.h"
#include "tensorrt_llm/kernels/trtllmGenKernels/fmha/fmhaRunner.h"

#include <memory>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail
{

struct TrtllmGenMaskSelection
{
    TrtllmGenAttentionMaskType maskType;
    int32_t leftSlidingWindow = -1;
    int32_t rightSlidingWindow = -1;
    int32_t chunkedAttentionSize = 0;
};

TrtllmGenMaskSelection selectTrtllmGenMask(ContextAttentionMaskType attentionMaskType, int32_t slidingWindowSize,
    int32_t chunkedAttentionSize, int32_t kvSeqLen);

} // namespace detail

////////////////////////////////////////////////////////////////////////////////////////////////////

class FmhaDispatcher
{
public:
    // Constructor.
    FmhaDispatcher(kernels::MHARunnerFixedParams fixedParams);

    // Deconstructor.
    ~FmhaDispatcher() = default;

    // Check if any fmha kernel meets the requirements.
    bool isSupported();

    // Check whether a kernel exists for one call, whose mask depends on the call itself: a sliding
    // window that covers the whole history selects the causal mask, one that does not selects a
    // windowed mask no fixed probe can speak for.
    bool isSupported(tensorrt_llm::kernels::MHARunnerParams const& runnerParams);

    // The same check for one call under a layout other than the fixed one.
    bool isSupported(tensorrt_llm::kernels::MHARunnerParams const& runnerParams, kernels::AttentionInputLayout layout);

    // Whether the kernels come from trtllm-gen rather than fmha v2.
    bool usesTrtllmGen() const
    {
        return mUseTllmGen;
    }

    // Does FMHA need a separate Q and Kv input ?
    bool isSeparateQAndKvInput() const
    {
        return mFixedParams.attentionInputLayout != kernels::AttentionInputLayout::PACKED_QKV;
    }

    // Run the fmha kernel.
    void run(tensorrt_llm::kernels::MHARunnerParams runnerParams);

    // Run the fmha kernel under a layout other than the fixed one.
    void run(tensorrt_llm::kernels::MHARunnerParams runnerParams, kernels::AttentionInputLayout layout);

private:
    // The kernel-selection parameters for one call: its own sequence lengths, window and chunking
    // decide the mask, so a verdict on one call does not stand for another.
    kernels::TllmGenFmhaRunnerParams makeTllmGenProbeParams(
        tensorrt_llm::kernels::MHARunnerParams const& runnerParams, kernels::AttentionInputLayout layout) const;

    // Run the fmha kernel under the given layout.
    void runImpl(tensorrt_llm::kernels::MHARunnerParams runnerParams, kernels::AttentionInputLayout layout);

    // The fixed fmha parameters.
    kernels::MHARunnerFixedParams mFixedParams;
    // Whether to enable trtllm-gen kernels.
    bool mUseTllmGen;
    // Runner for fmha v2 kernels (for SM <= 90)
    std::unique_ptr<kernels::FusedMHARunnerV2> mFMHARunner;
    // Runner for trtllm-gen fmha kernels (for SM == 100)
    std::unique_ptr<kernels::TllmGenFmhaRunner> mTllmGenFMHARunner;
    // Cached SM count to avoid repeated cudaDeviceGetAttribute calls in the per-iter
    // FMHA dispatch hot path (isSupported / run).
    int mMultiProcessorCount{0};
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernels

TRTLLM_NAMESPACE_END
