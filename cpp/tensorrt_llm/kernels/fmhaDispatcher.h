/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/kernels/contextFusedMultiHeadAttention/fmhaRunner.h"
#include "tensorrt_llm/kernels/contextFusedMultiHeadAttention/fused_multihead_attention_common.h"
#include "tensorrt_llm/kernels/trtllmGenKernels/fmha/fmhaRunner.h"

using tensorrt_llm::common::op::UniqPtrWNullCopy;

namespace tensorrt_llm::kernels
{

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

    // Does FMHA need a separate Q and Kv input ?
    bool isSeparateQAndKvInput() const
    {
        return mFixedParams.attentionInputLayout != kernels::AttentionInputLayout::PACKED_QKV;
    }

    // Run the fmha kernel.
    void run(tensorrt_llm::kernels::MHARunnerParams runnerParams);

private:
    // The fixed fmha parameters.
    kernels::MHARunnerFixedParams mFixedParams;
    // Whether to enable trtllm-gen kernels.
    bool mUseTllmGen;
    // Runner for fmha v2 kernels (for SM <= 90)
    UniqPtrWNullCopy<kernels::FusedMHARunnerV2> mFMHARunner;
    // Runner for trtllm-gen fmha kernels (for SM == 100)
    UniqPtrWNullCopy<kernels::TllmGenFmhaRunner> mTllmGenFMHARunner;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace tensorrt_llm::kernels
