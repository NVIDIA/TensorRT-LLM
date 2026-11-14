/*
 * Copyright (c) 2019-2026, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
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
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{
void invokeGlmRubinNoAuxTcFp4Quant(float* scores, float* bias, __nv_bfloat16 const* hiddenStates,
    float const* globalScale, float* topkValues, int32_t* topkIndices, uint32_t* fp4Out, uint8_t* sfOut,
    int64_t const numTokens, int64_t const numExperts, int64_t const topk, int64_t const hiddenSize,
    double const routedScalingFactor, cudaStream_t const stream);
} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
