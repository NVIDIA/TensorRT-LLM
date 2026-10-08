/*
 * Copyright (c) 2019-2026, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/common/cudaUtils.h"
#include <assert.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::dsv3MinLatencyKernels
{

template <typename T, int kNumTokens, int kNumExperts, int kHiddenDim>
void invokeRouterGemm(float* output, T const* mat_a, T const* mat_b, cudaStream_t stream);

// Tensor-core variant (mma.sync m16n8k16, bf16 in / fp32 out) for large expert counts such as Kimi K3's 896,
// where the scalar FFMA kernel above is instruction-bound. kNumTokens <= 16, kNumExperts % 8 == 0.
template <int kNumTokens, int kNumExperts, int kHiddenDim>
void invokeRouterGemmMma(float* output, __nv_bfloat16 const* mat_a, __nv_bfloat16 const* mat_b, cudaStream_t stream);

// BF16 input [M,7168], gate [896,7168], down [3584,7168]; FP32 logits and BF16 projection, M <= 16.
template <int kNumTokens>
void invokeRouterLatentGemmMma(float* logits, __nv_bfloat16* projection, __nv_bfloat16 const* input,
    __nv_bfloat16 const* gateWeight, __nv_bfloat16 const* downWeight, cudaStream_t stream);

} // namespace kernels::dsv3MinLatencyKernels

TRTLLM_NAMESPACE_END
