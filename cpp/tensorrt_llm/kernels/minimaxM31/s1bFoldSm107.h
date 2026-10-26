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

#include "tensorrt_llm/common/config.h"

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM31
{

//! S1b fold: the MiniMax-M3.1 router's split-K reduce + the trtllm-gen MiniMax2 top-4 selection in one launch.
//! partials: contiguous fp32 [S, M, 128], one strided-batched TF32 GEMM over S K-slices of the router projection; at
//! the split factor cuBLASLt's heuristic picks for the shape, summed in slice order they are bitwise its split-K
//! logits. Per row (one warp): the slice-order sum, then moe::dev::routing::TopKExpertSelect<SigmoidBias,
//! ScaledSumNormalize> (the production E128K8 instance), so ids (top-4, in slot order) and bf16 weights are bitwise
//! what routingIndicesClusterKernel selects from the same logits. bias: fp32 [128]; ids: int32 [M, 4]; weights: bf16
//! [M, 4]; logitsOut: fp32 [M, 128] or null. S must be 8 or 16. PDL: waits on its predecessor, releases its dependents
//! early.
cudaError_t invokeS1bFold(float const* partials, int32_t S, int32_t M, float const* bias, float routeScale,
    float* logitsOut, int32_t* ids, __nv_bfloat16* weights, bool pdl, cudaStream_t stream);

} // namespace minimaxM31
} // namespace kernels

TRTLLM_NAMESPACE_END
