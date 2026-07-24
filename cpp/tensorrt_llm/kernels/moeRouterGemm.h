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
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

// MoE router GEMM: logits[M, N] = act[M, K] @ weight[N, K]^T.
//
// The activation is loaded as bf16 and widened to fp32 on-chip, multiplied
// against an fp32 weight with fp32 accumulation. Keeping the weight in fp32
// preserves the routing-critical precision, and widening on-chip avoids a
// separate fp32 copy of the activation in global memory.
//
//   act    : [M, K] row-major, dtype T (bf16 or fp16).
//   weight : [N, K] row-major, fp32 (the raw [num_experts, hidden] tensor).
//   output : [M, N] row-major, fp32.
template <typename T>
void invokeMoeRouterGemm(float* output, T const* act, float const* weight, int num_tokens, int num_experts,
    int hidden_dim, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
