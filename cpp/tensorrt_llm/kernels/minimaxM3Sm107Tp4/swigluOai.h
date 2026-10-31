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
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM3Sm107Tp4
{

//! The dense MLP's swigluoai activation: out[m, j] = bf16(min(g, 7) * sigmoid(1.702 * min(g, 7)) * (clamp(u, -7, 7) +
//! 1)) with g = x[m, j], u = x[m, d + j] (x bf16 [rows, 2d], out bf16 [rows, d], both contiguous, d even): bit for bit
//! the Triton silu_and_mul kernel at swiglu_limit 7, swiglu_alpha 1.702, swiglu_beta 1. With `pdl` the launch allows
//! programmatic serialization and the grid waits on its dependency, then releases its dependents before it computes
//! (the down GEMV stages its weights during the activation and reads it only after its own grid wait).
cudaError_t invokeSwigluOai(
    uint32_t const* x, uint32_t* out, int64_t rows, int64_t halfD, bool pdl, cudaStream_t stream);

} // namespace minimaxM3Sm107Tp4
} // namespace kernels

TRTLLM_NAMESPACE_END
