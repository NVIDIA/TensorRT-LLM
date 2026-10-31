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
namespace minimaxM3Sm107Tp2
{

//! out[m, j] = bf16(silu(g) * (u + 0)) with g = x[m, j], u = x[m, d + j] (x bf16 [rows, 2d], out bf16 [rows, d], both
//! contiguous, d a multiple of 8, 16-byte aligned rows): bit for bit the Triton silu_and_mul kernel at its defaults
//! (no limit, alpha 1, beta 0), the plain SwiGLU of the Eagle3 drafter MLP.
cudaError_t invokeSiluAndMul(uint16_t const* x, uint16_t* out, int64_t rows, int64_t d, cudaStream_t stream);

} // namespace minimaxM3Sm107Tp2
} // namespace kernels

TRTLLM_NAMESPACE_END
