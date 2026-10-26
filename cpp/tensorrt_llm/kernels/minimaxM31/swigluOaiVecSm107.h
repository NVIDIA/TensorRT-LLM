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
namespace minimaxM31
{

//! SwiGLU-OAI (limit 7, alpha 1.702, beta 1) of x = [gate | up] bf16 [rows, 2d] -> out bf16 [rows, d], bit for bit the
//! Triton silu_and_mul kernel (the in-tree swigluOai pair sequence: ex2.approx, div.full, one rounding). d % 8 == 0 and
//! a 16-byte aligned x take 16-byte vectors with nv (1 / 2 / 4) output vectors per thread; otherwise one packed pair
//! per thread. d must be even. PDL: waits on its predecessor, releases its dependents early.
cudaError_t invokeSwigluOaiVec(
    void const* x, void* out, int64_t rows, int64_t d, int nv, bool pdl, cudaStream_t stream);

} // namespace minimaxM31
} // namespace kernels

TRTLLM_NAMESPACE_END
