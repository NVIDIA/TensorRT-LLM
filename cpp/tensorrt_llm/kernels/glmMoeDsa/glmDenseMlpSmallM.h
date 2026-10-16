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

#include <cuda_bf16.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::glm_moe_dsa::dense_mlp_small_m
{

//! Largest number of activation rows (tokens) the kernels accept.
constexpr int kMaxRows = 8;

//! Tiling argument that selects the default tiling of the shape.
constexpr int kDefaultTiling = -1;

//! Host-side eligibility checks (no device work). tiling = kDefaultTiling checks the default tiling of the shape.
//! \param numRows activation rows M.
//! \param hidden model hidden size (gate/up contraction size, down output size).
//! \param inter rank-local intermediate size (gate/up output size, down contraction size).
bool isGateUpSupported(int numRows, int hidden, int inter, int tiling = kDefaultTiling);
bool isDownSupported(int numRows, int inter, int hidden, int tiling = kDefaultTiling);

//! Gate/up projection fused with SwiGLU. For every row m < numRows and column j < inter:
//!   gate = bf16(sum_k x[m, k] * W[j, k]), up = bf16(sum_k x[m, k] * W[inter + j, k]),
//!   h[m, j] = bf16((gate * sigmoid(gate)) * up), sums in FP32, sigmoid as 1 / (1 + exp(-gate)) with the fast exp
//!   and division.
//! \param x activations [numRows, hidden], row-major, BF16, 16-byte aligned.
//! \param gateUpWeight [2 * inter, hidden], row-major, BF16 (rows [0, inter) gate, [inter, 2 * inter) up).
//! \param h output [numRows, inter], row-major, BF16.
cudaError_t launchGateUpSwiglu(__nv_bfloat16 const* x, __nv_bfloat16 const* gateUpWeight, __nv_bfloat16* h, int numRows,
    int hidden, int inter, int tiling, bool enablePdl, cudaStream_t stream);

//! Down projection: out[m, n] = bf16(sum_k h[m, k] * W[n, k]) with FP32 sums.
//! \param h activations [numRows, inter], row-major, BF16, 16-byte aligned.
//! \param downWeight [hidden, inter], row-major, BF16.
//! \param out output [numRows, hidden], row-major, BF16.
cudaError_t launchDown(__nv_bfloat16 const* h, __nv_bfloat16 const* downWeight, __nv_bfloat16* out, int numRows,
    int inter, int hidden, int tiling, bool enablePdl, cudaStream_t stream);

} // namespace kernels::glm_moe_dsa::dense_mlp_small_m

TRTLLM_NAMESPACE_END
