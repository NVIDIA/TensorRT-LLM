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

#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace dense_mlp
{

//! Largest number of activation rows (tokens) the small-M dense MLP kernels accept.
constexpr int kMaxRows = 8;

//! Number of tiling variants compiled for each kernel. Variant 0 is the default.
int getNumVariants();

//! Host-side eligibility checks (no device work).
//! \param numRows activation rows M.
//! \param hidden model hidden size (gate/up contraction size, down output size).
//! \param inter rank-local intermediate size (gate/up output size, down contraction size).
bool isGateUpSupported(int numRows, int hidden, int inter, int variant);
bool isDownSupported(int numRows, int inter, int hidden, int variant);

//! Gate/up projection fused with SwiGLU.
//! For every row m < numRows and column j < inter:
//!   gate = bf16(sum_k x[m, k] * W[j, k]), up = bf16(sum_k x[m, k] * W[inter + j, k]),
//!   h[m, j] = bf16((gate * sigmoid(gate)) * up), all sums in FP32.
//! \param x activations [numRows, hidden], row-major, BF16.
//! \param gateUpWeight [2 * inter, hidden], row-major, BF16 (rows [0, inter) gate, [inter, 2 * inter) up).
//! \param h output [numRows, inter], row-major, BF16.
cudaError_t invokeGateUpSwiglu(__nv_bfloat16 const* x, __nv_bfloat16 const* gateUpWeight, __nv_bfloat16* h, int numRows,
    int hidden, int inter, int variant, bool enablePdl, cudaStream_t stream);

//! Down projection: out[m, n] = bf16(sum_k h[m, k] * W[n, k]) with FP32 sums.
//! \param h activations [numRows, inter], row-major, BF16.
//! \param downWeight [hidden, inter], row-major, BF16.
//! \param out output [numRows, hidden], row-major, BF16.
cudaError_t invokeDown(__nv_bfloat16 const* h, __nv_bfloat16 const* downWeight, __nv_bfloat16* out, int numRows,
    int inter, int hidden, int variant, bool enablePdl, cudaStream_t stream);

} // namespace dense_mlp

TRTLLM_MV2_NAMESPACE_END
