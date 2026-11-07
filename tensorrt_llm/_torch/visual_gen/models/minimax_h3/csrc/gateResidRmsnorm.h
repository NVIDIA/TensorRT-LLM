/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace tensorrt_llm::visual_gen::minimax_h3
{

//! Hidden size the gated-residual + RMSNorm + modulation kernel is built for.
constexpr int kGateResidRmsnormHidden = 5376;

//! For each row r of x, attn [rows, kGateResidRmsnormHidden] and m = idx[r]:
//!   h[r] = bf16(x[r] + attn[r] * gate[m]),  y[r] = bf16(rmsnorm(h[r]) * w * (1 + scale[m]) + shift[m]).
//! gate, scale and shift are row views with unit column stride (gateStride / modStride elements between rows,
//! scale and shift sharing modStride); every pointer and row stride is 16-byte aligned. Needs sm_90 or newer.
cudaError_t invokeGateResidRmsnormIndexedShiftScale(__nv_bfloat16* y, __nv_bfloat16* h, __nv_bfloat16 const* x,
    __nv_bfloat16 const* attn, __nv_bfloat16 const* w, __nv_bfloat16 const* gate, __nv_bfloat16 const* scale,
    __nv_bfloat16 const* shift, int64_t const* idx, int64_t gateStride, int64_t modStride, int rows, float eps,
    cudaStream_t stream);

} // namespace tensorrt_llm::visual_gen::minimax_h3
