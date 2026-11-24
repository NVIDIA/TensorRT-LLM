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

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM31
{

constexpr int kAddRmsNormHidden = 6144;

//! Operands of invokeAddRmsNorm; every tensor contiguous, input / residual / weight / replicas 16-byte aligned.
struct AddRmsNormParams
{
    __nv_bfloat16* input;        // [numRows, 6144]: the rows in, the normalized rows out
    __nv_bfloat16* residual;     // [numRows, 6144]: the residual in, bf16(input + residual) out
    __nv_bfloat16 const* weight; // [6144]
    uint64_t* q;                 // [numRows, 6144] E4M3 as words of 8 bytes, or null (no MXFP8 form)
    uint32_t* sfWords;           // the swizzled UE8M0 scales (mxfp8_rows.scale_bytes) as words; null with q
    __nv_bfloat16* replicas;     // [numRows, numReplicas * 6144], or null with numReplicas 0
    int numReplicas;
    int numRows;
    float eps;
    // NORM_L2PF: up to two byte ranges (the next GEMM's weight and weight scales, never written by this
    // kernel) that every CTA's threads split and pull into L2 with cp.async.bulk.prefetch.L2 before the PDL wait;
    // null / 0 = none.
    char const* prefetch0;
    uint64_t prefetch0Bytes;
    char const* prefetch1;
    uint64_t prefetch1Bytes;
};

//! The MiniMax-M3.1 residual add + RMSNorm of BF16 rows of width 6144, bit for bit the served kernels
//! it replaces (cute_dsl_kernels add_rmsnorm_mxfp8 / add_rmsnorm_replicated / add_rmsnorm_replicated_mxfp8, and
//! FlashInfer's fused_add_rmsnorm for the plain form, q null and numReplicas 0). Per row: h = input + residual (fp32),
//! residual = bf16(h), input = bf16(h * rsqrt.approx(mean(h * h) + eps) * w), optionally its MXFP8 form (32-element
//! UE8M0 blocks, 128x4-swizzled scales, the last 128-row tile's padding rows zeroed) and numReplicas BF16 copies side
//! by side. The launch shape follows the launch table (4-CTA clusters up to 92 rows, 2-CTA up to 106, one CTA above);
//! numRows <= 65535. The MXFP8 forms need an sm_100+ arch- or family-specific build (sm_107a); elsewhere they trap.
//! PDL: waits on its predecessor and releases its dependents right after the wait (weights are read before it).
//! The prefetch ranges only warm L2 for the consumer GEMM: no output depends on them.
cudaError_t invokeAddRmsNorm(AddRmsNormParams const& params, bool pdl, cudaStream_t stream);

} // namespace minimaxM31
} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
