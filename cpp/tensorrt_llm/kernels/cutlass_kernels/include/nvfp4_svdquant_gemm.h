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

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::cutlass_kernels
{

// SVDQuant fused NVFP4 GEMM (SM100): the residual block-scaled NVFP4 GEMM out = alpha * (A @ Bᵀ)
// fused with the rank-r LoRA-up correction D @ L1ᵀ, computed by a 2nd bf16 tcgen05 MMA (K = r)
// into the SAME TMEM accumulator after the NVFP4 K-loop (a custom CUTLASS SM100 block-scaled
// collective; the stock nvfp4 GEMM template is untouched). 1/alpha is folded into L1 host-side so
// the fused epilogue yields alpha * residual + D @ L1ᵀ + bias.
//
//   A   [m, k]  NVFP4 (e2m1), row-major          SFA/SFB  block scale factors (ue4m3, swizzled)
//   B   [n, k]  NVFP4 (e2m1), column-major        alpha   per-tensor dequant scale (device f32[1])
//   D   [m, r]  bf16, row-major (r = 32)          out     [m, n] bf16
//   L1  [n, r]  bf16, row-major (= svdquant_lora_b / alpha)
enum class Nvfp4SvdquantGemmTactic : int
{
    // Preserve the original tactic IDs used by existing sweeps.
    k1Sm128x256x128 = 0,
    k2Sm256x256x128 = 1,
    k1Sm128x256x128Cluster1x2 = 2,
    k1Sm128x256x128Cluster1x4 = 3,
    k2Sm256x256x128Cluster2x2 = 4,
    k2Sm256x256x128Cluster2x4 = 5,
    k2Sm256x256x128Cluster4x2 = 6,
    k2Sm256x256x128Cluster4x4 = 7,
    k2Sm256x256x128Cluster4x1 = 8,
    k1Sm128x128x128 = 9,
    k1Sm128x128x128Cluster1x2 = 10,
    k2Sm256x192x128 = 11,
    k2Sm256x192x128Cluster2x2 = 12,
};

inline constexpr int kNvfp4SvdquantGemmNumTactics = 13;

// Tactic 0 preserves the original 1SM kernel and remains the source-compatible default.
size_t nvfp4_svdquant_gemm_workspace_size(int m, int n, int k, int tactic = 0);

void nvfp4_svdquant_gemm_run(void* out, void const* A, void const* B, void const* sfa, void const* sfb,
    float const* alpha, void const* D, void const* L1, void const* bias, int m, int n, int k, char* workspace,
    size_t workspaceBytes, cudaStream_t stream, int tactic = 0, int64_t dStride = 32);

} // namespace kernels::cutlass_kernels

TRTLLM_NAMESPACE_END
