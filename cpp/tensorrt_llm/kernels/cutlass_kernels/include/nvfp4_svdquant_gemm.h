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

#include <cstddef>
#include <cuda_runtime_api.h>

namespace tensorrt_llm::kernels::cutlass_kernels
{

// SVDQuant fused NVFP4 GEMM (SM100): the residual block-scaled NVFP4 GEMM out = alpha * (A @ Bᵀ)
// fused with the rank-r LoRA-up correction D @ L1ᵀ, computed by a 2nd bf16 tcgen05 MMA (K = r)
// into the SAME TMEM accumulator after the NVFP4 K-loop (a custom CUTLASS SM100 block-scaled
// collective; the stock nvfp4 GEMM template is untouched). 1/alpha is folded into L1 host-side so
// the unchanged LinearCombination epilogue (out = alpha * acc) yields alpha * residual + D @ L1ᵀ.
//
//   A   [m, k]  NVFP4 (e2m1), row-major          SFA/SFB  block scale factors (ue4m3, swizzled)
//   B   [n, k]  NVFP4 (e2m1), column-major        alpha   per-tensor dequant scale (device f32[1])
//   D   [m, r]  bf16, row-major (r = 32)          out     [m, n] bf16
//   L1  [n, r]  bf16, row-major (= svdquant_lora_b / alpha)
size_t nvfp4_svdquant_gemm_workspace_size(int m, int n, int k);

void nvfp4_svdquant_gemm_run(void* out, void const* A, void const* B, void const* sfa, void const* sfb,
    float const* alpha, void const* D, void const* L1, int m, int n, int k, char* workspace, size_t workspaceBytes,
    cudaStream_t stream);

} // namespace tensorrt_llm::kernels::cutlass_kernels
