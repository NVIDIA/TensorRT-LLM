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
#include <cuda_runtime.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{

// Cuda-core W4A16 NVFP4 GEMV for decode rows (1 <= m <= kNvfp4W4a16GemvMaxRows):
//   out[m, n] = bf16(g * sum_k x[m, k] * e2m1(w[n, k]) * e4m3(sf[n, k / 16]))
// x:   [m, k] BF16, row-major, contiguous
// w:   [n, k / 2] packed E2M1 (fp4_quantize layout: element 2b is the low nibble of byte b)
// sf:  [n, k / 16] E4M3 block scales, linear (not swizzled)
// out: [m, n] BF16
// g:   FP32 global scale
// Each CTA owns 2 output rows across the whole K; all weight and scale loads are
// issued before griddepcontrol.wait, and dependents are triggered before the
// reduction. Returns false (no launch) for unsupported shapes or devices (SM120/121,
// n even, k in {5120, 6144}); the caller must then use another path.
constexpr int kNvfp4W4a16GemvMaxRows = 4;

bool nvfp4W4a16GemvSupported(int n, int k);

bool nvfp4W4a16Gemv(void const* x, void const* w, void const* sf, void* out, float g, int m, int n, int k,
    bool enablePdl, cudaStream_t stream);

// The same GEMV with a fused GDN decode causal_conv1d_update epilogue (width 4, no bias, SiLU) on the
// output columns [0, convDim), for 1 <= m <= kNvfp4W4a16GemvConvMaxRows:
//   for row r with slot s = convStateIndices[r] != padSlotId and column c < convDim:
//     xv = bf16(GEMV output); out[r, c] = bf16(silu(sum_i w[c, i] * [state[s, c, 0..2], xv][i]))
//     state[s, c, 0..2] = [state[s, c, 1], state[s, c, 2], xv]
//   PAD rows keep the raw GEMV output and leave the state untouched; columns >= convDim are the plain
//   GEMV output. Bitwise equal to nvfp4W4a16Gemv followed by causal_conv1d_update_kernel_sl1.
// convState:  [slots, convDim, 3] BF16 with strides (convStateBatchStride, 3, 1)
// convWeight: [convDim, 4] BF16, contiguous, 8-byte aligned
// convDim must be even (a CTA's two columns are either both conv channels or both not).
struct Nvfp4GemvConvParams
{
    void* convState;
    void const* convWeight;
    int const* convStateIndices;
    int64_t convStateBatchStride;
    int convDim;
    int padSlotId;
};

constexpr int kNvfp4W4a16GemvConvMaxRows = 3;

bool nvfp4W4a16GemvConv(void const* x, void const* w, void const* sf, void* out, float g, int m, int n, int k,
    Nvfp4GemvConvParams const& conv, bool enablePdl, cudaStream_t stream);

} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
