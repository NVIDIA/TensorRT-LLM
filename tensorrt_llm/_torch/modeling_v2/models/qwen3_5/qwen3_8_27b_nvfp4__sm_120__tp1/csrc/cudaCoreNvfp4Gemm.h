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

#include <cuda_runtime.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{

// Cuda-core W4A4 NVFP4 GEMM for decode rows (1 <= m <= kCudaCoreNvfp4GemmMaxRows), the target's copy
// of TensorRT-LLM's cuda_core_nvfp4_gemm specialized to what it calls:
//   out[m, n] = bf16(alpha * sum_k e2m1(a[m, k]) * e4m3(sfA[m, k / 16]) * e2m1(w[n, k]) * e4m3(sfW[n, k / 16]))
// a:     [m, k / 2] packed E2M1
// sfA:   the 128x4 swizzled E4M3 scales written by fp4_quantize(isSfSwizzledLayout=True), read in place
// w:     [n, k / 2] packed E2M1
// sfW:   the 128x4 swizzled E4M3 weight scales
// alpha: device FP32 scalar
// out:   [m, n] BF16
// Weights stream with L1 no-allocate loads; with PDL, the first K step's weight loads are issued before
// griddepcontrol.wait. Returns false (no launch) unless n is even and k % 32 == 0.
constexpr int kCudaCoreNvfp4GemmMaxRows = 8;

bool cudaCoreNvfp4Gemm(void const* a, void const* w, void const* sfA, void const* sfW, float const* alpha, void* out,
    int m, int n, int k, bool enablePdl, cudaStream_t stream);

} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
