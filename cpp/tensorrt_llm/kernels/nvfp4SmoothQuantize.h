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
#include <cuda_runtime_api.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

// Fused smooth + NVFP4 quantize: (out, sf_out) = NVFP4-quantize(in * pqs) in a single pass over in,
// folding the per-input-channel pre_quant_scale smoothing into the quantize. Byte-identical to
// fp4_quantize(in * pqs) (same cvt_warp_fp16_to_fp4 + swizzled SF layout), so the residual GEMM
// consumes the output unchanged. in [m, n] bf16, pqs [n] bf16, sf_scale f32[1] (the per-tensor
// global scale). out [m, n/2] uint8 (packed e2m1), sf_out swizzled UE4M3 block scales (vec size 16).
void nvfp4_smooth_quantize(void* out, void* sf_out, void const* in, void const* pqs, float const* sf_scale, int m,
    int n, int multiProcessorCount, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
