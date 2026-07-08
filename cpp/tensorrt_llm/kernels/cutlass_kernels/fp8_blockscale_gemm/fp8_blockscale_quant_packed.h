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

// Host-callable launcher for the fused FP8 1x128 quantize + UE8M0-pack kernel.
// Kernel implementation lives in fp8_blockscale_quant_packed.cu and is built
// by nvcc; this header is safe to include from .cpp files compiled by g++.

#pragma once

#include "tensorrt_llm/common/config.h"

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::fp8_blockscale_gemm
{

// Launches the fused 1x128 FP8 quant + UE8M0 pack kernel.
//
// Inputs:
//   input  : BF16 [m, k] row-major contiguous
// Outputs:
//   fp8_output           : E4M3 [m, k] row-major contiguous
//   packed_scale_output  : uint32 [packed_sf_k, scale_leading_dim_uint32]
//                          where packed_sf_k = ceil(ceil(k/128)/4)
//
// `scale_leading_dim_uint32` is the (uint32) stride between consecutive
// packed_sf_k rows of the scale tensor; caller is responsible for choosing
// it (typically aligned to 4 uint32 = 16 bytes for TMA alignment).
void launch_fp8_quantize_1x128_packed_bf16_e4m3(__nv_fp8_e4m3* fp8_output, int32_t* packed_scale_output,
    __nv_bfloat16 const* input, int m, int k, int scale_leading_dim_uint32, cudaStream_t stream);

} // namespace kernels::fp8_blockscale_gemm

TRTLLM_NAMESPACE_END
