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

/*
 * Fused Gated RMSNorm + NVFP4 Quantization CUDA Kernel
 *
 * Fuses three operations for Nemotron-H NVFP4 quantized path:
 * 1. SiLU gating: gated = x * z * sigmoid(z)
 * 2. Group RMSNorm: y = norm(gated) * weight
 * 3. NVFP4 quantization with block scaling
 *
 * Key optimizations:
 * - Register-based storage: gated values in registers (single HBM pass for x, z)
 * - Inline float-to-FP4 quantization (skips intermediate bf16 conversion)
 * - Vectorized loads with uint4 (8 bf16 per load)
 *
 * Performance: 1.3x-3.4x faster than Triton + fp4_quantize baseline
 * Requires SM100 (Blackwell) or newer
 */

#pragma once

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

// FP4 E2M1 constants
constexpr float FP4_E2M1_MAX = 6.0f;
constexpr int FP4_BLOCK_SIZE = 16;

// Kernel parameters
template <typename T>
struct FusedGatedRMSNormQuantParams
{
    T const* x;            // Input [M, N]
    T const* z;            // Gate [M, N] (can be strided)
    T const* weight;       // RMSNorm weight [N]
    uint32_t* y_fp4;       // Output FP4 [M, N/8] (8 FP4 values packed per uint32)
    uint32_t* sf_out;      // Scale factors (swizzled layout)
    float const* sf_scale; // Global scale factor for FP4
    int M;                 // Number of rows
    int N;                 // Full hidden dimension
    int zRowStride;        // Row stride for z (allows non-contiguous z)
    int groupSize;         // Normalization group size
    float eps;             // Epsilon for RMSNorm
    cudaStream_t stream;
};

// Launch the fused gated RMSNorm + FP4 quantization kernel
template <typename T>
void invokeFusedGatedRMSNormQuant(FusedGatedRMSNormQuantParams<T> const& params, int multiProcessorCount);

// Explicit instantiations
extern template void invokeFusedGatedRMSNormQuant<half>(
    FusedGatedRMSNormQuantParams<half> const& params, int multiProcessorCount);

#ifdef ENABLE_BF16
extern template void invokeFusedGatedRMSNormQuant<__nv_bfloat16>(
    FusedGatedRMSNormQuantParams<__nv_bfloat16> const& params, int multiProcessorCount);
#endif

} // namespace kernels

TRTLLM_NAMESPACE_END
