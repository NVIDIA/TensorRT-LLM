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
 * Fused LayerNorm + NVFP4 Quantization CUDA Kernel
 *
 * Targets Wan 2.2 (and other DiT models) where a LayerNorm immediately
 * feeds a NVFP4-quantized Linear layer. Two configurations are supported:
 *
 *   1. AdaLN (norm1, norm3 in WanBlock):
 *        y = ((x - mean) * rstd) * (1 + scale_msa) + shift_msa
 *      No learned LN weight/bias; modulation comes from timestep embedding.
 *
 *   2. Plain LN with affine (norm2 in WanBlock):
 *        y = ((x - mean) * rstd) * ln_weight + ln_bias
 *      Learned weight and bias; no modulation.
 *
 * Both configurations end with: y -> NVFP4 quantize (per 16-element block) -> output.
 *
 * Modulation broadcasting: scale_msa / shift_msa are shaped [B, N] (one row
 * per batch element). The same vector applies to all sequence positions
 * within a batch element, so the kernel indexes:
 *     batch_idx = row / seq_len_per_batch
 * to find the right modulation row for each input row.
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

// FP4 E2M1 constants (must match the values in fusedGatedRMSNormQuant.cuh).
constexpr float LN_FP4_E2M1_MAX = 6.0f;
constexpr int LN_FP4_BLOCK_SIZE = 16;

// Runtime parameters for the fused LayerNorm + NVFP4 quant kernel.
// The HAS_LN_AFFINE / HAS_MODULATION switches are compile-time on the
// kernel template; we store runtime booleans here so the launcher can
// dispatch to the right specialization.
template <typename T>
struct FusedLayerNormQuantParams
{
    T const* x;                  // Input, shape [M, N], contiguous
    T const* ln_weight;          // [N] - used only if has_ln_affine
    T const* ln_bias;            // [N] - used only if has_ln_affine
    T const* scale_msa;          // [B, N] - used only if has_modulation
    T const* shift_msa;          // [B, N] - used only if has_modulation
    uint32_t* y_fp4;             // FP4 output, [M, N/8] (8 FP4 packed per uint32)
    uint32_t* sf_out;            // FP8 scale-factor output, swizzled layout
    float const* sf_scale;       // Scalar global activation scale (calibrated module.input_scale)
    int M;                       // Total rows = B * seq_len_per_batch
    int N;                       // Hidden dimension (5120 for Wan 2.2)
    int seq_len_per_batch;       // Used only if has_modulation; batch_idx = row / seq_len_per_batch
    bool has_ln_affine;          // Dispatch flag: norm2 -> true, norm1/norm3 -> false
    bool has_modulation;         // Dispatch flag: norm1/norm3 -> true, norm2 -> false
    float eps;                   // LayerNorm epsilon
    cudaStream_t stream;
};

// Launch the fused LayerNorm + FP4 quantization kernel.
template <typename T>
void invokeFusedLayerNormQuant(FusedLayerNormQuantParams<T> const& params, int multiProcessorCount);

// Explicit instantiations (defined in the .cu file).
extern template void invokeFusedLayerNormQuant<half>(
    FusedLayerNormQuantParams<half> const& params, int multiProcessorCount);

#ifdef ENABLE_BF16
extern template void invokeFusedLayerNormQuant<__nv_bfloat16>(
    FusedLayerNormQuantParams<__nv_bfloat16> const& params, int multiProcessorCount);
#endif

} // namespace kernels

TRTLLM_NAMESPACE_END
