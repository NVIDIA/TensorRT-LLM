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

#ifndef TRTLLM_FUSEDDITLAYERNORMSHIFTSCALEKERNEL_H
#define TRTLLM_FUSEDDITLAYERNORMSHIFTSCALEKERNEL_H

#include "tensorrt_llm/common/config.h"
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

// Parameters for the fused DiT LayerNorm + optional AdaLN/affine + optional NVFP4 quant kernel.
//
// Supported modes (compile-time template flags):
//   HAS_LN_AFFINE  : LayerNorm with learned weight/bias (norm2 in WanBlock)
//   HAS_MODULATION : LayerNorm + AdaLN: y = (1 + scale_msa) * x_hat + shift_msa (norm1/norm3)
//   HAS_QUANT      : write NVFP4 packed output + swizzled scale factors
//
// HAS_LN_AFFINE and HAS_MODULATION are mutually exclusive.
//
// Input shape:  x [M, D]
// Modulator shapes: scale_msa/shift_msa [B, D] where B = M / seq_len_per_batch
//                   ln_weight/ln_bias   [D]
// Output shapes:
//   HAS_QUANT=false: out_bf16 [M, D]
//   HAS_QUANT=true:  out_fp4  [M, D/8]  (8 FP4 nibbles packed per uint32)
//                    out_sf   swizzled NVFP4 scale factors (uint8 array)
struct DiTLayerNormShiftScaleParams
{
    // Input
    __nv_bfloat16 const* x = nullptr; // [M, D] bf16

    // Affine LN params (HAS_LN_AFFINE only)
    __nv_bfloat16 const* ln_weight = nullptr; // [D] bf16
    __nv_bfloat16 const* ln_bias = nullptr;   // [D] bf16

    // AdaLN modulation params (HAS_MODULATION only)
    __nv_bfloat16 const* scale_msa = nullptr; // [B, D] bf16
    __nv_bfloat16 const* shift_msa = nullptr; // [B, D] bf16

    // bf16 output path (HAS_QUANT=false)
    __nv_bfloat16* out_bf16 = nullptr; // [M, D] bf16

    // FP4 output path (HAS_QUANT=true)
    uint32_t* out_fp4 = nullptr;     // [M, D/8] packed FP4 (8 nibbles per uint32)
    uint32_t* out_sf = nullptr;      // swizzled NVFP4 scale factors (uint8 recast as uint32*)
    float const* sf_scale = nullptr; // scalar global activation scale (calibrated input_scale)

    // Shape
    int M = 0;                 // total rows
    int D = 0;                 // hidden dim (must equal 5120 for now)
    int seq_len_per_batch = 0; // HAS_MODULATION: batch_idx = row / seq_len_per_batch
    float eps = 1e-6f;
};

// Launch the fused kernel. The bool flags select the matching compile-time specialization.
// has_ln_affine and has_modulation must not both be true. Supported hidden_dim: 5120 (Wan 14B).
void launchFusedDiTLayerNormShiftScaleKernel(DiTLayerNormShiftScaleParams const& params, bool has_ln_affine,
    bool has_modulation, bool has_quant, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END

#endif // TRTLLM_FUSEDDITLAYERNORMSHIFTSCALEKERNEL_H
