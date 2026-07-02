/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "tensorrt_llm/common/config.h"

#include <cstdint>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

// Fused inverse-RoPE + 1x128 block-scaled FP8 quant for DeepSeek-V4
// absorption-mode attention output. Reads bf16 attention output
// [num_tokens, num_heads, head_dim], applies NEOX inverse rotary embedding
// to the rope segment (the LAST 64 elements of each head, i.e. the second
// half of the final quant chunk), then per-128-element-chunk
// absmax-quantizes the result to FP8 e4m3.
//
// Layout constraints (asserted by caller; the kernel hardcodes the matching
// constants):
//   * quant_group_size == 128
//   * rope_dim         == 64  (half_rope == 32, NEOX)
//   * head_dim         == chunks_per_head * 128, with chunks_per_head in
//                         {1, 2, 3, 4}. Production DSv4-{Flash,Pro} use
//                         head_dim = 512 (kv_lora_rank=448 nope + 64 rope).
//   * Per head: scales emitted for every quant chunk (so total scale slots
//     per token = num_heads * chunks_per_head).
//
// Hardware: targets SM89+ (uses cvt.rn.satfinite.e4m3x2.f32, __shfl_sync,
// bf16 intrinsics).  No TMA / cluster / tcgen05 dependencies.
//
// Layouts:
//   o            : bf16 [num_tokens, num_heads, head_dim]
//   positions    : int64 [num_tokens]
//   cos_sin_cache: fp32 [max_positions, 2, 32] (cos block then sin block, NEOX)
//   fp8_out      : fp8  [..., num_tokens, num_heads * head_dim]
//   scale_out    : fp32 with stride scale_stride_k between adjacent quant
//                  blocks (qb index = head_idx * chunks_per_head + chunk),
//                  layout [..., num_heads * chunks_per_head, scale_buf_m]
//                  where scale_buf_m = pad_up(num_tokens, 4) per the BMM
//                  dequant consumer's hard-coded m-dim stride.
void invokeInverseRopeFp8Quant(void const* o, //
    void const* positions,                    //
    void const* cos_sin_cache,                //
    void* fp8_out,                            //
    void* scale_out,                          //
    int num_tokens,                           //
    int num_heads,                            //
    int heads_per_group,                      //
    int chunks_per_head,                      //
    bool is_neox,                             //
    int scale_buf_m,                          //
    int o_stride_token,                       //
    int o_stride_head,                        //
    int fp8_stride_group,                     //
    int fp8_stride_token,                     //
    int scale_stride_group,                   //
    int scale_stride_k,                       //
    cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
