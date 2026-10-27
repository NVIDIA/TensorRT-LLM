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

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM31
{

//! The (value, index) partial of one CTA's slice of a row (the argmax reduction's carry).
struct MarkovEpilogueBest
{
    float value;
    int32_t index;
};

//! S4FE: one position of the in-tree DSpark Markov chain's elementwise tail, fused
//! (models/modeling_speculative.py dspark_markov_chain with cast_bias_to_logits, full-vocab argmax):
//!
//!   out[b, :]  = base[b, :] + float(bias[b, :])              (fp32 add.rn; the bf16 -> fp32 cast is exact)
//!   prev[b]    = torch.argmax(out[b, :])                      (NaN wins; ties and NaNs take the lowest index)
//!   embOut[b]  = w1[prev[b], :]                               (the next position's markov_prev_embeddings; prev is
//!                                                              always in vocab, so the mask is a no-op)
//!
//! Bitwise contract vs the in-tree path: each output element is the same single fp32 rounding as torch's add of the
//! cast bias, and the argmax uses torch's total order (GreaterOrNan: NaN beats any number, equal values or two NaNs
//! keep the lower index), so any split of a row gives torch's index. bias is the in-tree F.linear's own bf16 output.
//!
//! Layout: base and out are rows of d fp32 with row strides baseRowStride / outRowStride elements (a [B, K, d] tensor's
//! position k: stride K * d); bias is a contiguous [rows, d] bf16 tensor; d % 4 == 0 and every row 16-byte aligned
//! (base / out) or 8-byte aligned (bias). w1 is [vocab, rank] bf16 contiguous (rank <= 1024); embOut is [rows, rank]
//! or null. partials holds rows * splits entries; counters holds rows int32 that are zero before the first launch and
//! are left zero by every launch (the last CTA of a row resets its counter): CUDA-graph safe, stream-ordered reuse.
//! Grid (splits, rows), 256 threads; no co-residency requirement.
void invokeMarkovEpilogue(float const* base, int64_t baseRowStride, __nv_bfloat16 const* bias, float* out,
    int64_t outRowStride, int64_t* prev, __nv_bfloat16 const* w1, int32_t rank, __nv_bfloat16* embOut,
    MarkovEpilogueBest* partials, int32_t* counters, int32_t rows, int32_t d, int32_t splits, cudaStream_t stream);

//! B1: identical arithmetic with a BF16 base widened by payload bit shift.
void invokeMarkovEpilogue(__nv_bfloat16 const* base, int64_t baseRowStride, __nv_bfloat16 const* bias, float* out,
    int64_t outRowStride, int64_t* prev, __nv_bfloat16 const* w1, int32_t rank, __nv_bfloat16* embOut,
    MarkovEpilogueBest* partials, int32_t* counters, int32_t rows, int32_t d, int32_t splits, cudaStream_t stream);

//! The default split count: about two CTAs per SM over the rows (at least 1, at most one CTA per 1024 float4s).
int32_t markovEpilogueSplits(int32_t rows, int32_t d, int32_t smCount);

} // namespace minimaxM31
} // namespace kernels

TRTLLM_NAMESPACE_END
