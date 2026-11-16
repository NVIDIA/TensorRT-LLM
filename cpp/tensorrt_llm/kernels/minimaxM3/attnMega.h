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
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace minimax_m3_attn_mega
{
constexpr int kGrid = 112;           // CTAs of the persistent grid (7 clusters of 16)
constexpr int kMaxTokens = 8;        // rows of each call-parity buffer
constexpr int kQkvN = 2560;          // packed QKV(+index) width
constexpr int kQkvK = 6144;          // hidden size
constexpr int kOpN = 6144;           // o_proj output width
constexpr int kOpK = 2048;           // o_proj input width (16 heads x 128)
constexpr int kOpSfBytes = 16 * 512; // one parity of the MXFP8 hand-off's swizzled scales
} // namespace minimax_m3_attn_mega

//! Kernel parameters of the attention megakernel (the device code's Params).
struct MinimaxM3AttnMegaParams
{
    __nv_bfloat16 const* packed;
    int64_t packed_stride;
    uint8_t* kv_data;
    int64_t d_page, d_plane;
    uint8_t* kv_scale;
    int64_t s_page, s_plane;
    uint8_t* idx_k;
    int64_t i_page, i_tok;
    int64_t num_pages;
    int const* out_cache_loc;
    int const* position_ids;
    int const* block_table;
    int64_t bt_stride;
    int bt_cols;
    int const* seq_lens;
    float const* kv_quant_scale;
    float const* k_gs;
    float const* v_gs;
    __nv_bfloat16 const* q_w;
    __nv_bfloat16 const* k_w;
    __nv_bfloat16 const* ik_w;
    float const* rope;
    float eps;
    float sm_scale;
    __nv_bfloat16* out;
    int64_t out_stride;
    uint8_t* q8;
    uint8_t* sf;
    int num_sf_k_tiles;
    unsigned long long* prof; // optional per-CTA clock64 probes (-DM3FA_PROFILE builds only)
    // Fused QKV projection (FQKV kernels only): out[M, 2560] = mxfp8(x)[M, 6144] @ W^T.
    __nv_bfloat16 const* x; // [M, 6144] layer input (the all-reduce output)
    int ldx;
    int num_tokens;         // M
    uint8_t const* wsf;     // UE8M0 weight scales, 128x4-swizzled
    float* part;            // [2 parities][splits][8 tokens][2560] partial sums, sentinel at rest
    int* counters;          // [0]: call parity, [1]: CTAs done with this call (zero at rest)
    int attn_clusters;      // leading clusters that run the attention
    uint8_t const* xq;      // mega: the layer input's MXFP8 values [M, 6144] (row stride ldxq) ...
    uint8_t const* xsf_act; // ... and its 128x4-swizzled UE8M0 scale atoms (atom k at byte 512 k)
    int ldxq;
    __nv_bfloat16* qkv_out; // mega: the packed QKV rows [2 parities][8][2560] the attention reads
    uint8_t const* wsf_o;   // mega: the o_proj weight's scale atoms
    __nv_bfloat16* y_out;   // mega: the o_proj output [M, 6144]
    int* ar_ws;             // mega, ar_mode: the next m3_mc_allreduce's multicast exchange (int32 words)...
    int const* ar_flags;    // ... its call flags (parity of a (token, column slice) slot = flags[t * 8 + slice] & 1)
    int ar_rank;
    int ar_world;
    int ar_maxt;
    int ar_mode;         // 0: none; 2: push the o_proj rows into the exchange (as m3_mx_gemv's ar mode 2)
    unsigned int* flags; // mega: [0, 20) tile counters (4 per call), [32 + CTA] the attention CTAs' targets
    long long pf_a0, pf_b0, pf_a1, pf_b1; // mega: L2 prefetch regions issued in the attention window (bytes 0: none)
    int pf_hint;                          // 1: evict_last policy on the prefetch
    CUtensorMap wmap;                     // the [2560, 6144] E4M3 weight, 128 x 128 boxes, 128-byte swizzle
    CUtensorMap wmap_o;                   // mega: the [6144, 2048] E4M3 o_proj weight, same boxes
};

//! Launch the megakernel on `stream`. The caller fills `params` except `wmap`, `wmap_o`, `packed`,
//! `packed_stride` and `attn_clusters`, which this derives (the TMA descriptors of the two weights, cached per
//! address). Only (qlen, qg, bpc) = (1, 1, 1) with the MXFP8 hand-off is compiled; anything else, or a failed TMA
//! descriptor encode, returns cudaErrorInvalidValue.
cudaError_t invokeMinimaxM3AttnMega(MinimaxM3AttnMegaParams& params, void const* qkvWeight, void const* oWeight,
    int qlen, int qg, int bpc, int numReqs, bool pdl, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
