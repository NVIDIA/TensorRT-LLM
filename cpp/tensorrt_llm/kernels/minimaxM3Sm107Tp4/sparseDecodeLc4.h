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
namespace minimaxM3Sm107Tp4
{

//! SM107 TP4 copy of kernels/minimaxM3/sparseDecodeLc (the sm_103 targets keep that one), with the split-K merge
//! weights formed once per CTA. Operands of the MiniMax-M3 one-launch long-context sparse decode. Strides are in
//! elements of the pointed-to type. Geometry is fixed: 16 query heads on one KV head, head_dim 128, page 128, top-k 16.
struct MinimaxM3SparseDecodeLcParams
{
    uint8_t const* q; // e4m3 [T, 16, 128]
    int64_t q_tok, q_head;
    uint8_t const* k; // packed e2m1 [pages, 1, 128, 64]
    uint8_t const* v;
    int64_t k_blk, v_blk;
    uint8_t const* ksc; // e4m3 [pages, 1, 128, 8], token-major
    uint8_t const* vsc; // e4m3, 4x4 token-quad order
    int64_t ksc_blk, vsc_blk;
    float const* kgs;
    float const* vgs;
    int const* topk; // [T, 16] view (kv head 0)
    int64_t t_tok, t_k;
    int const* block_table;
    int64_t bt_stride;
    int const* seq_lens;
    int dql;
    float sm_scale;
    __nv_bfloat16* out; // [T, 2048]
    int64_t out_stride;
    uint8_t* q8;        // e4m3 [rows, num_sf_k_tiles * 128]
    uint8_t* sf;
    int num_sf_k_tiles;
    // Optional L2 prefetch of two read-only ranges (the layer's o_proj weight and scales), split across the grid's
    // CTAs and issued before the grid dependency: a hint, so the output does not depend on it. nullptr = none.
    uint8_t const* pf0;
    int64_t pf0_bytes;
    uint8_t const* pf1;
    int64_t pf1_bytes;
    // Optional next-layer KV prefetch: CTA (token, r) records its selected block id in this layer's [rows, 16] id
    // buffer (n1_self) and pulls into L2 the K/V pages and scales of the block the next sparse layer's CTA (token, r)
    // selected at the previous step (n1_next, from that layer's pools n1_k / n1_v / n1_ksc / n1_vsc, this layer's
    // strides). Hints only. nullptr = off.
    int* n1_self;
    int const* n1_next;
    uint8_t const* n1_k;
    uint8_t const* n1_v;
    uint8_t const* n1_ksc;
    uint8_t const* n1_vsc;
    int64_t n1_pages; // pages in the next layer's pool; a recorded block outside it is skipped
    // Identity selection (short-sequence steps, every request within 16 blocks): CTA r attends block r of its
    // token's request when r < min(16, blocks), from the device sequence length; topk is not read.
    int identity;
    // The step is an identity step (short sequences: no scorer / select ran this step), whether or not this kernel
    // derives its blocks from the lengths (identity) or reads the staged table: q is then read after the grid wait,
    // because the predecessor is the producer, which releases its dependents at entry, before it writes q.
    int identity_step;
    // One-division merge (MSA_ONE_DIV_MERGE): the CTAs push unnormalized partials with their row max and sum, and the
    // merge divides once per output. Deterministic, a different rounding sequence from the per-chunk normalization.
    int one_div;
    // BT_PREWAIT: stage the request's block-table row (up to 4096 entries) in shared memory before the grid
    // dependency, so the post-wait page lookup reads shared memory. Same output.
    int bt_prewait;
};

//! One cluster of 16 CTAs per query token: CTA r attends the token's r-th selected block, then the cluster merges
//! the 16 partials over DSMEM into the BF16 row (and, when mxfp8, its MXFP8 copy with swizzled UE8M0 scales).
//!
//! Instantiated variants: linearSoftmax in {false, true} x mxfp8 in {false, true}, with earlyTrigger (release PDL
//! dependents at entry). Returns cudaErrorNotSupported for any other variant. The device code is SM100-family only;
//! on other architectures the kernel compiles to an empty body, so callers check the device first.
cudaError_t invokeSparseDecodeLc4(MinimaxM3SparseDecodeLcParams const& params, int32_t numTokens, bool linearSoftmax,
    bool mxfp8, bool earlyTrigger, bool usePdl, cudaStream_t stream);

} // namespace minimaxM3Sm107Tp4
} // namespace kernels

TRTLLM_NAMESPACE_END
