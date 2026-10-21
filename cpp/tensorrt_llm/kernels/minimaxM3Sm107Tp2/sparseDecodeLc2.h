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

//! Operands of the MiniMax-M3 one-launch long-context sparse decode, SM107 TP2 geometry. Strides are in elements of
//! the pointed-to type. Geometry is fixed: 16 query heads per KV head, head_dim 128, page 128, top-k 16.
struct MinimaxM3Sm107Tp2SparseDecodeLcParams
{
    uint8_t const* q; // e4m3 [T, kv_heads * 16, 128]
    int64_t q_tok, q_head;
    uint8_t const* k; // packed e2m1 [pages, kv_heads, 128, 64]
    uint8_t const* v;
    int64_t k_blk, v_blk, k_h, v_h;
    uint8_t const* ksc; // e4m3 [pages, kv_heads, 128, 8], token-major
    uint8_t const* vsc; // e4m3, 4x4 token-quad order
    int64_t ksc_blk, vsc_blk, ksc_h, vsc_h;
    float const* kgs;
    float const* vgs;
    int const* topk; // [kv_heads, T, 16]
    int64_t t_h, t_tok, t_k;
    int const* block_table;
    int64_t bt_stride;
    int const* seq_lens;
    int dql;
    float sm_scale;
    __nv_bfloat16* out; // [T, kv_heads * 2048]
    int64_t out_stride;
    uint8_t* q8;        // e4m3 [rows, num_sf_k_tiles * 128]
    uint8_t* sf;
    int num_sf_k_tiles;
    int num_kv_heads;
    // N1 (optional): this layer's selected block id per (KV head, query row, slot) at its previous decode step,
    // int32 [kv_heads, n1_rows, 16]. Before the grid dependency each CTA pulls into L2 the K/V page and scales of the
    // block its slot selected last step, then records this step's block. A hint: no value changes.
    int* n1_ids;
    int64_t n1_h;
    int n1_rows;
    int64_t n1_pages;
};

//! One cluster of 16 CTAs per (query token, KV head): CTA r attends the token's r-th selected block of that head,
//! then the cluster merges the 16 partials over DSMEM into the BF16 rows (and, when mxfp8, the MXFP8 copy with
//! swizzled UE8M0 scales). prewait stages the request's block-table row in shared memory before the grid dependency
//! wait (rows longer than 4096 blocks read it from global memory); the arithmetic is the same in both variants.
//! The device code is SM100-family only; on other architectures the kernel compiles to an empty body.
cudaError_t invokeMinimaxM3Sm107Tp2SparseDecodeLc(MinimaxM3Sm107Tp2SparseDecodeLcParams const& params,
    int32_t numTokens, bool linearSoftmax, bool mxfp8, bool prewait, bool usePdl, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
