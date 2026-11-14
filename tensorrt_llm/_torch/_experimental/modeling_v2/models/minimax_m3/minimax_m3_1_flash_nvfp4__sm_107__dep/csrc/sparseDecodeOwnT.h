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

#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include <cstddef>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{

//! Operands of OWN-T, the MiniMax-M3 block-sparse decode over the packed-NVFP4 paged cache (SM107): LC2's operands
//! (MinimaxM3Sm107Tp2SparseDecodeLcParams) in the same layouts, plus the request count. Strides are in elements of the
//! pointed-to type. Geometry is fixed: 8 query rows per request, 16 query heads per KV head, head_dim 128, page 128,
//! top-k 16.
struct MinimaxM3Sm107SparseDecodeOwnTParams
{
    uint8_t const* q; // e4m3 [T, kv_heads * 16, 128]
    int64_t q_tok, q_head;
    uint8_t const* k; // packed e2m1 [pages, kv_heads, 128, 64], low nibble = even dim
    uint8_t const* v;
    int64_t k_blk, v_blk, k_h, v_h;
    uint8_t const* ksc; // e4m3 [pages, kv_heads, 128, 8], token-major
    uint8_t const* vsc; // e4m3, 4x4 token-quad order: (token n, column c) at (n >> 2) * 32 + (n & 3) + 4 c
    int64_t ksc_blk, vsc_blk, ksc_h, vsc_h;
    float const* kgs;
    float const* vgs;
    int const* topk;        // [kv_heads, T, 16]
    int64_t t_h, t_tok, t_k;
    int const* block_table; // [requests, bt_stride]
    int64_t bt_stride;
    int const* seq_lens;    // [requests]
    int dql;                // query rows per request (8)
    float sm_scale;
    __nv_bfloat16* out;     // [T, kv_heads * 16 * 128]
    int64_t out_stride;
    int num_kv_heads;
    int num_requests;
    // The MXFP8 copy of out, or nullptr: E4M3 rows [T, num_sf_k_tiles * 128] and their UE8M0 scales in the 128x4
    // swizzled layout of trtllm::mxfp8_quantize(out, true), bitwise that quantize of the BF16 rows.
    uint8_t* q8;
    uint8_t* sf;
    int num_sf_k_tiles;
};

//! The widest call: (request, KV head) items.
constexpr int32_t kMinimaxM3Sm107SparseDecodeOwnTMaxItems = 256;
//! Query rows per request.
constexpr int32_t kMinimaxM3Sm107SparseDecodeOwnTRows = 8;

//! Workspace bytes for `numItems` (request, KV head) items on a main grid of `grid` CTAs. Grows with both.
size_t minimaxM3Sm107SparseDecodeOwnTWorkspaceBytes(int32_t numItems, int32_t grid);

//! Prepares the current device for invokeMinimaxM3Sm107SparseDecodeOwnT (its shared-memory opt-in) and checks the main
//! kernel's launch registers, which its per-role register budgets assume. cudaErrorInvalidDeviceFunction means this
//! build carries no OWN-T kernel for the device's architecture.
cudaError_t minimaxM3Sm107SparseDecodeOwnTConfigure();

//! OWN-T on `stream`, three launches: the union of the blocks each (request, KV head) item's rows selected; the main
//! grid of `grid` CTAs (the SM count) over equal weighted shares of all items' union blocks, writing an item one CTA
//! covers as BF16 rows and the others as fp32 partials; the merge of the partials (and the zero rows of items without
//! blocks). When params.q8 is set, the merge also writes the MXFP8 copy of every row. The main and merge
//! grids, and the union grid after the previous kernel, launch with PDL when usePdl; values do not depend on it. Call
//! minimaxM3Sm107SparseDecodeOwnTConfigure on the device first.
cudaError_t invokeMinimaxM3Sm107SparseDecodeOwnT(MinimaxM3Sm107SparseDecodeOwnTParams const& params, void* workspace,
    size_t workspaceBytes, int32_t grid, bool usePdl, cudaStream_t stream);

} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
