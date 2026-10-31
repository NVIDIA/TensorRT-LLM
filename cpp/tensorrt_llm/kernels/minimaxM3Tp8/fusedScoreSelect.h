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
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

//! Largest number of requests (decode spans) one fused score + select launch takes.
constexpr int32_t kMinimaxM3Tp8FusedScoreSelectMaxBatch = 2;
//! Largest decode query length (query rows per request).
constexpr int32_t kMinimaxM3Tp8FusedScoreSelectMaxDql = 4;
//! Largest number of selected rows per request (num_kv_heads * dql).
constexpr int32_t kMinimaxM3Tp8FusedScoreSelectMaxRows = 8;
//! Largest number of blocks one CTA scores (ceil(max_tiles / splits) must not exceed it).
constexpr int32_t kMinimaxM3Tp8FusedScoreSelectMaxLocalBlocks = 128;
//! Bytes of the CTA's TMA stage ring, reused by the select: rows * ceil4(max_tiles) * 4 + rows * 1024 must fit.
constexpr int64_t kMinimaxM3Tp8FusedScoreSelectRingBytes = 12 * 128 * 128;

//! Operands of the MiniMax-M3 TP8 fused index score + top-16 block select. Strides are in elements.
struct MinimaxM3Tp8FusedScoreSelectParams
{
    uint8_t const* q;  // e4m3 [total_q, heads, 128]
    int64_t q_st, q_sh;
    int32_t const* bt; // [batch, bt_width]
    int64_t bt_stride;
    int bt_width;
    int32_t const* seq_lens; // [batch], including this step's dql query rows
    int32_t const* n_valid;  // [total_q] causal valid-block counts
    float* scr;              // [batch][rows][>= ceil4(max_tiles)] block scores
    int64_t scr_b, scr_r;
    int32_t* out;            // logical [total_q, kv_heads, 16]
    int64_t out_t, out_h;
    int32_t* counters;       // [batch], zero at rest
    int heads, kv_heads, dql, max_tiles, init_blocks, local_blocks;
};

//! One launch of index scoring and top-16 selection for at most 2 decode requests. Grid (splits, batch), one CTA of
//! 8 consumer warps + 1 TMA producer warp per SM: CTA (s, b) TMA-loads the e4m3 [128, 128] index-K pages of blocks
//! s, s + splits, ... of request b (12-stage ring; the pages holding only tokens older than this step are requested
//! before the grid dependency wait), scores them against the request's q rows on FP16 tensor cores, and stores each
//! block's score (max over its causal positions, amax over each KV head's index heads) to the scratch. The last CTA
//! of a request (per-request arrival counter) releases PDL dependents, resets that counter to zero, and selects the
//! top 16 of every (query row, KV head) row exactly like minimax_m3_select_blocks (init / local forcing, ties to the
//! smaller block, -inf picks dropped, ascending ids with a -1 tail).
//!
//! kPages is the e4m3 index-K page pool [numPages, 128, 128] with the given byte strides (row stride and page stride
//! multiples of 16, base 16-byte aligned); its TMA descriptor is encoded on first use and cached per (base, pages,
//! strides). The device code is SM100-family only; on other architectures the kernel compiles to an empty body.
cudaError_t invokeMinimaxM3Tp8FusedScoreSelect(MinimaxM3Tp8FusedScoreSelectParams const& params, void const* kPages,
    int64_t numPages, int64_t kRowStride, int64_t kPageStride, int32_t batch, int32_t splits, bool usePdl,
    cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
