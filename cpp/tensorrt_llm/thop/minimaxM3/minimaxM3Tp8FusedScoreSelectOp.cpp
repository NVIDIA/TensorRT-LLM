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

#include "tensorrt_llm/kernels/minimaxM3Tp8/fusedScoreSelect.h"

#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

constexpr int64_t kHeadDim = 128;
constexpr int64_t kBlockK = 128;
constexpr int64_t kTopK = 16;

} // namespace

//! MiniMax-M3 TP8 fused index score + top-16 block select for decode spans of at most 2 requests. Equals
//! minimax_m3_select_blocks applied to the block scores of cute_dsl_minimax_m3_index_decode_score. Grid (splits,
//! batch) with splits = max(SM count / batch, ceil(max_tiles / 128), 1).
//!
//! @param idxQ E4M3 [total_q, heads, 128] (total_q = batch * dql, heads * 4 <= 8), unit last-dim stride, row and head
//! strides multiples of 4 bytes, 4-byte aligned.
//! @param idxKPages E4M3 [num_pages, 128, 128], unit last-dim stride, page and row strides multiples of 16 bytes,
//! 16-byte aligned.
//! @param blockTable Int32 [batch, >= blocks], unit last-dim stride.
//! @param seqLens Int32 [batch] (1 or 2 requests), including this step's dql query rows.
//! @param nValid Contiguous int32 [total_q] per-token causal valid-block counts.
//! @param out Int32, logical [total_q, num_kv_heads, 16] with unit last-dim stride (a permuted head-major backing is
//! fine). Every entry is written: ascending block ids with a -1 tail.
//! @param scratch Contiguous float32 [>= batch, >= num_kv_heads * dql, L] with L >= max_tiles and L % 4 == 0, 16-byte
//! aligned; [batch, 8, ceil4(max_tiles)] covers every supported geometry. Contents need no initialization.
//! @param counters Contiguous int32 with >= batch elements, all zero before the first call. The kernel returns each
//! request's counter to zero before it exits, so the buffer stays reusable across calls and graph replays without a
//! reset (as long as launches sharing it are stream-ordered).
//! @param maxTiles Worst-case block count n_valid is clamped to; num_kv_heads * dql * ceil4(max_tiles) * 4 +
//! num_kv_heads * dql * 1024 must not exceed 196608 bytes.
void minimaxM3Tp8FusedScoreSelect(torch::Tensor const& idxQ, torch::Tensor const& idxKPages,
    torch::Tensor const& blockTable, torch::Tensor const& seqLens, torch::Tensor const& nValid, torch::Tensor& out,
    torch::Tensor& scratch, torch::Tensor& counters, int64_t decodeQueryLen, int64_t numKvHeads, int64_t initBlocks,
    int64_t localBlocks, int64_t maxTiles, bool usePdl)
{
    namespace tk = tensorrt_llm::kernels;
    TORCH_CHECK(idxQ.is_cuda(), "minimax_m3_tp8_fused_score_select: idx_q must be a CUDA tensor");
    c10::cuda::CUDAGuard const deviceGuard{idxQ.device()};
    auto const* props = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(props->major >= 10, "minimax_m3_tp8_fused_score_select requires an SM100-family GPU");

    TORCH_CHECK(idxQ.dim() == 3 && idxQ.size(2) == kHeadDim && idxQ.stride(2) == 1 && idxQ.element_size() == 1,
        "minimax_m3_tp8_fused_score_select: idx_q must be one-byte [total_q, heads, 128] with unit last-dim stride");
    TORCH_CHECK(
        idxQ.stride(0) % 4 == 0 && idxQ.stride(1) % 4 == 0 && (reinterpret_cast<uintptr_t>(idxQ.data_ptr()) & 3) == 0,
        "minimax_m3_tp8_fused_score_select: idx_q strides and base must be 4-byte aligned");
    TORCH_CHECK(idxKPages.is_cuda() && idxKPages.element_size() == 1 && idxKPages.dim() == 3
            && idxKPages.size(1) == kBlockK && idxKPages.size(2) == kHeadDim && idxKPages.stride(2) == 1,
        "minimax_m3_tp8_fused_score_select: idx_k_pages must be one-byte [num_pages, 128, 128] with unit last-dim "
        "stride");
    TORCH_CHECK(idxKPages.stride(1) % 16 == 0 && idxKPages.stride(0) % 16 == 0
            && (reinterpret_cast<uintptr_t>(idxKPages.data_ptr()) & 15) == 0,
        "minimax_m3_tp8_fused_score_select: idx_k_pages strides and base must be 16-byte aligned");
    TORCH_CHECK(blockTable.scalar_type() == torch::kInt32 && blockTable.dim() == 2 && blockTable.stride(1) == 1,
        "minimax_m3_tp8_fused_score_select: block_table must be int32 [batch, blocks] with unit last-dim stride");
    TORCH_CHECK(
        seqLens.scalar_type() == torch::kInt32 && nValid.scalar_type() == torch::kInt32 && nValid.is_contiguous(),
        "minimax_m3_tp8_fused_score_select: seq_lens and n_valid must be int32, n_valid contiguous");
    TORCH_CHECK(out.scalar_type() == torch::kInt32 && out.dim() == 3 && out.size(2) == kTopK && out.stride(2) == 1,
        "minimax_m3_tp8_fused_score_select: out must be int32 [total_q, num_kv_heads, 16] with unit last-dim stride");

    int64_t const batch = seqLens.size(0);
    int64_t const totalQ = idxQ.size(0);
    int64_t const heads = idxQ.size(1);
    int64_t const rows = decodeQueryLen * numKvHeads;
    TORCH_CHECK(batch >= 1 && batch <= tk::kMinimaxM3Tp8FusedScoreSelectMaxBatch,
        "minimax_m3_tp8_fused_score_select takes 1 to ", tk::kMinimaxM3Tp8FusedScoreSelectMaxBatch, " requests, got ",
        batch);
    TORCH_CHECK(decodeQueryLen >= 1 && decodeQueryLen <= tk::kMinimaxM3Tp8FusedScoreSelectMaxDql
            && totalQ == batch * decodeQueryLen,
        "minimax_m3_tp8_fused_score_select: decode_query_len must be in [1, 4] with total_q = batch * "
        "decode_query_len");
    TORCH_CHECK(numKvHeads >= 1 && heads % numKvHeads == 0 && heads * tk::kMinimaxM3Tp8FusedScoreSelectMaxDql <= 8
            && rows <= tk::kMinimaxM3Tp8FusedScoreSelectMaxRows,
        "minimax_m3_tp8_fused_score_select: unsupported head geometry");
    TORCH_CHECK(out.size(0) == totalQ && out.size(1) == numKvHeads,
        "minimax_m3_tp8_fused_score_select: out must be [total_q, num_kv_heads, 16]");
    TORCH_CHECK(nValid.numel() >= totalQ, "minimax_m3_tp8_fused_score_select: n_valid needs total_q elements");
    TORCH_CHECK(maxTiles >= 1, "minimax_m3_tp8_fused_score_select: max_tiles must be positive");

    TORCH_CHECK(scratch.is_cuda() && scratch.scalar_type() == torch::kFloat32 && scratch.dim() == 3
            && scratch.is_contiguous() && scratch.size(0) >= batch && scratch.size(1) >= rows
            && scratch.size(2) >= maxTiles && scratch.stride(1) % 4 == 0
            && (reinterpret_cast<uintptr_t>(scratch.data_ptr()) & 15) == 0,
        "minimax_m3_tp8_fused_score_select: scratch must be contiguous 16-byte aligned float32 [>= batch, >= rows, "
        "L] with L >= max_tiles and L % 4 == 0");
    TORCH_CHECK(counters.is_cuda() && counters.scalar_type() == torch::kInt32 && counters.is_contiguous()
            && counters.numel() >= batch,
        "minimax_m3_tp8_fused_score_select: counters must be contiguous int32 with >= batch elements");
    TORCH_CHECK(rows * ((maxTiles + 3) & ~int64_t{3}) * 4 + rows * 1024 <= tk::kMinimaxM3Tp8FusedScoreSelectRingBytes,
        "minimax_m3_tp8_fused_score_select: select scratch does not fit the stage ring");

    int64_t const splits = std::max<int64_t>({props->multiProcessorCount / batch,
        (maxTiles + tk::kMinimaxM3Tp8FusedScoreSelectMaxLocalBlocks - 1)
            / tk::kMinimaxM3Tp8FusedScoreSelectMaxLocalBlocks,
        int64_t{1}});
    TORCH_CHECK((maxTiles + splits - 1) / splits <= tk::kMinimaxM3Tp8FusedScoreSelectMaxLocalBlocks,
        "minimax_m3_tp8_fused_score_select: too few splits");

    tk::MinimaxM3Tp8FusedScoreSelectParams p{};
    p.q = reinterpret_cast<uint8_t const*>(idxQ.data_ptr());
    p.q_st = idxQ.stride(0);
    p.q_sh = idxQ.stride(1);
    p.bt = blockTable.data_ptr<int32_t>();
    p.bt_stride = blockTable.stride(0);
    p.bt_width = static_cast<int>(blockTable.size(1));
    p.seq_lens = seqLens.data_ptr<int32_t>();
    p.n_valid = nValid.data_ptr<int32_t>();
    p.scr = scratch.data_ptr<float>();
    p.scr_b = scratch.stride(0);
    p.scr_r = scratch.stride(1);
    p.out = out.data_ptr<int32_t>();
    p.out_t = out.stride(0);
    p.out_h = out.stride(1);
    p.counters = counters.data_ptr<int32_t>();
    p.heads = static_cast<int>(heads);
    p.kv_heads = static_cast<int>(numKvHeads);
    p.dql = static_cast<int>(decodeQueryLen);
    p.max_tiles = static_cast<int>(maxTiles);
    p.init_blocks = static_cast<int>(initBlocks);
    p.local_blocks = static_cast<int>(localBlocks);

    auto const stream = at::cuda::getCurrentCUDAStream(idxQ.get_device());
    cudaError_t const err
        = tk::invokeMinimaxM3Tp8FusedScoreSelect(p, idxKPages.data_ptr(), idxKPages.size(0), idxKPages.stride(1),
            idxKPages.stride(0), static_cast<int32_t>(batch), static_cast<int32_t>(splits), usePdl, stream);
    TORCH_CHECK(err == cudaSuccess, "minimax_m3_tp8_fused_score_select launch failed: ", cudaGetErrorString(err));
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "minimax_m3_tp8_fused_score_select(Tensor idx_q, Tensor idx_k_pages, Tensor block_table, Tensor seq_lens, "
        "Tensor n_valid, Tensor(a!) out, Tensor(b!) scratch, Tensor(c!) counters, int decode_query_len, "
        "int num_kv_heads, int init_blocks, int local_blocks, int max_tiles, bool use_pdl) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("minimax_m3_tp8_fused_score_select", &tensorrt_llm::torch_ext::minimaxM3Tp8FusedScoreSelect);
}
