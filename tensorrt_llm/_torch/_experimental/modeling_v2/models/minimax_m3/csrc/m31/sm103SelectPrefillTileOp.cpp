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

// MiniMax-M3.1 GB300 (sm_103) target-local copy of the #20629 prefill tile select
// (cpp/tensorrt_llm/kernels/minimaxM3/selectPrefillTile.*); namespace and op name are M3.1's.

#include "minimax_m3/csrc/m31/selectPrefillTileSm103.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include "tensorrt_llm/modeling_v2/common/torchLibrary.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace torch_ext
{

//! Prefill top-16 block select over per-KV-head scores with unit query stride, into a caller-owned table.
//!
//! @param scores Float32 CUDA tensor [num_kv_heads, num_blocks, total_q] with stride(2) == 1.
//! @param nValidBlocks Contiguous int32 CUDA tensor with at least total_q elements.
//! @param out Int32 CUDA tensor of logical shape [total_q, num_kv_heads, 16] with unit stride in the last dim (a
//! head-major backing is a permuted view). Every entry is written.
void minimaxM31SelectPrefillTile(torch::Tensor const& scores, torch::Tensor const& nValidBlocks, torch::Tensor& out,
    int64_t initBlocks, int64_t localBlocks)
{
    TORCH_CHECK(scores.is_cuda() && scores.scalar_type() == torch::kFloat32 && scores.dim() == 3,
        "minimax_m31_select_prefill_tile: scores must be a 3-D float32 CUDA tensor [num_kv_heads, num_blocks, "
        "total_q]");
    TORCH_CHECK(scores.stride(2) == 1, "minimax_m31_select_prefill_tile: scores must have unit query stride");
    TORCH_CHECK(nValidBlocks.is_cuda() && nValidBlocks.scalar_type() == torch::kInt32 && nValidBlocks.is_contiguous()
            && nValidBlocks.numel() >= scores.size(2),
        "minimax_m31_select_prefill_tile: n_valid_blocks must be contiguous int32 CUDA with at least total_q elements");
    int64_t const numKvHeads = scores.size(0);
    int64_t const numBlocks = scores.size(1);
    int64_t const totalQ = scores.size(2);
    TORCH_CHECK(numBlocks <= TRTLLM_MV2_NS::kernels::minimaxM31Sm103::kMinimaxM3SelectPrefillTileMaxBlocks,
        "minimax_m31_select_prefill_tile supports at most ",
        TRTLLM_MV2_NS::kernels::minimaxM31Sm103::kMinimaxM3SelectPrefillTileMaxBlocks, " blocks, got ", numBlocks);
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kInt32 && out.dim() == 3 && out.size(0) == totalQ
            && out.size(1) == numKvHeads && out.size(2) == 16 && out.stride(2) == 1,
        "minimax_m31_select_prefill_tile: out must be int32 [total_q, num_kv_heads, 16] with unit last-dim stride");
    TORCH_CHECK(nValidBlocks.device() == scores.device() && out.device() == scores.device(),
        "minimax_m31_select_prefill_tile: scores, n_valid_blocks and out must be on the same CUDA device");

    c10::cuda::CUDAGuard const deviceGuard{scores.device()};
    auto const stream = at::cuda::getCurrentCUDAStream(scores.get_device());
    cudaError_t const err
        = TRTLLM_MV2_NS::kernels::minimaxM31Sm103::invokeMinimaxM3SelectPrefillTile(scores.data_ptr<float>(),
            scores.stride(0), scores.stride(1), nValidBlocks.data_ptr<int32_t>(), out.data_ptr<int32_t>(),
            out.stride(0), out.stride(1), static_cast<int32_t>(numKvHeads), static_cast<int32_t>(numBlocks),
            static_cast<int32_t>(totalQ), static_cast<int32_t>(initBlocks), static_cast<int32_t>(localBlocks), stream);
    if (err != cudaSuccess)
    {
        cudaGetLastError();
        TORCH_CHECK(false, "minimax_m31_select_prefill_tile launch failed: ", cudaGetErrorString(err));
    }
}

//! DEP2CTX_TSEL_ROWS: the same select over row-major scores (unit block stride), into a caller-owned table.
//!
//! @param scores Float32 CUDA tensor [num_kv_heads, num_blocks, total_q] with stride(1) == 1 (blocks contiguous per
//! (head, query) row), e.g. the Q8KV8 prefill scorer's [total_q * num_kv_heads, pitch] buffer viewed as
//! view(total_q, num_kv_heads, pages).permute(1, 2, 0).
//! @param nValidBlocks Contiguous int32 CUDA tensor with at least total_q elements.
//! @param out As minimax_m31_select_prefill_tile. The table equals that op's on the same values bit for bit.
void minimaxM31SelectPrefillTileRows(torch::Tensor const& scores, torch::Tensor const& nValidBlocks, torch::Tensor& out,
    int64_t initBlocks, int64_t localBlocks)
{
    TORCH_CHECK(scores.is_cuda() && scores.scalar_type() == torch::kFloat32 && scores.dim() == 3,
        "minimax_m31_select_prefill_tile_rows: scores must be a 3-D float32 CUDA tensor [num_kv_heads, num_blocks, "
        "total_q]");
    TORCH_CHECK(scores.stride(1) == 1 && scores.stride(0) >= 0 && scores.stride(2) >= 0,
        "minimax_m31_select_prefill_tile_rows: scores must have unit block stride");
    TORCH_CHECK(nValidBlocks.is_cuda() && nValidBlocks.scalar_type() == torch::kInt32 && nValidBlocks.is_contiguous()
            && nValidBlocks.numel() >= scores.size(2),
        "minimax_m31_select_prefill_tile_rows: n_valid_blocks must be contiguous int32 CUDA with at least total_q "
        "elements");
    int64_t const numKvHeads = scores.size(0);
    int64_t const numBlocks = scores.size(1);
    int64_t const totalQ = scores.size(2);
    TORCH_CHECK(numBlocks <= TRTLLM_MV2_NS::kernels::minimaxM31Sm103::kMinimaxM3SelectPrefillTileMaxBlocks,
        "minimax_m31_select_prefill_tile_rows supports at most ",
        TRTLLM_MV2_NS::kernels::minimaxM31Sm103::kMinimaxM3SelectPrefillTileMaxBlocks, " blocks, got ", numBlocks);
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kInt32 && out.dim() == 3 && out.size(0) == totalQ
            && out.size(1) == numKvHeads && out.size(2) == 16 && out.stride(2) == 1,
        "minimax_m31_select_prefill_tile_rows: out must be int32 [total_q, num_kv_heads, 16] with unit last-dim "
        "stride");
    TORCH_CHECK(nValidBlocks.device() == scores.device() && out.device() == scores.device(),
        "minimax_m31_select_prefill_tile_rows: scores, n_valid_blocks and out must be on the same CUDA device");

    c10::cuda::CUDAGuard const deviceGuard{scores.device()};
    auto const stream = at::cuda::getCurrentCUDAStream(scores.get_device());
    cudaError_t const err
        = TRTLLM_MV2_NS::kernels::minimaxM31Sm103::invokeMinimaxM3SelectPrefillTileRows(scores.data_ptr<float>(),
            scores.stride(0), scores.stride(2), nValidBlocks.data_ptr<int32_t>(), out.data_ptr<int32_t>(),
            out.stride(0), out.stride(1), static_cast<int32_t>(numKvHeads), static_cast<int32_t>(numBlocks),
            static_cast<int32_t>(totalQ), static_cast<int32_t>(initBlocks), static_cast<int32_t>(localBlocks), stream);
    if (err != cudaSuccess)
    {
        cudaGetLastError();
        TORCH_CHECK(false, "minimax_m31_select_prefill_tile_rows launch failed: ", cudaGetErrorString(err));
    }
}

} // namespace torch_ext

TRTLLM_MV2_NAMESPACE_END

TRTLLM_MV2_LIBRARY_FRAGMENT(m)
{
    m.def(
        "minimax_m31_select_prefill_tile(Tensor scores, Tensor n_valid_blocks, Tensor(a!) out, int init_blocks, "
        "int local_blocks) -> ()");
    m.def(
        "minimax_m31_select_prefill_tile_rows(Tensor scores, Tensor n_valid_blocks, Tensor(a!) out, int init_blocks, "
        "int local_blocks) -> ()");
}

TRTLLM_MV2_LIBRARY_IMPL(CUDA, m)
{
    m.impl("minimax_m31_select_prefill_tile", &TRTLLM_MV2_NS::torch_ext::minimaxM31SelectPrefillTile);
    m.impl("minimax_m31_select_prefill_tile_rows", &TRTLLM_MV2_NS::torch_ext::minimaxM31SelectPrefillTileRows);
}
