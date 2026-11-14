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

#include "minimax_m3/csrc/m31/fselTopkSm107.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include "tensorrt_llm/modeling_v2/common/torchLibrary.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace torch_ext
{

//! FSEL top-16 block select for MiniMax-M3 decode rows of up to 9216 blocks, into a caller-owned table.
//!
//! @param scores Float32 CUDA tensor [num_kv_heads, num_blocks, total_q], any strides.
//! @param nValidBlocks Contiguous int32 CUDA tensor with at least total_q elements.
//! @param pageTable Int32 CUDA tensor of block ids: 1-D with at least num_blocks entries (shared by every row; the M3
//! target passes the identity table) or 2-D [>= total_q, >= num_blocks] with unit last-dim stride.
//! @param out Int32 CUDA tensor of logical shape [total_q, num_kv_heads, 16] with unit stride in the last dim (the
//! head-major backing the target hands over is a permuted view). Every entry is written.
//! @param triggerPhase Where the dependent grid is released: 0 right after this grid's dependency wait (the original
//! kernel's point), 1 once the row's scores are in registers, 2 at this grid's exit.
//! @param variant 6: 512-thread CTAs (the traced instance), 5: 256-thread CTAs, 7: variant 6 with rows above 4096 valid
//! blocks split over a 4-CTA cluster (the same table; variant 6 when num_blocks <= 4096).
void minimaxM3Sm107Tp4FselTopk(torch::Tensor const& scores, torch::Tensor const& nValidBlocks,
    torch::Tensor const& pageTable, torch::Tensor& out, int64_t initBlocks, int64_t localBlocks, bool usePdl,
    int64_t triggerPhase, int64_t variant)
{
    TORCH_CHECK(scores.is_cuda() && scores.scalar_type() == torch::kFloat32 && scores.dim() == 3,
        "minimax_m3_sm107tp4_fsel_topk: scores must be a 3-D float32 CUDA tensor [num_kv_heads, num_blocks, total_q]");
    TORCH_CHECK(nValidBlocks.is_cuda() && nValidBlocks.scalar_type() == torch::kInt32 && nValidBlocks.is_contiguous()
            && nValidBlocks.numel() >= scores.size(2),
        "minimax_m3_sm107tp4_fsel_topk: n_valid_blocks must be contiguous int32 CUDA with at least total_q elements");
    int64_t const numKvHeads = scores.size(0);
    int64_t const numBlocks = scores.size(1);
    int64_t const totalQ = scores.size(2);
    TORCH_CHECK(numBlocks <= TRTLLM_MV2_NS::kernels::minimaxM3::kFselTopkMaxBlocks,
        "minimax_m3_sm107tp4_fsel_topk supports at most ", TRTLLM_MV2_NS::kernels::minimaxM3::kFselTopkMaxBlocks,
        " blocks, got ", numBlocks);
    TORCH_CHECK(numKvHeads > 0 && totalQ > 0, "minimax_m3_sm107tp4_fsel_topk: empty scores");
    TORCH_CHECK(pageTable.is_cuda() && pageTable.scalar_type() == torch::kInt32
            && ((pageTable.dim() == 1 && pageTable.size(0) >= numBlocks && pageTable.stride(0) == 1)
                || (pageTable.dim() == 2 && pageTable.size(0) >= totalQ && pageTable.size(1) >= numBlocks
                    && pageTable.stride(1) == 1)),
        "minimax_m3_sm107tp4_fsel_topk: page_table must be int32 CUDA, 1-D [>= num_blocks] or "
        "2-D [>= total_q, >= num_blocks] with unit last-dim stride");
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kInt32 && out.dim() == 3 && out.size(0) == totalQ
            && out.size(1) == numKvHeads && out.size(2) == 16 && out.stride(2) == 1,
        "minimax_m3_sm107tp4_fsel_topk: out must be int32 [total_q, num_kv_heads, 16] with unit last-dim stride");
    TORCH_CHECK(
        triggerPhase >= 0 && triggerPhase <= 2, "minimax_m3_sm107tp4_fsel_topk: trigger_phase must be 0, 1 or 2");
    TORCH_CHECK(variant == 5 || variant == 6 || variant == TRTLLM_MV2_NS::kernels::minimaxM3::kFselTopkVariantMs4,
        "minimax_m3_sm107tp4_fsel_topk: variant must be 5, 6 or 7");
    TORCH_CHECK(
        initBlocks >= 0 && localBlocks >= 0, "minimax_m3_sm107tp4_fsel_topk: forcing counts must be non-negative");

    c10::cuda::CUDAGuard const deviceGuard{scores.device()};
    auto const stream = at::cuda::getCurrentCUDAStream(scores.get_device());
    int64_t const pageTableRowStride = pageTable.dim() == 1 ? 0 : pageTable.stride(0);
    cudaError_t const err = TRTLLM_MV2_NS::kernels::minimaxM3::invokeFselTopk(scores.data_ptr<float>(),
        scores.stride(0), scores.stride(1), scores.stride(2), nValidBlocks.data_ptr<int32_t>(),
        pageTable.data_ptr<int32_t>(), pageTableRowStride, static_cast<int32_t>(numBlocks), out.data_ptr<int32_t>(),
        out.stride(0), out.stride(1), static_cast<int32_t>(numKvHeads), static_cast<int32_t>(totalQ),
        static_cast<int32_t>(initBlocks), static_cast<int32_t>(localBlocks), static_cast<int32_t>(variant), usePdl,
        static_cast<int32_t>(triggerPhase), nullptr, nullptr, stream);
    if (err != cudaSuccess)
    {
        cudaGetLastError();
        TORCH_CHECK(false, "minimax_m3_sm107tp4_fsel_topk launch failed: ", cudaGetErrorString(err));
    }
}

} // namespace torch_ext

TRTLLM_MV2_NAMESPACE_END

TRTLLM_MV2_LIBRARY_FRAGMENT(m)
{
    m.def(
        "minimax_m3_sm107tp4_fsel_topk(Tensor scores, Tensor n_valid_blocks, Tensor page_table, Tensor(a!) out, int "
        "init_blocks, int local_blocks, bool use_pdl=False, int trigger_phase=0, int variant=6) -> ()");
}

TRTLLM_MV2_LIBRARY_IMPL(CUDA, m)
{
    m.impl("minimax_m3_sm107tp4_fsel_topk", &TRTLLM_MV2_NS::torch_ext::minimaxM3Sm107Tp4FselTopk);
}
