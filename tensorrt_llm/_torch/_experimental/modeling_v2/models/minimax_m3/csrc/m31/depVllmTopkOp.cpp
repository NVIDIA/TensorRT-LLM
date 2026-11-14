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

#include "minimax_m3/csrc/m31/depVllmTopk.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include "tensorrt_llm/modeling_v2/common/torchLibrary.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace torch_ext
{

//! DEP8 decode top-16 select with vLLM's MiniMax-M3.1 ranker, into a caller-owned head-major table.
//!
//! @param scores Float32 CUDA tensor [num_kv_heads, num_blocks, total_q] with unit block stride (the NVFP4 index-K
//! reader's native layout seen as a strided view); at most 9217 blocks and 2047 heads x rows.
//! @param nValidBlocks Contiguous int32 CUDA tensor with at least total_q elements: each query row's candidate blocks,
//! its own block last (shared by every head).
//! @param out Int32 CUDA tensor of logical shape [total_q, num_kv_heads, 16] on a head-major backing
//! (out.permute(1, 0, 2) contiguous). Every entry is written: ascending ids, -1 past a row's candidates.
void minimaxM3Sm107DepVllmTopk(
    torch::Tensor const& scores, torch::Tensor const& nValidBlocks, torch::Tensor& out, bool usePdl)
{
    namespace tk = TRTLLM_MV2_NS::kernels::minimaxM3Sm107Tp8;
    TORCH_CHECK(scores.is_cuda() && scores.scalar_type() == torch::kFloat32 && scores.dim() == 3,
        "minimax_m3_sm107_dep_vllm_topk: scores must be a 3-D float32 CUDA tensor [num_kv_heads, num_blocks, total_q]");
    int64_t const numHeads = scores.size(0);
    int64_t const numBlocks = scores.size(1);
    int64_t const totalQ = scores.size(2);
    TORCH_CHECK(scores.stride(1) == 1, "minimax_m3_sm107_dep_vllm_topk: scores need unit block stride");
    TORCH_CHECK(
        numBlocks >= 1 && numBlocks <= tk::kDepVllmTopkMaxColumns && numHeads * totalQ <= tk::kDepVllmTopkMaxRows,
        "minimax_m3_sm107_dep_vllm_topk: at most ", tk::kDepVllmTopkMaxColumns, " blocks and ", tk::kDepVllmTopkMaxRows,
        " heads x rows, got ", numBlocks, " blocks and ", numHeads * totalQ);
    TORCH_CHECK(
        totalQ <= 1 || scores.stride(2) >= numBlocks, "minimax_m3_sm107_dep_vllm_topk: query rows must not overlap");
    TORCH_CHECK(nValidBlocks.is_cuda() && nValidBlocks.scalar_type() == torch::kInt32 && nValidBlocks.is_contiguous()
            && nValidBlocks.numel() >= totalQ,
        "minimax_m3_sm107_dep_vllm_topk: n_valid_blocks must be contiguous int32 CUDA with at least total_q elements");
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kInt32 && out.dim() == 3 && out.size(0) == totalQ
            && out.size(1) == numHeads && out.size(2) == 16 && out.stride(2) == 1 && out.stride(0) == 16
            && out.stride(1) == totalQ * 16,
        "minimax_m3_sm107_dep_vllm_topk: out must be int32 [total_q, num_kv_heads, 16] on a head-major backing");
    TORCH_CHECK(scores.get_device() == nValidBlocks.get_device() && scores.get_device() == out.get_device(),
        "minimax_m3_sm107_dep_vllm_topk: tensors on one device");

    c10::cuda::CUDAGuard const deviceGuard{scores.device()};
    auto const stream = at::cuda::getCurrentCUDAStream(scores.get_device());
    cudaError_t const err = tk::invokeDepVllmTopk(scores.data_ptr<float>(), scores.stride(0),
        totalQ <= 1 ? numBlocks : scores.stride(2), nValidBlocks.data_ptr<int32_t>(), out.data_ptr<int32_t>(),
        static_cast<int32_t>(numHeads), static_cast<int32_t>(totalQ), static_cast<int32_t>(numBlocks), usePdl, stream);
    if (err != cudaSuccess)
    {
        cudaGetLastError();
        TORCH_CHECK(false, "minimax_m3_sm107_dep_vllm_topk launch failed: ", cudaGetErrorString(err));
    }
}

} // namespace torch_ext

TRTLLM_MV2_NAMESPACE_END

TRTLLM_MV2_LIBRARY_FRAGMENT(m)
{
    m.def(
        "minimax_m3_sm107_dep_vllm_topk(Tensor scores, Tensor n_valid_blocks, Tensor(a!) out, bool use_pdl=True) -> "
        "()");
}

TRTLLM_MV2_LIBRARY_IMPL(CUDA, m)
{
    m.impl("minimax_m3_sm107_dep_vllm_topk", &TRTLLM_MV2_NS::torch_ext::minimaxM3Sm107DepVllmTopk);
}
