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

#include "tensorrt_llm/kernels/minimaxM3/selectLong.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

//! Top-16 block select for MiniMax-M3 decode rows of up to 9216 blocks, into a caller-owned table.
//!
//! @param scores Float32 CUDA tensor [num_kv_heads, num_blocks, total_q], any strides.
//! @param nValidBlocks Contiguous int32 CUDA tensor with at least total_q elements.
//! @param out Int32 CUDA tensor of logical shape [total_q, num_kv_heads, 16] with unit stride in the last dim (the
//! head-major backing the target hands over is a permuted view). Every entry is written.
void minimaxM3SelectLong(torch::Tensor const& scores, torch::Tensor const& nValidBlocks, torch::Tensor& out,
    int64_t initBlocks, int64_t localBlocks, bool usePdl, bool earlyTrigger)
{
    TORCH_CHECK(scores.is_cuda() && scores.scalar_type() == torch::kFloat32 && scores.dim() == 3,
        "minimax_m3_select_long: scores must be a 3-D float32 CUDA tensor [num_kv_heads, num_blocks, total_q]");
    TORCH_CHECK(nValidBlocks.is_cuda() && nValidBlocks.scalar_type() == torch::kInt32 && nValidBlocks.is_contiguous()
            && nValidBlocks.numel() >= scores.size(2),
        "minimax_m3_select_long: n_valid_blocks must be contiguous int32 CUDA with at least total_q elements");
    int64_t const numKvHeads = scores.size(0);
    int64_t const numBlocks = scores.size(1);
    int64_t const totalQ = scores.size(2);
    TORCH_CHECK(numBlocks <= tensorrt_llm::kernels::kMinimaxM3SelectLongMaxBlocks,
        "minimax_m3_select_long supports at most ", tensorrt_llm::kernels::kMinimaxM3SelectLongMaxBlocks,
        " blocks, got ", numBlocks);
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kInt32 && out.dim() == 3 && out.size(0) == totalQ
            && out.size(1) == numKvHeads && out.size(2) == 16 && out.stride(2) == 1,
        "minimax_m3_select_long: out must be int32 [total_q, num_kv_heads, 16] with unit last-dim stride");

    c10::cuda::CUDAGuard const deviceGuard{scores.device()};
    auto const stream = at::cuda::getCurrentCUDAStream(scores.get_device());
    cudaError_t const err = tensorrt_llm::kernels::invokeMinimaxM3SelectLong(scores.data_ptr<float>(), scores.stride(0),
        scores.stride(1), scores.stride(2), nValidBlocks.data_ptr<int32_t>(), out.data_ptr<int32_t>(), out.stride(0),
        out.stride(1), static_cast<int32_t>(numKvHeads), static_cast<int32_t>(numBlocks), static_cast<int32_t>(totalQ),
        static_cast<int32_t>(initBlocks), static_cast<int32_t>(localBlocks), usePdl, earlyTrigger, stream);
    if (err != cudaSuccess)
    {
        cudaGetLastError();
        TORCH_CHECK(false, "minimax_m3_select_long launch failed: ", cudaGetErrorString(err));
    }
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "minimax_m3_select_long(Tensor scores, Tensor n_valid_blocks, Tensor(a!) out, int init_blocks, "
        "int local_blocks, bool use_pdl=False, bool early_trigger=False) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("minimax_m3_select_long", &tensorrt_llm::torch_ext::minimaxM3SelectLong);
}
