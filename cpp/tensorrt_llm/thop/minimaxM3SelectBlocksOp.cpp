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

#include "tensorrt_llm/kernels/minimaxM3SelectBlocks.h"

#include <ATen/cuda/CUDAContext.h>
#include <limits>
#include <torch/extension.h>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

torch::Tensor minimaxM3SelectBlocks(torch::Tensor const& scores, torch::Tensor const& nValidBlocks, int64_t topK,
    int64_t initBlocks, int64_t localBlocks)
{
    constexpr int64_t kRequiredTopK = 16;
    constexpr int64_t kMaxBlockIndex = 65'535;

    TORCH_CHECK(scores.is_cuda(), "minimax_m3_select_blocks expects CUDA scores");
    TORCH_CHECK(scores.scalar_type() == torch::kFloat32, "minimax_m3_select_blocks expects float32 scores, got ",
        scores.scalar_type());
    TORCH_CHECK(scores.dim() == 3, "scores must be [num_kv_heads, num_blocks, total_queries]");
    TORCH_CHECK(scores.stride(0) >= 0 && scores.stride(1) >= 0 && scores.stride(2) >= 0,
        "scores must have non-negative strides");

    TORCH_CHECK(nValidBlocks.is_cuda(), "minimax_m3_select_blocks expects CUDA n_valid_blocks");
    TORCH_CHECK(nValidBlocks.device() == scores.device(), "scores and n_valid_blocks must be on the same device");
    TORCH_CHECK(nValidBlocks.scalar_type() == torch::kInt32,
        "minimax_m3_select_blocks expects int32 n_valid_blocks, got ", nValidBlocks.scalar_type());
    TORCH_CHECK(
        nValidBlocks.dim() == 1 && nValidBlocks.size(0) == scores.size(2), "n_valid_blocks must be [total_queries]");
    TORCH_CHECK(nValidBlocks.is_contiguous(), "n_valid_blocks must be contiguous");

    TORCH_CHECK(topK == kRequiredTopK, "minimax_m3_select_blocks supports topk=16, got ", topK);
    TORCH_CHECK(initBlocks >= 0, "init_blocks must be non-negative");
    TORCH_CHECK(localBlocks >= 0, "local_blocks must be non-negative");
    TORCH_CHECK(scores.size(1) <= kMaxBlockIndex, "minimax_m3_select_blocks supports at most ", kMaxBlockIndex,
        " blocks, got ", scores.size(1));
    TORCH_CHECK(scores.size(0) <= std::numeric_limits<int32_t>::max()
            && scores.size(1) <= std::numeric_limits<int32_t>::max()
            && scores.size(2) <= std::numeric_limits<int32_t>::max(),
        "minimax_m3_select_blocks dimensions exceed int32 range");
    TORCH_CHECK(scores.size(0) * scores.size(2) <= std::numeric_limits<int32_t>::max(),
        "minimax_m3_select_blocks output rows exceed int32 range");
    TORCH_CHECK(initBlocks <= std::numeric_limits<int32_t>::max() && localBlocks <= std::numeric_limits<int32_t>::max(),
        "minimax_m3_select_blocks forcing ranges exceed int32 range");

    auto output = torch::empty({scores.size(2), scores.size(0), topK}, scores.options().dtype(torch::kInt32));
    auto const stream = at::cuda::getCurrentCUDAStream(scores.get_device());
    tensorrt_llm::kernels::invokeMinimaxM3SelectBlocks(scores.data_ptr<float>(), scores.stride(0), scores.stride(1),
        scores.stride(2), nValidBlocks.data_ptr<int32_t>(), output.data_ptr<int32_t>(),
        static_cast<int32_t>(scores.size(0)), static_cast<int32_t>(scores.size(1)),
        static_cast<int32_t>(scores.size(2)), static_cast<int32_t>(initBlocks), static_cast<int32_t>(localBlocks),
        stream);
    return output;
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "minimax_m3_select_blocks(Tensor scores, Tensor n_valid_blocks, int topk, int init_blocks, int "
        "local_blocks) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("minimax_m3_select_blocks", &tensorrt_llm::torch_ext::minimaxM3SelectBlocks);
}
