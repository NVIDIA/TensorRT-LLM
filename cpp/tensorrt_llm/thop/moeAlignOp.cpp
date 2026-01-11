/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/kernels/moeAlignKernels.h"
#include "thUtils.h"
#include <torch/extension.h>

namespace tk = tensorrt_llm::kernels;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

void moeAlignBlockSizeOp(torch::Tensor topk_ids, int64_t num_experts, int64_t block_size,
    torch::Tensor sorted_token_ids, torch::Tensor expert_ids, torch::Tensor num_tokens_post_pad)
{
    // Validate inputs
    CHECK_TH_CUDA(topk_ids);
    CHECK_CONTIGUOUS(topk_ids);
    CHECK_INPUT(sorted_token_ids, torch::kInt32);
    CHECK_INPUT(expert_ids, torch::kInt32);
    CHECK_INPUT(num_tokens_post_pad, torch::kInt32);

    TORCH_CHECK(topk_ids.scalar_type() == torch::kInt32 || topk_ids.scalar_type() == torch::kInt64,
        "topk_ids must be int32 or int64");

    auto stream = at::cuda::getCurrentCUDAStream();

    tk::invokeMoeAlignBlockSize(topk_ids.data_ptr(), topk_ids.element_size(), sorted_token_ids.data_ptr<int32_t>(),
        expert_ids.data_ptr<int32_t>(), num_tokens_post_pad.data_ptr<int32_t>(), static_cast<int32_t>(num_experts),
        static_cast<int32_t>(block_size), static_cast<int32_t>(topk_ids.numel()),
        static_cast<int32_t>(sorted_token_ids.size(0)), stream);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "moe_align_block_size(Tensor topk_ids, int num_experts, int block_size, "
        "Tensor(a!) sorted_token_ids, Tensor(a!) expert_ids, Tensor(a!) num_tokens_post_pad) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("moe_align_block_size", &tensorrt_llm::torch_ext::moeAlignBlockSizeOp);
}
