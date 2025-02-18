/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/runtime/torchUtils.h"

#include "tensorrt_llm/kernels/noAuxTcKernels.h"

// #include <NvInferRuntime.h>
// #include <c10/cuda/CUDAStream.h>
// #include <cassert>
// #include <set>
// #include <string>
// #include <torch/extension.h>
// #include <vector>

namespace th = torch;
namespace tl = tensorrt_llm;
namespace tk = tensorrt_llm::kernels;

namespace torch_ext
{
th::Tensor noaux_tc_op(th::Tensor const& scores, th::Tensor const& scores_with_bias, int64_t n_group,
    int64_t topk_group, int64_t topk, double routed_scaling_factor)
{
    auto data_type = scores_with_bias.scalar_type();
    auto input_size = scores_with_bias.sizes();
    int64_t num_tokens = input_size[0];
    int64_t num_experts = input_size[1];
    TORCH_CHECK(input_size.size() == 2, "scores_with_bias must be a 2D Tensor");
    TORCH_CHECK(num_experts % n_group == 0, "num_experts should be divisible by n_group");
    TORCH_CHECK(n_group < 32, "n_group should be smaller than 32 for now"); //@todo: remove this restriction later
    TORCH_CHECK(topk < 32, "topk should be smaller than 32 for now");       //@todo: remove this restriction later

    th::Tensor group_scores = th::empty({num_tokens, n_group}, th::dtype(data_type).device(torch::kCUDA));

    auto stream = at::cuda::getCurrentCUDAStream(scores_with_bias.get_device());

    switch (data_type)
    {
    case torch::kFloat16:
        // Handle Float16
        tk::invokeNoAuxTc<half>(reinterpret_cast<half*>(scores.mutable_data_ptr()),
            reinterpret_cast<half*>(group_scores.mutable_data_ptr()),
            reinterpret_cast<half*>(scores_with_bias.data_ptr()), num_tokens, num_experts, n_group, topk_group, topk,
            routed_scaling_factor, stream);
        break;
    case torch::kFloat32:
        // Handle Float32
        tk::invokeNoAuxTc<float>(reinterpret_cast<float*>(scores.mutable_data_ptr()),
            reinterpret_cast<float*>(group_scores.mutable_data_ptr()),
            reinterpret_cast<float*>(scores_with_bias.data_ptr()), num_tokens, num_experts, n_group, topk_group, topk,
            routed_scaling_factor, stream);
        break;
    case torch::kBFloat16:
        // Handle BFloat16
        tk::invokeNoAuxTc<__nv_bfloat16>(reinterpret_cast<__nv_bfloat16*>(scores.mutable_data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(group_scores.mutable_data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(scores_with_bias.data_ptr()), num_tokens, num_experts, n_group, topk_group,
            topk, routed_scaling_factor, stream);
        break;
    default:
        // Handle other data types
        throw std::invalid_argument("Invalid dtype, only supports float16, float32, and bfloat16");
        break;
    }
    return scores;
}

} // end namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "noaux_tc_op(Tensor scores, Tensor scores_with_bias, int n_group, int topk_group, int topk, float "
        "routed_scaling_factor) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("noaux_tc_op", &torch_ext::noaux_tc_op);
}
