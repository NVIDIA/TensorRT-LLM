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
std::tuple<at::Tensor, at::Tensor> noaux_tc_op(th::Tensor const& scores, th::Tensor const& bias, int64_t n_group,
    int64_t topk_group, int64_t topk, double routed_scaling_factor)
{
    auto data_type = scores.scalar_type();
    auto bias_type = bias.scalar_type();

    auto input_size = scores.sizes();
    int64_t num_tokens = input_size[0];
    int64_t num_experts = input_size[1];
    TORCH_CHECK(input_size.size() == 2, "scores must be a 2D Tensor");
    TORCH_CHECK(scores.is_cuda() && bias.is_cuda(), "scores and bias must be CUDA tensors");
    TORCH_CHECK(scores.get_device() == bias.get_device(), "scores and bias must be on the same device");
    TORCH_CHECK(bias.dim() == 1 && bias.numel() == num_experts,
        "bias must be 1D with length == number of experts (%ld)", num_experts);
    TORCH_CHECK(num_experts % n_group == 0, "num_experts should be divisible by n_group");
    TORCH_CHECK(
        n_group <= 32, "n_group should be smaller than or equal to 32 for now"); //@todo: remove this restriction later
    TORCH_CHECK(
        topk <= 32, "topk should be smaller than or equal to 32 for now");       //@todo: remove this restriction later

    th::Tensor topk_values = th::empty({num_tokens, topk}, th::dtype(data_type).device(torch::kCUDA));
    th::Tensor topk_indices = th::empty({num_tokens, topk}, th::dtype(torch::kInt32).device(torch::kCUDA));
    //@TODO check the data type of indices

    auto stream = at::cuda::getCurrentCUDAStream(scores.get_device());

    switch (data_type)
    {
    case torch::kFloat16:
        // Handle Float16
        switch (bias_type)
        {
        case torch::kFloat16:
            tk::invokeNoAuxTc<half, half, half, int32_t>(reinterpret_cast<half*>(scores.mutable_data_ptr()),
                reinterpret_cast<half*>(bias.mutable_data_ptr()),
                reinterpret_cast<half*>(topk_values.mutable_data_ptr()),
                reinterpret_cast<int32_t*>(topk_indices.mutable_data_ptr()), num_tokens, num_experts, n_group,
                topk_group, topk, routed_scaling_factor, stream);
            break;
        case torch::kFloat32:
            tk::invokeNoAuxTc<half, float, half, int32_t>(reinterpret_cast<half*>(scores.mutable_data_ptr()),
                reinterpret_cast<float*>(bias.mutable_data_ptr()),
                reinterpret_cast<half*>(topk_values.mutable_data_ptr()),
                reinterpret_cast<int32_t*>(topk_indices.mutable_data_ptr()), num_tokens, num_experts, n_group,
                topk_group, topk, routed_scaling_factor, stream);
            break;
        case torch::kBFloat16:
            tk::invokeNoAuxTc<half, __nv_bfloat16, half, int32_t>(reinterpret_cast<half*>(scores.mutable_data_ptr()),
                reinterpret_cast<__nv_bfloat16*>(bias.mutable_data_ptr()),
                reinterpret_cast<half*>(topk_values.mutable_data_ptr()),
                reinterpret_cast<int32_t*>(topk_indices.mutable_data_ptr()), num_tokens, num_experts, n_group,
                topk_group, topk, routed_scaling_factor, stream);
        default: throw std::invalid_argument("Invalid bias dtype, only supports float16, float32, and bfloat16"); break;
        }
        break;
    case torch::kFloat32:
        switch (bias_type)
        {
        case torch::kFloat32:
            tk::invokeNoAuxTc<float, float, float, int32_t>(reinterpret_cast<float*>(scores.mutable_data_ptr()),
                reinterpret_cast<float*>(bias.mutable_data_ptr()),
                reinterpret_cast<float*>(topk_values.mutable_data_ptr()),
                reinterpret_cast<int32_t*>(topk_indices.mutable_data_ptr()), num_tokens, num_experts, n_group,
                topk_group, topk, routed_scaling_factor, stream);
            break;
        case torch::kFloat16:
            tk::invokeNoAuxTc<float, half, float, int32_t>(reinterpret_cast<float*>(scores.mutable_data_ptr()),
                reinterpret_cast<half*>(bias.mutable_data_ptr()),
                reinterpret_cast<float*>(topk_values.mutable_data_ptr()),
                reinterpret_cast<int32_t*>(topk_indices.mutable_data_ptr()), num_tokens, num_experts, n_group,
                topk_group, topk, routed_scaling_factor, stream);
            break;
        case torch::kBFloat16:
            tk::invokeNoAuxTc<float, __nv_bfloat16, float, int32_t>(reinterpret_cast<float*>(scores.mutable_data_ptr()),
                reinterpret_cast<__nv_bfloat16*>(bias.mutable_data_ptr()),
                reinterpret_cast<float*>(topk_values.mutable_data_ptr()),
                reinterpret_cast<int32_t*>(topk_indices.mutable_data_ptr()), num_tokens, num_experts, n_group,
                topk_group, topk, routed_scaling_factor, stream);
            break;
        default: throw std::invalid_argument("Invalid bias dtype, only supports float16, float32, and bfloat16"); break;
        }
        break;
    case torch::kBFloat16:
        // Handle BFloat16
        switch (bias_type)
        {
        case torch::kBFloat16:
            tk::invokeNoAuxTc<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, int32_t>(
                reinterpret_cast<__nv_bfloat16*>(scores.mutable_data_ptr()),
                reinterpret_cast<__nv_bfloat16*>(bias.mutable_data_ptr()),
                reinterpret_cast<__nv_bfloat16*>(topk_values.mutable_data_ptr()),
                reinterpret_cast<int32_t*>(topk_indices.mutable_data_ptr()), num_tokens, num_experts, n_group,
                topk_group, topk, routed_scaling_factor, stream);
            break;
        case torch::kFloat16:
            tk::invokeNoAuxTc<__nv_bfloat16, half, __nv_bfloat16, int32_t>(
                reinterpret_cast<__nv_bfloat16*>(scores.mutable_data_ptr()),
                reinterpret_cast<half*>(bias.mutable_data_ptr()),
                reinterpret_cast<__nv_bfloat16*>(topk_values.mutable_data_ptr()),
                reinterpret_cast<int32_t*>(topk_indices.mutable_data_ptr()), num_tokens, num_experts, n_group,
                topk_group, topk, routed_scaling_factor, stream);
            break;
        case torch::kFloat32:
            tk::invokeNoAuxTc<__nv_bfloat16, float, __nv_bfloat16, int32_t>(
                reinterpret_cast<__nv_bfloat16*>(scores.mutable_data_ptr()),
                reinterpret_cast<float*>(bias.mutable_data_ptr()),
                reinterpret_cast<__nv_bfloat16*>(topk_values.mutable_data_ptr()),
                reinterpret_cast<int32_t*>(topk_indices.mutable_data_ptr()), num_tokens, num_experts, n_group,
                topk_group, topk, routed_scaling_factor, stream);
            break;
        default: throw std::invalid_argument("Invalid bias dtype, only supports bfloat16, float16, and float32"); break;
        }
        break;
    default:
        // Handle other data types
        throw std::invalid_argument("Invalid dtype, only supports float16, float32, and bfloat16");
        break;
    }
    return {topk_values, topk_indices};
}

} // end namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "noaux_tc_op(Tensor scores, Tensor bias, int n_group, int topk_group, int topk, float "
        "routed_scaling_factor) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("noaux_tc_op", &torch_ext::noaux_tc_op);
}
