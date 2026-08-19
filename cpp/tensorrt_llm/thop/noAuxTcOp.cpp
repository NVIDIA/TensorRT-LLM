/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/runtime/torchUtils.h"

#include "tensorrt_llm/kernels/noAuxTcKernels.h"
#include "tensorrt_llm/thop/thUtils.h"

// #include <c10/cuda/CUDAStream.h>
// #include <cassert>
// #include <set>
// #include <string>
// #include <torch/extension.h>
// #include <vector>

namespace th = torch;
namespace tl = tensorrt_llm;
namespace tk = tensorrt_llm::kernels;

TRTLLM_NAMESPACE_BEGIN

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

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> kimi_k3_noaux_tc_mxfp8_quant(
    th::Tensor const& scores, th::Tensor const& bias, th::Tensor const& hiddenStates, double routedScalingFactor)
{
    constexpr int64_t numExperts = 896;
    constexpr int64_t topK = 16;
    constexpr int64_t hiddenSize = 3584;
    constexpr int64_t maxNumTokens = 64;
    constexpr int64_t sfVecSize = 32;

    int const smVersion = tl::common::getSMVersion();
    TORCH_CHECK(smVersion >= 100 && smVersion < 110, "kimi_k3_noaux_tc_mxfp8_quant requires an SM10x architecture");
    TORCH_CHECK(scores.is_cuda() && bias.is_cuda() && hiddenStates.is_cuda(), "all inputs must be CUDA tensors");
    TORCH_CHECK(scores.get_device() == bias.get_device() && scores.get_device() == hiddenStates.get_device(),
        "all inputs must be on the same device");
    TORCH_CHECK(scores.scalar_type() == torch::kFloat32 && bias.scalar_type() == torch::kFloat32,
        "scores and bias must be float32");
    TORCH_CHECK(hiddenStates.scalar_type() == torch::kBFloat16, "hidden_states must be bfloat16");
    TORCH_CHECK(scores.is_contiguous() && bias.is_contiguous() && hiddenStates.is_contiguous(),
        "all inputs must be contiguous");
    TORCH_CHECK(scores.dim() == 2 && scores.size(1) == numExperts, "scores must have shape [M, 896]");
    TORCH_CHECK(bias.dim() == 1 && bias.numel() == numExperts, "bias must have shape [896]");
    TORCH_CHECK(hiddenStates.dim() == 2 && hiddenStates.size(0) == scores.size(0) && hiddenStates.size(1) == hiddenSize,
        "hidden_states must have shape [M, 3584] with the same M as scores");
    int64_t const numTokens = scores.size(0);
    TORCH_CHECK(numTokens > 0 && numTokens <= maxNumTokens, "M must be in [1, 64]");

    auto const device = th::Device(th::kCUDA, scores.get_device());
    th::Tensor topkValues = th::empty({numTokens, topK}, th::dtype(torch::kBFloat16).device(device));
    th::Tensor topkIndices = th::empty({numTokens, topK}, th::dtype(torch::kInt32).device(device));
    th::Tensor quantizedHiddenStates
        = th::empty({numTokens, hiddenSize}, th::dtype(torch::kFloat8_e4m3fn).device(device));
    th::Tensor hiddenStatesScale = th::empty({numTokens, hiddenSize / sfVecSize}, th::dtype(SF_DTYPE).device(device));

    auto stream = at::cuda::getCurrentCUDAStream(scores.get_device());
    tk::invokeKimiK3NoAuxTcMxFp8Quant(reinterpret_cast<float*>(scores.mutable_data_ptr()),
        reinterpret_cast<float*>(bias.mutable_data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(hiddenStates.mutable_data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(topkValues.mutable_data_ptr()),
        reinterpret_cast<int32_t*>(topkIndices.mutable_data_ptr()),
        reinterpret_cast<int64_t*>(quantizedHiddenStates.mutable_data_ptr()),
        reinterpret_cast<int32_t*>(hiddenStatesScale.mutable_data_ptr()), numTokens, routedScalingFactor, stream);
    return {topkIndices, topkValues, quantizedHiddenStates, hiddenStatesScale};
}

} // end namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "noaux_tc_op(Tensor scores, Tensor bias, int n_group, int topk_group, int topk, float "
        "routed_scaling_factor) -> (Tensor, Tensor)");
    m.def(
        "kimi_k3_noaux_tc_mxfp8_quant(Tensor scores, Tensor bias, Tensor hidden_states, float "
        "routed_scaling_factor) -> (Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("noaux_tc_op", &tensorrt_llm::torch_ext::noaux_tc_op);
    m.impl("kimi_k3_noaux_tc_mxfp8_quant", &tensorrt_llm::torch_ext::kimi_k3_noaux_tc_mxfp8_quant);
}
