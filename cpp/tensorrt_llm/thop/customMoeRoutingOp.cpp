/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/kernels/customMoeRoutingKernels.h"
#include "tensorrt_llm/runtime/torchUtils.h"

namespace th = torch;
namespace tl = tensorrt_llm;
namespace tk = tensorrt_llm::kernels;

namespace torch_ext
{
template <bool DoSoftmaxBeforeTopK>
std::tuple<at::Tensor, at::Tensor> custom_moe_routing_op(
    th::Tensor const& router_logits, int64_t topk, c10::optional<at::ScalarType> output_dtype)
{
    auto data_type = router_logits.scalar_type();
    auto input_size = router_logits.sizes();
    int64_t num_tokens = input_size[0];
    int64_t num_experts = input_size[1];
    TORCH_CHECK(input_size.size() == 2, "router_logits must be a 2D Tensor");
    TORCH_CHECK(topk <= 8, "topk should be smaller than or equal to 8 for now"); //@todo: remove this restriction later
    TORCH_CHECK(num_experts <= 128, "expert number should be smaller than or equal to 128 for now");

    // Determine output data type
    at::ScalarType topk_values_dtype = output_dtype.value_or(torch::kFloat32);
    TORCH_CHECK(topk_values_dtype == torch::kFloat32 || topk_values_dtype == torch::kBFloat16,
        "output_dtype must be float32 or bfloat16");

    auto opts = router_logits.options();
    th::Tensor topk_values = th::empty({num_tokens, topk}, opts.dtype(topk_values_dtype));
    th::Tensor topk_indices = th::empty({num_tokens, topk}, opts.dtype(torch::kInt32));

    auto stream = at::cuda::getCurrentCUDAStream(router_logits.get_device());

    switch (data_type)
    {
    case torch::kFloat32:
        // Handle Float32 input
        if (topk_values_dtype == torch::kFloat32)
        {
            tk::invokeCustomMoeRouting<float, float, int32_t, DoSoftmaxBeforeTopK>(
                reinterpret_cast<float*>(router_logits.mutable_data_ptr()),
                reinterpret_cast<float*>(topk_values.mutable_data_ptr()), topk_indices.data_ptr<int32_t>(), num_tokens,
                num_experts, topk, stream);
        }
        else
        { // bfloat16 output
            tk::invokeCustomMoeRouting<float, __nv_bfloat16, int32_t, DoSoftmaxBeforeTopK>(
                reinterpret_cast<float*>(router_logits.mutable_data_ptr()),
                reinterpret_cast<__nv_bfloat16*>(topk_values.mutable_data_ptr()), topk_indices.data_ptr<int32_t>(),
                num_tokens, num_experts, topk, stream);
        }
        break;
    case torch::kBFloat16:
        // Handle BFloat16 input
        if (topk_values_dtype == torch::kFloat32)
        {
            tk::invokeCustomMoeRouting<__nv_bfloat16, float, int32_t, DoSoftmaxBeforeTopK>(
                reinterpret_cast<__nv_bfloat16*>(router_logits.mutable_data_ptr()),
                reinterpret_cast<float*>(topk_values.mutable_data_ptr()), topk_indices.data_ptr<int32_t>(), num_tokens,
                num_experts, topk, stream);
        }
        else
        { // bfloat16 output
            tk::invokeCustomMoeRouting<__nv_bfloat16, __nv_bfloat16, int32_t, DoSoftmaxBeforeTopK>(
                reinterpret_cast<__nv_bfloat16*>(router_logits.mutable_data_ptr()),
                reinterpret_cast<__nv_bfloat16*>(topk_values.mutable_data_ptr()), topk_indices.data_ptr<int32_t>(),
                num_tokens, num_experts, topk, stream);
        }
        break;
    case torch::kHalf:
        // Handle Half input
        if (topk_values_dtype == torch::kFloat32)
        {
            tk::invokeCustomMoeRouting<half, float, int32_t, DoSoftmaxBeforeTopK>(
                reinterpret_cast<half*>(router_logits.mutable_data_ptr()),
                reinterpret_cast<float*>(topk_values.mutable_data_ptr()), topk_indices.data_ptr<int32_t>(), num_tokens,
                num_experts, topk, stream);
        }
        else
        { // bfloat16 output
            tk::invokeCustomMoeRouting<half, __nv_bfloat16, int32_t, DoSoftmaxBeforeTopK>(
                reinterpret_cast<half*>(router_logits.mutable_data_ptr()),
                reinterpret_cast<__nv_bfloat16*>(topk_values.mutable_data_ptr()), topk_indices.data_ptr<int32_t>(),
                num_tokens, num_experts, topk, stream);
        }
        break;
    default:
        // Handle other data types
        throw std::invalid_argument("Invalid dtype, only supports float32, float16 and bfloat16");
        break;
    }
    return {topk_indices, topk_values};
}

std::tuple<at::Tensor, at::Tensor> renorm_moe_routing_op(
    th::Tensor const& router_logits, int64_t topk, c10::optional<at::ScalarType> output_dtype)
{
    return custom_moe_routing_op<false>(router_logits, topk, output_dtype);
}

std::tuple<at::Tensor, at::Tensor> default_moe_routing_op(
    th::Tensor const& router_logits, int64_t topk, c10::optional<at::ScalarType> output_dtype)
{
    return custom_moe_routing_op<true>(router_logits, topk, output_dtype);
}
} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "renorm_moe_routing_op(Tensor router_logits, SymInt topk, ScalarType? output_dtype=None"
        ") -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("renorm_moe_routing_op", &torch_ext::renorm_moe_routing_op);
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "default_moe_routing_op(Tensor router_logits, SymInt topk, ScalarType? output_dtype=None"
        ") -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("default_moe_routing_op", &torch_ext::default_moe_routing_op);
}
