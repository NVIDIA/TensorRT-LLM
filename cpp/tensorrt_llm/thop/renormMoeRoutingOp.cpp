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
#include "tensorrt_llm/kernels/renormMoeRoutingKernels.h"
#include "tensorrt_llm/runtime/torchUtils.h"

namespace th = torch;
namespace tl = tensorrt_llm;
namespace tk = tensorrt_llm::kernels;

namespace torch_ext
{

std::tuple<at::Tensor, at::Tensor> renorm_moe_routing_op(th::Tensor const& router_logits, int64_t topk)
{
    auto data_type = router_logits.scalar_type();
    auto input_size = router_logits.sizes();
    int64_t num_tokens = input_size[0];
    int64_t num_experts = input_size[1];
    TORCH_CHECK(input_size.size() == 2, "router_logits must be a 2D Tensor");
    TORCH_CHECK(topk <= 8, "topk should be smaller than or equal to 8 for now"); //@todo: remove this restriction later
    TORCH_CHECK(num_experts <= 128, "expert number should be smaller than or equal to 128 for now");

    th::Tensor topk_values = th::empty({num_tokens, topk}, th::dtype(torch::kFloat32).device(torch::kCUDA));
    th::Tensor topk_indices = th::empty({num_tokens, topk}, th::dtype(torch::kInt32).device(torch::kCUDA));

    auto stream = at::cuda::getCurrentCUDAStream(router_logits.get_device());

    switch (data_type)
    {
    case torch::kFloat32:
        // Handle Float32
        tk::invokeRenormMoeRouting<float, float, int32_t>(reinterpret_cast<float*>(router_logits.mutable_data_ptr()),
            reinterpret_cast<float*>(topk_values.mutable_data_ptr()),
            reinterpret_cast<int32_t*>(topk_indices.mutable_data_ptr()), num_tokens, num_experts, topk, stream);
        break;
    case torch::kBFloat16:
        // Handle BFloat16
        tk::invokeRenormMoeRouting<__nv_bfloat16, float, int32_t>(
            reinterpret_cast<__nv_bfloat16*>(router_logits.mutable_data_ptr()),
            reinterpret_cast<float*>(topk_values.mutable_data_ptr()),
            reinterpret_cast<int32_t*>(topk_indices.mutable_data_ptr()), num_tokens, num_experts, topk, stream);
        break;
    case torch::kHalf:
        // Handle Half
        tk::invokeRenormMoeRouting<half, float, int32_t>(reinterpret_cast<half*>(router_logits.mutable_data_ptr()),
            reinterpret_cast<float*>(topk_values.mutable_data_ptr()),
            reinterpret_cast<int32_t*>(topk_indices.mutable_data_ptr()), num_tokens, num_experts, topk, stream);
        break;
    default:
        // Handle other data types
        throw std::invalid_argument("Invalid dtype, only supports float32, float16 and bfloat16");
        break;
    }
    return {topk_indices, topk_values};
}

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "renorm_moe_routing_op(Tensor router_logits, SymInt topk"
        ") -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("renorm_moe_routing_op", &torch_ext::renorm_moe_routing_op);
}
