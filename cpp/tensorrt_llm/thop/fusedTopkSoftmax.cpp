/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/workspace.h"
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/EmptyTensor.h>

#include <cuda_fp16.h>

#include <cstdint>

namespace torch_ext
{

std::tuple<torch::Tensor, torch::Tensor> fused_topk_softmax(torch::Tensor const& router_logits, int64_t const top_k,
    int64_t const num_experts_total, int64_t const start_expert, int64_t const end_expert)
{
    // TODO: enable once the kernel has been added to the internal CUTLASS library.
    TLLM_CHECK_WITH_INFO(false, "Fused topk/softmax op has not been enabled yet.");

    CHECK_INPUT(router_logits, torch::kBFloat16);

    auto const& router_logits_shape = router_logits.sizes();
    auto const& rank = router_logits_shape.size();

    TORCH_CHECK(rank == 2, "router_logits should be 2D tensor.");
    int64_t const num_rows = router_logits_shape[0];

    auto token_final_scales
        = torch::empty({num_rows, top_k}, torch::dtype(torch::kFloat32).device(router_logits.device()));
    auto token_selected_experts
        = torch::empty({num_rows, top_k}, torch::dtype(torch::kInt32).device(router_logits.device()));

    // auto stream = at::cuda::getCurrentCUDAStream(router_logits.get_device());
    // tensorrt_llm::kernels::topkGatingSoftmaxKernelLauncher(
    //     static_cast<__nv_bfloat16 const*>(router_logits.const_data_ptr()),
    //     static_cast<float*>(token_final_scales.data_ptr()), static_cast<int*>(token_selected_experts.data_ptr()),
    //     num_rows, top_k, num_experts_total, start_expert, end_expert, stream);
    return {token_final_scales, token_selected_experts};
}
} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "fused_topk_softmax(Tensor router_logits, int top_k, "
        "int num_experts_total, int start_expert, "
        "int end_expert) -> (Tensor, Tensor) ");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("fused_topk_softmax", &torch_ext::fused_topk_softmax);
}
