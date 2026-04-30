/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

static constexpr int kTOPK = 6;

// PyTorch wrapper function that calls the kernel implementation
void gate_forward(th::Tensor scores_in, // [batch_size, nExperts] - pre-computed from linear(x, weight)
    th::Tensor bias,                    // empty tensor if hash mode
    th::Tensor input_ids,               // empty tensor if non-hash mode
    th::Tensor tid2eid,                 // empty tensor if non-hash mode
    th::Tensor out_weights,             // [batch_size, topK] - pre-allocated
    th::Tensor out_indices,             // [batch_size, topK] - pre-allocated
    int64_t topk, double route_scale, bool is_hash)
{
    TORCH_CHECK(topk == kTOPK, "topk must be ", kTOPK);
    auto const n_experts = scores_in.size(1);
    TORCH_CHECK(n_experts == 256 || n_experts == 384, "n_experts must be 256 or 384");
    TORCH_CHECK(scores_in.scalar_type() == torch::kFloat32, "scores_in must be float32");
    TORCH_CHECK(out_weights.scalar_type() == torch::kFloat32, "out_weights must be float32");
    TORCH_CHECK(out_indices.scalar_type() == torch::kInt32, "out_indices must be int32");
    TORCH_CHECK(
        scores_in.is_cuda() && out_weights.is_cuda() && out_indices.is_cuda(), "All tensors must be CUDA tensors");
    TORCH_CHECK(
        scores_in.get_device() == out_weights.get_device() && scores_in.get_device() == out_indices.get_device(),
        "All tensors must be on the same device");

    auto const batch_size = scores_in.size(0);
    auto stream = at::cuda::getCurrentCUDAStream(scores_in.get_device());

    // Call the kernel implementation from kernels namespace
    kernels::gate_forward(scores_in.data_ptr<float>(), is_hash ? nullptr : bias.data_ptr<float>(),
        is_hash ? input_ids.data_ptr<int>() : nullptr, is_hash ? tid2eid.data_ptr<int>() : nullptr,
        out_weights.data_ptr<float>(), out_indices.data_ptr<int>(), batch_size, static_cast<int>(n_experts),
        static_cast<float>(route_scale), is_hash, stream);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "gate_forward(Tensor scores_in, Tensor bias, Tensor input_ids, Tensor tid2eid, Tensor out_weights, "
        "Tensor out_indices, int topk, float route_scale, bool is_hash) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("gate_forward", &tensorrt_llm::torch_ext::gate_forward);
}
