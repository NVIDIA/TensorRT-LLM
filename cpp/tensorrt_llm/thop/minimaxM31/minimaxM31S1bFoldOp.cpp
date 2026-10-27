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

#include "tensorrt_llm/kernels/minimaxM31/s1bFoldSm107.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <optional>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

//! S1b fold (see kernels/minimaxM31/s1bFoldSm107.h): partials fp32 [S, M, 128] (S 8 or 16), bias fp32 [128] ->
//! ids int32 [M, 4] and weights bf16 [M, 4] (written), logits_out fp32 [M, 128] or None.
void minimaxM31S1bFold(torch::Tensor const& partials, torch::Tensor const& bias, std::optional<torch::Tensor> logitsOut,
    torch::Tensor& ids, torch::Tensor& weights, double routeScale, bool pdl)
{
    TORCH_CHECK(partials.is_cuda() && partials.scalar_type() == torch::kFloat32 && partials.dim() == 3
            && partials.is_contiguous() && partials.size(2) == 128 && (partials.size(0) == 8 || partials.size(0) == 16),
        "minimax_m31_s1b_fold: partials must be a contiguous float32 [8 | 16, M, 128] CUDA tensor");
    int64_t const M = partials.size(1);
    TORCH_CHECK(M <= 65535 * 8, "minimax_m31_s1b_fold: too many rows");
    TORCH_CHECK(bias.is_cuda() && bias.scalar_type() == torch::kFloat32 && bias.is_contiguous() && bias.numel() == 128
            && bias.get_device() == partials.get_device(),
        "minimax_m31_s1b_fold: bias must be a contiguous float32 [128] tensor on partials' device");
    TORCH_CHECK(ids.is_cuda() && ids.scalar_type() == torch::kInt32 && ids.is_contiguous() && ids.numel() == M * 4,
        "minimax_m31_s1b_fold: ids must be a contiguous int32 [M, 4] tensor");
    TORCH_CHECK(weights.is_cuda() && weights.scalar_type() == torch::kBFloat16 && weights.is_contiguous()
            && weights.numel() == M * 4,
        "minimax_m31_s1b_fold: weights must be a contiguous bf16 [M, 4] tensor");
    float* lo = nullptr;
    if (logitsOut.has_value())
    {
        auto& t = logitsOut.value();
        TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kFloat32 && t.is_contiguous() && t.numel() == M * 128,
            "minimax_m31_s1b_fold: logits_out must be a contiguous float32 [M, 128] tensor");
        lo = t.data_ptr<float>();
    }
    at::cuda::CUDAGuard const guard(partials.device());
    auto e = tensorrt_llm::kernels::minimaxM31::invokeS1bFold(partials.data_ptr<float>(),
        static_cast<int32_t>(partials.size(0)), static_cast<int32_t>(M), bias.data_ptr<float>(),
        static_cast<float>(routeScale), lo, ids.data_ptr<int32_t>(),
        reinterpret_cast<__nv_bfloat16*>(weights.data_ptr()), pdl,
        at::cuda::getCurrentCUDAStream(partials.get_device()));
    TORCH_CHECK(e == cudaSuccess, "minimax_m31_s1b_fold: ", cudaGetErrorString(e));
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "minimax_m31_s1b_fold(Tensor partials, Tensor bias, Tensor(a!)? logits_out, Tensor(b!) ids, Tensor(c!) "
        "weights, "
        "float route_scale, bool pdl) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("minimax_m31_s1b_fold", &tensorrt_llm::torch_ext::minimaxM31S1bFold);
}
