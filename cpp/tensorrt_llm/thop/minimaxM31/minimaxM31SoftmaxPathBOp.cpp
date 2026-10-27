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

#include "tensorrt_llm/kernels/minimaxM31/softmaxPathBSm107.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>
#include <torch/extension.h>
TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{
namespace
{
bool overlaps(torch::Tensor const& lhs, torch::Tensor const& rhs)
{
    auto const lhsBegin = reinterpret_cast<uintptr_t>(lhs.data_ptr());
    auto const rhsBegin = reinterpret_cast<uintptr_t>(rhs.data_ptr());
    return lhsBegin < rhsBegin + rhs.nbytes() && rhsBegin < lhsBegin + lhs.nbytes();
}
} // namespace

void minimaxM31SoftmaxPathB(torch::Tensor const& logits, std::optional<torch::Tensor> const& temperature,
    double temperatureValue, torch::Tensor& output)
{
    using namespace tensorrt_llm::kernels::minimaxM31;
    TORCH_CHECK(
        logits.is_cuda() && logits.scalar_type() == torch::kFloat32 && logits.dim() == 2 && logits.is_contiguous(),
        "softmax_pathb: expected contiguous fp32 CUDA matrix");
    int64_t const rows = logits.size(0);
    TORCH_CHECK(rows >= kSoftmaxPathBMinRows && rows <= kSoftmaxPathBMaxRows && logits.size(1) == kSoftmaxPathBVocab
            && reinterpret_cast<uintptr_t>(logits.data_ptr()) % 16 == 0,
        "softmax_pathb: expected aligned [129..448, 200064] logits");
    TORCH_CHECK(output.device() == logits.device() && output.scalar_type() == torch::kFloat32
            && output.sizes() == logits.sizes() && output.is_contiguous()
            && reinterpret_cast<uintptr_t>(output.data_ptr()) % 16 == 0 && !overlaps(logits, output),
        "softmax_pathb: output must match logits and not overlap it");
    c10::cuda::CUDAGuard const deviceGuard{logits.device()};
    auto const* properties = at::cuda::getDeviceProperties(logits.get_device());
    TORCH_CHECK(properties->major == 10 && properties->minor == 7, "softmax_pathb: SM107 required");
    float* temperatureArray = nullptr;
    if (temperature.has_value())
    {
        auto const& t = temperature.value();
        TORCH_CHECK(t.device() == logits.device() && t.scalar_type() == torch::kFloat32 && t.dim() == 1
                && t.size(0) == rows && t.is_contiguous() && !overlaps(t, output),
            "softmax_pathb: temperature must be nonoverlapping contiguous fp32 [rows] on logits device");
        temperatureArray = t.data_ptr<float>();
    }
    auto const error = invokeSoftmaxPathB(logits.data_ptr<float>(), output.data_ptr<float>(), temperatureArray,
        static_cast<float>(temperatureValue), static_cast<int>(rows),
        at::cuda::getCurrentCUDAStream(logits.get_device()));
    TORCH_CHECK(error == cudaSuccess, "softmax_pathb launch failed: ", cudaGetErrorString(error));
}
} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("minimax_m31_softmax_pathb(Tensor logits, Tensor? temperature, float temperature_val, Tensor(a!) out) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("minimax_m31_softmax_pathb", &tensorrt_llm::torch_ext::minimaxM31SoftmaxPathB);
}
