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

#include "tensorrt_llm/kernels/minimaxM31/softmaxMcSm103.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cstdint>
#include <limits>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

//! ST-01-SM's multi-CTA softmax (see kernels/minimaxM31/softmaxMcSm103.h): logits fp32 [rows, d] -> out fp32
//! [rows, d], bitwise flashinfer.sampling.softmax(logits, temperature) on FlashInfer's Path A shapes (rows <= 128,
//! d >= 24576). temperature: an fp32 [rows] tensor, or None for the scalar temperature_val. partials: a CUDA buffer
//! of at least rows x ceil(d / 8192) float2, overwritten by every call; write_ctas: the CTAs per row of the write pass.
void minimaxM31SoftmaxMc(torch::Tensor const& logits, std::optional<torch::Tensor> const& temperature,
    double temperatureVal, torch::Tensor& partials, torch::Tensor& out, int64_t writeCtas)
{
    TORCH_CHECK(
        logits.is_cuda() && logits.scalar_type() == torch::kFloat32 && logits.dim() == 2 && logits.is_contiguous(),
        "minimax_m31_softmax_mc: logits must be a contiguous 2-D float32 CUDA tensor");
    int64_t const rows = logits.size(0);
    int64_t const d = logits.size(1);
    TORCH_CHECK(d % 4 == 0 && reinterpret_cast<uintptr_t>(logits.data_ptr()) % 16 == 0,
        "minimax_m31_softmax_mc: logits must be 16-byte aligned with a width divisible by 4");
    TORCH_CHECK(rows <= 128 && d >= 24576,
        "minimax_m31_softmax_mc: FlashInfer's Path A shapes only (rows <= 128, width >= 24576)");
    TORCH_CHECK(rows * d <= std::numeric_limits<int>::max(), "minimax_m31_softmax_mc: logits is too large");
    float const* temperatureArr = nullptr;
    if (temperature.has_value())
    {
        auto const& t = temperature.value();
        TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kFloat32 && t.dim() == 1 && t.is_contiguous()
                && t.size(0) == rows && t.get_device() == logits.get_device(),
            "minimax_m31_softmax_mc: temperature must be a contiguous float32 [rows] tensor on logits' device");
        temperatureArr = t.data_ptr<float>();
    }
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kFloat32 && out.dim() == 2 && out.is_contiguous()
            && out.size(0) == rows && out.size(1) == d && reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0
            && out.get_device() == logits.get_device(),
        "minimax_m31_softmax_mc: out must be a contiguous, 16-byte aligned float32 [rows, d] on logits' device");
    TORCH_CHECK(writeCtas >= 1 && writeCtas <= 65535, "minimax_m31_softmax_mc: write_ctas must be in [1, 65535]");
    auto const partialsBytes = static_cast<int64_t>(
        tensorrt_llm::kernels::minimaxM31::softmaxMcPartialsBytes(static_cast<int>(rows), static_cast<int>(d)));
    TORCH_CHECK(partials.is_cuda() && partials.is_contiguous() && partials.get_device() == logits.get_device()
            && reinterpret_cast<uintptr_t>(partials.data_ptr()) % 8 == 0
            && partials.numel() * partials.element_size() >= partialsBytes,
        "minimax_m31_softmax_mc: partials must be a contiguous, 8-byte aligned CUDA buffer of at least ", partialsBytes,
        " bytes on logits' device");
    if (rows == 0)
    {
        return;
    }
    c10::cuda::CUDAGuard const deviceGuard{logits.device()};
    auto const err = tensorrt_llm::kernels::minimaxM31::invokeSoftmaxMc(logits.data_ptr<float>(), out.data_ptr<float>(),
        temperatureArr, static_cast<float>(temperatureVal), static_cast<int>(rows), static_cast<int>(d),
        static_cast<int>(writeCtas), partials.data_ptr(), at::cuda::getCurrentCUDAStream(logits.get_device()));
    TORCH_CHECK(err == cudaSuccess, "minimax_m31_softmax_mc launch failed: ", cudaGetErrorString(err));
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "minimax_m31_softmax_mc(Tensor logits, Tensor? temperature, float temperature_val, Tensor(a!) partials, "
        "Tensor(b!) out, int write_ctas) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("minimax_m31_softmax_mc", &tensorrt_llm::torch_ext::minimaxM31SoftmaxMc);
}
