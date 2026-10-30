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

#include "tensorrt_llm/kernels/fusedResidualNormQuantKernel.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cstdint>
#include <tuple>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

namespace tk = tensorrt_llm::kernels;

void checkVector(torch::Tensor const& t, char const* name)
{
    TORCH_CHECK(t.is_cuda() && t.is_contiguous(), name, " must be a contiguous CUDA tensor");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, name, " must be 16-byte aligned");
}

// fp32 tensor of `rows` x hidden elements (rows = 1 for per-channel tensors).
float const* fp32Rows(c10::optional<torch::Tensor> const& t, int64_t rows, int64_t hidden, char const* name)
{
    if (!t.has_value())
    {
        return nullptr;
    }
    TORCH_CHECK(t->scalar_type() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(t->numel() == rows * hidden, name, " must have ", rows * hidden, " elements, got ", t->numel());
    checkVector(*t, name);
    return t->data_ptr<float>();
}

} // namespace

// Gated residual update and LayerNorm / AdaLN + static FP8 quantization of each row of x [rows, hidden].
// mode 0: x_out = x + gate * (y + y_bias), no normalized output.
// mode 1: LayerNorm(x_out or x) with affine weight / bias [hidden], quantized with inv_out_scale.
// mode 2: LayerNorm(x_out or x) * (1 + weight[b]) + bias[b] with weight / bias [batch, hidden].
// gate is [batch, hidden]; batch b covers rows [b * rows_per_batch, (b + 1) * rows_per_batch).
// Returns (x_out, normed); x_out is empty without y and normed is empty in mode 0.
std::tuple<torch::Tensor, torch::Tensor> fused_residual_norm_quant(torch::Tensor const& x,
    c10::optional<torch::Tensor> const& y, c10::optional<torch::Tensor> const& y_bias,
    c10::optional<torch::Tensor> const& gate, int64_t mode, c10::optional<torch::Tensor> const& weight,
    c10::optional<torch::Tensor> const& bias, int64_t rows_per_batch, double eps,
    c10::optional<torch::Tensor> const& inv_out_scale)
{
    TORCH_CHECK(mode >= 0 && mode <= 2, "fused_residual_norm_quant: mode must be 0, 1 or 2");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be bfloat16");
    checkVector(x, "x");
    int64_t const hidden = x.size(-1);
    int64_t const rows = x.numel() / hidden;
    TORCH_CHECK(tk::residualNormQuantSupportsHidden(static_cast<int>(hidden)),
        "fused_residual_norm_quant: unsupported hidden size ", hidden);
    TORCH_CHECK(rows_per_batch > 0 && rows % rows_per_batch == 0, "rows_per_batch (", rows_per_batch,
        ") must evenly divide the rows (", rows, ")");
    int64_t const batch = rows / rows_per_batch;
    bool const normalizes = mode != 0;
    TORCH_CHECK(y.has_value() || normalizes, "fused_residual_norm_quant: mode 0 needs y");
    TORCH_CHECK(!y_bias.has_value() || y.has_value(), "y_bias needs y");
    TORCH_CHECK(!gate.has_value() || y.has_value(), "gate needs y");

    tk::ResidualNormQuantParams p;
    p.x = x.data_ptr();
    if (y.has_value())
    {
        TORCH_CHECK(
            y->scalar_type() == torch::kBFloat16 && y->sizes() == x.sizes(), "y must be bfloat16 shaped like x");
        checkVector(*y, "y");
        p.y = y->data_ptr();
    }
    if (y_bias.has_value())
    {
        TORCH_CHECK(
            y_bias->scalar_type() == torch::kBFloat16 && y_bias->numel() == hidden, "y_bias must be bfloat16 [hidden]");
        checkVector(*y_bias, "y_bias");
        p.y_bias = y_bias->data_ptr();
    }
    p.gate = fp32Rows(gate, batch, hidden, "gate");
    int64_t const modRows = mode == 2 ? batch : 1;
    if (normalizes)
    {
        TORCH_CHECK(weight.has_value() && bias.has_value() && inv_out_scale.has_value(),
            "modes 1 and 2 need weight, bias and inv_out_scale");
        p.weight = fp32Rows(weight, modRows, hidden, "weight");
        p.bias = fp32Rows(bias, modRows, hidden, "bias");
        TORCH_CHECK(inv_out_scale->scalar_type() == torch::kFloat32 && inv_out_scale->numel() == 1,
            "inv_out_scale must be float32 [1]");
        p.inv_out_scale = inv_out_scale->data_ptr<float>();
    }

    auto x_out = y.has_value() ? torch::empty_like(x) : x.new_empty({0});
    auto normed = normalizes ? torch::empty(x.sizes(), x.options().dtype(torch::kFloat8_e4m3fn))
                             : x.new_empty({0}, x.options().dtype(torch::kFloat8_e4m3fn));
    p.x_out = y.has_value() ? x_out.data_ptr() : nullptr;
    p.normed = normalizes ? normed.data_ptr() : nullptr;
    p.rows = static_cast<int>(rows);
    p.hidden = static_cast<int>(hidden);
    p.rows_per_batch = static_cast<int>(rows_per_batch);
    p.eps = static_cast<float>(eps);
    p.mode = static_cast<tk::ResidualNormMode>(mode);
    tk::launchResidualNormQuant(p, at::cuda::getCurrentCUDAStream(x.get_device()));
    return {x_out, normed};
}

bool fused_residual_norm_quant_supported(int64_t hidden)
{
    return tk::residualNormQuantSupportsHidden(static_cast<int>(hidden));
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "fused_residual_norm_quant(Tensor x, Tensor? y, Tensor? y_bias, Tensor? gate, int mode, Tensor? weight, "
        "Tensor? bias, int rows_per_batch, float eps, Tensor? inv_out_scale) -> (Tensor, Tensor)");
    m.def("fused_residual_norm_quant_supported(int hidden) -> bool");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("fused_residual_norm_quant", &tensorrt_llm::torch_ext::fused_residual_norm_quant);
}

// No tensor arguments, so dispatch cannot pick a backend from the inputs.
TORCH_LIBRARY_IMPL(trtllm, CompositeExplicitAutograd, m)
{
    m.impl("fused_residual_norm_quant_supported", &tensorrt_llm::torch_ext::fused_residual_norm_quant_supported);
}
