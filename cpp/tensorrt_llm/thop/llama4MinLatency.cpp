/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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
#include "tensorrt_llm/common/dataType.h"
#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/kernels/llama4MinLatencyKernels/llama4Bf16Bf16Gemm.h"
#include "tensorrt_llm/kernels/llama4MinLatencyKernels/llama4Fp8Bf16Gemm.h"
#include "tensorrt_llm/kernels/llama4MinLatencyKernels/llama4Fp8Fp8GemmSwiGLU.h"
#include "tensorrt_llm/kernels/llama4MinLatencyKernels/llama4MinLatencyMoEOp.h"
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <nvml.h>
#include <torch/extension.h>

#include <cstddef>
#include <cstdint>
#include <unordered_set>

namespace torch_ext
{

torch::Tensor llama4_bf16_bf16_gemm(torch::Tensor const& inputA, torch::Tensor const& inputB)
{
    CHECK_INPUT(inputA, c10::ScalarType::BFloat16);
    CHECK_INPUT(inputB, c10::ScalarType::BFloat16);

    TORCH_CHECK(inputA.dim() == 2, "inputA must be 2D.");
    TORCH_CHECK(inputB.dim() == 2, "inputB must be 2D.");
    TORCH_CHECK(inputA.sizes()[1] == 5120, "inputA.size(1) must be 5120");
    TORCH_CHECK(inputB.sizes()[0] == 128, "inputB.size(0) must be 128");
    TORCH_CHECK(inputB.sizes()[1] == 5120, "inputB.size(1) must be 5120");

    auto const num_tokens = inputA.sizes()[0];
    auto const hidden_out = inputB.sizes()[0];

    auto stream = at::cuda::getCurrentCUDAStream(inputA.get_device());
    auto output = torch::empty(
        {num_tokens, hidden_out}, torch::TensorOptions().dtype(c10::ScalarType::BFloat16).device(inputA.device()));

    tensorrt_llm::kernels::llama4_min_latency::llama4_bf16_bf16_gemm::llama4_bf16_bf16_gemm_op(
        num_tokens, inputA.data_ptr(), inputB.data_ptr(), output.data_ptr(), stream);

    return output;
}

torch::Tensor llama4_fp8_bf16_gemm(torch::Tensor const& inputA, torch::Tensor const& inputB,
    torch::Tensor const& scaling_factor, torch::optional<torch::Tensor> const& position_ids)
{
    CHECK_INPUT(inputA, c10::ScalarType::Float8_e4m3fn);
    CHECK_INPUT(inputB, c10::ScalarType::Float8_e4m3fn);
    CHECK_INPUT(scaling_factor, c10::ScalarType::Float);

    bool const has_position_ids = position_ids.has_value();
    if (has_position_ids)
    {
        CHECK_TH_CUDA(position_ids.value());
        CHECK_CONTIGUOUS(position_ids.value());
        if (position_ids.value().scalar_type() != c10::ScalarType::Long
            && position_ids.value().scalar_type() != c10::ScalarType::Int)
        {
            TORCH_CHECK(false, "position_ids must be a Long or Int tensor.");
        }
    }

    bool const position_ids_int64 = has_position_ids && position_ids.value().dtype() == c10::ScalarType::Long;

    TORCH_CHECK(inputA.dim() == 2, "inputA must be 2D.");
    TORCH_CHECK(inputB.dim() == 2, "inputB must be 2D.");
    TORCH_CHECK(inputA.sizes()[1] == 5120, "inputA.size(1) must be 5120");
    TORCH_CHECK(inputB.sizes()[1] == 5120, "inputB.size(1) must be 5120");
    TORCH_CHECK(scaling_factor.dim() == 0, "scaling_factor must be a scalar tensor");

    auto const num_tokens = inputA.sizes()[0];
    auto const hidden_in = inputA.sizes()[1];
    auto const hidden_out = inputB.sizes()[0];

    auto stream = at::cuda::getCurrentCUDAStream(inputA.get_device());
    auto output = torch::empty(
        {num_tokens, hidden_out}, torch::TensorOptions().dtype(c10::ScalarType::BFloat16).device(inputA.device()));

    tensorrt_llm::kernels::llama4_min_latency::llama4_fp8_bf16_gemm::llama4_fp8_bf16_gemm_op(inputA.data_ptr(),
        inputB.data_ptr(), output.data_ptr(), scaling_factor.data_ptr(),
        has_position_ids ? position_ids.value().data_ptr() : nullptr, position_ids_int64, num_tokens, hidden_in,
        hidden_out, stream);

    return output;
}

torch::Tensor llama4_fp8_fp8_gemm_swiglu(torch::Tensor const& inputA, torch::Tensor const& inputB,
    torch::Tensor const& in_scale, torch::Tensor const& out_scale_inv)
{
    CHECK_INPUT(inputA, c10::ScalarType::Float8_e4m3fn);
    CHECK_INPUT(inputB, c10::ScalarType::Float8_e4m3fn);
    CHECK_INPUT(in_scale, c10::ScalarType::Float);
    CHECK_INPUT(out_scale_inv, c10::ScalarType::Float);

    TORCH_CHECK(inputA.dim() == 2, "inputA must be 2D.");
    TORCH_CHECK(inputB.dim() == 2, "inputB must be 2D.");
    TORCH_CHECK(inputA.sizes()[1] == 5120, "inputA.size(1) must be 5120");
    TORCH_CHECK(inputB.sizes()[1] == 5120, "inputB.size(1) must be 5120");
    TORCH_CHECK(in_scale.dim() == 0, "in_scale must be a scalar tensor");
    TORCH_CHECK(out_scale_inv.dim() == 0, "out_scale_inv must be a scalar tensor");

    auto stream = at::cuda::getCurrentCUDAStream(inputA.get_device());
    auto const num_tokens = inputA.sizes()[0];
    auto const hidden_in = inputA.sizes()[1];
    auto const hidden_out = inputB.sizes()[0] / 2;
    auto output = torch::empty(
        {num_tokens, hidden_out}, torch::TensorOptions().dtype(c10::ScalarType::Float8_e4m3fn).device(inputA.device()));

    tensorrt_llm::kernels::llama4_min_latency::llama4_fp8_fp8_gemm_swiglu::llama4_fp8_fp8_gemm_swiglu_op(num_tokens,
        hidden_in, hidden_out, inputA.data_ptr(), inputB.data_ptr(), output.data_ptr(), in_scale.data_ptr(),
        out_scale_inv.data_ptr(), stream);

    return output;
}

torch::Tensor llama4_moe_tp8ep1_min_latency(torch::Tensor const& input, torch::Tensor const& router_logits,
    torch::Tensor const& fc1_expert_weights, torch::Tensor const& fc2_expert_weights,
    torch::optional<c10::ArrayRef<torch::Tensor>> const& quant_scales)
{
    CHECK_INPUT(input, c10::ScalarType::Float8_e4m3fn)
    CHECK_INPUT(router_logits, c10::ScalarType::BFloat16)
    CHECK_INPUT(fc1_expert_weights, c10::ScalarType::Float8_e4m3fn)
    CHECK_INPUT(fc2_expert_weights, c10::ScalarType::Float8_e4m3fn)

    TORCH_CHECK(input.dim() == 2, "input must be 2D.");
    TORCH_CHECK(router_logits.dim() == 2, "router_logits must be 2D.");
    TORCH_CHECK(input.sizes()[0] == router_logits.sizes()[0], "input and router_logits must have the same num tokens.");

    TORCH_CHECK(fc1_expert_weights.dim() == 3, "fc1_expert_weights must be 3D.");
    TORCH_CHECK(fc2_expert_weights.dim() == 3, "fc2_expert_weights must be 3D.");
    TORCH_CHECK(fc1_expert_weights.sizes()[0] == fc2_expert_weights.sizes()[0],
        "fc1_expert_weights and fc2_expert_weights must have the same number of experts.");
    TORCH_CHECK(fc1_expert_weights.sizes()[1] == fc2_expert_weights.sizes()[2] * 2,
        "fc1_expert_weights inter size must be 2 times fc2_expert_weights inter size.");
    TORCH_CHECK(router_logits.sizes()[0] == input.sizes()[0], "router_logits and input must have the same num tokens.");
    TORCH_CHECK(router_logits.sizes()[1] == 128, "router_logits must have 128 experts.");
    TORCH_CHECK(input.sizes()[1] == 5120, "input must have 5120 hidden size.");
    TORCH_CHECK(fc1_expert_weights.sizes()[2] == 5120, "fc1_expert_weights must have 5120 hidden size.");

    int64_t num_rows = input.sizes()[0];
    int64_t hidden_size = fc2_expert_weights.sizes()[1];
    int64_t inter_size = fc2_expert_weights.sizes()[2];
    int64_t num_experts = fc2_expert_weights.sizes()[0];

    TORCH_CHECK(quant_scales.has_value(), "Expecting quant scales for fp8 quantization");
    TORCH_CHECK(quant_scales.value().size() == 4, "Expecting 4 quant scales for fp8 quantization");

    auto const fc1_dequant = quant_scales.value()[0];
    auto const fc2_quant = quant_scales.value()[1];
    auto const fc2_dequant = quant_scales.value()[2];

    CHECK_INPUT(fc1_dequant, c10::ScalarType::Float);
    CHECK_INPUT(fc2_quant, c10::ScalarType::Float);
    CHECK_INPUT(fc2_dequant, c10::ScalarType::Float);
    TORCH_CHECK(fc1_dequant.dim() == 1, "fc1 dequant must be 1D");
    TORCH_CHECK(fc2_quant.dim() == 0, "fc2 quant must be a scalar tensor");
    TORCH_CHECK(fc2_dequant.dim() == 1, "fc2 dequant must be 1D");
    TORCH_CHECK(fc1_dequant.sizes()[0] == num_experts, "fc1 dequant size must be (num_experts,)");
    TORCH_CHECK(fc2_dequant.sizes()[0] == num_experts, "fc2 dequant size must be (num_experts,)");

    auto stream = at::cuda::getCurrentCUDAStream(input.get_device());

    std::vector<int64_t> fc2_input_shape = {num_rows, inter_size};
    auto fc2_input = torch::empty(fc2_input_shape, input.options().dtype(c10::ScalarType::Float8_e4m3fn));
    std::vector<int64_t> exp_idx_shape = {num_rows};
    auto exp_idx = torch::empty(exp_idx_shape, input.options().dtype(c10::ScalarType::Int));
    std::vector<int64_t> output_shape = {num_rows, hidden_size};
    auto output = torch::empty(output_shape, input.options().dtype(c10::ScalarType::BFloat16));

    tensorrt_llm::kernels::llama4_min_latency::llama4_moe::run_moe_llama4_tp8ep1_min_latency(num_rows, num_experts,
        input.const_data_ptr(), router_logits.const_data_ptr(), fc1_expert_weights.const_data_ptr(),
        fc2_expert_weights.const_data_ptr(), static_cast<float const*>(fc1_dequant.data_ptr()),
        static_cast<float const*>(fc2_quant.data_ptr()), static_cast<float const*>(fc2_dequant.data_ptr()),
        fc2_input.data_ptr(), static_cast<int*>(exp_idx.data_ptr()), output.data_ptr(), stream);

    return output;
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("llama4_bf16_bf16_gemm(Tensor inputA, Tensor inputB) -> Tensor");
    m.def(
        "llama4_fp8_bf16_gemm(Tensor inputA, Tensor inputB, "
        "Tensor scaling_factor, Tensor? position_ids=None) -> Tensor");
    m.def("llama4_fp8_fp8_gemm_swiglu(Tensor inputA, Tensor inputB, Tensor in_scale, Tensor out_scale_inv) -> Tensor");
    m.def(
        "llama4_moe_tp8ep1_min_latency(Tensor input, Tensor router_logits, "
        "Tensor fc1_expert_weights, Tensor fc2_expert_weights, "
        "Tensor[]? quant_scales=None) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("llama4_bf16_bf16_gemm", &torch_ext::llama4_bf16_bf16_gemm);
    m.impl("llama4_fp8_bf16_gemm", &torch_ext::llama4_fp8_bf16_gemm);
    m.impl("llama4_fp8_fp8_gemm_swiglu", &torch_ext::llama4_fp8_fp8_gemm_swiglu);
    m.impl("llama4_moe_tp8ep1_min_latency", &torch_ext::llama4_moe_tp8ep1_min_latency);
}

} // namespace torch_ext
