/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION &
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
#include "tensorrt_llm/kernels/llama4GemmSwiGLU.h"
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"

#include <nvml.h>
#include <torch/extension.h>

// using namespace nvinfer1;
using tensorrt_llm::kernels::llama4_fc_swiglu::llama4_fc_swiglu_fp8_op;
using tensorrt_llm::kernels::llama4_fc_swiglu::llama4_fc_swiglu_bf16_op;

namespace torch_ext
{

namespace
{
class Llama4GemmSwiGLUOp
{
public:
    Llama4GemmSwiGLUOp()
    {
    }

    ~Llama4GemmSwiGLUOp() = default;

    torch::Tensor run_fp8(
        torch::Tensor inputA, torch::Tensor inputB, torch::Tensor in_scale, torch::Tensor out_scale_inv) noexcept
    {
        auto stream = at::cuda::getCurrentCUDAStream(inputA.get_device());
        auto hidden_out = inputB.size(1) / 2;
        auto output = torch::empty(
            {inputA.size(0), hidden_out}, torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(inputA.device()));

        llama4_fc_swiglu_fp8_op(inputA.size(0), hidden_out, inputA.data_ptr(), inputB.data_ptr(), output.data_ptr(),
            in_scale.data_ptr(), out_scale_inv.data_ptr(), stream);

        return output;
    }

    torch::Tensor run_bf16(
        torch::Tensor inputA, torch::Tensor inputB, torch::Tensor in_scale, torch::Tensor out_scale_inv) noexcept
    {
        auto stream = at::cuda::getCurrentCUDAStream(inputA.get_device());
        auto hidden_out = inputB.size(1) / 2;
        auto output = torch::empty(
            {inputA.size(0), hidden_out}, torch::TensorOptions().dtype(torch::kBFloat16).device(inputA.device()));

        llama4_fc_swiglu_bf16_op(inputA.size(0), hidden_out, inputA.data_ptr(), inputB.data_ptr(), output.data_ptr(),
            in_scale.data_ptr(), out_scale_inv.data_ptr(), stream);

        return output;
    }

    int initialize() noexcept {
        return 0;
    }

};
} // namespace

torch::Tensor llama4_fc_swiglu_fp8(torch::Tensor inputA, torch::Tensor inputB, torch::Tensor in_scale, torch::Tensor out_scale_inv) {
    Llama4GemmSwiGLUOp op;
    return op.run_fp8(inputA, inputB, in_scale, out_scale_inv);
}

torch::Tensor llama4_fc_swiglu_bf16(
    torch::Tensor inputA, torch::Tensor inputB, torch::Tensor in_scale, torch::Tensor out_scale_inv)
{
    Llama4GemmSwiGLUOp op;
    return op.run_bf16(inputA, inputB, in_scale, out_scale_inv);
}

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("llama4_fc_swiglu_fp8(Tensor inputA, Tensor inputB, Tensor in_scale, Tensor out_scale_inv) -> Tensor");
    m.def("llama4_fc_swiglu_bf16(Tensor inputA, Tensor inputB, Tensor in_scale, Tensor out_scale_inv) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("llama4_fc_swiglu_fp8", &torch_ext::llama4_fc_swiglu_fp8);
    m.impl("llama4_fc_swiglu_bf16", &torch_ext::llama4_fc_swiglu_bf16);
}
