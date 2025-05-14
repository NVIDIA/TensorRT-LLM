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

#include "tensorrt_llm/kernels/llama4RouterGemm.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/dataType.h"
#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"

#include <nvml.h>
#include <torch/extension.h>

#include <cstddef>
#include <cstdint>
#include <unordered_set>

#define NUM_EXPERT 128

// using namespace nvinfer1;
using tensorrt_llm::kernels::llama4_router_gemm::llama4_router_gemm_op;

namespace torch_ext
{

namespace
{
class Llama4RouterGemmOp
{
public:
    Llama4RouterGemmOp() {}

    ~Llama4RouterGemmOp() = default;

    torch::Tensor run(torch::Tensor inputA, torch::Tensor inputB) noexcept
    {
        auto stream = at::cuda::getCurrentCUDAStream(inputA.get_device());
        auto output = torch::empty(
            {inputA.size(0), NUM_EXPERT}, torch::TensorOptions().dtype(inputA.dtype()).device(inputA.device()));

        llama4_router_gemm_op(inputA.size(0), inputA.data_ptr(), inputB.data_ptr(), output.data_ptr(), stream);

        return output;
    }

    int initialize() noexcept
    {
        return 0;
    }
};
} // namespace

torch::Tensor llama4_router_gemm(torch::Tensor inputA, torch::Tensor inputB)
{
    Llama4RouterGemmOp op;
    return op.run(inputA, inputB);
}

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("llama4_router_gemm(Tensor inputA, Tensor inputB) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("llama4_router_gemm", &torch_ext::llama4_router_gemm);
}
