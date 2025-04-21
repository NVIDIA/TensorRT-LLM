/*
 * Copyright (c) 2025-2025, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/kernels/llama4QKVGemm.h"
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

// using namespace nvinfer1;
using tensorrt_llm::kernels::llama4_qkv_gemm::llama4_qkv_gemm_op;

namespace torch_ext
{

namespace
{
class Llama4QKVGemmOp
{
public:
    Llama4QKVGemmOp() {}

    ~Llama4QKVGemmOp() = default;

    torch::Tensor run(torch::Tensor inputA, torch::Tensor inputB, torch::Tensor scaling_factor,
                      torch::optional<torch::Tensor> position_ids) noexcept
    {
        auto stream = at::cuda::getCurrentCUDAStream(inputA.get_device());
        auto output = torch::empty(
            {inputA.size(0), inputB.size(1)}, torch::TensorOptions().dtype(torch::kBFloat16).device(inputA.device()));

        void const* position_ids_ptr = position_ids.has_value() ? position_ids.value().const_data_ptr() : nullptr;
        llama4_qkv_gemm_op(
            inputA.data_ptr(), inputB.data_ptr(), output.data_ptr(), scaling_factor.data_ptr(),
            position_ids_ptr, inputA.size(0), inputB.size(0), inputB.size(1), stream);

        return output;
    }

    int initialize() noexcept
    {
        return 0;
    }
};
} // namespace

torch::Tensor llama4_qkv_gemm(torch::Tensor const& inputA, torch::Tensor const& inputB,
                              torch::Tensor const& scaling_factor,
                              torch::optional<torch::Tensor> position_ids)
{
    Llama4QKVGemmOp op;
    return op.run(inputA, inputB, scaling_factor, position_ids);
}

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("llama4_qkv_gemm(Tensor inputA, Tensor inputB, "
          "Tensor scaling_factor, Tensor? position_ids=None) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("llama4_qkv_gemm", &torch_ext::llama4_qkv_gemm);
}
