/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/common/dataType.h"
#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/kernels/groupRmsNormKernels.h"
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAContext.h>
#include <cstdint>
#include <cuda_runtime.h>
#include <stdexcept>
#include <torch/extension.h>
#include <vector>

namespace torch_ext
{

template <typename DType>
void groupRMSNormImpl(torch::TensorList const& inputs, torch::TensorList const& weights,
    std::vector<torch::Tensor>& outputs, std::vector<uint32_t> input_dims, std::vector<uint32_t> input_strides,
    std::vector<uint32_t> output_strides, uint32_t batch_size, uint32_t num_inputs, double eps, double weight_bias,
    bool enable_weights, cudaStream_t stream)
{
    // Prepare pointer arrays on host
    std::vector<DType*> inputs_ptr(num_inputs);
    std::vector<DType*> weights_ptr(num_inputs, nullptr); // Initialize with nullptr
    std::vector<DType*> outputs_ptr(num_inputs);

    for (uint32_t i = 0; i < num_inputs; ++i)
    {
        inputs_ptr[i] = reinterpret_cast<DType*>(inputs[i].data_ptr());
        if (enable_weights)
        {
            weights_ptr[i] = reinterpret_cast<DType*>(weights[i].data_ptr());
        }
        outputs_ptr[i] = reinterpret_cast<DType*>(outputs[i].mutable_data_ptr());
    }

    // Pass raw pointers to vectors
    tensorrt_llm::kernels::group_rms_norm::GroupRMSNormKernel(inputs_ptr.data(), weights_ptr.data(), outputs_ptr.data(),
        input_dims.data(), input_strides.data(), output_strides.data(), batch_size, num_inputs, static_cast<float>(eps),
        static_cast<float>(weight_bias), enable_weights, stream);
}

std::vector<torch::Tensor> groupRMSNorm(torch::TensorList const& inputs, torch::TensorList const& weights, double eps,
    double weight_bias, bool enable_weights)
{
    TORCH_CHECK(inputs.size() > 0, "Input tensor list cannot be empty.");
    auto const first_input = inputs[0];
    auto const dtype = first_input.scalar_type();
    auto const device = first_input.device();
    uint32_t const num_inputs = inputs.size();
    uint32_t const batch_size = first_input.sizes()[0];

    TORCH_CHECK(first_input.dim() == 2, "Inputs must be 2D tensors [batch_size, hidden_dim].");
    TORCH_CHECK(first_input.is_cuda(), "Inputs must be CUDA tensors.");

    if (enable_weights)
    {
        TORCH_CHECK(weights.size() == num_inputs, "Weights list size must match inputs list size.");
    }

    std::vector<torch::Tensor> outputs;
    std::vector<uint32_t> input_dims(num_inputs);
    std::vector<uint32_t> input_strides(num_inputs);
    std::vector<uint32_t> output_strides(num_inputs);
    outputs.reserve(num_inputs);

    for (size_t i = 0; i < num_inputs; ++i)
    {
        auto const& current_input = inputs[i];
        TORCH_CHECK(current_input.dim() == 2, "All inputs must be 2D tensors.");
        TORCH_CHECK(current_input.sizes()[0] == batch_size, "All inputs must have the same batch size.");
        TORCH_CHECK(current_input.scalar_type() == dtype, "All inputs must have the same data type.");
        TORCH_CHECK(current_input.device() == device, "All inputs must be on the same CUDA device.");

        input_dims[i] = current_input.sizes()[1];
        input_strides[i] = current_input.strides()[0];
        output_strides[i] = current_input.strides()[0];

        if (enable_weights)
        {
            auto const& current_weight = weights[i];
            TORCH_CHECK(current_weight.dim() == 1, "Weights must be 1D tensors.");
            TORCH_CHECK(
                current_weight.sizes()[0] == current_input.sizes()[1], "Weight dimension must match input dimension.");
            TORCH_CHECK(current_weight.scalar_type() == dtype, "Weights must have the same data type as inputs.");
            TORCH_CHECK(current_weight.device() == device, "Weights must be on the same CUDA device as inputs.");
        }
        outputs.push_back(torch::empty_like(current_input));
        TORCH_CHECK(outputs[i].device() == device, "Outputs must be on the same CUDA device as inputs.");
    }

    auto stream = at::cuda::getCurrentCUDAStream(inputs[0].get_device());
    switch (dtype)
    {
    case torch::ScalarType::Half:
        groupRMSNormImpl<half>(inputs, weights, outputs, input_dims, input_strides, output_strides, batch_size,
            num_inputs, eps, weight_bias, enable_weights, stream);
        break;
#ifdef ENABLE_BF16
    case torch::ScalarType::BFloat16:
        groupRMSNormImpl<__nv_bfloat16>(inputs, weights, outputs, input_dims, input_strides, output_strides, batch_size,
            num_inputs, eps, weight_bias, enable_weights, stream);
        break;
#endif
    case torch::ScalarType::Float:
        groupRMSNormImpl<float>(inputs, weights, outputs, input_dims, input_strides, output_strides, batch_size,
            num_inputs, eps, weight_bias, enable_weights, stream);
        break;
    default: TORCH_CHECK(false, "Unsupported data type for GroupRMSNorm");
    }
    return outputs;
}

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "group_rms_norm(Tensor[] inputs, Tensor[] weights, float eps, float weight_bias, bool enable_weights) -> "
        "Tensor[]");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("group_rms_norm", &torch_ext::groupRMSNorm);
}
