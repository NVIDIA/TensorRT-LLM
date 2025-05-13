/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/kernels/groupRmsNormKernels/groupRmsNormKernels.h"
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

void groupRMSNormBase(torch::TensorList const& inputs, torch::TensorList const& outputs,
    torch::TensorList const& weights, double eps, double weight_bias)
{
    TORCH_CHECK(!inputs.empty(), "Input tensor list cannot be empty.");
    TORCH_CHECK(!outputs.empty(), "Output tensor list cannot be empty.");
    uint32_t const num_inputs = inputs.size();
    TORCH_CHECK(num_inputs <= 2, "Only up to 2 inputs are supported.");
    auto const first_input = inputs[0];
    TORCH_CHECK(first_input.dim() == 2, "Inputs must be 2D tensors [batch_size, hidden_dim].");
    TORCH_CHECK(first_input.sizes()[0] > 0, "Batch size must be greater than 0.");
    TORCH_CHECK(first_input.is_cuda(), "Inputs must be CUDA tensors.");
    uint32_t const batch_size = first_input.sizes()[0];
    auto const dtype = first_input.scalar_type();
    auto const device = first_input.device();

    if (!weights.empty())
    {
        TORCH_CHECK(weights.size() == num_inputs, "Weights list size must match inputs list size.");
    }

    for (size_t i = 0; i < num_inputs; ++i)
    {
        TORCH_CHECK(inputs[i].sizes()[0] == batch_size, "Inputs must have the same batch size.");
        TORCH_CHECK(inputs[i].dim() == 2, "Inputs must be 2D tensors [batch_size, hidden_dim].");
        TORCH_CHECK(inputs[i].device() == device, "Inputs must be on the same device.");
        TORCH_CHECK(inputs[i].scalar_type() == dtype, "Inputs must be of the same type.");
        TORCH_CHECK(outputs[i].dim() == 2, "Outputs must be 2D tensors [batch_size, hidden_dim].");
        TORCH_CHECK(outputs[i].device() == device, "Outputs must be on the same device.");
        TORCH_CHECK(outputs[i].scalar_type() == dtype, "Outputs must be of the same type.");
        TORCH_CHECK(outputs[i].sizes()[0] == batch_size, "Outputs must have the same batch size.");
        TORCH_CHECK(
            outputs[i].sizes()[1] == inputs[i].sizes()[1], "Outputs and inputs must have the same last dimension.");
        TORCH_CHECK(inputs[i].strides()[0] == outputs[i].strides()[0], "Inputs and outputs must have the same stride.");
        TORCH_CHECK(inputs[i].strides()[1] == 1, "Inputs must be contiguous along the last dimension.");
        TORCH_CHECK(outputs[i].strides()[1] == 1, "Outputs must be contiguous along the last dimension.");
        if (!weights.empty())
        {
            TORCH_CHECK(
                inputs[i].sizes()[1] == weights[i].sizes()[0], "Inputs and weights must have the same last dimension.");
        }
    }

#define DISPATCH_INPUT_SIZES(n)                                                                                        \
    {                                                                                                                  \
        tensorrt_llm::kernels::group_rms_norm::GroupRMSParams<n> params;                                               \
        for (size_t i = 0; i < n; ++i)                                                                                 \
        {                                                                                                              \
            params.inputs[i] = reinterpret_cast<float4*>(inputs[i].data_ptr());                                        \
            params.input_last_dims[i] = inputs[i].sizes()[1];                                                          \
            params.input_strides[i] = inputs[i].strides()[0];                                                          \
            params.output_strides[i] = outputs[i].strides()[0];                                                        \
            if (!weights.empty())                                                                                      \
            {                                                                                                          \
                params.weights[i] = reinterpret_cast<float4 const*>(weights[i].data_ptr());                            \
            }                                                                                                          \
            params.outputs[i] = reinterpret_cast<float4*>(outputs[i].mutable_data_ptr());                              \
        }                                                                                                              \
        /* Set remaining params */                                                                                     \
        params.batch_size = batch_size;                                                                                \
        params.num_inputs = n;                                                                                         \
        params.eps = static_cast<float>(eps);                                                                          \
        params.weight_bias = static_cast<float>(weight_bias);                                                          \
        params.enable_weights = !weights.empty();                                                                      \
        params.stream = at::cuda::getCurrentCUDAStream(inputs[0].get_device());                                        \
        /* Handle dtype conversion */                                                                                  \
        switch (dtype)                                                                                                 \
        {                                                                                                              \
        case torch::ScalarType::Half: params.dtype = nvinfer1::DataType::kHALF; break;                                 \
        case torch::ScalarType::BFloat16: params.dtype = nvinfer1::DataType::kBF16; break;                             \
        case torch::ScalarType::Float: params.dtype = nvinfer1::DataType::kFLOAT; break;                               \
        default: TORCH_CHECK(false, "Unsupported data type");                                                          \
        }                                                                                                              \
        tensorrt_llm::kernels::group_rms_norm::GroupRMSNormBaseKernelLauncher<n>(params);                              \
        break;                                                                                                         \
    }

    switch (num_inputs)
    {
    case 1: DISPATCH_INPUT_SIZES(1)
    case 2: DISPATCH_INPUT_SIZES(2)
    default: TORCH_CHECK(false, "Unsupported number of inputs (max 2)");
    }
#undef DISPATCH_INPUT_SIZES
}

void groupRMSNormLargeBatch(torch::TensorList const& inputs, torch::TensorList const& outputs,
    torch::TensorList const& weights, double eps, double weight_bias)
{
    TORCH_CHECK(!inputs.empty(), "Input tensor list cannot be empty.");
    TORCH_CHECK(!outputs.empty(), "Output tensor list cannot be empty.");
    TORCH_CHECK(inputs.size() == 2, "groupRMSNormLargeBatch requires exactly 2 input tensors.");
    auto const first_input = inputs[0];
    TORCH_CHECK(first_input.dim() == 2, "Inputs must be 2D tensors [batch_size, hidden_dim].");
    TORCH_CHECK(first_input.sizes()[0] > 0, "Batch size must be greater than 0.");
    TORCH_CHECK(first_input.is_cuda(), "Inputs must be CUDA tensors.");
    uint32_t const batch_size = first_input.sizes()[0];
    auto const dtype = first_input.scalar_type();
    auto const device = first_input.device();
    uint32_t const num_inputs = inputs.size();

    for (size_t i = 0; i < num_inputs; ++i)
    {
        TORCH_CHECK(inputs[i].sizes()[0] == batch_size, "Inputs must have the same batch size.");
        TORCH_CHECK(inputs[i].dim() == 2, "Inputs must be 2D tensors [batch_size, hidden_dim].");
        TORCH_CHECK(inputs[i].device() == device, "Inputs must be on the same device.");
        TORCH_CHECK(inputs[i].scalar_type() == dtype, "Inputs must be of the same type.");
        TORCH_CHECK(outputs[i].dim() == 2, "Outputs must be 2D tensors [batch_size, hidden_dim].");
        TORCH_CHECK(outputs[i].device() == device, "Outputs must be on the same device.");
        TORCH_CHECK(outputs[i].scalar_type() == dtype, "Outputs must be of the same type.");
        TORCH_CHECK(outputs[i].sizes()[0] == batch_size, "Outputs must have the same batch size.");
        TORCH_CHECK(
            outputs[i].sizes()[1] == inputs[i].sizes()[1], "Outputs and inputs must have the same last dimension.");
        TORCH_CHECK(inputs[i].strides()[0] == outputs[i].strides()[0], "Inputs and outputs must have the same stride.");
        TORCH_CHECK(inputs[i].strides()[1] == 1, "Inputs must be contiguous along the last dimension.");
        TORCH_CHECK(outputs[i].strides()[1] == 1, "Outputs must be contiguous along the last dimension.");
        if (!weights.empty())
        {
            TORCH_CHECK(
                inputs[i].sizes()[1] == weights[i].sizes()[0], "Inputs and weights must have the same last dimension.");
        }
    }

    tensorrt_llm::kernels::group_rms_norm::GroupRMSParams<2> params;

    for (size_t i = 0; i < 2; ++i)
    {
        params.inputs[i] = reinterpret_cast<float4*>(inputs[i].data_ptr());
        params.input_last_dims[i] = inputs[i].sizes()[1];
        params.input_strides[i] = inputs[i].strides()[0];
        params.output_strides[i] = outputs[i].strides()[0];
        params.outputs[i] = reinterpret_cast<float4*>(outputs[i].mutable_data_ptr());
        if (!weights.empty())
        {
            params.weights[i] = reinterpret_cast<float4 const*>(weights[i].data_ptr());
        }
    }

    // Set remaining params
    params.batch_size = batch_size;
    params.num_inputs = 2;
    params.eps = static_cast<float>(eps);
    params.weight_bias = static_cast<float>(weight_bias);
    params.enable_weights = !weights.empty();
    params.stream = at::cuda::getCurrentCUDAStream(inputs[0].get_device());

    // Handle dtype conversion
    switch (dtype)
    {
    case torch::ScalarType::Half: params.dtype = nvinfer1::DataType::kHALF; break;
    case torch::ScalarType::BFloat16: params.dtype = nvinfer1::DataType::kBF16; break;
    case torch::ScalarType::Float: params.dtype = nvinfer1::DataType::kFLOAT; break;
    default: TORCH_CHECK(false, "Unsupported data type");
    }

    tensorrt_llm::kernels::group_rms_norm::GroupRMSNormKernelLargeBatchLauncher<2>(params);
}

void groupRMSNormHeuristic(torch::TensorList const& inputs, torch::TensorList const& outputs,
    torch::TensorList const& weights, double eps, double weight_bias)
{
    TORCH_CHECK(!inputs.empty(), "Input tensor list cannot be empty.");
    TORCH_CHECK(!outputs.empty(), "Output tensor list cannot be empty.");
    uint32_t const num_inputs = inputs.size();
    TORCH_CHECK(num_inputs <= 2, "Heuristic kernel only supports up to 2 input tensors.");
    auto const first_input = inputs[0];
    TORCH_CHECK(first_input.dim() == 2, "Inputs must be 2D tensors [batch_size, hidden_dim].");
    TORCH_CHECK(first_input.sizes()[0] > 0, "Batch size must be greater than 0.");
    TORCH_CHECK(first_input.is_cuda(), "Inputs must be CUDA tensors.");
    uint32_t const batch_size = first_input.sizes()[0];
    auto const dtype = first_input.scalar_type();
    auto const device = first_input.device();

    if (!weights.empty())
    {
        TORCH_CHECK(weights.size() == num_inputs, "Weights list size must match inputs list size.");
    }

    for (size_t i = 0; i < num_inputs; ++i)
    {
        TORCH_CHECK(inputs[i].sizes()[0] == batch_size, "Inputs must have the same batch size.");
        TORCH_CHECK(inputs[i].dim() == 2, "Inputs must be 2D tensors [batch_size, hidden_dim].");
        TORCH_CHECK(inputs[i].device() == device, "Inputs must be on the same device.");
        TORCH_CHECK(inputs[i].scalar_type() == dtype, "Inputs must be of the same type.");
        TORCH_CHECK(outputs[i].dim() == 2, "Outputs must be 2D tensors [batch_size, hidden_dim].");
        TORCH_CHECK(outputs[i].device() == device, "Outputs must be on the same device.");
        TORCH_CHECK(outputs[i].scalar_type() == dtype, "Outputs must be of the same type.");
        TORCH_CHECK(outputs[i].sizes()[0] == batch_size, "Outputs must have the same batch size.");
        TORCH_CHECK(
            outputs[i].sizes()[1] == inputs[i].sizes()[1], "Outputs and inputs must have the same last dimension.");
        TORCH_CHECK(inputs[i].strides()[0] == outputs[i].strides()[0], "Inputs and outputs must have the same stride.");
        TORCH_CHECK(inputs[i].strides()[1] == 1, "Inputs must be contiguous along the last dimension.");
        TORCH_CHECK(outputs[i].strides()[1] == 1, "Outputs must be contiguous along the last dimension.");
        if (!weights.empty())
        {
            TORCH_CHECK(
                inputs[i].sizes()[1] == weights[i].sizes()[0], "Inputs and weights must have the same last dimension.");
        }
    }

    // Dispatch based on number of inputs - templates require compile-time constants
#define DISPATCH_HEURISTIC_INPUTS(n)                                                                                   \
    {                                                                                                                  \
        tensorrt_llm::kernels::group_rms_norm::GroupRMSParams<n> params;                                               \
        for (size_t i = 0; i < n; ++i)                                                                                 \
        {                                                                                                              \
            params.inputs[i] = reinterpret_cast<float4*>(inputs[i].data_ptr());                                        \
            params.input_last_dims[i] = inputs[i].sizes()[1];                                                          \
            params.input_strides[i] = inputs[i].strides()[0];                                                          \
            params.output_strides[i] = outputs[i].strides()[0];                                                        \
            params.outputs[i] = reinterpret_cast<float4*>(outputs[i].mutable_data_ptr());                              \
            if (!weights.empty())                                                                                      \
            {                                                                                                          \
                params.weights[i] = reinterpret_cast<float4 const*>(weights[i].data_ptr());                            \
            }                                                                                                          \
        }                                                                                                              \
                                                                                                                       \
        /* Set remaining params */                                                                                     \
        params.batch_size = batch_size;                                                                                \
        params.num_inputs = n;                                                                                         \
        params.eps = static_cast<float>(eps);                                                                          \
        params.weight_bias = static_cast<float>(weight_bias);                                                          \
        params.enable_weights = !weights.empty();                                                                      \
        params.stream = at::cuda::getCurrentCUDAStream(inputs[0].get_device());                                        \
                                                                                                                       \
        /* Handle dtype conversion */                                                                                  \
        switch (dtype)                                                                                                 \
        {                                                                                                              \
        case torch::ScalarType::Half: params.dtype = nvinfer1::DataType::kHALF; break;                                 \
        case torch::ScalarType::BFloat16: params.dtype = nvinfer1::DataType::kBF16; break;                             \
        case torch::ScalarType::Float: params.dtype = nvinfer1::DataType::kFLOAT; break;                               \
        default: TORCH_CHECK(false, "Unsupported data type");                                                          \
        }                                                                                                              \
                                                                                                                       \
        /* Use the heuristic launcher that will decide between regular and large batch kernels */                      \
        tensorrt_llm::kernels::group_rms_norm::GroupRMSNormKernelLauncherWithHeuristic<n>(params);                     \
        break;                                                                                                         \
    }

    switch (num_inputs)
    {
    case 1: DISPATCH_HEURISTIC_INPUTS(1)
    case 2: DISPATCH_HEURISTIC_INPUTS(2)
    default: TORCH_CHECK(false, "Unsupported number of inputs (max 2)");
    }
#undef DISPATCH_HEURISTIC_INPUTS
}

} // namespace torch_ext

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("group_rms_norm_base", &torch_ext::groupRMSNormBase);
    m.impl("group_rms_norm_large_batch", &torch_ext::groupRMSNormLargeBatch);
    // Use groupRMSNormHeuristic which automatically selects between regular and large batch kernels
    m.impl("group_rms_norm_heuristic", &torch_ext::groupRMSNormHeuristic);
}
