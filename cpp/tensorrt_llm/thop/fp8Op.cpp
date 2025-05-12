/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/thop/fp8Op.h"
#include "tensorrt_llm/common/cudaBf16Wrapper.h"
#include "tensorrt_llm/common/cudaFp8Utils.h"
#include "tensorrt_llm/thop/thUtils.h"
#include <torch/extension.h>

#if defined(TORCH_VERSION_MAJOR)                                                                                       \
    && ((TORCH_VERSION_MAJOR > 1) || ((TORCH_VERSION_MAJOR == 1) && (TORCH_VERSION_MINOR >= 9)))
#define TORCH_IS_AT_LEAST_v190
#endif

namespace torch_ext
{
using torch::Tensor;
using at::cuda::CUDAStream;
using namespace tensorrt_llm::common;

void e4m3_dynamic_quantize(
    Tensor& input, Tensor& quantized_input, Tensor& scales, CUDAStream& stream, QuantizeMode& quantize_mode)
{
    auto quantized_input_ptr = get_ptr<__nv_fp8_e4m3>(quantized_input);

    if (input.scalar_type() == at::ScalarType::Float)
    {
        invokeComputeScalesAndQuantizeMatrix(quantized_input_ptr, get_ptr<float>(scales), get_ptr<float const>(input),
            input.numel(), input.size(-1), quantize_mode, stream);
    }
    else if (input.scalar_type() == at::ScalarType::Half)
    {
        invokeComputeScalesAndQuantizeMatrix(quantized_input_ptr, get_ptr<half>(scales), get_ptr<half const>(input),
            input.numel(), input.size(-1), quantize_mode, stream);
    }
#ifdef ENABLE_BF16
    else if (input.scalar_type() == at::ScalarType::BFloat16)
    {
        invokeComputeScalesAndQuantizeMatrix(quantized_input_ptr, get_ptr<__nv_bfloat16>(scales),
            get_ptr<__nv_bfloat16 const>(input), input.numel(), input.size(-1), quantize_mode, stream);
    }
#endif
    else
    {
        TORCH_CHECK(false, "Invalid datatype. input must be BF16/FP16/FP32");
    }
}

void e4m3_static_quantize(
    Tensor& input, Tensor& quantized_input, Tensor& scales, CUDAStream& stream, QuantizeMode& quantize_mode)
{
    auto quantized_input_ptr = get_ptr<__nv_fp8_e4m3>(quantized_input);

    if (input.scalar_type() == at::ScalarType::Float)
    {
        invokeQuantizeMatrix(quantized_input_ptr, get_ptr<float>(scales), get_ptr<float const>(input), input.numel(),
            input.size(-1), quantize_mode, stream);
    }
    else if (input.scalar_type() == at::ScalarType::Half)
    {
        invokeQuantizeMatrix(quantized_input_ptr, get_ptr<float>(scales), get_ptr<half const>(input), input.numel(),
            input.size(-1), quantize_mode, stream);
    }
#ifdef ENABLE_BF16
    else if (input.scalar_type() == at::ScalarType::BFloat16)
    {
        invokeQuantizeMatrix(quantized_input_ptr, get_ptr<float>(scales), get_ptr<__nv_bfloat16 const>(input),
            input.numel(), input.size(-1), quantize_mode, stream);
    }
#endif
    else
    {
        TORCH_CHECK(false, "Invalid datatype. input must be BF16/FP16/FP32");
    }
}

std::tuple<Tensor, Tensor> e4m3_quantize_helper(Tensor input, at::optional<Tensor> scales, QuantizeMode quantize_mode)
{
    CHECK_CONTIGUOUS(input);
    CHECK_TH_CUDA(input);
    TORCH_CHECK(input.numel() != 0, "input should not be empty tensor");
    TORCH_CHECK(input.dim() >= 2 && (quantize_mode != QuantizeMode::PER_CHANNEL || input.dim() == 2),
        "Invalid dim. The dim of input should be greater than or equal to 2");

    auto _st = input.scalar_type();
    TORCH_CHECK(_st == torch::kFloat32 || _st == torch::kFloat16 || _st == torch::kBFloat16,
        "Invalid datatype. input must be FP16 or BF16 or FP32");

    Tensor quantized_input
        = torch::empty(input.sizes(), torch::dtype(torch::kFloat8_e4m3fn).device(torch::kCUDA).requires_grad(false));
    Tensor scales_;

    auto stream = at::cuda::getCurrentCUDAStream(input.get_device());

    if (scales.has_value())
    {
        // static quantization will use float scales by default.
        scales_ = scales.value();
        CHECK_TH_CUDA(scales_);
        CHECK_TYPE(scales_, torch::kFloat32);
        e4m3_static_quantize(input, quantized_input, scales_, stream, quantize_mode);
    }
    else
    {
        std::vector<int64_t> scale_shape;
        if (quantize_mode == QuantizeMode::PER_TOKEN)
        {
            for (int i = 0; i < input.dim() - 1; i++)
                scale_shape.push_back(input.size(i));
            scale_shape.push_back(1);
        }
        else if (quantize_mode == QuantizeMode::PER_CHANNEL)
        {
            for (int i = 0; i < input.dim() - 2; i++)
                scale_shape.push_back(input.size(i));
            scale_shape.push_back(1);
            scale_shape.push_back(input.size(-1));
        }
        else // must be PER_TENSOR
        {
            scale_shape.assign(input.dim(), 1);
        }
        scales_ = torch::empty(scale_shape, torch::dtype(input.dtype()).device(torch::kCUDA).requires_grad(false));
        e4m3_dynamic_quantize(input, quantized_input, scales_, stream, quantize_mode);
    }

    return {quantized_input, scales_};
}

Tensor e4m3_dequantize_helper(Tensor input, Tensor scales, QuantizeMode quantize_mode)
{
    CHECK_CONTIGUOUS(input);
    CHECK_TH_CUDA(input);
    CHECK_TH_CUDA(scales);
    TORCH_CHECK(input.numel() != 0, "input should not be empty tensor");
    TORCH_CHECK(input.dim() >= 2 && (quantize_mode != QuantizeMode::PER_CHANNEL || input.dim() == 2),
        "Invalid dim. The dim of input should be greater than or equal to 2");

    TORCH_CHECK(input.scalar_type() == torch::kFloat8_e4m3fn, "Invalid datatype. input must be Int8 (Fp8)");

    std::vector<int64_t> dequantized_input_shape;
    for (int i = 0; i < input.dim(); i++)
        dequantized_input_shape.push_back(input.size(i));
    TORCH_CHECK(scales.dim() == input.dim());
    if (quantize_mode == QuantizeMode::PER_TOKEN)
    {
        for (int i = 0; i < input.dim() - 1; i++)
            TORCH_CHECK(scales.size(i) == input.size(i));
        TORCH_CHECK(scales.size(-1) == 1)
    }
    else if (quantize_mode == QuantizeMode::PER_CHANNEL)
    {
        for (int i = 0; i < input.dim() - 2; i++)
            TORCH_CHECK(scales.size(i) == input.size(i));
        TORCH_CHECK(scales.size(-2) == 1);
        TORCH_CHECK(scales.size(-1) == input.size(-1));
    }
    else
    {
        for (int i = 0; i < input.dim(); i++)
            TORCH_CHECK(scales.size(i) == 1);
    }

    Tensor dequantized_input
        = torch::empty(dequantized_input_shape, torch::dtype(scales.dtype()).device(torch::kCUDA).requires_grad(false));

    auto input_ptr = get_ptr<__nv_fp8_e4m3>(input);

    auto stream = at::cuda::getCurrentCUDAStream(input.get_device());

    if (scales.scalar_type() == at::ScalarType::Float)
    {
        invokeDequantizeMatrix(get_ptr<float>(dequantized_input), get_ptr<float>(scales), input_ptr, input.numel(),
            input.size(-1), quantize_mode, stream);
    }
    else if (scales.scalar_type() == at::ScalarType::Half)
    {
        invokeDequantizeMatrix(get_ptr<half>(dequantized_input), get_ptr<half>(scales), input_ptr, input.numel(),
            input.size(-1), quantize_mode, stream);
    }
#ifdef ENABLE_BF16
    else if (scales.scalar_type() == at::ScalarType::BFloat16)
    {
        invokeDequantizeMatrix(get_ptr<__nv_bfloat16>(dequantized_input), get_ptr<__nv_bfloat16>(scales), input_ptr,
            input.numel(), input.size(-1), quantize_mode, stream);
    }
#endif
    else
    {
        TORCH_CHECK(false, "Invalid datatype. input must be BF16/FP16/FP32");
    }

    return dequantized_input;
}

std::tuple<Tensor, Tensor> symmetric_quantize_weight(Tensor weight)
{
    return e4m3_quantize_helper(weight, at::nullopt, QuantizeMode::PER_CHANNEL);
}

std::tuple<Tensor, Tensor> symmetric_quantize_activation(Tensor activation)
{
    return e4m3_quantize_helper(activation, at::nullopt, QuantizeMode::PER_TOKEN);
}

std::tuple<Tensor, Tensor> symmetric_quantize_per_tensor(Tensor input)
{
    return e4m3_quantize_helper(input, at::nullopt, QuantizeMode::PER_TENSOR);
}

std::tuple<Tensor, Tensor> symmetric_static_quantize_weight(Tensor weight, Tensor scales)
{
    return e4m3_quantize_helper(weight, scales, QuantizeMode::PER_CHANNEL);
}

std::tuple<Tensor, Tensor> symmetric_static_quantize_activation(Tensor activation, Tensor scales)
{
    return e4m3_quantize_helper(activation, scales, QuantizeMode::PER_TOKEN);
}

std::tuple<Tensor, Tensor> symmetric_static_quantize_per_tensor(Tensor input, Tensor scales)
{
    return e4m3_quantize_helper(input, scales, QuantizeMode::PER_TENSOR);
}

Tensor symmetric_dequantize_weight(Tensor weight, Tensor scales)
{
    return e4m3_dequantize_helper(weight, scales, QuantizeMode::PER_CHANNEL);
}

Tensor symmetric_dequantize_activation(Tensor activation, Tensor scales)
{
    return e4m3_dequantize_helper(activation, scales, QuantizeMode::PER_TOKEN);
}

Tensor symmetric_dequantize_per_tensor(Tensor input, Tensor scales)
{
    return e4m3_dequantize_helper(input, scales, QuantizeMode::PER_TENSOR);
}

} // namespace torch_ext

// Utility methods that may be useful for preprocessing weights in torch.
TORCH_LIBRARY_FRAGMENT(tensorrt_llm, m)
{
    m.def("quantize_e4m3_weight(Tensor weight) -> (Tensor, Tensor)");
    m.def("quantize_e4m3_activation(Tensor activation) -> (Tensor, Tensor)");
    m.def("quantize_e4m3_per_tensor(Tensor input) -> (Tensor, Tensor)");
    m.def("static_quantize_e4m3_weight(Tensor weight, Tensor scales) -> (Tensor, Tensor)");
    m.def("static_quantize_e4m3_activation(Tensor activation, Tensor scales) -> (Tensor, Tensor)");
    m.def("static_quantize_e4m3_per_tensor(Tensor input, Tensor scales) -> (Tensor, Tensor)");
    m.def("dequantize_e4m3_weight(Tensor weight, Tensor scales) -> Tensor");
    m.def("dequantize_e4m3_activation(Tensor activation, Tensor scales) -> Tensor");
    m.def("dequantize_e4m3_per_tensor(Tensor input, Tensor scales) -> Tensor");
}

TORCH_LIBRARY_IMPL(tensorrt_llm, CUDA, m)
{
    m.impl("quantize_e4m3_weight", &torch_ext::symmetric_quantize_weight);
    m.impl("quantize_e4m3_activation", &torch_ext::symmetric_quantize_activation);
    m.impl("quantize_e4m3_per_tensor", &torch_ext::symmetric_quantize_per_tensor);
    m.impl("static_quantize_e4m3_weight", &torch_ext::symmetric_static_quantize_weight);
    m.impl("static_quantize_e4m3_activation", &torch_ext::symmetric_static_quantize_activation);
    m.impl("static_quantize_e4m3_per_tensor", &torch_ext::symmetric_static_quantize_per_tensor);
    m.impl("dequantize_e4m3_weight", &torch_ext::symmetric_dequantize_weight);
    m.impl("dequantize_e4m3_activation", &torch_ext::symmetric_dequantize_activation);
    m.impl("dequantize_e4m3_per_tensor", &torch_ext::symmetric_dequantize_per_tensor);
}
