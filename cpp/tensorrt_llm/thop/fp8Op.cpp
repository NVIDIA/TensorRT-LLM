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

#include "tensorrt_llm/common/cudaBf16Wrapper.h"
#include "tensorrt_llm/common/cudaFp8Utils.h"
#include "tensorrt_llm/thop/thUtils.h"

#if defined(TORCH_VERSION_MAJOR)                                                                                       \
    && ((TORCH_VERSION_MAJOR > 1) || ((TORCH_VERSION_MAJOR == 1) && (TORCH_VERSION_MINOR >= 9)))
#define TORCH_IS_AT_LEAST_v190
#endif

namespace torch_ext
{
using torch::Tensor;
using namespace tensorrt_llm::common;

std::vector<Tensor> e4m3_quantize_helper(Tensor input, QuantizeMode quantize_mode)
{
    CHECK_CONTIGUOUS(input);
    TORCH_CHECK(input.numel() != 0, "input should not be empty tensor");
    TORCH_CHECK(input.dim() >= 2 && (quantize_mode != QuantizeMode::PER_CHANNEL || input.dim() == 2),
        "Invalid dim. The dim of input should be greater than or equal to 2");

    auto _st = input.scalar_type();
    TORCH_CHECK(_st == torch::kFloat32 || _st == torch::kFloat16 || _st == torch::kBFloat16,
        "Invalid datatype. input must be FP16 or BF16 or FP32");

    std::vector<int64_t> quantized_input_shape;
    for (int i = 0; i < input.dim(); i++)
        quantized_input_shape.push_back(input.size(i));
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

    const auto is_cuda = input.is_cuda();
    input = input.cuda();

    Tensor quantized_input
        = torch::empty(quantized_input_shape, torch::dtype(torch::kInt8).device(torch::kCUDA).requires_grad(false));

    Tensor scales = torch::empty(scale_shape, torch::dtype(input.dtype()).device(torch::kCUDA).requires_grad(false));

    auto quantized_input_ptr = reinterpret_cast<__nv_fp8_e4m3*>(get_ptr<int8_t>(quantized_input));

    auto stream = at::cuda::getDefaultCUDAStream();

    if (input.scalar_type() == at::ScalarType::Float)
    {
        invokeComputeScalesAndQuantizeMatrix(quantized_input_ptr, get_ptr<float>(scales), get_ptr<const float>(input),
            input.numel(), input.size(-1), quantize_mode, stream);
    }
    else if (input.scalar_type() == at::ScalarType::Half)
    {
        invokeComputeScalesAndQuantizeMatrix(quantized_input_ptr, get_ptr<half>(scales), get_ptr<const half>(input),
            input.numel(), input.size(-1), quantize_mode, stream);
    }
#ifdef ENABLE_BF16
    else if (input.scalar_type() == at::ScalarType::BFloat16)
    {
        invokeComputeScalesAndQuantizeMatrix(quantized_input_ptr, get_ptr<__nv_bfloat16>(scales),
            get_ptr<const __nv_bfloat16>(input), input.numel(), input.size(-1), quantize_mode, stream);
    }
#endif
    else
    {
        TORCH_CHECK(false, "Invalid datatype. input must be BF16/FP16/FP32");
    }

    if (!is_cuda)
    {
        quantized_input = quantized_input.cpu();
        scales = scales.cpu();
    }

    return std::vector<Tensor>{quantized_input, scales};
}

Tensor e4m3_dequantize_helper(Tensor input, Tensor scales, QuantizeMode quantize_mode)
{
    CHECK_CONTIGUOUS(input);
    TORCH_CHECK(input.numel() != 0, "input should not be empty tensor");
    TORCH_CHECK(input.dim() >= 2 && (quantize_mode != QuantizeMode::PER_CHANNEL || input.dim() == 2),
        "Invalid dim. The dim of input should be greater than or equal to 2");

    TORCH_CHECK(input.scalar_type() == torch::kInt8, "Invalid datatype. input must be Int8 (Fp8)");

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

    const auto w_is_cuda = input.is_cuda();
    input = input.cuda();
    scales = scales.cuda();

    Tensor dequantized_input
        = torch::empty(dequantized_input_shape, torch::dtype(scales.dtype()).device(torch::kCUDA).requires_grad(false));

    auto input_ptr = reinterpret_cast<__nv_fp8_e4m3*>(get_ptr<int8_t>(input));

    auto stream = at::cuda::getDefaultCUDAStream();

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

    if (!w_is_cuda)
        dequantized_input = dequantized_input.cpu();

    return dequantized_input;
}

std::vector<Tensor> symmetric_quantize_weight(Tensor weight)
{
    return e4m3_quantize_helper(weight, QuantizeMode::PER_CHANNEL);
}

std::vector<Tensor> symmetric_quantize_activation(Tensor activation)
{
    return e4m3_quantize_helper(activation, QuantizeMode::PER_TOKEN);
}

std::vector<Tensor> symmetric_quantize_per_tensor(Tensor input)
{
    return e4m3_quantize_helper(input, QuantizeMode::PER_TENSOR);
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
static auto symmetric_quantize_weight
    = torch::RegisterOperators("tensorrt_llm::quantize_e4m3_weight", &torch_ext::symmetric_quantize_weight);

static auto symmetric_quantize_activation
    = torch::RegisterOperators("tensorrt_llm::quantize_e4m3_activation", &torch_ext::symmetric_quantize_activation);

static auto symmetric_quantize_per_tensor
    = torch::RegisterOperators("tensorrt_llm::quantize_e4m3_per_tensor", &torch_ext::symmetric_quantize_per_tensor);

static auto symmetric_dequantize_weight
    = torch::RegisterOperators("tensorrt_llm::dequantize_e4m3_weight", &torch_ext::symmetric_dequantize_weight);

static auto symmetric_dequantize_activation
    = torch::RegisterOperators("tensorrt_llm::dequantize_e4m3_activation", &torch_ext::symmetric_dequantize_activation);

static auto symmetric_dequantize_per_tensor
    = torch::RegisterOperators("tensorrt_llm::dequantize_e4m3_per_tensor", &torch_ext::symmetric_dequantize_per_tensor);
