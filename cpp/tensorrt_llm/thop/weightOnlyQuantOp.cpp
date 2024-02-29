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
#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_preprocessors.h"
#include "tensorrt_llm/thop/thUtils.h"

#if defined(TORCH_VERSION_MAJOR)                                                                                       \
    && ((TORCH_VERSION_MAJOR > 1) || ((TORCH_VERSION_MAJOR == 1) && (TORCH_VERSION_MINOR >= 9)))
#define TORCH_IS_AT_LEAST_v190
#endif

namespace torch_ext
{
using torch::Tensor;
using namespace tensorrt_llm::kernels::cutlass_kernels;

void check_quant_type_allowed(torch::ScalarType quant_type)
{
#ifdef TORCH_IS_AT_LEAST_v190
    TORCH_CHECK(
        quant_type == torch::kInt8 || quant_type == at::ScalarType::QUInt4x2, "Must be int4 or int8 quantization");
#else
    TORCH_CHECK(quant_type == torch::kInt8, "Must be int8 quantization");
#endif
}

QuantType get_ft_quant_type(torch::ScalarType quant_type)
{
    if (quant_type == torch::kInt8)
    {
        return QuantType::INT8_WEIGHT_ONLY;
    }
#ifdef TORCH_IS_AT_LEAST_v190
    else if (quant_type == at::ScalarType::QUInt4x2)
    {
        return QuantType::PACKED_INT4_WEIGHT_ONLY;
    }
#endif
    else
    {
        TORCH_CHECK(false, "Invalid quantization type");
    }
}

// Permutes the rows of B for Turing and Ampere. Throws an error for other architectures.
Tensor permute_B_rows_for_mixed_gemm(Tensor quantized_tensor, torch::ScalarType quant_type, const int64_t arch_version)
{
    auto _st = quantized_tensor.scalar_type();
    CHECK_CPU(quantized_tensor);
    CHECK_CONTIGUOUS(quantized_tensor);
    TORCH_CHECK(_st == torch::kInt8, "Quantized tensor must be int8 dtype");
    check_quant_type_allowed(quant_type);
    TORCH_CHECK(
        quantized_tensor.dim() == 2 || quantized_tensor.dim() == 3, "Invalid dim. The dim of weight should be 2 or 3");

    QuantType ft_quant_type = get_ft_quant_type(quant_type);
    const size_t bits_in_quant_type = get_bits_in_quant_type(ft_quant_type);

    const size_t num_experts = quantized_tensor.dim() == 2 ? 1 : quantized_tensor.size(0);
    const size_t num_rows = quantized_tensor.size(-2);
    const size_t num_cols = (8 / bits_in_quant_type) * quantized_tensor.size(-1);

    Tensor transformed_tensor = torch::empty_like(quantized_tensor);

    int8_t* input_byte_ptr = get_ptr<int8_t>(quantized_tensor);
    int8_t* output_byte_ptr = get_ptr<int8_t>(transformed_tensor);

    permute_B_rows_for_mixed_gemm(
        output_byte_ptr, input_byte_ptr, {num_experts, num_rows, num_cols}, ft_quant_type, arch_version);

    return transformed_tensor;
}

// We need to use this transpose to correctly handle packed int4 and int8 data
Tensor subbyte_transpose(Tensor quantized_tensor, torch::ScalarType quant_type)
{

    auto _st = quantized_tensor.scalar_type();
    CHECK_CPU(quantized_tensor);
    CHECK_CONTIGUOUS(quantized_tensor);
    TORCH_CHECK(_st == torch::kInt8, "Quantized tensor must be int8 dtype");
    check_quant_type_allowed(quant_type);
    TORCH_CHECK(
        quantized_tensor.dim() == 2 || quantized_tensor.dim() == 3, "Invalid dim. The dim of weight should be 2 or 3");

    QuantType ft_quant_type = get_ft_quant_type(quant_type);
    const size_t bits_in_quant_type = get_bits_in_quant_type(ft_quant_type);

    const size_t num_experts = quantized_tensor.dim() == 2 ? 1 : quantized_tensor.size(0);
    const size_t num_rows = quantized_tensor.size(-2);
    const size_t num_cols = (8 / bits_in_quant_type) * quantized_tensor.size(-1);

    Tensor transposed_tensor = torch::empty_like(quantized_tensor);

    int8_t* input_byte_ptr = get_ptr<int8_t>(quantized_tensor);
    int8_t* output_byte_ptr = get_ptr<int8_t>(transposed_tensor);

    subbyte_transpose(output_byte_ptr, input_byte_ptr, {num_experts, num_rows, num_cols}, ft_quant_type);
    return transposed_tensor;
}

Tensor preprocess_weights_for_mixed_gemm(Tensor row_major_quantized_weight, torch::ScalarType quant_type)
{
    auto _st = row_major_quantized_weight.scalar_type();
    CHECK_CPU(row_major_quantized_weight);
    CHECK_CONTIGUOUS(row_major_quantized_weight);
    TORCH_CHECK(_st == torch::kInt8, "Quantized tensor must be int8 dtype");
    check_quant_type_allowed(quant_type);
    TORCH_CHECK(row_major_quantized_weight.dim() == 2 || row_major_quantized_weight.dim() == 3,
        "Invalid dim. The dim of weight should be 2 or 3");

    QuantType ft_quant_type = get_ft_quant_type(quant_type);
    const size_t bits_in_quant_type = get_bits_in_quant_type(ft_quant_type);

    const size_t num_experts = row_major_quantized_weight.dim() == 2 ? 1 : row_major_quantized_weight.size(0);
    const size_t num_rows = row_major_quantized_weight.size(-2);
    const size_t num_cols = (8 / bits_in_quant_type) * row_major_quantized_weight.size(-1);

    Tensor processed_tensor = torch::zeros_like(row_major_quantized_weight);
    int8_t* input_byte_ptr = get_ptr<int8_t>(row_major_quantized_weight);
    int8_t* output_byte_ptr = get_ptr<int8_t>(processed_tensor);

    preprocess_weights_for_mixed_gemm(
        output_byte_ptr, input_byte_ptr, {num_experts, num_rows, num_cols}, ft_quant_type);

    return processed_tensor;
}

std::vector<Tensor> symmetric_quantize_helper(
    Tensor weight, torch::ScalarType quant_type, bool return_unprocessed_quantized_tensor)
{
    CHECK_CPU(weight);
    CHECK_CONTIGUOUS(weight);
    TORCH_CHECK(weight.numel() != 0, "weight should not be empty tensor");
    TORCH_CHECK(weight.dim() == 2 || weight.dim() == 3, "Invalid dim. The dim of weight should be 2 or 3");

    auto _st = weight.scalar_type();
    TORCH_CHECK(_st == torch::kFloat32 || _st == torch::kFloat16 || _st == torch::kBFloat16,
        "Invalid datatype. Weight must be FP16 or BF16");
    check_quant_type_allowed(quant_type);
    QuantType ft_quant_type = get_ft_quant_type(quant_type);

    const size_t num_experts = weight.dim() == 2 ? 1 : weight.size(0);
    const size_t num_rows = weight.size(-2);
    const size_t num_cols = weight.size(-1);

    const size_t bits_in_type = get_bits_in_quant_type(ft_quant_type);
    const size_t bytes_per_out_col = num_cols * bits_in_type / 8;

    const size_t input_mat_size = num_rows * num_cols;
    const size_t quantized_mat_size = num_rows * bytes_per_out_col;

    std::vector<int64_t> quantized_weight_shape;
    std::vector<int64_t> scale_shape;
    if (weight.dim() == 2)
    {
        quantized_weight_shape = {int64_t(num_rows), int64_t(bytes_per_out_col)};
        scale_shape = {int64_t(num_cols)};
    }
    else if (weight.dim() == 3)
    {
        quantized_weight_shape = {int64_t(num_experts), int64_t(num_rows), int64_t(bytes_per_out_col)};
        scale_shape = {int64_t(num_experts), int64_t(num_cols)};
    }
    else
    {
        TORCH_CHECK(false, "Invalid weight dimension. Weight must have dim 2 or 3");
    }

    Tensor unprocessed_quantized_weight
        = torch::empty(quantized_weight_shape, torch::dtype(torch::kInt8).device(torch::kCPU).requires_grad(false));

    Tensor processed_quantized_weight = torch::empty_like(unprocessed_quantized_weight);

    Tensor scales = torch::empty(scale_shape, torch::dtype(weight.dtype()).device(torch::kCPU).requires_grad(false));

    int8_t* unprocessed_quantized_weight_ptr = get_ptr<int8_t>(unprocessed_quantized_weight);
    int8_t* processed_quantized_weight_ptr = get_ptr<int8_t>(processed_quantized_weight);

    if (weight.scalar_type() == at::ScalarType::Float)
    {
        symmetric_quantize<float, float>(processed_quantized_weight_ptr, unprocessed_quantized_weight_ptr,
            get_ptr<float>(scales), get_ptr<const float>(weight), {num_experts, num_rows, num_cols}, ft_quant_type);
    }
    else if (weight.scalar_type() == at::ScalarType::Half)
    {
        symmetric_quantize<half, half>(processed_quantized_weight_ptr, unprocessed_quantized_weight_ptr,
            get_ptr<half>(scales), get_ptr<const half>(weight), {num_experts, num_rows, num_cols}, ft_quant_type);
    }
#ifdef ENABLE_BF16
    else if (weight.scalar_type() == at::ScalarType::BFloat16)
    {
        symmetric_quantize<__nv_bfloat16, __nv_bfloat16>(processed_quantized_weight_ptr,
            unprocessed_quantized_weight_ptr, get_ptr<__nv_bfloat16>(scales), get_ptr<const __nv_bfloat16>(weight),
            {num_experts, num_rows, num_cols}, ft_quant_type);
    }
#endif
    else
    {
        TORCH_CHECK(false, "Invalid datatype. Weight must be BF16/FP16");
    }

    if (return_unprocessed_quantized_tensor)
    {
        return std::vector<Tensor>{unprocessed_quantized_weight, processed_quantized_weight, scales};
    }

    return std::vector<Tensor>{processed_quantized_weight, scales};
}

std::vector<Tensor> symmetric_quantize_last_axis_of_batched_matrix(Tensor weight, torch::ScalarType quant_type)
{
    return symmetric_quantize_helper(weight, quant_type, false);
}

// Same as symmetric_quantize_last_axis_of_batched_matrix but returns a tuple of:
// (unprocessed_quantized_weights, preprocessed_quantized_weights, scales)
// Exposed mainly for testing, so that the unprocessed weights can be passed to torch functions.
std::vector<Tensor> _symmetric_quantize_last_axis_of_batched_matrix(Tensor weight, torch::ScalarType quant_type)
{
    return symmetric_quantize_helper(weight, quant_type, true);
}

Tensor add_bias_and_interleave_int4s(Tensor weight)
{
    CHECK_CPU(weight);
    CHECK_CONTIGUOUS(weight);
    TORCH_CHECK(weight.numel() != 0, "weight should not be empty tensor");
    TORCH_CHECK(weight.dtype() == torch::kInt8, "Weight must be a packed int8 tensor");
    Tensor output = weight.clone().detach();

    int8_t* int4_tensor_ptr = get_ptr<int8_t>(output);
    const size_t num_bytes = output.numel();
    const size_t num_elts = 2 * num_bytes;
    add_bias_and_interleave_quantized_tensor_inplace(int4_tensor_ptr, num_elts, QuantType::PACKED_INT4_WEIGHT_ONLY);

    return output;
}

Tensor add_bias_and_interleave_int8s(Tensor weight)
{
    CHECK_CPU(weight);
    CHECK_CONTIGUOUS(weight);
    TORCH_CHECK(weight.numel() != 0, "weight should not be empty tensor");
    TORCH_CHECK(weight.dtype() == torch::kInt8, "Weight must be an int8 tensor");
    Tensor output = weight.clone().detach();

    int8_t* int8_tensor_ptr = get_ptr<int8_t>(output);
    const size_t num_elts = output.numel();
    add_bias_and_interleave_quantized_tensor_inplace(int8_tensor_ptr, num_elts, QuantType::INT8_WEIGHT_ONLY);

    return output;
}

Tensor unpack_int4_packed_tensor_to_int8(Tensor weight)
{
    CHECK_CPU(weight);
    CHECK_CONTIGUOUS(weight);
    TORCH_CHECK(weight.numel() != 0, "weight should not be empty tensor");
    TORCH_CHECK(weight.dtype() == torch::kInt8, "Weight must be a packed int8 tensor");

    std::vector<int64_t> int8_tensor_size(weight.dim());
    for (int i = 0; i < weight.dim(); ++i)
    {
        int8_tensor_size[i] = weight.size(i);
    }
    int8_tensor_size[weight.dim() - 1] *= 2;

    Tensor unpacked_weight
        = torch::zeros(int8_tensor_size, torch::dtype(torch::kInt8).device(torch::kCPU).requires_grad(false));

    int8_t* packed_ptr = get_ptr<int8_t>(weight);
    int8_t* unpacked_ptr = get_ptr<int8_t>(unpacked_weight);

    for (size_t packed_idx = 0; packed_idx < weight.numel(); ++packed_idx)
    {
        int8_t packed_data = packed_ptr[packed_idx];

        int8_t elt_0 = (int8_t(packed_data << 4) >> 4); // The double shift here is to ensure sign extension
        int8_t elt_1 = packed_data >> 4;

        unpacked_ptr[2 * packed_idx + 0] = elt_0;
        unpacked_ptr[2 * packed_idx + 1] = elt_1;
    }

    return unpacked_weight;
}

Tensor pack_int8_tensor_to_packed_int4(Tensor weight)
{
    CHECK_CPU(weight);
    CHECK_CONTIGUOUS(weight);
    TORCH_CHECK(weight.numel() != 0, "weight should not be empty tensor");
    TORCH_CHECK(weight.dtype() == torch::kInt8, "Weight must be a int8 tensor");

    std::vector<int64_t> packed_tensor_size(weight.dim());
    for (int i = 0; i < weight.dim(); ++i)
    {
        packed_tensor_size[i] = weight.size(i);
    }
    packed_tensor_size[weight.dim() - 1] = (packed_tensor_size[weight.dim() - 1] + 1) / 2;

    Tensor packed_weight
        = torch::zeros(packed_tensor_size, torch::dtype(torch::kInt8).device(torch::kCPU).requires_grad(false));

    int8_t* unpacked_ptr = get_ptr<int8_t>(weight);
    int8_t* packed_ptr = get_ptr<int8_t>(packed_weight);

    for (size_t packed_idx = 0; packed_idx < packed_weight.numel(); ++packed_idx)
    {
        int8_t packed_int4s = 0;
        int8_t elt_0 = unpacked_ptr[2 * packed_idx + 0];
        int8_t elt_1 = unpacked_ptr[2 * packed_idx + 1];

        TORCH_CHECK(elt_0 >= -8 && elt_0 <= 7, "Value in unpacked tensor not in int4 range");
        TORCH_CHECK(elt_1 >= -8 && elt_1 <= 7, "Value in unpacked tensor not in int4 range");

        packed_int4s |= ((elt_0 & 0x0F));
        packed_int4s |= int8_t(elt_1 << 4);

        packed_ptr[packed_idx] = packed_int4s;
    }
    return packed_weight;
}

} // namespace torch_ext

// Utility methods that may be useful for preprocessing weights in torch.
static auto symmetric_quantize_last_axis_of_batched_matrix
    = torch::RegisterOperators("trtllm::symmetric_quantize_last_axis_of_batched_matrix",
        &torch_ext::symmetric_quantize_last_axis_of_batched_matrix);

static auto preprocess_weights_for_mixed_gemm = torch::RegisterOperators(
    "trtllm::preprocess_weights_for_mixed_gemm", &torch_ext::preprocess_weights_for_mixed_gemm);

static auto unpack_int4_packed_tensor_to_int8 = torch::RegisterOperators(
    "trtllm::unpack_int4_packed_tensor_to_int8", &torch_ext::unpack_int4_packed_tensor_to_int8);

static auto pack_int8_tensor_to_packed_int4
    = torch::RegisterOperators("trtllm::pack_int8_tensor_to_packed_int4", &torch_ext::pack_int8_tensor_to_packed_int4);

// Utility methods exposed purely for unit tests in torch.
static auto _symmetric_quantize_last_axis_of_batched_matrix
    = torch::RegisterOperators("trtllm::_symmetric_quantize_last_axis_of_batched_matrix",
        &torch_ext::_symmetric_quantize_last_axis_of_batched_matrix);

static auto add_bias_and_interleave_int4s
    = torch::RegisterOperators("trtllm::_add_bias_and_interleave_int4s", &torch_ext::add_bias_and_interleave_int4s);

static auto add_bias_and_interleave_int8s
    = torch::RegisterOperators("trtllm::_add_bias_and_interleave_int8s", &torch_ext::add_bias_and_interleave_int8s);

static auto permute_B_rows_for_mixed_gemm
    = torch::RegisterOperators("trtllm::_permute_B_rows_for_mixed_gemm", &torch_ext::permute_B_rows_for_mixed_gemm);

static auto subbyte_transpose = torch::RegisterOperators("trtllm::_subbyte_transpose", &torch_ext::subbyte_transpose);
