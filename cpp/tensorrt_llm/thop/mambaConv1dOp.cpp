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
#include "tensorrt_llm/kernels/mambaConv1dKernels.h"
#include "tensorrt_llm/thop/thUtils.h"

namespace th = torch;
namespace tk = tensorrt_llm::kernels;

namespace torch_ext
{

std::tuple<th::Tensor, th::Tensor> mamba_conv1d(th::Tensor const& input, th::Tensor const& conv_weight,
    th::Tensor const& conv_bias, th::Tensor const& conv_state, th::Tensor const& host_request_types,
    th::Tensor const& last_token_ids, th::optional<th::Tensor> host_context_lengths,
    th::optional<th::Tensor> slot_mapping, int64_t const dim, int64_t const dconv, int64_t const pre_stride,
    int64_t const post_stride, bool const remove_padding, bool const apply_silu, bool const is_paged_state)
{
    // tensors info: [shapes] x [dtype]
    // input: [batch_size, seq_len, dim] or [num_tokens, dim] for remove_padding x [float16, float32, bfloat16]
    // conv_weight: [1, dconv, dim] x [float16, float32, bfloat16]
    // conv_bias: [dim] x [float16, float32, bfloat16]
    // conv_state: [batch_size, dconv-1, dim] x [float16, float32, bfloat16]
    // host_request_types: [batch_size] x [int32]
    // last_token_ids: [batch_size] x [int32]
    // host_context_lengths: [batch_size] x [int32] for remove_padding

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    tk::MambaConv1dParamsBase params;

    auto host_request_sizes = host_request_types.sizes();
    auto input_sizes = input.sizes();

    int batch_size = host_request_sizes[0];
    int max_seqlen;

    if (remove_padding && host_context_lengths.has_value())
    {
        max_seqlen = host_context_lengths.value().max().item<int>();
    }
    else
    {
        max_seqlen = input_sizes[1];
    }

    // req_type=0 -> context (prefill)
    // req_type=1 -> generation (decode)
    auto req_type = host_request_types[0].item<int>();

    int idx = (remove_padding) ? 1 : 2;
    int64_t out_dim = input_sizes[idx] - pre_stride - post_stride;

    std::vector<int64_t> out_shape;
    if (remove_padding)
    {
        out_shape = {input_sizes[0], out_dim};
    }
    else
    {
        out_shape = {input_sizes[0], input_sizes[1], out_dim};
    }

    auto out = torch::empty(out_shape, input.options());
    auto state_out = torch::empty_like(conv_state);

    params.batch = batch_size;
    params.dim = dim;
    params.max_seqlen = max_seqlen;
    params.dconv = dconv;
    params.pre_stride = pre_stride;
    params.post_stride = post_stride;
    params.remove_padding = remove_padding;
    params.apply_silu = apply_silu;

    // Set the pointers and strides.
    params.in_ptr = input.data_ptr();
    params.weight_ptr = conv_weight.data_ptr();
    params.bias_ptr = conv_bias.data_ptr();
    params.out_ptr = out.data_ptr();
    params.last_token_ids_ptr = static_cast<int const*>(last_token_ids.const_data_ptr());

    if (is_paged_state)
    {
        if (!slot_mapping.has_value())
        {
            throw std::invalid_argument("slot_mapping must be provided when paged state is enabled");
        }

        params.state_in_ptr = *reinterpret_cast<void**>(const_cast<void*>(conv_state.data_ptr()));
        params.state_out_ptr = *reinterpret_cast<void**>(const_cast<void*>(conv_state.data_ptr()));
        params.state_slot_mapping_ptr = static_cast<int const*>(slot_mapping.value().const_data_ptr());
    }
    else
    {
        params.state_in_ptr = conv_state.data_ptr();
        params.state_out_ptr = state_out.data_ptr();
        params.state_slot_mapping_ptr = nullptr;
    }

    c10::ScalarType dtype = input.scalar_type();

    if (req_type == 0)
    {
        switch (dtype)
        {
        case torch::kFloat16:
            // Handle Float16
            tk::invokeMambaConv1dContext<half>(params, stream);
            break;
        case torch::kFloat32:
            // Handle Float32
            tk::invokeMambaConv1dContext<float>(params, stream);
            break;
        case torch::kBFloat16:
            // Handle BFloat16
            tk::invokeMambaConv1dContext<__nv_bfloat16>(params, stream);
            break;
        default:
            // Handle other data types
            throw std::invalid_argument("Invalid dtype, only supports float16, float32, and bfloat16");
            break;
        }
    }
    else
    {
        switch (dtype)
        {
        case torch::kFloat16:
            // Handle Float16
            tk::invokeMambaConv1dGeneration<half>(params, stream);
            break;
        case torch::kFloat32:
            // Handle Float32
            tk::invokeMambaConv1dGeneration<float>(params, stream);
            break;
        case torch::kBFloat16:
            // Handle BFloat16
            tk::invokeMambaConv1dGeneration<__nv_bfloat16>(params, stream);
            break;
        default:
            // Handle other data types
            throw std::invalid_argument("Invalid dtype, only supports float16, float32, and bfloat16");
            break;
        }
    }

    sync_check_cuda_error(stream);

    if (is_paged_state)
    {
        return std::make_tuple(out, conv_state);
    }
    else
    {
        return std::make_tuple(out, state_out);
    }
}

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "mamba_conv1d(Tensor input, Tensor conv_weight, "
        "Tensor conv_bias, Tensor conv_state, "
        "Tensor host_request_types, Tensor last_token_ids, "
        "Tensor? host_context_lengths, Tensor? slot_mapping, "
        "int dim, int dconv, int pre_stride, int post_stride, "
        "bool remove_padding, bool apply_silu, "
        "bool is_paged_state) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("mamba_conv1d", &torch_ext::mamba_conv1d);
}
