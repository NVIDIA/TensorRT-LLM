/*
 * Adapted from https://github.com/Dao-AILab/causal-conv1d/blob/main/csrc/causal_conv1d.cpp
 * Copyright (c) 2024, Tri Dao.
 *
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
#include "tensorrt_llm/kernels/causalConv1d/causalConv1d.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

namespace torch_ext
{

#define CHECK_SHAPE(x, ...)                                                                                            \
    TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")

#define DISPATCH_WTYPE_ITYPE_FLOAT_AND_HALF_AND_BF16(ITYPE, NAME, ...)                                                 \
    if (ITYPE == at::ScalarType::Half)                                                                                 \
    {                                                                                                                  \
        using input_t = half;                                                                                          \
        using weight_t = half;                                                                                         \
        __VA_ARGS__();                                                                                                 \
    }                                                                                                                  \
    else if (ITYPE == at::ScalarType::BFloat16)                                                                        \
    {                                                                                                                  \
        using input_t = nv_bfloat16;                                                                                   \
        using weight_t = nv_bfloat16;                                                                                  \
        __VA_ARGS__();                                                                                                 \
    }                                                                                                                  \
    else if (ITYPE == at::ScalarType::Float)                                                                           \
    {                                                                                                                  \
        using input_t = float;                                                                                         \
        using weight_t = float;                                                                                        \
        __VA_ARGS__();                                                                                                 \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        AT_ERROR(#NAME, " not implemented for input type '", toString(ITYPE), "'");                                    \
    }

void set_conv_params_fwd(tensorrt_llm::kernels::causal_conv1d::ConvParamsBase& params,
    // sizes
    const size_t batch, const size_t dim, const size_t seqlen, const size_t width,
    // device pointers
    const at::Tensor x, const at::Tensor weight, const at::Tensor out, std::optional<at::Tensor> const& bias,
    bool silu_activation, int64_t pad_slot_id, std::optional<at::Tensor> const& query_start_loc = std::nullopt,
    std::optional<at::Tensor> const& cache_indices = std::nullopt,
    std::optional<at::Tensor> const& has_initial_state = std::nullopt)
{

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    params.batch = batch;
    params.dim = dim;
    params.seqlen = seqlen;
    params.width = width;
    params.pad_slot_id = pad_slot_id;

    params.silu_activation = silu_activation;

    // Set the pointers and strides.
    params.x_ptr = x.data_ptr();
    params.weight_ptr = weight.data_ptr();
    params.bias_ptr = bias.has_value() ? bias.value().data_ptr() : nullptr;
    params.out_ptr = out.data_ptr();
    // All stride are in elements, not bytes.
    params.query_start_loc_ptr = query_start_loc.has_value() ? query_start_loc.value().data_ptr() : nullptr;
    params.cache_indices_ptr = cache_indices.has_value() ? cache_indices.value().data_ptr() : nullptr;
    params.has_initial_state_ptr = has_initial_state.has_value() ? has_initial_state.value().data_ptr() : nullptr;
    bool const varlen = params.query_start_loc_ptr != nullptr;
    params.x_batch_stride = x.stride(varlen ? 1 : 0);
    params.x_c_stride = x.stride(varlen ? 0 : 1);
    params.x_l_stride = x.stride(varlen ? 1 : -1);
    params.weight_c_stride = weight.stride(0);
    params.weight_width_stride = weight.stride(1);
    params.out_batch_stride = out.stride(varlen ? 1 : 0);
    params.out_c_stride = out.stride(varlen ? 0 : 1);
    params.out_l_stride = out.stride(varlen ? 1 : -1);
}

void causalConv1dFwd(at::Tensor const& x, at::Tensor const& weight, std::optional<at::Tensor> const& bias_,
    std::optional<at::Tensor> const& conv_states, std::optional<at::Tensor> const& query_start_loc,
    std::optional<at::Tensor> const& cache_indices, std::optional<at::Tensor> const& has_initial_state,
    bool silu_activation,
    // used to identify padding entries if cache_indices provided
    // in case of padding, the kernel will return early
    int64_t pad_slot_id)
{
    auto input_type = x.scalar_type();
    auto weight_type = weight.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::Float || input_type == at::ScalarType::Half
        || input_type == at::ScalarType::BFloat16);
    TORCH_CHECK(weight_type == at::ScalarType::Float || weight_type == at::ScalarType::Half
        || weight_type == at::ScalarType::BFloat16);

    TORCH_CHECK(x.is_cuda());
    TORCH_CHECK(weight.is_cuda());

    bool const varlen = query_start_loc.has_value() ? true : false;
    auto const sizes = x.sizes();
    int const batch_size = varlen ? query_start_loc.value().sizes()[0] - 1 : sizes[0];
    int const dim = varlen ? sizes[0] : sizes[1];
    int const seqlen = varlen ? sizes[1] : sizes[2];
    int const width = weight.size(-1);
    if (varlen)
    {
        CHECK_SHAPE(x, dim, seqlen);
    }
    else
    {
        CHECK_SHAPE(x, batch_size, dim, seqlen);
    }
    CHECK_SHAPE(weight, dim, width);

    if (bias_.has_value())
    {
        auto bias = bias_.value();
        TORCH_CHECK(bias.scalar_type() == weight_type);
        TORCH_CHECK(bias.is_cuda());
        TORCH_CHECK(bias.stride(-1) == 1);
        CHECK_SHAPE(bias, dim);
    }

    if (has_initial_state.has_value())
    {
        auto has_initial_state_ = has_initial_state.value();
        TORCH_CHECK(has_initial_state_.scalar_type() == at::ScalarType::Bool);
        TORCH_CHECK(has_initial_state_.is_cuda());
        CHECK_SHAPE(has_initial_state_, batch_size);
    }

    if (query_start_loc.has_value())
    {
        auto query_start_loc_ = query_start_loc.value();
        TORCH_CHECK(query_start_loc_.scalar_type() == at::ScalarType::Int);
        TORCH_CHECK(query_start_loc_.is_cuda());
    }

    if (cache_indices.has_value())
    {
        auto cache_indices_ = cache_indices.value();
        TORCH_CHECK(cache_indices_.scalar_type() == at::ScalarType::Int);
        TORCH_CHECK(cache_indices_.is_cuda());
        CHECK_SHAPE(cache_indices_, batch_size);
    }

    at::Tensor out = x;

    tensorrt_llm::kernels::causal_conv1d::ConvParamsBase params;
    set_conv_params_fwd(params, batch_size, dim, seqlen, width, x, weight, out, bias_, silu_activation, pad_slot_id,
        query_start_loc, cache_indices, has_initial_state);

    if (conv_states.has_value())
    {
        auto conv_states_ = conv_states.value();
        TORCH_CHECK(conv_states_.scalar_type() == input_type);
        TORCH_CHECK(conv_states_.is_cuda());
        params.conv_states_ptr = conv_states_.data_ptr();
        params.conv_states_batch_stride = conv_states_.stride(0);
        params.conv_states_c_stride = conv_states_.stride(1);
        params.conv_states_l_stride = conv_states_.stride(2);
    }
    else
    {
        params.conv_states_ptr = nullptr;
    }

    // Otherwise the kernel will be launched from cuda:0 device
    // Static cast to signed char (AKA c10::DeviceIndex - the input to CUDAGuard) to avoid compiler warning about
    // narrowing
    at::cuda::CUDAGuard device_guard{static_cast<signed char>(x.get_device())};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    DISPATCH_WTYPE_ITYPE_FLOAT_AND_HALF_AND_BF16(x.scalar_type(), "causal_conv1d_fwd",
        [&] { tensorrt_llm::kernels::causal_conv1d::causal_conv1d_fwd_cuda<input_t, weight_t>(params, stream); });
}

void causalConv1dUpdate(at::Tensor const& x, at::Tensor const& conv_state, at::Tensor const& weight,
    std::optional<at::Tensor> const& bias_, bool silu_activation, std::optional<at::Tensor> const& cache_seqlens_,
    std::optional<at::Tensor> const& conv_state_indices_,
    // used to identify padding entries if cache_indices provided
    // in case of padding, the kernel will return early
    int64_t pad_slot_id)
{
    auto input_type = x.scalar_type();
    auto weight_type = weight.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::Float || input_type == at::ScalarType::Half
        || input_type == at::ScalarType::BFloat16);
    TORCH_CHECK(weight_type == at::ScalarType::Float || weight_type == at::ScalarType::Half
        || weight_type == at::ScalarType::BFloat16);
    TORCH_CHECK(weight_type == input_type,
        "weight type must equal to input type, other variations are disabled due to binary size limitations");
    TORCH_CHECK(conv_state.scalar_type() == input_type);

    TORCH_CHECK(x.is_cuda());
    TORCH_CHECK(conv_state.is_cuda());
    TORCH_CHECK(weight.is_cuda());

    auto const sizes = x.sizes();
    int const batch_size = sizes[0];
    int const dim = sizes[1];
    int const seqlen = sizes[2];
    int const width = weight.size(-1);
    int const conv_state_len = conv_state.size(2);
    TORCH_CHECK(conv_state_len >= width - 1);

    CHECK_SHAPE(x, batch_size, dim, seqlen);
    CHECK_SHAPE(weight, dim, width);

    TORCH_CHECK(width >= 2 && width <= 4, "causal_conv1d only supports width between 2 and 4");

    if (bias_.has_value())
    {
        auto bias = bias_.value();
        TORCH_CHECK(bias.scalar_type() == weight_type);
        TORCH_CHECK(bias.is_cuda());
        TORCH_CHECK(bias.stride(-1) == 1);
        CHECK_SHAPE(bias, dim);
    }

    at::Tensor out = x;

    tensorrt_llm::kernels::causal_conv1d::ConvParamsBase params;
    set_conv_params_fwd(params, batch_size, dim, seqlen, width, x, weight, out, bias_, silu_activation, pad_slot_id);
    params.conv_state_ptr = conv_state.data_ptr();
    params.conv_state_len = conv_state_len;
    // All stride are in elements, not bytes.
    params.conv_state_batch_stride = conv_state.stride(0);
    params.conv_state_c_stride = conv_state.stride(1);
    params.conv_state_l_stride = conv_state.stride(2);

    if (cache_seqlens_.has_value())
    {
        auto cache_seqlens = cache_seqlens_.value();
        TORCH_CHECK(cache_seqlens.scalar_type() == torch::kInt32);
        TORCH_CHECK(cache_seqlens.is_cuda());
        TORCH_CHECK(cache_seqlens.stride(-1) == 1);
        CHECK_SHAPE(cache_seqlens, batch_size);
        params.cache_seqlens = cache_seqlens.data_ptr<int32_t>();
    }
    else
    {
        params.cache_seqlens = nullptr;
    }

    if (conv_state_indices_.has_value())
    {
        auto conv_state_indices = conv_state_indices_.value();
        TORCH_CHECK(conv_state_indices.scalar_type() == torch::kInt32)
        TORCH_CHECK(conv_state_indices.is_cuda());
        TORCH_CHECK(conv_state_indices.stride(0) == 1)
        CHECK_SHAPE(conv_state_indices, batch_size);

        int conv_state_entries = conv_state.size(0);
        CHECK_SHAPE(conv_state, conv_state_entries, dim, conv_state_len);

        params.conv_state_indices_ptr = conv_state_indices.data_ptr<int32_t>();
    }
    else
    {
        CHECK_SHAPE(conv_state, batch_size, dim, conv_state_len);
        params.conv_state_indices_ptr = nullptr;
    }

    // Otherwise the kernel will be launched from cuda:0 device
    // Static cast to signed char (AKA c10::DeviceIndex - the input to CUDAGuard) to avoid compiler warning about
    // narrowing
    at::cuda::CUDAGuard device_guard{static_cast<signed char>(x.get_device())};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    DISPATCH_WTYPE_ITYPE_FLOAT_AND_HALF_AND_BF16(x.scalar_type(), "causal_conv1d_update",
        [&] { tensorrt_llm::kernels::causal_conv1d::causal_conv1d_update_cuda<input_t, weight_t>(params, stream); });
}

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "causal_conv1d_fwd(Tensor! x,"
        "Tensor! weight,"
        "Tensor? bias_,"
        "Tensor!? conv_states,"
        "Tensor? query_start_loc,"
        "Tensor? cache_indices,"
        "Tensor? has_initial_state,"
        "bool silu_activation,"
        "int pad_slot_id) -> ()");

    m.def(
        "causal_conv1d_update(Tensor! x,"
        "Tensor! conv_state,"
        "Tensor! weight,"
        "Tensor? bias_,"
        "bool silu_activation,"
        "Tensor? cache_seqlens_,"
        "Tensor? conv_state_indices,"
        "int pad_slot_id) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("causal_conv1d_fwd", &torch_ext::causalConv1dFwd);
    m.impl("causal_conv1d_update", &torch_ext::causalConv1dUpdate);
}
