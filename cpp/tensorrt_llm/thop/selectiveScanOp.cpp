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
#include "tensorrt_llm/kernels/selectiveScan/selectiveScan.h"
#include "tensorrt_llm/thop/thUtils.h"

namespace th = torch;
namespace tk = tensorrt_llm::kernels;

namespace torch_ext
{

template <typename T, c10::ScalarType U>
std::tuple<th::Tensor, th::Tensor> run_selective_scan(th::Tensor const& input, th::Tensor const& state,
    th::Tensor const& delta, th::Tensor const& delta_bias, th::Tensor const& A, th::Tensor const& BC,
    th::Tensor const& D, th::Tensor const& host_request_types, th::Tensor const& last_token_ids,
    th::optional<th::Tensor> z, th::optional<th::Tensor> host_context_lengths, th::optional<th::Tensor> slot_mapping,
    int64_t const dim, int64_t const dstate, int64_t const nheads, int64_t const ngroups, int64_t const chunk_size,
    int64_t const delta_rank, bool const delta_softplus, bool const remove_padding, bool const is_mamba2,
    bool const is_paged_state)
{

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto driver = tensorrt_llm::common::CUDADriverWrapper::getInstance();

    tk::SSMParamsBase params;

    auto host_request_sizes = host_request_types.sizes();
    auto input_sizes = input.sizes();
    auto last_token_ids_sizes = last_token_ids.sizes();

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

    int num_tokens = input_sizes[0];

    if (!remove_padding)
    {
        num_tokens *= input_sizes[1];
    }

    // req_type=0 -> context (prefill)
    // req_type=1 -> generation (decode)
    auto req_type = host_request_types[0].item<int>();

    std::vector<int64_t> out_shape;
    if (remove_padding)
    {
        out_shape = {input_sizes[0], dim};
    }
    else
    {
        out_shape = {input_sizes[0], input_sizes[1], dim};
    }

    auto out = torch::empty(out_shape, input.options());

    params.batch = batch_size;
    params.dim = dim;
    params.max_seqlen = max_seqlen;
    params.num_tokens = num_tokens;
    params.dstate = dstate;
    params.dt_rank = delta_rank;
    params.nheads = nheads;
    params.ngroups = ngroups;
    params.chunk_size = chunk_size;
    params.delta_softplus = delta_softplus;
    params.remove_padding = remove_padding;
    params.is_mamba2 = is_mamba2;

    params.u_ptr = input.data_ptr();
    params.delta_ptr = delta.data_ptr();
    params.delta_bias_ptr = delta_bias.data_ptr();
    params.A_ptr = A.data_ptr();
    params.BC_ptr = BC.data_ptr();
    params.D_ptr = D.data_ptr();
    params.last_token_ids_ptr = static_cast<int const*>(last_token_ids.const_data_ptr());
    params.out_ptr = out.data_ptr();

    if (is_paged_state)
    {
        if (!slot_mapping.has_value())
        {
            throw std::invalid_argument("slot_mapping must be provided when paged state is enabled");
        }

        params.x_ptr = *reinterpret_cast<void**>(const_cast<void*>(state.data_ptr()));
        params.slot_mapping_ptr = static_cast<int const*>(slot_mapping.value().const_data_ptr());
    }
    else
    {
        params.x_ptr = state.data_ptr();
        params.slot_mapping_ptr = nullptr;
    }

    if (z.has_value())
    {
        params.z_ptr = z.value().data_ptr();
    }
    else
    {
        params.z_ptr = nullptr;
    }

    // workspace tensors
    if (!is_mamba2 || req_type == 1)
    { // doesn't require workspace
        params.Os_ptr = nullptr;
        params.St_ptr = nullptr;
        params.dc_ptr = nullptr;
        params.dA_ptr = nullptr;
        params.CB_ptr = nullptr;
        params.desc_ptr = nullptr;
    }
    else if (remove_padding)
    {
        int B = last_token_ids_sizes[0];
        int BxL = input_sizes[0]; // num_tokens
        int H = nheads;
        int P = dim / H;
        int G = ngroups;
        int N = dstate;
        int Q = chunk_size;
        int BxC = (BxL + Q - 1) / Q + B;

        auto mxOs = torch::empty({long(BxC) * H * N * P * 2}, torch::dtype(U).device(torch::kCUDA));
        auto mxCB = torch::empty({long(BxC) * G * Q * Q * 2}, torch::dtype(U).device(torch::kCUDA));

        auto mxSt = torch::empty({long(BxC) * H * N * P * 4}, torch::dtype(torch::kFloat16).device(torch::kCUDA));
        auto mxdc = torch::empty({long(BxC) * H * Q * 4}, torch::dtype(torch::kFloat16).device(torch::kCUDA));
        auto mxdA = torch::empty({long(BxC) * H * Q * 4}, torch::dtype(torch::kFloat16).device(torch::kCUDA));
        auto desc = torch::empty({1024}, torch::dtype(torch::kInt64).device(torch::kCUDA));

        params.Os_ptr = mxOs.data_ptr();
        params.St_ptr = mxSt.data_ptr();
        params.dc_ptr = mxdc.data_ptr();
        params.dA_ptr = mxdA.data_ptr();
        params.CB_ptr = mxCB.data_ptr();
        params.desc_ptr = desc.data_ptr();
    }
    else
    {
        int B = input_sizes[0];
        int L = input_sizes[1];
        int H = nheads;
        int P = dim / H;
        int G = ngroups;
        int N = dstate;
        int Q = chunk_size;
        int C = (L + Q - 1) / Q;

        auto mxOs = torch::empty({long(B * C) * H * N * P * 2}, torch::dtype(U).device(torch::kCUDA));
        auto mxCB = torch::empty({long(B * C) * G * Q * Q * 2}, torch::dtype(U).device(torch::kCUDA));

        auto mxSt = torch::empty({long(B * C) * H * N * P * 4}, torch::dtype(torch::kFloat16).device(torch::kCUDA));
        auto mxdc = torch::empty({long(B * C) * H * Q * 4}, torch::dtype(torch::kFloat16).device(torch::kCUDA));
        auto mxdA = torch::empty({long(B * C) * H * Q * 4}, torch::dtype(torch::kFloat16).device(torch::kCUDA));
        auto desc = torch::empty({1024}, torch::dtype(torch::kInt64).device(torch::kCUDA));

        params.Os_ptr = mxOs.data_ptr();
        params.St_ptr = mxSt.data_ptr();
        params.dc_ptr = mxdc.data_ptr();
        params.dA_ptr = mxdA.data_ptr();
        params.CB_ptr = mxCB.data_ptr();
        params.desc_ptr = desc.data_ptr();
    }

    if (req_type == 0)
    {
        if (is_mamba2)
        {
            tk::invokeChunkScan<T, float>(params, stream, driver.get());
        }
        else
        {
            tk::invokeSelectiveScan<T, float>(params, stream);
        }
    }
    else if (req_type == 1)
    {
        tk::invokeSelectiveScanUpdate<T, float>(params, stream);
    }

    sync_check_cuda_error(stream);

    return std::make_tuple(out, state);
}

std::tuple<th::Tensor, th::Tensor> selective_scan(th::Tensor const& input, th::Tensor const& state,
    th::Tensor const& delta, th::Tensor const& delta_bias, th::Tensor const& A, th::Tensor const& BC,
    th::Tensor const& D, th::Tensor const& host_request_types, th::Tensor const& last_token_ids,
    th::optional<th::Tensor> z, th::optional<th::Tensor> host_context_lengths, th::optional<th::Tensor> slot_mapping,
    int64_t const dim, int64_t const dstate, int64_t const nheads, int64_t const ngroups, int64_t const chunk_size,
    int64_t const delta_rank, bool const delta_softplus, bool const remove_padding, bool const is_mamba2,
    bool const is_paged_state)
{
    c10::ScalarType dtype = input.scalar_type();

    switch (dtype)
    {
    case torch::kFloat16:
        // Handle Float16
        return run_selective_scan<half, torch::kFloat16>(input, state, delta, delta_bias, A, BC, D, host_request_types,
            last_token_ids, z, host_context_lengths, slot_mapping, dim, dstate, nheads, ngroups, chunk_size, delta_rank,
            delta_softplus, remove_padding, is_mamba2, is_paged_state);
    case torch::kFloat32:
        // Handle Float32
        return run_selective_scan<float, torch::kFloat32>(input, state, delta, delta_bias, A, BC, D, host_request_types,
            last_token_ids, z, host_context_lengths, slot_mapping, dim, dstate, nheads, ngroups, chunk_size, delta_rank,
            delta_softplus, remove_padding, is_mamba2, is_paged_state);
    case torch::kBFloat16:
        // Handle BFloat16
        return run_selective_scan<__nv_bfloat16, torch::kBFloat16>(input, state, delta, delta_bias, A, BC, D,
            host_request_types, last_token_ids, z, host_context_lengths, slot_mapping, dim, dstate, nheads, ngroups,
            chunk_size, delta_rank, delta_softplus, remove_padding, is_mamba2, is_paged_state);
    default:
        // Handle other data types
        throw std::invalid_argument("Invalid dtype, only supports float16, float32, and bfloat16");
    }
}

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "selective_scan(Tensor input, Tensor state, "
        "Tensor delta, Tensor delta_bias, Tensor A, "
        "Tensor BC, Tensor D, Tensor host_request_types, "
        "Tensor last_token_ids, Tensor? z, "
        "Tensor? host_context_lengths, Tensor? slot_mapping, "
        "int dim, int dstate, int nheads, int ngroups, "
        "int chunk_size, int delta_rank, bool delta_softplus, "
        "bool remove_padding, bool is_mamba2, bool is_paged_state) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("selective_scan", &torch_ext::selective_scan);
}
