/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/kernels/mamba2MTPSSMCache/mamba2MTPSSMCache.h"
#include "tensorrt_llm/thop/thUtils.h"

namespace th = torch;
namespace tk = tensorrt_llm::kernels;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

tk::Mamba2Dtype torchToMamba2Dtype(c10::ScalarType dtype)
{
    switch (dtype)
    {
    case c10::ScalarType::BFloat16: return tk::Mamba2Dtype::kBFloat16;
    case c10::ScalarType::Half: return tk::Mamba2Dtype::kFloat16;
    case c10::ScalarType::Float: return tk::Mamba2Dtype::kFloat32;
    default: TORCH_CHECK(false, "Unsupported dtype for mamba2 MTP SSM cache kernel"); return tk::Mamba2Dtype::kFloat32;
    }
}

} // anonymous namespace

void mamba2_mtp_ssm_cache_update(th::Tensor ssm, th::Tensor x, th::Tensor dt, th::Tensor A, th::Tensor B, th::Tensor C,
    th::Tensor out, th::Tensor intermediate_states, th::optional<th::Tensor> D, th::optional<th::Tensor> z,
    th::optional<th::Tensor> dt_bias, bool const dt_softplus, th::optional<th::Tensor> ssm_batch_indices,
    th::optional<th::Tensor> intermediate_states_indices, th::optional<th::Tensor> retrieve_parent_token,
    int64_t const cache_steps, int64_t const pad_slot_id, bool const disable_state_update)
{
    TORCH_CHECK(ssm.dim() == 4 && ssm.is_cuda() && ssm.is_contiguous(), "ssm should be a 4D contiguous CUDA tensor");
    TORCH_CHECK(x.dim() == 4, "x should be a 4D tensor");
    auto device = ssm.device();
    int const bs = x.size(0);
    int const nheads = ssm.size(1);
    int const head_dim = ssm.size(2);
    int const ssm_dim = ssm.size(3);

    TORCH_CHECK(intermediate_states.dim() == 5 && intermediate_states.size(0) == ssm.size(0)
            && intermediate_states.size(1) == cache_steps && intermediate_states.size(2) == nheads
            && intermediate_states.size(3) == head_dim && intermediate_states.size(4) == ssm_dim,
        "intermediate_states shape check failed");
    TORCH_CHECK(intermediate_states.device() == device && intermediate_states.is_contiguous(),
        "intermediate_states is not a contiguous tensor of the same device as ssm");

    TORCH_CHECK(
        x.size(1) == cache_steps && x.size(2) == nheads && x.size(3) == head_dim, "x tensor has incorrect shapes");
    TORCH_CHECK(x.device() == device && x.is_contiguous(), "x is not a contiguous tensor of the same device as ssm");

    TORCH_CHECK(dt.sizes() == x.sizes(), "dt tensor has the incorrect shape");
    TORCH_CHECK(dt.device() == device, "dt is not a tensor of the same device as ssm");
    TORCH_CHECK(dt.stride(2) == 1 && dt.stride(3) == 0, "dt tensor's strides should be (1, 0)");

    TORCH_CHECK(A.dim() == 3 && A.size(0) == nheads && A.size(1) == head_dim && A.size(2) == ssm_dim,
        "A tensor has incorrect shapes");
    TORCH_CHECK(A.device() == device, "A is not a tensor of the same device as ssm");
    TORCH_CHECK(A.stride(0) == 1 && A.stride(1) == 0 && A.stride(2) == 0, "A tensor's strides should be (1, 0, 0)");

    TORCH_CHECK(B.dim() == 4 && B.size(0) == bs && B.size(1) == cache_steps && B.size(3) == ssm_dim,
        "B tensor has incorrect shapes");
    TORCH_CHECK(B.device() == device && B.is_contiguous(), "B is not a contiguous tensor of the same device as ssm");
    int const ngroups = B.size(2);
    TORCH_CHECK(ngroups > 0 && (nheads % ngroups == 0), "unsupported pair of nheads and ngroups");

    TORCH_CHECK(C.sizes() == B.sizes(), "C tensor has the incorrect shape");
    TORCH_CHECK(C.device() == device && C.is_contiguous(), "C is not a contiguous tensor of the same device as ssm");

    TORCH_CHECK(out.sizes() == x.sizes(), "out tensor has the incorrect shape");
    TORCH_CHECK(
        out.device() == device && out.is_contiguous(), "out is not a contiguous tensor of the same device as ssm");

    auto in_out_dtype = x.dtype();
    TORCH_CHECK(B.dtype() == in_out_dtype && C.dtype() == in_out_dtype && out.dtype() == in_out_dtype,
        "In out tensors dtype check fail");
    TORCH_CHECK(intermediate_states.dtype() == ssm.dtype(), "intermediate states dtype check fail");

    // Build params
    tk::Mamba2MTPSSMCacheParams params;
    params.ssm = ssm.data_ptr();
    params.intermediate_states = intermediate_states.data_ptr();
    params.x = x.data_ptr();
    params.dt = dt.data_ptr();
    params.A = A.data_ptr();
    params.B = B.data_ptr();
    params.C = C.data_ptr();
    params.out = out.data_ptr();

    if (D.has_value())
    {
        auto D_val = D.value();
        TORCH_CHECK(
            D_val.dim() == 2 && D_val.size(0) == nheads && D_val.size(1) == head_dim && D_val.dtype() == dt.dtype(),
            "D tensor has the incorrect shape or dtype");
        TORCH_CHECK(D_val.device() == device, "D is not a tensor of the same device as ssm");
        TORCH_CHECK(D_val.stride(0) == 1 && D_val.stride(1) == 0, "D tensor's strides should be (1, 0)");
        params.D = D_val.data_ptr();
    }
    else
    {
        params.D = nullptr;
    }

    if (z.has_value())
    {
        auto z_val = z.value();
        TORCH_CHECK(z_val.sizes() == x.sizes() && z_val.dtype() == in_out_dtype,
            "z tensor doesn't have the same shape and dtype with x");
        TORCH_CHECK(z_val.device() == device && z_val.is_contiguous(),
            "z is not a contiguous tensor of the same device as ssm");
        params.z = z_val.data_ptr();
    }
    else
    {
        params.z = nullptr;
    }

    if (dt_bias.has_value())
    {
        auto dt_bias_val = dt_bias.value();
        TORCH_CHECK(dt_bias_val.dim() == 2 && dt_bias_val.size(0) == nheads && dt_bias_val.size(1) == head_dim
                && dt_bias_val.dtype() == dt.dtype(),
            "dt_bias tensor has the incorrect shape or dtype");
        TORCH_CHECK(dt_bias_val.device() == device, "dt_bias is not a tensor of the same device as ssm");
        TORCH_CHECK(
            dt_bias_val.stride(0) == 1 && dt_bias_val.stride(1) == 0, "dt_bias tensor's strides should be (1, 0)");
        params.dt_bias = dt_bias_val.data_ptr();
    }
    else
    {
        params.dt_bias = nullptr;
    }

    if (ssm_batch_indices.has_value())
    {
        auto idx = ssm_batch_indices.value();
        TORCH_CHECK(idx.device() == device && idx.is_contiguous(),
            "ssm_batch_indices is not a contiguous tensor of the same device as ssm");
        TORCH_CHECK(idx.dim() == 1 && idx.size(0) == bs && idx.dtype() == torch::kInt32,
            "ssm_indices is not an int32 [batch] tensor");
        params.ssm_batch_indices = static_cast<int32_t const*>(idx.const_data_ptr());
    }
    else
    {
        params.ssm_batch_indices = nullptr;
    }

    if (intermediate_states_indices.has_value())
    {
        auto idx = intermediate_states_indices.value();
        TORCH_CHECK(idx.device() == device && idx.is_contiguous(),
            "intermediate_states_indices is not a contiguous tensor of the same device as ssm");
        TORCH_CHECK(idx.dim() == 1 && idx.size(0) == bs && idx.dtype() == torch::kInt32,
            "intermediate_indices is not an int32 [batch] tensor");
        params.intermediate_states_indices = static_cast<int32_t const*>(idx.const_data_ptr());
    }
    else
    {
        params.intermediate_states_indices = nullptr;
    }

    if (retrieve_parent_token.has_value())
    {
        auto rpt = retrieve_parent_token.value();
        TORCH_CHECK(rpt.device() == device && rpt.is_contiguous(),
            "retrieve_parent_token is not a contiguous tensor of the same device as ssm");
        TORCH_CHECK(rpt.dim() == 2 && rpt.size(0) == bs && rpt.size(1) == cache_steps && rpt.dtype() == torch::kInt32,
            "retrieve_parent_token is not an int32 [batch, cache_steps] tensor");
        params.retrieve_parent_token = static_cast<int32_t const*>(rpt.const_data_ptr());
    }
    else
    {
        params.retrieve_parent_token = nullptr;
    }

    params.dt_softplus = dt_softplus;
    params.cache_steps = static_cast<int>(cache_steps);
    params.pad_slot_id = static_cast<int>(pad_slot_id);
    params.disable_state_update = disable_state_update;
    params.bs = bs;
    params.nheads = nheads;
    params.head_dim = head_dim;
    params.ssm_dim = ssm_dim;
    params.ngroups = ngroups;

    params.ssm_dtype = torchToMamba2Dtype(ssm.scalar_type());
    params.in_out_dtype = torchToMamba2Dtype(x.scalar_type());
    params.weight_dtype = torchToMamba2Dtype(dt.scalar_type());
    params.a_dtype = torchToMamba2Dtype(A.scalar_type());

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    tk::invokeMamba2MTPSSMCacheUpdate(params, stream);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "mamba2_mtp_ssm_cache_update(Tensor ssm, Tensor x, "
        "Tensor dt, Tensor A, Tensor B, Tensor C, "
        "Tensor out, Tensor intermediate_states, "
        "Tensor? D, Tensor? z, Tensor? dt_bias, "
        "bool dt_softplus, Tensor? ssm_batch_indices, "
        "Tensor? intermediate_states_indices, "
        "Tensor? retrieve_parent_token, "
        "int cache_steps, int pad_slot_id, bool disable_state_update) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("mamba2_mtp_ssm_cache_update", &tensorrt_llm::torch_ext::mamba2_mtp_ssm_cache_update);
}
