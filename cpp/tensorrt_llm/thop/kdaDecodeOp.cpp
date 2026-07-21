/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

#include "tensorrt_llm/kernels/kdaDecode/kdaDecode.h"

#include <ATen/Functions.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <optional>
#include <torch/library.h>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

constexpr int kDimK = 128;
constexpr int kDimV = 128;
constexpr int kKernelWidth = 4;

void validate_kda_decode_fusion_inputs(at::Tensor x_q, at::Tensor x_k, at::Tensor x_v, at::Tensor w_q_t,
    at::Tensor w_k_t, at::Tensor w_v_t, at::Tensor bias_q, at::Tensor bias_k, at::Tensor bias_v, at::Tensor cs_q,
    at::Tensor cs_k, at::Tensor cs_v, at::Tensor a_log, at::Tensor g, at::Tensor dt_bias, at::Tensor beta,
    at::Tensor onorm_g, at::Tensor onorm_weight, at::Tensor ssm_state_indices, at::Tensor cu_seqlens, at::Tensor state,
    at::Tensor out, bool update_conv_cache)
{
    TORCH_CHECK(x_q.is_cuda() && x_q.scalar_type() == at::kBFloat16, "x_q must be a CUDA bfloat16 tensor");
    TORCH_CHECK(x_k.is_cuda() && x_k.scalar_type() == at::kBFloat16, "x_k must be a CUDA bfloat16 tensor");
    TORCH_CHECK(x_v.is_cuda() && x_v.scalar_type() == at::kBFloat16, "x_v must be a CUDA bfloat16 tensor");
    TORCH_CHECK(w_q_t.is_cuda() && w_q_t.scalar_type() == at::kBFloat16, "w_q_t must be a CUDA bfloat16 tensor");
    TORCH_CHECK(w_k_t.is_cuda() && w_k_t.scalar_type() == at::kBFloat16, "w_k_t must be a CUDA bfloat16 tensor");
    TORCH_CHECK(w_v_t.is_cuda() && w_v_t.scalar_type() == at::kBFloat16, "w_v_t must be a CUDA bfloat16 tensor");
    TORCH_CHECK(bias_q.is_cuda() && bias_q.scalar_type() == at::kBFloat16, "bias_q must be a CUDA bfloat16 tensor");
    TORCH_CHECK(bias_k.is_cuda() && bias_k.scalar_type() == at::kBFloat16, "bias_k must be a CUDA bfloat16 tensor");
    TORCH_CHECK(bias_v.is_cuda() && bias_v.scalar_type() == at::kBFloat16, "bias_v must be a CUDA bfloat16 tensor");
    TORCH_CHECK(cs_q.is_cuda() && cs_q.scalar_type() == at::kBFloat16, "cs_q must be a CUDA bfloat16 tensor");
    TORCH_CHECK(cs_k.is_cuda() && cs_k.scalar_type() == at::kBFloat16, "cs_k must be a CUDA bfloat16 tensor");
    TORCH_CHECK(cs_v.is_cuda() && cs_v.scalar_type() == at::kBFloat16, "cs_v must be a CUDA bfloat16 tensor");
    TORCH_CHECK(g.is_cuda() && g.scalar_type() == at::kBFloat16, "g must be a CUDA bfloat16 tensor");
    TORCH_CHECK(beta.is_cuda() && beta.scalar_type() == at::kBFloat16, "beta must be a CUDA bfloat16 tensor");
    TORCH_CHECK(onorm_g.is_cuda() && onorm_g.scalar_type() == at::kBFloat16, "onorm_g must be a CUDA bfloat16 tensor");
    TORCH_CHECK(a_log.is_cuda() && a_log.scalar_type() == at::kFloat, "a_log must be a CUDA float32 tensor");
    TORCH_CHECK(dt_bias.is_cuda() && dt_bias.scalar_type() == at::kFloat, "dt_bias must be a CUDA float32 tensor");
    TORCH_CHECK(onorm_weight.is_cuda() && onorm_weight.scalar_type() == at::kFloat,
        "onorm_weight must be a CUDA float32 tensor");
    TORCH_CHECK(state.is_cuda() && state.scalar_type() == at::kFloat, "state must be a CUDA float32 tensor");
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kBFloat16, "out must be a CUDA bfloat16 tensor");

    TORCH_CHECK(x_q.dim() == 4 && x_k.dim() == 4 && x_v.dim() == 4, "x_q, x_k, and x_v must be rank-4 tensors");
    TORCH_CHECK(x_q.size(0) == 1 && x_k.size(0) == 1 && x_v.size(0) == 1, "only T=1 decode inputs are supported");
    TORCH_CHECK(x_q.size(3) == kDimK && x_k.size(3) == kDimK, "only K=128 is supported");
    TORCH_CHECK(x_v.size(3) == kDimV, "only V=128 is supported");
    TORCH_CHECK(w_q_t.size(0) == kKernelWidth && w_k_t.size(0) == kKernelWidth && w_v_t.size(0) == kKernelWidth,
        "only convolution width 4 is supported");

    int const B = static_cast<int>(x_q.size(1));
    int const H = static_cast<int>(x_q.size(2));
    int const HV = static_cast<int>(x_v.size(2));
    TORCH_CHECK((B == 128 && H == 2 && HV == 2) || (B == 32 && H == 12 && HV == 12),
        "KDA decode fusion CUDA supports only (B,H,HV)=(128,2,2) and (32,12,12)");
    TORCH_CHECK(x_k.size(1) == B && x_k.size(2) == H && x_v.size(1) == B,
        "x_q, x_k, and x_v batch/head dimensions are inconsistent");
    TORCH_CHECK(HV % H == 0, "HV must be divisible by H");
    TORCH_CHECK(state.is_contiguous(), "state must be contiguous");
    TORCH_CHECK(state.size(0) >= B && state.size(1) == HV && state.size(2) == kDimV && state.size(3) == kDimK,
        "state must have shape [slots, HV, 128, 128] with slots >= B");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
    TORCH_CHECK(out.size(0) == B && out.size(1) == 1 && out.size(2) == HV && out.size(3) == kDimV,
        "out must have shape [B, 1, HV, 128]");

    TORCH_CHECK(ssm_state_indices.is_cuda() && ssm_state_indices.scalar_type() == at::kInt,
        "ssm_state_indices must be a CUDA int32 tensor");
    TORCH_CHECK(cu_seqlens.is_cuda() && cu_seqlens.scalar_type() == at::kInt, "cu_seqlens must be a CUDA int32 tensor");

    if (update_conv_cache)
    {
        TORCH_CHECK(H == HV, "conv state update currently assumes H == HV");
        TORCH_CHECK(cs_q.stride(1) == 1 && cs_k.stride(1) == 1,
            "update_conv_cache expects cs_q/cs_k transposed layout with "
            "contiguous dim axis");
        TORCH_CHECK(cs_q.stride(2) == H * kDimK && cs_k.stride(2) == H * kDimK,
            "update_conv_cache expects cs_q/cs_k token stride H*K");
        TORCH_CHECK(cs_v.stride(1) == 1,
            "update_conv_cache expects cs_v transposed layout with "
            "contiguous dim axis");
        TORCH_CHECK(cs_v.stride(2) == HV * kDimV, "update_conv_cache expects cs_v token stride HV*V");
    }
    else
    {
        TORCH_CHECK(cs_q.is_contiguous(), "cs_q must be contiguous [B, H*K, 3]");
        TORCH_CHECK(cs_k.is_contiguous(), "cs_k must be contiguous [B, H*K, 3]");
        TORCH_CHECK(cs_v.is_contiguous(), "cs_v must be contiguous [B, HV*V, 3]");
    }
}

void launch_selected_kernel(at::Tensor x_q, at::Tensor x_k, at::Tensor x_v, at::Tensor w_q_t, at::Tensor w_k_t,
    at::Tensor w_v_t, at::Tensor bias_q, at::Tensor bias_k, at::Tensor bias_v, at::Tensor cs_q, at::Tensor cs_k,
    at::Tensor cs_v, at::Tensor a_log, at::Tensor g, at::Tensor dt_bias, at::Tensor beta, at::Tensor onorm_g,
    at::Tensor onorm_weight, at::Tensor ssm_state_indices, at::Tensor cu_seqlens, at::Tensor state, at::Tensor out,
    bool apply_onorm, bool update_conv_cache, bool use_lower_bound, bool apply_beta_sigmoid, double lower_bound,
    double scale, double onorm_eps)
{
    int const B = static_cast<int>(x_q.size(1));
    int const H = static_cast<int>(x_q.size(2));
    int const HV = static_cast<int>(x_v.size(2));

    const tensorrt_llm::kernels::kdaDecode::KdaDecodeParams params{x_q.data_ptr(), x_k.data_ptr(), x_v.data_ptr(),
        w_q_t.data_ptr(), w_k_t.data_ptr(), w_v_t.data_ptr(), bias_q.data_ptr(), bias_k.data_ptr(), bias_v.data_ptr(),
        cs_q.data_ptr(), cs_k.data_ptr(), cs_v.data_ptr(), a_log.data_ptr<float>(), g.data_ptr(),
        dt_bias.data_ptr<float>(), beta.data_ptr(), onorm_g.data_ptr(), onorm_weight.data_ptr<float>(),
        ssm_state_indices.data_ptr<int>(), cu_seqlens.data_ptr<int>(), state.data_ptr<float>(), out.data_ptr(), B, H,
        HV, apply_onorm, update_conv_cache, use_lower_bound, apply_beta_sigmoid, static_cast<float>(lower_bound),
        static_cast<float>(scale), static_cast<float>(onorm_eps)};

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    tensorrt_llm::kernels::kdaDecode::invokeKdaDecode(params, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor kda_decode_fusion_forward(at::Tensor x_q, at::Tensor x_k, at::Tensor x_v, at::Tensor w_q_t, at::Tensor w_k_t,
    at::Tensor w_v_t, at::Tensor bias_q, at::Tensor bias_k, at::Tensor bias_v, at::Tensor cs_q, at::Tensor cs_k,
    at::Tensor cs_v, at::Tensor a_log, at::Tensor g, at::Tensor dt_bias, at::Tensor beta, at::Tensor onorm_g,
    at::Tensor onorm_weight, at::Tensor ssm_state_indices, at::Tensor cu_seqlens, at::Tensor state, bool apply_onorm,
    bool update_conv_cache, bool use_lower_bound, bool apply_beta_sigmoid, double lower_bound, double scale,
    double onorm_eps, std::optional<at::Tensor> output)
{
    int const B = static_cast<int>(x_q.size(1));
    int const HV = static_cast<int>(x_v.size(2));
    auto out = output.has_value() ? *output : at::empty({B, 1, HV, kDimV}, x_q.options());
    validate_kda_decode_fusion_inputs(x_q, x_k, x_v, w_q_t, w_k_t, w_v_t, bias_q, bias_k, bias_v, cs_q, cs_k, cs_v,
        a_log, g, dt_bias, beta, onorm_g, onorm_weight, ssm_state_indices, cu_seqlens, state, out, update_conv_cache);
    launch_selected_kernel(x_q, x_k, x_v, w_q_t, w_k_t, w_v_t, bias_q, bias_k, bias_v, cs_q, cs_k, cs_v, a_log, g,
        dt_bias, beta, onorm_g, onorm_weight, ssm_state_indices, cu_seqlens, state, out, apply_onorm, update_conv_cache,
        use_lower_bound, apply_beta_sigmoid, lower_bound, scale, onorm_eps);
    return out;
}

} // namespace

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "kda_decode(Tensor x_q, Tensor x_k, Tensor x_v, Tensor w_q_t, "
        "Tensor w_k_t, Tensor w_v_t, Tensor bias_q, Tensor bias_k, "
        "Tensor bias_v, Tensor(a!) conv_state_q, Tensor(b!) conv_state_k, "
        "Tensor(c!) conv_state_v, Tensor a_log, Tensor g, Tensor dt_bias, "
        "Tensor beta, Tensor onorm_g, Tensor onorm_weight, "
        "Tensor ssm_state_indices, Tensor cu_seqlens, Tensor(d!) state, "
        "bool apply_onorm, bool update_conv_cache, bool use_lower_bound, "
        "bool apply_beta_sigmoid, float lower_bound, float scale, "
        "float onorm_eps, Tensor(e!)? output=None) -> Tensor(e!)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("kda_decode", &tensorrt_llm::torch_ext::kda_decode_fusion_forward);
}
