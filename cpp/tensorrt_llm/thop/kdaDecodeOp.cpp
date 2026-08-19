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
#include <cstdint>
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

//! Batch-row stride of a rank-4 ``[1, B, heads, dim]`` decode input.
//!
//! The kernel walks these tensors as ``row * rowStride + head * dim + i``.
//! Fused-projection column views may therefore keep a wider row stride as
//! long as their head and channel axes are packed.
int64_t token_row_stride(at::Tensor const& tensor, char const* name)
{
    TORCH_CHECK(tensor.stride(3) == 1 && tensor.stride(2) == tensor.size(3), name,
        " must be contiguous across its head and channel axes; only the batch-row stride may vary");
    return tensor.stride(1);
}

void validate_kda_decode_fusion_inputs(at::Tensor x_q, at::Tensor x_k, at::Tensor x_v, at::Tensor w_q_t,
    at::Tensor w_k_t, at::Tensor w_v_t, at::Tensor bias_q, at::Tensor bias_k, at::Tensor bias_v, at::Tensor cs_q,
    at::Tensor cs_k, at::Tensor cs_v, at::Tensor a_log, at::Tensor g, at::Tensor dt_bias, at::Tensor beta,
    at::Tensor onorm_g, at::Tensor onorm_weight, std::optional<at::Tensor> const& ssm_state_indices,
    at::Tensor cu_seqlens, at::Tensor state, bool apply_onorm, bool update_conv_cache)
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

    TORCH_CHECK(x_q.dim() == 4 && x_k.dim() == 4 && x_v.dim() == 4, "x_q, x_k, and x_v must be rank-4 tensors");
    TORCH_CHECK(x_q.size(0) == 1 && x_k.size(0) == 1 && x_v.size(0) == 1, "only T=1 decode inputs are supported");
    TORCH_CHECK(x_q.size(3) == kDimK && x_k.size(3) == kDimK, "only K=128 is supported");
    TORCH_CHECK(x_v.size(3) == kDimV, "only V=128 is supported");
    token_row_stride(x_q, "x_q");
    token_row_stride(x_k, "x_k");
    token_row_stride(x_v, "x_v");
    TORCH_CHECK(w_q_t.dim() == 2 && w_k_t.dim() == 2 && w_v_t.dim() == 2, "w_q_t, w_k_t, and w_v_t must be rank-2");
    TORCH_CHECK(w_q_t.size(0) == kKernelWidth && w_k_t.size(0) == kKernelWidth && w_v_t.size(0) == kKernelWidth,
        "only convolution width 4 is supported");
    TORCH_CHECK(w_q_t.is_contiguous() && w_k_t.is_contiguous() && w_v_t.is_contiguous(),
        "w_q_t, w_k_t, and w_v_t must be contiguous [4, dim] tensors");

    int const B = static_cast<int>(x_q.size(1));
    int const H = static_cast<int>(x_q.size(2));
    int const HV = static_cast<int>(x_v.size(2));
    TORCH_CHECK(B > 0, "KDA decode requires a non-empty batch");
    bool const supportedHeads = H == 1 || H == 2 || H == 3 || H == 4 || H == 6 || H == 8 || H == 12 || H == 16
        || H == 24 || H == 32 || H == 48 || H == 96;
    TORCH_CHECK(
        H == HV && supportedHeads, "KDA decode fusion CUDA supports H == HV in {1,2,3,4,6,8,12,16,24,32,48,96}");
    TORCH_CHECK(x_k.size(1) == B && x_k.size(2) == H && x_v.size(1) == B,
        "x_q, x_k, and x_v batch/head dimensions are inconsistent");
    TORCH_CHECK(HV % H == 0, "HV must be divisible by H");

    int64_t const qk_dim = static_cast<int64_t>(H) * kDimK;
    int64_t const v_dim = static_cast<int64_t>(HV) * kDimV;
    TORCH_CHECK(w_q_t.size(1) == qk_dim && w_k_t.size(1) == qk_dim && w_v_t.size(1) == v_dim,
        "w_q_t and w_k_t must be [4, H*128], w_v_t must be [4, HV*128]");
    TORCH_CHECK(bias_q.is_contiguous() && bias_k.is_contiguous() && bias_v.is_contiguous(),
        "bias_q, bias_k, and bias_v must be contiguous");
    TORCH_CHECK(bias_q.numel() == qk_dim && bias_k.numel() == qk_dim && bias_v.numel() == v_dim,
        "bias_q and bias_k must hold H*128 elements, bias_v must hold HV*128 elements");
    TORCH_CHECK(a_log.is_contiguous() && a_log.numel() == H, "a_log must be contiguous with H elements");
    TORCH_CHECK(dt_bias.is_contiguous() && dt_bias.numel() == qk_dim, "dt_bias must be contiguous with H*128 elements");
    TORCH_CHECK(g.dim() == 4 && g.size(0) == 1 && g.size(1) == B && g.size(2) == HV && g.size(3) == kDimK,
        "g must have shape [1, B, HV, 128]");
    token_row_stride(g, "g");
    TORCH_CHECK(beta.dim() == 3 && beta.size(0) == 1 && beta.size(1) == B && beta.size(2) == HV && beta.stride(2) == 1,
        "beta must have shape [1, B, HV] with a contiguous head axis");
    if (apply_onorm)
    {
        TORCH_CHECK(onorm_g.dim() == 4 && onorm_g.size(0) == 1 && onorm_g.size(1) == B && onorm_g.size(2) == HV
                && onorm_g.size(3) == kDimV,
            "onorm_g must have shape [1, B, HV, 128] when apply_onorm is set");
        token_row_stride(onorm_g, "onorm_g");
        TORCH_CHECK(onorm_weight.is_contiguous() && onorm_weight.numel() == kDimV,
            "onorm_weight must be contiguous with 128 elements when apply_onorm is set");
    }

    TORCH_CHECK(state.dim() == 4 && state.size(0) >= B && state.size(1) == HV && state.size(2) == kDimV
            && state.size(3) == kDimK,
        "state must have shape [slots, HV, 128, 128] with slots >= B");
    TORCH_CHECK(state.stride(3) == 1 && state.stride(2) == kDimK && state.stride(1) == kDimV * kDimK
            && state.stride(0) >= HV * kDimV * kDimK,
        "state must be contiguous within each [HV, 128, 128] slot and have a non-overlapping slot stride");
    // The kernel moves recurrent state with 16B cp.async loads and float4 stores at element
    // offsets of `slot * stride(0) + <multiple of 4>`, so both the slot stride and the base
    // pointer have to keep those accesses 16B aligned.
    TORCH_CHECK(state.stride(0) % 4 == 0,
        "state slot stride must be a multiple of 4 floats so that per-slot float4 accesses stay 16B aligned, got ",
        state.stride(0));
    TORCH_CHECK(reinterpret_cast<uintptr_t>(state.data_ptr()) % 16 == 0,
        "state must start at a 16B-aligned address (check the storage offset of the view passed in)");
    if (ssm_state_indices.has_value())
    {
        TORCH_CHECK(ssm_state_indices->is_cuda() && ssm_state_indices->scalar_type() == at::kInt,
            "ssm_state_indices must be a CUDA int32 tensor");
        TORCH_CHECK(
            ssm_state_indices->is_contiguous() && ssm_state_indices->dim() == 1 && ssm_state_indices->size(0) == B,
            "ssm_state_indices must be contiguous with shape [B]");
    }
    TORCH_CHECK(cu_seqlens.is_cuda() && cu_seqlens.scalar_type() == at::kInt, "cu_seqlens must be a CUDA int32 tensor");
    TORCH_CHECK(cu_seqlens.is_contiguous() && cu_seqlens.dim() == 1 && cu_seqlens.size(0) == B + 1,
        "cu_seqlens must be contiguous with shape [B + 1]");

    if (update_conv_cache)
    {
        TORCH_CHECK(H == HV, "conv state update currently assumes H == HV");
        TORCH_CHECK(
            cs_q.dim() == 3 && cs_k.dim() == 3 && cs_v.dim() == 3, "update_conv_cache expects rank-3 conv-state pools");
        TORCH_CHECK(cs_q.size(0) >= state.size(0) && cs_k.size(0) >= state.size(0) && cs_v.size(0) >= state.size(0),
            "conv-state pools must cover every recurrent-state slot");
        TORCH_CHECK(cs_q.size(1) == H * kDimK && cs_k.size(1) == H * kDimK && cs_v.size(1) == HV * kDimV
                && cs_q.size(2) == kKernelWidth - 1 && cs_k.size(2) == kKernelWidth - 1
                && cs_v.size(2) == kKernelWidth - 1,
            "update_conv_cache expects [slots, dim, 3] conv-state pools");
        int64_t const minConvSlotStride = 3 * qk_dim * (kKernelWidth - 1);
        TORCH_CHECK(
            cs_q.stride(0) == cs_k.stride(0) && cs_q.stride(0) == cs_v.stride(0) && cs_q.stride(0) >= minConvSlotStride,
            "update_conv_cache expects equal, non-overlapping packed conv-state slot strides");
        TORCH_CHECK(cs_q.stride(1) == kKernelWidth - 1 && cs_k.stride(1) == kKernelWidth - 1
                && cs_v.stride(1) == kKernelWidth - 1 && cs_q.stride(2) == 1 && cs_k.stride(2) == 1
                && cs_v.stride(2) == 1,
            "update_conv_cache expects section views of [slots, 3 * dim, 3] packed conv states");
    }
    else
    {
        TORCH_CHECK(cs_q.dim() == 3 && cs_k.dim() == 3 && cs_v.dim() == 3, "batch-local conv states must be rank-3");
        TORCH_CHECK(cs_q.size(0) == B && cs_k.size(0) == B && cs_v.size(0) == B && cs_q.size(1) == H * kDimK
                && cs_k.size(1) == H * kDimK && cs_v.size(1) == HV * kDimV && cs_q.size(2) == kKernelWidth - 1
                && cs_k.size(2) == kKernelWidth - 1 && cs_v.size(2) == kKernelWidth - 1,
            "batch-local conv states must have shapes [B, H*128, 3], "
            "[B, H*128, 3], and [B, HV*128, 3]");
        TORCH_CHECK(cs_q.is_contiguous(), "cs_q must be contiguous [B, H*K, 3]");
        TORCH_CHECK(cs_k.is_contiguous(), "cs_k must be contiguous [B, H*K, 3]");
        TORCH_CHECK(cs_v.is_contiguous(), "cs_v must be contiguous [B, HV*V, 3]");
    }
}

void launch_selected_kernel(at::Tensor x_q, at::Tensor x_k, at::Tensor x_v, at::Tensor w_q_t, at::Tensor w_k_t,
    at::Tensor w_v_t, at::Tensor bias_q, at::Tensor bias_k, at::Tensor bias_v, at::Tensor cs_q, at::Tensor cs_k,
    at::Tensor cs_v, at::Tensor a_log, at::Tensor g, at::Tensor dt_bias, at::Tensor beta, at::Tensor onorm_g,
    at::Tensor onorm_weight, std::optional<at::Tensor> const& ssm_state_indices, at::Tensor cu_seqlens,
    at::Tensor state, at::Tensor out, bool apply_onorm, bool update_conv_cache, bool use_lower_bound,
    bool apply_beta_sigmoid, double lower_bound, double scale, double onorm_eps, bool enable_pdl)
{
    int const B = static_cast<int>(x_q.size(1));
    int const H = static_cast<int>(x_q.size(2));
    int const HV = static_cast<int>(x_v.size(2));
    tensorrt_llm::kernels::kdaDecode::KdaDecodeIoLayout const layout{static_cast<int>(token_row_stride(x_q, "x_q")),
        static_cast<int>(token_row_stride(x_k, "x_k")), static_cast<int>(token_row_stride(x_v, "x_v")),
        static_cast<int>(token_row_stride(g, "g")), static_cast<int>(beta.stride(1)),
        static_cast<int>(token_row_stride(onorm_g, "onorm_g"))};

    tensorrt_llm::kernels::kdaDecode::KdaDecodeParams const params{x_q.data_ptr(), x_k.data_ptr(), x_v.data_ptr(),
        w_q_t.data_ptr(), w_k_t.data_ptr(), w_v_t.data_ptr(), bias_q.data_ptr(), bias_k.data_ptr(), bias_v.data_ptr(),
        cs_q.data_ptr(), cs_k.data_ptr(), cs_v.data_ptr(), a_log.data_ptr<float>(), g.data_ptr(),
        dt_bias.data_ptr<float>(), beta.data_ptr(), onorm_g.data_ptr(), onorm_weight.data_ptr<float>(),
        ssm_state_indices.has_value() ? ssm_state_indices->data_ptr<int>() : nullptr, cu_seqlens.data_ptr<int>(),
        state.data_ptr<float>(), state.stride(0), cs_q.stride(0), out.data_ptr(), B, H, HV, apply_onorm,
        update_conv_cache, use_lower_bound, apply_beta_sigmoid, enable_pdl, static_cast<float>(lower_bound),
        static_cast<float>(scale), static_cast<float>(onorm_eps), layout};

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    tensorrt_llm::kernels::kdaDecode::invokeKdaDecode(params, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor kda_decode_fusion_forward(at::Tensor x_q, at::Tensor x_k, at::Tensor x_v, at::Tensor w_q_t, at::Tensor w_k_t,
    at::Tensor w_v_t, at::Tensor bias_q, at::Tensor bias_k, at::Tensor bias_v, at::Tensor cs_q, at::Tensor cs_k,
    at::Tensor cs_v, at::Tensor a_log, at::Tensor g, at::Tensor dt_bias, at::Tensor beta, at::Tensor onorm_g,
    at::Tensor onorm_weight, std::optional<at::Tensor> ssm_state_indices, at::Tensor cu_seqlens, at::Tensor state,
    bool apply_onorm, bool update_conv_cache, bool use_lower_bound, bool apply_beta_sigmoid, double lower_bound,
    double scale, double onorm_eps, bool enable_pdl, std::optional<at::Tensor> output)
{
    validate_kda_decode_fusion_inputs(x_q, x_k, x_v, w_q_t, w_k_t, w_v_t, bias_q, bias_k, bias_v, cs_q, cs_k, cs_v,
        a_log, g, dt_bias, beta, onorm_g, onorm_weight, ssm_state_indices, cu_seqlens, state, apply_onorm,
        update_conv_cache);
    int const B = static_cast<int>(x_q.size(1));
    int const HV = static_cast<int>(x_v.size(2));
    auto out = output.has_value() ? *output : at::empty({B, 1, HV, kDimV}, x_q.options());
    if (output.has_value())
    {
        TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kBFloat16, "out must be a CUDA bfloat16 tensor");
        TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
        TORCH_CHECK(out.dim() == 4 && out.size(0) == B && out.size(1) == 1 && out.size(2) == HV && out.size(3) == kDimV,
            "out must have shape [B, 1, HV, 128]");
    }
    launch_selected_kernel(x_q, x_k, x_v, w_q_t, w_k_t, w_v_t, bias_q, bias_k, bias_v, cs_q, cs_k, cs_v, a_log, g,
        dt_bias, beta, onorm_g, onorm_weight, ssm_state_indices, cu_seqlens, state, out, apply_onorm, update_conv_cache,
        use_lower_bound, apply_beta_sigmoid, lower_bound, scale, onorm_eps, enable_pdl);
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
        "Tensor? ssm_state_indices, Tensor cu_seqlens, Tensor(d!) state, "
        "bool apply_onorm, bool update_conv_cache, bool use_lower_bound, "
        "bool apply_beta_sigmoid, float lower_bound, float scale, "
        "float onorm_eps, bool enable_pdl=True, Tensor(e!)? output=None) -> Tensor(e!)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("kda_decode", &tensorrt_llm::torch_ext::kda_decode_fusion_forward);
}
