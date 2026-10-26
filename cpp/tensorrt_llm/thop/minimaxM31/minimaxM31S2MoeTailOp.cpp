/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/kernels/minimaxM31/s2MoeTailSm107.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <tuple>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> m31S2MoeTail(torch::Tensor const& fc2,
    torch::Tensor const& scales, torch::Tensor const& perm, torch::Tensor const& shared, torch::Tensor const& residual,
    torch::Tensor const& norm_weight, torch::Tensor const& workspace, int64_t rank, int64_t nranks, double eps,
    bool trigger_completion_at_end)
{
    TORCH_CHECK(fc2.is_cuda() && fc2.scalar_type() == at::kBFloat16 && fc2.dim() == 2 && fc2.stride(1) == 1);
    int64_t const hidden = fc2.size(1);
    int64_t const tokens = shared.size(0);
    TORCH_CHECK(shared.is_contiguous() && shared.scalar_type() == at::kBFloat16 && shared.size(1) == hidden);
    TORCH_CHECK(
        residual.is_contiguous() && residual.scalar_type() == at::kBFloat16 && residual.sizes() == shared.sizes());
    TORCH_CHECK(
        norm_weight.is_contiguous() && norm_weight.scalar_type() == at::kBFloat16 && norm_weight.numel() == hidden);
    TORCH_CHECK(perm.is_contiguous() && perm.scalar_type() == at::kInt && perm.dim() == 2 && perm.size(0) == tokens);
    TORCH_CHECK(scales.is_contiguous() && scales.sizes() == perm.sizes()
        && (scales.scalar_type() == at::kBFloat16 || scales.scalar_type() == at::kFloat));
    TORCH_CHECK(tokens >= 1 && tokens <= tensorrt_llm::kernels::minimaxM31Sm107::s2tail::kMaxTokens,
        "one-shot Lamport rows only (<= 256)");
    TORCH_CHECK(hidden % 256 == 0 && (fc2.stride(0) % 8) == 0);
    TORCH_CHECK(nranks == 2, "TP2 only");
    at::cuda::CUDAGuard const guard(fc2.device());
    auto const opts = shared.options();
    torch::Tensor norm_out = torch::empty_like(shared);
    torch::Tensor residual_out = torch::empty_like(residual);
    torch::Tensor q = torch::empty({tokens, hidden}, opts.dtype(at::ScalarType::Float8_e4m3fn));
    int64_t const sf_size = (tokens + 127) / 128 * 128 * ((hidden / 32 + 3) / 4 * 4);
    torch::Tensor sf = torch::empty({sf_size}, opts.dtype(at::ScalarType::Byte));
    tensorrt_llm::kernels::minimaxM31Sm107::s2tail::Params p{};
    p.nranks = static_cast<int>(nranks);
    p.rank = static_cast<int>(rank);
    p.size = static_cast<int>(tokens * hidden);
    p.hidden_dim = static_cast<int>(hidden);
    p.top_k = static_cast<int>(perm.size(1));
    p.ld_fc2 = static_cast<int>(fc2.stride(0));
    p.workspace = reinterpret_cast<void**>(workspace.data_ptr());
    p.fc2 = reinterpret_cast<__nv_bfloat16 const*>(fc2.data_ptr());
    p.scales = scales.data_ptr();
    p.perm = perm.data_ptr<int>();
    p.shared = reinterpret_cast<__nv_bfloat16 const*>(shared.data_ptr());
    p.residual_in = reinterpret_cast<__nv_bfloat16 const*>(residual.data_ptr());
    p.rms_gamma = reinterpret_cast<__nv_bfloat16 const*>(norm_weight.data_ptr());
    p.rms_eps = static_cast<float>(eps);
    p.residual_out = reinterpret_cast<__nv_bfloat16*>(residual_out.data_ptr());
    p.norm_out = reinterpret_cast<__nv_bfloat16*>(norm_out.data_ptr());
    p.quant_out = reinterpret_cast<uint64_t*>(q.data_ptr());
    p.scale_out = reinterpret_cast<uint32_t*>(sf.data_ptr());
    auto stream = at::cuda::getCurrentCUDAStream(fc2.get_device());
    tensorrt_llm::kernels::minimaxM31Sm107::s2tail::invokeS2MoeTail(
        p, scales.scalar_type() == at::kBFloat16, trigger_completion_at_end, stream);
    return {norm_out, residual_out, q, sf};
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "minimax_m31_s2_moe_tail(Tensor fc2, Tensor scales, Tensor perm, Tensor shared, Tensor residual, "
        "Tensor norm_weight, Tensor workspace, int rank, int nranks, float eps, bool trigger_completion_at_end=True) "
        "-> (Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("minimax_m31_s2_moe_tail", &tensorrt_llm::torch_ext::m31S2MoeTail);
}
