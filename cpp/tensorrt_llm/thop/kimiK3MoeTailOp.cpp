/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/kernels/communicationKernels/kimiK3MoeTailFusion.h"

#include <ATen/Functions.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <torch/library.h>

#include <optional>
#include <vector>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

//! Fused K3 MoE-tail head: oneshot AR(latent)+RMSNorm concurrent with
//! RS(shared). Returns {z [T, L] reduced+normed latent, s_shard [T, H/nranks]
//! this rank's fully-reduced shared segment}.
std::vector<at::Tensor> kimi_k3_moe_tail_ar_rs_norm(at::Tensor const& latent, at::Tensor const& shared,
    at::Tensor const& norm_weight, at::Tensor workspace, int64_t const rank, int64_t const nranks, double const rms_eps,
    bool const trigger_completion_at_end)
{
    TORCH_CHECK(latent.dim() == 2 && shared.dim() == 2, "kimi_k3_moe_tail: latent/shared must be 2-D");
    int64_t const T = latent.size(0);
    int64_t const L = latent.size(1);
    int64_t const H = shared.size(1);
    TORCH_CHECK(nranks == 2 || nranks == 4 || nranks == 8, "kimi_k3_moe_tail: nranks must be 2/4/8");
    TORCH_CHECK(shared.size(0) == T, "kimi_k3_moe_tail: token counts must match");
    TORCH_CHECK(T >= 1 && T <= 64, "kimi_k3_moe_tail: token count must be in [1, 64]");
    TORCH_CHECK(L % 8 == 0 && L / 8 >= 128 && L / 8 <= 512, "kimi_k3_moe_tail: latent/8 must be in [128, 512]");
    TORCH_CHECK(H % (nranks * 8) == 0, "kimi_k3_moe_tail: hidden must split into 16B-aligned segments");
    TORCH_CHECK(latent.scalar_type() == at::kBFloat16 && shared.scalar_type() == at::kBFloat16
            && norm_weight.scalar_type() == at::kBFloat16,
        "kimi_k3_moe_tail: bf16 only");
    TORCH_CHECK(latent.is_contiguous() && shared.is_contiguous() && norm_weight.is_contiguous(),
        "kimi_k3_moe_tail: inputs must be contiguous");
    TORCH_CHECK(norm_weight.numel() == L, "kimi_k3_moe_tail: norm weight must have L elements");

    auto params = tensorrt_llm::kernels::kimi_k3_moe::KimiK3MoeTailParams();
    params.nranks = static_cast<int>(nranks);
    params.rank = static_cast<int>(rank);
    params.latent_dim = static_cast<int>(L);
    params.hidden_dim = static_cast<int>(H);
    params.token_num = static_cast<int>(T);
    params.workspace = reinterpret_cast<void**>(workspace.mutable_data_ptr());
    params.latent_in = latent.data_ptr();
    params.shared_in = shared.data_ptr();
    params.norm_weight = norm_weight.data_ptr();
    params.rms_eps = static_cast<float>(rms_eps);
    params.stream = at::cuda::getCurrentCUDAStream(latent.get_device());
    params.trigger_completion_at_end = trigger_completion_at_end;

    at::Tensor z = at::empty_like(latent);
    at::Tensor s_shard = at::empty({T, H / nranks}, shared.options());
    params.z_out = z.mutable_data_ptr();
    params.sshard_out = s_shard.mutable_data_ptr();

    tensorrt_llm::kernels::kimi_k3_moe::kimi_k3_moe_tail_op(params);

    return {z, s_shard};
}

//! Fused [oneshot AllGather + add] for the striped up-projection: gathers
//! each rank's output-column stripe [T, S] into the full hidden row. Optional
//! operands: shard_add [T, S] is folded into the pushed stripe (fp32 + single
//! rounding); shared [T, H] (strided rows OK) is added on the poll side.
at::Tensor kimi_k3_stripe_allgather_add(at::Tensor const& shard, std::optional<at::Tensor> const& shared,
    std::optional<at::Tensor> const& shard_add, at::Tensor workspace, int64_t const rank, int64_t const nranks,
    int64_t const hidden, bool const trigger_completion_at_end)
{
    TORCH_CHECK(shard.dim() == 2, "kimi_k3_stripe_allgather_add: shard must be 2-D");
    int64_t const T = shard.size(0);
    int64_t const S = shard.size(1);
    int64_t const H = shared.has_value() ? shared->size(1) : hidden;
    TORCH_CHECK(nranks == 2 || nranks == 4 || nranks == 8, "kimi_k3_stripe_allgather_add: nranks must be 2/4/8");
    TORCH_CHECK(H == S * nranks, "kimi_k3_stripe_allgather_add: hidden must equal stripe * nranks");
    TORCH_CHECK(T >= 1 && T <= 64, "kimi_k3_stripe_allgather_add: token count must be in [1, 64]");
    TORCH_CHECK(S % 8 == 0, "kimi_k3_stripe_allgather_add: stripe must be a multiple of 8");
    TORCH_CHECK(shard.scalar_type() == at::kBFloat16, "kimi_k3_stripe_allgather_add: bf16 only");
    TORCH_CHECK(shard.is_contiguous(), "kimi_k3_stripe_allgather_add: shard must be contiguous");
    if (shared.has_value())
    {
        TORCH_CHECK(shared->dim() == 2 && shared->size(0) == T && shared->scalar_type() == at::kBFloat16,
            "kimi_k3_stripe_allgather_add: shared must be bf16 [T, H]");
        TORCH_CHECK(shared->stride(1) == 1 && shared->stride(0) % 8 == 0,
            "kimi_k3_stripe_allgather_add: shared rows must be dense and 16B-aligned");
        TORCH_CHECK(reinterpret_cast<uintptr_t>(shared->data_ptr()) % 16 == 0,
            "kimi_k3_stripe_allgather_add: shared base must be 16B-aligned");
    }
    if (shard_add.has_value())
    {
        TORCH_CHECK(shard_add->sizes() == shard.sizes() && shard_add->scalar_type() == at::kBFloat16
                && shard_add->is_contiguous(),
            "kimi_k3_stripe_allgather_add: shard_add must be a contiguous bf16 [T, S] tensor");
    }

    auto params = tensorrt_llm::kernels::kimi_k3_moe::KimiK3StripeAgParams();
    params.nranks = static_cast<int>(nranks);
    params.rank = static_cast<int>(rank);
    params.size = static_cast<int>(T * S);
    params.stripe_dim = static_cast<int>(S);
    params.hidden_dim = static_cast<int>(H);
    params.shared_ld = shared.has_value() ? static_cast<int>(shared->stride(0)) : 0;
    params.workspace = reinterpret_cast<void**>(workspace.mutable_data_ptr());
    params.shard_in = shard.data_ptr();
    params.shard_add = shard_add.has_value() ? shard_add->data_ptr() : nullptr;
    params.shared_in = shared.has_value() ? shared->data_ptr() : nullptr;
    params.stream = at::cuda::getCurrentCUDAStream(shard.get_device());
    params.trigger_completion_at_end = trigger_completion_at_end;

    at::Tensor out = at::empty({T, H}, shard.options());
    params.out = out.mutable_data_ptr();

    tensorrt_llm::kernels::kimi_k3_moe::kimi_k3_stripe_ag_add_op(params);

    return out;
}

} // namespace

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "kimi_k3_moe_tail_ar_rs_norm(Tensor latent, Tensor shared, Tensor norm_weight, Tensor workspace, "
        "int rank, int nranks, float rms_eps, bool trigger_completion_at_end) -> Tensor[]");
    m.def(
        "kimi_k3_stripe_allgather_add(Tensor shard, Tensor? shared, Tensor? shard_add, Tensor workspace, "
        "int rank, int nranks, int hidden, bool trigger_completion_at_end) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("kimi_k3_moe_tail_ar_rs_norm", &tensorrt_llm::torch_ext::kimi_k3_moe_tail_ar_rs_norm);
    m.impl("kimi_k3_stripe_allgather_add", &tensorrt_llm::torch_ext::kimi_k3_stripe_allgather_add);
}
