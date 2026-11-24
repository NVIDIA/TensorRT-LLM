/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include "minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2/csrc/arPrepushedSm107.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include "tensorrt_llm/modeling_v2/common/torchLibrary.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <initializer_list>
#include <limits>
#include <optional>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace torch_ext
{

//! OPROJ_PUSH_GEMM attention boundary (TP2 decode): the one-shot Lamport all-reduce + residual + RMSNorm over the rows
//! the o_proj's Lamport-push GEMM epilogue already stored into the AllReduce workspace's data slots (no push loop),
//! + optionally the MXFP8 of norm_out (mx_out e4m3 [rows, hidden], mx_sf swizzled UE8M0 of pad128(rows) rows).
//! allreduce_in is shape-only (its bytes are not read). Bitwise to the stock one-shot allreduce_fusion
//! RESIDUAL_RMS_NORM (+ mxfp8_quantize).
void m31ArPrepushed(int64_t rank, torch::Tensor const& workspace, torch::Tensor const& allreduce_in,
    torch::Tensor const& residual_in, torch::Tensor& residual_out, torch::Tensor& norm_out, torch::Tensor const& gamma,
    double eps, bool trigger_completion_at_end, std::optional<torch::Tensor> mx_out, std::optional<torch::Tensor> mx_sf)
{
    TORCH_CHECK(allreduce_in.is_cuda() && allreduce_in.scalar_type() == at::kBFloat16 && allreduce_in.is_contiguous());
    TORCH_CHECK(allreduce_in.dim() >= 1 && allreduce_in.size(-1) > 0 && allreduce_in.size(-1) % 8 == 0
            && allreduce_in.numel() <= std::numeric_limits<int32_t>::max(),
        "minimax_m31_ar_prepushed: allreduce_in's hidden size must be a positive multiple of 8");
    TORCH_CHECK(rank >= 0 && rank < 2, "minimax_m31_ar_prepushed: rank ", rank, " outside [0, 2)");
    for (torch::Tensor const* t :
        std::initializer_list<torch::Tensor const*>{&workspace, &residual_in, &residual_out, &norm_out, &gamma})
    {
        TORCH_CHECK(t->is_cuda() && t->device() == allreduce_in.device(),
            "minimax_m31_ar_prepushed: every operand must be a CUDA tensor on allreduce_in's device");
    }
    // residual_in / residual_out / norm_out are read and written as packed BF16 vectors of allreduce_in's size;
    // gamma as one packed BF16 row of the hidden size.
    for (torch::Tensor const* t : std::initializer_list<torch::Tensor const*>{&residual_in, &residual_out, &norm_out})
    {
        TORCH_CHECK(t->scalar_type() == at::kBFloat16 && t->is_contiguous() && t->numel() == allreduce_in.numel()
                && reinterpret_cast<uintptr_t>(t->data_ptr()) % 16 == 0,
            "minimax_m31_ar_prepushed: residual_in, residual_out and norm_out must be contiguous 16-byte aligned "
            "bfloat16 tensors of allreduce_in's size");
    }
    TORCH_CHECK(gamma.scalar_type() == at::kBFloat16 && gamma.is_contiguous() && gamma.numel() == allreduce_in.size(-1)
            && reinterpret_cast<uintptr_t>(gamma.data_ptr()) % 16 == 0,
        "minimax_m31_ar_prepushed: gamma must be a contiguous 16-byte aligned bfloat16 [hidden] tensor");
    for (std::optional<torch::Tensor> const* t :
        std::initializer_list<std::optional<torch::Tensor> const*>{&mx_out, &mx_sf})
    {
        TORCH_CHECK(!t->has_value() || ((*t)->is_cuda() && (*t)->device() == allreduce_in.device()),
            "minimax_m31_ar_prepushed: mx_out and mx_sf must be CUDA tensors on allreduce_in's device");
    }
    TRTLLM_MV2_NS::kernels::minimaxM31Sm107::arPrepushed::Params p{};
    p.rank = static_cast<int>(rank);
    p.size = static_cast<int>(allreduce_in.numel());
    p.hidden_dim = static_cast<int>(allreduce_in.size(-1));
    p.workspace = reinterpret_cast<void**>(workspace.data_ptr());
    p.allreduce_in = allreduce_in.data_ptr();
    p.residual_in = residual_in.data_ptr();
    p.residual_out = residual_out.data_ptr();
    p.norm_out = norm_out.data_ptr();
    p.rms_gamma = gamma.data_ptr();
    p.rms_eps = static_cast<float>(eps);
    p.mx_out = nullptr;
    p.mx_sf = nullptr;
    if (mx_out.has_value())
    {
        TORCH_CHECK(mx_sf.has_value() && mx_out->is_contiguous() && mx_sf->is_contiguous());
        int64_t const rows = p.size / p.hidden_dim;
        TORCH_CHECK(mx_out->numel() * mx_out->element_size() == rows * p.hidden_dim, "mx_out must be [rows, H] e4m3");
        TORCH_CHECK(mx_sf->numel() * mx_sf->element_size() == (rows + 127) / 128 * 128 * (p.hidden_dim / 32),
            "mx_sf must be the swizzled [pad128(rows), H / 32] UE8M0 scales");
        p.mx_out = mx_out->data_ptr();
        p.mx_sf = mx_sf->data_ptr();
    }
    TORCH_CHECK(p.size % p.hidden_dim == 0 && p.hidden_dim % 8 == 0);
    at::cuda::CUDAGuard const guard(allreduce_in.device());
    C10_CUDA_CHECK(TRTLLM_MV2_NS::kernels::minimaxM31Sm107::arPrepushed::invokeArPrepushed(
        p, trigger_completion_at_end, at::cuda::getCurrentCUDAStream(allreduce_in.get_device())));
}

} // namespace torch_ext

TRTLLM_MV2_NAMESPACE_END

TRTLLM_MV2_LIBRARY_FRAGMENT(m)
{
    m.def(
        "minimax_m31_ar_prepushed(int rank, Tensor workspace, Tensor allreduce_in, Tensor residual_in, "
        "Tensor(a!) residual_out, Tensor(b!) norm_out, Tensor gamma, float eps, bool trigger_completion_at_end, "
        "Tensor(c!)? mx_out, Tensor(d!)? mx_sf) -> ()");
}

TRTLLM_MV2_LIBRARY_IMPL(CUDA, m)
{
    m.impl("minimax_m31_ar_prepushed", &TRTLLM_MV2_NS::torch_ext::m31ArPrepushed);
}
