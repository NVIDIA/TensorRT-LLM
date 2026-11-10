/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "qwen4_exp/qwen3_8_flash_next_fp8__sm_103__unified/csrc/hcAllreduceInjectNorm.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include "tensorrt_llm/modeling_v2/common/torchLibrary.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <cstdint>
#include <optional>
#include <vector>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace
{

//! Returns `(hc_out, normed)` for the block output summed over ranks; see kernels::HcInjectNormParams.
std::vector<at::Tensor> hcAllreduceInjectNorm(at::Tensor const& partial, at::Tensor workspace, int64_t const rank,
    int64_t const nranks, at::Tensor const& hc, at::Tensor const& inject, std::optional<at::Tensor> const& shared,
    std::optional<at::Tensor> const& gate, at::Tensor const& norm_weight, double const eps, int64_t const program_warps,
    bool const trigger_completion_at_end)
{
    auto const bf16 = at::ScalarType::BFloat16;
    int64_t const rows = hc.size(0);
    TORCH_CHECK(
        partial.dim() == 2 && partial.is_contiguous() && partial.scalar_type() == bf16 && partial.size(0) == rows,
        "hc_allreduce_inject_norm: partial must be a contiguous [rows, hidden] bf16 tensor");
    int64_t const hidden = partial.size(1);
    TORCH_CHECK(hc.dim() == 2 && hc.size(1) % hidden == 0 && hc.stride(1) == 1 && hc.scalar_type() == bf16,
        "hc_allreduce_inject_norm: hc must be [rows, hc_count * hidden] bf16 with unit column stride");
    int64_t const hc_count = hc.size(1) / hidden;
    TORCH_CHECK(hc.stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(hc.data_ptr()) % 16 == 0,
        "hc_allreduce_inject_norm: hc rows must be 16-byte aligned");
    TORCH_CHECK(inject.dim() == 2 && inject.size(0) == rows && inject.size(1) >= hc_count && inject.stride(1) == 1
            && inject.scalar_type() == bf16,
        "hc_allreduce_inject_norm: inject must be [rows, >= hc_count] bf16 with unit column stride");
    TORCH_CHECK(
        norm_weight.numel() == hc_count * hidden && norm_weight.is_contiguous() && norm_weight.scalar_type() == bf16,
        "hc_allreduce_inject_norm: norm_weight must be [hc_count * hidden] bf16");
    TORCH_CHECK(shared.has_value() == gate.has_value(), "hc_allreduce_inject_norm: shared and gate come together");
    if (shared.has_value())
    {
        TORCH_CHECK(shared->dim() == 2 && shared->size(0) == rows && shared->size(1) == hidden && shared->stride(1) == 1
                && shared->stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(shared->data_ptr()) % 16 == 0
                && shared->scalar_type() == bf16,
            "hc_allreduce_inject_norm: shared must be [rows, hidden] bf16 with 16-byte aligned rows");
        TORCH_CHECK(gate->dim() == 2 && gate->size(0) == rows && gate->scalar_type() == bf16,
            "hc_allreduce_inject_norm: gate must be [rows, 1] bf16");
    }
    auto hc_out = at::empty({rows, hc_count * hidden}, hc.options());
    auto normed = at::empty({rows, hc_count * hidden}, hc.options());

    kernels::HcInjectNormParams params;
    params.nranks = static_cast<int>(nranks);
    params.rank = static_cast<int>(rank);
    params.rows = static_cast<int>(rows);
    params.hidden_dim = static_cast<int>(hidden);
    params.hc_count = static_cast<int>(hc_count);
    params.program_warps = static_cast<int>(program_warps);
    params.workspace = reinterpret_cast<void**>(workspace.mutable_data_ptr());
    params.partial = partial.data_ptr();
    params.hc = hc.data_ptr();
    params.hc_stride = hc.stride(0);
    params.inject = inject.data_ptr();
    params.inject_stride = inject.stride(0);
    params.shared = shared.has_value() ? shared->data_ptr() : nullptr;
    params.shared_stride = shared.has_value() ? shared->stride(0) : 0;
    params.gate = gate.has_value() ? gate->data_ptr() : nullptr;
    params.gate_stride = gate.has_value() ? gate->stride(0) : 0;
    params.norm_weight = norm_weight.data_ptr();
    params.hc_out = hc_out.mutable_data_ptr();
    params.hc_out_stride = hc_out.stride(0);
    params.normed = normed.mutable_data_ptr();
    params.normed_stride = normed.stride(0);
    params.eps = static_cast<float>(eps);
    params.trigger_completion_at_end = trigger_completion_at_end;
    params.stream = at::cuda::getCurrentCUDAStream(hc.get_device());
    kernels::hc_allreduce_inject_norm_op(params);
    return {hc_out, normed};
}

} // namespace

TRTLLM_MV2_NAMESPACE_END

TRTLLM_MV2_LIBRARY_FRAGMENT(m)
{
    m.def(
        "hc_allreduce_inject_norm("
        "Tensor partial,"
        "Tensor workspace,"
        "int rank,"
        "int nranks,"
        "Tensor hc,"
        "Tensor inject,"
        "Tensor? shared,"
        "Tensor? gate,"
        "Tensor norm_weight,"
        "float eps,"
        "int program_warps,"
        "bool trigger_completion_at_end) -> Tensor[]");
}

TRTLLM_MV2_LIBRARY_IMPL(CUDA, m)
{
    m.impl("hc_allreduce_inject_norm", &TRTLLM_MV2_NS::hcAllreduceInjectNorm);
}
