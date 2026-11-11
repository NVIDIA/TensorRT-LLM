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
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>
#include <initializer_list>
#include <optional>
#include <vector>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace
{

//! Returns `(hcOut, normed)` for the block output summed over ranks; see kernels::HcInjectNormParams.
std::vector<at::Tensor> hcAllreduceInjectNorm(at::Tensor const& partial, at::Tensor workspace, int64_t const rank,
    int64_t const nranks, at::Tensor const& hc, at::Tensor const& inject, std::optional<at::Tensor> const& shared,
    std::optional<at::Tensor> const& gate, at::Tensor const& normWeight, double const eps, int64_t const programWarps,
    bool const triggerCompletionAtEnd)
{
    auto const bf16 = at::ScalarType::BFloat16;
    TORCH_CHECK(hc.is_cuda(), "hc_allreduce_inject_norm: hc must be a CUDA tensor");
    c10::cuda::CUDAGuard const guard(hc.device());
    for (at::Tensor const* t : std::initializer_list<at::Tensor const*>{&partial, &workspace, &inject, &normWeight})
    {
        TORCH_CHECK(t->device() == hc.device(), "hc_allreduce_inject_norm: every tensor must be on hc's device");
    }
    for (std::optional<at::Tensor> const* t : std::initializer_list<std::optional<at::Tensor> const*>{&shared, &gate})
    {
        TORCH_CHECK(!t->has_value() || (*t)->device() == hc.device(),
            "hc_allreduce_inject_norm: every tensor must be on hc's device");
    }
    TORCH_CHECK(workspace.scalar_type() == at::ScalarType::Long && workspace.is_contiguous(),
        "hc_allreduce_inject_norm: workspace must be the stock all-reduce fusion workspace (contiguous int64)");
    TORCH_CHECK(0 <= rank && rank < nranks, "hc_allreduce_inject_norm: rank must be in [0, nranks)");
    int64_t const rows = hc.size(0);
    TORCH_CHECK(
        partial.dim() == 2 && partial.is_contiguous() && partial.scalar_type() == bf16 && partial.size(0) == rows,
        "hc_allreduce_inject_norm: partial must be a contiguous [rows, hidden] bf16 tensor");
    int64_t const hidden = partial.size(1);
    TORCH_CHECK(hc.dim() == 2 && hc.size(1) % hidden == 0 && hc.stride(1) == 1 && hc.scalar_type() == bf16,
        "hc_allreduce_inject_norm: hc must be [rows, hc_count * hidden] bf16 with unit column stride");
    int64_t const hcCount = hc.size(1) / hidden;
    TORCH_CHECK(hc.stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(hc.data_ptr()) % 16 == 0,
        "hc_allreduce_inject_norm: hc rows must be 16-byte aligned");
    TORCH_CHECK(inject.dim() == 2 && inject.size(0) == rows && inject.size(1) >= hcCount && inject.stride(1) == 1
            && inject.scalar_type() == bf16,
        "hc_allreduce_inject_norm: inject must be [rows, >= hc_count] bf16 with unit column stride");
    TORCH_CHECK(
        normWeight.numel() == hcCount * hidden && normWeight.is_contiguous() && normWeight.scalar_type() == bf16,
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
    auto hcOut = at::empty({rows, hcCount * hidden}, hc.options());
    auto normed = at::empty({rows, hcCount * hidden}, hc.options());

    kernels::HcInjectNormParams params;
    params.nranks = static_cast<int>(nranks);
    params.rank = static_cast<int>(rank);
    params.rows = static_cast<int>(rows);
    params.hiddenDim = static_cast<int>(hidden);
    params.hcCount = static_cast<int>(hcCount);
    params.programWarps = static_cast<int>(programWarps);
    params.workspace = reinterpret_cast<void**>(workspace.mutable_data_ptr());
    params.partial = partial.data_ptr();
    params.hc = hc.data_ptr();
    params.hcStride = hc.stride(0);
    params.inject = inject.data_ptr();
    params.injectStride = inject.stride(0);
    params.shared = shared.has_value() ? shared->data_ptr() : nullptr;
    params.sharedStride = shared.has_value() ? shared->stride(0) : 0;
    params.gate = gate.has_value() ? gate->data_ptr() : nullptr;
    params.gateStride = gate.has_value() ? gate->stride(0) : 0;
    params.normWeight = normWeight.data_ptr();
    params.hcOut = hcOut.mutable_data_ptr();
    params.hcOutStride = hcOut.stride(0);
    params.normed = normed.mutable_data_ptr();
    params.normedStride = normed.stride(0);
    params.eps = static_cast<float>(eps);
    params.triggerCompletionAtEnd = triggerCompletionAtEnd;
    params.stream = at::cuda::getCurrentCUDAStream(hc.get_device());
    kernels::launchHcAllreduceInjectNorm(params);
    return {hcOut, normed};
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
