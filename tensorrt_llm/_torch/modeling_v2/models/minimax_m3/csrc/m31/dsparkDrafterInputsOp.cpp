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

#include "minimax_m3/csrc/m31/dsparkDrafterInputsSm107.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include "tensorrt_llm/modeling_v2/common/torchLibrary.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cstdint>
#include <limits>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

bool isIndex(torch::Tensor const& t)
{
    return t.scalar_type() == torch::kInt32 || t.scalar_type() == torch::kInt64;
}

//! The shared input / output checks of both ops; returns n.
int64_t checkInputs(torch::Tensor const& accepted, torch::Tensor const& nacc, torch::Tensor const& slots,
    torch::Tensor const& ctxLen, torch::Tensor const& table, torch::Tensor const& maskRow, torch::Tensor const& bonus,
    torch::Tensor const& qpos, torch::Tensor const& cpos, torch::Tensor const& noise, char const* op)
{
    int64_t const n = nacc.size(0);
    TORCH_CHECK(accepted.is_cuda() && isIndex(accepted) && accepted.dim() == 2 && accepted.size(0) == n
            && accepted.stride(1) == 1,
        op, ": accepted must be int32 / int64 [n, *] with unit column stride");
    TORCH_CHECK(nacc.is_cuda() && isIndex(nacc) && nacc.dim() == 1 && nacc.is_contiguous(), op,
        ": num_accepted must be a contiguous int32 / int64 [n] CUDA tensor");
    TORCH_CHECK(slots.scalar_type() == torch::kInt64 && slots.dim() == 1 && slots.size(0) == n && slots.is_contiguous(),
        op, ": slots must be a contiguous int64 [n] tensor");
    TORCH_CHECK(ctxLen.scalar_type() == torch::kInt64 && ctxLen.dim() == 1 && ctxLen.is_contiguous(), op,
        ": ctx_len must be a contiguous int64 1-D tensor");
    TORCH_CHECK(table.is_cuda() && table.dim() == 2 && table.is_contiguous(), op, ": table must be contiguous 2-D");
    int64_t const hidden = table.size(1);
    TORCH_CHECK(maskRow.scalar_type() == table.scalar_type() && maskRow.dim() == 1 && maskRow.size(0) == hidden
            && maskRow.is_contiguous(),
        op, ": mask_row must be a contiguous [hidden] tensor of the table's dtype");
    TORCH_CHECK(bonus.scalar_type() == torch::kInt64 && bonus.is_contiguous() && bonus.numel() >= n, op,
        ": bonus must be a contiguous int64 [n] tensor");
    TORCH_CHECK(qpos.scalar_type() == torch::kInt64 && qpos.is_contiguous() && qpos.dim() == 2 && qpos.size(0) == n, op,
        ": query positions must be a contiguous int64 [n, block] tensor");
    TORCH_CHECK(cpos.scalar_type() == torch::kInt64 && cpos.is_contiguous() && cpos.dim() == 2 && cpos.size(0) == n, op,
        ": ctx positions must be a contiguous int64 [n, tokens] tensor");
    TORCH_CHECK(noise.scalar_type() == table.scalar_type() && noise.is_contiguous() && noise.dim() == 3
            && noise.size(0) == n && noise.size(1) == qpos.size(1) && noise.size(2) == hidden,
        op, ": noise must be a contiguous [n, block, hidden] tensor of the table's dtype");
    TORCH_CHECK(qpos.size(1) >= 1 && cpos.size(1) >= 1 && n <= std::numeric_limits<int32_t>::max()
            && qpos.size(1) <= std::numeric_limits<int32_t>::max()
            && cpos.size(1) <= std::numeric_limits<int32_t>::max(),
        op, ": invalid sizes");
    return n;
}

} // namespace

//! ops/dspark/dspark_drafter_inputs.drafter_inputs: bonus / qpos / cpos / noise written in place.
void minimaxM31DsparkDrafterInputs(torch::Tensor const& accepted, torch::Tensor const& nacc, torch::Tensor const& slots,
    torch::Tensor const& ctxLen, int64_t maxCtx, torch::Tensor const& table, torch::Tensor const& maskRow,
    torch::Tensor& bonus, torch::Tensor& qpos, torch::Tensor& cpos, torch::Tensor& noise)
{
    char const* op = "minimax_m31_dspark_drafter_inputs";
    int64_t const n = checkInputs(accepted, nacc, slots, ctxLen, table, maskRow, bonus, qpos, cpos, noise, op);
    if (n == 0)
        return;
    at::cuda::CUDAGuard const guard(table.device());
    TRTLLM_MV2_NS::kernels::minimaxM31::invokeDsparkDrafterInputs(accepted.data_ptr(),
        accepted.scalar_type() == torch::kInt64, accepted.stride(0), nacc.data_ptr(),
        nacc.scalar_type() == torch::kInt64, slots.data_ptr<int64_t>(), ctxLen.data_ptr<int64_t>(), maxCtx,
        table.data_ptr(), maskRow.data_ptr(), bonus.data_ptr<int64_t>(), qpos.data_ptr<int64_t>(),
        cpos.data_ptr<int64_t>(), noise.data_ptr(), table.size(1) * table.element_size(), table.size(0),
        static_cast<int32_t>(n), static_cast<int32_t>(qpos.size(1)), static_cast<int32_t>(cpos.size(1)),
        at::cuda::getCurrentCUDAStream(table.get_device()));
}

//! drafter_inputs_and_refresh: + tables[:n] / counts[:n] from src = offsets[pool_idx, :n, 0] (int32 / int64, any
//! strides), in the same launch.
void minimaxM31DsparkDrafterInputsRefresh(torch::Tensor const& accepted, torch::Tensor const& nacc,
    torch::Tensor const& slots, torch::Tensor const& ctxLen, int64_t maxCtx, torch::Tensor const& table,
    torch::Tensor const& maskRow, torch::Tensor& bonus, torch::Tensor& qpos, torch::Tensor& cpos, torch::Tensor& noise,
    torch::Tensor const& src, int64_t divisor, torch::Tensor& tables, torch::Tensor& counts)
{
    char const* op = "minimax_m31_dspark_drafter_inputs_refresh";
    int64_t const n = checkInputs(accepted, nacc, slots, ctxLen, table, maskRow, bonus, qpos, cpos, noise, op);
    TORCH_CHECK(n >= 1, op, ": needs at least one request");
    TORCH_CHECK(src.is_cuda() && isIndex(src) && src.dim() == 2 && src.size(0) == n, op,
        ": offsets rows must be an int32 / int64 [n, width] CUDA tensor");
    TORCH_CHECK(tables.scalar_type() == torch::kInt32 && tables.dim() == 2 && tables.stride(1) == 1
            && tables.size(0) >= n && tables.size(1) == src.size(1),
        op, ": tables must be int32 [>= n, width] with unit column stride");
    TORCH_CHECK(
        counts.scalar_type() == torch::kInt64 && counts.dim() == 1 && counts.is_contiguous() && counts.size(0) >= n, op,
        ": counts must be a contiguous int64 [>= n] tensor");
    TORCH_CHECK(divisor >= 1 && src.size(1) <= std::numeric_limits<int32_t>::max(), op, ": invalid divisor / width");
    at::cuda::CUDAGuard const guard(table.device());
    TRTLLM_MV2_NS::kernels::minimaxM31::invokeDsparkDrafterInputsRefresh(accepted.data_ptr(),
        accepted.scalar_type() == torch::kInt64, accepted.stride(0), nacc.data_ptr(),
        nacc.scalar_type() == torch::kInt64, slots.data_ptr<int64_t>(), ctxLen.data_ptr<int64_t>(), maxCtx,
        table.data_ptr(), maskRow.data_ptr(), bonus.data_ptr<int64_t>(), qpos.data_ptr<int64_t>(),
        cpos.data_ptr<int64_t>(), noise.data_ptr(), table.size(1) * table.element_size(), table.size(0),
        static_cast<int32_t>(n), static_cast<int32_t>(qpos.size(1)), static_cast<int32_t>(cpos.size(1)), src.data_ptr(),
        src.scalar_type() == torch::kInt64, src.stride(0), src.stride(1), tables.data_ptr<int32_t>(), tables.stride(0),
        counts.data_ptr<int64_t>(), static_cast<int32_t>(src.size(1)), divisor,
        at::cuda::getCurrentCUDAStream(table.get_device()));
}

} // namespace torch_ext

TRTLLM_MV2_NAMESPACE_END

TRTLLM_MV2_LIBRARY_FRAGMENT(m)
{
    m.def(
        "minimax_m31_dspark_drafter_inputs(Tensor accepted, Tensor num_accepted, Tensor slots, Tensor ctx_len, "
        "int max_ctx, Tensor table, Tensor mask_row, Tensor(a!) bonus, Tensor(b!) query_positions, "
        "Tensor(c!) ctx_positions, Tensor(d!) noise) -> ()");
    m.def(
        "minimax_m31_dspark_drafter_inputs_refresh(Tensor accepted, Tensor num_accepted, Tensor slots, "
        "Tensor ctx_len, int max_ctx, Tensor table, Tensor mask_row, Tensor(a!) bonus, Tensor(b!) query_positions, "
        "Tensor(c!) ctx_positions, Tensor(d!) noise, Tensor offsets, int divisor, Tensor(e!) tables, "
        "Tensor(f!) counts) -> ()");
}

TRTLLM_MV2_LIBRARY_IMPL(CUDA, m)
{
    m.impl("minimax_m31_dspark_drafter_inputs", &TRTLLM_MV2_NS::torch_ext::minimaxM31DsparkDrafterInputs);
    m.impl(
        "minimax_m31_dspark_drafter_inputs_refresh", &TRTLLM_MV2_NS::torch_ext::minimaxM31DsparkDrafterInputsRefresh);
}
