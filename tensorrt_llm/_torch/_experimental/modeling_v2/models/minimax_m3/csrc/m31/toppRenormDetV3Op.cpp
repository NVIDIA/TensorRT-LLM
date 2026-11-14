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

#include "minimax_m3/csrc/m31/toppRenormDetV3Sm103.h"
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

static void checkV3Operands(torch::Tensor const& probs, torch::Tensor const& topP, torch::Tensor const& workspace,
    torch::Tensor const& out, int64_t ctasPerRow)
{
    TORCH_CHECK(probs.is_cuda() && probs.scalar_type() == torch::kFloat32 && probs.dim() == 2 && probs.is_contiguous(),
        "minimax_m31_topp_renorm_det_v3: probs must be a contiguous 2-D float32 CUDA tensor");
    int64_t const rows = probs.size(0);
    int64_t const d = probs.size(1);
    TORCH_CHECK(d % 4 == 0 && reinterpret_cast<uintptr_t>(probs.data_ptr()) % 16 == 0,
        "minimax_m31_topp_renorm_det_v3: probs must be 16-byte aligned with a width divisible by 4");
    TORCH_CHECK(d <= std::numeric_limits<int>::max() && rows <= 65535,
        "minimax_m31_topp_renorm_det_v3: probs is too large (rows <= 65535, width < 2^31)");
    TORCH_CHECK(topP.is_cuda() && topP.scalar_type() == torch::kFloat32 && topP.dim() == 1 && topP.is_contiguous()
            && topP.size(0) == rows && topP.get_device() == probs.get_device(),
        "minimax_m31_topp_renorm_det_v3: top_p must be a contiguous float32 [rows] tensor on probs' device");
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kFloat32 && out.dim() == 2 && out.is_contiguous()
            && out.size(0) == rows && out.size(1) == d && reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0
            && out.get_device() == probs.get_device(),
        "minimax_m31_topp_renorm_det_v3: out must be a contiguous, 16-byte aligned float32 [rows, d] on probs' device");
    TORCH_CHECK(ctasPerRow >= 1 && ctasPerRow <= 1024 && 1024 % ctasPerRow == 0,
        "minimax_m31_topp_renorm_det_v3: ctas_per_row must divide 1024");
    auto const wsBytesPerRow
        = static_cast<int64_t>(TRTLLM_MV2_NS::kernels::minimaxM31::toppRenormDetV3WorkspaceBytesPerRow());
    TORCH_CHECK(workspace.is_cuda() && workspace.is_contiguous() && workspace.get_device() == probs.get_device()
            && reinterpret_cast<uintptr_t>(workspace.data_ptr()) % 16 == 0
            && workspace.numel() * workspace.element_size() >= wsBytesPerRow * rows,
        "minimax_m31_topp_renorm_det_v3: workspace must be a contiguous, 16-byte aligned CUDA buffer of at least ",
        wsBytesPerRow * rows, " bytes on probs' device");
}

static void launchV3(torch::Tensor const& probs, torch::Tensor const& topP, torch::Tensor& workspace,
    torch::Tensor& out, int64_t ctasPerRow, bool pdl, bool candidateList, bool listPath, bool wideHist, bool keepAllGe1,
    TRTLLM_MV2_NS::kernels::minimaxM31::ToppRowStore const& rowStore, bool chainPrefetch)
{
    int64_t const rows = probs.size(0);
    int64_t const d = probs.size(1);
    if (rows == 0 || d == 0)
    {
        return;
    }
    c10::cuda::CUDAGuard const deviceGuard{probs.device()};
    auto const stream = at::cuda::getCurrentCUDAStream(probs.get_device());
    // A second row destination (FOLD_F1) runs the keep-all rule; keep_all_ge1 applies to the plain launch.
    TORCH_CHECK(rowStore.dst == nullptr || keepAllGe1,
        "minimax_m31_topp_renorm_det_v3: the FOLD_F1 row store runs the keep-all rule (keep_all_ge1)");
    auto const err = rowStore.dst == nullptr
        ? TRTLLM_MV2_NS::kernels::minimaxM31::invokeToppRenormDetV3(probs.data_ptr<float>(), out.data_ptr<float>(),
            topP.data_ptr<float>(), static_cast<int>(rows), static_cast<int>(d), static_cast<int>(ctasPerRow),
            workspace.data_ptr(), nullptr, pdl, candidateList, listPath, wideHist, keepAllGe1, stream, chainPrefetch)
        : TRTLLM_MV2_NS::kernels::minimaxM31::invokeToppRenormDetV3(probs.data_ptr<float>(), out.data_ptr<float>(),
            topP.data_ptr<float>(), static_cast<int>(rows), static_cast<int>(d), static_cast<int>(ctasPerRow),
            workspace.data_ptr(), nullptr, pdl, candidateList, listPath, wideHist, stream, rowStore, chainPrefetch);
    TORCH_CHECK(err == cudaSuccess, "minimax_m31_topp_renorm_det_v3 launch failed: ", cudaGetErrorString(err));
}

//! TOPP_DET v3 (see csrc/m31/toppRenormDetV3Sm103.h): minimax_m31_topp_renorm_det's renorm, bitwise its
//! output, with the redundant per-CTA selection and an optional programmatic dependent launch. workspace: a
//! zero-initialized byte buffer of at least rows x minimax_m31_topp_renorm_det_v3_ws_bytes(), reused across calls;
//! ctas_per_row divides 1024 and rows x ctas_per_row must not exceed the SM count when ctas_per_row > 1.
//! list_path (V4) / wide_hist (K8): the one-sync list path and 8 privatized pass histograms; both bitwise the same.
//! Both default to off, so a caller that does not pass them runs the v3 kernel unchanged.
//! chain_prefetch (TOPP_V6, default on): the sm_107 chain-prefetch summation; off, the staged loop (the same bits).
//! keep_all_ge1 (default on): rows with top_p >= 1 keep every element; off, they take FlashInfer's radix passes too, so
//! every row is bitwise FlashInfer's deterministic top_p_renorm_probs.
void minimaxM31ToppRenormDetV3(torch::Tensor const& probs, torch::Tensor const& topP, torch::Tensor& workspace,
    torch::Tensor& out, int64_t ctasPerRow, bool pdl, bool candidateList, bool listPath, bool wideHist, bool keepAllGe1,
    bool chainPrefetch)
{
    checkV3Operands(probs, topP, workspace, out, ctasPerRow);
    launchV3(probs, topP, workspace, out, ctasPerRow, pdl, candidateList, listPath, wideHist, keepAllGe1,
        TRTLLM_MV2_NS::kernels::minimaxM31::ToppRowStore{}, chainPrefetch);
}

//! FOLD_F1: minimax_m31_topp_renorm_det_v3 that also writes output row r, value for value, to
//! dst[slot_ids[r / k], r % k, :d] (the spec sampler's per-slot draft-prob rows; the store
//! `dst[slot_ids, :k, :d] = out.view(n, k, d)` done in the same launch). dst: float32 [slots, >= k, >= d] with unit
//! last-dim stride and 16-byte aligned rows; slot_ids: contiguous int64 [rows / k] (not bounds-checked).
void minimaxM31ToppRenormDetV3Rows(torch::Tensor const& probs, torch::Tensor const& topP, torch::Tensor& workspace,
    torch::Tensor& out, int64_t ctasPerRow, bool pdl, bool candidateList, bool listPath, bool wideHist,
    torch::Tensor& dst, torch::Tensor const& slotIds, int64_t k, bool chainPrefetch)
{
    checkV3Operands(probs, topP, workspace, out, ctasPerRow);
    int64_t const rows = probs.size(0);
    int64_t const d = probs.size(1);
    TORCH_CHECK(k >= 1 && rows % k == 0 && slotIds.is_cuda() && slotIds.scalar_type() == torch::kInt64
            && slotIds.dim() == 1 && slotIds.is_contiguous() && slotIds.size(0) * k == rows
            && slotIds.get_device() == probs.get_device(),
        "minimax_m31_topp_renorm_det_v3_rows: slot_ids must be a contiguous int64 [rows / k] tensor on probs' device");
    TORCH_CHECK(dst.is_cuda() && dst.scalar_type() == torch::kFloat32 && dst.dim() == 3 && dst.stride(2) == 1
            && dst.size(1) >= k && dst.size(2) >= d && dst.stride(0) % 4 == 0 && dst.stride(1) % 4 == 0
            && reinterpret_cast<uintptr_t>(dst.data_ptr()) % 16 == 0 && dst.get_device() == probs.get_device(),
        "minimax_m31_topp_renorm_det_v3_rows: dst must be a float32 [slots, >= k, >= d] tensor with unit last-dim "
        "stride and 16-byte aligned rows on probs' device");
    TRTLLM_MV2_NS::kernels::minimaxM31::ToppRowStore rowStore{};
    rowStore.dst = dst.data_ptr<float>();
    rowStore.slotIds = slotIds.data_ptr<int64_t>();
    rowStore.k = static_cast<int32_t>(k);
    rowStore.slotStride = dst.stride(0);
    rowStore.rowStride = dst.stride(1);
    launchV3(probs, topP, workspace, out, ctasPerRow, pdl, candidateList, listPath, wideHist, /*keepAllGe1=*/true,
        rowStore, chainPrefetch);
}

//! Workspace bytes per row of minimax_m31_topp_renorm_det_v3.
int64_t minimaxM31ToppRenormDetV3WsBytes()
{
    return static_cast<int64_t>(TRTLLM_MV2_NS::kernels::minimaxM31::toppRenormDetV3WorkspaceBytesPerRow());
}

} // namespace torch_ext

TRTLLM_MV2_NAMESPACE_END

TRTLLM_MV2_LIBRARY_FRAGMENT(m)
{
    m.def(
        "minimax_m31_topp_renorm_det_v3(Tensor probs, Tensor top_p, Tensor(a!) workspace, Tensor(b!) out, "
        "int ctas_per_row, bool pdl, bool candidate_list, bool list_path=False, bool wide_hist=False, "
        "bool keep_all_ge1=True, bool chain_prefetch=True) -> ()");
    m.def(
        "minimax_m31_topp_renorm_det_v3_rows(Tensor probs, Tensor top_p, Tensor(a!) workspace, Tensor(b!) out, "
        "int ctas_per_row, bool pdl, bool candidate_list, bool list_path, bool wide_hist, Tensor(c!) dst, "
        "Tensor slot_ids, int k, bool chain_prefetch=True) -> ()");
    // No tensor arguments: a catch-all kernel (as moe_a2a_get_timeout), callable without a device.
    m.def("minimax_m31_topp_renorm_det_v3_ws_bytes() -> int");
}

TRTLLM_MV2_LIBRARY_IMPL(CUDA, m)
{
    m.impl("minimax_m31_topp_renorm_det_v3", &TRTLLM_MV2_NS::torch_ext::minimaxM31ToppRenormDetV3);
    m.impl("minimax_m31_topp_renorm_det_v3_rows", &TRTLLM_MV2_NS::torch_ext::minimaxM31ToppRenormDetV3Rows);
}

TRTLLM_MV2_LIBRARY_IMPL(CompositeExplicitAutograd, m)
{
    m.impl("minimax_m31_topp_renorm_det_v3_ws_bytes", &TRTLLM_MV2_NS::torch_ext::minimaxM31ToppRenormDetV3WsBytes);
}
