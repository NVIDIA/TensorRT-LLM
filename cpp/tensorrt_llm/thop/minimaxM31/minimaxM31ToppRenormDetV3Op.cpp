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

#include "tensorrt_llm/kernels/minimaxM31/toppRenormDetV3Sm103.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cstdint>
#include <limits>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

//! TOPP_DET v3 (see kernels/minimaxM31/toppRenormDetV3Sm103.h): minimax_m31_topp_renorm_det's renorm, bitwise its
//! output, with the redundant per-CTA selection and an optional programmatic dependent launch. workspace: a
//! zero-initialized byte buffer of at least rows x minimax_m31_topp_renorm_det_v3_ws_bytes(), reused across calls;
//! ctas_per_row divides 1024 and rows x ctas_per_row must not exceed the SM count when ctas_per_row > 1.
//! list_path (V4) / wide_hist (K8): the one-sync list path and 8 privatized pass histograms; both bitwise the same.
//! Both default to off, so a caller that does not pass them runs the v3 kernel unchanged.
void minimaxM31ToppRenormDetV3(torch::Tensor const& probs, torch::Tensor const& topP, torch::Tensor& workspace,
    torch::Tensor& out, int64_t ctasPerRow, bool pdl, bool candidateList, bool listPath, bool wideHist)
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
        = static_cast<int64_t>(tensorrt_llm::kernels::minimaxM31::toppRenormDetV3WorkspaceBytesPerRow());
    TORCH_CHECK(workspace.is_cuda() && workspace.is_contiguous() && workspace.get_device() == probs.get_device()
            && reinterpret_cast<uintptr_t>(workspace.data_ptr()) % 16 == 0
            && workspace.numel() * workspace.element_size() >= wsBytesPerRow * rows,
        "minimax_m31_topp_renorm_det_v3: workspace must be a contiguous, 16-byte aligned CUDA buffer of at least ",
        wsBytesPerRow * rows, " bytes on probs' device");
    if (rows == 0 || d == 0)
    {
        return;
    }
    c10::cuda::CUDAGuard const deviceGuard{probs.device()};
    auto const err = tensorrt_llm::kernels::minimaxM31::invokeToppRenormDetV3(probs.data_ptr<float>(),
        out.data_ptr<float>(), topP.data_ptr<float>(), static_cast<int>(rows), static_cast<int>(d),
        static_cast<int>(ctasPerRow), workspace.data_ptr(), nullptr, pdl, candidateList, listPath, wideHist,
        at::cuda::getCurrentCUDAStream(probs.get_device()));
    TORCH_CHECK(err == cudaSuccess, "minimax_m31_topp_renorm_det_v3 launch failed: ", cudaGetErrorString(err));
}

//! Workspace bytes per row of minimax_m31_topp_renorm_det_v3.
int64_t minimaxM31ToppRenormDetV3WsBytes()
{
    return static_cast<int64_t>(tensorrt_llm::kernels::minimaxM31::toppRenormDetV3WorkspaceBytesPerRow());
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "minimax_m31_topp_renorm_det_v3(Tensor probs, Tensor top_p, Tensor(a!) workspace, Tensor(b!) out, "
        "int ctas_per_row, bool pdl, bool candidate_list, bool list_path=False, bool wide_hist=False) -> ()");
    // No tensor arguments: a catch-all kernel (as moe_a2a_get_timeout), callable without a device.
    m.def(
        "minimax_m31_topp_renorm_det_v3_ws_bytes() -> int", &tensorrt_llm::torch_ext::minimaxM31ToppRenormDetV3WsBytes);
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("minimax_m31_topp_renorm_det_v3", &tensorrt_llm::torch_ext::minimaxM31ToppRenormDetV3);
}
