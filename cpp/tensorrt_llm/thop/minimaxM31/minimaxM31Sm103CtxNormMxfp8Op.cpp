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

#include "tensorrt_llm/kernels/minimaxM31/ctxResidualRmsNormMxfp8Sm103.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cstdlib>
#include <optional>
#include <tuple>
#include <vector>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

//! MiniMax-M3.1 sm_103 targets (target-local copy of the MiniMax-M3 SM107 CTX op; PDL is a caller argument).
//! Residual add + RMSNorm of an all-reduce output (in place: inter becomes inter + residual), with the MXFP8
//! quantization of the normalized rows in the same pass. Returns (norm_out, mxfp8, sf, fp4, fp4_sf) where mxfp8 / sf
//! match mxfp8_quantize(norm_out, swizzled=True) bit for bit (mxfp8 [rows, hidden] Float8_e4m3fn, a 1-D UINT8 scale
//! tensor of the swizzled 128x4 layout). With fp4GlobalScale, fp4 / fp4_sf match
//! fp4_quantize(norm_out, fp4GlobalScale, 16, sf_use_ue8m0=False, is_sf_swizzled=False) bit for bit (fp4 [rows,
//! hidden / 2] UINT8, a 1-D UINT8 linear scale tensor of rows * hidden / 16); otherwise both are empty. norm_out is
//! empty when writeNorm is false.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> minimaxM31CtxResidualRmsNormMxfp8(
    torch::Tensor& inter, torch::Tensor const& residual, torch::Tensor const& weight, double eps, bool writeNorm,
    std::optional<torch::Tensor> const& fp4GlobalScale, bool enablePdl)
{
    TORCH_CHECK(inter.is_cuda() && inter.is_contiguous() && inter.scalar_type() == at::kBFloat16 && inter.dim() >= 2);
    TORCH_CHECK(
        residual.is_contiguous() && residual.scalar_type() == at::kBFloat16 && residual.sizes() == inter.sizes());
    int64_t const hidden = inter.size(-1);
    int64_t const rows = inter.numel() / hidden;
    TORCH_CHECK(weight.is_contiguous() && weight.scalar_type() == at::kBFloat16 && weight.numel() == hidden);
    TORCH_CHECK(hidden % 256 == 0 && hidden / 8 <= 1024, "hidden must be a multiple of 256 and at most 8192");
    at::cuda::CUDAGuard const guard(inter.device());
    auto const opts = inter.options();
    torch::Tensor normOut = writeNorm ? torch::empty_like(inter) : torch::empty({0}, opts);
    torch::Tensor qOut = torch::empty(inter.sizes(), opts.dtype(at::ScalarType::Float8_e4m3fn));
    int64_t const sfSize = (rows + 127) / 128 * 128 * ((hidden / 32 + 3) / 4 * 4);
    torch::Tensor sfOut = torch::empty({sfSize}, opts.dtype(at::ScalarType::Byte));
    bool const fp4 = fp4GlobalScale.has_value();
    if (fp4)
    {
        TORCH_CHECK(
            fp4GlobalScale->is_cuda() && fp4GlobalScale->scalar_type() == at::kFloat && fp4GlobalScale->numel() == 1);
    }
    std::vector<int64_t> fp4Shape(inter.sizes().begin(), inter.sizes().end());
    fp4Shape.back() = hidden / 2;
    torch::Tensor fp4Out = fp4 ? torch::empty(fp4Shape, opts.dtype(at::ScalarType::Byte)) : torch::empty({0}, opts);
    torch::Tensor fp4SfOut
        = fp4 ? torch::empty({rows * (hidden / 16)}, opts.dtype(at::ScalarType::Byte)) : torch::empty({0}, opts);
    tensorrt_llm::kernels::minimaxM31Sm103::invokeCtxResidualRmsNormMxfp8(
        reinterpret_cast<__nv_bfloat16*>(inter.data_ptr()), reinterpret_cast<__nv_bfloat16 const*>(residual.data_ptr()),
        reinterpret_cast<__nv_bfloat16 const*>(weight.data_ptr()),
        writeNorm ? reinterpret_cast<__nv_bfloat16*>(normOut.data_ptr()) : nullptr,
        reinterpret_cast<uint64_t*>(qOut.data_ptr()), reinterpret_cast<uint32_t*>(sfOut.data_ptr()),
        fp4 ? reinterpret_cast<uint32_t*>(fp4Out.data_ptr()) : nullptr,
        fp4 ? reinterpret_cast<uint32_t*>(fp4SfOut.data_ptr()) : nullptr,
        fp4 ? fp4GlobalScale->data_ptr<float>() : nullptr, /*f32Out=*/nullptr, static_cast<int>(rows),
        static_cast<int>(hidden), static_cast<float>(eps), enablePdl,
        at::cuda::getCurrentCUDAStream(inter.get_device()));
    return {normOut, qOut, sfOut, fp4Out, fp4SfOut};
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "minimax_m31_ctx_residual_rmsnorm_mxfp8(Tensor(a!) inter, Tensor residual, Tensor weight, float eps, "
        "bool write_norm, Tensor? fp4_global_scale=None, bool enable_pdl=False) -> (Tensor, Tensor, Tensor, Tensor, "
        "Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("minimax_m31_ctx_residual_rmsnorm_mxfp8", &tensorrt_llm::torch_ext::minimaxM31CtxResidualRmsNormMxfp8);
}
