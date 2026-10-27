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

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/minimaxM3Sm107Tp8/mcAllreduceNormTp8.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <algorithm>
#include <optional>
#include <vector>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

constexpr int64_t kMaxTokens = 48;
constexpr int64_t kCluster = 8;
constexpr int64_t kMaxWorld = 8;

} // namespace

//! MiniMax-M3 SM107 TP8 multicast all-reduce (stock rounding points: BF16 rank sum, then the residual add) + residual +
//! RMSNorm (+ optional MoE finalize, + optional MXFP8 output).
//!
//! @param x BF16 [M, H] partial, or with `perm` the permuted FC2 output [rows, H].
//! @param x2 Optional BF16 [M, H] second addend (the shared expert in finalize mode).
//! @param residual BF16 [M, H], 1 <= M <= 48.
//! @param normWeight BF16 [H].
//! @param wsUc This rank's unicast view of the exchange workspace (int32 words); re-armed in place.
//! @param wsMcPtr Device address of the workspace's multicast mapping.
//! @param flags Int32 per-CTA call-parity counters, at least maxt * 8; updated in place.
//! @param cluster Cluster size; 8 is the only one built.
//! @param perm, scales Finalize mode: int32 [M, topk] permuted-row index and fp32 / bf16 [M, topk] routing weights.
//! @param maxt Token capacity of the workspace bucket (8, 16 or 48).
//! @param quant Also return the MXFP8 copy of the normed rows (E4M3 [M, H], swizzled UE8M0 scales).
//! @param prepushed 0, or 2 / 3 when producers already pushed this rank's rows (see the kernel header).
//! @param opts Epilogue variant; every variant is bit-identical.
//! @param capOut Optional BF16 [>= M, H] view (row stride a multiple of 8, 16-byte aligned) that receives the Eagle3
//! capture of the updated residual: m3_spec_tail capture_norm with capW [H] and capEps, bit for bit (H 6144; not the
//! drafter boundary, opts bit 16).
//! @return [normed, updated residual] or, with `quant`, [normed, updated residual, E4M3 rows, scale bytes].
std::vector<torch::Tensor> minimaxM3Sm107Tp8McAllreduceNorm(torch::Tensor const& x,
    std::optional<torch::Tensor> const& x2, torch::Tensor const& residual, torch::Tensor const& normWeight, double eps,
    torch::Tensor& wsUc, int64_t wsMcPtr, torch::Tensor& flags, int64_t rank, int64_t world, int64_t cluster, bool pdl,
    std::optional<torch::Tensor> const& perm, std::optional<torch::Tensor> const& scales, int64_t maxt, bool quant,
    int64_t prepushed, int64_t opts, std::optional<torch::Tensor> const& capOut,
    std::optional<torch::Tensor> const& capW, double capEps)
{
    int const sm = tensorrt_llm::common::getSMVersion();
    TORCH_CHECK(sm == 107, "minimax_m3_sm107_tp8_mc_allreduce_norm requires SM107, got SM", sm);
    TORCH_CHECK(cluster == kCluster, "minimax_m3_sm107_tp8_mc_allreduce_norm: only cluster 8 is built, got ", cluster);
    TORCH_CHECK(tensorrt_llm::kernels::isMinimaxM3Sm107Tp8McAllreduceNormSupported(
                    static_cast<int32_t>(maxt), static_cast<int32_t>(opts)),
        "minimax_m3_sm107_tp8_mc_allreduce_norm: unsupported (bucket ", maxt, ", opts ", opts, ")");
    TORCH_CHECK(
        world >= 1 && world <= kMaxWorld && rank >= 0 && rank < world, "invalid rank ", rank, " / world ", world);

    tensorrt_llm::kernels::MinimaxM3Sm107Tp8McAllreduceNormParams p;
    if (x2.has_value())
    {
        TORCH_CHECK(x2->scalar_type() == at::kBFloat16 && x2->is_contiguous() && x2->sizes() == residual.sizes(),
            "x2 must be a contiguous bf16 tensor shaped like residual");
        p.x2 = reinterpret_cast<__nv_bfloat16 const*>(x2->data_ptr());
    }
    if (perm.has_value())
    {
        // Finalize mode: x is the permuted FC2 output [rows, H].
        TORCH_CHECK(scales.has_value() && perm->scalar_type() == at::kInt && perm->is_contiguous() && perm->dim() == 2
                && perm->size(0) == residual.size(0),
            "perm must be a contiguous int32 [M, topk] tensor, with scales");
        TORCH_CHECK(scales->is_contiguous() && scales->numel() == perm->numel()
                && (scales->scalar_type() == at::kFloat || scales->scalar_type() == at::kBFloat16),
            "scales must be a contiguous fp32 or bf16 tensor with one weight per perm entry");
        TORCH_CHECK(x.dim() == 2 && x.size(1) == residual.size(1), "fc2 must be [rows, H]");
        p.fc2 = reinterpret_cast<__nv_bfloat16 const*>(x.data_ptr());
        p.perm = perm->data_ptr<int32_t>();
        p.scales = scales->data_ptr();
        p.scaleFp32 = scales->scalar_type() == at::kFloat;
        p.topk = static_cast<int32_t>(perm->size(1));
    }
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.dim() == 2,
        "x must be a contiguous 2-D bf16 CUDA tensor");
    TORCH_CHECK(residual.dim() == 2 && residual.size(0) >= 1 && residual.size(0) <= std::min(kMaxTokens, maxt),
        "residual must hold 1..min(48, bucket) rows");
    TORCH_CHECK(residual.scalar_type() == at::kBFloat16 && residual.is_contiguous()
            && (perm.has_value() || residual.sizes() == x.sizes()),
        "residual must be a contiguous bf16 tensor shaped like x");
    TORCH_CHECK(
        normWeight.scalar_type() == at::kBFloat16 && normWeight.numel() == residual.size(1), "norm_w must be bf16 [H]");
    TORCH_CHECK(wsUc.scalar_type() == at::kInt && wsUc.is_cuda(), "ws_uc must be an int32 CUDA tensor");
    TORCH_CHECK(flags.scalar_type() == at::kInt && flags.is_contiguous() && flags.numel() >= maxt * kCluster,
        "flags must be a contiguous int32 tensor of at least bucket * 8 counters");

    int64_t const m = residual.size(0), h = residual.size(1);
    TORCH_CHECK(!quant || h % 128 == 0, "MXFP8 output needs H % 128 == 0");
    c10::cuda::CUDAGuard const deviceGuard{residual.device()};
    auto outNormed = torch::empty_like(residual);
    auto outResidual = torch::empty_like(residual);
    auto outQ = quant ? torch::empty({m, h}, residual.options().dtype(at::kFloat8_e4m3fn)) : torch::Tensor();
    auto outSf = quant ? torch::empty({(m + 127) / 128 * 128 * (h / 32)}, residual.options().dtype(at::kByte))
                       : torch::Tensor();

    p.x = reinterpret_cast<__nv_bfloat16 const*>(x.data_ptr());
    p.numTokens = static_cast<int32_t>(m);
    p.hidden = static_cast<int32_t>(h);
    p.residual = reinterpret_cast<__nv_bfloat16 const*>(residual.data_ptr());
    p.normWeight = reinterpret_cast<__nv_bfloat16 const*>(normWeight.data_ptr());
    p.eps = static_cast<float>(eps);
    p.wsUc = wsUc.data_ptr<int32_t>();
    p.wsMc = reinterpret_cast<int32_t*>(wsMcPtr);
    p.flags = flags.data_ptr<int32_t>();
    p.rank = static_cast<int32_t>(rank);
    p.world = static_cast<int32_t>(world);
    p.outNormed = reinterpret_cast<__nv_bfloat16*>(outNormed.data_ptr());
    p.outResidual = reinterpret_cast<__nv_bfloat16*>(outResidual.data_ptr());
    p.outQuant = quant ? reinterpret_cast<uint8_t*>(outQ.data_ptr()) : nullptr;
    p.outSf = quant ? outSf.data_ptr<uint8_t>() : nullptr;
    p.prepushed = static_cast<int32_t>(prepushed);
    if (capOut.has_value())
    {
        TORCH_CHECK((opts & 16) == 0, "the capture epilogue is not built for the drafter boundary (opts bit 16)");
        TORCH_CHECK(capOut->scalar_type() == at::kBFloat16 && capOut->dim() == 2 && capOut->size(0) >= m
                && capOut->size(1) == h && capOut->stride(1) == 1 && capOut->stride(0) % 8 == 0
                && reinterpret_cast<uintptr_t>(capOut->data_ptr()) % 16 == 0 && capOut->device() == residual.device(),
            "cap_out must be a bf16 [>= M, H] view with unit column stride, a row stride that is a multiple of 8 and "
            "16-byte alignment");
        TORCH_CHECK(
            capW.has_value() && capW->scalar_type() == at::kBFloat16 && capW->is_contiguous() && capW->numel() == h,
            "cap_w must be a contiguous bf16 [H] tensor");
        p.capOut = reinterpret_cast<__nv_bfloat16*>(capOut->data_ptr());
        p.capStride = capOut->stride(0);
        p.capWeight = reinterpret_cast<__nv_bfloat16 const*>(capW->data_ptr());
        p.capEps = static_cast<float>(capEps);
    }
    tensorrt_llm::kernels::invokeMinimaxM3Sm107Tp8McAllreduceNorm(p, static_cast<int32_t>(maxt),
        static_cast<int32_t>(opts), pdl, at::cuda::getCurrentCUDAStream(residual.get_device()));
    if (quant)
    {
        return {outNormed, outResidual, outQ, outSf};
    }
    return {outNormed, outResidual};
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "minimax_m3_sm107_tp8_mc_allreduce_norm(Tensor x, Tensor? x2, Tensor residual, Tensor norm_w, float eps, "
        "Tensor(a!) ws_uc, int ws_mc_ptr, Tensor(b!) flags, int rank, int world, int cluster, bool pdl, "
        "Tensor? perm, Tensor? scales, int maxt, bool quant, int prepushed, int opts, Tensor(c!)? cap_out=None, "
        "Tensor? cap_w=None, float cap_eps=0.) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("minimax_m3_sm107_tp8_mc_allreduce_norm", &tensorrt_llm::torch_ext::minimaxM3Sm107Tp8McAllreduceNorm);
}
