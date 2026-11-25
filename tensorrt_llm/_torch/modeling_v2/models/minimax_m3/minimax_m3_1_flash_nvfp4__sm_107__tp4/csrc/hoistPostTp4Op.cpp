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

#include "minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/csrc/hoistPostTp4.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include "tensorrt_llm/modeling_v2/common/torchLibrary.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <optional>
#include <vector>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace torch_ext
{

namespace hp = TRTLLM_MV2_NS::kernels::minimaxM3Sm107Tp4::hoistPost;

namespace
{

template <typename T>
T* optPtr(std::optional<torch::Tensor> const& t)
{
    return t.has_value() ? static_cast<T*>(t->data_ptr()) : nullptr;
}

} // namespace

//! The reference extension's allreduce_norm_quant_route (route_warp_tp4_fc1_per_token_ext.cu), argument for argument
//! and check for check, minus the rx workspace lookup (rh_find): rx_bases are the MNNVL rx slot's [world] bases, which
//! the Python host (kernels/hoist/seams.py) allocates and checks against the prx bucket (maxt, world, rank,
//! npart).
void minimaxM3Sm107Tp4HoistPost(torch::Tensor const& pp, int64_t rows, torch::Tensor const& residual,
    torch::Tensor const& gamma, double eps, int64_t gemma, std::optional<torch::Tensor> const& outNorm,
    torch::Tensor const& outRes, std::optional<torch::Tensor> const& outQ, std::optional<torch::Tensor> const& outSf,
    int64_t cl, std::vector<int64_t> const& rxBases, std::optional<torch::Tensor> const& route,
    std::optional<torch::Tensor> const& rctr, std::optional<torch::Tensor> const& bias,
    std::optional<torch::Tensor> const& tab, int64_t delayNs, std::optional<torch::Tensor> const& lgDbg,
    std::optional<torch::Tensor> const& tdbg, int64_t repl, int64_t npart, std::optional<torch::Tensor> const& rxc,
    int64_t deadlineNs, int64_t hangDbg, int64_t roll, std::optional<torch::Tensor> const& x4Q,
    std::optional<torch::Tensor> const& x4Sf, std::optional<torch::Tensor> const& x4Sfsc, int64_t usePrr, int64_t rx2,
    int64_t pt, std::optional<torch::Tensor> const& x4Pts, int64_t pwait)
{
    TORCH_CHECK(!pp.is_cuda() && pp.scalar_type() == torch::kUInt8 && pp.is_contiguous(),
        "hoist_post: pp = the push-params bytes (CPU uint8)");
    TORCH_CHECK(pp.numel() >= static_cast<int64_t>(hp::pushParamsBytes()), "push params size");
    hp::PushParamsView v{};
    TORCH_CHECK(hp::decodePushParams(pp.data_ptr(), static_cast<size_t>(pp.numel()), &v), "push params decode");
    TORCH_CHECK(v.cluster == 4 && v.ndst == v.world, "v3 consumer: unicast, cluster-4 parity layout only");
    TORCH_CHECK(v.world >= 1 && v.world <= hp::kMaxWorld && v.rank >= 0 && v.rank < v.world, "push params rank/world");
    TORCH_CHECK(residual.is_cuda() && residual.scalar_type() == torch::kBFloat16, "hoist_post: residual bf16");
    TORCH_CHECK(gamma.is_cuda() && gamma.scalar_type() == torch::kBFloat16, "hoist_post: gamma bf16");
    TORCH_CHECK(outRes.is_cuda() && outRes.scalar_type() == torch::kBFloat16, "hoist_post: out_res bf16");
    TORCH_CHECK(rows >= 0 && rows <= v.maxt);
    TORCH_CHECK(roll == 0 || (roll == 1 && v.world == 2),
        "allreduce_norm_quant_route: roll 1 is the TP2 (world 2) specialisation");
    hp::HoistPostParams a{};
    a.pushParams = pp.data_ptr();
    a.residual = residual.data_ptr();
    a.gamma = gamma.data_ptr();
    a.outRes = outRes.data_ptr();
    a.outNorm = optPtr<void>(outNorm);
    a.outQ = optPtr<void>(outQ);
    a.outSf = optPtr<void>(outSf);
    a.eps = static_cast<float>(eps);
    a.rows = static_cast<int>(rows);
    a.gemma = static_cast<int>(gemma);
    if (route.has_value())
    {
        TORCH_CHECK(
            repl >= 1 && repl <= 32, "allreduce_norm_quant_route: 1 <= repl <= 32 (one publishing lane per replica)");
        TORCH_CHECK(
            route->scalar_type() == torch::kInt64 && route->is_contiguous(), "allreduce_norm_quant_route: route int64");
        TORCH_CHECK(
            route->numel() % (repl * hp::kLineU64) == 0, "allreduce_norm_quant_route: route = [repl][maxt][16]");
        a.rmaxt = static_cast<int>(route->numel() / (repl * hp::kLineU64));
        TORCH_CHECK(rows <= a.rmaxt, "allreduce_norm_quant_route: route lines have fewer rows than rows");
        TORCH_CHECK(rctr.has_value() && rctr->scalar_type() == torch::kInt32
                && rctr->numel() >= static_cast<int64_t>(a.rmaxt) * hp::kRctrStride,
            "allreduce_norm_quant_route: rctr int32 [maxt][32]");
        TORCH_CHECK(reinterpret_cast<int64_t>(route->data_ptr()) % 128 == 0
                && reinterpret_cast<int64_t>(rctr->data_ptr()) % 128 == 0,
            "allreduce_norm_quant_route: route lines / rctr rows must be 128-byte aligned (one writer per line)");
        a.route = static_cast<uint64_t*>(route->data_ptr());
        a.rctr = static_cast<int*>(rctr->data_ptr());
        a.repl = static_cast<int>(repl);
        if (tab.has_value())
        {
            TORCH_CHECK(tab->scalar_type() == torch::kInt32 && tab->numel() >= rows * hp::kTopK,
                "allreduce_norm_quant_route: tab int32 [maxt][4]");
            a.tab = static_cast<unsigned const*>(tab->data_ptr());
            a.delayNs = static_cast<int>(delayNs);
        }
        else
        {
            TORCH_CHECK(bias.has_value() && bias->scalar_type() == torch::kFloat32 && bias->numel() == hp::kExperts,
                "allreduce_norm_quant_route: bias fp32 [128]");
            TORCH_CHECK(static_cast<int>(rxBases.size()) == v.world,
                "allreduce_norm_quant_route: rx_bases = the rx slot's [world] bases");
            TORCH_CHECK(rxBases[v.rank] != 0 && rxBases[v.rank] % 128 == 0,
                "allreduce_norm_quant_route: this rank's rx slot base (MNNVL carve, 128-byte aligned)");
            TORCH_CHECK(rxc.has_value() && rxc->scalar_type() == torch::kInt32 && rxc->numel() >= rows * hp::kRctrStride
                    && reinterpret_cast<int64_t>(rxc->data_ptr()) % 128 == 0,
                "allreduce_norm_quant_route: rxc int32 [maxt][32], 128-byte aligned");
            TORCH_CHECK(deadlineNs > 0, "allreduce_norm_quant_route: deadline_ns");
            a.rx = reinterpret_cast<uint64_t*>(rxBases[v.rank]);
            a.rxc = static_cast<int*>(rxc->data_ptr());
            a.npart = static_cast<int>(npart);
            a.deadlineNs = static_cast<unsigned long long>(deadlineNs);
            a.hangDbg = static_cast<int>(hangDbg);
            a.bias = static_cast<float const*>(bias->data_ptr());
        }
        if (lgDbg.has_value())
        {
            TORCH_CHECK(lgDbg->scalar_type() == torch::kFloat32 && lgDbg->numel() >= rows * hp::kExperts,
                "allreduce_norm_quant_route: lg_dbg fp32 [maxt][128]");
            a.lgDbg = static_cast<float*>(lgDbg->data_ptr());
        }
        if (tdbg.has_value())
        {
            TORCH_CHECK(tdbg->scalar_type() == torch::kInt64 && tdbg->numel() >= 16LL * a.rmaxt * hp::kTdbgW,
                "allreduce_norm_quant_route: tdbg int64 [16][maxt][4]");
            a.tdbg = static_cast<uint64_t*>(tdbg->data_ptr());
        }
    }
    bool const x4 = x4Q.has_value();
    if (x4)
    {
        // the engine's B1 inputs: row-major packed e2m1 [maxt][3072], e4m3 scales [maxt][384]
        TORCH_CHECK(
            x4Sf.has_value() && x4Sfsc.has_value(), "allreduce_norm_quant_route: x4 needs x4_q, x4_sf and x4_sfsc");
        TORCH_CHECK(x4Q->scalar_type() == torch::kUInt8 && x4Q->is_contiguous() && x4Q->numel() % (hp::kHidden / 2) == 0
                && x4Q->numel() / (hp::kHidden / 2) >= rows && reinterpret_cast<int64_t>(x4Q->data_ptr()) % 16 == 0,
            "allreduce_norm_quant_route: x4_q uint8 [>=rows][3072], 16 B aligned");
        TORCH_CHECK(x4Sf->scalar_type() == torch::kUInt8 && x4Sf->is_contiguous()
                && x4Sf->numel() % (hp::kHidden / 16) == 0 && x4Sf->numel() / (hp::kHidden / 16) >= rows
                && reinterpret_cast<int64_t>(x4Sf->data_ptr()) % 16 == 0,
            "allreduce_norm_quant_route: x4_sf uint8 [>=rows][384], 16 B aligned");
        TORCH_CHECK(x4Sfsc->scalar_type() == torch::kFloat32 && x4Sfsc->numel() == 1 && x4Sfsc->is_cuda(),
            "allreduce_norm_quant_route: x4_sfsc fp32 [1] cuda");
        a.x4Q = static_cast<uint8_t*>(x4Q->data_ptr());
        a.x4Sf = static_cast<uint8_t*>(x4Sf->data_ptr());
        a.x4Sfsc = static_cast<float const*>(x4Sfsc->data_ptr());
        if (pt)
        {
            TORCH_CHECK(usePrr && x4Pts.has_value() && x4Pts->scalar_type() == torch::kFloat32 && x4Pts->is_cuda()
                    && x4Pts->numel() >= rows,
                "allreduce_norm_quant_route: pt needs prr 1 and x4_pts fp32 [>= rows]");
            a.x4Pts = static_cast<float*>(x4Pts->data_ptr());
        }
    }
    else
    {
        TORCH_CHECK(!pt && !x4Pts.has_value(), "allreduce_norm_quant_route: pt / x4_pts without x4_q");
        TORCH_CHECK(
            !x4Sf.has_value() && !x4Sfsc.has_value(), "allreduce_norm_quant_route: x4_sf / x4_sfsc without x4_q");
    }
    TORCH_CHECK(usePrr, "allreduce_norm_quant_route: this extension launches only the routing-warp post (prr=1)");
    TORCH_CHECK(v.world == hp::kPrrWorld, "allreduce_norm_quant_route: prr4 is the world-4 path");
    TORCH_CHECK(!route.has_value() || tab.has_value() || npart == hp::kPrrNpart,
        "allreduce_norm_quant_route: prr4 needs npart 1");
    TORCH_CHECK(!pwait || cl == 8, "allreduce_norm_quant_route: pwait (M3LL_DEV_OPROJ_PTRIG=1) is built for cl 8 only");
    a.cl = static_cast<int>(cl);
    a.x4 = x4 ? (pt ? 2 : 1) : 0;
    a.rx2 = rx2 ? 1 : 0;
    a.pwait = pwait ? 1 : 0;
    at::cuda::CUDAGuard guard(residual.device());
    cudaError_t const err = hp::launch(a, at::cuda::getCurrentCUDAStream(residual.get_device()));
    TORCH_CHECK(err == cudaSuccess, "lbv3r launch: ", cudaGetErrorString(err));
}

} // namespace torch_ext

TRTLLM_MV2_NAMESPACE_END

TRTLLM_MV2_LIBRARY_FRAGMENT(m)
{
    m.def(
        "minimax_m3_sm107tp4_hoist_post(Tensor pp, int rows, Tensor residual, Tensor gamma, float eps, int gemma, "
        "Tensor(a!)? out_norm, Tensor(b!) out_res, Tensor(c!)? out_q, Tensor(d!)? out_sf, int cl, int[] rx_bases, "
        "Tensor(e!)? route, Tensor(f!)? rctr, Tensor? bias, Tensor? tab, int delay_ns, Tensor(g!)? lg_dbg, "
        "Tensor(h!)? tdbg, int repl, int npart, Tensor(i!)? rxc, int deadline_ns, int hang_dbg, int roll, "
        "Tensor(j!)? x4_q, Tensor(k!)? x4_sf, Tensor? x4_sfsc, int prr, int rx2, int pt, Tensor(l!)? x4_pts, "
        "int pwait) -> ()");
}

TRTLLM_MV2_LIBRARY_IMPL(CUDA, m)
{
    m.impl("minimax_m3_sm107tp4_hoist_post", &TRTLLM_MV2_NS::torch_ext::minimaxM3Sm107Tp4HoistPost);
}
