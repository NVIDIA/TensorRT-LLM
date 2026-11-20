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

// The reference headers declare their kernels and structs at global scope (and `using namespace lb;`): keep them in
// this translation unit only, ahead of the TensorRT-LLM namespace.
#include "minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/csrc/route_warp_tp4_fc1_per_token.cuh"
#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include "minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/csrc/hoistPostTp4.h"

#include <cstring>
#include <type_traits>

static_assert(lb::H == TRTLLM_MV2_NS::kernels::minimaxM3Sm107Tp4::hoistPost::kHidden, "peer lb::H");
static_assert(lb::MAXW == TRTLLM_MV2_NS::kernels::minimaxM3Sm107Tp4::hoistPost::kMaxWorld, "peer lb::MAXW");
static_assert(rh::E == TRTLLM_MV2_NS::kernels::minimaxM3Sm107Tp4::hoistPost::kExperts, "peer rh::E");
static_assert(rh::TOPK == TRTLLM_MV2_NS::kernels::minimaxM3Sm107Tp4::hoistPost::kTopK, "peer rh::TOPK");
static_assert(rh::LINE_U64 == TRTLLM_MV2_NS::kernels::minimaxM3Sm107Tp4::hoistPost::kLineU64, "peer rh::LINE_U64");
static_assert(rh::RCTR_STRIDE == TRTLLM_MV2_NS::kernels::minimaxM3Sm107Tp4::hoistPost::kRctrStride, "peer RCTR_STRIDE");
static_assert(rh::TDBG_W == TRTLLM_MV2_NS::kernels::minimaxM3Sm107Tp4::hoistPost::kTdbgW, "peer rh::TDBG_W");
static_assert(prr::W == TRTLLM_MV2_NS::kernels::minimaxM3Sm107Tp4::hoistPost::kPrrWorld, "peer prr::W");
static_assert(prr::NP == TRTLLM_MV2_NS::kernels::minimaxM3Sm107Tp4::hoistPost::kPrrNpart, "peer prr::NP");
static_assert(sizeof(PushParamsV3) == 8 * 8 + 8 + 5 * 4 + 4, "peer PushParamsV3 layout (8 peers, flags, 5 ints, pad)");

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{

namespace minimaxM3Sm107Tp4
{

namespace hoistPost
{

size_t pushParamsBytes()
{
    return sizeof(PushParamsV3);
}

bool decodePushParams(void const* bytes, size_t nbytes, PushParamsView* out)
{
    if (nbytes < sizeof(PushParamsV3))
    {
        return false;
    }
    PushParamsV3 pp;
    std::memcpy(&pp, bytes, sizeof(pp));
    for (int i = 0; i < 8; ++i)
    {
        out->peers[i] = reinterpret_cast<int64_t>(pp.peers[i]);
    }
    out->flags = reinterpret_cast<int64_t>(pp.flags);
    out->rank = pp.rank;
    out->world = pp.world;
    out->maxt = pp.maxt;
    out->cluster = pp.cluster;
    out->ndst = pp.ndst;
    return true;
}

namespace
{

template <typename Kern>
cudaError_t go(Kern kern, int clv, int thv, int rows, V3Params const& p, rh::RhParams const& q, cudaStream_t st)
{
    cudaError_t err = cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    if (err != cudaSuccess)
    {
        return err;
    }
    cudaLaunchConfig_t cfg = {};
    // rows 0: one idle cluster (build-time module load); every row predicate is t < rows, so it touches no rx, prx
    // slot, flag, counter or output (peer extension, verbatim semantics)
    cfg.gridDim = dim3((rows > 0 ? rows : 1) * clv);
    cfg.blockDim = dim3(thv);
    cfg.stream = st;
    cudaLaunchAttribute at[2];
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = clv;
    at[0].val.clusterDim.y = 1;
    at[0].val.clusterDim.z = 1;
    at[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[1].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = at;
    cfg.numAttrs = 2;
    return cudaLaunchKernelEx(&cfg, kern, p, q);
}

template <int X, int R2>
cudaError_t byCl(int cl, int pwait, int rows, V3Params const& p, rh::RhParams const& q, cudaStream_t st)
{
    if (pwait)
    {
        // DEV (M3LL_DEV_OPROJ_PTRIG=1): the cl-8 post with the end-of-CTA grid wait (the op checks cl == 8)
        return go(lbv3rp_kernel<8, 96, X, R2, 1>, 8, 128, rows, p, q, st);
    }
    if (cl == 4)
    {
        return go(lbv3rp_kernel<4, 192, X, R2>, 4, 224, rows, p, q, st);
    }
    if (cl == 16)
    {
        return go(lbv3rp_kernel<16, 64, X, R2>, 16, 96, rows, p, q, st);
    }
    return go(lbv3rp_kernel<8, 96, X, R2>, 8, 128, rows, p, q, st);
}

} // namespace

cudaError_t launch(HoistPostParams const& a, cudaStream_t stream)
{
    PushParamsV3 pp;
    std::memcpy(&pp, a.pushParams, sizeof(pp));
    V3Params p{};
    p.ex = pp.peers[pp.rank];
    p.flags = const_cast<int*>(pp.flags);
    p.maxt = pp.maxt;
    p.world = pp.world;
    p.rank = pp.rank;
    p.residual = static_cast<__nv_bfloat16 const*>(a.residual);
    p.gamma = static_cast<__nv_bfloat16 const*>(a.gamma);
    p.out_res = static_cast<__nv_bfloat16*>(a.outRes);
    p.out_norm = static_cast<__nv_bfloat16*>(a.outNorm);
    p.out_q = static_cast<uint8_t*>(a.outQ);
    p.out_sf = static_cast<uint8_t*>(a.outSf);
    p.eps = a.eps;
    p.rows = a.rows;
    p.gemma = a.gemma;
    p.sf_k_tiles = H / 32 / 4;
    rh::RhParams q{};
    q.rx = reinterpret_cast<unsigned long long*>(a.rx);
    q.rxc = a.rxc;
    q.bias = a.bias;
    q.route = reinterpret_cast<unsigned long long*>(a.route);
    q.rctr = a.rctr;
    q.lg_dbg = a.lgDbg;
    q.tdbg = reinterpret_cast<unsigned long long*>(a.tdbg);
    q.tab = a.tab;
    q.npart = a.npart;
    q.repl = a.repl;
    q.delay_ns = a.delayNs;
    q.rmaxt = a.rmaxt;
    q.deadline_ns = a.deadlineNs;
    q.hang_dbg = a.hangDbg;
    q.x4_q = a.x4Q;
    q.x4_sf = a.x4Sf;
    q.x4_sfsc = a.x4Sfsc;
    q.x4_pts = a.x4Pts;
    int const rows = a.rows;
    // ROLL is not a template of this kernel; X4 / RX2 / CL / PW select the instantiation as the reference extension
    // does.
    if (a.x4 == 2)
    {
        return a.rx2 ? byCl<2, 1>(a.cl, a.pwait, rows, p, q, stream) : byCl<2, 0>(a.cl, a.pwait, rows, p, q, stream);
    }
    if (a.x4 == 1)
    {
        return a.rx2 ? byCl<1, 1>(a.cl, a.pwait, rows, p, q, stream) : byCl<1, 0>(a.cl, a.pwait, rows, p, q, stream);
    }
    return a.rx2 ? byCl<0, 1>(a.cl, a.pwait, rows, p, q, stream) : byCl<0, 0>(a.cl, a.pwait, rows, p, q, stream);
}

} // namespace hoistPost

} // namespace minimaxM3Sm107Tp4

} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
