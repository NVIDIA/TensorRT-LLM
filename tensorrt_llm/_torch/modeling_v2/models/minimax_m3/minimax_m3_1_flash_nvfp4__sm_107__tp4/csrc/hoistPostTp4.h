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

#pragma once

#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include <cuda_runtime.h>
#include <stdint.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{

namespace minimaxM3Sm107Tp4
{

//! The TP4 C1 MoE-boundary post of the low-latency reference chain (router.lr4-prr4-q / post.prr4),
//! ported as-is: hoistPostTp4/ holds the reference route_warp_tp4_fc1_per_token.cuh, allreduce_norm_quant.cuh and
//! lamport_exchange.cuh byte for byte (their sha1 with the reference extension source is the reference H9_HDR_HASH
//! 40c1042cbf7d). The launch body is the reference extension's allreduce_norm_quant_route without its pybind / torch
//! layer and without the cudaIpc rx workspace (router_exchange_create / open / reset): the rx slots now live in an
//! MNNVL allocation made by the Python host (kernels/hoist), which also checks what rh_find checked.
namespace hoistPost
{

// Shapes the reference kernels are written for (the .cu static_asserts these against the reference headers).
constexpr int kHidden = 6144;   // lb::H
constexpr int kMaxWorld = 8;    // lb::MAXW
constexpr int kExperts = 128;   // rh::E
constexpr int kTopK = 4;        // rh::TOPK
constexpr int kLineU64 = 16;    // rh::LINE_U64
constexpr int kRctrStride = 32; // rh::RCTR_STRIDE
constexpr int kTdbgW = 4;       // rh::TDBG_W
constexpr int kPrrWorld = 4;    // prr::W
constexpr int kPrrNpart = 1;    // prr::NP

//! The reference PushParamsV3 (allreduce_norm_quant.cuh), decoded from the push-params bytes the o_proj producer uses.
struct PushParamsView
{
    int64_t peers[8];
    int64_t flags;
    int rank, world, maxt, cluster, ndst;
};

//! Size of the reference PushParamsV3 in bytes.
size_t pushParamsBytes();

//! Decode the push-params bytes (the layout of the reference PushParamsV3); false when nbytes is short.
bool decodePushParams(void const* bytes, size_t nbytes, PushParamsView* out);

//! Every field of the reference V3Params (the prepushed row path) and rh::RhParams (the routing warp + X4 tail) the
//! peer extension sets; pointers are device pointers, nullptr = off (as the reference c10::optional = None).
struct HoistPostParams
{
    // V3Params
    void const* pushParams; // the raw PushParamsV3 bytes (host memory)
    void const* residual;
    void const* gamma;
    void* outRes;
    void* outNorm;
    void* outQ;
    void* outSf;
    float eps;
    int rows;
    int gemma;
    // rh::RhParams
    uint64_t* rx;
    int* rxc;
    float const* bias;
    uint64_t* route;
    int* rctr;
    float* lgDbg;
    uint64_t* tdbg;
    unsigned const* tab;
    int npart;
    int repl;
    int delayNs;
    int rmaxt;
    unsigned long long deadlineNs;
    int hangDbg;
    uint8_t* x4Q;
    uint8_t* x4Sf;
    float const* x4Sfsc;
    float* x4Pts;
    // dispatch (the reference extension's template selection)
    int cl;    // 4, 8 (shipped H9_V3_CL) or 16
    int x4;    // 0 off, 1 static X4, 2 per-token X4 + s_t (pt)
    int rx2;   // PRR_RX2
    int pwait; // DEV M3LL_DEV_OPROJ_PTRIG: cl-8 post with the end-of-CTA grid wait
};

//! Launch the routing-warp post (lbv3rp_kernel) exactly as the reference extension does: grid (rows ? rows : 1) * cl
//! CTAs in clusters of cl, programmatic stream serialization (PDL) on. rows 0 launches one idle cluster.
cudaError_t launch(HoistPostParams const& params, cudaStream_t stream);

} // namespace hoistPost

} // namespace minimaxM3Sm107Tp4

} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
