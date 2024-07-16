/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuda_fp8.h>
#include <mma.h>

#include "tensorrt_llm/common/cudaBf16Fallbacks.cuh"

#include "Common.h"
#include "Poly.h"

namespace tensorrt_llm
{
namespace kernels
{

typedef void (*StatePassingKernelFuncFp16)(int B_, int L_, int H_, int P_, int N_,
    //  const half  *g_mxY_,  // B*L*H*P
    half* g_mxOs_,        // B*C*H*N*P
    half* g_mxFs_,        // B  *H*N*P
    float const* g_mxSt_, // B*C*H*N*P
                          //  const float *g_mxdc_, // B*C*H*Q
    float const* g_mxdA_, // B*C*H*Q
                          //  const half  *g_mxdt_, // B*L*(2*H*P+2*G*N+H) or B*L*(H*P+2*G*N+H)
                          //  const float *g_mxdb_, //     H
                          //  const float *g_mxA_,  //     H
                          //  const half  *g_mxCB_, // B*C*G*Q*Q
                          //  const float *g_mxD_,  //     H
                          //  const half  *g_mxXBC_,  // B*L*(H*P+2*G*N)
                          //  const half  *g_mxZ_,  // B*L*(2*H*P+2*G*N+H)
    bool removePadding_, int const* lastTokenIdsPtr_, int const* stateSlotMappingPtr_);

typedef void (*StatePassingKernelFuncBf16)(int B_, int L_, int H_, int P_, int N_,
    //  const bf16  *g_mxY_,  // B*L*H*P
    bf16* g_mxOs_,        // B*C*H*N*P
    bf16* g_mxFs_,        // B  *H*N*P
    float const* g_mxSt_, // B*C*H*N*P
                          //  const float *g_mxdc_, // B*C*H*Q
    float const* g_mxdA_, // B*C*H*Q
                          //  const bf16  *g_mxdt_, // B*L*(2*H*P+2*G*N+H) or B*L*(H*P+2*G*N+H)
                          //  const float *g_mxdb_, //     H
                          //  const float *g_mxA_,  //     H
                          //  const bf16  *g_mxCB_, // B*C*G*Q*Q
                          //  const float *g_mxD_,  //     H
                          //  const bf16  *g_mxXBC_,  // B*L*(H*P+2*G*N)
                          //  const bf16  *g_mxZ_,  // B*L*(2*H*P+2*G*N+H)
    bool removePadding_, int const* lastTokenIdsPtr_, int const* stateSlotMappingPtr_);

template <int Q_, int tileH_, int warpH_, class Tp_>
__global__ std::enable_if_t<std::is_same_v<Tp_, half> || std::is_same_v<Tp_, __nv_bfloat16>> state_passing_kernel(
    int B_, int L_, int H_, int P_, int N_,
    //  const Tp_   *g_mxY_,  // B*L*H*P
    Tp_* g_mxOs_,         // B*C*H*N*P
    Tp_* g_mxFs_,         // B  *H*N*P
    float const* g_mxSt_, // B*C*H*N*P
                          //  const float *g_mxdc_, // B*C*H*Q
    float const* g_mxdA_, // B*C*H*Q
                          //  const Tp_   *g_mxdt_, // B*L*(2*H*P+2*G*N+H) or B*L*(H*P+2*G*N+H)
                          //  const Wt_   *g_mxdb_, //     H
                          //  const Wt_   *g_mxA_,  //     H
                          //  const Tp_   *g_mxCB_, // B*C*G*Q*Q
                          //  const Wt_   *g_mxD_,  //     H
                          //  const Tp_   *g_mxXBC_,  // B*L*(H*P+2*G*N)
                          //  const Tp_   *g_mxZ_,  // B*L*(2*H*P+2*G*N+H)
    bool removePadding_, int const* lastTokenIdsPtr_, int const* stateSlotMappingPtr_)
{
    using namespace tensorrt_llm::common;

    auto blockIdx_x = Rn<ID>{int(blockIdx.x)};
    auto blockIdx_y = Rn<ID>{int(blockIdx.y)};
    auto blockIdx_z = Rn<ID>{int(blockIdx.z)};

    auto threadIdx_x = Rn<ID, 32>{int(threadIdx.x)};
    auto threadIdx_y = Rn<ID, warpH_>{int(threadIdx.y)};

    // auto B = Rn<ID>{B_};
    auto L = Rn<ID>{L_};
    auto H = Rn<ID>{H_};
    auto P = Rn<ID>{P_};
    // auto G = Rn<ID>{G_};
    auto N = Rn<ID>{N_};
    auto Q = cn<Q_>;
    auto C = Rn<ID>{div_up(L.var, Q_)};

    auto aStart = blockIdx_z * L;
    auto cStart = blockIdx_z * C;

    if (removePadding_)
    {
        aStart = Rn<ID>{int(blockIdx.z ? lastTokenIdsPtr_[blockIdx.z - 1] : 0)};
        cStart = Rn<ID>{int(blockIdx.z ? div_up(aStart.var, Q_) + blockIdx.z - 1 : 0)};
        L = Rn<ID>{lastTokenIdsPtr_[blockIdx.z] - aStart.var};
        C = Rn<ID>{div_up(L.var, Q_)};
    }
    else
    {
        L = Rn<ID>{lastTokenIdsPtr_[blockIdx.z]};
        C = Rn<ID>{div_up(L.var, Q_)};
    }

    if (stateSlotMappingPtr_)
    {
        g_mxFs_ += stateSlotMappingPtr_[blockIdx.z] * H_ * N_ * P_;
    }
    else
    {
        g_mxFs_ += blockIdx.z * H_ * N_ * P_;
    }

    auto hStart = Rn<ID>{blockIdx_x.var * tileH_ / N_ / P_};

    register Tp_ r_mxOs[tileH_ / (warpH_ * 32)] = {0};
    register float r_mxSt[tileH_ / (warpH_ * 32)] = {0};

    for (int iC = 0; iC < C.var; iC++)
    {
        if (std::is_same_v<Tp_, half>)
#pragma unroll
            for (int i = 0; i < tileH_ / (warpH_ * 32); i += 2)
                *(half2*) &r_mxOs[i] = __float22half2_rn(*(float2*) &r_mxSt[i]);
        else
#pragma unroll
            for (int i = 0; i < tileH_ / (warpH_ * 32); i += 2)
                *(bf162*) &r_mxOs[i] = __float22bfloat162_rn(*(float2*) &r_mxSt[i]);

#pragma unroll
        for (int i = 0; i < tileH_ / (warpH_ * 32); i += 2)
            *(int*) (g_mxOs_
                + get((cStart + Rn<>{iC}) * H * N * P + blockIdx_x * cn<tileH_>
                    + (threadIdx_y * cn<32> + threadIdx_x) * cn<tileH_ / (warpH_ * 32)> + Rn<UNROLL>{i}))
                = *(int*) &r_mxOs[i];

        float scale = expf(g_mxdA_[get((cStart + Rn<>{iC}) * H * Q + hStart * Q + Q - cn<1>)]);

#pragma unroll
        for (int i = 0; i < tileH_ / (warpH_ * 32); i++)
        {
            float tmp = g_mxSt_[get((cStart + Rn<>{iC}) * H * N * P + blockIdx_x * cn<tileH_>
                + (threadIdx_y * cn<32> + threadIdx_x) * cn<tileH_ / (warpH_ * 32)> + Rn<UNROLL>{i})];

            r_mxSt[i] = scale * r_mxSt[i] + tmp;
        }
    }

    if (std::is_same_v<Tp_, half>)
#pragma unroll
        for (int i = 0; i < tileH_ / (warpH_ * 32); i += 2)
            *(half2*) &r_mxOs[i] = __float22half2_rn(*(float2*) &r_mxSt[i]);
    else
#pragma unroll
        for (int i = 0; i < tileH_ / (warpH_ * 32); i += 2)
            *(bf162*) &r_mxOs[i] = __float22bfloat162_rn(*(float2*) &r_mxSt[i]);

#pragma unroll
    for (int i = 0; i < tileH_ / (warpH_ * 32); i += 8)
        *(int4*) (g_mxFs_
            + get(blockIdx_x * cn<tileH_> + (threadIdx_y * cn<32> + threadIdx_x) * cn<tileH_ / (warpH_ * 32)>
                + Rn<UNROLL>{i}))
            = *(int4*) &r_mxOs[i];
}

StatePassingKernelFuncFp16 getStatePassingKernelFp16(
    int B_, int L_, int H_, int P_, int N_, int Q_, dim3* blockDims_, dim3* threadDims_, int* sharedMem_)
{
    int B = B_;
    int L = L_;
    int H = H_;
    int P = P_;
    // int G = G_;
    int N = N_;
    int Q = Q_;
    int C = div_up(L, Q);

    int tileH = 1024;
    int warpH = 8;

    auto sharedMem = 0;

    *blockDims_ = dim3(H * N * P / tileH, 1, B);
    *threadDims_ = dim3(32, warpH);
    *sharedMem_ = sharedMem;

    if (Q_ == 128)
        return state_passing_kernel<128, 1024, 8, half>;
    else if (Q_ == 256)
        return state_passing_kernel<256, 1024, 8, half>;
    else
        return nullptr;
}

StatePassingKernelFuncBf16 getStatePassingKernelBf16(
    int B_, int L_, int H_, int P_, int N_, int Q_, dim3* blockDims_, dim3* threadDims_, int* sharedMem_)
{
    int B = B_;
    int L = L_;
    int H = H_;
    int P = P_;
    // int G = G_;
    int N = N_;
    int Q = Q_;
    int C = div_up(L, Q);

    int tileH = 1024;
    int warpH = 8;

    auto sharedMem = 0;

    *blockDims_ = dim3(H * N * P / tileH, 1, B);
    *threadDims_ = dim3(32, warpH);
    *sharedMem_ = sharedMem;

    if (Q_ == 128)
        return state_passing_kernel<128, 1024, 8, bf16>;
    else if (Q_ == 256)
        return state_passing_kernel<256, 1024, 8, bf16>;
    else
        return nullptr;
}

} // namespace kernels
} // namespace tensorrt_llm

// vim: ts=2 sw=2 sts=2 et sta
