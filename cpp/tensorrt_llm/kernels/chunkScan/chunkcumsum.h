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

typedef void (*ChunkCumsumKernelFuncFp16)(int B_, int L_, int H_,
    //  const half  *g_mxY_,  // B*L*H*P
    //  const half  *g_mxOs_, // B*C*H*N*P
    //  const half  *g_mxFs_, // B  *H*N*P
    //  const float *g_mxSt_, // B*C*H*N*P
    float* g_mxdc_,       // B*C*H*Q
    float* g_mxdA_,       // B*C*H*Q
    half const* g_mxdt_,  // B*L*H
    float const* g_mxdb_, //     H
    float const* g_mxA_,  //     H
                          //  const half  *g_mxCB_, // B*C*G*Q*Q
                          //  const half  *g_mxBC_, // B*L*2*G*N
                          //  const float *g_mxD_,  //     H
                          //  const half  *g_mxX_,  // B*L*H*P
                          //  const half  *g_mxZ_,  // B*L*H*P
    bool removePadding_, int const* lastTokenIdsPtr_);

typedef void (*ChunkCumsumKernelFuncBf16)(int B_, int L_, int H_,
    //  const bf16  *g_mxY_,  // B*L*H*P
    //  const bf16  *g_mxOs_, // B*C*H*N*P
    //  const bf16  *g_mxFs_, // B  *H*N*P
    //  const float *g_mxSt_, // B*C*H*N*P
    float* g_mxdc_,       // B*C*H*Q
    float* g_mxdA_,       // B*C*H*Q
    bf16 const* g_mxdt_,  // B*L*H
    float const* g_mxdb_, //     H
    float const* g_mxA_,  //     H
                          //  const bf16  *g_mxCB_, // B*C*G*Q*Q
                          //  const bf16  *g_mxBC_, // B*L*2*G*N
                          //  const float *g_mxD_,  //     H
                          //  const bf16  *g_mxX_,  // B*L*H*P
                          //  const bf16  *g_mxZ_,  // B*L*H*P
    bool removePadding_, int const* lastTokenIdsPtr_);

template <int Q_, int tileH_, int warpH_, bool dtSoftplus_, class Tp_, class Wt_ = float>
__global__ std::enable_if_t<std::is_same_v<Tp_, half> || std::is_same_v<Tp_, __nv_bfloat16>> chunk_cumsum_kernel(int B_,
    int L_, int H_,
    //  const Tp_   *g_mxY_,  // B*L*H*P
    //  const Tp_   *g_mxOs_, // B*C*H*N*P
    //  const Tp_   *g_mxFs_, // B  *H*N*P
    //  const float *g_mxSt_, // B*C*H*N*P
    float* g_mxdc_,     // B*C*H*Q
    float* g_mxdA_,     // B*C*H*Q
    Tp_ const* g_mxdt_, // B*L*H
    Wt_ const* g_mxdb_, //     H
    Wt_ const* g_mxA_,  //     H
                        //  const Tp_   *g_mxCB_, // B*C*G*Q*Q
                        //  const Tp_   *g_mxBC_, // B*L*2*G*N
                        //  const Wt_   *g_mxD_,  //     H
                        //  const Tp_   *g_mxX_,  // B*L*H*P
                        //  const Tp_   *g_mxZ_,  // B*L*H*P
    bool removePadding_, int const* lastTokenIdsPtr_)
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
    // auto P = Rn<ID>{P_};
    // auto G = Rn<ID>{G_};
    // auto N = Rn<ID>{N_};
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

    if (blockIdx_y * Q >= L)
        return;

    auto thread = [=](auto iStep) { return iStep * cn<warpH_ * 32> + threadIdx_y * cn<32> + threadIdx_x; };

#pragma unroll
    for (Rn<UNROLL, div_up(tileH_, warpH_ * 32)> iStep; iStep.var < iStep.size; iStep.var++)
    {
        float r_A = 0.f, r_db = 0.f, sum = 0.f;

        if (thread(iStep) < cn<tileH_>)
            r_A = g_mxA_[get(blockIdx_x * cn<tileH_> + thread(iStep))];
        if (thread(iStep) < cn<tileH_> && g_mxdb_)
            r_db = g_mxdb_[get(blockIdx_x * cn<tileH_> + thread(iStep))];

#pragma unroll
        for (Rn<UNROLL, Q_> iQ; iQ.var < iQ.size; iQ.var++)
        {
            float r_dt = 0.f;

            if (thread(iStep) < cn<tileH_> && blockIdx_y * Q + iQ < L)
            {
                r_dt = float(g_mxdt_[get((aStart + blockIdx_y * Q + iQ) * H + blockIdx_x * cn<tileH_> + thread(iStep))])
                    + r_db;

                if (dtSoftplus_)
                    r_dt = r_dt > 32.f ? r_dt : log1p(expf(r_dt));

                sum += r_dt;
            }

            if (thread(iStep) < cn<tileH_>)
            {
                g_mxdc_[get((cStart + blockIdx_y) * H * Q + (blockIdx_x * cn<tileH_> + thread(iStep)) * Q + iQ)] = r_dt;
                g_mxdA_[get((cStart + blockIdx_y) * H * Q + (blockIdx_x * cn<tileH_> + thread(iStep)) * Q + iQ)]
                    = sum * r_A;
            }
        }
    }
}

ChunkCumsumKernelFuncFp16 getChunkCumsumKernelFp16(
    int B_, int L_, int H_, int Q_, bool dtSoftPlus_, dim3* blockDims_, dim3* threadDims_, int* sharedMem_)
{
    int B = B_;
    int L = L_;
    int H = H_;
    // int P = P_;
    // int G = G_;
    // int N = N_;
    int Q = Q_;
    int C = div_up(L, Q);

    int tileH = 1;
    int warpH = 1;

    auto sharedMem = 0;

    *blockDims_ = dim3(H / tileH, C, B);
    *threadDims_ = dim3(32, warpH);
    *sharedMem_ = sharedMem;

    if (dtSoftPlus_)
    {
        if (Q_ == 128)
            return chunk_cumsum_kernel<128, 1, 1, true, half>;
        else if (Q_ == 256)
            return chunk_cumsum_kernel<256, 1, 1, true, half>;
        else
            return nullptr;
    }
    else
    {
        if (Q_ == 128)
            return chunk_cumsum_kernel<128, 1, 1, false, half>;
        else if (Q_ == 256)
            return chunk_cumsum_kernel<256, 1, 1, false, half>;
        else
            return nullptr;
    }
}

ChunkCumsumKernelFuncBf16 getChunkCumsumKernelBf16(
    int B_, int L_, int H_, int Q_, bool dtSoftPlus_, dim3* blockDims_, dim3* threadDims_, int* sharedMem_)
{
    int B = B_;
    int L = L_;
    int H = H_;
    // int P = P_;
    // int G = G_;
    // int N = N_;
    int Q = Q_;
    int C = div_up(L, Q);

    int tileH = 1;
    int warpH = 1;

    auto sharedMem = 0;

    *blockDims_ = dim3(H / tileH, C, B);
    *threadDims_ = dim3(32, warpH);
    *sharedMem_ = sharedMem;

    if (dtSoftPlus_)
    {
        if (Q_ == 128)
            return chunk_cumsum_kernel<128, 1, 1, true, bf16>;
        else if (Q_ == 256)
            return chunk_cumsum_kernel<256, 1, 1, true, bf16>;
        else
            return nullptr;
    }
    else
    {
        if (Q_ == 128)
            return chunk_cumsum_kernel<128, 1, 1, false, bf16>;
        else if (Q_ == 256)
            return chunk_cumsum_kernel<256, 1, 1, false, bf16>;
        else
            return nullptr;
    }
}

} // namespace kernels
} // namespace tensorrt_llm

// vim: ts=2 sw=2 sts=2 et sta
