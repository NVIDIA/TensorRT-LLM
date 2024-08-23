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

#include <cuda.h>
#include <cuda_fp8.h>
#include <mma.h>

#include "tensorrt_llm/common/cudaBf16Fallbacks.cuh"
#include "tensorrt_llm/common/cudaDriverWrapper.h"

#include "Common.h"
#include "CudaType.h"
#include "Poly.h"

namespace tensorrt_llm
{
namespace kernels
{

typedef void (*ChunkScanKernelFunc)(int B_, int L_, int H_, int P_, int G_, int N_,
    void* g_mxY_,        // Tp_   B*L*H*P
    void const* g_mxOs_, // Tp_   B*C*H*N*P
                         //  const void *g_mxFs_, // Tp_   B  *H*N*P
                         //  const void *g_mxSt_, // float B*C*H*N*P
    void const* g_mxdc_, // float B*C*H*Q
    void const* g_mxdA_, // float B*C*H*Q
                         //  const void *g_mxdt_, // Tp_   B*L*((g_mxZ?2:1)*H*P+2*G+H)
                         //  const void *g_mxdb_, // Wt_       H
                         //  const void *g_mxA_,  // Wt_       H
    void const* g_mxCB_, // Tp_   B*C*G*Q*Q
    void const* g_mxD_,  // Wt_       H
    void const* g_mxX_,  // Tp_   B*L*(H*P+2*G*N)
    void const* g_mxZ_,  // g_mxdt_ or nullptr
    bool removePadding_, int const* lastTokenIdsPtr_);

template <int Q_, int tileM_, int tileN_, int tileK_, // smem size, per sm
    int wmmaM_, int wmmaN_, int wmmaK_,               // wmma size, per instruction
    int warpM_, int warpN_,                           // warp number
    int pipeS_, class Tp_, class Wt_>
__global__ std::enable_if_t<std::is_same_v<Tp_, half> || std::is_same_v<Tp_, __nv_bfloat16>> chunk_scan_kernel(int B_,
    int L_, int H_, int P_, int G_, int N_,
    void* g_mxY_,        // Tp_   B*L*H*P
    void const* g_mxOs_, // Tp_   B*C*H*N*P
                         //  const void *g_mxFs_, // Tp_   B  *H*N*P
                         //  const void *g_mxSt_, // float B*C*H*N*P
    void const* g_mxdc_, // float B*C*H*Q
    void const* g_mxdA_, // float B*C*H*Q
                         //  const void *g_mxdt_, // Tp_   B*L*((g_mxZ?2:1)*H*P+2*G+H)
                         //  const void *g_mxdb_, // Wt_       H
                         //  const void *g_mxA_,  // Wt_       H
    void const* g_mxCB_, // Tp_   B*C*G*Q*Q
    void const* g_mxD_,  // Wt_       H
    void const* g_mxX_,  // Tp_   B*L*(H*P+2*G*N)
    void const* g_mxZ_,  // g_mxdt_ or nullptr
    bool removePadding_, int const* lastTokenIdsPtr_)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    using namespace tensorrt_llm::common;

    auto blockIdx_x = Rn<ID>{int(blockIdx.x)};
    auto blockIdx_y = Rn<ID>{int(blockIdx.y)};
    auto blockIdx_z = Rn<ID>{int(blockIdx.z)};

    auto threadIdx_x = Rn<ID, 32>{int(threadIdx.x)};
    auto threadIdx_y = Rn<ID, warpN_>{int(threadIdx.y)};
    auto threadIdx_z = Rn<ID, warpM_>{int(threadIdx.z)};

    // auto B = Rn<ID>{B_};
    auto L = Rn<ID>{L_};
    auto H = Rn<ID>{H_};
    auto P = Rn<ID>{P_};
    auto G = Rn<ID>{G_};
    auto N = Rn<ID>{N_};
    auto Q = cn<Q_>;
    auto C = Rn<ID>{div_up(L.var, Q_)};

    auto X_stride = Rn<ID>{H_ * P_ + 2 * G_ * N_};
    auto Z_stride = Rn<ID>{(g_mxZ_ ? 2 : 1) * H_ * P_ + 2 * G_ * N_ + H_};

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

    auto hStart = Rn<ID>{blockIdx_x.var / (P_ / cn<tileN_>) / (Q / cn<tileM_>) };
    auto mStart = Rn<ID>{blockIdx_x.var / (P_ / cn<tileN_>) % (Q / cn<tileM_>) };
    auto nStart = Rn<ID>{blockIdx_x.var % (P_ / cn<tileN_>) };
    auto gStart = Rn<ID>{hStart.var / (H_ / G_)};

    if (blockIdx_y * Q + mStart * cn<tileM_> >= L)
        return;

    Tp_* g_mxY = (Tp_*) g_mxY_ + int64_t(get(aStart + blockIdx_y * Q)) * get(H * P);
    Tp_ const* g_mxOs = (Tp_ const*) g_mxOs_ + int64_t(get(cStart + blockIdx_y)) * get(H * N * P);
    //  const Tp_   *g_mxFs = (const Tp_   *)g_mxFs_;
    //  const float *g_mxSt = (const float *)g_mxSt_;
    float const* g_mxdc = (float const*) g_mxdc_ + int64_t(get(cStart + blockIdx_y)) * get(H * Q);
    float const* g_mxdA = (float const*) g_mxdA_ + int64_t(get(cStart + blockIdx_y)) * get(H * Q);
    //  const Tp_   *g_mxdt = (const Tp_   *)g_mxdt_;
    //  const Wt_   *g_mxdb = (const Wt_   *)g_mxdb_;
    //  const Wt_   *g_mxA  = (const Wt_   *)g_mxA_;
    Tp_ const* g_mxCB = (Tp_ const*) g_mxCB_ + int64_t(get(cStart + blockIdx_y)) * get(G * Q * Q);
    Wt_ const* g_mxD = (Wt_ const*) g_mxD_;
    Tp_ const* g_mxX = (Tp_ const*) g_mxX_ + int64_t(get(aStart + blockIdx_y * Q)) * get(X_stride);
    Tp_ const* g_mxZ = (Tp_ const*) g_mxZ_ + (g_mxZ_ ? int64_t(get(aStart + blockIdx_y * Q)) * get(Z_stride) : 0);

    extern __shared__ float smem[];

    Tp_* s_mxL = (Tp_*) smem;
    Tp_* s_mxR = (Tp_*) smem + tileM_ * tileK_ * pipeS_;
    Tp_* s_mxAcc = (Tp_*) smem;

    float* s_mxdc = smem + (tileM_ + tileN_) * tileK_ * pipeS_ / 2;
    float* s_mxdA = smem + (tileM_ + tileN_) * tileK_ * pipeS_ / 2 + Q_;

    unsigned b_base = __nvvm_get_smem_pointer(smem);

    unsigned b_mxL = b_base;
    unsigned b_mxR = b_base + tileM_ * tileK_ * pipeS_ * sizeof(Tp_);
    unsigned b_mxAcc = b_base;

    using std::array;

    register array<array<array<float, wmmaM_ * wmmaN_ / 32>, tileN_ / wmmaN_ / warpN_>, tileM_ / wmmaM_ / warpM_>
        r_mxAcc
        = array<array<array<float, wmmaM_ * wmmaN_ / 32>, tileN_ / wmmaN_ / warpN_>, tileM_ / wmmaM_ / warpM_>();
    register array<array<unsigned, wmmaM_ * wmmaK_ / 64>, tileM_ / wmmaM_ / warpM_> r_mxL;
    register array<array<unsigned, wmmaK_ * wmmaN_ / 64>, tileN_ / wmmaN_ / warpN_> r_mxR;

    constexpr int step = std::max(
        1, tileM_ / wmmaM_ / warpM_ * tileN_ / wmmaN_ / warpN_ / (tileM_ / wmmaM_ / warpM_ + tileN_ / wmmaN_ / warpN_));

    auto baseL = [](auto iK) { return iK % cn<pipeS_> * cn<tileM_> * cn<tileK_>; };
    auto baseR = [](auto iK) { return iK % cn<pipeS_> * cn<tileN_> * cn<tileK_>; };

    auto thread = [=](auto iStep)
    {
        return iStep * cn<warpM_ * warpN_ * 256> + threadIdx_z * cn<warpN_ * 256> + threadIdx_y * cn<256>
            + threadIdx_x * cn<8>;
    };

#pragma unroll
    for (Rn<UNROLL, div_up(Q_, warpM_ * warpN_ * 256)> iStep; iStep.var < iStep.size; iStep.var++)
        if (thread(iStep) < cn<Q_>)
        {
#pragma unroll
            for (int i = 0; i < 8; i += 4)
            {
                // dc should be set to zero out of seq length, but this is done
                // by chunkcumsum, so no need here.
                *(int4*) (s_mxdc + get(thread(iStep)) + i) = *(int4*) (g_mxdc + get(hStart * Q + thread(iStep)) + i);
                *(int4*) (s_mxdA + get(thread(iStep)) + i) = *(int4*) (g_mxdA + get(hStart * Q + thread(iStep)) + i);
            }
        }

#pragma unroll
    for (Rn<UNROLL, pipeS_> iK; iK.var < iK.size; iK.var++)
    {
#pragma unroll
        for (Rn<UNROLL, div_up(tileM_ * tileK_, warpM_ * warpN_ * 256)> iStep; iStep.var < iStep.size; iStep.var++)
            if (thread(iStep) < cn<tileM_ * tileK_>
                && thread(iStep) / cn<tileK_> < L - blockIdx_y * Q - mStart * cn<tileM_>)
                cp_shared_global<16>(b_mxL + swizzle<tileK_ * 2, tileK_ * 2>(thread(iStep) * cn<2>, baseL(iK) * cn<2>),
                    g_mxX
                        + get((mStart * cn<tileM_> + thread(iStep) / cn<tileK_>) *X_stride + H * P + cn<1> * G * N
                            + gStart * N + iK * cn<tileK_> + thread(iStep) % cn<tileK_>));
            else if (thread(iStep) < cn<tileM_ * tileK_>)
                *(int4*) ((char*) s_mxL + swizzle<tileK_ * 2, tileK_ * 2>(thread(iStep) * cn<2>, baseL(iK) * cn<2>))
                    = int4{0, 0, 0, 0};

#pragma unroll
        for (Rn<UNROLL, div_up(tileN_ * tileK_, warpM_ * warpN_ * 256)> iStep; iStep.var < iStep.size; iStep.var++)
            if (thread(iStep) < cn<tileN_ * tileK_>)
                cp_shared_global<16>(b_mxR + swizzle<tileN_ * 2, tileN_ * 2>(thread(iStep) * cn<2>, baseR(iK) * cn<2>),
                    g_mxOs
                        + get(hStart * N * P + (iK * cn<tileK_> + thread(iStep) / cn<tileN_>) *P + nStart * cn<tileN_>
                            + thread(iStep) % cn<tileN_>));

        cp_commit_group();
    }

    cp_wait_group<pipeS_ - 1>();

    __syncthreads();

    int Q1 = std::min(Q_, (mStart.var + 1) * tileM_);

    for (int iK = pipeS_; iK < (N_ + Q1) / tileK_ + pipeS_; iK++)
    {
        auto jK = Rn<>{iK};
        if ((iK - pipeS_) * cn<tileK_> == N_)
        {

#pragma unroll
            for (int y = 0; y < tileM_ / wmmaM_ / warpM_; y++)
#pragma unroll
                for (int x = 0; x < tileN_ / wmmaN_ / warpN_; x++)
                {
                    float2 tmp2 = float2{expf(s_mxdA[get(mStart * cn<tileM_> + Rn<UNROLL>{y} * cn<warpM_ * wmmaM_>
                                             + threadIdx_z * cn<wmmaM_> + threadIdx_x / cn<4>)]),
                        expf(s_mxdA[get(mStart * cn<tileM_> + Rn<UNROLL>{y} * cn<warpM_ * wmmaM_> + cn<8>
                            + threadIdx_z * cn<wmmaM_> + threadIdx_x / cn<4>)])};

                    r_mxAcc[y][x][0] *= tmp2.x;
                    r_mxAcc[y][x][1] *= tmp2.x;
                    r_mxAcc[y][x][2] *= tmp2.y;
                    r_mxAcc[y][x][3] *= tmp2.y;
                }
        }

        if ((iK - pipeS_) * cn<tileK_> >= N_)
        {

#pragma unroll
            for (Rn<UNROLL, div_up(tileM_ * tileK_, warpM_ * warpN_ * 256)> iStep; iStep.var < iStep.size; iStep.var++)
                if (thread(iStep) < cn<tileM_ * tileK_>)
                {
                    register Tp_ tmpL[8];

                    *(int4*) &tmpL[0] = *(int4*) ((char*) s_mxL
                        + swizzle<tileK_ * 2, tileK_ * 2>(thread(iStep) * cn<2>, baseL(jK) * cn<2>));

#pragma unroll
                    for (int i = 0; i < 8; i += 2)
                    {
                        float2 tmp2 = std::is_same_v<Tp_, half> ? __half22float2(*(half2*) &tmpL[i])
                                                                : bf1622float2(*(bf162*) &tmpL[i]);

                        int kStart = (iK - pipeS_) * cn<tileK_> - N_;

                        // dc should be set to zero out of seq length, but this is done
                        // by chunkcumsum, so no need here.
                        tmp2.x *= expf(s_mxdA[get(mStart * cn<tileM_> + thread(iStep) / cn<tileK_>)]
                                      - s_mxdA[kStart + get(thread(iStep) % cn<tileK_> + Rn<UNROLL>{i})])
                            * s_mxdc[kStart + get(thread(iStep) % cn<tileK_> + Rn<UNROLL>{i})];
                        tmp2.y *= expf(s_mxdA[get(mStart * cn<tileM_> + thread(iStep) / cn<tileK_>)]
                                      - s_mxdA[kStart + get(thread(iStep) % cn<tileK_> + Rn<UNROLL>{i + 1})])
                            * s_mxdc[kStart + get(thread(iStep) % cn<tileK_> + Rn<UNROLL>{i + 1})];

                        if (get(mStart * cn<tileM_> + thread(iStep) / cn<tileK_>)
                            < kStart + get(thread(iStep) % cn<tileK_> + Rn<UNROLL>{i}))
                            tmp2.x = 0;
                        if (get(mStart * cn<tileM_> + thread(iStep) / cn<tileK_>)
                            < kStart + get(thread(iStep) % cn<tileK_> + Rn<UNROLL>{i + 1}))
                            tmp2.y = 0;

                        if (std::is_same_v<Tp_, half>)
                            *(half2*) &tmpL[i] = __float22half2_rn(tmp2);
                        else
                            *(bf162*) &tmpL[i] = __float22bfloat162_rn(tmp2);
                    }

                    *(int4*) ((char*) s_mxL + swizzle<tileK_ * 2, tileK_ * 2>(thread(iStep) * cn<2>, baseL(jK) * cn<2>))
                        = *(int4*) &tmpL[0];
                }

            __syncthreads();
        }

#pragma unroll
        for (int k = 0; k < tileK_ / wmmaK_; k++)
        {
#pragma unroll
            for (int y = 0; y < tileM_ / wmmaM_ / warpM_; y++)
#pragma unroll
                for (int x = 0; x < tileN_ / wmmaN_ / warpN_; x++)
                {
                    if ((y * tileN_ / wmmaN_ / warpN_ + x) % step == 0)
                    {
                        int x1 = (y * tileN_ / wmmaN_ / warpN_ + x) / step;
                        int y1 = x1 - tileN_ / wmmaN_ / warpN_
                            + (tileM_ / wmmaM_ / warpM_ == 1 || tileN_ / wmmaN_ / warpN_ == 1);

                        if (y1 >= 0 && y1 < tileM_ / wmmaM_ / warpM_)
                        {
                            if (wmmaK_ == 16)
                                ldmatrix_x4<>(r_mxL[y1][0], r_mxL[y1][1], r_mxL[y1][2], r_mxL[y1][3],
                                    b_mxL + iK % pipeS_ * (tileM_ * tileK_ * 2)
                                        + 2
                                            * swz<tileK_ * 2, tileK_>(y1 * warpM_ * wmmaM_ * tileK_ + k * wmmaK_
                                                + threadIdx.z * wmmaM_ * tileK_ + threadIdx.x % 16 * tileK_
                                                + threadIdx.x / 16 * 8));
                        }

                        if (x1 >= 0 && x1 < tileN_ / wmmaN_ / warpN_)
                        {
                            if (wmmaK_ == 16 && x1 % 2 == 0)
                                ldmatrix_x4</* trans_ = */ true>(r_mxR[x1][0], r_mxR[x1][1], r_mxR[x1 + 1][0],
                                    r_mxR[x1 + 1][1],
                                    b_mxR + iK % pipeS_ * (tileK_ * tileN_ * 2)
                                        + 2
                                            * swz<tileN_ * 2, tileN_>(x1 * warpN_ * wmmaN_ + k * wmmaK_ * tileN_
                                                + threadIdx.y * wmmaN_ + threadIdx.x % wmmaK_ * tileN_
                                                + threadIdx.x / wmmaK_ * warpN_ * wmmaN_));
                        }
                    }
                }

#pragma unroll
            for (int y = 0; y < tileM_ / wmmaM_ / warpM_; y++)
#pragma unroll
                for (int x = 0; x < tileN_ / wmmaN_ / warpN_; x++)
                {
                    mma<Tp_>(r_mxAcc[y][x], r_mxL[y], r_mxR[x]);
                }
        }

        __syncthreads();

        if (iK * tileK_ < N_)
        {

#pragma unroll
            for (Rn<UNROLL, div_up(tileM_ * tileK_, warpM_ * warpN_ * 256)> iStep; iStep.var < iStep.size; iStep.var++)
                if (thread(iStep) < cn<tileM_ * tileK_>
                    && thread(iStep) / cn<tileK_> < L - blockIdx_y * Q - mStart * cn<tileM_>)
                    cp_shared_global<16>(
                        b_mxL + swizzle<tileK_ * 2, tileK_ * 2>(thread(iStep) * cn<2>, baseL(jK) * cn<2>),
                        g_mxX
                            + get((mStart * cn<tileM_> + thread(iStep) / cn<tileK_>) *X_stride + H * P + cn<1> * G * N
                                + gStart * N + jK * cn<tileK_> + thread(iStep) % cn<tileK_>));
                else if (thread(iStep) < cn<tileM_ * tileK_>)
                    *(int4*) ((char*) s_mxL + swizzle<tileK_ * 2, tileK_ * 2>(thread(iStep) * cn<2>, baseL(jK) * cn<2>))
                        = int4{0, 0, 0, 0};

#pragma unroll
            for (Rn<UNROLL, div_up(tileN_ * tileK_, warpM_ * warpN_ * 256)> iStep; iStep.var < iStep.size; iStep.var++)
                if (thread(iStep) < cn<tileN_ * tileK_>)
                    cp_shared_global<16>(
                        b_mxR + swizzle<tileN_ * 2, tileN_ * 2>(thread(iStep) * cn<2>, baseR(jK) * cn<2>),
                        g_mxOs
                            + get(hStart * N * P + (jK * cn<tileK_> + thread(iStep) / cn<tileN_>) *P
                                + nStart * cn<tileN_> + thread(iStep) % cn<tileN_>));
        }
        else if (iK * tileK_ < N_ + Q1)
        {

#pragma unroll
            for (Rn<UNROLL, div_up(tileM_ * tileK_, warpM_ * warpN_ * 256)> iStep; iStep.var < iStep.size; iStep.var++)
                if (thread(iStep) < cn<tileM_ * tileK_>)
                    cp_shared_global<16>(
                        b_mxL + swizzle<tileK_ * 2, tileK_ * 2>(thread(iStep) * cn<2>, baseL(jK) * cn<2>),
                        g_mxCB
                            + get(gStart * Q * Q + (mStart * cn<tileM_> + thread(iStep) / cn<tileK_>) *Q
                                + jK * cn<tileK_> - N + thread(iStep) % cn<tileK_>));

#pragma unroll
            for (Rn<UNROLL, div_up(tileN_ * tileK_, warpM_ * warpN_ * 256)> iStep; iStep.var < iStep.size; iStep.var++)
                if (thread(iStep) < cn<tileN_ * tileK_>
                    && thread(iStep) / cn<tileN_> < L - blockIdx_y * Q - jK * cn<tileK_> + N)
                    cp_shared_global<16>(
                        b_mxR + swizzle<tileN_ * 2, tileN_ * 2>(thread(iStep) * cn<2>, baseR(jK) * cn<2>),
                        g_mxX
                            + get((jK * cn<tileK_> - N + thread(iStep) / cn<tileN_>) *X_stride + hStart * P
                                + nStart * cn<tileN_> + thread(iStep) % cn<tileN_>));
                else if (thread(iStep) < cn<tileN_ * tileK_>)
                    *(int4*) ((char*) s_mxR + swizzle<tileN_ * 2, tileN_ * 2>(thread(iStep) * cn<2>, baseR(jK) * cn<2>))
                        = int4{0, 0, 0, 0};
        }

        cp_commit_group();
        cp_wait_group<pipeS_ - 1>();

        __syncthreads();
    }

    cp_wait_group<0>();

    if (g_mxD)
    {
        float r_D = g_mxD[hStart.var];

#pragma unroll
        for (int y = 0; y < tileM_ / wmmaM_ / warpM_; y++)
#pragma unroll
            for (int x = 0; x < tileN_ / wmmaN_ / warpN_; x++)
            {
                Tp_ tmp16[4] = {0};
                float tmp32[4] = {0};

                if (blockIdx_y * Q + mStart * cn<tileM_> + Rn<UNROLL>{y} * cn<warpM_ * wmmaM_>
                        + threadIdx_z * cn<wmmaM_> + threadIdx_x / cn<4>
                    < L)
                {
                    *(int*) &tmp16[0] = *(int*) (g_mxX
                        + get((mStart * cn<tileM_> + Rn<UNROLL>{y} * cn<warpM_ * wmmaM_> + threadIdx_z * cn<wmmaM_>
                                  + threadIdx_x / cn<4>) *X_stride
                            + hStart * P + nStart * cn<tileN_> + Rn<UNROLL>{x} * cn<warpN_ * wmmaN_>
                            + threadIdx_y * cn<wmmaN_> + threadIdx_x % cn<4> * cn<2>));

                    *(float2*) &tmp32[0] = std::is_same_v<Tp_, half> ? __half22float2(*(half2*) &tmp16[0])
                                                                     : bf1622float2(*(bf162*) &tmp16[0]);

                    r_mxAcc[y][x][0] += r_D * tmp32[0];
                    r_mxAcc[y][x][1] += r_D * tmp32[1];
                }

                if (blockIdx_y * Q + mStart * cn<tileM_> + Rn<UNROLL>{y} * cn<warpM_ * wmmaM_> + cn<8>
                        + threadIdx_z * cn<wmmaM_> + threadIdx_x / cn<4>
                    < L)
                {
                    *(int*) &tmp16[2] = *(int*) (g_mxX
                        + get((mStart * cn<tileM_> + Rn<UNROLL>{y} * cn<warpM_ * wmmaM_> + cn<8>
                                  + threadIdx_z * cn<wmmaM_> + threadIdx_x / cn<4>) *X_stride
                            + hStart * P + nStart * cn<tileN_> + Rn<UNROLL>{x} * cn<warpN_ * wmmaN_>
                            + threadIdx_y * cn<wmmaN_> + threadIdx_x % cn<4> * cn<2>));

                    *(float2*) &tmp32[2] = std::is_same_v<Tp_, half> ? __half22float2(*(half2*) &tmp16[2])
                                                                     : bf1622float2(*(bf162*) &tmp16[2]);

                    r_mxAcc[y][x][2] += r_D * tmp32[2];
                    r_mxAcc[y][x][3] += r_D * tmp32[3];
                }
            }
    }

    if (g_mxZ)
    {
#pragma unroll
        for (int y = 0; y < tileM_ / wmmaM_ / warpM_; y++)
#pragma unroll
            for (int x = 0; x < tileN_ / wmmaN_ / warpN_; x++)
            {
                Tp_ tmp16[4] = {0};
                float tmp32[4] = {0};

                if (blockIdx_y * Q + mStart * cn<tileM_> + Rn<UNROLL>{y} * cn<warpM_ * wmmaM_>
                        + threadIdx_z * cn<wmmaM_> + threadIdx_x / cn<4>
                    < L)
                {
                    *(int*) &tmp16[0] = *(int*) (g_mxZ
                        + get((mStart * cn<tileM_> + Rn<UNROLL>{y} * cn<warpM_ * wmmaM_> + threadIdx_z * cn<wmmaM_>
                                  + threadIdx_x / cn<4>) *Z_stride
                            + hStart * P + nStart * cn<tileN_> + Rn<UNROLL>{x} * cn<warpN_ * wmmaN_>
                            + threadIdx_y * cn<wmmaN_> + threadIdx_x % cn<4> * cn<2>));

                    *(float2*) &tmp32[0] = std::is_same_v<Tp_, half> ? __half22float2(*(half2*) &tmp16[0])
                                                                     : bf1622float2(*(bf162*) &tmp16[0]);

                    r_mxAcc[y][x][0] *= tmp32[0] > 32.f ? tmp32[0] : tmp32[0] / (1.f + expf(-tmp32[0]));
                    r_mxAcc[y][x][1] *= tmp32[1] > 32.f ? tmp32[1] : tmp32[1] / (1.f + expf(-tmp32[1]));
                }

                if (blockIdx_y * Q + mStart * cn<tileM_> + Rn<UNROLL>{y} * cn<warpM_ * wmmaM_> + cn<8>
                        + threadIdx_z * cn<wmmaM_> + threadIdx_x / cn<4>
                    < L)
                {
                    *(int*) &tmp16[2] = *(int*) (g_mxZ
                        + get((mStart * cn<tileM_> + Rn<UNROLL>{y} * cn<warpM_ * wmmaM_> + cn<8>
                                  + threadIdx_z * cn<wmmaM_> + threadIdx_x / cn<4>) *Z_stride
                            + hStart * P + nStart * cn<tileN_> + Rn<UNROLL>{x} * cn<warpN_ * wmmaN_>
                            + threadIdx_y * cn<wmmaN_> + threadIdx_x % cn<4> * cn<2>));

                    *(float2*) &tmp32[2] = std::is_same_v<Tp_, half> ? __half22float2(*(half2*) &tmp16[2])
                                                                     : bf1622float2(*(bf162*) &tmp16[2]);

                    r_mxAcc[y][x][2] *= tmp32[2] > 32.f ? tmp32[2] : tmp32[2] / (1.f + expf(-tmp32[2]));
                    r_mxAcc[y][x][3] *= tmp32[3] > 32.f ? tmp32[3] : tmp32[3] / (1.f + expf(-tmp32[3]));
                }
            }
    }

#pragma unroll
    for (int y = 0; y < tileM_ / wmmaM_ / warpM_; y++)
#pragma unroll
        for (int x = 0; x < tileN_ / wmmaN_ / warpN_; x++)
        {
            if (std::is_same_v<Tp_, half>)
            {
                *(half2*) &r_mxAcc[y][x][0] = __floats2half2_rn(r_mxAcc[y][x][0], r_mxAcc[y][x][1]);
                *(half2*) &r_mxAcc[y][x][2] = __floats2half2_rn(r_mxAcc[y][x][2], r_mxAcc[y][x][3]);
            }
            else
            {
                *(bf162*) &r_mxAcc[y][x][0] = __floats2bfloat162_rn(r_mxAcc[y][x][0], r_mxAcc[y][x][1]);
                *(bf162*) &r_mxAcc[y][x][2] = __floats2bfloat162_rn(r_mxAcc[y][x][2], r_mxAcc[y][x][3]);
            }
        }

#pragma unroll
    for (int y = 0; y < tileM_ / wmmaM_ / warpM_; y++)
#pragma unroll
        for (int x = 0; x < tileN_ / wmmaN_ / warpN_; x++)
        {
            asm volatile("st.shared.b32 [%0], %1;\n" ::"r"(b_mxAcc
                             + 2
                                 * swz<tileN_ * 2, tileN_>(y * warpM_ * wmmaM_ * tileN_ + x * warpN_ * wmmaN_
                                     + (threadIdx.z * wmmaM_ + threadIdx.x / 4) * tileN_
                                     + (threadIdx.y * wmmaN_ + threadIdx.x % 4 * 2))),
                "r"(*(unsigned*) &r_mxAcc[y][x][0]));
            asm volatile("st.shared.b32 [%0], %1;\n" ::"r"(b_mxAcc
                             + 2
                                 * swz<tileN_ * 2, tileN_>(y * warpM_ * wmmaM_ * tileN_ + 8 * tileN_
                                     + x * warpN_ * wmmaN_ + (threadIdx.z * wmmaM_ + threadIdx.x / 4) * tileN_
                                     + (threadIdx.y * wmmaN_ + threadIdx.x % 4 * 2))),
                "r"(*(unsigned*) &r_mxAcc[y][x][2]));
        }

    __syncthreads();

#pragma unroll
    for (Rn<UNROLL, div_up(tileM_ * tileN_, warpM_ * warpN_ * 256)> iStep; iStep.var < iStep.size; iStep.var++)
        if (thread(iStep) < cn<tileM_ * tileN_>
            && thread(iStep) / cn<tileN_> < L - blockIdx_y * Q - mStart * cn<tileM_>)
            *(int4*) (g_mxY
                + get((mStart * cn<tileM_> + thread(iStep) / cn<tileN_>) *H * P + hStart * P + nStart * cn<tileN_>
                    + thread(iStep) % cn<tileN_>))
                = *(int4*) ((char*) s_mxAcc + swizzle<tileN_ * 2, tileN_ * 2>(thread(iStep) * cn<2>));
#endif
}

template <int Q_, int tileM_, int tileN_, int tileK_, // smem size, per sm
    int wmmaM_, int wmmaN_, int wmmaK_,               // wmma size, per instruction
    int warpM_, int warpN_,                           // warp number
    int pipeS_, class Tp_, class Wt_>
__global__ std::enable_if_t<std::is_same_v<Tp_, half> || std::is_same_v<Tp_, __nv_bfloat16>> chunk_scan_hopper(int B_,
    int L_, int H_, int P_, int G_, int N_,
    void* g_mxY_,        // Tp_   B*L*H*P
    void const* g_mxOs_, // Tp_   B*C*H*N*P
                         //  const void *g_mxFs_, // Tp_   B  *H*N*P
                         //  const void *g_mxSt_, // float B*C*H*N*P
    void const* g_mxdc_, // float B*C*H*Q
    void const* g_mxdA_, // float B*C*H*Q
                         //  const void *g_mxdt_, // Tp_   B*L*((g_mxZ?2:1)*H*P+2*G+H)
                         //  const void *g_mxdb_, // Wt_       H
                         //  const void *g_mxA_,  // Wt_       H
    void const* g_mxCB_, // Tp_   B*C*G*Q*Q
    void const* g_mxD_,  // Wt_       H
    void const* g_mxX_,  // Tp_   B*L*(H*P+2*G*N)
    void const* g_mxZ_,  // g_mxdt_ or nullptr
    bool removePadding_, int const* lastTokenIdsPtr_)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)
    using namespace tensorrt_llm::common;

    auto blockIdx_x = Rn<ID>{int(blockIdx.x)};
    auto blockIdx_y = Rn<ID>{int(blockIdx.y)};
    auto blockIdx_z = Rn<ID>{int(blockIdx.z)};

    auto threadIdx_x = Rn<ID, 32>{int(threadIdx.x)};
    auto threadIdx_y = Rn<ID, warpM_>{int(threadIdx.y)};
    auto threadIdx_z = Rn<ID, warpN_>{int(threadIdx.z)};

    // auto B = Rn<ID>{B_};
    auto L = Rn<ID>{L_};
    auto H = Rn<ID>{H_};
    auto P = Rn<ID>{P_};
    auto G = Rn<ID>{G_};
    auto N = Rn<ID>{N_};
    auto Q = cn<Q_>;
    auto C = Rn<ID>{div_up(L.var, Q_)};

    auto X_stride = Rn<ID>{H_ * P_ + 2 * G_ * N_};
    auto Z_stride = Rn<ID>{(g_mxZ_ ? 2 : 1) * H_ * P_ + 2 * G_ * N_ + H_};

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

    auto hStart = Rn<ID>{blockIdx_x.var / (P_ / cn<tileN_>) / (Q / cn<tileM_>) };
    auto mStart = Rn<ID>{blockIdx_x.var / (P_ / cn<tileN_>) % (Q / cn<tileM_>) };
    auto nStart = Rn<ID>{blockIdx_x.var % (P_ / cn<tileN_>) };
    auto gStart = Rn<ID>{hStart.var / (H_ / G_)};

    if (blockIdx_y * Q + mStart * cn<tileM_> >= L)
        return;

    Tp_* g_mxY = (Tp_*) g_mxY_ + int64_t(get(aStart + blockIdx_y * Q)) * get(H * P);
    //  const Tp_   *g_mxOs = (const Tp_   *)g_mxOs_;
    //  const Tp_   *g_mxFs = (const Tp_   *)g_mxFs_;
    //  const float *g_mxSt = (const float *)g_mxSt_;
    float const* g_mxdc = (float const*) g_mxdc_ + int64_t(get(cStart + blockIdx_y)) * get(H * Q);
    float const* g_mxdA = (float const*) g_mxdA_ + int64_t(get(cStart + blockIdx_y)) * get(H * Q);
    //  const Tp_   *g_mxdt = (const Tp_   *)g_mxdt_;
    //  const Wt_   *g_mxdb = (const Wt_   *)g_mxdb_;
    //  const Wt_   *g_mxA  = (const Wt_   *)g_mxA_;
    //  const Tp_   *g_mxCB = (const Tp_   *)g_mxCB_;
    Wt_ const* g_mxD = (Wt_ const*) g_mxD_;
    Tp_ const* g_mxX = (Tp_ const*) (*(uint64_t const*) ((CUtensorMap const*) g_mxX_ + 3))
        + int64_t(get(aStart + blockIdx_y * Q)) * get(X_stride);
    Tp_ const* g_mxZ = (Tp_ const*) g_mxZ_ + (g_mxZ_ ? int64_t(get(aStart + blockIdx_y * Q)) * get(Z_stride) : 0);

    extern __shared__ float smem[];

    Tp_* s_mxL = (Tp_*) smem + 512;
    Tp_* s_mxR = (Tp_*) smem + 512 + tileM_ * tileK_ * pipeS_;
    Tp_* s_mxAcc = (Tp_*) smem + 512;

    float* s_mxdc = smem + 256 + (tileM_ + tileN_) * tileK_ * pipeS_ / 2;
    float* s_mxdA = smem + 256 + (tileM_ + tileN_) * tileK_ * pipeS_ / 2 + Q_;

    unsigned b_base = __nvvm_get_smem_pointer(smem);

    unsigned b_mxL = b_base + 1024;
    unsigned b_mxR = b_base + 1024 + tileM_ * tileK_ * pipeS_ * sizeof(Tp_);
    unsigned b_mxAcc = b_base + 1024;
    unsigned b_mbar = b_base;

    uint32_t formatL = tileK_ >= 64 ? 1 : (tileK_ == 32 ? 2 : 3);
    uint32_t formatR = 1;
    uint32_t baseOffsetL = 0;
    uint32_t baseOffsetR = 0;
    uint32_t strideDimL = tileK_ >= 64 ? warpM_ / 4 * 64 : warpM_ / 4 * tileK_;
    uint32_t strideDimR = 64;
    uint32_t leadingDimL = 0;
    uint32_t leadingDimR = tileK_ * 8;
    uint32_t startAddrL = (b_mxL + threadIdx.y / 4 * 8 * tileK_ * 2) / 16;
    uint32_t startAddrR
        = (b_mxR + threadIdx.z * tileN_ / warpN_ % 64 * 2 + threadIdx.z * tileN_ / warpN_ / 64 * tileK_ * 128) / 16;

    uint64_t d_mxL = ((uint64_t) formatL << 62) + ((uint64_t) baseOffsetL << 49) + ((uint64_t) strideDimL << 32)
        + ((uint64_t) leadingDimL << 16) + ((uint64_t) startAddrL);
    uint64_t d_mxR = ((uint64_t) formatR << 62) + ((uint64_t) baseOffsetR << 49) + ((uint64_t) strideDimR << 32)
        + ((uint64_t) leadingDimR << 16) + ((uint64_t) startAddrR);

    using std::array;

    register array<array<array<float, wmmaM_ * wmmaN_ / 32>, tileN_ / wmmaN_ / warpN_>, tileM_ / wmmaM_ / warpM_>
        r_mxAcc
        = array<array<array<float, wmmaM_ * wmmaN_ / 32>, tileN_ / wmmaN_ / warpN_>, tileM_ / wmmaM_ / warpM_>();

    auto baseL = [](auto iK) { return iK % cn<pipeS_> * cn<tileM_> * cn<tileK_>; };
    auto baseR = [](auto iK) { return iK % cn<pipeS_> * cn<tileN_> * cn<tileK_>; };

    auto thread = [=](auto iStep)
    {
        return iStep * cn<warpM_ * warpN_ * 256> + threadIdx_y * cn<warpN_ * 256> + threadIdx_z * cn<256>
            + threadIdx_x * cn<8>;
    };

    asm volatile(
        ".reg .pred elected_one;\n"
        ".reg .pred done;\n"
        "elect.sync _|elected_one, 0xFFFFFFFF;\n"
        "@!elected_one setp.eq.b32 done, 0, 0;\n" ::);

    if (threadIdx.y == 0 && threadIdx.z == 0)
    {
#pragma unroll
        for (Rn<UNROLL, pipeS_> iK; iK.var < iK.size; iK.var++)
            asm volatile("@elected_one mbarrier.init.shared.b64 [%0], %1;\n" ::"r"(b_mbar + iK.var * 8), "r"(1));
    }

    __syncthreads();

#pragma unroll
    for (Rn<UNROLL, div_up(Q_, warpM_ * warpN_ * 256)> iStep; iStep.var < iStep.size; iStep.var++)
        if (thread(iStep) < cn<Q_>)
        {
#pragma unroll
            for (int i = 0; i < 8; i += 4)
            {
                // dc should be set to zero out of seq length, but this is done
                // by chunkcumsum, so no need here.
                *(int4*) (s_mxdc + get(thread(iStep)) + i) = *(int4*) (g_mxdc + get(hStart * Q + thread(iStep)) + i);
                *(int4*) (s_mxdA + get(thread(iStep)) + i) = *(int4*) (g_mxdA + get(hStart * Q + thread(iStep)) + i);
            }
        }

#pragma unroll
    for (Rn<UNROLL, pipeS_> iK; iK.var < iK.size; iK.var++)
    {
        if (threadIdx.y == 0 && threadIdx.z == 0)
        {
            asm volatile("@elected_one mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n" ::"r"(b_mbar + iK.var * 8),
                "r"((tileM_ + tileN_) * tileK_ * 2));
            asm volatile(
                "@elected_one cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes"
                " [%0], [%1, {%2, %3, %4}], [%5];\n" ::"r"(b_mxL + iK.var % pipeS_ * (tileM_ * tileK_ * 2)),
                "l"((CUtensorMap const*) g_mxX_), "r"(iK.var * tileK_), "r"(gStart.var),
                "r"(get(aStart + blockIdx_y * Q + mStart * cn<tileM_>)), "r"(b_mbar + iK.var * 8));
#pragma unroll
            for (int iN = 0; iN < tileN_; iN += 64)
                asm volatile(
                    "@elected_one cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1, {%2, %3, %4, %5}], [%6];\n" ::"r"(
                        b_mxR + iK.var % pipeS_ * (tileN_ * tileK_ * 2) + iN * tileK_ * 2),
                    "l"((CUtensorMap const*) g_mxX_ + 1), "r"(get(nStart * cn<tileN_>) + iN), "r"(iK.var * tileK_),
                    "r"(hStart.var), "r"(get(cStart + blockIdx_y)), "r"(b_mbar + iK.var * 8));
        }
    }

    __syncthreads();

    int Q1 = std::min(Q_, (mStart.var + 1) * tileM_);

    for (int iK = pipeS_; iK < (N_ + Q1) / tileK_ + pipeS_; iK++)
    {
        uint64_t d_mxL0 = d_mxL + iK % pipeS_ * (tileM_ * tileK_ * 2) / 16;
        uint64_t d_mxR0 = d_mxR + iK % pipeS_ * (tileN_ * tileK_ * 2) / 16;

        asm volatile(
            "{\n"
            "TRY_AGAIN:\n"
            "mbarrier.try_wait.parity.shared.b64 done, [%0], %1;\n"
            "@!done nanosleep.u32 %2;\n"
            "@!done bra TRY_AGAIN;\n"
            "}\n" ::"r"(b_mbar + iK % pipeS_ * 8),
            "r"(1 - iK / pipeS_ % 2), "r"(threadIdx.x % 6));

        auto jK = Rn<>{iK};
        if ((iK - pipeS_) * cn<tileK_> == N_)
        {

#pragma unroll
            for (int y = 0; y < tileM_ / wmmaM_ / warpM_; y++)
#pragma unroll
                for (int x = 0; x < tileN_ / wmmaN_ / warpN_; x++)
                {
                    float2 tmp2 = float2{expf(s_mxdA[get(mStart * cn<tileM_> + Rn<UNROLL>{y} * cn<warpM_ * wmmaM_>
                                             + threadIdx_y / cn<4> * cn<8>
                                             + threadIdx_y % cn<4> * cn<warpM_ / 4 * wmmaM_> + threadIdx_x / cn<4>)]),
                        expf(s_mxdA[get(mStart * cn<tileM_> + Rn<UNROLL>{y} * cn<warpM_ * wmmaM_> + cn<warpM_ / 4 * 8>
                            + threadIdx_y / cn<4> * cn<8> + threadIdx_y % cn<4> * cn<warpM_ / 4 * wmmaM_>
                            + threadIdx_x / cn<4>)])};

                    r_mxAcc[y][x][0] *= tmp2.x;
                    r_mxAcc[y][x][1] *= tmp2.x;
                    r_mxAcc[y][x][2] *= tmp2.y;
                    r_mxAcc[y][x][3] *= tmp2.y;
                }
        }

        if ((iK - pipeS_) * cn<tileK_> >= N_)
        {

#pragma unroll
            for (Rn<UNROLL, div_up(tileM_ * tileK_, warpM_ * warpN_ * 256)> iStep; iStep.var < iStep.size; iStep.var++)
                if (thread(iStep) < cn<tileM_ * tileK_>)
                {
                    register Tp_ tmpL[8];

                    *(int4*) &tmpL[0]
                        = *(int4*) ((char*) s_mxL + swizzle<tileK_ * 2, 128>(thread(iStep) * cn<2>, baseL(jK) * cn<2>));

#pragma unroll
                    for (int i = 0; i < 8; i += 2)
                    {
                        float2 tmp2 = std::is_same_v<Tp_, half> ? __half22float2(*(half2*) &tmpL[i])
                                                                : bf1622float2(*(bf162*) &tmpL[i]);

                        int kStart = (iK - pipeS_) * cn<tileK_> - N_;

                        // dc should be set to zero out of seq length, but this is done
                        // by chunkcumsum, so no need here.
                        tmp2.x *= expf(s_mxdA[get(mStart * cn<tileM_> + thread(iStep) / cn<tileK_>)]
                                      - s_mxdA[kStart + get(thread(iStep) % cn<tileK_> + Rn<UNROLL>{i})])
                            * s_mxdc[kStart + get(thread(iStep) % cn<tileK_> + Rn<UNROLL>{i})];
                        tmp2.y *= expf(s_mxdA[get(mStart * cn<tileM_> + thread(iStep) / cn<tileK_>)]
                                      - s_mxdA[kStart + get(thread(iStep) % cn<tileK_> + Rn<UNROLL>{i + 1})])
                            * s_mxdc[kStart + get(thread(iStep) % cn<tileK_> + Rn<UNROLL>{i + 1})];

                        if (std::min(
                                get(L - blockIdx_y * Q - cn<1>), get(mStart * cn<tileM_> + thread(iStep) / cn<tileK_>))
                            < kStart + get(thread(iStep) % cn<tileK_> + Rn<UNROLL>{i}))
                            tmp2.x = 0;
                        if (std::min(
                                get(L - blockIdx_y * Q - cn<1>), get(mStart * cn<tileM_> + thread(iStep) / cn<tileK_>))
                            < kStart + get(thread(iStep) % cn<tileK_> + Rn<UNROLL>{i + 1}))
                            tmp2.y = 0;

                        if (std::is_same_v<Tp_, half>)
                            *(half2*) &tmpL[i] = __float22half2_rn(tmp2);
                        else
                            *(bf162*) &tmpL[i] = __float22bfloat162_rn(tmp2);
                    }

                    *(int4*) ((char*) s_mxL + swizzle<tileK_ * 2, 128>(thread(iStep) * cn<2>, baseL(jK) * cn<2>))
                        = *(int4*) &tmpL[0];
                }

            __syncthreads();
        }

        asm volatile("fence.proxy.async.shared::cta; \n");
        asm volatile("wgmma.fence.sync.aligned; \n");

#pragma unroll
        for (int k = 0; k < tileK_ / wmmaK_; k++)
        {
#pragma unroll
            for (int y = 0; y < tileM_ / wmmaM_ / warpM_; y++)
            {
                wgmma<Tp_, 0, 1>(
                    r_mxAcc[y], d_mxL0 + k * 2 + y * warpM_ * wmmaM_ * (tileK_ * 2 / 16), d_mxR0 + k * 128);
            }
        }

        asm volatile("wgmma.commit_group.sync.aligned;");
        asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(0));

        __syncthreads();

        if (iK * tileK_ < N_)
        {

            if (threadIdx.y == 0 && threadIdx.z == 0)
            {
                asm volatile("@elected_one mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n" ::"r"(
                                 b_mbar + jK.var % pipeS_ * 8),
                    "r"((tileM_ + tileN_) * tileK_ * 2));
                asm volatile(
                    "@elected_one cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1, {%2, %3, %4}], [%5];\n" ::"r"(b_mxL + jK.var % pipeS_ * (tileM_ * tileK_ * 2)),
                    "l"((CUtensorMap const*) g_mxX_), "r"(jK.var * tileK_), "r"(gStart.var),
                    "r"(get(aStart + blockIdx_y * Q + mStart * cn<tileM_>)), "r"(b_mbar + jK.var % pipeS_ * 8));
#pragma unroll
                for (int iN = 0; iN < tileN_; iN += 64)
                    asm volatile(
                        "@elected_one cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes"
                        " [%0], [%1, {%2, %3, %4, %5}], [%6];\n" ::"r"(
                            b_mxR + jK.var % pipeS_ * (tileN_ * tileK_ * 2) + iN * tileK_ * 2),
                        "l"((CUtensorMap const*) g_mxX_ + 1), "r"(get(nStart * cn<tileN_>) + iN), "r"(jK.var * tileK_),
                        "r"(hStart.var), "r"(get(cStart + blockIdx_y)), "r"(b_mbar + jK.var % pipeS_ * 8));
            }
        }
        else if (iK * tileK_ < N_ + Q1)
        {

            if (threadIdx.y == 0 && threadIdx.z == 0)
            {
                asm volatile("@elected_one mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n" ::"r"(
                                 b_mbar + jK.var % pipeS_ * 8),
                    "r"((tileM_ + tileN_) * tileK_ * 2));
                asm volatile(
                    "@elected_one cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1, {%2, %3, %4, %5}], [%6];\n" ::"r"(b_mxL + jK.var % pipeS_ * (tileM_ * tileK_ * 2)),
                    "l"((CUtensorMap const*) g_mxX_ + 2), "r"(jK.var * tileK_ - N_), "r"(get(mStart * cn<tileM_>)),
                    "r"(gStart.var), "r"(get(cStart + blockIdx_y)), "r"(b_mbar + jK.var % pipeS_ * 8));
#pragma unroll
                for (int iN = 0; iN < tileN_; iN += 64)
                    asm volatile(
                        "@elected_one cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes"
                        " [%0], [%1, {%2, %3, %4}], [%5];\n" ::"r"(
                            b_mxR + jK.var % pipeS_ * (tileN_ * tileK_ * 2) + iN * tileK_ * 2),
                        "l"((CUtensorMap const*) g_mxX_ + 3), "r"(get(nStart * cn<tileN_>) + iN), "r"(hStart.var),
                        "r"(get(aStart + blockIdx_y * Q + jK * cn<tileK_> - N)), "r"(b_mbar + jK.var % pipeS_ * 8));
            }
        }
    }

    if (threadIdx.y == 0 && threadIdx.z == 0)
    {
#pragma unroll
        for (Rn<UNROLL, pipeS_> iK; iK.var < iK.size; iK.var++)
            asm volatile("@elected_one mbarrier.inval.shared.b64 [%0];\n" ::"r"(b_mbar + iK.var * 8));
    }

    if (g_mxD)
    {
        float r_D = g_mxD[hStart.var];

#pragma unroll
        for (int y = 0; y < tileM_ / wmmaM_ / warpM_; y++)
#pragma unroll
            for (int x = 0; x < tileN_ / wmmaN_ / warpN_; x++)
            {
                Tp_ tmp16[4] = {0};
                float tmp32[4] = {0};

                if (blockIdx_y * Q + mStart * cn<tileM_> + Rn<UNROLL>{y} * cn<warpM_ * wmmaM_>
                        + threadIdx_y / cn<4> * cn<8> + threadIdx_y % cn<4> * cn<warpM_ / 4 * wmmaM_>
                        + threadIdx_x / cn<4>
                    < L)
                {
                    *(int*) &tmp16[0] = *(int*) (g_mxX
                        + get((mStart * cn<tileM_> + Rn<UNROLL>{y} * cn<warpM_ * wmmaM_> + threadIdx_y / cn<4> * cn<8>
                                  + threadIdx_y % cn<4> * cn<warpM_ / 4 * wmmaM_> + threadIdx_x / cn<4>) *X_stride
                            + hStart * P + nStart * cn<tileN_> + Rn<UNROLL>{x} * cn<wmmaN_>
                            + threadIdx_z * cn<tileN_ / warpN_> + threadIdx_x % cn<4> * cn<2>));

                    *(float2*) &tmp32[0] = std::is_same_v<Tp_, half> ? __half22float2(*(half2*) &tmp16[0])
                                                                     : bf1622float2(*(bf162*) &tmp16[0]);

                    r_mxAcc[y][x][0] += r_D * tmp32[0];
                    r_mxAcc[y][x][1] += r_D * tmp32[1];
                }

                if (blockIdx_y * Q + mStart * cn<tileM_> + Rn<UNROLL>{y} * cn<warpM_ * wmmaM_> + cn<warpM_ / 4 * 8>
                        + threadIdx_y / cn<4> * cn<8> + threadIdx_y % cn<4> * cn<warpM_ / 4 * wmmaM_>
                        + threadIdx_x / cn<4>
                    < L)
                {
                    *(int*) &tmp16[2] = *(int*) (g_mxX
                        + get((mStart * cn<tileM_> + Rn<UNROLL>{y} * cn<warpM_ * wmmaM_> + cn<warpM_ / 4 * 8>
                                  + threadIdx_y / cn<4> * cn<8> + threadIdx_y % cn<4> * cn<warpM_ / 4 * wmmaM_>
                                  + threadIdx_x / cn<4>) *X_stride
                            + hStart * P + nStart * cn<tileN_> + Rn<UNROLL>{x} * cn<wmmaN_>
                            + threadIdx_z * cn<tileN_ / warpN_> + threadIdx_x % cn<4> * cn<2>));

                    *(float2*) &tmp32[2] = std::is_same_v<Tp_, half> ? __half22float2(*(half2*) &tmp16[2])
                                                                     : bf1622float2(*(bf162*) &tmp16[2]);

                    r_mxAcc[y][x][2] += r_D * tmp32[2];
                    r_mxAcc[y][x][3] += r_D * tmp32[3];
                }
            }
    }

    if (g_mxZ)
    {
#pragma unroll
        for (int y = 0; y < tileM_ / wmmaM_ / warpM_; y++)
#pragma unroll
            for (int x = 0; x < tileN_ / wmmaN_ / warpN_; x++)
            {
                Tp_ tmp16[4] = {0};
                float tmp32[4] = {0};

                if (blockIdx_y * Q + mStart * cn<tileM_> + Rn<UNROLL>{y} * cn<warpM_ * wmmaM_>
                        + threadIdx_y / cn<4> * cn<8> + threadIdx_y % cn<4> * cn<warpM_ / 4 * wmmaM_>
                        + threadIdx_x / cn<4>
                    < L)
                {
                    *(int*) &tmp16[0] = *(int*) (g_mxZ
                        + get((mStart * cn<tileM_> + Rn<UNROLL>{y} * cn<warpM_ * wmmaM_> + threadIdx_y / cn<4> * cn<8>
                                  + threadIdx_y % cn<4> * cn<warpM_ / 4 * wmmaM_> + threadIdx_x / cn<4>) *Z_stride
                            + hStart * P + nStart * cn<tileN_> + Rn<UNROLL>{x} * cn<wmmaN_>
                            + threadIdx_z * cn<tileN_ / warpN_> + threadIdx_x % cn<4> * cn<2>));

                    *(float2*) &tmp32[0] = std::is_same_v<Tp_, half> ? __half22float2(*(half2*) &tmp16[0])
                                                                     : bf1622float2(*(bf162*) &tmp16[0]);

                    r_mxAcc[y][x][0] *= tmp32[0] > 32.f ? tmp32[0] : tmp32[0] / (1.f + expf(-tmp32[0]));
                    r_mxAcc[y][x][1] *= tmp32[1] > 32.f ? tmp32[1] : tmp32[1] / (1.f + expf(-tmp32[1]));
                }

                if (blockIdx_y * Q + mStart * cn<tileM_> + Rn<UNROLL>{y} * cn<warpM_ * wmmaM_> + cn<warpM_ / 4 * 8>
                        + threadIdx_y / cn<4> * cn<8> + threadIdx_y % cn<4> * cn<warpM_ / 4 * wmmaM_>
                        + threadIdx_x / cn<4>
                    < L)
                {
                    *(int*) &tmp16[2] = *(int*) (g_mxZ
                        + get((mStart * cn<tileM_> + Rn<UNROLL>{y} * cn<warpM_ * wmmaM_> + cn<warpM_ / 4 * 8>
                                  + threadIdx_y / cn<4> * cn<8> + threadIdx_y % cn<4> * cn<warpM_ / 4 * wmmaM_>
                                  + threadIdx_x / cn<4>) *Z_stride
                            + hStart * P + nStart * cn<tileN_> + Rn<UNROLL>{x} * cn<wmmaN_>
                            + threadIdx_z * cn<tileN_ / warpN_> + threadIdx_x % cn<4> * cn<2>));

                    *(float2*) &tmp32[2] = std::is_same_v<Tp_, half> ? __half22float2(*(half2*) &tmp16[2])
                                                                     : bf1622float2(*(bf162*) &tmp16[2]);

                    r_mxAcc[y][x][2] *= tmp32[2] > 32.f ? tmp32[2] : tmp32[2] / (1.f + expf(-tmp32[2]));
                    r_mxAcc[y][x][3] *= tmp32[3] > 32.f ? tmp32[3] : tmp32[3] / (1.f + expf(-tmp32[3]));
                }
            }
    }

#pragma unroll
    for (int y = 0; y < tileM_ / wmmaM_ / warpM_; y++)
#pragma unroll
        for (int x = 0; x < tileN_ / wmmaN_ / warpN_; x++)
        {
            if (std::is_same_v<Tp_, half>)
            {
                *(half2*) &r_mxAcc[y][x][0] = __floats2half2_rn(r_mxAcc[y][x][0], r_mxAcc[y][x][1]);
                *(half2*) &r_mxAcc[y][x][2] = __floats2half2_rn(r_mxAcc[y][x][2], r_mxAcc[y][x][3]);
            }
            else
            {
                *(bf162*) &r_mxAcc[y][x][0] = __floats2bfloat162_rn(r_mxAcc[y][x][0], r_mxAcc[y][x][1]);
                *(bf162*) &r_mxAcc[y][x][2] = __floats2bfloat162_rn(r_mxAcc[y][x][2], r_mxAcc[y][x][3]);
            }
        }

#pragma unroll
    for (int y = 0; y < tileM_ / wmmaM_ / warpM_; y++)
#pragma unroll
        for (int x = 0; x < tileN_ / wmmaN_ / warpN_; x++)
        {
            asm volatile(
                "st.shared.b32 [%0], %1;\n" ::"r"(b_mxAcc
                    + 2
                        * swz<tileN_ * 2, tileN_>(y * warpM_ * wmmaM_ * tileN_ + x * wmmaN_
                            + (threadIdx.y / 4 * 8 + threadIdx.y % 4 * (warpM_ / 4 * wmmaM_) + +threadIdx.x / 4)
                                * tileN_
                            + (threadIdx.z * tileN_ / warpN_ + threadIdx.x % 4 * 2))),
                "r"(*(unsigned*) &r_mxAcc[y][x][0]));
            asm volatile(
                "st.shared.b32 [%0], %1;\n" ::"r"(b_mxAcc
                    + 2
                        * swz<tileN_ * 2, tileN_>(y * warpM_ * wmmaM_ * tileN_ + warpM_ / 4 * 8 * tileN_ + x * wmmaN_
                            + (threadIdx.y / 4 * 8 + threadIdx.y % 4 * (warpM_ / 4 * wmmaM_) + +threadIdx.x / 4)
                                * tileN_
                            + (threadIdx.z * tileN_ / warpN_ + threadIdx.x % 4 * 2))),
                "r"(*(unsigned*) &r_mxAcc[y][x][2]));
        }

    __syncthreads();

#pragma unroll
    for (Rn<UNROLL, div_up(tileM_ * tileN_, warpM_ * warpN_ * 256)> iStep; iStep.var < iStep.size; iStep.var++)
        if (thread(iStep) < cn<tileM_ * tileN_>
            && thread(iStep) / cn<tileN_> < L - blockIdx_y * Q - mStart * cn<tileM_>)
            *(int4*) (g_mxY
                + get((mStart * cn<tileM_> + thread(iStep) / cn<tileN_>) *H * P + hStart * P + nStart * cn<tileN_>
                    + thread(iStep) % cn<tileN_>))
                = *(int4*) ((char*) s_mxAcc + swizzle<tileN_ * 2, tileN_ * 2>(thread(iStep) * cn<2>));
#endif
}

typedef ChunkScanKernelFunc (*GetChunkScanKernelFunc)(int B_, int L_, int H_, int P_, int G_, int N_, int Q_,
    int numTokens_, int isHopper_, tensorrt_llm::common::CUDADriverWrapper* cudaDriver_, dim3* blockDims_,
    dim3* threadDims_, int* sharedMem_, int* useTma_, CUtensorMap* descs_);

template <class Tp_, class Wt_>
ChunkScanKernelFunc getChunkScanKernel(int B_, int L_, int H_, int P_, int G_, int N_, int Q_, int numTokens_,
    int isHopper_, tensorrt_llm::common::CUDADriverWrapper* cudaDriver_, dim3* blockDims_, dim3* threadDims_,
    int* sharedMem_, int* useTma_, CUtensorMap* descs_)
{
    int B = B_;
    int L = L_;
    int H = H_;
    int P = P_;
    int G = G_;
    int N = N_;
    int Q = Q_;
    int C = div_up(L, Q);

    int64_t compute = int64_t(numTokens_) * H * Q * P * (N + Q);

    auto setLaunchParams = [&](int tileM, int tileN, int tileK, int warpM, int warpN, int pipeS, int useTma)
    {
        auto sharedMem
            = useTma * 1024 + std::max((tileM * tileK + tileK * tileN) * pipeS * 2 + Q * 8, (tileM * tileN) * 2);

        *blockDims_ = dim3(H * P / tileN * Q / tileM, C, B);
        *threadDims_ = dim3(32, useTma ? warpM : warpN, useTma ? warpN : warpM);
        *sharedMem_ = sharedMem;
        *useTma_ = useTma;

        if (useTma)
        {
            {
                std::array<uint64_t, 2> tensorStrides{2uL * N, 2uL * (H * P + 2 * G * N)};
                std::array<uint64_t, 3> tensorSizes{1uL * N, 1uL * G, 1uL * numTokens_};
                std::array<uint32_t, 3> traveralStrides{1u, 1u, 1u};
                std::array<uint32_t, 3> boxSizes{(uint32_t) tileK, 1u, (uint32_t) tileM};

                cudaDriver_->cuTensorMapEncodeTiled(&descs_[0], CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 3, nullptr,
                    &tensorSizes[0], &tensorStrides[0], &boxSizes[0], &traveralStrides[0],
                    CU_TENSOR_MAP_INTERLEAVE_NONE, tileK == 64 ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_64B,
                    CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
            }
            {
                std::array<uint64_t, 3> tensorStrides{2uL * P, 2uL * N * P, 2uL * H * N * P};
                std::array<uint64_t, 4> tensorSizes{1uL * P, 1uL * N, 1uL * H, 1uL * div_up(numTokens_ + B * Q - Q, Q)};
                std::array<uint32_t, 4> traveralStrides{1u, 1u, 1u, 1u};
                std::array<uint32_t, 4> boxSizes{64u, (uint32_t) tileK, 1u, 1u};

                cudaDriver_->cuTensorMapEncodeTiled(&descs_[1], CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 4, nullptr,
                    &tensorSizes[0], &tensorStrides[0], &boxSizes[0], &traveralStrides[0],
                    CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
            }
            {
                std::array<uint64_t, 3> tensorStrides{2uL * Q, 2uL * Q * Q, 2uL * G * Q * Q};
                std::array<uint64_t, 4> tensorSizes{1uL * Q, 1uL * Q, 1uL * G, 1uL * div_up(numTokens_ + B * Q - Q, Q)};
                std::array<uint32_t, 4> traveralStrides{1u, 1u, 1u, 1u};
                std::array<uint32_t, 4> boxSizes{(uint32_t) tileK, (uint32_t) tileM, 1u, 1u};

                cudaDriver_->cuTensorMapEncodeTiled(&descs_[2], CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 4, nullptr,
                    &tensorSizes[0], &tensorStrides[0], &boxSizes[0], &traveralStrides[0],
                    CU_TENSOR_MAP_INTERLEAVE_NONE, tileK == 64 ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_64B,
                    CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
            }
            {
                std::array<uint64_t, 2> tensorStrides{2uL * P, 2uL * (H * P + 2 * G * N)};
                std::array<uint64_t, 3> tensorSizes{1uL * P, 1uL * H, 1uL * numTokens_};
                std::array<uint32_t, 3> traveralStrides{1u, 1u, 1u};
                std::array<uint32_t, 3> boxSizes{64u, 1u, (uint32_t) tileK};

                cudaDriver_->cuTensorMapEncodeTiled(&descs_[3], CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 3, nullptr,
                    &tensorSizes[0], &tensorStrides[0], &boxSizes[0], &traveralStrides[0],
                    CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
            }
        }
    };

    if (isHopper_)
    {
#ifndef FAST_BUILD
        if (Q_ == 256 && P_ % 256 == 0)
        {
            if (compute >= (1LL << 44))
                setLaunchParams(64, 128, 32, 4, 2, 2, 1);
            else if (compute >= (1LL << 41))
                setLaunchParams(64, 64, 32, 4, 1, 2, 1);
            else if (compute >= (1LL << 38))
                setLaunchParams(64, 64, 32, 4, 2, 2, 1);
            else if (compute >= (1LL << 36))
                setLaunchParams(64, 64, 32, 4, 1, 2, 0);
            else if (compute >= (1LL << 0))
                setLaunchParams(64, 64, 32, 4, 2, 2, 0);

            if (compute >= (1LL << 44))
                return chunk_scan_hopper<256, 64, 128, 32, 16, 8, 16, 4, 2, 2, Tp_, Wt_>;
            else if (compute >= (1LL << 41))
                return chunk_scan_hopper<256, 64, 64, 32, 16, 8, 16, 4, 1, 2, Tp_, Wt_>;
            else if (compute >= (1LL << 38))
                return chunk_scan_hopper<256, 64, 64, 32, 16, 8, 16, 4, 2, 2, Tp_, Wt_>;
            else if (compute >= (1LL << 36))
                return chunk_scan_kernel<256, 64, 64, 32, 16, 8, 16, 4, 1, 2, Tp_, Wt_>;
            else if (compute >= (1LL << 0))
                return chunk_scan_kernel<256, 64, 64, 32, 16, 8, 16, 4, 2, 2, Tp_, Wt_>;
        }

        if (Q_ == 256 && P_ % 128 == 0)
        {
            if (compute >= (1LL << 44))
                setLaunchParams(64, 128, 32, 4, 2, 2, 1);
            else if (compute >= (1LL << 41))
                setLaunchParams(64, 64, 32, 4, 1, 2, 1);
            else if (compute >= (1LL << 38))
                setLaunchParams(64, 64, 32, 4, 2, 2, 1);
            else if (compute >= (1LL << 36))
                setLaunchParams(64, 64, 32, 4, 1, 2, 0);
            else if (compute >= (1LL << 0))
                setLaunchParams(64, 64, 32, 4, 2, 2, 0);

            if (compute >= (1LL << 44))
                return chunk_scan_hopper<256, 64, 128, 32, 16, 8, 16, 4, 2, 2, Tp_, Wt_>;
            else if (compute >= (1LL << 41))
                return chunk_scan_hopper<256, 64, 64, 32, 16, 8, 16, 4, 1, 2, Tp_, Wt_>;
            else if (compute >= (1LL << 38))
                return chunk_scan_hopper<256, 64, 64, 32, 16, 8, 16, 4, 2, 2, Tp_, Wt_>;
            else if (compute >= (1LL << 36))
                return chunk_scan_kernel<256, 64, 64, 32, 16, 8, 16, 4, 1, 2, Tp_, Wt_>;
            else if (compute >= (1LL << 0))
                return chunk_scan_kernel<256, 64, 64, 32, 16, 8, 16, 4, 2, 2, Tp_, Wt_>;
        }
#endif

        if (Q_ == 256 && P_ % 64 == 0)
        {
            if (compute >= (1LL << 45))
                setLaunchParams(64, 64, 32, 4, 1, 3, 1);
            else if (compute >= (1LL << 41))
                setLaunchParams(64, 64, 32, 4, 1, 2, 1);
            else if (compute >= (1LL << 38))
                setLaunchParams(64, 64, 32, 4, 2, 2, 1);
            else if (compute >= (1LL << 36))
                setLaunchParams(64, 64, 32, 4, 1, 2, 0);
            else if (compute >= (1LL << 0))
                setLaunchParams(64, 64, 32, 4, 2, 2, 0);

            if (compute >= (1LL << 45))
                return chunk_scan_hopper<256, 64, 64, 32, 16, 8, 16, 4, 1, 3, Tp_, Wt_>;
            else if (compute >= (1LL << 41))
                return chunk_scan_hopper<256, 64, 64, 32, 16, 8, 16, 4, 1, 2, Tp_, Wt_>;
            else if (compute >= (1LL << 38))
                return chunk_scan_hopper<256, 64, 64, 32, 16, 8, 16, 4, 2, 2, Tp_, Wt_>;
            else if (compute >= (1LL << 36))
                return chunk_scan_kernel<256, 64, 64, 32, 16, 8, 16, 4, 1, 2, Tp_, Wt_>;
            else if (compute >= (1LL << 0))
                return chunk_scan_kernel<256, 64, 64, 32, 16, 8, 16, 4, 2, 2, Tp_, Wt_>;
        }

#ifndef FAST_BUILD
        if (Q_ == 128 && P_ % 256 == 0)
        {
            if (compute >= (1LL << 43))
                setLaunchParams(64, 128, 32, 4, 2, 2, 1);
            else if (compute >= (1LL << 41))
                setLaunchParams(128, 64, 32, 4, 2, 2, 1);
            else if (compute >= (1LL << 36))
                setLaunchParams(64, 64, 32, 4, 2, 2, 1);
            else if (compute >= (1LL << 35))
                setLaunchParams(64, 64, 32, 2, 2, 2, 0);
            else if (compute >= (1LL << 33))
                setLaunchParams(64, 64, 32, 4, 2, 2, 0);
            else if (compute >= (1LL << 31))
                setLaunchParams(64, 64, 32, 2, 4, 3, 0);
            else if (compute >= (1LL << 0))
                setLaunchParams(64, 64, 32, 4, 2, 2, 0);

            if (compute >= (1LL << 43))
                return chunk_scan_hopper<128, 64, 128, 32, 16, 8, 16, 4, 2, 2, Tp_, Wt_>;
            else if (compute >= (1LL << 41))
                return chunk_scan_hopper<128, 128, 64, 32, 16, 8, 16, 4, 2, 2, Tp_, Wt_>;
            else if (compute >= (1LL << 36))
                return chunk_scan_hopper<128, 64, 64, 32, 16, 8, 16, 4, 2, 2, Tp_, Wt_>;
            else if (compute >= (1LL << 35))
                return chunk_scan_kernel<128, 64, 64, 32, 16, 8, 16, 2, 2, 2, Tp_, Wt_>;
            else if (compute >= (1LL << 33))
                return chunk_scan_kernel<128, 64, 64, 32, 16, 8, 16, 4, 2, 2, Tp_, Wt_>;
            else if (compute >= (1LL << 31))
                return chunk_scan_kernel<128, 64, 64, 32, 16, 8, 16, 2, 4, 3, Tp_, Wt_>;
            else if (compute >= (1LL << 0))
                return chunk_scan_kernel<128, 64, 64, 32, 16, 8, 16, 4, 2, 2, Tp_, Wt_>;
        }

        if (Q_ == 128 && P_ % 128 == 0)
        {
            if (compute >= (1LL << 43))
                setLaunchParams(64, 128, 32, 4, 2, 2, 1);
            else if (compute >= (1LL << 41))
                setLaunchParams(128, 64, 32, 4, 2, 2, 1);
            else if (compute >= (1LL << 36))
                setLaunchParams(64, 64, 32, 4, 2, 2, 1);
            else if (compute >= (1LL << 35))
                setLaunchParams(64, 64, 32, 2, 2, 2, 0);
            else if (compute >= (1LL << 33))
                setLaunchParams(64, 64, 32, 4, 2, 2, 0);
            else if (compute >= (1LL << 31))
                setLaunchParams(64, 64, 32, 2, 4, 3, 0);
            else if (compute >= (1LL << 0))
                setLaunchParams(64, 64, 32, 4, 2, 2, 0);

            if (compute >= (1LL << 43))
                return chunk_scan_hopper<128, 64, 128, 32, 16, 8, 16, 4, 2, 2, Tp_, Wt_>;
            else if (compute >= (1LL << 41))
                return chunk_scan_hopper<128, 128, 64, 32, 16, 8, 16, 4, 2, 2, Tp_, Wt_>;
            else if (compute >= (1LL << 36))
                return chunk_scan_hopper<128, 64, 64, 32, 16, 8, 16, 4, 2, 2, Tp_, Wt_>;
            else if (compute >= (1LL << 35))
                return chunk_scan_kernel<128, 64, 64, 32, 16, 8, 16, 2, 2, 2, Tp_, Wt_>;
            else if (compute >= (1LL << 33))
                return chunk_scan_kernel<128, 64, 64, 32, 16, 8, 16, 4, 2, 2, Tp_, Wt_>;
            else if (compute >= (1LL << 31))
                return chunk_scan_kernel<128, 64, 64, 32, 16, 8, 16, 2, 4, 3, Tp_, Wt_>;
            else if (compute >= (1LL << 0))
                return chunk_scan_kernel<128, 64, 64, 32, 16, 8, 16, 4, 2, 2, Tp_, Wt_>;
        }

        if (Q_ == 128 && P_ % 64 == 0)
        {
            if (compute >= (1LL << 41))
                setLaunchParams(128, 64, 32, 4, 2, 2, 1);
            else if (compute >= (1LL << 36))
                setLaunchParams(64, 64, 32, 4, 2, 2, 1);
            else if (compute >= (1LL << 35))
                setLaunchParams(64, 64, 32, 2, 2, 2, 0);
            else if (compute >= (1LL << 33))
                setLaunchParams(64, 64, 32, 4, 2, 2, 0);
            else if (compute >= (1LL << 31))
                setLaunchParams(64, 64, 32, 2, 4, 3, 0);
            else if (compute >= (1LL << 0))
                setLaunchParams(64, 64, 32, 4, 2, 2, 0);

            if (compute >= (1LL << 41))
                return chunk_scan_hopper<128, 128, 64, 32, 16, 8, 16, 4, 2, 2, Tp_, Wt_>;
            else if (compute >= (1LL << 36))
                return chunk_scan_hopper<128, 64, 64, 32, 16, 8, 16, 4, 2, 2, Tp_, Wt_>;
            else if (compute >= (1LL << 35))
                return chunk_scan_kernel<128, 64, 64, 32, 16, 8, 16, 2, 2, 2, Tp_, Wt_>;
            else if (compute >= (1LL << 33))
                return chunk_scan_kernel<128, 64, 64, 32, 16, 8, 16, 4, 2, 2, Tp_, Wt_>;
            else if (compute >= (1LL << 31))
                return chunk_scan_kernel<128, 64, 64, 32, 16, 8, 16, 2, 4, 3, Tp_, Wt_>;
            else if (compute >= (1LL << 0))
                return chunk_scan_kernel<128, 64, 64, 32, 16, 8, 16, 4, 2, 2, Tp_, Wt_>;
        }

        if (Q_ == 64 && P_ % 256 == 0)
        {
            if (compute >= (1LL << 42))
                setLaunchParams(64, 64, 32, 4, 2, 4, 1);
            else if (compute >= (1LL << 36))
                setLaunchParams(64, 64, 32, 4, 2, 2, 1);
            else if (compute >= (1LL << 0))
                setLaunchParams(64, 64, 32, 4, 2, 2, 0);

            if (compute >= (1LL << 42))
                return chunk_scan_hopper<64, 64, 64, 32, 16, 8, 16, 4, 2, 4, Tp_, Wt_>;
            else if (compute >= (1LL << 36))
                return chunk_scan_hopper<64, 64, 64, 32, 16, 8, 16, 4, 2, 2, Tp_, Wt_>;
            else if (compute >= (1LL << 0))
                return chunk_scan_kernel<64, 64, 64, 32, 16, 8, 16, 4, 2, 2, Tp_, Wt_>;
        }

        if (Q_ == 64 && P_ % 128 == 0)
        {
            if (compute >= (1LL << 42))
                setLaunchParams(64, 64, 32, 4, 2, 4, 1);
            else if (compute >= (1LL << 36))
                setLaunchParams(64, 64, 32, 4, 2, 2, 1);
            else if (compute >= (1LL << 0))
                setLaunchParams(64, 64, 32, 4, 2, 2, 0);

            if (compute >= (1LL << 42))
                return chunk_scan_hopper<64, 64, 64, 32, 16, 8, 16, 4, 2, 4, Tp_, Wt_>;
            else if (compute >= (1LL << 36))
                return chunk_scan_hopper<64, 64, 64, 32, 16, 8, 16, 4, 2, 2, Tp_, Wt_>;
            else if (compute >= (1LL << 0))
                return chunk_scan_kernel<64, 64, 64, 32, 16, 8, 16, 4, 2, 2, Tp_, Wt_>;
        }

        if (Q_ == 64 && P_ % 64 == 0)
        {
            if (compute >= (1LL << 42))
                setLaunchParams(64, 64, 32, 4, 2, 4, 1);
            else if (compute >= (1LL << 36))
                setLaunchParams(64, 64, 32, 4, 2, 2, 1);
            else if (compute >= (1LL << 0))
                setLaunchParams(64, 64, 32, 4, 2, 2, 0);

            if (compute >= (1LL << 42))
                return chunk_scan_hopper<64, 64, 64, 32, 16, 8, 16, 4, 2, 4, Tp_, Wt_>;
            else if (compute >= (1LL << 36))
                return chunk_scan_hopper<64, 64, 64, 32, 16, 8, 16, 4, 2, 2, Tp_, Wt_>;
            else if (compute >= (1LL << 0))
                return chunk_scan_kernel<64, 64, 64, 32, 16, 8, 16, 4, 2, 2, Tp_, Wt_>;
        }
#endif
    }
    else
    {
#ifndef FAST_BUILD
        if (Q_ == 256 && P_ % 256 == 0)
        {
            if (compute >= (1LL << 44))
                setLaunchParams(64, 128, 32, 2, 4, 2, 0);
            else if (compute >= (1LL << 37))
                setLaunchParams(64, 64, 32, 2, 2, 2, 0);
            else if (compute >= (1LL << 0))
                setLaunchParams(64, 64, 32, 4, 2, 2, 0);

            if (compute >= (1LL << 44))
                return chunk_scan_kernel<256, 64, 128, 32, 16, 8, 16, 2, 4, 2, Tp_, Wt_>;
            else if (compute >= (1LL << 37))
                return chunk_scan_kernel<256, 64, 64, 32, 16, 8, 16, 2, 2, 2, Tp_, Wt_>;
            else if (compute >= (1LL << 0))
                return chunk_scan_kernel<256, 64, 64, 32, 16, 8, 16, 4, 2, 2, Tp_, Wt_>;
        }

        if (Q_ == 256 && P_ % 128 == 0)
        {
            if (compute >= (1LL << 44))
                setLaunchParams(64, 128, 32, 2, 4, 2, 0);
            else if (compute >= (1LL << 37))
                setLaunchParams(64, 64, 32, 2, 2, 2, 0);
            else if (compute >= (1LL << 0))
                setLaunchParams(64, 64, 32, 4, 2, 2, 0);

            if (compute >= (1LL << 44))
                return chunk_scan_kernel<256, 64, 128, 32, 16, 8, 16, 2, 4, 2, Tp_, Wt_>;
            else if (compute >= (1LL << 37))
                return chunk_scan_kernel<256, 64, 64, 32, 16, 8, 16, 2, 2, 2, Tp_, Wt_>;
            else if (compute >= (1LL << 0))
                return chunk_scan_kernel<256, 64, 64, 32, 16, 8, 16, 4, 2, 2, Tp_, Wt_>;
        }
#endif

        if (Q_ == 256 && P_ % 64 == 0)
        {
            if (compute >= (1LL << 37))
                setLaunchParams(64, 64, 32, 2, 2, 2, 0);
            else if (compute >= (1LL << 0))
                setLaunchParams(64, 64, 32, 4, 2, 2, 0);

            if (compute >= (1LL << 37))
                return chunk_scan_kernel<256, 64, 64, 32, 16, 8, 16, 2, 2, 2, Tp_, Wt_>;
            else if (compute >= (1LL << 0))
                return chunk_scan_kernel<256, 64, 64, 32, 16, 8, 16, 4, 2, 2, Tp_, Wt_>;
        }

#ifndef FAST_BUILD
        if (Q_ == 128 && P_ % 256 == 0)
        {
            if (compute >= (1LL << 35))
                setLaunchParams(64, 64, 32, 2, 2, 2, 0);
            else if (compute >= (1LL << 0))
                setLaunchParams(64, 64, 32, 4, 2, 2, 0);

            if (compute >= (1LL << 35))
                return chunk_scan_kernel<128, 64, 64, 32, 16, 8, 16, 2, 2, 2, Tp_, Wt_>;
            else if (compute >= (1LL << 0))
                return chunk_scan_kernel<128, 64, 64, 32, 16, 8, 16, 4, 2, 2, Tp_, Wt_>;
        }

        if (Q_ == 128 && P_ % 128 == 0)
        {
            if (compute >= (1LL << 35))
                setLaunchParams(64, 64, 32, 2, 2, 2, 0);
            else if (compute >= (1LL << 0))
                setLaunchParams(64, 64, 32, 4, 2, 2, 0);

            if (compute >= (1LL << 35))
                return chunk_scan_kernel<128, 64, 64, 32, 16, 8, 16, 2, 2, 2, Tp_, Wt_>;
            else if (compute >= (1LL << 0))
                return chunk_scan_kernel<128, 64, 64, 32, 16, 8, 16, 4, 2, 2, Tp_, Wt_>;
        }

        if (Q_ == 128 && P_ % 64 == 0)
        {
            if (compute >= (1LL << 35))
                setLaunchParams(64, 64, 32, 2, 2, 2, 0);
            else if (compute >= (1LL << 0))
                setLaunchParams(64, 64, 32, 4, 2, 2, 0);

            if (compute >= (1LL << 35))
                return chunk_scan_kernel<128, 64, 64, 32, 16, 8, 16, 2, 2, 2, Tp_, Wt_>;
            else if (compute >= (1LL << 0))
                return chunk_scan_kernel<128, 64, 64, 32, 16, 8, 16, 4, 2, 2, Tp_, Wt_>;
        }

        if (Q_ == 64 && P_ % 256 == 0)
        {
            if (compute >= (1LL << 36))
                setLaunchParams(64, 64, 32, 2, 2, 2, 0);
            else if (compute >= (1LL << 0))
                setLaunchParams(64, 64, 32, 4, 2, 2, 0);

            if (compute >= (1LL << 36))
                return chunk_scan_kernel<64, 64, 64, 32, 16, 8, 16, 2, 2, 2, Tp_, Wt_>;
            else if (compute >= (1LL << 0))
                return chunk_scan_kernel<64, 64, 64, 32, 16, 8, 16, 4, 2, 2, Tp_, Wt_>;
        }

        if (Q_ == 64 && P_ % 128 == 0)
        {
            if (compute >= (1LL << 36))
                setLaunchParams(64, 64, 32, 2, 2, 2, 0);
            else if (compute >= (1LL << 0))
                setLaunchParams(64, 64, 32, 4, 2, 2, 0);

            if (compute >= (1LL << 36))
                return chunk_scan_kernel<64, 64, 64, 32, 16, 8, 16, 2, 2, 2, Tp_, Wt_>;
            else if (compute >= (1LL << 0))
                return chunk_scan_kernel<64, 64, 64, 32, 16, 8, 16, 4, 2, 2, Tp_, Wt_>;
        }

        if (Q_ == 64 && P_ % 64 == 0)
        {
            if (compute >= (1LL << 36))
                setLaunchParams(64, 64, 32, 2, 2, 2, 0);
            else if (compute >= (1LL << 0))
                setLaunchParams(64, 64, 32, 4, 2, 2, 0);

            if (compute >= (1LL << 36))
                return chunk_scan_kernel<64, 64, 64, 32, 16, 8, 16, 2, 2, 2, Tp_, Wt_>;
            else if (compute >= (1LL << 0))
                return chunk_scan_kernel<64, 64, 64, 32, 16, 8, 16, 4, 2, 2, Tp_, Wt_>;
        }
#endif
    }

    return nullptr;
}

extern GetChunkScanKernelFunc getChunkScanKernel_fp16_fp16;
extern GetChunkScanKernelFunc getChunkScanKernel_fp16_fp32;
extern GetChunkScanKernelFunc getChunkScanKernel_bf16_bf16;
extern GetChunkScanKernelFunc getChunkScanKernel_bf16_fp32;

static inline ChunkScanKernelFunc getChunkScanKernel(int B_, int L_, int H_, int P_, int G_, int N_, int Q_,
    int numTokens_, int isHopper_, tensorrt_llm::common::CUDADriverWrapper* cudaDriver_, dim3* blockDims_,
    dim3* threadDims_, int* sharedMem_, int* useTma_, CUtensorMap* descs_, CudaType tp_ = CT_FP16,
    CudaType wt_ = CT_FP32)
{
    if (tp_ == CT_FP16 && wt_ == CT_FP16)
        return getChunkScanKernel_fp16_fp16(B_, L_, H_, P_, G_, N_, Q_, numTokens_, isHopper_, cudaDriver_, blockDims_,
            threadDims_, sharedMem_, useTma_, descs_);
    else if (tp_ == CT_FP16 && wt_ == CT_FP32)
        return getChunkScanKernel_fp16_fp32(B_, L_, H_, P_, G_, N_, Q_, numTokens_, isHopper_, cudaDriver_, blockDims_,
            threadDims_, sharedMem_, useTma_, descs_);
    else if (tp_ == CT_BF16 && wt_ == CT_BF16)
        return getChunkScanKernel_bf16_bf16(B_, L_, H_, P_, G_, N_, Q_, numTokens_, isHopper_, cudaDriver_, blockDims_,
            threadDims_, sharedMem_, useTma_, descs_);
    else if (tp_ == CT_BF16 && wt_ == CT_FP32)
        return getChunkScanKernel_bf16_fp32(B_, L_, H_, P_, G_, N_, Q_, numTokens_, isHopper_, cudaDriver_, blockDims_,
            threadDims_, sharedMem_, useTma_, descs_);

    return nullptr;
}

} // namespace kernels
} // namespace tensorrt_llm

// vim: ts=2 sw=2 sts=2 et sta
