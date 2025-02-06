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

typedef void (*ChunkStateKernelFunc)(int B_, int L_, int H_, int P_, int G_, int N_,
    //  const void *g_mxY_,  // Tp_   B*L*H*P
    //  const void *g_mxOs_, // Tp_   B*C*H*N*P
    //  const void *g_mxFs_, // Tp_   B  *H*N*P
    void* g_mxSt_,       // float B*C*H*N*P
    void const* g_mxdc_, // float B*C*H*Q
    void const* g_mxdA_, // float B*C*H*Q
                         //  const void *g_mxdt_, // Tp_   B*L*((g_mxZ?2:1)*H*P+2*G+round_up(H,8))
                         //  const void *g_mxdb_, // Wt_       H
                         //  const void *g_mxA_,  // Wt_       H
                         //  const void *g_mxCB_, // Tp_   B*C*G*Q*Q
                         //  const void *g_mxD_,  // Wt_       H
    void const* g_mxX_,  // Tp_   B*L*(H*P+2*G*N)
                         //  const void *g_mxZ_,  // g_mxdt_ or nullptr
    bool removePadding_, int const* lastTokenIdsPtr_);

template <int Q_, int tileM_, int tileN_, int tileK_, // smem size, per sm
    int warpM_, int warpN_,                           // warp number
    int pipeS_, int pipeR_, int group_, int bnChk_, class Tp_>
__global__ std::enable_if_t<std::is_same_v<Tp_, half> || std::is_same_v<Tp_, __nv_bfloat16>> chunk_state_kernel(int B_,
    int L_, int H_, int P_, int G_, int N_,
    //  const void *g_mxY_,  // Tp_   B*L*H*P
    //  const void *g_mxOs_, // Tp_   B*C*H*N*P
    //  const void *g_mxFs_, // Tp_   B  *H*N*P
    void* g_mxSt_,       // float B*C*H*N*P
    void const* g_mxdc_, // float B*C*H*Q
    void const* g_mxdA_, // float B*C*H*Q
                         //  const void *g_mxdt_, // Tp_   B*L*((g_mxZ?2:1)*H*P+2*G+round_up(H,8))
                         //  const void *g_mxdb_, // Wt_       H
                         //  const void *g_mxA_,  // Wt_       H
                         //  const void *g_mxCB_, // Tp_   B*C*G*Q*Q
                         //  const void *g_mxD_,  // Wt_       H
    void const* g_mxX_,  // Tp_   B*L*(H*P+2*G*N)
                         //  const void *g_mxZ_,  // g_mxdt_ or nullptr
    bool removePadding_, int const* lastTokenIdsPtr_)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    using namespace tensorrt_llm::common;

    constexpr int wmmaM_ = 16; // wmma size, per instruction
    constexpr int wmmaN_ = 8;
    constexpr int wmmaK_ = 16;

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

    int numBlocks_m = H_ * (N_ / tileM_);
    int numBlocks_n = div_up(P_, tileN_);

    int numBlocks_group = numBlocks_n * group_;
    int blockIdx_0 = blockIdx.x / numBlocks_group * group_;
    int group_m = std::min(group_, numBlocks_m - blockIdx_0);

    int mStartInt = blockIdx.x % numBlocks_group % group_m + blockIdx_0;
    int nStartInt = blockIdx.x % numBlocks_group / group_m;

    auto hStart = Rn<ID>{mStartInt / (N_ / tileM_)};
    auto mStart = Rn<ID>{mStartInt % (N_ / tileM_)};
    auto nStart = Rn<ID>{nStartInt};
    auto gStart = Rn<ID>{hStart.var / (H_ / G_)};

    //  const Tp_   *g_mxY  = (const Tp_   *)g_mxY_;
    //  const Tp_   *g_mxOs = (const Tp_   *)g_mxOs_;
    //  const Tp_   *g_mxFs = (const Tp_   *)g_mxFs_;
    float* g_mxSt = (float*) g_mxSt_ + int64_t(get(cStart + blockIdx_y)) * get(H * N * P);
    float const* g_mxdc = (float const*) g_mxdc_ + int64_t(get(cStart + blockIdx_y)) * get(H * Q);
    float const* g_mxdA = (float const*) g_mxdA_ + int64_t(get(cStart + blockIdx_y)) * get(H * Q);
    //  const Tp_   *g_mxdt = (const Tp_   *)g_mxdt_;
    //  const Wt_   *g_mxdb = (const Wt_   *)g_mxdb_;
    //  const Wt_   *g_mxA  = (const Wt_   *)g_mxA_;
    //  const Tp_   *g_mxCB = (const Tp_   *)g_mxCB_;
    //  const Wt_   *g_mxD  = (const Wt_   *)g_mxD_;
    Tp_ const* g_mxX = (Tp_ const*) g_mxX_ + int64_t(get(aStart + blockIdx_y * Q)) * get(X_stride);
    //  const Tp_   *g_mxZ  = (const Tp_   *)g_mxZ_;

    extern __shared__ float smem[];

    Tp_* s_mxL = (Tp_*) smem;
    Tp_* s_mxR = (Tp_*) smem + tileM_ * tileK_ * pipeS_;

    float* s_mxdc = smem + (tileM_ + tileN_) * tileK_ * pipeS_ / 2;
    float* s_mxdA = smem + (tileM_ + tileN_) * tileK_ * pipeS_ / 2 + Q_;

    unsigned b_base = __nvvm_get_smem_pointer(smem);

    unsigned b_mxL = b_base;
    unsigned b_mxR = b_base + tileM_ * tileK_ * pipeS_ * sizeof(Tp_);

    using std::array;

    array<array<array<float, wmmaM_ * wmmaN_ / 32>, tileN_ / wmmaN_ / warpN_>, tileM_ / wmmaM_ / warpM_> r_mxAcc
        = array<array<array<float, wmmaM_ * wmmaN_ / 32>, tileN_ / wmmaN_ / warpN_>, tileM_ / wmmaM_ / warpM_>();
    array<array<array<unsigned, wmmaM_ * wmmaK_ / 64>, tileM_ / wmmaM_ / warpM_>, pipeR_> r_mxL;
    array<array<array<unsigned, wmmaK_ * wmmaN_ / 64>, tileN_ / wmmaN_ / warpN_>, pipeR_> r_mxR;

    if (pipeR_ == 2)
    {
        r_mxL = array<array<array<unsigned, wmmaM_ * wmmaK_ / 64>, tileM_ / wmmaM_ / warpM_>, pipeR_>();
        r_mxR = array<array<array<unsigned, wmmaK_ * wmmaN_ / 64>, tileN_ / wmmaN_ / warpN_>, pipeR_>();
    }

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
                // dc & dA should be set to zero out of seq length, but this is done to
                // dc by chunkcumsum, so no need here.
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
                && thread(iStep) / cn<tileM_> < L - blockIdx_y * Q - iK * cn<tileK_>)
                cp_shared_global<16>(b_mxL + swizzle<tileM_ * 2, tileM_ * 2>(thread(iStep) * cn<2>, baseL(iK) * cn<2>),
                    g_mxX
                        + get((iK * cn<tileK_> + thread(iStep) / cn<tileM_>) *X_stride + H * P + gStart * N
                            + mStart * cn<tileM_> + thread(iStep) % cn<tileM_>));
            else if (thread(iStep) < cn<tileM_ * tileK_>)
                *(int4*) ((char*) s_mxL + swizzle<tileM_ * 2, tileM_ * 2>(thread(iStep) * cn<2>, baseL(iK) * cn<2>))
                    = int4{0, 0, 0, 0};

#pragma unroll
        for (Rn<UNROLL, div_up(tileN_ * tileK_, warpM_ * warpN_ * 256)> iStep; iStep.var < iStep.size; iStep.var++)
            if (thread(iStep) < cn<tileN_ * tileK_> && thread(iStep) / cn<tileN_> < L - blockIdx_y * Q - iK * cn<tileK_>
                && (!bnChk_ || nStart * cn<tileN_> + thread(iStep) % cn<tileN_> < P))
                cp_shared_global<16>(b_mxR + swizzle<tileN_ * 2, tileN_ * 2>(thread(iStep) * cn<2>, baseR(iK) * cn<2>),
                    g_mxX
                        + get((iK * cn<tileK_> + thread(iStep) / cn<tileN_>) *X_stride + hStart * P
                            + nStart * cn<tileN_> + thread(iStep) % cn<tileN_>));
            else if (thread(iStep) < cn<tileN_ * tileK_>)
                *(int4*) ((char*) s_mxR + swizzle<tileN_ * 2, tileN_ * 2>(thread(iStep) * cn<2>, baseR(iK) * cn<2>))
                    = int4{0, 0, 0, 0};

        cp_commit_group();
    }

    cp_wait_group<pipeS_ - 1>();

    __syncthreads();

    for (int iK = pipeS_; iK < Q_ / tileK_ + pipeS_; iK++)
    {
        auto jK = Rn<>{iK};
#pragma unroll
        for (Rn<UNROLL, div_up(tileM_ * tileK_, warpM_ * warpN_ * 256)> iStep; iStep.var < iStep.size; iStep.var++)
            if (thread(iStep) < cn<tileM_ * tileK_>)
            {
                Tp_ tmpL[8];

                *(int4*) &tmpL[0] = *(
                    int4*) ((char*) s_mxL + swizzle<tileM_ * 2, tileM_ * 2>(thread(iStep) * cn<2>, baseL(jK) * cn<2>));

#pragma unroll
                for (int i = 0; i < 8; i += 2)
                {
                    float2 tmp2 = std::is_same_v<Tp_, half> ? __half22float2(*(half2*) &tmpL[i])
                                                            : bf1622float2(*(bf162*) &tmpL[i]);

                    int kStart = (iK - pipeS_) * cn<tileK_>;

                    // dc & dA should be set to zero out of seq length, but this is done to
                    // dc by chunkcumsum, so no need here.
                    tmp2.x *= expf(s_mxdA[Q_ - 1] - s_mxdA[kStart + get(thread(iStep) / cn<tileM_>)])
                        * s_mxdc[kStart + get(thread(iStep) / cn<tileM_>)];
                    tmp2.y *= expf(s_mxdA[Q_ - 1] - s_mxdA[kStart + get(thread(iStep) / cn<tileM_>)])
                        * s_mxdc[kStart + get(thread(iStep) / cn<tileM_>)];

                    if (std::is_same_v<Tp_, half>)
                        *(half2*) &tmpL[i] = __float22half2_rn(tmp2);
                    else
                        *(bf162*) &tmpL[i] = __float22bfloat162_rn(tmp2);
                }

                *(int4*) ((char*) s_mxL + swizzle<tileM_ * 2, tileM_ * 2>(thread(iStep) * cn<2>, baseL(jK) * cn<2>))
                    = *(int4*) &tmpL[0];
            }

        __syncthreads();

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
                                ldmatrix_x4</* trans_ = */ true>(r_mxL[k % pipeR_][y1][0], r_mxL[k % pipeR_][y1][1],
                                    r_mxL[k % pipeR_][y1][2], r_mxL[k % pipeR_][y1][3],
                                    b_mxL + iK % pipeS_ * (tileM_ * tileK_ * 2)
                                        + 2
                                            * swz<tileM_ * 2, tileM_>(y1 * warpM_ * wmmaM_ + k * wmmaK_ * tileM_
                                                + threadIdx.z * wmmaM_ + threadIdx.x % 8 * tileM_
                                                + threadIdx.x / 8 % 2 * 8 + threadIdx.x / wmmaK_ * 8 * tileM_));
                        }

                        if (x1 >= 0 && x1 < tileN_ / wmmaN_ / warpN_)
                        {
                            if (wmmaK_ == 16 && x1 % 2 == 0)
                                ldmatrix_x4</* trans_ = */ true>(r_mxR[k % pipeR_][x1][0], r_mxR[k % pipeR_][x1][1],
                                    r_mxR[k % pipeR_][x1 + 1][0], r_mxR[k % pipeR_][x1 + 1][1],
                                    b_mxR + iK % pipeS_ * (tileK_ * tileN_ * 2)
                                        + 2
                                            * swz<tileN_ * 2, tileN_>(x1 * warpN_ * wmmaN_ + k * wmmaK_ * tileN_
                                                + threadIdx.y * wmmaN_ + threadIdx.x % wmmaK_ * tileN_
                                                + threadIdx.x / wmmaK_ * warpN_ * wmmaN_));
                        }
                    }

                    if (pipeR_ == 2)
                    {
                        if (wmmaK_ == 16)
                            mma<Tp_>(r_mxAcc[y][x], r_mxL[(k + 1) % pipeR_][y], r_mxR[(k + 1) % pipeR_][x]);
                    }
                }

#pragma unroll
            for (int y = 0; y < tileM_ / wmmaM_ / warpM_; y++)
#pragma unroll
                for (int x = 0; x < tileN_ / wmmaN_ / warpN_; x++)
                {
                    if (pipeR_ == 1)
                    {
                        if (wmmaK_ == 16)
                            mma<Tp_>(r_mxAcc[y][x], r_mxL[(k + 1) % pipeR_][y], r_mxR[(k + 1) % pipeR_][x]);
                    }
                }
        }

        __syncthreads();

#pragma unroll
        for (Rn<UNROLL, div_up(tileM_ * tileK_, warpM_ * warpN_ * 256)> iStep; iStep.var < iStep.size; iStep.var++)
            if (thread(iStep) < cn<tileM_ * tileK_> && thread(iStep) / cn<tileM_> < L - blockIdx_y * Q - jK * cn<tileK_>
                && jK * cn<tileK_> < Q)
                cp_shared_global<16>(b_mxL + swizzle<tileM_ * 2, tileM_ * 2>(thread(iStep) * cn<2>, baseL(jK) * cn<2>),
                    g_mxX
                        + get((jK * cn<tileK_> + thread(iStep) / cn<tileM_>) *X_stride + H * P + gStart * N
                            + mStart * cn<tileM_> + thread(iStep) % cn<tileM_>));
            else if (thread(iStep) < cn<tileM_ * tileK_> && jK * cn<tileK_> < Q)
                *(int4*) ((char*) s_mxL + swizzle<tileM_ * 2, tileM_ * 2>(thread(iStep) * cn<2>, baseL(jK) * cn<2>))
                    = int4{0, 0, 0, 0};

#pragma unroll
        for (Rn<UNROLL, div_up(tileN_ * tileK_, warpM_ * warpN_ * 256)> iStep; iStep.var < iStep.size; iStep.var++)
            if (thread(iStep) < cn<tileN_ * tileK_> && thread(iStep) / cn<tileN_> < L - blockIdx_y * Q - jK * cn<tileK_>
                && (!bnChk_ || nStart * cn<tileN_> + thread(iStep) % cn<tileN_> < P) && jK * cn<tileK_> < Q)
                cp_shared_global<16>(b_mxR + swizzle<tileN_ * 2, tileN_ * 2>(thread(iStep) * cn<2>, baseR(jK) * cn<2>),
                    g_mxX
                        + get((jK * cn<tileK_> + thread(iStep) / cn<tileN_>) *X_stride + hStart * P
                            + nStart * cn<tileN_> + thread(iStep) % cn<tileN_>));
            else if (thread(iStep) < cn<tileN_ * tileK_> && jK * cn<tileK_> < Q)
                *(int4*) ((char*) s_mxR + swizzle<tileN_ * 2, tileN_ * 2>(thread(iStep) * cn<2>, baseR(jK) * cn<2>))
                    = int4{0, 0, 0, 0};

        cp_commit_group();
        cp_wait_group<pipeS_ - 1>();

        __syncthreads();
    }

    cp_wait_group<0>();

#pragma unroll
    for (int y = 0; y < tileM_ / wmmaM_ / warpM_; y++)
#pragma unroll
        for (int x = 0; x < tileN_ / wmmaN_ / warpN_; x++)
        {
            if (pipeR_ == 2)
            {
                if (wmmaK_ == 16)
                    mma<Tp_>(r_mxAcc[y][x], r_mxL[1][y], r_mxR[1][x]);
            }

            if (!bnChk_
                || nStart * cn<tileN_> + Rn<UNROLL>{x} * cn<warpN_ * wmmaN_> + threadIdx_y * cn<wmmaN_>
                        + threadIdx_x % cn<4> * cn<2>
                    < P)
            {
                *(float2*) (g_mxSt
                    + get(hStart * N * P
                        + (mStart * cn<tileM_> + Rn<UNROLL>{y} * cn<warpM_ * wmmaM_> + threadIdx_z * cn<wmmaM_>
                            + threadIdx_x / cn<4>) *P
                        + nStart * cn<tileN_> + Rn<UNROLL>{x} * cn<warpN_ * wmmaN_> + threadIdx_y * cn<wmmaN_>
                        + threadIdx_x % cn<4> * cn<2>))
                    = *(float2*) &r_mxAcc[y][x][0];

                *(float2*) (g_mxSt
                    + get(hStart * N * P
                        + (mStart * cn<tileM_> + Rn<UNROLL>{y} * cn<warpM_ * wmmaM_> + cn<8> + threadIdx_z * cn<wmmaM_>
                            + threadIdx_x / cn<4>) *P
                        + nStart * cn<tileN_> + Rn<UNROLL>{x} * cn<warpN_ * wmmaN_> + threadIdx_y * cn<wmmaN_>
                        + threadIdx_x % cn<4> * cn<2>))
                    = *(float2*) &r_mxAcc[y][x][2];
            }
        }
#endif
}

template <int Q_, int tileM_, int tileN_, int tileK_, // smem size, per sm
    int warpM_, int warpN_,                           // warp number
    int pipeS_, int pipeR_, int group_, int bnChk_, class Tp_>
__global__ std::enable_if_t<std::is_same_v<Tp_, half> || std::is_same_v<Tp_, __nv_bfloat16>> chunk_state_hopper(int B_,
    int L_, int H_, int P_, int G_, int N_,
    //  const void *g_mxY_,  // Tp_   B*L*H*P
    //  const void *g_mxOs_, // Tp_   B*C*H*N*P
    //  const void *g_mxFs_, // Tp_   B  *H*N*P
    void* g_mxSt_,       // float B*C*H*N*P
    void const* g_mxdc_, // float B*C*H*Q
    void const* g_mxdA_, // float B*C*H*Q
                         //  const void *g_mxdt_, // Tp_   B*L*((g_mxZ?2:1)*H*P+2*G+round_up(H,8))
                         //  const void *g_mxdb_, // Wt_       H
                         //  const void *g_mxA_,  // Wt_       H
                         //  const void *g_mxCB_, // Tp_   B*C*G*Q*Q
                         //  const void *g_mxD_,  // Wt_       H
    void const* g_mxX_,  // Tp_   B*L*(H*P+2*G*N)
                         //  const void *g_mxZ_,  // g_mxdt_ or nullptr
    bool removePadding_, int const* lastTokenIdsPtr_)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)
    using namespace tensorrt_llm::common;

    constexpr int wmmaM_ = 16; // wmma size, per instruction
    constexpr int wmmaN_ = 8;
    constexpr int wmmaK_ = 16;

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

    int numBlocks_m = H_ * (N_ / tileM_);
    int numBlocks_n = div_up(P_, tileN_);

    int numBlocks_group = numBlocks_n * group_;
    int blockIdx_0 = blockIdx.x / numBlocks_group * group_;
    int group_m = std::min(group_, numBlocks_m - blockIdx_0);

    int mStartInt = blockIdx.x % numBlocks_group % group_m + blockIdx_0;
    int nStartInt = blockIdx.x % numBlocks_group / group_m;

    auto hStart = Rn<ID>{mStartInt / (N_ / tileM_)};
    auto mStart = Rn<ID>{mStartInt % (N_ / tileM_)};
    auto nStart = Rn<ID>{nStartInt};
    auto gStart = Rn<ID>{hStart.var / (H_ / G_)};

    //  const Tp_   *g_mxY  = (const Tp_   *)g_mxY_;
    //  const Tp_   *g_mxOs = (const Tp_   *)g_mxOs_;
    //  const Tp_   *g_mxFs = (const Tp_   *)g_mxFs_;
    float* g_mxSt = (float*) g_mxSt_ + int64_t(get(cStart + blockIdx_y)) * get(H * N * P);
    float const* g_mxdc = (float const*) g_mxdc_ + int64_t(get(cStart + blockIdx_y)) * get(H * Q);
    float const* g_mxdA = (float const*) g_mxdA_ + int64_t(get(cStart + blockIdx_y)) * get(H * Q);
    //  const Tp_   *g_mxdt = (const Tp_   *)g_mxdt_;
    //  const Wt_   *g_mxdb = (const Wt_   *)g_mxdb_;
    //  const Wt_   *g_mxA  = (const Wt_   *)g_mxA_;
    //  const Tp_   *g_mxCB = (const Tp_   *)g_mxCB_;
    //  const Wt_   *g_mxD  = (const Wt_   *)g_mxD_;
    //  const Tp_   *g_mxX  = (const Tp_   *)g_mxX_;
    //  const Tp_   *g_mxZ  = (const Tp_   *)g_mxZ_;

    extern __shared__ float smem[];

    Tp_* s_mxL = (Tp_*) smem + 512;
    Tp_* s_mxR = (Tp_*) smem + 512 + tileM_ * tileK_ * pipeS_;

    float* s_mxdc = smem + 256 + (tileM_ + tileN_) * tileK_ * pipeS_ / 2;
    float* s_mxdA = smem + 256 + (tileM_ + tileN_) * tileK_ * pipeS_ / 2 + Q_;

    unsigned b_base = __nvvm_get_smem_pointer(smem);

    unsigned b_mxL = b_base + 1024;
    unsigned b_mxR = b_base + 1024 + tileM_ * tileK_ * pipeS_ * sizeof(Tp_);
    unsigned b_mbar = b_base;

    uint32_t formatL = 1;
    uint32_t formatR = tileN_ >= 64 ? 1 : (tileN_ == 32 ? 2 : 3);
    uint32_t baseOffsetL = 0;
    uint32_t baseOffsetR = 0;
    uint32_t strideDimL = 64;
    uint32_t strideDimR = tileN_ >= 64 ? 64 : tileN_;
    uint32_t leadingDimL = tileK_ * 8 * warpM_ / 4;
    uint32_t leadingDimR = tileN_ >= 64 ? tileK_ * 8 : 0;
    uint32_t startAddrL = (b_mxL + threadIdx.y / 4 * tileK_ * 128) / 16;
    uint32_t startAddrR = (b_mxR + threadIdx.z * tileN_ / warpN_ % strideDimR * 2
                              + threadIdx.z * tileN_ / warpN_ / strideDimR * tileK_ * strideDimR * 2)
        / 16;

    uint64_t d_mxL = ((uint64_t) formatL << 62) + ((uint64_t) baseOffsetL << 49) + ((uint64_t) strideDimL << 32)
        + ((uint64_t) leadingDimL << 16) + ((uint64_t) startAddrL);
    uint64_t d_mxR = ((uint64_t) formatR << 62) + ((uint64_t) baseOffsetR << 49) + ((uint64_t) strideDimR << 32)
        + ((uint64_t) leadingDimR << 16) + ((uint64_t) startAddrR);

    using std::array;

    array<array<array<float, wmmaM_ * wmmaN_ / 32>, tileN_ / wmmaN_ / warpN_>, tileM_ / wmmaM_ / warpM_> r_mxAcc
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
                // dc & dA should be set to zero out of seq length, but this is done to
                // dc by chunkcumsum, so no need here.
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
#pragma unroll
            for (int iM = 0; iM < tileM_; iM += 64)
                asm volatile(
                    "@elected_one cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1, {%2, %3, %4}], [%5];\n" ::"r"(
                        b_mxL + iK.var % pipeS_ * (tileM_ * tileK_ * 2) + iM * tileK_ * 2),
                    "l"((CUtensorMap const*) g_mxX_), "r"(get(mStart * cn<tileM_>) + iM), "r"(gStart.var),
                    "r"(get(aStart + blockIdx_y * Q + iK * cn<tileK_>)), "r"(b_mbar + iK.var * 8));
#pragma unroll
            for (int iN = 0; iN < tileN_; iN += 64)
                asm volatile(
                    "@elected_one cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1, {%2, %3, %4}], [%5];\n" ::"r"(
                        b_mxR + iK.var % pipeS_ * (tileN_ * tileK_ * 2) + iN * tileK_ * 2),
                    "l"((CUtensorMap const*) g_mxX_ + 1), "r"(get(nStart * cn<tileN_>) + iN), "r"(hStart.var),
                    "r"(get(aStart + blockIdx_y * Q + iK * cn<tileK_>)), "r"(b_mbar + iK.var * 8));
        }
    }

    __syncthreads();

    for (int iK = pipeS_; iK < Q_ / tileK_ + pipeS_; iK++)
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
#pragma unroll
        for (Rn<UNROLL, div_up(tileM_ * tileK_, warpM_ * warpN_ * 256)> iStep; iStep.var < iStep.size; iStep.var++)
            if (thread(iStep) < cn<tileM_ * tileK_>)
            {
                Tp_ tmpL[8];

                *(int4*) &tmpL[0] = *(int4*) ((char*) s_mxL
                    + swizzle<128, 128>(
                        thread(iStep) * cn<2> / cn<tileM_ * 2> * cn<128> + thread(iStep) * cn<2> % cn<128>,
                        baseL(jK) * cn<2> + thread(iStep) * cn<2> % cn<tileM_ * 2> / cn<128> * cn<128 * tileK_>));

#pragma unroll
                for (int i = 0; i < 8; i += 2)
                {
                    float2 tmp2 = std::is_same_v<Tp_, half> ? __half22float2(*(half2*) &tmpL[i])
                                                            : bf1622float2(*(bf162*) &tmpL[i]);

                    int kStart = (iK - pipeS_) * cn<tileK_>;

                    // dc & dA should be set to zero out of seq length, but this is done to
                    // dc by chunkcumsum, so no need here.
                    tmp2.x *= expf(s_mxdA[Q_ - 1] - s_mxdA[kStart + get(thread(iStep) / cn<tileM_>)])
                        * s_mxdc[kStart + get(thread(iStep) / cn<tileM_>)];
                    tmp2.y *= expf(s_mxdA[Q_ - 1] - s_mxdA[kStart + get(thread(iStep) / cn<tileM_>)])
                        * s_mxdc[kStart + get(thread(iStep) / cn<tileM_>)];

                    if (std::is_same_v<Tp_, half>)
                        *(half2*) &tmpL[i] = __float22half2_rn(tmp2);
                    else
                        *(bf162*) &tmpL[i] = __float22bfloat162_rn(tmp2);
                }

                *(int4*) ((char*) s_mxL
                    + swizzle<128, 128>(
                        thread(iStep) * cn<2> / cn<tileM_ * 2> * cn<128> + thread(iStep) * cn<2> % cn<128>,
                        baseL(jK) * cn<2> + thread(iStep) * cn<2> % cn<tileM_ * 2> / cn<128> * cn<128 * tileK_>))
                    = *(int4*) &tmpL[0];
            }

        __syncthreads();

        asm volatile("fence.proxy.async.shared::cta; \n");
        asm volatile("wgmma.fence.sync.aligned; \n");

#pragma unroll
        for (int k = 0; k < tileK_ / wmmaK_; k++)
        {
#pragma unroll
            for (int y = 0; y < tileM_ / wmmaM_ / warpM_; y++)
            {
                wgmma<Tp_, 1, 1>(r_mxAcc[y], d_mxL0 + k * 128 + y * tileK_ * 128 * warpM_ / 4 / 16,
                    d_mxR0 + k * (tileN_ >= 64 ? 128 : 64));
            }
        }

        asm volatile("wgmma.commit_group.sync.aligned;");
        asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(0));

        __syncthreads();

        if (iK * tileK_ < Q_)
        {

            if (threadIdx.y == 0 && threadIdx.z == 0)
            {
                asm volatile("@elected_one mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n" ::"r"(
                                 b_mbar + jK.var % pipeS_ * 8),
                    "r"((tileM_ + tileN_) * tileK_ * 2));
#pragma unroll
                for (int iM = 0; iM < tileM_; iM += 64)
                    asm volatile(
                        "@elected_one cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes"
                        " [%0], [%1, {%2, %3, %4}], [%5];\n" ::"r"(
                            b_mxL + jK.var % pipeS_ * (tileM_ * tileK_ * 2) + iM * tileK_ * 2),
                        "l"((CUtensorMap const*) g_mxX_), "r"(get(mStart * cn<tileM_>) + iM), "r"(gStart.var),
                        "r"(get(aStart + blockIdx_y * Q + jK * cn<tileK_>)), "r"(b_mbar + jK.var % pipeS_ * 8));
#pragma unroll
                for (int iN = 0; iN < tileN_; iN += 64)
                    asm volatile(
                        "@elected_one cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes"
                        " [%0], [%1, {%2, %3, %4}], [%5];\n" ::"r"(
                            b_mxR + jK.var % pipeS_ * (tileN_ * tileK_ * 2) + iN * tileK_ * 2),
                        "l"((CUtensorMap const*) g_mxX_ + 1), "r"(get(nStart * cn<tileN_>) + iN), "r"(hStart.var),
                        "r"(get(aStart + blockIdx_y * Q + jK * cn<tileK_>)), "r"(b_mbar + jK.var % pipeS_ * 8));
            }
        }
    }

    if (threadIdx.y == 0 && threadIdx.z == 0)
    {
#pragma unroll
        for (Rn<UNROLL, pipeS_> iK; iK.var < iK.size; iK.var++)
            asm volatile("@elected_one mbarrier.inval.shared.b64 [%0];\n" ::"r"(b_mbar + iK.var * 8));
    }

#pragma unroll
    for (int y = 0; y < tileM_ / wmmaM_ / warpM_; y++)
#pragma unroll
        for (int x = 0; x < tileN_ / wmmaN_ / warpN_; x++)
            if (!bnChk_
                || nStart * cn<tileN_> + Rn<UNROLL>{x} * cn<wmmaN_> + threadIdx_z * cn<tileN_ / warpN_>
                        + threadIdx_x % cn<4> * cn<2>
                    < P)
            {
                *(float2*) (g_mxSt
                    + get(hStart * N * P
                        + (mStart * cn<tileM_> + Rn<UNROLL>{y} * cn<warpM_ * wmmaM_> + threadIdx_y * cn<wmmaM_>
                            + threadIdx_x / cn<4>) *P
                        + nStart * cn<tileN_> + Rn<UNROLL>{x} * cn<wmmaN_> + threadIdx_z * cn<tileN_ / warpN_>
                        + threadIdx_x % cn<4> * cn<2>))
                    = *(float2*) &r_mxAcc[y][x][0];

                *(float2*) (g_mxSt
                    + get(hStart * N * P
                        + (mStart * cn<tileM_> + Rn<UNROLL>{y} * cn<warpM_ * wmmaM_> + cn<8> + threadIdx_y * cn<wmmaM_>
                            + threadIdx_x / cn<4>) *P
                        + nStart * cn<tileN_> + Rn<UNROLL>{x} * cn<wmmaN_> + threadIdx_z * cn<tileN_ / warpN_>
                        + threadIdx_x % cn<4> * cn<2>))
                    = *(float2*) &r_mxAcc[y][x][2];
            }
#endif
}

typedef ChunkStateKernelFunc (*GetChunkStateKernelFunc)(int B_, int L_, int H_, int P_, int G_, int N_, int Q_,
    int numTokens_, int isHopper_, tensorrt_llm::common::CUDADriverWrapper* cudaDriver_, dim3* blockDims_,
    dim3* threadDims_, int* sharedMem_, int* useTma_, CUtensorMap* descs_);

template <class Tp_>
ChunkStateKernelFunc getChunkStateKernel(int B_, int L_, int H_, int P_, int G_, int N_, int Q_, int numTokens_,
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

    int64_t compute = int64_t(numTokens_) * H * N * P_ * Q;

    auto set
        = [&](int tileM, int tileN, int tileK, int warpM, int warpN, int pipeS, int useTma, ChunkStateKernelFunc func)
    {
        auto sharedMem = useTma * 1024 + (tileM * tileK + tileK * tileN) * pipeS * 2 + Q * 8;

        *blockDims_ = dim3(H * div_up(P_, tileN) * N / tileM, C, B);
        *threadDims_ = dim3(32, useTma ? warpM : warpN, useTma ? warpN : warpM);
        *sharedMem_ = sharedMem;
        *useTma_ = useTma;

        if (useTma)
        {
            {
                std::array<uint64_t, 2> tensorStrides{2uL * N, 2uL * (H * P_ + 2 * G * N)};
                std::array<uint64_t, 3> tensorSizes{1uL * N, 1uL * G, 1uL * numTokens_};
                std::array<uint32_t, 3> traveralStrides{1u, 1u, 1u};
                std::array<uint32_t, 3> boxSizes{64u, 1u, (uint32_t) tileK};

                cudaDriver_->cuTensorMapEncodeTiled(&descs_[0], CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 3, nullptr,
                    &tensorSizes[0], &tensorStrides[0], &boxSizes[0], &traveralStrides[0],
                    CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
            }
            {
                std::array<uint64_t, 2> tensorStrides{2uL * P_, 2uL * (H * P_ + 2 * G * N)};
                std::array<uint64_t, 3> tensorSizes{1uL * P_, 1uL * H, 1uL * numTokens_};
                std::array<uint32_t, 3> traveralStrides{1u, 1u, 1u};
                std::array<uint32_t, 3> boxSizes{tileN >= 64 ? 64u : (uint32_t) tileN, 1u, (uint32_t) tileK};

                cudaDriver_->cuTensorMapEncodeTiled(&descs_[1], CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 3, nullptr,
                    &tensorSizes[0], &tensorStrides[0], &boxSizes[0], &traveralStrides[0],
                    CU_TENSOR_MAP_INTERLEAVE_NONE, tileN >= 64 ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_64B,
                    CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
            }
        }

        return func;
    };

    if (Q == 256 && isHopper_ == false)
    {
#ifndef FAST_BUILD
        if (P == 32 && N % 256 == 0)
        {
            if (compute >= (1LL << 38))
                return set(64, 32, 32, 2, 1, 2, 0, chunk_state_kernel<256, 64, 32, 32, 2, 1, 2, 1, 1, 1, Tp_>);
            else if (compute >= (1LL << 35))
                return set(64, 32, 64, 2, 2, 2, 0, chunk_state_kernel<256, 64, 32, 64, 2, 2, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 33))
                return set(64, 32, 64, 2, 2, 4, 0, chunk_state_kernel<256, 64, 32, 64, 2, 2, 4, 2, 1, 1, Tp_>);
            else
                return set(64, 32, 64, 2, 2, 2, 0, chunk_state_kernel<256, 64, 32, 64, 2, 2, 2, 1, 8, 1, Tp_>);
        }
        if (P == 32 && N % 128 == 0)
        {
            if (compute >= (1LL << 38))
                return set(64, 32, 32, 2, 1, 2, 0, chunk_state_kernel<256, 64, 32, 32, 2, 1, 2, 1, 1, 1, Tp_>);
            else if (compute >= (1LL << 35))
                return set(64, 32, 64, 2, 2, 2, 0, chunk_state_kernel<256, 64, 32, 64, 2, 2, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 33))
                return set(64, 32, 64, 2, 2, 4, 0, chunk_state_kernel<256, 64, 32, 64, 2, 2, 4, 2, 1, 1, Tp_>);
            else
                return set(64, 32, 64, 2, 2, 2, 0, chunk_state_kernel<256, 64, 32, 64, 2, 2, 2, 1, 8, 1, Tp_>);
        }
        if (P == 32 && N % 64 == 0)
        {
            if (compute >= (1LL << 38))
                return set(64, 32, 32, 2, 1, 2, 0, chunk_state_kernel<256, 64, 32, 32, 2, 1, 2, 1, 1, 1, Tp_>);
            else if (compute >= (1LL << 35))
                return set(64, 32, 64, 2, 2, 2, 0, chunk_state_kernel<256, 64, 32, 64, 2, 2, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 33))
                return set(64, 32, 64, 2, 2, 4, 0, chunk_state_kernel<256, 64, 32, 64, 2, 2, 4, 2, 1, 1, Tp_>);
            else
                return set(64, 32, 64, 2, 2, 2, 0, chunk_state_kernel<256, 64, 32, 64, 2, 2, 2, 1, 8, 1, Tp_>);
        }
        if (P % 256 == 0 && N % 256 == 0)
        {
            if (compute >= (1LL << 44))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 2, 1, 2, 1, Tp_>);
            else if (compute >= (1LL << 43))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 42))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 1, 1, Tp_>);
            else if (compute >= (1LL << 39))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 38))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 1, 1, Tp_>);
            else if (compute >= (1LL << 37))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 2, 1, 2, 1, Tp_>);
            else if (compute >= (1LL << 36))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 1, 1, Tp_>);
            else if (compute >= (1LL << 35))
                return set(64, 64, 32, 2, 2, 2, 0, chunk_state_kernel<256, 64, 64, 32, 2, 2, 2, 2, 2, 1, Tp_>);
            else
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 4, 1, Tp_>);
        }
        if (P % 256 == 0 && N % 128 == 0)
        {
            if (compute >= (1LL << 44))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 2, 1, 2, 1, Tp_>);
            else if (compute >= (1LL << 43))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 42))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 1, 1, Tp_>);
            else if (compute >= (1LL << 39))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 38))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 1, 1, Tp_>);
            else if (compute >= (1LL << 37))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 2, 1, 2, 1, Tp_>);
            else if (compute >= (1LL << 36))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 1, 1, Tp_>);
            else if (compute >= (1LL << 35))
                return set(64, 64, 32, 2, 2, 2, 0, chunk_state_kernel<256, 64, 64, 32, 2, 2, 2, 2, 2, 1, Tp_>);
            else
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 4, 1, Tp_>);
        }
        if (P % 256 == 0 && N % 64 == 0)
        {
            if (compute >= (1LL << 44))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 2, 1, 2, 1, Tp_>);
            else if (compute >= (1LL << 43))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 42))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 1, 1, Tp_>);
            else if (compute >= (1LL << 39))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 38))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 1, 1, Tp_>);
            else if (compute >= (1LL << 37))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 2, 1, 2, 1, Tp_>);
            else if (compute >= (1LL << 36))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 1, 1, Tp_>);
            else if (compute >= (1LL << 35))
                return set(64, 64, 32, 2, 2, 2, 0, chunk_state_kernel<256, 64, 64, 32, 2, 2, 2, 2, 2, 1, Tp_>);
            else
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 4, 1, Tp_>);
        }
        if (P % 128 == 0 && N % 256 == 0)
        {
            if (compute >= (1LL << 44))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 2, 1, 2, 1, Tp_>);
            else if (compute >= (1LL << 43))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 42))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 1, 1, Tp_>);
            else if (compute >= (1LL << 38))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 34))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 2, 1, 2, 1, Tp_>);
            else
                return set(64, 64, 64, 2, 2, 4, 0, chunk_state_kernel<256, 64, 64, 64, 2, 2, 4, 2, 1, 1, Tp_>);
        }
        if (P % 128 == 0 && N % 128 == 0)
        {
            if (compute >= (1LL << 44))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 2, 1, 2, 1, Tp_>);
            else if (compute >= (1LL << 43))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 42))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 1, 1, Tp_>);
            else if (compute >= (1LL << 38))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 34))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 2, 1, 2, 1, Tp_>);
            else
                return set(64, 64, 64, 2, 2, 4, 0, chunk_state_kernel<256, 64, 64, 64, 2, 2, 4, 2, 1, 1, Tp_>);
        }
        if (P % 128 == 0 && N % 64 == 0)
        {
            if (compute >= (1LL << 44))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 2, 1, 2, 1, Tp_>);
            else if (compute >= (1LL << 43))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 42))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 1, 1, Tp_>);
            else if (compute >= (1LL << 38))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 34))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 2, 1, 2, 1, Tp_>);
            else
                return set(64, 64, 64, 2, 2, 4, 0, chunk_state_kernel<256, 64, 64, 64, 2, 2, 4, 2, 1, 1, Tp_>);
        }
        if (P % 64 == 0 && N % 256 == 0)
        {
            if (compute >= (1LL << 39))
                return set(64, 64, 32, 1, 2, 2, 0, chunk_state_kernel<256, 64, 64, 32, 1, 2, 2, 1, 1, 1, Tp_>);
            else if (compute >= (1LL << 36))
                return set(64, 64, 32, 2, 2, 2, 0, chunk_state_kernel<256, 64, 64, 32, 2, 2, 2, 2, 2, 1, Tp_>);
            else if (compute >= (1LL << 35))
                return set(64, 64, 64, 2, 2, 2, 0, chunk_state_kernel<256, 64, 64, 64, 2, 2, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 34))
                return set(64, 64, 32, 2, 2, 2, 0, chunk_state_kernel<256, 64, 64, 32, 2, 2, 2, 2, 2, 1, Tp_>);
            else if (compute >= (1LL << 33))
                return set(64, 64, 64, 2, 2, 4, 0, chunk_state_kernel<256, 64, 64, 64, 2, 2, 4, 2, 1, 1, Tp_>);
            else
                return set(64, 64, 64, 2, 2, 2, 0, chunk_state_kernel<256, 64, 64, 64, 2, 2, 2, 1, 8, 1, Tp_>);
        }
        if (P % 64 == 0 && N % 128 == 0)
        {
            if (compute >= (1LL << 39))
                return set(64, 64, 32, 1, 2, 2, 0, chunk_state_kernel<256, 64, 64, 32, 1, 2, 2, 1, 1, 1, Tp_>);
            else if (compute >= (1LL << 36))
                return set(64, 64, 32, 2, 2, 2, 0, chunk_state_kernel<256, 64, 64, 32, 2, 2, 2, 2, 2, 1, Tp_>);
            else if (compute >= (1LL << 35))
                return set(64, 64, 64, 2, 2, 2, 0, chunk_state_kernel<256, 64, 64, 64, 2, 2, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 34))
                return set(64, 64, 32, 2, 2, 2, 0, chunk_state_kernel<256, 64, 64, 32, 2, 2, 2, 2, 2, 1, Tp_>);
            else if (compute >= (1LL << 33))
                return set(64, 64, 64, 2, 2, 4, 0, chunk_state_kernel<256, 64, 64, 64, 2, 2, 4, 2, 1, 1, Tp_>);
            else
                return set(64, 64, 64, 2, 2, 2, 0, chunk_state_kernel<256, 64, 64, 64, 2, 2, 2, 1, 8, 1, Tp_>);
        }
#endif
        if (P % 64 == 0 && N % 64 == 0)
        {
            if (compute >= (1LL << 39))
                return set(64, 64, 32, 1, 2, 2, 0, chunk_state_kernel<256, 64, 64, 32, 1, 2, 2, 1, 1, 1, Tp_>);
            else if (compute >= (1LL << 36))
                return set(64, 64, 32, 2, 2, 2, 0, chunk_state_kernel<256, 64, 64, 32, 2, 2, 2, 2, 2, 1, Tp_>);
            else if (compute >= (1LL << 35))
                return set(64, 64, 64, 2, 2, 2, 0, chunk_state_kernel<256, 64, 64, 64, 2, 2, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 34))
                return set(64, 64, 32, 2, 2, 2, 0, chunk_state_kernel<256, 64, 64, 32, 2, 2, 2, 2, 2, 1, Tp_>);
            else if (compute >= (1LL << 33))
                return set(64, 64, 64, 2, 2, 4, 0, chunk_state_kernel<256, 64, 64, 64, 2, 2, 4, 2, 1, 1, Tp_>);
            else
                return set(64, 64, 64, 2, 2, 2, 0, chunk_state_kernel<256, 64, 64, 64, 2, 2, 2, 1, 8, 1, Tp_>);
        }
        if (round_up(P, 32) == 32)
            P = 32;
        else
            P = round_up(P, 64);
#ifndef FAST_BUILD
        if (P == 32 && N % 256 == 0)
        {
            if (compute >= (1LL << 38))
                return set(64, 32, 32, 2, 1, 2, 0, chunk_state_kernel<256, 64, 32, 32, 2, 1, 2, 1, 1, 1, Tp_>);
            else if (compute >= (1LL << 35))
                return set(64, 32, 64, 2, 2, 2, 0, chunk_state_kernel<256, 64, 32, 64, 2, 2, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 33))
                return set(64, 32, 64, 2, 2, 4, 0, chunk_state_kernel<256, 64, 32, 64, 2, 2, 4, 2, 1, 1, Tp_>);
            else
                return set(64, 32, 64, 2, 2, 2, 0, chunk_state_kernel<256, 64, 32, 64, 2, 2, 2, 1, 8, 1, Tp_>);
        }
        if (P == 32 && N % 128 == 0)
        {
            if (compute >= (1LL << 38))
                return set(64, 32, 32, 2, 1, 2, 0, chunk_state_kernel<256, 64, 32, 32, 2, 1, 2, 1, 1, 1, Tp_>);
            else if (compute >= (1LL << 35))
                return set(64, 32, 64, 2, 2, 2, 0, chunk_state_kernel<256, 64, 32, 64, 2, 2, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 33))
                return set(64, 32, 64, 2, 2, 4, 0, chunk_state_kernel<256, 64, 32, 64, 2, 2, 4, 2, 1, 1, Tp_>);
            else
                return set(64, 32, 64, 2, 2, 2, 0, chunk_state_kernel<256, 64, 32, 64, 2, 2, 2, 1, 8, 1, Tp_>);
        }
        if (P == 32 && N % 64 == 0)
        {
            if (compute >= (1LL << 38))
                return set(64, 32, 32, 2, 1, 2, 0, chunk_state_kernel<256, 64, 32, 32, 2, 1, 2, 1, 1, 1, Tp_>);
            else if (compute >= (1LL << 35))
                return set(64, 32, 64, 2, 2, 2, 0, chunk_state_kernel<256, 64, 32, 64, 2, 2, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 33))
                return set(64, 32, 64, 2, 2, 4, 0, chunk_state_kernel<256, 64, 32, 64, 2, 2, 4, 2, 1, 1, Tp_>);
            else
                return set(64, 32, 64, 2, 2, 2, 0, chunk_state_kernel<256, 64, 32, 64, 2, 2, 2, 1, 8, 1, Tp_>);
        }
        if (P % 256 == 0 && N % 256 == 0)
        {
            if (compute >= (1LL << 44))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 2, 1, 2, 1, Tp_>);
            else if (compute >= (1LL << 43))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 42))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 1, 1, Tp_>);
            else if (compute >= (1LL << 39))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 38))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 1, 1, Tp_>);
            else if (compute >= (1LL << 37))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 2, 1, 2, 1, Tp_>);
            else if (compute >= (1LL << 36))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 1, 1, Tp_>);
            else if (compute >= (1LL << 35))
                return set(64, 64, 32, 2, 2, 2, 0, chunk_state_kernel<256, 64, 64, 32, 2, 2, 2, 2, 2, 1, Tp_>);
            else
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 4, 1, Tp_>);
        }
        if (P % 256 == 0 && N % 128 == 0)
        {
            if (compute >= (1LL << 44))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 2, 1, 2, 1, Tp_>);
            else if (compute >= (1LL << 43))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 42))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 1, 1, Tp_>);
            else if (compute >= (1LL << 39))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 38))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 1, 1, Tp_>);
            else if (compute >= (1LL << 37))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 2, 1, 2, 1, Tp_>);
            else if (compute >= (1LL << 36))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 1, 1, Tp_>);
            else if (compute >= (1LL << 35))
                return set(64, 64, 32, 2, 2, 2, 0, chunk_state_kernel<256, 64, 64, 32, 2, 2, 2, 2, 2, 1, Tp_>);
            else
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 4, 1, Tp_>);
        }
        if (P % 256 == 0 && N % 64 == 0)
        {
            if (compute >= (1LL << 44))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 2, 1, 2, 1, Tp_>);
            else if (compute >= (1LL << 43))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 42))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 1, 1, Tp_>);
            else if (compute >= (1LL << 39))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 38))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 1, 1, Tp_>);
            else if (compute >= (1LL << 37))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 2, 1, 2, 1, Tp_>);
            else if (compute >= (1LL << 36))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 1, 1, Tp_>);
            else if (compute >= (1LL << 35))
                return set(64, 64, 32, 2, 2, 2, 0, chunk_state_kernel<256, 64, 64, 32, 2, 2, 2, 2, 2, 1, Tp_>);
            else
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 4, 1, Tp_>);
        }
        if (P % 128 == 0 && N % 256 == 0)
        {
            if (compute >= (1LL << 44))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 2, 1, 2, 1, Tp_>);
            else if (compute >= (1LL << 42))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 41))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 2, 1, 2, 1, Tp_>);
            else if (compute >= (1LL << 40))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 39))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 1, 1, Tp_>);
            else if (compute >= (1LL << 38))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 34))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 2, 1, 2, 1, Tp_>);
            else if (compute >= (1LL << 33))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 32))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 2, 1, 2, 1, Tp_>);
            else
                return set(64, 64, 32, 2, 2, 2, 0, chunk_state_kernel<256, 64, 64, 32, 2, 2, 2, 2, 2, 1, Tp_>);
        }
        if (P % 128 == 0 && N % 128 == 0)
        {
            if (compute >= (1LL << 44))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 2, 1, 2, 1, Tp_>);
            else if (compute >= (1LL << 42))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 41))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 2, 1, 2, 1, Tp_>);
            else if (compute >= (1LL << 40))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 39))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 1, 1, Tp_>);
            else if (compute >= (1LL << 38))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 34))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 2, 1, 2, 1, Tp_>);
            else if (compute >= (1LL << 33))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 32))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 2, 1, 2, 1, Tp_>);
            else
                return set(64, 64, 32, 2, 2, 2, 0, chunk_state_kernel<256, 64, 64, 32, 2, 2, 2, 2, 2, 1, Tp_>);
        }
        if (P % 128 == 0 && N % 64 == 0)
        {
            if (compute >= (1LL << 44))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 2, 1, 2, 1, Tp_>);
            else if (compute >= (1LL << 42))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 41))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 2, 1, 2, 1, Tp_>);
            else if (compute >= (1LL << 40))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 39))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 1, 1, Tp_>);
            else if (compute >= (1LL << 38))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 34))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 2, 1, 2, 1, Tp_>);
            else if (compute >= (1LL << 33))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 32))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 2, 1, 2, 1, Tp_>);
            else
                return set(64, 64, 32, 2, 2, 2, 0, chunk_state_kernel<256, 64, 64, 32, 2, 2, 2, 2, 2, 1, Tp_>);
        }
        if (P % 64 == 0 && N % 256 == 0)
        {
            if (compute >= (1LL << 39))
                return set(64, 64, 32, 1, 2, 2, 0, chunk_state_kernel<256, 64, 64, 32, 1, 2, 2, 1, 1, 1, Tp_>);
            else if (compute >= (1LL << 36))
                return set(64, 64, 32, 2, 2, 2, 0, chunk_state_kernel<256, 64, 64, 32, 2, 2, 2, 2, 2, 1, Tp_>);
            else if (compute >= (1LL << 35))
                return set(64, 64, 64, 2, 2, 2, 0, chunk_state_kernel<256, 64, 64, 64, 2, 2, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 34))
                return set(64, 64, 64, 2, 2, 4, 0, chunk_state_kernel<256, 64, 64, 64, 2, 2, 4, 2, 1, 1, Tp_>);
            else if (compute >= (1LL << 32))
                return set(64, 64, 32, 2, 2, 2, 0, chunk_state_kernel<256, 64, 64, 32, 2, 2, 2, 2, 2, 1, Tp_>);
            else if (compute >= (1LL << 30))
                return set(64, 64, 64, 2, 2, 4, 0, chunk_state_kernel<256, 64, 64, 64, 2, 2, 4, 2, 1, 1, Tp_>);
            else
                return set(64, 64, 32, 2, 2, 2, 0, chunk_state_kernel<256, 64, 64, 32, 2, 2, 2, 2, 2, 1, Tp_>);
        }
        if (P % 64 == 0 && N % 128 == 0)
        {
            if (compute >= (1LL << 39))
                return set(64, 64, 32, 1, 2, 2, 0, chunk_state_kernel<256, 64, 64, 32, 1, 2, 2, 1, 1, 1, Tp_>);
            else if (compute >= (1LL << 36))
                return set(64, 64, 32, 2, 2, 2, 0, chunk_state_kernel<256, 64, 64, 32, 2, 2, 2, 2, 2, 1, Tp_>);
            else if (compute >= (1LL << 35))
                return set(64, 64, 64, 2, 2, 2, 0, chunk_state_kernel<256, 64, 64, 64, 2, 2, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 34))
                return set(64, 64, 64, 2, 2, 4, 0, chunk_state_kernel<256, 64, 64, 64, 2, 2, 4, 2, 1, 1, Tp_>);
            else if (compute >= (1LL << 32))
                return set(64, 64, 32, 2, 2, 2, 0, chunk_state_kernel<256, 64, 64, 32, 2, 2, 2, 2, 2, 1, Tp_>);
            else if (compute >= (1LL << 30))
                return set(64, 64, 64, 2, 2, 4, 0, chunk_state_kernel<256, 64, 64, 64, 2, 2, 4, 2, 1, 1, Tp_>);
            else
                return set(64, 64, 32, 2, 2, 2, 0, chunk_state_kernel<256, 64, 64, 32, 2, 2, 2, 2, 2, 1, Tp_>);
        }
#endif
        if (P % 64 == 0 && N % 64 == 0)
        {
            if (compute >= (1LL << 39))
                return set(64, 64, 32, 1, 2, 2, 0, chunk_state_kernel<256, 64, 64, 32, 1, 2, 2, 1, 1, 1, Tp_>);
            else if (compute >= (1LL << 36))
                return set(64, 64, 32, 2, 2, 2, 0, chunk_state_kernel<256, 64, 64, 32, 2, 2, 2, 2, 2, 1, Tp_>);
            else if (compute >= (1LL << 35))
                return set(64, 64, 64, 2, 2, 2, 0, chunk_state_kernel<256, 64, 64, 64, 2, 2, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 34))
                return set(64, 64, 64, 2, 2, 4, 0, chunk_state_kernel<256, 64, 64, 64, 2, 2, 4, 2, 1, 1, Tp_>);
            else if (compute >= (1LL << 32))
                return set(64, 64, 32, 2, 2, 2, 0, chunk_state_kernel<256, 64, 64, 32, 2, 2, 2, 2, 2, 1, Tp_>);
            else if (compute >= (1LL << 30))
                return set(64, 64, 64, 2, 2, 4, 0, chunk_state_kernel<256, 64, 64, 64, 2, 2, 4, 2, 1, 1, Tp_>);
            else
                return set(64, 64, 32, 2, 2, 2, 0, chunk_state_kernel<256, 64, 64, 32, 2, 2, 2, 2, 2, 1, Tp_>);
        }
    }
    else if (Q == 256 && isHopper_)
    {
#ifndef FAST_BUILD
        if (P == 32 && N % 256 == 0)
        {
            if (compute >= (1LL << 39))
                return set(64, 32, 64, 4, 1, 2, 1, chunk_state_hopper<256, 64, 32, 64, 4, 1, 2, 0, 1, 1, Tp_>);
            else if (compute >= (1LL << 38))
                return set(64, 32, 32, 2, 1, 2, 0, chunk_state_kernel<256, 64, 32, 32, 2, 1, 2, 1, 1, 1, Tp_>);
            else if (compute >= (1LL << 36))
                return set(64, 32, 64, 2, 2, 2, 0, chunk_state_kernel<256, 64, 32, 64, 2, 2, 2, 1, 8, 1, Tp_>);
            else
                return set(64, 32, 64, 2, 2, 4, 0, chunk_state_kernel<256, 64, 32, 64, 2, 2, 4, 2, 1, 1, Tp_>);
        }
        if (P == 32 && N % 128 == 0)
        {
            if (compute >= (1LL << 39))
                return set(64, 32, 64, 4, 1, 2, 1, chunk_state_hopper<256, 64, 32, 64, 4, 1, 2, 0, 1, 1, Tp_>);
            else if (compute >= (1LL << 38))
                return set(64, 32, 32, 2, 1, 2, 0, chunk_state_kernel<256, 64, 32, 32, 2, 1, 2, 1, 1, 1, Tp_>);
            else if (compute >= (1LL << 36))
                return set(64, 32, 64, 2, 2, 2, 0, chunk_state_kernel<256, 64, 32, 64, 2, 2, 2, 1, 8, 1, Tp_>);
            else
                return set(64, 32, 64, 2, 2, 4, 0, chunk_state_kernel<256, 64, 32, 64, 2, 2, 4, 2, 1, 1, Tp_>);
        }
        if (P == 32 && N % 64 == 0)
        {
            if (compute >= (1LL << 39))
                return set(64, 32, 64, 4, 1, 2, 1, chunk_state_hopper<256, 64, 32, 64, 4, 1, 2, 0, 1, 1, Tp_>);
            else if (compute >= (1LL << 38))
                return set(64, 32, 32, 2, 1, 2, 0, chunk_state_kernel<256, 64, 32, 32, 2, 1, 2, 1, 1, 1, Tp_>);
            else if (compute >= (1LL << 36))
                return set(64, 32, 64, 2, 2, 2, 0, chunk_state_kernel<256, 64, 32, 64, 2, 2, 2, 1, 8, 1, Tp_>);
            else
                return set(64, 32, 64, 2, 2, 4, 0, chunk_state_kernel<256, 64, 32, 64, 2, 2, 4, 2, 1, 1, Tp_>);
        }
        if (P % 256 == 0 && N % 256 == 0)
        {
            if (compute >= (1LL << 41))
                return set(64, 256, 32, 4, 1, 2, 1, chunk_state_hopper<256, 64, 256, 32, 4, 1, 2, 0, 4, 1, Tp_>);
            else if (compute >= (1LL << 39))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 35))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 1, 1, Tp_>);
            else if (compute >= (1LL << 34))
                return set(64, 64, 64, 2, 2, 4, 0, chunk_state_kernel<256, 64, 64, 64, 2, 2, 4, 2, 1, 1, Tp_>);
            else
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 1, 1, Tp_>);
        }
        if (P % 256 == 0 && N % 128 == 0)
        {
            if (compute >= (1LL << 41))
                return set(64, 256, 32, 4, 1, 2, 1, chunk_state_hopper<256, 64, 256, 32, 4, 1, 2, 0, 4, 1, Tp_>);
            else if (compute >= (1LL << 39))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 35))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 1, 1, Tp_>);
            else if (compute >= (1LL << 34))
                return set(64, 64, 64, 2, 2, 4, 0, chunk_state_kernel<256, 64, 64, 64, 2, 2, 4, 2, 1, 1, Tp_>);
            else
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 1, 1, Tp_>);
        }
        if (P % 256 == 0 && N % 64 == 0)
        {
            if (compute >= (1LL << 41))
                return set(64, 256, 32, 4, 1, 2, 1, chunk_state_hopper<256, 64, 256, 32, 4, 1, 2, 0, 4, 1, Tp_>);
            else if (compute >= (1LL << 39))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 35))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 1, 1, Tp_>);
            else if (compute >= (1LL << 34))
                return set(64, 64, 64, 2, 2, 4, 0, chunk_state_kernel<256, 64, 64, 64, 2, 2, 4, 2, 1, 1, Tp_>);
            else
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 1, 1, Tp_>);
        }
        if (P % 128 == 0 && N % 256 == 0)
        {
            if (compute >= (1LL << 41))
                return set(64, 128, 64, 4, 1, 2, 1, chunk_state_hopper<256, 64, 128, 64, 4, 1, 2, 0, 2, 1, Tp_>);
            else if (compute >= (1LL << 39))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 35))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 1, 1, Tp_>);
            else if (compute >= (1LL << 33))
                return set(64, 64, 64, 2, 2, 4, 0, chunk_state_kernel<256, 64, 64, 64, 2, 2, 4, 2, 1, 1, Tp_>);
            else
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 1, 1, Tp_>);
        }
        if (P % 128 == 0 && N % 128 == 0)
        {
            if (compute >= (1LL << 41))
                return set(64, 128, 64, 4, 1, 2, 1, chunk_state_hopper<256, 64, 128, 64, 4, 1, 2, 0, 2, 1, Tp_>);
            else if (compute >= (1LL << 39))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 35))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 1, 1, Tp_>);
            else if (compute >= (1LL << 33))
                return set(64, 64, 64, 2, 2, 4, 0, chunk_state_kernel<256, 64, 64, 64, 2, 2, 4, 2, 1, 1, Tp_>);
            else
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 1, 1, Tp_>);
        }
        if (P % 128 == 0 && N % 64 == 0)
        {
            if (compute >= (1LL << 41))
                return set(64, 128, 64, 4, 1, 2, 1, chunk_state_hopper<256, 64, 128, 64, 4, 1, 2, 0, 2, 1, Tp_>);
            else if (compute >= (1LL << 39))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 35))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 1, 1, Tp_>);
            else if (compute >= (1LL << 33))
                return set(64, 64, 64, 2, 2, 4, 0, chunk_state_kernel<256, 64, 64, 64, 2, 2, 4, 2, 1, 1, Tp_>);
            else
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 1, 1, Tp_>);
        }
        if (P % 64 == 0 && N % 256 == 0)
        {
            if (compute >= (1LL << 40))
                return set(64, 64, 32, 4, 1, 3, 1, chunk_state_hopper<256, 64, 64, 32, 4, 1, 3, 0, 4, 1, Tp_>);
            else if (compute >= (1LL << 37))
                return set(64, 64, 32, 1, 2, 2, 0, chunk_state_kernel<256, 64, 64, 32, 1, 2, 2, 1, 1, 1, Tp_>);
            else if (compute >= (1LL << 36))
                return set(64, 64, 64, 2, 2, 2, 0, chunk_state_kernel<256, 64, 64, 64, 2, 2, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 33))
                return set(64, 64, 64, 2, 2, 4, 0, chunk_state_kernel<256, 64, 64, 64, 2, 2, 4, 2, 1, 1, Tp_>);
            else
                return set(64, 64, 64, 2, 2, 2, 0, chunk_state_kernel<256, 64, 64, 64, 2, 2, 2, 1, 8, 1, Tp_>);
        }
        if (P % 64 == 0 && N % 128 == 0)
        {
            if (compute >= (1LL << 40))
                return set(64, 64, 32, 4, 1, 3, 1, chunk_state_hopper<256, 64, 64, 32, 4, 1, 3, 0, 4, 1, Tp_>);
            else if (compute >= (1LL << 37))
                return set(64, 64, 32, 1, 2, 2, 0, chunk_state_kernel<256, 64, 64, 32, 1, 2, 2, 1, 1, 1, Tp_>);
            else if (compute >= (1LL << 36))
                return set(64, 64, 64, 2, 2, 2, 0, chunk_state_kernel<256, 64, 64, 64, 2, 2, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 33))
                return set(64, 64, 64, 2, 2, 4, 0, chunk_state_kernel<256, 64, 64, 64, 2, 2, 4, 2, 1, 1, Tp_>);
            else
                return set(64, 64, 64, 2, 2, 2, 0, chunk_state_kernel<256, 64, 64, 64, 2, 2, 2, 1, 8, 1, Tp_>);
        }
#endif
        if (P % 64 == 0 && N % 64 == 0)
        {
            if (compute >= (1LL << 40))
                return set(64, 64, 32, 4, 1, 3, 1, chunk_state_hopper<256, 64, 64, 32, 4, 1, 3, 0, 4, 1, Tp_>);
            else if (compute >= (1LL << 37))
                return set(64, 64, 32, 1, 2, 2, 0, chunk_state_kernel<256, 64, 64, 32, 1, 2, 2, 1, 1, 1, Tp_>);
            else if (compute >= (1LL << 36))
                return set(64, 64, 64, 2, 2, 2, 0, chunk_state_kernel<256, 64, 64, 64, 2, 2, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 33))
                return set(64, 64, 64, 2, 2, 4, 0, chunk_state_kernel<256, 64, 64, 64, 2, 2, 4, 2, 1, 1, Tp_>);
            else
                return set(64, 64, 64, 2, 2, 2, 0, chunk_state_kernel<256, 64, 64, 64, 2, 2, 2, 1, 8, 1, Tp_>);
        }
        if (round_up(P, 32) == 32)
            P = 32;
        else
            P = round_up(P, 64);
#ifndef FAST_BUILD
        if (P == 32 && N % 256 == 0)
        {
            if (compute >= (1LL << 39))
                return set(64, 32, 64, 4, 1, 2, 1, chunk_state_hopper<256, 64, 32, 64, 4, 1, 2, 0, 1, 1, Tp_>);
            else if (compute >= (1LL << 38))
                return set(64, 32, 32, 2, 1, 2, 0, chunk_state_kernel<256, 64, 32, 32, 2, 1, 2, 1, 1, 1, Tp_>);
            else if (compute >= (1LL << 36))
                return set(64, 32, 64, 2, 2, 2, 0, chunk_state_kernel<256, 64, 32, 64, 2, 2, 2, 1, 8, 1, Tp_>);
            else
                return set(64, 32, 64, 2, 2, 4, 0, chunk_state_kernel<256, 64, 32, 64, 2, 2, 4, 2, 1, 1, Tp_>);
        }
        if (P == 32 && N % 128 == 0)
        {
            if (compute >= (1LL << 39))
                return set(64, 32, 64, 4, 1, 2, 1, chunk_state_hopper<256, 64, 32, 64, 4, 1, 2, 0, 1, 1, Tp_>);
            else if (compute >= (1LL << 38))
                return set(64, 32, 32, 2, 1, 2, 0, chunk_state_kernel<256, 64, 32, 32, 2, 1, 2, 1, 1, 1, Tp_>);
            else if (compute >= (1LL << 36))
                return set(64, 32, 64, 2, 2, 2, 0, chunk_state_kernel<256, 64, 32, 64, 2, 2, 2, 1, 8, 1, Tp_>);
            else
                return set(64, 32, 64, 2, 2, 4, 0, chunk_state_kernel<256, 64, 32, 64, 2, 2, 4, 2, 1, 1, Tp_>);
        }
        if (P == 32 && N % 64 == 0)
        {
            if (compute >= (1LL << 39))
                return set(64, 32, 64, 4, 1, 2, 1, chunk_state_hopper<256, 64, 32, 64, 4, 1, 2, 0, 1, 1, Tp_>);
            else if (compute >= (1LL << 38))
                return set(64, 32, 32, 2, 1, 2, 0, chunk_state_kernel<256, 64, 32, 32, 2, 1, 2, 1, 1, 1, Tp_>);
            else if (compute >= (1LL << 36))
                return set(64, 32, 64, 2, 2, 2, 0, chunk_state_kernel<256, 64, 32, 64, 2, 2, 2, 1, 8, 1, Tp_>);
            else
                return set(64, 32, 64, 2, 2, 4, 0, chunk_state_kernel<256, 64, 32, 64, 2, 2, 4, 2, 1, 1, Tp_>);
        }
        if (P % 256 == 0 && N % 256 == 0)
        {
            if (compute >= (1LL << 41))
                return set(64, 256, 32, 4, 1, 2, 1, chunk_state_hopper<256, 64, 256, 32, 4, 1, 2, 0, 4, 1, Tp_>);
            else if (compute >= (1LL << 39))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 35))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 1, 1, Tp_>);
            else if (compute >= (1LL << 34))
                return set(64, 64, 64, 2, 2, 4, 0, chunk_state_kernel<256, 64, 64, 64, 2, 2, 4, 2, 1, 1, Tp_>);
            else
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 1, 1, Tp_>);
        }
        if (P % 256 == 0 && N % 128 == 0)
        {
            if (compute >= (1LL << 41))
                return set(64, 256, 32, 4, 1, 2, 1, chunk_state_hopper<256, 64, 256, 32, 4, 1, 2, 0, 4, 1, Tp_>);
            else if (compute >= (1LL << 39))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 35))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 1, 1, Tp_>);
            else if (compute >= (1LL << 34))
                return set(64, 64, 64, 2, 2, 4, 0, chunk_state_kernel<256, 64, 64, 64, 2, 2, 4, 2, 1, 1, Tp_>);
            else
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 1, 1, Tp_>);
        }
        if (P % 256 == 0 && N % 64 == 0)
        {
            if (compute >= (1LL << 41))
                return set(64, 256, 32, 4, 1, 2, 1, chunk_state_hopper<256, 64, 256, 32, 4, 1, 2, 0, 4, 1, Tp_>);
            else if (compute >= (1LL << 39))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 35))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 1, 1, Tp_>);
            else if (compute >= (1LL << 34))
                return set(64, 64, 64, 2, 2, 4, 0, chunk_state_kernel<256, 64, 64, 64, 2, 2, 4, 2, 1, 1, Tp_>);
            else
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 1, 1, Tp_>);
        }
        if (P % 128 == 0 && N % 256 == 0)
        {
            if (compute >= (1LL << 40))
                return set(64, 128, 64, 4, 1, 2, 1, chunk_state_hopper<256, 64, 128, 64, 4, 1, 2, 0, 2, 1, Tp_>);
            else if (compute >= (1LL << 39))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 35))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 1, 1, Tp_>);
            else if (compute >= (1LL << 34))
                return set(64, 64, 64, 2, 2, 4, 0, chunk_state_kernel<256, 64, 64, 64, 2, 2, 4, 2, 1, 1, Tp_>);
            else if (compute >= (1LL << 33))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 32))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 1, 1, Tp_>);
            else
                return set(64, 64, 64, 2, 2, 4, 0, chunk_state_kernel<256, 64, 64, 64, 2, 2, 4, 2, 1, 1, Tp_>);
        }
        if (P % 128 == 0 && N % 128 == 0)
        {
            if (compute >= (1LL << 40))
                return set(64, 128, 64, 4, 1, 2, 1, chunk_state_hopper<256, 64, 128, 64, 4, 1, 2, 0, 2, 1, Tp_>);
            else if (compute >= (1LL << 39))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 35))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 1, 1, Tp_>);
            else if (compute >= (1LL << 34))
                return set(64, 64, 64, 2, 2, 4, 0, chunk_state_kernel<256, 64, 64, 64, 2, 2, 4, 2, 1, 1, Tp_>);
            else if (compute >= (1LL << 33))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 32))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 1, 1, Tp_>);
            else
                return set(64, 64, 64, 2, 2, 4, 0, chunk_state_kernel<256, 64, 64, 64, 2, 2, 4, 2, 1, 1, Tp_>);
        }
        if (P % 128 == 0 && N % 64 == 0)
        {
            if (compute >= (1LL << 40))
                return set(64, 128, 64, 4, 1, 2, 1, chunk_state_hopper<256, 64, 128, 64, 4, 1, 2, 0, 2, 1, Tp_>);
            else if (compute >= (1LL << 39))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 35))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 1, 1, Tp_>);
            else if (compute >= (1LL << 34))
                return set(64, 64, 64, 2, 2, 4, 0, chunk_state_kernel<256, 64, 64, 64, 2, 2, 4, 2, 1, 1, Tp_>);
            else if (compute >= (1LL << 33))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 32))
                return set(64, 128, 32, 1, 4, 3, 0, chunk_state_kernel<256, 64, 128, 32, 1, 4, 3, 1, 1, 1, Tp_>);
            else
                return set(64, 64, 64, 2, 2, 4, 0, chunk_state_kernel<256, 64, 64, 64, 2, 2, 4, 2, 1, 1, Tp_>);
        }
        if (P % 64 == 0 && N % 256 == 0)
        {
            if (compute >= (1LL << 40))
                return set(64, 64, 32, 4, 1, 3, 1, chunk_state_hopper<256, 64, 64, 32, 4, 1, 3, 0, 4, 1, Tp_>);
            else if (compute >= (1LL << 38))
                return set(64, 64, 32, 1, 2, 2, 0, chunk_state_kernel<256, 64, 64, 32, 1, 2, 2, 1, 1, 1, Tp_>);
            else if (compute >= (1LL << 36))
                return set(64, 64, 64, 2, 2, 2, 0, chunk_state_kernel<256, 64, 64, 64, 2, 2, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 34))
                return set(64, 64, 64, 2, 2, 4, 0, chunk_state_kernel<256, 64, 64, 64, 2, 2, 4, 2, 1, 1, Tp_>);
            else if (compute >= (1LL << 32))
                return set(64, 64, 64, 2, 2, 2, 0, chunk_state_kernel<256, 64, 64, 64, 2, 2, 2, 1, 8, 1, Tp_>);
            else
                return set(64, 64, 64, 2, 2, 4, 0, chunk_state_kernel<256, 64, 64, 64, 2, 2, 4, 2, 1, 1, Tp_>);
        }
        if (P % 64 == 0 && N % 128 == 0)
        {
            if (compute >= (1LL << 40))
                return set(64, 64, 32, 4, 1, 3, 1, chunk_state_hopper<256, 64, 64, 32, 4, 1, 3, 0, 4, 1, Tp_>);
            else if (compute >= (1LL << 38))
                return set(64, 64, 32, 1, 2, 2, 0, chunk_state_kernel<256, 64, 64, 32, 1, 2, 2, 1, 1, 1, Tp_>);
            else if (compute >= (1LL << 36))
                return set(64, 64, 64, 2, 2, 2, 0, chunk_state_kernel<256, 64, 64, 64, 2, 2, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 34))
                return set(64, 64, 64, 2, 2, 4, 0, chunk_state_kernel<256, 64, 64, 64, 2, 2, 4, 2, 1, 1, Tp_>);
            else if (compute >= (1LL << 32))
                return set(64, 64, 64, 2, 2, 2, 0, chunk_state_kernel<256, 64, 64, 64, 2, 2, 2, 1, 8, 1, Tp_>);
            else
                return set(64, 64, 64, 2, 2, 4, 0, chunk_state_kernel<256, 64, 64, 64, 2, 2, 4, 2, 1, 1, Tp_>);
        }
#endif
        if (P % 64 == 0 && N % 64 == 0)
        {
            if (compute >= (1LL << 40))
                return set(64, 64, 32, 4, 1, 3, 1, chunk_state_hopper<256, 64, 64, 32, 4, 1, 3, 0, 4, 1, Tp_>);
            else if (compute >= (1LL << 38))
                return set(64, 64, 32, 1, 2, 2, 0, chunk_state_kernel<256, 64, 64, 32, 1, 2, 2, 1, 1, 1, Tp_>);
            else if (compute >= (1LL << 36))
                return set(64, 64, 64, 2, 2, 2, 0, chunk_state_kernel<256, 64, 64, 64, 2, 2, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 34))
                return set(64, 64, 64, 2, 2, 4, 0, chunk_state_kernel<256, 64, 64, 64, 2, 2, 4, 2, 1, 1, Tp_>);
            else if (compute >= (1LL << 32))
                return set(64, 64, 64, 2, 2, 2, 0, chunk_state_kernel<256, 64, 64, 64, 2, 2, 2, 1, 8, 1, Tp_>);
            else
                return set(64, 64, 64, 2, 2, 4, 0, chunk_state_kernel<256, 64, 64, 64, 2, 2, 4, 2, 1, 1, Tp_>);
        }
    }
    if (Q == 128 && isHopper_ == false)
    {
#ifndef FAST_BUILD
        if (P == 32 && N % 256 == 0)
        {
            if (compute >= (1LL << 34))
                return set(64, 32, 32, 2, 1, 2, 0, chunk_state_kernel<128, 64, 32, 32, 2, 1, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 31))
                return set(64, 32, 32, 1, 2, 4, 0, chunk_state_kernel<128, 64, 32, 32, 1, 2, 4, 1, 2, 1, Tp_>);
            else
                return set(64, 64, 32, 1, 2, 4, 0, chunk_state_kernel<128, 64, 64, 32, 1, 2, 4, 2, 1, 1, Tp_>);
        }
        if (P == 32 && N % 128 == 0)
        {
            if (compute >= (1LL << 34))
                return set(64, 32, 32, 2, 1, 2, 0, chunk_state_kernel<128, 64, 32, 32, 2, 1, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 31))
                return set(64, 32, 32, 1, 2, 4, 0, chunk_state_kernel<128, 64, 32, 32, 1, 2, 4, 1, 2, 1, Tp_>);
            else
                return set(64, 64, 32, 1, 2, 4, 0, chunk_state_kernel<128, 64, 64, 32, 1, 2, 4, 2, 1, 1, Tp_>);
        }
        if (P == 32 && N % 64 == 0)
        {
            if (compute >= (1LL << 34))
                return set(64, 32, 32, 2, 1, 2, 0, chunk_state_kernel<128, 64, 32, 32, 2, 1, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 31))
                return set(64, 32, 32, 1, 2, 4, 0, chunk_state_kernel<128, 64, 32, 32, 1, 2, 4, 1, 2, 1, Tp_>);
            else
                return set(64, 64, 32, 1, 2, 4, 0, chunk_state_kernel<128, 64, 64, 32, 1, 2, 4, 2, 1, 1, Tp_>);
        }
        if (P % 256 == 0 && N % 256 == 0)
        {
            if (compute >= (1LL << 44))
                return set(64, 128, 32, 1, 4, 4, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 4, 2, 4, 1, Tp_>);
            else if (compute >= (1LL << 36))
                return set(64, 128, 32, 2, 2, 3, 0, chunk_state_kernel<128, 64, 128, 32, 2, 2, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 35))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 34))
                return set(64, 64, 32, 1, 2, 2, 0, chunk_state_kernel<128, 64, 64, 32, 1, 2, 2, 1, 8, 0, Tp_>);
            else
                return set(64, 128, 32, 2, 2, 3, 0, chunk_state_kernel<128, 64, 128, 32, 2, 2, 3, 1, 4, 1, Tp_>);
        }
        if (P % 256 == 0 && N % 128 == 0)
        {
            if (compute >= (1LL << 44))
                return set(64, 128, 32, 1, 4, 4, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 4, 2, 4, 1, Tp_>);
            else if (compute >= (1LL << 36))
                return set(64, 128, 32, 2, 2, 3, 0, chunk_state_kernel<128, 64, 128, 32, 2, 2, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 35))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 34))
                return set(64, 64, 32, 1, 2, 2, 0, chunk_state_kernel<128, 64, 64, 32, 1, 2, 2, 1, 8, 0, Tp_>);
            else
                return set(64, 128, 32, 2, 2, 3, 0, chunk_state_kernel<128, 64, 128, 32, 2, 2, 3, 1, 4, 1, Tp_>);
        }
        if (P % 256 == 0 && N % 64 == 0)
        {
            if (compute >= (1LL << 44))
                return set(64, 128, 32, 1, 4, 4, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 4, 2, 4, 1, Tp_>);
            else if (compute >= (1LL << 36))
                return set(64, 128, 32, 2, 2, 3, 0, chunk_state_kernel<128, 64, 128, 32, 2, 2, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 35))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 34))
                return set(64, 64, 32, 1, 2, 2, 0, chunk_state_kernel<128, 64, 64, 32, 1, 2, 2, 1, 8, 0, Tp_>);
            else
                return set(64, 128, 32, 2, 2, 3, 0, chunk_state_kernel<128, 64, 128, 32, 2, 2, 3, 1, 4, 1, Tp_>);
        }
        if (P % 128 == 0 && N % 256 == 0)
        {
            if (compute >= (1LL << 44))
                return set(64, 128, 32, 1, 4, 4, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 4, 2, 4, 1, Tp_>);
            else if (compute >= (1LL << 35))
                return set(64, 128, 32, 2, 2, 3, 0, chunk_state_kernel<128, 64, 128, 32, 2, 2, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 34))
                return set(64, 128, 32, 1, 4, 4, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 4, 2, 4, 1, Tp_>);
            else if (compute >= (1LL << 33))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 2, 1, 8, 1, Tp_>);
            else
                return set(64, 64, 32, 2, 2, 3, 0, chunk_state_kernel<128, 64, 64, 32, 2, 2, 3, 1, 4, 1, Tp_>);
        }
        if (P % 128 == 0 && N % 128 == 0)
        {
            if (compute >= (1LL << 44))
                return set(64, 128, 32, 1, 4, 4, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 4, 2, 4, 1, Tp_>);
            else if (compute >= (1LL << 35))
                return set(64, 128, 32, 2, 2, 3, 0, chunk_state_kernel<128, 64, 128, 32, 2, 2, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 34))
                return set(64, 128, 32, 1, 4, 4, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 4, 2, 4, 1, Tp_>);
            else if (compute >= (1LL << 33))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 2, 1, 8, 1, Tp_>);
            else
                return set(64, 64, 32, 2, 2, 3, 0, chunk_state_kernel<128, 64, 64, 32, 2, 2, 3, 1, 4, 1, Tp_>);
        }
        if (P % 128 == 0 && N % 64 == 0)
        {
            if (compute >= (1LL << 44))
                return set(64, 128, 32, 1, 4, 4, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 4, 2, 4, 1, Tp_>);
            else if (compute >= (1LL << 35))
                return set(64, 128, 32, 2, 2, 3, 0, chunk_state_kernel<128, 64, 128, 32, 2, 2, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 34))
                return set(64, 128, 32, 1, 4, 4, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 4, 2, 4, 1, Tp_>);
            else if (compute >= (1LL << 33))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 2, 1, 8, 1, Tp_>);
            else
                return set(64, 64, 32, 2, 2, 3, 0, chunk_state_kernel<128, 64, 64, 32, 2, 2, 3, 1, 4, 1, Tp_>);
        }
        if (P % 64 == 0 && N % 256 == 0)
        {
            if (compute >= (1LL << 44))
                return set(64, 64, 32, 1, 2, 4, 0, chunk_state_kernel<128, 64, 64, 32, 1, 2, 4, 2, 1, 1, Tp_>);
            else if (compute >= (1LL << 36))
                return set(64, 64, 32, 1, 2, 2, 0, chunk_state_kernel<128, 64, 64, 32, 1, 2, 2, 1, 8, 0, Tp_>);
            else if (compute >= (1LL << 34))
                return set(64, 64, 32, 2, 2, 3, 0, chunk_state_kernel<128, 64, 64, 32, 2, 2, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 33))
                return set(64, 64, 32, 1, 2, 4, 0, chunk_state_kernel<128, 64, 64, 32, 1, 2, 4, 2, 1, 1, Tp_>);
            else if (compute >= (1LL << 32))
                return set(64, 64, 32, 2, 2, 3, 0, chunk_state_kernel<128, 64, 64, 32, 2, 2, 3, 1, 4, 1, Tp_>);
            else
                return set(64, 64, 32, 1, 2, 2, 0, chunk_state_kernel<128, 64, 64, 32, 1, 2, 2, 1, 8, 0, Tp_>);
        }
        if (P % 64 == 0 && N % 128 == 0)
        {
            if (compute >= (1LL << 44))
                return set(64, 64, 32, 1, 2, 4, 0, chunk_state_kernel<128, 64, 64, 32, 1, 2, 4, 2, 1, 1, Tp_>);
            else if (compute >= (1LL << 36))
                return set(64, 64, 32, 1, 2, 2, 0, chunk_state_kernel<128, 64, 64, 32, 1, 2, 2, 1, 8, 0, Tp_>);
            else if (compute >= (1LL << 34))
                return set(64, 64, 32, 2, 2, 3, 0, chunk_state_kernel<128, 64, 64, 32, 2, 2, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 33))
                return set(64, 64, 32, 1, 2, 4, 0, chunk_state_kernel<128, 64, 64, 32, 1, 2, 4, 2, 1, 1, Tp_>);
            else if (compute >= (1LL << 32))
                return set(64, 64, 32, 2, 2, 3, 0, chunk_state_kernel<128, 64, 64, 32, 2, 2, 3, 1, 4, 1, Tp_>);
            else
                return set(64, 64, 32, 1, 2, 2, 0, chunk_state_kernel<128, 64, 64, 32, 1, 2, 2, 1, 8, 0, Tp_>);
        }
#endif
        if (P % 64 == 0 && N % 64 == 0)
        {
            if (compute >= (1LL << 44))
                return set(64, 64, 32, 1, 2, 4, 0, chunk_state_kernel<128, 64, 64, 32, 1, 2, 4, 2, 1, 1, Tp_>);
            else if (compute >= (1LL << 36))
                return set(64, 64, 32, 1, 2, 2, 0, chunk_state_kernel<128, 64, 64, 32, 1, 2, 2, 1, 8, 0, Tp_>);
            else if (compute >= (1LL << 34))
                return set(64, 64, 32, 2, 2, 3, 0, chunk_state_kernel<128, 64, 64, 32, 2, 2, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 33))
                return set(64, 64, 32, 1, 2, 4, 0, chunk_state_kernel<128, 64, 64, 32, 1, 2, 4, 2, 1, 1, Tp_>);
            else if (compute >= (1LL << 32))
                return set(64, 64, 32, 2, 2, 3, 0, chunk_state_kernel<128, 64, 64, 32, 2, 2, 3, 1, 4, 1, Tp_>);
            else
                return set(64, 64, 32, 1, 2, 2, 0, chunk_state_kernel<128, 64, 64, 32, 1, 2, 2, 1, 8, 0, Tp_>);
        }
        if (round_up(P, 32) == 32)
            P = 32;
        else
            P = round_up(P, 64);
#ifndef FAST_BUILD
        if (P == 32 && N % 256 == 0)
        {
            if (compute >= (1LL << 34))
                return set(64, 32, 32, 2, 1, 2, 0, chunk_state_kernel<128, 64, 32, 32, 2, 1, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 31))
                return set(64, 32, 32, 1, 2, 4, 0, chunk_state_kernel<128, 64, 32, 32, 1, 2, 4, 1, 2, 1, Tp_>);
            else
                return set(64, 64, 32, 1, 2, 4, 0, chunk_state_kernel<128, 64, 64, 32, 1, 2, 4, 2, 1, 1, Tp_>);
        }
        if (P == 32 && N % 128 == 0)
        {
            if (compute >= (1LL << 34))
                return set(64, 32, 32, 2, 1, 2, 0, chunk_state_kernel<128, 64, 32, 32, 2, 1, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 31))
                return set(64, 32, 32, 1, 2, 4, 0, chunk_state_kernel<128, 64, 32, 32, 1, 2, 4, 1, 2, 1, Tp_>);
            else
                return set(64, 64, 32, 1, 2, 4, 0, chunk_state_kernel<128, 64, 64, 32, 1, 2, 4, 2, 1, 1, Tp_>);
        }
        if (P == 32 && N % 64 == 0)
        {
            if (compute >= (1LL << 34))
                return set(64, 32, 32, 2, 1, 2, 0, chunk_state_kernel<128, 64, 32, 32, 2, 1, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 31))
                return set(64, 32, 32, 1, 2, 4, 0, chunk_state_kernel<128, 64, 32, 32, 1, 2, 4, 1, 2, 1, Tp_>);
            else
                return set(64, 64, 32, 1, 2, 4, 0, chunk_state_kernel<128, 64, 64, 32, 1, 2, 4, 2, 1, 1, Tp_>);
        }
        if (P % 256 == 0 && N % 256 == 0)
        {
            if (compute >= (1LL << 44))
                return set(64, 128, 32, 1, 4, 4, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 4, 2, 4, 1, Tp_>);
            else if (compute >= (1LL << 36))
                return set(64, 128, 32, 2, 2, 3, 0, chunk_state_kernel<128, 64, 128, 32, 2, 2, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 34))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 2, 1, 8, 1, Tp_>);
            else
                return set(64, 128, 32, 2, 2, 3, 0, chunk_state_kernel<128, 64, 128, 32, 2, 2, 3, 1, 4, 1, Tp_>);
        }
        if (P % 256 == 0 && N % 128 == 0)
        {
            if (compute >= (1LL << 44))
                return set(64, 128, 32, 1, 4, 4, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 4, 2, 4, 1, Tp_>);
            else if (compute >= (1LL << 36))
                return set(64, 128, 32, 2, 2, 3, 0, chunk_state_kernel<128, 64, 128, 32, 2, 2, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 34))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 2, 1, 8, 1, Tp_>);
            else
                return set(64, 128, 32, 2, 2, 3, 0, chunk_state_kernel<128, 64, 128, 32, 2, 2, 3, 1, 4, 1, Tp_>);
        }
        if (P % 256 == 0 && N % 64 == 0)
        {
            if (compute >= (1LL << 44))
                return set(64, 128, 32, 1, 4, 4, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 4, 2, 4, 1, Tp_>);
            else if (compute >= (1LL << 36))
                return set(64, 128, 32, 2, 2, 3, 0, chunk_state_kernel<128, 64, 128, 32, 2, 2, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 34))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 2, 1, 8, 1, Tp_>);
            else
                return set(64, 128, 32, 2, 2, 3, 0, chunk_state_kernel<128, 64, 128, 32, 2, 2, 3, 1, 4, 1, Tp_>);
        }
        if (P % 128 == 0 && N % 256 == 0)
        {
            if (compute >= (1LL << 44))
                return set(64, 128, 32, 1, 4, 4, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 4, 2, 4, 1, Tp_>);
            else if (compute >= (1LL << 35))
                return set(64, 128, 32, 2, 2, 3, 0, chunk_state_kernel<128, 64, 128, 32, 2, 2, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 34))
                return set(64, 128, 32, 1, 4, 4, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 4, 2, 4, 1, Tp_>);
            else if (compute >= (1LL << 32))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 31))
                return set(64, 128, 32, 2, 2, 3, 0, chunk_state_kernel<128, 64, 128, 32, 2, 2, 3, 1, 4, 1, Tp_>);
            else
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 2, 1, 8, 1, Tp_>);
        }
        if (P % 128 == 0 && N % 128 == 0)
        {
            if (compute >= (1LL << 44))
                return set(64, 128, 32, 1, 4, 4, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 4, 2, 4, 1, Tp_>);
            else if (compute >= (1LL << 35))
                return set(64, 128, 32, 2, 2, 3, 0, chunk_state_kernel<128, 64, 128, 32, 2, 2, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 34))
                return set(64, 128, 32, 1, 4, 4, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 4, 2, 4, 1, Tp_>);
            else if (compute >= (1LL << 32))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 31))
                return set(64, 128, 32, 2, 2, 3, 0, chunk_state_kernel<128, 64, 128, 32, 2, 2, 3, 1, 4, 1, Tp_>);
            else
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 2, 1, 8, 1, Tp_>);
        }
        if (P % 128 == 0 && N % 64 == 0)
        {
            if (compute >= (1LL << 44))
                return set(64, 128, 32, 1, 4, 4, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 4, 2, 4, 1, Tp_>);
            else if (compute >= (1LL << 35))
                return set(64, 128, 32, 2, 2, 3, 0, chunk_state_kernel<128, 64, 128, 32, 2, 2, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 34))
                return set(64, 128, 32, 1, 4, 4, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 4, 2, 4, 1, Tp_>);
            else if (compute >= (1LL << 32))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 31))
                return set(64, 128, 32, 2, 2, 3, 0, chunk_state_kernel<128, 64, 128, 32, 2, 2, 3, 1, 4, 1, Tp_>);
            else
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 2, 1, 8, 1, Tp_>);
        }
        if (P % 64 == 0 && N % 256 == 0)
        {
            if (compute >= (1LL << 44))
                return set(64, 64, 32, 1, 2, 4, 0, chunk_state_kernel<128, 64, 64, 32, 1, 2, 4, 2, 1, 1, Tp_>);
            else if (compute >= (1LL << 34))
                return set(64, 64, 32, 2, 2, 3, 0, chunk_state_kernel<128, 64, 64, 32, 2, 2, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 33))
                return set(64, 64, 32, 1, 2, 4, 0, chunk_state_kernel<128, 64, 64, 32, 1, 2, 4, 2, 1, 1, Tp_>);
            else if (compute >= (1LL << 32))
                return set(64, 64, 32, 2, 2, 3, 0, chunk_state_kernel<128, 64, 64, 32, 2, 2, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 31))
                return set(64, 64, 32, 1, 2, 4, 0, chunk_state_kernel<128, 64, 64, 32, 1, 2, 4, 2, 1, 1, Tp_>);
            else if (compute >= (1LL << 30))
                return set(64, 32, 32, 2, 1, 2, 0, chunk_state_kernel<128, 64, 32, 32, 2, 1, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 27))
                return set(64, 64, 32, 2, 2, 3, 0, chunk_state_kernel<128, 64, 64, 32, 2, 2, 3, 1, 4, 1, Tp_>);
            else
                return set(64, 64, 32, 1, 2, 4, 0, chunk_state_kernel<128, 64, 64, 32, 1, 2, 4, 2, 1, 1, Tp_>);
        }
        if (P % 64 == 0 && N % 128 == 0)
        {
            if (compute >= (1LL << 44))
                return set(64, 64, 32, 1, 2, 4, 0, chunk_state_kernel<128, 64, 64, 32, 1, 2, 4, 2, 1, 1, Tp_>);
            else if (compute >= (1LL << 34))
                return set(64, 64, 32, 2, 2, 3, 0, chunk_state_kernel<128, 64, 64, 32, 2, 2, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 33))
                return set(64, 64, 32, 1, 2, 4, 0, chunk_state_kernel<128, 64, 64, 32, 1, 2, 4, 2, 1, 1, Tp_>);
            else if (compute >= (1LL << 32))
                return set(64, 64, 32, 2, 2, 3, 0, chunk_state_kernel<128, 64, 64, 32, 2, 2, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 31))
                return set(64, 64, 32, 1, 2, 4, 0, chunk_state_kernel<128, 64, 64, 32, 1, 2, 4, 2, 1, 1, Tp_>);
            else if (compute >= (1LL << 30))
                return set(64, 32, 32, 2, 1, 2, 0, chunk_state_kernel<128, 64, 32, 32, 2, 1, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 27))
                return set(64, 64, 32, 2, 2, 3, 0, chunk_state_kernel<128, 64, 64, 32, 2, 2, 3, 1, 4, 1, Tp_>);
            else
                return set(64, 64, 32, 1, 2, 4, 0, chunk_state_kernel<128, 64, 64, 32, 1, 2, 4, 2, 1, 1, Tp_>);
        }
#endif
        if (P % 64 == 0 && N % 64 == 0)
        {
            if (compute >= (1LL << 44))
                return set(64, 64, 32, 1, 2, 4, 0, chunk_state_kernel<128, 64, 64, 32, 1, 2, 4, 2, 1, 1, Tp_>);
            else if (compute >= (1LL << 34))
                return set(64, 64, 32, 2, 2, 3, 0, chunk_state_kernel<128, 64, 64, 32, 2, 2, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 33))
                return set(64, 64, 32, 1, 2, 4, 0, chunk_state_kernel<128, 64, 64, 32, 1, 2, 4, 2, 1, 1, Tp_>);
            else if (compute >= (1LL << 32))
                return set(64, 64, 32, 2, 2, 3, 0, chunk_state_kernel<128, 64, 64, 32, 2, 2, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 31))
                return set(64, 64, 32, 1, 2, 4, 0, chunk_state_kernel<128, 64, 64, 32, 1, 2, 4, 2, 1, 1, Tp_>);
            else if (compute >= (1LL << 30))
                return set(64, 32, 32, 2, 1, 2, 0, chunk_state_kernel<128, 64, 32, 32, 2, 1, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 27))
                return set(64, 64, 32, 2, 2, 3, 0, chunk_state_kernel<128, 64, 64, 32, 2, 2, 3, 1, 4, 1, Tp_>);
            else
                return set(64, 64, 32, 1, 2, 4, 0, chunk_state_kernel<128, 64, 64, 32, 1, 2, 4, 2, 1, 1, Tp_>);
        }
    }
    else if (Q == 128 && isHopper_)
    {
#ifndef FAST_BUILD
        if (P == 32 && N % 256 == 0)
        {
            if (compute >= (1LL << 37))
                return set(64, 32, 32, 2, 1, 2, 0, chunk_state_kernel<128, 64, 32, 32, 2, 1, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 34))
                return set(64, 32, 32, 1, 2, 4, 0, chunk_state_kernel<128, 64, 32, 32, 1, 2, 4, 1, 2, 1, Tp_>);
            else if (compute >= (1LL << 33))
                return set(64, 32, 32, 2, 1, 2, 0, chunk_state_kernel<128, 64, 32, 32, 2, 1, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 32))
                return set(64, 32, 32, 1, 2, 4, 0, chunk_state_kernel<128, 64, 32, 32, 1, 2, 4, 1, 2, 1, Tp_>);
            else
                return set(64, 64, 32, 2, 2, 3, 0, chunk_state_kernel<128, 64, 64, 32, 2, 2, 3, 1, 4, 1, Tp_>);
        }
        if (P == 32 && N % 128 == 0)
        {
            if (compute >= (1LL << 37))
                return set(64, 32, 32, 2, 1, 2, 0, chunk_state_kernel<128, 64, 32, 32, 2, 1, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 34))
                return set(64, 32, 32, 1, 2, 4, 0, chunk_state_kernel<128, 64, 32, 32, 1, 2, 4, 1, 2, 1, Tp_>);
            else if (compute >= (1LL << 33))
                return set(64, 32, 32, 2, 1, 2, 0, chunk_state_kernel<128, 64, 32, 32, 2, 1, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 32))
                return set(64, 32, 32, 1, 2, 4, 0, chunk_state_kernel<128, 64, 32, 32, 1, 2, 4, 1, 2, 1, Tp_>);
            else
                return set(64, 64, 32, 2, 2, 3, 0, chunk_state_kernel<128, 64, 64, 32, 2, 2, 3, 1, 4, 1, Tp_>);
        }
        if (P == 32 && N % 64 == 0)
        {
            if (compute >= (1LL << 37))
                return set(64, 32, 32, 2, 1, 2, 0, chunk_state_kernel<128, 64, 32, 32, 2, 1, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 34))
                return set(64, 32, 32, 1, 2, 4, 0, chunk_state_kernel<128, 64, 32, 32, 1, 2, 4, 1, 2, 1, Tp_>);
            else if (compute >= (1LL << 33))
                return set(64, 32, 32, 2, 1, 2, 0, chunk_state_kernel<128, 64, 32, 32, 2, 1, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 32))
                return set(64, 32, 32, 1, 2, 4, 0, chunk_state_kernel<128, 64, 32, 32, 1, 2, 4, 1, 2, 1, Tp_>);
            else
                return set(64, 64, 32, 2, 2, 3, 0, chunk_state_kernel<128, 64, 64, 32, 2, 2, 3, 1, 4, 1, Tp_>);
        }
        if (P % 256 == 0 && N % 256 == 0)
        {
            if (compute >= (1LL << 43))
                return set(64, 128, 32, 1, 4, 4, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 4, 2, 4, 1, Tp_>);
            else if (compute >= (1LL << 42))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 40))
                return set(64, 128, 32, 1, 4, 4, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 4, 2, 4, 1, Tp_>);
            else if (compute >= (1LL << 38))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 37))
                return set(64, 128, 32, 1, 4, 4, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 4, 2, 4, 1, Tp_>);
            else if (compute >= (1LL << 36))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 33))
                return set(64, 128, 32, 1, 4, 4, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 4, 2, 4, 1, Tp_>);
            else
                return set(64, 64, 32, 2, 2, 3, 0, chunk_state_kernel<128, 64, 64, 32, 2, 2, 3, 1, 4, 1, Tp_>);
        }
        if (P % 256 == 0 && N % 128 == 0)
        {
            if (compute >= (1LL << 43))
                return set(64, 128, 32, 1, 4, 4, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 4, 2, 4, 1, Tp_>);
            else if (compute >= (1LL << 42))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 40))
                return set(64, 128, 32, 1, 4, 4, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 4, 2, 4, 1, Tp_>);
            else if (compute >= (1LL << 38))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 37))
                return set(64, 128, 32, 1, 4, 4, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 4, 2, 4, 1, Tp_>);
            else if (compute >= (1LL << 36))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 33))
                return set(64, 128, 32, 1, 4, 4, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 4, 2, 4, 1, Tp_>);
            else
                return set(64, 64, 32, 2, 2, 3, 0, chunk_state_kernel<128, 64, 64, 32, 2, 2, 3, 1, 4, 1, Tp_>);
        }
        if (P % 256 == 0 && N % 64 == 0)
        {
            if (compute >= (1LL << 43))
                return set(64, 128, 32, 1, 4, 4, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 4, 2, 4, 1, Tp_>);
            else if (compute >= (1LL << 42))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 40))
                return set(64, 128, 32, 1, 4, 4, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 4, 2, 4, 1, Tp_>);
            else if (compute >= (1LL << 38))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 37))
                return set(64, 128, 32, 1, 4, 4, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 4, 2, 4, 1, Tp_>);
            else if (compute >= (1LL << 36))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 33))
                return set(64, 128, 32, 1, 4, 4, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 4, 2, 4, 1, Tp_>);
            else
                return set(64, 64, 32, 2, 2, 3, 0, chunk_state_kernel<128, 64, 64, 32, 2, 2, 3, 1, 4, 1, Tp_>);
        }
        if (P % 128 == 0 && N % 256 == 0)
        {
            if (compute >= (1LL << 43))
                return set(64, 128, 32, 1, 4, 4, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 4, 2, 4, 1, Tp_>);
            else if (compute >= (1LL << 42))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 40))
                return set(64, 128, 32, 1, 4, 4, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 4, 2, 4, 1, Tp_>);
            else if (compute >= (1LL << 36))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 35))
                return set(64, 128, 32, 1, 4, 4, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 4, 2, 4, 1, Tp_>);
            else if (compute >= (1LL << 34))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 32))
                return set(64, 128, 32, 1, 4, 4, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 4, 2, 4, 1, Tp_>);
            else
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 2, 1, 8, 1, Tp_>);
        }
        if (P % 128 == 0 && N % 128 == 0)
        {
            if (compute >= (1LL << 43))
                return set(64, 128, 32, 1, 4, 4, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 4, 2, 4, 1, Tp_>);
            else if (compute >= (1LL << 42))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 40))
                return set(64, 128, 32, 1, 4, 4, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 4, 2, 4, 1, Tp_>);
            else if (compute >= (1LL << 36))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 35))
                return set(64, 128, 32, 1, 4, 4, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 4, 2, 4, 1, Tp_>);
            else if (compute >= (1LL << 34))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 32))
                return set(64, 128, 32, 1, 4, 4, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 4, 2, 4, 1, Tp_>);
            else
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 2, 1, 8, 1, Tp_>);
        }
        if (P % 128 == 0 && N % 64 == 0)
        {
            if (compute >= (1LL << 43))
                return set(64, 128, 32, 1, 4, 4, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 4, 2, 4, 1, Tp_>);
            else if (compute >= (1LL << 42))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 40))
                return set(64, 128, 32, 1, 4, 4, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 4, 2, 4, 1, Tp_>);
            else if (compute >= (1LL << 36))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 35))
                return set(64, 128, 32, 1, 4, 4, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 4, 2, 4, 1, Tp_>);
            else if (compute >= (1LL << 34))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 32))
                return set(64, 128, 32, 1, 4, 4, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 4, 2, 4, 1, Tp_>);
            else
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 2, 1, 8, 1, Tp_>);
        }
        if (P % 64 == 0 && N % 256 == 0)
        {
            if (compute >= (1LL << 40))
                return set(64, 64, 32, 4, 1, 3, 1, chunk_state_hopper<128, 64, 64, 32, 4, 1, 3, 0, 8, 1, Tp_>);
            else if (compute >= (1LL << 36))
                return set(64, 64, 32, 1, 2, 4, 0, chunk_state_kernel<128, 64, 64, 32, 1, 2, 4, 2, 1, 1, Tp_>);
            else if (compute >= (1LL << 31))
                return set(64, 64, 32, 2, 2, 3, 0, chunk_state_kernel<128, 64, 64, 32, 2, 2, 3, 1, 4, 1, Tp_>);
            else
                return set(64, 64, 32, 1, 2, 4, 0, chunk_state_kernel<128, 64, 64, 32, 1, 2, 4, 2, 1, 1, Tp_>);
        }
        if (P % 64 == 0 && N % 128 == 0)
        {
            if (compute >= (1LL << 40))
                return set(64, 64, 32, 4, 1, 3, 1, chunk_state_hopper<128, 64, 64, 32, 4, 1, 3, 0, 8, 1, Tp_>);
            else if (compute >= (1LL << 36))
                return set(64, 64, 32, 1, 2, 4, 0, chunk_state_kernel<128, 64, 64, 32, 1, 2, 4, 2, 1, 1, Tp_>);
            else if (compute >= (1LL << 31))
                return set(64, 64, 32, 2, 2, 3, 0, chunk_state_kernel<128, 64, 64, 32, 2, 2, 3, 1, 4, 1, Tp_>);
            else
                return set(64, 64, 32, 1, 2, 4, 0, chunk_state_kernel<128, 64, 64, 32, 1, 2, 4, 2, 1, 1, Tp_>);
        }
#endif
        if (P % 64 == 0 && N % 64 == 0)
        {
            if (compute >= (1LL << 40))
                return set(64, 64, 32, 4, 1, 3, 1, chunk_state_hopper<128, 64, 64, 32, 4, 1, 3, 0, 8, 1, Tp_>);
            else if (compute >= (1LL << 36))
                return set(64, 64, 32, 1, 2, 4, 0, chunk_state_kernel<128, 64, 64, 32, 1, 2, 4, 2, 1, 1, Tp_>);
            else if (compute >= (1LL << 31))
                return set(64, 64, 32, 2, 2, 3, 0, chunk_state_kernel<128, 64, 64, 32, 2, 2, 3, 1, 4, 1, Tp_>);
            else
                return set(64, 64, 32, 1, 2, 4, 0, chunk_state_kernel<128, 64, 64, 32, 1, 2, 4, 2, 1, 1, Tp_>);
        }
        if (round_up(P, 32) == 32)
            P = 32;
        else
            P = round_up(P, 64);
#ifndef FAST_BUILD
        if (P == 32 && N % 256 == 0)
        {
            if (compute >= (1LL << 37))
                return set(64, 32, 32, 2, 1, 2, 0, chunk_state_kernel<128, 64, 32, 32, 2, 1, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 34))
                return set(64, 32, 32, 1, 2, 4, 0, chunk_state_kernel<128, 64, 32, 32, 1, 2, 4, 1, 2, 1, Tp_>);
            else if (compute >= (1LL << 33))
                return set(64, 32, 32, 2, 1, 2, 0, chunk_state_kernel<128, 64, 32, 32, 2, 1, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 32))
                return set(64, 32, 32, 1, 2, 4, 0, chunk_state_kernel<128, 64, 32, 32, 1, 2, 4, 1, 2, 1, Tp_>);
            else
                return set(64, 64, 32, 2, 2, 3, 0, chunk_state_kernel<128, 64, 64, 32, 2, 2, 3, 1, 4, 1, Tp_>);
        }
        if (P == 32 && N % 128 == 0)
        {
            if (compute >= (1LL << 37))
                return set(64, 32, 32, 2, 1, 2, 0, chunk_state_kernel<128, 64, 32, 32, 2, 1, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 34))
                return set(64, 32, 32, 1, 2, 4, 0, chunk_state_kernel<128, 64, 32, 32, 1, 2, 4, 1, 2, 1, Tp_>);
            else if (compute >= (1LL << 33))
                return set(64, 32, 32, 2, 1, 2, 0, chunk_state_kernel<128, 64, 32, 32, 2, 1, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 32))
                return set(64, 32, 32, 1, 2, 4, 0, chunk_state_kernel<128, 64, 32, 32, 1, 2, 4, 1, 2, 1, Tp_>);
            else
                return set(64, 64, 32, 2, 2, 3, 0, chunk_state_kernel<128, 64, 64, 32, 2, 2, 3, 1, 4, 1, Tp_>);
        }
        if (P == 32 && N % 64 == 0)
        {
            if (compute >= (1LL << 37))
                return set(64, 32, 32, 2, 1, 2, 0, chunk_state_kernel<128, 64, 32, 32, 2, 1, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 34))
                return set(64, 32, 32, 1, 2, 4, 0, chunk_state_kernel<128, 64, 32, 32, 1, 2, 4, 1, 2, 1, Tp_>);
            else if (compute >= (1LL << 33))
                return set(64, 32, 32, 2, 1, 2, 0, chunk_state_kernel<128, 64, 32, 32, 2, 1, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 32))
                return set(64, 32, 32, 1, 2, 4, 0, chunk_state_kernel<128, 64, 32, 32, 1, 2, 4, 1, 2, 1, Tp_>);
            else
                return set(64, 64, 32, 2, 2, 3, 0, chunk_state_kernel<128, 64, 64, 32, 2, 2, 3, 1, 4, 1, Tp_>);
        }
        if (P % 256 == 0 && N % 256 == 0)
        {
            if (compute >= (1LL << 43))
                return set(64, 128, 32, 1, 4, 4, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 4, 2, 4, 1, Tp_>);
            else if (compute >= (1LL << 42))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 40))
                return set(64, 128, 32, 1, 4, 4, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 4, 2, 4, 1, Tp_>);
            else if (compute >= (1LL << 38))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 37))
                return set(64, 128, 32, 1, 4, 4, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 4, 2, 4, 1, Tp_>);
            else if (compute >= (1LL << 36))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 33))
                return set(64, 128, 32, 1, 4, 4, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 4, 2, 4, 1, Tp_>);
            else
                return set(64, 64, 32, 2, 2, 3, 0, chunk_state_kernel<128, 64, 64, 32, 2, 2, 3, 1, 4, 1, Tp_>);
        }
        if (P % 256 == 0 && N % 128 == 0)
        {
            if (compute >= (1LL << 43))
                return set(64, 128, 32, 1, 4, 4, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 4, 2, 4, 1, Tp_>);
            else if (compute >= (1LL << 42))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 40))
                return set(64, 128, 32, 1, 4, 4, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 4, 2, 4, 1, Tp_>);
            else if (compute >= (1LL << 38))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 37))
                return set(64, 128, 32, 1, 4, 4, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 4, 2, 4, 1, Tp_>);
            else if (compute >= (1LL << 36))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 33))
                return set(64, 128, 32, 1, 4, 4, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 4, 2, 4, 1, Tp_>);
            else
                return set(64, 64, 32, 2, 2, 3, 0, chunk_state_kernel<128, 64, 64, 32, 2, 2, 3, 1, 4, 1, Tp_>);
        }
        if (P % 256 == 0 && N % 64 == 0)
        {
            if (compute >= (1LL << 43))
                return set(64, 128, 32, 1, 4, 4, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 4, 2, 4, 1, Tp_>);
            else if (compute >= (1LL << 42))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 40))
                return set(64, 128, 32, 1, 4, 4, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 4, 2, 4, 1, Tp_>);
            else if (compute >= (1LL << 38))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 37))
                return set(64, 128, 32, 1, 4, 4, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 4, 2, 4, 1, Tp_>);
            else if (compute >= (1LL << 36))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 33))
                return set(64, 128, 32, 1, 4, 4, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 4, 2, 4, 1, Tp_>);
            else
                return set(64, 64, 32, 2, 2, 3, 0, chunk_state_kernel<128, 64, 64, 32, 2, 2, 3, 1, 4, 1, Tp_>);
        }
        if (P % 128 == 0 && N % 256 == 0)
        {
            if (compute >= (1LL << 42))
                return set(64, 128, 32, 1, 4, 4, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 4, 2, 4, 1, Tp_>);
            else if (compute >= (1LL << 41))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 39))
                return set(64, 128, 32, 1, 4, 4, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 4, 2, 4, 1, Tp_>);
            else if (compute >= (1LL << 36))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 29))
                return set(64, 128, 32, 1, 4, 4, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 4, 2, 4, 1, Tp_>);
            else
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 2, 1, 8, 1, Tp_>);
        }
        if (P % 128 == 0 && N % 128 == 0)
        {
            if (compute >= (1LL << 42))
                return set(64, 128, 32, 1, 4, 4, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 4, 2, 4, 1, Tp_>);
            else if (compute >= (1LL << 41))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 39))
                return set(64, 128, 32, 1, 4, 4, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 4, 2, 4, 1, Tp_>);
            else if (compute >= (1LL << 36))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 29))
                return set(64, 128, 32, 1, 4, 4, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 4, 2, 4, 1, Tp_>);
            else
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 2, 1, 8, 1, Tp_>);
        }
        if (P % 128 == 0 && N % 64 == 0)
        {
            if (compute >= (1LL << 42))
                return set(64, 128, 32, 1, 4, 4, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 4, 2, 4, 1, Tp_>);
            else if (compute >= (1LL << 41))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 39))
                return set(64, 128, 32, 1, 4, 4, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 4, 2, 4, 1, Tp_>);
            else if (compute >= (1LL << 36))
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 2, 1, 8, 1, Tp_>);
            else if (compute >= (1LL << 29))
                return set(64, 128, 32, 1, 4, 4, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 4, 2, 4, 1, Tp_>);
            else
                return set(64, 128, 32, 1, 4, 2, 0, chunk_state_kernel<128, 64, 128, 32, 1, 4, 2, 1, 8, 1, Tp_>);
        }
        if (P % 64 == 0 && N % 256 == 0)
        {
            if (compute >= (1LL << 39))
                return set(64, 64, 32, 4, 1, 3, 1, chunk_state_hopper<128, 64, 64, 32, 4, 1, 3, 0, 8, 1, Tp_>);
            else if (compute >= (1LL << 35))
                return set(64, 64, 32, 1, 2, 4, 0, chunk_state_kernel<128, 64, 64, 32, 1, 2, 4, 2, 1, 1, Tp_>);
            else if (compute >= (1LL << 31))
                return set(64, 64, 32, 2, 2, 3, 0, chunk_state_kernel<128, 64, 64, 32, 2, 2, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 30))
                return set(64, 32, 32, 1, 2, 4, 0, chunk_state_kernel<128, 64, 32, 32, 1, 2, 4, 1, 2, 1, Tp_>);
            else if (compute >= (1LL << 27))
                return set(64, 64, 32, 2, 2, 3, 0, chunk_state_kernel<128, 64, 64, 32, 2, 2, 3, 1, 4, 1, Tp_>);
            else
                return set(64, 64, 32, 1, 2, 4, 0, chunk_state_kernel<128, 64, 64, 32, 1, 2, 4, 2, 1, 1, Tp_>);
        }
        if (P % 64 == 0 && N % 128 == 0)
        {
            if (compute >= (1LL << 39))
                return set(64, 64, 32, 4, 1, 3, 1, chunk_state_hopper<128, 64, 64, 32, 4, 1, 3, 0, 8, 1, Tp_>);
            else if (compute >= (1LL << 35))
                return set(64, 64, 32, 1, 2, 4, 0, chunk_state_kernel<128, 64, 64, 32, 1, 2, 4, 2, 1, 1, Tp_>);
            else if (compute >= (1LL << 31))
                return set(64, 64, 32, 2, 2, 3, 0, chunk_state_kernel<128, 64, 64, 32, 2, 2, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 30))
                return set(64, 32, 32, 1, 2, 4, 0, chunk_state_kernel<128, 64, 32, 32, 1, 2, 4, 1, 2, 1, Tp_>);
            else if (compute >= (1LL << 27))
                return set(64, 64, 32, 2, 2, 3, 0, chunk_state_kernel<128, 64, 64, 32, 2, 2, 3, 1, 4, 1, Tp_>);
            else
                return set(64, 64, 32, 1, 2, 4, 0, chunk_state_kernel<128, 64, 64, 32, 1, 2, 4, 2, 1, 1, Tp_>);
        }
#endif
        if (P % 64 == 0 && N % 64 == 0)
        {
            if (compute >= (1LL << 39))
                return set(64, 64, 32, 4, 1, 3, 1, chunk_state_hopper<128, 64, 64, 32, 4, 1, 3, 0, 8, 1, Tp_>);
            else if (compute >= (1LL << 35))
                return set(64, 64, 32, 1, 2, 4, 0, chunk_state_kernel<128, 64, 64, 32, 1, 2, 4, 2, 1, 1, Tp_>);
            else if (compute >= (1LL << 31))
                return set(64, 64, 32, 2, 2, 3, 0, chunk_state_kernel<128, 64, 64, 32, 2, 2, 3, 1, 4, 1, Tp_>);
            else if (compute >= (1LL << 30))
                return set(64, 32, 32, 1, 2, 4, 0, chunk_state_kernel<128, 64, 32, 32, 1, 2, 4, 1, 2, 1, Tp_>);
            else if (compute >= (1LL << 27))
                return set(64, 64, 32, 2, 2, 3, 0, chunk_state_kernel<128, 64, 64, 32, 2, 2, 3, 1, 4, 1, Tp_>);
            else
                return set(64, 64, 32, 1, 2, 4, 0, chunk_state_kernel<128, 64, 64, 32, 1, 2, 4, 2, 1, 1, Tp_>);
        }
    }

    return nullptr;
}

extern GetChunkStateKernelFunc getChunkStateKernel_fp16;
extern GetChunkStateKernelFunc getChunkStateKernel_bf16;

static inline ChunkStateKernelFunc getChunkStateKernel(int B_, int L_, int H_, int P_, int G_, int N_, int Q_,
    int numTokens_, int isHopper_, tensorrt_llm::common::CUDADriverWrapper* cudaDriver_, dim3* blockDims_,
    dim3* threadDims_, int* sharedMem_, int* useTma_, CUtensorMap* descs_, CudaType tp_ = CT_FP16)
{
    if (tp_ == CT_FP16)
        return getChunkStateKernel_fp16(B_, L_, H_, P_, G_, N_, Q_, numTokens_, isHopper_, cudaDriver_, blockDims_,
            threadDims_, sharedMem_, useTma_, descs_);
    else if (tp_ == CT_BF16)
        return getChunkStateKernel_bf16(B_, L_, H_, P_, G_, N_, Q_, numTokens_, isHopper_, cudaDriver_, blockDims_,
            threadDims_, sharedMem_, useTma_, descs_);

    return nullptr;
}

} // namespace kernels
} // namespace tensorrt_llm

// vim: ts=2 sw=2 sts=2 et sta
