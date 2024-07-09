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

typedef void (*ChunkStateKernelFuncFp16)(int B_, int L_, int H_, int P_, int G_, int N_,
    //  const half  *g_mxY_,  // B*L*H*P
    //  const half  *g_mxOs_, // B*C*H*N*P
    //  const half  *g_mxFs_, // B  *H*N*P
    float* g_mxSt_,       // B*C*H*N*P
    float const* g_mxdc_, // B*C*H*Q
    float const* g_mxdA_, // B*C*H*Q
                          //  const half  *g_mxdt_, // B*L*H
                          //  const float *g_mxdb_, //     H
                          //  const float *g_mxA_,  //     H
                          //  const half  *g_mxCB_, // B*C*G*Q*Q
    half const* g_mxBC_,  // B*L*2*G*N
                          //  const float *g_mxD_,  //     H
    half const* g_mxX_,   // B*L*H*P
                          //  const half  *g_mxZ_,  // B*L*H*P
    bool removePadding_, int const* lastTokenIdsPtr_);

typedef void (*ChunkStateKernelFuncBf16)(int B_, int L_, int H_, int P_, int G_, int N_,
    //  const bf16  *g_mxY_,  // B*L*H*P
    //  const bf16  *g_mxOs_, // B*C*H*N*P
    //  const bf16  *g_mxFs_, // B  *H*N*P
    float* g_mxSt_,       // B*C*H*N*P
    float const* g_mxdc_, // B*C*H*Q
    float const* g_mxdA_, // B*C*H*Q
                          //  const bf16  *g_mxdt_, // B*L*H
                          //  const float *g_mxdb_, //     H
                          //  const float *g_mxA_,  //     H
                          //  const bf16  *g_mxCB_, // B*C*G*Q*Q
    bf16 const* g_mxBC_,  // B*L*2*G*N
                          //  const float *g_mxD_,  //     H
    bf16 const* g_mxX_,   // B*L*H*P
                          //  const bf16  *g_mxZ_,  // B*L*H*P
    bool removePadding_, int const* lastTokenIdsPtr_);

template <int Q_, int tileM_, int tileN_, int tileK_, // smem size, per sm
    int wmmaM_, int wmmaN_, int wmmaK_,               // wmma size, per instruction
    int warpM_, int warpN_,                           // warp number
    int pipeS_, class Tp_>
__global__ std::enable_if_t<std::is_same_v<Tp_, half> || std::is_same_v<Tp_, __nv_bfloat16>> chunk_state_kernel(int B_,
    int L_, int H_, int P_, int G_, int N_,
    //  const Tp_   *g_mxY_,  // B*L*H*P
    //  const Tp_   *g_mxOs_, // B*C*H*N*P
    //  const Tp_   *g_mxFs_, // B  *H*N*P
    float* g_mxSt_,       // B*C*H*N*P
    float const* g_mxdc_, // B*C*H*Q
    float const* g_mxdA_, // B*C*H*Q
                          //  const Tp_   *g_mxdt_, // B*L*H
                          //  const Wt_   *g_mxdb_, //     H
                          //  const Wt_   *g_mxA_,  //     H
                          //  const Tp_   *g_mxCB_, // B*C*G*Q*Q
    Tp_ const* g_mxBC_,   // B*L*2*G*N
                          //  const Wt_   *g_mxD_,  //     H
    Tp_ const* g_mxX_,    // B*L*H*P
                          //  const Tp_   *g_mxZ_,  // B*L*H*P
    bool removePadding_, int const* lastTokenIdsPtr_)
{
#if __CUDA_ARCH__ >= 800
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

    auto hStart = Rn<ID>{blockIdx_x.var / (P_ / cn<tileN_>) / (N_ / cn<tileM_>) };
    auto mStart = Rn<ID>{blockIdx_x.var / (P_ / cn<tileN_>) % (N_ / cn<tileM_>) };
    auto nStart = Rn<ID>{blockIdx_x.var % (P_ / cn<tileN_>) };
    auto gStart = Rn<ID>{hStart.var / (H_ / G_)};

    extern __shared__ float smem[];

    Tp_* s_mxB = (Tp_*) smem;
    Tp_* s_mxX = (Tp_*) smem + tileM_ * tileK_ * pipeS_;

    float* s_mxdc = smem + (tileM_ + tileN_) * tileK_ * pipeS_ / 2;
    float* s_mxdA = smem + (tileM_ + tileN_) * tileK_ * pipeS_ / 2 + Q_;

    unsigned b_base = __nvvm_get_smem_pointer(smem);

    unsigned b_mxB = b_base;
    unsigned b_mxX = b_base + tileM_ * tileK_ * pipeS_ * sizeof(Tp_);

    using std::array;

    register array<array<array<float, wmmaM_ * wmmaN_ / 32>, tileN_ / wmmaN_ / warpN_>, tileM_ / wmmaM_ / warpM_> r_mxSt
        = array<array<array<float, wmmaM_ * wmmaN_ / 32>, tileN_ / wmmaN_ / warpN_>, tileM_ / wmmaM_ / warpM_>();
    register array<array<unsigned, wmmaM_ * wmmaK_ / 64>, tileM_ / wmmaM_ / warpM_> r_mxB;
    register array<array<unsigned, wmmaK_ * wmmaN_ / 64>, tileN_ / wmmaN_ / warpN_> r_mxX;

    constexpr int step = std::max(
        1, tileM_ / wmmaM_ / warpM_ * tileN_ / wmmaN_ / warpN_ / (tileM_ / wmmaM_ / warpM_ + tileN_ / wmmaN_ / warpN_));

    auto baseB = [](auto iK) { return iK % cn<pipeS_> * cn<tileM_> * cn<tileK_>; };
    auto baseX = [](auto iK) { return iK % cn<pipeS_> * cn<tileN_> * cn<tileK_>; };

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
                *(int4*) (s_mxdc + get(thread(iStep)) + i)
                    = *(int4*) (g_mxdc_ + get((cStart + blockIdx_y) * H * Q + hStart * Q + thread(iStep)) + i);
                *(int4*) (s_mxdA + get(thread(iStep)) + i)
                    = *(int4*) (g_mxdA_ + get((cStart + blockIdx_y) * H * Q + hStart * Q + thread(iStep)) + i);
            }
        }

#pragma unroll
    for (Rn<UNROLL, pipeS_> iK; iK.var < iK.size; iK.var++)
    {
#pragma unroll
        for (Rn<UNROLL, div_up(tileM_ * tileK_, warpM_ * warpN_ * 256)> iStep; iStep.var < iStep.size; iStep.var++)
            if (thread(iStep) < cn<tileM_ * tileK_>
                && thread(iStep) / cn<tileM_> < L - blockIdx_y * Q - iK * cn<tileK_>)
                cp_shared_global<16>(b_mxB + swizzle<tileM_ * 2, tileM_ * 2>(thread(iStep) * cn<2>, baseB(iK) * cn<2>),
                    g_mxBC_
                        + get((aStart + blockIdx_y * Q + iK * cn<tileK_> + thread(iStep) / cn<tileM_>) *cn<2> * G * N
                            + gStart * N + mStart * cn<tileM_> + thread(iStep) % cn<tileM_>));
            else if (thread(iStep) < cn<tileM_ * tileK_>)
                *(int4*) ((char*) s_mxB + swizzle<tileM_ * 2, tileM_ * 2>(thread(iStep) * cn<2>, baseB(iK) * cn<2>))
                    = int4{0, 0, 0, 0};

#pragma unroll
        for (Rn<UNROLL, div_up(tileN_ * tileK_, warpM_ * warpN_ * 256)> iStep; iStep.var < iStep.size; iStep.var++)
            if (thread(iStep) < cn<tileN_ * tileK_>
                && thread(iStep) / cn<tileN_> < L - blockIdx_y * Q - iK * cn<tileK_>)
                cp_shared_global<16>(b_mxX + swizzle<tileN_ * 2, tileN_ * 2>(thread(iStep) * cn<2>, baseX(iK) * cn<2>),
                    g_mxX_
                        + get((aStart + blockIdx_y * Q + iK * cn<tileK_> + thread(iStep) / cn<tileN_>) *H * P
                            + hStart * P + nStart * cn<tileN_> + thread(iStep) % cn<tileN_>));
            else if (thread(iStep) < cn<tileN_ * tileK_>)
                *(int4*) ((char*) s_mxX + swizzle<tileN_ * 2, tileN_ * 2>(thread(iStep) * cn<2>, baseX(iK) * cn<2>))
                    = int4{0, 0, 0, 0};

        cp_commit_group();
    }

    asm volatile("cp.async.wait_group %0;\n" ::"n"(pipeS_ - 1));

    __syncthreads();

    for (int iK = pipeS_; iK < Q_ / tileK_ + pipeS_; iK++)
    {
        auto jK = Rn<>{iK};
#pragma unroll
        for (Rn<UNROLL, div_up(tileM_ * tileK_, warpM_ * warpN_ * 256)> iStep; iStep.var < iStep.size; iStep.var++)
            if (thread(iStep) < cn<tileM_ * tileK_>)
            {
                register Tp_ tmpB[8];

                *(int4*) &tmpB[0] = *(
                    int4*) ((char*) s_mxB + swizzle<tileM_ * 2, tileM_ * 2>(thread(iStep) * cn<2>, baseB(jK) * cn<2>));

#pragma unroll
                for (int i = 0; i < 8; i += 2)
                {
                    float2 tmp2 = std::is_same_v<Tp_, half> ? __half22float2(*(half2*) &tmpB[i])
                                                            : bf1622float2(*(bf162*) &tmpB[i]);

                    int kStart = (iK - pipeS_) * cn<tileK_>;

                    tmp2.x *= expf(s_mxdA[Q_ - 1] - s_mxdA[kStart + get(thread(iStep) / cn<tileM_>)])
                        * s_mxdc[kStart + get(thread(iStep) / cn<tileM_>)];
                    tmp2.y *= expf(s_mxdA[Q_ - 1] - s_mxdA[kStart + get(thread(iStep) / cn<tileM_>)])
                        * s_mxdc[kStart + get(thread(iStep) / cn<tileM_>)];

                    if (std::is_same_v<Tp_, half>)
                        *(half2*) &tmpB[i] = __float22half2_rn(tmp2);
                    else
                        *(bf162*) &tmpB[i] = __float22bfloat162_rn(tmp2);
                }

                *(int4*) ((char*) s_mxB + swizzle<tileM_ * 2, tileM_ * 2>(thread(iStep) * cn<2>, baseB(jK) * cn<2>))
                    = *(int4*) &tmpB[0];
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
                                asm volatile(
                                    "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                    : "=r"(r_mxB[y1][0]), "=r"(r_mxB[y1][1]), "=r"(r_mxB[y1][2]), "=r"(r_mxB[y1][3])
                                    : "r"(b_mxB + iK % pipeS_ * (tileM_ * tileK_ * 2)
                                        + 2
                                            * swz<tileM_ * 2, tileM_>(y1 * warpM_ * wmmaM_ + k * wmmaK_ * tileM_
                                                + threadIdx.z * wmmaM_ + threadIdx.x % 8 * tileM_
                                                + threadIdx.x / 8 % 2 * 8 + threadIdx.x / wmmaK_ * 8 * tileM_)));
                        }

                        if (x1 >= 0 && x1 < tileN_ / wmmaN_ / warpN_)
                        {
                            if (wmmaK_ == 16 && x1 % 2 == 0)
                                asm volatile(
                                    "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                    : "=r"(r_mxX[x1][0]), "=r"(r_mxX[x1][1]), "=r"(r_mxX[x1 + 1][0]),
                                    "=r"(r_mxX[x1 + 1][1])
                                    : "r"(b_mxX + iK % pipeS_ * (tileK_ * tileN_ * 2)
                                        + 2
                                            * swz<tileN_ * 2, tileN_>(x1 * warpN_ * wmmaN_ + k * wmmaK_ * tileN_
                                                + threadIdx.y * wmmaN_ + threadIdx.x % wmmaK_ * tileN_
                                                + threadIdx.x / wmmaK_ * warpN_ * wmmaN_)));
                        }
                    }
                }

#pragma unroll
            for (int y = 0; y < tileM_ / wmmaM_ / warpM_; y++)
#pragma unroll
                for (int x = 0; x < tileN_ / wmmaN_ / warpN_; x++)
                {
                    if (wmmaK_ == 16)
                    {
                        if (std::is_same_v<Tp_, half>)
                            asm volatile(
                                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 \n"
                                "    {%0, %1, %2, %3}, \n"
                                "    {%4, %5, %6, %7}, \n"
                                "    {%8, %9}, \n"
                                "    {%0, %1, %2, %3}; \n"
                                : "+f"(r_mxSt[y][x][0]), "+f"(r_mxSt[y][x][1]), "+f"(r_mxSt[y][x][2]),
                                "+f"(r_mxSt[y][x][3])
                                : "r"(r_mxB[y][0]), "r"(r_mxB[y][1]), "r"(r_mxB[y][2]), "r"(r_mxB[y][3]),
                                "r"(r_mxX[x][0]), "r"(r_mxX[x][1]));
                        else
                            asm volatile(
                                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 \n"
                                "    {%0, %1, %2, %3}, \n"
                                "    {%4, %5, %6, %7}, \n"
                                "    {%8, %9}, \n"
                                "    {%0, %1, %2, %3}; \n"
                                : "+f"(r_mxSt[y][x][0]), "+f"(r_mxSt[y][x][1]), "+f"(r_mxSt[y][x][2]),
                                "+f"(r_mxSt[y][x][3])
                                : "r"(r_mxB[y][0]), "r"(r_mxB[y][1]), "r"(r_mxB[y][2]), "r"(r_mxB[y][3]),
                                "r"(r_mxX[x][0]), "r"(r_mxX[x][1]));
                    }
                }
        }

        __syncthreads();

#pragma unroll
        for (Rn<UNROLL, div_up(tileM_ * tileK_, warpM_ * warpN_ * 256)> iStep; iStep.var < iStep.size; iStep.var++)
            if (thread(iStep) < cn<tileM_ * tileK_> && thread(iStep) / cn<tileM_> < L - blockIdx_y * Q - jK * cn<tileK_>
                && jK * cn<tileK_> < Q)
                cp_shared_global<16>(b_mxB + swizzle<tileM_ * 2, tileM_ * 2>(thread(iStep) * cn<2>, baseB(jK) * cn<2>),
                    g_mxBC_
                        + get((aStart + blockIdx_y * Q + jK * cn<tileK_> + thread(iStep) / cn<tileM_>) *cn<2> * G * N
                            + gStart * N + mStart * cn<tileM_> + thread(iStep) % cn<tileM_>));
            else if (thread(iStep) < cn<tileM_ * tileK_> && jK * cn<tileK_> < Q)
                *(int4*) ((char*) s_mxB + swizzle<tileM_ * 2, tileM_ * 2>(thread(iStep) * cn<2>, baseB(jK) * cn<2>))
                    = int4{0, 0, 0, 0};

#pragma unroll
        for (Rn<UNROLL, div_up(tileN_ * tileK_, warpM_ * warpN_ * 256)> iStep; iStep.var < iStep.size; iStep.var++)
            if (thread(iStep) < cn<tileN_ * tileK_> && thread(iStep) / cn<tileN_> < L - blockIdx_y * Q - jK * cn<tileK_>
                && jK * cn<tileK_> < Q)
                cp_shared_global<16>(b_mxX + swizzle<tileN_ * 2, tileN_ * 2>(thread(iStep) * cn<2>, baseX(jK) * cn<2>),
                    g_mxX_
                        + get((aStart + blockIdx_y * Q + jK * cn<tileK_> + thread(iStep) / cn<tileN_>) *H * P
                            + hStart * P + nStart * cn<tileN_> + thread(iStep) % cn<tileN_>));
            else if (thread(iStep) < cn<tileN_ * tileK_> && jK * cn<tileK_> < Q)
                *(int4*) ((char*) s_mxX + swizzle<tileN_ * 2, tileN_ * 2>(thread(iStep) * cn<2>, baseX(jK) * cn<2>))
                    = int4{0, 0, 0, 0};

        asm volatile("cp.async.commit_group;\n" ::);

        asm volatile("cp.async.wait_group %0;\n" ::"n"(pipeS_ - 1));

        __syncthreads();
    }

#pragma unroll
    for (int y = 0; y < tileM_ / wmmaM_ / warpM_; y++)
#pragma unroll
        for (int x = 0; x < tileN_ / wmmaN_ / warpN_; x++)
        {
            *(float2*) (g_mxSt_
                + get((cStart + blockIdx_y) * H * N * P + hStart * N * P
                    + (mStart * cn<tileM_> + Rn<UNROLL>{y} * cn<warpM_ * wmmaM_> + threadIdx_z * cn<wmmaM_>
                        + threadIdx_x / cn<4>) *P
                    + nStart * cn<tileN_> + Rn<UNROLL>{x} * cn<warpN_ * wmmaN_> + threadIdx_y * cn<wmmaN_>
                    + threadIdx_x % cn<4> * cn<2>))
                = *(float2*) &r_mxSt[y][x][0];

            *(float2*) (g_mxSt_
                + get((cStart + blockIdx_y) * H * N * P + hStart * N * P
                    + (mStart * cn<tileM_> + Rn<UNROLL>{y} * cn<warpM_ * wmmaM_> + cn<8> + threadIdx_z * cn<wmmaM_>
                        + threadIdx_x / cn<4>) *P
                    + nStart * cn<tileN_> + Rn<UNROLL>{x} * cn<warpN_ * wmmaN_> + threadIdx_y * cn<wmmaN_>
                    + threadIdx_x % cn<4> * cn<2>))
                = *(float2*) &r_mxSt[y][x][2];
        }

    asm volatile("cp.async.wait_group %0;\n" ::"n"(0));
#endif
}

ChunkStateKernelFuncFp16 getChunkStateKernelFp16(
    int B_, int L_, int H_, int P_, int G_, int N_, int Q_, dim3* blockDims_, dim3* threadDims_, int* sharedMem_)
{
    int B = B_;
    int L = L_;
    int H = H_;
    int P = P_;
    // int G = G_;
    int N = N_;
    int Q = Q_;
    int C = div_up(L, Q);

    int tileM = 64;
    int tileN = 64;
    int tileK = 32;
    int warpM = 1;
    int warpN = 2;
    int pipeS = 3;

    auto sharedMem = (tileM * tileK + tileK * tileN) * pipeS * 2 + Q * 8;

    *blockDims_ = dim3(H * P / tileN * N / tileM, C, B);
    *threadDims_ = dim3(32, warpN, warpM);
    *sharedMem_ = sharedMem;

    if (Q_ == 128)
        return chunk_state_kernel<128, 64, 64, 32, 16, 8, 16, 1, 2, 3, half>;
    else if (Q_ == 256)
        return chunk_state_kernel<256, 64, 64, 32, 16, 8, 16, 1, 2, 3, half>;
    else
        return nullptr;
}

ChunkStateKernelFuncBf16 getChunkStateKernelBf16(
    int B_, int L_, int H_, int P_, int G_, int N_, int Q_, dim3* blockDims_, dim3* threadDims_, int* sharedMem_)
{
    int B = B_;
    int L = L_;
    int H = H_;
    int P = P_;
    // int G = G_;
    int N = N_;
    int Q = Q_;
    int C = div_up(L, Q);

    int tileM = 64;
    int tileN = 64;
    int tileK = 32;
    int warpM = 1;
    int warpN = 2;
    int pipeS = 3;

    auto sharedMem = (tileM * tileK + tileK * tileN) * pipeS * 2 + Q * 8;

    *blockDims_ = dim3(H * P / tileN * N / tileM, C, B);
    *threadDims_ = dim3(32, warpN, warpM);
    *sharedMem_ = sharedMem;

    if (Q_ == 128)
        return chunk_state_kernel<128, 64, 64, 32, 16, 8, 16, 1, 2, 3, bf16>;
    else if (Q_ == 256)
        return chunk_state_kernel<256, 64, 64, 32, 16, 8, 16, 1, 2, 3, bf16>;
    else
        return nullptr;
}

} // namespace kernels
} // namespace tensorrt_llm

// vim: ts=2 sw=2 sts=2 et sta
