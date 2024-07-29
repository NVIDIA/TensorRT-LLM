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

typedef void (*BmmChunkKernelFuncFp16)(int B_, int L_, int H_, int P_, int G_, int N_,
    //  const half  *g_mxY_,  // B*L*H*P
    //  const half  *g_mxOs_, // B*C*H*N*P
    //  const half  *g_mxFs_, // B  *H*N*P
    //  const float *g_mxSt_, // B*C*H*N*P
    //  const float *g_mxdc_, // B*C*H*Q
    //  const float *g_mxdA_, // B*C*H*Q
    //  const half  *g_mxdt_, // B*L*(2*H*P+2*G*N+H) or B*L*(H*P+2*G*N+H)
    //  const float *g_mxdb_, //     H
    //  const float *g_mxA_,  //     H
    half* g_mxCB_,        // B*C*G*Q*Q
    half const* g_mxXBC_, // B*L*(H*P+2*G*N)
                          //  const float *g_mxD_,  //     H
                          //  const half  *g_mxX_,  // B*L*H*P
                          //  const half  *g_mxZ_,  // B*L*(2*H*P+2*G*N+H)
    bool removePadding_, int const* lastTokenIdsPtr_);

typedef void (*BmmChunkKernelFuncBf16)(int B_, int L_, int H_, int P_, int G_, int N_,
    //  const bf16  *g_mxY_,  // B*L*H*P
    //  const bf16  *g_mxOs_, // B*C*H*N*P
    //  const bf16  *g_mxFs_, // B  *H*N*P
    //  const float *g_mxSt_, // B*C*H*N*P
    //  const float *g_mxdc_, // B*C*H*Q
    //  const float *g_mxdA_, // B*C*H*Q
    //  const bf16  *g_mxdt_, // B*L*(2*H*P+2*G*N+H) or B*L*(H*P+2*G*N+H)
    //  const float *g_mxdb_, //     H
    //  const float *g_mxA_,  //     H
    bf16* g_mxCB_,        // B*C*G*Q*Q
    bf16 const* g_mxXBC_, // B*L*(H*P+2*G*N)
                          //  const float *g_mxD_,  //     H
                          //  const bf16  *g_mxX_,  // B*L*H*P
                          //  const bf16  *g_mxZ_,  // B*L*(2*H*P+2*G*N+H)
    bool removePadding_, int const* lastTokenIdsPtr_);

template <int Q_, int tileM_, int tileN_, int tileK_, // smem size, per sm
    int wmmaM_, int wmmaN_, int wmmaK_,               // wmma size, per instruction
    int warpM_, int warpN_,                           // warp number
    int pipeS_, class Tp_>
__global__ std::enable_if_t<std::is_same_v<Tp_, half> || std::is_same_v<Tp_, __nv_bfloat16>> bmm_chunk_kernel(int B_,
    int L_, int H_, int P_, int G_, int N_,
    //  const Tp_   *g_mxY_,  // B*L*H*P
    //  const Tp_   *g_mxOs_, // B*C*H*N*P
    //  const Tp_   *g_mxFs_, // B  *H*N*P
    //  const float *g_mxSt_, // B*C*H*N*P
    //  const float *g_mxdc_, // B*C*H*Q
    //  const float *g_mxdA_, // B*C*H*Q
    //  const Tp_   *g_mxdt_, // B*L*(2*H*P+2*G*N+H) or B*L*(H*P+2*G*N+H)
    //  const Wt_   *g_mxdb_, //     H
    //  const Wt_   *g_mxA_,  //     H
    Tp_* g_mxCB_,        // B*C*G*Q*Q
    Tp_ const* g_mxXBC_, // B*L*(H*P+2*G*N)
                         //  const Wt_   *g_mxD_,  //     H
                         //  const Tp_   *g_mxX_,  // B*L*H*P
                         //  const Tp_   *g_mxZ_,  // B*L*(2*H*P+2*G*N+H)
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

    auto xbcDim = Rn<ID>{H_ * P_ + 2 * G_ * N_};
    auto bOffset = Rn<ID>{H_ * P_};
    auto cOffset = Rn<ID>{H_ * P_ + G_ * N_};

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

    auto gStart = blockIdx_x / (Q / cn<tileN_>) / (Q / cn<tileM_>);
    auto mStart = blockIdx_x / (Q / cn<tileN_>) % (Q / cn<tileM_>);
    auto nStart = blockIdx_x % (Q / cn<tileN_>);

    extern __shared__ float smem[];

    Tp_* s_mxC = (Tp_*) smem;
    Tp_* s_mxB = (Tp_*) smem + tileM_ * tileK_ * pipeS_;
    Tp_* s_mxCB = (Tp_*) smem;

    unsigned b_base = __nvvm_get_smem_pointer(smem);

    unsigned b_mxC = b_base;
    unsigned b_mxB = b_base + tileM_ * tileK_ * pipeS_ * sizeof(Tp_);
    unsigned b_mxCB = b_base;

    using std::array;

    register array<array<array<float, wmmaM_ * wmmaN_ / 32>, tileN_ / wmmaN_ / warpN_>, tileM_ / wmmaM_ / warpM_> r_mxCB
        = array<array<array<float, wmmaM_ * wmmaN_ / 32>, tileN_ / wmmaN_ / warpN_>, tileM_ / wmmaM_ / warpM_>();
    register array<array<unsigned, wmmaM_ * wmmaK_ / 64>, tileM_ / wmmaM_ / warpM_> r_mxC;
    register array<array<unsigned, wmmaK_ * wmmaN_ / 64>, tileN_ / wmmaN_ / warpN_> r_mxB;

    constexpr int step = std::max(
        1, tileM_ / wmmaM_ / warpM_ * tileN_ / wmmaN_ / warpN_ / (tileM_ / wmmaM_ / warpM_ + tileN_ / wmmaN_ / warpN_));

    auto baseC = [](auto iK) { return iK % cn<pipeS_> * cn<tileM_> * cn<tileK_>; };
    auto baseB = [](auto iK) { return iK % cn<pipeS_> * cn<tileN_> * cn<tileK_>; };

    auto thread = [=](auto iStep)
    {
        return iStep * cn<warpM_ * warpN_ * 256> + threadIdx_z * cn<warpN_ * 256> + threadIdx_y * cn<256>
            + threadIdx_x * cn<8>;
    };

#pragma unroll
    for (Rn<UNROLL, pipeS_> iK; iK.var < iK.size; iK.var++)
    {
#pragma unroll
        for (Rn<UNROLL, div_up(tileM_ * tileK_, warpM_ * warpN_ * 256)> iStep; iStep.var < iStep.size; iStep.var++)
            if (thread(iStep) < cn<tileM_ * tileK_>
                && thread(iStep) / cn<tileK_> < L - blockIdx_y * Q - mStart * cn<tileM_>)
                cp_shared_global<16>(b_mxC + swizzle<tileK_ * 2, tileK_ * 2>(thread(iStep) * cn<2>, baseC(iK) * cn<2>),
                    g_mxXBC_
                        + get((aStart + blockIdx_y * Q + mStart * cn<tileM_> + thread(iStep) / cn<tileK_>) *xbcDim
                            + cOffset + gStart * N + iK * cn<tileK_> + thread(iStep) % cn<tileK_>));
            else if (thread(iStep) < cn<tileM_ * tileK_>)
                *(int4*) ((char*) s_mxC + swizzle<tileK_ * 2, tileK_ * 2>(thread(iStep) * cn<2>, baseC(iK) * cn<2>))
                    = int4{0, 0, 0, 0};

#pragma unroll
        for (Rn<UNROLL, div_up(tileN_ * tileK_, warpM_ * warpN_ * 256)> iStep; iStep.var < iStep.size; iStep.var++)
            if (thread(iStep) < cn<tileN_ * tileK_>
                && thread(iStep) / cn<tileK_> < L - blockIdx_y * Q - nStart * cn<tileN_>)
                cp_shared_global<16>(b_mxB + swizzle<tileK_ * 2, tileK_ * 2>(thread(iStep) * cn<2>, baseB(iK) * cn<2>),
                    g_mxXBC_
                        + get((aStart + blockIdx_y * Q + nStart * cn<tileN_> + thread(iStep) / cn<tileK_>) *xbcDim
                            + bOffset + gStart * N + iK * cn<tileK_> + thread(iStep) % cn<tileK_>));
            else if (thread(iStep) < cn<tileN_ * tileK_>)
                *(int4*) ((char*) s_mxB + swizzle<tileK_ * 2, tileK_ * 2>(thread(iStep) * cn<2>, baseB(iK) * cn<2>))
                    = int4{0, 0, 0, 0};

        cp_commit_group();
    }

    asm volatile("cp.async.wait_group %0;\n" ::"n"(pipeS_ - 1));

    __syncthreads();

    for (int iK = pipeS_; iK < N_ / tileK_ + pipeS_; iK++)
    {
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
                                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                    : "=r"(r_mxC[y1][0]), "=r"(r_mxC[y1][1]), "=r"(r_mxC[y1][2]), "=r"(r_mxC[y1][3])
                                    : "r"(b_mxC + iK % pipeS_ * (tileM_ * tileK_ * 2)
                                        + 2
                                            * swz<tileK_ * 2, tileK_>(y1 * warpM_ * wmmaM_ * tileK_ + k * wmmaK_
                                                + threadIdx.z * wmmaM_ * tileK_ + threadIdx.x % 16 * tileK_
                                                + threadIdx.x / 16 * 8)));
                        }

                        if (x1 >= 0 && x1 < tileN_ / wmmaN_ / warpN_)
                        {
                            if (wmmaK_ == 16 && x1 % 2 == 0)
                                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                             : "=r"(r_mxB[x1][0]), "=r"(r_mxB[x1][1]), "=r"(r_mxB[x1 + 1][0]),
                                             "=r"(r_mxB[x1 + 1][1])
                                             : "r"(b_mxB + iK % pipeS_ * (tileK_ * tileN_ * 2)
                                                 + 2
                                                     * swz<tileK_ * 2, tileK_>(x1 * warpN_ * wmmaN_ * tileK_
                                                         + k * wmmaK_ + threadIdx.y * wmmaN_ * tileK_
                                                         + threadIdx.x % 8 * tileK_ + threadIdx.x / 8 % 2 * 8
                                                         + threadIdx.x / wmmaK_ * warpN_ * wmmaN_ * tileK_)));
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
                                : "+f"(r_mxCB[y][x][0]), "+f"(r_mxCB[y][x][1]), "+f"(r_mxCB[y][x][2]),
                                "+f"(r_mxCB[y][x][3])
                                : "r"(r_mxC[y][0]), "r"(r_mxC[y][1]), "r"(r_mxC[y][2]), "r"(r_mxC[y][3]),
                                "r"(r_mxB[x][0]), "r"(r_mxB[x][1]));
                        else
                            asm volatile(
                                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 \n"
                                "    {%0, %1, %2, %3}, \n"
                                "    {%4, %5, %6, %7}, \n"
                                "    {%8, %9}, \n"
                                "    {%0, %1, %2, %3}; \n"
                                : "+f"(r_mxCB[y][x][0]), "+f"(r_mxCB[y][x][1]), "+f"(r_mxCB[y][x][2]),
                                "+f"(r_mxCB[y][x][3])
                                : "r"(r_mxC[y][0]), "r"(r_mxC[y][1]), "r"(r_mxC[y][2]), "r"(r_mxC[y][3]),
                                "r"(r_mxB[x][0]), "r"(r_mxB[x][1]));
                    }
                }
        }

        __syncthreads();

        if (iK * tileK_ < N_)
        {

            auto jK = Rn<>{iK};
#pragma unroll
            for (Rn<UNROLL, div_up(tileM_ * tileK_, warpM_ * warpN_ * 256)> iStep; iStep.var < iStep.size; iStep.var++)
                if (thread(iStep) < cn<tileM_ * tileK_>
                    && thread(iStep) / cn<tileK_> < L - blockIdx_y * Q - mStart * cn<tileM_>)
                    cp_shared_global<16>(
                        b_mxC + swizzle<tileK_ * 2, tileK_ * 2>(thread(iStep) * cn<2>, baseC(jK) * cn<2>),
                        g_mxXBC_
                            + get((aStart + blockIdx_y * Q + mStart * cn<tileM_> + thread(iStep) / cn<tileK_>) *xbcDim
                                + cOffset + gStart * N + jK * cn<tileK_> + thread(iStep) % cn<tileK_>));
                else if (thread(iStep) < cn<tileM_ * tileK_>)
                    *(int4*) ((char*) s_mxC + swizzle<tileK_ * 2, tileK_ * 2>(thread(iStep) * cn<2>, baseC(jK) * cn<2>))
                        = int4{0, 0, 0, 0};

#pragma unroll
            for (Rn<UNROLL, div_up(tileN_ * tileK_, warpM_ * warpN_ * 256)> iStep; iStep.var < iStep.size; iStep.var++)
                if (thread(iStep) < cn<tileN_ * tileK_>
                    && thread(iStep) / cn<tileK_> < L - blockIdx_y * Q - nStart * cn<tileN_>)
                    cp_shared_global<16>(
                        b_mxB + swizzle<tileK_ * 2, tileK_ * 2>(thread(iStep) * cn<2>, baseB(jK) * cn<2>),
                        g_mxXBC_
                            + get((aStart + blockIdx_y * Q + nStart * cn<tileN_> + thread(iStep) / cn<tileK_>) *xbcDim
                                + bOffset + gStart * N + jK * cn<tileK_> + thread(iStep) % cn<tileK_>));
                else if (thread(iStep) < cn<tileN_ * tileK_>)
                    *(int4*) ((char*) s_mxB + swizzle<tileK_ * 2, tileK_ * 2>(thread(iStep) * cn<2>, baseB(jK) * cn<2>))
                        = int4{0, 0, 0, 0};
        }

        asm volatile("cp.async.commit_group;\n" ::);

        asm volatile("cp.async.wait_group %0;\n" ::"n"(pipeS_ - 1));

        __syncthreads();
    }

#pragma unroll
    for (int y = 0; y < tileM_ / wmmaM_ / warpM_; y++)
#pragma unroll
        for (int x = 0; x < tileN_ / wmmaN_ / warpN_; x++)
        {
            if (std::is_same_v<Tp_, half>)
            {
                *(half2*) &r_mxCB[y][x][0] = __floats2half2_rn(r_mxCB[y][x][0], r_mxCB[y][x][1]);
                *(half2*) &r_mxCB[y][x][2] = __floats2half2_rn(r_mxCB[y][x][2], r_mxCB[y][x][3]);
            }
            else
            {
                *(bf162*) &r_mxCB[y][x][0] = __floats2bfloat162_rn(r_mxCB[y][x][0], r_mxCB[y][x][1]);
                *(bf162*) &r_mxCB[y][x][2] = __floats2bfloat162_rn(r_mxCB[y][x][2], r_mxCB[y][x][3]);
            }
        }

#pragma unroll
    for (int y = 0; y < tileM_ / wmmaM_ / warpM_; y++)
#pragma unroll
        for (int x = 0; x < tileN_ / wmmaN_ / warpN_; x++)
        {
            asm volatile("st.shared.b32 [%0], %1;\n" ::"r"(b_mxCB
                             + 2
                                 * swz<tileN_ * 2, tileN_>(y * warpM_ * wmmaM_ * tileN_ + x * warpN_ * wmmaN_
                                     + (threadIdx.z * wmmaM_ + threadIdx.x / 4) * tileN_
                                     + (threadIdx.y * wmmaN_ + threadIdx.x % 4 * 2))),
                "r"(*(unsigned*) &r_mxCB[y][x][0]));
            asm volatile("st.shared.b32 [%0], %1;\n" ::"r"(b_mxCB
                             + 2
                                 * swz<tileN_ * 2, tileN_>(y * warpM_ * wmmaM_ * tileN_ + 8 * tileN_
                                     + x * warpN_ * wmmaN_ + (threadIdx.z * wmmaM_ + threadIdx.x / 4) * tileN_
                                     + (threadIdx.y * wmmaN_ + threadIdx.x % 4 * 2))),
                "r"(*(unsigned*) &r_mxCB[y][x][2]));
        }

    __syncthreads();

#pragma unroll
    for (Rn<UNROLL, div_up(tileM_ * tileN_, warpM_ * warpN_ * 256)> iStep; iStep.var < iStep.size; iStep.var++)
        if (thread(iStep) < cn<tileM_ * tileN_>)
            *(int4*) (g_mxCB_
                + get(cStart * G * Q * Q + blockIdx_y * G * Q * Q + gStart * Q * Q
                    + (mStart * cn<tileM_> + thread(iStep) / cn<tileN_>) *Q + nStart * cn<tileN_>
                    + thread(iStep) % cn<tileN_>))
                = *(int4*) ((char*) s_mxCB + swizzle<tileN_ * 2, tileN_ * 2>(thread(iStep) * cn<2>));

    asm volatile("cp.async.wait_group %0;\n" ::"n"(0));
#endif
}

BmmChunkKernelFuncFp16 getBmmChunkKernelFp16(
    int B_, int L_, int G_, int N_, int Q_, dim3* blockDims_, dim3* threadDims_, int* sharedMem_)
{
    int B = B_;
    int L = L_;
    // int H = H_;
    // int P = P_;
    int G = G_;
    // int N = N_;
    int Q = Q_;
    int C = div_up(L, Q);

    int tileM = 128;
    int tileN = 64;
    int tileK = 32;
    int warpM = 2;
    int warpN = 1;
    int pipeS = 2;

    auto sharedMem = std::max((tileM * tileK + tileK * tileN) * pipeS * 2, (tileM * tileN) * 2);

    *blockDims_ = dim3(G * Q / tileN * Q / tileM, C, B);
    *threadDims_ = dim3(32, warpN, warpM);
    *sharedMem_ = sharedMem;

    if (Q_ == 128)
        return bmm_chunk_kernel<128, 128, 64, 32, 16, 8, 16, 2, 1, 2, half>;
    else if (Q_ == 256)
        return bmm_chunk_kernel<256, 128, 64, 32, 16, 8, 16, 2, 1, 2, half>;
    else
        return nullptr;
}

BmmChunkKernelFuncBf16 getBmmChunkKernelBf16(
    int B_, int L_, int G_, int N_, int Q_, dim3* blockDims_, dim3* threadDims_, int* sharedMem_)
{
    int B = B_;
    int L = L_;
    // int H = H_;
    // int P = P_;
    int G = G_;
    // int N = N_;
    int Q = Q_;
    int C = div_up(L, Q);

    int tileM = 128;
    int tileN = 64;
    int tileK = 32;
    int warpM = 2;
    int warpN = 1;
    int pipeS = 2;

    auto sharedMem = std::max((tileM * tileK + tileK * tileN) * pipeS * 2, (tileM * tileN) * 2);

    *blockDims_ = dim3(G * Q / tileN * Q / tileM, C, B);
    *threadDims_ = dim3(32, warpN, warpM);
    *sharedMem_ = sharedMem;

    if (Q_ == 128)
        return bmm_chunk_kernel<128, 128, 64, 32, 16, 8, 16, 2, 1, 2, bf16>;
    else if (Q_ == 256)
        return bmm_chunk_kernel<256, 128, 64, 32, 16, 8, 16, 2, 1, 2, bf16>;
    else
        return nullptr;
}

} // namespace kernels
} // namespace tensorrt_llm

// vim: ts=2 sw=2 sts=2 et sta
