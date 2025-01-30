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
#include "CudaType.h"
#include "Poly.h"

namespace tensorrt_llm
{
namespace kernels
{

typedef void (*ChunkCumsumKernelFunc)(int B_, int L_, int H_, int P_, int G_, int N_,
    //  const void *g_mxY_,  // Tp_   B*L*H*P
    //  const void *g_mxOs_, // Tp_   B*C*H*N*P
    //  const void *g_mxFs_, // Tp_   B  *H*N*P
    //  const void *g_mxSt_, // float B*C*H*N*P
    void* g_mxdc_,       // float B*C*H*Q
    void* g_mxdA_,       // float B*C*H*Q
    void const* g_mxdt_, // Tp_   B*L*((g_mxZ?2:1)*H*P+2*G+round_up(H,8))
    void const* g_mxdb_, // Wt_       H
    void const* g_mxA_,  // Wt_       H
                         //  const void *g_mxCB_, // Tp_   B*C*G*Q*Q
                         //  const void *g_mxD_,  // Wt_       H
                         //  const void *g_mxX_,  // Tp_   B*L*(H*P+2*G*N)
    void const* g_mxZ_,  // g_mxdt_ or nullptr
    bool removePadding_, int const* lastTokenIdsPtr_, bool dtSoftplus_);

template <int Q_, int tileH_, int warpH_, class Tp_, class Wt_>
__global__ std::enable_if_t<std::is_same_v<Tp_, half> || std::is_same_v<Tp_, __nv_bfloat16>> chunk_cumsum_kernel(int B_,
    int L_, int H_, int P_, int G_, int N_,
    //  const void *g_mxY_,  // Tp_   B*L*H*P
    //  const void *g_mxOs_, // Tp_   B*C*H*N*P
    //  const void *g_mxFs_, // Tp_   B  *H*N*P
    //  const void *g_mxSt_, // float B*C*H*N*P
    void* g_mxdc_,       // float B*C*H*Q
    void* g_mxdA_,       // float B*C*H*Q
    void const* g_mxdt_, // Tp_   B*L*((g_mxZ?2:1)*H*P+2*G+round_up(H,8))
    void const* g_mxdb_, // Wt_       H
    void const* g_mxA_,  // Wt_       H
                         //  const void *g_mxCB_, // Tp_   B*C*G*Q*Q
                         //  const void *g_mxD_,  // Wt_       H
                         //  const void *g_mxX_,  // Tp_   B*L*(H*P+2*G*N)
    void const* g_mxZ_,  // g_mxdt_ or nullptr
    bool removePadding_, int const* lastTokenIdsPtr_, bool dtSoftplus_)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
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

    auto Z_stride = Rn<ID>{(g_mxZ_ ? 2 : 1) * H_ * P_ + 2 * G_ * N_ + round_up(H_, 8)};

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

    //  const Tp_   *g_mxY  = (const Tp_   *)g_mxY_;
    //  const Tp_   *g_mxOs = (const Tp_   *)g_mxOs_;
    //  const Tp_   *g_mxFs = (const Tp_   *)g_mxFs_;
    //  const float *g_mxSt = (const float *)g_mxSt_;
    float* g_mxdc = (float*) g_mxdc_ + int64_t(get(cStart + blockIdx_y)) * get(H * Q);
    float* g_mxdA = (float*) g_mxdA_ + int64_t(get(cStart + blockIdx_y)) * get(H * Q);
    Tp_ const* g_mxdt = (Tp_ const*) g_mxdt_ + int64_t(get(aStart + blockIdx_y * Q)) * get(Z_stride);
    Wt_ const* g_mxdb = (Wt_ const*) g_mxdb_;
    Wt_ const* g_mxA = (Wt_ const*) g_mxA_;
    //  const Tp_   *g_mxCB = (const Tp_   *)g_mxCB_;
    //  const Wt_   *g_mxD  = (const Wt_   *)g_mxD_;
    //  const Tp_   *g_mxX  = (const Tp_   *)g_mxX_;
    //  const Tp_   *g_mxZ  = (const Tp_   *)g_mxZ_;

    extern __shared__ float smem[];

    float* s_mxdc = smem;
    float* s_mxdb = smem + Q_ * tileH_;
    float* s_mxA = smem + Q_ * tileH_ + tileH_;

    auto thread = [=](auto iStep) { return iStep * cn<warpH_ * 32> + threadIdx_y * cn<32> + threadIdx_x; };

#pragma unroll
    for (Rn<UNROLL, div_up(tileH_, warpH_ * 32)> iStep; iStep.var < iStep.size; iStep.var++)
        if (thread(iStep) < cn<tileH_>)
            if (blockIdx_x * cn<tileH_> + thread(iStep) < H)
            {
                s_mxdb[get(thread(iStep))] = g_mxdb ? float(g_mxdb[get(blockIdx_x * cn<tileH_> + thread(iStep))]) : 0.f;
                s_mxA[get(thread(iStep))] = float(g_mxA[get(blockIdx_x * cn<tileH_> + thread(iStep))]);
            }
            else
            {
                s_mxdb[get(thread(iStep))] = 0.f;
                s_mxA[get(thread(iStep))] = 0.f;
            }

    __syncthreads();

#pragma unroll
    for (Rn<UNROLL, div_up(Q_ * tileH_, warpH_ * 256)> iStep; iStep.var < iStep.size; iStep.var++)
        if (thread(iStep) * cn<8> < cn<Q_ * tileH_>)
        {
            if (blockIdx_y * Q + thread(iStep) * cn<8> / cn<tileH_> < L)
            {
                Tp_ tmp[8];

#pragma unroll
                for (int i = 0; i < 8; i += 2)
                    if (blockIdx_x * cn<tileH_> + thread(iStep) * cn<8> % cn<tileH_> + Rn<UNROLL, 8>{i} < H)
                        *(int*) &tmp[i]
                            = *(int*) &g_mxdt[get(thread(iStep) * cn<8> / cn<tileH_> * Z_stride + Z_stride
                                                  + blockIdx_x * cn<tileH_> + thread(iStep) * cn<8> % cn<tileH_>)
                                - round_up(H_, 8) + i];
                    else
                        *(int*) &tmp[i] = 0;

#pragma unroll
                for (int i = 0; i < 8; i += 2)
                {
                    float2 tmp2 = std::is_same_v<Tp_, half> ? __half22float2(*(half2*) &tmp[i])
                                                            : bf1622float2(*(bf162*) &tmp[i]);

                    tmp2.x += s_mxdb[get(thread(iStep) * cn<8> % cn<tileH_> + Rn<UNROLL, 8>{i})];
                    tmp2.y += s_mxdb[get(thread(iStep) * cn<8> % cn<tileH_> + Rn<UNROLL, 8>{i + 1})];

                    if (dtSoftplus_)
                    {
                        float softplusx = log1p(expf(tmp2.x));
                        float softplusy = log1p(expf(tmp2.y));
                        tmp2.x = tmp2.x > 32.f ? tmp2.x : softplusx;
                        tmp2.y = tmp2.y > 32.f ? tmp2.y : softplusy;
                    }
                    else
                    {
                        tmp2.x = tmp2.x > 0.f ? tmp2.x : 0.f;
                        tmp2.y = tmp2.y > 0.f ? tmp2.y : 0.f;
                    }

                    s_mxdc[get((thread(iStep) * cn<8> + Rn<UNROLL, 8>{i}) % cn<tileH_> * Q
                        + (thread(iStep) * cn<8> + Rn<UNROLL, 8>{i}) / cn<tileH_>)]
                        = tmp2.x;
                    s_mxdc[get((thread(iStep) * cn<8> + Rn<UNROLL, 8>{i + 1}) % cn<tileH_> * Q
                        + (thread(iStep) * cn<8> + Rn<UNROLL, 8>{i + 1}) / cn<tileH_>)]
                        = tmp2.y;
                }
            }
            else
            {
#pragma unroll
                for (int i = 0; i < 8; i++)
                {
                    // Set dc to zero out of seq length, a must for chunkstate & chunkscan.
                    s_mxdc[get((thread(iStep) * cn<8> + Rn<UNROLL, 8>{i}) % cn<tileH_> * Q
                        + (thread(iStep) * cn<8> + Rn<UNROLL, 8>{i}) / cn<tileH_>)]
                        = 0.f;
                }
            }
        }

    __syncthreads();

#pragma unroll
    for (Rn<UNROLL, div_up(Q_ * tileH_, warpH_ * 128)> iStep; iStep.var < iStep.size; iStep.var++)
        if (thread(iStep) * cn<4> < cn<Q_ * tileH_>)
            if (blockIdx_x * cn<tileH_> + thread(iStep) * cn<4> / Q < H)
            {
                float4 tmp4 = *(float4*) &s_mxdc[get(thread(iStep) * cn<4>)];

                *(float4*) &g_mxdc[get(
                    (thread(iStep) * cn<4> / Q + blockIdx_x * cn<tileH_>) *Q + thread(iStep) * cn<4> % Q)]
                    = tmp4;
            }

    __syncthreads();

#pragma unroll
    for (Rn<UNROLL, div_up(tileH_, warpH_ * 32)> iStep; iStep.var < iStep.size; iStep.var++)
        if (thread(iStep) < cn<tileH_>)
        {
            float sum = 0.f;

#pragma unroll
            for (Rn<UNROLL, Q_> iQ; iQ.var < iQ.size; iQ.var++)
            {
                sum += s_mxdc[get(thread(iStep) * Q + iQ)];
                s_mxdc[get(thread(iStep) * Q + iQ)] = sum;
            }
        }

    __syncthreads();

#pragma unroll
    for (Rn<UNROLL, div_up(Q_ * tileH_, warpH_ * 128)> iStep; iStep.var < iStep.size; iStep.var++)
        if (thread(iStep) * cn<4> < cn<Q_ * tileH_>)
            if (blockIdx_x * cn<tileH_> + thread(iStep) * cn<4> / Q < H)
            {
                float r_A = s_mxA[get(thread(iStep) * cn<4> / Q)];

                float4 tmp4 = *(float4*) &s_mxdc[get(thread(iStep) * cn<4>)];

                tmp4.x *= r_A;
                tmp4.y *= r_A;
                tmp4.z *= r_A;
                tmp4.w *= r_A;

                *(float4*) &g_mxdA[get(
                    (thread(iStep) * cn<4> / Q + blockIdx_x * cn<tileH_>) *Q + thread(iStep) * cn<4> % Q)]
                    = tmp4;
            }
#endif
}

typedef ChunkCumsumKernelFunc (*GetChunkCumsumKernelFunc)(int B_, int L_, int H_, int P_, int G_, int N_, int Q_,
    int numTokens_, dim3* blockDims_, dim3* threadDims_, int* sharedMem_);

template <class Tp_, class Wt_>
ChunkCumsumKernelFunc getChunkCumsumKernel(int B_, int L_, int H_, int P_, int G_, int N_, int Q_, int numTokens_,
    dim3* blockDims_, dim3* threadDims_, int* sharedMem_)
{
    int B = B_;
    int L = L_;
    int H = round_up(H_, 8);
    // int P = P_;
    // int G = G_;
    // int N = N_;
    int Q = Q_;
    int C = div_up(L, Q);

    int64_t compute = int64_t(numTokens_) * H * Q;

    auto set = [&](int tileH, int warpH, ChunkCumsumKernelFunc func)
    {
        auto sharedMem = (Q + 2) * tileH * 4;

        *blockDims_ = dim3(div_up(H, tileH), C, B);
        *threadDims_ = dim3(32, warpH);
        *sharedMem_ = sharedMem;

        return func;
    };

    if (Q == 256)
    {
        if (H % 16 == 0)
        {
            if (compute >= (1LL << 29))
                return set(16, 8, chunk_cumsum_kernel<256, 16, 8, Tp_, Wt_>);
            else
                return set(8, 8, chunk_cumsum_kernel<256, 8, 8, Tp_, Wt_>);
        }
        if (H % 8 == 0)
        {
            if (compute >= (1LL << 29))
                return set(16, 8, chunk_cumsum_kernel<256, 16, 8, Tp_, Wt_>);
            else
                return set(8, 8, chunk_cumsum_kernel<256, 8, 8, Tp_, Wt_>);
        }
    }
    if (Q == 128)
    {
        if (H % 16 == 0)
        {
            if (compute >= (1LL << 27))
                return set(16, 8, chunk_cumsum_kernel<128, 16, 8, Tp_, Wt_>);
            else if (compute >= (1LL << 26))
                return set(8, 8, chunk_cumsum_kernel<128, 8, 8, Tp_, Wt_>);
            else if (compute >= (1LL << 12))
                return set(8, 4, chunk_cumsum_kernel<128, 8, 4, Tp_, Wt_>);
            else if (compute >= (1LL << 9))
                return set(8, 8, chunk_cumsum_kernel<128, 8, 8, Tp_, Wt_>);
            else
                return set(8, 4, chunk_cumsum_kernel<128, 8, 4, Tp_, Wt_>);
        }
        if (H % 8 == 0)
        {
            if (compute >= (1LL << 27))
                return set(16, 8, chunk_cumsum_kernel<128, 16, 8, Tp_, Wt_>);
            else if (compute >= (1LL << 26))
                return set(8, 8, chunk_cumsum_kernel<128, 8, 8, Tp_, Wt_>);
            else if (compute >= (1LL << 12))
                return set(8, 4, chunk_cumsum_kernel<128, 8, 4, Tp_, Wt_>);
            else if (compute >= (1LL << 9))
                return set(8, 8, chunk_cumsum_kernel<128, 8, 8, Tp_, Wt_>);
            else
                return set(8, 4, chunk_cumsum_kernel<128, 8, 4, Tp_, Wt_>);
        }
    }

    return nullptr;
}

extern GetChunkCumsumKernelFunc getChunkCumsumKernel_fp16_fp16;
extern GetChunkCumsumKernelFunc getChunkCumsumKernel_fp16_fp32;
extern GetChunkCumsumKernelFunc getChunkCumsumKernel_bf16_bf16;
extern GetChunkCumsumKernelFunc getChunkCumsumKernel_bf16_fp32;

static inline ChunkCumsumKernelFunc getChunkCumsumKernel(int B_, int L_, int H_, int P_, int G_, int N_, int Q_,
    int numTokens_, dim3* blockDims_, dim3* threadDims_, int* sharedMem_, CudaType tp_ = CT_FP16,
    CudaType wt_ = CT_FP32)
{
    if (tp_ == CT_FP16 && wt_ == CT_FP16)
        return getChunkCumsumKernel_fp16_fp16(
            B_, L_, H_, P_, G_, N_, Q_, numTokens_, blockDims_, threadDims_, sharedMem_);
    else if (tp_ == CT_FP16 && wt_ == CT_FP32)
        return getChunkCumsumKernel_fp16_fp32(
            B_, L_, H_, P_, G_, N_, Q_, numTokens_, blockDims_, threadDims_, sharedMem_);
    else if (tp_ == CT_BF16 && wt_ == CT_BF16)
        return getChunkCumsumKernel_bf16_bf16(
            B_, L_, H_, P_, G_, N_, Q_, numTokens_, blockDims_, threadDims_, sharedMem_);
    else if (tp_ == CT_BF16 && wt_ == CT_FP32)
        return getChunkCumsumKernel_bf16_fp32(
            B_, L_, H_, P_, G_, N_, Q_, numTokens_, blockDims_, threadDims_, sharedMem_);

    return nullptr;
}

} // namespace kernels
} // namespace tensorrt_llm

// vim: ts=2 sw=2 sts=2 et sta
