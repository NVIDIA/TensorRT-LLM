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

typedef void (*StatePassingKernelFunc)(int B_, int L_, int H_, int P_, int G_, int N_,
    //  const void *g_mxY_,  // Tp_   B*L*H*P
    void* g_mxOs_,       // Tp_   B*C*H*N*P
    void* g_mxFs_,       // Tp_   B  *H*N*P
    void const* g_mxSt_, // float B*C*H*N*P
                         //  const void *g_mxdc_, // float B*C*H*Q
    void const* g_mxdA_, // float B*C*H*Q
                         //  const void *g_mxdt_, // Tp_   B*L*((g_mxZ?2:1)*H*P+2*G+round_up(H,8))
                         //  const void *g_mxdb_, // Wt_       H
                         //  const void *g_mxA_,  // Wt_       H
                         //  const void *g_mxCB_, // Tp_   B*C*G*Q*Q
                         //  const void *g_mxD_,  // Wt_       H
                         //  const void *g_mxX_,  // Tp_   B*L*(H*P+2*G*N)
                         //  const void *g_mxZ_,  // g_mxdt_ or nullptr
    bool removePadding_, int const* lastTokenIdsPtr_, int const* stateSlotMappingPtr_);

template <int Q_, int tileH_, int warpH_, class Tp_>
__global__ std::enable_if_t<std::is_same_v<Tp_, half> || std::is_same_v<Tp_, __nv_bfloat16>> state_passing_kernel(
    int B_, int L_, int H_, int P_, int G_, int N_,
    //  const void *g_mxY_,  // Tp_   B*L*H*P
    void* g_mxOs_,       // Tp_   B*C*H*N*P
    void* g_mxFs_,       // Tp_   B  *H*N*P
    void const* g_mxSt_, // float B*C*H*N*P
                         //  const void *g_mxdc_, // float B*C*H*Q
    void const* g_mxdA_, // float B*C*H*Q
                         //  const void *g_mxdt_, // Tp_   B*L*((g_mxZ?2:1)*H*P+2*G+round_up(H,8))
                         //  const void *g_mxdb_, // Wt_       H
                         //  const void *g_mxA_,  // Wt_       H
                         //  const void *g_mxCB_, // Tp_   B*C*G*Q*Q
                         //  const void *g_mxD_,  // Wt_       H
                         //  const void *g_mxX_,  // Tp_   B*L*(H*P+2*G*N)
                         //  const void *g_mxZ_,  // g_mxdt_ or nullptr
    bool removePadding_, int const* lastTokenIdsPtr_, int const* stateSlotMappingPtr_)
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
        g_mxFs_ = (Tp_*) g_mxFs_ + int64_t(stateSlotMappingPtr_[blockIdx.z]) * H_ * N_ * P_;
    }
    else
    {
        g_mxFs_ = (Tp_*) g_mxFs_ + int64_t(blockIdx.z) * H_ * N_ * P_;
    }

    auto hStart = Rn<ID>{blockIdx_x.var * tileH_ / N_ / P_};

    //  const Tp_   *g_mxY  = (const Tp_   *)g_mxY_;
    //        Tp_   *g_mxOs = (      Tp_   *)g_mxOs_;
    Tp_* g_mxFs = (Tp_*) g_mxFs_;
    //  const float *g_mxSt = (const float *)g_mxSt_;
    //  const float *g_mxdc = (const float *)g_mxdc_;
    //  const float *g_mxdA = (const float *)g_mxdA_;
    //  const Tp_   *g_mxdt = (const Tp_   *)g_mxdt_;
    //  const Wt_   *g_mxdb = (const Wt_   *)g_mxdb_;
    //  const Wt_   *g_mxA  = (const Wt_   *)g_mxA_;
    //  const Tp_   *g_mxCB = (const Tp_   *)g_mxCB_;
    //  const Wt_   *g_mxD  = (const Wt_   *)g_mxD_;
    //  const Tp_   *g_mxX  = (const Tp_   *)g_mxX_;
    //  const Tp_   *g_mxZ  = (const Tp_   *)g_mxZ_;

    Tp_ r_mxOs[tileH_ / (warpH_ * 32)] = {0};
    float r_mxSt[tileH_ / (warpH_ * 32)] = {0};

    for (int iC = 0; iC < C.var; iC++)
    {
        Tp_* g_mxOs = (Tp_*) g_mxOs_ + int64_t(get(cStart + Rn<>{iC})) * get(H * N * P);
        float const* g_mxSt = (float const*) g_mxSt_ + int64_t(get(cStart + Rn<>{iC})) * get(H * N * P);
        float const* g_mxdA = (float const*) g_mxdA_ + int64_t(get(cStart + Rn<>{iC})) * get(H * Q);

        if (std::is_same_v<Tp_, half>)
#pragma unroll
            for (int i = 0; i < tileH_ / (warpH_ * 32); i += 2)
                *(half2*) &r_mxOs[i] = __float22half2_rn(*(float2*) &r_mxSt[i]);
        else
#pragma unroll
            for (int i = 0; i < tileH_ / (warpH_ * 32); i += 2)
                *(bf162*) &r_mxOs[i] = __float22bfloat162_rn(*(float2*) &r_mxSt[i]);

        if (tileH_ / (warpH_ * 32) % 8 == 0)
#pragma unroll
            for (int i = 0; i < tileH_ / (warpH_ * 32); i += 8)
                *(int4*) (g_mxOs
                    + get(blockIdx_x * cn<tileH_> + (threadIdx_y * cn<32> + threadIdx_x) * cn<tileH_ / (warpH_ * 32)>
                        + Rn<UNROLL>{i}))
                    = *(int4*) &r_mxOs[i];
        else if (tileH_ / (warpH_ * 32) % 4 == 0)
#pragma unroll
            for (int i = 0; i < tileH_ / (warpH_ * 32); i += 4)
                *(int2*) (g_mxOs
                    + get(blockIdx_x * cn<tileH_> + (threadIdx_y * cn<32> + threadIdx_x) * cn<tileH_ / (warpH_ * 32)>
                        + Rn<UNROLL>{i}))
                    = *(int2*) &r_mxOs[i];
        else
#pragma unroll
            for (int i = 0; i < tileH_ / (warpH_ * 32); i += 2)
                *(int*) (g_mxOs
                    + get(blockIdx_x * cn<tileH_> + (threadIdx_y * cn<32> + threadIdx_x) * cn<tileH_ / (warpH_ * 32)>
                        + Rn<UNROLL>{i}))
                    = *(int*) &r_mxOs[i];

        float scale = expf(g_mxdA[get(hStart * Q + Q - cn<1>)]);

#pragma unroll
        for (int i = 0; i < tileH_ / (warpH_ * 32); i++)
        {
            float tmp = g_mxSt[get(blockIdx_x * cn<tileH_>
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

    if (tileH_ / (warpH_ * 32) % 8 == 0)
#pragma unroll
        for (int i = 0; i < tileH_ / (warpH_ * 32); i += 8)
            *(int4*) (g_mxFs
                + get(blockIdx_x * cn<tileH_> + (threadIdx_y * cn<32> + threadIdx_x) * cn<tileH_ / (warpH_ * 32)>
                    + Rn<UNROLL>{i}))
                = *(int4*) &r_mxOs[i];
    else if (tileH_ / (warpH_ * 32) % 4 == 0)
#pragma unroll
        for (int i = 0; i < tileH_ / (warpH_ * 32); i += 4)
            *(int2*) (g_mxFs
                + get(blockIdx_x * cn<tileH_> + (threadIdx_y * cn<32> + threadIdx_x) * cn<tileH_ / (warpH_ * 32)>
                    + Rn<UNROLL>{i}))
                = *(int2*) &r_mxOs[i];
    else
#pragma unroll
        for (int i = 0; i < tileH_ / (warpH_ * 32); i += 2)
            *(int*) (g_mxFs
                + get(blockIdx_x * cn<tileH_> + (threadIdx_y * cn<32> + threadIdx_x) * cn<tileH_ / (warpH_ * 32)>
                    + Rn<UNROLL>{i}))
                = *(int*) &r_mxOs[i];
#endif
}

typedef StatePassingKernelFunc (*GetStatePassingKernelFunc)(int B_, int L_, int H_, int P_, int G_, int N_, int Q_,
    int numTokens_, dim3* blockDims_, dim3* threadDims_, int* sharedMem_);

template <class Tp_>
StatePassingKernelFunc getStatePassingKernel(int B_, int L_, int H_, int P_, int G_, int N_, int Q_, int numTokens_,
    dim3* blockDims_, dim3* threadDims_, int* sharedMem_)
{
    int B = B_;
    int L = L_;
    int H = H_;
    int P = P_;
    // int G = G_;
    int N = N_;
    int Q = Q_;
    int C = div_up(L, Q);

    // int64_t compute = int64_t(numTokens_) * H * N * P_;

    auto set = [&](int tileH, int warpH, StatePassingKernelFunc func)
    {
        auto sharedMem = 0;

        *blockDims_ = dim3(H * N * P_ / tileH, 1, B);
        *threadDims_ = dim3(32, warpH);
        *sharedMem_ = sharedMem;

        return func;
    };

    if (Q == 256)
    {
        return set(512, 4, state_passing_kernel<256, 512, 4, Tp_>);
    }
    if (Q == 128)
    {
        return set(512, 4, state_passing_kernel<128, 512, 4, Tp_>);
    }

    return nullptr;
}

extern GetStatePassingKernelFunc getStatePassingKernel_fp16;
extern GetStatePassingKernelFunc getStatePassingKernel_bf16;

static inline StatePassingKernelFunc getStatePassingKernel(int B_, int L_, int H_, int P_, int G_, int N_, int Q_,
    int numTokens_, dim3* blockDims_, dim3* threadDims_, int* sharedMem_, CudaType tp_ = CT_FP16)
{
    if (tp_ == CT_FP16)
        return getStatePassingKernel_fp16(B_, L_, H_, P_, G_, N_, Q_, numTokens_, blockDims_, threadDims_, sharedMem_);
    else if (tp_ == CT_BF16)
        return getStatePassingKernel_bf16(B_, L_, H_, P_, G_, N_, Q_, numTokens_, blockDims_, threadDims_, sharedMem_);

    return nullptr;
}

} // namespace kernels
} // namespace tensorrt_llm

// vim: ts=2 sw=2 sts=2 et sta
