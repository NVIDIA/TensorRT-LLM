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

#include "tensorrt_llm/kernels/minimaxM31/tp2ctxPeer.h"

#include <algorithm>
#include <cuda_bf16.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM31
{
namespace
{

constexpr int kThreads = 512;

__global__ void tp2ctxPeerAddKernel(
    uint4 const* __restrict__ a, uint4 const* __restrict__ b, uint4* __restrict__ out, int64_t n)
{
    for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x)
    {
        uint4 const x = a[i];
        uint4 const y = b[i];
        uint4 r;
        auto const* xa = reinterpret_cast<__nv_bfloat162 const*>(&x);
        auto const* ya = reinterpret_cast<__nv_bfloat162 const*>(&y);
        auto* ra = reinterpret_cast<__nv_bfloat162*>(&r);
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            float2 const fx = __bfloat1622float2(xa[j]);
            float2 const fy = __bfloat1622float2(ya[j]);
            ra[j] = __floats2bfloat162_rn(fx.x + fy.x, fx.y + fy.y);
        }
        out[i] = r;
    }
}

__global__ void tp2ctxPeerPushKernel(uint4 const* __restrict__ src, uint4* __restrict__ dst, int64_t n)
{
    for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x)
    {
        dst[i] = src[i];
    }
}

__global__ void tp2ctxIpcBarrierKernel(int* localFlags, int* peerFlags, int* state, int rank)
{
    if (threadIdx.x != 0)
    {
        return;
    }
    int const epoch = state[0] + 1;
    state[0] = epoch;
    __threadfence_system();
    asm volatile("st.release.sys.global.b32 [%0], %1;" ::"l"(peerFlags + rank), "r"(epoch) : "memory");
    int seen = 0;
    do
    {
        asm volatile("ld.acquire.sys.global.b32 %0, [%1];" : "=r"(seen) : "l"(localFlags + (1 - rank)) : "memory");
    } while (seen < epoch);
}

int gridFor(int64_t nVec, int smCount)
{
    // 2 x #SMs saturates NVLink at every measured size (GEMM-AR MB 836817 / 836907); fewer CTAs for small calls.
    int64_t const want = (nVec + kThreads - 1) / kThreads;
    return static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(want, 2 * static_cast<int64_t>(smCount))));
}

} // namespace

void launchTp2ctxPeerAdd(void const* a, void const* b, void* out, int64_t n, int smCount, cudaStream_t stream)
{
    int64_t const nVec = n / 8;
    if (nVec == 0)
    {
        return;
    }
    tp2ctxPeerAddKernel<<<gridFor(nVec, smCount), kThreads, 0, stream>>>(
        static_cast<uint4 const*>(a), static_cast<uint4 const*>(b), static_cast<uint4*>(out), nVec);
}

void launchTp2ctxPeerPush(void const* src, void* dst, int64_t nBytes, int smCount, cudaStream_t stream)
{
    int64_t const nVec = nBytes / 16;
    if (nVec == 0)
    {
        return;
    }
    tp2ctxPeerPushKernel<<<gridFor(nVec, smCount), kThreads, 0, stream>>>(
        static_cast<uint4 const*>(src), static_cast<uint4*>(dst), nVec);
}

void launchTp2ctxIpcBarrier(int* localFlags, int* peerFlags, int* state, int rank, cudaStream_t stream)
{
    tp2ctxIpcBarrierKernel<<<1, 32, 0, stream>>>(localFlags, peerFlags, state, rank);
}

} // namespace minimaxM31
} // namespace kernels

TRTLLM_NAMESPACE_END
