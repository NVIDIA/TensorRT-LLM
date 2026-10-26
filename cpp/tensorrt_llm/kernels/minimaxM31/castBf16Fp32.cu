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

#include "tensorrt_llm/kernels/minimaxM31/castBf16Fp32.h"

#include <algorithm>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM31
{

namespace
{

constexpr int kThreads = 256;
constexpr int kVec = 8; // bf16 per 16-byte load

__device__ __forceinline__ uint32_t widen(uint32_t bf16Bits)
{
    return bf16Bits << 16;
}

// Eight bf16 per thread per iteration: one 16-byte load, two 16-byte stores; grid-stride over the vectors.
__global__ void __launch_bounds__(kThreads) castBf16ToFp32Vec(uint4 const* in, uint4* out, int64_t nVec)
{
    for (int64_t v = int64_t(blockIdx.x) * kThreads + threadIdx.x; v < nVec; v += int64_t(gridDim.x) * kThreads)
    {
        uint4 const x = in[v];
        uint4 lo, hi;
        lo.x = widen(x.x & 0xFFFFu);
        lo.y = widen(x.x >> 16);
        lo.z = widen(x.y & 0xFFFFu);
        lo.w = widen(x.y >> 16);
        hi.x = widen(x.z & 0xFFFFu);
        hi.y = widen(x.z >> 16);
        hi.z = widen(x.w & 0xFFFFu);
        hi.w = widen(x.w >> 16);
        out[2 * v] = lo;
        out[2 * v + 1] = hi;
    }
}

__global__ void __launch_bounds__(kThreads)
    castBf16ToFp32Scalar(uint16_t const* in, uint32_t* out, int64_t begin, int64_t n)
{
    for (int64_t i = begin + int64_t(blockIdx.x) * kThreads + threadIdx.x; i < n; i += int64_t(gridDim.x) * kThreads)
    {
        out[i] = widen(in[i]);
    }
}

} // namespace

void launchCastBf16ToFp32(uint16_t const* in, float* out, int64_t n, cudaStream_t stream)
{
    if (n <= 0)
    {
        return;
    }
    int device = 0;
    cudaGetDevice(&device);
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    int64_t const maxBlocks = int64_t(sms) * 8;
    bool const aligned = (reinterpret_cast<uintptr_t>(in) % 16 == 0) && (reinterpret_cast<uintptr_t>(out) % 16 == 0);
    int64_t done = 0;
    if (aligned)
    {
        int64_t const nVec = n / kVec;
        if (nVec > 0)
        {
            int64_t const blocks = std::min<int64_t>((nVec + kThreads - 1) / kThreads, maxBlocks);
            castBf16ToFp32Vec<<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
                reinterpret_cast<uint4 const*>(in), reinterpret_cast<uint4*>(out), nVec);
        }
        done = nVec * kVec;
    }
    if (done < n)
    {
        int64_t const rest = n - done;
        int64_t const blocks = std::min<int64_t>((rest + kThreads - 1) / kThreads, maxBlocks);
        castBf16ToFp32Scalar<<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
            in, reinterpret_cast<uint32_t*>(out), done, n);
    }
}

} // namespace minimaxM31
} // namespace kernels

TRTLLM_NAMESPACE_END
