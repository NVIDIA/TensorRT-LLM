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

#include "tensorrt_llm/kernels/minimaxM3Sm107Tp2/stepPrologue.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM3Sm107Tp2
{

namespace
{

constexpr int kThreads = 512;
constexpr int kWarps = kThreads / 32;

// One warp per copy. Plain (coherent) loads: a device copy may read bytes another warp of this CTA just stored, which
// the CTA barrier between the two kinds makes visible; a non-coherent load could return the stale line.
__device__ __forceinline__ void copyWarp(uint8_t* dst, uint8_t const* src, int64_t bytes, int lane)
{
    auto const d = reinterpret_cast<uintptr_t>(dst);
    auto const s = reinterpret_cast<uintptr_t>(src);
    if (((d | s | static_cast<uintptr_t>(bytes)) & 15) == 0)
    {
        auto* d16 = reinterpret_cast<uint4*>(dst);
        auto const* s16 = reinterpret_cast<uint4 const*>(src);
        for (int64_t i = lane; i < bytes / 16; i += 32)
        {
            d16[i] = s16[i];
        }
    }
    else if (((d | s | static_cast<uintptr_t>(bytes)) & 3) == 0)
    {
        auto* d4 = reinterpret_cast<uint32_t*>(dst);
        auto const* s4 = reinterpret_cast<uint32_t const*>(src);
        for (int64_t i = lane; i < bytes / 4; i += 32)
        {
            d4[i] = s4[i];
        }
    }
    else
    {
        for (int64_t i = lane; i < bytes; i += 32)
        {
            dst[i] = src[i];
        }
    }
}

// One CTA: a step's window is a few dozen copies of a few hundred bytes, so the launch is the cost; a CTA barrier
// orders the host copies before the device copies where the parallel scatter needed a second launch.
__global__ void __launch_bounds__(kThreads) stepPrologueKernel(uint8_t* window, int64_t* epoch, int64_t maxDesc)
{
    auto* header = reinterpret_cast<int64_t*>(window);
    int64_t const numHost = min(header[0], maxDesc);
    int64_t const numDevice = min(header[1], maxDesc);
    int64_t const* desc = header + 2;
    int const warp = threadIdx.x / 32;
    int const lane = threadIdx.x % 32;
    if (epoch != nullptr && threadIdx.x == 0)
    {
        *epoch += 1;
    }
    for (int64_t i = warp; i < numHost; i += kWarps)
    {
        int64_t const* c = desc + 3 * i;
        copyWarp(reinterpret_cast<uint8_t*>(c[0]), window + c[1], c[2], lane);
    }
    __syncthreads();
    for (int64_t i = warp; i < numDevice; i += kWarps)
    {
        // The device copies' slots follow all header[0] host copies, as the parallel scatter indexes them.
        int64_t const* c = desc + 3 * (header[0] + i);
        copyWarp(reinterpret_cast<uint8_t*>(c[0]), reinterpret_cast<uint8_t const*>(c[1]), c[2], lane);
    }
    // Every thread has read the header and the descriptors: consume the window.
    __syncthreads();
    if (threadIdx.x < 2)
    {
        header[threadIdx.x] = 0;
    }
}

} // namespace

cudaError_t invokeStepPrologue(uint8_t* window, int64_t* epoch, int64_t maxDesc, cudaStream_t stream)
{
    stepPrologueKernel<<<1, kThreads, 0, stream>>>(window, epoch, maxDesc);
    return cudaGetLastError();
}

} // namespace minimaxM3Sm107Tp2
} // namespace kernels

TRTLLM_NAMESPACE_END
