/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

#include <hip/hip_runtime.h>

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1200__) && !defined(__gfx1201__)
#error "These kernels require the real RDNA4 gfx1200 or gfx1201 target."
#endif
#if defined(__AMDGCN_WAVEFRONT_SIZE) && __AMDGCN_WAVEFRONT_SIZE != 32
#error "RDNA4 reductions must be compiled with -mwavefrontsize32."
#endif

namespace tensorrt_llm::rdna4
{
constexpr int kWaveSize = 32;
constexpr int kBlockSize = 256;
constexpr int kWavesPerBlock = kBlockSize / kWaveSize;

//! Sum a full wave32. All 32 lanes must participate, with zero for inactive data elements.
__device__ __forceinline__ float waveSum(float value)
{
    for (int offset = kWaveSize / 2; offset > 0; offset /= 2)
    {
        value += __shfl_down(value, offset, kWaveSize);
    }
    return value;
}

//! Reduce a complete 256-thread work-group, returning the result to every thread.
//! The final barrier prevents scratch reuse until every wave has read the result.
__device__ __forceinline__ float blockSum(float value, float* scratch)
{
    int const lane = static_cast<int>(threadIdx.x) % kWaveSize;
    int const wave = static_cast<int>(threadIdx.x) / kWaveSize;
    value = waveSum(value);
    if (lane == 0)
    {
        scratch[wave] = value;
    }
    __syncthreads();
    if (wave == 0)
    {
        value = lane < kWavesPerBlock ? scratch[lane] : 0.0F;
        value = waveSum(value);
        if (lane == 0)
        {
            scratch[0] = value;
        }
    }
    __syncthreads();
    float const result = scratch[0];
    __syncthreads();
    return result;
}
} // namespace tensorrt_llm::rdna4
