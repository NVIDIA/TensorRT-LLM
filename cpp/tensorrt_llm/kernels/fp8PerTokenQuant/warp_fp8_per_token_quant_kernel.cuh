/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/config.h"
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace detail
{
constexpr int kFp8QuantWarpSize = 32;
constexpr int kFp8QuantRowsPerBlock = 8;
constexpr int kFp8QuantBlockSize = kFp8QuantWarpSize * kFp8QuantRowsPerBlock;

template <typename T>
struct alignas(16) Fp8QuantInputVector
{
    T values[16 / sizeof(T)];
};

template <int kElements>
struct alignas(kElements) Fp8QuantOutputVector
{
    uint16_t pairs[kElements / 2];
};

template <typename T, bool kUsePdl>
__global__ __launch_bounds__(kFp8QuantBlockSize, 8) void warpPerTokenFp8QuantKernel(uint8_t* __restrict__ output,
    float* __restrict__ scales, T const* __restrict__ input, int hiddenSize, int64_t numTokens, float minScale)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if constexpr (kUsePdl)
    {
        cudaGridDependencySynchronize();
        cudaTriggerProgrammaticLaunchCompletion();
    }
#endif
    constexpr int kVectorElements = 16 / sizeof(T);
    int const lane = threadIdx.x % kFp8QuantWarpSize;
    int64_t const row = static_cast<int64_t>(blockIdx.x) * kFp8QuantRowsPerBlock + threadIdx.x / kFp8QuantWarpSize;
    if (row >= numTokens)
    {
        return;
    }
    auto const* rowInput = reinterpret_cast<Fp8QuantInputVector<T> const*>(input + row * hiddenSize);
    auto* rowOutput = reinterpret_cast<Fp8QuantOutputVector<kVectorElements>*>(output + row * hiddenSize);
    int const vectors = hiddenSize / kVectorElements;
    float maxAbs = 0.0f;
    for (int col = lane; col < vectors; col += kFp8QuantWarpSize)
    {
        auto const values = rowInput[col];
#pragma unroll
        for (int i = 0; i < kVectorElements; ++i)
        {
            maxAbs = fmaxf(maxAbs, fabsf(static_cast<float>(values.values[i])));
        }
    }
#pragma unroll
    for (int offset = kFp8QuantWarpSize / 2; offset > 0; offset /= 2)
    {
        maxAbs = fmaxf(maxAbs, __shfl_down_sync(0xffffffff, maxAbs, offset));
    }
    maxAbs = __shfl_sync(0xffffffff, maxAbs, 0);
    constexpr float kFp8Max = 448.0f;
    float const scale = fmaxf(__fdiv_rn(maxAbs, kFp8Max), minScale);
    if (lane == 0)
    {
        scales[row] = scale;
    }
    for (int col = lane; col < vectors; col += kFp8QuantWarpSize)
    {
        auto const values = rowInput[col];
        Fp8QuantOutputVector<kVectorElements> packed;
#pragma unroll
        for (int i = 0; i < kVectorElements; i += 2)
        {
            float const x = fmaxf(-kFp8Max, fminf(__fdiv_rn(static_cast<float>(values.values[i]), scale), kFp8Max));
            float const y = fmaxf(-kFp8Max, fminf(__fdiv_rn(static_cast<float>(values.values[i + 1]), scale), kFp8Max));
            packed.pairs[i / 2] = __nv_cvt_float2_to_fp8x2(make_float2(x, y), __NV_SATFINITE, __NV_E4M3);
        }
        rowOutput[col] = packed;
    }
}
} // namespace detail
} // namespace kernels

TRTLLM_NAMESPACE_END
