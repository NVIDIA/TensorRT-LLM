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

#include "minimax_m3/csrc/m31/widenAddRowsSm107.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include <cuda_bf16.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM31
{
namespace
{

// 128 threads x 4 columns x 8 iterations = 4096 columns per CTA (markov_glue: 1.80 vs 2.44 us for Triton at
// 12 rows, 2.06 vs 2.90 us at 16 rows of the 200064-wide vocabulary).
constexpr int kThreads = 128;
constexpr int kIters = 8;
constexpr int64_t kCols = static_cast<int64_t>(kThreads) * 4 * kIters;

__device__ __forceinline__ float toFloat(float v)
{
    return v;
}

__device__ __forceinline__ float toFloat(__nv_bfloat16 v)
{
    return __uint_as_float(static_cast<uint32_t>(__bfloat16_as_ushort(v)) << 16);
}

template <typename BiasT>
__device__ __forceinline__ float4 load4Bias(BiasT const* p);

template <>
__device__ __forceinline__ float4 load4Bias<float>(float const* p)
{
    return *reinterpret_cast<float4 const*>(p);
}

template <>
__device__ __forceinline__ float4 load4Bias<__nv_bfloat16>(__nv_bfloat16 const* p)
{
    uint2 const w = *reinterpret_cast<uint2 const*>(p);
    float4 f;
    f.x = __uint_as_float(w.x << 16);
    f.y = __uint_as_float(w.x & 0xFFFF0000u);
    f.z = __uint_as_float(w.y << 16);
    f.w = __uint_as_float(w.y & 0xFFFF0000u);
    return f;
}

template <typename BiasT, bool kVecPath>
__global__ void __launch_bounds__(kThreads) widenAddRowsKernel(float const* __restrict__ base, int64_t baseRowStride,
    BiasT const* __restrict__ bias, float* __restrict__ out, int64_t cols, bool pdl)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    if (pdl)
    {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }
#endif
    int64_t const row = blockIdx.y;
    float const* b = base + row * baseRowStride;
    BiasT const* s = bias + row * cols;
    float* o = out + row * cols;
    int64_t const cta0 = static_cast<int64_t>(blockIdx.x) * kCols;
#pragma unroll
    for (int it = 0; it < kIters; ++it)
    {
        int64_t const c0 = cta0 + (static_cast<int64_t>(it) * kThreads + threadIdx.x) * 4;
        if constexpr (kVecPath)
        {
            if (c0 + 4 <= cols)
            {
                float4 const x = *reinterpret_cast<float4 const*>(b + c0);
                float4 const y = load4Bias<BiasT>(s + c0);
                *reinterpret_cast<float4*>(o + c0) = make_float4(x.x + y.x, x.y + y.y, x.z + y.z, x.w + y.w);
            }
        }
        else
        {
            for (int64_t c = c0; c < c0 + 4 && c < cols; ++c)
            {
                o[c] = b[c] + toFloat(s[c]);
            }
        }
    }
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    if (pdl)
    {
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
#endif
}

template <typename BiasT>
cudaError_t launch(float const* base, int64_t baseRowStride, BiasT const* bias, float* out, int64_t rows, int64_t cols,
    bool pdl, cudaStream_t stream)
{
    bool const vec = (reinterpret_cast<uintptr_t>(base) % 16 == 0) && (baseRowStride % 4 == 0)
        && (reinterpret_cast<uintptr_t>(bias) % (4 * sizeof(BiasT)) == 0)
        && (reinterpret_cast<uintptr_t>(out) % 16 == 0) && (cols % 4 == 0);
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(static_cast<unsigned>((cols + kCols - 1) / kCols), static_cast<unsigned>(rows));
    cfg.blockDim = dim3(kThreads);
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    return vec ? cudaLaunchKernelEx(&cfg, widenAddRowsKernel<BiasT, true>, base, baseRowStride, bias, out, cols, pdl)
               : cudaLaunchKernelEx(&cfg, widenAddRowsKernel<BiasT, false>, base, baseRowStride, bias, out, cols, pdl);
}

} // namespace

cudaError_t invokeWidenAddRows(float const* base, int64_t baseRowStride, void const* bias, bool biasIsBf16, float* out,
    int64_t rows, int64_t cols, bool pdl, cudaStream_t stream)
{
    if (rows <= 0 || cols <= 0)
    {
        return cudaSuccess;
    }
    if (rows > 65535)
    {
        return cudaErrorInvalidValue;
    }
    if (biasIsBf16)
    {
        return launch(base, baseRowStride, static_cast<__nv_bfloat16 const*>(bias), out, rows, cols, pdl, stream);
    }
    return launch(base, baseRowStride, static_cast<float const*>(bias), out, rows, cols, pdl, stream);
}

} // namespace minimaxM31
} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
