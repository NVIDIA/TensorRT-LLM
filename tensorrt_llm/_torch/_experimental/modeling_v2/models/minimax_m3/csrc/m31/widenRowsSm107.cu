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

#include "minimax_m3/csrc/m31/widenRowsSm107.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include <cuda_fp16.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM31
{
namespace
{

// 256 threads x 8 columns x 2 iterations = 4096 columns per CTA (8.79 vs 9.73 us for Triton at 84 rows,
// 9.96 vs 11.99 us at 96 rows of the 200064-wide vocabulary).
constexpr int kThreads = 256;
constexpr int kIters = 2;
constexpr int kVec = 8; // 8 x 16-bit = one 16-byte load
constexpr int64_t kCols = static_cast<int64_t>(kThreads) * kVec * kIters;

__device__ __forceinline__ void storeStreaming16(float* p, uint4 v)
{
    asm volatile("st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
                 : "memory");
}

__device__ __forceinline__ float widenOne(uint16_t bits, bool isBf16)
{
    if (isBf16)
    {
        return __uint_as_float(static_cast<uint32_t>(bits) << 16);
    }
    return __half2float(__ushort_as_half(bits));
}

template <bool kVecPath>
__global__ void __launch_bounds__(kThreads) widenRowsKernel(
    uint16_t const* __restrict__ in, int64_t inRowStride, float* __restrict__ out, int64_t cols, bool isBf16, bool pdl)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    if (pdl)
    {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }
#endif
    int64_t const row = blockIdx.y;
    uint16_t const* src = in + row * inRowStride;
    float* dst = out + row * cols;
    int64_t const cta0 = static_cast<int64_t>(blockIdx.x) * kCols;
#pragma unroll
    for (int it = 0; it < kIters; ++it)
    {
        int64_t const c0 = cta0 + (static_cast<int64_t>(it) * kThreads + threadIdx.x) * kVec;
        if constexpr (kVecPath)
        {
            if (c0 + kVec <= cols)
            {
                uint4 const v = *reinterpret_cast<uint4 const*>(src + c0);
                uint16_t const* h = reinterpret_cast<uint16_t const*>(&v);
                float f[kVec];
#pragma unroll
                for (int i = 0; i < kVec; ++i)
                {
                    f[i] = widenOne(h[i], isBf16);
                }
                storeStreaming16(dst + c0, *reinterpret_cast<uint4 const*>(f));
                storeStreaming16(dst + c0 + 4, *reinterpret_cast<uint4 const*>(f + 4));
            }
        }
        else
        {
            for (int64_t c = c0; c < c0 + kVec && c < cols; ++c)
            {
                dst[c] = widenOne(src[c], isBf16);
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

} // namespace

cudaError_t invokeWidenRows(uint16_t const* in, int64_t inRowStride, float* out, int64_t rows, int64_t cols,
    bool isBf16, bool pdl, cudaStream_t stream)
{
    if (rows <= 0 || cols <= 0)
    {
        return cudaSuccess;
    }
    if (rows > 65535)
    {
        return cudaErrorInvalidValue;
    }
    bool const vec = (reinterpret_cast<uintptr_t>(in) % 16 == 0) && (inRowStride % kVec == 0)
        && (reinterpret_cast<uintptr_t>(out) % 16 == 0) && (cols % kVec == 0);
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(static_cast<unsigned>((cols + kCols - 1) / kCols), static_cast<unsigned>(rows));
    cfg.blockDim = dim3(kThreads);
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    return vec ? cudaLaunchKernelEx(&cfg, widenRowsKernel<true>, in, inRowStride, out, cols, isBf16, pdl)
               : cudaLaunchKernelEx(&cfg, widenRowsKernel<false>, in, inRowStride, out, cols, isBf16, pdl);
}

} // namespace minimaxM31
} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
