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

#include "minimax_m3/csrc/m31/fillPaddingRowsSm107.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM31
{
namespace
{

// Compact kernel: 128 threads x 2 16-byte vectors per thread (beat Triton on every DEP8 shape).
constexpr int kCompactThreads = 128;
constexpr int kCompactVecs = 2;
constexpr int kMaxCompactRows = 2048;
// Per-row grid fallback: 256 threads x 8 iterations of one 16-byte vector.
constexpr int kGridThreads = 256;
constexpr int kGridIters = 8;

__device__ __forceinline__ void pdlWait(bool pdl)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    if (pdl)
    {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }
#endif
}

__device__ __forceinline__ void pdlTrigger(bool pdl)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    if (pdl)
    {
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
#endif
}

template <typename T>
__device__ __forceinline__ T fromFloat(float v);

template <>
__device__ __forceinline__ float fromFloat<float>(float v)
{
    return v;
}

template <>
__device__ __forceinline__ __nv_bfloat16 fromFloat<__nv_bfloat16>(float v)
{
    return __float2bfloat16_rn(v);
}

template <>
__device__ __forceinline__ __half fromFloat<__half>(float v)
{
    return __float2half_rn(v);
}

//! One CTA covers kGridThreads * kElems * kGridIters columns of one row, kElems = 16 bytes / sizeof(T).
template <typename T>
__global__ void __launch_bounds__(kGridThreads) fillPaddingRowsKernel(T* __restrict__ rows,
    bool const* __restrict__ isPadding, int64_t cols, float value, float col0, bool hasCol0, bool vecPath, bool pdl)
{
    constexpr int kElems = 16 / sizeof(T);
    pdlWait(pdl);
    int64_t const row = blockIdx.y;
    if (isPadding[row])
    {
        T const v = fromFloat<T>(value);
        T const v0 = fromFloat<T>(col0);
        T buf[kElems];
#pragma unroll
        for (int i = 0; i < kElems; ++i)
        {
            buf[i] = v;
        }
        uint4 const word = *reinterpret_cast<uint4 const*>(buf);
        T* dst = rows + row * cols;
        int64_t const cta0 = static_cast<int64_t>(blockIdx.x) * (kGridThreads * kElems * kGridIters);
#pragma unroll
        for (int it = 0; it < kGridIters; ++it)
        {
            int64_t const c0 = cta0 + (static_cast<int64_t>(it) * kGridThreads + threadIdx.x) * kElems;
            if (vecPath && c0 + kElems <= cols && c0 != 0)
            {
                *reinterpret_cast<uint4*>(dst + c0) = word;
            }
            else
            {
                for (int64_t c = c0; c < c0 + kElems && c < cols; ++c)
                {
                    dst[c] = (hasCol0 && c == 0) ? v0 : v;
                }
            }
        }
    }
    pdlTrigger(pdl);
}

//! Each CTA: the padded-row indices from the mask (block ballot + prefix, in mask order) into shared memory, then its
//! column slice [blockIdx.x * span, +span) of vectors in every padded row, span = kCompactThreads * kCompactVecs.
template <typename T>
__global__ void __launch_bounds__(kCompactThreads) fillPaddingRowsCompactKernel(T* __restrict__ rows,
    bool const* __restrict__ isPadding, int nRows, int64_t vecsPerRow, float value, float col0, bool hasCol0, bool pdl)
{
    constexpr int kElems = 16 / sizeof(T);
    constexpr int kWarps = kCompactThreads / 32;
    __shared__ int16_t padRows[kMaxCompactRows];
    __shared__ int warpCount[kWarps];
    pdlWait(pdl);
    int const warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int total = 0;
    for (int base = 0; base < nRows; base += kCompactThreads)
    {
        int const r = base + threadIdx.x;
        bool const p = r < nRows && isPadding[r];
        unsigned const m = __ballot_sync(0xffffffffu, p);
        if (lane == 0)
        {
            warpCount[warp] = __popc(m);
        }
        __syncthreads();
        int before = 0, sum = 0;
#pragma unroll
        for (int w = 0; w < kWarps; ++w)
        {
            int const c = warpCount[w];
            before += w < warp ? c : 0;
            sum += c;
        }
        if (p)
        {
            padRows[total + before + __popc(m & ((1u << lane) - 1u))] = static_cast<int16_t>(r);
        }
        total += sum;
        __syncthreads();
    }
    if (total > 0)
    {
        T const v = fromFloat<T>(value);
        T buf[kElems];
#pragma unroll
        for (int i = 0; i < kElems; ++i)
        {
            buf[i] = v;
        }
        uint4 const word = *reinterpret_cast<uint4 const*>(buf);
        buf[0] = fromFloat<T>(col0);
        uint4 const word0 = hasCol0 ? *reinterpret_cast<uint4 const*>(buf) : word;
        int64_t const cols = vecsPerRow * kElems;
        int64_t const v0 = static_cast<int64_t>(blockIdx.x) * (kCompactThreads * kCompactVecs) + threadIdx.x;
        for (int i = 0; i < total; ++i)
        {
            T* dst = rows + static_cast<int64_t>(padRows[i]) * cols;
#pragma unroll
            for (int k = 0; k < kCompactVecs; ++k)
            {
                int64_t const vi = v0 + static_cast<int64_t>(k) * kCompactThreads;
                if (vi < vecsPerRow)
                {
                    *reinterpret_cast<uint4*>(dst + vi * kElems) = vi == 0 ? word0 : word;
                }
            }
        }
    }
    pdlTrigger(pdl);
}

template <typename T>
cudaError_t launch(T* rows, bool const* isPadding, int64_t nRows, int64_t cols, float value, float col0, bool hasCol0,
    bool pdl, cudaStream_t stream)
{
    constexpr int kElems = 16 / sizeof(T);
    bool const vecPath
        = (reinterpret_cast<uintptr_t>(rows) % 16 == 0) && ((cols * static_cast<int64_t>(sizeof(T))) % 16 == 0);
    cudaLaunchConfig_t cfg{};
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    if (vecPath && nRows <= kMaxCompactRows)
    {
        int64_t const vecsPerRow = cols / kElems;
        constexpr int64_t kSpan = static_cast<int64_t>(kCompactThreads) * kCompactVecs;
        cfg.gridDim = dim3(static_cast<unsigned>((vecsPerRow + kSpan - 1) / kSpan));
        cfg.blockDim = dim3(kCompactThreads);
        return cudaLaunchKernelEx(&cfg, fillPaddingRowsCompactKernel<T>, rows, isPadding, static_cast<int>(nRows),
            vecsPerRow, value, col0, hasCol0, pdl);
    }
    if (nRows > 65535)
    {
        return cudaErrorInvalidValue;
    }
    constexpr int64_t kCols = static_cast<int64_t>(kGridThreads) * kElems * kGridIters;
    cfg.gridDim = dim3(static_cast<unsigned>((cols + kCols - 1) / kCols), static_cast<unsigned>(nRows));
    cfg.blockDim = dim3(kGridThreads);
    return cudaLaunchKernelEx(
        &cfg, fillPaddingRowsKernel<T>, rows, isPadding, cols, value, col0, hasCol0, vecPath, pdl);
}

} // namespace

cudaError_t invokeFillPaddingRows(void* rows, int elemSize, bool isBf16, bool const* isPadding, int64_t nRows,
    int64_t cols, float value, float col0, bool hasCol0, bool pdl, cudaStream_t stream)
{
    if (nRows <= 0 || cols <= 0)
    {
        return cudaSuccess;
    }
    if (elemSize == 4)
    {
        return launch(static_cast<float*>(rows), isPadding, nRows, cols, value, col0, hasCol0, pdl, stream);
    }
    if (elemSize != 2)
    {
        return cudaErrorInvalidValue;
    }
    if (isBf16)
    {
        return launch(static_cast<__nv_bfloat16*>(rows), isPadding, nRows, cols, value, col0, hasCol0, pdl, stream);
    }
    return launch(static_cast<__half*>(rows), isPadding, nRows, cols, value, col0, hasCol0, pdl, stream);
}

} // namespace minimaxM31
} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
