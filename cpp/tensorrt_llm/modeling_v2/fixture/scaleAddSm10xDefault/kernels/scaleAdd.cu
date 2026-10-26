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

#include "tensorrt_llm/modeling_v2/fixture/scaleAddSm10xDefault/kernels/scaleAdd.h"

#include <algorithm>
#include <cuda_bf16.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{

namespace
{

constexpr int kThreads = 256;
constexpr int64_t kMaxBlocks = 65535;

__device__ __forceinline__ float toFloat(float value)
{
    return value;
}

__device__ __forceinline__ float toFloat(__nv_bfloat16 value)
{
    return __bfloat162float(value);
}

template <typename T>
__device__ __forceinline__ T fromFloat(float value);

template <>
__device__ __forceinline__ float fromFloat<float>(float value)
{
    return value;
}

template <>
__device__ __forceinline__ __nv_bfloat16 fromFloat<__nv_bfloat16>(float value)
{
    return __float2bfloat16_rn(value);
}

template <typename T>
__global__ void scaleAddKernel(T const* x, T* out, int64_t numel, float alpha, float beta)
{
    int64_t const stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < numel; i += stride)
    {
        out[i] = fromFloat<T>(__fadd_rn(__fmul_rn(toFloat(x[i]), alpha), beta));
    }
}

template <typename T>
cudaError_t launchScaleAdd(void const* x, void* out, int64_t numel, float alpha, float beta, cudaStream_t stream)
{
    int64_t const blocks = std::min<int64_t>((numel + kThreads - 1) / kThreads, kMaxBlocks);
    scaleAddKernel<T><<<static_cast<unsigned int>(blocks), kThreads, 0, stream>>>(
        static_cast<T const*>(x), static_cast<T*>(out), numel, alpha, beta);
    return cudaGetLastError();
}

} // namespace

cudaError_t invokeScaleAdd(
    void const* x, void* out, int64_t numel, ScaleAddDtype dtype, float alpha, float beta, cudaStream_t stream)
{
    if (numel == 0)
    {
        return cudaSuccess;
    }
    switch (dtype)
    {
    case ScaleAddDtype::kFloat32: return launchScaleAdd<float>(x, out, numel, alpha, beta, stream);
    case ScaleAddDtype::kBfloat16: return launchScaleAdd<__nv_bfloat16>(x, out, numel, alpha, beta, stream);
    }
    return cudaErrorInvalidValue;
}

} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
