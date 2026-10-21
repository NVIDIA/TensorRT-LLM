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

#include "tensorrt_llm/kernels/minimaxM3/stepInputsScatter.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{

constexpr int kThreads = 256;

__device__ __forceinline__ void copyBytes(uint8_t* dst, uint8_t const* src, int64_t bytes)
{
    for (int64_t b = threadIdx.x; b < bytes; b += blockDim.x)
    {
        dst[b] = src[b];
    }
}

// One CTA: the batches this serves are a step's few hundred bytes of metadata, where a single launch is the point and
// the copy itself is negligible.
__global__ void __launch_bounds__(kThreads) stepInputsScatterKernel(uint8_t const* __restrict__ stage)
{
    auto const* header = reinterpret_cast<int64_t const*>(stage);
    int64_t const numHostCopies = header[0];
    int64_t const numDeviceCopies = header[1];
    int64_t const* desc = header + 2;
    // Copies complete in descriptor order: one may overwrite or read another's destination.
    for (int64_t i = 0; i < numHostCopies; ++i)
    {
        copyBytes(reinterpret_cast<uint8_t*>(desc[3 * i]), stage + desc[3 * i + 1], desc[3 * i + 2]);
        __syncthreads();
    }
    for (int64_t i = numHostCopies; i < numHostCopies + numDeviceCopies; ++i)
    {
        copyBytes(reinterpret_cast<uint8_t*>(desc[3 * i]), reinterpret_cast<uint8_t const*>(desc[3 * i + 1]),
            desc[3 * i + 2]);
        __syncthreads();
    }
}

} // namespace

void invokeStepInputsScatter(uint8_t const* stage, cudaStream_t stream)
{
    stepInputsScatterKernel<<<1, kThreads, 0, stream>>>(stage);
}

} // namespace kernels

TRTLLM_NAMESPACE_END
