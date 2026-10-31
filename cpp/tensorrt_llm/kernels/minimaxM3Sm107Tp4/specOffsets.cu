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

#include "tensorrt_llm/kernels/minimaxM3Sm107Tp4/specOffsets.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM3Sm107Tp4
{

namespace
{

constexpr int kThreads = 256;

__global__ void __launch_bounds__(kThreads)
    specOffsetsKernel(int32_t* __restrict__ pos, int32_t const* __restrict__ posOff, int64_t nPos,
        int32_t* __restrict__ kv, int32_t const* __restrict__ kvOff, int64_t nKv)
{
    for (int64_t i = threadIdx.x; i < nPos; i += kThreads)
    {
        pos[i] = static_cast<int32_t>(static_cast<uint32_t>(pos[i]) + static_cast<uint32_t>(posOff[i]));
    }
    for (int64_t j = threadIdx.x; j < nKv; j += kThreads)
    {
        kv[j] = static_cast<int32_t>(static_cast<uint32_t>(kv[j]) + static_cast<uint32_t>(kvOff[j]));
    }
}

} // namespace

cudaError_t invokeSpecOffsets(int32_t* pos, int32_t const* posOff, int64_t nPos, int32_t* kv, int32_t const* kvOff,
    int64_t nKv, cudaStream_t stream)
{
    specOffsetsKernel<<<1, kThreads, 0, stream>>>(pos, posOff, nPos, kv, kvOff, nKv);
    return cudaGetLastError();
}

} // namespace minimaxM3Sm107Tp4
} // namespace kernels

TRTLLM_NAMESPACE_END
