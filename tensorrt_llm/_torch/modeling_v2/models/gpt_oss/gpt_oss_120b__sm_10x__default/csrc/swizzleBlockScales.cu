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

#include "gpt_oss/gpt_oss_120b__sm_10x__default/csrc/swizzleBlockScales.h"

#include <algorithm>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{

namespace
{

// One 32-bit word per thread: the four scales (m, c..c+3) of a 4-column group stay together in the swizzled
// layout, so each word moves whole. In words, one 128x4 tile is 128 words.
__global__ void swizzleBlockScalesKernel(
    uint32_t const* __restrict__ in, uint32_t* __restrict__ out, int64_t rows, int64_t words, int64_t total)
{
    for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < total;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x)
    {
        int64_t const word = i % words;
        int64_t const m = (i / words) % rows;
        int64_t const expert = i / (words * rows);
        int64_t const dst
            = expert * rows * words + (m / 128) * 128 * words + word * 128 + (m % 32) * 4 + (m % 128) / 32;
        out[dst] = in[i];
    }
}

} // namespace

cudaError_t invokeSwizzleBlockScales(
    uint8_t const* in, uint8_t* out, int64_t experts, int64_t rows, int64_t cols, cudaStream_t stream)
{
    int64_t const words = cols / 4;
    int64_t const total = experts * rows * words;
    if (total == 0)
    {
        return cudaSuccess;
    }
    constexpr int kThreads = 256;
    int64_t const blocks = std::min<int64_t>((total + kThreads - 1) / kThreads, 65535);
    swizzleBlockScalesKernel<<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
        reinterpret_cast<uint32_t const*>(in), reinterpret_cast<uint32_t*>(out), rows, words, total);
    return cudaGetLastError();
}

} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
