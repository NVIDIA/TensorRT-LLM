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
#include "tensorrt_llm/kernels/minimaxM31/tailBk3Sm107.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM31
{
namespace
{

// One block per row; 16-byte copies when the row allows them (rank % 8 == 0 and aligned rows), else element copies.
__global__ void markovPrevEmbedKernel(
    int64_t const* prev, __nv_bfloat16 const* w1, int64_t vocab, int32_t rank, __nv_bfloat16* out)
{
    int32_t const i = blockIdx.x;
    int64_t const p = prev[i];
    bool const valid = p >= 0 && p < vocab;
    __nv_bfloat16* dst = out + static_cast<int64_t>(i) * rank;
    __nv_bfloat16 const* src = w1 + (valid ? p : 0) * static_cast<int64_t>(rank);
    bool const vec = (rank % 8) == 0 && (reinterpret_cast<uintptr_t>(src) % 16) == 0
        && (reinterpret_cast<uintptr_t>(dst) % 16) == 0;
    if (vec)
    {
        int32_t const nv = rank / 8;
        for (int32_t c = threadIdx.x; c < nv; c += blockDim.x)
        {
            uint4 v = valid ? reinterpret_cast<uint4 const*>(src)[c] : make_uint4(0u, 0u, 0u, 0u);
            reinterpret_cast<uint4*>(dst)[c] = v;
        }
    }
    else
    {
        for (int32_t c = threadIdx.x; c < rank; c += blockDim.x)
        {
            dst[c] = valid ? src[c] : __float2bfloat16_rn(0.0f);
        }
    }
}

__global__ void expandBlockSamplingKernel(float const* temps, int32_t const* topKs, float const* topPs,
    float const* minPs, int64_t const* seeds, int64_t const* offsets, int32_t n, int32_t K, int64_t slotOffset,
    int64_t span, float* tempsOut, int32_t* topKsOut, float* topPsOut, float* minPsOut, int64_t* seedsOut,
    int64_t* offsetsOut)
{
    int32_t const r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n * K)
    {
        return;
    }
    int32_t const i = r / K;
    int32_t const j = r - i * K;
    tempsOut[r] = temps[i];
    topKsOut[r] = topKs[i];
    topPsOut[r] = topPs[i];
    minPsOut[r] = minPs[i];
    seedsOut[r] = seeds[i];
    // torch: (offsets + slot * SPAN).repeat_interleave(K) + arange(K).repeat(n) * SPAN, int64 two's-complement wrap.
    uint64_t const o = static_cast<uint64_t>(offsets[i]) + static_cast<uint64_t>(slotOffset)
        + static_cast<uint64_t>(static_cast<int64_t>(j) * span);
    offsetsOut[r] = static_cast<int64_t>(o);
}

} // namespace

void invokeMarkovPrevEmbed(int64_t const* prev, int32_t n, __nv_bfloat16 const* w1, int64_t vocab, int32_t rank,
    __nv_bfloat16* out, cudaStream_t stream)
{
    if (n <= 0 || rank <= 0)
    {
        return;
    }
    int32_t const threads = rank / 8 >= 128 ? 128 : 64;
    markovPrevEmbedKernel<<<n, threads, 0, stream>>>(prev, w1, vocab, rank, out);
}

void invokeExpandBlockSampling(float const* temps, int32_t const* topKs, float const* topPs, float const* minPs,
    int64_t const* seeds, int64_t const* offsets, int32_t n, int32_t K, int64_t slotOffset, int64_t span,
    float* tempsOut, int32_t* topKsOut, float* topPsOut, float* minPsOut, int64_t* seedsOut, int64_t* offsetsOut,
    cudaStream_t stream)
{
    int32_t const total = n * K;
    if (total <= 0)
    {
        return;
    }
    int32_t const threads = 128;
    expandBlockSamplingKernel<<<(total + threads - 1) / threads, threads, 0, stream>>>(temps, topKs, topPs, minPs,
        seeds, offsets, n, K, slotOffset, span, tempsOut, topKsOut, topPsOut, minPsOut, seedsOut, offsetsOut);
}

} // namespace minimaxM31
} // namespace kernels

TRTLLM_NAMESPACE_END
