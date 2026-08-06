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

#include "tensorrt_llm/kernels/deepseekV4Indices.h"

#include <algorithm>
#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{
constexpr int32_t kInvalidIndex = -1;
constexpr int32_t kThreadsPerBlock = 256;

// One block per token row; threads stride along the window / compressed columns.
// Replaces the @maybe_compile(max-autotune) python graph which materialised
// [num_tokens, window_size] and [num_tokens, max_compressed_indices] temporaries
// plus one [num_tokens] tensor per compression ratio.
__global__ void computeDeepseekV4IndicesKernel(int32_t const* __restrict__ tokenPositions,
    int32_t* __restrict__ swaLocalIndices, int32_t* __restrict__ compressedLocalIndices,
    int32_t* __restrict__ topkLensRatio1, int32_t* __restrict__ topkLensRatio4, int32_t* __restrict__ topkLensRatio128,
    int32_t numTokens, int32_t windowSize, int32_t maxCompressedIndices, int32_t sparseMlaTopk, int32_t swaStride,
    int32_t compressedStride)
{
    int32_t const tokenId = static_cast<int32_t>(blockIdx.x);
    if (tokenId >= numTokens)
    {
        return;
    }

    int32_t const position = tokenPositions[tokenId];
    int32_t const kvLen = position + 1;

    // ── SWA local indices: start = clamp(position - window + 1, min=0); entries
    //    beyond `position` are invalid.
    int32_t const swaStart = max(position - windowSize + 1, 0);
    int32_t* swaRow = swaLocalIndices + static_cast<int64_t>(tokenId) * swaStride;
    for (int32_t col = static_cast<int32_t>(threadIdx.x); col < windowSize; col += static_cast<int32_t>(blockDim.x))
    {
        int32_t const idx = swaStart + col;
        swaRow[col] = idx > position ? kInvalidIndex : idx;
    }

    // ── Compressed local indices (ratio 128): first `kvLen / 128` columns are
    //    their own index, the rest are invalid.
    int32_t const numValid = kvLen / 128;
    int32_t* compRow = compressedLocalIndices + static_cast<int64_t>(tokenId) * compressedStride;
    for (int32_t col = static_cast<int32_t>(threadIdx.x); col < maxCompressedIndices;
         col += static_cast<int32_t>(blockDim.x))
    {
        compRow[col] = col < numValid ? col : kInvalidIndex;
    }

    // ── sparse_mla_topk_lens per compression ratio (one scalar per token).
    if (threadIdx.x == 0)
    {
        if (topkLensRatio1 != nullptr)
        {
            topkLensRatio1[tokenId] = min(kvLen, windowSize);
        }
        if (topkLensRatio4 != nullptr)
        {
            topkLensRatio4[tokenId] = windowSize + min(kvLen / 4, sparseMlaTopk);
        }
        if (topkLensRatio128 != nullptr)
        {
            topkLensRatio128[tokenId] = windowSize + kvLen / 128;
        }
    }
}
} // namespace

void invokeDeepseekV4ComputeIndices(int32_t const* tokenPositions, int32_t* swaLocalIndices,
    int32_t* compressedLocalIndices, int32_t* topkLensRatio1, int32_t* topkLensRatio4, int32_t* topkLensRatio128,
    int32_t numTokens, int32_t windowSize, int32_t maxCompressedIndices, int32_t sparseMlaTopk, int32_t swaStride,
    int32_t compressedStride, cudaStream_t stream)
{
    if (numTokens <= 0)
    {
        return;
    }
    int32_t const widest = max(windowSize, maxCompressedIndices);
    int32_t threads = widest >= kThreadsPerBlock ? kThreadsPerBlock : max(widest, 32);
    dim3 const block(static_cast<uint32_t>(threads));
    dim3 const grid(static_cast<uint32_t>(numTokens));
    computeDeepseekV4IndicesKernel<<<grid, block, 0, stream>>>(tokenPositions, swaLocalIndices, compressedLocalIndices,
        topkLensRatio1, topkLensRatio4, topkLensRatio128, numTokens, windowSize, maxCompressedIndices, sparseMlaTopk,
        swaStride, compressedStride);
}

} // namespace kernels

TRTLLM_NAMESPACE_END
