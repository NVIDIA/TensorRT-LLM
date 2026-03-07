/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "fusedDiTQKNormRopeKernel.h"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/mathUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include <cuda_bf16.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace common
{
// Specialization for packed_as used in vectorized loads/stores.
template <>
struct packed_as<uint, 1>
{
    using type = uint;
};

template <>
struct packed_as<uint, 2>
{
    using type = uint2;
};

template <>
struct packed_as<uint, 4>
{
    using type = uint4;
};
} // namespace common

TRTLLM_NAMESPACE_END
TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

////////////////////////////////////////////////////////////////////////////////////////////////////

// Fused per-head QK Norm + RoPE kernel for Diffusion Transformers.
//
// Each warp processes one head of one token (Q or K only; V is untouched).
// Uses precomputed cos/sin embeddings and supports dual-stream attention
// where text tokens (tokenIdx < num_txt_tokens) use add_weights instead of
// primary weights for RMSNorm.
//
// RoPE mode: interleaved (pairs elements [2i, 2i+1]), matching FLUX's
// repeat_interleave_real=True convention.
template <int head_dim>
__global__ void fusedDiTQKNormRopeKernel(__nv_bfloat16* qkv, // [num_tokens, total_heads * head_dim]
    int const num_heads_q, int const num_heads_k, int const num_heads_v, float const eps,
    __nv_bfloat16 const* q_weight,                           // [head_dim]
    __nv_bfloat16 const* k_weight,                           // [head_dim]
    __nv_bfloat16 const* q_add_weight,                       // [head_dim] or nullptr
    __nv_bfloat16 const* k_add_weight,                       // [head_dim] or nullptr
    float const* cos_emb,                                    // [num_tokens, head_dim]
    float const* sin_emb,                                    // [num_tokens, head_dim]
    int const num_tokens, int const num_txt_tokens)
{
    int const warpsPerBlock = blockDim.x / 32;
    int const warpId = threadIdx.x / 32;
    int const laneId = threadIdx.x % 32;

    int const globalWarpIdx = blockIdx.x * warpsPerBlock + warpId;

    int const total_qk_heads = num_heads_q + num_heads_k;

    // Map warp → (token, head type)
    int const tokenIdx = globalWarpIdx / total_qk_heads;
    int const localHeadIdx = globalWarpIdx % total_qk_heads;

    if (tokenIdx >= num_tokens)
    {
        return;
    }

    bool const isQ = localHeadIdx < num_heads_q;
    int const headIdx = isQ ? localHeadIdx : localHeadIdx - num_heads_q;

    int const num_heads = num_heads_q + num_heads_k + num_heads_v;

    // Each warp (32 threads) processes one head of head_dim elements.
    static_assert(
        head_dim % (32 * 2) == 0, "head_dim must be divisible by 64 (each warp thread gets even number of elements)");
    constexpr int numElemsPerThread = head_dim / 32;
    float elements[numElemsPerThread];
    constexpr int elemSizeBytes = numElemsPerThread * sizeof(__nv_bfloat16);
    static_assert(elemSizeBytes % 4 == 0, "elemSizeBytes must be a multiple of 4");
    constexpr int vecSize = elemSizeBytes / 4;
    using vec_T = typename tensorrt_llm::common::packed_as<uint, vecSize>::type;

    // Compute offset into packed QKV tensor
    int offsetWarp;
    if (isQ)
    {
        offsetWarp = tokenIdx * num_heads * head_dim + headIdx * head_dim;
    }
    else
    {
        offsetWarp = tokenIdx * num_heads * head_dim + num_heads_q * head_dim + headIdx * head_dim;
    }
    int offsetThread = offsetWarp + laneId * numElemsPerThread;

    // ---- Step 1: Load elements and compute sum of squares ----
    float sumOfSquares = 0.0f;
    {
        vec_T vec = *reinterpret_cast<vec_T const*>(&qkv[offsetThread]);
        for (int i = 0; i < vecSize; i++)
        {
            float2 vals = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<uint*>(&vec) + i));
            sumOfSquares += vals.x * vals.x;
            sumOfSquares += vals.y * vals.y;
            elements[2 * i] = vals.x;
            elements[2 * i + 1] = vals.y;
        }
    }

    // ---- Step 2: RMS normalization with dual-stream weight selection ----
    sumOfSquares = tensorrt_llm::common::warpReduceSum(sumOfSquares);
    float rms_rcp = rsqrtf(sumOfSquares / static_cast<float>(head_dim) + eps);

    // Select norm weight: text tokens use add_weight (if provided), image tokens use primary weight
    bool const useAddWeight = (num_txt_tokens > 0) && (tokenIdx < num_txt_tokens);

    __nv_bfloat16 const* weight_ptr;
    if (isQ)
    {
        weight_ptr = (useAddWeight && q_add_weight != nullptr) ? q_add_weight : q_weight;
    }
    else
    {
        weight_ptr = (useAddWeight && k_add_weight != nullptr) ? k_add_weight : k_weight;
    }

    for (int i = 0; i < numElemsPerThread; i++)
    {
        int dim = laneId * numElemsPerThread + i;
        float weight = __bfloat162float(weight_ptr[dim]);
        elements[i] *= rms_rcp * weight;
    }

    // ---- Step 3: Apply interleaved RoPE with precomputed cos/sin ----
    // cos_emb/sin_emb layout: [num_tokens, head_dim]
    // Interleaved pairing: (element[2i], element[2i+1])
    // Process pairs together to avoid in-place overwrite of the first element
    // before the second element reads it.
    int const embOffset = tokenIdx * head_dim;

    for (int i = 0; i < numElemsPerThread; i += 2)
    {
        int dim = laneId * numElemsPerThread + i;
        float cos0 = cos_emb[embOffset + dim];
        float sin0 = sin_emb[embOffset + dim];
        float cos1 = cos_emb[embOffset + dim + 1];
        float sin1 = sin_emb[embOffset + dim + 1];

        float x = elements[i];
        float y = elements[i + 1];

        elements[i] = x * cos0 + (-y) * sin0;
        elements[i + 1] = y * cos1 + x * sin1;
    }

    // ---- Step 4: Store back ----
    {
        vec_T vec;
        for (int i = 0; i < vecSize; i++)
        {
            __nv_bfloat162 vals = __float22bfloat162_rn(make_float2(elements[2 * i], elements[2 * i + 1]));
            reinterpret_cast<__nv_bfloat162&>(*(reinterpret_cast<uint*>(&vec) + i)) = vals;
        }
        vec_T* outputPtr = reinterpret_cast<vec_T*>(&qkv[offsetThread]);
        *outputPtr = vec;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void launchFusedDiTQKNormRope(void* qkv, int num_tokens, int num_heads_q, int num_heads_k, int num_heads_v,
    int head_dim, float eps, void const* q_weight, void const* k_weight, void const* q_add_weight,
    void const* k_add_weight, float const* cos_emb, float const* sin_emb, int num_txt_tokens, cudaStream_t stream)
{
    constexpr int blockSize = 256;

    int const warpsPerBlock = blockSize / 32;
    int const totalQKHeads = num_heads_q + num_heads_k;
    int const totalWarps = num_tokens * totalQKHeads;

    int const gridSize = common::divUp(totalWarps, warpsPerBlock);
    dim3 gridDim(gridSize);
    dim3 blockDim(blockSize);

    switch (head_dim)
    {
    case 64:
        fusedDiTQKNormRopeKernel<64><<<gridDim, blockDim, 0, stream>>>(reinterpret_cast<__nv_bfloat16*>(qkv),
            num_heads_q, num_heads_k, num_heads_v, eps, reinterpret_cast<__nv_bfloat16 const*>(q_weight),
            reinterpret_cast<__nv_bfloat16 const*>(k_weight), reinterpret_cast<__nv_bfloat16 const*>(q_add_weight),
            reinterpret_cast<__nv_bfloat16 const*>(k_add_weight), cos_emb, sin_emb, num_tokens, num_txt_tokens);
        break;
    case 128:
        fusedDiTQKNormRopeKernel<128><<<gridDim, blockDim, 0, stream>>>(reinterpret_cast<__nv_bfloat16*>(qkv),
            num_heads_q, num_heads_k, num_heads_v, eps, reinterpret_cast<__nv_bfloat16 const*>(q_weight),
            reinterpret_cast<__nv_bfloat16 const*>(k_weight), reinterpret_cast<__nv_bfloat16 const*>(q_add_weight),
            reinterpret_cast<__nv_bfloat16 const*>(k_add_weight), cos_emb, sin_emb, num_tokens, num_txt_tokens);
        break;
    case 256:
        fusedDiTQKNormRopeKernel<256><<<gridDim, blockDim, 0, stream>>>(reinterpret_cast<__nv_bfloat16*>(qkv),
            num_heads_q, num_heads_k, num_heads_v, eps, reinterpret_cast<__nv_bfloat16 const*>(q_weight),
            reinterpret_cast<__nv_bfloat16 const*>(k_weight), reinterpret_cast<__nv_bfloat16 const*>(q_add_weight),
            reinterpret_cast<__nv_bfloat16 const*>(k_add_weight), cos_emb, sin_emb, num_tokens, num_txt_tokens);
        break;
    default: TLLM_THROW("Unsupported head dimension for fusedDiTQKNormRope: %d", head_dim);
    }
}

} // namespace kernels

TRTLLM_NAMESPACE_END
