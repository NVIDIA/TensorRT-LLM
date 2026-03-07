/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

#include "fusedQKNormRopeKernel.h"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/mathUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace common
{
// Specialization for packed_as used in this kernel.
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
// Apply the RoPE rotation to a single element.
////////////////////////////////////////////////////////////////////////////////////////////////////
template <bool full_rotary>
__device__ __forceinline__ float applyRoPERotation(float element, float paired_element, float sin_val, float cos_val,
    float attention_factor, int dim_idx, int rotary_dim)
{
    if constexpr (full_rotary)
    {
        return (element * cos_val + paired_element * sin_val) * attention_factor;
    }

    float rotated_element{element};
    if (dim_idx < rotary_dim)
    {
        rotated_element = (element * cos_val + paired_element * sin_val) * attention_factor;
    }

    return rotated_element;
}

// Apply YaRN frequency scaling to adjust the RoPE frequency
__device__ __forceinline__ float applyYaRNScaling(float freq, int half_dim, float factor, float low, float high)
{
    float const inv_freq_extrapolation = freq;
    float const inv_freq_interpolation = freq / factor;

    // Inverse lerp
    float const linear_func = (static_cast<float>(half_dim) - low) / (high - low);
    // clamp linear_func to [0.0f, 1.0f]
    float const ramp_func = fminf(fmaxf(linear_func, 0.0f), 1.0f);
    return inv_freq_interpolation * ramp_func + inv_freq_extrapolation * (1.0f - ramp_func);
}

// Determine the RoPE frequency for a given dimension index, applying YaRN scaling if needed
__device__ __forceinline__ float computeRoPEFrequency(
    float base, int half_dim, int rotary_dim, float factor, float low, float high)
{
    float freq = powf(base, -2.0f * half_dim / static_cast<float>(rotary_dim));
    if (factor != 1.0f)
    {
        freq = applyYaRNScaling(freq, half_dim, factor, low, high);
    }
    return freq;
}

// Perform per-head QK Norm and RoPE in a single kernel.
// head_dim: the dimension of each head
// interleave: interleave=!is_neox.
// full_rotary: true when rotary_dim == head_dim (no partial-RoPE predicate needed in the kernel)
template <int head_dim, bool interleave, bool full_rotary>
__global__ void fusedQKNormRopeKernel(
    __nv_bfloat16* qkv,            // Combined QKV tensor [num_tokens, (num_heads_q+num_heads_k+num_heads_v)*head_dim]
    int const num_heads_q,         // Number of query heads
    int const num_heads_k,         // Number of key heads
    int const num_heads_v,         // Number of value heads
    int const rotary_dim,          // Dimension for RoPE
    float const eps,               // Epsilon for RMS normalization
    __nv_bfloat16 const* q_weight, // RMSNorm weights for query
    __nv_bfloat16 const* k_weight, // RMSNorm weights for key
    float const base,              // Base for RoPE computation
    int const* position_ids,       // Position IDs for RoPE
    int const num_tokens,          // Number of tokens
    // parameters for yarn
    float factor, // factor in rope_scaling in config.json. When it is not 1.0, it means the model is using yarn.
    float low,    // threshold for high frequency
    float high,   // threshold for low frequency
    float attention_factor, // attention_factor applied on cos and sin
    // stop of parameters for yarn
    bool is_qk_norm // Whether to apply QK norm
)
{
    int const warpsPerBlock = blockDim.x / 32;
    int const warpId = threadIdx.x / 32;
    int const laneId = threadIdx.x % 32;

    // Calculate global warp index to determine which head/token this warp processes
    int const globalWarpIdx = blockIdx.x * warpsPerBlock + warpId;

    // Total number of attention heads (Q and K)
    int const total_qk_heads = num_heads_q + num_heads_k;

    // Determine which token and head type (Q or K) this warp processes
    int const tokenIdx = globalWarpIdx / total_qk_heads;
    int const localHeadIdx = globalWarpIdx % total_qk_heads;

    // Skip if this warp is assigned beyond the number of tokens
    if (tokenIdx >= num_tokens)
    {
        return;
    }

    bool const isQ = localHeadIdx < num_heads_q;
    int const headIdx = isQ ? localHeadIdx : localHeadIdx - num_heads_q;

    int const num_heads = num_heads_q + num_heads_k + num_heads_v;

    static_assert(head_dim % (32 * 2) == 0,
        "head_dim must be divisible by 64 (each warp processes one head, and each thread gets even number of "
        "elements)");
    constexpr int numElemsPerThread = head_dim / 32;
    float elements[numElemsPerThread];
    constexpr int elemSizeBytes = numElemsPerThread * sizeof(__nv_bfloat16);
    static_assert(elemSizeBytes % 4 == 0, "numSizeBytes must be a multiple of 4");
    constexpr int vecSize = elemSizeBytes / 4; // Use packed_as<uint, vecSize> to perform loading/saving.
    using vec_T = typename tensorrt_llm::common::packed_as<uint, vecSize>::type;

    int offsetWarp; // Offset for the warp
    if (isQ)
    {
        // Q segment: token offset + head offset within Q segment
        offsetWarp = tokenIdx * num_heads * head_dim + headIdx * head_dim;
    }
    else
    {
        // K segment: token offset + entire Q segment + head offset within K segment
        offsetWarp = tokenIdx * num_heads * head_dim + num_heads_q * head_dim + headIdx * head_dim;
    }
    int offsetThread = offsetWarp + laneId * numElemsPerThread;

    // Sum of squares for RMSNorm
    float sumOfSquares = 0.0f;

    // Load.
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

    if (is_qk_norm)
    {
        __syncwarp();
        // Reduce sum across warp using the utility function
        sumOfSquares = tensorrt_llm::common::warpReduceSum(sumOfSquares);

        // Compute RMS normalization factor
        float rms_rcp = rsqrtf(sumOfSquares / static_cast<float>(head_dim) + eps);

        // Normalize elements
        for (int i = 0; i < numElemsPerThread; i++)
        {
            int dim = laneId * numElemsPerThread + i;
            float weight = isQ ? __bfloat162float(q_weight[dim]) : __bfloat162float(k_weight[dim]);
            elements[i] *= rms_rcp * weight;
        }
    }

    float const pos_id = static_cast<float>(position_ids[tokenIdx]);

    if constexpr (interleave)
    {
        // Compute RoPE with interleaving. Even/odd pairs share the same half_dim
        // so sin/cos are computed once per pair and reused
        for (int i = 0; i < numElemsPerThread; i += 2)
        {
            int const dim_idx = laneId * numElemsPerThread + i;
            int const half_dim = dim_idx / 2;
            float const freq{computeRoPEFrequency(base, half_dim, rotary_dim, factor, low, high)};
            float const theta = pos_id * freq;
            float sin_val, cos_val;
            __sincosf(theta, &sin_val, &cos_val);

            // Save originals before mutation
            float const orig_even = elements[i];
            float const orig_odd = elements[i + 1];

            // Even element: paired with next
            elements[i] = applyRoPERotation<full_rotary>(
                orig_even, -orig_odd, sin_val, cos_val, attention_factor, dim_idx, rotary_dim);
            // Odd element: paired with previous (reuse same sin/cos)
            elements[i + 1] = applyRoPERotation<full_rotary>(
                orig_odd, orig_even, sin_val, cos_val, attention_factor, dim_idx + 1, rotary_dim);
        }
    }
    else
    {
        // Before data exchange with in warp, we need to sync.
        __syncwarp();
        int const pairOffset = (rotary_dim / 2) / numElemsPerThread;
        for (int i = 0; i < numElemsPerThread; i++)
        {
            float elem2 = __shfl_xor_sync(0xffffffff, elements[i], pairOffset);
            if (laneId < pairOffset)
            {
                elem2 = -elem2;
            }

            int const orig_dim_idx = laneId * numElemsPerThread + i;
            int const dim_idx = (orig_dim_idx * 2) % rotary_dim;
            int const half_dim = dim_idx / 2;

            float const freq{computeRoPEFrequency(base, half_dim, rotary_dim, factor, low, high)};

            float const theta = pos_id * freq;
            float sin_val, cos_val;
            __sincosf(theta, &sin_val, &cos_val);

            elements[i] = applyRoPERotation<full_rotary>(
                elements[i], elem2, sin_val, cos_val, attention_factor, orig_dim_idx, rotary_dim);
        }
        // __shfl_xor_sync does not provide memfence. Need to sync again.
        __syncwarp();
    }

    // Store.
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

// Borrowed from
// https://github.com/flashinfer-ai/flashinfer/blob/8125d079a43e9a0ba463a4ed1b639cefd084cec9/include/flashinfer/pos_enc.cuh#L568
#define DISPATCH_INTERLEAVE(interleave, INTERLEAVE, ...)                                                               \
    if (interleave)                                                                                                    \
    {                                                                                                                  \
        bool const INTERLEAVE = true;                                                                                  \
        __VA_ARGS__                                                                                                    \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        bool const INTERLEAVE = false;                                                                                 \
        __VA_ARGS__                                                                                                    \
    }

#define DISPATCH_FULL_ROTARY(full_rotary, FULL_ROTARY, ...)                                                            \
    if (full_rotary)                                                                                                   \
    {                                                                                                                  \
        bool const FULL_ROTARY = true;                                                                                 \
        __VA_ARGS__                                                                                                    \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        bool const FULL_ROTARY = false;                                                                                \
        __VA_ARGS__                                                                                                    \
    }

// Head dimensions should be a multiple of 64
// Add more cases as needed
#define DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, ...)                                                                     \
    switch (head_dim)                                                                                                  \
    {                                                                                                                  \
    case 64:                                                                                                           \
    {                                                                                                                  \
        constexpr int HEAD_DIM = 64;                                                                                   \
        __VA_ARGS__                                                                                                    \
    }                                                                                                                  \
    break;                                                                                                             \
    case 128:                                                                                                          \
    {                                                                                                                  \
        constexpr int HEAD_DIM = 128;                                                                                  \
        __VA_ARGS__                                                                                                    \
    }                                                                                                                  \
    break;                                                                                                             \
    case 256:                                                                                                          \
    {                                                                                                                  \
        constexpr int HEAD_DIM = 256;                                                                                  \
        __VA_ARGS__                                                                                                    \
    }                                                                                                                  \
    break;                                                                                                             \
    default: TLLM_THROW("Unsupported head dimension for fusedQKNormRope: %d", head_dim);                               \
    }

void launchFusedQKNormRope(void* qkv, int const num_tokens, int const num_heads_q, int const num_heads_k,
    int const num_heads_v, int const head_dim, int const rotary_dim, float const eps, void const* q_weight,
    void const* k_weight, float const base, bool const interleave, int const* position_ids, float factor, float low,
    float high, float attention_factor, cudaStream_t stream, bool is_qk_norm)
{
    if (factor == 1.0f)
    {
        TLLM_CHECK(attention_factor == 1.0f);
    }

    TLLM_CHECK_WITH_INFO(rotary_dim % 2 == 0, "rotary_dim must be even");
    if (!interleave)
    {
        // To allow warp-level pairing for partial rope
        TLLM_CHECK_WITH_INFO(
            (rotary_dim * 16) % head_dim == 0, "Unsupported rotary dimension for fusedQKNormRope: %d", rotary_dim);
    }

    constexpr dim3 blockSize(256, 1, 1);
    constexpr int warpsPerBlock = blockSize.x / 32;

    int const totalQKHeads = num_heads_q + num_heads_k;
    int const totalWarps = num_tokens * totalQKHeads;

    int const gridSize = common::divUp(totalWarps, warpsPerBlock);

    // linear_ramp_factor
    if (factor != 1.0f)
    {
        high += 0.001f; // Prevent singularity
    }

    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
        DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
            DISPATCH_FULL_ROTARY(rotary_dim == head_dim, FULL_ROTARY, {
                fusedQKNormRopeKernel<HEAD_DIM, INTERLEAVE, FULL_ROTARY><<<gridSize, blockSize, 0, stream>>>(
                    reinterpret_cast<__nv_bfloat16*>(qkv), num_heads_q, num_heads_k, num_heads_v, rotary_dim, eps,
                    reinterpret_cast<__nv_bfloat16 const*>(q_weight), static_cast<__nv_bfloat16 const*>(k_weight), base,
                    position_ids, num_tokens, factor, low, high, attention_factor, is_qk_norm);
            });
        });
    });
}
} // namespace kernels

TRTLLM_NAMESPACE_END
