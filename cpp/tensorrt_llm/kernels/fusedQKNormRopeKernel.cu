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
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/mathUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace tensorrt_llm::common
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
} // namespace tensorrt_llm::common

namespace tensorrt_llm::kernels
{

////////////////////////////////////////////////////////////////////////////////////////////////////

// Perform per-head QK Norm and RoPE in a single kernel.
// head_dim: the dimension of each head
// interleave: interleave=!is_neox.
template <int head_dim, bool interleave>
__global__ void fusedQKNormRopeKernel(
    __nv_bfloat16* qkv,            // Combined QKV tensor [num_tokens, (num_heads_q+num_heads_k+num_heads_v)*head_dim]
    int const num_heads_q,         // Number of query heads
    int const num_heads_k,         // Number of key heads
    int const num_heads_v,         // Number of value heads
    float const eps,               // Epsilon for RMS normalization
    __nv_bfloat16 const* q_weight, // RMSNorm weights for query
    __nv_bfloat16 const* k_weight, // RMSNorm weights for key
    float const base,              // Base for RoPE computation
    int const* position_ids,       // Position IDs for RoPE
    int const num_tokens           // Number of tokens
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
        return;

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

    // Apply RoPE to normalized elements
    float elements2[numElemsPerThread]; // Additional buffer required for RoPE.
    float cos_vals[numElemsPerThread];
    float sin_vals[numElemsPerThread];

    float pos_id = static_cast<float>(position_ids[tokenIdx]);

    // TODO: cos sin calculation could be halved.
    if constexpr (interleave)
    {
        // Perform interleaving. Fill cos_vals and sin_vals.
        for (int i = 0; i < numElemsPerThread; i++)
        {
            if (i % 2 == 0)
            {
                elements2[i] = -elements[i + 1];
            }
            else
            {
                elements2[i] = elements[i - 1];
            }

            int dim_idx = laneId * numElemsPerThread + i;
            int half_dim = dim_idx / 2;
            float freq = powf(base, -2.0f * half_dim / static_cast<float>(head_dim));
            float theta = pos_id * freq;
            __sincosf(theta, &sin_vals[i], &cos_vals[i]);
        }
    }
    else
    {
        // Before data exchange with in warp, we need to sync.
        __syncwarp();
        // Get the data from the other half of the warp. Fill cos_vals and sin_vals.
        for (int i = 0; i < numElemsPerThread; i++)
        {
            elements2[i] = __shfl_xor_sync(0xffffffff, elements[i], 16);
            if (laneId < 16)
            {
                elements2[i] = -elements2[i];
            }

            int dim_idx = laneId * numElemsPerThread + i;
            dim_idx = (dim_idx * 2) % head_dim;
            int half_dim = dim_idx / 2;
            float freq = powf(base, -2.0f * half_dim / static_cast<float>(head_dim));
            float theta = pos_id * freq;
            __sincosf(theta, &sin_vals[i], &cos_vals[i]);
        }
        // __shfl_xor_sync does not provide memfence. Need to sync again.
        __syncwarp();
    }

    for (int i = 0; i < numElemsPerThread; i++)
    {
        elements[i] = elements[i] * cos_vals[i] + elements2[i] * sin_vals[i];
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
        const bool INTERLEAVE = true;                                                                                  \
        __VA_ARGS__                                                                                                    \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        const bool INTERLEAVE = false;                                                                                 \
        __VA_ARGS__                                                                                                    \
    }

void launchFusedQKNormRope(void* qkv, int const num_tokens, int const num_heads_q, int const num_heads_k,
    int const num_heads_v, int const head_dim, float const eps, void const* q_weight, void const* k_weight,
    float const base, bool const interleave, int const* position_ids, cudaStream_t stream)
{
    constexpr int blockSize = 256;

    int const warpsPerBlock = blockSize / 32;
    int const totalQKHeads = num_heads_q + num_heads_k;
    int const totalWarps = num_tokens * totalQKHeads;

    int const gridSize = common::divUp(totalWarps, warpsPerBlock);
    dim3 gridDim(gridSize);
    dim3 blockDim(blockSize);

    // Head dimensions should be a multiple of 64
    // Add more cases as needed
    switch (head_dim)
    {
    case 64:
        DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
            fusedQKNormRopeKernel<64, INTERLEAVE>
                <<<gridDim, blockDim, 0, stream>>>(reinterpret_cast<__nv_bfloat16*>(qkv), num_heads_q, num_heads_k,
                    num_heads_v, eps, reinterpret_cast<__nv_bfloat16 const*>(q_weight),
                    reinterpret_cast<__nv_bfloat16 const*>(k_weight), base, position_ids, num_tokens);
        });
        break;
    case 128:
        DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
            fusedQKNormRopeKernel<128, INTERLEAVE>
                <<<gridDim, blockDim, 0, stream>>>(reinterpret_cast<__nv_bfloat16*>(qkv), num_heads_q, num_heads_k,
                    num_heads_v, eps, reinterpret_cast<__nv_bfloat16 const*>(q_weight),
                    reinterpret_cast<__nv_bfloat16 const*>(k_weight), base, position_ids, num_tokens);
        });
        break;
    default: TLLM_THROW("Unsupported head dimension for fusedQKNormRope: %d", head_dim);
    }
}
} // namespace tensorrt_llm::kernels
