/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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
#include <type_traits>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

////////////////////////////////////////////////////////////////////////////////////////////////////

// Select the RoPE position id for a given rotary half-dim under interleaved mRoPE.
// Mirrors MRotaryEmbedding.apply_interleaved_rope: section 1 (height) drives
// dims {1,4,7,...} up to mrope_section1*3, section 2 (width) drives {2,5,8,...}
// up to mrope_section2*3, everything else uses section 0 (temporal).
// position_ids is [num_tokens] for the non-mRoPE case (sec is always 0) and
// [3, num_tokens] (row-major: sec*num_tokens + tokenIdx) for mRoPE.
__device__ __forceinline__ float selectMRopePosId(int const* position_ids, int tokenIdx, int num_tokens, int half_dim,
    bool use_mrope, int mrope_section1, int mrope_section2)
{
    int sec = 0;
    if (use_mrope)
    {
        if (half_dim % 3 == 1 && half_dim < mrope_section1 * 3)
        {
            sec = 1;
        }
        else if (half_dim % 3 == 2 && half_dim < mrope_section2 * 3)
        {
            sec = 2;
        }
    }
    return static_cast<float>(position_ids[sec * num_tokens + tokenIdx]);
}

// Store a per-thread run of `numElemsPerThread` float elements, converting to
// OutT. The FP8 path saturates to +/-448, whereas torch's .to(float8_e4m3fn)
// produces NaN for out-of-range values.
template <typename OutT, int numElemsPerThread, int vecSize>
__device__ __forceinline__ void storeHeadElements(
    OutT* out, int offsetThread, float const (&elements)[numElemsPerThread])
{
    using vec_T = typename tensorrt_llm::common::packed_as<uint, vecSize>::type;
    if constexpr (std::is_same_v<OutT, __nv_bfloat16>)
    {
        vec_T vec;
#pragma unroll
        for (int i = 0; i < vecSize; i++)
        {
            __nv_bfloat162 vals = __float22bfloat162_rn(make_float2(elements[2 * i], elements[2 * i + 1]));
            reinterpret_cast<__nv_bfloat162&>(*(reinterpret_cast<uint*>(&vec) + i)) = vals;
        }
        *reinterpret_cast<vec_T*>(&out[offsetThread]) = vec;
    }
    else // __nv_fp8_e4m3
    {
        static_assert(numElemsPerThread % 2 == 0, "FP8 store expects an even element count per thread");
#pragma unroll
        for (int i = 0; i < numElemsPerThread; i += 2)
        {
            __nv_fp8x2_e4m3 packed(make_float2(elements[i], elements[i + 1]));
            reinterpret_cast<__nv_fp8x2_storage_t*>(&out[offsetThread])[i / 2] = packed.__x;
        }
    }
}

// Perform per-head QK Norm and RoPE in a single kernel, reading a BF16 input and
// writing to a (possibly different-dtype) output buffer.
// head_dim: the dimension of each head
// interleave: interleave=!is_neox.
// OutT: output element type (__nv_bfloat16 or __nv_fp8_e4m3).
template <int head_dim, bool interleave, typename OutT>
__global__ void fusedQKNormRopeKernel(
    __nv_bfloat16 const* qkv_in,   // Combined QKV input [num_tokens, (num_heads_q+num_heads_k+num_heads_v)*head_dim]
    OutT* qkv_out,                 // Output buffer, same layout as qkv_in
    int const num_heads_q,         // Number of query heads
    int const num_heads_k,         // Number of key heads
    int const num_heads_v,         // Number of value heads
    bool const process_v,          // Whether to copy-cast V heads into qkv_out
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
    bool is_qk_norm, // Whether to apply QK norm
    bool use_gemma,  // Whether QK norm uses Gemma-style RMSNorm (scale by (1 + weight))
    // parameters for interleaved mRoPE (use_mrope=false -> plain RoPE, single position per token)
    bool use_mrope,     // Whether to use interleaved mRoPE position selection
    int mrope_section1, // mrope_section[1] (height); section 0 (temporal) is implied
    int mrope_section2  // mrope_section[2] (width)
)
{
    int const warpsPerBlock = blockDim.x / 32;
    int const warpId = threadIdx.x / 32;
    int const laneId = threadIdx.x % 32;

    // Calculate global warp index to determine which head/token this warp processes
    int const globalWarpIdx = blockIdx.x * warpsPerBlock + warpId;

    // Total number of attention heads (Q and K)
    int const total_qk_heads = num_heads_q + num_heads_k;
    int const total_proc_heads = total_qk_heads + (process_v ? num_heads_v : 0);

    // Determine which token and head this warp processes
    int const tokenIdx = globalWarpIdx / total_proc_heads;
    int const localHeadIdx = globalWarpIdx % total_proc_heads;

    // Skip if this warp is assigned beyond the number of tokens
    if (tokenIdx >= num_tokens)
        return;

    bool const isQ = localHeadIdx < num_heads_q;
    bool const isV = localHeadIdx >= total_qk_heads;
    int headIdx;  // index within the head's own Q/K/V segment
    int segStart; // element offset of the segment start within a token row
    if (isQ)
    {
        headIdx = localHeadIdx;
        segStart = 0;
    }
    else if (!isV)
    {
        headIdx = localHeadIdx - num_heads_q;
        segStart = num_heads_q * head_dim;
    }
    else
    {
        headIdx = localHeadIdx - total_qk_heads;
        segStart = total_qk_heads * head_dim;
    }

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

    int const offsetWarp = tokenIdx * num_heads * head_dim + segStart + headIdx * head_dim;
    int offsetThread = offsetWarp + laneId * numElemsPerThread;

    // Sum of squares for RMSNorm
    float sumOfSquares = 0.0f;

    // Load.
    {
        vec_T vec = *reinterpret_cast<vec_T const*>(&qkv_in[offsetThread]);
#pragma unroll
        for (int i = 0; i < vecSize; i++)
        {
            float2 vals = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<uint*>(&vec) + i));
            sumOfSquares += vals.x * vals.x;
            sumOfSquares += vals.y * vals.y;

            elements[2 * i] = vals.x;
            elements[2 * i + 1] = vals.y;
        }
    }

    // V heads are copy-cast only: no norm, no RoPE.
    if (isV)
    {
        storeHeadElements<OutT, numElemsPerThread, vecSize>(qkv_out, offsetThread, elements);
        return;
    }

    if (is_qk_norm)
    {
        // Reduce sum across warp using the utility function
        sumOfSquares = tensorrt_llm::common::warpReduceSum(sumOfSquares);

        // Compute RMS normalization factor
        float rms_rcp = rsqrtf(sumOfSquares / static_cast<float>(head_dim) + eps);

        // Normalize elements
        for (int i = 0; i < numElemsPerThread; i++)
        {
            int dim = laneId * numElemsPerThread + i;
            float weight = isQ ? __bfloat162float(q_weight[dim]) : __bfloat162float(k_weight[dim]);
            // Gemma RMSNorm scales by (1 + weight); standard RMSNorm scales by weight.
            elements[i] *= rms_rcp * (use_gemma ? (1.0f + weight) : weight);
        }
    }
    // Apply RoPE to normalized elements
    float elements2[numElemsPerThread]; // Additional buffer required for RoPE.
    float cos_vals[numElemsPerThread];
    float sin_vals[numElemsPerThread];

    // pos_id is selected per rotary half-dim (interleaved mRoPE); for plain RoPE
    // selectMRopePosId always returns position_ids[tokenIdx].

    // Hoist log2(base) and the loop-invariant constants out of the per-thread
    // per-elem loop. powf(base, -2*hd/rd) == exp2f(-2*hd/rd * log2(base)); base
    // and rotary_dim are kernel-uniform, so one MUFU.LG2 per warp instead of
    // one per (thread, iter). Uses the fast __log2f intrinsic (a few ULPs of
    // error, absorbed by bf16 downcast at store time).
    float const neg2_log2base_over_rd = -2.0f * __log2f(base) / static_cast<float>(rotary_dim);
    // rotary_dim is even by contract; when it's also a power of 2 (always in
    // practice — 64/128/256) '% rotary_dim' becomes '& (rotary_dim - 1)'.
    // The bool is warp-uniform → predicated select, no branch divergence.
    int const rd_mask = rotary_dim - 1;
    bool const rd_is_pow2 = ((rotary_dim & rd_mask) == 0);
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
            float freq = exp2f(static_cast<float>(half_dim) * neg2_log2base_over_rd);

            if (factor != 1.0f)
            {
                float inv_freq_extrapolation = freq;
                float inv_freq_interpolation = freq / factor;

                // linear_ramp_factor
                if (fabsf(low - high) <= 1e-6f)
                {
                    high += 0.001; // Prevent singularity
                }
                float linear_func = (static_cast<float>(half_dim) - low) / (high - low);
                // clamp linear_func to [0.0f, 1.0f]
                float ramp_func = fmin(fmax(linear_func, 0.0f), 1.0f);
                float inv_freq_extrapolation_factor = 1.0f - ramp_func;
                freq = inv_freq_interpolation * (1.0f - inv_freq_extrapolation_factor)
                    + inv_freq_extrapolation * inv_freq_extrapolation_factor;
            }

            float pos_id = selectMRopePosId(
                position_ids, tokenIdx, num_tokens, half_dim, use_mrope, mrope_section1, mrope_section2);
            float theta = pos_id * freq;
            __sincosf(theta, &sin_vals[i], &cos_vals[i]);
        }
    }
    else
    {
        // Before data exchange with in warp, we need to sync.
        __syncwarp();
        int pairOffset = (rotary_dim / 2) / numElemsPerThread;
        // Get the data from the other half of the warp. Fill cos_vals and sin_vals.
        for (int i = 0; i < numElemsPerThread; i++)
        {
            elements2[i] = __shfl_xor_sync(0xffffffff, elements[i], pairOffset);
            if (laneId < pairOffset)
            {
                elements2[i] = -elements2[i];
            }

            int dim_idx = laneId * numElemsPerThread + i;
            dim_idx = rd_is_pow2 ? ((dim_idx * 2) & rd_mask) : ((dim_idx * 2) % rotary_dim);
            int half_dim = dim_idx / 2;
            float freq = exp2f(static_cast<float>(half_dim) * neg2_log2base_over_rd);

            if (factor != 1.0f)
            {
                float inv_freq_extrapolation = freq;
                float inv_freq_interpolation = freq / factor;

                // linear_ramp_factor
                if (fabsf(low - high) <= 1e-6f)
                {
                    high += 0.001; // Prevent singularity
                }
                float linear_func = (static_cast<float>(half_dim) - low) / (high - low);
                // clamp linear_func to [0.0f, 1.0f]
                float ramp_func = fmin(fmax(linear_func, 0.0f), 1.0f);
                float inv_freq_extrapolation_factor = 1.0f - ramp_func;
                freq = inv_freq_interpolation * (1.0f - inv_freq_extrapolation_factor)
                    + inv_freq_extrapolation * inv_freq_extrapolation_factor;
            }

            float pos_id = selectMRopePosId(
                position_ids, tokenIdx, num_tokens, half_dim, use_mrope, mrope_section1, mrope_section2);
            float theta = pos_id * freq;
            __sincosf(theta, &sin_vals[i], &cos_vals[i]);
        }
        // __shfl_xor_sync does not provide memfence. Need to sync again.
        __syncwarp();
    }

    bool const is_full_rope = (rotary_dim == head_dim);
    if (is_full_rope)
    {
        for (int i = 0; i < numElemsPerThread; i++)
        {
            elements[i] = (elements[i] * cos_vals[i] + elements2[i] * sin_vals[i]) * attention_factor;
        }
    }
    else
    {
        for (int i = 0; i < numElemsPerThread; i++)
        {
            int dim_idx = laneId * numElemsPerThread + i;

            if (dim_idx < rotary_dim)
            {
                elements[i] = (elements[i] * cos_vals[i] + elements2[i] * sin_vals[i]) * attention_factor;
            }
        }
    }

    // Store.
    storeHeadElements<OutT, numElemsPerThread, vecSize>(qkv_out, offsetThread, elements);
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

template <typename OutT>
static void launchFusedQKNormRopeImpl(__nv_bfloat16 const* qkv_in, OutT* qkv_out, bool const process_v,
    int const num_tokens, int const num_heads_q, int const num_heads_k, int const num_heads_v, int const head_dim,
    int const rotary_dim, float const eps, __nv_bfloat16 const* q_weight, __nv_bfloat16 const* k_weight,
    float const base, bool const interleave, int const* position_ids, float factor, float low, float high,
    float attention_factor, cudaStream_t stream, bool is_qk_norm, bool use_gemma, bool use_mrope, int mrope_section1,
    int mrope_section2)
{
    if (factor == 1.0f)
    {
        TLLM_CHECK(attention_factor == 1.0f);
    }

    TLLM_CHECK_WITH_INFO(rotary_dim > 0 && rotary_dim <= head_dim && rotary_dim % 2 == 0,
        "rotary_dim must be positive, even and no greater than head_dim (got rotary_dim=%d, head_dim=%d)", rotary_dim,
        head_dim);
    // Skipping V leaves the output's V slots untouched, which is only meaningful in place.
    TLLM_CHECK_WITH_INFO(process_v || static_cast<void const*>(qkv_in) == static_cast<void const*>(qkv_out),
        "process_v=false requires qkv_in and qkv_out to alias");
    if (!interleave)
    {
        // To allow warp-level pairing for partial rope
        TLLM_CHECK_WITH_INFO(
            (rotary_dim * 16) % head_dim == 0, "Unsupported rotary dimension for fusedQKNormRope: %d", rotary_dim);
    }

    constexpr int blockSize = 256;

    int const warpsPerBlock = blockSize / 32;
    int const totalProcHeads = num_heads_q + num_heads_k + (process_v ? num_heads_v : 0);
    int const totalWarps = num_tokens * totalProcHeads;

    int const gridSize = common::divUp(totalWarps, warpsPerBlock);
    dim3 gridDim(gridSize);
    dim3 blockDim(blockSize);

    // Head dimensions should be a multiple of 64
    // Add more cases as needed
    switch (head_dim)
    {
    case 64:
        DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
            fusedQKNormRopeKernel<64, INTERLEAVE, OutT><<<gridDim, blockDim, 0, stream>>>(qkv_in, qkv_out, num_heads_q,
                num_heads_k, num_heads_v, process_v, rotary_dim, eps, q_weight, k_weight, base, position_ids,
                num_tokens, factor, low, high, attention_factor, is_qk_norm, use_gemma, use_mrope, mrope_section1,
                mrope_section2);
        });
        break;
    case 128:
        DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
            fusedQKNormRopeKernel<128, INTERLEAVE, OutT><<<gridDim, blockDim, 0, stream>>>(qkv_in, qkv_out, num_heads_q,
                num_heads_k, num_heads_v, process_v, rotary_dim, eps, q_weight, k_weight, base, position_ids,
                num_tokens, factor, low, high, attention_factor, is_qk_norm, use_gemma, use_mrope, mrope_section1,
                mrope_section2);
        });
        break;
    case 256:
        DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
            fusedQKNormRopeKernel<256, INTERLEAVE, OutT><<<gridDim, blockDim, 0, stream>>>(qkv_in, qkv_out, num_heads_q,
                num_heads_k, num_heads_v, process_v, rotary_dim, eps, q_weight, k_weight, base, position_ids,
                num_tokens, factor, low, high, attention_factor, is_qk_norm, use_gemma, use_mrope, mrope_section1,
                mrope_section2);
        });
        break;
    default: TLLM_THROW("Unsupported head dimension for fusedQKNormRope: %d", head_dim);
    }
}

void launchFusedQKNormRope(void* qkv, int const num_tokens, int const num_heads_q, int const num_heads_k,
    int const num_heads_v, int const head_dim, int const rotary_dim, float const eps, void const* q_weight,
    void const* k_weight, float const base, bool const interleave, int const* position_ids, float factor, float low,
    float high, float attention_factor, cudaStream_t stream, bool is_qk_norm, bool use_gemma, bool use_mrope,
    int mrope_section1, int mrope_section2)
{
    launchFusedQKNormRopeImpl<__nv_bfloat16>(static_cast<__nv_bfloat16 const*>(qkv), static_cast<__nv_bfloat16*>(qkv),
        /*process_v=*/false, num_tokens, num_heads_q, num_heads_k, num_heads_v, head_dim, rotary_dim, eps,
        static_cast<__nv_bfloat16 const*>(q_weight), static_cast<__nv_bfloat16 const*>(k_weight), base, interleave,
        position_ids, factor, low, high, attention_factor, stream, is_qk_norm, use_gemma, use_mrope, mrope_section1,
        mrope_section2);
}

void launchFusedQKNormRopeToFp8(void const* qkv_in, void* qkv_out, int const num_tokens, int const num_heads_q,
    int const num_heads_k, int const num_heads_v, int const head_dim, int const rotary_dim, float const eps,
    void const* q_weight, void const* k_weight, float const base, bool const interleave, int const* position_ids,
    float factor, float low, float high, float attention_factor, cudaStream_t stream, bool is_qk_norm, bool use_gemma,
    bool use_mrope, int mrope_section1, int mrope_section2)
{
    // Out-of-place, so V has to be copy-cast rather than left untouched.
    launchFusedQKNormRopeImpl<__nv_fp8_e4m3>(static_cast<__nv_bfloat16 const*>(qkv_in),
        static_cast<__nv_fp8_e4m3*>(qkv_out), /*process_v=*/true, num_tokens, num_heads_q, num_heads_k, num_heads_v,
        head_dim, rotary_dim, eps, static_cast<__nv_bfloat16 const*>(q_weight),
        static_cast<__nv_bfloat16 const*>(k_weight), base, interleave, position_ids, factor, low, high,
        attention_factor, stream, is_qk_norm, use_gemma, use_mrope, mrope_section1, mrope_section2);
}
} // namespace kernels

TRTLLM_NAMESPACE_END
