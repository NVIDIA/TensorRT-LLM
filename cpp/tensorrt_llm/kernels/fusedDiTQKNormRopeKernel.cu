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

#include "fusedDiTQKNormRopeKernel.h"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/mathUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include <cuda_bf16.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Per-head QK Norm + RoPE kernel (FLUX, Cosmos3, UniVideo)
//
// Each warp processes one head of one token (Q or K only; V is untouched).
// Supports:
//   - Precomputed cos/sin embeddings
//   - Dual-stream attention (text vs image norm weights)
//   - Interleaved or rotate_half RoPE modes
//
template <int head_dim, bool interleave>
__global__ void fusedDiTQKNormRopeKernel(__nv_bfloat16* qkv, // [num_tokens, total_heads * head_dim]
    int const num_heads_q, int const num_heads_k, int const num_heads_v, float const eps,
    __nv_bfloat16 const* q_weight,                           // [head_dim]
    __nv_bfloat16 const* k_weight,                           // [head_dim]
    __nv_bfloat16 const* q_add_weight,                       // [head_dim] or nullptr
    __nv_bfloat16 const* k_add_weight,                       // [head_dim] or nullptr
    float const* cos_emb,                                    // [num_tokens, head_dim]
    float const* sin_emb,                                    // [num_tokens, head_dim]
    int const num_tokens, int const num_txt_tokens,
    int const tokens_per_batch)                              // seq_len per batch element; 0 = flat (no batching)
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

    // Compute offset into packed QKV tensor (use int64_t to avoid overflow
    // when num_tokens * num_heads * head_dim > INT_MAX, e.g. WAN I2V 14B)
    int64_t offsetWarp;
    if (isQ)
    {
        offsetWarp = static_cast<int64_t>(tokenIdx) * num_heads * head_dim + headIdx * head_dim;
    }
    else
    {
        offsetWarp
            = static_cast<int64_t>(tokenIdx) * num_heads * head_dim + num_heads_q * head_dim + headIdx * head_dim;
    }
    int64_t offsetThread = offsetWarp + laneId * numElemsPerThread;

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

    // Select norm weight: text tokens use add_weight (if provided), image tokens use primary weight.
    // For batched input (B*S flattened), use modulo to get local token index within each batch element.
    int const localTokenIdx = (tokens_per_batch > 0) ? (tokenIdx % tokens_per_batch) : tokenIdx;
    bool const useAddWeight = (num_txt_tokens > 0) && (localTokenIdx < num_txt_tokens);

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

    // ---- Step 3: Apply RoPE with precomputed cos/sin ----
    int64_t const embOffset = static_cast<int64_t>(tokenIdx) * head_dim;

    if constexpr (interleave)
    {
        // Interleaved pairing: (element[2i], element[2i+1])
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
    }
    else
    {
        // rotate_half pairing: element[i] pairs with element[i + D/2].
        // Each of the 32 lanes owns numElemsPerThread = D/32 consecutive elements,
        // so the partner element at offset D/2 lives in the lane that is
        // (D/2) / (D/32) = 16 lanes away. XOR with 16 swaps the two halves.
        __syncwarp();
        constexpr int pairOffset = 16;

        float partner[numElemsPerThread];
        for (int i = 0; i < numElemsPerThread; i++)
        {
            partner[i] = __shfl_xor_sync(0xffffffff, elements[i], pairOffset);
            // First half (laneId < pairOffset): rotate_half = [-partner, self]
            // result[i] = elements[i] * cos - partner[i] * sin
            if (laneId < pairOffset)
            {
                partner[i] = -partner[i];
            }
        }
        __syncwarp();

        for (int i = 0; i < numElemsPerThread; i++)
        {
            int dim = laneId * numElemsPerThread + i;
            float cos_val = cos_emb[embOffset + dim];
            float sin_val = sin_emb[embOffset + dim];
            elements[i] = elements[i] * cos_val + partner[i] * sin_val;
        }
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
    void const* k_add_weight, float const* cos_emb, float const* sin_emb, int num_txt_tokens, bool interleave,
    int tokens_per_batch, cudaStream_t stream)
{
    constexpr int blockSize = 256;

    int const warpsPerBlock = blockSize / 32;
    int const totalQKHeads = num_heads_q + num_heads_k;
    int const totalWarps = num_tokens * totalQKHeads;

    int const gridSize = common::divUp(totalWarps, warpsPerBlock);
    dim3 gridDim(gridSize);
    dim3 blockDim(blockSize);

#define LAUNCH_PER_HEAD_KERNEL(HEAD_DIM, INTERLEAVE)                                                                   \
    fusedDiTQKNormRopeKernel<HEAD_DIM, INTERLEAVE><<<gridDim, blockDim, 0, stream>>>(                                  \
        reinterpret_cast<__nv_bfloat16*>(qkv), num_heads_q, num_heads_k, num_heads_v, eps,                             \
        reinterpret_cast<__nv_bfloat16 const*>(q_weight), reinterpret_cast<__nv_bfloat16 const*>(k_weight),            \
        reinterpret_cast<__nv_bfloat16 const*>(q_add_weight), reinterpret_cast<__nv_bfloat16 const*>(k_add_weight),    \
        cos_emb, sin_emb, num_tokens, num_txt_tokens, tokens_per_batch)

    if (interleave)
    {
        switch (head_dim)
        {
        case 64: LAUNCH_PER_HEAD_KERNEL(64, true); break;
        case 128: LAUNCH_PER_HEAD_KERNEL(128, true); break;
        case 256: LAUNCH_PER_HEAD_KERNEL(256, true); break;
        default: TLLM_THROW("Unsupported head dimension for fusedDiTQKNormRope: %d", head_dim);
        }
    }
    else
    {
        switch (head_dim)
        {
        case 64: LAUNCH_PER_HEAD_KERNEL(64, false); break;
        case 128: LAUNCH_PER_HEAD_KERNEL(128, false); break;
        case 256: LAUNCH_PER_HEAD_KERNEL(256, false); break;
        default: TLLM_THROW("Unsupported head dimension for fusedDiTQKNormRope: %d", head_dim);
        }
    }
#undef LAUNCH_PER_HEAD_KERNEL
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Cross-head QK Norm + RoPE kernel (WAN, LTX-2)
//
// One CTA per (token, Q-or-K). RMSNorm is computed across ALL heads combined
// (norm dim = num_heads * head_dim), then RoPE is applied in-register.
//
// Grid:  (num_tokens, 2)   — blockIdx.y==0 → Q,  blockIdx.y==1 → K
// Block: up to 1024 threads; each thread handles multiple bf16x2 pairs
//        at stride blockDim.x.
//
// The packed QKV layout is [num_tokens, (Hq + Hk + Hv) * head_dim].
// Only Q and K regions are read/written; V is untouched.
//
template <bool interleave>
__global__ void fusedDiTCrossHeadQKNormRopeKernel(__nv_bfloat16* qkv, // [num_tokens, (Hq+Hk+Hv)*head_dim], in-place
    int const q_dim,                                                  // num_heads_q * head_dim
    int const k_dim,                                                  // num_heads_k * head_dim
    int const total_row,                                              // (Hq+Hk+Hv) * head_dim
    int const head_dim, float const eps,
    __nv_bfloat16 const* q_weight,                                    // [q_dim]
    __nv_bfloat16 const* k_weight,                                    // [k_dim]
    float const* cos_emb,                                             // [num_tokens, head_dim]
    float const* sin_emb,                                             // [num_tokens, head_dim]
    int const num_tokens)
{
    int const tokenIdx = blockIdx.x;
    if (tokenIdx >= num_tokens)
    {
        return;
    }
    bool const isQ = (blockIdx.y == 0);
    int const norm_dim = isQ ? q_dim : k_dim;
    __nv_bfloat16 const* weight = isQ ? q_weight : k_weight;

    // Base offset into packed QKV for this token's Q or K region
    int64_t const baseOffset = static_cast<int64_t>(tokenIdx) * total_row + (isQ ? 0 : q_dim);

    // ---- Step 1: Load elements and accumulate sum-of-squares ----
    int const halfDim = norm_dim / 2; // number of bf16x2 pairs
    float threadSumSq = 0.0f;

    for (int i = threadIdx.x; i < halfDim; i += blockDim.x)
    {
        int const elemIdx = i * 2;
        __nv_bfloat162 val = *reinterpret_cast<__nv_bfloat162 const*>(&qkv[baseOffset + elemIdx]);
        float2 valf = __bfloat1622float2(val);
        threadSumSq += valf.x * valf.x + valf.y * valf.y;
    }

    // ---- Step 2: CTA-level reduction for sum-of-squares ----
    float totalSumSq = tensorrt_llm::common::blockReduceSum(threadSumSq);

    __shared__ float s_rms_rcp;
    if (threadIdx.x == 0)
    {
        s_rms_rcp = rsqrtf(totalSumSq / static_cast<float>(norm_dim) + eps);
    }
    __syncthreads();
    float const rms_rcp = s_rms_rcp;

    // ---- Step 3: Apply norm weight and interleaved RoPE, then store ----
    // cos/sin embeddings are [num_tokens, head_dim] and broadcast across heads.
    int64_t const embBase = static_cast<int64_t>(tokenIdx) * head_dim;

    // Reload from global memory rather than caching in registers: norm_dim can be
    // very large (e.g. 40 heads * 128 = 5120), so hoarding per-thread register arrays
    // would tank occupancy.  The second load likely hits L1/L2 cache from Step 1.
    if constexpr (interleave)
    {
        for (int i = threadIdx.x; i < halfDim; i += blockDim.x)
        {
            int const elemIdx = i * 2;
            //
            __nv_bfloat162 val = *reinterpret_cast<__nv_bfloat162 const*>(&qkv[baseOffset + elemIdx]);
            __nv_bfloat162 w = *reinterpret_cast<__nv_bfloat162 const*>(&weight[elemIdx]);
            float2 valf = __bfloat1622float2(val);
            float2 wf = __bfloat1622float2(w);

            float x0 = valf.x * rms_rcp * wf.x;
            float x1 = valf.y * rms_rcp * wf.y;

            // Interleaved RoPE: pair (x[2j], x[2j+1]) within each head
            int const dimInHead0 = elemIdx % head_dim;
            int const dimInHead1 = dimInHead0 + 1;

            float cos0 = cos_emb[embBase + dimInHead0];
            float sin0 = sin_emb[embBase + dimInHead0];
            float cos1 = cos_emb[embBase + dimInHead1];
            float sin1 = sin_emb[embBase + dimInHead1];

            float out0 = x0 * cos0 - x1 * sin0;
            float out1 = x1 * cos1 + x0 * sin1;

            *reinterpret_cast<__nv_bfloat162*>(&qkv[baseOffset + elemIdx])
                = __float22bfloat162_rn(make_float2(out0, out1));
        }
    }
    else
    {
        // rotate_half requires a two-pass approach: within the same head, element i
        // is paired with element i + head_dim/2, which is processed by a different
        // thread.  We stage normalized values in shared memory so that Pass 2 can
        // read partners without racing against other threads' RoPE-phase writes to
        // global memory.
        //
        // Layout: s_norm[0..norm_dim) holds the post-normalize (pre-RoPE) values.
        // Allocated as dynamic shared memory sized = norm_dim * sizeof(bf16).
        extern __shared__ __nv_bfloat16 s_norm[];

        // Pass 1: normalize all elements and store them to shared memory.
        for (int i = threadIdx.x; i < halfDim; i += blockDim.x)
        {
            int const elemIdx = i * 2;
            __nv_bfloat162 val = *reinterpret_cast<__nv_bfloat162 const*>(&qkv[baseOffset + elemIdx]);
            __nv_bfloat162 w = *reinterpret_cast<__nv_bfloat162 const*>(&weight[elemIdx]);
            float2 valf = __bfloat1622float2(val);
            float2 wf = __bfloat1622float2(w);

            float x0 = valf.x * rms_rcp * wf.x;
            float x1 = valf.y * rms_rcp * wf.y;

            *reinterpret_cast<__nv_bfloat162*>(&s_norm[elemIdx]) = __float22bfloat162_rn(make_float2(x0, x1));
        }
        __syncthreads();

        // Pass 2: apply rotate_half RoPE. Self + partner are both read from
        // s_norm (read-only in this pass), and results are written to qkv in
        // global memory. Since different threads write disjoint (elemIdx,
        // elemIdx+1) pairs, and partner reads come from shared memory which is
        // never written during Pass 2, there is no read/write race.
        int const halfHead = head_dim / 2;
        for (int i = threadIdx.x; i < halfDim; i += blockDim.x)
        {
            int const elemIdx = i * 2;

            __nv_bfloat162 selfVal = *reinterpret_cast<__nv_bfloat162 const*>(&s_norm[elemIdx]);
            float2 selfF = __bfloat1622float2(selfVal);
            float x0 = selfF.x;
            float x1 = selfF.y;

            int const localDim0 = elemIdx % head_dim;
            int const localDim1 = localDim0 + 1;
            // elemIdx is always even (i*2) and head_dim is always even (it must be
            // a power of 2: 64/128/256 for supported models), so head boundaries
            // fall on even indices.  Therefore elemIdx and elemIdx+1 always belong
            // to the same head and share the same head offset.
            int const headOff = elemIdx - localDim0;

            // Partner is head_dim/2 away within the same head
            int pDim0 = (localDim0 < halfHead) ? (headOff + localDim0 + halfHead) : (headOff + localDim0 - halfHead);

            // pDim0 is always even: localDim0 is even (elemIdx = i*2), halfHead is
            // even (head_dim is a power of 2), and headOff is a multiple of
            // head_dim.  So we can load the (pDim0, pDim0+1) partner pair as a
            // single bf16x2 vector load from shared memory.
            __nv_bfloat162 partnerVal = *reinterpret_cast<__nv_bfloat162 const*>(&s_norm[pDim0]);
            float2 partnerF = __bfloat1622float2(partnerVal);
            float p0 = partnerF.x;
            float p1 = partnerF.y;

            // First half: result = self * cos - partner * sin
            // Second half: result = self * cos + partner * sin
            // A single sign suffices for both elements: localDim0 is always even
            // (elemIdx = i*2) and halfHead is always even (head_dim is power of 2),
            // so the half-boundary (halfHead-1 / halfHead = odd/even) can never
            // fall between localDim0 and localDim1.  Both are always in the same half.
            float sign = (localDim0 < halfHead) ? -1.0f : 1.0f;

            float cos0 = cos_emb[embBase + localDim0];
            float sin0 = sin_emb[embBase + localDim0];
            float cos1 = cos_emb[embBase + localDim1];
            float sin1 = sin_emb[embBase + localDim1];

            float out0 = x0 * cos0 + sign * p0 * sin0;
            float out1 = x1 * cos1 + sign * p1 * sin1;

            *reinterpret_cast<__nv_bfloat162*>(&qkv[baseOffset + elemIdx])
                = __float22bfloat162_rn(make_float2(out0, out1));
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void launchFusedDiTCrossHeadQKNormRope(void* qkv, int num_tokens, int num_heads_q, int num_heads_k, int num_heads_v,
    int head_dim, float eps, void const* q_weight, void const* k_weight, float const* cos_emb, float const* sin_emb,
    bool interleave, cudaStream_t stream)
{
    int const q_dim = num_heads_q * head_dim;
    int const k_dim = num_heads_k * head_dim;
    int const total_row = (num_heads_q + num_heads_k + num_heads_v) * head_dim;
    int const max_dim = (q_dim > k_dim) ? q_dim : k_dim;

    // Block size: enough threads to cover max_dim/2 bf16x2 pairs, capped at 1024
    int const halfDim = max_dim / 2;
    int blockSize = 256;
    if (halfDim > 256)
    {
        blockSize = 512;
    }
    if (halfDim > 512)
    {
        blockSize = 1024;
    }

    dim3 grid(num_tokens, 2); // y=0 → Q, y=1 → K
    dim3 block(blockSize);

    // rotate_half requires shared memory to stage normalized values (avoids a
    // read/write race on partner elements in global memory); the interleaved path
    // computes each output pair from the thread's own values and needs no smem.
    size_t const smem_bytes = interleave ? 0 : static_cast<size_t>(max_dim) * sizeof(__nv_bfloat16);

#define LAUNCH_CROSS_HEAD_KERNEL(INTERLEAVE)                                                                           \
    fusedDiTCrossHeadQKNormRopeKernel<INTERLEAVE>                                                                      \
        <<<grid, block, smem_bytes, stream>>>(reinterpret_cast<__nv_bfloat16*>(qkv), q_dim, k_dim, total_row,          \
            head_dim, eps, reinterpret_cast<__nv_bfloat16 const*>(q_weight),                                           \
            reinterpret_cast<__nv_bfloat16 const*>(k_weight), cos_emb, sin_emb, num_tokens)

    if (interleave)
    {
        LAUNCH_CROSS_HEAD_KERNEL(true);
    }
    else
    {
        LAUNCH_CROSS_HEAD_KERNEL(false);
    }
#undef LAUNCH_CROSS_HEAD_KERNEL
}

} // namespace kernels

TRTLLM_NAMESPACE_END
