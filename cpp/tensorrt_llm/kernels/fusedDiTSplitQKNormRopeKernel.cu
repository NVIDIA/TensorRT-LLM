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

#include "fusedDiTSplitQKNormRopeKernel.h"
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
// Split single-tensor RMSNorm + RoPE for DiT (LTX-2).
//
// Layout: 1 block = 1 (token), block_size = num_heads * 32, 1 warp = 1 head.
// Each warp does its own warp-level reduce. For full_dim_norm, partial sums
// are written to shared memory and a single warp does inter-warp reduce +
// broadcasts the rms_rcp back via shared memory.
//
template <int HEAD_DIM, bool INTERLEAVE, bool FULL_DIM_NORM, bool DO_NORM>
__global__ void fusedDiTSplitNormRopeKernel(__nv_bfloat16* tensor, int const num_tokens, int const num_heads,
    float const eps, __nv_bfloat16 const* weight, float const* cos_emb, float const* sin_emb)
{
    int const tokenIdx = blockIdx.x;
    int const warpId = threadIdx.x >> 5;
    int const laneId = threadIdx.x & 31;

    if (tokenIdx >= num_tokens)
    {
        return;
    }

    int const headIdx = warpId; // 1 warp == 1 head

    static_assert(HEAD_DIM % (32 * 2) == 0, "head_dim must be divisible by 64");
    constexpr int numElemsPerThread = HEAD_DIM / 32;
    float elements[numElemsPerThread];
    constexpr int elemSizeBytes = numElemsPerThread * sizeof(__nv_bfloat16);
    static_assert(elemSizeBytes % 4 == 0, "elemSizeBytes must be a multiple of 4");
    constexpr int vecSize = elemSizeBytes / 4;
    using vec_T = typename tensorrt_llm::common::packed_as<uint, vecSize>::type;

    // Contiguous SEPARATE_QKV input: row_stride = num_heads * head_dim.
    int64_t const offsetWarp = static_cast<int64_t>(tokenIdx) * num_heads * HEAD_DIM + headIdx * HEAD_DIM;
    int64_t const offsetThread = offsetWarp + laneId * numElemsPerThread;

    // ---- Step 1: Load + sum_of_squares ----
    float sumOfSquares = 0.0f;
    {
        vec_T vec = *reinterpret_cast<vec_T const*>(&tensor[offsetThread]);
        for (int i = 0; i < vecSize; i++)
        {
            float2 vals = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<uint*>(&vec) + i));
            sumOfSquares += vals.x * vals.x;
            sumOfSquares += vals.y * vals.y;
            elements[2 * i] = vals.x;
            elements[2 * i + 1] = vals.y;
        }
    }

    // ---- Step 2: RMS normalization ----
    // Shared memory used by the full-dim path; declared at function scope so
    // its lifetime is unambiguous regardless of constexpr branching.
    __shared__ float warp_sums[32];
    __shared__ float shared_rms_rcp;

    if constexpr (DO_NORM)
    {
        float rms_rcp;

        if constexpr (FULL_DIM_NORM)
        {
            // 1) Warp-level reduce (per head)
            sumOfSquares = tensorrt_llm::common::warpReduceSum(sumOfSquares);
            // 2) Cross-warp reduce via shared memory
            if (laneId == 0)
            {
                warp_sums[warpId] = sumOfSquares;
            }
            __syncthreads();
            if (warpId == 0)
            {
                float v = (laneId < num_heads) ? warp_sums[laneId] : 0.f;
                v = tensorrt_llm::common::warpReduceSum(v);
                if (laneId == 0)
                {
                    float const norm_dim = static_cast<float>(num_heads) * static_cast<float>(HEAD_DIM);
                    shared_rms_rcp = rsqrtf(v / norm_dim + eps);
                }
            }
            __syncthreads();
            rms_rcp = shared_rms_rcp;
        }
        else
        {
            // Per-head: just warp reduce
            sumOfSquares = tensorrt_llm::common::warpReduceSum(sumOfSquares);
            rms_rcp = rsqrtf(sumOfSquares / static_cast<float>(HEAD_DIM) + eps);
        }

        // Apply weight + scale
        for (int i = 0; i < numElemsPerThread; i++)
        {
            int const dim = laneId * numElemsPerThread + i;
            int const weight_offset = FULL_DIM_NORM ? (headIdx * HEAD_DIM + dim) : dim;
            float const w = __bfloat162float(weight[weight_offset]);
            elements[i] *= rms_rcp * w;
        }
    }

    // ---- Step 3: Apply RoPE ----
    int64_t const embOffset = static_cast<int64_t>(tokenIdx) * HEAD_DIM;

    if constexpr (INTERLEAVE)
    {
        // Pair (2i, 2i+1) — neighbor pairing (LTX-2 INTERLEAVED mode)
        for (int i = 0; i < numElemsPerThread; i += 2)
        {
            int const dim = laneId * numElemsPerThread + i;
            float const cos0 = cos_emb[embOffset + dim];
            float const sin0 = sin_emb[embOffset + dim];
            float const cos1 = cos_emb[embOffset + dim + 1];
            float const sin1 = sin_emb[embOffset + dim + 1];

            float const x = elements[i];
            float const y = elements[i + 1];

            elements[i] = x * cos0 + (-y) * sin0;
            elements[i + 1] = y * cos1 + x * sin1;
        }
    }
    else
    {
        // rotate_half via __shfl_xor_sync(pairOffset=16): each lane's partner
        // element at offset HEAD_DIM/2 lives in (HEAD_DIM/2)/numElemsPerThread = 16 lanes away.
        __syncwarp();
        constexpr int pairOffset = 16;

        float partner[numElemsPerThread];
        for (int i = 0; i < numElemsPerThread; i++)
        {
            partner[i] = __shfl_xor_sync(0xffffffff, elements[i], pairOffset);
            if (laneId < pairOffset)
            {
                partner[i] = -partner[i];
            }
        }
        __syncwarp();

        for (int i = 0; i < numElemsPerThread; i++)
        {
            int const dim = laneId * numElemsPerThread + i;
            float const cos_val = cos_emb[embOffset + dim];
            float const sin_val = sin_emb[embOffset + dim];
            elements[i] = elements[i] * cos_val + partner[i] * sin_val;
        }
    }

    // ---- Step 4: Store ----
    {
        vec_T vec;
        for (int i = 0; i < vecSize; i++)
        {
            __nv_bfloat162 vals = __float22bfloat162_rn(make_float2(elements[2 * i], elements[2 * i + 1]));
            reinterpret_cast<__nv_bfloat162&>(*(reinterpret_cast<uint*>(&vec) + i)) = vals;
        }
        vec_T* outputPtr = reinterpret_cast<vec_T*>(&tensor[offsetThread]);
        *outputPtr = vec;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Inductor-style full-dim split kernel (LTX-2 INTERLEAVE only):
// block=256 (8 warps), grid=(num_tokens,). One block per token.
// Chunked reduce loop holds only one chunk's data in registers at a time
// → ~32 regs/thread, 8 blocks/SM, ~94% warps_active.
// Mirror of fusedDiTQKNormRopeFullDimKernelInductorLike but for one tensor (Q xor K).

// PER_HEAD_COS=false: cos/sin shape [num_tokens, HEAD_DIM] (FLUX-style, head broadcast).
// PER_HEAD_COS=true:  cos/sin shape [num_tokens, num_heads * HEAD_DIM] (LTX-2 3D RoPE).
template <int HEAD_DIM, bool INTERLEAVE, bool PER_HEAD_COS>
__global__ void fusedDiTSplitNormRopeKernelInductorLike(__nv_bfloat16* tensor, int const num_tokens,
    int const num_heads, float const eps, __nv_bfloat16 const* weight, float const* cos_emb, float const* sin_emb)
{
    constexpr int BLOCK_SIZE = 256;
    constexpr int N_WARPS = BLOCK_SIZE / 32;                    // 8
    constexpr int CHUNK_ELEMS = 8;                              // 1 uint4 load = 8 bf16
    constexpr int CHUNK_BLOCK_ELEMS = BLOCK_SIZE * CHUNK_ELEMS; // 2048

#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;");
#endif

    int const tid = threadIdx.x;
    int const warpId = tid >> 5;
    int const laneId = tid & 31;
    int const tokenIdx = blockIdx.x;
    if (tokenIdx >= num_tokens)
        return;

    int const N = num_heads * HEAD_DIM;
    int const num_chunks = (N + CHUNK_BLOCK_ELEMS - 1) / CHUNK_BLOCK_ELEMS;

    int64_t const tokenBase = static_cast<int64_t>(tokenIdx) * N;
    // cos/sin per-token base; PER_HEAD_COS adds head_idx * HEAD_DIM in the load.
    int64_t const embBase = PER_HEAD_COS ? static_cast<int64_t>(tokenIdx) * num_heads * HEAD_DIM
                                         : static_cast<int64_t>(tokenIdx) * HEAD_DIM;

    __shared__ float warp_sums[32];

    // Phase 1: sum²
    float sum2 = 0.0f;
    for (int chunk = 0; chunk < num_chunks; chunk++)
    {
        int const elemBase = chunk * CHUNK_BLOCK_ELEMS + tid * CHUNK_ELEMS;
        if (elemBase >= N)
            continue;
        uint4 const v = *reinterpret_cast<uint4 const*>(&tensor[tokenBase + elemBase]);
        uint const* uints = reinterpret_cast<uint const*>(&v);
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            float2 vals = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&uints[i]));
            sum2 += vals.x * vals.x + vals.y * vals.y;
        }
    }

    // 1-sync replicated cross-warp reduce.
    sum2 = tensorrt_llm::common::warpReduceSum(sum2);
    if (laneId == 0)
        warp_sums[warpId] = sum2;
    __syncthreads();
    float total = 0.0f;
#pragma unroll
    for (int w = 0; w < N_WARPS; w++)
        total += warp_sums[w];
    float const rms_rcp = rsqrtf(total / static_cast<float>(N) + eps);

    // Phase 2: re-read input + apply scale + RoPE + store.
    for (int chunk = 0; chunk < num_chunks; chunk++)
    {
        int const elemBase = chunk * CHUNK_BLOCK_ELEMS + tid * CHUNK_ELEMS;
        if (elemBase >= N)
            continue;
        int const headIdx = elemBase / HEAD_DIM;
        int const baseDim = elemBase - headIdx * HEAD_DIM;

        uint4 in_vec = *reinterpret_cast<uint4 const*>(&tensor[tokenBase + elemBase]);
        float w_vals[CHUNK_ELEMS], cos_vals[CHUNK_ELEMS], sin_vals[CHUNK_ELEMS];
#pragma unroll
        for (int i = 0; i < CHUNK_ELEMS; i++)
            w_vals[i] = __bfloat162float(weight[headIdx * HEAD_DIM + baseDim + i]);
        int const cosHeadOff = PER_HEAD_COS ? headIdx * HEAD_DIM : 0;
#pragma unroll
        for (int i = 0; i < CHUNK_ELEMS; i++)
        {
            cos_vals[i] = cos_emb[embBase + cosHeadOff + baseDim + i];
            sin_vals[i] = sin_emb[embBase + cosHeadOff + baseDim + i];
        }

        float elements[CHUNK_ELEMS];
        uint const* uints = reinterpret_cast<uint const*>(&in_vec);
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            float2 vals = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&uints[i]));
            elements[2 * i] = vals.x;
            elements[2 * i + 1] = vals.y;
        }

#pragma unroll
        for (int i = 0; i < CHUNK_ELEMS; i++)
            elements[i] *= rms_rcp * w_vals[i];

        if constexpr (INTERLEAVE)
        {
#pragma unroll
            for (int i = 0; i < CHUNK_ELEMS; i += 2)
            {
                float const x = elements[i], y = elements[i + 1];
                elements[i] = x * cos_vals[i] + (-y) * sin_vals[i];
                elements[i + 1] = y * cos_vals[i + 1] + x * sin_vals[i + 1];
            }
        }

        uint4 out_vec;
        uint* o_uints = reinterpret_cast<uint*>(&out_vec);
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            __nv_bfloat162 vals = __float22bfloat162_rn(make_float2(elements[2 * i], elements[2 * i + 1]));
            reinterpret_cast<__nv_bfloat162&>(o_uints[i]) = vals;
        }
        *reinterpret_cast<uint4*>(&tensor[tokenBase + elemBase]) = out_vec;
    }

#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void launchFusedDiTSplitNormRope(void* tensor, int num_tokens, int num_heads, int head_dim, float eps,
    void const* weight, float const* cos_emb, float const* sin_emb, bool full_dim_norm, bool do_norm, bool interleave,
    bool per_head_cos, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(num_heads <= 32,
        "fusedDiTSplitNormRope: num_heads (%d) must be <= 32 (block_size = num_heads*32 <= 1024)", num_heads);
    TLLM_CHECK_WITH_INFO(num_heads >= 1, "fusedDiTSplitNormRope: num_heads must be >= 1, got %d", num_heads);

    // do_norm=false (K-skip-norm for AV cross-attn) is Phase 2 work.
    TLLM_CHECK_WITH_INFO(do_norm, "fusedDiTSplitNormRope: do_norm=false not yet supported");

    // Dispatch:
    //   full_dim_norm + interleave  → InductorLike (block=256, chunked, Q-or-K)
    //                                  templated on per_head_cos.
    //   full_dim_norm + rotate_half → 32-warp template (cross-warp shfl partner)
    //   per_head     (any RoPE)     → 32-warp template, warp-only reduce.
    if (full_dim_norm && interleave)
    {
        dim3 const gridDim(num_tokens);
        dim3 const blockDim(256);
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = gridDim;
        cfg.blockDim = blockDim;
        cfg.dynamicSmemBytes = 0;
        cfg.stream = stream;
        cudaLaunchAttribute attrs[1] = {};
        attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed = 1;
        cfg.attrs = attrs;
        cfg.numAttrs = 1;
#define LAUNCH_INDUCTOR_LIKE(HEAD_DIM, PER_HEAD)                                                                       \
    cudaLaunchKernelEx(&cfg, fusedDiTSplitNormRopeKernelInductorLike<HEAD_DIM, true, PER_HEAD>,                        \
        reinterpret_cast<__nv_bfloat16*>(tensor), num_tokens, num_heads, eps,                                          \
        reinterpret_cast<__nv_bfloat16 const*>(weight), cos_emb, sin_emb)
        if (per_head_cos)
        {
            switch (head_dim)
            {
            case 64: LAUNCH_INDUCTOR_LIKE(64, true); break;
            case 128: LAUNCH_INDUCTOR_LIKE(128, true); break;
            default: TLLM_THROW("Unsupported head_dim for fusedDiTSplitNormRope: %d (only 64, 128)", head_dim);
            }
        }
        else
        {
            switch (head_dim)
            {
            case 64: LAUNCH_INDUCTOR_LIKE(64, false); break;
            case 128: LAUNCH_INDUCTOR_LIKE(128, false); break;
            default: TLLM_THROW("Unsupported head_dim for fusedDiTSplitNormRope: %d (only 64, 128)", head_dim);
            }
        }
#undef LAUNCH_INDUCTOR_LIKE
    }
    else
    {
        int const blockSize = num_heads * 32;
        dim3 const gridDim(num_tokens);
        dim3 const blockDim(blockSize);
#define LAUNCH_GENERIC(HEAD_DIM, INTERLEAVE, FULL_DIM)                                                                 \
    fusedDiTSplitNormRopeKernel<HEAD_DIM, INTERLEAVE, FULL_DIM, true>                                                  \
        <<<gridDim, blockDim, 0, stream>>>(reinterpret_cast<__nv_bfloat16*>(tensor), num_tokens, num_heads, eps,       \
            reinterpret_cast<__nv_bfloat16 const*>(weight), cos_emb, sin_emb)
        if (interleave && !full_dim_norm)
        {
            switch (head_dim)
            {
            case 64: LAUNCH_GENERIC(64, true, false); break;
            case 128: LAUNCH_GENERIC(128, true, false); break;
            default: TLLM_THROW("Unsupported head_dim for fusedDiTSplitNormRope: %d (only 64, 128)", head_dim);
            }
        }
        else if (!interleave && full_dim_norm)
        {
            switch (head_dim)
            {
            case 64: LAUNCH_GENERIC(64, false, true); break;
            case 128: LAUNCH_GENERIC(128, false, true); break;
            default: TLLM_THROW("Unsupported head_dim for fusedDiTSplitNormRope: %d (only 64, 128)", head_dim);
            }
        }
        else // !interleave && !full_dim_norm
        {
            switch (head_dim)
            {
            case 64: LAUNCH_GENERIC(64, false, false); break;
            case 128: LAUNCH_GENERIC(128, false, false); break;
            default: TLLM_THROW("Unsupported head_dim for fusedDiTSplitNormRope: %d (only 64, 128)", head_dim);
            }
        }
#undef LAUNCH_GENERIC
    }
}

} // namespace kernels

TRTLLM_NAMESPACE_END
