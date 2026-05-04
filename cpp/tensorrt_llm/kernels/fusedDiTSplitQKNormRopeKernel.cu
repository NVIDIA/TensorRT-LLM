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
#include <type_traits>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Fused full-dim RMSNorm + RoPE on a SINGLE Q or K tensor (DiT, e.g. LTX-2 cross-attn).
// Layout: block=256 (8 warps), grid=(num_tokens,). Chunked reduce loop holds
// only one chunk's data in registers at a time → ~32 regs/thread, 8 blocks/SM,
// ~94% warps_active. Mirror of fusedDiTQKNormFullDimRopeKernel but for one tensor.
//
// PER_HEAD_COS=false: cos/sin shape [num_tokens, HEAD_DIM] (FLUX-style, head broadcast).
// PER_HEAD_COS=true:  cos/sin shape [num_tokens, num_heads * HEAD_DIM] (LTX-2 3D RoPE).
// CosT: float (fp32 cos) or __nv_bfloat16 (B-2: kernel upcasts in registers).
template <int HEAD_DIM, bool INTERLEAVE, bool PER_HEAD_COS, typename CosT>
__global__ void fusedDiTSplitNormFullDimRopeKernel(__nv_bfloat16* __restrict__ tensor, int const num_tokens,
    int const num_heads, float const eps, __nv_bfloat16 const* __restrict__ weight, CosT const* __restrict__ cos_emb,
    CosT const* __restrict__ sin_emb)
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
        static_assert(CHUNK_ELEMS == 8, "CHUNK_ELEMS=8 required for vectorized cos/sin load");
        if constexpr (std::is_same_v<CosT, float>)
        {
            // fp32 cos: 8 scalar loads → 2 LDG.128 per array (float4 × 2).
            float4 const* cos_v = reinterpret_cast<float4 const*>(&cos_emb[embBase + cosHeadOff + baseDim]);
            float4 const* sin_v = reinterpret_cast<float4 const*>(&sin_emb[embBase + cosHeadOff + baseDim]);
            float4 c0 = cos_v[0], c1 = cos_v[1];
            float4 s0 = sin_v[0], s1 = sin_v[1];
            cos_vals[0] = c0.x;
            cos_vals[1] = c0.y;
            cos_vals[2] = c0.z;
            cos_vals[3] = c0.w;
            cos_vals[4] = c1.x;
            cos_vals[5] = c1.y;
            cos_vals[6] = c1.z;
            cos_vals[7] = c1.w;
            sin_vals[0] = s0.x;
            sin_vals[1] = s0.y;
            sin_vals[2] = s0.z;
            sin_vals[3] = s0.w;
            sin_vals[4] = s1.x;
            sin_vals[5] = s1.y;
            sin_vals[6] = s1.z;
            sin_vals[7] = s1.w;
        }
        else
        {
            // bf16 cos: 8 bf16 = 16 bytes → 1 LDG.128 (uint4) per array (halves cos/sin
            // memory traffic vs fp32). Upcast in registers via __bfloat1622float2.
            uint4 const cos_packed = *reinterpret_cast<uint4 const*>(&cos_emb[embBase + cosHeadOff + baseDim]);
            uint4 const sin_packed = *reinterpret_cast<uint4 const*>(&sin_emb[embBase + cosHeadOff + baseDim]);
            uint const* cos_uints = reinterpret_cast<uint const*>(&cos_packed);
            uint const* sin_uints = reinterpret_cast<uint const*>(&sin_packed);
#pragma unroll
            for (int i = 0; i < 4; i++)
            {
                float2 cv = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&cos_uints[i]));
                float2 sv = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&sin_uints[i]));
                cos_vals[2 * i] = cv.x;
                cos_vals[2 * i + 1] = cv.y;
                sin_vals[2 * i] = sv.x;
                sin_vals[2 * i + 1] = sv.y;
            }
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
        else
        {
            // rotate-half: partner element at +HEAD_DIM/2 within the same head.
            // Inline partner exchange (single reg `p` per iter, no array).
            constexpr int xor_mask = HEAD_DIM / 16;
            bool const negate = ((laneId & xor_mask) == 0);
#pragma unroll
            for (int i = 0; i < CHUNK_ELEMS; i++)
            {
                float p = __shfl_xor_sync(0xffffffff, elements[i], xor_mask);
                if (negate)
                {
                    p = -p;
                }
                elements[i] = elements[i] * cos_vals[i] + p * sin_vals[i];
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

void launchFusedDiTSplitNormFullDimRope(void* tensor, int num_tokens, int num_heads, int head_dim, float eps,
    void const* weight, void const* cos_emb, void const* sin_emb, bool interleave, bool per_head_cos, bool cos_is_bf16,
    cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(num_heads <= 32,
        "fusedDiTSplitNormFullDimRope: num_heads (%d) must be <= 32 (block_size = num_heads*32 <= 1024)", num_heads);
    TLLM_CHECK_WITH_INFO(num_heads >= 1, "fusedDiTSplitNormFullDimRope: num_heads must be >= 1, got %d", num_heads);

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
#define LAUNCH(HEAD_DIM, INTERLEAVE, PER_HEAD, COS_T)                                                                  \
    cudaLaunchKernelEx(&cfg, fusedDiTSplitNormFullDimRopeKernel<HEAD_DIM, INTERLEAVE, PER_HEAD, COS_T>,                \
        reinterpret_cast<__nv_bfloat16*>(tensor), num_tokens, num_heads, eps,                                          \
        reinterpret_cast<__nv_bfloat16 const*>(weight), reinterpret_cast<COS_T const*>(cos_emb),                       \
        reinterpret_cast<COS_T const*>(sin_emb))
#define DISPATCH(INTERLEAVE, PER_HEAD, COS_T)                                                                          \
    do                                                                                                                 \
    {                                                                                                                  \
        switch (head_dim)                                                                                              \
        {                                                                                                              \
        case 64: LAUNCH(64, INTERLEAVE, PER_HEAD, COS_T); break;                                                       \
        case 128: LAUNCH(128, INTERLEAVE, PER_HEAD, COS_T); break;                                                     \
        default: TLLM_THROW("Unsupported head_dim for fusedDiTSplitNormFullDimRope: %d (only 64, 128)", head_dim);     \
        }                                                                                                              \
    } while (0)
#define DISPATCH_DTYPE(INTERLEAVE, PER_HEAD)                                                                           \
    do                                                                                                                 \
    {                                                                                                                  \
        if (cos_is_bf16)                                                                                               \
            DISPATCH(INTERLEAVE, PER_HEAD, __nv_bfloat16);                                                             \
        else                                                                                                           \
            DISPATCH(INTERLEAVE, PER_HEAD, float);                                                                     \
    } while (0)
    if (interleave)
    {
        if (per_head_cos)
            DISPATCH_DTYPE(true, true);
        else
            DISPATCH_DTYPE(true, false);
    }
    else
    {
        if (per_head_cos)
            DISPATCH_DTYPE(false, true);
        else
            DISPATCH_DTYPE(false, false);
    }
#undef DISPATCH_DTYPE
#undef DISPATCH
#undef LAUNCH
}

} // namespace kernels

TRTLLM_NAMESPACE_END
