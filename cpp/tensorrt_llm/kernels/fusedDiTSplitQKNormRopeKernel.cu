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
#include <cuda_pipeline.h>
#include <cuda_runtime.h>
#include <type_traits>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

// Fused full-dim RMSNorm + RoPE on a SINGLE Q or K tensor (LTX-2 SEPARATE_QKV cross-attn).
// Strategy:
//   - 2 rows per CTA (256 threads = 2 rows x 128 threads x 4 warps).
//   - Phase 0a: cp.async X + cos + sin (HBM -> SMEM) in one commit group.
//   - Phase 0b: sync load weight -> regs (overlaps the cp.async transfers).
//   - Phase 1: sum^2 reads X from SMEM (no HBM re-read).
//   - Phase 2: reads X + cos + sin from SMEM, multiplies by cached weight regs, writes HBM.
template <int HEAD_DIM, bool INTERLEAVE, bool PER_HEAD_COS, typename CosT>
__global__ void fusedDiTSplitNormFullDimRopeKernel(__nv_bfloat16* __restrict__ tensor, int const num_tokens,
    int const num_heads, float const eps, __nv_bfloat16 const* __restrict__ weight, CosT const* __restrict__ cos_emb,
    CosT const* __restrict__ sin_emb, int const cos_seq_per_batch)
{
    constexpr int BLOCK_SIZE = 256;
    constexpr int ROWS_PER_BLOCK = 2;
    constexpr int THREADS_PER_ROW = BLOCK_SIZE / ROWS_PER_BLOCK; // 128
    constexpr int WARPS_PER_ROW = THREADS_PER_ROW / 32;          // 4
    constexpr int CHUNK_ELEMS = 8;                               // uint4 = 8 bf16
    constexpr int MAX_N = 32 * HEAD_DIM;
    constexpr int MAX_CHUNKS = (MAX_N + THREADS_PER_ROW * CHUNK_ELEMS - 1) / (THREADS_PER_ROW * CHUNK_ELEMS);

#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;");
#endif

    int const tid = threadIdx.x;
    int const row_in_block = tid / THREADS_PER_ROW; // 0 or 1
    int const lane_in_row = tid % THREADS_PER_ROW;  // 0..127
    int const row_warp = lane_in_row >> 5;          // 0..3
    int const row_lane = lane_in_row & 31;

    int const tokenIdx = blockIdx.x * ROWS_PER_BLOCK + row_in_block;
    if (tokenIdx >= num_tokens)
        return;

    int const N = num_heads * HEAD_DIM;
    int const chunks_per_row = (N + THREADS_PER_ROW * CHUNK_ELEMS - 1) / (THREADS_PER_ROW * CHUNK_ELEMS);
    int64_t const tokenBase = static_cast<int64_t>(tokenIdx) * N;
    int const cos_tokenIdx = (cos_seq_per_batch > 0) ? (tokenIdx % cos_seq_per_batch) : tokenIdx;
    int64_t const embBase = PER_HEAD_COS ? static_cast<int64_t>(cos_tokenIdx) * num_heads * HEAD_DIM
                                         : static_cast<int64_t>(cos_tokenIdx) * HEAD_DIM;

    // SMEM layout: [X bf16 stage][cos CosT stage][sin CosT stage][warp_sums fp32].
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __nv_bfloat16* smem_input = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    CosT* smem_cos = reinterpret_cast<CosT*>(smem_raw + ROWS_PER_BLOCK * N * sizeof(__nv_bfloat16));
    CosT* smem_sin = smem_cos + ROWS_PER_BLOCK * N;
    float* warp_sums = reinterpret_cast<float*>(
        smem_raw + ROWS_PER_BLOCK * N * sizeof(__nv_bfloat16) + 2 * ROWS_PER_BLOCK * N * sizeof(CosT));

    // Phase 0a: cp.async X + cos + sin HBM -> SMEM.
#pragma unroll
    for (int chunk = 0; chunk < MAX_CHUNKS; chunk++)
    {
        if (chunk >= chunks_per_row)
            continue;
        int const elemBase = chunk * THREADS_PER_ROW * CHUNK_ELEMS + lane_in_row * CHUNK_ELEMS;
        if (elemBase >= N)
            continue;
        __pipeline_memcpy_async(smem_input + row_in_block * N + elemBase, tensor + tokenBase + elemBase, 16);
        int const headIdx = elemBase / HEAD_DIM;
        int const baseDim = elemBase - headIdx * HEAD_DIM;
        int const cosHeadOff = PER_HEAD_COS ? headIdx * HEAD_DIM : 0;
        if constexpr (std::is_same_v<CosT, float>)
        {
            // fp32 cos: 8 floats per chunk = 32 bytes = 2x 16-byte cp.async per array.
            __pipeline_memcpy_async(
                smem_cos + row_in_block * N + elemBase, cos_emb + embBase + cosHeadOff + baseDim, 16);
            __pipeline_memcpy_async(
                smem_cos + row_in_block * N + elemBase + 4, cos_emb + embBase + cosHeadOff + baseDim + 4, 16);
            __pipeline_memcpy_async(
                smem_sin + row_in_block * N + elemBase, sin_emb + embBase + cosHeadOff + baseDim, 16);
            __pipeline_memcpy_async(
                smem_sin + row_in_block * N + elemBase + 4, sin_emb + embBase + cosHeadOff + baseDim + 4, 16);
        }
        else
        {
            // bf16 cos: 8 bf16 = 16 bytes = 1x cp.async per array.
            __pipeline_memcpy_async(
                smem_cos + row_in_block * N + elemBase, cos_emb + embBase + cosHeadOff + baseDim, 16);
            __pipeline_memcpy_async(
                smem_sin + row_in_block * N + elemBase, sin_emb + embBase + cosHeadOff + baseDim, 16);
        }
    }
    __pipeline_commit();

    // Phase 0b: sync load weight -> regs (overlaps with cp.async transfers above).
    uint4 weight_cache[MAX_CHUNKS];
#pragma unroll
    for (int chunk = 0; chunk < MAX_CHUNKS; chunk++)
    {
        if (chunk >= chunks_per_row)
            continue;
        int const elemBase = chunk * THREADS_PER_ROW * CHUNK_ELEMS + lane_in_row * CHUNK_ELEMS;
        if (elemBase >= N)
            continue;
        int const headIdx = elemBase / HEAD_DIM;
        int const baseDim = elemBase - headIdx * HEAD_DIM;
        weight_cache[chunk] = *reinterpret_cast<uint4 const*>(&weight[headIdx * HEAD_DIM + baseDim]);
    }

    // Phase 0c: wait + sync.
    __pipeline_wait_prior(0);
    __syncthreads();

    // Phase 1: sum^2 from SMEM.
    float sum2 = 0.0f;
#pragma unroll
    for (int chunk = 0; chunk < MAX_CHUNKS; chunk++)
    {
        if (chunk >= chunks_per_row)
            continue;
        int const elemBase = chunk * THREADS_PER_ROW * CHUNK_ELEMS + lane_in_row * CHUNK_ELEMS;
        if (elemBase >= N)
            continue;
        uint4 const v = *reinterpret_cast<uint4 const*>(&smem_input[row_in_block * N + elemBase]);
        uint const* uints = reinterpret_cast<uint const*>(&v);
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            float2 vals = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&uints[i]));
            sum2 += vals.x * vals.x + vals.y * vals.y;
        }
    }

    // Per-row warp reduce + cross-warp reduce.
    sum2 = tensorrt_llm::common::warpReduceSum(sum2);
    if (row_lane == 0)
        warp_sums[row_in_block * WARPS_PER_ROW + row_warp] = sum2;
    __syncthreads();
    float total = 0.0f;
#pragma unroll
    for (int w = 0; w < WARPS_PER_ROW; w++)
        total += warp_sums[row_in_block * WARPS_PER_ROW + w];
    float const rms_rcp = rsqrtf(total / static_cast<float>(N) + eps);

    // Phase 2: read X + cos + sin from SMEM, apply cached weight, RoPE, write to HBM.
#pragma unroll
    for (int chunk = 0; chunk < MAX_CHUNKS; chunk++)
    {
        if (chunk >= chunks_per_row)
            continue;
        int const elemBase = chunk * THREADS_PER_ROW * CHUNK_ELEMS + lane_in_row * CHUNK_ELEMS;
        if (elemBase >= N)
            continue;

        uint4 const in_vec = *reinterpret_cast<uint4 const*>(&smem_input[row_in_block * N + elemBase]);
        uint4 const w_vec = weight_cache[chunk];

        float cos_vals[CHUNK_ELEMS];
        float sin_vals[CHUNK_ELEMS];
        if constexpr (std::is_same_v<CosT, float>)
        {
            float4 const* cs = reinterpret_cast<float4 const*>(&smem_cos[row_in_block * N + elemBase]);
            float4 const* ss = reinterpret_cast<float4 const*>(&smem_sin[row_in_block * N + elemBase]);
            float4 c0 = cs[0], c1 = cs[1];
            float4 s0 = ss[0], s1 = ss[1];
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
            uint4 const cp = *reinterpret_cast<uint4 const*>(&smem_cos[row_in_block * N + elemBase]);
            uint4 const sp = *reinterpret_cast<uint4 const*>(&smem_sin[row_in_block * N + elemBase]);
            uint const* cu = reinterpret_cast<uint const*>(&cp);
            uint const* su = reinterpret_cast<uint const*>(&sp);
#pragma unroll
            for (int i = 0; i < 4; i++)
            {
                float2 cv = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&cu[i]));
                float2 sv = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&su[i]));
                cos_vals[2 * i] = cv.x;
                cos_vals[2 * i + 1] = cv.y;
                sin_vals[2 * i] = sv.x;
                sin_vals[2 * i + 1] = sv.y;
            }
        }

        float elements[CHUNK_ELEMS];
        float w_vals[CHUNK_ELEMS];
        uint const* x_uints = reinterpret_cast<uint const*>(&in_vec);
        uint const* w_uints = reinterpret_cast<uint const*>(&w_vec);
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            float2 xv = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&x_uints[i]));
            float2 wv = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&w_uints[i]));
            elements[2 * i] = xv.x;
            elements[2 * i + 1] = xv.y;
            w_vals[2 * i] = wv.x;
            w_vals[2 * i + 1] = wv.y;
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
            constexpr int xor_mask = HEAD_DIM / 16;
            bool const negate = ((row_lane & xor_mask) == 0);
            unsigned const activeMask = __activemask();
#pragma unroll
            for (int i = 0; i < CHUNK_ELEMS; i++)
            {
                float p = __shfl_xor_sync(activeMask, elements[i], xor_mask);
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
    int cos_seq_per_batch, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(num_heads <= 32,
        "fusedDiTSplitNormFullDimRope: num_heads (%d) must be <= 32 (block_size = num_heads*32 <= 1024)", num_heads);
    TLLM_CHECK_WITH_INFO(num_heads >= 1, "fusedDiTSplitNormFullDimRope: num_heads must be >= 1, got %d", num_heads);

    int const N = num_heads * head_dim;
    constexpr int ROWS_PER_BLOCK = 2;

    cudaLaunchAttribute attrs[1] = {};
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;

    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3((num_tokens + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);
    cfg.blockDim = dim3(256);
    // SMEM = X stage (bf16) + cos stage (CosT) + sin stage (CosT) + warp_sums.
    size_t const cos_elem_size = cos_is_bf16 ? sizeof(__nv_bfloat16) : sizeof(float);
    cfg.dynamicSmemBytes = ROWS_PER_BLOCK * N * sizeof(__nv_bfloat16) + 2 * ROWS_PER_BLOCK * N * cos_elem_size
        + ROWS_PER_BLOCK * 4 /*WARPS_PER_ROW*/ * sizeof(float);
    cfg.stream = stream;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;
    // Default per-CTA dynamic SMEM cap is 48 KB; raise it per kernel specialization.
#define LAUNCH(HEAD_DIM, INTERLEAVE, PER_HEAD, COS_T)                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        auto* kptr = fusedDiTSplitNormFullDimRopeKernel<HEAD_DIM, INTERLEAVE, PER_HEAD, COS_T>;                        \
        cudaFuncSetAttribute(                                                                                          \
            reinterpret_cast<void const*>(kptr), cudaFuncAttributeMaxDynamicSharedMemorySize, cfg.dynamicSmemBytes);   \
        cudaLaunchKernelEx(&cfg, kptr, reinterpret_cast<__nv_bfloat16*>(tensor), num_tokens, num_heads, eps,           \
            reinterpret_cast<__nv_bfloat16 const*>(weight), reinterpret_cast<COS_T const*>(cos_emb),                   \
            reinterpret_cast<COS_T const*>(sin_emb), cos_seq_per_batch);                                               \
    } while (0)
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
