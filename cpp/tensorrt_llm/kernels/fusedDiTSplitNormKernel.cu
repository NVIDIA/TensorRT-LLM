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

#include "fusedDiTSplitNormKernel.h"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/mathUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Fused full-dim RMSNorm only (no RoPE) on a SINGLE Q or K tensor.
// Strategy:
//   - 2 rows per CTA (256 threads = 2 rows × 128 threads × 4 warps).
//   - cp.async X HBM → SMEM (Phase 0a) overlaps with sync weight load → regs (Phase 0b).
//   - Phase 1: sum^2 from SMEM, per-row reduce.
//   - Phase 2: re-read X from SMEM, multiply by cached weight regs, write HBM.
template <int HEAD_DIM>
__global__ void fusedDiTSplitNormFullDimKernel(__nv_bfloat16* __restrict__ tensor, int const num_tokens,
    int const num_heads, float const eps, __nv_bfloat16 const* __restrict__ weight)
{
    constexpr int BLOCK_SIZE = 256;
    constexpr int ROWS_PER_BLOCK = 2;
    constexpr int THREADS_PER_ROW = BLOCK_SIZE / ROWS_PER_BLOCK; // 128
    constexpr int WARPS_PER_ROW = THREADS_PER_ROW / 32;          // 4
    constexpr int CHUNK_ELEMS = 8;                               // uint4 = 8 bf16
    constexpr int MAX_N = 32 * HEAD_DIM;
    constexpr int MAX_CHUNKS_PER_ROW = (MAX_N + THREADS_PER_ROW * CHUNK_ELEMS - 1) / (THREADS_PER_ROW * CHUNK_ELEMS);

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

    extern __shared__ __align__(16) unsigned char smem_raw[];
    __nv_bfloat16* smem_input = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    float* warp_sums = reinterpret_cast<float*>(smem_raw + ROWS_PER_BLOCK * N * sizeof(__nv_bfloat16));

    // Phase 0a: issue cp.async X HBM -> SMEM (one commit group).
#pragma unroll
    for (int chunk = 0; chunk < MAX_CHUNKS_PER_ROW; chunk++)
    {
        if (chunk >= chunks_per_row)
            continue;
        int const elemBase = chunk * THREADS_PER_ROW * CHUNK_ELEMS + lane_in_row * CHUNK_ELEMS;
        if (elemBase >= N)
            continue;
        __pipeline_memcpy_async(smem_input + row_in_block * N + elemBase, tensor + tokenBase + elemBase, 16);
    }
    __pipeline_commit();

    // Phase 0b: SYNC load weight into registers (overlaps with cp.async X HBM transfer).
    // weight_cache[chunk] holds 8 bf16 weight elements per chunk.
    uint4 weight_cache[MAX_CHUNKS_PER_ROW];
#pragma unroll
    for (int chunk = 0; chunk < MAX_CHUNKS_PER_ROW; chunk++)
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

    // Phase 0c: wait for X cp.async + sync block.
    __pipeline_wait_prior(0);
    __syncthreads();

    // Phase 1: sum^2 from SMEM.
    float sum2 = 0.0f;
#pragma unroll
    for (int chunk = 0; chunk < MAX_CHUNKS_PER_ROW; chunk++)
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

    // Phase 2: re-read X from SMEM, multiply by cached weight regs, write to HBM.
#pragma unroll
    for (int chunk = 0; chunk < MAX_CHUNKS_PER_ROW; chunk++)
    {
        if (chunk >= chunks_per_row)
            continue;
        int const elemBase = chunk * THREADS_PER_ROW * CHUNK_ELEMS + lane_in_row * CHUNK_ELEMS;
        if (elemBase >= N)
            continue;

        uint4 const in_vec = *reinterpret_cast<uint4 const*>(&smem_input[row_in_block * N + elemBase]);
        uint4 const w_vec = weight_cache[chunk];

        uint const* x_uints = reinterpret_cast<uint const*>(&in_vec);
        uint const* w_uints = reinterpret_cast<uint const*>(&w_vec);
        uint4 out_vec;
        uint* o_uints = reinterpret_cast<uint*>(&out_vec);
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            float2 x_vals = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&x_uints[i]));
            float2 w_vals = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&w_uints[i]));
            float2 y_vals;
            y_vals.x = x_vals.x * rms_rcp * w_vals.x;
            y_vals.y = x_vals.y * rms_rcp * w_vals.y;
            __nv_bfloat162 bf = __float22bfloat162_rn(y_vals);
            reinterpret_cast<__nv_bfloat162&>(o_uints[i]) = bf;
        }
        *reinterpret_cast<uint4*>(&tensor[tokenBase + elemBase]) = out_vec;
    }

#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void launchFusedDiTSplitNormFullDim(
    void* tensor, int num_tokens, int num_heads, int head_dim, float eps, void const* weight, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(num_heads <= 32,
        "fusedDiTSplitNormFullDim: num_heads (%d) must be <= 32 (block_size = num_heads*32 <= 1024)", num_heads);
    TLLM_CHECK_WITH_INFO(num_heads >= 1, "fusedDiTSplitNormFullDim: num_heads must be >= 1, got %d", num_heads);

    int const N = num_heads * head_dim;
    constexpr int ROWS_PER_BLOCK = 2;

    cudaLaunchAttribute attrs[1] = {};
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;

    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3((num_tokens + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);
    cfg.blockDim = dim3(256);
    cfg.dynamicSmemBytes
        = ROWS_PER_BLOCK * N * sizeof(__nv_bfloat16) + ROWS_PER_BLOCK * 4 /*warps_per_row*/ * sizeof(float);
    cfg.stream = stream;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;
#define LAUNCH(HEAD_DIM)                                                                                               \
    cudaLaunchKernelEx(&cfg, fusedDiTSplitNormFullDimKernel<HEAD_DIM>, reinterpret_cast<__nv_bfloat16*>(tensor),       \
        num_tokens, num_heads, eps, reinterpret_cast<__nv_bfloat16 const*>(weight))
    switch (head_dim)
    {
    case 64: LAUNCH(64); break;
    case 128: LAUNCH(128); break;
    default: TLLM_THROW("Unsupported head_dim for fusedDiTSplitNormFullDim: %d (only 64, 128)", head_dim);
    }
#undef LAUNCH
}

} // namespace kernels

TRTLLM_NAMESPACE_END
