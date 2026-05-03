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
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Fused full-dim RMSNorm only (no RoPE) on a SINGLE Q or K tensor.
// Mirror of fusedDiTSplitNormFullDimRopeKernel sans RoPE step:
//   Phase 1: sum^2 reduce over full inner dim (num_heads * HEAD_DIM).
//   Phase 2: scale by rsqrt(mean+eps) * weight, write back.
// Block=256 (8 warps), grid=(num_tokens,). No cos/sin LDG; no rotation.
template <int HEAD_DIM>
__global__ void fusedDiTSplitNormFullDimKernel(__nv_bfloat16* __restrict__ tensor, int const num_tokens,
    int const num_heads, float const eps, __nv_bfloat16 const* __restrict__ weight)
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

    __shared__ float warp_sums[32];

    // Phase 1: sum^2
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

    // Phase 2: re-read input + apply scale + store (no RoPE).
    for (int chunk = 0; chunk < num_chunks; chunk++)
    {
        int const elemBase = chunk * CHUNK_BLOCK_ELEMS + tid * CHUNK_ELEMS;
        if (elemBase >= N)
            continue;
        int const headIdx = elemBase / HEAD_DIM;
        int const baseDim = elemBase - headIdx * HEAD_DIM;

        uint4 in_vec = *reinterpret_cast<uint4 const*>(&tensor[tokenBase + elemBase]);
        float w_vals[CHUNK_ELEMS];
#pragma unroll
        for (int i = 0; i < CHUNK_ELEMS; i++)
            w_vals[i] = __bfloat162float(weight[headIdx * HEAD_DIM + baseDim + i]);

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

void launchFusedDiTSplitNormFullDim(
    void* tensor, int num_tokens, int num_heads, int head_dim, float eps, void const* weight, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(num_heads <= 32,
        "fusedDiTSplitNormFullDim: num_heads (%d) must be <= 32 (block_size = num_heads*32 <= 1024)", num_heads);
    TLLM_CHECK_WITH_INFO(num_heads >= 1, "fusedDiTSplitNormFullDim: num_heads must be >= 1, got %d", num_heads);

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
