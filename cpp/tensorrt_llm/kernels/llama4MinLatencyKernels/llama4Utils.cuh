/*
 * Copyright (c) 2025-2025, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include <cuda_fp8.h>

#include "tensorrt_llm/common/envUtils.h"

namespace tensorrt_llm::kernels::llama4_min_latency
{

namespace llama4_bf16_bf16_gemm
{
constexpr int GEMM_K = 5120;
constexpr int BLOCK_SIZE = 256;
constexpr int NUM_EXPERTS = 128;
constexpr int VEC_SIZE = 4;

} // namespace llama4_bf16_bf16_gemm

namespace llama4_fp8_bf16_gemm
{

constexpr int HIDDEN_IN = 5120;
constexpr int HIDDEN_OUT = 896;
constexpr int Q_HIDDEN_OUT = 640;

constexpr float FLOOR_SCALE = 8192.0;
constexpr float ATTN_SCALE = 0.1;

constexpr int BLOCK_SIZE = 128;
constexpr int WARP_SIZE = 32;
constexpr int WARP_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;
constexpr int VEC_SIZE = 8;

constexpr bool ENABLE_ACQBULK = 1;
constexpr bool ENABLE_PREFETCH = 1;
constexpr bool ENABLE_PREEXIT = 0;

} // namespace llama4_fp8_bf16_gemm

namespace llama4_fp8_fp8_gemm_swiglu
{

constexpr int BLOCK_SIZE = 128;
constexpr int WARP_SIZE = 32;
constexpr int VEC_SIZE = 8;

constexpr bool ENABLE_ACQBULK = 1;
constexpr bool ENABLE_PREFETCH = 0;
constexpr bool ENABLE_PREEXIT = 0;

} // namespace llama4_fp8_fp8_gemm_swiglu

inline void launch_kernel_pdl(
    dim3 grid_dim, dim3 block_dim, cudaStream_t stream, void* kernel_func, void* args[], int num_args)
{
    cudaLaunchConfig_t config;
    config.gridDim = grid_dim;
    config.blockDim = block_dim;
    config.dynamicSmemBytes = 0;
    config.stream = stream;

    cudaLaunchAttribute attrs[1];
    config.attrs = attrs;
    config.numAttrs = 0;
    attrs[config.numAttrs].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[config.numAttrs++].val.programmaticStreamSerializationAllowed
        = (tensorrt_llm::common::getEnvEnablePDL() ? 1 : 0);

    cudaLaunchKernelExC(&config, (void const*) kernel_func, args);
}

inline int div_up(int x, int y)
{
    return (x + y - 1) / y;
}

__device__ __forceinline__ float2 ffma2(float2 x, float2 y, float2 acc)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000))
    return __ffma2_rn(x, y, acc);
#else
    return make_float2(x.x * y.x + acc.x, x.y * y.y + acc.y);
#endif
}

__device__ __forceinline__ float silu(float x)
{
    return x / (1.0f + __expf(-x));
}

__device__ __forceinline__ float sigmoid(float x)
{
    return 1.0f / (1.0f + __expf(-x));
}

struct __align__(8) aligned_fp8x8
{
    __align__(8) __nv_fp8x4_e4m3 data[2];
};

struct __align__(8) aligned_bfloat16x4
{
    __align__(8) __nv_bfloat16 data[4];
};

} // namespace tensorrt_llm::kernels::llama4_min_latency
