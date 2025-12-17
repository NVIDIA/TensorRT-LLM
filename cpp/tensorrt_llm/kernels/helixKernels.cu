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

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/kernels/helixKernels.h"

#include <cstdint>
#include <cstdio>

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

using namespace tensorrt_llm::common;

namespace cg = cooperative_groups;

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{
static constexpr int WARP_SIZE = 32;

// Utility: warp-level corrected sum
template <int N>
__device__ inline void warpReduceCorrectedSum(float (&correctedVal)[N], float (&maxVal)[N], float (&sumVal)[N])
{
    float warp_max = maxVal[0];
#pragma unroll
    for (int nn = 1; nn < N; ++nn)
        warp_max = fmaxf(warp_max, maxVal[nn]);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000 && defined(__CUDA_ARCH_FEAT_SM100_ALL))
    asm("redux.sync.max.f32 %0, %1, 0xffffffff;\n" : "=f"(warp_max) : "f"(warp_max));
#else
#pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset *= 2)
        warp_max = fmaxf(warp_max, __shfl_xor_sync(0xffffffff, warp_max, offset));
#endif
    float global_sum = 0.F;
    float corrected_max_exp[N];
#pragma unroll
    for (int nn = 0; nn < N; ++nn)
    {
        corrected_max_exp[nn] = sumVal[nn] * expf(maxVal[nn] - warp_max);
        global_sum += corrected_max_exp[nn];
    }
#pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset *= 2)
        global_sum += __shfl_xor_sync(0xffffffff, global_sum, offset);
    auto norm = 1.F / global_sum;
#pragma unroll
    for (int nn = 0; nn < N; ++nn)
        correctedVal[nn] = corrected_max_exp[nn] * norm;
}

static constexpr int MAX_CP_VAL_PER_THREAD = 8;
static constexpr int MAX_CP = WARP_SIZE * MAX_CP_VAL_PER_THREAD;
static constexpr int BYTES_O_PER_THREAD = 16;
static constexpr int NUM_PRE_LOAD = 8;

// Kernel: fused helix post-processing
// output: [num_tokens, num_heads * kv_lora_rank] (half)
// gathered_o: [cp_size, num_tokens, num_heads * kv_lora_rank] (half)
// gathered_stats: [cp_size, num_tokens, num_heads, 2] (fp32)
// note: we explicitly avoid using restrict here, to avoid getting ld.global.nc
// which may have longer latency
template <typename T>
__global__ void helix_postprocess_kernel(
    T* output, T const* gathered_o, float2 const* gathered_stats, int cp_size, int kv_lora_rank)
{
    // Each block processes one (token, head)
    // gridDim.x: num_tokens, gridDim.y: num_heads
    // there are two separate types of warps:
    // warp 0 calculates the correction values (one per cp_size)
    // all other warps pre-load the gathered_o elements for the current token/head
    // and once warp 0 is done, all other warps can start accumulating the output
    static constexpr int NUM_O_PER_THREAD = BYTES_O_PER_THREAD / sizeof(T);

    int tok_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int num_tokens = gridDim.x;
    int num_heads = gridDim.y;

    int const cp_size_aligned = ((cp_size + NUM_PRE_LOAD - 1) / NUM_PRE_LOAD) * NUM_PRE_LOAD;
    __shared__ float smem_correction[MAX_CP];

    int lane_idx = threadIdx.x % WARP_SIZE;
    int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / WARP_SIZE, 0);
    // here we have to wait for memory operations of the previous kernel to complete
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif

    if (warp_idx == 0)
    {
        // the warp collectively calculates the correction values
        float max_values[MAX_CP_VAL_PER_THREAD];
        float sum_values[MAX_CP_VAL_PER_THREAD];
#pragma unroll
        for (int cp_val_idx = 0; cp_val_idx < MAX_CP_VAL_PER_THREAD; ++cp_val_idx)
        {
            auto cp_idx = cp_val_idx * WARP_SIZE + lane_idx;
            auto stats_offset = cp_idx * num_tokens * num_heads + tok_idx * num_heads + head_idx;
            float2 stats = cp_idx < cp_size ? gathered_stats[stats_offset] : make_float2(-INFINITY, 0.F);
            max_values[cp_val_idx] = stats.x;
            sum_values[cp_val_idx] = stats.y;
        }
        float corrected_values[MAX_CP_VAL_PER_THREAD];
        warpReduceCorrectedSum(corrected_values, max_values, sum_values);
#pragma unroll
        for (int cp_val_idx = 0; cp_val_idx < MAX_CP_VAL_PER_THREAD; ++cp_val_idx)
        {
            auto cp_idx = cp_val_idx * WARP_SIZE + lane_idx;
            smem_correction[cp_idx] = corrected_values[cp_val_idx];
        }
        cg::this_thread_block().sync();
    }
    else
    {
        // all other warps pre-load the gathered_o elements for the current token/head
        auto const* gathered_o_off = gathered_o + tok_idx * num_heads * kv_lora_rank + head_idx * kv_lora_rank;
        // we subtract WARP_SIZE because first warp is not participating here
        gathered_o_off += (threadIdx.x - WARP_SIZE) * NUM_O_PER_THREAD;
        float4 const* gathered_o_16b = reinterpret_cast<float4 const*>(gathered_o_off);
        auto gathered_16b_stride = (num_tokens * num_heads * kv_lora_rank) / NUM_O_PER_THREAD;
        T vals[NUM_PRE_LOAD][NUM_O_PER_THREAD];
#pragma unroll
        for (int cp_idx = 0; cp_idx < NUM_PRE_LOAD && cp_idx < cp_size; ++cp_idx)
        {
            auto val
                = cp_idx < cp_size ? gathered_o_16b[cp_idx * gathered_16b_stride] : make_float4(0.F, 0.F, 0.F, 0.F);
            *reinterpret_cast<float4*>(vals[cp_idx]) = val;
        }
        float final_sum[NUM_O_PER_THREAD];
#pragma unroll
        for (int o_idx = 0; o_idx < NUM_O_PER_THREAD; ++o_idx)
        {
            final_sum[o_idx] = 0.F;
        }
        cg::this_thread_block().sync();

        // here we can trigger the dependent kernels to start
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
        cudaTriggerProgrammaticLaunchCompletion();
#endif

        float corr_vals[NUM_PRE_LOAD];
#pragma unroll
        for (int cp_idx = 0; cp_idx < NUM_PRE_LOAD && cp_idx < cp_size; ++cp_idx)
        {
            corr_vals[cp_idx] = smem_correction[cp_idx];
        }

        for (int cp_idx_base = NUM_PRE_LOAD; cp_idx_base < cp_size_aligned; cp_idx_base += NUM_PRE_LOAD)
        {
#pragma unroll
            for (int cp_idx = 0; cp_idx < NUM_PRE_LOAD; ++cp_idx)
            {
#pragma unroll
                for (int o_idx = 0; o_idx < NUM_O_PER_THREAD; ++o_idx)
                {
                    final_sum[o_idx] += static_cast<float>(vals[cp_idx][o_idx]) * corr_vals[cp_idx];
                }
            }
#pragma unroll
            for (int cp_idx = 0; cp_idx < NUM_PRE_LOAD; ++cp_idx)
            {
                *reinterpret_cast<float4*>(vals[cp_idx]) = cp_idx_base + cp_idx < cp_size
                    ? gathered_o_16b[(cp_idx_base + cp_idx) * gathered_16b_stride]
                    : make_float4(0.F, 0.F, 0.F, 0.F);
                corr_vals[cp_idx] = cp_idx_base + cp_idx < cp_size ? smem_correction[cp_idx_base + cp_idx] : 0.F;
            }
        }
#pragma unroll
        for (int cp_idx = 0; cp_idx < NUM_PRE_LOAD && cp_idx < cp_size; ++cp_idx)
        {
#pragma unroll
            for (int o_idx = 0; o_idx < NUM_O_PER_THREAD; ++o_idx)
            {
                final_sum[o_idx] += static_cast<float>(vals[cp_idx][o_idx]) * corr_vals[cp_idx];
            }
        }
        T output_typed[NUM_O_PER_THREAD];
#pragma unroll
        for (int o_idx = 0; o_idx < NUM_O_PER_THREAD; ++o_idx)
        {
            output_typed[o_idx] = static_cast<T>(final_sum[o_idx]);
        }
        auto* output_off = output + tok_idx * num_heads * kv_lora_rank + head_idx * kv_lora_rank;
        output_off += (threadIdx.x - WARP_SIZE) * NUM_O_PER_THREAD;
        *reinterpret_cast<float4*>(output_off) = *reinterpret_cast<float4*>(output_typed);
    }
}

static constexpr int MAX_THREADS = 256;
static constexpr int MAX_KV_LORA_BYTES = (MAX_THREADS - WARP_SIZE) * BYTES_O_PER_THREAD;

// Kernel: fused helix post-processing
// output: [num_tokens, num_heads * kv_lora_rank] (half)
// gathered_o: [num_tokens, num_heads, cp_size, kv_lora_rank] (half)
// gathered_stats: [num_tokens, num_heads, cp_size, 2] (fp32)
// note: we explicitly avoid using restrict here, to avoid getting ld.global.nc
// which may have longer latency
template <typename T>
__global__ void __launch_bounds__(MAX_THREADS) helix_postprocess_kernel_native(
    T* output, T const* gathered_o, float2 const* gathered_stats, int cp_size, int kv_lora_rank)
{
    // Each block processes one (token, head)
    // gridDim.x: num_tokens, gridDim.y: num_heads
    // there are two separate types of warps:
    // warp 0 calculates the correction values (one per cp_size)
    // all other warps pre-load the gathered_o elements for the current token/head
    // and once warp 0 is done, all other warps can start accumulating the output
    static constexpr int NUM_O_PER_THREAD = BYTES_O_PER_THREAD / sizeof(T);

    int tok_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int num_tokens = gridDim.x;
    int num_heads = gridDim.y;

    int const cp_size_aligned = ((cp_size + NUM_PRE_LOAD - 1) / NUM_PRE_LOAD) * NUM_PRE_LOAD;
    __shared__ float smem_correction[MAX_CP];

    int lane_idx = threadIdx.x % WARP_SIZE;
    int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / WARP_SIZE, 0);

    // all warps except first pre-load the gathered_o elements for the current
    // token/head
    T const* gathered_o_off;
    gathered_o_off = gathered_o + tok_idx * num_heads * cp_size * kv_lora_rank + head_idx * cp_size * kv_lora_rank;
    // we subtract WARP_SIZE because first warp is not participating in pre-load
    gathered_o_off += (threadIdx.x - WARP_SIZE) * NUM_O_PER_THREAD;
    float4 const* gathered_o_16b = reinterpret_cast<float4 const*>(gathered_o_off);
    int gathered_16b_stride = (kv_lora_rank) / NUM_O_PER_THREAD;
    int stats_offset = tok_idx * num_heads * cp_size + head_idx * cp_size;
    int stats_stride = 1;

    // here we have to wait for memory operations of the previous kernel to
    // complete
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif

    float max_values[MAX_CP_VAL_PER_THREAD];
    float sum_values[MAX_CP_VAL_PER_THREAD];
    T vals[NUM_PRE_LOAD][NUM_O_PER_THREAD];
    float final_sum[NUM_O_PER_THREAD];
    float corr_vals[NUM_PRE_LOAD];
    T output_typed[NUM_O_PER_THREAD];

    if (warp_idx == 0)
    {
        // the warp collectively calculates the correction values
#pragma unroll
        for (int cp_val_idx = 0; cp_val_idx < MAX_CP_VAL_PER_THREAD; ++cp_val_idx)
        {
            auto cp_idx = cp_val_idx * WARP_SIZE + lane_idx;
            auto stats_idx = stats_offset + cp_idx * stats_stride;
            float2 stats = cp_idx < cp_size ? gathered_stats[stats_idx] : make_float2(-INFINITY, 0.F);
            max_values[cp_val_idx] = stats.x;
            sum_values[cp_val_idx] = stats.y;
        }
        float corrected_values[MAX_CP_VAL_PER_THREAD];
        warpReduceCorrectedSum(corrected_values, max_values, sum_values);
#pragma unroll
        for (int cp_val_idx = 0; cp_val_idx < MAX_CP_VAL_PER_THREAD; ++cp_val_idx)
        {
            auto cp_idx = cp_val_idx * WARP_SIZE + lane_idx;
            smem_correction[cp_idx] = corrected_values[cp_val_idx];
        }
    }
    else
    {
        // all other warps pre-load the gathered_o elements
#pragma unroll
        for (int cp_idx = 0; cp_idx < NUM_PRE_LOAD && cp_idx < cp_size; ++cp_idx)
        {
            auto val = gathered_o_16b[cp_idx * gathered_16b_stride];
            *reinterpret_cast<float4*>(vals[cp_idx]) = val;
        }
#pragma unroll
        for (int o_idx = 0; o_idx < NUM_O_PER_THREAD; ++o_idx)
        {
            final_sum[o_idx] = 0.F;
        }
    }
    __syncthreads();

    // warp 0 exits early
    if (warp_idx == 0)
        return;

        // here we can trigger the dependent kernels to start
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif

#pragma unroll
    for (int cp_idx = 0; cp_idx < NUM_PRE_LOAD && cp_idx < cp_size; ++cp_idx)
    {
        corr_vals[cp_idx] = smem_correction[cp_idx];
    }

    for (int cp_idx_base = NUM_PRE_LOAD; cp_idx_base < cp_size_aligned; cp_idx_base += NUM_PRE_LOAD)
    {
#pragma unroll
        for (int cp_idx = 0; cp_idx < NUM_PRE_LOAD; ++cp_idx)
        {
#pragma unroll
            for (int o_idx = 0; o_idx < NUM_O_PER_THREAD; ++o_idx)
            {
                final_sum[o_idx] += static_cast<float>(vals[cp_idx][o_idx]) * corr_vals[cp_idx];
            }
        }
#pragma unroll
        for (int cp_idx = 0; cp_idx < NUM_PRE_LOAD; ++cp_idx)
        {
            *reinterpret_cast<float4*>(vals[cp_idx]) = cp_idx_base + cp_idx < cp_size
                ? gathered_o_16b[(cp_idx_base + cp_idx) * gathered_16b_stride]
                : make_float4(0.F, 0.F, 0.F, 0.F);
            corr_vals[cp_idx] = cp_idx_base + cp_idx < cp_size ? smem_correction[cp_idx_base + cp_idx] : 0.F;
        }
    }
#pragma unroll
    for (int cp_idx = 0; cp_idx < NUM_PRE_LOAD && cp_idx < cp_size; ++cp_idx)
    {
#pragma unroll
        for (int o_idx = 0; o_idx < NUM_O_PER_THREAD; ++o_idx)
        {
            final_sum[o_idx] += static_cast<float>(vals[cp_idx][o_idx]) * corr_vals[cp_idx];
        }
    }
#pragma unroll
    for (int o_idx = 0; o_idx < NUM_O_PER_THREAD; ++o_idx)
    {
        output_typed[o_idx] = static_cast<T>(final_sum[o_idx]);
    }
    auto* output_off = output + tok_idx * num_heads * kv_lora_rank + head_idx * kv_lora_rank;
    output_off += (threadIdx.x - WARP_SIZE) * NUM_O_PER_THREAD;
    *reinterpret_cast<float4*>(output_off) = *reinterpret_cast<float4*>(output_typed);
}

} // anonymous namespace

template <typename T>
void helixPostProcess(HelixPostProcParams<T> const& params, cudaStream_t stream)
{
    // Check that gathered_o is 16-byte aligned
    TLLM_CHECK_WITH_INFO(reinterpret_cast<uintptr_t>(params.gathered_o) % 16 == 0,
        "gathered_o must be 16-byte aligned for async memcpy");
    // Check that kv_lora_rank * sizeof(T) is a multiple of 16
    TLLM_CHECK_WITH_INFO((params.kv_lora_rank * sizeof(T)) % 16 == 0,
        "kv_lora_rank * sizeof(T) must be a multiple of 16 for async memcpy");
    // Check that cp_size is not larger than the max fallback CP size
    TLLM_CHECK_WITH_INFO(params.cp_size <= MAX_CP, "cp_size > fallback max CP size");

    auto* kernel_instance = &helix_postprocess_kernel<T>;
    cudaLaunchConfig_t config;
    config.gridDim = dim3(params.num_tokens, params.num_heads);
    config.blockDim = WARP_SIZE + params.kv_lora_rank * sizeof(T) / 16;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
    config.numAttrs = 1;
    config.attrs = attrs;
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel_instance, params.output, params.gathered_o,
        params.gathered_stats, params.cp_size, params.kv_lora_rank));
}

#define INSTANTIATE_POST_PROC(T)                                                                                       \
    template void helixPostProcess<T>(HelixPostProcParams<T> const& params, cudaStream_t stream);

INSTANTIATE_POST_PROC(__half);
INSTANTIATE_POST_PROC(__nv_bfloat16);

template <typename T>
void helixPostProcessNative(HelixPostProcParams<T> const& params, cudaStream_t stream)
{
    // Check that gathered_o is 16-byte aligned
    TLLM_CHECK_WITH_INFO(reinterpret_cast<uintptr_t>(params.gathered_o) % 16 == 0,
        "gathered_o must be 16-byte aligned for async memcpy");
    // TODO: Figure why this constraint is specific to this implementation and not legacy one.
    TLLM_CHECK_WITH_INFO((params.kv_lora_rank * sizeof(T)) <= MAX_KV_LORA_BYTES,
        "kv_lora_rank * sizeof(T) must be <= %zu bytes", MAX_KV_LORA_BYTES);
    // Check that kv_lora_rank * sizeof(T) is a multiple of 16
    TLLM_CHECK_WITH_INFO((params.kv_lora_rank * sizeof(T)) % 16 == 0,
        "kv_lora_rank * sizeof(T) must be a multiple of 16 for async memcpy");
    // Check that cp_size is not larger than the max fallback CP size
    TLLM_CHECK_WITH_INFO(params.cp_size <= MAX_CP, "cp_size > fallback max CP size");

    auto kernel_instance = helix_postprocess_kernel_native<T>;
    cudaLaunchConfig_t config;
    config.gridDim = dim3(params.num_tokens, params.num_heads);
    config.blockDim = WARP_SIZE + params.kv_lora_rank * sizeof(T) / 16;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = common::getEnvEnablePDL();
    config.numAttrs = 1;
    config.attrs = attrs;
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel_instance, params.output, params.gathered_o,
        params.gathered_stats, params.cp_size, params.kv_lora_rank));
}

#define INSTANTIATE_POST_PROC_NATIVE(T)                                                                                \
    template void helixPostProcessNative<T>(HelixPostProcParams<T> const& params, cudaStream_t stream);

INSTANTIATE_POST_PROC_NATIVE(__half);
INSTANTIATE_POST_PROC_NATIVE(__nv_bfloat16);

} // namespace kernels

TRTLLM_NAMESPACE_END
