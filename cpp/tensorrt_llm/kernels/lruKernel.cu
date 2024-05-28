/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include <cuda_runtime_api.h>

#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#ifdef ENABLE_FP8
#include <cuda_fp8.h>
#endif

#include "lruKernel.h"
#include "tensorrt_llm/common/cudaTypeUtils.cuh"

using namespace tensorrt_llm::common;

namespace tensorrt_llm
{
namespace kernels
{

__forceinline__ __device__ float copysignf_pos(float a, float b)
{
    float r;
    r = __int_as_float(__float_as_int(a) | (__float_as_int(b) & 0x80000000));
    return r;
}

#pragma nv_diag_suppress static_var_with_dynamic_init

template <typename T, int CHANNELS_PER_BLOCK = 128, int STAGES = 20, int SEQ_UNROLL = 10>
__launch_bounds__(256, 1) __global__ void rg_lru_kernel(lruParams params)
{
    T* output = reinterpret_cast<T*>(params.out_ptr);
    float* state = reinterpret_cast<float*>(params.state_ptr);
    T* x = reinterpret_cast<T*>(params.x_ptr);
    T* y = reinterpret_cast<T*>(params.y_ptr);
    T* y_bias = reinterpret_cast<T*>(params.y_bias_ptr);
    T* A = reinterpret_cast<T*>(params.A_ptr);
    int num_channels = params.width;
    int block_size = params.block_size;

    bool enable_fuse_gate = (params.gate_ptr != nullptr);
    bool enable_gate_bias;
    T *gate_x, *gate_a, *gate_x_bias, *gate_a_bias;
    if (enable_fuse_gate)
    {
        enable_gate_bias = (params.gate_bias_ptr != nullptr);
        gate_x = reinterpret_cast<T*>(params.gate_ptr);
        gate_a = reinterpret_cast<T*>(params.gate_ptr);
        if (enable_gate_bias)
        {
            gate_x_bias = reinterpret_cast<T*>(params.gate_bias_ptr);
            gate_a_bias = reinterpret_cast<T*>(params.gate_bias_ptr);
        }
    }
    else
    {
        enable_gate_bias = (params.gate_x_bias_ptr != nullptr);
        gate_x = reinterpret_cast<T*>(params.gate_x_ptr);
        gate_a = reinterpret_cast<T*>(params.gate_a_ptr);
        if (enable_gate_bias)
        {
            gate_x_bias = reinterpret_cast<T*>(params.gate_x_bias_ptr);
            gate_a_bias = reinterpret_cast<T*>(params.gate_a_bias_ptr);
        }
    }

    __shared__ cuda::pipeline_shared_state<cuda::thread_scope::thread_scope_block, STAGES / SEQ_UNROLL> pipeline_state;
    auto block = cooperative_groups::this_thread_block();

    __shared__ __align__(128) T sh_gx[STAGES][CHANNELS_PER_BLOCK];
    __shared__ __align__(128) T sh_gate_x_bias[CHANNELS_PER_BLOCK];
    __shared__ __align__(128) T sh_ga[STAGES][CHANNELS_PER_BLOCK];
    __shared__ __align__(128) T sh_gate_a_bias[CHANNELS_PER_BLOCK];
    __shared__ __align__(128) T sh_x[STAGES][CHANNELS_PER_BLOCK];
    __shared__ __align__(128) T sh_y[STAGES][CHANNELS_PER_BLOCK];
    __shared__ __align__(128) T sh_y_bias[CHANNELS_PER_BLOCK];
    __shared__ __align__(128) float sh_a[CHANNELS_PER_BLOCK];

    int const channel = blockIdx.x * blockDim.x + threadIdx.x;
    int const sample = blockIdx.y; // batch id
    int const tid = threadIdx.x;

    int const slot_idx = params.slot_mapping_ptr == nullptr ? sample : params.slot_mapping_ptr[sample];
    int num_tokens;
    int start_token_idx;
    if (params.remove_padding)
    {
        start_token_idx = sample == 0 ? 0 : params.last_token_ids_ptr[sample - 1];
        int end_token_idx = params.last_token_ids_ptr[sample];
        num_tokens = end_token_idx - start_token_idx;
    }
    else
    {
        start_token_idx = sample * params.max_seqlen;
        num_tokens = params.last_token_ids_ptr[sample];
    }

    int const seq_loops = (num_tokens + SEQ_UNROLL - 1) / SEQ_UNROLL;
    int const block_channel_base = start_token_idx * num_channels + blockIdx.x * blockDim.x;
    int const gate_num_channels = enable_fuse_gate ? num_channels * 2 : num_channels;
    int const gate_block_channel_base = start_token_idx * gate_num_channels;
    int const tid_offset = tid < 64 ? 32 : 64;
    int const gchannel = sizeof(T) == 4 ? channel : blockIdx.x * blockDim.x + (threadIdx.x - tid_offset) * 4;
    int const gx_dim_idx = enable_fuse_gate ? gchannel / block_size * block_size * 2 + gchannel % block_size : gchannel;
    int const ga_dim_idx = enable_fuse_gate ? gx_dim_idx + block_size : gchannel;
    int const gx_bias_idx = enable_fuse_gate ? channel / block_size * block_size * 2 + channel % block_size : channel;
    int const ga_bias_idx = enable_fuse_gate ? gx_bias_idx + block_size : channel;

    if (threadIdx.y == 1)
    {
        // Data loading warps

        // Bias and param A are independent of token
        if (y_bias)
            sh_y_bias[tid] = y_bias[channel];
        if (enable_gate_bias)
        {
            sh_gate_x_bias[tid] = gate_x_bias[gx_bias_idx];
            sh_gate_a_bias[tid] = gate_a_bias[ga_bias_idx];
        }
        float param_a = cuda_cast<float>(A[channel]);
        sh_a[tid] = param_a <= 20.f ? -8.0f * __logf(1.0f + __expf(param_a)) : -8.0f * param_a;

        cuda::pipeline pipeline = cuda::make_pipeline(block, &pipeline_state, cuda::pipeline_role::producer);

        int stage = 0;
        for (int si = 0; si < seq_loops; si++)
        {
            pipeline.producer_acquire();
#pragma unroll
            for (int token_id = si * SEQ_UNROLL; token_id < num_tokens && token_id < (si + 1) * SEQ_UNROLL; token_id++)
            {
                int block_channel = block_channel_base + token_id * num_channels;
                int gate_block_channel = gate_block_channel_base + token_id * gate_num_channels;
                if (sizeof(T) == 4)
                {
                    cuda::memcpy_async(
                        &sh_gx[stage][tid], &gate_x[gate_block_channel + gx_dim_idx], sizeof(T), pipeline);
                    cuda::memcpy_async(
                        &sh_ga[stage][tid], &gate_a[gate_block_channel + ga_dim_idx], sizeof(T), pipeline);
                    cuda::memcpy_async(&sh_x[stage][tid], &x[block_channel + tid], sizeof(T), pipeline);
                    if (y)
                        cuda::memcpy_async(&sh_y[stage][tid], &y[block_channel + tid], sizeof(T), pipeline);
                }
                else
                {
                    if (tid < 32)
                    {
                        float2* block_x = (float2*) &x[block_channel];
                        cuda::memcpy_async((float2*) &sh_x[stage][tid * 4], &block_x[tid], sizeof(float2), pipeline);
                    }
                    else if (tid < 64)
                    {
                        int tid_tmp = tid - 32;
                        float2* block_gx = (float2*) &gate_x[gate_block_channel];
                        cuda::memcpy_async(
                            (float2*) &sh_gx[stage][tid_tmp * 4], &block_gx[gx_dim_idx >> 2], sizeof(float2), pipeline);
                    }
                    else if (tid < 96)
                    {
                        int tid_tmp = tid - 64;
                        float2* block_ga = (float2*) &gate_a[gate_block_channel];
                        cuda::memcpy_async(
                            (float2*) &sh_ga[stage][tid_tmp * 4], &block_ga[ga_dim_idx >> 2], sizeof(float2), pipeline);
                    }
                    else if (tid < 128)
                    {
                        if (y)
                        {
                            int tid_tmp = tid - 96;
                            float2* block_y = (float2*) &y[block_channel];
                            cuda::memcpy_async(
                                (float2*) &sh_y[stage][tid_tmp * 4], &block_y[tid_tmp], sizeof(float2), pipeline);
                        }
                    }
                }
                stage++;
                if (stage >= STAGES)
                    stage = 0;
            }
            pipeline.producer_commit();
        }
    }
    else
    {
        // Compute warps

        cuda::pipeline pipeline = cuda::make_pipeline(block, &pipeline_state, cuda::pipeline_role::consumer);

        float state_reg = 0.f;
        int stage = 0;

        for (int si = 0; si < seq_loops; si++)
        {
            pipeline.consumer_wait();
#pragma unroll
            for (int token_id = si * SEQ_UNROLL; token_id < num_tokens && token_id < (si + 1) * SEQ_UNROLL; token_id++)
            {
                // Read y
                float y_reg;
                if (y_bias)
                {
                    y_reg = cuda_cast<float>(sh_y[stage][tid] + sh_y_bias[tid]);
                    // GELU
                    float k0 = float(0.7978845608028654);
                    float k1 = float(0.044715);
                    float y_tanh = k0 * y_reg * (1.0 + k1 * y_reg * y_reg);
                    float exp_val = -1.f * cuda_abs(y_tanh * 2);
                    y_reg = 0.5f * y_reg
                        * (1.f + copysignf_pos(__fdividef((1.f - __expf(exp_val)), (1.f + __expf(exp_val))), y_tanh));
                }
                else if (y)
                {
                    y_reg = cuda_cast<float>(sh_y[stage][tid]);
                }
                else
                {
                    y_reg = 1.f;
                }
                // Read gate_x
                float gate_x_reg, gate_a_reg;
                if (enable_gate_bias)
                {
                    gate_x_reg = cuda_cast<float>(-sh_gx[stage][tid] - sh_gate_x_bias[tid]);
                    gate_a_reg = cuda_cast<float>(-sh_ga[stage][tid] - sh_gate_a_bias[tid]);
                }
                else
                {
                    gate_x_reg = cuda_cast<float>(-sh_gx[stage][tid]);
                    gate_a_reg = cuda_cast<float>(-sh_ga[stage][tid]);
                }
                // Get gated inputs
                float x_reg = cuda_cast<float>(sh_x[stage][tid]);
                float sigmoid_x = __fdividef(1.0f, (1.0f + __expf(gate_x_reg)));
                float sigmoid_a = __fdividef(1.0f, (1.0f + __expf(gate_a_reg)));
                float log_a = sigmoid_a * sh_a[tid];
                float a = __expf(log_a);
                float a_square = __expf(2.0 * log_a);
                float outf = y_reg;
                float normalized_x = x_reg * sigmoid_x;
                if (si != 0 || token_id != 0)
                    normalized_x *= sqrtf(1 - a_square);

                // RNN scan
                state_reg = a * state_reg + normalized_x;
                outf *= state_reg;

                // Write output
                T* out = &output[start_token_idx * num_channels + token_id * num_channels];
                out[channel] = cuda_cast<T>(outf);

                stage++;
                if (stage >= STAGES)
                    stage = 0;
            }
            pipeline.consumer_release();
        }
        // Write the new state back out to the cache
        state[slot_idx * num_channels + channel] = state_reg;
    }
}

template <typename T>
void invokeRGLRU(lruParams& params, cudaStream_t stream)
{
    int samples = params.batch;
    int channels = params.width;

    int const threads = 128;
    int const blocks = (channels + threads - 1) / threads;
    dim3 block(threads, 2);
    dim3 grid(blocks, samples);
    TLLM_CHECK((channels % block.x) == 0);
    TLLM_CHECK(!(params.block_size % 4 != 0 && sizeof(T) == 2));

    rg_lru_kernel<T><<<grid, block, 0, stream>>>(params);
}

#define INSTANTIATE_RGLRU_DATA_TYPE(T) template void invokeRGLRU<T>(lruParams & params, cudaStream_t stream);

INSTANTIATE_RGLRU_DATA_TYPE(float);
INSTANTIATE_RGLRU_DATA_TYPE(half);
#ifdef ENABLE_BF16
INSTANTIATE_RGLRU_DATA_TYPE(__nv_bfloat16);
#endif
#undef INSTANTIATE_RGLRU_DATA_TYPE

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
__launch_bounds__(128, 2) __global__ void rg_lru_update_kernel(lruParams params)
{
    T* output = reinterpret_cast<T*>(params.out_ptr);
    float* state = reinterpret_cast<float*>(params.state_ptr);
    T* x = reinterpret_cast<T*>(params.x_ptr);
    T* y = reinterpret_cast<T*>(params.y_ptr);
    T* y_bias = reinterpret_cast<T*>(params.y_bias_ptr);
    T* A = reinterpret_cast<T*>(params.A_ptr);
    int num_channels = params.width;
    int block_size = params.block_size;

    bool enable_fuse_gate = (params.gate_ptr != nullptr);
    bool enable_gate_bias;
    T *gate_x, *gate_a, *gate_x_bias, *gate_a_bias;
    if (enable_fuse_gate)
    {
        enable_gate_bias = (params.gate_bias_ptr != nullptr);
        gate_x = reinterpret_cast<T*>(params.gate_ptr);
        gate_a = reinterpret_cast<T*>(params.gate_ptr);
        if (enable_gate_bias)
        {
            gate_x_bias = reinterpret_cast<T*>(params.gate_bias_ptr);
            gate_a_bias = reinterpret_cast<T*>(params.gate_bias_ptr);
        }
    }
    else
    {
        enable_gate_bias = (params.gate_x_bias_ptr != nullptr);
        gate_x = reinterpret_cast<T*>(params.gate_x_ptr);
        gate_a = reinterpret_cast<T*>(params.gate_a_ptr);
        if (enable_gate_bias)
        {
            gate_x_bias = reinterpret_cast<T*>(params.gate_x_bias_ptr);
            gate_a_bias = reinterpret_cast<T*>(params.gate_a_bias_ptr);
        }
    }

    int const channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= num_channels)
        return;
    int const sample = blockIdx.y; // batch id
    int const slot_idx = params.slot_mapping_ptr == nullptr ? sample : params.slot_mapping_ptr[sample];
    int const idx = sample * num_channels + channel;
    int const gate_num_channels = enable_fuse_gate ? num_channels * 2 : num_channels;
    int const gate_base_idx = sample * gate_num_channels;
    int const gx_dim_idx = enable_fuse_gate ? channel / block_size * block_size * 2 + channel % block_size : channel;
    int const ga_dim_idx = enable_fuse_gate ? gx_dim_idx + block_size : channel;

    float state_reg = state[slot_idx * num_channels + channel];

    // Read a
    float param_a = cuda_cast<float>(A[channel]);
    float c = param_a <= 20.f ? -8.0f * __logf(1.0f + __expf(param_a)) : -8.0f * param_a;

    // Read y
    float y_reg;
    if (y_bias)
    {
        y_reg = cuda_cast<float>(y[idx] + y_bias[channel]);
        // GELU
        float k0 = float(0.7978845608028654);
        float k1 = float(0.044715);
        float y_tanh = k0 * y_reg * (1.0 + k1 * y_reg * y_reg);
        float exp_val = -1.f * cuda_abs(y_tanh * 2);
        y_reg = 0.5f * y_reg
            * (1.f + copysignf_pos(__fdividef((1.f - __expf(exp_val)), (1.f + __expf(exp_val))), y_tanh));
    }
    else if (y)
    {
        y_reg = cuda_cast<float>(y[idx]);
    }
    else
    {
        y_reg = 1.f;
    }
    // Read gate_x
    float gate_x_reg, gate_a_reg;
    if (enable_gate_bias)
    {
        gate_x_reg = cuda_cast<float>(-gate_x[gate_base_idx + gx_dim_idx] - gate_x_bias[gx_dim_idx]);
        gate_a_reg = cuda_cast<float>(-gate_a[gate_base_idx + ga_dim_idx] - gate_a_bias[ga_dim_idx]);
    }
    else
    {
        gate_x_reg = cuda_cast<float>(-gate_x[gate_base_idx + gx_dim_idx]);
        gate_a_reg = cuda_cast<float>(-gate_a[gate_base_idx + ga_dim_idx]);
    }
    // Get gated inputs
    float sigmoid_x = __fdividef(1.0f, (1.0f + __expf(gate_x_reg)));
    float sigmoid_a = __fdividef(1.0f, (1.0f + __expf(gate_a_reg)));
    float log_a = sigmoid_a * c;
    float a = __expf(log_a);
    float a_square = __expf(2.0 * log_a);
    float outf = y_reg;
    float normalized_x = cuda_cast<float>(x[idx]) * sigmoid_x * sqrtf(1 - a_square);

    // RNN update
    state_reg = a * state_reg + normalized_x;
    outf *= state_reg;

    // Write output and state
    output[sample * num_channels + channel] = cuda_cast<T>(outf);
    state[slot_idx * num_channels + channel] = state_reg;
}

template <typename T>
void invokeRGLRUUpdate(lruParams& params, cudaStream_t stream)
{
    int samples = params.batch;
    int channels = params.width;

    int const threads = 128;
    int const blocks = (channels + threads - 1) / threads;
    dim3 block(threads, 1);
    dim3 grid(blocks, samples);

    rg_lru_update_kernel<T><<<grid, block, 0, stream>>>(params);
}

#define INSTANTIATE_RGLRU_UPDATE_DATA_TYPE(T)                                                                          \
    template void invokeRGLRUUpdate<T>(lruParams & params, cudaStream_t stream)

INSTANTIATE_RGLRU_UPDATE_DATA_TYPE(float);
INSTANTIATE_RGLRU_UPDATE_DATA_TYPE(half);
#ifdef ENABLE_BF16
INSTANTIATE_RGLRU_UPDATE_DATA_TYPE(__nv_bfloat16);
#endif
#undef INSTANTIATE_RGLRU_UPDATE_DATA_TYPE

} // namespace kernels
} // namespace tensorrt_llm
