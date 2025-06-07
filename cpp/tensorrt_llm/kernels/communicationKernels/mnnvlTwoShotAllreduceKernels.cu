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

#include "mnnvlTwoShotAllreduceKernels.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/logger.h"
#include <cstddef>
#include <cstdint>
#include <cuda/atomic>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <nvml.h>

namespace tensorrt_llm::kernels::mnnvl
{
__device__ bool isNegZero(float v)
{
    return v == 0.f && signbit(v);
}

__device__ bool isNegZero(__nv_bfloat16 val)
{
    return isNegZero(__bfloat162float(val));
}

template <typename T>
inline __device__ float toFloat(T val)
{
    return val;
}

template <>
inline __device__ float toFloat<__nv_bfloat16>(__nv_bfloat16 val)
{
    return __bfloat162float(val);
}

template <typename T>
inline __device__ T fromFloat(float val)
{
    return val;
}

template <>
inline __device__ __nv_bfloat16 fromFloat<__nv_bfloat16>(float val)
{
    return __float2bfloat16(val);
}

template <int WORLD_SIZE, typename T>
__global__ void twoshot_allreduce_kernel(T* output_ptr, T* shard_ptr, T** input_ptrs, T* mcast_ptr, int num_tokens,
    int buffer_M, int token_dim, int rank, uint32_t* buffer_flags, bool wait_for_results)
{
    int elt = blockIdx.y * blockDim.x + threadIdx.x;
    if (elt >= token_dim)
        return;
    int token = blockIdx.x;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif

    uint32_t* offset_access_ptr = &buffer_flags[3];
    // Buffer size is M * N, and we need two buffers for reduce-scatter and allgather
    uint32_t buffer_size = (buffer_flags[2] << 1);
    uint32_t input_offset = buffer_flags[0] * buffer_size;
    uint32_t clear_offset = buffer_flags[1] * buffer_size;

    if (wait_for_results)
    {
        __syncthreads();
        if (threadIdx.x == 0)
        {
            atomicAdd(offset_access_ptr, 1);
        }
    }

    if (elt < token_dim)
    {
        // Scatter token
        int dest_rank = token % WORLD_SIZE;
        int dest_token_offset = token / WORLD_SIZE;
        T val = shard_ptr[token * token_dim + elt];
        if (isNegZero(val))
            val = fromFloat<T>(0.f);
        input_ptrs[dest_rank][input_offset + dest_token_offset * token_dim * WORLD_SIZE + rank * token_dim + elt] = val;

        // Reduce and broadcast

        int global_token = token * WORLD_SIZE + rank;
        if (global_token < num_tokens)
        {

            float accum = 0.f;

            T values[WORLD_SIZE];

            for (int r = 0; r < WORLD_SIZE; r++)
            {
                input_ptrs[rank][clear_offset + token * token_dim * WORLD_SIZE + r * token_dim + elt]
                    = fromFloat<T>(-0.f);
            }

            while (1)
            {
                bool valid = true;
                for (int r = 0; r < WORLD_SIZE; r++)
                {
                    T volatile* lamport_ptr = (T volatile*) &input_ptrs[rank][input_offset
                        + token * token_dim * WORLD_SIZE + r * token_dim + elt];
                    values[r] = *lamport_ptr;
                    valid &= !isNegZero(values[r]);
                }
                if (valid)
                    break;
            }
            for (int r = 0; r < WORLD_SIZE; r++)
            {
                accum += toFloat<T>(values[r]);
            }
            mcast_ptr[input_offset + buffer_M * token_dim + global_token * token_dim + elt] = fromFloat<T>(accum);
        }
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif

    input_ptrs[rank][clear_offset + buffer_M * token_dim + token * token_dim + elt] = fromFloat<T>(-0.f);

    // Optionally wait for results if the next layer isn't doing the Lamport check
    if (wait_for_results)
    {
        T volatile* lamport_ptr
            = (T volatile*) &input_ptrs[rank][input_offset + buffer_M * token_dim + token * token_dim + elt];
        T val = *lamport_ptr;
        while (isNegZero(val))
            val = *lamport_ptr;

        // Copy if requested
        if (output_ptr)
            output_ptr[token * token_dim + elt] = val;
        if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0)
        {
            // Make sure all blocks have finished reading the offsets, 2-D grid
            while (*reinterpret_cast<uint32_t volatile*>(offset_access_ptr) < gridDim.x * gridDim.y)
            {
            }
            buffer_flags[0] = (buffer_flags[0] + 1) % 3;
            buffer_flags[1] = (buffer_flags[1] + 1) % 3;
            *(offset_access_ptr) = 0;
        }
    }
}

#define LAUNCH_ALL_REDUCE_KERNEL(WORLD_SIZE, T)                                                                        \
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&config, &twoshot_allreduce_kernel<WORLD_SIZE, T>,                              \
        reinterpret_cast<T*>(params.output), reinterpret_cast<T*>(params.input),                                       \
        reinterpret_cast<T**>(params.buffer_ptrs_dev), (T*) params.multicast_ptr, params.num_tokens, params.buffer_M,  \
        params.token_dim, params.rank, reinterpret_cast<uint32_t*>(params.buffer_flags), params.wait_for_results));

void twoshot_allreduce_op(AllReduceParams const& params)
{
    int const world_size = params.nranks;
    int const rank = params.rank;
    auto const dtype = params.dtype;
    int const buffer_M = params.buffer_M;
    int const num_tokens = params.num_tokens;
    int const token_dim = params.token_dim;
    bool const wait_for_results = params.wait_for_results;

    int const num_threads = 128;
    int const num_blocks = (token_dim + num_threads - 1) / num_threads;

    dim3 grid(num_tokens, num_blocks);
    TLLM_LOG_DEBUG(
        "[TwoShot AllReduce] twoshot allreduce on rank %d, world_size: %d, buffer_M: %d, num_tokens: %d, token_dim: "
        "%d, wait_for_results: %d",
        rank, world_size, buffer_M, num_tokens, token_dim, wait_for_results);

    cudaLaunchConfig_t config;
    cudaLaunchAttribute attrs[1];
    config.dynamicSmemBytes = 0;
    config.stream = params.stream;
    config.gridDim = grid;
    config.blockDim = num_threads;
    config.attrs = attrs;
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL() ? 1 : 0;
    config.numAttrs = 1;

    if (dtype == nvinfer1::DataType::kFLOAT)
    {
        switch (world_size)
        {
        case 2: LAUNCH_ALL_REDUCE_KERNEL(2, float); break;
        case 4: LAUNCH_ALL_REDUCE_KERNEL(4, float); break;
        case 8: LAUNCH_ALL_REDUCE_KERNEL(8, float); break;
        case 16: LAUNCH_ALL_REDUCE_KERNEL(16, float); break;
        case 32: LAUNCH_ALL_REDUCE_KERNEL(32, float); break;
        case 64: LAUNCH_ALL_REDUCE_KERNEL(64, float); break;
        default: TLLM_CHECK_WITH_INFO(false, "TwoShot AllReduce]: unsupported world_size.");
        }
    }
    else if (dtype == nvinfer1::DataType::kBF16)
    {
        switch (world_size)
        {
        case 2: LAUNCH_ALL_REDUCE_KERNEL(2, __nv_bfloat16); break;
        case 4: LAUNCH_ALL_REDUCE_KERNEL(4, __nv_bfloat16); break;
        case 8: LAUNCH_ALL_REDUCE_KERNEL(8, __nv_bfloat16); break;
        case 16: LAUNCH_ALL_REDUCE_KERNEL(16, __nv_bfloat16); break;
        case 32: LAUNCH_ALL_REDUCE_KERNEL(32, __nv_bfloat16); break;
        case 64: LAUNCH_ALL_REDUCE_KERNEL(64, __nv_bfloat16); break;
        default: TLLM_CHECK_WITH_INFO(false, "TwoShot AllReduce]: unsupported world_size.");
        }
    }
    else
    {
        TLLM_CHECK_WITH_INFO(false, "TwoShot AllReduce]: unsupported dtype.");
    }
}

template <typename T_IN>
__device__ void copy_f4(T_IN* dst, T_IN const* src)
{
    float4* dst4 = (float4*) dst;
    float4 const* src4 = (float4 const*) src;
    __pipeline_memcpy_async(dst4, src4, sizeof(float4));
}

template <typename T_IN>
__device__ void copy_f4_ldg(T_IN* dst, T_IN const* src)
{
    float4* dst4 = (float4*) dst;
    float4 const* src4 = (float4*) src;
    *dst4 = *src4;
}

__device__ float4 loadfloat4(void const* ptr)
{

    float return_value[4];

    asm volatile("ld.volatile.global.v4.f32 {%0, %1, %2, %3}, [%4];\n"
                 : "=f"(return_value[0]), "=f"(return_value[1]), "=f"(return_value[2]), "=f"(return_value[3])
                 : "l"(ptr));

    return *(float4*) return_value;
}

template <typename T>
inline __device__ T add(T a, T b)
{
    return a + b;
}

#define FINAL_MASK 0xffffffff

template <typename T>
__inline__ __device__ T warpReduceSum(T val)
{
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val = add<T>(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32)); //__shfl_sync bf16 return float when sm < 80
    return val;
}

inline __device__ float block_reduce_sum(float val)
{
    __shared__ float smem[32];
    int lane_id = threadIdx.x % 32, warp_id = threadIdx.x / 32, warp_num = blockDim.x / 32;
    val = warpReduceSum(val);
    if (lane_id == 0)
    {
        smem[warp_id] = val;
    }
    __syncthreads();
    val = lane_id < warp_num ? smem[lane_id] : 0.f;
    val = warpReduceSum(val);
    return val;
}

template <int DIM, int NUM_THREADS, int NUM_INPUTS, typename T_OUT, typename T_IN>
__global__ void __launch_bounds__(128, 1)
    RMSNorm(T_IN* input_plus_residual, T_OUT* output_norm, T_IN const* buffer_input, T_IN const* gamma, float epsilon,
        T_IN const* residual, int batch_size, uint32_t* buffer_flags)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    static bool const LAMPORT = true;

    extern __shared__ uint8_t smem[];

    int sample = blockIdx.y;

    static int const CGA_THREADS = NUM_THREADS * 1;

    static int const ITERS = DIM / CGA_THREADS;
    float r_input[ITERS];
    float r_gamma[ITERS];

    T_IN* sh_input = (T_IN*) &smem[0];
    T_IN* sh_residual = (T_IN*) &smem[NUM_INPUTS * NUM_THREADS * ITERS * sizeof(T_IN)];
    T_IN* sh_gamma = (T_IN*) &smem[(NUM_INPUTS + 1) * NUM_THREADS * ITERS * sizeof(T_IN)];

    static int const ELTS_PER_THREAD = sizeof(float4) / sizeof(T_IN);

    int offsets[NUM_INPUTS][DIM / (1 * ELTS_PER_THREAD * NUM_THREADS)];

    uint32_t* offset_access_ptr = &buffer_flags[3];
    // Buffer size is M * N, and we need two buffers for reduce-scatter and allgather
    uint32_t buffer_size = buffer_flags[2];
    uint32_t buffer_offset = buffer_flags[0] * (buffer_size << 1);
    T_IN const* input = &buffer_input[buffer_offset + buffer_size];

    cudaTriggerProgrammaticLaunchCompletion();

    __syncthreads();
    if (threadIdx.x == 0)
    {
        atomicAdd(offset_access_ptr, 1);
    }

    for (int i = 0; i < NUM_INPUTS; i++)
    {
        for (int j = 0; j < DIM / (1 * ELTS_PER_THREAD * NUM_THREADS); j++)
        {
            int k = j * NUM_THREADS + threadIdx.x;
            offsets[i][j] = i * batch_size * DIM + sample * DIM + blockIdx.x * DIM / 1 + k * ELTS_PER_THREAD;
        }
    }

#pragma unroll
    for (int j = 0; j < DIM / (1 * ELTS_PER_THREAD * NUM_THREADS); j++)
    {
        int i = j * NUM_THREADS + threadIdx.x;
        copy_f4(&sh_residual[i * ELTS_PER_THREAD], &residual[sample * DIM + blockIdx.x * DIM + i * ELTS_PER_THREAD]);
    }

    __pipeline_commit();

#pragma unroll
    for (int j = 0; j < DIM / (ELTS_PER_THREAD * NUM_THREADS); j++)
    {
        int i = j * NUM_THREADS + threadIdx.x;
        copy_f4(&sh_gamma[i * ELTS_PER_THREAD], &gamma[blockIdx.x * DIM + i * ELTS_PER_THREAD]);
    }

    __pipeline_commit();

    // Load all inputs
    bool valid = false;

    if (!LAMPORT)
        cudaGridDependencySynchronize();

    while (!valid)
    {
        valid = true;
#pragma unroll
        for (int i = 0; i < NUM_INPUTS; i++)
        {
            for (int j = 0; j < DIM / (ELTS_PER_THREAD * NUM_THREADS); j++)
            {
                int k = j * NUM_THREADS + threadIdx.x;

                float4* dst4 = (float4*) &sh_input[i * NUM_THREADS * ITERS + k * ELTS_PER_THREAD];
                float4* src4 = (float4*) &input[offsets[i][j]];

                float4 value = loadfloat4(src4);
                if (LAMPORT)
                {
                    // Assume that the 16B were written atomically, so we only need to check one value
                    T_IN lowest_val = *(T_IN*) &value;
                    valid &= !isNegZero(lowest_val);
                }
                *dst4 = value;
            }
        }
    }

    __syncthreads();

    // Perform the initial input reduction
    if (NUM_INPUTS > 0)
    {

        T_IN accum[ELTS_PER_THREAD];
        float4* accum4 = (float4*) &accum;

        for (int j = 0; j < DIM / (ELTS_PER_THREAD * NUM_THREADS); j++)
        {
            int k = j * NUM_THREADS + threadIdx.x;

            *accum4 = *(float4*) &sh_input[k * ELTS_PER_THREAD];

            for (int i = 1; i < NUM_INPUTS; i++)
            {
                float4 data = *(float4*) &sh_input[i * NUM_THREADS * ITERS + k * ELTS_PER_THREAD];
                T_IN* p_d = (T_IN*) &data;
                for (int x = 0; x < ELTS_PER_THREAD; x++)
                {
                    accum[x] += p_d[x];
                }
            }

            // Write back to input 0's staging location.  No sync needed since all data localized to thread.
            *(float4*) &sh_input[k * ELTS_PER_THREAD] = *accum4;
        }
    }

    // Wait for residual
    __pipeline_wait_prior(1);
    __syncthreads();

    float thread_sum = 0.f;

#pragma unroll
    for (int io = 0; io < ITERS / ELTS_PER_THREAD; io++)
    {

        float4 inp4 = *(float4*) &sh_input[io * NUM_THREADS * ELTS_PER_THREAD + threadIdx.x * ELTS_PER_THREAD];
        float4 res4 = *(float4*) &sh_residual[io * NUM_THREADS * ELTS_PER_THREAD + threadIdx.x * ELTS_PER_THREAD];

        T_IN* r_inp = (T_IN*) &inp4;
        T_IN* r_res = (T_IN*) &res4;

        float4 out4;

        T_IN* r_out = (T_IN*) &out4;

        for (int ii = 0; ii < ELTS_PER_THREAD; ii++)
        {

            int i = io * ELTS_PER_THREAD + ii;

            T_IN inp_plus_resid = r_inp[ii] + r_res[ii];
            r_out[ii] = inp_plus_resid;
            r_input[i] = toFloat(inp_plus_resid);

            // Accumulate the squares for RMSNorm
            thread_sum += toFloat(inp_plus_resid * inp_plus_resid);
        }

        *(float4*) &input_plus_residual[sample * DIM + blockIdx.x * DIM + io * NUM_THREADS * ELTS_PER_THREAD
            + threadIdx.x * ELTS_PER_THREAD]
            = out4;
    }

    // Wait for Gamma.  There will be a global synchronization as part of the reduction
    __pipeline_wait_prior(0);

    float cluster_sum = block_reduce_sum(thread_sum);

    float rcp_rms = rsqrtf(cluster_sum / DIM + epsilon);

#pragma unroll
    for (int io = 0; io < ITERS / ELTS_PER_THREAD; io++)
    {

        float4 gamma4 = *(float4*) &sh_gamma[io * NUM_THREADS * ELTS_PER_THREAD + threadIdx.x * ELTS_PER_THREAD];
        T_IN* r_g4 = (T_IN*) &gamma4;

        float4 out4;
        // FIXME: this only works if T_OUT == T_IN
        T_OUT* r_out = (T_OUT*) &out4;

        for (int ii = 0; ii < ELTS_PER_THREAD; ii++)
        {
            int i = io * ELTS_PER_THREAD + ii;
            r_gamma[i] = toFloat(r_g4[ii]);
            r_out[ii] = fromFloat<T_OUT>(r_gamma[i] * r_input[i] * rcp_rms);
        }

        *(float4*) &output_norm[sample * DIM + blockIdx.x * DIM + io * NUM_THREADS * ELTS_PER_THREAD
            + threadIdx.x * ELTS_PER_THREAD]
            = out4;
    }
    // Update the buffer pointers
    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0)
    {
        // Make sure all blocks have finished accessing the buffer
        while (*reinterpret_cast<uint32_t volatile*>(offset_access_ptr) != gridDim.x * gridDim.y)
        {
        }
        buffer_flags[0] = (buffer_flags[0] + 1) % 3;
        buffer_flags[1] = (buffer_flags[1] + 1) % 3;
        *(offset_access_ptr) = 0;
    }
    __syncthreads();
#endif
}

template <typename T, int H_DIM>
void twoshot_rmsnorm(T* prenorm_output, T* normed_output, T const* input, T const* gamma, double epsilon,
    T const* residual, uint32_t* buffer_flags, int batch, cudaStream_t stream)
{

    // input to rmsnorm is the buffer in the twoshot ar
    // We should use prenorm output to determine the actual used size
    // int batch = normed_output.sizes()[0];
    // int dim = normed_output.sizes()[1];
    float _epsilon{static_cast<float>(epsilon)};

    static constexpr int NUM_THREADS = 128;
    static constexpr int CGA_THREADS = NUM_THREADS;
    constexpr int iters = H_DIM / CGA_THREADS;

    dim3 grid(1, batch, 1);

    cudaLaunchConfig_t config;
    cudaLaunchAttribute attrs[1];
    config.stream = stream;
    config.gridDim = grid;
    config.blockDim = NUM_THREADS;
    config.attrs = attrs;
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL() ? 1 : 0;
    config.numAttrs = 1;

    size_t shmem_size = 3 * NUM_THREADS * iters * sizeof(T);
    cudaFuncSetAttribute(
        &RMSNorm<H_DIM, NUM_THREADS, 1, T, T>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
    config.dynamicSmemBytes = shmem_size;
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&config, &RMSNorm<H_DIM, NUM_THREADS, 1, T, T>, prenorm_output, normed_output,
        input, gamma, _epsilon, residual, batch, buffer_flags));
}

#define LAUNCH_RMSNORM_KERNEL(T, H_DIM)                                                                                \
    twoshot_rmsnorm<T, H_DIM>(static_cast<T*>(params.residual_output), static_cast<T*>(params.output),                 \
        static_cast<T const*>(params.input), static_cast<T const*>(params.gamma), params.epsilon,                      \
        static_cast<T const*>(params.residual), params.buffer_flags, params.batch, params.stream)

void twoshot_rmsnorm_op(RMSNormParams const& params)
{
    auto dtype = params.dtype;
    if (dtype == nvinfer1::DataType::kFLOAT)
    {
        switch (params.hidden_dim)
        {
        case 2048: LAUNCH_RMSNORM_KERNEL(float, 2048); break;
        case 4096: LAUNCH_RMSNORM_KERNEL(float, 4096); break;
        // Llama-4 Hidden Dimension
        case 5120: LAUNCH_RMSNORM_KERNEL(float, 5120); break;
        // DeepSeek Hidden Dimension
        case 7168: LAUNCH_RMSNORM_KERNEL(float, 7168); break;
        case 8192: LAUNCH_RMSNORM_KERNEL(float, 8192); break;
        default: TLLM_CHECK_WITH_INFO(false, "[MNNVL TwoShot RMSNorm]: unsupported hidden_dim.");
        }
    }
    else if (dtype == nvinfer1::DataType::kBF16)
    {
        switch (params.hidden_dim)
        {
        case 2048: LAUNCH_RMSNORM_KERNEL(__nv_bfloat16, 2048); break;
        case 4096: LAUNCH_RMSNORM_KERNEL(__nv_bfloat16, 4096); break;
        // Llama-4 Hidden Dimension
        case 5120: LAUNCH_RMSNORM_KERNEL(__nv_bfloat16, 5120); break;
        // DeepSeek Hidden Dimension
        case 7168: LAUNCH_RMSNORM_KERNEL(__nv_bfloat16, 7168); break;
        case 8192: LAUNCH_RMSNORM_KERNEL(__nv_bfloat16, 8192); break;
        default: TLLM_CHECK_WITH_INFO(false, "[MNNVL TwoShot RMSNorm]: unsupported hidden_dim.");
        }
    }
    else
    {
        TLLM_CHECK_WITH_INFO(false, "[MNNVL TwoShot RMSNorm]: unsupported dtype.");
    }
}

} // namespace tensorrt_llm::kernels::mnnvl
