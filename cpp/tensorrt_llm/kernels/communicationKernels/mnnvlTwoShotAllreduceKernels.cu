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

// Guard for internal helper functions
namespace
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

template <>
inline __device__ float toFloat<__nv_half>(__nv_half val)
{
    return __half2float(val);
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

template <>
inline __device__ __nv_half fromFloat<__nv_half>(float val)
{
    return __float2half(val);
}

inline __device__ float2 loadfloat2(void const* ptr)
{
    float2 return_value;
    asm volatile("ld.volatile.global.v2.f32 {%0, %1}, [%2];\n" : "=f"(return_value.x), "=f"(return_value.y) : "l"(ptr));
    return return_value;
}

template <typename T>
inline __device__ T divUp(T val, T divisor)
{
    return (val + divisor - 1) / divisor;
}

__device__ struct __attribute__((aligned(32))) LamportFlags
{
    uint32_t buffer_size;
    uint32_t input_offset;
    uint32_t clear_offset;
    uint32_t num_tokens_prev;
    uint32_t* offset_access_ptr;
    uint32_t* buffer_flags;

    __device__ explicit LamportFlags(uint32_t* buffer_flags, uint32_t buffer_size)
        : offset_access_ptr(&buffer_flags[4])
        , buffer_flags(buffer_flags)
        , buffer_size(buffer_size)
    {
        uint4 flag = reinterpret_cast<uint4*>(buffer_flags)[0];
        input_offset = flag.x * (buffer_size << 1U);
        clear_offset = flag.y * (buffer_size << 1U);
        num_tokens_prev = flag.z;
    }

    __device__ void cta_arrive()
    {
        __syncthreads();
        if (threadIdx.x == 0)
        {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000))
            asm volatile("red.async.release.global.gpu.add.u32 [%0], %1;" ::"l"(offset_access_ptr), "r"(1) : "memory");
#elif (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
            asm volatile("red.global.gpu.add.u32 [%0], %1;" ::"l"(offset_access_ptr), "r"(1) : "memory");
#else
            atomicAdd(offset_access_ptr, 1);
#endif
        }
    }

    __device__ void wait_and_update(uint32_t num_tokens)
    {
        if (threadIdx.x == 0 && blockIdx.x == gridDim.x - 1 && blockIdx.y == 0)
        {
            while (*reinterpret_cast<uint32_t volatile*>(offset_access_ptr) < gridDim.x * gridDim.y)
            {
            }
            uint4 flag = reinterpret_cast<uint4*>(buffer_flags)[0];
            buffer_flags[0] = (flag.x + 1) % 3;
            buffer_flags[1] = (flag.y + 1) % 3;
            buffer_flags[2] = num_tokens;
            *(offset_access_ptr) = 0;
        }
    }
};
} // namespace

template <int WORLD_SIZE, typename T>
__global__ void twoshot_allreduce_kernel(T* output_ptr, T* shard_ptr, T** input_ptrs, T* mcast_ptr, int num_tokens,
    int buffer_M, int token_dim, int rank, uint32_t buffer_size, uint32_t* buffer_flags, bool wait_for_results)
{
    int elt = blockIdx.y * blockDim.x + threadIdx.x;
    if (elt >= token_dim)
        return;
    int token = blockIdx.x;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif

    LamportFlags flags(buffer_flags, buffer_size);

    // Capture the number of tokens in previous iteration so that we can properly clear the buffer
    // The scatter stage will use the buffer in WORLD_SIZE granularity, thus we need to round up
    uint32_t clr_toks_cta
        = divUp<uint32_t>(flags.num_tokens_prev > num_tokens ? flags.num_tokens_prev : num_tokens, WORLD_SIZE)
        * WORLD_SIZE;
    clr_toks_cta = divUp<uint32_t>(clr_toks_cta, gridDim.x);

    if (elt < token_dim)
    {
        // Scatter token
        int dest_rank = token % WORLD_SIZE;
        int dest_token_offset = token / WORLD_SIZE;
        T val = shard_ptr[token * token_dim + elt];
        if (isNegZero(val))
            val = fromFloat<T>(0.f);
        input_ptrs[dest_rank][flags.input_offset + dest_token_offset * token_dim * WORLD_SIZE + rank * token_dim + elt]
            = val;

        // Clear the buffer used by the previous call. Note the number of tokens to clear could be larger than the
        // number of tokens in the current call.
        for (int clr_tok = 0; clr_tok < clr_toks_cta; clr_tok++)
        {
            uint32_t clr_token_idx = token + clr_tok * gridDim.x;
            if (clr_token_idx < buffer_M)
            {
                input_ptrs[rank][flags.clear_offset + clr_token_idx * token_dim + elt] = fromFloat<T>(-0.f);
            }
        }

        // Reduce and broadcast
        if ((token % WORLD_SIZE) == rank)
        {
            int local_token = token / WORLD_SIZE;
            float accum = 0.f;

            T values[WORLD_SIZE];
            while (1)
            {
                bool valid = true;
                for (int r = 0; r < WORLD_SIZE; r++)
                {
                    T volatile* lamport_ptr = (T volatile*) &input_ptrs[rank][flags.input_offset
                        + local_token * token_dim * WORLD_SIZE + r * token_dim + elt];
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
            mcast_ptr[flags.input_offset + buffer_M * token_dim + token * token_dim + elt] = fromFloat<T>(accum);
        }
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
    if (elt < token_dim)
    {
        // Similarly clear broadcast buffer here
        for (int clr_tok = 0; clr_tok < clr_toks_cta; clr_tok++)
        {
            uint32_t clr_token_idx = token + clr_tok * gridDim.x;
            if (clr_token_idx < buffer_M)
            {
                input_ptrs[rank][flags.clear_offset + buffer_M * token_dim + clr_token_idx * token_dim + elt]
                    = fromFloat<T>(-0.f);
            }
        }
    }

    // Optionally wait for results if the next layer isn't doing the Lamport check
    if (wait_for_results)
    {
        // Update the atomic counter to indicate the block has read the offsets
        flags.cta_arrive();

        // Only use a set of CTAs for lamport sync, reargange the grid
        constexpr int ELTS_PER_LOAD = sizeof(float2) / sizeof(T);
        // blockDim.x / ELTS_PER_LOAD should be at least the size of a warp (32)
        if (threadIdx.x < (blockDim.x / ELTS_PER_LOAD))
        {
            uint64_t elt_load_offset = blockIdx.y * blockDim.x + threadIdx.x * ELTS_PER_LOAD;
            if (elt_load_offset < token_dim)
            {
                uint64_t current_pos = blockIdx.x * token_dim + elt_load_offset;

                void* lamport_ptr = (void*) &input_ptrs[rank][flags.input_offset + buffer_M * token_dim + current_pos];
                // We have 2 assumptions here:
                // 1. The write is atomic in 8B granularity -> Each buffer in the buffer group should be aligned to 8B
                // 2. The num_token * token_dim is divisible by ELTS_PER_LOAD (4 for BF16 and 2 for FP32)
                float2 val = loadfloat2(lamport_ptr);
                while (isNegZero(*(T*) &val))
                {
                    val = loadfloat2(lamport_ptr);
                }
                if (output_ptr)
                {
                    *((float2*) &output_ptr[current_pos]) = val;
                }
            }
        }

        // Update the buffer flags
        flags.wait_and_update(num_tokens);
    }
}

#define LAUNCH_ALL_REDUCE_KERNEL(WORLD_SIZE, T)                                                                        \
    TLLM_CUDA_CHECK(                                                                                                   \
        cudaLaunchKernelEx(&config, &twoshot_allreduce_kernel<WORLD_SIZE, T>, reinterpret_cast<T*>(params.output),     \
            reinterpret_cast<T*>(params.input), reinterpret_cast<T**>(params.buffer_ptrs_dev),                         \
            (T*) params.multicast_ptr, params.num_tokens, params.buffer_M, params.token_dim, params.rank,              \
            params.buffer_size, reinterpret_cast<uint32_t*>(params.buffer_flags), params.wait_for_results));

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
    else if (dtype == nvinfer1::DataType::kHALF)
    {
        switch (world_size)
        {
        case 2: LAUNCH_ALL_REDUCE_KERNEL(2, __nv_half); break;
        case 4: LAUNCH_ALL_REDUCE_KERNEL(4, __nv_half); break;
        case 8: LAUNCH_ALL_REDUCE_KERNEL(8, __nv_half); break;
        case 16: LAUNCH_ALL_REDUCE_KERNEL(16, __nv_half); break;
        case 32: LAUNCH_ALL_REDUCE_KERNEL(32, __nv_half); break;
        case 64: LAUNCH_ALL_REDUCE_KERNEL(64, __nv_half); break;
        default: TLLM_CHECK_WITH_INFO(false, "TwoShot AllReduce]: unsupported world_size.");
        }
    }
    else
    {
        TLLM_CHECK_WITH_INFO(false, "TwoShot AllReduce]: unsupported dtype.");
    }
}

// Guard for internal helper functions
namespace
{
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

template <typename T>
inline __device__ T add(T a, T b)
{
    return a + b;
}

#define FINAL_MASK 0xffffffff
#define WARP_SIZE 32

template <typename T>
__inline__ __device__ T warpReduceSum(T val)
{
    // Get the actual number of active threads in this warp
    int active_warp_size = min(WARP_SIZE, blockDim.x - (threadIdx.x & ~(WARP_SIZE - 1)));
    unsigned int mask = (1U << active_warp_size) - 1;

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        if (offset < active_warp_size)
        {
            val = add<T>(val, __shfl_xor_sync(mask, val, offset, WARP_SIZE));
        }
    }
    return val;
}

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
inline __device__ float block_reduce_sum(float val)
{
    __shared__ float smem[WARP_SIZE];
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int warp_num = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE; // Ceiling division to include partial warps

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

__device__ float4 loadfloat4(void const* ptr)
{

    float4 return_value;

    asm volatile("ld.volatile.global.v4.f32 {%0, %1, %2, %3}, [%4];\n"
                 : "=f"(return_value.x), "=f"(return_value.y), "=f"(return_value.z), "=f"(return_value.w)
                 : "l"(ptr));

    return return_value;
}
#endif
} // namespace

template <int DIM, int NUM_THREADS, int NUM_INPUTS, typename T_OUT, typename T_IN>
__global__ void __launch_bounds__(128, 1)
    RMSNorm(T_IN* input_plus_residual, T_OUT* output_norm, T_IN const* buffer_input, T_IN const* gamma, float epsilon,
        T_IN const* residual, int batch_size, uint32_t buffer_size, uint32_t* buffer_flags)
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

    LamportFlags flags(buffer_flags, buffer_size);
    T_IN const* input = &buffer_input[flags.input_offset + flags.buffer_size];

    cudaTriggerProgrammaticLaunchCompletion();

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
    flags.cta_arrive();
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
    flags.wait_and_update(batch_size);
#endif
}

template <typename T, int H_DIM, int NUM_THREADS>
void twoshot_rmsnorm(T* prenorm_output, T* normed_output, T const* input, T const* gamma, double epsilon,
    T const* residual, uint32_t buffer_size, uint32_t* buffer_flags, int batch, cudaStream_t stream)
{

    // input to rmsnorm is the buffer in the twoshot ar
    // We should use prenorm output to determine the actual used size
    float _epsilon{static_cast<float>(epsilon)};

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
        input, gamma, _epsilon, residual, batch, buffer_size, buffer_flags));
}

#define LAUNCH_RMSNORM_KERNEL(T, H_DIM, NUM_THREADS)                                                                   \
    twoshot_rmsnorm<T, H_DIM, NUM_THREADS>(static_cast<T*>(params.residual_output), static_cast<T*>(params.output),    \
        static_cast<T const*>(params.input), static_cast<T const*>(params.gamma), params.epsilon,                      \
        static_cast<T const*>(params.residual), params.buffer_size, params.buffer_flags, params.batch, params.stream)

void twoshot_rmsnorm_op(RMSNormParams const& params)
{
    auto dtype = params.dtype;

#define CASE_DISPATCH_RMSNORM(T, H_DIM, NUM_THREADS)                                                                   \
    case H_DIM: LAUNCH_RMSNORM_KERNEL(T, H_DIM, NUM_THREADS); break;

#define TYPE_DISPATCH_RMSNORM(T)                                                                                       \
    CASE_DISPATCH_RMSNORM(T, 2048, 128)                                                                                \
    CASE_DISPATCH_RMSNORM(T, 2880, 120)                                                                                \
    CASE_DISPATCH_RMSNORM(T, 4096, 128)                                                                                \
    CASE_DISPATCH_RMSNORM(T, 5120, 128)                                                                                \
    CASE_DISPATCH_RMSNORM(T, 7168, 128)                                                                                \
    CASE_DISPATCH_RMSNORM(T, 8192, 128)

    if (dtype == nvinfer1::DataType::kFLOAT)
    {
        switch (params.hidden_dim)
        {
            TYPE_DISPATCH_RMSNORM(float);
        default: TLLM_CHECK_WITH_INFO(false, "[MNNVL TwoShot RMSNorm]: unsupported hidden_dim.");
        }
    }
    else if (dtype == nvinfer1::DataType::kBF16)
    {
        switch (params.hidden_dim)
        {
            TYPE_DISPATCH_RMSNORM(__nv_bfloat16);
        default: TLLM_CHECK_WITH_INFO(false, "[MNNVL TwoShot RMSNorm]: unsupported hidden_dim.");
        }
    }
    else if (dtype == nvinfer1::DataType::kHALF)
    {
        switch (params.hidden_dim)
        {
            TYPE_DISPATCH_RMSNORM(__nv_half);
        default: TLLM_CHECK_WITH_INFO(false, "[MNNVL TwoShot RMSNorm]: unsupported hidden_dim.");
        }
    }
    else
    {
        TLLM_CHECK_WITH_INFO(false, "[MNNVL TwoShot RMSNorm]: unsupported dtype.");
    }
#undef TYPE_DISPATCH_RMSNORM
#undef CASE_DISPATCH_RMSNORM
}

} // namespace tensorrt_llm::kernels::mnnvl
